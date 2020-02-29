import logging
import os
import pickle
import time
from collections import deque
from pprint import pformat
from numpy.random import randint

import redis

logger = logging.getLogger(__name__)

EXP_KEY = 'es:exp'
TASK_ID_KEY = 'es:task_id'
TASK_DATA_KEY = 'es:task_data'
TASK_CHANNEL = 'es:agent_task_channel'
TASK_ID_CHANNEL = 'es:task_id_channel'
TASK_CLEAR_CHANNEL = 'es:task_clear_channel'
RESULTS_KEY = 'es:results'

def serialize(x):
    return pickle.dumps(x, protocol=-1)


def deserialize(x):
    return pickle.loads(x)


def retry_connect(redis_cfg, tries=300, base_delay=4.):
    for i in range(tries):
        try:
            r = redis.StrictRedis(**redis_cfg)
            r.ping()
            return r
        except redis.ConnectionError as e:
            if i == tries - 1:
                raise
            else:
                delay = base_delay * (1 + (os.getpid() % 10) / 9)
                print('Could not connect to {}. Retrying after {:.2f} sec ({}/{}). Error: {}'.format(
                    redis_cfg, delay, i + 2, tries, e))
                time.sleep(delay)


def retry_get(pipe, key, tries=300, base_delay=4.):
    for i in range(tries):
        # Try to (m)get
        if isinstance(key, (list, tuple)):
            vals = pipe.mget(key)
            if all(v is not None for v in vals):
                return vals
        else:
            val = pipe.get(key)
            if val is not None:
                return val
        # Sleep and retry if any key wasn't available
        if i != tries - 1:
            delay = base_delay * (1 + (os.getpid() % 10) / 9)
            logger.warning('{} not set. Retrying after {:.2f} sec ({}/{})'.format(key, delay, i + 2, tries))
            time.sleep(delay)
    raise RuntimeError('{} not set'.format(key))

def retry_pop(pipe, key, tries=300, base_delay=4.):
    for i in range(tries):
        val = pipe.brpop(key).execute() # brpop returns a list (?), also need to run execute otherwise we don't know whats in there.
        #print("[popper] val popped:", type(val), type(val[0]))
        if val[0] is not None:
            return val
        # Sleep and retry if any key wasn't available
        if i != tries - 1:
            delay = base_delay * (1 + (os.getpid() % 10) / 9)
            logger.warning('{} not set. Retrying after {:.2f} sec ({}/{})'.format(key, delay, i + 2, tries))
            time.sleep(delay)
    raise RuntimeError('{} not set'.format(key))


class MasterClient:
    def __init__(self, master_redis_cfg):
        self.task_counter = 0
        self.master_redis = retry_connect(master_redis_cfg)
        logger.info('[master] Connected to Redis: {}'.format(self.master_redis))

    def declare_experiment(self, exp):
        self.master_redis.set(EXP_KEY, serialize(exp))
        logger.info('[master] Declared experiment {}'.format(pformat(exp)))
        print("declared experiment:", deserialize(self.master_redis.get(EXP_KEY)))

    def increment_task(self):
        """increments task counter (iteration number) and updates redis"""
        self.task_counter += 1
        logger.debug('[master] Updated task counter to {}'.format(self.task_counter))
        (self.master_redis.pipeline()
         .mset({TASK_ID_KEY: self.task_counter})
         .publish(TASK_ID_CHANNEL, self.task_counter)
         .execute()
        )
        return self.task_counter

    def declare_agent_tasks_batched(self, curr_task_id, tasks):
        serialized_tasks = [serialize(task) for task in tasks]
        pipe = self.master_redis.pipeline()
        for task in serialized_tasks:
            (pipe.rpush(TASK_DATA_KEY, task)
            .publish(TASK_CHANNEL, task)
            )
        pipe.execute()


    def clear_agent_tasks(self):
        (self.master_redis.pipeline()
         .delete(TASK_DATA_KEY)
         .publish(TASK_CLEAR_CHANNEL, True)
         .execute()
         )
        print("[master] cleared all tasks.")

    def pop_result(self):
        res = self.master_redis.blpop(RESULTS_KEY, timeout=2*60)

        if res is None:
            return None
        else:
            res = deserialize(res[1])
        task_id, agent_id, result = res
        logger.debug('[master] Popped a result for agent {} on task {}'.format(agent_id, task_id))
        return task_id, agent_id, result


class RelayClient:
    """
    Receives and stores task broadcasts from the master
    Batches and pushes results from workers to the master
    """

    def __init__(self, master_redis_cfg, relay_redis_cfg):
        self.master_redis = retry_connect(master_redis_cfg)
        print('[relay] Connected to master: {}'.format(self.master_redis))
        self.local_redis = retry_connect(relay_redis_cfg)
        print('[relay] Connected to relay: {}'.format(self.local_redis))

    def run(self):
        # Initialization: read exp and latest task from master
        self.local_redis.set(EXP_KEY, retry_get(self.master_redis, EXP_KEY))
        task_id = self._declare_task_id_local(retry_get(self.master_redis,
                                                        TASK_ID_KEY))

        #TODO: one reason why distributed crap might not work is because
        # there's a relay on EACH worker.
        # so then instead of pulling ALL tasks at the beginning, just pull one.
        # then the relay should pop from the MASTER.
        tasks = self.master_redis.lrange(TASK_DATA_KEY, 0, -1)
        print("[relay] num tasks:", len(tasks))
        if len(tasks) == 0:
            print("[relay] no tasks from master, waiting for pub/sub...")
        else:
            for task in tasks:
                self._declare_task_local(task)

        # Start subscribing to tasks
        p = self.master_redis.pubsub(ignore_subscribe_messages=True)
        p.subscribe(**{TASK_CHANNEL: lambda msg: self._declare_task_local(msg['data'])})
        p.subscribe(**{TASK_ID_CHANNEL: lambda msg: self._declare_task_id_local(msg['data'])})
        p.subscribe(**{TASK_CLEAR_CHANNEL: lambda msg: self._clear_tasks_local()})
        p.run_in_thread(sleep_time=0.001)

        # Loop on RESULTS_KEY and push to master
        batch_sizes, last_print_time = deque(maxlen=20), time.time()  # for logging
        while True:
            results = []
            start_time = curr_time = time.time()
            while curr_time - start_time < 0.001:
                print("[relay] popping results from local redis...")
                results.append(self.local_redis.blpop(RESULTS_KEY)[1])
                curr_time = time.time()
            print("[relay] pushing results to master...")
            self.master_redis.rpush(RESULTS_KEY, *results)
            # Log
            batch_sizes.append(len(results))
            if curr_time - last_print_time > 5.0:
                print('[relay] Average batch size {:.3f}'.format(sum(batch_sizes) / len(batch_sizes)))
                last_print_time = curr_time

    def _clear_tasks_local(self):
        self.local_redis.delete(TASK_DATA_KEY)
        # also delete results on master so we stop having redundant results.
        self.master_redis.delete(RESULTS_KEY)

    def _declare_task_local(self, task):
         """tasks should be a list of serialized tasks"""
         print("[relay] declaring tasks locally...")
         self.local_redis.rpush(TASK_DATA_KEY, task)

    def _declare_task_id_local(self, task_id):
        print('[relay] Updating task to {}'.format(task_id))
        self.local_redis.mset({TASK_ID_KEY: task_id})
        return task_id


class WorkerClient:
    def __init__(self, relay_redis_cfg):
        self.local_redis = retry_connect(relay_redis_cfg)
        logger.info('[worker] Connected to relay: {}'.format(self.local_redis))

        self.task_id, self.task_data, self.agent_id = None, None, None

    def get_experiment(self):
        # Grab experiment info
        exp = deserialize(retry_get(self.local_redis, EXP_KEY))
        logger.info('[worker] Experiment: {}'.format(exp))
        return exp

    def get_current_task(self, num_agents):

        with self.local_redis.pipeline() as pipe:
            while True:
                # each worker has a different self.task_id
                #self.task_id = int(retry_get(pipe, TASK_ID_KEY).execute()[0])
                self.task_data = None

                while self.task_data is None:
                    pipe.watch(TASK_ID_KEY)
                    # sometimes retry_get returns a value, sometimes a pipe. ugh.
                    # TODO: make less ugly, understand pipes better
                    try:
                        self.task_id = int(retry_get(pipe, TASK_ID_KEY))
#                        print("[worker]: got task id: {}".format(self.task_id))
                    except TypeError:
                        self.task_id = int(retry_get(pipe, TASK_ID_KEY).execute()[0])
#                        print("[worker]: got task id: {}".format(self.task_id))

                    try:
                        pipe.watch(TASK_ID_KEY)
                        pipe.multi()
                        pipe.blpop(TASK_DATA_KEY, timeout=2*60)
                        self.task_data = pipe.execute()[0]
                    except TypeError as e:
                        print("[worker] ... got a type error:", e)
                        continue
                    except redis.WatchError:
                        print("[worker] multi break (watch error). Setting task_data to None and continuing.")
                        self.task_data = None
                        continue

                self.task_data = deserialize(self.task_data[1])
                if self.task_data.task_id != self.task_id:
                    print("[worker] the task data I pulled isn't the same as the task ID in redis, continuing...")
                    continue

                self.task_id, self.agent_id = self.task_data.task_id, self.task_data.agent_id
                #pipe.rpush(TASK_DATA_KEY, serialize(self.task_data))
                pipe.watch(TASK_ID_KEY)
                print('[worker] Got new task {} for agent {}'.format(self.task_id, self.agent_id))

                return self.task_id, self.task_data, self.agent_id


    def push_result(self, task_id, agent_id, result):
        self.local_redis.rpush(RESULTS_KEY, serialize((task_id, agent_id, result)))
        print('[worker] Pushed result for task {}'.format(task_id))
