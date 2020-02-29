import logging
import time
from collections import namedtuple
import os.path as osp
import os
import itertools
#import roboschool
import copy
import sys
import time
import pandas as pd
import pickle
import h5py
import json
import math

import numpy as np
from scipy.spatial.distance import euclidean

#from pathos.multiprocessing import ProcessingPool, ThreadingPool
from multiprocessing import Process, Queue
from .dist import MasterClient, WorkerClient
from . import agent_updaters
from . import multi
from . import networks

#logger = logging.getLogger('')

dayandtime = str(time.strftime("%Y-%m-%d")+'_'+time.strftime("%H:%M:%S"))


logging.basicConfig(
        filename='events.log',
        format='[%(asctime)s pid=%(process)d] %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logger = logging.getLogger('')



Config = namedtuple('Config', [
    'num_agents',
    'l2coeff', 'noise_stdev', 'episodes_per_batch', 'timesteps_per_batch',
    'calc_obstat_prob', 'eval_prob', 'snapshot_freq',
    'return_proc_mode', 'episode_cutoff_mode', 'num_threads', 'broadcast_prob', 'experiments_filename', 'experiment_group_name'
])

Task = namedtuple('Task', ['agent_id', 'task_id', 'params', 'eval_job', 'best_theta', 'ob_mean', 'ob_std', 'timestep_limit'])
# Task = namedtuple('Task', ['agent_id', 'task_id', 'params', 'ob_mean', 'ob_std', 'timestep_limit'])
Result = namedtuple('Result', [
    'agent_id',
    'task_id',
    'worker_id',
    'noise_inds_n', 'returns_n2', 'signreturns_n2', 'lengths_n2',
    'eval_return', 'eval_length',
    'ob_sum', 'ob_sumsq', 'ob_count'
])

Agent = namedtuple('Agent', ['optimizer', 'theta'])

class RunningStat(object):
    def __init__(self, shape, eps):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.full(shape, eps, dtype=np.float32)
        self.count = eps

    def increment(self, s, ssq, c):
        self.sum += s
        self.sumsq += ssq
        self.count += c

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.mean), 1e-2))

    def set_from_init(self, init_mean, init_std, init_count):
        self.sum[:] = init_mean * init_count
        self.sumsq[:] = (np.square(init_mean) + np.square(init_std)) * init_count
        self.count = init_count


class SharedNoiseTable(object):
    def __init__(self):
        import ctypes, multiprocessing
        seed = 123
        count = 250000000  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        logger.info('Sampling {} random numbers with seed {}'.format(count, seed))
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        assert self.noise.dtype == np.float32
        self.noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here
        logger.info('Sampled {} bytes'.format(self.noise.size * 4))

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)


def make_session(single_threaded):
    import tensorflow as tf
    if not single_threaded:
        return tf.InteractiveSession()
    return tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1))


def setup(exp, single_threaded, snapshot_file=None):
    import gym
    gym.undo_logger_setup()
    from . import policies, tf_util

    config = Config(**exp['config'])
    env = gym.make(exp['env_id'])
    session = make_session(single_threaded=single_threaded)
    if snapshot_file is None:
        policy = getattr(policies, exp['policy']['type'])(env.observation_space,
                                                          env.action_space,
                                                          **exp['policy']['args'])
        running_from_snapshot = False
    else:
        print("[master] Initializing agent weights using a snapshot...")
        policy = getattr(policies, exp['policy']['type']).Load(snapshot_file)
        running_from_snapshot = True
    tf_util.initialize()
    return config, env, session, policy, running_from_snapshot


def setup_worker(exp, single_threaded):
    import gym
    gym.undo_logger_setup()
    from . import policies, tf_util

    config = Config(**exp['config'])
    env = gym.make(exp['env_id'])
    sess = make_session(single_threaded=single_threaded)
    policy = getattr(policies, exp['policy']['type'])(env.observation_space,
                                                      env.action_space,
                                                      **exp['policy']['args'])
    tf_util.initialize()

    return config, env, sess, policy

def run_master_from_snapshot(master_redis_cfg, snapshot, log_dir):
    from . import policies, es
    print('[master] trying to run master from a snapshot at: {}'.format(snapshot))
    with h5py.File(snapshot, 'r') as f:
        exp = pickle.loads(f.attrs['exp'].tostring())
        config = es.Config(**exp['config'])
        iteration = f.attrs['iteration']
        exp_name = os.path.splitext(exp['config']['experiments_filename'][0])
        exp_group_name = exp['config']['experiment_group_name']

    run_master(master_redis_cfg, log_dir, exp, exp_name, exp_group_name,
                snapshot_fname=snapshot, snapshot_iteration=iteration)

def run_master(master_redis_cfg, log_dir, exp, exp_name, exp_group_name, snapshot_fname=None, snapshot_iteration=None):
    logger.info('run_master: {}'.format(locals()))
    from .optimizers import SGD, Adam
    from . import tabular_logger as tlogger
    logger.info('Tabular logging to {}'.format(log_dir))
    tlogger.start(log_dir)

    config, env, session, policy, running_from_snapshot = setup(exp, single_threaded=False, snapshot_file=snapshot_fname)

    num_params = policy.num_params

    master = MasterClient(master_redis_cfg)

    dayandtime = str(time.strftime("%Y-%m-%d")+'_'+time.strftime("%H:%M:%S"))
    all_results = pd.DataFrame()
    each_agent_results = pd.DataFrame()
    each_agent_results_mean = pd.DataFrame()


    def optimizer(thetas):
        return {'sgd': SGD, 'adam': Adam}[exp['optimizer']['type']](thetas, **exp['optimizer']['args'])


    if running_from_snapshot:
        print("[master] Done!")
        print("[master] Continuing experiment from iteration {}".format(snapshot_iteration))
        # update number of iterations in task counter (dont forget to cast to int!)
        master.task_counter = int(snapshot_iteration)
        time.sleep(2)


    tlogger.log('[master] Generating shared noise table...')
    np.random.seed(1337)
    noise = SharedNoiseTable()
    rs = np.random.RandomState()
    ob_stat = RunningStat(
        env.observation_space.shape, # all the same
        eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
    )
    tlogger.log('[master] Generated shared noise table...')

    def all_weights_same(l):
        for i,a in enumerate(l):
            for j,b in enumerate(l):
                if not all(np.isclose(a.theta,
                                      b.theta)):
                    print("--- Policies are different in fully-connected network. ---")
                    print("Agent 1: {} | Agent 2: {}".format(i, j))
                    return False
        print("--- All weights are the same before update ---")
        return True

    if not exp["policy"]["args"]["same_param_start"]:
        print("--- Starting agents with different parameters by adding noise ---")
        theta = policy.get_trainable_flat()

        thetas = []
        for agent in range(int(config.num_agents)):
            noise_idx = noise.sample_index(rs, policy.num_params)
            v = config.noise_stdev * noise.get(noise_idx, policy.num_params)
            thetas.append(theta + v)

        agents = [Agent(optimizer=optimizer(theta),
                        theta=thetas[n])
                  for n in range(int(config.num_agents))]
        # assert not all_weights_same(agents) #takes way too long with 1000 agents (7 mins with 300 agents)

    else:
        theta = policy.get_trainable_flat()
        agents = [Agent(optimizer=optimizer(theta),
                        theta=theta)
                  for n in range(int(config.num_agents))]
        # assert all_weights_same(agents)   #takes way too long with 1000 agents (7 mins with 300 agents)

    if config.episode_cutoff_mode.startswith('adaptive:'):
        _, args = config.episode_cutoff_mode.split(':')
        arg0, arg1, arg2 = args.split(',')
        tslimit, incr_tslimit_threshold, tslimit_incr_ratio = int(arg0), float(arg1), float(arg2)
        adaptive_tslimit = True
        logger.info(
            'Starting timestep limit set to {}. When {}% of rollouts hit the limit, it will be increased by {}'.format(
                tslimit, incr_tslimit_threshold * 100, tslimit_incr_ratio))
    elif config.episode_cutoff_mode == 'env_default':
        tslimit, incr_tslimit_threshold, tslimit_incr_ratio = None, None, None
        adaptive_tslimit = False
    else:
        raise NotImplementedError(config.episode_cutoff_mode)

    episodes_so_far = 0
    timesteps_so_far = 0
    tstart = time.time()
    tlogger.log('[master] declaring experiment...')
    master.declare_experiment(exp)
    tlogger.log('[master] declared experiment.')

    network_file = exp['agent_update']['args']['network_file']
    network_filename = '{}/networkx_graph_files/{}'.format(os.environ['HOME'], network_file)

    if exp["agent_update"]["on_the_fly_topology"]:
        print("\n>>Generating a network for this run\n")
        net_type = exp["agent_update"]["args"]["network_type"]
        n_agents = exp["config"]["num_agents"]
        net_p = exp["agent_update"]["args"]["network_args"]["p"]
        print("\n>>TYPE: {}".format(net_type))
        if net_type == "fully-connected":
            adjacency_matrix = networks.fully_connected(n_agents)
        elif net_type == "erdos":
            adjacency_matrix = networks.erdos_renyi_connected(n_agents, net_p)
        else:
            # just break everything with this line lol
            adjacency_matrix=None
    else:
        print("\n>>Using a network file for this run\n")
        print('Network file being used: {}'.format(network_filename))
        with open(network_filename, "rb") as input_file:
            adjacency_matrix = pickle.load(input_file)


    agent_updater = getattr(agent_updaters, exp['agent_update']['strategy'])(noise,
                                                                             num_params,
                                                                             config.l2coeff,
                                                                             config.noise_stdev,
                                                                             adjacency_matrix=adjacency_matrix,
                                                                             num_agents=config.num_agents,
                                                                             **exp['agent_update']['args'])

    exp_group_name  =   exp['config']['experiment_group_name']
    exp_name        =   exp['config']['experiments_filename'][:-5] # to remove .json extension

    logdir= "./logs/{}".format(exp_group_name)
    snapshot_dir = './snapshots/{}'.format(exp_group_name)
    summary_log = osp.join(logdir, "summary_{}_{}.csv".format(exp_name, dayandtime))
    # filename_each_agent_results_mean = osp.join(logdir, "mean_each_agent_results{}_{}.csv".format(exp_name, dayandtime))
    # filename_each_agent_gradNorm = osp.join(logdir, "mean_each_agent_gradNorm{}_{}.csv".format(exp_name, dayandtime))
    # filename_each_agent_gradient = osp.join(logdir, "mean_each_agent_gradient{}_{}.csv".format(exp_name, dayandtime))

    if not osp.exists(logdir):
        os.makedirs(logdir)
    if not osp.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    best_theta = agents[0].theta #just a placeholder until it is re-written every iteration with the actual best theta (theta producing max results)

    while True:

        step_tstart = time.time()
        curr_task_id = master.increment_task()
        tlogger.log('********** Iteration {} **********'.format(curr_task_id))

        tlogger.log('[master] declaring all initial tasks...')
        # declare a task for each agent

        eval_job = False
        if rs.rand() < config.eval_prob:
            eval_job = True
            print('>>>>>>>>>>>>>>> THIS IS AN EVAL JOB')

        tasks = [Task(
            agent_id=agent_id,
            task_id=curr_task_id,
            params=agent.theta,
            eval_job=eval_job,
            best_theta=best_theta, #for first iteration this is essentially random
            ob_mean=ob_stat.mean if policy.needs_ob_stat else None,
            ob_std=ob_stat.std if policy.needs_ob_stat else None,
            timestep_limit=tslimit
        ) for agent_id, agent in enumerate(agents)]

        master.declare_agent_tasks_batched(curr_task_id, tasks)
        tlogger.log('[master] Declared tasks.')

        # Pop off results for the current task
        curr_task_results, eval_rets, eval_lens, worker_ids = [], [], [], []
        num_results_skipped, num_episodes_popped, num_timesteps_popped, ob_count_this_batch = 0, 0, 0, 0
        #while num_episodes_popped < config.episodes_per_batch or num_timesteps_popped < config.timesteps_per_batch:


        while len(curr_task_results) < config.num_agents:
            # Wait for a result
            res = master.pop_result()

            # if None, it means we've been blocking (not getting results)
            # for at least 10 seconds
            if res is None:
                print("[master] {} at least 5min secs since last result. Re-posting needed tasks...".format(time.strftime("%X", time.localtime())))
                agents_received = [r.agent_id for r in curr_task_results]
                tasks_left = [task for task in tasks if task.agent_id not in agents_received]
                print('[master] have {} tasks left, posting...'.format(len(tasks_left)))
                # TODO: check if we have all our tasks already in redis; if so, don't post!
                # check by doing the same thing relays do to pull all existing tasks....
                master.declare_agent_tasks_batched(curr_task_id, tasks_left)
                continue
            else:
                task_id, agent_id, result = res

            agent = agents[agent_id]

            # assert isinstance(task_id, int) and isinstance(result, Result)
            # assert (result.eval_return is None) == (result.eval_length is None)
            worker_ids.append(result.worker_id)

            # continue if we've already gotten a result for this agent
            #agent_task = master.get_agent_task(agent_id)
            agents_collected = [r.agent_id for r in curr_task_results]


            if agent_id in agents_collected or task_id != curr_task_id:
                print("[master] {} redundant result for agent".format(curr_task_id), agent_id, task_id)
                print("[master] {} have".format(curr_task_id), len(curr_task_results), " / ", config.num_agents, "results")
                continue

            if eval_job:
                print("[master] EVAL getting result for agent", agent_id, "waiting for {} more results...".format(config.num_agents - len(curr_task_results)))
            else:
                print("[master] getting result for agent", agent_id, "waiting for {} more results...".format(config.num_agents - len(curr_task_results)))

            #TODO: how to do eval
            if result.eval_length is not None:
                # This was an eval job
                # print('>>>> This was an eval job')
                episodes_so_far += 1
                timesteps_so_far += result.eval_length
                # Store the result only for current tasks
                if task_id == curr_task_id:
                    curr_task_results.append(result)
                    eval_rets.append(result.eval_return)
                    eval_lens.append(result.eval_length)
            else:
                # The real shit
                # assert (result.noise_inds_n.ndim == 1 and
                #         result.returns_n2.shape == result.lengths_n2.shape == (len(result.noise_inds_n), 2))
                # assert result.returns_n2.dtype == np.float32
                # Update counts
                result_num_eps = result.lengths_n2.size
                result_num_timesteps = result.lengths_n2.sum()
                episodes_so_far += result_num_eps
                timesteps_so_far += result_num_timesteps
                # Store results only for current tasks
                if task_id == curr_task_id:
                    curr_task_results.append(result)
                    num_episodes_popped += result_num_eps
                    num_timesteps_popped += result_num_timesteps

                    # Update ob stats
                    if policy.needs_ob_stat and result.ob_count > 0:
                        ob_stat.increment(result.ob_sum, result.ob_sumsq, result.ob_count)
                        ob_count_this_batch += result.ob_count
                else:
                    num_results_skipped += 1

        print('>>> clearing agent tasks')
        master.clear_agent_tasks()
        print('>>> done clearing agent tasks')

        # # Compute skip fraction
        # frac_results_skipped = num_results_skipped / (num_results_skipped + len(curr_task_results))
        # if num_results_skipped > 0:
        #     logger.warning('Skipped {} out of date results ({:.2f}%)'.format(
        #         num_results_skipped, 100. * frac_results_skipped))
        # print('>>> done frac results')

        # adjacency_matrix = np.ones((config.num_agents, config.num_agents))

        # keep track of all game lengths to figure out if we should
        # adapt the timestep limit


        if eval_job == False:
            global_noise_inds_n = np.concatenate([r.noise_inds_n for r in curr_task_results])
            global_returns_n2 = np.concatenate([r.returns_n2 for r in curr_task_results])
            global_lengths_n2 = np.concatenate([r.lengths_n2 for r in curr_task_results])
            global_agent_ids_n = [r.agent_id for r in curr_task_results]



            # calculate search diameter. We do this before agent update so that
            # search diameter is measured as a function of both noise (individual
            # exploration) and network effects. This also enables us to compare
            # networked strategies to fully-connected strategies.
            # thetas = np.array([agents[agent_id].optimizer.theta + noise.get(noise_idx, num_params)
            #                    for agent_id, noise_idx in zip(global_agent_ids_n, global_noise_inds_n)])
            # centroid = np.apply_along_axis(np.median, 0, thetas)
            # distances = np.apply_along_axis(lambda v: euclidean(centroid, v), 1, thetas)
            # print(distances)
            # print(np.mean(distances), np.std(distances))
            # search_diameter = np.max(distances)

            # save best result
            # if global_returns_n2.max() > 500:
            #     # save best_theta
            #     best_result = max(curr_task_results, key=lambda r: np.max(r.returns_n2))
            #     best_theta = agents[best_result.agent_id].optimizer.theta
            #     antithetic_idx = np.argmax([max(best_result.returns_n2[:,0]),
            #                                 max(best_result.returns_n2[:,1])])
            #     noise_idx = np.argmax(np.apply_along_axis(np.max, 1, best_result.returns_n2))
            #     v = config.noise_stdev * noise.get(best_result.noise_inds_n[noise_idx], num_params)
            #
            #     if antithetic_idx == 0:
            #         best_theta = best_theta + v
            #     else:
            #         best_theta = best_theta - v
            #
            #     snap_filename = osp.join(snapshot_dir, 'snapshot_{}_{}_iter{:05d}_rew{}'.format(
            #     # filename = osp.join(snapshot_dir, 'snapshot_{}_rew{}.h5'.format(
            #         exp_name,
            #         dayandtime,
            #         curr_task_id,
            #         global_returns_n2.max()
            #         # np.nan if not eval_rets else int(np.mean(eval_rets))
            #     ))
            #
            #     print("saving h5 file...")
            #     policy.set_trainable_flat(best_theta)
            #     policy.save(snap_filename + '.h5', exp, iteration=curr_task_id)

            # perform actual update


            arglist = [(agent_id, agents, curr_task_results) for agent_id in range(len(agents))]
            process_ids = range(len(agents))
            results = multi.mp_factorizer(arglist, process_ids, config.num_threads, agent_updater.update)
            agent_ids, new_agents, gs, update_ratios = zip(*results)
            agents = new_agents
            print("num agents:", len(agents))

            # Update ob stat (we're never running the policy in the master, but we might be snapshotting the policy)
            if policy.needs_ob_stat:
                policy.set_ob_stat(ob_stat.mean, ob_stat.std)

            # Update number of steps to take
            if adaptive_tslimit and (global_lengths_n2 == tslimit).mean() >= incr_tslimit_threshold:
                old_tslimit = tslimit
                tslimit = int(tslimit_incr_ratio * tslimit)
                logger.info('Increased timestep limit from {} to {}'.format(old_tslimit, tslimit))

            # if (config.broadcast_prob != 'na' and np.random.randn() < config.broadcast_prob): #normal random
            current_iteration_broadcast = np.nan
            if (config.broadcast_prob != 'na' and np.random.random() < config.broadcast_prob): current_iteration_broadcast = True #uniform random

            # being used in case of eval
            best_result = max(curr_task_results, key=lambda r: np.max(r.returns_n2))
            best_theta = agents[best_result.agent_id].optimizer.theta

            if current_iteration_broadcast == True:
                tlogger.log("[master] Broadcasting best parameters from network...")
                best_result = max(curr_task_results, key=lambda r: np.max(r.returns_n2))
                best_theta = agents[best_result.agent_id].optimizer.theta
                antithetic_idx = np.argmax([max(best_result.returns_n2[:,0]),
                                            max(best_result.returns_n2[:,1])])
                best_noise_idx = np.argmax(np.apply_along_axis(np.max, 1, best_result.returns_n2))
                new_agents = []
                for agent_id, agent in enumerate(agents):
                    agent.optimizer.theta = best_theta
                    new_agents.append(Agent(agent.optimizer, agent.optimizer.theta))

            # # so the best result is stored for each iteration, even without broadcast
            # best_result = max(curr_task_results, key=lambda r: np.max(r.returns_n2))


            # ## save current results
            # current_iteration_results_mean = dict()
            # for agent in curr_task_results:
            #     current_iteration_results_mean[agent.agent_id] = np.mean(list(itertools.chain(*agent.returns_n2))) #the complicated chain thing flattens the results
            # with open(filename_each_agent_results_mean, 'a') as file:
            #     file.write(str(current_iteration_results_mean)+"\n")
            #
            # # save GradNorm
            # getNorm = lambda g: float(np.linalg.norm(g))
            # current_iteration_gradNorms = {agent_id: getNorm(g) for agent_id, g in zip(agent_ids, gs)}
            # # with open(filename_each_agent_gradNorm, 'a') as file:
            # #     file.write(str(current_iteration_gradNorms)+"\n")
            # current_iteration_gradNorms = list(current_iteration_gradNorms.values())
            # #
            # # save actual gradient
            # #getNorm = lambda g: float(np.linalg.norm(g))
            # current_iteration_gradients = {agent_id: g for agent_id, g in zip(agent_ids, gs)}
            # with open(filename_each_agent_gradient, 'a') as file:
            #     file.write(str(current_iteration_gradients)+"\n")
            # current_iteration_gradients = list(current_iteration_gradients.values())
        #
        # # save a snapshot?
        # if config.snapshot_freq != 0 and curr_task_id % config.snapshot_freq == 0:
        #
        #     snap_filename = osp.join(snapshot_dir, 'snapshot_{}_{}_iter{:05d}_rew{}.h5'.format(
        #     # filename = osp.join(snapshot_dir, 'snapshot_{}_rew{}.h5'.format(
        #         exp_name,
        #         dayandtime,
        #         curr_task_id,
        #         np.nan if not eval_rets else int(np.mean(eval_rets))
        #     ))
        #
        #     print("[master] {} saving parameters to file: {}".format(dayandtime, snap_filename))
        #     # assert not osp.exists(filename)
        #
        #     with h5py.File(snap_filename, 'w') as f:
        #         for agent_id, agent in enumerate(agents):
        #             f.create_dataset(str(agent_id), data=agent.theta)
        #         f.attrs['num_agents'] = config.num_agents
        #         f.attrs['network_file'] = exp['agent_update']['args']['network_file']
        #         f.attrs['exp'] = json.dumps(exp)
        #         f.attrs['exp_name'] = exp_name
        #         f.attrs['exp_group_name'] = exp_group_name
        #         f.attrs['iteration'] = curr_task_id
        #         f.attrs['date'] = dayandtime
        #
        #     tlogger.log('Saved snapshot {}'.format(snap_filename))

        step_tend = time.time()
        # tlogger.record_tabular("EpRewMean", global_returns_n2.mean())
        # tlogger.record_tabular("EpRewMax", global_returns_n2.max())
        # tlogger.record_tabular("EpRewMin", global_returns_n2.min())
        # tlogger.record_tabular("EpRewStd", global_returns_n2.std())
        # tlogger.dump_tabular()

        print('>>>> saving to csv..')
        all_results = all_results.append({
            "Iteration" : curr_task_id,
            "EpRewMean" : np.nan if eval_job else global_returns_n2.mean(),
            # "EpRewStd" : global_returns_n2.std(),
            "EpRewMax": np.nan if eval_job else global_returns_n2.max(),
            # "EpRewMin": global_returns_n2.min(),
            # "EpLenMean" : global_lengths_n2.mean(),
            "Norm" : np.nan if eval_job else float(np.mean([np.square(agent.theta).sum() for agent in agents])),
            # "GradNorm" : np.nan if eval_job else float(np.square(np.mean(gs)).sum()),
            # "UpdateRatio" : np.nan if eval_job else float(np.mean(update_ratios)),
            # "UniqueWorkers" : num_unique_workers,
            "TimeElapsedThisIter" : step_tend - step_tstart,
            # "TimeElapsed" : step_tend - tstart,
            "CurrentTime": str(time.strftime("%Y-%m-%d")+'_'+time.strftime("%H:%M:%S")),
            "current_iteration_broadcast": np.nan if eval_job else current_iteration_broadcast,
            "best_agent": np.nan if eval_job else best_result.agent_id,
            "network_type": exp['agent_update']['args']['network_type'],
            "on_the_fly_topology": exp['agent_update']['on_the_fly_topology'],
            "same_param_start": exp['policy']['args']['same_param_start'],
            "strategy": exp['agent_update']['strategy'],
            "num_agents" : config.num_agents,
            "broadcast_prob": np.nan if eval_job else config.broadcast_prob,
            "network_parameter": exp['agent_update']['args']['network_args']['p'],
            # "search_diameter": search_diameter,
            # "grad_norm_mean": np.nan if eval_job else np.mean(current_iteration_gradNorms),
            # "grad_norm_std": np.nan if eval_job else np.std(current_iteration_gradNorms),
            "env": exp['env_id'],
            "EvalEpRewMean": np.nan if not eval_rets else np.mean(eval_rets),
            "EvalEpLenMean": np.nan if not eval_rets else np.mean(eval_lens),
            "EvalEpRewStd": np.nan if not eval_rets else np.std(eval_rets),
            # "EvalPopRank": np.nan if not eval_rets else (
            #                                 np.searchsorted(np.sort(returns_n2.ravel()): eval_rets).mean() / returns_n2.size)),
            "EvalEpCount": len(eval_rets),
            "eval_job": eval_job


        }, ignore_index=True)

        all_results.to_csv(summary_log)


def rollout_and_update_ob_stat(policy, env, timestep_limit, rs, task_ob_stat, calc_obstat_prob):
    if policy.needs_ob_stat and calc_obstat_prob != 0 and rs.rand() < calc_obstat_prob:
        rollout_rews, rollout_len, obs = policy.rollout(
            env, timestep_limit=timestep_limit, save_obs=True, random_stream=rs)
        task_ob_stat.increment(obs.sum(axis=0), np.square(obs).sum(axis=0), len(obs))
    else:
        rollout_rews, rollout_len = policy.rollout(env, timestep_limit=timestep_limit, random_stream=rs)
    return rollout_rews, rollout_len


def run_worker(master_redis_cfg, noise, *, min_task_runtime=.2):
    logger.info('run_worker: {}'.format(locals()))
    assert isinstance(noise, SharedNoiseTable)

    worker = WorkerClient(master_redis_cfg)
    exp = worker.get_experiment()
    config, env, session, policy = setup_worker(exp, single_threaded=True)
    num_agents = config.num_agents

    worker_id = None

    while True:
        task_id, task_data, agent_id = worker.get_current_task(num_agents)
        assert policy.needs_ob_stat == (config.calc_obstat_prob != 0)

        rs = np.random.RandomState()

        import random
        random_val = str(random.randint(0,9))
        seed = str(task_id)+random_val+str(agent_id)
        seed = int(seed)

        rs.seed(seed)
        np.random.seed(seed)

        if worker_id is None:
            worker_id = rs.randint(2 ** 31)

        #print("[worker] got task {} for agent {}".format(task_id, agent_id))
        task_tstart = time.time()
        assert isinstance(task_id, int) and isinstance(task_data, Task)
        if policy.needs_ob_stat:
            policy.set_ob_stat(task_data.ob_mean, task_data.ob_std)


        if task_data.eval_job == True:
        # if rs.rand() < config.eval_prob:
            # Evaluation: noiseless weights and noiseless actions
            print('>>>>> THIS IS AN EVAL JOB')
            print("[worker] evaluating task {} for agent {}".format(task_id, agent_id))
            policy.set_trainable_flat(task_data.best_theta)
            eval_rews, eval_length = policy.rollout(env)  # eval rollouts don't obey task_data.timestep_limit
            eval_return = eval_rews.sum()
            logger.info('Eval result: task={} return={:.3f} length={}'.format(task_id, eval_return, eval_length))
            worker.push_result(task_id, agent_id, Result(
                agent_id=agent_id,
                task_id=task_id,
                worker_id=worker_id,
                noise_inds_n=None,
                returns_n2=None,
                signreturns_n2=None,
                lengths_n2=None,
                eval_return=eval_return,
                eval_length=eval_length,
                ob_sum=None,
                ob_sumsq=None,
                ob_count=None
            ))
        else:
            # Rollouts with noise
            print("[worker] evaluating task {} for agent {}".format(task_id, agent_id))
            noise_inds, returns, signreturns, lengths = [], [], [], []
            task_ob_stat = RunningStat(env.observation_space.shape, eps=0.)  # eps=0 because we're incrementing only

            while not noise_inds or time.time() - task_tstart < min_task_runtime:
                noise_idx = noise.sample_index(rs, policy.num_params)
                v = config.noise_stdev * noise.get(noise_idx, policy.num_params)

                policy.set_trainable_flat(task_data.params + v)
                rews_pos, len_pos = rollout_and_update_ob_stat(
                    policy, env, task_data.timestep_limit, rs, task_ob_stat, config.calc_obstat_prob)

                policy.set_trainable_flat(task_data.params - v)
                rews_neg, len_neg = rollout_and_update_ob_stat(
                    policy, env, task_data.timestep_limit, rs, task_ob_stat, config.calc_obstat_prob)

                noise_inds.append(noise_idx)
                returns.append([rews_pos.sum(), rews_neg.sum()])
                signreturns.append([np.sign(rews_pos).sum(), np.sign(rews_neg).sum()])
                lengths.append([len_pos, len_neg])

            worker.push_result(task_id, agent_id, Result(
                agent_id=agent_id,
                task_id=task_id,
                worker_id=worker_id,
                noise_inds_n=np.array(noise_inds),
                returns_n2=np.array(returns, dtype=np.float32),
                signreturns_n2=np.array(signreturns, dtype=np.float32),
                lengths_n2=np.array(lengths, dtype=np.int32),
                eval_return=None,
                eval_length=None,
                ob_sum=None if task_ob_stat.count == 0 else task_ob_stat.sum,
                ob_sumsq=None if task_ob_stat.count == 0 else task_ob_stat.sumsq,
                ob_count=task_ob_stat.count
            ))
