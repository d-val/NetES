import numpy as np
from collections import namedtuple
from copy import deepcopy
# from . import networks
# import networkx
import pickle

Agent = namedtuple('Agent', ['optimizer', 'theta'])

# def load_adjacency_matrix_file(network_type, num_agents):
#     """Loads an adjacency matrix from a pre-determined filename.
#
#     :param fpath: Filepath of matrix
#     :returns: A numpy array that represents an adjacency matrix
#     :rtype: np.Array
#
#     """
#     # TODO: placeholder for now, load from "networks" directory in the future.
#     return np.ones((20000,20000))

# def generate_adjacency_matrix(network_type, num_agents, **kwargs):
#     # TODO: actually generate a network here.
#     return networks.create_graph(network_type, num_agents, **kwargs)

def compute_ranks(x): #this just returns the sorted ranks of an array e.g. compute_ranks( np.array([1,2,5,0,3,1,9]) ) returns [1 3 5 0 4 2 6]
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks

def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y

def median_normalized(x):
    y = x.ravel()
    output = (y - np.median(y))/(np.max(y) - np.min(y))
    return output.reshape(x.shape).astype(np.float32)

def median_simple(x):
    y = x.ravel()
    output = (y - np.median(y))
    return output.reshape(x.shape).astype(np.float32)


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)

def batched_weighted_sum(weights, vecs, batch_size):
    total = 0.
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float32),
                        np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed

class AgentUpdater:
    """Updates Agents parameters... ?!?!?!?"""
    def __init__(self, *args, **kwargs):

        # self.network_source = kwargs.pop('network_source')
        self.network_type = kwargs.pop('network_type')
        self.network_file = kwargs.pop('network_file')
        self.num_agents = kwargs.pop('num_agents')
        self.network_args = kwargs.pop('network_args')
        self.return_proc_mode = kwargs.pop('return_proc_mode')

        self.adjacency_matrix = kwargs.pop('adjacency_matrix')

        print(args, kwargs)
        #
        # with open(network_file, "rb") as input_file:
        #     self.adjacency_matrix = pickle.load(input_file)
        # if self.network_source == 'file':
        #     self.adjacency_matrix = load_adjacency_matrix_file(self.network_type, self.num_agents)
        # elif self.network_source == 'generated':
        #     self.adjacency_matrix = generate_adjacency_matrix(self.network_type,
        #                                                       self.num_agents,
        #                                                       **self.network_args)


        # else:
        #     raise NotImplementedError("'network_source' must be one of 'generated' or 'file'.")

        self._initialize(*args, **kwargs)

    def _initialize(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, agent_id):
        raise NotImplementedError


class CopyUpdater(AgentUpdater):
    def _initialize(self, noise_block, num_params, l2coeff, noise_stdev, comms_noise_stdev):
        self.comms_noise_std = comms_noise_stdev
        self.noise_stdev = noise_stdev
        self.noise_block = noise_block
        self.num_params = num_params
        self.l2coeff = l2coeff



    def update(self, agent_id, agents, task_results):
        assert (self.adjacency_matrix.shape[0] >= len(agents) and
                self.adjacency_matrix.shape[1] >= len(agents))

        agent = deepcopy(agents[agent_id])
        assert not agent is agents[agent_id]

        adjacency_vec = self.adjacency_matrix[:, agent_id]
        this_agent_results = list(filter(lambda result: adjacency_vec[result.agent_id] == 1,
                                         task_results))

        theta = agent.optimizer.theta

        best_result = max(this_agent_results, key=lambda r: np.max(r.returns_n2))
        best_theta = agents[best_result.agent_id].optimizer.theta
        antithetic_idx = np.argmax([max(best_result.returns_n2[:,0]),
                                    max(best_result.returns_n2[:,1])])
        noise_idx = np.argmax(np.apply_along_axis(np.max, 1, best_result.returns_n2))

        v = self.noise_stdev * self.noise_block.get(best_result.noise_inds_n[noise_idx], self.num_params)

        if antithetic_idx == 0:
            new_theta = best_theta + v
        else:
            new_theta = best_theta - v

        agent.optimizer.theta = new_theta

        new_agent = Agent(agent.optimizer, agent.optimizer.theta)

        update_ratio = np.linalg.norm(new_theta - theta) / np.linalg.norm(theta)

        return agent_id, new_agent, v, update_ratio


class Neighborhood(AgentUpdater):
    def _initialize(self, noise_block, num_params, l2coeff, noise_stdev, comms_noise_stdev):
        self.comms_noise_std = comms_noise_stdev
        self.noise_stdev = noise_stdev
        self.noise_block = noise_block
        self.num_params = num_params
        self.l2coeff = l2coeff

    def update(self, agent_id, agents, task_results):
        agent = deepcopy(agents[agent_id])
        adjacency_vec = self.adjacency_matrix[:, agent_id]

        this_agent_results = list(filter(lambda result: adjacency_vec[result.agent_id] == 1,
                                         task_results))

        print("Agent {} using {} / {} other agents to perform \
        update.".format(agent_id, len(this_agent_results),
                        len(task_results)))

        theta = agent.optimizer.theta

        noise_inds_n = np.concatenate([r.noise_inds_n for r in this_agent_results])
        returns_n2 = np.concatenate([r.returns_n2 for r in this_agent_results])
        lengths_n2 = np.concatenate([r.lengths_n2 for r in this_agent_results])
        agent_ids_n = np.concatenate([[r.agent_id] * len(r.returns_n2) for r in this_agent_results])
        thetas = np.concatenate([agents[r.agent_id].optimizer.theta
                                 for r in this_agent_results])

        assert noise_inds_n.shape[0] == returns_n2.shape[0] == lengths_n2.shape[0]

        # Process returns
        if self.return_proc_mode == 'centered_rank':
            proc_returns_n2 = compute_centered_ranks(returns_n2)

        elif self.return_proc_mode == 'median_normalized':
            proc_returns_n2 = median_normalized(returns_n2)

        elif self.return_proc_mode == 'median_simple':
            proc_returns_n2 = median_simple(returns_n2)

        elif self.return_proc_mode == 'sign':
            proc_returns_n2 = np.concatenate([r.signreturns_n2 for r in this_agent_results])
        elif self.return_proc_mode == 'centered_sign_rank':
            proc_returns_n2 = compute_centered_ranks(np.concatenate([r.signreturns_n2 for r in this_agent_results]))
        else:
            raise NotImplementedError(self.return_proc_mode)

        # vecs = []
        # for n_idx, r in zip(noise_inds_n, returns_n2):
        #     noise = self.noise_block.get(n_idx, self.num_params)
        #     theta_vec = agents[r.agent_id].optimizer.theta - theta
        #     vecs.append(noise + theta_vec)

        noise_vecs = list((self.noise_block.get(idx, self.num_params) for idx in noise_inds_n))
        theta_vecs = list((agents[agent_id].optimizer.theta - theta for agent_id in agent_ids_n))

        delta_theta, count = batched_weighted_sum(
            proc_returns_n2[:, 0] + proc_returns_n2[:, 1],
            theta_vecs,
            batch_size=500
        )

        delta_noise, count = batched_weighted_sum(
            proc_returns_n2[:, 0] - proc_returns_n2[:, 1],
            noise_vecs,
            batch_size=500
        )

        g = (delta_noise + delta_theta) / returns_n2.size

        assert g.shape == (self.num_params,) and g.dtype == np.float32 and count == len(noise_inds_n)
        update_ratio = agent.optimizer.update(-g + self.l2coeff * agent.optimizer.theta)
        assert not all(np.isclose(theta,
                                  agent.optimizer.theta))

        new_agent = Agent(agent.optimizer, agent.optimizer.theta)
        return agent_id, new_agent, g, update_ratio


class ESUpdater(AgentUpdater):
    def _initialize(self, noise_block, num_params, l2coeff, noise_stdev, comms_noise_stdev):
        self.comms_noise_std = comms_noise_stdev
        self.noise_stdev = noise_stdev
        self.noise_block = noise_block
        self.num_params = num_params
        self.l2coeff = l2coeff
        self.adjacency_matrix = np.ones(self.adjacency_matrix.shape)

    def update(self, agent_id, agents, task_results):
        # make sure its fully-connected
        agent = deepcopy(agents[agent_id])
        adjacency_vec = self.adjacency_matrix[:, agent_id]

        this_agent_results = list(filter(lambda result: adjacency_vec[result.agent_id] == 1,
                                         task_results))

        print("Agent {} using {} / {} other agents to perform \
        update.".format(agent_id, len(this_agent_results),
                        len(task_results)))

        theta = agent.optimizer.theta

        noise_inds_n = np.concatenate([r.noise_inds_n for r in this_agent_results])
        returns_n2 = np.concatenate([r.returns_n2 for r in this_agent_results])
        lengths_n2 = np.concatenate([r.lengths_n2 for r in this_agent_results])

        assert noise_inds_n.shape[0] == returns_n2.shape[0] == lengths_n2.shape[0]

             # Process returns
        if self.return_proc_mode == 'centered_rank':
            proc_returns_n2 = compute_centered_ranks(returns_n2)
        elif self.return_proc_mode == 'sign':
            proc_returns_n2 = np.concatenate([r.signreturns_n2 for r in this_agent_results])
        elif self.return_proc_mode == 'centered_sign_rank':
            proc_returns_n2 = compute_centered_ranks(np.concatenate([r.signreturns_n2 for r in this_agent_results]))
        else:
            raise NotImplementedError(self.return_proc_mode)

        g, count = batched_weighted_sum(
            proc_returns_n2[:, 0] - proc_returns_n2[:, 1],
            (self.noise_block.get(idx, self.num_params) for idx in noise_inds_n),
            batch_size=500
        )


        g /= returns_n2.size
        assert g.shape == (self.num_params,) and g.dtype == np.float32 and count == len(noise_inds_n)
        update_ratio = agent.optimizer.update(-g + self.l2coeff * agent.optimizer.theta)
        assert not all(np.isclose(theta,
                                  agent.optimizer.theta))

        new_agent = Agent(agent.optimizer, agent.optimizer.theta)
        return agent_id, new_agent, g, update_ratio
