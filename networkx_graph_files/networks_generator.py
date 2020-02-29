# make sure random networks are not disconnected (save network to file in es.py)
# also return average path length of network
# this should save networks to file
# for small world use connected_watts_strogatz_graph(n, k, p, tries=100, seed=None) with p = 0.04 as per paper
# average_shortest_path_length(G, weight=None) gives


import sys
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import itertools
import numpy as np

def save_endos_renyi_connected(n, p):
    graph = nx.erdos_renyi_graph(n,p) # where p is the probability of any edge being created
    counter = 0
    while nx.is_connected(graph) == False: #making sure we only use fully connected graphs
        graph = nx.erdos_renyi_graph(n,p)
        print("iter {}: trying to find a connected graph for erdos n={}, p={}..".format(counter, n, p))
        counter = counter + 1
    adj = nx.adjacency_matrix(graph)
    adj[range(n), range(n)] = 1 #adding self-loops to the adj matrix

    adjacency_matrix_file = '../networkx_graph_files/erdos_n_{}_p_{}.pickle'.format(n, p)
    with open(adjacency_matrix_file, 'wb') as handle:
        pickle.dump(adj, handle, -1)
    # return adj

def save_fully_connected(n):
    adjacency_matrix_file = '../networkx_graph_files/fully_{}.pickle'.format(n)
    adj = np.ones((n, n))
    with open(adjacency_matrix_file, 'wb') as handle:
        pickle.dump(adj, handle, -1)

def save_small_world(n, k, save=False):
    seed=89123891203
    # graph = nx.connected_watts_strogatz_graph(n, k, p, tries=100, seed=89123891203)
    G = nx.random_regular_graph(d = k, n = n, seed=seed)
    assert nx.is_connected(G) == True
    assert np.std(G.degree().values()) == 0.0
    assert np.mean(G.degree().values()) == k
    # graph = nx.connected_watts_strogatz_graph(n, k, p, tries=100)
    adj = nx.adjacency_matrix(graph)
    adj[range(n), range(n)] = 1 #adding self-loops to the adj matrix
    print('avg path length: ', nx.average_shortest_path_length(graph))
    # print('density: ', nx.density(graph))
    if save:
        average_shortest_path_length = nx.average_shortest_path_length(graph)
        adjacency_matrix_file = '../networkx_graph_files/smallworld_n_{}_pathL_{}.pickle'.format(n, average_shortest_path_length)
        with open(adjacency_matrix_file, 'wb') as handle:
            pickle.dump(adj, handle, -1)

def save_linear_network(n, save=False):
    graph = nx.Graph()
    graph.add_node(0)
    for i in range(1, n):
        graph.add_node(i)
        graph.add_edge(i-1,i)
    adj = nx.adjacency_matrix(graph)
    adj[range(n), range(n)] = 1 #adding self-loops to the adj matrix
    # print('avg path length: ', nx.average_shortest_path_length(graph))
    # print('density: ', nx.density(graph))
    if save:
        # average_shortest_path_length = nx.average_shortest_path_length(graph)
        adjacency_matrix_file = '../networkx_graph_files/linear_n_{}.pickle'.format(n)
        with open(adjacency_matrix_file, 'wb') as handle:
            pickle.dump(adj, handle, -1)

def save_power_law(n, exponent, save=False):
    sequence = nx.random_powerlaw_tree_sequence(n = n, gamma = exponent, tries=50000)
    graph = nx.configuration_model(sequence)
    print(sequence)
    adj = nx.adjacency_matrix(graph)
    adj[range(n), range(n)] = 1 #adding self-loops to the adj matrix
    # print('avg path length: ', nx.average_shortest_path_length(graph))
    print('density: ', nx.density(graph))
    if save:
        # average_shortest_path_length = nx.average_shortest_path_length(graph)
        print('saving...')
        adjacency_matrix_file = 'power_law_n_{}_exp_{}.pickle'.format(n, exponent)
        with open(adjacency_matrix_file, 'wb') as handle:
            pickle.dump(adj, handle, -1)



def save_self_loop(n, save=False):
    adj = np.zeros((n, n))
    adj[range(n), range(n)] = 1 #adding self-loops to the adj matrix
    # print(adj)
    # print('avg path length: ', nx.average_shortest_path_length(graph))
    if save:
        # average_shortest_path_length = nx.average_shortest_path_length(graph)
        print('saving...')
        adjacency_matrix_file = 'self_loop_{}.pickle'.format(n)
        with open(adjacency_matrix_file, 'wb') as handle:
            pickle.dump(adj, handle, -1)

# save_self_loop(10)
save_self_loop(100, save = True)

def save_barabasi_albert_graph(n, m, seed, save=False):
    # sequence = nx.random_powerlaw_tree_sequence(n = n, gamma = exponent, tries=50000)
    # graph = nx.configuration_model(sequence)
    graph = nx.barabasi_albert_graph(n, m, seed)
    if nx.is_connected(graph):
        print('graph is connected')
        # print(sequence)
        adj = nx.adjacency_matrix(graph)
        adj[range(n), range(n)] = 1 #adding self-loops to the adj matrix
        # print('avg path length: ', nx.average_shortest_path_length(graph))
        # print('density: ', nx.density(graph))
        if save:
            # average_shortest_path_length = nx.average_shortest_path_length(graph)
            print('saving...')
            adjacency_matrix_file = 'scale_free_{}_m{}_seed{}.pickle'.format(n, m, seed)
            with open(adjacency_matrix_file, 'wb') as handle:
                pickle.dump(adj, handle, -1)
