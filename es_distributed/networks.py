import networkx as nx
import numpy as np

def erdos_renyi_connected(n, p):
    graph = nx.erdos_renyi_graph(n,p) # where p is the probability of any edge being created
    counter = 0
    while nx.is_connected(graph) == False: #making sure we only use fully connected graphs
        graph = nx.erdos_renyi_graph(n,p)
        print("iter {}: trying to find a connected graph for erdos n={}, p={}..".format(counter, n, p))
        counter = counter + 1
    adj = nx.adjacency_matrix(graph)
    adj[range(n), range(n)] = 1 #adding self-loops to the adj matrix
    return adj


def fully_connected(n):
    adj = np.ones((n, n))
    return adj
