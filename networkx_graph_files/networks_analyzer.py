    # make sure random networks are not disconnected (save network to file in es.py)
# also return average path length of network
# this should save networks to file
# for small world use connected_watts_strogatz_G(n, k, p, tries=100, seed=None) with p = 0.04 as per paper
# average_shortest_path_length(G, weight=None) gives


import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import itertools
import numpy as np
import pandas as pd

current_dir = os.getcwd()+"/pickles/"
# current_dir =  current_dir+"/data"
dirs = os.listdir( current_dir )
print(len(dirs))
counter = 0

for file in dirs:
    # if '10000' in file: continue
    if '.pickle' not in file: continue
    print('Network file being used: {}'.format(file))
    counter = counter + 1

    network_type = file[:-7] #removing the ".pickle" extension
    with open(current_dir+file, "rb") as input_file:
        adjacency_matrix = pickle.load(input_file)
    G = nx.Graph(adjacency_matrix)
    numpy_adjacency_matrix = nx.to_numpy_matrix(G)
    number_nodes = numpy_adjacency_matrix.shape[0]


    all_network_data = pd.DataFrame()


    ## node level only
    print('calculating degree..')
    degree_dict = dict()
    for node in G.degree():
        # degree_dict[node[0]] = node[1]
        degree_dict[node[0]] = node[1] - 1
    # print(degree_dict)

    current_df = pd.DataFrame()
    current_df = current_df.append(degree_dict, ignore_index=True)
    current_df = current_df.transpose()
    current_df.columns = ['degree']
    current_df['nodes'] = current_df.index.values
    all_network_data = current_df
    # print(current_df)

    #
    ## local level
    current_df = pd.DataFrame()
    print('calculating burt..')
    # print(nx.algorithms.structuralholes.constraint(G)) # dict of burt network constraints
    burt = nx.algorithms.structuralholes.constraint(G) # dict of burt network constraints
    current_df = current_df.append(burt, ignore_index=True)
    # print(current_df)
    current_df = current_df.transpose()
    current_df.columns = ['burt']
    current_df['nodes'] = current_df.index.values
    # print(current_df)

    all_network_data = pd.merge(
                                all_network_data,
                                current_df,
                                how='inner',
                                )



    # print(nx.clustering(G)) # dict of node clustering triangles
    current_df = pd.DataFrame()
    print('calculating clustering..')
    burt =  nx.clustering(G)
    current_df = current_df.append(burt, ignore_index=True)
    # print(current_df)
    current_df = current_df.transpose()
    current_df.columns = ['clustering']
    current_df['nodes'] = current_df.index.values
    # print(current_df)

    all_network_data = pd.merge(
                                all_network_data,
                                current_df,
                                how='inner',
                                )

    ## global level
    # print(nx.closeness_centrality(G)) # dict of node closeness_centrality
    current_df = pd.DataFrame()
    print('calculating closeness_centrality..')
    burt =  nx.closeness_centrality(G)
    current_df = current_df.append(burt, ignore_index=True)
    # print(current_df)
    current_df = current_df.transpose()
    current_df.columns = ['closeness_centrality']
    current_df['nodes'] = current_df.index.values
    # print(current_df)

    all_network_data = pd.merge(
                                all_network_data,
                                current_df,
                                how='inner',
                                )

    # print(nx.betweenness_centrality(G)) # dict of node betweenness centrality
    current_df = pd.DataFrame()
    print('calculating betweenness_centrality..')
    burt =  nx.betweenness_centrality(G)
    current_df = current_df.append(burt, ignore_index=True)
    # print(current_df)
    current_df = current_df.transpose()
    current_df.columns = ['betweenness_centrality']
    current_df['nodes'] = current_df.index.values
    # print(current_df)

    all_network_data = pd.merge(
                                all_network_data,
                                current_df,
                                how='inner',
                                )

    #not node level, purely global
    # print(nx.transitivity(G)) #Possible triangles are identified by the number of “triads” (two edges with a shared vertex).
    print('calculating transitivity..')
    all_network_data['transitivity'] = nx.transitivity(G)

    ## modularity
    import community
    print('calculating modularity..')
    part = community.best_partition(G)
    # print(community.modularity(part, G))
    all_network_data['modularity'] = community.modularity(part, G)

    G.remove_edges_from(nx.selfloop_edges(G))
    # print(nx.core_number(G))
    current_df = pd.DataFrame()
    print('calculating core_number..')
    burt =  nx.core_number(G)
    current_df = current_df.append(burt, ignore_index=True)
    current_df = current_df.transpose()
    current_df.columns = ['core_number']
    current_df['nodes'] = current_df.index.values
    # print(current_df)

    all_network_data = pd.merge(
                                all_network_data,
                                current_df,
                                how='inner',
                                )



    # nx.draw(G, pos=nx.spring_layout(G), with_labels=True)
    # plt.savefig("../Graph.png")


    print(all_network_data)
    all_network_data.to_csv('metrics/{}_constraint_only.csv'.format(file))
