#!/usr/bin/python
# -*- coding: utf-8 -*-

from build_networks import build_adjacency_matrix
import networkx as nx
import pandas as pd

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


def sweep_connected_components():
    '''
    Starting from the embeddings of abstracts using tf-idf, this function builds 10 different networks, using as threshold distance ten
    values in the range 0.001-0.01. Then, the number of connected components for each network are calculated, and the results stored
    in a csv file.

    Since we have 14986 connected components with the threshold 0.001, while a fully connected graph with the threshold 0.01, the aim
    of this function is to find the optimal value of the threshold to have a number of connected components as close as possible to the
    number of different topics in arxiv physics papers (22 topics).
    '''
    connected_components = pd.DataFrame(columns = ['Threshold', 'Connected_components'])

    for threshold in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]:
        adjacency_matrix = build_adjacency_matrix(threshold = threshold)
        tf_idf_graph = nx.from_scipy_sparse_array(adjacency_matrix)
        connected_components.loc[len(connected_components)] = [threshold, nx.number_connected_components(tf_idf_graph)]
        print("Iteration")

    connected_components.to_csv("networks/results/connected_components.csv")

if(__name__ == '__main__'):
    sweep_connected_components()