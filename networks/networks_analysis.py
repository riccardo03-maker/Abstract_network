#!/usr/bin/python
# -*- coding: utf-8 -*-

from build_networks import build_adjacency_matrix
import networkx as nx
import pandas as pd
import numpy as np

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


def sweep_connected_components():
    '''
    Starting from the embeddings of abstracts using tf-idf, this function builds 20 different networks, using as threshold distance twenty
    values in the range 0.0001-0.01. Then, the number of connected components for each network is calculated, as well as the size of
    the largest component, and the results stored in a csv file.
    '''
    connected_components = pd.DataFrame(columns = ['Threshold', 'Connected_components', 'Largest_component'])

    for threshold in np.linspace(start = 0.0001, stop = 0.01, num = 20):
        adjacency_matrix = build_adjacency_matrix(threshold = threshold)
        abstract_network = nx.from_scipy_sparse_array(adjacency_matrix)
    
        connected_components.loc[len(connected_components)] = [threshold, nx.number_connected_components(abstract_network),
                                                           max([len(c) for c in list(nx.connected_components(abstract_network))])]
        print("Iteration")

    connected_components.to_csv("networks/results/connected_components.csv")


if(__name__ == '__main__'):
    sweep_connected_components()