#!/usr/bin/python
# -*- coding: utf-8 -*-

from build_networks import build_adjacency_matrix
import networkx as nx
import pandas as pd
import numpy as np
from scipy.sparse import load_npz

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


all_physics_topics = ["Accelerator Physics", "Applied Physics", "Atmospheric and Oceanic Physics", "Atomic and Molecular Clusters", 
                      "Atomic Physics", "Biological Physics", "Chemical Physics", "Classical Physics", "Computational Physics", 
                      "Data Analysis, Statistics and Probability", "Fluid Dynamics", "General Physics", "Geophysics", 
                      "History and Philosophy of Physics", "Instrumentation and Detectors", "Medical Physics", "Optics",
                        "Physics and Society", "Physics Education", "Plasma Physics", "Popular Physics", "Space Physics"]
all_papers = pd.read_csv("data/all_papers.csv")


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


def degree_distribution():
    '''
    Calculate the degree of each node in the network.

    All the degrees are stored in a csv file, together with the topic of the paper (the primary cathegory if this is one of the "Physics"
    topics, otherwise the secondary cathegory).
    '''
    #load network
    adjacency_matrix = load_npz("networks/abstract_tfidf_adjacency_0_01.npz")
    #calculate degree by summing rows (or columns, it is symmetric) of adjacency matrix
    degree_distribution = adjacency_matrix.sum(axis = 0)

    #create list of topics
    topics_list = [all_papers['primary_cathegory'][i] if all_papers['primary_cathegory'][i] in all_physics_topics 
               else all_papers['secondary_cathegory'][i] for i in range(25877)]

    #merge everything in a dataset and save
    data = {"Degree" : degree_distribution, "Topic" : topics_list}
    dataset = pd.DataFrame(data = data)
    dataset.to_csv("networks/results/degree_distribution.csv")


if(__name__ == '__main__'):
    degree_distribution()