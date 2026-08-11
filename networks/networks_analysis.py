#!/usr/bin/python
# -*- coding: utf-8 -*-

from .build_networks import build_adjacency_matrix
import networkx as nx
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import eigsh

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
    Starting from the embeddings of abstracts using tf-idf, this function builds 20 different networks, using as threshold distance ten
    values in the range 0.15-0.5. Then, the number of connected components for each network is calculated, as well as the size of
    the largest component, and the results stored in a csv file.
    '''
    connected_components = pd.DataFrame(columns = ['Threshold', 'Connected_components', 'Largest_component'])

    for threshold in np.linspace(start = 0.15, stop = 0.5, num = 10):
        adjacency_matrix = build_adjacency_matrix(threshold = threshold)
        abstract_network = nx.from_scipy_sparse_array(adjacency_matrix)
    
        connected_components.loc[len(connected_components)] = [threshold, nx.number_connected_components(abstract_network),
                                                           max([len(c) for c in list(nx.connected_components(abstract_network))])]
        print("Iteration")

    connected_components.to_csv("networks/results/connected_components.csv")


def centrality_measures(network: str):
    '''
    Calculate some centrality measures of each node in the network given as input.

    In particular, this function calculates the degree, clustering coefficient and eigenvector centrality of each node, as well as
    the mean degree of the neighbours of each node.
    
    All the measures are stored in a csv file, together with the topic of the paper (the primary cathegory if this is one of the "Physics"
    topics, otherwise the secondary cathegory).

    Parameters
    ----------
        network: str
            The path to the file with the adjacency matrix of the network to be analyzed
    '''
    #load network
    adjacency_matrix = load_npz(network)
    #calculate degree by summing rows (or columns, it is symmetric) of adjacency matrix
    degree_distribution = adjacency_matrix.sum(axis = 0)

    #create list of topics
    #this has no meaning if the network is built from a null model
    topics_list = [all_papers['primary_cathegory'][i] if all_papers['primary_cathegory'][i] in all_physics_topics 
               else all_papers['secondary_cathegory'][i] for i in range(25877)]

    #calculate clustering coefficient and eigenvector centrality using networkx
    G = nx.from_scipy_sparse_array(adjacency_matrix)
    clustering_coefficients = list(nx.clustering(G).values())
    eigenvector_centrality = list(nx.eigenvector_centrality(G).values())

    #calculate mean degree of nearest neighbours
    knn = list(nx.average_neighbor_degree(G).values())

    #merge everything in a dataset and save
    data = {"Degree" : degree_distribution, "Clustering_coefficient" : clustering_coefficients, 
            "Eigenvector_centrality" : eigenvector_centrality, "Mean_degree_NN" : knn, "Topic" : topics_list}
    dataset = pd.DataFrame(data = data)

    #save dataset in the same folder of the adjacency matrix
    dataset.to_csv(network[::-1].split('/', 1)[1][::-1] + "/centrality_measures.csv")


def split_fiedler_eigenvector():
    '''
    Divide the abstract embeddings graph into communities using the sign of the components of the Fiedler eigenvector.

    During the first iteration, the nodes with degree 0 are removed from the graph. Then the graph is divided into two parts following
    the signs of the Fiedler eigenvector, and the two induced subgraphs are built from this division.

    During the following iterations, the induced subgraph with the higher number of nodes is divided into two more parts following 
    again the signs of the Fiedler eigenvector of that graph. This procedure is repeated until we have a total of 22 subgraphs (same
    number as the topics of physics papers).

    Returns
    -------
        subgraphs_list: list of nx.Graph
            List of subgraphs obtained through Fiedler eigenvector division.
    '''
    full_graph = nx.from_scipy_sparse_array(load_npz("networks/results/abstract_embeddings/abstract_tfidf_adjacency_0_2.npz"))

    #create list of topics and set them as node attributes
    topics_list = [all_papers['primary_cathegory'][i] if all_papers['primary_cathegory'][i] in all_physics_topics 
               else all_papers['secondary_cathegory'][i] for i in range(25877)]
    topics_dictionary = dict(zip(list(range(25877)), topics_list))
    nx.set_node_attributes(full_graph, topics_dictionary, name = "Topic")

    #keep only the largest component and remove the other nodes
    largest_component = max(nx.connected_components(full_graph), key=len)
    G = full_graph.subgraph(largest_component).copy()

    subgraphs_list = []
    G_max = G
    for _ in range(21): #after N iterations we have N+1 subgraphs
        #create fiedler eigenvector
        laplacian_matrix = nx.laplacian_matrix(G_max).tocsr()
        _, evecs = eigsh(laplacian_matrix, k = 2, which ='SM', tol = 1e-6)
        fiedler_vector = evecs[:, 1]

        #create subgraphs following the sign of Fiedler eigenvector
        subgraph_1 = [node for index, node in enumerate(list(G_max.nodes)) if fiedler_vector[index] > 0]
        subgraph_2 = [node for index, node in enumerate(list(G_max.nodes)) if fiedler_vector[index] < 0]

        #append the smallest subgraph to the list of subgraphs and keep the other one for the next iteration
        if len(subgraph_1) >= len(subgraph_2):
            G_min = G_max.subgraph(subgraph_2).copy()
            G_max = G_max.subgraph(subgraph_1)
        else:
            G_min = G_max.subgraph(subgraph_1).copy()
            G_max = G_max.subgraph(subgraph_2)
        subgraphs_list.append(G_min)

        print("Iteration")

    #append the remaining subgraph to the list of subgraphs
    subgraphs_list.append(G_max)
    return subgraphs_list


if(__name__ == '__main__'):
    #centrality_measures("networks/results/scale_free_network/scale_free_network.npz")
    split_fiedler_eigenvector()