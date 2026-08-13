#!/usr/bin/python
# -*- coding: utf-8 -*-

from build_networks import build_adjacency_matrix
import networkx as nx
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import pickle

__author__= ['Riccardo Grandicelli']
__email__= ['riccardograndicelli03@gmail.com']


all_physics_topics = ["Accelerator Physics", "Applied Physics", "Atmospheric and Oceanic Physics", "Atomic and Molecular Clusters", 
                      "Atomic Physics", "Biological Physics", "Chemical Physics", "Classical Physics", "Computational Physics", 
                      "Data Analysis, Statistics and Probability", "Fluid Dynamics", "General Physics", "Geophysics", 
                      "History and Philosophy of Physics", "Instrumentation and Detectors", "Medical Physics", "Optics",
                        "Physics and Society", "Physics Education", "Plasma Physics", "Popular Physics", "Space Physics"]
all_papers = pd.read_csv("data/all_papers.csv")


def sweep_connected_components():
    '''
    Starting from the embeddings of abstracts using tf-idf, this function builds ten different networks, using as threshold distance ten
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

    connected_components.to_csv("tf_idf_network/results/connected_components.csv")


def centrality_measures(network: str):
    '''
    Calculate some centrality measures of each node in the network given as input.

    In particular, this function calculates the degree, clustering coefficient and eigenvector centrality of each node, as well as
    the mean degree of the neighbours of each node.
    
    All the measures are stored in a csv file.

    Parameters
    ----------
        network: str
            The path to the file with the adjacency matrix of the network to be analyzed
    '''
    #load network
    adjacency_matrix = load_npz(network)
    #calculate degree by summing rows (or columns, it is symmetric) of adjacency matrix
    degree_distribution = adjacency_matrix.sum(axis = 0)

    #calculate clustering coefficient and eigenvector centrality using networkx
    G = nx.from_scipy_sparse_array(adjacency_matrix)
    clustering_coefficients = list(nx.clustering(G).values())
    eigenvector_centrality = list(nx.eigenvector_centrality(G).values())

    #calculate mean degree of nearest neighbours
    knn = list(nx.average_neighbor_degree(G).values())

    #merge everything in a dataset and save
    data = {"Degree" : degree_distribution, "Clustering_coefficient" : clustering_coefficients, 
            "Eigenvector_centrality" : eigenvector_centrality, "Mean_degree_NN" : knn}
    dataset = pd.DataFrame(data = data)

    #save dataset in the same folder of the adjacency matrix
    dataset.to_csv(network[::-1].split('/', 1)[1][::-1] + "/centrality_measures.csv")


def connectivity_between_cathegories():
    '''
    Create a 22 x 22 matrix, where each row and each column represent one of the 22 cathegories of physics papers, and the entries
    are the number of links in the network between nodes belonging to the two cathegories. So the matrix is symmetric, and the
    diagonal elements are the links between papers of the same cathegory.

    The matrix is saved as a dataset in an external csv file, together with a column representing the number of papers of each
    cathegory.
    '''
    # create the graph and the 22 x 22 matrix
    G = nx.from_scipy_sparse_array(load_npz("tf_idf_network/results/abstract_embeddings/abstract_tfidf_adjacency_0_2.npz"))
    connectivity_matrix = np.zeros(shape = (22, 22), dtype = np.int32)

    #create list of topics and set them as node attributes
    topics_list = [all_papers['primary_cathegory'][i] if all_papers['primary_cathegory'][i] in all_physics_topics 
               else all_papers['secondary_cathegory'][i] for i in range(25877)]
    topics_dictionary = dict(zip(list(range(25877)), topics_list))
    nx.set_node_attributes(G, topics_dictionary, name = "Topic")

    #create a list for the number of papers for each cathegory
    papers_for_cathegory = []

    for i, first_topic in enumerate(all_physics_topics):
        papers_for_cathegory.append(len([node for node in list(G.nodes) if G.nodes[node]["Topic"] == first_topic]))

        for j, second_topic in enumerate(all_physics_topics):
            edges = [edge for edge in list(G.edges) if G.nodes[edge[0]]["Topic"] == first_topic and 
                   G.nodes[edge[1]]["Topic"] == second_topic]
            connectivity_matrix[i][j] = len(edges)

    connections_between_cathegories = pd.DataFrame(data = connectivity_matrix, index = all_physics_topics, columns = all_physics_topics)
    connections_between_cathegories.insert(loc = len(connections_between_cathegories), column = "Number_of_papers",
                                           value = papers_for_cathegory)
    connections_between_cathegories.to_csv("tf_idf_network/results/connections_between_cathegories.csv")


def split_fiedler_eigenvector():
    '''
    Divide the abstract embeddings graph into communities using the sign of the components of the Fiedler eigenvector.

    During the first iteration, the nodes with degree 0 are removed from the graph. Then the graph is divided into two parts following
    the signs of the Fiedler eigenvector, and the two induced subgraphs are built from this division.

    During the following iterations, the induced subgraph with the higher number of nodes is divided into two more parts following 
    again the signs of the Fiedler eigenvector of that graph. This procedure is repeated until we have a total of 22 subgraphs (same
    number as the topics of physics papers).

    The list of all subgraphs is saved in an external file.
    '''
    full_graph = nx.from_scipy_sparse_array(load_npz("tf_idf_network/results/abstract_embeddings/abstract_tfidf_adjacency_0_2.npz"))

    #create list of topics and set them as node attributes
    topics_list = [all_papers['primary_cathegory'][i] if all_papers['primary_cathegory'][i] in all_physics_topics 
               else all_papers['secondary_cathegory'][i] for i in range(25877)]
    topics_dictionary = dict(zip(list(range(25877)), topics_list))
    nx.set_node_attributes(full_graph, topics_dictionary, name = "Topic")

    #create list of titles and set them as node attributes
    titles_list = [all_papers['title'][i] for i in range(25877)]
    titles_dictionary = dict(zip(list(range(25877)), titles_list))
    nx.set_node_attributes(full_graph, titles_dictionary, name = "Title")

    #keep only the largest component and remove the other nodes
    largest_component = max(nx.connected_components(full_graph), key=len)
    G = full_graph.subgraph(largest_component).copy()

    subgraphs_list = []
    G_max = G
    for _ in range(21): #after N iterations we have N+1 subgraphs
        #create fiedler eigenvector
        laplacian_matrix = nx.laplacian_matrix(G_max)
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

    #save the list of subgraphs
    with open("tf_idf_network/results/abstract_embeddings/fiedler_split", 'wb') as file:
        pickle.dump(subgraphs_list, file)


def split_k_means():
    '''
    Split the network built from abstract tf-idf embeddings into clusters using K-Means clustering.

    First, the nodes that do not belong to the largest connected component are removed. Then, the first 30 eigenvectors of the
    Laplacian matrix of the remaining network are calculated and their components used as features for network nodes (30 features for
    node). After that, a K-Means clustering algorithm (with K = 22) is used to divide the network into 22 clusters.

    The list of all 22 subgraphs induced with this algorithm is saved in an external file.
    '''
    full_graph = nx.from_scipy_sparse_array(load_npz("tf_idf_network/results/abstract_embeddings/abstract_tfidf_adjacency_0_2.npz"))

    #create list of topics and set them as node attributes
    topics_list = [all_papers['primary_cathegory'][i] if all_papers['primary_cathegory'][i] in all_physics_topics 
                    else all_papers['secondary_cathegory'][i] for i in range(25877)]
    topics_dictionary = dict(zip(list(range(25877)), topics_list))
    nx.set_node_attributes(full_graph, topics_dictionary, name = "Topic")

    #create list of titles and set them as node attributes
    titles_list = [all_papers['title'][i] for i in range(25877)]
    titles_dictionary = dict(zip(list(range(25877)), titles_list))
    nx.set_node_attributes(full_graph, titles_dictionary, name = "Title")

    #keep only the largest component and remove the other nodes
    largest_component = max(nx.connected_components(full_graph), key=len)
    G = full_graph.subgraph(largest_component).copy()

    #create laplacian matrix and calculate eigenvalues and eigenvectors
    laplacian_matrix = nx.laplacian_matrix(G)
    _, evecs = eigsh(laplacian_matrix, k = 30, which ='SM', tol = 1e-6)

    kmeans = KMeans(n_clusters = 22, random_state = 42)
    kmeans.fit(evecs)

    #create the list of subgraphs
    subgraphs_list = []
    for i in range(22):
        subgraph_nodes = [node for index, node in enumerate(list(G.nodes)) if kmeans.labels_[index] == i]
        subgraph = G.subgraph(subgraph_nodes).copy()
        subgraphs_list.append(subgraph)

    #save the list of subgraphs
    with open("tf_idf_network/results/abstract_embeddings/k_means_split", 'wb') as file:
        pickle.dump(subgraphs_list, file)


def split_louvain_method():
    '''
    Split the network built from abstract tf-idf embeddings using the Louvain method.

    First, the nodes that do not belong to the largest connected component are removed. Then, the Luovain method is applied as
    explained in the networkx documentation.

    The list with all subgraphs induced with this algorithm is saved in an external file.

    References
    ----------
        Networkx documentation louvain_communities: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html
    '''
    full_graph = nx.from_scipy_sparse_array(load_npz("tf_idf_network/results/abstract_embeddings/abstract_tfidf_adjacency_0_2.npz"))

    #create list of topics and set them as node attributes
    topics_list = [all_papers['primary_cathegory'][i] if all_papers['primary_cathegory'][i] in all_physics_topics 
                    else all_papers['secondary_cathegory'][i] for i in range(25877)]
    topics_dictionary = dict(zip(list(range(25877)), topics_list))
    nx.set_node_attributes(full_graph, topics_dictionary, name = "Topic")

    #create list of titles and set them as node attributes
    titles_list = [all_papers['title'][i] for i in range(25877)]
    titles_dictionary = dict(zip(list(range(25877)), titles_list))
    nx.set_node_attributes(full_graph, titles_dictionary, name = "Title")

    #keep only the largest component and remove the other nodes
    largest_component = max(nx.connected_components(full_graph), key=len)
    G = full_graph.subgraph(largest_component).copy()

    subgraphs_list = nx.community.louvain_communities(G, seed = 42)

    #save the list of subgraphs
    with open("tf_idf_network/results/abstract_embeddings/louvain_split", 'wb') as file:
        pickle.dump(subgraphs_list, file)   


if(__name__ == '__main__'):
    centrality_measures("networks/results/link_shuffle/link_shuffle.npz")
    #split_fiedler_eigenvector()
    #split_k_means()
    #split_louvain_method()