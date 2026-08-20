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
from collections import Counter

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
    Starting from the embeddings of paper titles using SciBERT, this function builds ten different networks, using as threshold 
    distance five values in the range 0.8-0.9. Then, the number of connected components for each network is calculated, as well as
    the size of the largest component, and the results stored in a csv file
    "scibert_network/results/title_embeddings/connected_components.csv".
    '''
    connected_components = pd.DataFrame(columns = ['Threshold', 'Connected_components', 'Largest_component'])

    for threshold in np.linspace(start = 0.8, stop = 0.9, num = 5):
        adjacency_matrix = build_adjacency_matrix(threshold = threshold)
        abstract_network = nx.from_scipy_sparse_array(adjacency_matrix)
    
        connected_components.loc[len(connected_components)] = [threshold, nx.number_connected_components(abstract_network),
                                                           max([len(c) for c in list(nx.connected_components(abstract_network))])]
        print("Iteration")

    connected_components.to_csv("scibert_network/results/title_embeddings/connected_components.csv")


def centrality_measures(network: str):
    '''
    Calculate some centrality measures of each node in the network given as input.

    In particular, this function calculates the degree and clustering coefficient of each node, as well as the mean degree of the 
    neighbours of each node.
    
    All the measures are stored in the csv file "centrality_measures.csv", stored in the same folder of the adjacency matrix
    given as input.

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

    #calculate mean degree of nearest neighbours
    knn = list(nx.average_neighbor_degree(G).values())

    #merge everything in a dataset and save
    data = {"Degree" : degree_distribution, "Clustering_coefficient" : clustering_coefficients, "Mean_degree_NN" : knn}
    dataset = pd.DataFrame(data = data)

    #save dataset in the same folder of the adjacency matrix
    dataset.to_csv(network[::-1].split('/', 1)[1][::-1] + "/centrality_measures.csv")


def connectivity_between_cathegories():
    '''
    Create a 22 x 22 matrix, where each row and each column represent one of the 22 cathegories of physics papers, and the entries
    are the number of links in the SciBERT network between nodes belonging to the two cathegories. So the matrix is symmetric, and
    the diagonal elements are the links between papers of the same cathegory.

    The matrix is saved as a dataset in the csv file "scibert_network/results/connections_between_cathegories.csv", where also 
    a column representing the number of papers of each cathegory is included.
    '''
    # create the graph and the 22 x 22 matrix
    G = nx.from_scipy_sparse_array(load_npz("scibert_network/results/title_embeddings/title_scibert_adjacency_0_85.npz"))
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
    connections_between_cathegories.to_csv("scibert_network/results/connections_between_cathegories.csv")


def split_louvain_method():
    '''
    Split the network built from title SciBERT embeddings using the Louvain algorithm.

    First, the nodes that do not belong to the largest connected component are removed. Then, the Louvain method is applied as
    explained in the networkx documentation.

    The list with all subgraphs induced with this algorithm is saved in an external file.

    All subgraphs induced by the division of the network into communities are saved in the Python list 
    "scibert_network/results/abstract_embeddings/louvain_split".

    References
    ----------
        Networkx documentation louvain_communities: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html
    '''
    full_graph = nx.from_scipy_sparse_array(load_npz("scibert_network/results/title_embeddings/title_scibert_adjacency_0_85.npz"))

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

    communities_list = nx.community.louvain_communities(G, seed = 42)

    #the list obtained with the Louvain algorithm contains just the nodes of each subgraph. Now we need to create the
    #subgraphs starting from these nodes.
    subgraphs_list = []
    for community in communities_list:
        subgraphs_list.append(G.subgraph(community).copy())

    #save the list of subgraphs
    with open("scibert_network/results/title_embeddings/louvain_split", 'wb') as file:
        pickle.dump(subgraphs_list, file)


def cathegories_by_community():
    '''
    Create a table where the rows are the communities obtained using the Louvain algorithm, the columns are the cathegories 
    of Physics papers and each entry is the number of papers of a certain cathegory in a certain community.

    The table is saved in the csv file "scibert_network/results/cathegories_by_community_louvain.csv".
    '''
    cathegories_by_community = pd.DataFrame(columns = all_physics_topics)

    #load subgraphs
    with open("scibert_network/results/title_embeddings/louvain_split", "rb") as file:
        subgraphs_list = pickle.load(file)

    for community in subgraphs_list:
        topics_of_community = nx.get_node_attributes(community, "Topic")
        number_of_papers_per_topic = dict(Counter(list(topics_of_community.values())))
        list_number_of_papers_per_topic = []

        for cathegory in all_physics_topics:
            #count the number of nodes of each cathegory in the community
            if cathegory not in number_of_papers_per_topic.keys():
                list_number_of_papers_per_topic.append(0)
            else:
                list_number_of_papers_per_topic.append(number_of_papers_per_topic[cathegory])

        #put the number of papers of each topic for a community as a row in the dataset
        cathegories_by_community.loc[len(cathegories_by_community)] = list_number_of_papers_per_topic

    cathegories_by_community.to_csv("scibert_network/results/cathegories_by_community_louvain.csv")


if(__name__ == '__main__'):
    #sweep_connected_components()
    #centrality_measures("scibert_network/results/link_shuffle/link_shuffle.npz")
    #split_louvain_method()
    #connectivity_between_cathegories()
    cathegories_by_community()