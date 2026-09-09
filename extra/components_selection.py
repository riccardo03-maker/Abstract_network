#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_array, lil_array, load_npz
import pandas as pd
import networkx as nx
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import pickle
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

__author__= ['Riccardo Grandicelli']
__email__= ['riccardograndicelli03@gmail.com']


all_physics_topics = ["Accelerator Physics", "Applied Physics", "Atmospheric and Oceanic Physics", "Atomic and Molecular Clusters", 
                      "Atomic Physics", "Biological Physics", "Chemical Physics", "Classical Physics", "Computational Physics", 
                      "Data Analysis, Statistics and Probability", "Fluid Dynamics", "General Physics", "Geophysics", 
                      "History and Philosophy of Physics", "Instrumentation and Detectors", "Medical Physics", "Optics",
                        "Physics and Society", "Physics Education", "Plasma Physics", "Popular Physics", "Space Physics"]
all_papers = pd.read_csv("../data/all_papers.csv")



def entropy_component(embedding_matrix: np.ndarray):
    '''
    Calculate the information entropy of all the columns of the matrix given as input.

    Parameters
    ----------
        embedding_matrix: np.ndarray
            The matrix for which we want to calculate the entropy of each column.
    Returns
    -------
        entropy: np.ndarray
            Array of length equal to the number of rows of the input matrix, containing the entropies of all the columns of the
            input matrix.
    '''
    #normalize the rows of the matrix so that each row sums up to 1. In this way each element of a row of the matrix is the value
    #of p to use in the entropy calculation
    sum_of_rows = np.sum(embedding_matrix, axis = 0)
    embedding_matrix_normalized = embedding_matrix / sum_of_rows


    #to have -plog(p), multiply each element of the normalized matrix by the log of that element, and then take the negative
    embedding_matrix_normalized = np.where(np.isclose(embedding_matrix_normalized, 0.), 0., (embedding_matrix_normalized * np.log(embedding_matrix_normalized)) * (-1))

    #sum over the rows to obtain the entropy
    entropy = embedding_matrix_normalized.sum(axis = 0)
    return entropy


def build_adjacency_matrix(embedding_matrix: csr_array, threshold: float) -> csr_array:
    '''
    Create an adjacency matrix, starting from distance matrix between abstract embeddings obtained using tf-idf, filtered to keep
    only the components with an entropy (calculated across all abstracts) higher than a certain threshold.

    The similarity between vectors representing abstracts is calculated using cosine similarity. Since some components have
    been filtered, the vectors are no more normalized to 1, so we cannot use the scalar product instead.

    All similarity values are put in a matrix called distance matrix, where each entry represents the distance between two
    abstract vectors. However, since we used the similarity as a score, a higher value means a lower distance between two
    papers.

    After the calculation of the distance matrix, all values lower than the threshold distance given as input are converted
    into 0, while higher values are converted into 1. In this way the distance matrix is converted into the adjacency matrix of
    a network.

    All values on the diagonal of the adjacency matrix are manually set to 0, to avoid the creation of self-links in the network.
    
    Parameters
    ---------
        embedding_matrix: csr_array
            The matrix with the embeddings of all abstracts.
        threshold: float
            The threshold distance to transform distance matrix into adjacency matrix.
    Returns
    -------
        adjacency_matrix: scipy.sparse.csr_array
            The adjacency matrix of the network built from abstracts TF-IDF embeddings
    '''
    adjacency_matrix = lil_array((embedding_matrix.shape[0], embedding_matrix.shape[0]), dtype = np.int32)

    #25877 = 113 * 229, so we iterate 229 times and at each iteration consider 113 abstracts
    for i in range(229):
        #calculate the cosine similarity of 113 embedded abstracts with all the others
        distance_matrix_row = cosine_similarity(embedding_matrix[(i * 113):((i+1) * 113)], embedding_matrix)

        #apply the threshold on similarity to create a row of the adjacency matrix
        adjacency_matrix_row = np.array(distance_matrix_row > threshold, dtype = np.int32)
        adjacency_matrix[(i * 113):((i+1) * 113)] = adjacency_matrix_row

    #since the similarity of a vector with itself is always 1, each node in the network has a link with itself (the diagonal elements
    #of the adjacency matrix are all 1). So we set all diagonal elements to 0 to remove these links
    adjacency_matrix.setdiag(0)
    return adjacency_matrix


def sweep_connected_components(embedding_matrix: csr_array):
    '''
    Starting from the embeddings of paper abstracts using tf-idf, filtered on the basis of entropy, this function builds ten different
    networks, using as threshold distance ten values in the range 0.15-0.5. Then, the number of connected components for each network
    is calculated, as well as the size of the largest component, and the results stored in the csv file
    "extra/results/connected_components.csv".

    Parameters
    ----------
        embedding_matrix: csr_array
            The matrix with the embeddings of papers (filtered on the basis of entropy).
    '''
    connected_components = pd.DataFrame(columns = ['Threshold', 'Connected_components', 'Largest_component'])

    for threshold in np.linspace(start = 0.15, stop = 0.5, num = 10):
        adjacency_matrix = build_adjacency_matrix(embedding_matrix = embedding_matrix, threshold = threshold)
        abstract_network = nx.from_scipy_sparse_array(adjacency_matrix)
    
        connected_components.loc[len(connected_components)] = [threshold, nx.number_connected_components(abstract_network),
                                                           max([len(c) for c in list(nx.connected_components(abstract_network))])]
        print("Iteration")

    connected_components.to_csv("./results/connected_components.csv")


def sweep_entropy_threshold():
    '''
    Starting from the embeddings of paper abstracts using tf-idf, this function builds ten different networks, using a threshold 
    distance of 0.2, and keeping only the components of the vector embeddings with an entropy higher than a certain threshold, which
    is given by ten different values in the range 1-5. Then, the number of embedding vector components remaining is calculated,
    as well as the number of connected components for each network and the size of the largest component, and the results stored 
    in the csv file "extra/results/connected_components_entropy.csv".    
    '''
    connected_components = pd.DataFrame(columns = ['Threshold', 'Embedding_components','Connected_components', 'Largest_component'])

    #load matrix with embeddings and array with entropy of each component
    abstract_embeddings = load_npz("../embeddings/abstract_embeddings_tfidf.npz")
    entropy_array = np.load("./entropy_array.npz")['arr_0']

    for threshold in np.linspace(start = 1, stop = 5, num = 10):
        #filter the embedding matrix using the entropy vector with the selected threshold
        embedding_matrix = abstract_embeddings[:, np.where(entropy_array > threshold)[0]]

        adjacency_matrix = build_adjacency_matrix(embedding_matrix = embedding_matrix, threshold = 0.2)
        abstract_network = nx.from_scipy_sparse_array(adjacency_matrix)
    
        connected_components.loc[len(connected_components)] = [threshold, embedding_matrix.shape[1], nx.number_connected_components(abstract_network),
                                                           max([len(c) for c in list(nx.connected_components(abstract_network))])]
        print("Iteration")

    connected_components.to_csv("./results/connected_components_entropy.csv")


def connectivity_between_cathegories():
    '''
    Create a 22 x 22 matrix, where each row and each column represent one of the 22 cathegories of physics papers, and the entries
    are the number of links in the TF-IDF network between nodes belonging to the two cathegories. So the matrix is symmetric, and
    the diagonal elements are the links between papers of the same cathegory.

    The matrix is saved as a dataset in the csv file "extra/results/connections_between_cathegories.csv", where also 
    a column representing the number of papers of each cathegory is included.
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
    Split the abstract embeddings network into communities using the sign of the components of the Fiedler eigenvector.

    During the first iteration, the nodes with degree 0 are removed from the graph. Then the network is divided into two parts following
    the signs of the Fiedler eigenvector, and the two induced subgraphs are built from this division.

    During the following iterations, the induced subgraph with the higher number of nodes is divided into two more parts following 
    again the signs of the Fiedler eigenvector of that graph. This procedure is repeated until we have a total of 22 subgraphs (same
    number as the topics of Physics papers).

    All subgraphs obtained are saved in the Python list "tf_idf_network/results/abstract_embeddings/fiedler_split".
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
        #calculate fiedler eigenvector
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
    Laplacian matrix of the remaining network are calculated and their components used as features for network nodes (30 features
    per node). After that, a K-Means clustering algorithm (with K = 22) is used to divide the network into 22 clusters.

    All subgraphs induced by the division of the network into clusters are saved in the Python list 
    "tf_idf_network/results/abstract_embeddings/k_means_split".
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
    Split the network built from abstract tf-idf embeddings using the Louvain algorithm.

    First, the nodes that do not belong to the largest connected component are removed. Then, the Louvain method is applied as
    explained in the networkx documentation.

    All subgraphs induced by the division of the network into communities are saved in the Python list 
    "tf_idf_network/results/abstract_embeddings/louvain_split".
    
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

    communities_list = nx.community.louvain_communities(G, seed = 42)

    #the list obtained with the Louvain algorithm contains just the nodes of each subgraph. Now we need to create the
    #subgraphs starting from these nodes.
    subgraphs_list = []
    for community in communities_list:
        subgraphs_list.append(G.subgraph(community).copy())

    #save the list of subgraphs
    with open("tf_idf_network/results/abstract_embeddings/louvain_split", 'wb') as file:
        pickle.dump(subgraphs_list, file)


def cathegories_by_community(split_method: str):
    '''
    Create a table where the rows are the communities obtained using the input algorithm, the columns are the cathegories 
    of Physics papers and each entry is the number of papers of a certain cathegory in a certain community.

    The table is saved in dedicated csv files, one for each algorithm used to obtain the communities.

    Parameters
    ----------
        split_method: {'fiedler', 'kmeans', 'louvain'}
            The method used to obtain the division of the network into communities
    '''
    cathegories_by_community = pd.DataFrame(columns = all_physics_topics)

    if split_method == 'fiedler':
        subgraphs_list_path = "tf_idf_network/results/abstract_embeddings/fiedler_split"
    elif split_method == 'kmeans':
        subgraphs_list_path = "tf_idf_network/results/abstract_embeddings/k_means_split"
    elif split_method == 'louvain':    
        subgraphs_list_path = "tf_idf_network/results/abstract_embeddings/louvain_split"

    #load subgraphs
    with open(subgraphs_list_path, "rb") as file:
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

    cathegories_by_community.to_csv("tf_idf_network/results/cathegories_by_community_" + split_method + ".csv")
