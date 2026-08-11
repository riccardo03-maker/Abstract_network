#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.sparse import load_npz, csr_array, lil_array, save_npz
import numpy as np
import networkx as nx

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


def build_adjacency_matrix(threshold: float) -> csr_array:
    '''
    Create an adjacency matrix, starting from distance matrix between abstract embeddings.

    The similarity between vectors representing abstracts are calculated using cosine similarity. Since all vectors have been normalized
    by the tf-idf transformer function, the cosine similarity between two vectors is equivalent to their scalar product.
    All similarity values are put in a matrix called distance matrix, where each entry represents the distance between two
    abstract vectors. However, since we used the similarity as a score, a higher value means a lower distance between two
    papers.

    After the calculation of the distance matrix, all values lower than the threshold distance given as input are converted
    into 0, while higher values are converted into 1. In this way the distance matrix is converted into the adjacency matrix of
    a network.

    The adjacency matrix is saved in an npz format.

    Parameters
    ---------
        threshold: float
            The threshold distance to transform distance matrix into adjacency matrix.
    '''
    abstract_embeddings = csr_array(load_npz("embeddings/abstract_embeddings_tfidf.npz"))
    adjacency_matrix = lil_array((abstract_embeddings.shape[0], abstract_embeddings.shape[0]), dtype = np.int32)

    #25877 = 113 * 229, so we iterate 229 times and at each iteration consider 113 abstracts
    for i in range(229):
        #calculate the scalar product of 113 embedded abstracts with all the others
        distance_matrix_row = abstract_embeddings[(i * 113):((i+1) * 113)] @ abstract_embeddings.T

        #apply the threshold on similarity to create a row of the adjacency matrix
        adjacency_matrix_row = np.array(distance_matrix_row.toarray() > threshold, dtype = np.int32)
        adjacency_matrix[(i * 113):((i+1) * 113)] = adjacency_matrix_row

    #since the similarity of a vector with itself is always 1, each node in the network has a link with itself (the diagonal elements
    #of the adjacency matrix are all 1). So we set all diagonal elements to 0 to remove these links
    adjacency_matrix.setdiag(0)
    return adjacency_matrix


def build_null_model(model: str):
    '''
    Build a network of the model specified as input, using the same number of nodes (N) and links (M) of the network built from the embeddings
    of the paper abstracts.

    The adjacency matrix of the new network of the specified model is saved in a npz file.

    Parameters
    ----------
        model: {'random', 'scale_free'}
            The name of the null model to use to build this network:
                - Random: a random network built starting from N nodes and putting randomly M links, with N and M fixed.
                - Scale_free: a Barabasi-Albert network. At each time step in the building process a new node is added, and a number
                    links equal to M/N is added to that node. The process is repeated until there are N nodes. If M/N is not integer,
                    there could be less links with respect to the original network.
    '''
    starting_network = nx.from_scipy_sparse_array(load_npz("networks/results/abstract_embeddings/abstract_tfidf_adjacency_0_2.npz"))
    #number of nodes and of links
    N = len(starting_network)
    M = starting_network.size()

    if model == 'random':
        G = nx.gnm_random_graph(N, M, seed = 42)
    if model == 'scale_free':
        G = nx.barabasi_albert_graph(N, (M // N) + 1, seed = 42)

    adjacency_matrix = nx.to_scipy_sparse_array(G, format = 'csr')
    save_npz("networks/results/" + model + "_network/" + model +"_network.npz", adjacency_matrix)


def pendant_node_removal(network: str):
    '''
    Implement an algorithm that progressively removes all nodes of degree 0 and 1 from the chosen network.

    At each iteration, all nodes with degree equal to 0 or 1 are removed from the network. The algorithm is repeated until no node has
    remained or until all remaining nodes have degree greater or equal than 2. In the first case, a message tells that nothing has remained,
    while in the second case the adjacency matrix of the remaining network is saved in a npz file.

    Parameters
    ----------
        network: str
            The path to the file with the adjacency matrix of the network to which the algorithm is applied  
    '''
    G = nx.from_scipy_sparse_array(load_npz(network))
    nodes_last_iteration = 0
    
    while len(G) != nodes_last_iteration:
        #register the number of nodes before the next removal step. If after the removal the number of nodes does not change,
        #stop the cycle
        nodes_last_iteration = len(G)

        degree_list = list(G.degree)
        G.remove_nodes_from([degree_list[index][0] for index, _ in enumerate(list(G.nodes)) if degree_list[index][1] in [0, 1]])

    if len(G) == 0:
        print("No nodes remained in the network after iterated pendant nodes removal")
    else:
        save_npz("networks/results/abstract_embeddings/abstract_core.npz", nx.to_scipy_sparse_array(G, format = 'csr'))


def network_node_removal():
    '''
    Remove the ten highest degree nodes from the abstract embeddings network, to see how robust is the network to the removal of
    highest degree nodes.

    The adjacency matrix of the new network without the three nodes is saved as a npz file.
    '''
    G = nx.from_scipy_sparse_array(load_npz("networks/results/abstract_embeddings/abstract_tfidf_adjacency_0_2.npz"))
    G.remove_nodes_from([17128, 10651, 9132, 16526, 9091, 12460, 3787, 20387, 16369, 2966]) #these are the nodes with higher degrees
    adjacency_matrix = nx.to_scipy_sparse_array(G, format = "csr")
    save_npz("networks/results/abstract_node_removal/abstract_node_removal.npz", adjacency_matrix)


if(__name__ == '__main__'):
    #adjacency_matrix = build_adjacency_matrix(threshold = 0.2)
    #save_npz("networks/adjacency_matrices/abstract_tfidf_adjacency_0_2.npz", matrix = adjacency_matrix.tocsr())
    #build_null_model(model = 'scale_free')
    #pendant_node_removal("networks/results/abstract_embeddings/abstract_tfidf_adjacency_0_2.npz")
    network_node_removal()