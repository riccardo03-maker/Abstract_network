#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.sparse import load_npz, csr_array, lil_array, save_npz
import numpy as np
import networkx as nx

__author__= ['Riccardo Grandicelli']
__email__= ['riccardograndicelli03@gmail.com']


def build_adjacency_matrix(threshold: float) -> csr_array:
    '''
    Create an adjacency matrix, starting from distance matrix between abstract embeddings obtained using tf-idf.

    The similarity between vectors representing abstracts is calculated using cosine similarity. Since all vectors have been normalized
    by the tf-idf transformer function, the cosine similarity between two vectors is equivalent to their scalar product.
    All similarity values are put in a matrix called distance matrix, where each entry represents the distance between two
    abstract vectors. However, since we used the similarity as a score, a higher value means a lower distance between two
    papers.

    After the calculation of the distance matrix, all values lower than the threshold distance given as input are converted
    into 0, while higher values are converted into 1. In this way the distance matrix is converted into the adjacency matrix of
    a network.

    All values on the diagonal of the adjacency matrix are manually set to 0, to avoid the creation of self-links in the network.
    
    Parameters
    ---------
        threshold: float
            The threshold distance to transform distance matrix into adjacency matrix.
    Returns
    -------
        adjacency_matrix: scipy.sparse.csr_array
            The adjacency matrix of the network built from abstracts TF-IDF embeddings
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


def build_scale_free():
    '''
    Build a Barabasi-Albert network using the same number of nodes of the network built from the embeddings of the paper abstracts.

    At each time step in the building process a new node is added, and a number links equal to M/N is added to that node, where N
    and M are respectively the number of nodes and the number of links in the TF-IDF network. The process is repeated until there
    are N nodes. If M/N is not integer, it is approximated to the next integer, and so there could be more links with respect to 
    the original network.

    The adjacency matrix of the Barabasi-Albert network is saved in the npz file 
    "tf_idf_network/results/scale_free_network/scale_free_network.npz".
    
    References
    ----------
        Barabasi-Albert algorithm networkx: https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.barabasi_albert_graph.html    
    '''
    starting_network = nx.from_scipy_sparse_array(load_npz("tf_idf_network/results/abstract_embeddings/abstract_tfidf_adjacency_0_2.npz"))
    #number of nodes and of links
    N = len(starting_network)
    M = starting_network.size()

    G = nx.barabasi_albert_graph(N, (M // N) + 1, seed = 42)

    adjacency_matrix = nx.to_scipy_sparse_array(G, format = 'csr')
    save_npz("tf_idf_network/results/scale_free_network/scale_free_network.npz", adjacency_matrix)


def pendant_node_removal():
    '''
    Implement an algorithm that progressively removes all nodes of degree 0 and 1 from the network built from TF-IDF embeddings.

    At each iteration, all nodes with degree equal to 0 or 1 are removed from the network. The algorithm is repeated until no node
    remains or until all remaining nodes have degree greater or equal than 2. In the first case, a message tells that nothing has remained,
    while in the second case the adjacency matrix of the remaining network is saved in the npz file
    "tf_idf_network/results/abstract_embeddings/abstract_core.npz".
    '''
    G = nx.from_scipy_sparse_array(load_npz("networks/results/abstract_embeddings/abstract_tfidf_adjacency_0_2.npz"))
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
        save_npz("tf_idf_network/results/abstract_embeddings/abstract_core.npz", nx.to_scipy_sparse_array(G, format = 'csr'))


def link_shuffling():
    '''
    Build a new network by randomly swapping the links of the original network built from abstract tf-idf embeddings.

    The adjacency matrix of the new network is saved in the npz file "tf_idf_network/results/link_shuffle/link_shuffle.npz".
    
    References
    ----------
        Link shuffling algorithm: https://doi.org/10.1126/science.1065103
    '''
    G = nx.from_scipy_sparse_array(load_npz("tf_idf_network/results/abstract_embeddings/abstract_tfidf_adjacency_0_2.npz"))
    G = nx.random_reference(G, seed = 42, connectivity = False)
    adjacency_matrix = nx.to_scipy_sparse_array(G, format = 'csr')
    save_npz("tf_idf_network/results/link_shuffle/link_shuffle.npz", adjacency_matrix)


if(__name__ == '__main__'):
    #adjacency_matrix = build_adjacency_matrix(threshold = 0.2)
    #save_npz("tf_idf_network/adjacency_matrices/abstract_tfidf_adjacency_0_2.npz", matrix = adjacency_matrix.tocsr())
    #build_scale_free()
    #pendant_node_removal()
    link_shuffling()