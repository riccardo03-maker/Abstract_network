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
    Build a network of the model specified as input, using the same number of nodes and links of the network built from the embeddings
    of the paper abstracts.

    The adjacency matrix of the new network of the specified model is saved in a npz file.

    Parameters
    ----------
        model: {'random', 'scale_free'}
            The name of the null model to use to build this network:
                - Random: a random network built starting from N nodes and putting randomly M links, with N and M fixed.
    '''
    starting_network = nx.from_scipy_sparse_array(load_npz("networks/adjacency_matrices/abstract_tfidf_adjacency_0_2.npz"))
    #number of nodes and of links
    N = len(starting_network)
    M = starting_network.size()

    if model == 'random':
        G = nx.gnm_random_graph(N, M, seed = 42)
    if model == 'scale_free':
        print("Currently under development")
        return None

    adjacency_matrix = nx.to_scipy_sparse_array(G, format = 'csr')
    save_npz("networks/adjacency_matrices/" + model + "_network.npz", adjacency_matrix)


if(__name__ == '__main__'):
    #adjacency_matrix = build_adjacency_matrix(threshold = 0.2)
    #save_npz("networks/adjacency_matrices/abstract_tfidf_adjacency_0_2.npz", matrix = adjacency_matrix.tocsr())
    build_null_model(model = 'random')