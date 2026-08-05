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

    The distance between vectors representing abstracts are calculated using cosine similarity. Since all vectors have been normalized
    by the tf-idf transformer function, the distance between two vectors given by cosine similarity is equivalent to 
    their scalar product.

    After the calculation of the distance matrix, all distances greater than the threshold distance given as input are converted
    into 0, while lower distances are converted into 1. In this way the distance matrix is converted into the adjacency matrix of
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

        #apply the threshold on distance to create a row of the adjacency matrix
        adjacency_matrix_row = np.array(distance_matrix_row.toarray() < threshold, dtype = np.int32)
        adjacency_matrix[(i * 113):((i+1) * 113)] = adjacency_matrix_row
        print(i)

    save_npz(file = "networks/abstract_tfidf_adjacency_0_01.npz", matrix = adjacency_matrix.tocsr())


if(__name__ == '__main__'):
    build_adjacency_matrix(threshold = 0.01)