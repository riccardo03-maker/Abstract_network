#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_array, lil_array

__author__= ['Riccardo Grandicelli']
__email__= ['riccardograndicelli03@gmail.com']


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
        #calculate the scalar product of 113 embedded abstracts with all the others
        distance_matrix_row = embedding_matrix[(i * 113):((i+1) * 113)] @ embedding_matrix.T

        #apply the threshold on similarity to create a row of the adjacency matrix
        adjacency_matrix_row = np.array(distance_matrix_row.toarray() > threshold, dtype = np.int32)
        adjacency_matrix[(i * 113):((i+1) * 113)] = adjacency_matrix_row

    #since the similarity of a vector with itself is always 1, each node in the network has a link with itself (the diagonal elements
    #of the adjacency matrix are all 1). So we set all diagonal elements to 0 to remove these links
    adjacency_matrix.setdiag(0)
    return adjacency_matrix