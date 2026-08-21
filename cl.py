#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse

__author__= ['Riccardo Grandicelli']
__email__= ['riccardograndicelli03@gmail.com']


if(__name__ == '__main__'):
    '''
    Implement a command line interface that allows the execution of all the code used for the project.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--plot', '-p',
        dest = 'plot',
        required = False,
        action = 'store',
        default = None,
        help = '''The number of the figure in the 'plots' folder to re-create using the ggplot2 library of R. 
        Only the single plots are created, since the complete figures were built from the single plots using an image 
        editor, and therefore they cannot be re-created using ggplot2.
        ''',
        choices = ['1', '2', '3', '4']
    )

    parser.add_argument(
        '--table', '-t',
        dest = 'table',
        required = False,
        action = 'store',
        default = None,
        help = '''The number of the table in the 'tables' folder to re-create
        ''',
        choices = ['1', '2', '3', '4', '5', '6', '7']
    )
    parser.add_argument(
        '--download', '-d',
        dest = 'download',
        required = False,
        action = 'store_true',
        default = False,
        help = '''Download authors, titles, abstracts, primary and secondary cathegory of all 25877 papers used for the analysis.
        '''
    )

    parser.add_argument(
        '--network', '-n',
        dest = 'network',
        required = False,
        action = 'store',
        default = None,
        help = '''The network used for the algorithm chosen: TF-IDF or SciBERT network. If one option between --build, --embed,
        --sweep, --analysis and --community is provided, and this option is not provided, an error is raised.
        ''',
        choices = ['tf_idf', 'scibert']
    )

    parser.add_argument(
        '--build', '-b',
        dest = 'build',
        required = False,
        action = 'store_true',
        default = False,
        help = '''Build the adjacency matrix of the network chosen as the --network option, and calculate its centrality measures.
        If the --network option is not provided, an error is raised.
        '''
    )

    parser.add_argument(
        '--embed', '-e',
        dest = 'embed',
        required = False,
        action = 'store_true',
        default = False,
        help = '''If the --network option provided is 'tf_idf", create the text embeddings of all paper abstracts using tf-idf.
        If the --network option provided is 'scibert', create the text embeddings of all paper titles using SciBERT.
        If the --network option is not provided, an error is raised.
        '''
    )

    parser.add_argument(
        '--analysis', '-a',
        dest = 'analysis',
        required = False,
        action = 'store',
        default = None,
        help = '''Do the chosen analysis on the network chosen as the --network option. 'pendant' implements the pendant node removal
        algorithm, and prints the number of nodes remained in the network at the end of the algorithm. 'scale_free' builds a 
        Barabasi-Albert network and calculates its centrality measures. 'shuffle' implements the link shuffling algorithm, and 
        calculates the centrality measures of the network obtained. 'connections' calculates the matrix of connections of the network
        (number of links between each couple of cathegories).
        If the --network option is not provided, an error is raised.
        ''',
        choices = ['pendant', 'scale_free', 'shuffle', 'connections']
    )

    parser.add_argument(
        '--sweep', '-s',
        dest = 'sweep',
        required = False,
        action = 'store_true',
        default = False,
        help = '''Build many adjacency matrices for the network provided as the --network option, for different values of threshold
        distance, and calculate for each adjacency matrix the number of connected components of the corresponding network and the
        size of the largest connected component.
        If the --network option is not provided, an error is raised.
        '''
    )

    parser.add_argument(
        '--community', '-c',
        dest = 'comunity',
        required = False,
        action = 'store',
        default = None,
        help = '''Apply the division into communities using the chosen algorithm on the network provided as the --network option,
        and calculate the number of papers of each cathegory in each community.
        If the --network option is 'scibert' and this option is different from 'louvain', an error is raised.
        If the --network option is not provided, an error is raised.
        ''',
        choices = ['fiedler', 'kmeans', 'louvain']
    )

    args = parser.parse_args()
