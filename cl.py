#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import subprocess
from scipy.sparse import save_npz, load_npz
import networkx as nx

from download.download_papers import download_papers
from embeddings.embedding_papers import embed_titles_scibert, embed_abstracts_tf_idf
import tf_idf_network.build_networks as build_tf_idf
import tf_idf_network.network_analysis as analyze_tf_idf
import scibert_network.build_networks as build_scibert
import scibert_network.network_analysis as analyze_scibert
from tables import tables

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
        dest = 'community',
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


    if args.download:
        download_papers()


    if args.embed:
        if args.network == 'tf_idf':
            embed_abstracts_tf_idf()
        elif args.network == 'scibert':
            embed_titles_scibert()
        else:
            print("Option --network needed")
            exit(1)


    if args.sweep:
        if args.network == 'tf_idf':
            analyze_tf_idf.sweep_connected_components()
        elif args.network == 'scibert':
            analyze_scibert.sweep_connected_components()
        else:
            print("Option --network needed")
            exit(1)


    if args.build:
        if args.network == 'tf_idf':
            adjacency_matrix = build_tf_idf.build_adjacency_matrix(threshold = 0.2)
            save_npz("tf_idf_network/results/abstract_embeddings/abstract_tfidf_adjacency_0_2.npz", matrix = adjacency_matrix.tocsr())
            analyze_tf_idf.centrality_measures("tf_idf_network/results/abstract_embeddings/abstract_tfidf_adjacency_0_2.npz")
        elif args.network == 'scibert':
            adjacency_matrix = build_scibert.build_adjacency_matrix(threshold = 0.85)
            save_npz("scibert_network/results/title_embeddings/title_scibert_adjacency_0_85.npz", matrix = adjacency_matrix.tocsr())
            analyze_scibert.centrality_measures("scibert_network/results/title_embeddings/title_scibert_adjacency_0_85.npz")
        else:
            print("Option --network needed")
            exit(1)


    if args.analysis == 'pendant':
        if args.network == 'tf_idf':
            build_tf_idf.pendant_node_removal()
            G = nx.from_scipy_sparse_array(load_npz("tf_idf_network/results/abstract_embeddings/abstract_core.npz"))
            print(len(G))
        elif args.network == 'scibert':
            build_scibert.pendant_node_removal()
            G = nx.from_scipy_sparse_array(load_npz("scibert_network/results/title_embeddings/title_core.npz"))
            print(len(G))
        else:
            print("Option --network needed")
            exit(1)

    elif args.analysis == 'scale_free':
        if args.network == 'tf_idf':
            build_tf_idf.build_scale_free()
            analyze_tf_idf.centrality_measures("tf_idf_network/results/scale_free_network/scale_free_network.npz")
        elif args.network == 'scibert':
            build_scibert.build_scale_free()
            analyze_scibert.centrality_measures("scibert_network/results/scale_free_network/scale_free_network.npz")
        else:
            print("Option --network needed")
            exit(1)

    elif args.analysis == 'shuffle':
        if args.network == 'tf_idf':
            build_tf_idf.link_shuffling()
            analyze_tf_idf.centrality_measures("tf_idf_network/results/link_shuffle/link_shuffle.npz")
        elif args.network == 'scibert':
            build_scibert.link_shuffling()
            analyze_scibert.centrality_measures("scibert_network/results/link_shuffle/link_shuffle.npz")
        else:
            print("Option --network needed")
            exit(1)

    elif args.analysis == 'connections':
        if args.network == 'tf_idf':
            analyze_tf_idf.connectivity_between_cathegories()
        elif args.network == 'scibert':
            analyze_scibert.connectivity_between_cathegories()
        else:
            print("Option --network needed")
            exit(1)


    if args.community == 'fiedler':        
        if args.network == 'tf_idf':
            analyze_tf_idf.split_fiedler_eigenvector()
            analyze_tf_idf.cathegories_by_community(split_method = 'fiedler')
        elif args.network == 'scibert':
            print("Fiedler algorithm was not implemented for SciBERT network")
            exit(1)
        else:
            print("Option --network needed")
            exit(1)

    elif args.community == 'kmeans':
        if args.network == 'tf_idf':
            analyze_tf_idf.split_k_means()
            analyze_tf_idf.cathegories_by_community(split_method = 'kmeans')
        elif args.network == 'scibert':
            print("K-Means clustering was not implemented for SciBERT network")
        else:
            print("Option --network needed")
            exit(1)

    elif args.community == 'louvain':
        if args.network == 'tf_idf':
            analyze_tf_idf.split_louvain_method()
            analyze_tf_idf.cathegories_by_community(split_method = 'louvain')
        elif args.network == 'scibert':
            analyze_scibert.split_louvain_method()
            analyze_scibert.cathegories_by_community()
        else:
            print("Option --network needed")
            exit(1)

     
    if args.plot == '1':
        subprocess.run(["Rscript", "plots/figure_1/figure_1.R"])
    if args.plot == '2':
        subprocess.run(["Rscript", "plots/figure_2/figure_2.R"])
    if args.plot == '3':
        subprocess.run(["Rscript", "plots/figure_3/figure_3.R"])
    if args.plot == '4':
        subprocess.run(["Rscript", "plots/figure_4/figure_4.R"])


    if args.table == '1':
        tables.table_of_cathegories()
    if args.table == '2':
        tables.table_connected_components_tf_idf()
    if args.table == '3':
        subprocess.run(["Rscript", "tables/table_3.R"])
    if args.table == '4':
        tables.table_of_cathegories_by_community_tf_idf()
    if args.table == '5':
        tables.table_connected_components_scibert()
    if args.table == '6':
        subprocess.run(["Rscript", "tables/table_6.R"])
    if args.table == '7':
        tables.table_of_cathegories_by_community_scibert()