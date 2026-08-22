# Integration of multiomic data to predict antimicrobial resistance

[![GitHub pull-requests](https://img.shields.io/github/issues-pr/riccardo03-maker/Abstract_network.svg?style=plastic)](https://github.com/riccardo03-maker/Abstract_network/pulls)
[![GitHub issues](https://img.shields.io/github/issues/riccardo03-maker/Abstract_network.svg?style=plastic)](https://github.com/riccardo03-maker/Abstract_network/issues)

[![GitHub stars](https://img.shields.io/github/stars/riccardo03-maker/Abstract_network.svg?label=Stars&style=social)](https://github.com/riccardo03-maker/Abstract_network/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/riccardo03-maker/Abstract_network.svg?label=Watch&style=social)](https://github.com/riccardo03-maker/Abstract_network/watchers)

This is the GitHub repository of the project "Network-based classification of scientific papers using text embedding". It contains all the code used to download data from arXiv, to build and analyze the networks and to do all the plots and tables present in the report of the project.

## Prerequisites

This repository is written in both Python and R languages. R is used for all the plots and some of the tables, while the rest of the code is written in Python.

A Python version of 3.10 or higher is required for the correct usage of this repository. All the required Python packages are reported in the [requirements.txt](https://github.com/riccardo03-maker/Abstract_network/blob/main/requirements.txt), and they can be installed as described in the [Configuration](#configuration) section.

To re-create the plots, as well as Tables 3 and 6, a R version of 4.3.1 or higher is required, together with the package `tidyverse` (which also includes the `ggplot2` package used for the plots). If R is already installed, `tidyverse` can be installed as described in the [Configuration](#configuration) section.

## Configuration

To use this repository, first clone it into your working directory

```bash
git clone https://github.com/riccardo03-maker/Abstract_network
```
and move to the project root directory

```bash
cd Abstract_network
```
Then, you need to install all the required packages for both Python and R. Python packages can be installed using the command

```bash
pip install -r requirements.txt
```

while the package `tidyverse` for R can be installed with

```bash
Rscript -e 'install.packages("tidyverse", repos="https://cloud.r-project.org")'
```

Now you are ready to use the code in this repository.

## Usage

The entire code in this repository can be executed directly from the `ml.py` script. This Python script works as a command line application, with the following syntax:

```bash
python cl.py --help

usage: cl.py [-h] [--plot {1,2,3,4}] [--table {1,2,3,4,5,6,7}] [--download] [--network {tf_idf,scibert}] [--build]
             [--embed] [--analysis {pendant,scale_free,shuffle,connections}] [--sweep]
             [--community {fiedler,kmeans,louvain}]

options:
  -h, --help            show this help message and exit
  --plot, -p {1,2,3,4}  The number of the figure in the 'plots' folder to re-create using the ggplot2 library of R.
                        Only the single plots are created, since the complete figures were built from the single plots
                        using an image editor, and therefore they cannot be re-created using ggplot2.
  --table, -t {1,2,3,4,5,6,7}
                        The number of the table in the 'tables' folder to re-create
  --download, -d        Download authors, titles, abstracts, primary and secondary cathegory of all 25877 papers used
                        for the analysis.
  --network, -n {tf_idf,scibert}
                        The network used for the algorithm chosen: TF-IDF or SciBERT network. If one option between
                        --build, --embed, --sweep, --analysis and --community is provided, and this option is not
                        provided, an error is raised.
  --build, -b           Build the adjacency matrix of the network chosen as the --network option, and calculate its
                        centrality measures. If the --network option is not provided, an error is raised.
  --embed, -e           If the --network option provided is 'tf_idf", create the text embeddings of all paper
                        abstracts using tf-idf. If the --network option provided is 'scibert', create the text
                        embeddings of all paper titles using SciBERT. If the --network option is not provided, an
                        error is raised.
  --analysis, -a {pendant,scale_free,shuffle,connections}
                        Do the chosen analysis on the network chosen as the --network option. 'pendant' implements the
                        pendant node removal algorithm, and prints the number of nodes remained in the network at the
                        end of the algorithm. 'scale_free' builds a Barabasi-Albert network and calculates its
                        centrality measures. 'shuffle' implements the link shuffling algorithm, and calculates the
                        centrality measures of the network obtained. 'connections' calculates the matrix of
                        connections of the network (number of links between each couple of cathegories). If the
                        --network option is not provided, an error is raised.
  --sweep, -s           Build many adjacency matrices for the network provided as the --network option, for different
                        values of threshold distance, and calculate for each adjacency matrix the number of connected
                        components of the corresponding network and the size of the largest connected component. If
                        the --network option is not provided, an error is raised.
  --community, -c {fiedler,kmeans,louvain}
                        Apply the division into communities using the chosen algorithm on the network provided as the
                        --network option, and calculate the number of papers of each cathegory in each community. If
                        the --network option is 'scibert' and this option is different from 'louvain', an error is
                        raised. If the --network option is not provided, an error is raised.
```
Each output of the code is saved in a dedicated folder. So, the `cl.py` must always be executed from the project root directory to have all outputs stored in the correct folders.


## Table of contents

| Directory | Description |
|---|---|
| [data](https://github.com/riccardo03-maker/Abstract_network/tree/main/data)| Data on papers used for this project|
|[download](https://github.com/riccardo03-maker/Abstract_network/tree/main/download)| Python script to download from arXiv the papers used for this project|
|[embeddings](https://github.com/riccardo03-maker/Abstract_network/tree/main/embeddings)| Embeddings of abstracts/titles using TF-IDF/SciBERT|
|[plots](https://github.com/riccardo03-maker/Abstract_network/tree/main/plots)| All figures used in the report (and the code to re-create them)|
|[scibert_network](https://github.com/riccardo03-maker/Abstract_network/tree/main/scibert_network)|Code and results for the analysis of the network built from SciBERT embeddings|
|[tables](https://github.com/riccardo03-maker/Abstract_network/tree/main/tables)| All tables used in the report (and the code to re-create them)|
|[tf_idf_network](https://github.com/riccardo03-maker/Abstract_network/tree/main/tf_idf_network)|Code and results for the analysis of the network built from TF_IDF embeddings|
|[cl.py](https://github.com/riccardo03-maker/Abstract_network/blob/main/cl.py)| Command line interface to execute all functions in this repository|

## Authors

*  **Riccardo Grandicelli**