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

(TODO)

## Table of contents

Description of the folders of this repository

(TODO)
## Authors

*  **Riccardo Grandicelli**