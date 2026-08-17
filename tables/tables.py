#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from collections import Counter
import subprocess

__author__= ['Riccardo Grandicelli']
__email__= ['riccardograndicelli03@gmail.com']

all_physics_topics = ["Accelerator Physics", "Applied Physics", "Atmospheric and Oceanic Physics", "Atomic and Molecular Clusters", 
                      "Atomic Physics", "Biological Physics", "Chemical Physics", "Classical Physics", "Computational Physics", 
                      "Data Analysis, Statistics and Probability", "Fluid Dynamics", "General Physics", "Geophysics", 
                      "History and Philosophy of Physics", "Instrumentation and Detectors", "Medical Physics", "Optics",
                        "Physics and Society", "Physics Education", "Plasma Physics", "Popular Physics", "Space Physics"]
all_papers = pd.read_csv("data/all_papers.csv")



def table_of_cathegories():
    '''
    Create the first table of the report. 
    '''
    #create list of cathegories, using the second cathegory when the first one is not a Physics cathegory
    topics_list = [all_papers['primary_cathegory'][i] if all_papers['primary_cathegory'][i] in all_physics_topics 
                    else all_papers['secondary_cathegory'][i] for i in range(25877)]

    papers_cathegories = dict(Counter(topics_list))
    papers_cathegories_dict = {'Cathegory' : papers_cathegories.keys(), "Number_of_papers" : papers_cathegories.values()}
    papers_cathegories_data = pd.DataFrame(data = papers_cathegories_dict)
    papers_cathegories_data.to_csv("tables/table_1.csv")


def table_connected_components_tf_idf():
    '''
    Create the second table of the report.
    '''
    subprocess.run("cp tf_idf_network/results/abstract_embeddings/connected_components.csv tables/table_2.csv", shell = True)


if(__name__ == '__main__'):
    table_connected_components_tf_idf()
