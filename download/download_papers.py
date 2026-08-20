#!/usr/bin/python
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import pickle

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']

all_physics_topics = ["Accelerator Physics", "Applied Physics", "Atmospheric and Oceanic Physics", "Atomic and Molecular Clusters", 
                      "Atomic Physics", "Biological Physics", "Chemical Physics", "Classical Physics", "Computational Physics", 
                      "Data Analysis, Statistics and Probability", "Fluid Dynamics", "General Physics", "Geophysics", 
                      "History and Philosophy of Physics", "Instrumentation and Detectors", "Medical Physics", "Optics",
                        "Physics and Society", "Physics Education", "Plasma Physics", "Popular Physics", "Space Physics"]


def download_papers():
    '''
    Download from arXiv the papers used for the analysis. The IDs of papers used are provided in 13 Python lists, saved in the 
    folder "data/list_of_papers".

    For each paper, title, authors and abstract are downloaded, together with the primary and secondary cathegory. The primary
    cathegory is the one reported in bold characters in the arXiv page of the paper. The secondary cathegory is the first cathegory
    belonging to the Physics archive among those that are not written in bold characters. If the paper has only one cathegory, or
    if no secondary cathegory belongs to the Physics archive, the secondary cathegory is just identical to the primary cathegory.

    All papers downloaded in this way are saved in the csv file "data/all_papers.csv".
    '''
    papers = pd.DataFrame(columns = ['paper', 'authors', 'title', 'abstract', 'primary_cathegory', 'secondary_cathegory'])

    for i in range(13):
        with open("data/list_of_papers/all_papers_" + str(i) + "_list", 'rb') as file:
            paper_id_list = pickle.load(file)

        for k in range(len(paper_id_list)):
            article_number = paper_id_list[k] #already a string
            article_response = requests.get("https://arxiv.org/abs/" + article_number)
            article_soup = BeautifulSoup(article_response.text, 'html.parser')

            #find authors
            authors_tag = article_soup.find_all('meta', attrs = {'name' : 'citation_author'})
            authors = [tag.attrs['content'] for tag in authors_tag]

            #find title
            title = article_soup.find('meta', attrs = {'name' : 'citation_title'}).attrs['content']

            #find abstract
            abstract = article_soup.find('meta', attrs = {'name' : 'citation_abstract'}).attrs['content']

            #find primary cathegory
            cathegory_tag = article_soup.find('span', attrs = {'class' : 'primary-subject'})
            cathegory = re.findall(r'(.*)\s\(', cathegory_tag.text)[0]

            #find secondary cathegory
            other_cathegories_tag = article_soup.find('td', attrs = {'class' : 'tablecell subjects'})
            other_cathegories = re.findall(r';\s*([^()]+?)\s*\(', other_cathegories_tag.text)
            secondary_cathegory = [topic for topic in other_cathegories if topic in all_physics_topics]
            if not secondary_cathegory:
                secondary_cathegory = cathegory
            else:
                secondary_cathegory = secondary_cathegory[0]

            #save everything in a dataset
            papers.loc[len(papers)] = [paper_id_list[k], authors, title, abstract, cathegory, secondary_cathegory]
        print("Iteration")

    papers.to_csv("data/all_papers.csv")


if(__name__ == '__main__'):
    download_papers()