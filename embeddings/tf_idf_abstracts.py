#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import save_npz

__author__= ['Riccardo Grandicelli']
__email__= ['riccardograndicelli03@gmail.com']


def embed_abstracts_tf_idf():
    '''
    Transform the abstracts of all papers into vectors, using the term frequency-inverse document frequency method.

    First, this function counts, for eachone of the words in all abstracts, how many times it appears in each abstract. Words that
    contain a number or a '_' are not considered. Then, for each word the tf-idf score in each abstract is calculated, using the formula
    specified in the TfidfTrasformer documentation of scikit-learn.

    So, each abstract is embedded in a vector with a number of elements equal to the length of the vocabulary (total number of words
    appearing in all abstracts). The embeddings of the abstracts are stored in a sparse matrix and saved in the npz format.

    References
    ----------
        TfidfTransformer documentation: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
    '''
    #create a list with all abstracts
    papers = pd.read_csv("data/all_papers.csv")
    list_of_abstract = list(papers["abstract"])

    #remove words with numbers and underscores from the vocabulary
    vectorizer = CountVectorizer()
    vectorizer.fit(list_of_abstract)
    vocabulary = list(vectorizer.get_feature_names_out())
    no_numbers_vocabulary = [word for word in vocabulary 
                             if not any(character in word for character in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_'])]

    #create the embeddings of the abstracts and save them
    vectorizer = CountVectorizer(vocabulary = no_numbers_vocabulary)
    X = vectorizer.fit_transform(list_of_abstract)
    transformer = TfidfTransformer()
    abstract_embeddings = transformer.fit_transform(X)
    save_npz(file = "embeddings/abstract_embeddings_tfidf", matrix = abstract_embeddings)


if(__name__ == '__main__'):
    embed_abstracts_tf_idf()