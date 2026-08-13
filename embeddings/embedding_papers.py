#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import save_npz
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

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
    list_of_abstracts = list(papers["abstract"])

    #remove words with numbers and underscores from the vocabulary
    vectorizer = CountVectorizer()
    vectorizer.fit(list_of_abstracts)
    vocabulary = list(vectorizer.get_feature_names_out())
    no_numbers_vocabulary = [word for word in vocabulary 
                             if not any(character in word for character in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_'])]

    #create the embeddings of the abstracts and save them
    vectorizer = CountVectorizer(vocabulary = no_numbers_vocabulary)
    X = vectorizer.fit_transform(list_of_abstracts)
    transformer = TfidfTransformer()
    abstract_embeddings = transformer.fit_transform(X)
    save_npz(file = "embeddings/abstract_embeddings_tfidf", matrix = abstract_embeddings)


def embed_titles_scibert():
    '''
    Transform the titles of all papers into vectors, using the SciBERT language model.

    Each word of each title is embedded in a 784-dimensional vector. Then, the mean of the single-word embeddings is taken for each 
    title, in order to obtain a single vector for each paper.

    The embeddings of titles are saved as numpy arrays in a npz file.

    References
    ----------
        SciBERT paper: https://arxiv.org/pdf/1903.10676
    '''
    #load the pretrained model
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

    #create a list with all titles
    papers = pd.read_csv("data/all_papers.csv")
    list_of_titles = list(papers["title"])

    #create the embedding for each word in each title, using 113 titles at a time
    for i in range(229): #229 * 113 = 25877, the total number of papers
        batch_of_titles = list_of_titles[(i * 113):((i+1) * 113)]

        inputs = tokenizer(batch_of_titles, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state

        #do the mean of the word embeddings for each title. When taking the mean, we need to neglect the additional tokens added by the model to
        #have the same number of tokens in each title
        attention_mask = inputs['attention_mask'] #the attention mask has a 1 for each real word and a 0 for each additional token
        #make the attention mask of the same size of the embeddings tensor
        attention_mask = attention_mask.unsqueeze(-1).expand(embeddings.size())

        #do the weighted mean using the attention mask as weights (so that additional tokens are not considered)
        title_embeddings = torch.sum(embeddings * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)

        #create or update vector of title embeddings
        if i==0:
            embeddings_array = title_embeddings.numpy()
        else:
            embeddings_array = np.concatenate((embeddings_array, title_embeddings.numpy()), axis = 0)

        print("Iteration")

    #save the embeddings in a npz file
    np.savez("embeddings/title_embeddings_scibert.npz", embeddings_array)


if(__name__ == '__main__'):
    embed_titles_scibert()
