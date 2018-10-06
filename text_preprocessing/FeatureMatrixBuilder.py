# -*- coding: utf-8 -*-
# Literature: Dipanjan S., "Text Analytics with python", 2016, Apress

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from text_preprocessing.TextPreprocessing import TextPreprocessing
import pandas as pd
import numpy as np

class FeatureMatrixBuilder:

    def build_feature_matrix(self, documents, feature_type="tfidf", vocabulary=[], stopwords=set(), n_range=(1, 1)):
        """ Build a feature matrix based on parameters"""
        tp = TextPreprocessing()
        if feature_type == 'frequency':
            # bag of words model
            vectorizer = CountVectorizer(binary=True, min_df=3, max_df=0.25, ngram_range=n_range,
                                         tokenizer=tp.get_tokens_no_punct, stop_words=tp.STOPWORDS, vocabulary=vocabulary)
        elif feature_type == 'tfidf':
            # tfidf model
            vectorizer = TfidfVectorizer(ngram_range=n_range,
                                         vocabulary=vocabulary, tokenizer=tp.get_tokens_no_punct, min_df=3,
                                         stop_words=tp.STOPWORDS, sublinear_tf=True)
        elif feature_type == 'no_custom':
            vectorizer = TfidfVectorizer(analyzer="word", stop_words=tp.STOPWORDS)
        else:
            raise Exception("Falsche Feature-Typ angegeben!")

        feature_matrix = vectorizer.fit_transform(documents).astype(float)
        return vectorizer, feature_matrix

    def build_bowModel(self, corpus, min_df=0., max_df = 1.):
        count_vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
        count_vec_matrix = count_vectorizer.fit_transform(corpus)
        return count_vec_matrix

    def build_nGram_model(self, corpus, ngram_range = (2,2)):
        count_vectorizer = CountVectorizer(ngram_range=ngram_range)
        count_vec_matrix = count_vectorizer.fit_transform(corpus)
        count_vec_matrix = count_vec_matrix.toarray()
        voc = count_vectorizer.get_feature_names()
        df = pd.DataFrame(count_vec_matrix, columns = voc)
        return df

    def build_TfIdfModel(self, corpus, min_df=0., max_df = 1.):
        tfidf_vectorizer = TfidfVectorizer(min_df = min_df, max_df = max_df, use_idf=True)
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
        tfidf_matrix = tfidf_matrix.toarray()
        vocab = tfidf_vectorizer.get_feature_names()
        df = pd.DataFrame(np.round(tfidf_matrix,2), columns=vocab)
        return df