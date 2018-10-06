# -*- coding: utf-8 -*-
# Literature: Dipanjan S., "Text Analytics with python", 2016, Apress

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from text_preprocessing.TextPreprocessing import TextPreprocessing

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