from spacy.tests.pipeline.test_pipe_methods import nlp

from text_preprocessing.TextPreprocessing import TextPreprocessing
from text_preprocessing.CSVDataReader import CSVDataReader
from text_preprocessing.FeatureMatrixBuilder import FeatureMatrixBuilder
import itertools as it

class TextSimilarity:

    def similarity(self, text1, text2, vocabulary = False):
        text_preprocessor = TextPreprocessing()
        feature_matrix_builder = FeatureMatrixBuilder()

        cleaned_text1 = text_preprocessor.normalize_for_other_py_libraries(text1)
        cleaned_text2 = text_preprocessor.normalize_for_other_py_libraries(text2)

        # normalize
        sents1 = text_preprocessor.parse_sentences_nltk(cleaned_text1)
        sents2 = text_preprocessor.parse_sentences_nltk(cleaned_text2)

        # compute similarity
        VOC1 = None
        VOC2 = None
        if(vocabulary == True):
            VOC1 = text_preprocessor.build_vocabulary(sents1)
            VOC2 = text_preprocessor.build_vocabulary(sents2)

        #build the feature matrix for the texts
        vectorizer, features = feature_matrix_builder.build_feature_matrix()


    def compute_similarity_spacy(self, text1, text2):
        sents1 = nlp(text1).sents
        sents2 = nlp(text2).sents
        #Get all possible cartesian combination of the sentences
        combis = list(it.product(sents1, sents2))
        for s1, s2 in combis:
            sim = s1.similarity(s2)




