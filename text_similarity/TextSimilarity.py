
"""
Sources/Inspiration:
https://sites.temple.edu/tudsc/2017/03/30/measuring-similarity-between-texts-in-python/
http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/
"""

from spacy.tests.pipeline.test_pipe_methods import nlp
from text_preprocessing.TextPreprocessing import TextPreprocessing
from feature_extraction.FeatureMatrixBuilder import FeatureMatrixBuilder
import itertools as it


class TextSimilarity:

    def similarity(self, corpus, vocabulary = False, tokenizer= "no punct"):
        text_preprocessor = TextPreprocessing()
        feature_matrix_builder = FeatureMatrixBuilder()
        if tokenizer == "no punct":
            tokenizer = text_preprocessor.get_tokens_no_punct
        elif tokenizer == "important":
            tokenizer = text_preprocessor.get_important_tokens
        elif tokenizer == "chunks":
            tokenizer = text_preprocessor.get_valid_chunks_text
        similarities = feature_matrix_builder.buildSimilarityModel(corpus,tokenizer=tokenizer, stopwords=text_preprocessor.stopwords, ngram_range=(1,1))
        return similarities



    def compute_similarity_spacy(self, text1, text2):
        sents1 = nlp(text1).sents
        sents2 = nlp(text2).sents
        #Get all possible cartesian combination of the sentences
        combis = list(it.product(sents1, sents2))
        for s1, s2 in combis:
            sim = s1.similarity(s2)



CORPUS = [
"Das rote Auto hält an der roten Ampel. Das rote Auto fährt zu schnell.",
"Das schwarze Auto wurde rot eingefärbt, weil die Farbe rot lebendiger wirkt",
"Das schwarze Mofa fährt an der grünen Ampel durch, hält aber am roten Schild nicht und wird vom roten Auto angefahren",
"Morgen scheint die Sonne und wir fahren ans Meer"
]

ts = TextSimilarity()
similarities = ts.similarity(CORPUS, tokenizer="important")

print(similarities[3])
