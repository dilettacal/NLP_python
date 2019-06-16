"""
Sources/Inspiration:
https://sites.temple.edu/tudsc/2017/03/30/measuring-similarity-between-texts-in-python/
http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/
"""
import spacy

from ..text_preprocessing.CSVDataReader import CSVDataReader
from ..text_preprocessing.TextPreprocessing import TextPreprocessing
from ..feature_extraction.FeatureMatrixBuilder import FeatureMatrixBuilder
import itertools as it


class TextSimilarity:
    nlp = spacy.load('de_core_news_sm')

    def similarity(self, corpus, vocabulary=False, tokenizer="no punct"):
        text_preprocessor = TextPreprocessing()
        feature_matrix_builder = FeatureMatrixBuilder()
        if tokenizer == "no punct":
            tokenizer = text_preprocessor.get_tokens_no_punct
        elif tokenizer == "important":
            tokenizer = text_preprocessor.get_important_tokens
        elif tokenizer == "chunks":
            tokenizer = text_preprocessor.get_valid_chunks_text
        similarities = feature_matrix_builder.buildSimilarityModel(corpus, tokenizer=tokenizer,
                                                                   stopwords=text_preprocessor.stopwords,
                                                                   ngram_range=(1, 1))
        return similarities

    def compute_similarity_spacy(self, text1, text2):
        tp = TextPreprocessing()
        processed_text1 = tp.normalize_document(text1)
        processed_text2 = tp.normalize_document(text2)
        doc1 = self.nlp(processed_text1)
        doc2 = self.nlp(processed_text2)
        return doc1.similarity(doc2)


    def compute_similarity_news_paper_corpus(self):
        nlp = spacy.load('de_core_news_sm')
        datareader = CSVDataReader()
        corpus = datareader.get_train()
        print(corpus.head())
        all_text1 = corpus["text1"].values
        all_text2 = corpus["text2"].values
        complete_corpus = zip(all_text1, all_text2)
        similarities = []
        for t1, t2 in complete_corpus:
            doc1 = nlp(t1)
            doc2 = nlp(t2)
            print(doc1.similarity(doc2))
            similarities.append(doc1.similarity(doc2))
        return similarities


CORPUS = [
    "Das rote Auto hält an der roten Ampel. Das rote Auto fährt zu schnell.",
    "Das schwarze Auto wurde rot eingefärbt, weil die Farbe rot lebendiger wirkt",
    "Das schwarze Mofa fährt an der grünen Ampel durch, hält aber am roten Schild nicht und wird vom roten Auto angefahren",
    "Morgen scheint die Sonne und wir fahren ans Meer"
]

ts = TextSimilarity()
similarities = ts.similarity(CORPUS, tokenizer="important")

print(similarities[3])

datareader = CSVDataReader()
corpus = datareader.get_train()
all_text1 = corpus["text1"].values
all_text2 = corpus["text2"].values
first_text1 = all_text1[:1][0]
first_text2 = all_text2[:1][0]
print(first_text1)

print(ts.compute_similarity_spacy(first_text1,first_text2))

example = []
example.append(first_text1)
example.append(first_text1[:100])

ex = ts.similarity(example)
print(ex[0])
print(ex[3])

all_texts = []
all_texts.append(all_text1)
all_texts.append(all_text2)
similarity_sklearn = ts.similarity(all_texts)


