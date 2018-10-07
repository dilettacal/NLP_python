# -*- coding: utf-8 -*-
import re, string
import pandas as pd
import spacy
import numpy as np
from nltk.corpus import stopwords
from spacy.lemmatizer import Lemmatizer
import pandas as pd
import nltk
from string import punctuation
from spacy.lang.de.stop_words import STOP_WORDS as spacy_stops

class TextPreprocessing:
    # Get stopwords from nltk
    stopwords = stopwords.words("german")
    stopwords.append("dass")
    # Create final set of stopwords
    STOPWORDS = set(stopwords + list(spacy_stops) + list(punctuation))
    # Load German spacy model
    nlp = spacy.load('de_core_news_sm')

    """" Preprocessing methods - clean, stop words removal, lemmatization """
    def clean_text(self, text: str):
        """ Removes newlines, tabs and double spaces"""
        cleaned_text = re.sub('\n', '', text)
        cleaned_text = re.sub('\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s([,?.!"](?:\s|$))', r'\1', cleaned_text)
        cleaned_text = re.sub('Mehr zum Thema\:(.*)', '', cleaned_text)
        cleaned_text = re.sub('Berliner Morgenpost [0-9]{4} (.*) Alle Rechte vorbehalten\.', '', cleaned_text)
        # Text specific elements to be removed
        REDAKTIONEN = ["( mit epd )", "( BM )", "( dpa )"]
        cleaned_text = cleaned_text.strip("+++")
        cleaned_text = cleaned_text.replace("+++", ".")
        cleaned_text = cleaned_text.replace("0 0", "")
        for x in REDAKTIONEN:
            cleaned_text = cleaned_text.replace(x, "")
        return cleaned_text

    """ Sentence segmentation """
    def parse_sentences(self, document: str):
        # Parsing mit spacy
        doc = self.nlp(document)
        # Sentence segmentation
        sentences = [sent.text for sent in doc.sents]
        for index, sent in enumerate(sentences):
            if len(sent.split()) == 1 and re.match("[-.,!?]", sent):
                sentences.remove(sent)
        if len(sentences[0].split()) == 1:
            sentences.remove(sentences[0])
        return sentences


    def parse_sentences_nltk(self, document: str):
        """ Sent segmentation with nltk """
        sentences = nltk.sent_tokenize(document, language='german')
        return sentences

    """ Lemmatization """

    def lemmatization(self, sent):
        doc = self.nlp(sent)
        lemmatized_tokens = [token.lemma_ for token in doc]
        lemmatized_text = ' '.join(lemmatized_tokens)
        return lemmatized_text

    def lemmatize_tokens_no_punct(self, sent):
        doc = self.nlp(sent)
        lemmatized_tokens = [token.lemma_ for token in doc if not token.is_punct]
        return lemmatized_tokens

    """ Tokenization """
    def get_tokens_no_punct(self, sent):
        doc = self.nlp(sent)
        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        return [token.lemma_.lower().translate(remove_punct_dict) for token in doc if not token.is_punct or not token.like_num]

    def get_important_tokens(self, sent, valid_pos=["PROPN", "VERB", "NOUN", "ADJ"]):
        """Returns a list of important tokens depending on their pos value"""
        valid_pos = valid_pos
        doc = self.nlp(sent)
        tokens = [token.text for token in doc if (token.is_alpha and token.pos_ in valid_pos)]
        return tokens

    """ Nominal phrases """
    def get_valid_chunks_text(self, sentences):
        """Return all noun chunks in a list of sentences"""
        valid_chunks = []
        for sentence in sentences:
            doc = self.nlp(sentence)
            chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
            valid_chunks.append(chunks)
        return valid_chunks

    def get_chunks_in_sent(self, sent: str):
        """ Returns chunks in a sent as str"""
        doc = self.nlp(sent)
        return [chunk.text.lower() for chunk in doc.noun_chunks]

    """ Special chars removal """
    def remove_spec_chars(self, sent):
        """ Removes special chars based on string.punctuations and on tokenizer method"""
        # tokens = get_important_tokens(sent)
        tokens = self.get_tokens_no_punct(sent)
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
        filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])  # filter object
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def remove_stopwords(self, sent):
        # tokens = get_important_tokens(sent)
        tokens = self.get_tokens_no_punct(sent)
        filtered_tokens = [token for token in tokens if token.lower() not in self.STOPWORDS]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text


    """ Normalize corpus of sentences"""

    def normalize_sents_corpus(self, corpus, std_lemmatize=True, tokenize=False):
        """Param corpus = normalized sentences """
        normalized_corpus = []
        for text in corpus:
            if std_lemmatize:
                text = self.lemmatization(text)
            else:
                text = text.lower()
            text = self.remove_spec_chars(text)
            text = self.remove_stopwords(text)
            # For Problem 2, hier immer false am besten
            if tokenize:
                text = self.get_tokens_no_punct(text)
                normalized_corpus.append(text)
            else:
                normalized_corpus.append(text)
        return normalized_corpus

    def normalize_for_other_py_libraries(self, doc):
        """ Prepare texts for use in other python libraries"""
        cleaned_doc = self.clean_text(doc)
        parsed_sents = self.parse_sentences(cleaned_doc)
        cleaned_doc = self.rebuild_document(parsed_sents)
        return cleaned_doc


    def rebuild_document(self, elem_array):
        return ''.join(elem_array)

    """ general normalization """
    def normalize_document(self, document):
        #1. clean the text
        document = self.normalize_for_other_py_libraries(document)
        #2. normalize corpus
        document = self.remove_spec_chars(document)
        document = self.remove_stopwords(document)
        sents = self.parse_sentences(document)
        filtered_tokens = [self.get_tokens_no_punct(sent) for sent in sents]
        document = [' '.join(token) for token in filtered_tokens]
        return document


    def similarity_pipeline(self, corpus):
        norm_corpus =[self.normalize_document(doc) for doc in corpus]
        all_doc_tokenized = [self.parse_sentences(doc) for doc in corpus]
        return norm_corpus, all_doc_tokenized


    """ Build vocabulary """

    def build_vocabulary(self, sents):
        vocabulary = set()
        for sent in sents:
            chunks = self.get_chunks_in_sent(sent)
            toks = self.get_important_tokens(sent)
            vocabulary.update(chunks)
            vocabulary.update(toks)
        vocabulary = list(vocabulary)
        for i in range(len(vocabulary)):
            splits = vocabulary[i].split()
            tokens = [token for token in splits if token and token not in self.STOPWORDS]
            vocabulary[i] = ' '.join(tokens)
        return list(set(filter(bool, vocabulary)))
