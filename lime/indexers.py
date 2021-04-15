from abc import ABC
import nltk
from nltk.util import ngrams


class Indexer(ABC):
    def __init__(self):
        pass

    def __call__(self, text):
        pass


class StringTokenizer(Indexer):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text):
        return self.tokenizer(text)


class NgramTokenizer(Indexer):
    def __init__(self):
        nltk.download('punkt')

    def extract_ngrams_x(self, data, num):
        n_grams = ngrams(nltk.word_tokenize(data), num)
        return [' '.join(grams) for grams in n_grams]

    def extract_ngrams(self, data, num):
        return [self.extract_ngrams_x(doc, num) for doc in data]

    def __call__(self, text):
        return self.extract_ngrams(text, 1) + self.extract_ngrams(text, 2)


class StringIndexer(Indexer):
    def __init__(self, mask_string='<UNKNOWN>'):
        pass

    def __call__(self, text):
        pass


class CharIndexer(Indexer):
    def __init__(self, mask_string='#'):
        pass

    def __call__(self, text):
        pass
