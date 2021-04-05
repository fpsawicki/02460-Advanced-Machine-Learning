from abc import ABC


class Indexer(ABC):
    def __init__(self):
        pass

    def __call__(self, text):
        pass


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
