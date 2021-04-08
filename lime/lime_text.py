from lime.lime_base import BaseLIME
from lime.explainers import TextExplainer
from lime.indexers import Indexer, StringIndexer

import numpy as np

from sklearn.metrics import pairwise_distances


class TextLIME:
    KERNEL_MULTIPLIER = 0.75

    def __init__(self,
                 random_state=123,
                 simple_model=None,
                 kernel_width=None,
                 indexer=None,
                 alpha_penalty=None,
                 distance_metric='l2',
                 feature_selection='highest_weights'):
        """
            random_state: integer randomness seed value
            simple_model: sklearn model for local explainations
            kernel_width: float
            indexer: object of subclass Indexer with callable indexation function
            alpha_penalty: float L2 penalty term in ridge model for feature selection
            distance_metric: str type of metric used in kernel
            feature_selection: str type of feature selection method
        """
        self.base = BaseLIME(random_state, alpha_penalty)
        self.simple_model = simple_model
        self.kernel_width = kernel_width
        self.distance_metric = distance_metric
        self.feature_selection = feature_selection

        if not indexer:
            # hyperparameters from baseline implementation
            self.indexer_fn = StringIndexer()
        #msg = 'Invalid indexer type, use one implemented in indexer.py'
        #assert issubclass(indexer, Indexer), msg
        else: 
            self.indexer_fn = indexer

    def _kernel_fn(self, active_tokens):
        all_tokens = np.ones(active_tokens[0].shape[0])[np.newaxis, :]
        distances = pairwise_distances(
            active_tokens,
            all_tokens,
            metric=self.distance_metric).ravel()

        kernel = self.kernel_width
        if kernel is None:
            kernel = np.sqrt(len(distances)) * TextLIME.KERNEL_MULTIPLIER
        kernel = float(kernel)

        kernel = np.sqrt(np.exp(-(distances ** 2) / kernel ** 2))
        return kernel

    def _neighborhood_generation(self, text, num_samples):
        neighborhood_data = []
        split_text = text.strip().replace('  ', '').replace(',', '').replace('.','').lower()
        num_indexes = np.unique(split_text).shape[0]
        active_indexes = np.random.binomial(1, 0.5, size=(num_samples, num_indexes))
        for k in range(num_samples):
            active = np.argwhere(active_indexes[k])
            sample = np.array(split_text) * 0
            for i in active:
                sample = sample + active * split_text
            neighborhood_data.append(sample)

        return np.array(neighborhood_data), active_indexes

    def explain_instance(self, text, main_model, labels=(1,), num_features=100000, num_samples=1000):
        """
            text: numpy array of a single image (RGB or Grayscale)
            main_model: callable object or function returning prediction of an image
            labels: iterable with labels to be explained
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the simple model

            returns: ImageExplainer object with convenient access to instance explainations
        """
        indexed_string = self.indexer_fn(text)
        neigh_data, active_words = self._neighborhood_generation(text, num_samples)
        neigh_weights = self._kernel_fn(active_words)
        neigh_labl = []
        for neigh in neigh_data:
            neigh_labl.append(main_model(neigh))

        neigh_labl = np.array(neigh_labl)

        results = {}
        for label in labels:
            res = self.base.explain_instance(
                active_words, neigh_weights, neigh_labl, label, num_features,
                feature_selection=self.feature_selection, simple_model=self.simple_model
            )
            results[label] = {
                'intercept': res[0],
                'feature_importance': res[1],
                'prediction_score': res[2],
                'local_prediction': res[3]
            }
        return TextExplainer(text, active_words, results)
