from lime.lime_base import BaseLIME
from lime.explainers import TextExplainer
from lime.indexers import Indexer, StringIndexer

import copy
import numpy as np

from sklearn.metrics import pairwise_distances
from scipy.linalg import circulant


class TextLIME:
    KERNEL_MULTIPLIER = 0.75

    def __init__(self,
                 random_state=123,
                 simple_model=None,
                 kernel_width=None,
                 indexer=None,
                 alpha_penalty=None,
                 distance_metric='l2',
                 feature_selection='highest_weights',
                 inactive_string='',
                 neighbour_version='random_uniform',
                 neighbour_parameter=None):
        """
            random_state: integer randomness seed value
            simple_model: sklearn model for local explainations
            kernel_width: float
            indexer: object of subclass Indexer with callable indexation function
            alpha_penalty: float L2 penalty term in ridge model for feature selection
            distance_metric: str type of metric used in kernel
            feature_selection: str type of feature selection method
            inactive_string: str replacement of inactive words in neighborhood generation
            neighbour_version: str version of the neighbourhood generation function to use,
            neighbour_parameter: any parameters related to the neighbourhood generation function
        """
        self.base = BaseLIME(random_state, alpha_penalty)
        self.simple_model = simple_model
        self.kernel_width = kernel_width
        self.distance_metric = distance_metric
        self.feature_selection = feature_selection
        self.inactive_string = inactive_string

        if not neighbour_parameter:
            if neighbour_version == 'random_uniform':
                neighbour_parameter = 0.5
            elif neighbour_version == 'consecutive':
                neighbour_parameter = 3
            elif neighbour_version == 'random_normal':
                neighbour_parameter = 2
        self.neighbour_parameter = neighbour_parameter
        self.neighbour_version = neighbour_version

        if not indexer:
            # hyperparameters from baseline implementation
            self.indexer_fn = StringIndexer()
        # msg = 'Invalid indexer type, use one implemented in indexer.py'
        # assert issubclass(indexer, Indexer), msg
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

    def _normal_tokens(self, num_samples, num_tokens, spread):
        center = np.random.randint(0, num_tokens + 1, num_samples)
        idx_normal = []
        for c in center:
            idx_normal.append(np.round((np.random.normal(c, spread, num_tokens))).astype(int))
        lis = list(range(num_tokens))
        return np.array([[int(i in idx) for i in lis] for idx in idx_normal])

    def _neighborhood_generation(self, token_string, num_samples):
        neighborhood_data = []
        num_indexes = len(token_string)
        if self.neighbour_version == 'random_uniform':
            active_indexes = np.random.binomial(1, self.neighbour_parameter, size=(num_samples, num_indexes))
        elif self.neighbour_version == 'consecutive':
            tmp = np.zeros(num_indexes)
            tmp[0:self.neighbour_parameter] = 1
            active_indexes = circulant(tmp)[:num_samples]
        elif self.neighbour_version == 'one_on':
            active_indexes = np.eye(num_indexes)[:num_samples]
        elif self.neighbour_version == 'one_off':
            active_indexes = (np.ones(num_indexes)-np.eye(num_indexes))[:num_samples]
        elif self.neighbour_version == 'random_normal':
            active_indexes = self._normal_tokens(num_samples, num_indexes, self.neighbour_parameter)
        for act_idx in active_indexes:
            inactive = np.argwhere(act_idx == 0)
            sample = copy.deepcopy(token_string)
            for inact in inactive:
                sample[inact[0]] = self.inactive_string
            neighborhood_data.append(sample)
        print(neighborhood_data)
        return neighborhood_data, active_indexes

    def explain_instance(self, text, main_model, labels=(0,1,2,3), num_features=100000, num_samples=1000):
        """
            text: numpy array of a single image (RGB or Grayscale)
            main_model: callable object or function returning prediction of an image
            labels: iterable with labels to be explained
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the simple model

            returns: ImageExplainer object with convenient access to instance explainations
        """
        token_string = self.indexer_fn(text)
        neigh_data, active_words = self._neighborhood_generation(token_string, num_samples)
        neigh_weights = self._kernel_fn(active_words)
        neigh_labl = []
        for neigh in neigh_data:
            neigh_text = ' '.join(neigh)
            neigh_labl.append(main_model(neigh_text))
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
