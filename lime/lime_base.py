import warnings
import numpy as np
import scipy as sp

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone
from sklearn.svm import LinearSVR  # other SVM methods have no _coefs


class BaseLIME:
    MODELS = [Ridge, Lasso, DecisionTreeRegressor, LinearSVR, RandomForestRegressor]
    METHODS = [None, 'forward_selection', 'highest_weights', 'lars_path']

    def __init__(self, random_state=123, alpha_penalty=None):
        """
            random_state: integer seed value for randomness
            alpha_penalty: float L2 penalty term in ridge model for feature selection
        """
        self.random_state = random_state
        self.alpha_penalty = alpha_penalty

    def _forward_selection(self, data, labels, weights, num_features):
        penalty = (self.alpha_penalty if self.alpha_penalty else 0)  # try different alpha?
        clf = Ridge(alpha=penalty, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]],
                        labels, sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels, sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def _highest_weights(self, data, labels, weights, num_features):
        penalty = (self.alpha_penalty if self.alpha_penalty else 0.01)  # try different alpha?
        clf = Ridge(alpha=penalty, fit_intercept=True, random_state=self.random_state)
        clf.fit(data, labels, sample_weight=weights)
        coef = clf.coef_
        if sp.sparse.issparse(data):
            # Optimization for sparse data
            coef = sp.sparse.csr_matrix(clf.coef_)
            weighted_data = coef
            sdata = len(weighted_data.data)
            argsort_data = np.abs(weighted_data.data).argsort()
            if sdata < num_features:
                nnz_indexes = argsort_data[::-1]
                indices = weighted_data.indices[nnz_indexes]
                num_to_pad = num_features - sdata
                indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                indices_set = set(indices)
                pad_counter = 0
                for i in range(data.shape[1]):
                    if i not in indices_set:
                        indices[pad_counter + sdata] = i
                        pad_counter += 1
                        if pad_counter >= num_to_pad:
                            break
            else:
                nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                indices = weighted_data.indices[nnz_indexes]
            return indices
        else:
            weighted_data = coef
            feature_weights = sorted(
                zip(range(data.shape[1]), weighted_data),
                key=lambda x: np.abs(x[1]), reverse=True)
            return np.array([x[0] for x in feature_weights[:num_features]])

    def _feature_selection(self, data, labels, weights, num_features, method):
        assert method in BaseLIME.METHODS, f'Invalid method type choose from: {BaseLIME.METHODS}'
        if method is None:
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self._forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            return self._highest_weights(data, labels, weights, num_features)
        elif method == 'lars_path':
            raise Exception('Not Implemented')

    def explain_instance(self, neighborhood_data, neighborhood_weights, neighborhood_labels,
                         label, num_features, feature_selection, simple_model=None):
        if simple_model is None:
            simple_model = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)
        assert type(simple_model) in BaseLIME.MODELS, f'Invalid simple_model type choose from: {BaseLIME.MODELS}'
        if simple_model.random_state != self.random_state:
            warnings.warn('random_state of a simple_model is not equal to LIME random_state!')
        labels_column = neighborhood_labels[:, label]
        used_features = self._feature_selection(
            neighborhood_data, labels_column,
            neighborhood_weights, num_features, feature_selection)

        # makes sure that weights are not passed in multiclass classification
        simple_model = clone(simple_model)

        simple_model.fit(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=neighborhood_weights)
        prediction_score = simple_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=neighborhood_weights)

        local_pred = simple_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
        if type(simple_model) in [DecisionTreeRegressor, RandomForestRegressor]:
            feature_importance = sorted(zip(used_features, simple_model.feature_importances_),
                                        key=lambda x: np.abs(x[1]), reverse=True)
            simple_model.intercept_ = np.zeros(neighborhood_labels.shape[1])
        else:
            feature_importance = sorted(zip(used_features, simple_model.coef_),
                                        # key=lambda x: np.abs(x[1]), reverse=True)
                                        # Changed to take into account only positive coefficients
                                        key=lambda x: x[1], reverse=True)
        return (simple_model.intercept_, feature_importance, prediction_score, local_pred)
