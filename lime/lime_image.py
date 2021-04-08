from lime_base import BaseLIME
from segmentations import QuickShift, Segmentation
from explainers import ImageExplainer

import numpy as np

from sklearn.metrics import pairwise_distances
from skimage.color import gray2rgb


class ImageLIME(BaseLIME):
    KERNEL_MULTIPLIER = 0.75

    def __init__(self,
                 random_state=123,
                 simple_model=None,
                 kernel_width=None,
                 segmentation=None,
                 alpha_penalty=None,
                 distance_metric='cosine',
                 feature_selection='highest_weights'):
        """
            random_state: integer randomness seed value
            simple_model: sklearn model for local explainations
            kernel_width: float
            segmentation: object of subclass Segmentation with callable segmentation function
            alpha_penalty: float L2 penalty term in ridge model for feature selection
            distance_metric: str type of metric used in kernel
            feature_selection: str type of feature selection method
        """
        self.base = BaseLIME(random_state, alpha_penalty)
        self.simple_model = simple_model
        self.kernel_width = kernel_width
        self.distance_metric = distance_metric
        self.feature_selection = feature_selection

        if not segmentation:
            # hyperparameters from baseline implementation
            self.segmentation_fn = QuickShift(
                kernel_size=4, max_dist=200, ratio=0.2, random_seed=random_state)
        # msg = 'Invalid segmentation type, use one implemented in segmentations.py'
        # assert issubclass(Segmentation, segmentation), msg

    def _kernel_fn(self, segmentation, active_segments):
        kernel = self.kernel_width
        if kernel is None:
            kernel = np.sqrt(len(active_segments)) * ImageLIME.KERNEL_MULTIPLIER
        kernel = float(kernel)

        num_segments = np.unique(segmentation).shape[0]
        all_segments = np.ones(num_segments)[np.newaxis, :]

        distances = pairwise_distances(
            active_segments,
            all_segments,
            metric=self.distance_metric).ravel()

        kernel = np.sqrt(np.exp(-(distances ** 2) / kernel ** 2))
        return kernel

    def _neighborhood_generation(self, instance, segmentation, num_samples):
        def get_seg_x(seg, x):
            return (seg == x) * 1

        neighborhood_data = []
        num_segmemts = np.unique(segmentation).shape[0]
        active_segments = np.random.binomial(1, 0.5, size=(num_samples, num_segmemts))
        for k in range(num_samples):
            active = np.argwhere(active_segments[k])
            sample = instance * 0
            for i in active:
                sample = sample + get_seg_x(segmentation, i)[:,:,np.newaxis] * instance
            neighborhood_data.append(sample)

        return np.array(neighborhood_data), active_segments

    def explain_instance(self, image, main_model, labels=(1,), num_features=100000, num_samples=50):
        """
            image: numpy array of a single image (RGB or Grayscale)
            main_model: callable object or function returning prediction of an image
            labels: iterable with labels to be explained
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the simple model

            returns: ImageExplainer object with convenient access to instance explainations
        """
        segs = self.segmentation_fn(image)  # segmentations before rgb2gray (some algorithms require 3 chanels)
        # check if rgb then change to grayscale
        if (len(image.shape) == 2):
            image = gray2rgb(image)
        neigh_data, active_segs = self._neighborhood_generation(image, segs, num_samples)
        neigh_weights = self._kernel_fn(segs, active_segs)
        neigh_labl = []
        for neigh in neigh_data:
            neigh_labl.append(main_model(neigh))

        neigh_labl = np.array(neigh_labl)

        results = {}
        for label in labels:
            res = self.base.explain_instance(
                active_segs, neigh_weights, neigh_labl, label, num_features,
                feature_selection=self.feature_selection, simple_model=self.simple_model
            )
            results[label] = {
                'intercept': res[0],
                'feature_importance': res[1],
                'prediction_score': res[2],
                'local_prediction': res[3]
            }
        return ImageExplainer(image, segs, results)
