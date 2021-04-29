from lime.lime_base import BaseLIME
from lime.explainers import ImageExplainer
from lime.segmentations import QuickShift, Segmentation
from skimage.measure import regionprops
import numpy as np
from sklearn.metrics import pairwise_distances
from skimage.color import gray2rgb


class ImageLIME(BaseLIME):
    KERNEL_MULTIPLIER = 0.75

    def __init__(self,
                 random_state=123,
                 simple_model=None,
                 kernel_width=None,
                 kernel_active=True,
                 segmentation=None,
                 alpha_penalty=None,
                 distance_metric='cosine',
                 feature_selection='highest_weights'):
        """
            random_state: integer randomness seed value
            simple_model: sklearn model for local explainations
            kernel_width: float
            kernel_active: bool if false return 1 for all neighbourhood distances
            segmentation: object of subclass Segmentation with callable segmentation function
            alpha_penalty: float L2 penalty term in ridge model for feature selection
            distance_metric: str type of metric used in kernel
            feature_selection: str type of feature selection method
        """
        self.base = BaseLIME(random_state, alpha_penalty)
        self.simple_model = simple_model
        self.kernel_width = kernel_width
        self.kernel_active = kernel_active
        self.distance_metric = distance_metric
        self.feature_selection = feature_selection

        if not segmentation:
            # hyperparameters from baseline implementation
            self.segmentation_fn = QuickShift(
                kernel_size=4, max_dist=200, ratio=0.2, random_seed=random_state)
        else:
            self.segmentation_fn = segmentation
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

    def _neighborhood_generation_random(self, instance, segmentation, num_samples):
        # Randomly turning each pixel on and off
        
        def get_seg_x(seg, x):
            return (seg == x) * 1

        neighborhood_data = []
        num_segmemts = np.unique(segmentation).shape[0]
        active_segments = np.random.binomial(1, 0.5, size=(num_samples, num_segmemts))
        for k in range(num_samples):
            active = np.argwhere(active_segments[k])
            sample = instance * 0
            for i in active:
                sample = sample + get_seg_x(segmentation, i)[:, :, np.newaxis] * instance
            neighborhood_data.append(sample)

        return np.array(neighborhood_data), active_segments
    
    def _neighborhood_generation_one(self, instance, segmentation, num_samples):
        # Randomly turning one pixel on at a time
        
        # Instance the image whoes prediction we want to explain
        # Segmentation (fx from quickshift(im))
        # num_samples the number of samples
        
        def get_seg_x(seg, x):
            return (seg == x) * 1
        
        num_segments = np.unique(segmentation).shape[0]
        
        # Get the samples
        neighborhood_data = []
        active_segments = np.zeros((num_samples,num_segments))
        active_segments[:,0] = 1
        np.apply_along_axis(np.random.shuffle,1,active_segments) 
      
        for i in range(active_segments.shape[0]):
          main_pixel = np.where(active_segments[i] == 1)[0][0]
      
        for k in range(num_samples):
          active = np.argwhere(active_segments[k])
          sample = instance*0
          for i in active:
            sample = sample + get_seg_x(segmentation,i)[:,:,np.newaxis]*instance
          neighborhood_data.append(sample)
      
        # Get lables for the samples 
        #neighborhood_labels = model_predict(neighborhood_data)
      
        return np.array(neighborhood_data), active_segments
    
    def _neighborhood_generation_radio(self, instance, segmentation, num_samples, radio):
        # Instance the image whoes prediction we want to explain
        # Segmentation (fx from quickshift(im))
        # num_samples the number of samples
        
        def get_seg_x(seg, x):
            return (seg == x) * 1
      
        segmentation_plus1 = segmentation + 1 # because regionprops ignores label 0, so it yields 1 superpixel less
        seg_3d = np.moveaxis(np.array([segmentation_plus1,segmentation_plus1,segmentation_plus1]),0,-1)
        properties = regionprops(seg_3d, instance)
        coords = np.array([properties[i].weighted_centroid for i in range(len(properties))]).T
        x_s = coords[1]
        y_s = coords[0]
      
        # plt.imshow(seg)
        # plt.scatter(y = y_s, x = x_s)
      
        x_s = x_s.reshape((1,x_s.shape[0]))
        y_s = y_s.reshape((1,y_s.shape[0]))
      
        x_s_ax1 = np.tile(x_s.T,(1,x_s.shape[1])) # matrix with x_s repeated along axis 1 direction (x_s is in column vector shape)
        x_s_ax0 = np.tile(x_s,(x_s.shape[1],1)) # x_s repeated along axis 0 direction (x_s is in row vector shape)#
        y_s_ax1 = np.tile(y_s.T,(1,y_s.shape[1])) # y_s repeated along axis 1 direction (y_s is in column vector shape)
        y_s_ax0 = np.tile(y_s,(y_s.shape[1],1)) # y_s repeated along axis 0 direction (y_s is in in row vector shape)
      
        x_dist = x_s_ax1 - x_s_ax0
        y_dist = y_s_ax1 - y_s_ax0
      
        dist = np.sqrt(np.power(x_dist,2) + np.power(y_dist,2))
      
        num_segments = np.unique(segmentation).shape[0]
      
        # Get the samples
        neighborhood_data = []
        active_segments = np.zeros((num_samples,num_segments))
        active_segments[:,0] = 1
        np.apply_along_axis(np.random.shuffle,1,active_segments)
      
        for i in range(active_segments.shape[0]):
          main_pixel = np.where(active_segments[i] == 1)[0][0]
          neighboring_pixels = np.where(dist[main_pixel,:] <= radio)[0]
          active_segments[i,neighboring_pixels] = 1
      
        for k in range(num_samples):
          active = np.argwhere(active_segments[k])
          sample = instance*0
          for i in active:
            sample = sample + get_seg_x(segmentation,i)[:,:,np.newaxis]*instance
          neighborhood_data.append(sample)
      
        # Get lables for the samples 
        #neighborhood_labels = model_predict(neighborhood_data)
      
        return neighborhood_data, active_segments# neighborhood_labels

    def explain_instance(self, image, main_model, neighborhood_type='random', radio=None, segs=None, labels=(0,), num_features=100000, num_samples=50):
        """
            image: numpy array of a single image (RGB or Grayscale)
            main_model: callable object or function returning prediction of an image
            neighborhood_type: techinque used to generate neighbors
            radio: only when neighborhood_type = 'radio'. Radio for generating the neighbors
            segs: numpy array with image segmentations
            labels: iterable with labels to be explained
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the simple model

            returns: ImageExplainer object with convenient access to instance explainations
        """
        if segs is None:
            segs = self.segmentation_fn(image)  # segmentations before rgb2gray (some algorithms require 3 chanels)
        
        if neighborhood_type == 'random':
             neigh_data, active_segs = self._neighborhood_generation_random(image, segs, num_samples)
        elif neighborhood_type == 'one':
             neigh_data, active_segs = self._neighborhood_generation_one(image, segs, num_samples)
        elif neighborhood_type == 'radio':
             neigh_data, active_segs = self._neighborhood_generation_radio(image, segs, num_samples, radio) 
        
        # check if rgb then change to grayscale
        if (len(image.shape) == 2):
            image = gray2rgb(image)
        # neigh_data, active_segs = neigh_function()
        neigh_weights = self._kernel_fn(segs, active_segs)
        if not self.kernel_active:
            neigh_weights = np.ones_like(neigh_weights)
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
