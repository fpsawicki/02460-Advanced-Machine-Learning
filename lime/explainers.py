from abc import ABC
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import numpy as np

class Explainer(ABC):
    def __init__(self):
        pass

    def visualize(self):
        pass

    def describe(self):
        pass


class ImageExplainer(Explainer):
    def __init__(self, image, segments, results):
        self.image = image
        self.segs = segments
        self.results = results

    def visualize(self, n_top_features, label):
        def get_seg_x(seg, x):
            return (seg == x) * 1
        
        plt.figure
        plt.imshow(mark_boundaries(self.image, self.segs))
        plt.show()
        
        top_features = self.results[label]['feature_importance'][:n_top_features]

        top_indexes = []
        for feature in top_features:
            top_indexes.append(feature[0])

        explained_image = 0
        for i in range(len(top_indexes)):
            explained_image += get_seg_x(self.segs, top_indexes[i])

        img_to_show = self.image.copy()
        img_to_show[explained_image == 0] = 0
        plt.figure()
        plt.imshow(img_to_show)
        plt.show()

        return img_to_show
    
    def liqud_visualize(self, label, pos_color = (153, 255, 153), neg_color = (255, 77, 77)):
        np_result = np.array(self.results[label]['feature_importance'][:])
        
        pos_idx=[]
        neg_idx=[]
        pos_coef=[]
        neg_coef=[]
        for i in range(len(np_result)):
          if np_result[i,1]>0:
              pos_idx.append(np_result[i,0])
              pos_coef.append(np_result[i,1])
          else:
              neg_idx.append(np_result[i,0])
              neg_coef.append(np_result[i,1])
      
        pos_idx=np.array(pos_idx)
        neg_idx=np.array(neg_idx)
        pos_coef=np.array(pos_coef)
        neg_coef=np.array(neg_coef)
        
        norm = 1/pos_coef.ravel().max()
        pos_coef = pos_coef.ravel()*norm
        norm = 1/neg_coef.ravel().min()
        neg_coef = neg_coef.ravel()*norm
      
        # Positive part 
        color = [col/255 for col in pos_color]
        color = np.array([np.ones_like(self.segs)*color[0], np.ones_like(self.segs)*color[1], np.ones_like(self.segs)*color[2]])
        color = np.moveaxis(color, 0, -1)
      
        seg_pos = np.array([(self.segs == idx)*abs(pos_coef[i]) for i, idx in enumerate(pos_idx)]).sum(axis = 0)
        img_pos = np.array([seg_pos, seg_pos, seg_pos])
        img_pos = np.moveaxis(img_pos, 0, -1)
        img_pos = np.round(self.image*img_pos*color).astype(int)
      
        # Negative part 
        color = [col/255 for col in neg_color]
        color = np.array([np.ones_like(self.segs)*color[0], np.ones_like(self.segs)*color[1], np.ones_like(self.segs)*color[2]])
        color = np.moveaxis(color, 0, -1)
      
        seg_neg = np.array([(self.segs == idx)*abs(neg_coef[i]) for i, idx in enumerate(neg_idx)]).sum(axis = 0)
        img_neg = np.array([seg_neg, seg_neg, seg_neg])
        img_neg = np.moveaxis(img_neg, 0, -1)
        img_neg = np.round(self.image*img_neg*color).astype(int)
        
        plt.imshow(img_pos+img_neg)
        plt.show()
  
    def describe(self):
        return self.results


class TextExplainer(Explainer):
    def __init__(self, text, active_words, results):
        self.text = text
        self.active_words = active_words
        self.results = results

    def visualize(self):
        pass

    def describe(self):
        return self.results
