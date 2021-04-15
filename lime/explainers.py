from abc import ABC
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

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
