from abc import ABC

import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries
import numpy as np
from IPython.core.display import HTML


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

        # plt.figure
        # plt.imshow(mark_boundaries(self.image, self.segs))
        # plt.show()

        top_features = self.results[label]['feature_importance'][:n_top_features]

        top_indexes = []
        for feature in top_features:
            top_indexes.append(feature[0])

        explained_image = 0
        for i in range(len(top_indexes)):
            explained_image += get_seg_x(self.segs, top_indexes[i])

        img_to_show = self.image.copy()
        img_to_show[explained_image == 0] = 0
        # plt.figure()
        # plt.imshow(img_to_show)
        # plt.show()

        return img_to_show
    
    def liqud_visualize(self, label, pos_color = (153, 255, 153), neg_color = (255, 77, 77), retur = False):
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
        
        colored_explanation = img_pos+img_neg
        colored_explanation = mark_boundaries(colored_explanation.astype(np.uint8), self.segs)
        if retur==False:
            plt.imshow(colored_explanation)
            plt.show()
        else:
            return colored_explanation
            
        
  
    def describe(self):
        return self.results


class TextExplainer(Explainer):
    def __init__(self, text, active_words, results):
        self.text = text
        self.active_words = active_words
        self.results = results

    def visualize_text(self, labels=None, min_importance=0.01):
        """
            labels: dict where key is index of prediction class and value is description of that class.
                    - this can be a subset of all labels if ex. one class is of interest
                    - default value will print all labels in prediction without descriptions
            min_importance: float minimum absolute value of coefficient below which text is black

            returns: HTML object with text coloring to be viewed in jupyter notebook
        """
        text = []
        if not labels:
            labels = {i: f'Label #{i}' for i in self.results.keys()}
        for label_id in labels.keys():
            text.append(f"<h1>{labels[label_id]}</h1>")
            for idx, word in enumerate(self.text):
                coef = [i[1] for i in self.results[label_id]['feature_importance'] if i[0] == idx]
                if not coef:
                    color = 'black'
                else:
                    color = 'red' if coef[0] < 0 else 'green'
                    color = 'black' if abs(coef[0]) < min_importance else color
                text.append(f"<span style='color: {color}'>{word}</span>")
        return HTML(' '.join(text))

    def visualize_words(self, labels=None, n_top_words=5, show_coefs='all'):
        """
            labels: dict where key is index of prediction class and value is description of that class.
                    - this can be a subset of all labels if ex. one class is of interest
                    - default value will print all labels in prediction without descriptions
            n_top_words: int number of words with highest absolute value in coefficients
            show_coefs: str (all, negative, positive) to show only coefficients with provided sign

            returns: pyplot with word importance per each label
        """
        if not labels:
            labels = {i: f'Label #{i}' for i in self.results.keys()}

        total_labels = len(labels.keys())
        n_cols = 3 if total_labels >= 3 else total_labels % 3
        n_rows = 1 if total_labels <= 1 else (total_labels + 2) // 3

        fig, ax = plt.subplots(n_rows, n_cols, sharex=False, sharey='all')
        fig.suptitle('Word Importances')

        def set_invisible(x):
            x.set_visible(False)
        vf = np.vectorize(set_invisible)
        vf(ax)  # sets all subplots as invisible

        for idx, label_id in enumerate(labels.keys()):
            coefs = self.results[label_id]['feature_importance']
            if show_coefs == 'positive':
                coefs = [c for c in coefs if c[1] >= 0]
            if show_coefs == 'negative':
                coefs = [c for c in coefs if c[1] <= 0]
            coefs = sorted(coefs, key=lambda row: np.abs(row[1]), reverse=True)[:n_top_words]

            words = [self.text[c[0]] for c in coefs]
            importances = np.array([c[1] for c in coefs])
            x = np.arange(len(words))

            n_col = (idx % 3)
            n_row = idx // 3

            mask_1 = importances >= 0
            mask_2 = importances < 0

            plot = None
            if (n_rows == 1) and (n_cols == 1):
                plot = ax
            elif (n_rows == 1):
                plot = ax[n_col]
            else:
                plot = ax[n_row, n_col]

            plot.bar(x[mask_1], importances[mask_1], 0.35, label='Word', color='green')
            plot.bar(x[mask_2], importances[mask_2], 0.35, label='Word', color='red')
            plot.axhline(0, color='black')

            plot.set_xticks(np.arange(len(words)))
            plot.set_xticklabels(words, rotation=(45))
            plot.set_title(f'{labels[label_id]}')
            plot.set_visible(True)

        fig.tight_layout()
        plt.show()

    def describe(self):
        return self.results
