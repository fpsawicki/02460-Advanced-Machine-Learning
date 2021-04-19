from abc import ABC
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries
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

    def visualize_text(self, labels=None, min_importance=0.01):
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

    def visualize_words(self, labels=None, n_top_words=5):
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
            coefs = sorted(coefs, key=lambda row: np.abs(row[1]))[:n_top_words]

            words = [self.text[c[0]] for c in coefs]
            print(words)
            importances = np.array([c[1] for c in coefs])
            x = np.arange(len(words))

            n_col = (idx % 3)
            n_row = (idx - 1) // 3

            mask_1 = importances >= 0
            mask_2 = importances < 0

            ax[n_row, n_col].bar(x[mask_1] - 0.35, importances[mask_1], 0.35, label='Word', color='green')
            ax[n_row, n_col].bar(x[mask_2] - 0.35, importances[mask_2], 0.35, label='Word', color='red')
            ax[n_row, n_col].axhline(0, color='black')

            ax[n_row, n_col].set_xticks(np.arange(len(words)))
            ax[n_row, n_col].set_xticklabels(words, rotation=(45))
            ax[n_row, n_col].set_title(f'{labels[label_id]}')
            ax[n_row, n_col].set_visible(True)

        fig.tight_layout()
        plt.show()

    def describe(self):
        return self.results
