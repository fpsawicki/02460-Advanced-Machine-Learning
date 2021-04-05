from abc import ABC


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

    def visualize(self):
        pass

    def describe(self):
        return self.results


class TextExplainer(Explainer):
    def __init__(self, image, segments):
        pass

    def visualize(self):
        pass

    def describe(self):
        pass
