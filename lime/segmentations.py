from skimage.segmentation import quickshift, slic
from functools import partial
from abc import ABC


class Segmentation(ABC):
    def __init__(self):
        pass

    def __call__(self, image):
        pass


class Grid(Segmentation):
    def __init__(self):
        pass

    def __call__(self, image):
        pass


class QuickShift(Segmentation):
    def __init__(self, **kwargs):
        self.function = partial(quickshift, **kwargs)

    def __call__(self, image):
        return self.function(image)


class Slic(Segmentation):
    def __init__(self, **kwargs):
        self.function = partial(slic, **kwargs)

    def __call__(self, image):
        return self.function(image)
