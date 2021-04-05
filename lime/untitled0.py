# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 16:45:03 2021

@author: javig
"""

import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from lime_image import ImageLIME
from skimage import io
from PIL import Image

im = io.imread('C:/Users/javig/Desktop/index.jpg')

model = torchvision.models.inception_v3(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def cnn_predict(input):
  'Returns the predicted label encoded as integer'
  input = Image.fromarray(input.astype('uint8'), 'RGB')
  input = preprocess(input)
  input = input.unsqueeze(0) # Returns a new tensor with a dimension of size one inserted at the specified position: [1,3,299,299]

  output = model(input)

  # The output has unnormalized scores. To get probabilities, run a softmax on it.
  probabilities = torch.nn.functional.softmax(output[0], dim=0)
  top_prob, top_catid = torch.topk(probabilities, 1)
 
  return probabilities.detach().numpy()

explainer = ImageLIME()

image_explainer = explainer.explain_instance(im, cnn_predict, labels = (0,1))

image_explainer.visualize(5, 1)

print(image_explainer.describe())


