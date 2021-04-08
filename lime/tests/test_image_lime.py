import torch
import pytest
import numpy as np
from PIL import Image
from functools import partial
from torchvision import transforms


@pytest.fixture(scope="module")
def pytorch_model():
    """Loads and returns pytorch model only once (shufflenet was chosen due to its speed)"""
    model = torch.hub.load('pytorch/vision:v0.9.0', 'shufflenet_v2_x1_0', pretrained=True)
    model.eval()

    if torch.cuda.is_available():
        model.cuda()
    return model


def run_pytorch_model(model, input_image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = Image.fromarray(input_image)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch.cuda()
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities.detach().numpy()


def run_dummy_model(image):
    return 

def run_dummy_model_multiclass(image):
    pass


class Test_Image_Lime:
    def test_dummy_rgb(self):
        image = rgb
        lime = ImageLime()
        lime.explain_instance(self, image, run_dummy_model)

    def test_dummy_grayscale(self):
        image = grayscale
        lime = ImageLime()
        lime.explain_instance(self, image, run_dummy_model)

    def test_dummy_rgb_multiclass(self):
        image = grayscale
        lime = ImageLime()
        lime.explain_instance(self, image, run_dummy_model_multiclass)

    def test_pytorch_rgb_ridge(self, pytorch_model):
        model = partial(run_pytorch_model, pytorch_model)
        lime = ImageLime()
        lime.explain_instance(self, image, model)

    def test_pytorch_rgb_svm(self, pytorch_model):
        model = partial(run_pytorch_model, pytorch_model)
        lime = ImageLime()
        lime.explain_instance(self, image, model)

    def test_pytorch_rgb_tree(self, pytorch_model):
        model = partial(run_pytorch_model, pytorch_model)
        lime = ImageLime()
        lime.explain_instance(self, image, model)
