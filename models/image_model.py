import torch
import torch.nn.functional as F

from torch import nn, optim
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.mnist_test = torchvision.datasets.FashionMNIST(
            self.data_dir, download=True, train=False, 
            transform=transforms.Compose([transforms.ToTensor()]))
        mnist_full = torchvision.datasets.FashionMNIST(
            self.data_dir, download=True, train=True, 
            transform=transforms.Compose([transforms.ToTensor()]))
        # Train / Validation Set split
        threshold = round(len(mnist_full) * 0.7)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [threshold, len(mnist_full) - threshold])

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )


class CNN_Net(nn.Module):
    def __init__(self, output_dims, cnn_dims, dropout):
        super(CNN_Net, self).__init__()

        conv_layers = []
        input_channels = 1
        for cnn_dim in cnn_dims:
            cnn_dim = int(cnn_dim)  # discrete uniform somehow outputs float
            conv_layers.append(
                nn.Conv2d(in_channels=input_channels, out_channels=cnn_dim, kernel_size=3, padding=1)
            )
            conv_layers.append(nn.BatchNorm2d(cnn_dim))
            conv_layers.append(nn.Dropout2d(dropout))
            input_channels = cnn_dim
        self.conv_layers = nn.Sequential(*conv_layers)

        layers = []
        input_features = self.conv_output() 
        for output_dim in output_dims:
            output_dim = int(output_dim)
            layers.append(
                nn.Linear(in_features=input_features, out_features=output_dim)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_features = output_dim

        layers.append(nn.Linear(input_features, 10))
        self.lin_layers = nn.Sequential(*layers)

    def conv_output(self):
        # helper function to calculate output dims of convolution layer
        x_dummy = torch.zeros_like(torch.empty(32, 1, 28, 28))
        x_dummy = self.conv_layers(x_dummy)
        return x_dummy.view(x_dummy.size(0), -1).shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.lin_layers(x)
        return x


class LightningCNN_Net(pl.LightningModule):
    def __init__(self, output_dims, cnn_dims, dropout, lr):
        super().__init__()
        self.model = CNN_Net(output_dims, cnn_dims, dropout)
        self.lr = lr
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        self.log('training_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1)
        self.val_acc.update(pred, target)

    def validation_epoch_end(self, outputs):
        accuracy = self.val_acc.compute()
        self.log("val_acc", accuracy)
        self.log("hp_metric", accuracy)

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        self.test_acc.update(pred, target)

    def test_epoch_end(self, outputs):
        self.log("test_acc", self.test_acc.compute())

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)
