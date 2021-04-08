from collections import Counter

import torch
import torch.nn.functional as F

from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

import pytorch_lightning as pl


class MyCollator:
    # Helper class for batch preprocessing (a more flexible collate_fn)
    def __init__(self, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __call__(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(int(_label) - 1)
            _text = [self.vocab[token] for token in self.tokenizer(_text)]
            processed_text = torch.tensor(_text, dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list, text_list, offsets


class AGNewsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.data_test = list(AG_NEWS(split='test'))
        data_full = list(AG_NEWS(split='train'))
        # Train / Validation Set split
        threshold = round(len(data_full) * 0.8)
        self.data_train, self.data_val = random_split(
            data_full, [threshold, len(data_full) - threshold])
        # Vocab and Tokenizer for data processing in collate batch
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = self.get_vocab(data_full, self.tokenizer)
        self.collate_batch = MyCollator(self.vocab, self.tokenizer)

    def get_vocab(self, data_full, tokenizer):
        counter = Counter()
        for (label, line) in data_full:
            counter.update(tokenizer(line))
        return Vocab(counter, min_freq=1)

    def train_dataloader(self):
        return DataLoader(
            self.data_train, batch_size=self.batch_size, shuffle=True, pin_memory=True,
            collate_fn=self.collate_batch
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val, batch_size=self.batch_size, shuffle=False, pin_memory=True,
            collate_fn=self.collate_batch
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test, batch_size=self.batch_size, shuffle=False, pin_memory=True,
            collate_fn=self.collate_batch
        )


class Text_Net(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dims, dropout):
        super(Text_Net, self).__init__()
        embed_dim = int(embed_dim)
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)

        layers = []
        input_features = embed_dim
        for dim in output_dims:
            dim = int(dim)
            layers.append(nn.Linear(input_features, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_features = dim

        layers.append(nn.Linear(input_features, 4))
        self.lin_layers = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5

        def init_linear(layer):
            if type(layer) == nn.Linear:
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.lin_layers.apply(init_linear)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.lin_layers(embedded)


class LightningText_Net(pl.LightningModule):
    def __init__(self, vocab_size, embed_dim, output_dims, dropout, lr):
        super().__init__()
        self.model = Text_Net(vocab_size, embed_dim, output_dims, dropout)
        self.lr = lr
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, text, offsets):
        return self.model(text, offsets)

    def training_step(self, batch, batch_idx):
        label, text, offsets = batch
        output = self(text, offsets)
        loss = F.cross_entropy(output, label)
        self.log('training_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        label, text, offsets = batch
        output = self(text, offsets)
        pred = output.argmax(dim=1)
        self.val_acc.update(pred, label)

    def validation_epoch_end(self, outputs):
        accuracy = self.val_acc.compute()
        self.log("val_acc", accuracy)
        self.log("hp_metric", accuracy)

    def test_step(self, batch, batch_idx):
        label, text, offsets = batch
        output = self(text, offsets)
        pred = output.argmax(dim=1, keepdim=True)
        self.test_acc.update(pred, label)

    def test_epoch_end(self, outputs):
        self.log("test_acc", self.test_acc.compute())

    def configure_optimizers(self):
        # Adam optimizer doesn't work on sparse emedding-bag layer
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, threshold=0.00001
        )
        # Weird... https://github.com/PyTorchLightning/pytorch-lightning/issues/2976#issuecomment-680219124
        # val_checkpoint_on in monitor wont work
        # val_acc in monitor works only if reduce_on_plateau is true
        scheduler = {'scheduler': lr_scheduler, 'reduce_on_plateau': True, 'monitor': 'val_acc'}
        return [optimizer], [scheduler]
