import re

import matplotlib as plt
import numpy as np
import pandas as pd
import torch
from gensim import downloader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from preprocessing import NERDataset

# -------------------------------
# Define the Nueral Network Model
# -------------------------------


class NER_NN(nn.Module):
    # TODO: change hidden dim?
    def __init__(self, input_size, num_classes, hidden_dim=100):
        super(NER_NN, self).__init__()
        self.first_layer = nn.Linear(input_size, hidden_dim)
        # TODO: add layer? (hidden, hidden)
        self.second_layer = nn.Linear(hidden_dim, num_classes)
        # TODO: check also other activations (tanh?)
        self.activation = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()  # for classification

    def forward(self, input_ids, labels=None):
        # NOTE: question - where do we call this function?
        x = self.first_layer(input_ids)
        x = self.activation(x)
        x = self.second_layer(x)
        if labels is None:
            return x, None
        loss = self.loss(x, labels)
        return x, loss


# -------------------------
# Train loop
# -----------------------


def train_and_plot(NN_model, train_loader, num_epochs: int, batch_size: int):

    # -------
    # GPU
    # -------

    # First checking if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("Training on GPU.")
    else:
        print("No GPU available, training on CPU.")

    NN_model.to(device)

    # ----------------------------------
    # Define Optimizer and Loss Function
    # ----------------------------------
    # many are available such as SGD, Adam, RMSprop, Adagrad..
    # TODO: change optimizer?
    optimizer = torch.optim.SGD(params=NN_model.parameters(), lr=0.2)
    clip = 1000  # gradient clipping
    # optimizer = torch.optim.Adam(params=NN_model.parameters())

    loss_func = torch.nn.CrossEntropyLoss()

    # ----------------------------------
    # Epoch Loop
    # ----------------------------------

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        y_true = []
        y_pred = []
        loss_batches_list = []
        f1_batches_list = []
        accuracy_batches_list = []

        for batch_num, (inputs, labels) in enumerate(train_loader):
            # if training on gpu
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = labels.shape[0]

            # optimize
            optimizer.zero_grad()

            # output
            # IMPORTANT - change the dimensions of x before it enters the NN,
            # batch size must always be first
            x = inputs.unsqueeze(0)  # x.size() -> [1, batch_size]
            x = x.view(batch_size, -1)  # x.size() -> [batch_size, 1]
            outputs = NN_model(x)

            # calculate the loss and perform backprop
            loss = loss_func(outputs.squeeze(), labels.float())
            loss_batches_list.append(loss)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(NN_model.parameters(), clip)
            optimizer.step()

            # predictions
            preds = outputs.argmax(dim=-1).clone().detach().cpu()
            y_true.append(labels.cpu().view(-1))
            y_pred.append(preds.view(-1))

            if batch_num % 1000 == 0:
                cur_accuracy = accuracy_score(
                    labels.cpu().view(-1), preds.view(-1), normalize=False
                )
                accuracy_batches_list.append(cur_accuracy)
                cur_f1 = f1_score(
                    labels.cpu().view(-1), preds.view(-1), normalize=False
                )
                f1_batches_list.append(cur_f1)

                plot_loss(inputs, labels, outputs, loss_batches_list)

        print_epoch_results(num_epochs, epoch, y_true, y_pred, loss_batches_list)


def print_epoch_results(num_epochs, epoch, y_true, y_pred, loss_batches_list):
    loss_batches = np.array(loss_batches_list)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    accuracy = accuracy_score(y_true, y_pred, normalize=False)
    f1 = f1_score(y_true, y_pred, normalize=False)
    print(
        "Epoch: {}/{}...".format(epoch + 1, num_epochs),
        "Avg Loss: {:.6f}...".format(loss_batches.mean()),
        "Accuracy: {:.3f}".format(accuracy),
        "F1: {:.3f}".format(f1),
    )


def plot_loss(inputs, labels, outputs, losses):
    # plot to show learning process
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.cla()
    ax.scatter(inputs.cpu().data.numpy(), labels.cpu().data.numpy())
    ax.plot(
        inputs.cpu().data.numpy(),
        outputs.cpu().data.numpy(),
        "r-",
        lw=2,
    )
    ax.text(
        0.5,
        0,
        "Loss=%.4f" % np.mean(losses),
        fontdict={"size": 10, "color": "red"},
    )
    plt.pause(0.1)
    ax.clear()
    plt.show()


# -------------------------
# Putting it all together
# -------------------------
embedding_type = "glove"
NER_dataset = NERDataset(embedding_model_type=embedding_type)
train_loader, dev_loader, test_loader = NER_dataset.get_preprocessed_data()

# TODO: change params?

# option 1:
# is identity - yes / no
num_classes = 2
# option 2:
# different classification for differnet identities
# num_classes = ?

batch_size = 32
num_epochs = 15
hidden_dim = 100
# TODO: change according to the embedding
input_size = 125  # single vector size


NN_model = NER_NN(input_size=input_size, num_classes=num_classes, hidden_dim=hidden_dim)

train_and_plot(
    NN_model=NN_model,
    train_loader=train_loader,
    num_epochs=num_epochs,
    batch_size=batch_size,
)
