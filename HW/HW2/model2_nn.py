import re

import numpy as np
import pandas as pd
import torch
from gensim import downloader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

# -----------------------
# Define the data set
# -----------------------




# -------------------------------
# Define the Nueral Network Model
# -------------------------------


class NER_NN(nn.Module):
    # TODO: change hidden dim?
    def __init__(self, vocab_size, num_classes, hidden_dim=100):
        super(NER_NN, self).__init__()
        self.first_layer = nn.Linear(vocab_size, hidden_dim)
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

def train_and_plot(NN_model, data_sets, num_epochs: int, batch_size=32):

    # -------
    # GPU
    # -------

    # First checking if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("Training on GPU.")
    else:
        print("No GPU available, training on CPU.")

    # ---------
    # Load data
    # ---------

    # NOTE: question - what does the dataloader do?
    data_loaders = {
        "train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
        "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False),
    }
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

    # ----------------------
    # Define training params
    # ----------------------

    counter = 0
    print_every = 100

    best_f1 = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            # TODO: get y_true in numbers
            y_true = []
            y_pred = []

            for batch in data_loaders[phase]:
                batch_size = 0
                for k, v in batch.items():
                    batch[k] = v.to(device)
                    batch_size = v.shape[0]

                optimizer.zero_grad()
                if phase == "train":
                    outputs, loss = NN_model(**batch)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs, loss = NN_model(**batch)
                pred = outputs.argmax(dim=-1).clone().detach().cpu()

                cur_num_correct = f1_score(
                    batch["labels"].cpu().view(-1), pred.view(-1), normalize=False
                )

                running_loss += loss.item() * batch_size
                running_acc += cur_num_correct

            epoch_loss = running_loss / len(data_sets[phase])
            epoch_acc = running_acc / len(data_sets[phase])
            # TODO: calc f1
            # epoch_f1 = ?

            epoch_f1 = round(epoch_f1, 5)
            print(f"{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}")
            if phase == "test" and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                with open("NN_model.pkl", "wb") as f:
                    torch.save(NN_model, f)
        print()

    print(f"Best Validation F1: {best_f1:4f}")
    with open("NN_model.pkl", "rb") as f:
        NN_model = torch.load(f)
    return NN_model



for e in range(epochs):
    # batch loop
    for inputs, labels in train_loader:

        # if training on gpu
        inputs, labels = inputs.to(device), labels.to(device)

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        # x.size() -> [batch_size]
        batch_size = inputs.size(0)
        # IMPORTANT - change the dimensions of x before it enters the NN,
        # batch size must always be first
        x = inputs.unsqueeze(0)  # x.size() -> [1, batch_size]
        x = x.view(batch_size, -1)  # x.size() -> [batch_size, 1]
        predictions = net(x)

        # calculate the loss and perform backprop
        loss = loss_func(predictions.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_losses = []
            net.eval()
            print_flag = True
            for inputs, labels in valid_loader:
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                if print_flag:
                    inputs, labels = zip(*sorted(zip(inputs.numpy(), labels.numpy())))
                    inputs = torch.from_numpy(np.asarray(inputs))
                    labels = torch.from_numpy(np.asarray(labels))
                inputs, labels = inputs.to(device), labels.to(device)

                # get the output from the model
                # x.size() -> [batch_size]
                batch_size = inputs.size(0)
                # IMPORTANT - change the dimensions of x before it enters the NN,
                # batch size must always be first
                x = inputs.unsqueeze(0)  # x.size() -> [1, batch_size]
                x = x.view(batch_size, -1)  # x.size() -> [batch_size, 1]
                val_predictions = net(x)
                val_loss = loss_func(val_predictions.squeeze(), labels.float())

                val_losses.append(val_loss.item())
                if print_flag:
                    print_flag = False
                    # plot and show learning process
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.cla()
                    ax.scatter(inputs.cpu().data.numpy(), labels.cpu().data.numpy())
                    ax.plot(
                        inputs.cpu().data.numpy(),
                        val_predictions.cpu().data.numpy(),
                        "r-",
                        lw=2,
                    )
                    ax.text(
                        0.5,
                        0,
                        "Loss=%.4f" % np.mean(val_losses),
                        fontdict={"size": 10, "color": "red"},
                    )
                    plt.pause(0.1)
                    ax.clear()

            net.train()
            print(
                "Epoch: {}/{}...".format(e + 1, epochs),
                "Step: {}...".format(counter),
                "Loss: {:.6f}...".format(loss.item()),
                "Val Loss: {:.6f}".format(np.mean(val_losses)),
            )
plt.show()


def print_loss_stats():
    

# -------------------------
# Putting it all together
# -------------------------

train_ds = NER_DataSet("data/train.tagged")
print("created train dataset")
dev_ds = NER_DataSet("data/dev.csv", tokenizer=train_ds.tokenizer)
print("created dev dataset")
ds_to_check = dev_ds

datasets = {"train": train_ds, "test": ds_to_check}

# TODO: change params?

# option 1:
# is identity - yes / no
num_classes = 2
# option 2:
# different classification for differnet identities
# num_classes = ?

batch_size = 32
num_epochs = 15

NN_model = NER_NN(num_classes, vocab_size=train_ds.vocabulary_size)

train_and_plot(
    NN_model=NN_model, data_sets=datasets, num_epochs=num_epochs, batch_size=batch_size
)
