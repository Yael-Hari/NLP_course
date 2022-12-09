from abc import ABC
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from preprocessing import EmbeddingDataset
from model2_nn import train_and_plot


class LSTM_NER_NN(nn.Module):
    def __init__(self, input_size=30, hidden_dim=50, num_classes=2, model_save_path=""):
        super().__init__()
        self.embedding_dim = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Sequential(nn.ReLU(),
                                        nn.Linear(self.hidden_dim, num_classes))
        self.loss_fn = nn.NLLLoss()
        self.model_save_path = model_save_path
        self.num_classes = num_classes

    def forward(self, embeds, sentence_len, tags=None):
        lstm_out, _ = self.lstm(embeds.view(len(embeds), -1, self.embedding_dim))
        tag_space = self.hidden2tag(lstm_out[range(len(embeds)), sentence_len - 1, :])
        tag_score = F.softmax(tag_space, dim=1)
        if tags is None:
            return tag_score, None
        loss = self.loss_fn(tag_score, tags)
        return tag_score, loss


# -------------------------
# Putting it all together
# -------------------------
def main():
    embedding_type = "glove"
    batch_size = 32
    NER_dataset = EmbeddingDataset(embedding_model_type=embedding_type)
    train_loader, dev_loader = NER_dataset.get_data_loaders(batch_size=batch_size)

    # option 1:
    # is identity - yes / no
    num_classes = 2
    # option 2:
    # different classification for different identities
    # num_classes = ?

    num_epochs = 5
    hidden_dim = 64
    # single vector size
    input_size = NER_dataset.VEC_DIM
    lr = 0.001

    model_save_path = f"LSTM_model_stateDict_batchSize_{batch_size}_hidden_{hidden_dim}_lr_{lr}.pt"
    LSTM_model = LSTM_NER_NN(
        input_size=input_size,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        model_save_path=model_save_path,
    )

    train_and_plot(
        NN_model=LSTM_model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        batch_size=batch_size,
        val_loader=dev_loader,
        lr=lr
    )


if __name__ == "__main__":
    main()
