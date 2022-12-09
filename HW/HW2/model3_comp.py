# from abc import ABC
# import numpy as np
# import torch
import torch.nn as nn
import torch.nn.functional as F

from model2_nn import train_and_plot

# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm, trange
from preprocessing import SentencesEmbeddingDataset


class LSTM_NER_NN(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes, model_save_path):
        super().__init__()
        self.embedding_dim = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )
        self.hidden2tag = nn.Sequential(
            nn.ReLU(), nn.Linear(self.hidden_dim, num_classes)
        )
        self.loss_func = nn.NLLLoss()
        self.model_save_path = model_save_path
        self.num_classes = num_classes

    def forward(self, embeds, tags=None):
        # LSTM
        batch_size = len(embeds)
        sequence_length = -1
        embedding_size = self.embedding_dim
        lstm_out, _ = self.lstm(
            input=embeds.view(batch_size, sequence_length, embedding_size)
        )
        # hidden -> tag score -> prediction -> loss
        tag_space = self.hidden2tag(lstm_out)
        tag_score = F.softmax(tag_space, dim=1)
        if tags is None:
            return tag_score, None
        loss = self.loss_func(tag_score, tags)
        return tag_score, loss


# -------------------------
# Putting it all together
# -------------------------
def main():
    embedding_type = "glove"
    batch_size = 32
    NER_dataset = SentencesEmbeddingDataset(embedding_model_type=embedding_type)
    train_loader, dev_loader = NER_dataset.get_data_loaders(batch_size=batch_size)

    # option 1:
    # is identity - yes / no
    num_classes = 2

    num_epochs = 5
    hidden_dim = 64
    # single vector size
    input_size = NER_dataset.VEC_DIM
    lr = 0.001

    model_save_path = (
        f"LSTM_model_stateDict_batchSize_{batch_size}_hidden_{hidden_dim}_lr_{lr}.pt"
    )
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
        lr=lr,
    )


if __name__ == "__main__":
    main()
