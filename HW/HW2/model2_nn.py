import pandas as pd
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset

# -----------------------

# -----------------------
# Define the Model




class SentimentNN(nn.Module):
    def __init__(self, vocab_size, num_classes, hidden_dim=100):
        super(SentimentNN, self).__init__()
        self.first_layer = nn.Linear(vocab_size, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        x = self.first_layer(input_ids)
        x = self.activation(x)
        x = self.second_layer(x)
        if labels is None:
            return x, None
        loss = self.loss(x, labels)
        return x, loss
