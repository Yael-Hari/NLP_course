import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam

from preprocessing import SentencesEmbeddingDataset
from train_loop_model3 import train_and_plot_LSTM
from utils import plot_epochs_results, remove_padding


class LSTM_NER_NN(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        num_classes,
        model_save_path,
        activation,
        num_layers,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )
        self.hidden2tag = nn.Sequential(
            activation, nn.Linear(self.hidden_dim * 2, num_classes)
        )
        self.model_save_path = model_save_path
        self.num_classes = num_classes
        print(f"{hidden_dim=} | {activation=} | {num_layers=}")

    def forward(self, sentences_embeddings, sen_lengths):
        # pack
        packed_input = pack_padded_sequence(
            sentences_embeddings, sen_lengths, batch_first=True, enforce_sorted=False
        )
        lstm_packed_output, (ht, ct) = self.lstm(input=packed_input)
        # unpack
        lstm_out_padded, out_lengths = pad_packed_sequence(
            lstm_packed_output, batch_first=True
        )
        # reshape from sentences to words
        words_lstm_out_unpadded = remove_padding(lstm_out_padded, out_lengths)
        # hidden -> tag score -> prediction -> loss
        tag_space = self.hidden2tag(words_lstm_out_unpadded)
        tag_score = F.softmax(tag_space, dim=1)
        return tag_score


# -------------------------
# Putting it all together
# -------------------------

# fasttext-wiki-news-subwords-300
# glove-twitter-100
# glove-twitter-200
# glove-twitter-25
# glove-twitter-50
# glove-wiki-gigaword-100
# glove-wiki-gigaword-200
# glove-wiki-gigaword-300
# glove-wiki-gigaword-50
# word2vec-google-news-300
# word2vec-ruscorpora-300


def main():
    batch_size = 32
    embedding_name = "glove-wiki-gigaword-300"
    NER_dataset = SentencesEmbeddingDataset(
        embedding_model_path=embedding_name, vec_dim=300
    )
    train_loader, dev_loader = NER_dataset.get_data_loaders(batch_size=batch_size)

    num_classes = 2
    num_epochs = 30
    hidden_dim = 64
    embedding_dim = NER_dataset.vec_dim
    lr = 0.001
    # activation = nn.ReLU()
    # activation = nn.Sigmoid()
    activation = nn.Tanh()
    activation_name = "Tanh"

    class_weights = torch.tensor([0.25, 0.75])

    num_layers = 1
    model_save_path = (
        f"LSTM_model_stateDict_batchSize_{batch_size}_hidden_{hidden_dim}_lr_{lr}.pt"
    )

    LSTM_model = LSTM_NER_NN(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        model_save_path=model_save_path,
        activation=activation,
        num_layers=num_layers,
    )

    optimizer = Adam(params=LSTM_model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss(weight=class_weights)

    epoch_dict = train_and_plot_LSTM(
        LSTM_model=LSTM_model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        val_loader=dev_loader,
        optimizer=optimizer,
        loss_func=loss_func,
    )

    plot_epochs_results(
        epoch_dict=epoch_dict,
        lr=lr,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        embedding_name=embedding_name,
        activation_name=activation_name,
        class_weights=list(class_weights),
    )


if __name__ == "__main__":
    main()
