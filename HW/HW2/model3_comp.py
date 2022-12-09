import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam

from preprocessing import SentencesEmbeddingDataset
from train_loop_model3 import train_and_plot_LSTM


class LSTM_NER_NN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, model_save_path):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )
        self.hidden2tag = nn.Sequential(
            nn.ReLU(), nn.Linear(self.hidden_dim, num_classes)
        )
        self.model_save_path = model_save_path
        self.num_classes = num_classes

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
        # hidden -> tag score -> prediction -> loss
        tag_space = self.hidden2tag(lstm_out_padded)
        tag_score = F.softmax(tag_space, dim=1)
        return tag_score


# -------------------------
# Putting it all together
# -------------------------
def main():
    embedding_type = "glove"
    batch_size = 32
    NER_dataset = SentencesEmbeddingDataset(embedding_model_type=embedding_type)
    train_loader, dev_loader = NER_dataset.get_data_loaders(batch_size=batch_size)

    num_classes = 2
    num_epochs = 5
    hidden_dim = 64
    embedding_dim = NER_dataset.VEC_DIM
    lr = 0.001
    model_save_path = (
        f"LSTM_model_stateDict_batchSize_{batch_size}_hidden_{hidden_dim}_lr_{lr}.pt"
    )

    LSTM_model = LSTM_NER_NN(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        model_save_path=model_save_path,
    )

    optimizer = Adam(params=LSTM_NER_NN.parameters(), lr=lr)
    loss_func = nn.NLLLoss()

    train_and_plot_LSTM(
        LSTM_model=LSTM_model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        val_loader=dev_loader,
        optimizer=optimizer,
        loss_func=loss_func,
    )


if __name__ == "__main__":
    main()
