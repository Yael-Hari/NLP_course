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
        dropout,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.hidden2tag = nn.Sequential(
            activation, nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        self.hidden2tag_layer2 = nn.Sequential(
            # activation,
            nn.Linear(self.hidden_dim, num_classes)
        )
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.max_pooling = nn.MaxPool1d(2, stride=2)
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
        # reshape from sentences to words
        words_lstm_out_unpadded = remove_padding(lstm_out_padded, out_lengths)
        # hidden -> tag score -> prediction -> loss
        tag_space = self.hidden2tag(words_lstm_out_unpadded)
        tag_space = self.activation(tag_space)
        tag_space = self.dropout(tag_space)
        tag_space = self.hidden2tag_layer2(tag_space)
        tag_score = F.softmax(tag_space, dim=1)
        return tag_score


# -------------------------
# Putting it all together
# -------------------------


def run(
    NER_dataset,
    train_loader,
    dev_loader,
    embedding_name,
    vec_dim,
    hidden_dim,
    dropout,
    class_weights,
    loss_func,
    loss_func_name,
    batch_size
):

    num_classes = 2
    num_epochs = 10
    lr = 0.001
    activation = nn.Tanh()
    num_layers = 1

    embedding_dim = vec_dim

    # train_loader, dev_loader = NER_dataset.get_data_loaders(batch_size=batch_size)

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
        dropout=dropout,
    )

    optimizer = Adam(params=LSTM_model.parameters(), lr=lr)

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
        hidden=hidden_dim,
        embedding_name=embedding_name,
        dropout=dropout,
        loss_func_name=loss_func_name,
        class_weights=list(class_weights),
    )


def main():
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

    # ("glove-twitter-200", 200),
    # ("glove-wiki-gigaword-300", 300),
    # ("word2vec-google-news-300", 300),
    # hidden_list = [64, 128, 256, 280]
    # (nn.BCEWithLogitsLoss(pos_weight=class_weights), "BCELogit")

    embed_list = [
        (
            "concated",
            ["glove-twitter-200", "word2vec-google-news-300"],
            [200, 300],
            500,
        ),
        # ("glove-twitter-200", None, None, 200),
        # ("word2vec-google-news-300", None, None, 300),
        # ("glove-wiki-gigaword-300", None, None, 300)
    ]
    # hidden_list = [32, 64, 128, 256]
    hidden_list = [128]
    # dropout_list = [0, 0.2, 0.4, 0.5]
    # dropout_list = [0.2, 0.4, 0.5, 0.0]
    dropout_list = [0.2]
    w_list = [
        torch.tensor([0.2, 0.8]),
        # torch.tensor([0.15, 0.85]),
        # torch.tensor([0.1, 0.9]),
        # torch.tensor([0.4, 0.6]),
    ]

    for embedding_name, embedding_paths, vec_dims_list, vec_dim in embed_list:
        torch.manual_seed(42)
        # NER_dataset = SentencesEmbeddingDataset(
        #     vec_dim=vec_dim,
        #     list_embedding_paths=embedding_paths,
        #     list_vec_dims=vec_dims_list,
        #     embedding_model_path=embedding_name,
        # )
        batch_size = 32
        train_loader, dev_loader, test_dataloader = torch.load(f'concated_ds_{batch_size}.pkl')
        for hidden_dim in hidden_list:
            for dropout in dropout_list:
                for class_weights in w_list:
                    for loss_func, loss_func_name in [
                        (nn.CrossEntropyLoss(weight=class_weights), "CrossEntropy"),
                    ]:
                        print(
                            "----------------------------------------------------------"
                        )
                        print(
                            f"{embedding_name=} | {hidden_dim=} | {dropout=} \
                                \n{class_weights=} | {loss_func=}"
                        )
                        run(
                            NER_dataset=None,
                            train_loader=train_loader,
                            dev_loader=dev_loader,
                            embedding_name=embedding_name + "2022-12-12",
                            vec_dim=vec_dim,
                            hidden_dim=hidden_dim,
                            dropout=dropout,
                            class_weights=class_weights,
                            loss_func=loss_func,
                            loss_func_name=loss_func_name,
                            batch_size=batch_size
                        )


if __name__ == "__main__":
    main()
