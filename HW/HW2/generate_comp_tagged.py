import torch
import torch.nn as nn

from model3_comp import LSTM_NER_NN
from predict_model3 import predict_LSTM
from preprocessing import SentencesEmbeddingDataset
from utils import write_to_tagged_file


def main():
    # predict
    embedding_dim = 500
    hidden_dim = 128
    dropout = 0.2
    num_classes = 2
    num_epochs = 10
    activation = nn.Tanh()
    num_layers = 1
    batch_size = 32
    predictions_path = "data/comp_206014482_316375872.tagged"
    model_save_path = f"LSTM_model_stateDict_hidden_{hidden_dim}.pt"

    # test_loader
    torch.manual_seed(42)
    NER_dataset = SentencesEmbeddingDataset(
        vec_dim=embedding_dim,
        list_embedding_paths=["glove-twitter-200", "word2vec-google-news-300"],
        list_vec_dims=[200, 300],
        embedding_model_path="concated",
    )
    _, _, test_loader = NER_dataset.get_data_loaders(batch_size=batch_size)

    # LSTM model
    LSTM_model = LSTM_NER_NN(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        model_save_path="test.pt",
        activation=activation,
        num_layers=num_layers,
        dropout=dropout,
    )

    LSTM_model.load_state_dict(torch.load(model_save_path))

    # predict
    y_pred = predict_LSTM(
        LSTM_model=LSTM_model,
        test_loader=test_loader,
        num_epochs=num_epochs,
    )

    # save tagged file
    write_to_tagged_file(y_pred, predictions_path)


if __name__ == "__main__":
    main()
