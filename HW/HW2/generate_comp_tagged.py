import torch.nn as nn

from model3_comp import LSTM_NER_NN
from train_loop_model3 import predict_LSTM


def main():
    # test_loader
    # TODO

    # predict
    embedding_dim = 500
    hidden_dim = 128
    dropout = 0.2
    num_classes = 2
    num_epochs = 10
    activation = nn.Tanh()
    num_layers = 1

    LSTM_model = LSTM_NER_NN(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        model_save_path="",
        activation=activation,
        num_layers=num_layers,
        dropout=dropout,
    )

    y_pred = predict_LSTM(
        LSTM_model=LSTM_model,
        test_loader=test_loader,
        num_epochs=num_epochs,
    )

    # save tagged file
    # TODO


if __name__ == "__main__":
    main()
