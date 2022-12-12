import torch
import torch.nn as nn

from model3_comp import LSTM_NER_NN
from predict_model3 import predict_LSTM
from preprocessing import SentencesEmbeddingDataset


def main():
    # predict
    embedding_dim = 500
    hidden_dim = 128
    dropout = 0.2
    num_classes = 2
    num_epochs = 10
    activation = nn.Tanh()
    num_layers = 1
    batch_size = 64
    predictions_path = "data/comp_206014482_316375872.tagged"

    # test_loader
    # TODO

    # LSTM model
    LSTM_model = LSTM_NER_NN(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        model_save_path="",
        activation=activation,
        num_layers=num_layers,
        dropout=dropout,
    )

    # predict
    y_pred = predict_LSTM(
        LSTM_model=LSTM_model,
        test_loader=test_loader,
        num_epochs=num_epochs,
    )

    # save tagged file
    # TODO
    output_file = open(predictions_path, "a+")
    for i in range(y_pred):
        if y_pred == 1:
            pred = "1"
        elif y_pred == 0:
            pred = "O"
        output_file.write(f"{words[i]}\t{pred}")
        output_file.write("\n")
    output_file.close()
    


if __name__ == "__main__":
    main()
