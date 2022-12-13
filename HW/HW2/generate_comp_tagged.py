import torch
import torch.nn as nn

from model3_comp import LSTM_NER_NN
from predict_model3 import predict_LSTM
from preprocessing import SentencesEmbeddingDataset
from utils import calc_f1, write_to_tagged_file


def main():
    # predict
    hidden_dim = 500
    batch_size = 32
    num_layers = 1
    w_list = [0.1, 0.9]
    O_str = "withO"  # or "noO"
    predictions_path = "data/comp_206014482_316375872.tagged"
    model_save_path = f"LSTM_model_stateDict_hidden={hidden_dim}_layers={num_layers}_w={w_list}_{O_str}.pt"

    file_path_no_tag = "data/test.untagged"

    torch.manual_seed(42)
    NER_dataset = SentencesEmbeddingDataset(
        vec_dim=500,
        list_embedding_paths=["glove-twitter-200", "word2vec-google-news-300"],
        list_vec_dims=[200, 300],
        embedding_model_path="concated",
        O_str=O_str,
    )
    _, _, test_loader = NER_dataset.get_data_loaders(batch_size=batch_size)

    # LSTM model
    LSTM_model = LSTM_NER_NN(
        embedding_dim=500,
        num_classes=2,
        hidden_dim=hidden_dim,
        model_save_path="test.pt",
        activation=nn.Tanh(),
        num_layers=1,
        dropout=0.2,
    )

    LSTM_model.load_state_dict(torch.load(model_save_path))

    # predict
    y_pred = predict_LSTM(LSTM_model=LSTM_model, test_loader=test_loader)

    # save tagged file
    write_to_tagged_file(y_pred, predictions_path, file_path_no_tag)

    # for DEBUG
    # calc_f1(predictions_path, tagged_real="data/dev.tagged")


if __name__ == "__main__":
    main()
