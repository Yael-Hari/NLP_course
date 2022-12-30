import torch
import torch.nn as nn

from model import DependencyParser
from train_predict_plot import predict


def main():
    # word embedding
    word_embedding_name = "glove-wiki-gigaword-200"
    word_embedding_dim = 200
    list_embedding_paths = None
    word_embedding_dim_list = None
    # pos embedding
    pos_embedding_name = "onehot"  # or "learn"
    pos_embedding_dim = 25
    batch_size = 1
    # lstm params
    lstm_hidden_dim = 250
    lstm_num_layers = 2
    lstm_dropout = 0.25
    activation = nn.Tanh()

    torch.manual_seed(42)
    load_dataset_from_pkl = False

    # get embeddings
    if load_dataset_from_pkl:
        train_loader, val_loader, comp_loader = torch.load(
            f"{word_embedding_name}_{pos_embedding_name}.pkl"
        )
    else:
        Dataset = SentencesEmbeddingDataset(
            embedding_model_path=word_embedding_name,
            list_embedding_paths=list_embedding_paths,
            word_embedding_dim_list=word_embedding_dim_list,
            word_embedding_dim=word_embedding_dim,
            pos_embedding_name=pos_embedding_name,
            pos_embedding_dim=pos_embedding_dim,
        )
        train_loader, test_loader, _ = Dataset.get_data_loaders(batch_size=batch_size)

    print("----------------------------------------------------------")

    hyper_params_title = f"{word_embedding_name=} | {pos_embedding_name=} | hidden={lstm_hidden_dim} \
            \nnum_layers={lstm_num_layers} | dropout={lstm_dropout}"
    print(hyper_params_title)
    model_save_path = f"{hyper_params_title}.pt"

    dependency_model = DependencyParser(
        embedding_dim=word_embedding_dim + pos_embedding_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=lstm_dropout,
        activation=activation,
    )

    dependency_model.load_state_dict(torch.load(model_save_path))

    # predict
    pred_deps = predict(dependency_model=dependency_model, loader_to_tag=comp_loader)

    # save tagged file
    file_path_no_tag = "comp.unlabeled"
    predictions_path = "comp_316375872_206014482.labeled"
    write_to_tagged_file(pred_deps, predictions_path, file_path_no_tag)

    # for DEBUG
    # calc_UAC(predictions_path, tagged_real="test.labeled")


def write_to_tagged_file(pred_deps, predictions_path, file_path_no_tag):
    empty_lines = ["\t", ""]
    print(f"tagging this file: {file_path_no_tag}")
    with open(file_path_no_tag, "r", encoding="utf8") as untagged_file:
        with open(predictions_path, "w", encoding="utf8") as preds_file:
            heads = pred_deps[:, 1]
            for head in heads:
                untagged_line = untagged_file.readline()
                # TODO: complete
                preds_file.write("something")
    pass
