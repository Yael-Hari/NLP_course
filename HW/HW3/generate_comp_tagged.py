import torch
import torch.nn as nn

from model import DependencyParser
from preprocess import SentencesEmbeddingDataset
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
    # lstm params
    lstm_hidden_dim = 250
    lstm_num_layers = 1
    lstm_dropout = 0
    activation = nn.Tanh()
    optimizer_name = "ADAM"

    torch.manual_seed(42)
    load_dataset_from_pkl = False

    # get embeddings
    if load_dataset_from_pkl:
        _, val_dataset, comp_dataset = torch.load(
            f"{word_embedding_name}_{pos_embedding_name}.pkl"
        )
    else:
        Dataset = SentencesEmbeddingDataset(
            word_embedding_name=word_embedding_name,
            list_embedding_paths=list_embedding_paths,
            word_embedding_dim_list=word_embedding_dim_list,
            word_embedding_dim=word_embedding_dim,
            pos_embedding_name=pos_embedding_name,
            pos_embedding_dim=pos_embedding_dim,
        )
        train_dataset, test_dataset, comp_dateset = Dataset.get_datasets()

    # paths
    # test
    tagged = True
    file_path_no_tag = "mini_train3.labeled"
    predictions_path = "mini_train3_316375872_206014482.labeled"
    dataset_to_tag = train_dataset
    # comp
    # tagged = False
    # file_path_no_tag = "comp.unlabeled"
    # predictions_path = "comp_316375872_206014482.labeled"
    # dataset_to_tag = comp_dataset

    print("----------------------------------------------------------")

    hyper_params_title = f"{word_embedding_name}"
    hyper_params_title += f" | pos={pos_embedding_name}"
    hyper_params_title += f" | hidden={lstm_hidden_dim}"
    hyper_params_title += f" \nnum_layers={lstm_num_layers}"
    hyper_params_title += f" | dropout={lstm_dropout}"
    hyper_params_title += f" | opt={optimizer_name}"
    model_name = "mini_train3 | "
    model_name += f"word_embedding_name={word_embedding_name}"
    model_name += f" | pos={pos_embedding_name}"
    model_name += f" | hidden={lstm_hidden_dim}"
    model_name += f" | num_layers={lstm_num_layers}"
    model_name += f" | dropout={lstm_dropout}"
    model_name += f" | opt={optimizer_name}"
    print(hyper_params_title)
    model_save_path = f"{model_name}.pt"

    dependency_model = DependencyParser(
        embedding_dim=word_embedding_dim + Dataset.pos_embedding_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_num_layers=lstm_num_layers,
        fc_hidden_dim=int(lstm_hidden_dim / 2),
        lstm_dropout=lstm_dropout,
        activation=activation,
        tagged=tagged,
    )

    dependency_model.load_state_dict(torch.load(model_save_path))

    # predict
    pred_deps = predict(
        dependency_model=dependency_model,
        dataset_to_tag=dataset_to_tag,
        tagged=tagged,
        model_save_path=model_save_path,
    )
    # save tagged file
    write_to_tagged_file(pred_deps, predictions_path, file_path_no_tag, tagged=tagged)


def write_to_tagged_file(pred_deps, predictions_path, file_path_no_tag, tagged):
    print(f"tagging this file: {file_path_no_tag}")
    # empty lines
    empty_lines = ["", "\t"]
    heads = pred_deps[:, 1]
    head_curr_idx = 0
    num_corrct = 0
    num_total = 0

    with open(file_path_no_tag, "r", encoding="utf8") as untagged_file:
        untagged_lines = untagged_file.read()
        with open(predictions_path, "w", encoding="utf8") as preds_file:
            for untagged_line in untagged_lines.split("\n"):
                if untagged_line not in empty_lines:
                    values = untagged_line.split("\t")
                    if tagged:
                        t = values[6]
                        p = str(int(heads[head_curr_idx]))
                        num_total += 1
                        if t == p:
                            num_corrct += 1
                    # write pred
                    values[6] = str(int(heads[head_curr_idx]))
                    new_line = "\t".join(values)
                    new_line += "\n"
                    head_curr_idx += 1
                else:
                    new_line = "\n"
                preds_file.write(new_line)
    print("UAC:", num_corrct / num_total)
    print(f"saved preds to file: {predictions_path}")


if __name__ == "__main__":
    main()
