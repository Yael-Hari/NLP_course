import torch
import torch.nn as nn
import pickle
from model import DependencyParser
from preprocess import SentencesEmbeddingDataset
from train_predict_plot import predict, train_and_plot


def main_gen_comp():
    num_epochs = 10
    # word embedding
    word_embedding_name = "glove-wiki-gigaword-200"
    word_embedding_dim = 200
    # pos embedding
    pos_embedding_name = "onehot"
    # lstm params
    lstm_hidden_dim = 350
    lstm_num_layers = 4
    lstm_dropout = 0

    activation = nn.Tanh()
    optimizer_name = "ADAM"
    torch.manual_seed(42)

    # get embeddings
    gen_dataset = SentencesEmbeddingDataset(
        word_embedding_name=word_embedding_name,
        list_embedding_paths=None,
        word_embedding_dim_list=None,
        word_embedding_dim=word_embedding_dim,
        pos_embedding_name=pos_embedding_name,
        pos_embedding_dim=None,
    )
    gen_train_dataset, gen_val_dataset, gen_comp_dateset = gen_dataset.get_datasets()

    # paths
    # test
    tagged = True
    plot = False  # TODO change to False
    file_path_no_tag = "mini_train3.labeled"
    predictions_path = "mini_train3_316375872_206014482.labeled"
    dataset_to_tag = gen_val_dataset

    print("----------------------------------------------------------")

    hyper_params_title = f"{word_embedding_name}"
    hyper_params_title += f" | pos={pos_embedding_name}"
    hyper_params_title += f" | hidden={lstm_hidden_dim}"
    hyper_params_title += f" \nnum_layers={lstm_num_layers}"
    hyper_params_title += f" | dropout={lstm_dropout}"
    hyper_params_title += f" | opt={optimizer_name}"
    model_name = "comp | "
    model_name += f"word_embedding_name={word_embedding_name}"
    model_name += f" | pos={pos_embedding_name}"
    model_name += f" | hidden={lstm_hidden_dim}"
    model_name += f" | num_layers={lstm_num_layers}"
    model_name += f" | dropout={lstm_dropout}"
    model_name += f" | opt={optimizer_name}"
    print(hyper_params_title)
    model_save_path = f"{model_name}.pt"

    embedding_dim = word_embedding_dim + gen_dataset.pos_embedding_dim
    dep_model = DependencyParser(
        embedding_dim=embedding_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_num_layers=lstm_num_layers,
        fc_hidden_dim=int(lstm_hidden_dim / 2),
        lstm_dropout=lstm_dropout,
        activation=activation,
        tagged=tagged,
    )
    optimizer = torch.optim.Adam(
        params=dep_model.parameters()
    )

    train_and_plot(
        dependency_model=dep_model,
        model_save_path=model_save_path,
        train_dataset=gen_train_dataset,
        val_dataset=gen_val_dataset,
        num_epochs=num_epochs,
        optimizer=optimizer,
        hyper_params_title=hyper_params_title,
        plot=plot
    )

    # dep_model.load_state_dict(torch.load(model_save_path))
    # dep_model = torch.load(model_save_path)
    # with open(model_save_path + '.pkl', 'rb') as f:
    #     dep_model = pickle.load(f)["model"]

    # comp
    tagged = False
    plot = False
    file_path_no_tag = "comp.unlabeled"
    predictions_path = "comp_316375872_206014482.labeled"
    dataset_to_tag = gen_comp_dateset

    # predict
    pred_deps = predict(
        dependency_model=dep_model,
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
    if tagged:
        print("UAC:", num_corrct / num_total)
    print(f"saved preds to file: {predictions_path}")


if __name__ == "__main__":
    main_gen_comp()
