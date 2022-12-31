import matplotlib.pyplot as plt
import numpy as np
import torch

from chu_liu_edmonds import decode_mst


def train_and_plot(
    dependency_model,
    model_save_path: str,
    train_dataset,
    val_dataset,
    num_epochs: int,
    optimizer,
    hyper_params_title: str,
):
    # GPU - checking if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Training on GPU.")
    else:
        print("No GPU available, training on CPU.")
    device = torch.device("cpu")
    dependency_model.to(device)

    # Epoch Loop
    # prepare for evaluate
    epoch_dict = {}
    epoch_dict["total_epochs"] = num_epochs
    dataset_types = ["train", "validate"]
    for dataset_type in dataset_types:
        epoch_dict[dataset_type] = {}
        epoch_dict[dataset_type]["num_correct_def_list"] = []
        epoch_dict[dataset_type]["num_total_deps_list"] = []
        epoch_dict[dataset_type]["avg_loss_list"] = []

    for epoch_num in range(num_epochs):
        # prepare for evaluate
        for dataset_type in dataset_types:
            epoch_dict[dataset_type]["loss_list"] = []
            epoch_dict[dataset_type]["UAS_list"] = []

        datasets = {"train": train_dataset}
        if val_dataset:
            datasets["validate"] = val_dataset

        for dataset_type, (X, y) in datasets.items():
            for sentence, true_deps in zip(X, y):
                # if training on gpu
                sentence, true_deps = (
                    sentence.to(device),
                    true_deps.to(device),
                )
                # forward
                loss, scores_matrix = dependency_model((sentence, true_deps))
                # dependencies predictions
                pred_deps = decode_mst(
                    energy=scores_matrix.clone().detach().cpu(),
                    length=scores_matrix.size(0)
                )
                correct_deps = calc_correct_deps(pred_deps, true_deps)
                # update epoch dict
                epoch_dict[dataset_type]["num_correct_def_list"].append(correct_deps)
                epoch_dict[dataset_type]["num_total_deps_list"].append(len(true_deps))
                epoch_dict[dataset_type]["batch_loss_list"].append(loss.detach().cpu())

                if dataset_type == "train":
                    # backprop
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_dict = print_epoch_details(
                epoch_dict,
                epoch_num,
                dataset_type,
            )
    torch.save(dependency_model.state_dict(), model_save_path)
    print(f"saved model to file {model_save_path}")
    plot_epochs_results(epoch_dict=epoch_dict, hyper_params_title=hyper_params_title)
    print("--- FINISH ---")


def predict(dependency_model, dataset_to_tag):
    # GPU - checking if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Training on GPU.")
    else:
        print("No GPU available, training on CPU.")
    device = torch.device("cpu")
    dependency_model.to(device)
    # run predict
    pred_deps_all = []
    for sentence in dataset_to_tag:
        # if training on gpu
        sentence, true_deps = (sentence.to(device),)
        # forward
        _, scores_matrix = dependency_model(sentence, true_deps)
        # dependencies predictions
        pred_deps = decode_mst(scores_matrix.clone().detach().cpu())
        pred_deps_all.append(pred_deps)
    pred_deps_all = np.concatenate(pred_deps_all)
    return pred_deps_all


def calc_correct_deps(pred_deps, true_deps):
    """
    Example:
    True   Pred  Correct
    1 2     1 2     V
    2 0     2 0     V
    3 5     3 4     X      ==> UAS = 4 / 5
    4 5     4 5     V
    5 2     5 2     V

    assuming first column is the same (token idx)
    """
    return (pred_deps[:, 1] == true_deps[:, 1]).sum()


def print_epoch_details(
    epoch_dict,
    epoch_num,
    dataset_type,
):
    num_correct = epoch_dict[dataset_type]["num_correct_def_list"][-1]
    num_total = epoch_dict[dataset_type]["num_total_deps_list"][-1]
    UAS = num_correct / num_total
    loss_list = epoch_dict[dataset_type]["batch_loss_list"]
    epoch_loss = np.array(loss_list).sum()
    total_epochs_num = epoch_dict["total_epochs"]

    print(
        "Epoch: {}/{} |".format(epoch_num + 1, total_epochs_num),
        "{} UAS: {:.3f} |".format(dataset_type, UAS),
    )

    epoch_dict[dataset_type]["UAS_list"].append(UAS)
    epoch_dict[dataset_type]["loss_list"].append(epoch_loss)
    return epoch_dict


def plot_epochs_results(epoch_dict, hyper_params_title):
    epochs_nums_list = np.arange(1, epoch_dict["total_epochs"] + 1)

    dataset_types = ["train", "validate"]
    UAS_colors = {"train": "seagreen", "validate": "crimson"}
    loss_colors = {"train": "purple", "validate": "chocolate"}
    for dataset_type in dataset_types:
        UAS_vals = epoch_dict[dataset_type]["UAS_list"]
        loss_vals = epoch_dict[dataset_type]["loss_list"]
        if dataset_type == "validate":
            val_UAS = round(UAS_vals[-1], 3)

        plt.plot(
            epochs_nums_list,
            UAS_vals,
            label=f"{dataset_type}_UAS",
            color=UAS_colors[dataset_type],
        )
        plt.plot(
            epochs_nums_list,
            loss_vals,
            label=f"{dataset_type}_avg_loss",
            color=loss_colors[dataset_type],
        )
    plt.legend()
    plt.title(f"{val_UAS=} | {hyper_params_title}")
    plt.ylabel("Score")
    plt.xlabel("Epoch")
    file_name = f"plots/{val_UAS=}_{hyper_params_title}.png"

    plt.savefig(file_name)
    plt.clf()
    plt.cla()
