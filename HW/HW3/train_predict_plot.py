import time
import pickle
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
    plot=True
):
    # GPU - checking if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Training on GPU.")
    else:
        print("No GPU available, training on CPU.")
    # device = torch.device("cpu")
    dependency_model.to(device)

    # Epoch Loop
    # prepare for evaluate
    epoch_dict = {}
    epoch_dict["total_epochs"] = num_epochs
    dataset_types = ["train", "validate"]
    for dataset_type in dataset_types:
        epoch_dict[dataset_type] = {}
        epoch_dict[dataset_type]["all_epochs_loss_list"] = []
        epoch_dict[dataset_type]["all_epochs_UAS_list"] = []

    for epoch_num in range(num_epochs):
        datasets = {"train": train_dataset}
        if val_dataset:
            datasets["validate"] = val_dataset
        pred_deps_all_val = []
        true_deps_all_val = []

        # prepare for evaluate
        for dataset_type in dataset_types:
            epoch_dict[dataset_type]["epoch_num_correct_def_list"] = []
            epoch_dict[dataset_type]["epoch_num_total_deps_list"] = []
            epoch_dict[dataset_type]["epoch_loss_list"] = []

        for dataset_type, (X, y) in datasets.items():
            start = time.time()
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
                    length=scores_matrix.size(0),
                    has_labels=False,
                )
                pred_deps_in_format = torch.Tensor(
                    [[mod, head] for mod, head in enumerate(pred_deps[0])][1:]
                )
                if dataset_type == "validate":
                    pred_deps_all_val.append(pred_deps_in_format)
                    true_deps_all_val.append(true_deps)

                correct_deps = calc_correct_deps(pred_deps_in_format, true_deps.clone().detach().cpu())
                # update epoch dict
                epoch_dict[dataset_type]["epoch_num_correct_def_list"].append(
                    correct_deps
                )
                epoch_dict[dataset_type]["epoch_num_total_deps_list"].append(
                    len(true_deps)
                )
                epoch_dict[dataset_type]["epoch_loss_list"].append(
                    float(loss.clone().detach().cpu())
                )

                if dataset_type == "train":
                    # backprop
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_dict = print_epoch_details(
                epoch_dict,
                epoch_num,
                dataset_type,
            )
            print("epoch_time:", time.time() - start)

        # if epoch_num + 1 == 1:
        #     # torch.save(dependency_model, "epoch1_model.pkl")
        #     print(epoch_dict["validate"]["all_epochs_UAS_list"])
        #     print(epoch_dict["validate"]["all_epochs_loss_list"])
        #     exit()

            # if epoch_num + 1 == 5:
            #     dependency_model = torch.load("epoch3_model.pkl")
    print(epoch_dict["validate"]["all_epochs_UAS_list"])
    # [0.17834836918806385, 0.5454545454545454, 0.6943095072866066, 0.7702984038861902, 0.8400416377515614,
    #  0.8639833448993754]

    torch.save(dependency_model.state_dict(), model_save_path)
    # torch.save(dependency_model, model_save_path)

    with open(f"pred_deps/pred_deps_all 1_{model_save_path}.txt", "w") as f:
        for i in pred_deps_all_val:
            f.write(str(i) + "\n")
    with open(f"true_deps/true_deps_all 1_{model_save_path}.txt", "w") as f:
        for i in true_deps_all_val:
            f.write(str(i) + "\n")

    if plot:
        plot_epochs_results(
            epoch_dict=epoch_dict,
            hyper_params_title=hyper_params_title,
            model_save_path=model_save_path,
        )
    print("--- FINISH ---")


def predict(dependency_model, dataset_to_tag, tagged, model_save_path):
    # GPU - checking if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Training on GPU.")
    else:
        print("No GPU available, training on CPU.")
    dependency_model.to(device)
    # run predict
    pred_deps_all = []
    if tagged:
        (X, y) = dataset_to_tag
        true_deps_all = []
        loss_list = []
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
                length=scores_matrix.size(0),
                has_labels=False,
            )
            pred_deps_in_format = torch.Tensor(
                [[mod, head] for mod, head in enumerate(pred_deps[0])][1:]
            )
            pred_deps_all.append(pred_deps_in_format)
            true_deps_all.append(true_deps)
            loss_list.append(loss.clone().detach().cpu())
        pred_deps_all = np.concatenate(pred_deps_all)
        true_deps_all = np.concatenate(true_deps_all)
        UAC = calc_correct_deps(pred_deps_all, true_deps_all) / len(true_deps_all)
        print("loss:", np.round(np.sum(loss_list), 3))
        print("UAC:", UAC)
        # with open(f"pred_deps/pred_deps_all 2_{model_save_path}.txt", "w") as f:
        #     for i in pred_deps_all:
        #         f.write(str(i) + "\n")
        # with open(f"true_deps/true_deps_all 2_{model_save_path}.txt", "w") as f:
        #     for i in true_deps_all:
        #         f.write(str(i) + "\n")
    else:
        true_deps = None
        X = dataset_to_tag
        for sentence in X[0]:
            # if training on gpu
            sentence = sentence.to(device)
            # forward
            dependency_model.tagged = tagged
            _, scores_matrix = dependency_model((sentence, true_deps))
            # dependencies predictions
            pred_deps = decode_mst(
                energy=scores_matrix.clone().detach().cpu(),
                length=scores_matrix.size(0),
                has_labels=False,
            )
            pred_deps_in_format = torch.Tensor(
                [[mod, head] for mod, head in enumerate(pred_deps[0])][1:]
            )
            pred_deps_all.append(pred_deps_in_format)
        pred_deps_all = np.concatenate(pred_deps_all)
    return pred_deps_all


def calc_correct_deps(pred_deps, true_deps):
    """
    Example: (modifier, head)
    True   Pred  Correct
    1 2     1 2     V
    2 0     2 0     V
    3 5     3 4     X      ==> UAS = 4 / 5
    4 5     4 5     V
    5 2     5 2     V

    assuming first column is the same (token idx)
    """
    return int((pred_deps[:, 1] == true_deps[:, 1]).sum())


def print_epoch_details(
    epoch_dict,
    epoch_num,
    dataset_type,
):
    num_correct = sum(epoch_dict[dataset_type]["epoch_num_correct_def_list"])
    num_total = sum(epoch_dict[dataset_type]["epoch_num_total_deps_list"])
    epoch_UAS = num_correct / num_total
    loss_list = epoch_dict[dataset_type]["epoch_loss_list"]
    epoch_loss = np.array(loss_list).sum()
    total_epochs_num = epoch_dict["total_epochs"]

    print(
        "Epoch: {}/{} |".format(epoch_num + 1, total_epochs_num),
        "{} UAS: {:.3f} |".format(dataset_type, epoch_UAS),
        "{} loss: {:.3f} |".format(dataset_type, epoch_loss),
    )

    epoch_dict[dataset_type]["all_epochs_UAS_list"].append(epoch_UAS)
    epoch_dict[dataset_type]["all_epochs_loss_list"].append(epoch_loss)
    return epoch_dict


def plot_epochs_results(epoch_dict, hyper_params_title, model_save_path):
    epochs_nums_list = np.arange(1, epoch_dict["total_epochs"] + 1)

    dataset_types = ["train", "validate"]
    UAS_colors = {"train": "seagreen", "validate": "crimson"}
    loss_colors = {"train": "purple", "validate": "chocolate"}
    for dataset_type in dataset_types:
        UAS_vals = epoch_dict[dataset_type]["all_epochs_UAS_list"]
        if dataset_type == "validate":
            UAS = round(float(UAS_vals[-1]), 3)

        plt.plot(
            epochs_nums_list,
            UAS_vals,
            label=f"{dataset_type}_UAS",
            color=UAS_colors[dataset_type],
        )
    plt.legend()
    plt.title(f"{UAS=} | {hyper_params_title}")
    plt.ylabel("Score")
    plt.xlabel("Epoch")
    file_name = f"plots/{UAS=}_{model_save_path}.png"

    plt.savefig(file_name)
    plt.clf()
    plt.cla()

    for dataset_type in dataset_types:
        loss_vals = epoch_dict[dataset_type]["all_epochs_loss_list"]
        if dataset_type == "validate":
            loss = round(float(loss_vals[-1]), 3)

        plt.plot(
            epochs_nums_list,
            loss_vals,
            label=f"{dataset_type}_loss",
            color=loss_colors[dataset_type],
        )
    plt.legend()
    plt.title(f"{loss=} | {hyper_params_title}")
    plt.ylabel("Score")
    plt.xlabel("Epoch")
    file_name = f"plots/{loss=}_{model_save_path}.png"

    plt.savefig(file_name)
    plt.clf()
    plt.cla()
