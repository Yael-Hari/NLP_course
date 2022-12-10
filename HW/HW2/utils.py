import matplotlib.pyplot as plt
import numpy as np
import torch


def get_f1_accuracy_by_confusion_matrix(confusion_matrix):
    # only for 2 classes !
    Tn = confusion_matrix[0][0]
    Fn = confusion_matrix[1][0]
    Tp = confusion_matrix[1][1]
    Fp = confusion_matrix[0][1]

    accuracy = (Tn + Tp) / (Tn + Tp + Fn + Fp)
    precision = Tp / (Tp + Fp)
    recall = Tp / (Tp + Fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, f1


def print_batch_details(
    num_of_batches,
    batch_num,
    loss,
    train_confusion_matrix,
    val_confusion_matrix,
    loader_type,
):
    train_accuracy, train_f1 = get_f1_accuracy_by_confusion_matrix(
        train_confusion_matrix
    )
    if loader_type == "train":
        print(
            "Batch: {}/{} |".format(batch_num + 1, num_of_batches),
            "Batch Loss: {:.3f} |".format(loss),
            "Train Accuracy: {:.3f} |".format(train_accuracy),
            "Train F1: {:.3f}".format(train_f1),
        )
    if loader_type == "validate":
        val_accuracy, val_f1 = get_f1_accuracy_by_confusion_matrix(val_confusion_matrix)
        print(
            "Val Accuracy: {:.3f} |".format(val_accuracy),
            "Val F1: {:.3f}".format(val_f1),
        )


def print_epoch_details(
    epoch_dict,
    epoch_num,
    loader_type,
):
    confusion_matrix = epoch_dict[loader_type]["confusion_matrix"]
    loss_list = epoch_dict[loader_type]["loss_list"]
    avg_loss = np.array(loss_list).mean()
    total_epochs_num = epoch_dict["total_epochs"]

    accuracy, f1 = get_f1_accuracy_by_confusion_matrix(confusion_matrix)
    print(
        "Epoch: {}/{} |".format(epoch_num + 1, total_epochs_num),
        "{} Accuracy: {:.3f} |".format(loader_type, accuracy),
        "{} F1: {:.3f}".format(loader_type, f1),
    )

    epoch_dict[loader_type]["accuracy_list"].append(accuracy)
    epoch_dict[loader_type]["f1_list"].append(f1)
    epoch_dict[loader_type]["avg_loss_list"].append(avg_loss)
    return epoch_dict


def remove_padding(padded, lengths):
    unpadded = []
    for x, len_x in zip(padded, lengths):
        x_unpad = torch.Tensor(x[:len_x])
        unpadded.append(x_unpad)
    return torch.concat(unpadded)


def plot_epochs_results(
    epoch_dict,
    lr,
    hidden_size,
    num_layers,
    embedding_name,
    activation_name,
    class_weights,
):
    class_weights = [round(float(w), 2) for w in class_weights]
    hyper_params_str = f"{lr=} | {hidden_size=} | {num_layers=} | {activation_name=}\
        \n{embedding_name=} | {class_weights=}"

    epochs_nums_list = np.arange(1, epoch_dict["total_epochs"] + 1)

    loader_types = ["train", "validate"]
    acc_colors = {"train": "steelblue", "validate": "darkorange"}
    f1_colors = {"train": "seagreen", "validate": "crimson"}
    loss_colors = {"train": "purple", "validate": "chocolate"}
    for loader_type in loader_types:
        acc_vals = epoch_dict[loader_type]["accuracy_list"]
        f1_vals = epoch_dict[loader_type]["f1_list"]
        avg_loss_vals = epoch_dict[loader_type]["avg_loss_list"]
        plt.plot(
            epochs_nums_list,
            acc_vals,
            label=f"{loader_type}_Accuracy",
            color=acc_colors[loader_type],
        )
        plt.plot(
            epochs_nums_list,
            f1_vals,
            label=f"{loader_type}_F1",
            color=f1_colors[loader_type],
        )
        plt.plot(
            epochs_nums_list,
            avg_loss_vals,
            label=f"{loader_type}_avg_loss",
            color=loss_colors[loader_type],
        )
    plt.legend()
    plt.title(hyper_params_str)
    plt.ylabel("Score")
    plt.xlabel("Epoch")
    plt.show()
