# import matplotlib.pyplot as plt
import numpy as np


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
    num_epochs,
    epoch,
    train_confusion_matrix,
    loss_batches_list,
    val_confusion_matrix,
    loader_type,
):
    loss_batches = np.array(loss_batches_list)
    num_of_batches = len(loss_batches)
    train_accuracy, train_f1 = get_f1_accuracy_by_confusion_matrix(
        train_confusion_matrix
    )
    if loader_type == "train":
        print(
            "Epoch: {}/{} |".format(epoch + 1, num_epochs),
            "Train Avg Loss: {:.3f} |".format(loss_batches.mean()),
            "Train Accuracy: {:.3f} |".format(train_accuracy),
            "Train F1: {:.3f}".format(train_f1),
        )
    if loader_type == "validate":
        val_accuracy, val_f1 = get_f1_accuracy_by_confusion_matrix(val_confusion_matrix)
        print(
            "Val Accuracy: {:.3f} |".format(val_accuracy),
            "Val F1: {:.3f}".format(val_f1),
        )

    # plt.plot(np.arange(num_of_batches), loss_batches)
    # plt.title(f"Loss of epoch {epoch} by batch num")
    # plt.xlabel("Batch Num")
    # plt.ylabel("Loss")
    # plt.show()
