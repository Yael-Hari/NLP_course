import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import print_epoch_details, remove_padding


def train_and_plot_LSTM(
    LSTM_model,
    train_loader,
    num_epochs: int,
    optimizer,
    loss_func,
    val_loader=None,
):

    # -------
    # GPU
    # -------
    # First checking if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Training on GPU.")
    else:
        print("No GPU available, training on CPU.")
    LSTM_model.to(device)

    # ----------------------------------
    # Epoch Loop
    # ----------------------------------
    # prepare for evaluate
    epoch_dict = {}
    epoch_dict["total_epochs"] = num_epochs
    loader_types = ["train", "validate"]
    num_classes = LSTM_model.num_classes
    for loader_type in loader_types:
        epoch_dict[loader_type] = {}
        epoch_dict[loader_type]["accuracy_list"] = []
        epoch_dict[loader_type]["f1_list"] = []
        epoch_dict[loader_type]["avg_loss_list"] = []

    for epoch_num in range(num_epochs):
        # prepare for evaluate
        for loader_type in loader_types:
            epoch_dict[loader_type]["confusion_matrix"] = np.zeros(
                [num_classes, num_classes]
            )
            epoch_dict[loader_type]["loss_list"] = []

        data_loaders = {"train": train_loader}
        if val_loader:
            data_loaders["validate"] = val_loader

        for loader_type, data_loader in data_loaders.items():
            for batch_num, (sentences, labels, sen_lengths) in enumerate(data_loader):
                # if training on gpu
                sentences, labels, sen_lengths = (
                    sentences.to(device),
                    labels.to(device),
                    sen_lengths.to(device),
                )

                # forward
                outputs = LSTM_model(sentences, sen_lengths)

                # labels
                packed_labels = pack_padded_sequence(
                    labels, sen_lengths, batch_first=True, enforce_sorted=False
                )
                unpacked_labels, labels_lengths = pad_packed_sequence(
                    packed_labels, batch_first=True
                )
                unpadded_labels = remove_padding(unpacked_labels, labels_lengths).long()
                labels_one_hot = one_hot(unpadded_labels, num_classes=num_classes)

                # loss
                loss = loss_func(outputs, labels_one_hot.float())
                epoch_dict[loader_type]["loss_list"].append(loss.detach().cpu())

                if loader_type == "train":
                    # backprop
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    optimizer.zero_grad()

                # predictions
                preds = outputs.argmax(dim=1).clone().detach().cpu()
                y_true = np.array(unpadded_labels.cpu().view(-1).int())
                y_pred = np.array(preds.view(-1))
                n_preds = len(y_pred)
                for i in range(n_preds):
                    epoch_dict[loader_type]["confusion_matrix"][y_true[i]][
                        y_pred[i]
                    ] += 1

            epoch_dict = print_epoch_details(
                epoch_dict,
                epoch_num,
                loader_type,
            )

    torch.save(LSTM_model.state_dict(), LSTM_model.model_save_path)
    return epoch_dict
