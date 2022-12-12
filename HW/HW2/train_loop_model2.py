import numpy as np
import torch

from utils import print_batch_details, print_epoch_details


def train_and_plot(
    NN_model,
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
    NN_model.to(device)

    # ----------------------------------
    # Epoch Loop
    # ----------------------------------
    # prepare for evaluate
    epoch_dict = {}
    epoch_dict["total_epochs"] = num_epochs
    loader_types = ["train", "validate"]
    num_classes = NN_model.num_classes
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
            num_of_batches = len(data_loader)

            for batch_num, (inputs, labels) in enumerate(data_loader):
                # if training on gpu
                inputs, labels = inputs.to(device), labels.to(device)
                batch_size = labels.shape[0]

                # forward
                # IMPORTANT - change the dimensions of x before it enters the NN,
                # batch size must always be first
                x = inputs.unsqueeze(0)  # x.size() -> [1, batch_size]
                x = x.view(batch_size, -1)  # x.size() -> [batch_size, 1]
                outputs = NN_model(x)

                # loss
                loss = loss_func(outputs.squeeze(), labels.long())
                epoch_dict[loader_type]["loss_list"].append(loss.detach().cpu())

                if loader_type == "train":
                    # backprop
                    loss.backward(retain_graph=True)
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    # nn.utils.clip_grad_norm_(NN_model.parameters(), clip)
                    optimizer.step()
                    optimizer.zero_grad()

                # predictions
                preds = outputs.argmax(dim=-1).clone().detach().cpu()
                y_true = np.array(labels.cpu().view(-1).int())
                y_pred = np.array(preds.view(-1))
                n_preds = len(y_pred)
                for i in range(n_preds):
                    epoch_dict[loader_type]["confusion_matrix"][y_true[i]][
                        y_pred[i]
                    ] += 1

                # print
                # if batch_num % 50 == 0:
                #     print_batch_details(
                #         num_of_batches,
                #         batch_num,
                #         loss,
                #         train_confusion_matrix,
                #         val_confusion_matrix,
                #         loader_type,
                #     )

            epoch_dict = print_epoch_details(
                epoch_dict,
                epoch_num,
                loader_type,
            )
    # torch.save(NN_model.state_dict(), NN_model.model_save_path)
    return epoch_dict
