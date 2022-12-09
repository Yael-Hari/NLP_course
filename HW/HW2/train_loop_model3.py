import numpy as np
import torch
from tqdm import tqdm

from utils import print_batch_details, print_epoch_details


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
    for epoch in range(num_epochs):
        data_loaders = {"train": train_loader}
        if val_loader:
            data_loaders["validate"] = val_loader

        # prepare for evaluate
        num_classes = LSTM_model.num_classes
        train_confusion_matrix = np.zeros([num_classes, num_classes])
        val_confusion_matrix = None
        if val_loader:
            val_confusion_matrix = np.zeros([num_classes, num_classes])
        train_loss_batches_list = []

        for loader_type, data_loader in data_loaders.items():
            num_of_batches = len(data_loader)

            for batch_num, (sentences, labels, sen_lengths) in enumerate(
                tqdm(data_loader)
            ):
                # if training on gpu
                sentences, labels, sen_lengths = (
                    sentences.to(device),
                    labels.to(device),
                    sen_lengths.to(device),
                )

                # forward
                # ??????????????????????
                # IMPORTANT - change the dimensions of x before it enters the NN,
                # batch size must always be first
                # x = inputs.unsqueeze(0)  # x.size() -> [1, batch_size]
                # batch_size = labels.shape[0]
                # x = x.view(batch_size, -1)  # x.size() -> [batch_size, 1]
                # ??????????????????????
                outputs = LSTM_model(sentences, sen_lengths)

                # loss
                loss = loss_func(outputs.squeeze(), labels.long())

                if loader_type == "train":
                    train_loss_batches_list.append(loss.detach().cpu())
                    # backprop
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    optimizer.zero_grad()

                # predictions
                preds = outputs.argmax(dim=-1).clone().detach().cpu()
                y_true = np.array(labels.cpu().view(-1).int())
                y_pred = np.array(preds.view(-1))
                n_preds = len(y_pred)
                for i in range(n_preds):
                    if loader_type == "train":
                        train_confusion_matrix[y_true[i]][y_pred[i]] += 1
                    if loader_type == "validate":
                        val_confusion_matrix[y_true[i]][y_pred[i]] += 1
                # print
                if batch_num % 50 == 0:
                    print_batch_details(
                        num_of_batches,
                        batch_num,
                        loss,
                        train_confusion_matrix,
                        val_confusion_matrix,
                        loader_type,
                    )

            print_epoch_details(
                num_epochs,
                epoch,
                train_confusion_matrix,
                train_loss_batches_list,
                val_confusion_matrix,
                loader_type,
            )
    torch.save(LSTM_model.state_dict(), LSTM_model.model_save_path)
