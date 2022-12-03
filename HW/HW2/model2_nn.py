import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from preprocessing import NERDataset

# -------------------------------
# Define the Nueral Network Model
# -------------------------------


class NER_NN(nn.Module):
    # TODO: change hidden dim?
    def __init__(self, input_size, num_classes, hidden_dim=100):
        super(NER_NN, self).__init__()
        self.num_classes = num_classes
        self.first_layer = nn.Linear(input_size, hidden_dim)
        second_hidden_size = int(hidden_dim / 2)
        self.second_layer = nn.Linear(hidden_dim, second_hidden_size)
        self.third_layer = nn.Linear(second_hidden_size, num_classes)
        # TODO: check also other activations (tanh?)
        self.activation = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()  # for classification

    def forward(self, input_ids, labels=None):
        x = self.first_layer(input_ids)
        x = self.activation(x)
        x = self.second_layer(x)
        x = self.activation(x)
        x = self.third_layer(x)
        return x


# -------------------------
# Train loop
# -----------------------


def train_and_plot(
    NN_model, train_loader, num_epochs: int, batch_size: int, val_loader=None
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
    # Define Optimizer and Loss Function
    # ----------------------------------
    # many are available such as SGD, Adam, RMSprop, Adagrad..
    # TODO: change optimizer?
    # optimizer = torch.optim.SGD(params=NN_model.parameters(), lr=0.2)
    # clip = 1000  # gradient clipping
    optimizer = torch.optim.Adam(params=NN_model.parameters())

    loss_func = torch.nn.CrossEntropyLoss()

    # ----------------------------------
    # Epoch Loop
    # ----------------------------------

    for epoch in range(num_epochs):
        data_loaders = {"train": train_loader}
        if val_loader:
            data_loaders["validate"] = val_loader

        for loader_type, data_loader in data_loaders.items():
            # prepare for evaluate
            num_classes = NN_model.num_classes
            confusion_matrix = np.zeros([num_classes, num_classes])
            loss_batches_list = []
            num_of_batches = len(data_loader)

            for batch_num, (inputs, labels) in enumerate(tqdm(data_loader)):
                # if training on gpu
                inputs, labels = inputs.to(device), labels.to(device)
                batch_size = labels.shape[0]

                # optimize
                optimizer.zero_grad()

                # output
                # IMPORTANT - change the dimensions of x before it enters the NN,
                # batch size must always be first
                x = inputs.unsqueeze(0)  # x.size() -> [1, batch_size]
                x = x.view(batch_size, -1)  # x.size() -> [batch_size, 1]
                outputs = NN_model(x)

                # loss
                loss = loss_func(outputs.squeeze(), labels.long())
                loss_batches_list.append(loss.detach().cpu())

                if loader_type == "train":
                    # backprop
                    loss.backward()
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    # nn.utils.clip_grad_norm_(NN_model.parameters(), clip)
                    optimizer.step()

                # predictions
                preds = outputs.argmax(dim=-1).clone().detach().cpu()
                y_true = np.array(labels.cpu().view(-1).int())
                y_pred = np.array(preds.view(-1))
                n_preds = len(y_pred)
                for i in range(n_preds):
                    confusion_matrix[y_true[i]][y_pred[i]] += 1
                # print
                if batch_num % 50 == 0:
                    print_batch_details(
                        num_of_batches, batch_num, loss, confusion_matrix
                    )

            print_epoch_details(num_epochs, epoch, confusion_matrix, loss_batches_list)


def get_f1_accuracy_by_confusion_matrix(confusion_matrix):
    # only for 2 classes !!!
    # TODO: adapt fot more classes
    Tn = confusion_matrix[0][0]
    Fn = confusion_matrix[1][0]
    Tp = confusion_matrix[1][1]
    Fp = confusion_matrix[0][1]

    accuracy = (Tn + Tp) / (Tn + Tp + Fn + Fp)
    precision = Tp / (Tp + Fp)
    recall = Tp / (Tp + Fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, f1


def print_batch_details(num_of_batches, batch_num, loss, confusion_matrix):
    accuracy, f1 = get_f1_accuracy_by_confusion_matrix(confusion_matrix)
    print(
        "Batch: {}/{} |".format(batch_num + 1, num_of_batches),
        "Batch Loss: {:.3f} |".format(loss),
        "Accuracy: {:.3f} |".format(accuracy),
        "F1: {:.3f}".format(f1),
    )


def print_epoch_details(num_epochs, epoch, confusion_matrix, loss_batches_list):
    loss_batches = np.array(loss_batches_list)
    num_of_batches = len(loss_batches)
    accuracy, f1 = get_f1_accuracy_by_confusion_matrix(confusion_matrix)

    print(
        "Epoch: {}/{}...".format(epoch + 1, num_epochs),
        "Avg Loss: {:.3f}...".format(loss_batches.mean()),
        "Accuracy: {:.3f}".format(accuracy),
        "F1: {:.3f}".format(f1),
    )
    plt.plot(x=np.arange(num_of_batches), y=loss_batches)
    plt.title(f"Loss of epoch {epoch} by batch num")
    plt.xlabel("Batch Num")
    plt.ylabel("Loss")
    plt.show()


# -------------------------
# Putting it all together
# -------------------------
def main():
    embedding_type = "glove"
    batch_size = 32
    NER_dataset = NERDataset(embedding_model_type=embedding_type, batch_size=batch_size)
    train_loader, dev_loader, test_loader = NER_dataset.get_preprocessed_data()

    # TODO: change params?

    # option 1:
    # is identity - yes / no
    num_classes = 2
    # option 2:
    # different classification for differnet identities
    # num_classes = ?

    num_epochs = 5
    hidden_dim = 64
    # TODO: change according to the embedding
    # single vector size
    input_size = NERDataset.VEC_DIM * (1 + 2 * NERDataset.WINDOW_R)

    NN_model = NER_NN(
        input_size=input_size, num_classes=num_classes, hidden_dim=hidden_dim
    )

    train_and_plot(
        NN_model=NN_model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        batch_size=batch_size,
        val_loader=dev_loader,
    )


if __name__ == "__main__":
    main()
