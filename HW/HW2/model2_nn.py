import torch
from torch import nn

from preprocessing import WordsEmbeddingDataset
from train_loop_model2 import train_and_plot
from utils import plot_epochs_results

# -------------------------------
# Define the Nueral Network Model
# -------------------------------


class NER_NN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_dim, model_save_path, dropout):
        super(NER_NN, self).__init__()
        self.num_classes = num_classes
        self.first_layer = nn.Linear(input_size, hidden_dim)
        second_hidden_size = int(hidden_dim / 2)
        self.second_layer = nn.Linear(hidden_dim, second_hidden_size)
        self.third_layer = nn.Linear(second_hidden_size, num_classes)
        self.activation = nn.Tanh()
        self.model_save_path = model_save_path
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids, labels=None):
        x = self.first_layer(input_ids)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.second_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.third_layer(x)
        return x


# -------------------------
# Run
# -------------------------
def main():
    embedding_type = "glove"
    batch_size = 32
    num_classes = 2
    num_epochs = 5
    hidden_dim = 64
    lr = 0.001
    dropout = 0.2
    torch.manual_seed(42)
    NER_dataset = WordsEmbeddingDataset(
        embedding_model_type=embedding_type, learn_unknown=False
    )
    train_loader, dev_loader = NER_dataset.get_data_loaders(batch_size=batch_size)
    input_size = NER_dataset.vec_dim  # single vector size
    model_save_path = (
        f"model2_stateDict_batchSize_{batch_size}_hidden_{hidden_dim}_lr_{lr}.pt"
    )

    NN_model = NER_NN(
        input_size=input_size,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        model_save_path=model_save_path,
        dropout=dropout,
    )

    optimizer = torch.optim.Adam(params=NN_model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()
    loss_func_name = "CrossEntropy"
    # optimizer = torch.optim.SGD(params=NN_model.parameters(), lr=0.2)
    # clip = 1000  # gradient clipping

    epoch_dict = train_and_plot(
        NN_model=NN_model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        val_loader=dev_loader,
        optimizer=optimizer,
        loss_func=loss_func,
    )

    plot_epochs_results(
        epoch_dict=epoch_dict,
        hidden=hidden_dim,
        embedding_name=embedding_type,
        dropout=dropout,
        loss_func_name=loss_func_name,
        class_weights=[0],
    )


if __name__ == "__main__":
    main()
