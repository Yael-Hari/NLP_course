import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------
# Step 1 - Creating our neural-net model

# We create a FC regression network, with 2 layers.


class RegressioNet(nn.Module):
    def __init__(self):
        super(RegressioNet, self).__init__()
        self.hidden_dim = 10
        self.layer_1 = torch.nn.Linear(1, self.hidden_dim)
        self.layer_2 = torch.nn.Linear(self.hidden_dim, 1)
        self.activation = F.relu

    def forward(self, x):
        x = self.layer_1(x)  # x.size() -> [batch_size, self.hidden_dim]
        x = self.activation(x)  # x.size() -> [batch_size, self.hidden_dim]
        x = self.layer_2(x)  # x.size() -> [batch_size, 1]
        return x


net = RegressioNet()
print(net)


# -----------------------------------------------------
# Step 2 - Data Creation and split to train, val, test

# Visualize our data
import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(100000)

# y = exp(x) + 5 + noise
y = np.exp(x) + np.random.rand(100000) * 0.001 + 5

plt.scatter(x, y)
plt.show()

split_frac = 0.8

# split data into training, validation, and test data (x and y)

split_idx = int(len(x) * split_frac)
train_x, remaining_x = x[:split_idx], x[split_idx:]
train_y, remaining_y = y[:split_idx], y[split_idx:]

test_idx = int(len(remaining_x) * 0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

# print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print(
    "Train set: \t\t{}".format(train_x.shape),
    "\nValidation set: \t{}".format(val_x.shape),
    "\nTest set: \t\t{}".format(test_x.shape),
)


# -----------------------------------------------------
# Step 3 - Optimizer and Loss

# Define Optimizer and Loss Function
optimizer = torch.optim.SGD(
    net.parameters(), lr=0.2
)  # many others are available (such as Adam, RMSprop, Adagrad..)
loss_func = torch.nn.CrossEntropyLoss()

# ------------------------------------
# Step 4 - Training

batch_size = 20

# create Tensor datasets
train_data = TensorDataset(
    torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()
)
valid_data = TensorDataset(
    torch.from_numpy(val_x).float(), torch.from_numpy(val_y).float()
)
test_data = TensorDataset(
    torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float()
)

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)


# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print("Sample input size: ", sample_x.size())  # batch_size
print("Sample input: \n", sample_x)
print()
print("Sample label size: ", sample_y.size())  # batch_size
print("Sample label: \n", sample_y)


# First checking if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("Training on GPU.")
else:
    print("No GPU available, training on CPU.")


# Define training params
epochs = 1

counter = 0
print_every = 100
clip = 1000  # gradient clipping

# move model to GPU, if available
net = net.float()
net.to(device)

net.train()
# train for some number of epochs


for e in range(epochs):
    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        # if training on gpu
        inputs, labels = inputs.to(device), labels.to(device)

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        # x.size() -> [batch_size]
        batch_size = inputs.size(0)
        # IMPORTANT - change the dimensions of x before it enters the NN,
        # batch size must always be first
        x = inputs.unsqueeze(0)  # x.size() -> [1, batch_size]
        x = x.view(batch_size, -1)  # x.size() -> [batch_size, 1]
        predictions = net(x)

        # calculate the loss and perform backprop
        loss = loss_func(predictions.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_losses = []
            net.eval()
            print_flag = True
            for inputs, labels in valid_loader:
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                if print_flag:
                    inputs, labels = zip(*sorted(zip(inputs.numpy(), labels.numpy())))
                    inputs = torch.from_numpy(np.asarray(inputs))
                    labels = torch.from_numpy(np.asarray(labels))
                inputs, labels = inputs.to(device), labels.to(device)

                # get the output from the model
                # x.size() -> [batch_size]
                batch_size = inputs.size(0)
                # IMPORTANT - change the dimensions of x before it enters the NN,
                # batch size must always be first
                x = inputs.unsqueeze(0)  # x.size() -> [1, batch_size]
                x = x.view(batch_size, -1)  # x.size() -> [batch_size, 1]
                val_predictions = net(x)
                val_loss = loss_func(val_predictions.squeeze(), labels.float())

                val_losses.append(val_loss.item())
                if print_flag:
                    print_flag = False
                    # plot and show learning process
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.cla()
                    ax.scatter(inputs.cpu().data.numpy(), labels.cpu().data.numpy())
                    ax.plot(
                        inputs.cpu().data.numpy(),
                        val_predictions.cpu().data.numpy(),
                        "r-",
                        lw=2,
                    )
                    ax.text(
                        0.5,
                        0,
                        "Loss=%.4f" % np.mean(val_losses),
                        fontdict={"size": 10, "color": "red"},
                    )
                    plt.pause(0.1)
                    ax.clear()

            net.train()
            print(
                "Epoch: {}/{}...".format(e + 1, epochs),
                "Step: {}...".format(counter),
                "Loss: {:.6f}...".format(loss.item()),
                "Val Loss: {:.6f}".format(np.mean(val_losses)),
            )
plt.show()
