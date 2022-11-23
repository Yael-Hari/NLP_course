import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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
       x = self.layer_1(x)        # x.size() -> [batch_size, self.hidden_dim]
       x = self.activation(x)     # x.size() -> [batch_size, self.hidden_dim]
       x = self.layer_2(x)        # x.size() -> [batch_size, 1]
       return x

net = RegressioNet()
print(net)


# -----------------------------------------------------
# Step 2 - Data Creation and split to train, val, test

# get vectors representaions?


# -----------------------------------------------------
# Step 3 - Optimizer and Loss

# Define Optimizer and Loss Function
optimizer = torch.optim.SGD(net.parameters(), lr=0.2) # many others are available (such as Adam, RMSprop, Adagrad..)
loss_func = torch.nn.CrossEntropyLoss()

# ------------------------------------
# Step 4 - Training

batch_size = 20

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
valid_data = TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y).float())
test_data = TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float())

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)


# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size()) # batch_size
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)


# First checking if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')


# Define training params
epochs = 1

counter = 0
print_every = 100
clip = 1000 # gradient clipping

# move model to GPU, if available
net = net.float()
net.to(device)

net.train()
# train for some number of epochs
