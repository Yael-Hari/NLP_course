import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

# -----------------------
# Define the data set
# -----------------------


class NER_DataSet(Dataset):
    def __init__(self, file_path, tokenizer=None):
        # TODO: remove this section of open file?
        self.file_path = file_path
        data = pd.read_csv(self.file_path)
        self.sentences = data["reviewText"].tolist()
        self.labels = data["label"].tolist()

        self.tags_to_idx = {
            tag: idx for idx, tag in enumerate(sorted(list(set(self.labels))))
        }
        self.idx_to_tag = {idx: tag for tag, idx in self.tags_to_idx.items()}
        if tokenizer is None:
            # TODO: change tfidf to other vector representation?
            self.tokenizer = TfidfVectorizer(lowercase=True, stop_words=None)
            self.tokenized_sen = self.tokenizer.fit_transform(self.sentences)
        else:
            self.tokenizer = tokenizer
            # TODO: transform word or sentence?
            self.tokenized_sen = self.tokenizer.transform(self.sentences)
        # TODO: change?
        self.vocabulary_size = len(self.tokenizer.vocabulary_)

    def __getitem__(self, item):
        # NOTE: question - where does the function call to this?
        cur_sen = self.tokenized_sen[item]
        cur_sen = torch.FloatTensor(cur_sen.toarray()).squeeze()
        label = self.labels[item]
        label = self.tags_to_idx[label]
        # label = torch.Tensor(label)
        data = {"input_ids": cur_sen, "labels": label}
        return data

    def __len__(self):
        return len(self.sentences)


# -----------------------
# Define the Model
# -----------------------


class NER_NN(nn.Module):
    # TODO: change hidden dim?
    def __init__(self, vocab_size, num_classes, hidden_dim=100):
        super(NER_NN, self).__init__()
        self.first_layer = nn.Linear(vocab_size, hidden_dim)
        # TODO: add layer? (hidden, hidden)
        self.second_layer = nn.Linear(hidden_dim, num_classes)
        # TODO: check also other activations (tanh?)
        self.activation = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()  # for classification

    def forward(self, input_ids, labels=None):
        # NOTE: question - where do we call this function?
        x = self.first_layer(input_ids)
        x = self.activation(x)
        x = self.second_layer(x)
        if labels is None:
            return x, None
        loss = self.loss(x, labels)
        return x, loss


# -------------------------
# Train loop
# -----------------------

# TODO: change batch size?
def train(model, data_sets, optimizer, num_epochs: int, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NOTE: question - what does the dataloader do?
    data_loaders = {
        "train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
        "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False),
    }
    model.to(device)

    best_f1 = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc = 0.0
            # TODO: add running f1

            for batch in data_loaders[phase]:
                batch_size = 0
                for k, v in batch.items():
                    batch[k] = v.to(device)
                    batch_size = v.shape[0]

                optimizer.zero_grad()
                if phase == "train":
                    outputs, loss = model(**batch)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs, loss = model(**batch)
                pred = outputs.argmax(dim=-1).clone().detach().cpu()

                cur_num_correct = f1_score(
                    batch["labels"].cpu().view(-1), pred.view(-1), normalize=False
                )

                running_loss += loss.item() * batch_size
                running_acc += cur_num_correct

            epoch_loss = running_loss / len(data_sets[phase])
            epoch_acc = running_acc / len(data_sets[phase])
            # TODO: calc f1
            # epoch_f1 = ?

            epoch_f1 = round(epoch_f1, 5)
            print(f"{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}")
            if phase == "test" and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                with open("model.pkl", "wb") as f:
                    torch.save(model, f)
        print()

    print(f"Best Validation F1: {best_f1:4f}")
    with open("model.pkl", "rb") as f:
        model = torch.load(f)
    return model


# -------------------------
# Putting it all together
# -------------------------

train_ds = NER_DataSet("data/train.tagged")
print("created train")
dev_ds = NER_DataSet("data/dev.csv", tokenizer=train_ds.tokenizer)
ds_to_check = dev_ds

datasets = {"train": train_ds, "test": ds_to_check}
# TODO: change num classes?
model = NER_NN(num_classes=2, vocab_size=train_ds.vocabulary_size)
# TODO: change optimizer?
optimizer = Adam(params=model.parameters())
# TODO: change num epochs?
train(model=model, data_sets=datasets, optimizer=optimizer, num_epochs=15)
