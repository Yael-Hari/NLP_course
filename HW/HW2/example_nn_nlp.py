import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.optim import Adam
from torch.utils.data import Dataset

# -----------------------
# Define the data set


class SentimentDataSet(Dataset):
    def __init__(self, file_path, tokenizer=None):
        self.file_path = file_path
        data = pd.read_csv(self.file_path)
        self.sentences = data["reviewText"].tolist()
        self.labels = data["label"].tolist()
        self.tags_to_idx = {
            tag: idx for idx, tag in enumerate(sorted(list(set(self.labels))))
        }
        self.idx_to_tag = {idx: tag for tag, idx in self.tags_to_idx.items()}
        if tokenizer is None:
            self.tokenizer = TfidfVectorizer(lowercase=True, stop_words=None)
            self.tokenized_sen = self.tokenizer.fit_transform(self.sentences)
        else:
            self.tokenizer = tokenizer
            self.tokenized_sen = self.tokenizer.transform(self.sentences)
        self.vocabulary_size = len(self.tokenizer.vocabulary_)

    def __getitem__(self, item):
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

from torch import nn


class SentimentNN(nn.Module):
    def __init__(self, vocab_size, num_classes, hidden_dim=100):
        super(SentimentNN, self).__init__()
        self.first_layer = nn.Linear(vocab_size, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        x = self.first_layer(input_ids)
        x = self.activation(x)
        x = self.second_layer(x)
        if labels is None:
            return x, None
        loss = self.loss(x, labels)
        return x, loss


# -------------------------
# Train loop

from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader


def train(model, data_sets, optimizer, num_epochs: int, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loaders = {
        "train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
        "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False),
    }
    model.to(device)

    best_acc = 0.0

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

                cur_num_correct = accuracy_score(
                    batch["labels"].cpu().view(-1), pred.view(-1), normalize=False
                )

                running_loss += loss.item() * batch_size
                running_acc += cur_num_correct

            epoch_loss = running_loss / len(data_sets[phase])
            epoch_acc = running_acc / len(data_sets[phase])

            epoch_acc = round(epoch_acc, 5)
            print(f"{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}")
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                with open("model.pkl", "wb") as f:
                    torch.save(model, f)
        print()

    print(f"Best Validation Accuracy: {best_acc:4f}")
    with open("model.pkl", "rb") as f:
        model = torch.load(f)
    return model


# -------------------------
# Putting it all together

train_ds = SentimentDataSet("amazon_sa/train.csv")
print("created train")
test_ds = SentimentDataSet("amazon_sa/test.csv", tokenizer=train_ds.tokenizer)
datasets = {"train": train_ds, "test": test_ds}
model = SentimentNN(num_classes=2, vocab_size=train_ds.vocabulary_size)
optimizer = Adam(params=model.parameters())
train(model=model, data_sets=datasets, optimizer=optimizer, num_epochs=15)
