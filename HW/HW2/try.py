import re

import numpy as np
import pandas as pd
import torch
from gensim import downloader
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset

VEC_DIM = 200
GLOVE_PATH = f"glove-twitter-{VEC_DIM}"
WORD_2_VEC_PATH = "word2vec-google-news-300"


class SimpleDataSet(Dataset):
    def __init__(self, file_path, vector_type, tokenizer=None):
        self.file_path = file_path
        data = pd.read_csv(self.file_path, sep='\t')
        data.rename(columns={0: "words", 1: "label"}, inplace=True)
        data["label"] = data["label"].apply(lambda x: 0 if x == "O" else 1)
        self.sentences = data["words"].tolist()
        self.labels = data["label"].tolist()
        self.tags_to_idx = {
            tag: idx for idx, tag in enumerate(sorted(list(set(self.labels))))
        }
        self.idx_to_tag = {idx: tag for tag, idx in self.tags_to_idx.items()}
        self.vector_type = vector_type
        if vector_type == "tf-idf":
            if tokenizer is None:
                self.tokenizer = TfidfVectorizer(lowercase=True, stop_words=None)
                self.tokenized_sen = self.tokenizer.fit_transform(self.sentences)
            else:
                self.tokenizer = tokenizer
                self.tokenized_sen = self.tokenizer.transform(self.sentences)
            self.vector_dim = len(self.tokenizer.vocabulary_)
        else:
            if vector_type == "w2v":
                model = downloader.load(WORD_2_VEC_PATH)
            elif vector_type == "glove":
                model = downloader.load(GLOVE_PATH)
            else:
                raise KeyError(f"{vector_type} is not a supported vector type")
            representation, labels = [], []
            for sen, cur_labels in zip(self.sentences, self.labels):
                cur_rep = []
                for word in sen.split():
                    word = re.sub(r"\W+", "", word.lower())
                    if word not in model.key_to_index:
                        continue
                    vec = model[word]
                    cur_rep.append(vec)
                if len(cur_rep) == 0:
                    print(f"Sentence {sen} cannot be represented!")
                    continue
                cur_rep = np.stack(cur_rep).mean(
                    axis=0
                )  # HW TODO: change to token level classification
                representation.append(cur_rep)
                labels.append(cur_labels)
            self.labels = labels
            representation = np.stack(representation)
            self.tokenized_sen = representation
            self.vector_dim = representation.shape[-1]

    def __getitem__(self, item):
        cur_sen = self.tokenized_sen[item]
        if self.vector_type == "tf-idf":
            cur_sen = torch.FloatTensor(cur_sen.toarray()).squeeze()
        else:
            cur_sen = torch.FloatTensor(cur_sen).squeeze()
        label = self.labels[item]
        label = self.tags_to_idx[label]
        data = {"input_ids": cur_sen, "labels": label}
        return data

    def __len__(self):
        return len(self.labels)


def main():
    batch_size = 64
    train_ds = SimpleDataSet("data/train.tagged", vector_type="glove")
    print("created train")
    test_ds = SimpleDataSet("data/dev.tagged", vector_type="glove")
    data_sets = {"train": train_ds, "test": test_ds}
    data_loaders = {
        "train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
        "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False),
    }


def try():


if __name__ == "__main__":
    main()
