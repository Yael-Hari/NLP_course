import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer
from project_evaluate import read_file


class DeEnPairsData:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/bibert-ende")  # TODO change to german bert
        self.bibert_model = AutoModel.from_pretrained("jhu-clsp/bibert-ende")   # TODO change to german bert

        self.train_path   = 'data/train.labeled'
        self.val_path     = 'data/val.labeled'

    def get_embedding_data_loader_labeled(self, file_path):
        file_en, file_de = read_file(file_path)

        de_bibert_embedding = []
        for de_str in file_de:
            tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(de_str))
            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

            # Obtain the contextual embeddings of BiBERT
            layer = -1  # Last layer
            de_bibert_emb = self.bibert_model(input_ids, output_hidden_states=True)[2][layer]
            de_bibert_embedding.append(de_bibert_emb)
        print()

    def get_tokens_labeled_data_loaders(self):
        X_train, y_train = self.get_tokens_labeled(self.train_path)
        X_val, y_val = self.get_tokens_labeled(self.val_path)

        # create datasets
        train_dataset = [*zip(X_train, y_train)]
        val_dataset = [*zip(X_val, y_val)]

        # create dataloader
        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False
        )

        return train_loader, val_loader

    def get_tokens_labeled(self, file_path):
        file_en, file_de = read_file(file_path)

        de_tokens = []
        en_tokens = []
        for de_str, en_str in zip(file_de, file_en):
            # tokenize german
            # tokens = de_str.replace('.', ' ').replace('?', ' ').split()
            tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(de_str))
            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            de_tokens.append(input_ids)

            # tokenize english
            # tokens = en_str.replace('.', ' ').replace('?', ' ').split()
            tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(en_str))
            input_ids = torch.tensor(tokens, dtype=torch.long)
            en_tokens.append(input_ids)

        return de_tokens, en_tokens



if __name__ == '__main__':
    dataset = DeEnPairsData()
    dataset.get_data_loader_labeled(dataset.train_path)
