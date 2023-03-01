import torch
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer
from project_evaluate import read_file


class DeEnPairsData:
    def __init__(self):
        self.model_name   = "bert-base-german-cased"
        self.tokenizer    = AutoTokenizer.from_pretrained(self.model_name)  # tokenizer.decode(["input_ids"])
        self.bert_model   = AutoModel.from_pretrained(self.model_name)

        self.train_path   = 'data/train.labeled'
        self.val_path     = 'data/val.labeled'

        # ~~~~~~~~~~~~~~~~ for DEBUG ~~~~~~~~~~~~~~~~
        self.train_path   = 'data/mini_train.labeled'
        self.val_path     = 'data/mini_val.labeled'

    # ~~~~~~~~~~~~~~~~~ FINE TUNE BERT

    def get_labeled_loaders(self):

        X_train, y_train = self.get_labeled_data_bert(self.train_path)
        X_val, y_val = self.get_labeled_data_bert(self.val_path)

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

    def get_labeled_data_bert(self, file_path):

        # read data files
        file_en, file_de = read_file(file_path)

        de_bert_embedding = []
        for de_str in file_de:
            # Transform input to tokens
            tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(de_str))
            input_ids = torch.tensor(tokens, dtype=torch.long)  #.unsqueeze(0)

            # Obtain the contextual embeddings of BERT
            layer = -1  # Last layer
            de_bert_emb = self.bert_model(input_ids, output_hidden_states=True)[2][layer]
            de_bert_embedding.append(de_bert_emb)

            # Transform input tokens
            inputs = self.tokenizer("Hello world!", return_tensors="pt")

            # Model apply
            outputs = self.bert_model(**inputs)
        # TODO
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
    train_loader, val_loader = dataset.get_labeled_loaders()
