import torch
from gensim import downloader
from torch.utils.data import DataLoader

"""
Get Dataset:
    input: file with words and POS for each word.
    output: for each word - concat embddings of word and embedding of POS.
        * word embedding - glove + word2vec concatenated
        * POS embedding - ONE HOT / learnable
"""


class SentencesEmbeddingDataset:
    """
    Options of words embeddings:
    fasttext-wiki-news-subwords-300
    glove-twitter-100
    glove-twitter-200
    glove-twitter-25
    glove-twitter-50
    glove-wiki-gigaword-100
    glove-wiki-gigaword-200
    glove-wiki-gigaword-300
    glove-wiki-gigaword-50
    word2vec-google-news-300
    word2vec-ruscorpora-300
    """

    def __init__(
        self,
        embedding_model_path,
        list_embedding_paths,
        word_embedding_dim_list,
        word_embedding_dim,
        pos_embedding_name,
        pos_embedding_dim,
    ):
        """_summary_
        pos_embedding_name: "onehot" / "learn"
        """
        self.embedding_model_path = embedding_model_path
        self.list_embedding_paths = list_embedding_paths
        self.word_embedding_dim_list = word_embedding_dim_list
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_name = pos_embedding_name
        self.pos_embedding_dim = pos_embedding_dim

        print("preparing embedding...")
        if self.list_embedding_paths is None:
            self.embedding_model = downloader.load(self.embedding_model_path)
        else:
            self.embedding_model = [
                downloader.load(self.list_embedding_paths[0]),
                downloader.load(self.list_embedding_paths[1]),
            ]

        # paths to data
        self.train_path = "train.labeled"
        self.val_path = "test.labeled"
        self.comp_path = "comp.unlabeled"

        # embeddings

    def get_pos_embeddings(self):
        if self.pos_embedding_name == "learn":
            pos_embeddings = {}
            pos_values = self.get_pos_values()
            for pos in pos_values:
                pos_embeddings[pos] = torch.rand(
                    self.pos_embedding_dim, requires_grad=True
                )
        elif self.pos_embedding_name == "onehot":
            pos_embeddings = {}
        else:
            raise ValueError(f"unvalid {self.pos_embedding_name=}")

    def get_pos_values(self):
        # TODO: complete
        pass

    def get_data_loaders(self, batch_size):
        (
            X_train,
            y_train,
            sentences_lengths_train,
            X_val,
            y_val,
            sentences_lengths_val,
        ) = self.get_datasets()

        # create datasets
        train_dataset = [*zip(X_train, y_train, sentences_lengths_train)]
        val_dataset = [*zip(X_val, y_val, sentences_lengths_val)]

        # create dataloader
        torch.manual_seed(42)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        comp_dataloader = self.get_comp_loader(batch_size=batch_size)

        return train_dataloader, val_dataloader, comp_dataloader

    def get_comp_loader(self, batch_size):
        X_comp, _, sentences_lengths_comp = self._get_dataset_from_path(
            self.comp_path, tagged=False
        )
        comp_dataset = [*zip(X_comp, sentences_lengths_comp)]
        comp_dataloader = DataLoader(comp_dataset, batch_size=batch_size, shuffle=False)

        return comp_dataloader

    def get_datasets(self):
        X_train, y_train, sentences_lengths_train = self._get_dataset_from_path(
            self.train_path, tagged=True
        )
        X_val, y_val, sentences_lengths_val = self._get_dataset_from_path(
            self.val_path, tagged=True
        )
        return (
            X_train,
            y_train,
            sentences_lengths_train,
            X_val,
            y_val,
            sentences_lengths_val,
        )

    def _get_dataset_from_path(self, path: str, tagged: bool):
        # TODO: change

        EOF = "\ufeff"
        empty_lines = ["", "\t", EOF]

        # load data
        with open(path, "r", encoding="utf8") as f:
            raw_lines = f.read()

        # split to sentences
        sentences = []
        curr_s = []
        for word_tag in raw_lines.split("\n"):
            if word_tag not in empty_lines:
                if EOF == word_tag:  # EOF
                    continue
                if tagged:
                    word_tag = tuple(word_tag.split("\t"))
                curr_s.append(word_tag)
            else:
                if len(curr_s) > 0:
                    sentences.append(curr_s)
                curr_s = []

        X = []
        y = []
        sentences_lengths = []

        # embeddings
        for sentence in sentences:

            X_curr_sentence = []
            y_curr_sentence = []

            for word in sentence:
                if tagged:
                    word, tag = word
                    y_curr_sentence.append(tag)
                word = word.lower()

                # single embedding
                if self.list_embedding_paths is None:
                    if word not in self.embedding_model.key_to_index:
                        if self.learn_unknown:
                            word_vec = torch.rand(self.vec_dim, requires_grad=True)
                        else:
                            word_vec = torch.zeros(self.vec_dim)
                    else:
                        word_vec = torch.tensor(self.embedding_model[word])

                # concatenated embeddings
                else:
                    word_vec = []
                    # embedding #0:
                    for i in range(len(self.embedding_model)):

                        if word not in self.embedding_model[i].key_to_index:
                            if self.learn_unknown:
                                word_vec.append(
                                    torch.rand(
                                        self.list_vec_dims[i], requires_grad=True
                                    )
                                )
                            else:
                                word_vec.append(torch.zeros(self.list_vec_dims[i]))
                        else:
                            word_vec.append(torch.tensor(self.embedding_model[i][word]))

                    word_vec = torch.concat(word_vec)
                X_curr_sentence.append(word_vec)

            X_curr_sentence = torch.stack(X_curr_sentence)
            X.append(X_curr_sentence)
            sentences_lengths.append(len(X_curr_sentence))
            if tagged:
                y.append(y_curr_sentence)

        if tagged:
            # make labels binary
            y = self.get_true_deps()

        return X, y, sentences_lengths

    def get_true_deps(self):
        # TODO: complete
        pass
