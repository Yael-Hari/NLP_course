import torch
from gensim import downloader
from torch.nn import functional as F
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
        word_embedding_name,
        list_embedding_paths,
        word_embedding_dim_list,
        word_embedding_dim,
        pos_embedding_name,
        pos_embedding_dim,
    ):
        """
        pos_embedding_name: "onehot" / "learn"
        """
        self.word_embedding_name = word_embedding_name
        self.list_embedding_paths = list_embedding_paths
        self.word_embedding_dim_list = word_embedding_dim_list
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_name = pos_embedding_name
        self.pos_embedding_dim = pos_embedding_dim
        self.pos_embeddings = None
        self.learn_unknown_word = True

        print("preparing embedding...")
        if self.list_embedding_paths is None:
            self.embedding_model = downloader.load(self.word_embedding_name)
        else:
            self.embedding_model = [
                downloader.load(self.list_embedding_paths[0]),
                downloader.load(self.list_embedding_paths[1]),
            ]

        torch.manual_seed(42)

        # paths to data
        self.train_path = "train.labeled"
        self.val_path = "test.labeled"
        self.comp_path = "comp.unlabeled"

    def get_pos_embeddings(self, pos_values):
        if self.pos_embedding_name == "learn":
            pos_embeddings = {}
            pos_values.append("unknown")
            for pos in pos_values:
                pos_embeddings[pos] = torch.rand(
                    self.pos_embedding_dim, requires_grad=True
                )
        elif self.pos_embedding_name == "onehot":
            pos_values.append("unknown")
            pos_one_hot = F.one_hot(torch.arange(0, len(pos_values)))
            pos_embeddings = {pos: pos_one_hot[i] for i, pos in enumerate(pos_values)}
            # update pos_embedding_dim
            self.pos_embedding_dim = len(pos_one_hot[0])
        else:
            raise ValueError(f"unvalid {self.pos_embedding_name=}")
        return pos_embeddings

    def get_data_loaders(self, batch_size):
        """
        not in use
        """
        (train_dataset, val_dataset, comp_dataset) = self.get_datasets()
        X_train, y_train = train_dataset
        X_val, y_val = val_dataset
        X_comp, _ = comp_dataset

        # create datasets
        train_dataset = [*zip(X_train, y_train)]
        val_dataset = [*zip(X_val, y_val)]
        comp_dataset = X_comp

        # create dataloader
        shuffle_train = True
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle_train
        )
        shuffle_validate = False
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=shuffle_validate
        )
        shuffle_comp = False
        comp_dataloader = DataLoader(
            comp_dataset, batch_size=batch_size, shuffle=shuffle_comp
        )
        return train_dataloader, val_dataloader, comp_dataloader

    def get_datasets(self):
        """
        output: for each dataset: X and y tensors
            X - tensor of sentences embeddings tensors
            y - tensor of dentences true dependencies tensors
        """
        # ! need to call first with train to init self.pos_embeddings
        train_dataset = self._get_dataset_from_path(self.train_path, tagged=True)
        val_dataset = self._get_dataset_from_path(self.val_path, tagged=True)
        comp_dataset = self._get_dataset_from_path(self.comp_path, tagged=False)
        return (train_dataset, val_dataset, comp_dataset)

    def _get_dataset_from_path(self, path: str, tagged: bool):
        """
        input: path, tagged or not
        output: X and y tensors
            X - tensor of sentences embeddings tensors
            y - tensor of dentences true dependencies tensors

        ! need to call first with train to init self.pos_embeddings
        """
        X = []
        y = []
        sentences_lengths = []

        # get raw lines
        with open(path, "r", encoding="utf8") as f:
            raw_lines = f.read()
        # split to sentences
        sentences_words_and_pos, sentences_deps = self.get_sentences(raw_lines)
        # get pos embeddings by train data set
        if "train" in path:
            pos_values = [[pos for _, pos in sentence] for sentence in sentences_words_and_pos]
            pos_values = list(set([j for i in pos_values for j in i]))
            self.pos_embeddings = self.get_pos_embeddings(pos_values)
        # get embedding for each sentence
        for sentence in sentences_words_and_pos:
            sentence_embedding = self.get_sentence_embedding(sentence)
            X.append(sentence_embedding)
            sentences_lengths.append(len(sentence_embedding))
        if tagged:
            # convert to tensor
            y = torch.stack(sentences_deps)

        return X, y

    def get_sentences(self, raw_lines):
        """
        input:
            lines looks like:
            __________________________________________________________________________
            |     0        |   1    | 2 |     3     | 4 | 5 |      6     | 7 | 8 | 9 |
            __________________________________________________________________________
            token_counter  | token  | - | token POS | - | - | token head | - | - | - |
            __________________________________________________________________________
            columns are splitted by \t
            raws are splitted by \n
        output:
            sentences_words_and_pos - list of couples:  (token, token_POS)
            sentences_deps - list of couples:           tensor(token_counter, token_head)
        """
        # init
        sentences_words_and_pos = []
        sentences_deps = []
        curr_s_words_and_pos = []
        curr_s_deps = []
        # empty lines
        EOF = "\ufeff"
        empty_lines = ["", "\t", EOF]

        for raw_line in raw_lines.split("\n"):
            if raw_line not in empty_lines:
                input_values = raw_line.split("\t")
                curr_s_words_and_pos.append((input_values[1], input_values[3]))
                curr_s_deps.append(torch.tensor([int(input_values[0]), int(input_values[6])]))
            else:
                # got empty line -> finish current sentence
                if len(curr_s_words_and_pos) > 0:
                    sentences_words_and_pos.append(curr_s_words_and_pos)
                    sentences_deps.append(curr_s_deps)
                # init for next sentence
                curr_s_words_and_pos = []
                curr_s_deps = []
        return sentences_words_and_pos, sentences_deps

    def get_sentence_embedding(self, sentence):
        """
        input: sentence
        output: sentence embedding
            concat word_embdding + pos embdding
        """
        senetnce_embeddings = []

        for word, pos in sentence:
            # pos embedding
            pos_embedding = self.get_pos_embedding(pos)
            # word embedding
            word_embedding = self.get_word_embedding(word)
            # concat
            word_pos_embedding = torch.concat([word_embedding, pos_embedding])
            senetnce_embeddings.append(word_pos_embedding)

        senetnce_embeddings = torch.stack(senetnce_embeddings)
        return senetnce_embeddings

    def get_pos_embedding(self, pos):
        """
        input: pos
        output: pos embedding
        """
        if pos in self.pos_embeddings.keys():
            pos_embedding = self.pos_embeddings[pos]
        else:
            pos_embedding = self.pos_embeddings["unknown"]
        return pos_embedding

    def get_word_embedding(self, word):
        """
        input: word
        output: word embedding
            can be by single embdding path like glove only
            can be by multiple embeddings paths like glove and word2vec
        """
        # lower
        word = word.lower()

        # single embedding
        if self.list_embedding_paths is None:
            if word not in self.embedding_model.key_to_index:
                if self.learn_unknown_word:
                    word_vec = torch.rand(self.word_embedding_dim, requires_grad=True)
                else:
                    word_vec = torch.zeros(self.word_embedding_dim)
            else:
                word_vec = torch.tensor(self.embedding_model[word])

        # concatenated embeddings
        else:
            word_vec = []
            # embedding #0:
            for i in range(len(self.embedding_model)):
                if word not in self.embedding_model[i].key_to_index:
                    if self.learn_unknown_word:
                        word_vec.append(
                            torch.rand(self.list_vec_dims[i], requires_grad=True)
                        )
                    else:
                        word_vec.append(torch.zeros(self.list_vec_dims[i]))
                else:
                    word_vec.append(torch.tensor(self.embedding_model[i][word]))

            word_vec = torch.concat(word_vec)

        return word_vec
