import numpy as np
from gensim import downloader
import pandas as pd
from sklearn import preprocessing
import csv
import torch


class NERDataset:
    WINDOW_R = 2
    VEC_DIM = 25  # TODO change to 200
    GLOVE_PATH = f'glove-twitter-{VEC_DIM}'
    WORD2VEC_PATH = 'word2vec-google-news-300'

    def __init__(self, embedding_model_type="glove"):
        self.embedding_model_type = embedding_model_type

        if self.embedding_model_type == "glove":
            self.embedding_model_path = NERDataset.GLOVE_PATH
        elif self.embedding_model_type == "word2vec":
            self.embedding_model_path = NERDataset.WORD2VEC_PATH
        else:
            raise Exception("invalid model name")

        self.embedding_model = self._load_embedding_model()

        self.label_encoder = preprocessing.LabelEncoder()

        self.train_path = "data/train.tagged"
        self.dev_path = "data/dev.tagged"
        self.test_path = "data/test.untagged"

    def _load_embedding_model(self):
        # return 1
        print(f"preparing {self.embedding_model_type}...")
        glove = downloader.load(self.embedding_model_path)
        return glove

    def _get_dataset(self, path: str, tagged: bool):
        W = NERDataset.WINDOW_R

        if tagged:
            # load data
            df = pd.read_csv(path, sep='\t', header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')
            raw_tuples = df.to_records(index=False)

            # split to sentences
            sentences = []
            curr_s = []
            for word_tag in raw_tuples:
                if not pd.isna(word_tag[0]):
                    curr_s.append(word_tag)
                else:
                    sentences.append(curr_s)
                    curr_s = []

            # concat every word vec to WINDOW_R words behind and WINDOW_R after
            sentences = [[("*", "*")] * W + s + [("~", "~")] * W for s in sentences]

            # "*", "~" are 0 vecs for now. TODO: something else?
            # also, if a word not found in embedding_model, we put 0 vec. TODO: maybe there is a better idea?
            # TODO: if selected vector is not in embedding_model: for now we give it 0
            X = []
            y = []
            for sentence in sentences:
                sentence_len = len(sentence)
                for i_s in range(W, sentence_len - W):
                    s_word, s_tag = sentence[i_s]
                    vecs_list = []
                    for c_word, c_tag in sentence[(i_s - W): (i_s + W + 1)]:
                        if (c_word == "*") or (c_word == "~") or (c_word not in self.embedding_model.key_to_index):
                            u_c = np.zeros(NERDataset.VEC_DIM)
                        else:
                            u_c = self.embedding_model[c_word]
                        vecs_list.append(u_c)
                    concated_vec = np.concatenate(vecs_list)
                    X.append(concated_vec)
                    y.append(s_tag)
            return X, y

        if not tagged:
            with open(path, 'r', encoding="utf8") as f:
                raw_lines = f.readlines()
            # split to sentences
            sentences = []
            curr_s = []
            for word_tag in raw_lines:
                if word_tag[0] != '\n':
                    curr_s.append(word_tag)
                else:
                    sentences.append(curr_s)
                    curr_s = []

            sentences = [["*"] * W + \
                         [s[:-1] if s[-1] == '\n' else s for s in sen] + \
                         ["~"] * W for sen in sentences]

            # concat every word vec to WINDOW_R words behind and WINDOW_R after
            # "*", "~" are 0 vecs for now. TODO: something else?
            # also, if a word not found in embedding_model, we put 0 vec. TODO: maybe there is a better idea?
            # TODO: if selected vector is not in embedding_model: for now we give it 0
            X = []
            for sentence in sentences:
                sentence_len = len(sentence)
                for i_s in range(W, sentence_len - W):
                    vecs_list = []
                    for c_word in sentence[(i_s - W): (i_s + W + 1)]:
                        if (c_word == "*") or (c_word == "~") or (c_word not in self.embedding_model.key_to_index):
                            u_c = np.zeros(NERDataset.VEC_DIM)
                        else:
                            u_c = self.embedding_model[c_word]
                        vecs_list.append(u_c)
                    concated_vec = np.concatenate(vecs_list)
                    X.append(concated_vec)
            return X

    def get_preprocessed_data(self):
        X_train, y_train = self._get_dataset(path=self.train_path, tagged=True)
        X_dev, y_dev = self._get_dataset(path=self.dev_path, tagged=True)
        X_test = self._get_dataset(path=self.test_path, tagged=False)

        # encode labels:
        self.label_encoder.fit(y_train)
        y_train = self.label_encoder.transform(y_train)
        y_dev = self.label_encoder.transform(y_dev)
        # NOTE self.label_encoder.inverse_transform(y_dev) ===> with get the labels back

        return X_train, y_train, X_dev, y_dev, X_test




class NER_DataSet(Dataset):
    VEC_DIM = 25  # TODO change to 200
    GLOVE_PATH = f"glove-twitter-{VEC_DIM}"
    WORD_2_VEC_PATH = "word2vec-google-news-300"

    def __init__(self, file_path, vector_type, tokenizer=None):
        # TODO: remove this section of open file?
        self.file_path = file_path
        data = pd.read_csv(self.file_path)
        self.sentences = data["reviewText"].tolist()
        self.labels = data["label"].tolist()

        self.tags_to_idx = {
            tag: idx for idx, tag in enumerate(sorted(list(set(self.labels))))
        }
        self.idx_to_tag = {idx: tag for tag, idx in self.tags_to_idx.items()}

        self.vector_type = vector_type
        if vector_type == "tf-idf":
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
        else:
            if vector_type == "w2v":
                model = downloader.load(self.WORD_2_VEC_PATH)
            elif vector_type == "glove":
                model = downloader.load(self.GLOVE_PATH)
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
                cur_rep = np.stack(cur_rep).mean(axis=0)  # HW TODO: change to token lev
                representation.append(cur_rep)
                labels.append(cur_labels)
            self.labels = labels
            representation = np.stack(representation)
            self.tokenized_sen = representation
            self.vector_dim = representation.shape[-1]

    def __getitem__(self, item):
        # NOTE: question - where does the function call to this?
        cur_sen = self.tokenized_sen[item]
        if self.vector_type == "tf-idf":
            cur_sen = torch.FloatTensor(cur_sen.toarray()).squeeze()
        else:
            cur_sen = torch.FloatTensor(cur_sen).squeeze()
        label = self.labels[item]
        label = self.tags_to_idx[label]
        # label = torch.Tensor(label)
        data = {"input_ids": cur_sen, "labels": label}
        return data

    def __len__(self):
        return len(self.sentences)
