import torch
import torch.nn as nn

from preprocess import SentencesEmbeddingDataset
from train_predict_plot import train_and_plot

""" 
    ### Code Structure

    0. Prepare Hyper Parametrs
    1. Get Dataset:
        input: file with words and POS for each word.
        output: for each word - concat embddings of word and embedding of POS.
            * word embedding - glove + word2vec concatenated
            * POS embedding - ONE HOT / learnable
    2. Run DependencyParser Model
        forward
        predict - Run chu_liu_edmonds to get Predicted Tree
        evaluate - UAS & Loss pronts and plots

    * Calculate Negative Log Likelihood Loss
        a. Get Prob Matrix
            do softmax for each column of Scores Matrix
            (assuming each column represants modifier word and each row represants head word)
        b. loss = sum over all sentences (Xi, Yi):
            sum over all couples of (head, modifier) in Yi (true dependencies):
                - log(Prob[head][modifier]) / |Yi|
"""


class DependencyParser(nn.Module):
    def __init__(
        self,
        embedding_dim,
        lstm_hidden_dim,
        lstm_num_layers,
        fc_hidden_dim,
        tagged,
        lstm_dropout=0.25,
        activation=nn.Tanh(),
    ):

        super(DependencyParser, self).__init__()

        # ~~~~~~~~~ variables
        self.fc_hidden_dim = fc_hidden_dim
        self.embedding_dim = embedding_dim  # embedding dim: word2vec/glove + POS
        self.lstm_hidden_dim = lstm_hidden_dim
        self.tagged = tagged
        # self.root_vec = torch.rand(embedding_dim, requires_grad=False).unsqueeze(0)
        self.root_vec = torch.zeros(embedding_dim).unsqueeze(0)

        # ~~~~~~~~~ layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(self.lstm_hidden_dim * 4, self.fc_hidden_dim)
        self.activation = activation
        self.fc2 = nn.Linear(self.fc_hidden_dim, 1)
        self.mlp = nn.Sequential(self.fc1, self.activation, self.fc2)

        # ~~~~~~~~~ final funcs
        self.log_softmax = nn.LogSoftmax(dim=0)  # dim=0 for cols, dim=1 for rows
        self.loss_func = nn.NLLLoss()

    def forward(self, sentence):
        sentence_embedded, true_dependencies = sentence
        sentence_len = sentence_embedded.size(0)
        sentence_embedded = torch.concat([sentence_embedded, self.root_vec])
        lstm_output = self.prepare_lstm_output(input=sentence_embedded)  # (n+1) X 2h
        concated_pairs = self.concat_pairs(
            lstm_output, sentence_len
        )  # -> ((n+1)^2 - (n+1) - n) X 4h
        mlp_output = self.mlp(concated_pairs)  # -> ((n+1)^2 - (2n+1)) X 1

        # construct score matrix, with diag: torch.exp(torch.Tensor([float('-inf')]))
        scores_matrix = self.reshape_scores_vec_to_scores_mat(
            mlp_output, sentence_len
        )  # -> (n+1) X n

        # Calculate the negative log likelihood loss ---  only for tagged
        if self.tagged:
            loss = self.loss_func(
                input=self.log_softmax(scores_matrix).T,
                target=torch.stack([x[1] for x in true_dependencies]),
            )
        else:
            loss = None

        # add root column on index 0 with scores -inf
        col_to_add = torch.ones((sentence_len + 1, 1)) * float("-inf")
        scores_matrix = torch.concat([col_to_add, scores_matrix], 1)
        return loss, scores_matrix

    def prepare_lstm_output(self, input):
        """
        for each word concatenate hidden vectors
        """
        lstm_output, (hn, cn) = self.lstm(input=input)
        # TODO: concat outputs from different layers?
        return lstm_output

    def concat_pairs(self, mat, sentence_len):
        """
        input of size: (n+1)Xh;
        sentence_len = n

        Prepare Matrix for MLP (multi layer perceptron):
            shape: ((n+ 1) * n - n, 2 * word_dim)
            each row is concatination of couple of words v_i v_j
        """
        concated_vecs_list = []
        for i in range(sentence_len + 1):  # including the root vec
            for j in range(sentence_len):  # not including the root vec
                if i == j:
                    continue
                curr_vec = torch.concat([mat[i], mat[j]])
                concated_vecs_list.append(curr_vec)
        concated_vecs = torch.stack(concated_vecs_list)
        return concated_vecs

    def reshape_scores_vec_to_scores_mat(self, vec_to_reshape, sentence_len):
        """
        Constract Scores Matrix
            shape: (n + 1, n)
            matrix[i][j] = score of (v_i, v_j)
        """
        output_mat = torch.zeros(sentence_len + 1, sentence_len)
        running_index = 0
        for i in range(sentence_len + 1):
            for j in range(sentence_len):
                if i == j:
                    output_mat[i, j] = float("-inf")
                else:
                    output_mat[i, j] = vec_to_reshape[running_index]
                    running_index += 1

        # move root row to be the first row in mat
        output_mat = torch.concat([output_mat[-1].unsqueeze(0), output_mat[:-1]])
        return output_mat


def main():
    # Hyper parameters
    words_embedding_list = [("glove-wiki-gigaword-200", None, None, 200)]
    # embed_list = [
    #     (
    #         "concated",
    #         ["glove-twitter-200", "word2vec-google-news-300"],
    #         [200, 300],
    #         500,
    #     )
    # ]
    pos_embedding_name_list = ["onehot", "learn"]
    pos_embedding_dim = 25

    lstm_hidden_dim_list = [250, 300]
    lstm_num_layers_list = [1, 2, 3, 4, 5]
    lstm_dropout_list = [0, 0.25, 0.1, 0.3]

    optimizer = "ADAM"
    activation = nn.Tanh()
    num_epochs = 10
    torch.manual_seed(42)

    load_dataset_from_pkl = False

    for (
        word_embedding_name,
        list_embedding_paths,
        word_embedding_dim_list,
        word_embedding_dim,
    ) in words_embedding_list:
        for pos_embedding_name in pos_embedding_name_list:
            # get embeddings
            if load_dataset_from_pkl:
                train_dataset, val_dataset, _ = torch.load(
                    f"{word_embedding_name}_{pos_embedding_name}.pkl"
                )
            else:
                Dataset = SentencesEmbeddingDataset(
                    word_embedding_name=word_embedding_name,
                    list_embedding_paths=list_embedding_paths,
                    word_embedding_dim_list=word_embedding_dim_list,
                    word_embedding_dim=word_embedding_dim,
                    pos_embedding_name=pos_embedding_name,
                    pos_embedding_dim=pos_embedding_dim,
                )
                train_dataset, val_dataset, _ = Dataset.get_datasets()

            # run
            for lstm_dropout in lstm_dropout_list:
                for lstm_hidden_dim in lstm_hidden_dim_list:
                    for lstm_num_layers in lstm_num_layers_list:
                        print(
                            "----------------------------------------------------------"
                        )

                        hyper_params_title = (
                            f"{word_embedding_name}"
                        )
                        hyper_params_title += f" | pos={pos_embedding_name}"
                        hyper_params_title += f" | hidden={lstm_hidden_dim}"
                        hyper_params_title += f" \nnum_layers={lstm_num_layers}"
                        hyper_params_title += f" | dropout={lstm_dropout}"
                        hyper_params_title += f" | opt={optimizer}"
                        model_name = "mini_train3 | "
                        model_name += f"word_embedding_name={word_embedding_name}"
                        model_name += f" | pos={pos_embedding_name}"
                        model_name += f" | hidden={lstm_hidden_dim}"
                        model_name += f" | num_layers={lstm_num_layers}"
                        model_name += f" | dropout={lstm_dropout}"
                        model_name += f" | opt={optimizer}"
                        print(hyper_params_title)
                        model_save_path = f"{model_name}.pt"

                        dependency_model = DependencyParser(
                            embedding_dim=word_embedding_dim
                            + Dataset.pos_embedding_dim,
                            lstm_hidden_dim=lstm_hidden_dim,
                            lstm_num_layers=lstm_num_layers,
                            fc_hidden_dim=int(lstm_hidden_dim / 2),
                            lstm_dropout=lstm_dropout,
                            activation=activation,
                            tagged=True,
                        )
                        if optimizer == "ADAM":
                            optimizer = torch.optim.Adam(
                                params=dependency_model.parameters()
                            )
                        elif optimizer == "SGD":
                            optimizer = torch.optim.SGD(
                                dependency_model.parameters(), lr=0.1
                            )
                        train_and_plot(
                            dependency_model=dependency_model,
                            model_save_path=model_save_path,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            num_epochs=num_epochs,
                            optimizer=optimizer,
                            hyper_params_title=hyper_params_title,
                        )


if __name__ == "__main__":
    main()
