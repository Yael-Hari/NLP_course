import torch
import torch.nn as nn
import torch.nn.functional as F
from train_and_plot import train_and_plot

""" 
    ### Code Structure

    0. Prepare Hyper Parametrs
    1. Get Dataset:
        input: file with words and POS for each word.
        output: for each word - concat embddings of word and embedding of POS.
            * word embedding - glove + word2vec concatenated
            * POS embedding - ONE HOT / learnable
    2. Train and Plot:
        run - DependencyParser Model
        predict - use chu_liu_edmonds algorithm to get predicted defendencies
        evaluate - plot UAS & Loss by epoch
"""


class DependencyParser(nn.Module):
    def __init__(self, embedding_dim, lstm_hidden_dim, lstm_num_layers, lstm_dropout, activation):
        super(DependencyParser, self).__init__()
        self.embedding_dim = embedding_dim # Implement embedding layer for words (can be new or pretrained - word2vec/glove)
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.activation = activation
        
        self.lstm = # Implement BiLSTM module which is fed with word embeddings and outputs hidden representations
        self.mlp = # Implement a sub-module to calculate the scores for all possible edges in sentence dependency graph

    def forward(self, sentence):
        # input: Xi - sentence, Yi - true dependencies of this sentence
        # output: Scores Matrix
        word_idx_tensor, pos_idx_tensor, true_tree_heads = sentence

        # Pass word_idx through their embedding layer

        # Get Bi-LSTM hidden representation for each word in sentence
        # input: Xi - sentence, Yi - true dependencies of this sentence
            # output: hidden vectors for each word from each layer - 2 directions
        self.prepare_word_vectors()
        self.get_scores(scores_matrix)

        # Get score for each possible edge in the parsing graph, construct score matrix     
        
        # Calculate the negative log likelihood loss described above
      
        return loss, score_mat


    def prepare_word_vectors(self):
        """
        for each word concatenate hidden vectors
        """
        # TODO: complete
        pass

    def get_scores(self):
        """
        a. Prepare Matrix for MLP (multi layer perceptron):
            shape: (n**2 - n, 2 * word_dim)
            each row is concatination of couple of words v_i v_j
            * prepare dict {idx: (v_i, v_j)}
        b. MLP: Run matrix in FC with 2 layers and tanh activation between them
        c. Constract Scores Matrix
            shape: (n, n)
            matrix[i][j] = score of (v_i, v_j)
        """
        # TODO: complete
        pass

    def loss_function(self, scores_matrix):
        """
        1. Calculate Negative Log Likelihood Loss
            a. Get Prob Matrix
                do softmax for each column of Scores Matrix 
                (assuming each column represants modifier word and each row represants head word)
            b. loss = sum over all sentences (Xi, Yi):
                sum over all couples of (head, modifier) in Yi (true dependencies):
                    - log(Prob[head][modifier]) / |Yi|
        """
        # TODO: complete
        pass

    def predict(self):
        # TODO: complete
        pass


def main():
    ## Hyper parameters
    
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
    batch_size = 1

    lstm_hidden_dim_list = [250, 300]
    lstm_num_layers_list = [1, 2, 3]
    lstm_dropout_list = [0.25, 0.1, 0.3]
    
    activation = nn.Tanh()
    optimizer = torch.optim.SGD(lr=0.1)
    num_epochs = 10
    torch.manual_seed(42)

    load_dataset_from_pkl = False
    
    for word_embedding_name, list_embedding_paths, word_embedding_dim_list, word_embedding_dim in words_embedding_list:
        for pos_embedding_name in pos_embedding_name_list:
            # get embeddings
            if load_dataset_from_pkl:
                train_loader, val_loader, _ = torch.load(
                    f"{word_embedding_name}_{pos_embedding_name}.pkl"
                )
            else:
                Dataset = SentencesEmbeddingDataset(
                    embedding_model_path=word_embedding_name,
                    list_embedding_paths=list_embedding_paths,
                    word_embedding_dim_list=word_embedding_dim_list,
                    word_embedding_dim=word_embedding_dim,
                    pos_embedding_name=pos_embedding_name,
                    pos_embedding_dim=pos_embedding_dim
                )
                train_loader, test_loader, _ = Dataset.get_data_loaders(
                    batch_size=batch_size
                )
            # run
            for lstm_hidden_dim in lstm_hidden_dim_list:
                for lstm_num_layers in lstm_num_layers_list:
                    for lstm_dropout in lstm_dropout_list:
                        print(
                            "----------------------------------------------------------"
                        )
                        
                        hyper_params_title = f"{word_embedding_name=} | {pos_embedding_name=} | hidden={lstm_hidden_dim} \
                                \nnum_layers={lstm_num_layers} | dropout={lstm_dropout}"
                        print(hyper_params_title)
                        model_save_path=f"{hyper_params_title}.pt"

                        dependency_model = DependencyParser(
                            embedding_dim = word_embedding_dim+pos_embedding_dim,
                            lstm_hidden_dim=lstm_hidden_dim,
                            lstm_num_layers=lstm_num_layers,
                            lstm_dropout=lstm_dropout,
                            activation=activation,
                        )
                        train_and_plot(
                            dependency_model=dependency_model,
                            model_save_path=model_save_path,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            num_epochs=num_epochs,
                            optimizer=optimizer,
                            hyper_params_title=hyper_params_title
                        )


if __name__ == "__main__":
    main()
