import torch
import torch.nn as nn
import torch.nn.functional as F

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
"""


class DependencyParser(nn.Module):
    def __init__(self, *args):
        super(DependencyParser, self).__init__()
        self.word_embedding = # Implement embedding layer for words (can be new or pretrained - word2vec/glove)
        self.hidden_dim = self.word_embedding.embedding_dim
        
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

    def evaluate(self):
        #TODO: complete
        pass


def main():
    


def run(
    train_loader,
    dev_loader,
    embedding_name,
    vec_dim,
    hidden_dim,
    dropout,
    class_weights,
    loss_func,
    loss_func_name,
    batch_size,
    O_str,
    num_layers,
):
    num_classes = 2
    num_epochs = 10
    lr = 0.001
    activation = nn.Tanh()
    embedding_dim = vec_dim
    w_list = [round(float(w), 2) for w in class_weights]

    model_save_path = f"LSTM_model_stateDict_hidden={hidden_dim}_\
        layers={num_layers}_w={w_list}_{O_str}.pt"

    LSTM_model = LSTM_NER_NN(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        model_save_path=model_save_path,
        activation=activation,
        num_layers=num_layers,
        dropout=dropout,
    )

    optimizer = Adam(params=LSTM_model.parameters(), lr=lr)

    epoch_dict = train_and_plot_LSTM(
        LSTM_model=LSTM_model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        val_loader=dev_loader,
        optimizer=optimizer,
        loss_func=loss_func,
    )

   plot_epochs_results(
       epoch_dict=epoch_dict,
       hidden=hidden_dim,
       embedding_name=embedding_name,
       dropout=dropout,
       loss_func_name=loss_func_name,
       class_weights=list(class_weights),
       num_layers=num_layers,
   )


def main():
    # embed_list = [
    #     (
    #         "concated",
    #         ["glove-twitter-200", "word2vec-google-news-300"],
    #         [200, 300],
    #         500,
    #     )
    # ]
    # glove-wiki-gigaword-100

    hidden_list = [256, 600, 1000]
    hidden_list = [500]
    dropout_list = [0.2]
    w_list = [
        torch.tensor([0.1, 0.9]),
        # torch.tensor([0.2, 0.8]),
    ]
    batch_size = 32
    # num_layers_list = [1, 2, 3, 4]
    num_layers_list = [2]

    for embedding_name, embedding_paths, vec_dims_list, vec_dim in embed_list:
        torch.manual_seed(42)
        # option 1: make
        Dataset = SentencesEmbeddingDataset(
            vec_dim=vec_dim,
            list_embedding_paths=embedding_paths,
            list_vec_dims=vec_dims_list,
            embedding_model_path=embedding_name,
        )
        train_loader, dev_loader, _ = Dataset.get_data_loaders(
            batch_size=batch_size
        )
        # option 2: load
        # train_loader, dev_loader, _ = torch.load(
        #     f"{embedding_name}.pkl"
        # )

        # run
        for hidden_dim in hidden_list:
            for num_layers in num_layers_list:
                for dropout in dropout_list:
                    for class_weights in w_list:
                        for loss_func, loss_func_name in [
                            (
                                nn.CrossEntropyLoss(weight=class_weights),
                                "CrossEntropy",
                            ),
                        ]:
                            print(
                                "----------------------------------------------------------"
                            )
                            print(
                                f"{embedding_name=} | {hidden_dim=} | {dropout=} \
                                    \n{class_weights=} | {loss_func=}"
                            )
                            run(
                                train_loader=train_loader,
                                dev_loader=dev_loader,
                                embedding_name=embedding_name,
                                vec_dim=vec_dim,
                                hidden_dim=hidden_dim,
                                dropout=dropout,
                                class_weights=class_weights,
                                loss_func=loss_func,
                                loss_func_name=loss_func_name,
                                batch_size=batch_size,
                                num_layers=num_layers,
                            )




if __name__ == "__main__":
    main()
