# import numpy as np
# import gensim
from model1_simple import Model1
# import torch
from preprocessing import NERDataset


def main():
    ##############
    ##  PARAMS  ##
    ##############
    model_type = "KNN"
    embedding_type = "glove"

    model1 = Model1(model_type, n_neighbors=5)
    ner_dataset = NERDataset(embedding_model_type=embedding_type)
    train_loader, dev_loader, test_loader = ner_dataset.get_preprocessed_data()
    X_train = train_loader.dataset.tensors[0]
    y_train = train_loader.dataset.tensors[1]
    X_dev = dev_loader.dataset.tensors[0]
    y_dev = dev_loader.dataset.tensors[1]
    X_test = test_loader.dataset.tensors[0]

    model1.fit(X_train.detach().numpy(), y_train.detach().numpy())
    y_dev_pred = model1.predict(X_dev.detach().numpy())
    f1_score = model1.f1_score(y_dev.detach().numpy(), y_dev_pred)
    print(f"=== {model_type=} === {embedding_type} ===")
    print(f"=== {f1_score} ===")


if __name__ == '__main__':
    main()
