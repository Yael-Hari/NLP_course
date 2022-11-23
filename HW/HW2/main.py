import numpy as np
import gensim
from model1_simple import Model1
import torch
from preprocessing import NERDataset


def main():
    ############
    #  PARAMS  #
    ############
    model_type = "KNN"
    embedding_type = "glove"

    model1 = Model1(model_type, n_neighbors=5)
    ner_dataset = NERDataset(embedding_model_type=embedding_type)
    X_train, y_train, X_dev, y_dev, X_test = ner_dataset.get_preprocessed_data()
    model1.fit(X_train, y_train)
    y_dev_pred = model1.predict(X_dev)
    f1_score = model1.f1_score(y_dev, y_dev_pred)
    print(f"=== {model_type=} === {embedding_type} ===")
    print(f"=== {f1_score} ===")


if __name__ == '__main__':
    main()
