# import numpy as np
# import gensim
from model1_simple import Model1
# import torch
from preprocessing import EmbeddingDataset


def main():
    ds = EmbeddingDataset()
    train_dataloader, dev_dataloader = ds.get_dataloaders(batch_size=64, shuffle=True)
    # ##############
    # ##  PARAMS  ##
    # ##############
    # model_type = "KNN"
    # embedding_type = "glove"
    #
    # model1 = Model1(model_type, n_neighbors=5)
    # ner_dataset = EmbeddingDataset(embedding_model_type=embedding_type)
    # train_loader, dev_loader, test_loader = ner_dataset.get_datasets()
    # X_train = train_loader.dataset.tensors[0]
    # y_train = train_loader.dataset.tensors[1]
    # X_dev = dev_loader.dataset.tensors[0]
    # y_dev = dev_loader.dataset.tensors[1]
    # X_test = test_loader.dataset.tensors[0]
    #
    # model_type = "KNN"
    # model1 = Model1(model_type)
    # model1.fit(X_train.detach().numpy(), y_train.detach().numpy())
    # y_dev_pred = model1.predict(X_dev.detach().numpy())
    # f1_score = model1.f1_score(y_dev.detach().numpy(), y_dev_pred)
    # print(f"=== {model_type=} === {embedding_type=} ===")
    # print(f"=== {f1_score=} ===")
    #
    # model_type = "SVM"
    # model1 = Model1(model_type, kernel='rbf')
    # model1.fit(X_train.detach().numpy(), y_train.detach().numpy())
    # y_dev_pred = model1.predict(X_dev.detach().numpy())
    # f1_score = model1.f1_score(y_dev.detach().numpy(), y_dev_pred)
    # print(f"=== {model_type=} === {embedding_type} ===")
    # print(f"=== {f1_score} ===")
    #
    # model_type = "RF"
    # model1 = Model1(model_type)
    # model1.fit(X_train.detach().numpy(), y_train.detach().numpy())
    # y_dev_pred = model1.predict(X_dev.detach().numpy())
    # f1_score = model1.f1_score(y_dev.detach().numpy(), y_dev_pred)
    # print(f"=== {model_type=} === {embedding_type} ===")
    # print(f"=== {f1_score} ===")


if __name__ == '__main__':
    main()
