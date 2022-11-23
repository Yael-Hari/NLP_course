import numpy as np
import gensim
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score, f1_score


class Model1:
    def __init__(self, model="KNN", n_neighbors=5):
        self.model_type = model
        if self.model_type == "KNN":
            self.model = KNN(n_neighbors=n_neighbors)

    def fit(self, X, y):
        self.model = KNN(n_neighbors=5)
        self.model.fit(X, y)

    def predict(self, x_test):
        preds = self.model.predict(x_test)
        return preds


    # represent each word iwth word2vec / glove vector

    # use sklearn model to predict
