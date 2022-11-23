import numpy as np
import gensim
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


class Model1:
    def __init__(self, model="KNN", n_neighbors=5, kernel="linear"):
        self.model_type = model
        if self.model_type == "KNN":
            self.model = KNN(n_neighbors=n_neighbors)
        elif self.model_type == "SVM":
            self.model = SVC(kernel=kernel)

    def fit(self, X, y):
        self.model = KNN(n_neighbors=5)
        self.model.fit(X, y)

    def predict(self, x_test):
        preds = self.model.predict(x_test)
        return preds

    def f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred)


    # represent each word iwth word2vec / glove vector

    # use sklearn model to predict
