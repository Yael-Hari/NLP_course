import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor



class Model1:
    def __init__(self, model="KNN", n_neighbors=5, metric='cosine', kernel="linear"):
        self.model_type = model
        if self.model_type == "KNN":
            self.model = KNN(n_neighbors=n_neighbors, metric=metric)
        elif self.model_type == "SVM":
            self.model = SVC(kernel=kernel)
        elif self.model_type == "RF":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, x_test):
        preds = self.model.predict(x_test)
        return preds

    def f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred)


    # represent each word iwth word2vec / glove vector

    # use sklearn model to predict
