import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from preprocessing import EmbeddingDataset
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


if __name__ == '__main__':
    dataset = EmbeddingDataset()
    X_train, y_train, X_dev, y_dev = dataset.get_preprocessed_data()

    model_type = "KNN"
    model1 = Model1(model_type, n_neighbors=7)
    model1.fit(X_train, y_train)
    y_dev_pred = model1.predict(X_dev)
    f1_score = model1.f1_score(y_dev, y_dev_pred)
    print(f"=== {model_type=} === ")
    print(f"=== {f1_score=} ===")
