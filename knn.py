import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X.reshape([-1, X.shape[-1]])
        self.y_train = y.reshape([-1])

    def predict(self, X):
        dim = X.ndim
        shape = X.shape
        logits = np.expand_dims(X, -2) - np.expand_dims(self.X_train, tuple([-i for i in range(3, dim + 2)]))
        distances = np.sum(logits ** 2, -1) ** 0.5
        knn_indices = np.argsort(distances)[..., :self.k].reshape([-1, self.k]).tolist()
        for i in range(len(knn_indices)):
            knn_indices[i] = self.y_train[np.argmax(np.bincount(knn_indices[i]))]
        return np.array(knn_indices).reshape(shape[:-1])

    def evaluate(self, X_test, y_test):
        return np.sum(self.predict(X_test) == y_test) / y_test.size

   
if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    mean, std = np.mean(X_train, -2), np.std(X_train, -2)
    X_train = (X_train - mean) / (std + 0.0001)
    X_test = (X_test - mean) / (std + 0.0001)
    clf = KNN(1)
    clf.fit(X_train, y_train)
    acc = clf.evaluate(X_test, y_test)
    print(f"Accuracy = {acc * 100:.1f} %")







