import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X_train, y_train):
        X_train, y_train = X_train.reshape([-1, X_train.shape[-1]]), y_train.reshape([-1])
        n_points, n_features = X_train.shape[-2], X_train.shape[-1]
        self.weights = np.random.normal(0, 1, (n_features,))
        self.bias = np.random.normal(0, 1, (1,))
        
        for _ in range(self.n_iter):
            dw = X_train.T @ (self._sigmoid(X_train @ self.weights + self.bias) - y_train) / n_points
            db = np.sum(self._sigmoid(X_train @ self.weights + self.bias) - y_train) / n_points
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X_test):
        return (X_test @ self.weights + self.bias > 0).astype(int)

    def evaluate(self, X_test, y_test):
        return np.sum(self.predict(X_test) == y_test) / y_test.size

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


if __name__ == '__main__':
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    mean, std = np.mean(X_train, -2), np.std(X_train, -2)
    X_train = (X_train - mean) / (std + 0.00001)
    X_test = (X_test - mean) / (std + 0.00001)
    clf = LogisticRegression(lr=1, n_iter=1000)
    clf.fit(X_train, y_train)
    acc = clf.evaluate(X_test, y_test)
    print(f"Accuracy = {acc * 100:.1f} %")
