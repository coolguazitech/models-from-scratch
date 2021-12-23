import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class Pocket:
    def __init__(self, n_iter=1000):
        """ pocket algorithm implementation

        Parameters
        ----------
        n_iter : int, optional (default=1000)
            The number of iterations to update weights.

        """
        self.n_iter = n_iter

    def fit(self, X_train, y_train):
        X_train, y_train = X_train.reshape([-1, X_train.shape[-1]]), y_train.reshape([-1])
        n_points, n_features = X_train.shape[-2], X_train.shape[-1]
        self.weights = np.random.normal(0, 1, (n_features,))
        self.bias = np.random.normal(0, 1, (1,))
        _w = np.concatenate([self.weights, self.bias], axis=-1)
        _X = np.hstack([X_train, np.ones([n_points, 1])])
        n_corrects = 0

        for _ in range(self.n_iter):
            logits = _X @ _w * y_train >= 0
            if np.sum(logits) == n_points:
                self.weights, self.bias = _w[:-1], _w[-1]
                return
            else:
                indices = np.where(logits == 0)[0]
                random_index = np.random.choice(indices)
                temp_w = _w + _X[random_index] * y_train[random_index]
                logits = _X @ temp_w * y_train >= 0
                if np.sum(logits) > n_corrects:
                    n_corrects = np.sum(logits)
                    _w = temp_w

        self.weights, self.bias = _w[:-1], _w[-1]

    def predict(self, X_test):
        return np.where(X_test @ self.weights + self.bias > 0, 1, -1)

    def evaluate(self, X_test, y_test):
        return np.sum(self.predict(X_test) == y_test) / y_test.size

if __name__ == '__main__':
    X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=1234)
    X_train, X_test, y_train, y_test = train_test_split(X, (y - 0.5) * 2, test_size=0.2, random_state=1234)
    mean, std = np.mean(X_train, -2), np.std(X_train, -2)
    X_train = (X_train - mean) / (std + 0.00001)
    X_test = (X_test - mean) / (std + 0.00001)
    clf = Pocket(n_iter=100000)
    clf.fit(X_train, y_train)
    acc = clf.evaluate(X_test, y_test)
    print(f"Accuracy = {acc * 100:.1f} %")