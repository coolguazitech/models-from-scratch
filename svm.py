import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, n_iter=1000, lr=0.001, c=1.0):
        """ Naive hard margin support vector machine implementation

        Parameters
        ----------
        n_iter : int, optional (default=1000)
            The number of iterations to perform gradient descent.

        lr : float, optional (default=0.001)
            Determines the step size at each iteration while moving toward a minimum.

        c : float, optional (default=1.0)
            Bigger c implies large VC dimension.

        """
        self.n_iter = n_iter
        self.lr = lr
        self.c = c

    def fit(self, X_train, y_train):
        X_train, y_train = X_train.reshape([-1, X_train.shape[-1]]), y_train.reshape([-1])
        n_features = X_train.shape[-1]
        self.weights = np.random.normal(0, 1, (n_features,))
        self.bias = np.random.normal(0, 1, (1,))

        for _ in range(self.n_iter):
            conditional_indices = 1 - y_train * (X_train @ self.weights + self.bias) >= 0
            dw = (1 / self.c) * self.weights - (y_train * conditional_indices) @ X_train
            db = - np.sum(y_train * conditional_indices)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X_test):
        return np.where(X_test @ self.weights + self.bias > 0, 1, -1)

    def evaluate(self, X_test, y_test):
        return np.sum(self.predict(X_test) == y_test) / y_test.size

if __name__ == '__main__':
    # test
    X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.25, random_state=124)
    X_train, X_test, y_train, y_test = train_test_split(X, (y - 0.5) * 2, test_size=0.2, random_state=1234)
    mean, std = np.mean(X_train, -2), np.std(X_train, -2)
    X_train_std = (X_train - mean) / (std + 0.00001)
    X_test_std = (X_test - mean) / (std + 0.00001)
    clf = SVM(n_iter=1000, lr=0.001, c=1)
    clf.fit(X_train_std, y_train)
    acc = clf.evaluate(X_test_std, y_test)
    print(f"Accuracy = {acc * 100:.1f} %")

    # visualization
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(X_train_std[:, 0], X_train_std[:, 1], c=(y_train * 50 + 50).astype(int), s=15)
    x = np.linspace(-2, 2, 50)
    y = (-clf.weights[0] * x - clf.bias) / clf.weights[1]
    plt.plot(x, y, color="black", linewidth=2, label="Prediction")
    plt.show()