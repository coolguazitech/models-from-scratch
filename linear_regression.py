import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, lr=0.001, n_iter=1000):
        """ Linear regression implementation

        Parameters
        ----------
        n_iter : int, optional (default=1000)
            The number of iterations to perform gradient descent.

        lr : float, optional (default=0.001)
            Determines the step size at each iteration while moving toward a minimum.
            
        """
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X_train, y_train):
        X_train, y_train = X_train.reshape([-1, X_train.shape[-1]]), y_train.reshape([-1])
        n_points, n_features = X_train.shape[-2], X_train.shape[-1]
        self.weights = np.random.normal(0, 1, (n_features,))
        self.bias = np.random.normal(0, 1, (1,))
        
        for _ in range(self.n_iter):
            dw = X_train.T @ (2 * (X_train @ self.weights + self.bias - y_train)) / n_points
            db = np.sum(2 * (X_train @ self.weights + self.bias - y_train)) / n_points
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X_test):
        return X_test @ self.weights + self.bias

    def evaluate(self, X_test, y_test, mode='mae'):
        if mode == 'mae':
            beta = 1
        elif mode == 'mse':
            beta = 2
        else:
            beta = 1
        beta = 1 if mode == 'mae' else 2
        return np.sum(np.abs(self.predict(X_test) - y_test) ** beta) / y_test.size


if __name__ == '__main__':
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1234)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    mean, std = np.mean(X_train, -2), np.std(X_train, -2)
    X_train = (X_train - mean) / (std + 0.00001)
    X_test = (X_test - mean) / (std + 0.00001)
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    mae = clf.evaluate(X_test, y_test)
    print(f"MAE = {mae:.4f}")
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X_test, clf.predict(X_test), color="black", linewidth=2, label="Prediction")
    plt.show()