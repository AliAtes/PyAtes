import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    alpha = 0.0001
    iteration = 1000
    d = X.shape[0]  # d: number of the features
    n = X.shape[1]  # n: number of the observations
    w = np.random.rand(1, d)  # random initial d values
    b = np.random.rand()  # random initial b value

    def __int__(self):
        i = 1

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy_loss(self, a, y):
        return -(y * np.log(a) + (1 - y) * (np.log(1 - a)))

    def fit(self):
        self.Js = []  # store cost value for each 100th iteration 

        for i in range(self.iteration):
            # forward propagation
            z = np.dot(self.w, self.X) + self.b  # Model z = wX + b
            a = self.sigmoid(z)
            # backward propagation
            dw = (1 / self.n) * np.dot((a - self.y), self.X.T)
            db = (1 / self.n) * np.sum(a - self.y)
            # gradient descen
            self.w = self.w - self.alpha * dw
            self.b = self.b - self.alpha * db

            if i % 100:
                cost = self.cross_entropy_loss(a, self.y)
                self.Js.append(np.sum(cost))

    def predict(self, X_test):
        z = np.dot(self.w, X_test) + self.b
        a = self.sigmoid(z)
        return 1 * (a > 0.5)
