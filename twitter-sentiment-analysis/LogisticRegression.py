import os
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScalar

class LogisticRegression:
    def __init__(self):
        self.num_features = num_features
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, data):
        return 1 / (1 + np.exp(-data))

    def cost(self, y, ypred):
        """
        cost = -np.sum(y * log(ypred) + (1-y) * log(1-ypred))/num_samples
        :param y:
        :param ypred:
        :return:
        """
        num_samples = y.shape[0]
        cost = -np.sum(y * log(ypred) + (1-y) * log(1-ypred))/num_samples
        return cost

    def fitGD(self, X, y, learning_rate=0.001, num_epochs=100):
        """
        Logistic regression using Gradient Descent GD
        :param X:
        :param y:
        :param learning_rate: 0.001
        :param num_epochs: 100
        :return: None
        """
        optimizer = Optimizer("GradientDescent")
        self.weights, self.bias = optimizer.run(X,y)

    def predict(self, X, y):
        h = np.dot(X, self.weights) + self.bias
        ypred = self.sigmoid(h)
        return np.round(ypred).astype(int)

    def accuracy(self, y, ypred):
        acc = np.sum(ypred == y)/y.shape[0]
        return acc

    def plot_losses(self):
        fig, ax = plt.subplot(figsize = (8,8))
        ax.plot(losses)
