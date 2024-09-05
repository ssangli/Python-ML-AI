import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_input(filename):
    df = pd.read_csv(filename)
    df1 = df.dropna(axis=0)
    np_array = df1.to_numpy()
    xs = np_array[:, 3]
    ys = np_array[:, 2]
    return xs, ys

##### Logistic Regression #####
def generate_feature_vector(tweets, freq, include_bias = True):
    """
        feature vector : [ bias, positive, negative]
    :param tweets:
    :param ys:
    :param freq:
    :return:
    """
    res = []
    for tweet in tweets:
        tmp = []
        pcnt = 0
        ncnt = 0
        for word in tweet.split(' '):
            pcnt += freq[(word, 1)] if (word, 1) in freq else 0
            ncnt += freq[(word, 0)] if (word, 0) in freq else 0
        if include_bias:
            tmp = [1, pcnt, ncnt]
        else:
            tmp = [ pcnt, ncnt ]
        res.append(tmp)
    return np.array(res)
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))
def compute_cost(y, ypred):
    m = y.shape[0]
    epsilon = 1e-5 # remove log 0 error
    cost = -(np.sum(y * np.log(ypred + epsilon) + (1-y) * np.log(1-ypred + epsilon))) / m
    return cost

def gradient(x, y, ypred):
    error = ypred - y
    return np.dot(x.T, error)/y.shape[0]

def accuracy(y, ypred):
    y = y.reshape((y.shape[0], 1))
    ypred = ypred.reshape((ypred.shape[0], 1))
    acc = np.sum(ypred == y)/y.shape[0]
    return acc

def train(x, y, num_features, learning_rate = 0.001, num_epochs = 100):
    epoch_loss = []
    accs =[]
    y = y.reshape((y.shape[0], 1))
    theta = np.zeros((num_features, 1))
    print("X shape {} Y shape {} Theta {}".format(x.shape, y.shape, theta.shape))
    for epoch in range(num_epochs):
        h = np.dot(x, theta)
        ypred = sigmoid(h)
        loss = compute_cost(y, ypred)
        acc = accuracy(y, ypred)
        accs.append(acc)
        epoch_loss.append(loss)
        dw = gradient(x, y, ypred)
        # update weights
        theta = theta - learning_rate * dw
        print("Epoch : {}, Loss : {}, Accuracy {}".format(epoch, loss, acc))
    print(theta)
    return epoch_loss, accs, theta

def run_logistic_regression(include_bias = True, learning_rate = 0.001, num_epochs = 10):
    xs, ys = read_input('./data/twitter_training.csv')
    xs = process_tweets(xs)
    xs, ys = cleanup_data(xs, ys)
    freq = build_freq(xs, ys)
    X = generate_feature_vector(xs, freq, include_bias)
    num_features = 3
    if include_bias is False:
        num_features = 2
    losses, accs, theta = train(X, ys, num_features, learning_rate, num_epochs)
    return losses, accs, theta

def predict(X, theta):
    h = np.dot(X, theta)
    ypred = sigmoid(h)
    return ypred

##### Logistic Regression #####






