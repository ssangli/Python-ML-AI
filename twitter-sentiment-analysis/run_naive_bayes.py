import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ProcessTweets as pt
import NaiveBayes as nb

import nltk
from nltk.corpus import twitter_samples
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')



# using sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

vectorizer = CountVectorizer()

def read_input(filename):
    df = pd.read_csv(filename)
    df1 = df.dropna(axis=0)
    np_array = df1.to_numpy()
    xs = np_array[:, 3]
    ys = np_array[:, 2]
    process_tweets = pt.ProcessTweets(xs, ys)
    xs_mod, ys = process_tweets.get_processed_tweets()
    xs_mod_arr = np.array(xs_mod).reshape(-1, 1)
    ys_arr = np.array(ys)
    return xs_mod_arr, ys_arr

# Training
xs, ys = read_input('./data/twitter_training.csv')

def run_custom_model(xs, ys):
    ## Custom model ##
    naive_bayes = nb.NaiveBayes()
    naive_bayes.fit(xs, ys)
    accuracy = naive_bayes.accuracy(xs, ys)
    print("Training accuracy custom {}".format(accuracy))

def run_sklearn_nb_bow(xs, ys):
    # Sklearn
    xs_list = []
    ys_list = []
    xs_list = xs.reshape((xs.shape[0],)).tolist()
    ys_list = ys.reshape((ys.shape[0],)).tolist()
    X = vectorizer.fit_transform(xs_list)
    gnb = GaussianNB().fit(X.toarray(),ys_list)
    ypred = gnb.predict(X.toarray())
    acc = accuracy_score(ypred, ys)
    print("Sklearn Gaussian NB {}".format(acc))

def run_validation():
    # Validation
    xtest, ytest = read_input('./data/twitter_validation.csv')
    naive_bayes_val = nb.NaiveBayes()
    naive_bayes_val.fit(xtest_mod, ytest_mod)
    val_acc = naive_bayes_val.accuracy(xtest_mod, ytest_mod)
    print("Validation Accuracy {}".format(val_acc))

