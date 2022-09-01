#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Natural Language Processing."""
import re

import matplotlib.pyplot as plt

import nltk

import numpy as np

import pandas as pd

import sklearn.ensemble
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.tree
import sklearn.svm


def clean_reviews(reviews):
    """Clean the text of the reviews."""
    nltk.download("stopwords")

    corpus = []
    for review in reviews:
        review = re.sub("[^a-zA-Z]", " ", review).lower().split()

        ps = nltk.stem.porter.PorterStemmer()
        all_stopwords = nltk.corpus.stopwords.words("english")
        all_stopwords.remove("not")
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = " ".join(review)

        corpus.append(review)

    return corpus


def bag_of_words_model(corpus):
    """Define the sparse matrix."""
    cv = sklearn.feature_extraction.text.CountVectorizer(max_features=1500)
    cv.fit(corpus)
    return cv


def try_models(X_train, X_test, y_train, y_test, models):
    """Try different classification models."""
    for classifier in models:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
        accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
        print(cm, accuracy)


def main():
    # Import dataset
    dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

    # Preprocess data
    corpus = clean_reviews(dataset["Review"])

    # Create the model
    cv = bag_of_words_model(corpus)

    # Adapt data for some sklearn model
    X = cv.transform(corpus).toarray()
    y = dataset["Liked"].to_numpy()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Different models
    classifiers = (
        sklearn.linear_model.LogisticRegression(random_state=0),
        sklearn.neighbors.KNeighborsClassifier(),
        sklearn.svm.SVC(kernel="linear", random_state=0),
        sklearn.svm.SVC(kernel="rbf", random_state=0),
        sklearn.naive_bayes.GaussianNB(),
        sklearn.tree.DecisionTreeClassifier(criterion="entropy", random_state=0),
        sklearn.ensemble.RandomForestClassifier(criterion="entropy", random_state=0),
    )

    try_models(X_train, X_test, y_train, y_test, classifiers)


if __name__ == "__main__":
    main()
