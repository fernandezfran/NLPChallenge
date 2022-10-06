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

import xgboost


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


def train_models(X_train, X_test, y_train, y_test, models, models_names, verbose=1):
    """Try different classification models."""
    if verbose:
        print("model,accuracy,precision,recall,f1score")

    accuracy = []
    for classifier, name in zip(models, models_names):
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        acc = sklearn.metrics.accuracy_score(y_test, y_pred)

        prec, rec, fscor, sup = sklearn.metrics.precision_recall_fscore_support(
            y_test, y_pred, average="macro"
        )

        if verbose:
            print(f"{name},{acc},{prec},{rec},{fscor}")

        accuracy.append(acc)

    return accuracy


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
    classifiers_names = (
        "Logistic Regression",
        "KNN",
        "SVM linear",
        "SVM rbf",
        "Naive Bayes",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
    )
    classifiers = (
        sklearn.linear_model.LogisticRegression(random_state=0),
        sklearn.neighbors.KNeighborsClassifier(),
        sklearn.svm.SVC(kernel="linear", random_state=0),
        sklearn.svm.SVC(kernel="rbf", random_state=0),
        sklearn.naive_bayes.GaussianNB(),
        sklearn.tree.DecisionTreeClassifier(criterion="entropy", random_state=0),
        sklearn.ensemble.RandomForestClassifier(criterion="entropy", random_state=0),
        xgboost.XGBClassifier(),
    )
    accuracy = train_models(
        X_train, X_test, y_train, y_test, classifiers, classifiers_names
    )

    # Plot accuracy
    fig, ax = plt.subplots()

    x = np.arange(0, len(classifiers))

    ax.bar(x, accuracy)

    ax.set_xticks(x, classifiers_names, rotation=45)
    ax.set_ylabel("Accuracy")

    fig.tight_layout()
    fig.savefig("models_accuracy.png", dpi=600)


if __name__ == "__main__":
    main()
