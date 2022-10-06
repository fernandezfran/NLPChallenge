#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grid Search and Cross Validation of the best NLP model."""
import numpy as np

import pandas as pd

import sklearn.model_selection
import sklearn.svm

import natural_language_processing


def main():
    # Import dataset
    dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

    # Preprocess data
    corpus = natural_language_processing.clean_reviews(dataset["Review"])

    # Create the model
    cv = natural_language_processing.bag_of_words_model(corpus)

    # Adapt data for some sklearn model
    X = cv.transform(corpus).toarray()
    y = dataset["Liked"].to_numpy()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # SVM with the best accuracy
    clf = sklearn.svm.SVC(kernel="linear", random_state=0)
    clf.fit(X_train, y_train)
    old_accuracy = sklearn.metrics.accuracy_score(y_test, clf.predict(X_test))

    # Grid Search with Cross Validation
    cvalues = np.arange(0.25, 1.25, 0.25)
    grid_search = sklearn.model_selection.GridSearchCV(
        estimator=sklearn.svm.SVC(),
        param_grid=[
            {"C": cvalues, "kernel": ["linear"]},
            {"C": cvalues, "kernel": ["rbf"], "gamma": np.arange(0.1, 1, 0.1)},
        ],
        scoring="accuracy",
        cv=10,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    print("With the parameters of SVM:")
    print(best_params)
    print(f"an accuracy of {best_score:.2f} is obtained")
    print(f"the old one was: {old_accuracy:.2f}")


if __name__ == "__main__":
    main()
