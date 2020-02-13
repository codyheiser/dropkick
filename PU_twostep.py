# -*- coding: utf-8 -*-
"""
Two-step classifier for PU learning.

@author: C Heiser
Adapted from https://github.com/roywright/pu_learning, 
inspired by Kaboutari, et al., Int. Journal of Computer Applications Technology and Research (2014)
"""
import numpy as np


def twoStep(clf, X, y, n_iter=10, thresh="min"):
    """
    two-step positive-unlabeled (PU) classifier technique

    Parameters:
        clf (sklearn): sklearn classifier object
            e.g. clf=sklearn.ensemble.RandomForestClassifier(n_estimators=1000, n_jobs=-1)
        X (np.array): data in (obs, features) format
        y (np.array): labels in (obs, 1) format.
            should contain (-1, 0, 1) for unlabeled, reliable negative, reliable positive, respectively
        n_iter (int): maximum number of iterations to run
        thresh (str): how to determine which labels to switch.
            'min' = relabel all cells above minimum score for each label
            'mean' = relabel all cells within 1SD of mean score for each label

    Returns:
        pred (np.array): predict_proba scores from final trained clf
        y (np.array): final labels permuted from input array y
    """
    # train a classifier on the 0s and 1s from labeled datapoints
    clf.fit(X=X[y >= 0], y=y[y >= 0])

    # get scores for all points
    pred = clf.predict_proba(X)[:, 1]

    if thresh == "min":
        # find min/max scores for each label (positive and negative)
        label_thresh = [np.max(pred[y == 0]), np.min(pred[y == 1])]
    elif thresh == "mean":
        # find mean scores +/- one stdev for each label (positive and negative)
        label_thresh = [
            np.mean(pred[y == 0]) + np.std(pred[y == 0]),
            np.mean(pred[y == 1]) - np.std(pred[y == 1]),
        ]

    # STEP 1
    # if any unlabeled point has a score above the mean of positives or below the mean of negatives, label it accordingly
    iP_new = y[(y < 0) & (pred >= label_thresh[1])].index
    iN_new = y[(y < 0) & (pred <= label_thresh[0])].index
    y.loc[iP_new] = 1
    y.loc[iN_new] = 0

    # limit to n_iter iterations
    for i in range(n_iter):
        # if STEP 1 didn't find new labels, we're done
        if len(iP_new) + len(iN_new) == 0 and i > 0:
            break

        print(
            "\tStep 1 labeled {} new positives and {} new negatives. {} unlabeled remaining.".format(
                len(iP_new), len(iN_new), len(y[y == -1])
            )
        )
        print("\tIteration {}: Doing step 2...".format(i + 1), end="")

        # STEP 2
        # retrain on new labels and get new scores
        clf.fit(X[y >= 0], y[y >= 0])
        pred = clf.predict_proba(X)[:, -1]

        if thresh == "min":
            # find min/max scores for each label (positive and negative)
            label_thresh = [np.max(pred[y == 0]), np.min(pred[y == 1])]
        elif thresh == "mean":
            # find mean scores +/- one stdev for each label (positive and negative)
            label_thresh = [
                np.mean(pred[y == 0]) + np.std(pred[y == 0]),
                np.mean(pred[y == 1]) - np.std(pred[y == 1]),
            ]

        # repeat STEP 1
        iP_new = y[(y < 0) & (pred >= label_thresh[1])].index
        iN_new = y[(y < 0) & (pred <= label_thresh[0])].index
        y.loc[iP_new] = 1
        y.loc[iN_new] = 0

    # get scores assigned
    return (pred, y)
