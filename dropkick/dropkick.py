# -*- coding: utf-8 -*-
"""
Automated QC classifier pipeline

@author: C Heiser

usage: dropkick.py [-h] -c COUNTS [--obs-cols OBS_COLS [OBS_COLS ...]]
                   [--directions DIRECTIONS [DIRECTIONS ...]]
                   [--thresh-method THRESH_METHOD] [--mito-names MITO_NAMES]
                   [--n-hvgs N_HVGS] [--seed SEED] [--output-dir [OUTPUT_DIR]]
                   [--alphas [ALPHAS [ALPHAS ...]]] [--n-lambda N_LAMBDA]
                   [--cut-point CUT_POINT] [--n-splits N_SPLITS]
                   [--n-iter N_ITER] [--n-jobs N_JOBS] [--pos-frac POS_FRAC]
                   [--neg-frac NEG_FRAC]
                   {regression,twostep}

positional arguments:
  {regression,twostep}

optional arguments:
  -h, --help            show this help message and exit
  -c COUNTS, --counts COUNTS
                        [all] Input (cell x gene) counts matrix as .h5ad or
                        tab delimited text file
  --obs-cols OBS_COLS [OBS_COLS ...]
                        [all] Heuristics for thresholding. Several can be
                        specified with '--obs-cols arcsinh_n_genes_by_counts
                        pct_counts_ambient'
  --directions DIRECTIONS [DIRECTIONS ...]
                        [all] Direction of thresholding for each heuristic.
                        Several can be specified with '--obs-cols above below'
  --thresh-method THRESH_METHOD
                        [all] Method used for automatic thresholding on
                        heuristics. One of ['otsu','li','mean']
  --mito-names MITO_NAMES
                        [all] Substring or regex defining mitochondrial genes
  --n-hvgs N_HVGS       [all] Number of highly variable genes for training
                        model
  --seed SEED           [all] Random state for cross validation [regression]
                        or sampling training set [twostep]
  --output-dir [OUTPUT_DIR]
                        [all] Output directory. Output will be placed in
                        [output-dir]/[name]...
  --alphas [ALPHAS [ALPHAS ...]]
                        [regression] Ratios between l1 and l2 regularization
                        for regression model
  --n-lambda N_LAMBDA   [regression] Number of lambda (regularization
                        strength) values to test
  --cut-point CUT_POINT
                        [regression] The cut point to use for selecting
                        lambda_best
  --n-splits N_SPLITS   [regression] Number of splits for cross validation
  --n-iter N_ITER       [regression] Maximum number of iterations for
                        optimization
  --n-jobs N_JOBS       [regression] Maximum number of threads for cross
                        validation
  --pos-frac POS_FRAC   [twostep] Fraction of cells below threshold to sample
                        for training set
  --neg-frac NEG_FRAC   [twostep] Fraction of cells above threshold to sample
                        for training set
"""
import argparse
import sys
import os, errno
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import time
import threading
from skimage.filters import threshold_li, threshold_otsu, threshold_mean
from sklearn.ensemble import RandomForestClassifier

from PU_twostep import twoStep
from logistic import LogitNet


class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1: 
            for cursor in '|/-\\': yield cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()

    def __enter__(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def __exit__(self, exception, value, tb):
        self.busy = False
        time.sleep(self.delay)
        if exception is not None:
            return False



def check_dir_exists(path):
    """
    Checks if directory already exists or not and creates it if it doesn't
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def recipe_dropkick(
    adata,
    X_final="raw_counts",
    filter=True,
    calc_metrics=True,
    mito_names="^mt-|^MT-",
    n_ambient=10,
    target_sum=None,
    n_hvgs=2000,
    verbose=True,
):
    """
    scanpy preprocessing recipe

    Parameters:
        adata (AnnData.AnnData): object with raw counts data in .X
        X_final (str): which normalization should be left in .X slot?
            ("raw_counts","arcsinh_norm","norm_counts")
        filter (bool): remove cells and genes with zero total counts
        calc_metrics (bool): if False, do not calculate metrics in .obs/.var
        mito_names (str): substring encompassing mitochondrial gene names for
            calculation of mito expression
        n_ambient (int): number of ambient genes to call. top genes by cells.
        target_sum (int): total sum of counts for each cell prior to arcsinh 
            and log1p transformations; default None to use median counts.
        n_hvgs (int or None): number of HVGs to calculate using Seurat method
            if None, do not calculate HVGs
        verbose (bool): print updates to the console?

    Returns:
        AnnData.AnnData: adata is edited in place to include:
        - useful .obs and .var columns
            ("total_counts", "pct_counts_mito", "n_genes_by_counts", etc.)
        - raw counts (adata.layers["raw_counts"])
        - normalized counts (adata.layers["norm_counts"])
        - arcsinh transformation of normalized counts (adata.X)
        - highly variable genes if desired (adata.var["highly_variable"])
    """
    if filter:
        # remove cells and genes with zero total counts
        orig_shape = adata.shape
        sc.pp.filter_cells(adata, min_genes=10)
        sc.pp.filter_genes(adata, min_counts=1)
        if adata.shape[0] != orig_shape[0]:
            print("Removed {} cells with zero total counts".format(orig_shape[0]-adata.shape[0]))
        if adata.shape[1] != orig_shape[1]:
            print("Removed {} genes with zero total counts".format(orig_shape[1]-adata.shape[1]))

    # store raw counts before manipulation
    adata.layers["raw_counts"] = adata.X.copy()

    if calc_metrics:
        if verbose:
            print("Calculating metrics:")
        # identify mitochondrial genes
        adata.var["mito"] = adata.var_names.str.contains(mito_names)
        # identify putative ambient genes by lowest dropout pct (top 10)
        adata.var["ambient"] = np.array(adata.X.astype(bool).sum(axis=0) / adata.n_obs).squeeze()
        if verbose:
            print(
                "Top {} ambient genes have dropout rates between {} and {} percent:\n\t{}".format(
                    n_ambient,
                    round((1 - adata.var.ambient.nlargest(n=n_ambient).max()) * 100, 2),
                    round((1 - adata.var.ambient.nlargest(n=n_ambient).min()) * 100, 2),
                    adata.var.ambient.nlargest(n=n_ambient).index.tolist(),
                )
            )
        adata.var["ambient"] = (
            adata.var.ambient >= adata.var.ambient.nlargest(n=n_ambient).min()
        )
        # calculate standard qc .obs and .var
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mito", "ambient"], inplace=True, percent_top=[10, 50, 100]
        )
        # other arcsinh-transformed metrics
        adata.obs["arcsinh_total_counts"] = np.arcsinh(adata.obs["total_counts"])
        adata.obs["arcsinh_n_genes_by_counts"] = np.arcsinh(
            adata.obs["n_genes_by_counts"]
        )

    # log1p transform (adata.layers["log1p_norm"])
    sc.pp.normalize_total(adata, target_sum=target_sum, layers=None, layer_norm=None)
    adata.layers["norm_counts"] = adata.X.copy()  # save to .layers
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()  # save to .layers

    # HVGs
    if n_hvgs is not None:
        if verbose:
            print("Determining {} highly variable genes".format(n_hvgs))
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_hvgs, n_bins=20, flavor="seurat"
        )

    # arcsinh-transform normalized counts to leave in .X
    adata.X = np.arcsinh(adata.layers["norm_counts"])
    sc.pp.scale(adata)  # scale genes for feeding into model
    adata.layers[
        "arcsinh_norm"
    ] = adata.X.copy()  # save arcsinh scaled counts in .layers

    # set .X as desired for downstream processing; default raw_counts
    adata.X = adata.layers[X_final].copy()


def auto_thresh_obs(
    adata, obs_cols=["arcsinh_n_genes_by_counts", "pct_counts_ambient"], method="otsu",
):
    """
    automated thresholding on metrics in adata.obs

    Parameters:
        adata (anndata.AnnData): object containing unfiltered scRNA-seq data
        obs_cols (list of str): name of column(s) to threshold from adata.obs
        method (str): one of 'otsu' (default), 'li', or 'mean'

    Returns:
        thresholds (dict): keys are obs_cols and values are threshold results
    """
    thresholds = dict.fromkeys(obs_cols)  # initiate output dictionary
    for col in obs_cols:
        tmp = np.array(adata.obs[col])
        if method == "otsu":
            thresholds[col] = threshold_otsu(tmp)
        elif method == "li":
            thresholds[col] = threshold_li(tmp)
        elif method == "mean":
            thresholds[col] = threshold_mean(tmp)
        else:
            raise ValueError(
                "Please provide a valid threshold method ('otsu', 'li', 'mean')."
            )

    return thresholds


def plot_thresh_obs(adata, thresholds, bins=40, show=True):
    """
    plot automated thresholding on metrics in adata.obs as output by auto_thresh_obs()

    Parameters:
        adata (anndata.AnnData): object containing unfiltered scRNA-seq data
        thresholds (dict): output of auto_thresh_obs() function
        bins (int): number of bins for histogram
        show (bool): show plot or return object

    Returns:
        plot of distributions of obs_cols in thresholds dictionary with corresponding threshold values
    """
    fig, axes = plt.subplots(
        ncols=len(thresholds), nrows=1, figsize=(len(thresholds) * 4, 4), sharey=True
    )
    axes[0].set_ylabel("cells")
    for i in range(len(thresholds)):
        axes[i].hist(adata.obs[list(thresholds.keys())[i]], bins=bins)
        axes[i].axvline(list(thresholds.values())[i], color="r")
        axes[i].set_title(list(thresholds.keys())[i])
    fig.tight_layout()
    if show:
        plt.show()
    else:
        return fig


def filter_thresh_obs(
    adata,
    thresholds,
    obs_cols=["arcsinh_n_genes_by_counts", "pct_counts_ambient"],
    directions=["above", "below"],
    inclusive=True,
    name="thresh_filter",
):
    """
    filter cells by thresholding on metrics in adata.obs as output by auto_thresh_obs()

    Parameters:
        adata (anndata.AnnData): object containing unfiltered scRNA-seq data
        thresholds (dict): output of auto_thresh_obs() function
        obs_cols (list of str): name of column(s) to threshold from adata.obs
        directions (list of str): 'below' or 'above', indicating which direction to keep (label=1)
        inclusive (bool): include cells at the thresholds? default True.
        name (str): name of .obs col containing final labels

    Returns:
        updated adata with filter labels in adata.obs[name]
    """
    # initialize .obs column as all "good" cells
    adata.obs[name] = 1
    # if any criteria are NOT met, label cells "bad"
    for i in range(len(obs_cols)):
        if directions[i] == "above":
            if inclusive:
                adata.obs.loc[
                    (adata.obs[name] == 1)
                    & (adata.obs[obs_cols[i]] <= thresholds[obs_cols[i]]),
                    name,
                ] = 0
            else:
                adata.obs.loc[
                    (adata.obs[name] == 1)
                    & (adata.obs[obs_cols[i]] < thresholds[obs_cols[i]]),
                    name,
                ] = 0
        elif directions[i] == "below":
            if inclusive:
                adata.obs.loc[
                    (adata.obs[name] == 1)
                    & (adata.obs[obs_cols[i]] >= thresholds[obs_cols[i]]),
                    name,
                ] = 0
            else:
                adata.obs.loc[
                    (adata.obs[name] == 1)
                    & (adata.obs[obs_cols[i]] > thresholds[obs_cols[i]]),
                    name,
                ] = 0


def regression_pipe(
    adata,
    mito_names="^mt-|^MT-",
    n_hvgs=2000,
    thresh_method="otsu",
    metrics=["arcsinh_n_genes_by_counts", "pct_counts_ambient",],
    directions=["above", "below"],
    alphas=(0.1, 0.15, 0.2),
    n_lambda=10,
    cut_point=1,
    n_splits=3,
    max_iter=1000,
    n_jobs=-1,
    seed=18,
):
    """
    generate logistic regression model of cell quality

    Parameters:
        adata (anndata.AnnData): object containing unfiltered, raw scRNA-seq
            counts in .X layer
        mito_names (str): substring encompassing mitochondrial gene names for
            calculation of mito expression
        n_hvgs (int or None): number of HVGs to calculate using Seurat method
            if None, do not calculate HVGs
        thresh_method (str): one of 'otsu' (default), 'li', or 'mean'
        metrics (list of str): name of column(s) to threshold from adata.obs
        directions (list of str): 'below' or 'above', indicating which
            direction to keep (label=1)
        alphas (tuple of int): alpha values to test using glmnet with n-fold
            cross validation
        n_lambda (int): number of lambda values to test in glmnet
        cut_point (float): The cut point to use for selecting lambda_best.
            arg_max lambda
            cv_score(lambda)>=cv_score(lambda_max)-cut_point*standard_error(lambda_max)
        n_splits (int): number of splits for n-fold cross validation
        max_iter (int): number of iterations for glmnet optimization
        n_jobs (int): number of threads for cross validation by glmnet
        seed (int): random state for cross validation by glmnet

    Returns:
        adata_thresh (dict): dictionary of automated thresholds on heuristics
        rc (LogisticRegression): trained logistic regression classifier

        updated adata inplace to include 'train', 'dropkick_score', and
            'dropkick_label' columns in .obs
    """
    # 0) preprocess counts and calculate required QC metrics
    recipe_dropkick(
        adata,
        X_final="arcsinh_norm",
        filter=True,
        calc_metrics=True,
        mito_names=mito_names,
        n_hvgs=n_hvgs,
        target_sum=None,
        verbose=True,
    )

    # 1) threshold chosen heuristics using automated method
    print("Thresholding on heuristics for training labels: {}".format(metrics))
    adata_thresh = auto_thresh_obs(adata, method=thresh_method, obs_cols=metrics)

    # 2) create labels from combination of thresholds
    filter_thresh_obs(
        adata,
        adata_thresh,
        obs_cols=metrics,
        directions=directions,
        inclusive=True,
        name="train",
    )

    X = adata.X[:, adata.var.highly_variable].copy()  # final X is HVGs
    y = adata.obs["train"].copy(deep=True)  # final y is "train" labels from step 2

    if len(alphas)>1:
        # 3.1) cross-validation to choose alpha and lambda values
        cv_scores = {"rc": [], "lambda": [], "alpha": [], "score": []}  # dictionary o/p
        for alpha in alphas:
            print("Training LogitNet with alpha: {}".format(alpha), end="  ")
            rc = LogitNet(alpha=alpha, n_lambda=n_lambda, cut_point=cut_point, n_splits=n_splits, max_iter=max_iter, n_jobs=n_jobs, random_state=seed)
            with Spinner():
                rc.fit(adata=adata, y=y, n_hvgs=n_hvgs)
            cv_scores["rc"].append(rc)
            cv_scores["alpha"].append(alpha)
            cv_scores["lambda"].append(rc.lambda_best_)
            cv_scores["score"].append(rc.score(X, y, lamb=rc.lambda_best_))
        # determine optimal lambda and alpha values by accuracy score
        lambda_ = cv_scores["lambda"][
            cv_scores["score"].index(max(cv_scores["score"]))
        ]  # choose alpha value
        alpha_ = cv_scores["alpha"][
            cv_scores["score"].index(max(cv_scores["score"]))
        ]  # choose l1 ratio
        rc_ = cv_scores["rc"][
            cv_scores["score"].index(max(cv_scores["score"]))
        ]  # choose classifier
        print("Chosen lambda value: {}; Chosen alpha value: {}".format(lambda_, alpha_))
    else:
        # 3.2) train model with single alpha value
        print("Training LogitNet with alpha: {}".format(alphas[0]), end="  ")
        rc_ = LogitNet(alpha=alphas[0], n_lambda=n_lambda, cut_point=cut_point, n_splits=n_splits, max_iter=max_iter, n_jobs=n_jobs, random_state=seed)
        with Spinner():
            rc_.fit(adata=adata, y=y, n_hvgs=n_hvgs)
        lambda_, alpha_ = rc_.lambda_best_, alphas[0]

    # 5) use ridge model to assign scores and labels
    print("Assigning scores and labels")
    adata.obs["dropkick_score"] = rc_.predict_proba(X)[:, 1]
    adata.obs["dropkick_label"] = rc_.predict(X)
    adata.var.loc[adata.var.highly_variable, "dropkick_coef"] = rc_.coef_.squeeze()

    print("Done!\n")
    return adata_thresh, rc_, lambda_, alpha_


def sampling_probabilities(
    adata, obs_col, thresh, direction="below", inclusive=True, suffix="prob", plot=True
):
    """
    generate sampling probabilities from threshold on metric in adata.obs

    Parameters:
        adata (anndata.AnnData): object containing unfiltered scRNA-seq data
        obs_col (str): name of column to threshold from adata.obs
        thresh (float): threshold value for corresponding obs_col
        direction (str): 'below' or 'above', indicating which direction to make probabilities for
        inclusive (bool): include cells at the threshold? default True.
        suffix (str): string to append to end of obs_col name for making new adata.obs column

    Returns:
        updated adata with new .obs column containing sampling probabilities
    """
    # initialize new .obs column
    adata.obs["{}_{}".format(obs_col, suffix)] = 0
    # if below threshold, make probabilities inversely proportional to values
    if direction == "below":
        if inclusive:
            adata.obs.loc[
                adata.obs[obs_col] <= thresh, "{}_{}".format(obs_col, suffix)
            ] = np.reciprocal(adata.obs.loc[adata.obs[obs_col] <= thresh, obs_col] + 1)
        else:
            adata.obs.loc[
                adata.obs[obs_col] < thresh, "{}_{}".format(obs_col, suffix)
            ] = np.reciprocal(adata.obs.loc[adata.obs[obs_col] < thresh, obs_col] + 1)
    # if above threshold, probabilities are proportional to values
    elif direction == "above":
        if inclusive:
            adata.obs.loc[
                adata.obs[obs_col] >= thresh, "{}_{}".format(obs_col, suffix)
            ] = adata.obs.loc[adata.obs[obs_col] >= thresh, obs_col]
        else:
            adata.obs.loc[
                adata.obs[obs_col] > thresh, "{}_{}".format(obs_col, suffix)
            ] = adata.obs.loc[adata.obs[obs_col] > thresh, obs_col]
    else:
        raise ValueError(
            "Please provide a valid threshold direction ('above' or 'below')."
        )
    # normalize values to create probabilities (sum to 1)
    adata.obs["{}_{}".format(obs_col, suffix)] /= adata.obs[
        "{}_{}".format(obs_col, suffix)
    ].sum()
    # plot results on histogram
    if plot:
        new_order = np.argsort(adata.obs[obs_col])[::-1]
        tmp = adata[new_order, :].copy()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_title(obs_col)
        ax.set_ylabel("cells", color="b")
        ax.hist(tmp.obs[obs_col], bins=40, color="b")
        ax.tick_params(axis="y", labelcolor="b")
        ax2 = ax.twinx()
        ax2.set_ylabel("{}_{}".format(obs_col, suffix), color="g")
        ax2.plot(tmp.obs[obs_col], tmp.obs["{}_{}".format(obs_col, suffix)], color="g")
        ax2.tick_params(axis="y", labelcolor="g")
        plt.axvline(thresh, color="r")
        fig.tight_layout()
        plt.show()


def generate_training_labels(
    adata, pos_prob, pos_size, neg_prob=None, neg_size=None, name="train", seed=18
):
    """
    sample cells for training set

    Parameters:
        adata (anndata.AnnData): object containing unfiltered scRNA-seq data
        pos_prob (pd.Series): column of adata.obs containing probabilities of drawing positive label (1)
        pos_size (int): number of cells to assign positive label (1) to
        neg_prob (pd.Series or None): column of adata.obs containing probabilities of drawing negative label (0)
        neg_size (int or None): number of cells to assign negative label (0) to
        name (str): name of .obs col containing final labels
        seed (int): random seed for sampling

    Returns:
        updated adata with new .obs column containing sampled training labels
    """
    np.random.seed(seed=seed)  # set seed for np.random.choice
    adata.obs[name] = -1  # initialize column with all cells unlabeled (-1)
    if neg_size is not None:
        adata.obs.iloc[
            np.random.choice(
                a=len(adata.obs), size=neg_size, replace=False, p=neg_prob
            ),
            adata.obs.columns.get_loc(name),
        ] = 0
    adata.obs.iloc[
        np.random.choice(a=len(adata.obs), size=pos_size, replace=False, p=pos_prob),
        adata.obs.columns.get_loc(name),
    ] = 1


def twostep_pipe(
    adata,
    clf,
    mito_names="^mt-|^MT-",
    n_hvgs=2000,
    thresh_method="li",
    obs_cols=["arcsinh_n_genes_by_counts", "pct_counts_mito",],
    directions=["above", "below"],
    pos_frac=0.7,
    neg_frac=0.3,
    seed=18,
):
    """
    generate iteratively-trained RandomForest model of cell quality

    Parameters:
        adata (anndata.AnnData): object containing unfiltered, raw scRNA-seq
            counts in .X layer
        clf (sklearn classifier): classifier object such as RandomForestClassifier
        mito_names (str): substring encompassing mitochondrial gene names for
            calculation of mito expression
        n_hvgs (int or None): number of HVGs to calculate using Seurat method
            if None, do not calculate HVGs
        thresh_method (str): one of 'li' (default), 'otsu', or 'mean'
        obs_cols (list of str): name of column(s) to threshold from adata.obs
        directions (list of str): 'below' or 'above', indicating which
            direction to keep (label=1)
        pos_frac (float): fraction of positives (bad cells) to sample for training
        neg_frac (float): fraction of negative (good cells) to sample for training
        seed (int): random state for sampling training set

    Returns:
        adata_thresh (dict): dictionary of automated thresholds on heuristics
        rc (RidgeClassifierCV): trained ridge classifier

        updated adata inplace to include 'train', 'dropkick_score', and
            'dropkick_label' columns in .obs
    """
    # 1) preprocess counts and calculate required QC metrics
    print("Preprocessing counts and calculating metrics")
    recipe_dropkick(
        adata, mito_names=mito_names, n_hvgs=n_hvgs, target_sum=None, verbose=True
    )

    # 2) threshold chosen heuristics using automated method
    print("Thresholding on heuristics for training labels")
    adata_thresh = auto_thresh_obs(adata, method=thresh_method, obs_cols=obs_cols)

    # 3) create labels from combination of thresholds
    filter_thresh_obs(
        adata,
        adata_thresh,
        obs_cols=obs_cols,
        directions=directions,
        inclusive=True,
        name="thresh_filter",
    )

    # 4) calculate sampling probabilities from thresholded heuristics
    adata.obs["neg_prob"] = 0
    adata.obs["pos_prob"] = 0
    for i in range(len(obs_cols)):
        print("Generating sampling probabilities from {}".format(obs_cols[i]))
        sampling_probabilities(
            adata,
            obs_col=obs_cols[i],
            thresh=adata_thresh[obs_cols[i]],
            direction=directions[i],
            inclusive=True,
            suffix="neg",
            plot=False,
        )
        adata.obs["neg_prob"] += adata.obs[
            "{}_neg".format(obs_cols[i])
        ]  # add probabilities to combined vector
        if directions[i] == "above":
            pos_dir = "below"
        elif directions[i] == "below":
            pos_dir = "above"
        sampling_probabilities(
            adata,
            obs_col=obs_cols[i],
            thresh=adata_thresh[obs_cols[i]],
            direction=pos_dir,
            inclusive=True,
            suffix="pos",
            plot=False,
        )
        adata.obs["pos_prob"] += adata.obs[
            "{}_pos".format(obs_cols[i])
        ]  # add probabilities to combined vector
    # normalize combined probabilities
    adata.obs["neg_prob"] /= adata.obs["neg_prob"].sum()
    adata.obs["pos_prob"] /= adata.obs["pos_prob"].sum()

    # 5) generate training labels
    print(
        "Picking {} positives (empty droplets/dead cells) and {} negatives (live cells) for training".format(
            int(pos_frac * (adata.n_obs - adata.obs["thresh_filter"].sum())),
            int(neg_frac * (adata.n_obs - adata.obs["thresh_filter"].sum())),
        )
    )
    generate_training_labels(
        adata,
        pos_prob=adata.obs["pos_prob"],
        pos_size=int(pos_frac * (adata.n_obs - adata.obs["thresh_filter"].sum())),
        neg_prob=adata.obs["neg_prob"],
        neg_size=int(neg_frac * (adata.n_obs - adata.obs["thresh_filter"].sum())),
        name="train",
        seed=seed,
    )

    # 4) train two-step classifier
    y = adata.obs["train"].copy(deep=True)  # training labels defined above
    if n_hvgs is None:
        # if no HVGs, train on all genes. NOTE: this slows computation considerably.
        X = adata.X
    else:
        # use HVGs if provided
        X = adata.X[:, adata.var["highly_variable"] == True]
    print("Training two-step classifier:")
    adata.obs["dropkick_score"], adata.obs["dropkick_label"] = twoStep(
        clf=clf, X=X, y=y, thresh="min", n_iter=18
    )
    print(
        "Predicting remaining {} unlabeled barcodes with trained classifier.".format(
            (y == -1).sum()
        )
    )
    adata.obs.loc[y == -1, "dropkick_label"] = clf.predict(
        X[y == -1]
    )  # predict remaining unlabeled cells using trained clf
    adata.obs["dropkick_label"] = (~adata.obs["dropkick_label"].astype(bool)).astype(
        int
    )  # flip labels so 1 is good cell

    print("Done!\n")
    return adata_thresh, clf



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", type=str, choices=["regression", "twostep"],
    )
    parser.add_argument(
        "-c",
        "--counts",
        type=str,
        help="[all] Input (cell x gene) counts matrix as .h5ad or tab delimited text file",
        required=True,
    )
    parser.add_argument(
        "--obs-cols",
        type=str,
        help="[all] Heuristics for thresholding. Several can be specified with '--obs-cols arcsinh_n_genes_by_counts pct_counts_ambient'",
        nargs="+",
        default=["arcsinh_n_genes_by_counts", "pct_counts_ambient"],
    )
    parser.add_argument(
        "--directions",
        type=str,
        help="[all] Direction of thresholding for each heuristic. Several can be specified with '--obs-cols above below'",
        nargs="+",
        default=["above", "below"],
    )
    parser.add_argument(
        "--thresh-method",
        type=str,
        help="[all] Method used for automatic thresholding on heuristics. One of ['otsu','li','mean']",
        default="otsu",
    )
    parser.add_argument(
        "--mito-names",
        type=str,
        help="[all] Substring or regex defining mitochondrial genes",
        default="^mt-|^MT-",
    )
    parser.add_argument(
        "--n-hvgs",
        type=int,
        help="[all] Number of highly variable genes for training model",
        default=2000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="[all] Random state for cross validation [regression] or sampling training set [twostep]",
        default=18,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="[all] Output directory. Output will be placed in [output-dir]/[name]...",
        nargs="?",
        default=".",
    )

    parser.add_argument(
        "--alphas",
        type=float,
        help="[regression] Ratios between l1 and l2 regularization for regression model",
        nargs="*",
        default=[0.1, 0.15, 0.2],
    )
    parser.add_argument(
        "--n-lambda",
        type=int,
        help="[regression] Number of lambda (regularization strength) values to test",
        default=10,
    )
    parser.add_argument(
        "--cut-point",
        type=float,
        help="[regression] The cut point to use for selecting lambda_best",
        default=1.0,
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        help="[regression] Number of splits for cross validation",
        default=3,
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        help="[regression] Maximum number of iterations for optimization",
        default=100000,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="[regression] Maximum number of threads for cross validation",
        default=-1,
    )

    parser.add_argument(
        "--pos-frac",
        type=float,
        help="[twostep] Fraction of cells below threshold to sample for training set",
        default=0.7,
    )
    parser.add_argument(
        "--neg-frac",
        type=float,
        help="[twostep] Fraction of cells above threshold to sample for training set",
        default=0.3,
    )

    args = parser.parse_args()

    # read in counts data
    print("\nReading in unfiltered counts from {}".format(args.counts), end="")
    adata = sc.read(args.counts)
    print(" - {} barcodes and {} genes".format(adata.shape[0], adata.shape[1]))

    # create copy of AnnData to manipulate
    tmp = adata.copy()

    # check that output directory exists, create it if needed.
    check_dir_exists(args.output_dir)
    # get basename of file for writing outputs
    name = os.path.splitext(os.path.basename(args.counts))[0]

    if args.command == "regression":
        thresholds, regression_model, lambda_, alpha_ = regression_pipe(
            tmp,
            mito_names=args.mito_names,
            n_hvgs=args.n_hvgs,
            thresh_method=args.thresh_method,
            metrics=args.obs_cols,
            directions=args.directions,
            alphas=args.alphas,
            n_lambda=args.n_lambda,
            cut_point=args.cut_point,
            n_splits=args.n_splits,
            max_iter=args.n_iter,
            n_jobs=args.n_jobs,
            seed=args.seed,
        )
        # generate plot of chosen training thresholds on heuristics
        print(
            "Saving threshold plots to {}/{}_{}_thresholds.png".format(
                args.output_dir, name, args.thresh_method
            )
        )
        thresh_plt = plot_thresh_obs(tmp, thresholds, bins=40, show=False)
        plt.savefig(
            "{}/{}_{}_thresholds.png".format(args.output_dir, name, args.thresh_method)
        )
        # save new labels
        print(
            "Writing updated counts to {}/{}_{}.h5ad".format(
                args.output_dir, name, args.command
            )
        )
        (
            adata.obs["dropkick_train"],
            adata.obs["dropkick_score"],
            adata.obs["dropkick_label"],
            adata.var["dropkick_hvgs"],
            adata.var["dropkick_coef"],
        ) = (
            tmp.obs["train"],
            tmp.obs["dropkick_score"],
            tmp.obs["dropkick_label"],
            tmp.var["highly_variable"],
            tmp.var["dropkick_coef"],
        )
        adata.uns["pipeline_args"] = {
            "counts": args.counts,
            "n_hvgs": args.n_hvgs,
            "thresh_method": args.thresh_method,
            "obs_cols": args.obs_cols,
            "directions": args.directions,
            "alphas": args.alphas,
            "chosen_alpha": alpha_,
            "chosen_lambda": lambda_,
            "n_lambda": args.n_lambda,
            "cut_point": args.cut_point,
            "n_splits": args.n_splits,
            "max_iter": args.n_iter,
            "seed": args.seed,
        }  # save command-line arguments to .uns for reference
        adata.write(
            "{}/{}_{}.h5ad".format(args.output_dir, name, args.command),
            compression="gzip",
        )

    elif args.command == "twostep":
        thresholds, twostep_model = twostep_pipe(
            tmp,
            clf=RandomForestClassifier(
                n_estimators=500, n_jobs=-1
            ),  # use default clf for now
            mito_names=args.mito_names,
            n_hvgs=args.n_hvgs,
            thresh_method=args.thresh_method,
            obs_cols=args.obs_cols,
            directions=args.directions,
            pos_frac=args.pos_frac,
            neg_frac=args.neg_frac,
            seed=args.seed,
        )
        # generate plot of chosen training thresholds on heuristics
        print(
            "Saving threshold plots to {}/{}_{}_thresholds.png".format(
                args.output_dir, name, args.thresh_method
            )
        )
        thresh_plt = plot_thresh_obs(tmp, thresholds, bins=40, show=False)
        plt.savefig(
            "{}/{}_{}_thresholds.png".format(args.output_dir, name, args.thresh_method)
        )
        # save new labels
        print(
            "Writing updated counts to {}/{}_{}.h5ad".format(
                args.output_dir, name, args.command
            )
        )
        (
            adata.obs["train"],
            adata.obs["dropkick_score"],
            adata.obs["dropkick_label"],
            adata.obs["pos_prob"],
            adata.obs["neg_prob"],
        ) = (
            tmp.obs["train"],
            tmp.obs["dropkick_score"],
            tmp.obs["dropkick_label"],
            tmp.obs["pos_prob"],
            tmp.obs["neg_prob"],
        )
        adata.uns["pipeline_args"] = {
            "counts": args.counts,
            "obs_cols": args.obs_cols,
            "directions": args.directions,
            "mito_names": args.mito_names,
            "n_hvgs": args.n_hvgs,
            "thresh_method": args.thresh_method,
            "pos_frac": args.pos_frac,
            "neg_frac": args.neg_frac,
            "seed": args.seed,
        }  # save command-line arguments to .uns for reference
        adata.write(
            "{}/{}_{}.h5ad".format(args.output_dir, name, args.command),
            compression="gzip",
        )

    else:
        raise ValueError(
            "Please provide a valid filtering command ('regression', 'twostep')"
        )
