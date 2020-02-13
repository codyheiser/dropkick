# -*- coding: utf-8 -*-
"""
Automated QC classifier pipeline

@author: C Heiser
"""
import argparse
import os, errno
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from skimage.filters import threshold_li, threshold_otsu, threshold_mean
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from PU_twostep import twoStep
from QC import reorder_adata, arcsinh_norm, gf_icf, recipe_fcc


def check_dir_exists(path):
    """
    Checks if directory already exists or not and creates it if it doesn't
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


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
    directions=["above", "below",],
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


def ridge_pipe(
    adata,
    mito_names="^mt-|^MT-",
    n_hvgs=2000,
    thresh_method="otsu",
    metrics=["arcsinh_n_genes_by_counts", "pct_counts_ambient",],
    directions=["above", "below"],
    alphas=(100, 200, 300, 400, 500),
):
    """
    generate ridge regression model of cell quality

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
        alphas (tuple of int): alpha values to test using RidgeClassifierCV

    Returns:
        adata_thresh (dict): dictionary of automated thresholds on heuristics
        rc (RidgeClassifierCV): trained ridge classifier

        updated adata inplace to include 'train', 'dropkeeper_score', and
            'dropkeeper_label' columns in .obs
    """
    # 1) preprocess counts and calculate required QC metrics
    print("Preprocessing counts and calculating metrics")
    recipe_fcc(
        adata,
        X_final="arcsinh_norm",
        mito_names=mito_names,
        n_hvgs=n_hvgs,
        target_sum=None,
    )

    # 2) threshold chosen heuristics using automated method
    print("Thresholding on heuristics for training labels: {}".format(metrics))
    adata_thresh = auto_thresh_obs(adata, method=thresh_method, obs_cols=metrics)

    # 3) create labels from combination of thresholds
    filter_thresh_obs(
        adata,
        adata_thresh,
        obs_cols=metrics,
        directions=directions,
        inclusive=True,
        name="train",
    )

    # 4) train ridge regression classifier with cross validation
    y = adata.obs["train"].copy(deep=True)  # training labels defined above
    if n_hvgs is None:
        # if no HVGs, train on all genes. NOTE: this slows computation considerably.
        X = adata.X
    else:
        # use HVGs if provided
        X = adata.X[:, adata.var["highly_variable"] == True]
    print("Training ridge classifier with alpha values: {}".format(alphas))
    rc = RidgeClassifierCV(alphas=alphas, store_cv_values=True)
    rc.fit(X, y)
    print("Chosen alpha value: {}".format(rc.alpha_))

    # 5) use ridge model to assign scores and labels
    print("Assigning scores and labels from model")
    adata.obs["dropkeeper_score"] = rc.decision_function(X)
    adata.obs["dropkeeper_label"] = rc.predict(X)

    print("Done!")
    return adata_thresh, rc


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
    generate ridge regression model of cell quality

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

        updated adata inplace to include 'train', 'dropkeeper_score', and
            'dropkeeper_label' columns in .obs
    """
    # 1) preprocess counts and calculate required QC metrics
    print("Preprocessing counts and calculating metrics")
    recipe_fcc(
        adata,
        X_final="arcsinh_norm",
        mito_names=mito_names,
        n_hvgs=n_hvgs,
        target_sum=None,
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
    adata.obs["dropkeeper_score"], adata.obs["dropkeeper_label"] = twoStep(
        clf=clf, X=X, y=y, thresh="min", n_iter=18
    )
    print(
        "Predicting remaining {} unlabeled barcodes with trained classifier.".format(
            (y == -1).sum()
        )
    )
    adata.obs.loc[y == -1, "dropkeeper_label"] = clf.predict(
        X[y == -1]
    )  # predict remaining unlabeled cells using trained clf
    adata.obs["dropkeeper_label"] = (~adata.obs["dropkeeper_label"].astype(bool)).astype(
        int
    )  # flip labels so 1 is good cell

    print("Done!")
    return adata_thresh, clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", type=str, choices=["ridge", "twostep"],
    )
    parser.add_argument(
        "-c",
        "--counts",
        type=str,
        help="[all] Input (cell x gene) counts matrix as .h5ad or tab delimited text file",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="[all] Output directory. Output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default=".",
    )
    parser.add_argument(
        "--obs-cols",
        type=str,
        help="[all] Heuristics for thresholding. Several can be specified with '--obs-cols arcsinh_n_genes_by_counts pct_counts_mito'",
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
        "--alphas",
        type=float,
        help="[ridge] Alpha values for ridge regression model. Several can be specified with '--alphas 100 200 300 400'",
        nargs="*",
        default=[100, 200, 300, 400, 500],
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
    parser.add_argument(
        "--seed",
        type=int,
        help="[twostep] Random state for sampling training set",
        default=18,
    )

    args = parser.parse_args()

    # read in counts data
    print("\nReading in unfiltered counts from {}".format(args.counts), end="")
    adata = sc.read(args.counts)
    print(" - {} barcodes and {} genes".format(adata.shape[0], adata.shape[1]))
    tmp = adata.copy()  # copy of AnnData to manipulate
    # Check that output directory exists, create it if needed.
    check_dir_exists(args.output_dir)
    # get basename of file for writing outputs
    name = os.path.splitext(os.path.basename(args.counts))[0]

    if args.command == "ridge":
        thresholds, ridge_model = ridge_pipe(
            tmp,
            mito_names=args.mito_names,
            n_hvgs=args.n_hvgs,
            thresh_method=args.thresh_method,
            metrics=args.obs_cols,
            directions=args.directions,
            alphas=args.alphas,
        )
        # generate plot of chosen training thresholds on heuristics
        print(
            "Saving threshold plots to {}/{}_{}_thresholds.png".format(
                args.output_dir, name, args.thresh_method
            )
        )
        thresh_plt = plot_thresh_obs(tmp, thresholds, bins=40, show=False)
        plt.savefig(
            "{}/{}_{}_thresholds.png".format(
                args.output_dir, name, args.thresh_method
            )
        )
        # save new labels
        print(
            "Writing updated counts to {}/{}_{}.h5ad".format(
                args.output_dir, name, args.command
            )
        )
        adata.obs["train"], adata.obs["dropkeeper_score"], adata.obs["dropkeeper_label"] = (
            tmp.obs["train"],
            tmp.obs["dropkeeper_score"],
            tmp.obs["dropkeeper_label"],
        )
        adata.uns["pipeline_args"] = {
            "counts": args.counts,
            "obs_cols": args.obs_cols,
            "directions": args.directions,
            "alphas": args.alphas,
            "chosen_alpha": ridge_model.alpha_,
            "mito_names": args.mito_names,
            "n_hvgs": args.n_hvgs,
            "thresh_method": args.thresh_method,
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
            "{}/{}_{}_thresholds.png".format(
                args.output_dir, name, args.thresh_method
            )
        )
        # save new labels
        print(
            "Writing updated counts to {}/{}_{}.h5ad".format(
                args.output_dir, name, args.command
            )
        )
        (
            adata.obs["train"],
            adata.obs["dropkeeper_score"],
            adata.obs["dropkeeper_label"],
            adata.obs["pos_prob"],
            adata.obs["neg_prob"],
        ) = (
            tmp.obs["train"],
            tmp.obs["dropkeeper_score"],
            tmp.obs["dropkeeper_label"],
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
            "Please provide a valid filtering command ('ridge', 'twostep')"
        )
