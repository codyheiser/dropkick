# -*- coding: utf-8 -*-
"""
Automated QC classifier pipeline

@author: C Heiser
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_li, threshold_otsu, threshold_mean


def auto_thresh_obs(
    adata,
    obs_cols=[
        "arcsinh_total_counts",
        "arcsinh_n_genes_by_counts",
        "gf_icf_total",
        "pct_counts_mito",
        "pct_counts_in_top_50_genes",
    ],
    method="otsu",
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


def plot_thresh_obs(adata, thresholds, bins=40):
    """
    plot automated thresholding on metrics in adata.obs as output by auto_thresh_obs()

    Parameters:
        adata (anndata.AnnData): object containing unfiltered scRNA-seq data
        thresholds (dict): output of auto_thresh_obs() function

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
    plt.show()


def filter_thresh_obs(
    adata,
    thresholds,
    obs_cols=[
        "arcsinh_total_counts",
        "arcsinh_n_genes_by_counts",
        "gf_icf_total",
        "pct_counts_mito",
        "pct_counts_in_top_50_genes",
    ],
    directions=["above", "above", "above", "below", "below",],
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
