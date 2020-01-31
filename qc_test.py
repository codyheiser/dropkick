# -*- coding: utf-8 -*-
"""
Automated testing of cell filtering labels

@author: C Heiser
"""
import argparse
import matplotlib.pyplot as plt
import scanpy as sc
from QC import reorder_adata, arcsinh_norm, gf_icf, recipe_fcc


def set_diff(adata, labels, metrics=None):
    """
    return number of cells different between two labels

    Parameters:
        adata (anndata.AnnData): object with cell labels in .obs
        labels (list of str): two labels (columns of .obs) to compare the cell sets
            1 = real cell, 0 = empty or dead

    Returns:
        prints results
    """
    if len(labels)!=2:
        raise ValueError(
                "Please provide exactly two cell labels."
            )
    unique_0 = len(set(adata.obs_names[adata.obs[labels[0]]==1]).difference(set(adata.obs_names[adata.obs[labels[1]]==1])))
    unique_1 = len(set(adata.obs_names[adata.obs[labels[1]]==1]).difference(set(adata.obs_names[adata.obs[labels[0]]==1])))
    print("{} cells in {} - {} unique".format(adata.obs[labels[0]].sum(), labels[0], unique_0))
    if metrics is not None:
        for m in metrics:
            print("\t{}: {}".format(m, round(adata.obs.loc[adata.obs[labels[0]]==1, m].mean(),3)), end=" ")
        print("\n")
    print("{} cells in {} - {} unique".format(adata.obs[labels[1]].sum(), labels[1], unique_1))
    if metrics is not None:
        for m in metrics:
            print("\t{}: {}".format(m, round(adata.obs.loc[adata.obs[labels[1]]==1, m].mean(),3)), end=" ")
        print("\n")


def plot_set_obs(adata, labels, metrics=["arcsinh_total_counts","arcsinh_n_genes_by_counts","pct_counts_mito"], bins=40, show=True):
    """
    plot distribution of metrics in adata.obs for different labeled cell populations

    Parameters:
        adata (anndata.AnnData): object with cell labels and metrics in .obs
        labels (list of str): two labels (columns of .obs) to compare the cell sets
        metrics (list of str): .obs columns to plot distributions of
        bins (int): number of bins for histogram
        show (bool): show plot or return object

    Returns:
        plot of distributions of obs_cols split by cell labels
    """
    fig, axes = plt.subplots(ncols=len(metrics), nrows=1, figsize=(len(metrics) * 4, 4))
    axes[0].set_ylabel("cells")
    for i in range(len(metrics)):
        axes[i].hist(adata.obs.loc[adata.obs_names[adata.obs[labels[0]]==1],metrics[i]], alpha=0.5, label=labels[0], bins=bins)
        axes[i].hist(adata.obs.loc[adata.obs_names[adata.obs[labels[1]]==1],metrics[i]], alpha=0.5, label=labels[1], bins=bins)
        axes[i].hist(adata.obs.loc[set(adata.obs_names[adata.obs[labels[0]]==1]).difference(set(adata.obs_names[adata.obs[labels[1]]==1])),metrics[i]], alpha=0.5, label="{} unique".format(labels[0]), bins=bins)
        axes[i].hist(adata.obs.loc[set(adata.obs_names[adata.obs[labels[1]]==1]).difference(set(adata.obs_names[adata.obs[labels[0]]==1])),metrics[i]], alpha=0.5, label="{} unique".format(labels[1]), bins=bins)
        axes[i].set_title(metrics[i])
    axes[i].legend()
    fig.tight_layout()
    if show:
        plt.show()
    else:
        return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        help="Name for analysis. All output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default="dropkeeper",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory. All output will be placed in [output-dir]/[name]/...",
        nargs="?",
        default=".",
    )
    parser.add_argument(
        "-c",
        "--counts",
        type=str,
        help="Input (cell x gene) counts matrix as .h5ad file",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        help="Labels defining cell sets to compare",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--metrics",
        type=str,
        help="Heuristics for comparing. Several can be specified with e.g. '--metrics arcsinh_total_counts arcsinh_n_genes_by_counts pct_counts_mito'",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--mito-names",
        type=str,
        help="Substring or regex defining mitochondrial genes",
        default="^mt-",
    )

    args = parser.parse_args()
    # read in AnnData object
    print("\nReading in counts data from {}\n".format(args.counts))
    adata = sc.read(args.counts)
    # preprocess data and calculate metrics
    recipe_fcc(adata, mito_names=args.mito_names)
    # print set differences to console
    set_diff(adata, labels=args.labels, metrics=args.metrics)
    # generate plot of chosen metrics' distribution in two cell label populations
    print(
        "Saving distribution plots to {}/{}_metrics.png".format(
            args.output_dir, args.name
        )
    )
    plot_set_obs(adata, labels=args.labels, metrics=args.metrics, bins=40, show=False)
    plt.savefig(
        "{}/{}_metrics.png".format(
            args.output_dir, args.name
        )
    )
