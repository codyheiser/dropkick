# -*- coding: utf-8 -*-
"""
Utility functions for scRNA-seq quality control classifiers

@authors: B Chen, C Heiser
"""
import warnings

# basic matrix/dataframe manipulation
import numpy as np
import pandas as pd
from scipy import stats, interp
from itertools import cycle
import scanpy as sc

# sklearn tools
from sklearn.preprocessing import normalize, label_binarize
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier

# plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white")


# scanpy functions
def reorder_adata(adata, descending=True):
    """
    place cells in descending order of total counts
    
    Parameters:
        adata (AnnData.AnnData): AnnData object
        descending (bool): highest counts first

    Returns:
        AnnData.AnnData: adata cells are reordered in place
    """
    if descending:
        new_order = np.argsort(adata.X.sum(axis=1))[::-1]
    elif not descending:
        new_order = np.argsort(adata.X.sum(axis=1))[:]
    adata = adata[new_order, :].copy()


def reorder_AnnData(AnnData, descending=True):
    AnnData.obs["total_counts"] = AnnData.X.sum(axis=1)
    if descending == True:
        new_order = np.argsort(AnnData.obs["total_counts"])[::-1]
    elif descending == False:
        new_order = np.argsort(AnnData.obs["total_counts"])[:]
    AnnData.X = AnnData.X[new_order, :].copy()
    AnnData.obs = AnnData.obs.iloc[new_order].copy()


def arcsinh_norm(adata, layer=None, norm="l1", scale=None):
    """
    return arcsinh-normalized values for each element in anndata counts matrix

    Parameters:
        adata (AnnData.AnnData): AnnData object
        layer (str or None): name of layer to perform arcsinh-normalization on. if None, use AnnData.X
        norm (str or None): normalization strategy following GF-ICF transform.
            None: do not normalize counts
            "l1": divide each count by sum of counts for each cell (analogous to sc.pp.normalize_total)
            "l2": divide each count by sqrt of sum of squares of counts for each cell
        scale (int): factor to scale normalized counts to; if None, median total counts across all cells

    Returns:
        AnnData.AnnData: adata is edited in place to add arcsinh normalization to .layers
    """
    if layer is None:
        mat = adata.X
    else:
        mat = adata.layers[layer]

    if scale is None:
        scale = np.median(mat.sum(axis=1))

    if norm is None:
        adata.layers["arcsinh_norm"] = np.arcsinh(mat * scale)
    else:
        adata.layers["arcsinh_norm"] = np.arcsinh(
            normalize(mat, axis=1, norm=norm) * scale
        )


def gf_icf(adata, layer=None, transform="arcsinh", norm=None):
    """
    return GF-ICF scores for each element in anndata counts matrix

    Parameters:
        adata (AnnData.AnnData): AnnData object
        layer (str or None): name of layer to perform GF-ICF normalization on. if None, use AnnData.X
        transform (str): how to transform ICF weights. arcsinh is recommended to retain counts of genes
            expressed in all cells. log transform eliminates these genes from the dataset.
        norm (str or None): normalization strategy following GF-ICF transform.
            None: do not normalize GF-ICF scores
            "l1": divide each score by sum of scores for each cell (analogous to sc.pp.normalize_total)
            "l2": divide each score by sqrt of sum of squares of scores for each cell

    Returns:
        AnnData.AnnData: adata is edited in place to add GF-ICF normalization to .layers["gf_icf"]
    """
    if layer is None:
        m = adata.X
    else:
        m = adata.layers[layer]

    # gene frequency in each cell (l1 norm along rows)
    tf = m / m.sum(axis=1)[:, None]

    # number of cells containing each gene (sum nonzero along columns)
    nt = m.astype(bool).sum(axis=0)

    # if there are genes detected in zero cells, use "classic" method of pseudocount = 1
    if not np.all(nt):
        warnings.warn(
            "Encountered {} genes with 0 cells by counts. Consider removing these before proceeding (i.e. sc.pp.filter_genes(adata,min_cells=1))".format(
                np.size(nt) - np.count_nonzero(nt)
            )
        )
        # inverse cell frequency (total cells / number of cells containing each gene)
        if transform == "arcsinh":
            idf = np.arcsinh((adata.n_obs + 1) / (nt + 1))
        elif transform == "log":
            idf = np.log((adata.n_obs + 1) / (nt + 1))
        else:
            raise ValueError("Please provide a valid transform (log or arcsinh).")
    # otherwise, we can use "pure" GF-ICF transformation without potentially harmful pseudocount (preferred)
    else:
        # inverse cell frequency (total cells / number of cells containing each gene)
        if transform == "arcsinh":
            idf = np.arcsinh(adata.n_obs / nt)
        elif transform == "log":
            idf = np.log(adata.n_obs / nt)
        else:
            raise ValueError("Please provide a valid transform (log or arcsinh).")

    # save GF-ICF scores to .layers and total GF-ICF per cell in .obs
    tf_idf = tf * idf
    adata.obs["gf_icf_total"] = tf_idf.sum(axis=1)
    if norm is None:
        adata.layers["gf_icf"] = tf_idf
    else:
        adata.layers["gf_icf"] = normalize(tf_idf, norm=norm, axis=1)


def recipe_fcc(
    adata, X_final="raw_counts", mito_names="MT-", target_sum=None, n_hvgs=2000
):
    """
    scanpy preprocessing recipe

    Parameters:
        adata (AnnData.AnnData): object with raw counts data in .X
        X_final (str): which normalization should be left in .X slot?
            ("raw_counts","gf_icf","log1p_norm","arcsinh_norm")
        mito_names (str): substring encompassing mitochondrial gene names for
            calculation of mito expression
        target_sum (int): total sum of counts for each cell prior to arcsinh 
            and log1p transformations; default 1e6 for TPM
        n_hvgs (int or None): number of HVGs to calculate using Seurat method
            if None, do not calculate HVGs

    Returns:
        AnnData.AnnData: adata is edited in place to include:
        - useful .obs and .var columns
            ("total_counts", "pct_counts_mito", "n_genes_by_counts", etc.)
        - cells ordered by "total_counts"
        - raw counts (adata.layers["raw_counts"])
        - GF-ICF transformation of counts (adata.layers["gf_icf"])
        - arcsinh transformation of normalized counts
            (adata.layers["arcsinh_norm"])
        - log1p transformation of normalized counts
            (adata.X, adata.layers["log1p_norm"])
        - highly variable genes if desired (adata.var["highly_variable"])
    """
    # reorder cells by total counts descending
    reorder_adata(adata, descending=True)

    # store raw counts before manipulation
    adata.layers["raw_counts"] = adata.X.copy()

    # identify mitochondrial genes
    adata.var["mito"] = adata.var_names.str.contains(mito_names)
    # calculate standard qc .obs and .var
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mito"], inplace=True, percent_top=[10, 50, 100, 200, 500]
    )
    # rank cells by total counts
    adata.obs["ranked_total_counts"] = np.argsort(adata.obs["total_counts"])
    # arcsinh-transformed total counts
    adata.obs["arcsinh_n_genes_by_counts"] = np.arcsinh(adata.obs["n_genes_by_counts"])

    # arcsinh transform (adata.layers["arcsinh_norm"]) and add total for visualization
    arcsinh_norm(adata, norm="l1", scale=target_sum)
    adata.obs["arcsinh_total_counts"] = np.arcsinh(adata.obs["total_counts"])

    # GF-ICF transform (adata.layers["gf_icf"], adata.obs["gf_icf_total"])
    gf_icf(adata)

    # log1p transform (adata.layers["log1p_norm"])
    sc.pp.normalize_total(
        adata,
        target_sum=target_sum,
        layers=None,
        layer_norm=None,
        key_added="log1p_norm_factor",
    )
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()  # save to .layers

    # HVGs
    if n_hvgs is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, n_bins=20, flavor="seurat")

    # set .X as desired for downstream processing; default raw_counts
    adata.X = adata.layers[X_final].copy()


def find_inflection(
    ann_data, inflection_percentiles=[0, 15, 30, 100], output_prefix="Output"
):
    ann_data_cumsum = np.cumsum(ann_data.obs["total_counts"])
    x_vals = np.arange(0, ann_data.obs.shape[0])
    secant_coef = ann_data_cumsum[ann_data.obs.shape[0] - 1] / ann_data.obs.shape[0]
    secant_line = secant_coef * x_vals
    secant_dist = abs(ann_data_cumsum - secant_line)
    inflection_percentiles_inds = np.percentile(x_vals, inflection_percentiles).astype(
        int
    )
    inflection_points = secant_dist.argsort()[::-1]
    percentile_points = inflection_points[inflection_percentiles_inds]
    color = plt.cm.tab10(np.linspace(0, 1, ann_data.obs.shape[0]))
    plt.figure(figsize=(20, 10))
    plt.plot(np.array(ann_data_cumsum), label="Cumulative Sum")
    # plt.plot(np.array(secant_line), label="Secant Line")
    plt.plot(np.array(secant_dist), label="Secant Distance")
    for percentile in percentile_points:
        plt.axvline(
            x=percentile,
            ymin=0,
            c=color[percentile],
            linestyle="--",
            linewidth=2,
            label="Inflection point {}".format(percentile),
        )
    plt.legend()
    # save to file
    if output_prefix != "":
        plt.savefig(output_prefix + "_inflectionCheck.png", bbox_inches="tight")
    else:
        plt.show()
    print(
        "Inflection point at {} for {} percentiles of greatest secant distances".format(
            percentile_points, inflection_percentiles
        )
    )
    # SJCG: added the ability to return a dictionary of points
    return dict(zip(inflection_percentiles, percentile_points))


def cluster_summary_stats(AnnData, raw=False):
    cluster_means = np.zeros((len(np.unique(AnnData.obs["louvain"])), AnnData.n_vars))
    cluster_medians = np.zeros((len(np.unique(AnnData.obs["louvain"])), AnnData.n_vars))
    cluster_stdev = np.zeros((len(np.unique(AnnData.obs["louvain"])), AnnData.n_vars))
    if raw == True:
        for cluster in range(len(np.unique(AnnData.obs["louvain"]))):
            cluster_means[cluster] = np.array(
                np.mean(AnnData[AnnData.obs["louvain"].isin([str(cluster)])].X, axis=0)
            )
            cluster_medians[cluster] = np.array(
                np.median(
                    AnnData[AnnData.obs["louvain"].isin([str(cluster)])].X, axis=0
                )
            )
            cluster_stdev[cluster] = np.array(
                np.std(AnnData[AnnData.obs["louvain"].isin([str(cluster)])].X, axis=0)
            )
    elif raw == False:
        for cluster in range(len(np.unique(AnnData.obs["louvain"]))):
            cluster_means[cluster] = np.array(
                np.mean(
                    AnnData[AnnData.obs["louvain"].isin([str(cluster)])].raw.X, axis=0
                )
            )
            cluster_medians[cluster] = np.array(
                np.median(
                    AnnData[AnnData.obs["louvain"].isin([str(cluster)])].raw.X, axis=0
                )
            )
            cluster_stdev[cluster] = np.array(
                np.std(
                    AnnData[AnnData.obs["louvain"].isin([str(cluster)])].raw.X, axis=0
                )
            )
    AnnData.layers["Cluster_Medians"] = np.array(
        cluster_medians[AnnData.obs["louvain"].astype(int)]
    )
    AnnData.layers["Cluster_Means"] = cluster_means[AnnData.obs["louvain"].astype(int)]
    AnnData.layers["Cluster_Stdevs"] = cluster_stdev[AnnData.obs["louvain"].astype(int)]


def cluster_wilcoxon_rank_sum(AnnData, feature_list, alternative="greater"):
    cluster_list = AnnData.obs["louvain"]
    p_values = np.zeros((len(np.unique(cluster_list)), len(feature_list)))
    for cluster in range(len(np.unique(cluster_list))):
        for feature in range(len(feature_list)):
            p_values[cluster, feature] = stats.mannwhitneyu(
                AnnData[cluster_list.isin([str(cluster)])].obs_vector(
                    feature_list[feature]
                ),
                AnnData.obs_vector(feature_list[feature]),
                alternative="greater",
                use_continuity=True,
            )[1]
    AnnData.uns["Cluster_p_values"] = pd.DataFrame(
        p_values, np.arange(len(np.unique(cluster_list))), feature_list
    )


def cluster_p_threshold(AnnData, threshold=0.05):
    for columns in AnnData.uns["Cluster_p_values"]:
        AnnData.obs[columns + "_threshold"] = (
            (AnnData.uns["Cluster_p_values"] < threshold)[columns][
                AnnData.obs["louvain"].astype(int)
            ]
            .astype(int)
            .values
        )
        AnnData.obs[columns + "_enrichment"] = (AnnData.uns["Cluster_p_values"])[
            columns
        ][AnnData.obs["louvain"].astype(int)].values


# machine learning evaluation functions
def kfold_split(data, labels, n_splits, seed=None, shuffle=True):
    """
        split obs using k-fold strategy to cross-validate
            returns: dictionary with keys ['train','test'], which each contain a dictionary with keys ['data','labels'].
                values for ['data','labels'] are list of matrices/vectors
            ex: train data for the 3rd split can be indexed by `split['train']['data'][2]`,
                and its corresponding labels by `split['train']['labels'][2]`
        """
    kf = KFold(
        n_splits=n_splits, shuffle=shuffle, random_state=seed
    )  # generate KFold object for splitting data
    splits = {
        "train": {"data": [], "labels": []},
        "test": {"data": [], "labels": []},
    }  # initiate empty dictionary to dump matrix subsets into

    for train_i, test_i in kf.split(data):
        splits["train"]["data"].append(data[train_i, :])
        splits["train"]["labels"].append(labels[train_i])
        splits["test"]["data"].append(data[test_i, :])
        splits["test"]["labels"].append(labels[test_i])

    return splits


def validator(splits, classifier):
    """loops through kfold_split object and calculates confusion matrix and accuracy scores for given classifier"""
    for split in range(0, len(splits["train"]["data"])):
        classifier.fit(splits["train"]["data"][split], splits["train"]["labels"][split])
        prediction = classifier.predict(splits["test"]["data"][split])
        conf_matrix = confusion_matrix(splits["test"]["labels"][split], prediction)
        score = classifier.score(
            splits["test"]["data"][split], splits["test"]["labels"][split]
        )

        print("\nSplit {}: {}\n{}".format(split, score, conf_matrix))


def plot_cm(cm):
    """plot confusion matrix using seaborn for pretty output"""
    plt.figure(figsize=(3, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        square=True,
        cmap="Blues_r",
        cbar=False,
        annot_kws={"fontsize": 18},
    )
    plt.ylabel("Actual Label", fontsize=14)
    plt.xlabel("Predicted label", fontsize=14)
    plt.tick_params(axis="both", which="major", labelsize=14)
    score = cm.diagonal().sum() / cm.sum()
    plt.title("Accuracy: {0} %".format(np.round(score * 100, 2)), size=14)
    plt.show()
    plt.close()


def cm_metrics(cm, pretty_print=False):
    """calculate common metrics based on confusion matrix (e.g. accuracy, precision, sensitivity, specificity)"""
    assert cm.shape == (2, 2), "Confusion matrix must be 2 x 2."

    acc = cm.diagonal().sum() / cm.sum()
    prec = cm[1, 1] / cm[:, 1].sum()
    sens = cm[1, 1] / cm[1, :].sum()
    spec = cm[0, 0] / cm[0, :].sum()

    if pretty_print:
        print(
            "Accuracy: {}\nPrecision: {}\nSensitivity: {}\nSpecificity: {}".format(
                acc, prec, sens, spec
            )
        )

    return acc, prec, sens, spec


def roc_kfold(clf, X, y, k, seed=None):
    """Run binary classifier with cross-validation and plot ROC curves"""
    tprs = []
    aucs = []
    cm = np.zeros((len(y.unique()), len(y.unique())))
    out = {"acc": [], "prec": [], "sens": [], "spec": []}
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    i = 0
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for train, test in cv.split(X, y):
        clf.fit(X[train], y[train])
        prediction = clf.predict(X[test])
        conf_matrix = confusion_matrix(y[test], prediction)
        # print metrics to the console
        acc, prec, sens, spec = cm_metrics(conf_matrix, pretty_print=False)
        # append to outputs
        out["acc"].append(acc)
        out["prec"].append(prec)
        out["sens"].append(sens)
        out["spec"].append(spec)
        cm = cm + conf_matrix

        probas_ = clf.predict_proba(X[test])
        # compute ROC curve and area the curve
        fpr, tpr, _ = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(
            fpr, tpr, lw=1, alpha=0.5, label="ROC fold %d (AUC = %0.2f)" % (i, roc_auc)
        )
        i += 1

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()
    plt.close()

    plot_cm(cm)
    return out


def multiclass_roc(clf, X_train, X_test, y_train, y_test, plot_out=True):
    """Run multiclass classifier and plot ROC curves"""
    # binarize output
    y_train = label_binarize(y_train, classes=y_train.unique())
    y_test = label_binarize(y_test, classes=y_test.unique())

    # get expected number of classes from labels
    n_classes = y_train.shape[1]

    # use onevsrest method
    clf = OneVsRestClassifier(clf)

    # determine ROC scores
    y_score = clf.fit(X_train, y_train).decision_function(X_test)

    # compute ROC curve and area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # compute micro-avg ROC curve and area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # compute macro-avg ROC curve and area
    # aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # interpolate all ROC curves
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # average and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    if plot_out:
        # plot all ROC curves
        plt.figure()
        plt.plot([0, 1], [0, 1], "k--", lw=3)
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-avg ROC (area = {0:0.2f})".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-avg ROC (area = {0:0.2f})".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=3,
                label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
            )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="best")
        plt.show()

    return roc_auc
