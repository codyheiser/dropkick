import pandas as pd
import numpy as np
import scanpy as sc
# sklearn tools
from sklearn.preprocessing import normalize, label_binarize
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
# load sklearn classifiers
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
# PU bagging method adapted from sklearn
from PU_bagging import BaggingClassifierPU
# custom PU two-step method
from PU_twostep import twoStep
# plotting tools
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'white')



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


def reorder_AnnData(AnnData, descending = True):
    AnnData.obs['total_counts'] = AnnData.X.sum(axis=1)
    if(descending==True):
        new_order = np.argsort(AnnData.obs['total_counts'])[::-1]
    elif(descending==False):
        new_order = np.argsort(AnnData.obs['total_counts'])[:]
    AnnData.X = AnnData.X[new_order,:].copy()
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

    # number of cells containing each gene (sum nonzero along columns)
    nt = m.astype(bool).sum(axis=0)
    assert np.all(
        nt
    ), "Encountered {} genes with 0 cells by counts. Remove these before proceeding (i.e. sc.pp.filter_genes(adata,min_cells=1))".format(
        np.size(nt) - np.count_nonzero(nt)
    )
    # gene frequency in each cell (l1 norm along rows)
    tf = m / m.sum(axis=1)[:, None]

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
        n_hvgs (int): number of HVGs to calculate using Seurat method

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
        - highly variable genes (adata.var["highly_variable"])
    """
    # reorder cells by total counts descending
    reorder_adata(adata, descending=True)

    # store raw counts before manipulation
    adata.layers["raw_counts"] = adata.X.copy()

    # identify mitochondrial genes
    adata.var["mito"] = adata.var_names.str.contains(mito_names)
    # calculate standard qc .obs and .var
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mito"], inplace=True)
    # rank cells by total counts
    adata.obs["ranked_total_counts"] = np.argsort(adata.obs["total_counts"])

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
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, n_bins=20, flavor="seurat")

    # set .X as desired for downstream processing; default raw_counts
    adata.X = adata.layers[X_final].copy()


def find_inflection(ann_data, inflection_percentiles = [0,15,30,100], output_prefix='Output'):
    ann_data_cumsum = np.cumsum(ann_data.obs['total_counts'])
    x_vals=np.arange(0,ann_data.obs.shape[0])
    secant_coef=ann_data_cumsum[ann_data.obs.shape[0]-1]/ann_data.obs.shape[0]
    secant_line=secant_coef*x_vals
    secant_dist=abs(ann_data_cumsum-secant_line)
    inflection_percentiles_inds = np.percentile(x_vals,inflection_percentiles).astype(int)
    inflection_points = secant_dist.argsort()[::-1]
    percentile_points = inflection_points[inflection_percentiles_inds]
    color=plt.cm.tab10(np.linspace(0,1,ann_data.obs.shape[0]))
    plt.figure(figsize=(20,10))
    plt.plot(np.array(ann_data_cumsum), label="Cumulative Sum")
    #plt.plot(np.array(secant_line), label="Secant Line")
    plt.plot(np.array(secant_dist), label="Secant Distance")
    for percentile in percentile_points:
        plt.axvline(x=percentile,ymin=0,c=color[percentile],linestyle='--',linewidth=2,label="Inflection point {}".format(percentile))
    plt.legend()
    #save to file
    if(output_prefix!=''):
        plt.savefig(output_prefix+'_inflectionCheck.png',bbox_inches='tight')
    else:
        plt.show()
    print("Inflection point at {} for {} percentiles of greatest secant distances".format(percentile_points,inflection_percentiles))
    #SJCG: added the ability to return a dictionary of points
    return(dict(zip(inflection_percentiles, percentile_points)))
    
def reorder_AnnData(AnnData, descending = True):
    AnnData.obs['n_counts'] = AnnData.X.sum(axis=1)
    if(descending==True):
        AnnData = AnnData[np.argsort(AnnData.obs['n_counts'])[::-1]].copy()
    elif(descending==False):
        AnnData = AnnData[np.argsort(AnnData.obs['n_counts'])[:]].copy()
    return(AnnData)
    
def arcsinh_transform(AnnData, cofactor = 1000):
    AnnData.X = np.arcsinh(AnnData.X*cofactor,dtype='float')

def cluster_summary_stats(AnnData,raw=False):
    cluster_means = np.zeros((len(np.unique(AnnData.obs['louvain'])),AnnData.n_vars))
    cluster_medians = np.zeros((len(np.unique(AnnData.obs['louvain'])),AnnData.n_vars))
    cluster_stdev = np.zeros((len(np.unique(AnnData.obs['louvain'])),AnnData.n_vars))
    if(raw == True):
        for cluster in range(len(np.unique(AnnData.obs['louvain']))):
            cluster_means[cluster]=np.array(np.mean(AnnData[AnnData.obs['louvain'].isin([str(cluster)])].X,axis = 0))
            cluster_medians[cluster]=np.array(np.median(AnnData[AnnData.obs['louvain'].isin([str(cluster)])].X,axis = 0))
            cluster_stdev[cluster]=np.array(np.std(AnnData[AnnData.obs['louvain'].isin([str(cluster)])].X,axis = 0))
    elif(raw == False):    
        for cluster in range(len(np.unique(AnnData.obs['louvain']))):
            cluster_means[cluster]=np.array(np.mean(AnnData[AnnData.obs['louvain'].isin([str(cluster)])].raw.X,axis = 0))
            cluster_medians[cluster]=np.array(np.median(AnnData[AnnData.obs['louvain'].isin([str(cluster)])].raw.X,axis = 0))
            cluster_stdev[cluster]=np.array(np.std(AnnData[AnnData.obs['louvain'].isin([str(cluster)])].raw.X,axis = 0))
    AnnData.layers['Cluster_Medians'] = np.array(cluster_medians[AnnData.obs['louvain'].astype(int)])
    AnnData.layers['Cluster_Means'] = cluster_means[AnnData.obs['louvain'].astype(int)]
    AnnData.layers['Cluster_Stdevs'] = cluster_stdev[AnnData.obs['louvain'].astype(int)]

def cluster_wilcoxon_rank_sum(AnnData,feature_list,alternative='greater'):
    cluster_list = AnnData.obs['louvain']
    p_values = np.zeros((len(np.unique(cluster_list)),len(feature_list)))
    for cluster in range(len(np.unique(cluster_list))):
        for feature in range(len(feature_list)):
            p_values[cluster,feature]=stats.mannwhitneyu(AnnData[cluster_list.isin([str(cluster)])].obs_vector(feature_list[feature]),AnnData.obs_vector(feature_list[feature]),alternative='greater',use_continuity=True)[1]
    AnnData.uns['Cluster_p_values'] = pd.DataFrame(p_values,np.arange(len(np.unique(cluster_list))),feature_list)

def cluster_p_threshold(AnnData,threshold = 0.05):
    for columns in AnnData.uns['Cluster_p_values']:
        AnnData.obs[columns+'_threshold'] = (AnnData.uns['Cluster_p_values']<threshold)[columns][AnnData.obs['louvain'].astype(int)].astype(int).values
        AnnData.obs[columns+'_enrichment'] = (AnnData.uns['Cluster_p_values'])[columns][AnnData.obs['louvain'].astype(int)].values
