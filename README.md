# scRNAseqQC

Class and function definitions are contained in the [`QC.py`](QC.py) file  

Tutorial contained in [`QCVer6Demo.ipynb`](QCVer6Demo.ipynb)  

This scRNA-seq quality control pipeline requires python 3, numpy, pandas, scipy, and scanpy installed.  
From command line:  
```
pip install -r requirements.txt
```

## Under development
Currently we are still updating the interactive mode of this to enable it to run unsupervised.


## Requirements
To run this you will need to install:
* Docker: this is required to run the container
* Workflow engine: the code written in `cwl` will require something like `cwl-runner` or `toil`

You will also need a sample-by-gene comma-delimited matrix. Genes are expected to be in the columsn and samples by row.

## To run interactively
`python3 run-qc.py -f test_file.csv -q 30 -o test_output`

## To run via CWL
[add instructions here]


## This scRNA-seq quality control pipeline requires python 3, numpy, pandas, scipy, and scanpy installed.

`conda install seaborn scikit-learn statsmodels numba pytables`

`conda install -c conda-forge python-igraph louvain`
