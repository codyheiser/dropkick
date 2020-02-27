# dropkick
Filtering tool for single-cell RNA sequencing data using logistic regression models.

## Requirements
From command line:  
```
pip install -r requirements.txt
```

You will need an unfiltered barcode-by-gene counts matrix in `.h5ad` (`scanpy.anndata`) format.

## Usage
```bash
python dropkick.py regression -c <path/to/.h5ad>
```

Output will be saved in a new `.h5ad` file containing __dropkick__ scores, labels, and model parameters.