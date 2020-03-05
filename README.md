# dropkick
Automated cell filtering for single-cell RNA sequencing data.

`dropkick` works primarily with [**Scanpy**](https://icb-scanpy.readthedocs-hosted.com/en/stable/)'s `AnnData` objects, and accepts input files in `.h5ad` or flat (`.csv`, `.tsv`) format. It also writes outputs to `.h5ad` files when called from the command line.

## Requirements
From command line:  
```bash
pip install -r requirements.txt
```

You will need an unfiltered barcode-by-gene counts matrix in `.h5ad` (`scanpy.anndata`) format.

## Usage
From command line:
```bash
python dropkick.py regression -c <path/to/.h5ad>
```

Output will be saved in a new `.h5ad` file containing __dropkick__ scores, labels, and model parameters.

See [`dropkick_tutorial.ipynb`](dropkick_tutorial.ipynb) for an interactive walkthrough of the `dropkick` pipeline and its outputs.
