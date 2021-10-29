# Predicting lncRNA–protein interactions based on graph autoencoders and collaborative training

Code for our paper "Predicting lncRNA–protein interactions based on graph autoencoders and collaborative training" (IEEE BIBM 2021)

## Requirements

The code has been tested running under Python 3.7.4, with the following packages and their dependencies installed:

```
numpy==1.16.5
pytorch==1.3.1
sklearn==0.21.3
```

## Usage

```bash
git clone https://github.com/zhanglabNKU/LPIGAC.git
cd LPIGAC
python fivefoldcv.py
```

## Options

We adopt an argument parser by package  `argparse` in Python, and the options for running code are defined as follow:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-7,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=144,                    help='Dimension of representations')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Weight between lncRNA space and protein space')
parser.add_argument('--beta', type=float, default=1.0,
                    help='Hyperparameter beta')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
```

## Data

Files of data are listed as follow:

- `LncRNAName.txt`  includes the names of all lncRNAs.
- `ProteinName.txt`  includes the names of all proteins.
- `interaction.txt` is a matrix  `Y`  that shows lncRNA-protein associations. `Y[i,j]=1`  if lncRNA `i`  and protein `j` are known to be associated, otherwise 0.
- `protfeat.txt` is the feature matrix of proteins.
- `rnafeat.txt` is the feature matrix of lncRNAs.

## Citation

```
@inproceedings{jin2021lpigac,
    author = {Jin, Chen and Shi, Zhuangwei and Zhang, Han and Yin, Yanbin},
    title = {Predicting lncRNA–protein interactions based on graph autoencoders and collaborative training},
    year = {2021},
    booktitle = {IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
}
```