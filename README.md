# WalkLM-style Graph Language Modeling with PyG

This repository contains a compact WalkLM-style experiment built with
PyTorch Geometric and HuggingFace Transformers.

The pipeline is:

```text
text-attributed graph
→ attributed random walks
→ walk text corpus
→ masked language model fine-tuning
→ node embeddings
→ node classification
```

The example supports:

- `Cora`, `CiteSeer`, and `PubMed` as small Planetoid smoke tests.
- `ogbn-arxiv` and `ogbn-products` as meaningful text-attributed graph
  experiments.
- comparison of raw node features, WalkLM embeddings, and WalkLM + raw
  features.
- optional saving of WalkLM embeddings for downstream GNN experiments.

This is not a SOTA claim. The goal is to demonstrate that WalkLM-style
attributed random-walk language modeling can provide complementary signal on
real text-attributed graphs.

## Install

```bash
python -m pip install -r requirements.txt
```

For GPU PyG experiments that need compiled operators, install PyTorch and PyG
wheels that match your CUDA version. For example, on H100 with CUDA 12.8:

```bash
python -m pip install torch==2.8.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

## Run WalkLM on `ogbn-arxiv`

```bash
PYTHONPATH=. python -u walklm.py \
  --dataset ogbn-arxiv \
  --hf_model google/bert_uncased_L-2_H-128_A-2 \
  --max_nodes 10000 \
  --mlm_epochs 2 \
  --epochs 100 \
  --batch_size 16 \
  --num_walks 4 \
  --walk_length 4 \
  --max_text_chars 256 \
  --use_raw_features
```

## Run WalkLM on `ogbn-products`

```bash
PYTHONPATH=. python -u walklm.py \
  --dataset ogbn-products \
  --hf_model google/bert_uncased_L-2_H-128_A-2 \
  --max_nodes 100000 \
  --mlm_epochs 1 \
  --epochs 50 \
  --batch_size 32 \
  --num_walks 2 \
  --walk_length 3 \
  --max_text_chars 128 \
  --use_raw_features
```

To save embeddings for downstream GNN baselines, add:

```bash
--save_embeddings
```

Then run:

```bash
PYTHONPATH=. python -u products_walklm_gnn.py \
  --embedding_path walklm_products_100000_embeddings.pt
```

## Results

### `ogbn-arxiv`, 10k subset

| Method | Test Accuracy |
|---|---:|
| Node2Vec | 0.1643 |
| Raw features + MLP | 0.5063 |
| GCN | 0.5109 |
| GraphSAGE | 0.5045 |
| WalkLM embeddings + MLP | 0.5145 |
| WalkLM + raw features + MLP | 0.5651 |

### `ogbn-products` scaling

| Subset | Raw features + MLP | WalkLM embeddings + MLP | WalkLM + raw features + MLP |
|---:|---:|---:|---:|
| 10k | 0.5360 | 0.5855 | 0.6335 |
| 20k | 0.5480 | 0.6035 | 0.6462 |
| 50k | 0.5570 | 0.6166 | 0.6580 |
| 100k | 0.5566 | 0.6185 | 0.6666 |

### `ogbn-products`, 100k subset with GNN classifiers

| Method | Test Accuracy |
|---|---:|
| GCN + Raw features | 0.6493 |
| GCN + WalkLM embeddings | 0.6607 |
| GCN + WalkLM + raw features | 0.6999 |
| GraphSAGE + Raw features | 0.6325 |
| GraphSAGE + WalkLM embeddings | 0.6407 |
| GraphSAGE + WalkLM + raw features | 0.6756 |

### Existing PyG full-dataset reference

For context, PyG already includes full `ogbn-products` examples such as
`examples/ogbn_train.py`, `examples/rev_gnn.py`, and
`examples/correct_and_smooth.py`. In one full-dataset run of
`examples/correct_and_smooth.py`, the final result was:

| Example | Dataset | Test Accuracy |
|---|---|---:|
| `examples/correct_and_smooth.py` | full `ogbn-products` | 0.8377 |

This full-dataset result is not directly comparable to the subset WalkLM
experiments, but it is useful context for future work that combines WalkLM
embeddings with full PyG training pipelines.
