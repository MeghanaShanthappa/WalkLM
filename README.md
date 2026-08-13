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


## What each script does

### `walklm.py`

This is the main WalkLM-style experiment. It loads a graph dataset, samples
attributed random walks, converts those walks into text, fine-tunes a small
masked language model, extracts node embeddings, and evaluates them with an
MLP classifier. It compares:

- raw node features,
- WalkLM embeddings,
- WalkLM embeddings concatenated with raw node features.

For `ogbn-arxiv` and `ogbn-products`, it uses real node text. With
`--save_embeddings`, it also saves the learned WalkLM node embeddings for
downstream experiments.

### `products_walklm_gnn.py`

This is the follow-up GNN baseline script for `ogbn-products`. It does not
train the language model again. Instead, it loads saved embeddings from
`walklm.py` and checks whether those embeddings improve GNN classifiers. It
compares:

- GCN with raw features, WalkLM embeddings, and WalkLM + raw features,
- GraphSAGE with raw features, WalkLM embeddings, and WalkLM + raw features.

In short: `walklm.py` creates and evaluates WalkLM embeddings with an MLP, while
`products_walklm_gnn.py` tests whether those embeddings also help GCN and
GraphSAGE baselines.

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

These results come from `walklm.py`, with Node2Vec/GCN/GraphSAGE run as
separate same-subset baselines.

| Method | Test Accuracy |
|---|---:|
| Node2Vec | 0.1643 |
| Raw features + MLP | 0.5063 |
| GCN | 0.5109 |
| GraphSAGE | 0.5045 |
| WalkLM embeddings + MLP | 0.5145 |
| WalkLM + raw features + MLP | 0.5651 |

### `ogbn-products` scaling

These MLP-based results come from `walklm.py`.

| Subset | Raw features + MLP | WalkLM embeddings + MLP | WalkLM + raw features + MLP |
|---:|---:|---:|---:|
| 10k | 0.5360 | 0.5855 | 0.6335 |
| 20k | 0.5480 | 0.6035 | 0.6462 |
| 50k | 0.5570 | 0.6166 | 0.6580 |
| 100k | 0.5566 | 0.6185 | 0.6666 |

### `ogbn-products`, 100k subset with GNN classifiers

These results come from `products_walklm_gnn.py`, using embeddings saved by
`walklm.py --save_embeddings`.

| Method | Test Accuracy |
|---|---:|
| GCN + Raw features | 0.6493 |
| GCN + WalkLM embeddings | 0.6607 |
| GCN + WalkLM + raw features | 0.6999 |
| GraphSAGE + Raw features | 0.6325 |
| GraphSAGE + WalkLM embeddings | 0.6407 |
| GraphSAGE + WalkLM + raw features | 0.6756 |

### `ogbn-products`, 500k custom random split with Correct & Smooth

These exploratory results use a larger custom random 500k split
(300k train / 100k validation / 100k test). They are not directly comparable to
the smaller official-split subset tables above, but they test whether WalkLM
features still help in a stronger Correct & Smooth pipeline.

| Method | Test Accuracy |
|---|---:|
| MLP + Raw features + Correct & Smooth | 0.8625 |
| MLP + WalkLM + raw features + Correct & Smooth | 0.8750 |
| GCN + Raw features + Correct & Smooth | 0.8584 |
| GCN + WalkLM + raw features + Correct & Smooth | 0.8701 |
| GraphSAGE + Raw features + Correct & Smooth | 0.8487 |
| GraphSAGE + WalkLM + raw features + Correct & Smooth | 0.8677 |

All reported comparisons above use the same subset/split family and focus on
raw-feature MLP, GCN, GraphSAGE, and WalkLM-based variants. Stronger
full-dataset post-processing pipelines are complementary and left for future
work.
