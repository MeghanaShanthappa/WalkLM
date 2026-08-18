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

The goal is to demonstrate that WalkLM-style
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

This comparison asks whether WalkLM still helps once a real GNN already uses
graph structure. `GCN + raw features` and `GraphSAGE + raw features` use the
original node features with graph message passing. The WalkLM variants first
concatenate each node's raw features with its WalkLM embedding, then pass the
enriched features into the same GCN or GraphSAGE classifier.

In short, GCN/GraphSAGE read the graph numerically through message passing,
while WalkLM reads graph neighborhoods as text through attributed random walks.
The improvement from raw features to WalkLM + raw features suggests that the
textualized random-walk embeddings add complementary signal beyond standard GNN
aggregation.

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
full-dataset post-processing pipelines are complementary.

### `ogbn-products`, full official OGB split with SAGN/SLE-style training

This result uses the full `ogbn-products` graph with the official OGB
train/validation/test split:

- 2,449,029 nodes
- 123,718,280 undirected edges after preprocessing
- 196,615 train nodes
- 39,323 validation nodes
- 2,213,091 test nodes

The experiment uses saved WalkLM embeddings from `walklm.py`, concatenated with
the raw OGB node features:

```text
WalkLM embeddings: [2449029, 128]
Raw OGB features:  [2449029, 100]
Combined features: [2449029, 228]
```

The combined features were used with the external
[`SAGN_with_SLE`](https://github.com/skepsun/SAGN_with_SLE) implementation using
multi-stage SAGN training and Correct & Smooth post-processing.

| Method | Train Acc | Val Acc | Test Acc |
|---|---:|---:|---:|
| SAGN + WalkLM + raw, stage 3 before postprocess | 0.9425 | 0.9256 | 0.8481 |
| SAGN + WalkLM + raw + Correct & Smooth | 0.9780 | 0.9290 | 0.8517 |

This is a single-run result (`seed=0`), not a 10-run leaderboard submission.
Still, it shows that WalkLM features can improve a scalable leaderboard-style
graph model on the full official `ogbn-products` split and cross 85% test
accuracy in this run.

### `ogbn-products`, full official OGB split with LD-style LM features + WalkLM

This experiment adds WalkLM embeddings to the extracted `bert-base-uncased`
`products_sagn/hidden_state.pt` features released by the
[`LD`](https://github.com/MIRALab-USTC/LD) project, then trains the external
[`SAGN_with_SLE`](https://github.com/skepsun/SAGN_with_SLE) implementation with
multi-stage SAGN/SLE training and Correct & Smooth/SCR-style post-processing.

Feature stack:

```text
LD released hidden_state.pt: [2449029, 768]
WalkLM embeddings:           [2449029, 128]
Raw OGB features:            [2449029, 100]
Combined features:           [2449029, 996]
```

| Method | Train Acc | Val Acc | Test Acc |
|---|---:|---:|---:|
| LD-style + WalkLM + raw + SAGN/SLE, stage 3 before postprocess | 0.9569 | 0.9385 | 0.8716 |
| LD-style + WalkLM + raw + SAGN/SLE + Correct & Smooth/SCR | 0.9767 | 0.9383 | 0.8738 |

The table above is the best single-run result observed with `seed=0`.

A 5-seed run (`seed=0..4`) on the same official OGB split gives:

| Method | Train Acc | Val Acc | Test Acc |
|---|---:|---:|---:|
| LD-style + WalkLM + raw + SAGN/SLE, stage 3 before postprocess | 0.9570 ± 0.0001 | 0.9391 ± 0.0006 | 0.8713 ± 0.0008 |
| LD-style + WalkLM + raw + SAGN/SLE + Correct & Smooth/SCR | 0.9768 ± 0.0001 | 0.9383 ± 0.0001 | 0.8733 ± 0.0008 |
| LD-style + WalkLM + raw + SAGN/SLE + tuned Correct & Smooth/SCR | 0.9846 ± 0.0001 | 0.9397 ± 0.0002 | 0.8734 ± 0.0008 |

The tuned Correct & Smooth/SCR setting used:

```text
correction_alpha = 0.30
smoothing_alpha  = 0.65
scale            = 25
correction layers = 50
smoothing layers  = 50
adjacency          = DAD
```

These are leaderboard-style research results on the official split, but not yet
an official leaderboard submission because the run currently reports 5 seeds
rather than the full 10-run protocol.
