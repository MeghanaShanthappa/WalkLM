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

- `Cora` as a small smoke test.
- `ogbn-arxiv` as the meaningful text-attributed graph experiment.
- comparison of raw node features, WalkLM embeddings, and WalkLM + raw
  features.

## Install

```bash
python -m pip install -r requirements.txt
```

If using a fresh PyG checkout instead of an installed package, run with:

```bash
PYTHONPATH=/path/to/pytorch_geometric python -u walklm.py ...
```

## Run on `ogbn-arxiv`

Small 10k-node experiment:

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
  --use_raw_features | tee walklm_arxiv_10k.log
```

## Result from initial experiment

On a 10k-node `ogbn-arxiv` subset:

| Method | Test Accuracy |
|---|---:|
| Raw features + MLP | 0.5063 |
| WalkLM embeddings + MLP | 0.5145 |
| WalkLM + raw features + MLP | 0.5651 |

Additional graph baselines on the same subset:

| Method | Test Accuracy |
|---|---:|
| GCN | 0.5109 |
| GraphSAGE | 0.5045 |

This is not a SOTA claim. The goal is to demonstrate that WalkLM-style
attributed random-walk language modeling can provide complementary signal on a
real text-attributed graph.

