# Initial results

Dataset: `ogbn-arxiv`

Subset: first 10,000 nodes induced subgraph

Language model: `google/bert_uncased_L-2_H-128_A-2`

WalkLM command:

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

| Method | Test Accuracy |
|---|---:|
| Raw features + MLP | 0.5063 |
| WalkLM embeddings + MLP | 0.5145 |
| WalkLM + raw features + MLP | 0.5651 |
| GCN | 0.5109 |
| GraphSAGE | 0.5045 |

Takeaway: WalkLM embeddings alone slightly outperform raw features in this
subset experiment, and combining WalkLM embeddings with raw features gives the
best result.

