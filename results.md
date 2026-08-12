# Results

All WalkLM subset experiments use a controlled split sampled from the official
OGB train/validation/test partitions. These are lightweight experiments for
method exploration, not leaderboard submissions.

## `ogbn-arxiv`, 10k subset

| Method | Test Accuracy |
|---|---:|
| Node2Vec | 0.1643 |
| Raw features + MLP | 0.5063 |
| GCN | 0.5109 |
| GraphSAGE | 0.5045 |
| WalkLM embeddings + MLP | 0.5145 |
| WalkLM + raw features + MLP | 0.5651 |

## `ogbn-products` scaling

| Subset | Raw features + MLP | WalkLM embeddings + MLP | WalkLM + raw features + MLP |
|---:|---:|---:|---:|
| 10k | 0.5360 | 0.5855 | 0.6335 |
| 20k | 0.5480 | 0.6035 | 0.6462 |
| 50k | 0.5570 | 0.6166 | 0.6580 |
| 100k | 0.5566 | 0.6185 | 0.6666 |

Takeaway: WalkLM embeddings consistently outperform raw features in the MLP
setting, and concatenating WalkLM embeddings with raw features is best at every
subset size tested.

## `ogbn-products`, 100k subset with GNN classifiers

| Method | Test Accuracy |
|---|---:|
| GCN + Raw features | 0.6493 |
| GCN + WalkLM embeddings | 0.6607 |
| GCN + WalkLM + raw features | 0.6999 |
| GraphSAGE + Raw features | 0.6325 |
| GraphSAGE + WalkLM embeddings | 0.6407 |
| GraphSAGE + WalkLM + raw features | 0.6756 |

Takeaway: WalkLM also helps when used as input features for GNN classifiers.
The best 100k-subset result observed was `GCN + WalkLM + raw features` at
`0.6999`.

## `ogbn-products`, 500k custom random split with Correct & Smooth

These exploratory results use a larger custom random 500k split
(300k train / 100k validation / 100k test). They are not directly comparable to
the smaller official-split subset tables above.

| Method | Test Accuracy |
|---|---:|
| MLP + Raw features + Correct & Smooth | 0.8625 |
| GCN + Raw features + Correct & Smooth | 0.8580 |
| MLP + WalkLM + raw features + Correct & Smooth | 0.8746 |
| GCN + WalkLM + raw features + Correct & Smooth | 0.8703 |

Takeaway: WalkLM features improve both MLP and GCN Correct & Smooth pipelines
on this larger exploratory split.

## Scope

The tables above intentionally compare against same-subset MLP, GCN, and
GraphSAGE baselines. Full-dataset post-processing methods are complementary and
are left for future work.
