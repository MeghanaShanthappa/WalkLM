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

## `ogbn-products`, 100k subset with GNN classifiers

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
| MLP + WalkLM + raw features + Correct & Smooth | 0.8750 |
| GCN + Raw features + Correct & Smooth | 0.8584 |
| GCN + WalkLM + raw features + Correct & Smooth | 0.8701 |
| GraphSAGE + Raw features + Correct & Smooth | 0.8487 |
| GraphSAGE + WalkLM + raw features + Correct & Smooth | 0.8677 |

Takeaway: WalkLM features improve MLP, GCN, and GraphSAGE Correct &
Smooth pipelines on this larger exploratory split.

## `ogbn-products`, full official OGB split with SAGN/SLE-style training

This full-scale experiment uses saved WalkLM embeddings concatenated with raw
OGB features, then trains the external `SAGN_with_SLE` implementation with
multi-stage SAGN training and Correct & Smooth post-processing.

Dataset/split:

- Nodes: 2,449,029
- Undirected edges after preprocessing: 123,718,280
- Train/validation/test: 196,615 / 39,323 / 2,213,091
- Feature shape after concatenation: `[2449029, 228]`

| Method | Train Acc | Val Acc | Test Acc |
|---|---:|---:|---:|
| SAGN + WalkLM + raw, stage 3 before postprocess | 0.9425 | 0.9256 | 0.8481 |
| SAGN + WalkLM + raw + Correct & Smooth | 0.9780 | 0.9290 | 0.8517 |

This is a single-run result (`seed=0`), not a 10-run leaderboard submission.

## `ogbn-products`, full official OGB split with LD-style LM features + WalkLM

This experiment concatenates three feature sources:

- released `bert-base-uncased/products_sagn/hidden_state.pt` features from the
  external [`LD`](https://github.com/MIRALab-USTC/LD) project,
- WalkLM embeddings generated in this repository,
- raw OGB node features.

Feature shapes:

| Feature source | Shape |
|---|---:|
| LD released hidden states | `[2449029, 768]` |
| WalkLM embeddings | `[2449029, 128]` |
| Raw OGB features | `[2449029, 100]` |
| Combined features | `[2449029, 996]` |

The combined features are then used with the external
[`SAGN_with_SLE`](https://github.com/skepsun/SAGN_with_SLE) implementation and
Correct & Smooth/SCR-style post-processing.

| Method | Train Acc | Val Acc | Test Acc |
|---|---:|---:|---:|
| LD-style + WalkLM + raw + SAGN/SLE, stage 3 before postprocess | 0.9569 | 0.9385 | 0.8716 |
| LD-style + WalkLM + raw + SAGN/SLE + Correct & Smooth/SCR | 0.9767 | 0.9383 | 0.8738 |

This is a single-run result (`seed=0`), not a 10-run leaderboard submission.
It is best described as a leaderboard-style research result until repeated
across additional seeds.

## Scope

The tables above include both subset experiments and one full official-split
SAGN/SLE-style experiment. The full official-split result should be treated as a
single-run research result until repeated across more seeds.
