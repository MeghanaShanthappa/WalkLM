"""WalkLM-style attributed random-walk language modeling example.

This example demonstrates a compact WalkLM-style pipeline:

1. Load an attributed graph.
2. Sample random walks from each node.
3. Convert each walk into a text sequence.
4. Fine-tune a masked language model on the walk corpus.
5. Extract node embeddings from the fine-tuned language model.
6. Compare raw features, WalkLM embeddings, and their concatenation for node
   classification.

The Cora path is a lightweight smoke test. The ``ogbn-arxiv`` and
``ogbn-products`` paths use real text and are the more meaningful
text-attributed graph experiments.

Requirements on top of basic PyG:

    pip install ogb pandas transformers

Example:

    python examples/llm/walklm.py --dataset ogbn-arxiv --max_nodes 10000 \
        --hf_model google/bert_uncased_L-2_H-128_A-2 --use_raw_features
"""

import argparse
import os.path as osp
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torch_geometric import seed_everything
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph

# TODO: Remove once OGB supports PyTorch >= 2.6 weights_only defaults.
_original_torch_load = torch.load


def _torch_load_with_weights_only_false(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_with_weights_only_false


class TextDataset(Dataset):
    def __init__(self, encodings: Dict[str, Tensor]) -> None:
        self.encodings = encodings

    def __len__(self) -> int:
        return self.encodings['input_ids'].size(0)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {key: value[idx] for key, value in self.encodings.items()}


def build_adj_list(edge_index: Tensor, num_nodes: int) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    row, col = edge_index.cpu()
    for src, dst in zip(row.tolist(), col.tolist()):
        adj[src].append(dst)
    return adj


def sample_walk(
    start: int,
    adj: List[List[int]],
    walk_length: int,
    generator: torch.Generator,
) -> List[int]:
    walk = [start]
    current = start
    for _ in range(walk_length - 1):
        neighbors = adj[current]
        if len(neighbors) == 0:
            break
        idx = int(torch.randint(len(neighbors), (1, ), generator=generator))
        current = neighbors[idx]
        walk.append(current)
    return walk


def textualize_node(x: Tensor, node_idx: int, top_k: int) -> str:
    values = x[node_idx]
    _, indices = values.topk(min(top_k, values.numel()))
    features = [f'feature_{int(idx)}' for idx in indices if values[idx] > 0]
    if len(features) == 0:
        features = ['no_active_features']
    return 'paper has topics ' + ', '.join(features)


def textualize_walk(x: Tensor, walk: List[int], top_k: int) -> str:
    return ' cites '.join(
        textualize_node(x, node_idx, top_k) for node_idx in walk)


def build_walk_corpus(
    x: Tensor,
    edge_index: Tensor,
    num_walks: int,
    walk_length: int,
    top_k: int,
    seed: int,
) -> List[str]:
    generator = torch.Generator().manual_seed(seed)
    adj = build_adj_list(edge_index, x.size(0))

    corpus: List[str] = []
    for node_idx in range(x.size(0)):
        walks = [
            sample_walk(node_idx, adj, walk_length, generator)
            for _ in range(num_walks)
        ]
        corpus.append(' '.join(
            textualize_walk(x, walk, top_k=top_k) for walk in walks))
    return corpus


def build_text_walk_corpus(
    text: List[str],
    edge_index: Tensor,
    num_walks: int,
    walk_length: int,
    seed: int,
    max_text_chars: int,
) -> List[str]:
    generator = torch.Generator().manual_seed(seed)
    adj = build_adj_list(edge_index, len(text))

    corpus: List[str] = []
    for node_idx in range(len(text)):
        walks = [
            sample_walk(node_idx, adj, walk_length, generator)
            for _ in range(num_walks)
        ]
        walk_texts = []
        for walk in walks:
            walk_texts.append(' cites '.join(
                text[node_id][:max_text_chars].replace('\n', ' ')
                for node_id in walk))
        corpus.append(' '.join(walk_texts))
    return corpus


def tokenize_corpus(corpus: List[str], tokenizer, max_length: int):
    return tokenizer(
        corpus,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )


def train_language_model(
    model,
    tokenizer,
    encodings: Dict[str, Tensor],
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    from transformers import DataCollatorForLanguageModeling

    dataset = TextDataset(encodings)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        print(f'MLM Epoch: {epoch:03d}, Loss: {total_loss / len(loader):.4f}')


@torch.no_grad()
def mean_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    out = (hidden * mask).sum(dim=1)
    return out / mask.sum(dim=1).clamp(min=1e-9)


@torch.no_grad()
def encode_nodes(
    model,
    encodings: Dict[str, Tensor],
    batch_size: int,
    device: torch.device,
) -> Tensor:
    dataset = TextDataset(encodings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    xs: List[Tensor] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        out = model(**batch, output_hidden_states=True)
        x = mean_pool(out.hidden_states[-1], batch['attention_mask'])
        xs.append(x.cpu())

    return torch.cat(xs, dim=0)


class Classifier(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
    ):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


def train_classifier(
    model: Classifier,
    x: Tensor,
    y: Tensor,
    train_mask: Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    optimizer.zero_grad()
    out = model(x)
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def test(
    model: Classifier,
    x: Tensor,
    y: Tensor,
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
) -> List[float]:
    model.eval()
    pred = model(x).argmax(dim=-1)

    accs = []
    for mask in [train_mask, val_mask, test_mask]:
        acc = int((pred[mask] == y[mask]).sum()) / int(mask.sum())
        accs.append(acc)
    return accs


def run_classifier(
    name: str,
    x: Tensor,
    y: Tensor,
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
    num_classes: int,
    hidden_channels: int,
    dropout: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> float:
    x = x.to(device)
    model = Classifier(
        x.size(-1),
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_acc = final_test_acc = 0.0
    print(f'\nTraining classifier on {name}...')
    for epoch in range(1, epochs + 1):
        loss = train_classifier(model, x, y, train_mask, optimizer)
        train_acc, val_acc, test_acc = test(
            model, x, y, train_mask, val_mask, test_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc

        print(f'{name} | Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
              f'Test: {test_acc:.4f}')

    print(f'{name} Final Test: {final_test_acc:.4f}')
    return final_test_acc


def load_dataset(args) -> Tuple[object, object, Optional[List[str]]]:
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data',
                        'Planetoid')
        dataset = Planetoid(path, name=args.dataset)
        data = dataset[0]
        return dataset, data, None

    if args.dataset in ['ogbn-arxiv', 'ogbn-products']:
        from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
        from pandas import read_csv
        from torch_geometric.data import download_google_url

        root = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ogb')
        dataset = PygNodePropPredDataset(name=args.dataset, root=root)
        data = dataset[0]

        raw_text_id = {
            'ogbn-arxiv': '1g3OOVhRyiyKv13LY6gbp8GLITocOUr_3',
            'ogbn-products': '1I-S176-W4Bm1iPDjQv3hYwQBtxE0v8mt',
        }[args.dataset]
        raw_dir_name = args.dataset.replace('-', '_')

        raw_text_path = download_google_url(
            id=raw_text_id,
            folder=osp.join(root, raw_dir_name, 'raw'),
            filename='node-text.csv.gz',
            log=True,
        )
        text = list(read_csv(raw_text_path)['text'])

        split_idx = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[split_idx['train']] = True
        data.val_mask[split_idx['valid']] = True
        data.test_mask[split_idx['test']] = True
        data.y = data.y.view(-1)
        return dataset, data, text

    raise ValueError(f'Unsupported dataset: {args.dataset}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=[
                            'Cora',
                            'CiteSeer',
                            'PubMed',
                            'ogbn-arxiv',
                            'ogbn-products',
                        ])
    parser.add_argument('--hf_model', type=str,
                        default='google/bert_uncased_L-2_H-128_A-2')
    parser.add_argument('--num_walks', type=int, default=8)
    parser.add_argument('--walk_length', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mlm_epochs', type=int, default=1)
    parser.add_argument('--mlm_lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_raw_features', action='store_true')
    parser.add_argument('--max_nodes', type=int, default=None)
    parser.add_argument('--max_text_chars', type=int, default=256)
    parser.add_argument('--save_embeddings', action='store_true')
    args = parser.parse_args()

    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            'This example requires HuggingFace Transformers. Please install '
            'it via `pip install transformers`.') from e

    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, data, text = load_dataset(args)

    if args.max_nodes is not None and data.num_nodes > args.max_nodes:
        if args.dataset in ['ogbn-arxiv', 'ogbn-products']:
            split_idx = dataset.get_idx_split()

            num_train = int(args.max_nodes * 0.6)
            num_val = int(args.max_nodes * 0.2)
            num_test = args.max_nodes - num_train - num_val

            train_nodes = split_idx['train'][:num_train]
            val_nodes = split_idx['valid'][:num_val]
            test_nodes = split_idx['test'][:num_test]

            node_idx = torch.cat([train_nodes, val_nodes, test_nodes], dim=0)
            edge_index, _ = subgraph(
                node_idx,
                data.edge_index,
                relabel_nodes=True,
            )

            data.x = data.x[node_idx]
            data.y = data.y[node_idx].view(-1)
            data.edge_index = edge_index
            data.num_nodes = node_idx.numel()

            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

            data.train_mask[:num_train] = True
            data.val_mask[num_train:num_train + num_val] = True
            data.test_mask[num_train + num_val:] = True

            if text is not None:
                text = [text[int(i)] for i in node_idx]
        else:
            node_idx = torch.arange(args.max_nodes)
            data = data.subgraph(node_idx)
            if text is not None:
                text = text[:args.max_nodes]

    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    print('Device:', device)
    print('Nodes:', data.num_nodes)
    print('Edges:', data.edge_index.size(1))
    print('Train/Val/Test:', int(train_mask.sum()), int(val_mask.sum()),
          int(test_mask.sum()))

    raw_test = run_classifier(
        'Raw features',
        data.x,
        y,
        train_mask,
        val_mask,
        test_mask,
        dataset.num_classes,
        args.hidden_channels,
        args.dropout,
        args.epochs,
        args.lr,
        args.weight_decay,
        device,
    )

    print('\nBuilding attributed random-walk corpus...')
    if text is None:
        corpus = build_walk_corpus(
            data.x,
            data.edge_index,
            num_walks=args.num_walks,
            walk_length=args.walk_length,
            top_k=args.top_k,
            seed=args.seed,
        )
    else:
        corpus = build_text_walk_corpus(
            text,
            data.edge_index,
            num_walks=args.num_walks,
            walk_length=args.walk_length,
            seed=args.seed,
            max_text_chars=args.max_text_chars,
        )
    print(f'Example walk text: {corpus[0][:500]}...')

    print(f'\nLoading masked language model: {args.hf_model}')
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    model = AutoModelForMaskedLM.from_pretrained(
        args.hf_model,
        use_safetensors=True,
    ).to(device)

    encodings = tokenize_corpus(
        corpus,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    print('Fine-tuning masked language model on walk corpus...')
    train_language_model(
        model,
        tokenizer,
        encodings,
        batch_size=args.batch_size,
        epochs=args.mlm_epochs,
        lr=args.mlm_lr,
        device=device,
    )

    print('Encoding nodes with fine-tuned language model...')
    walklm_x = encode_nodes(
        model,
        encodings,
        batch_size=args.batch_size,
        device=device,
    )

    if args.save_embeddings and args.dataset == 'ogbn-products':
        save_path = f'walklm_products_{data.num_nodes}_embeddings.pt'
        torch.save({
            'walklm_x': walklm_x.cpu(),
            'raw_x': data.x.cpu(),
        }, save_path)
        print(f'Saved WalkLM embeddings to {save_path}')

    walklm_test = run_classifier(
        'WalkLM embeddings',
        walklm_x,
        y,
        train_mask,
        val_mask,
        test_mask,
        dataset.num_classes,
        args.hidden_channels,
        args.dropout,
        args.epochs,
        args.lr,
        args.weight_decay,
        device,
    )

    combined_test = None
    if args.use_raw_features:
        combined_x = torch.cat([walklm_x, data.x.cpu()], dim=-1)
        combined_test = run_classifier(
            'WalkLM + raw features',
            combined_x,
            y,
            train_mask,
            val_mask,
            test_mask,
            dataset.num_classes,
            args.hidden_channels,
            args.dropout,
            args.epochs,
            args.lr,
            args.weight_decay,
            device,
        )

    print('\nSummary:')
    print(f'Raw features Final Test: {raw_test:.4f}')
    print(f'WalkLM embeddings Final Test: {walklm_test:.4f}')
    if combined_test is not None:
        print(f'WalkLM + raw features Final Test: {combined_test:.4f}')


if __name__ == '__main__':
    main()
