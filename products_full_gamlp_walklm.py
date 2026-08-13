import os.path as osp
import random

import numpy as np
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import degree


_original_torch_load = torch.load


def _torch_load_with_weights_only_false(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_with_weights_only_false


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_adj(edge_index, num_nodes):
    row, col = edge_index
    deg = degree(col, num_nodes=num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.clamp(min=1).pow(-0.5)
    return deg_inv_sqrt[row] * deg_inv_sqrt[col]


def propagate(x, edge_index, edge_weight):
    row, col = edge_index
    out = x.new_zeros(x.size())
    out.index_add_(0, col, x[row] * edge_weight.view(-1, 1))
    return out


def precompute_hops(x, edge_index, num_hops):
    print(f'Precomputing {num_hops} propagation hops...')
    edge_weight = normalize_adj(edge_index, x.size(0)).to(x.device)
    xs = [x]

    for hop in range(1, num_hops + 1):
        x = propagate(x, edge_index, edge_weight)
        xs.append(x)
        print(f'  hop {hop}: {tuple(x.shape)}')

    return xs


class SimpleGAMLP(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_hops,
        dropout=0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.hop_lins = torch.nn.ModuleList([
            torch.nn.Linear(in_channels, hidden_channels)
            for _ in range(num_hops + 1)
        ])
        self.att = torch.nn.Linear(hidden_channels, 1)
        self.out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, xs):
        hs = []
        for x, lin in zip(xs, self.hop_lins):
            h = lin(x).relu()
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)

        h_stack = torch.stack(hs, dim=1)
        alpha = self.att(h_stack).softmax(dim=1)
        h = (h_stack * alpha).sum(dim=1)
        return self.out(h)


@torch.no_grad()
def evaluate(logits, y, split_idx, evaluator):
    pred = logits.argmax(dim=-1, keepdim=True)
    out = {}

    for split in ['train', 'valid', 'test']:
        idx = split_idx[split]
        out[split] = evaluator.eval({
            'y_true': y[idx].view(-1, 1).cpu(),
            'y_pred': pred[idx].cpu(),
        })['acc']

    return out['train'], out['valid'], out['test']


def run_one(name, xs, y, edge_index, split_idx, evaluator, num_classes, seed):
    seed_everything(seed)

    device = y.device
    model = SimpleGAMLP(
        in_channels=xs[0].size(-1),
        hidden_channels=512,
        out_channels=num_classes,
        num_hops=len(xs) - 1,
        dropout=0.5,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,
        weight_decay=1e-4,
    )

    best_valid = 0.0
    final_test = 0.0
    best_logits = None

    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()

        logits = model(xs)
        loss = F.cross_entropy(
            logits[split_idx['train']],
            y[split_idx['train']],
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(xs)
            train_acc, valid_acc, test_acc = evaluate(
                logits,
                y,
                split_idx,
                evaluator,
            )

        if valid_acc > best_valid:
            best_valid = valid_acc
            final_test = test_acc
            best_logits = logits.detach()

        if epoch == 1 or epoch % 10 == 0:
            print(f'{name} | Seed {seed} | Epoch {epoch:03d} | '
                  f'Loss {loss:.4f} | Train {train_acc:.4f} | '
                  f'Valid {valid_acc:.4f} | Test {test_acc:.4f}')

    print(f'{name} | Seed {seed} Base Final Test: {final_test:.4f}')

    train_mask = torch.zeros(y.size(0), dtype=torch.bool, device=device)
    train_mask[split_idx['train']] = True

    cs = CorrectAndSmooth(
        num_correction_layers=50,
        correction_alpha=0.5,
        num_smoothing_layers=50,
        smoothing_alpha=0.8,
        autoscale=True,
        scale=1.0,
    )

    y_soft = best_logits.softmax(dim=-1)
    y_soft = cs.correct(y_soft, y[split_idx['train']], train_mask, edge_index)
    y_soft = cs.smooth(y_soft, y[split_idx['train']], train_mask, edge_index)

    _, cs_valid, cs_test = evaluate(y_soft, y, split_idx, evaluator)

    print(f'{name} | Seed {seed} + C&S Valid: {cs_valid:.4f}, '
          f'Test: {cs_test:.4f}')

    return final_test, cs_test


def summarize(name, values):
    values = torch.tensor(values)
    print(f'{name}: {values.mean():.4f} +/- {values.std():.4f}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ogb')

    dataset = PygNodePropPredDataset(
        name='ogbn-products',
        root=root,
        transform=ToUndirected(),
    )
    data = dataset[0]

    obj = torch.load(
        'walklm_products_2449029_embeddings.pt',
        map_location='cpu',
        weights_only=False,
    )

    y = data.y.view(-1).to(device)
    edge_index = data.edge_index.to(device)
    split_idx = dataset.get_idx_split()
    split_idx = {key: value.to(device) for key, value in split_idx.items()}
    evaluator = Evaluator(name='ogbn-products')

    print('Device:', device)
    print('Nodes:', data.num_nodes)
    print('Edges:', edge_index.size(1))
    print('Train/Valid/Test:',
          split_idx['train'].numel(),
          split_idx['valid'].numel(),
          split_idx['test'].numel())

    raw_x = data.x.to(device)
    walklm_raw_x = torch.cat([obj['walklm_x'], data.x], dim=-1).to(device)

    experiments = [
        ('SimpleGAMLP + raw', raw_x),
        ('SimpleGAMLP + WalkLM + raw', walklm_raw_x),
    ]

    seeds = [0]
    num_hops = 3
    results = {}

    for name, x in experiments:
        xs = precompute_hops(x, edge_index, num_hops)
        base_tests = []
        cs_tests = []

        for seed in seeds:
            base, cs = run_one(
                name,
                xs,
                y,
                edge_index,
                split_idx,
                evaluator,
                dataset.num_classes,
                seed,
            )
            base_tests.append(base)
            cs_tests.append(cs)

        results[f'{name} Base'] = base_tests
        results[f'{name} + C&S'] = cs_tests

        del xs
        torch.cuda.empty_cache()

    print('\nSummary')
    for name, values in results.items():
        summarize(name, values)


if __name__ == '__main__':
    main()
