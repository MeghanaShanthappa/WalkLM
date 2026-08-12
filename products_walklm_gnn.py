import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import subgraph


# OGB stores processed PyG Data objects. PyTorch 2.6+ defaults to
# torch.load(weights_only=True), which cannot load these objects.
_original_torch_load = torch.load


def _torch_load_with_weights_only_false(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_with_weights_only_false


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


@torch.no_grad()
def test(model, x, edge_index, y, train_mask, val_mask, test_mask):
    model.eval()
    out = model(x, edge_index)
    pred = out.argmax(dim=-1)

    train_acc = float((pred[train_mask] == y[train_mask]).float().mean())
    val_acc = float((pred[val_mask] == y[val_mask]).float().mean())
    test_acc = float((pred[test_mask] == y[test_mask]).float().mean())
    return train_acc, val_acc, test_acc


def run(name, model, x, edge_index, y, train_mask, val_mask, test_mask, epochs, lr, weight_decay):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = final_test = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        train_acc, val_acc, test_acc = test(
            model, x, edge_index, y, train_mask, val_mask, test_mask)
        if val_acc > best_val:
            best_val = val_acc
            final_test = test_acc

        print(f'{name} | Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
              f'Test: {test_acc:.4f}')

    print(f'{name} Final Test: {final_test:.4f}')
    return final_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_nodes', type=int, default=100000)
    parser.add_argument('--embedding_path', type=str,
                        default='walklm_products_100000_embeddings.pt')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ogb')

    dataset = PygNodePropPredDataset(name='ogbn-products', root=root)
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    num_train = int(args.max_nodes * 0.6)
    num_val = int(args.max_nodes * 0.2)
    num_test = args.max_nodes - num_train - num_val

    train_nodes = split_idx['train'][:num_train]
    val_nodes = split_idx['valid'][:num_val]
    test_nodes = split_idx['test'][:num_test]

    node_idx = torch.cat([train_nodes, val_nodes, test_nodes], dim=0)
    edge_index, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True)

    raw_x = data.x[node_idx]
    y = data.y[node_idx].view(-1)

    obj = torch.load(args.embedding_path, map_location='cpu', weights_only=False)
    walklm_x = obj['walklm_x']
    combined_x = torch.cat([walklm_x, raw_x], dim=-1)

    train_mask = torch.zeros(node_idx.numel(), dtype=torch.bool)
    val_mask = torch.zeros(node_idx.numel(), dtype=torch.bool)
    test_mask = torch.zeros(node_idx.numel(), dtype=torch.bool)

    train_mask[:num_train] = True
    val_mask[num_train:num_train + num_val] = True
    test_mask[num_train + num_val:] = True

    edge_index = edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    print('Device:', device)
    print('Nodes:', node_idx.numel())
    print('Edges:', edge_index.size(1))
    print('Train/Val/Test:', int(train_mask.sum()), int(val_mask.sum()),
          int(test_mask.sum()))

    features = {
        'Raw features': raw_x,
        'WalkLM embeddings': walklm_x,
        'WalkLM + raw features': combined_x,
    }

    results = {}
    for model_name, model_cls in [('GCN', GCN), ('GraphSAGE', GraphSAGE)]:
        for feature_name, x in features.items():
            x = x.to(device)
            model = model_cls(x.size(-1), args.hidden_channels,
                              dataset.num_classes).to(device)
            name = f'{model_name} + {feature_name}'
            results[name] = run(name, model, x, edge_index, y, train_mask,
                                val_mask, test_mask, args.epochs, args.lr,
                                args.weight_decay)

    print('\nSummary:')
    for name, acc in results.items():
        print(f'{name}: {acc:.4f}')


if __name__ == '__main__':
    main()
