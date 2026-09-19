"""
M1: Environment Sanity Check
Runs PyG's canonical GNNExplainer on the BA-Shapes dataset.
Validates PyTorch, PyTorch Geometric, torch_geometric.explain, and scikit-learn.
"""

import sys
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCN
from torch_geometric.utils import k_hop_subgraph
import captum
import networkx
import community as community_louvain


def print_env_info():
    print("=" * 60)
    print("ENVIRONMENT SANITY CHECK")
    print("=" * 60)
    print(f"Python Version:    {sys.version.split()[0]}")
    print(f"PyTorch Version:   {torch.__version__}")
    print(f"PyG Version:       {torch_geometric.__version__}")
    print(f"Captum Version:    {captum.__version__}")
    print(f"NetworkX Version:  {networkx.__version__}")
    print(f"Louvain Available: True ({community_louvain.__name__})")
    print(f"Device:            {'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'}")
    print("=" * 60)


def main():
    print_env_info()

    # Reproducibility
    torch.manual_seed(42)

    print("\n[Step 1/3] Generating synthetic BA-Shapes ExplainerDataset...")
    dataset = ExplainerDataset(
        graph_generator=BAGraph(num_nodes=300, num_edges=5),
        motif_generator='house',
        num_motifs=80,
        transform=T.Constant(),
    )
    data = dataset[0]
    print(f"Dataset generated: {data.num_nodes} nodes, {data.num_edges} edges, {dataset.num_classes} classes.")

    idx = torch.arange(data.num_nodes)
    train_idx, test_idx = train_test_split(idx, train_size=0.8, random_state=42, stratify=data.y)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    print("\n[Step 2/3] Training 3-layer GCN backbone on BA-Shapes...")
    model = GCN(
        in_channels=data.num_node_features,
        hidden_channels=20,
        num_layers=3,
        out_channels=dataset.num_classes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

    for epoch in range(1, 401):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_idx], data.y[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        if epoch % 100 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=-1)
                train_acc = int((pred[train_idx] == data.y[train_idx]).sum()) / train_idx.size(0)
                test_acc = int((pred[test_idx] == data.y[test_idx]).sum()) / test_idx.size(0)
            print(f"  Epoch {epoch:03d} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    model.eval()

    print("\n[Step 3/3] Running PyG GNNExplainer (phenomenon & model modes)...")
    for explanation_type in ['phenomenon', 'model']:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=150),
            explanation_type=explanation_type,
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='raw',
            ),
        )

        targets, preds = [], []
        # Sample evaluation nodes
        eval_node_indices = list(range(400, min(data.num_nodes, 460), 5))
        for node_index in tqdm(eval_node_indices, desc=f"Explaining ({explanation_type})"):
            target = data.y if explanation_type == 'phenomenon' else None
            explanation = explainer(data.x, data.edge_index, index=node_index, target=target)

            _, _, _, hard_edge_mask = k_hop_subgraph(
                node_index, num_hops=3, edge_index=data.edge_index
            )

            targets.append(data.edge_mask[hard_edge_mask].cpu())
            preds.append(explanation.edge_mask[hard_edge_mask].cpu())

        y_true = torch.cat(targets).numpy()
        y_score = torch.cat(preds).detach().numpy()

        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)
            print(f"--> Mean Edge ROC-AUC ({explanation_type:10}): {auc:.4f}")
        else:
            print(f"--> Explanation completed for {explanation_type} (only 1 class in sample subset)")

    print("\n" + "=" * 60)
    print("M1 ENVIRONMENT SANITY CHECK PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == '__main__':
    main()
