"""
train_backbone.py - Train and Freeze GNN Backbone Detector

Trains a 3-layer GNN (GCN/GraphSAGE) + MLP classifier on StreamSpot provenance subgraphs
for benign vs. malicious subgraph classification.
Once trained, completely freezes all model weights for all subsequent milestones (M3-M8).

Theoretical Grounding:
- Wu et al., "ProvX: Toward Explainable Threat Detection on Provenance Graphs", arXiv:2508.06073, 2025.
- Ying et al., "GNNExplainer", NeurIPS 2019.
"""

import os
import random
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


class ProvenanceGNN(nn.Module):
    """
    3-layer GNN Backbone with global mean-pooling and MLP classification head.
    Designed with explicit conv layers to facilitate forward-hook intermediate
    embedding extraction without architectural changes (Milestone 3).
    """
    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 64,
        out_channels: int = 2,
        conv_type: str = 'gcn',
        dropout: float = 0.2
    ):
        super(ProvenanceGNN, self).__init__()
        self.conv_type = conv_type.lower()
        self.dropout = dropout

        # Backbone convolutional layers
        if self.conv_type == 'gcn':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
        elif self.conv_type == 'sage':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        elif self.conv_type == 'gat':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=1)
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1)
            self.conv3 = GATConv(hidden_channels, hidden_channels, heads=1)
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")

        # Classification MLP head
        self.fc1 = nn.Linear(hidden_channels, 32)
        self.fc2 = nn.Linear(32, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Layer 1 message passing
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Layer 2 message passing
        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        # Layer 3 message passing
        h3 = self.conv3(h2, edge_index)
        h3 = F.relu(h3)

        # Global graph readout
        hg = global_mean_pool(h3, batch)

        # MLP classification head
        out = self.fc1(hg)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        logits = self.fc2(out)
        return logits


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.squeeze())
        total_loss += loss.item() * batch.num_graphs

        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(batch.y.squeeze().cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())

    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': p,
        'recall': r,
        'f1': f1,
        'auc': auc,
        'fpr': fpr,
        'targets': all_targets,
        'preds': all_preds,
        'probs': all_probs
    }
    return metrics



def train_and_freeze(
    data_path: str = 'data/processed_subgraphs.pt',
    checkpoint_dir: str = 'checkpoints',
    splits_path: str = 'data/splits.pt',
    conv_type: str = 'gcn',
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.005,
    seed: int = 42
) -> Tuple[ProvenanceGNN, Dict[str, float]]:
    # Reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using compute device: {device}")

    print(f"Loading preprocessed provenance subgraphs from {data_path}...")
    dataset = torch.load(data_path, weights_only=False)
    print(f"Loaded {len(dataset)} subgraphs.")

    # Prepare indices and labels for stratified split
    indices = list(range(len(dataset)))
    labels = [d.y.item() for d in dataset]

    # Stratified 70% Train, 15% Val, 15% Test split
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    test_data = [dataset[i] for i in test_idx]

    print(f"Splits: Train={len(train_data)} | Val={len(val_data)} | Test={len(test_data)}")

    # Save splits for downstream milestones (M3-M8)
    splits = {
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }
    torch.save(splits, splits_path)
    print(f"Dataset split indices saved to {splits_path}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, exclude_keys=['node_to_log', 'node_map'])
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, exclude_keys=['node_to_log', 'node_map'])
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, exclude_keys=['node_to_log', 'node_map'])

    # Instantiate Model
    in_dim = dataset[0].x.size(1)
    model = ProvenanceGNN(
        in_channels=in_dim,
        hidden_channels=64,
        out_channels=2,
        conv_type=conv_type,
        dropout=0.2
    ).to(device)

    # Class weighting for balanced gradient signal
    num_benign = sum(1 for y in y_train if y == 0)
    num_malicious = sum(1 for y in y_train if y == 1)
    weights = torch.tensor([1.0, float(num_benign) / max(1, num_malicious)], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_f1 = -1.0
    best_state_dict = None

    print("\nTraining GNN Backbone Detector...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics['f1'])

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} (Best: {best_val_f1:.4f})"
            )

    # Load best checkpoint
    model.load_state_dict(best_state_dict)
    model = model.to(device)

    # Evaluate on held-out test set
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 60)
    print("FROZEN DETECTOR TEST EVALUATION REPORT")
    print("=" * 60)
    print(f"Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-Score:  {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {test_metrics['auc']:.4f}")
    print(f"FPR:       {test_metrics['fpr']:.4f} ({test_metrics['fpr']*100:.2f}%)")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_metrics['targets'], test_metrics['preds'])
    print(f"[[TN={cm[0,0]}, FP={cm[0,1]}],\n [FN={cm[1,0]}, TP={cm[1,1]}]]")
    print("\nClassification Report:")
    print(classification_report(test_metrics['targets'], test_metrics['preds'], target_names=['Benign (0)', 'Malicious (1)']))

    # FREEZE ALL WEIGHTS STRICTLY (Project Rule)
    print("=" * 60)
    print("FREEZING GNN BACKBONE MODEL WEIGHTS...")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Verify frozen status
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters:     {total_params}")
    print(f"Trainable Parameters: {trainable_params} (MUST BE 0)")
    assert trainable_params == 0, "Error: Model parameters were not completely frozen!"
    print("STATUS: MODEL FULLY FROZEN (eval mode, requires_grad=False).")
    print("=" * 60)

    # Save frozen model artifact
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'frozen_backbone.pt')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'in_channels': in_dim,
            'hidden_channels': 64,
            'out_channels': 2,
            'conv_type': conv_type,
            'dropout': 0.2
        },
        'test_metrics': {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc'],
            'fpr': test_metrics['fpr']
        }
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Frozen backbone checkpoint saved to: {checkpoint_path}")

    return model, test_metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train and Freeze Provenance GNN Backbone Detector')
    parser.add_argument('--dataset', type=str, default='streamspot', choices=['streamspot', 'darpa_cadets'])
    parser.add_argument('--conv_type', type=str, default='sage')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.dataset == 'streamspot':
        train_and_freeze(
            data_path='data/processed_subgraphs.pt',
            checkpoint_dir='checkpoints',
            splits_path='data/splits.pt',
            conv_type=args.conv_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed
        )
    elif args.dataset == 'darpa_cadets':
        train_and_freeze(
            data_path='data/darpa_cadets/processed_subgraphs.pt',
            checkpoint_dir='checkpoints/darpa_cadets',
            splits_path='data/darpa_cadets/splits.pt',
            conv_type=args.conv_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed
        )

