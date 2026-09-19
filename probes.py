"""
probes.py - Per-Layer Linear Probe Classifiers

Milestone 4: Mean-pools each layer's node embeddings into one graph-level vector per subgraph.
Trains an independent LogisticRegression probe per layer against the true label.
Reports per-layer probe accuracy as a table and a plot (accuracy vs. layer index) saved to disk.

Theoretical Grounding:
- Pelletreau-Duris et al., "Do Graph Neural Network States Contain Graph Properties?",
  PMLR vol. 284, NeSy 2025.
  Source of the per-layer linear probing methodology. The novel adaptation here probes
  for the task-specific intrusion label (benign vs. malicious threat) rather than
  graph-theoretic properties, revealing how threat detection commits across layers.
"""

import os
import pickle
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report
)

from extract_embeddings import LayerActivationExtractor, load_frozen_backbone


def extract_pooled_embeddings_for_dataset(
    model: torch.nn.Module,
    dataset: List[Any],
    indices: List[int],
    device: torch.device
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Extracts mean-pooled graph-level representations from intermediate layers
    for all subgraphs in a given split.
    Also records the frozen GNN's actual prediction for fidelity evaluation (M5).

    Returns:
        layer_features: Dict mapping layer_name ('layer_0', 'layer_1', ...) to [N, D] array
        true_labels: [N] array of ground truth labels
        gnn_preds: [N] array of frozen GNN final predictions
    """
    extractor = LayerActivationExtractor(model)

    layer_features: Dict[str, List[np.ndarray]] = {
        'layer_0 (input)': [],
        'layer_1': [],
        'layer_2': [],
        'layer_3': []
    }
    true_labels: List[int] = []
    gnn_preds: List[int] = []

    for idx in indices:
        subgraph = dataset[idx]
        layer_embeddings, logits = extractor.extract_subgraph_embeddings(subgraph, device)

        # Layer 0: Mean-pooled raw input node features (baseline)
        h0 = subgraph.x.mean(dim=0).cpu().numpy()
        layer_features['layer_0 (input)'].append(h0)

        # Layers 1, 2, 3: Mean-pooled intermediate node representations
        for l_name in ['layer_1', 'layer_2', 'layer_3']:
            hl = layer_embeddings[l_name].mean(dim=0).cpu().numpy()
            layer_features[l_name].append(hl)

        true_labels.append(subgraph.y.item())
        gnn_preds.append(logits.argmax(dim=-1).item())

    extractor.remove_hooks()

    feature_arrays = {k: np.array(v) for k, v in layer_features.items()}
    return feature_arrays, np.array(true_labels), np.array(gnn_preds)


def train_and_evaluate_probes(
    checkpoint_path: str = 'checkpoints/frozen_backbone.pt',
    data_path: str = 'data/processed_subgraphs.pt',
    splits_path: str = 'data/splits.pt',
    output_dir: str = 'results',
    probe_checkpoint_path: str = 'checkpoints/probes.pkl'
) -> Dict[str, Dict[str, float]]:
    """
    Trains one independent linear probe per layer against the true label,
    evaluates on the test set, prints a summary table, and plots accuracy vs. layer index.
    """
    print("=" * 70)
    print("MILESTONE 4: PER-LAYER LINEAR PROBE CLASSIFIERS")
    print("Theoretical Basis: Adapted from Pelletreau-Duris et al. (NeSy 2025)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model, checkpoint = load_frozen_backbone(checkpoint_path, device)
    backbone_acc = checkpoint['test_metrics']['accuracy']
    backbone_f1 = checkpoint['test_metrics']['f1']
    print(f"Frozen GNN Backbone Loaded: Test Acc = {backbone_acc:.4f}, Test F1 = {backbone_f1:.4f}")

    dataset = torch.load(data_path, weights_only=False)
    splits = torch.load(splits_path, weights_only=False)
    train_idx = splits['train_idx']
    test_idx = splits['test_idx']

    print(f"Extracting pooled layer representations for {len(train_idx)} Train & {len(test_idx)} Test subgraphs...")
    X_train, y_train, _ = extract_pooled_embeddings_for_dataset(model, dataset, train_idx, device)
    X_test, y_test, gnn_test_preds = extract_pooled_embeddings_for_dataset(model, dataset, test_idx, device)

    # Train independent LogisticRegression probes
    layers_to_probe = ['layer_0 (input)', 'layer_1', 'layer_2', 'layer_3']
    probe_models: Dict[str, LogisticRegression] = {}
    probe_metrics: Dict[str, Dict[str, float]] = {}

    print("\nTraining independent linear probes per layer...")
    for layer in layers_to_probe:
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(X_train[layer], y_train)
        probe_models[layer] = clf

        # Evaluate on test set
        y_pred = clf.predict(X_test[layer])
        y_prob = clf.predict_proba(X_test[layer])[:, 1]

        acc = accuracy_score(y_test, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        probe_metrics[layer] = {
            'accuracy': acc,
            'precision': p,
            'recall': r,
            'f1': f1,
            'roc_auc': auc,
            'preds': y_pred,
            'probs': y_prob
        }

    # Print Formatted Results Table
    print("\n" + "=" * 70)
    print("PER-LAYER PROBE ACCURACY & PERFORMANCE TABLE")
    print("=" * 70)
    header = f"{'Layer':<18} | {'Dim':<5} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}"
    print(header)
    print("-" * len(header))
    for layer in layers_to_probe:
        m = probe_metrics[layer]
        dim = X_train[layer].shape[1]
        print(f"{layer:<18} | {dim:<5} | {m['accuracy']*100:>8.2f}% | {m['precision']:>10.4f} | {m['recall']:>10.4f} | {m['f1']:>10.4f} | {m['roc_auc']:>10.4f}")
    print("-" * len(header))
    print(f"{'Frozen GNN Full':<18} | {'-':<5} | {backbone_acc*100:>8.2f}% | {checkpoint['test_metrics']['precision']:>10.4f} | {checkpoint['test_metrics']['recall']:>10.4f} | {backbone_f1:>10.4f} | {'-':>10}")
    print("=" * 70)

    # Save probes to checkpoint for M5 and M6
    os.makedirs(os.path.dirname(probe_checkpoint_path), exist_ok=True)
    with open(probe_checkpoint_path, 'wb') as f:
        pickle.dump({
            'probe_models': probe_models,
            'layers': layers_to_probe,
            'probe_metrics': {k: {m_k: m_v for m_k, m_v in v.items() if m_k not in ['preds', 'probs']} for k, v in probe_metrics.items()}
        }, f)
    print(f"\nTrained probes saved to: {probe_checkpoint_path}")

    # Plot Accuracy vs. Layer Index
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'm4_probe_accuracy.png')

    x_labels = ['Input (L0)', 'Layer 1', 'Layer 2', 'Layer 3']
    accuracies = [probe_metrics[l]['accuracy'] * 100 for l in layers_to_probe]
    f1_scores = [probe_metrics[l]['f1'] * 100 for l in layers_to_probe]

    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    plt.plot(x_labels, accuracies, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='Probe Accuracy (%)')
    plt.plot(x_labels, f1_scores, marker='s', linewidth=2.5, markersize=8, color='#ff7f0e', linestyle='--', label='Probe F1-Score (%)')
    plt.axhline(y=backbone_acc * 100, color='#2ca02c', linestyle=':', linewidth=2, label=f'Frozen GNN Final Acc ({backbone_acc*100:.1f}%)')

    for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
        plt.annotate(f"{acc:.1f}%", (x_labels[i], acc), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color='#1f77b4')

    plt.title("Layer-Wise Linear Probing: Threat Information Emergence", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("GNN Architecture Layer", fontsize=11, labelpad=10)
    plt.ylabel("Score (%)", fontsize=11, labelpad=10)
    plt.ylim(min(accuracies + f1_scores) - 8, 102)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right', fontsize=10)
    plt.tight_layout()

    plt.savefig(plot_path)
    plt.close()
    print(f"Per-layer probe accuracy plot saved to: {plot_path}")

    return probe_metrics


if __name__ == '__main__':
    train_and_evaluate_probes()
