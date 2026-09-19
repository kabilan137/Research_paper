"""
fidelity.py - Fidelity Validation & Decision Phase-Transition Layer (l*) Identification

Milestone 5: Compares each layer probe's predictions against the frozen GNN's actual
final predictions (not ground truth) to compute per-layer fidelity/agreement rates.
Identifies and reports the decision phase-transition layer l* where the decision settles.

Theoretical Grounding:
- Pelletreau-Duris et al., "Do Graph Neural Network States Contain Graph Properties?",
  PMLR vol. 284, NeSy 2025 (probing methodology).
- Fidelity-to-model principle: explanations and probes must be evaluated with respect
  to the frozen model's actual reasoning, not external labels.
"""

import os
import json
import pickle
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr, spearmanr

from extract_embeddings import LayerActivationExtractor, load_frozen_backbone
from probes import extract_pooled_embeddings_for_dataset


def compute_fidelity_metrics(
    y_true: np.ndarray,
    gnn_preds: np.ndarray,
    gnn_probs: np.ndarray,
    probe_preds_dict: Dict[str, np.ndarray],
    probe_probs_dict: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Computes fidelity metrics comparing probe outputs against the frozen GNN's predictions:
    1. Agreement Rate: fraction of samples where probe prediction equals GNN prediction.
    2. Benign Fidelity: agreement when GNN predicts Benign (0).
    3. Malicious Fidelity: agreement when GNN predicts Malicious (1).
    4. Cohen's Kappa: inter-rater agreement above chance between probe and GNN.
    5. Confidence Correlation: Pearson r between probe probability and GNN probability.
    """
    metrics = {}
    N = len(gnn_preds)

    gnn_benign_mask = (gnn_preds == 0)
    gnn_malicious_mask = (gnn_preds == 1)

    for layer_name, probe_preds in probe_preds_dict.items():
        # Overall Agreement
        agreement = np.mean(probe_preds == gnn_preds)

        # Class-conditional agreement
        benign_agreement = np.mean(probe_preds[gnn_benign_mask] == gnn_preds[gnn_benign_mask]) if np.any(gnn_benign_mask) else 0.0
        malicious_agreement = np.mean(probe_preds[gnn_malicious_mask] == gnn_preds[gnn_malicious_mask]) if np.any(gnn_malicious_mask) else 0.0

        # Cohen's Kappa against GNN
        kappa = cohen_kappa_score(probe_preds, gnn_preds)

        # Correlation between probability scores
        probe_prob = probe_probs_dict[layer_name]
        p_corr, _ = pearsonr(probe_prob, gnn_probs)
        s_corr, _ = spearmanr(probe_prob, gnn_probs)

        # Ground-truth accuracy for comparison
        gt_accuracy = np.mean(probe_preds == y_true)

        metrics[layer_name] = {
            'agreement': float(agreement),
            'benign_agreement': float(benign_agreement),
            'malicious_agreement': float(malicious_agreement),
            'cohen_kappa': float(kappa),
            'pearson_corr': float(p_corr),
            'spearman_corr': float(s_corr),
            'gt_accuracy': float(gt_accuracy)
        }

    return metrics


def identify_phase_transition_layer(
    fidelity_metrics: Dict[str, Dict[str, float]],
    layers: List[str],
    plateau_threshold: float = 0.90,
    plateau_delta: float = 0.03
) -> Tuple[str, int, str]:
    """
    Identifies the layer l* where agreement first spikes/plateaus:
    l* is defined as the earliest conv layer (l >= 1) where:
    1. Agreement exceeds plateau_threshold (e.g. 90%), or
    2. Delta in agreement to the next layer is less than plateau_delta (decision settles).
    """
    conv_layers = [l for l in layers if l.startswith('layer_') and not l.startswith('layer_0')]

    l_star_name = conv_layers[-1]
    l_star_idx = len(conv_layers)
    reason = "Defaulted to final layer."

    for i, l_name in enumerate(conv_layers):
        agr = fidelity_metrics[l_name]['agreement']
        # Check if next layer exists
        if i < len(conv_layers) - 1:
            next_l_name = conv_layers[i + 1]
            next_agr = fidelity_metrics[next_l_name]['agreement']
            delta = next_agr - agr
            if agr >= plateau_threshold or delta <= plateau_delta:
                l_star_name = l_name
                l_star_idx = i + 1
                reason = f"Agreement reached {agr*100:.2f}% with minimal subsequent gain (Delta to next layer = {delta*100:+.2f}%)."
                break
        else:
            if agr >= plateau_threshold:
                l_star_name = l_name
                l_star_idx = i + 1
                reason = f"Final layer achieved {agr*100:.2f}% agreement."

    return l_star_name, l_star_idx, reason


def evaluate_fidelity(
    checkpoint_path: str = 'checkpoints/frozen_backbone.pt',
    probe_checkpoint_path: str = 'checkpoints/probes.pkl',
    data_path: str = 'data/processed_subgraphs.pt',
    splits_path: str = 'data/splits.pt',
    output_dir: str = 'results',
    l_star_path: str = 'checkpoints/l_star.json'
) -> Dict[str, Any]:
    """
    Main execution pipeline for Milestone 5:
    Loads frozen backbone and probes, evaluates per-layer fidelity on the test set,
    identifies l*, plots agreement curves, and outputs formatted report.
    """
    print("=" * 75)
    print("MILESTONE 5: FIDELITY VALIDATION & PHASE-TRANSITION LAYER (l*)")
    print("Evaluating Agreement with Frozen GNN Final Predictions")
    print("=" * 75)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model, checkpoint = load_frozen_backbone(checkpoint_path, device)

    # Load probes
    if not os.path.exists(probe_checkpoint_path):
        raise FileNotFoundError(f"Probes not found at {probe_checkpoint_path}. Run probes.py first.")

    with open(probe_checkpoint_path, 'rb') as f:
        probe_data = pickle.load(f)

    probe_models = probe_data['probe_models']
    layers = probe_data['layers']

    # Load test split
    dataset = torch.load(data_path, weights_only=False)
    splits = torch.load(splits_path, weights_only=False)
    test_idx = splits['test_idx']

    print(f"Loaded {len(layers)} trained probes. Evaluating on {len(test_idx)} test subgraphs...")

    # Extract test embeddings and frozen GNN predictions
    X_test, y_test, gnn_test_preds = extract_pooled_embeddings_for_dataset(model, dataset, test_idx, device)

    # Compute GNN output probabilities
    gnn_test_probs = []
    extractor = LayerActivationExtractor(model)
    for idx in test_idx:
        _, logits = extractor.extract_subgraph_embeddings(dataset[idx], device)
        prob_malicious = torch.softmax(logits, dim=-1)[0, 1].item()
        gnn_test_probs.append(prob_malicious)
    extractor.remove_hooks()
    gnn_test_probs = np.array(gnn_test_probs)

    # Gather probe predictions and probabilities
    probe_preds_dict = {}
    probe_probs_dict = {}
    for layer in layers:
        clf = probe_models[layer]
        probe_preds_dict[layer] = clf.predict(X_test[layer])
        probe_probs_dict[layer] = clf.predict_proba(X_test[layer])[:, 1]

    # Compute fidelity metrics against frozen GNN
    metrics = compute_fidelity_metrics(
        y_true=y_test,
        gnn_preds=gnn_test_preds,
        gnn_probs=gnn_test_probs,
        probe_preds_dict=probe_preds_dict,
        probe_probs_dict=probe_probs_dict
    )

    # Identify phase transition layer l*
    l_star_name, l_star_idx, l_star_reason = identify_phase_transition_layer(metrics, layers)

    # Print Table
    print("\n" + "=" * 75)
    print("PER-LAYER FIDELITY / AGREEMENT WITH FROZEN GNN")
    print("=" * 75)
    header = f"{'Layer':<18} | {'Fidelity (Agr)':<15} | {'Benign Agr':<12} | {'Attack Agr':<12} | {'Cohen Kappa':<12} | {'Prob Corr (r)':<12}"
    print(header)
    print("-" * len(header))
    for layer in layers:
        m = metrics[layer]
        mark = " <--- l*" if layer == l_star_name else ""
        print(f"{layer:<18} | {m['agreement']*100:>13.2f}% | {m['benign_agreement']*100:>10.2f}% | {m['malicious_agreement']*100:>10.2f}% | {m['cohen_kappa']:>12.4f} | {m['pearson_corr']:>12.4f}{mark}")
    print("-" * len(header))
    print(f"\n[PHASE-TRANSITION DISCOVERY]")
    print(f"--> Decision Phase-Transition Layer: l* = {l_star_idx} ({l_star_name})")
    print(f"--> Discovery Rationale: {l_star_reason}")
    print("=" * 75)

    # Save l* metadata
    os.makedirs(os.path.dirname(l_star_path), exist_ok=True)
    l_star_data = {
        'l_star_name': l_star_name,
        'l_star_index': l_star_idx,
        'reason': l_star_reason,
        'fidelity_metrics': metrics
    }
    with open(l_star_path, 'w') as f:
        json.dump(l_star_data, f, indent=2)
    print(f"Phase-transition metadata saved to: {l_star_path}")

    # Plot Fidelity vs. Layer Index
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'm5_fidelity_agreement.png')

    x_labels = ['Input (L0)', 'Layer 1', 'Layer 2', 'Layer 3']
    agreements = [metrics[l]['agreement'] * 100 for l in layers]
    kappas = [metrics[l]['cohen_kappa'] for l in layers]
    gt_accs = [metrics[l]['gt_accuracy'] * 100 for l in layers]

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    color1 = '#1f77b4'
    ax1.set_xlabel('GNN Architecture Layer', fontsize=11, labelpad=10)
    ax1.set_ylabel('Fidelity / Agreement with Frozen GNN (%)', color=color1, fontsize=11, labelpad=10)
    line1 = ax1.plot(x_labels, agreements, marker='o', linewidth=2.5, markersize=8, color=color1, label='Fidelity (Agreement with GNN %)')
    line2 = ax1.plot(x_labels, gt_accs, marker='^', linewidth=2.0, markersize=7, color='#2ca02c', linestyle=':', label='Ground Truth Accuracy (%)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(min(agreements + gt_accs) - 10, 105)

    # Annotate agreement points
    for i, txt in enumerate(agreements):
        ax1.annotate(f"{txt:.1f}%", (x_labels[i], txt), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color=color1)

    # Secondary axis for Cohen's Kappa
    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.set_ylabel("Cohen's Kappa Score (κ)", color=color2, fontsize=11, labelpad=10)
    line3 = ax2.plot(x_labels, kappas, marker='s', linewidth=2.0, markersize=7, color=color2, linestyle='--', label="Cohen's Kappa (κ)")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(False)

    # Highlight l*
    plt.axvline(x=l_star_idx, color='#9467bd', linestyle='-.', linewidth=2.0, label=f'Phase-Transition Layer l* = {l_star_idx}')

    # Combined legend
    lines = line1 + line2 + line3 + [plt.Line2D([0], [0], color='#9467bd', linestyle='-.', linewidth=2.0, label=f'Decision Settles at l*={l_star_idx}')]
    labels_legend = [l.get_label() for l in lines]
    ax1.legend(lines, labels_legend, loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)

    plt.title(f"Decision Phase Transition: Per-Layer Fidelity to Frozen GNN (l* = {l_star_idx})", fontsize=12.5, fontweight='bold', pad=15)
    plt.tight_layout()

    plt.savefig(plot_path)
    plt.close()
    print(f"Fidelity agreement plot saved to: {plot_path}")

    return {
        'l_star_name': l_star_name,
        'l_star_index': l_star_idx,
        'metrics': metrics,
        'plot_path': plot_path
    }


if __name__ == '__main__':
    evaluate_fidelity()
