"""
attribution.py - Node Attribution at Layer l* and Audit Log Grounding

Milestone 6:
1. At layer l* (identified in M5), uses Captum (Integrated Gradients / Saliency) to rank
   nodes by their contribution to the probe classifier's decision.
2. Resolves top-ranked nodes back to their originating audit log lines via the preserved
   node-ID -> log-line lookup table.
3. Formats and prints progressive, human-readable explanations:
   "By layer l*, the model had already committed to [decision], driven primarily by: [log lines...]"

Theoretical Grounding:
- Wu et al., "ProvX: Toward Explainable Threat Detection on Provenance Graphs", arXiv:2508.06073, 2025:
  Convention for provenance graph grounding and node-to-audit-log resolution.
- Sundararajan et al., "Axiomatic Attribution for Deep Networks" (Integrated Gradients), ICML 2017 / Captum.
- Pelletreau-Duris et al., NeSy 2025 (intermediate state probing).
"""

import os
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from captum.attr import IntegratedGradients, Saliency
from torch_geometric.data import Data

from extract_embeddings import LayerActivationExtractor, load_frozen_backbone
from data_prep import NODE_TYPE_MAP


class ProbeScorer(nn.Module):
    """
    Differentiable wrapper scoring node embeddings through the linear probe's
    learned weights and bias. Facilitates Captum gradient-based attribution.
    """
    def __init__(self, weight: np.ndarray, bias: np.ndarray, target_class: int = 1):
        super(ProbeScorer, self).__init__()
        # weight: [1, D] or [C, D], bias: [1] or [C]
        if weight.ndim == 1:
            weight = weight.reshape(1, -1)
        if bias.ndim == 0:
            bias = bias.reshape(1)

        self.w = nn.Parameter(torch.tensor(weight, dtype=torch.float32), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(bias, dtype=torch.float32), requires_grad=False)
        self.target_class = target_class

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Intermediate node embeddings of shape [1, N, D]
        Returns:
            Logit score for the target class: [1]
        """
        # Global mean pooling over nodes: [1, D]
        pooled = h.mean(dim=1)
        # Linear projection: [1, C]
        if self.w.size(0) == 1:
            logits = pooled @ self.w.t() + self.b
            return logits[:, 0]
        else:
            logits = pooled @ self.w.t() + self.b
            return logits[:, self.target_class]


class LogGroundedExplainer:
    """
    Combines layer l* intermediate activations with Captum Integrated Gradients
    and the preserved node-to-log lookup table to synthesize grounded explanations.
    """
    def __init__(
        self,
        checkpoint_path: str = 'checkpoints/frozen_backbone.pt',
        probe_checkpoint_path: str = 'checkpoints/probes.pkl',
        l_star_path: str = 'checkpoints/l_star.json',
        device: Optional[torch.device] = None
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = device

        self.model, self.checkpoint = load_frozen_backbone(checkpoint_path, self.device)
        self.extractor = LayerActivationExtractor(self.model)

        with open(probe_checkpoint_path, 'rb') as f:
            probe_data = pickle.load(f)
        self.probe_models = probe_data['probe_models']

        with open(l_star_path, 'r') as f:
            l_star_data = json.load(f)
        self.l_star_name = l_star_data['l_star_name']
        self.l_star_idx = l_star_data['l_star_index']
        self.l_star_probe = self.probe_models[self.l_star_name]

    def explain_subgraph(
        self,
        subgraph: Data,
        top_k_nodes: int = 3,
        method: str = 'ig',
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Computes node attributions at layer l*, maps them back to source log lines,
        and constructs a human-readable explanation string.
        """
        # Step 1: Single forward pass to retrieve l* activations and GNN prediction
        layer_embeddings, gnn_logits = self.extractor.extract_subgraph_embeddings(subgraph, self.device)
        gnn_probs = torch.softmax(gnn_logits, dim=-1)[0]
        gnn_pred = gnn_logits.argmax(dim=-1).item()

        # Step 2: Layer l* probe prediction
        h_l_star = layer_embeddings[self.l_star_name]  # [N, D]
        pooled_l_star = h_l_star.mean(dim=0, keepdim=True).numpy()
        probe_pred = self.l_star_probe.predict(pooled_l_star)[0]
        probe_probs = self.l_star_probe.predict_proba(pooled_l_star)[0]

        # Determine target explanation class (default: probe's decision)
        exp_target = probe_pred if target_class is None else target_class
        decision_str = "malicious" if exp_target == 1 else "benign"
        conf = float(probe_probs[exp_target] * 100)

        # Step 3: Attribution via Captum
        probe_w = self.l_star_probe.coef_
        probe_b = self.l_star_probe.intercept_
        scorer = ProbeScorer(probe_w, probe_b, target_class=exp_target).to(self.device)

        h_input = h_l_star.unsqueeze(0).to(self.device).requires_grad_(True)

        if method == 'saliency':
            explainer = Saliency(scorer)
            attr = explainer.attribute(h_input)
        else:
            # Integrated Gradients
            explainer = IntegratedGradients(scorer)
            baseline = torch.zeros_like(h_input)
            attr = explainer.attribute(h_input, baselines=baseline, n_steps=30)

        # Node importance score: L2 norm of attribution vector per node
        node_scores = attr.squeeze(0).norm(dim=-1).detach().cpu().numpy()

        # Rank nodes in descending order of attribution
        ranked_indices = np.argsort(-node_scores)
        top_indices = ranked_indices[:min(top_k_nodes, len(node_scores))]

        # Step 4: Map top nodes back to original audit log lines via lookup table
        top_node_details = []
        for rank, node_idx in enumerate(top_indices, 1):
            if hasattr(subgraph, 'raw_attributes') and node_idx in subgraph.raw_attributes:
                entity_id = subgraph.raw_attributes[node_idx].get('name', str(node_idx))
                entity_type = subgraph.raw_attributes[node_idx].get('type', 'entity')
            else:
                entity_id = subgraph.node_map[node_idx] if hasattr(subgraph, 'node_map') else str(node_idx)
                raw_type_idx = subgraph.x[node_idx, :8].argmax().item() if subgraph.x.size(1) >= 8 else 0
                type_keys = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
                type_char = type_keys[raw_type_idx] if raw_type_idx < len(type_keys) else 'a'
                entity_type = NODE_TYPE_MAP.get(type_char, 'entity')

            log_lines = subgraph.node_to_log.get(node_idx, []) if hasattr(subgraph, 'node_to_log') else []

            # Deduplicate log lines while preserving chronological order
            seen = set()
            unique_logs = []
            for l in log_lines:
                if l not in seen:
                    seen.add(l)
                    unique_logs.append(l)

            top_node_details.append({
                'rank': rank,
                'subgraph_node_idx': int(node_idx),
                'entity_id': str(entity_id),
                'entity_type': entity_type,
                'attribution_score': float(node_scores[node_idx]),
                'relative_importance_pct': float(node_scores[node_idx] / (node_scores.sum() + 1e-8) * 100),
                'associated_log_lines': unique_logs
            })

        # Step 5: Format progressive human-readable explanation string
        log_snippets = []
        for nd in top_node_details:
            snippet = f"Entity '{nd['entity_type']}:{nd['entity_id']}' (Attrib: {nd['attribution_score']:.3f})"
            if nd['associated_log_lines']:
                # Show most salient log events
                events = ", ".join([f'"{l}"' for l in nd['associated_log_lines'][:2]])
                snippet += f" participating in: [{events}]"
            log_snippets.append(snippet)

        explanation_string = (
            f"By layer {self.l_star_idx} ({self.l_star_name}), the model had already committed "
            f"to '{decision_str}' (Confidence: {conf:.1f}%), driven primarily by: "
            + "; ".join(log_snippets) + "."
        )

        return {
            'graph_id': getattr(subgraph, 'graph_id', -1),
            'subgraph_id': getattr(subgraph, 'subgraph_id', -1),
            'true_label': int(subgraph.y.item()),
            'gnn_prediction': gnn_pred,
            'gnn_confidence': float(gnn_probs[gnn_pred] * 100),
            'probe_decision': decision_str,
            'probe_confidence': conf,
            'l_star_index': self.l_star_idx,
            'l_star_name': self.l_star_name,
            'explanation_string': explanation_string,
            'top_attributed_nodes': top_node_details
        }


def run_m6_attribution_demo(
    checkpoint_path: str = 'checkpoints/frozen_backbone.pt',
    probe_checkpoint_path: str = 'checkpoints/probes.pkl',
    l_star_path: str = 'checkpoints/l_star.json',
    data_path: str = 'data/processed_subgraphs.pt',
    splits_path: str = 'data/splits.pt',
    output_json: str = 'results/m6_explanations.json'
):
    """
    Executes Milestone 6 demonstration across representative test subgraphs
    (both Malicious and Benign) and outputs human-readable progressive explanations.
    """
    print("=" * 75)
    print("MILESTONE 6: NODE ATTRIBUTION AT LAYER l* & AUDIT LOG GROUNDING")
    print("Theoretical Basis: ProvX (Wu et al. 2025) & Integrated Gradients (Captum)")
    print("=" * 75)

    explainer = LogGroundedExplainer(
        checkpoint_path=checkpoint_path,
        probe_checkpoint_path=probe_checkpoint_path,
        l_star_path=l_star_path
    )
    print(f"Loaded Frozen Detector with target phase-transition layer l* = {explainer.l_star_idx} ({explainer.l_star_name}).")

    dataset = torch.load(data_path, weights_only=False)
    splits = torch.load(splits_path, weights_only=False)
    test_idx = splits['test_idx']

    # Sample representative malicious and benign test subgraphs
    malicious_candidates = [i for i in test_idx if dataset[i].y.item() == 1]
    benign_candidates = [i for i in test_idx if dataset[i].y.item() == 0]

    selected_indices = malicious_candidates[:3] + benign_candidates[:2]
    explanations = []

    print(f"\nGenerating progressive, log-grounded explanations for {len(selected_indices)} sample subgraphs...\n")

    for i, idx in enumerate(selected_indices, 1):
        sample = dataset[idx]
        exp = explainer.explain_subgraph(sample, top_k_nodes=3, method='ig')
        explanations.append(exp)

        lbl_str = "MALICIOUS (y=1)" if exp['true_label'] == 1 else "BENIGN (y=0)"
        print("*" * 75)
        print(f"CASE {i}: Graph {exp['graph_id']} | Subgraph {exp['subgraph_id']} | Ground Truth: {lbl_str}")
        print(f"Frozen GNN Output: Pred={exp['gnn_prediction']} ({'Malicious' if exp['gnn_prediction']==1 else 'Benign'}) | Conf={exp['gnn_confidence']:.1f}%")
        print(f"Layer l* Probe:    Pred={exp['probe_decision'].upper()} | Conf={exp['probe_confidence']:.1f}%")
        print("-" * 75)
        print("HUMAN-READABLE PROGRESSIVE EXPLANATION:")
        print(f"\"{exp['explanation_string']}\"")
        print("\nATTRIBUTED AUDIT LOG EVIDENCE BREAKDOWN:")
        for node in exp['top_attributed_nodes']:
            print(f"  [Rank {node['rank']}] Entity '{node['entity_id']}' ({node['entity_type']}) | Attr Score: {node['attribution_score']:.4f} ({node['relative_importance_pct']:.1f}% of total):")
            if node['associated_log_lines']:
                for log_line in node['associated_log_lines'][:3]:
                    print(f"    --> {log_line}")
            else:
                print("    --> (Isolated structure / background entity)")
        print("*" * 75 + "\n")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(explanations, f, indent=2)
    print(f"Sample grounded explanations saved to: {output_json}")
    print("=" * 75)
    print("MILESTONE 6 ATTRIBUTION AND LOG GROUNDING COMPLETE!")
    print("=" * 75)

    return explanations


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Node Attribution & Log Grounding at Layer l*')
    parser.add_argument('--dataset', type=str, default='streamspot', choices=['streamspot', 'darpa_cadets'])
    args = parser.parse_args()

    if args.dataset == 'streamspot':
        run_m6_attribution_demo(
            checkpoint_path='checkpoints/frozen_backbone.pt',
            probe_checkpoint_path='checkpoints/probes.pkl',
            l_star_path='checkpoints/l_star.json',
            data_path='data/processed_subgraphs.pt',
            splits_path='data/splits.pt',
            output_json='results/m6_explanations.json'
        )
    elif args.dataset == 'darpa_cadets':
        run_m6_attribution_demo(
            checkpoint_path='checkpoints/darpa_cadets/frozen_backbone.pt',
            probe_checkpoint_path='checkpoints/darpa_cadets/probes.pkl',
            l_star_path='checkpoints/darpa_cadets/l_star.json',
            data_path='data/darpa_cadets/processed_subgraphs.pt',
            splits_path='data/darpa_cadets/splits.pt',
            output_json='results/darpa_cadets/m6_explanations.json'
        )

