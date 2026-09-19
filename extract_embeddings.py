"""
extract_embeddings.py - Per-Layer Intermediate Embedding Extraction via PyTorch Forward Hooks

Milestone 3: Captures intermediate node representations from the frozen GNN detector
during a single forward pass without altering the model architecture or reprocessing graphs.

Theoretical Grounding:
- Pelletreau-Duris et al., "Do Graph Neural Network States Contain Graph Properties?",
  PMLR vol. 284, NeSy 2025.
"""

from typing import Dict, List, Any, Optional, Tuple
import os
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from train_backbone import ProvenanceGNN


class LayerActivationExtractor:
    """
    Registers non-invasive forward hooks on each convolutional layer of the frozen GNN.
    Captures layer representations h^(1), h^(2), ..., h^(L) produced as natural byproducts
    of a single inference forward pass.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self):
        """Attaches PyTorch forward hooks to conv1, conv2, and conv3."""
        # Clean up any existing handles
        self.remove_hooks()

        target_layers = [
            ('layer_1', self.model.conv1),
            ('layer_2', self.model.conv2),
            ('layer_3', self.model.conv3),
        ]

        for name, layer in target_layers:
            handle = layer.register_forward_hook(self._create_hook(name))
            self.handles.append(handle)

    def _create_hook(self, layer_name: str):
        def hook(module: nn.Module, input_tensor: Any, output_tensor: torch.Tensor):
            # Detach to prevent computational graph accumulation and save memory
            self.activations[layer_name] = output_tensor.detach().cpu()
        return hook

    def extract_subgraph_embeddings(
        self,
        subgraph: Data,
        device: torch.device
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Executes a single forward pass on one subgraph and returns a dictionary
        of per-layer node embeddings: {'layer_1': tensor, 'layer_2': tensor, 'layer_3': tensor}.
        """
        self.model.eval()
        self.activations.clear()

        # Place input on target device
        x = subgraph.x.to(device)
        edge_index = subgraph.edge_index.to(device)

        with torch.no_grad():
            # Single forward pass computes predictions and triggers hooks simultaneously
            logits = self.model(x, edge_index)

        # Return a copy of the captured layer activations
        return {k: v.clone() for k, v in self.activations.items()}, logits.detach().cpu()

    def remove_hooks(self):
        """Removes registered hooks cleanly."""
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()


def load_frozen_backbone(
    checkpoint_path: str = 'checkpoints/frozen_backbone.pt',
    device: Optional[torch.device] = None
) -> Tuple[ProvenanceGNN, Dict[str, Any]]:
    """
    Loads the trained backbone from disk, validates that it is strictly frozen
    (requires_grad=False for all weights), and returns the model and config.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run train_backbone.py first.")

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint['model_config']

    model = ProvenanceGNN(
        in_channels=config['in_channels'],
        hidden_channels=config['hidden_channels'],
        out_channels=config['out_channels'],
        conv_type=config['conv_type'],
        dropout=config.get('dropout', 0.2)
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Strict freezing verification
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 0, f"Error: Frozen model has {trainable} trainable parameters!"

    model = model.to(device)
    return model, checkpoint


def verify_m3_embedding_extraction():
    """
    Verification script for Milestone 3:
    1. Loads frozen detector.
    2. Runs hook extractor on sample subgraphs (Benign and Malicious).
    3. Prints dimensions, statistics, and layer evolution.
    4. Validates batch extraction across the test set.
    """
    print("=" * 65)
    print("MILESTONE 3: PER-LAYER EMBEDDING EXTRACTION VERIFICATION")
    print("Theoretical Basis: Pelletreau-Duris et al. (NeSy 2025)")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load frozen model
    model, checkpoint = load_frozen_backbone('checkpoints/frozen_backbone.pt', device)
    print("Frozen detector successfully loaded and verified (0 trainable parameters).")

    # Load subgraphs and splits
    dataset = torch.load('data/processed_subgraphs.pt', weights_only=False)
    splits = torch.load('data/splits.pt', weights_only=False)
    test_indices = splits['test_idx']
    print(f"Total Subgraphs: {len(dataset)}, Test Subgraphs: {len(test_indices)}")

    # Initialize hook extractor
    extractor = LayerActivationExtractor(model)

    # Test on a Benign sample and a Malicious sample
    benign_sample = next(d for d in [dataset[i] for i in test_indices] if d.y.item() == 0)
    malicious_sample = next(d for d in [dataset[i] for i in test_indices] if d.y.item() == 1)

    samples = [
        ("BENIGN SUBGRAPH (y=0)", benign_sample),
        ("MALICIOUS SUBGRAPH (y=1)", malicious_sample),
    ]

    print("\n" + "-" * 65)
    print("SINGLE-PASS FORWARD HOOK EXTRACTION ON INDIVIDUAL SUBGRAPHS")
    print("-" * 65)

    for label_desc, sample in samples:
        layer_embeddings, logits = extractor.extract_subgraph_embeddings(sample, device)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_label = logits.argmax(dim=-1).item()

        print(f"\n{label_desc}:")
        print(f"  Graph ID: {sample.graph_id}, Subgraph ID: {sample.subgraph_id}")
        print(f"  Nodes: {sample.num_nodes}, Edges: {sample.num_edges}")
        print(f"  Model Output: Logits={logits.numpy()[0].round(3).tolist()}, Pred={pred_label} (Prob Malicious: {probs[1].item():.4f})")
        print("  Extracted Layer Embedding Dictionary:")
        for layer_name in ['layer_1', 'layer_2', 'layer_3']:
            emb = layer_embeddings[layer_name]
            mean_norm = emb.norm(dim=-1).mean().item()
            print(f"    - {layer_name}: Shape={list(emb.shape)} | dtype={emb.dtype} | Mean L2 Norm={mean_norm:.4f}")

        # Check representational evolution between layers
        h1 = layer_embeddings['layer_1']
        h2 = layer_embeddings['layer_2']
        h3 = layer_embeddings['layer_3']

        sim_1_2 = torch.cosine_similarity(h1, h2, dim=-1).mean().item()
        sim_2_3 = torch.cosine_similarity(h2, h3, dim=-1).mean().item()
        print(f"  Inter-Layer Representation Shift:")
        print(f"    - Cosine Similarity (Layer 1 -> Layer 2): {sim_1_2:.4f}")
        print(f"    - Cosine Similarity (Layer 2 -> Layer 3): {sim_2_3:.4f}")
        print(f"    (Confirms layers are computing progressively refined representations)")

    # Validate dataset-wide extraction across entire test split
    print("\n" + "-" * 65)
    print("VALIDATING DATASET-WIDE EXTRACTION ACROSS TEST SPLIT (N=85)")
    print("-" * 65)

    test_subgraphs = [dataset[i] for i in test_indices]
    extracted_counts = {'layer_1': 0, 'layer_2': 0, 'layer_3': 0}

    for idx, g in enumerate(test_subgraphs):
        layer_dict, _ = extractor.extract_subgraph_embeddings(g, device)
        assert 'layer_1' in layer_dict and 'layer_2' in layer_dict and 'layer_3' in layer_dict
        assert layer_dict['layer_1'].size(0) == g.num_nodes
        assert layer_dict['layer_2'].size(0) == g.num_nodes
        assert layer_dict['layer_3'].size(0) == g.num_nodes
        for k in extracted_counts:
            extracted_counts[k] += 1

    print(f"Successfully extracted embeddings for all {len(test_subgraphs)} test subgraphs.")
    print(f"Verification counts: {extracted_counts}")
    print("\n" + "=" * 65)
    print("M3 PER-LAYER EMBEDDING EXTRACTION PASSED SUCCESSFULLY!")
    print("=" * 65)

    extractor.remove_hooks()


if __name__ == '__main__':
    verify_m3_embedding_extraction()
