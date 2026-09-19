# Layer-Wise, Log-Grounded Explainability for GNN-Based Threat Detection

A research prototype providing progressive, audit-log-grounded explainability for Graph Neural Network intrusion detectors operating on system provenance graphs.

---

## 1. Problem & Proposed Methodology

### The Problem
GNN-based intrusion detection systems built on provenance graphs (derived from OS audit logs) achieve high classification accuracy, but operate as black boxes. Existing state-of-the-art explainers (e.g., GNNExplainer, PGExplainer, XGNN) only explain the model's final-layer prediction; none reveal how the model's decision evolves layer-by-layer across message-passing steps, and none ground explanations in human-readable OS audit log records.

### The Proposed Solution
1. **Provenance Graph Construction & Subgraph Partitioning**: Ingest system audit logs (StreamSpot), construct provenance graphs (nodes = processes, files, sockets, threads; edges = syscalls), and partition them into cohesive subgraphs using the Louvain community detection algorithm, maintaining a persistent `node_id -> audit_log` lookup table.
2. **Train and Freeze GNN Detector**: Train a 3-layer GraphSAGE classifier on benign vs. malicious subgraphs, then **freeze it completely** (`eval()`, `no_grad()`).
3. **Per-Layer Embedding Extraction**: Intercept intermediate node representations computed during a single forward pass via non-invasive PyTorch forward hooks.
4. **Per-Layer Linear Probing**: Apply global mean-pooling to intermediate representations and train independent `LogisticRegression` probes per layer.
5. **Fidelity Validation & Phase-Transition Layer ($l^*$)**: Measure the agreement rate between each layer's probe and the frozen GNN's actual prediction. Identify the decision phase-transition layer $l^*$ where agreement first plateaus.
6. **Node Attribution & Log Grounding**: At layer $l^*$, apply Captum Integrated Gradients to rank nodes by their contribution to the probe decision, and map top nodes back to the original source audit log lines.
7. **Progressive Explanation Synthesis**: Generate human-readable explanations:
   > *"By layer 2, the model had already committed to 'malicious' (Confidence: 99.4%), driven primarily by: [LogLine #27420127 thread --open--> file], [LogLine #27420065 thread --open--> file]."*

---

## 2. Theoretical Grounding & Citations

The prototype builds upon and cites the following literature:
- **Pelletreau-Duris et al., "Do Graph Neural Network States Contain Graph Properties?", PMLR vol. 284, NeSy 2025**:
  Source of the per-layer linear probing methodology. Adapted here to probe for downstream threat classification labels rather than graph-theoretic properties.
- **Wu et al., "ProvX: Toward Explainable Threat Detection on Provenance Graphs", arXiv:2508.06073, 2025**:
  Source of provenance graph construction, Louvain-based community subgraph extraction, and the node-to-audit-log grounding convention.
- **Ying et al., "GNNExplainer: Generating Explanations for Graph Neural Networks", NeurIPS 2019**:
  Mutual-information mask optimization baseline.
- **Luo et al., "Parameterized Explainer for Graph Neural Networks", NeurIPS 2020 (PGExplainer)**:
  Parameterized neural explainer baseline.
- **Schnake et al., "Higher-Order Explanations of GNNs via Relevant Walks", IEEE TPAMI, vol. 44, no. 11, 2021 (GNN-LRP)**:
  Designated IEEE-journal base paper. Decomposes layer-wise relevance via fixed mathematical rules ($z$-rule, $\epsilon$-rule), whereas our method uses learned, statistically validated probes with empirical fidelity verification ($\kappa$).

---

## 3. Repository Structure

```
├── requirements.txt            # Environment dependencies
├── m1_sanity_check.py          # M1: PyG GNNExplainer environment verification
├── data_prep.py                # M2: Provenance graph ingestion & Louvain partitioning
├── train_backbone.py           # M2: 3-layer GraphSAGE backbone training & freezing
├── extract_embeddings.py       # M3: Forward hook intermediate representation extractor
├── probes.py                   # M4: Per-layer linear probes & accuracy evaluation
├── fidelity.py                 # M5: Model fidelity validation & l* identification
├── attribution.py              # M6: Captum node attribution & log grounding engine
├── baselines.py                # M7: PyG GNNExplainer & PGExplainer baseline comparison
├── app.py                      # M8: Interactive Streamlit demo dashboard
├── checkpoints/
│   ├── frozen_backbone.pt      # Read-only frozen GNN weights & config
│   ├── probes.pkl              # Trained LogisticRegression layer probes
│   └── l_star.json             # Phase-transition metadata (l* = 2)
├── results/
│   ├── m4_probe_accuracy.png   # Accuracy vs. layer plot
│   ├── m5_fidelity_agreement.png # Agreement & Cohen's Kappa curve
│   ├── m6_explanations.json    # Sample progressive grounded explanations
│   └── m7_baseline_comparison.json # Baseline comparison matrix
└── data/
    ├── processed_subgraphs.pt  # 561 preprocessed provenance subgraphs
    └── splits.pt               # Train/val/test stratified split indices
```

---

## 4. Reproduction Instructions

### Installation
```bash
pip install -r requirements.txt
```

### Milestone 1: Environment Sanity Check
Runs PyG's canonical GNNExplainer on synthetic BA-Shapes:
```bash
python3 m1_sanity_check.py
```

### Milestone 2: Data Preprocessing & Backbone Training
Preprocesses StreamSpot into Louvain subgraphs and trains the 3-layer GNN:
```bash
python3 data_prep.py
python3 train_backbone.py
```
- **Test Metrics**: Accuracy: 91.76%, Precision: 85.71%, Recall: 97.30%, F1: 91.14%.
- **Frozen Status**: All 25,890 parameters locked (`requires_grad = False`).

### Milestone 3: Per-Layer Embedding Extraction
Verifies forward-hook extraction during a single forward pass:
```bash
python3 extract_embeddings.py
```

### Milestone 4: Train Linear Probes
Trains per-layer logistic regression probes and generates accuracy curves:
```bash
python3 probes.py --dataset=streamspot
python3 probes.py --dataset=darpa_cadets
```
- **StreamSpot**: L0: 68.24% Acc | L1: 81.18% Acc | L2: 91.76% Acc | L3: 91.76% Acc
- **DARPA Cadets**: L0: 95.00% Acc | L1: 98.33% Acc | L2: 98.33% Acc | L3: 100.00% Acc

### Milestone 5: Fidelity Validation & $l^*$ Identification
Measures agreement against the frozen GNN's actual predictions and generates comparative plots:
```bash
python3 fidelity.py --dataset=streamspot
python3 fidelity.py --dataset=darpa_cadets
```
- **StreamSpot Phase-Transition**: $l^* = 2$ (Layer 2 achieves 92.94% agreement, $\kappa = 0.8589$, $r = 0.9719$).
- **DARPA Cadets Phase-Transition**: $l^* = 1$ (Layer 1 achieves 96.67% agreement, $\kappa = 0.9331$, $r = 0.9617$).
- Comparative plot saved to `results/side_by_side_fidelity.png`.

### Milestone 6: Node Attribution & Audit Log Grounding
Attributes influential nodes at $l^*$ using Captum and maps back to audit logs / process paths:
```bash
python3 attribution.py --dataset=streamspot
python3 attribution.py --dataset=darpa_cadets
```

### Milestone 7: Baseline Comparison & Probability of Necessity (PN)
Runs PyG's official GNNExplainer and PGExplainer, and evaluates prediction flip-rates:
```bash
python3 baselines.py --dataset=streamspot
python3 baselines.py --dataset=darpa_cadets
```

### Milestone 8: Interactive Demo Dashboard
Launches the interactive Streamlit web dashboard with live dataset switching:
```bash
streamlit run app.py
```
The dashboard allows live inspection of any subgraph across both StreamSpot and DARPA Cadets, renders per-layer fidelity and correlation plots, displays top-attributed nodes via Captum, and prints human-readable audit log evidence.

---

## 5. Comparative Evaluation Summary

For detailed numerical tables, ProvX metrics (Acc, Prec, Rec, F1, ROC-AUC, FPR), Probability of Necessity (PN) flip-rate comparisons, and an end-to-end case study of the Cadets Drakon APT attack, see:
👉 **[RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)**

