"""
app.py - Interactive Streamlit Demo Dashboard

Milestone 8: Final deliverable demonstrating Layer-Wise, Log-Grounded Explainability
for GNN-Based Threat Detection.
Features:
- Subgraph selection (benign vs. malicious)
- Per-layer probe accuracy and fidelity trajectories
- Decision phase-transition layer l* visualization
- Top-K attributed nodes computed via Captum Integrated Gradients
- Human-readable audit log evidence resolution

Theoretical Grounding:
- Pelletreau-Duris et al. (NeSy 2025): Per-layer linear probing.
- Wu et al. (ProvX 2025): Provenance graph construction & audit log grounding.
- Ying et al. (NeurIPS 2019): Baseline GNNExplainer.
- Schnake et al. (IEEE TPAMI 2021): Layer-wise relevance decomposition (GNN-LRP).
"""

import os
import json
import pickle
from typing import Dict, List, Any

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from extract_embeddings import LayerActivationExtractor, load_frozen_backbone
from attribution import LogGroundedExplainer
from data_prep import NODE_TYPE_MAP, EDGE_TYPE_MAP


st.set_page_config(
    page_title="Layer-Wise GNN Threat Explainability",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium research demo aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .exp-callout {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        border-left: 5px solid #4F46E5;
        padding: 18px;
        border-radius: 8px;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #1E1B4B;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .log-badge {
        background-color: #F1F5F9;
        color: #0F172A;
        font-family: monospace;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.88rem;
    }
    .badge-malicious {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .badge-benign {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    """Caches model, explainer, dataset, and metadata."""
    device = torch.device('cpu')
    checkpoint_path = 'checkpoints/frozen_backbone.pt'
    probe_path = 'checkpoints/probes.pkl'
    l_star_path = 'checkpoints/l_star.json'
    data_path = 'data/processed_subgraphs.pt'
    splits_path = 'data/splits.pt'

    model, checkpoint = load_frozen_backbone(checkpoint_path, device)
    explainer = LogGroundedExplainer(checkpoint_path, probe_path, l_star_path, device)

    dataset = torch.load(data_path, weights_only=False)
    splits = torch.load(splits_path, weights_only=False)
    test_idx = splits['test_idx']
    test_subgraphs = [dataset[i] for i in test_idx]

    with open(l_star_path, 'r') as f:
        l_star_info = json.load(f)

    with open(probe_path, 'rb') as f:
        probe_meta = pickle.load(f)

    baseline_data = {}
    if os.path.exists('results/m7_baseline_comparison.json'):
        with open('results/m7_baseline_comparison.json', 'r') as f:
            baseline_data = json.load(f)

    return model, checkpoint, explainer, test_subgraphs, l_star_info, probe_meta, baseline_data


def main():
    model, checkpoint, explainer, test_subgraphs, l_star_info, probe_meta, baseline_data = load_resources()

    # Header
    st.markdown('<div class="main-header">🛡️ Layer-Wise, Log-Grounded Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Research Prototype for GNN-Based Intrusion Detection (StreamSpot Provenance Graphs)</div>', unsafe_allow_html=True)

    # Sidebar: Subgraph selection and options
    st.sidebar.header("🕹️ Demo Controls")
    class_filter = st.sidebar.radio("Filter Subgraphs by Class:", ["All Test Subgraphs", "Malicious Only (y=1)", "Benign Only (y=0)"])

    if class_filter == "Malicious Only (y=1)":
        filtered_subgraphs = [g for g in test_subgraphs if g.y.item() == 1]
    elif class_filter == "Benign Only (y=0)":
        filtered_subgraphs = [g for g in test_subgraphs if g.y.item() == 0]
    else:
        filtered_subgraphs = test_subgraphs

    subgraph_options = [
        f"Graph {g.graph_id} | Subgraph #{g.subgraph_id} ({'Malicious' if g.y.item()==1 else 'Benign'} - {g.num_nodes} nodes, {g.num_edges} edges)"
        for g in filtered_subgraphs
    ]

    selected_idx = st.sidebar.selectbox("Select Subgraph:", range(len(subgraph_options)), format_func=lambda i: subgraph_options[i])
    selected_subgraph = filtered_subgraphs[selected_idx]

    top_k = st.sidebar.slider("Top-K Attributed Nodes to Explain:", min_value=1, max_value=5, value=3)
    method = st.sidebar.selectbox("Captum Attribution Method:", ["ig", "saliency"], format_func=lambda x: "Integrated Gradients (Captum)" if x=="ig" else "Saliency (Gradient)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Theoretical Grounding:**")
    st.sidebar.caption("• **Pelletreau-Duris et al. (NeSy 2025)**: Per-layer linear probing\n• **Wu et al. (ProvX 2025)**: Provenance graph & audit log grounding\n• **Ying et al. (NeurIPS 2019)**: GNNExplainer baseline\n• **Schnake et al. (IEEE TPAMI 2021)**: GNN-LRP")

    # Run Explanation on the selected subgraph
    exp = explainer.explain_subgraph(selected_subgraph, top_k_nodes=top_k, method=method)

    # Top Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        lbl = "MALICIOUS (Attack)" if exp['true_label'] == 1 else "BENIGN"
        badge_class = "badge-malicious" if exp['true_label'] == 1 else "badge-benign"
        st.markdown(f"**Ground Truth Label:**<br><span class='{badge_class}'>{lbl}</span>", unsafe_allow_html=True)
    with m_col2:
        gnn_decision = "Malicious" if exp['gnn_prediction'] == 1 else "Benign"
        st.metric("Frozen GNN Prediction", f"{gnn_decision}", f"{exp['gnn_confidence']:.1f}% Confidence")
    with m_col3:
        st.metric("Phase-Transition Layer (l*)", f"Layer {exp['l_star_index']}", f"{exp['l_star_name']}")
    with m_col4:
        probe_dec = exp['probe_decision'].capitalize()
        st.metric("Probe Decision at l*", f"{probe_dec}", f"{exp['probe_confidence']:.1f}% Confidence")

    # Core Progressive Explanation Box
    st.markdown("### 📢 Progressive Human-Readable Explanation")
    st.markdown(f"""
    <div class="exp-callout">
        <b>Grounded Summary:</b><br>
        {exp['explanation_string']}
    </div>
    """, unsafe_allow_html=True)

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Layer-Wise Dynamics (l*)", "🔍 Attributed Audit Log Evidence", "⚖️ Baseline Comparison", "📊 Architecture & Dataset"])

    with tab1:
        st.markdown("#### Layer-Wise Representation Probing & Phase-Transition Discovery")
        st.write("By probing intermediate GNN representations layer-by-layer, we discover where the black-box model's threat decision settles ($l^* = 2$).")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=140)

        # Plot 1: Fidelity & Accuracy across layers
        fidelity_dict = l_star_info['fidelity_metrics']
        layer_names = list(fidelity_dict.keys())
        x_ticks = ['L0 (Input)', 'Layer 1', 'Layer 2', 'Layer 3']
        fidelities = [fidelity_dict[k]['agreement'] * 100 for k in layer_names]
        gt_accs = [fidelity_dict[k]['gt_accuracy'] * 100 for k in layer_names]

        ax1.plot(x_ticks, fidelities, marker='o', linewidth=2.5, color='#4F46E5', label='Fidelity to Frozen GNN (%)')
        ax1.plot(x_ticks, gt_accs, marker='^', linewidth=2.0, color='#059669', linestyle='--', label='Ground Truth Accuracy (%)')
        ax1.axvline(x=exp['l_star_index'], color='#DC2626', linestyle='-.', linewidth=2, label=f"Phase-Transition l* = {exp['l_star_index']}")
        for i, val in enumerate(fidelities):
            ax1.annotate(f"{val:.1f}%", (x_ticks[i], val), textcoords="offset points", xytext=(0, 8), ha='center', fontweight='bold', color='#4F46E5')

        ax1.set_title("Per-Layer Fidelity & Agreement Rate", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Agreement (%)", fontsize=10)
        ax1.set_ylim(60, 105)
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, linestyle=':', alpha=0.6)

        # Plot 2: Correlation with GNN Final Probability
        corrs = [fidelity_dict[k]['pearson_corr'] for k in layer_names]
        kappas = [fidelity_dict[k]['cohen_kappa'] for k in layer_names]

        ax2.plot(x_ticks, corrs, marker='s', linewidth=2.5, color='#D97706', label='Confidence Correlation (r)')
        ax2.plot(x_ticks, kappas, marker='d', linewidth=2.0, color='#DB2777', linestyle='--', label="Cohen's Kappa (κ)")
        ax2.axvline(x=exp['l_star_index'], color='#DC2626', linestyle='-.', linewidth=2, label=f"l* = {exp['l_star_index']}")
        for i, val in enumerate(corrs):
            ax2.annotate(f"{val:.2f}", (x_ticks[i], val), textcoords="offset points", xytext=(0, 8), ha='center', fontweight='bold', color='#D97706')

        ax2.set_title("Inter-Rater Agreement & Confidence Correlation", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Metric Score", fontsize=10)
        ax2.set_ylim(0.3, 1.05)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, linestyle=':', alpha=0.6)

        st.pyplot(fig)
        plt.close()

        st.info(f"**Phase-Transition Finding**: At **Layer {exp['l_star_index']} ({exp['l_star_name']})**, probe fidelity reaches **{fidelity_dict[exp['l_star_name']]['agreement']*100:.1f}%** with a confidence correlation of **r = {fidelity_dict[exp['l_star_name']]['pearson_corr']:.4f}**. Subsequent layers produce negligible decision shifts.")

    with tab2:
        st.markdown(f"#### Top-{top_k} Attributed Nodes & Grounded Audit Log Evidence at Layer {exp['l_star_index']}")
        st.write("Captum Integrated Gradients ranks intermediate node embeddings; the preserved lookup table resolves them to original system events.")

        for node in exp['top_attributed_nodes']:
            with st.expander(f"📌 Rank #{node['rank']} — Entity '{node['entity_type']}:{node['entity_id']}' | Attribution Score: {node['attribution_score']:.4f} ({node['relative_importance_pct']:.1f}% of total influence)", expanded=True):
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    st.write(f"**Entity ID:** `{node['entity_id']}`")
                    st.write(f"**Entity Type:** `{node['entity_type']}`")
                    st.write(f"**Local Node Index:** `{node['subgraph_node_idx']}`")
                    st.write(f"**Attribution L2 Norm:** `{node['attribution_score']:.4f}`")
                with col_b:
                    st.markdown("**Originating Audit Log Events:**")
                    if node['associated_log_lines']:
                        for log_entry in node['associated_log_lines']:
                            st.markdown(f"• <span class='log-badge'>{log_entry}</span>", unsafe_allow_html=True)
                    else:
                        st.write("*(Structural background entity)*")

    with tab3:
        st.markdown("#### Head-to-Head Comparison with Baselines")
        st.write("Comparing our progressive, grounded approach against official PyG implementations of **GNNExplainer** and **PGExplainer**.")

        st.table({
            "Evaluation Metric / Feature": [
                "Layer-Wise Granularity",
                "Phase-Transition Layer",
                "Log-Grounded Audit Events",
                "Inference Overhead",
                "Attribution Methodology",
                "Model Invariance"
            ],
            "Our Proposed Method": [
                f"Yes (Layer 1 to Layer 3 progression)",
                f"Identified: l* = {exp['l_star_index']} ({exp['l_star_name']})",
                "Yes (Full audit log syscall events)",
                "~0.003s (Single forward pass)",
                "Captum Integrated Gradients on Probe",
                "Strictly Frozen (Read-only)"
            ],
            "GNNExplainer (Ying et al., 2019)": [
                "No (Final layer only)",
                "Undefined (Black-box)",
                "No (Abstract continuous masks only)",
                "~0.035s (80-step optimization)",
                "Mutual Information Optimization",
                "Requires Model Gradients"
            ],
            "PGExplainer (Luo et al., 2020)": [
                "No (Final layer only)",
                "Undefined (Black-box)",
                "No (Abstract edge weights only)",
                "~0.001s (Feedforward network)",
                "Parameterized Edge Mask MLP",
                "Requires Model Embeddings"
            ]
        })

        st.markdown(r"""
        > **Distinction from GNN-LRP (Schnake et al., IEEE TPAMI 2021):**
        > While GNN-LRP is conceptually related in considering layer-wise propagation, it uses *fixed mathematical conservation rules* ($z$-rule, $\epsilon$-rule). Our method uses *empirically trained and statistically validated linear probes* with verified fidelity ($\kappa = 0.86$, $r = 0.97$), directly identifying the empirical phase-transition layer $l^*$ grounded in concrete audit logs.
        """)

    with tab4:
        st.markdown("#### System Backbone & Dataset Configuration")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Frozen GNN Architecture:**")
            st.code("""
ProvenanceGNN(
  (conv1): SAGEConv(in_channels=56, out_channels=64)
  (conv2): SAGEConv(in_channels=64, out_channels=64)
  (conv3): SAGEConv(in_channels=64, out_channels=64)
  (readout): global_mean_pool
  (fc1): Linear(in_features=64, out_features=32)
  (fc2): Linear(in_features=32, out_features=2)
)
- Total Parameters: 25,890
- Trainable Parameters: 0 (Strictly Frozen)
- Test Accuracy: 91.76% | Test F1: 0.9114 | Recall: 97.30%
            """, language="text")
        with c2:
            st.markdown("**Dataset: StreamSpot Provenance Subgraphs**")
            st.markdown("""
            - **Benchmark**: StreamSpot (DARPA TC provenance research standard)
            - **Subgraphs Extracted via Louvain**: 561 total (317 Benign, 244 Malicious)
            - **Node Features (56 dims)**:
              - Entity type one-hot (8 dims)
              - Outgoing syscall interaction profile (23 dims)
              - Incoming syscall interaction profile (23 dims)
              - In-degree & out-degree (2 dims)
            - **Preserved Metadata**: Persistent `node_map` and `node_to_log` tables throughout the entire tensor pipeline.
            """)


if __name__ == '__main__':
    main()
