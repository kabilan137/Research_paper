"""
app.py - Interactive Streamlit Demo Dashboard

Research Prototype for:
"Layer-Wise, Log-Grounded Explainability for GNN-Based Threat Detection"

Dual-Benchmark Architecture:
1. DARPA TC E3 (Cadets Scenario) - Enterprise host audit benchmark (Drakon APT C2 & Nginx backdoor)
2. StreamSpot - Web-browsing provenance graph benchmark (Drive-by downloads & Flash exploits)

Features:
- Dataset selection (DARPA Cadets vs. StreamSpot)
- Subgraph selector (benign vs. malicious)
- Layer-wise probe fidelity trajectories and phase-transition layer l*
- Top-K attributed entities computed via Captum Integrated Gradients
- Grounded system audit log evidence resolution
- Head-to-head baseline comparison against GNNExplainer & PGExplainer
- Probability of Necessity (PN) / Prediction flip-rate analysis
- Cross-dataset side-by-side comparison (ProvX benchmark metrics)
"""

import os
import json
import pickle
from typing import Dict, List, Any, Tuple

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from extract_embeddings import LayerActivationExtractor, load_frozen_backbone
from attribution import LogGroundedExplainer
from data_prep import NODE_TYPE_MAP, EDGE_TYPE_MAP


st.set_page_config(
    page_title="Layer-Wise GNN Threat Explainability | DARPA & StreamSpot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern research demo aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .exp-callout {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        border-left: 5px solid #4F46E5;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 1.02rem;
        line-height: 1.6;
        color: #1E1B4B;
        margin-top: 12px;
        margin-bottom: 20px;
    }
    .log-badge {
        background-color: #F1F5F9;
        color: #0F172A;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.86rem;
        display: inline-block;
        margin-bottom: 4px;
    }
    .badge-malicious {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 4px;
        display: inline-block;
    }
    .badge-benign {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 4px;
        display: inline-block;
    }
    .dataset-badge {
        background-color: #EDE9FE;
        color: #5B21B6;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 4px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


DATASET_CONFIGS = {
    'darpa_cadets': {
        'name': 'DARPA TC E3 (Cadets Scenario)',
        'benchmark_role': 'Primary Benchmark (Literature Standard)',
        'description': 'Enterprise host provenance audit (FreeBSD OpenBSM / Drakon APT C2 & Nginx backdoor)',
        'checkpoint_path': 'checkpoints/darpa_cadets/frozen_backbone.pt',
        'probe_path': 'checkpoints/darpa_cadets/probes.pkl',
        'l_star_path': 'checkpoints/darpa_cadets/l_star.json',
        'data_path': 'data/darpa_cadets/processed_subgraphs.pt',
        'splits_path': 'data/darpa_cadets/splits.pt',
        'baseline_json': 'results/darpa_cadets/m7_baseline_comparison.json',
        'fidelity_img': 'results/darpa_cadets/m5_fidelity_agreement.png',
        'input_dim': 62,
        'edge_dim': 27,
        'subgraph_count': '400 subgraphs (200 Benign, 200 Malicious)',
        'detection_metrics': {
            'Accuracy': '95.00%',
            'Precision': '90.91%',
            'Recall': '100.00%',
            'F1-Score': '95.24%',
            'ROC-AUC': '0.9978',
            'FPR': '10.00%'
        },
        'l_star_nominal': 1,
        'source_authority': 'DARPA Transparent Computing / USENIX Security \'25 / ProHunter'
    },
    'streamspot': {
        'name': 'StreamSpot',
        'benchmark_role': 'Baseline Benchmark',
        'description': 'Web-browsing provenance graph benchmark (Drive-by downloads & Flash exploits)',
        'checkpoint_path': 'checkpoints/frozen_backbone.pt',
        'probe_path': 'checkpoints/probes.pkl',
        'l_star_path': 'checkpoints/l_star.json',
        'data_path': 'data/processed_subgraphs.pt',
        'splits_path': 'data/splits.pt',
        'baseline_json': 'results/m7_baseline_comparison.json',
        'fidelity_img': 'results/m5_fidelity_agreement.png',
        'input_dim': 56,
        'edge_dim': 23,
        'subgraph_count': '561 subgraphs (317 Benign, 244 Malicious)',
        'detection_metrics': {
            'Accuracy': '91.76%',
            'Precision': '85.71%',
            'Recall': '97.30%',
            'F1-Score': '91.14%',
            'ROC-AUC': '0.9840',
            'FPR': '12.50%'
        },
        'l_star_nominal': 2,
        'source_authority': 'StreamSpot (Manzoor et al., IEEE TKDE 2016)'
    }
}


@st.cache_resource
def load_dataset_resources(dataset_key: str):
    """Caches model, explainer, test cohort, and baseline data for the specified dataset."""
    cfg = DATASET_CONFIGS[dataset_key]
    device = torch.device('cpu')

    model, checkpoint = load_frozen_backbone(cfg['checkpoint_path'], device)
    explainer = LogGroundedExplainer(
        checkpoint_path=cfg['checkpoint_path'],
        probe_checkpoint_path=cfg['probe_path'],
        l_star_path=cfg['l_star_path'],
        device=device
    )

    dataset = torch.load(cfg['data_path'], weights_only=False)
    splits = torch.load(cfg['splits_path'], weights_only=False)
    test_idx = splits['test_idx']
    test_subgraphs = [dataset[i] for i in test_idx]

    with open(cfg['l_star_path'], 'r') as f:
        l_star_info = json.load(f)

    with open(cfg['probe_path'], 'rb') as f:
        probe_meta = pickle.load(f)

    baseline_data = {}
    if os.path.exists(cfg['baseline_json']):
        with open(cfg['baseline_json'], 'r') as f:
            baseline_data = json.load(f)

    return model, checkpoint, explainer, test_subgraphs, l_star_info, probe_meta, baseline_data


def main():
    # Sidebar: Benchmark Selection
    st.sidebar.title("🛡️ Benchmark Controls")
    dataset_choice = st.sidebar.selectbox(
        "Active Evaluation Dataset:",
        options=['darpa_cadets', 'streamspot'],
        format_func=lambda k: f"{'⭐ ' if k=='darpa_cadets' else ''}{DATASET_CONFIGS[k]['name']} ({'Benchmark' if k=='darpa_cadets' else 'Baseline'})"
    )

    cfg = DATASET_CONFIGS[dataset_choice]
    model, checkpoint, explainer, test_subgraphs, l_star_info, probe_meta, baseline_data = load_dataset_resources(dataset_choice)

    # Header
    st.markdown('<div class="main-header">🛡️ Layer-Wise, Log-Grounded Explainability</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Evaluating GNN Threat Detection on <span class="dataset-badge">{cfg["name"]}</span> — {cfg["description"]}</div>', unsafe_allow_html=True)

    # Sidebar: Subgraph selection and options
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔎 Subgraph Exploration")

    class_filter = st.sidebar.radio("Filter Test Subgraphs:", ["All Test Subgraphs", "Malicious Only (y=1)", "Benign Only (y=0)"])

    if class_filter == "Malicious Only (y=1)":
        filtered_subgraphs = [g for g in test_subgraphs if g.y.item() == 1]
    elif class_filter == "Benign Only (y=0)":
        filtered_subgraphs = [g for g in test_subgraphs if g.y.item() == 0]
    else:
        filtered_subgraphs = test_subgraphs

    def format_subgraph_label(idx: int) -> str:
        g = filtered_subgraphs[idx]
        lbl = "Malicious" if g.y.item() == 1 else "Benign"
        gid = getattr(g, 'graph_id', 'G')
        sid = getattr(g, 'subgraph_id', idx)
        # Check for prominent entity in Cadets
        extra = ""
        if hasattr(g, 'raw_attributes') and g.y.item() == 1:
            for attr in g.raw_attributes.values():
                nm = attr.get('name', '')
                if any(x in nm for x in ['/tmp/test', '/tmp/minions', '/tmp/main', '128.55']):
                    extra = f" | {nm}"
                    break
        return f"Graph {gid} #Sub {sid} [{lbl}] ({g.num_nodes} nodes, {g.num_edges} edges{extra})"

    selected_idx = st.sidebar.selectbox(
        "Select Subgraph Sample:",
        range(len(filtered_subgraphs)),
        format_func=format_subgraph_label
    )
    selected_subgraph = filtered_subgraphs[selected_idx]

    top_k = st.sidebar.slider("Top-K Attributed Nodes to Explain:", min_value=1, max_value=5, value=3)
    method = st.sidebar.selectbox(
        "Captum Attribution Method:",
        ["ig", "saliency"],
        format_func=lambda x: "Integrated Gradients (Captum)" if x == "ig" else "Saliency (Vanilla Gradient)"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Theoretical Grounding:**")
    st.sidebar.caption(
        "• **Pelletreau-Duris et al. (NeSy 2025)**: Layer-wise representation probing\n"
        "• **Wu et al. (ProvX 2025)**: Provenance graph log grounding & ProvX metric set\n"
        "• **Ying et al. (NeurIPS 2019)**: GNNExplainer baseline\n"
        "• **Schnake et al. (IEEE TPAMI 2021)**: GNN-LRP layer relevance"
    )

    # Execute Explanation on the selected subgraph
    exp = explainer.explain_subgraph(selected_subgraph, top_k_nodes=top_k, method=method)

    # Top Metric KPI Row
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

    # Core Progressive Explanation Callout
    st.markdown("### 📢 Progressive Human-Readable Explanation")
    st.markdown(f"""
    <div class="exp-callout">
        <b>Grounded Decision Narrative:</b><br>
        {exp['explanation_string']}
    </div>
    """, unsafe_allow_html=True)

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔬 Layer-Wise Dynamics (l*)",
        "🔍 Attributed Audit Log Evidence",
        "⚖️ Baseline Comparison & PN",
        "⚔️ Cross-Dataset Comparison",
        "📊 Architecture & Dataset"
    ])

    # ==================== TAB 1 ====================
    with tab1:
        st.markdown(f"#### Layer-Wise Representation Probing & Phase-Transition on {cfg['name']}")
        st.write(
            f"By training linear diagnostic probes at each layer of the frozen GraphSAGE backbone, "
            f"we determine the exact message-passing depth at which threat features become linearly separable ($l^* = {exp['l_star_index']}$)."
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4), dpi=140)

        fidelity_dict = l_star_info['fidelity_metrics']
        layer_names = list(fidelity_dict.keys())
        x_ticks = ['L0 (Input)', 'Layer 1', 'Layer 2', 'Layer 3']
        fidelities = [fidelity_dict[k]['agreement'] * 100 for k in layer_names]
        gt_accs = [fidelity_dict[k]['gt_accuracy'] * 100 for k in layer_names]

        # Plot 1: Fidelity & Accuracy
        ax1.plot(x_ticks, fidelities, marker='o', linewidth=2.5, color='#4F46E5', label='Fidelity to Frozen GNN (%)')
        ax1.plot(x_ticks, gt_accs, marker='^', linewidth=2.0, color='#059669', linestyle='--', label='Ground Truth Accuracy (%)')
        ax1.axvline(x=exp['l_star_index'], color='#DC2626', linestyle='-.', linewidth=2, label=f"Phase-Transition l* = {exp['l_star_index']}")
        for i, val in enumerate(fidelities):
            ax1.annotate(f"{val:.1f}%", (x_ticks[i], val), textcoords="offset points", xytext=(0, 7), ha='center', fontweight='bold', color='#4F46E5')

        ax1.set_title(f"Per-Layer Fidelity & Agreement ({cfg['name']})", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Agreement (%)", fontsize=10)
        ax1.set_ylim(60, 105)
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, linestyle=':', alpha=0.6)

        # Plot 2: Correlation & Kappa
        corrs = [fidelity_dict[k]['pearson_corr'] for k in layer_names]
        kappas = [fidelity_dict[k]['cohen_kappa'] for k in layer_names]

        ax2.plot(x_ticks, corrs, marker='s', linewidth=2.5, color='#D97706', label='Confidence Correlation (Pearson r)')
        ax2.plot(x_ticks, kappas, marker='d', linewidth=2.0, color='#DB2777', linestyle='--', label="Inter-Rater Agreement (Cohen's κ)")
        ax2.axvline(x=exp['l_star_index'], color='#DC2626', linestyle='-.', linewidth=2, label=f"l* = {exp['l_star_index']}")
        for i, val in enumerate(corrs):
            ax2.annotate(f"{val:.2f}", (x_ticks[i], val), textcoords="offset points", xytext=(0, 7), ha='center', fontweight='bold', color='#D97706')

        ax2.set_title("Agreement Strength & Probability Correlation", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Score [0, 1]", fontsize=10)
        ax2.set_ylim(0.3, 1.05)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, linestyle=':', alpha=0.6)

        st.pyplot(fig)
        plt.close()

        l_star_acc = fidelity_dict[exp['l_star_name']]['agreement'] * 100
        l_star_r = fidelity_dict[exp['l_star_name']]['pearson_corr']
        l_star_k = fidelity_dict[exp['l_star_name']]['cohen_kappa']

        st.info(
            f"**Phase-Transition Finding ($l^* = {exp['l_star_index']}$)**: "
            f"At **Layer {exp['l_star_index']} ({exp['l_star_name']})**, probe fidelity reaches **{l_star_acc:.1f}%** "
            f"with inter-rater agreement **κ = {l_star_k:.4f}** and probability correlation **r = {l_star_r:.4f}**. "
            f"{'On DARPA Cadets, enterprise APT behaviors are isolated within a single message-passing step (l*=1).' if dataset_choice=='darpa_cadets' else 'On StreamSpot, multi-hop structural context is required before the decision converges (l*=2).'}"
        )

    # ==================== TAB 2 ====================
    with tab2:
        st.markdown(f"#### Top-{top_k} Attributed Entities & Grounded Audit Log Events at Layer {exp['l_star_index']}")
        st.write(
            f"Captum Integrated Gradients is applied to the linear probe at $l^* = {exp['l_star_index']}$ to attribute importance "
            f"to intermediate representations. Attributed nodes are mapped back to concrete provenance audit log events."
        )

        for node in exp['top_attributed_nodes']:
            with st.expander(f"📌 Rank #{node['rank']} — Entity '{node['entity_type']}:{node['entity_id']}' | Attribution Score: {node['attribution_score']:.4f} ({node['relative_importance_pct']:.1f}% influence)", expanded=True):
                col_a, col_b = st.columns([1, 2.8])
                with col_a:
                    st.write(f"**Entity ID / Name:** `{node['entity_id']}`")
                    st.write(f"**Entity Type:** `{node['entity_type']}`")
                    st.write(f"**Subgraph Node Index:** `{node['subgraph_node_idx']}`")
                    st.write(f"**Attribution L2 Norm:** `{node['attribution_score']:.4f}`")
                    st.write(f"**Relative Share:** `{node['relative_importance_pct']:.1f}%`")
                with col_b:
                    st.markdown("**Associated Audit Log Events (Chronological):**")
                    if node['associated_log_lines']:
                        for log_entry in node['associated_log_lines']:
                            st.markdown(f"<span class='log-badge'>{log_entry}</span>", unsafe_allow_html=True)
                    else:
                        st.caption("*(Background structural entity; no directly attributed syscall event)*")

    # ==================== TAB 3 ====================
    with tab3:
        st.markdown("#### Head-to-Head Baseline Comparison & Probability of Necessity (PN)")
        st.write(
            "Comparing our layer-wise, log-grounded approach against official PyTorch Geometric implementations of "
            "**GNNExplainer** (Ying et al., NeurIPS 2019) and **PGExplainer** (Luo et al., NeurIPS 2020)."
        )

        st.subheader("1. Qualitative & Functional Comparison")
        st.table({
            "Feature / Dimension": [
                "Layer-Wise Granularity",
                "Phase-Transition Detection (l*)",
                "Log-Grounded Audit Events",
                "Explanation Output Type",
                "Inference Latency",
                "Backbone Invariance",
                "Empirical Fidelity Validation"
            ],
            "Our Layer-Wise Explainer": [
                f"Yes (Tracks L0 to L3; commits at l* = {exp['l_star_index']})",
                f"Identified: l* = {exp['l_star_index']} ({exp['l_star_name']})",
                "Yes (Named processes, file paths, IP sockets)",
                "Natural language narrative + audit log events",
                "~0.003s (Single forward pass + probe)",
                "Strictly Frozen (Read-only parameter access)",
                f"Verified: {l_star_acc:.1f}% agreement (κ = {l_star_k:.2f})"
            ],
            "GNNExplainer (NeurIPS 2019)": [
                "No (Final layer readout only)",
                "Undefined (Black-box)",
                "No (Abstract continuous mask tensor only)",
                "Sub-graph node/edge saliency heatmaps",
                "~0.026s (40-step mutual-info optimization)",
                "Requires dynamic backpropagation gradients",
                "No (Assumes mask fidelity heuristically)"
            ],
            "PGExplainer (NeurIPS 2020)": [
                "No (Final layer readout only)",
                "Undefined (Black-box)",
                "No (Abstract continuous edge mask only)",
                "Edge probability weights",
                "~0.001s (Feedforward neural pass)",
                "Requires backbone training embeddings",
                "No (Heuristic graph-level objective)"
            ]
        })

        st.subheader("2. Probability of Necessity (PN) / Prediction Flip-Rate")
        st.caption(
            "Evaluation protocol (ProvX / Pelletreau-Duris 2025): Remove the top-K (K=3) critical nodes identified by each "
            "explainer, re-run the frozen GNN, and measure prediction flip-rate and mean output probability drop."
        )

        if 'pn_results' in baseline_data:
            pn = baseline_data['pn_results']
            st.table({
                "Explainer Method": [
                    "Our Method (Probe at l*)",
                    "GNNExplainer (Ying et al. 2019)",
                    "Random Baseline"
                ],
                "Top-K Removed": [f"K = {pn['k_nodes']}"] * 3,
                "PN Flip-Rate (%)": [
                    f"{pn['our_method']['flip_rate']*100:.1f}%",
                    f"{pn['gnn_explainer']['flip_rate']*100:.1f}%",
                    f"{pn['random']['flip_rate']*100:.1f}%"
                ],
                "Flips / Cohort": [
                    f"{pn['our_method']['flip_count']} / {pn['cohort_size']}",
                    f"{pn['gnn_explainer']['flip_count']} / {pn['cohort_size']}",
                    f"{pn['random']['flip_count']} / {pn['cohort_size']}"
                ],
                "Mean Probability Drop (Δp)": [
                    f"{pn['our_method']['mean_prob_drop']:.4f}",
                    f"{pn['gnn_explainer']['mean_prob_drop']:.4f}",
                    f"{pn['random']['mean_prob_drop']:.4f}"
                ]
            })

            st.markdown(r"""
            > **Scientific Insight on Necessity vs. Grounding:**
            > GNNExplainer achieves high flip-rates by aggressively pruning high-degree structural bridge nodes, which disrupts message passing globally but yields ungrounded, abstract node masks.
            > Our method pinpoints the **semantically causative audit events** (e.g., executing `/tmp/test` and reading `/etc/libmap.conf`) while maintaining <5ms latency and producing human-actionable alerts for SOC analysts.
            """)

    # ==================== TAB 4 ====================
    with tab4:
        st.markdown("#### ⚔️ Cross-Dataset Side-by-Side Comparison: DARPA Cadets vs. StreamSpot")
        st.write(
            "Parallel evaluation demonstrating generalization across contrasting provenance paradigms: "
            "Enterprise host audit logs (DARPA Cadets) versus web-browsing execution logs (StreamSpot)."
        )

        st.subheader("1. ProvX Benchmark Performance Comparison")
        st.table({
            "Evaluation Dimension / Metric": [
                "Dataset Domain",
                "Audit Logging Source",
                "Evaluated Subgraphs",
                "Input Node Feature Dims",
                "Edge Attribute Dims",
                "Frozen Backbone Parameters",
                "Detection Accuracy",
                "Precision",
                "Recall (Detection Rate)",
                "F1-Score",
                "ROC-AUC",
                "False Positive Rate (FPR)",
                "Phase-Transition Layer (l*)",
                "Probe Agreement at l*",
                "Inter-Rater Cohen's κ at l*"
            ],
            "DARPA TC E3 (Cadets) [Benchmark]": [
                "Enterprise Linux/FreeBSD Host Audit",
                "OpenBSM Kernel Audit (Drakon APT)",
                "400 (200 Benign, 200 Malicious)",
                "62 dimensions",
                "27 dimensions",
                "26,658 (100% frozen)",
                "95.00%",
                "90.91%",
                "100.00%",
                "95.24%",
                "0.9978",
                "10.00%",
                "Layer 1 (l* = 1)",
                "96.67%",
                "0.9331 (Almost Perfect)"
            ],
            "StreamSpot [Baseline]": [
                "Web-Browsing Execution Logs",
                "Linux System-Call Audit (Flash/Drive-by)",
                "561 (317 Benign, 244 Malicious)",
                "56 dimensions",
                "23 dimensions",
                "25,890 (100% frozen)",
                "91.76%",
                "85.71%",
                "97.30%",
                "91.14%",
                "0.9840",
                "12.50%",
                "Layer 2 (l* = 2)",
                "92.94%",
                "0.8589 (Almost Perfect)"
            ]
        })

        st.subheader("2. Side-by-Side Fidelity Trajectories")
        if os.path.exists('results/side_by_side_fidelity.png'):
            st.image('results/side_by_side_fidelity.png', caption="Comparative Fidelity and Ground Truth Accuracy Trajectories: StreamSpot (Left) vs. DARPA Cadets (Right)")

        st.markdown("""
        ### Key Scientific Finding: Phase-Transition Shift ($l^* = 2 \\rightarrow l^* = 1$)
        - **In StreamSpot ($l^* = 2$)**: Benign browsing and drive-by attacks share similar browser process trees. A 1-hop neighborhood is insufficient to distinguish attack downloads from benign downloads; the model requires **2-hop structural context** ($l^*=2$) for its decision to settle.
        - **In DARPA Cadets ($l^* = 1$)**: Enterprise APT attacks (such as the Drakon C2 dropper and Nginx backdoor) execute distinct anomalous sequences (e.g. `/tmp/test` calling `/libexec/ld-elf.so.1` and reading `/etc/libmap.conf`). The immediate **1-hop message passing** ($l^*=1$) creates distinct linear separability, where agreement reaches 96.67% and Cohen's $\\kappa = 0.933$.
        """)

    # ==================== TAB 5 ====================
    with tab5:
        st.markdown(f"#### 📊 System Backbone & Dataset Configuration ({cfg['name']})")
        col_arch, col_data = st.columns(2)

        with col_arch:
            st.markdown(f"**Frozen 3-Layer GraphSAGE Architecture ({cfg['name']}):**")
            st.code(f"""
ProvenanceGNN(
  (conv1): SAGEConv(in_channels={cfg['input_dim']}, out_channels=64)
  (conv2): SAGEConv(in_channels=64, out_channels=64)
  (conv3): SAGEConv(in_channels=64, out_channels=64)
  (readout): global_mean_pool
  (fc1): Linear(in_features=64, out_features=32)
  (fc2): Linear(in_features=32, out_features=2)
)
- Input Feature Dimension: {cfg['input_dim']}
- Intermediate Channels: 64
- Total Parameters: {checkpoint['total_params'] if 'total_params' in checkpoint else '26,658'}
- Trainable Parameters: 0 (Strictly Frozen requires_grad=False)
- Device Evaluated: Apple Silicon GPU (MPS) / CPU
            """, language="text")

        with col_data:
            st.markdown(f"**Dataset & Provenance Graph Pipeline:**")
            st.markdown(f"""
            - **Benchmark Suite**: {cfg['name']}
            - **Literature Authority**: {cfg['source_authority']}
            - **Evaluation Cohort**: {cfg['subgraph_count']}
            - **Node Representation ({cfg['input_dim']} dims)**:
              - Entity type one-hot encoding
              - Outgoing syscall event interaction profile
              - Incoming syscall event interaction profile
              - In-degree & out-degree structural features
            - **Edge Representation ({cfg['edge_dim']} dims)**: One-hot encoded audit event syscalls.
            - **Log Grounding**: Persistent raw attributes and `node_to_log` lookup mapping intermediate representations directly back to source audit events.
            """)


if __name__ == '__main__':
    main()
