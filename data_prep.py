"""
data_prep.py - Provenance Graph Ingestion & Subgraph Partitioning

Theoretical Grounding:
- Wu et al., "ProvX: Toward Explainable Threat Detection on Provenance Graphs", arXiv:2508.06073, 2025:
  Provenance graph construction pipeline, Louvain-based community subgraph extraction,
  and node->log-entry grounding convention.
- StreamSpot dataset preprocessing conventions adapted from FDUDSDE/MAGIC (USENIX Security '24).
"""

import os
import tarfile
import random
import pickle
from typing import Dict, List, Tuple, Any

import torch
import networkx as nx
import community as community_louvain
from torch_geometric.data import Data
from tqdm import tqdm


# StreamSpot entity and interaction mappings
NODE_TYPE_MAP = {
    'a': 'process',
    'b': 'thread',
    'c': 'file',
    'd': 'MAP_ANONYMOUS',
    'e': 'NA',
    'f': 'stdin',
    'g': 'stdout',
    'h': 'stderr',
}
NODE_TYPES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPES)}

EDGE_TYPE_MAP = {
    'i': 'accept',
    'j': 'access',
    'k': 'bind',
    'l': 'chmod',
    'm': 'clone',
    'n': 'close',
    'o': 'connect',
    'p': 'execve',
    'q': 'fstat',
    'r': 'ftruncate',
    's': 'listen',
    't': 'mmap2',
    'u': 'open',
    'v': 'read',
    'w': 'recv',
    'x': 'recvfrom',
    'y': 'recvmsg',
    'z': 'send',
    'A': 'sendmsg',
    'B': 'sendto',
    'C': 'stat',
    'D': 'truncate',
    'E': 'unlink',
    'F': 'waitpid',
    'G': 'write',
    'H': 'writev',
}
EDGE_TYPES = ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 't', 'u', 'v', 'w', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
EDGE_TYPE_TO_IDX = {t: i for i, t in enumerate(EDGE_TYPES)}


def extract_louvain_subgraphs(
    g: nx.DiGraph,
    graph_id: int,
    label: int,
    min_nodes: int = 15,
    max_nodes: int = 200,
    max_subgraphs_per_graph: int = 8,
    random_seed: int = 42
) -> List[Data]:
    """
    Partitions a large provenance graph into cohesive subgraphs via Louvain
    community detection (Wu et al., ProvX 2025) while preserving the exact
    node-ID -> log-line lookup table for log-grounded explainability.
    """
    if g.number_of_nodes() < min_nodes:
        return []

    undirected = g.to_undirected()
    try:
        partition = community_louvain.best_partition(undirected, random_state=random_seed)
    except Exception:
        return []

    communities: Dict[int, List[str]] = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    subgraphs: List[Data] = []
    
    # Sort communities by size
    sorted_comm_ids = sorted(communities.keys(), key=lambda c: len(communities[c]), reverse=True)
    
    candidate_node_sets: List[List[str]] = []
    for cid in sorted_comm_ids:
        nodes = communities[cid]
        if min_nodes <= len(nodes) <= max_nodes:
            candidate_node_sets.append(nodes)
        elif len(nodes) > max_nodes:
            # Hierarchical Louvain partition for oversized communities
            sub_g = undirected.subgraph(nodes)
            try:
                sub_part = community_louvain.best_partition(sub_g, random_state=random_seed)
                sub_comms: Dict[int, List[str]] = {}
                for snode, scid in sub_part.items():
                    sub_comms.setdefault(scid, []).append(snode)
                for scid, snodes in sub_comms.items():
                    if min_nodes <= len(snodes) <= max_nodes:
                        candidate_node_sets.append(snodes)
            except Exception:
                pass

    # Shuffle candidates to get diverse coverage
    rng = random.Random(random_seed + graph_id)
    rng.shuffle(candidate_node_sets)

    for sub_idx, nodes in enumerate(candidate_node_sets[:max_subgraphs_per_graph]):
        sub_digraph = g.subgraph(nodes)
        if sub_digraph.number_of_edges() < 5:
            continue

        node_list = list(sub_digraph.nodes())
        node_to_local = {node_id: idx for idx, node_id in enumerate(node_list)}

        # Edge index and edge features (one-hot of edge types)
        edges = list(sub_digraph.edges(data=True))
        edge_index_list: List[Tuple[int, int]] = []
        edge_attr_list: List[int] = []

        # Preserved node_id -> original log line mapping
        node_to_log: Dict[int, List[str]] = {i: [] for i in range(len(node_list))}
        out_event_counts = torch.zeros((len(node_list), len(EDGE_TYPES)), dtype=torch.float)
        in_event_counts = torch.zeros((len(node_list), len(EDGE_TYPES)), dtype=torch.float)

        for u, v, d in edges:
            src_idx = node_to_local[u]
            dst_idx = node_to_local[v]
            edge_index_list.append((src_idx, dst_idx))
            etype = d.get('type', 'v')
            etype_idx = EDGE_TYPE_TO_IDX.get(etype, 0)
            edge_attr_list.append(etype_idx)

            out_event_counts[src_idx, etype_idx] += 1.0
            in_event_counts[dst_idx, etype_idx] += 1.0

            log_desc = d.get('desc', '')
            if log_desc:
                node_to_log[src_idx].append(log_desc)
                node_to_log[dst_idx].append(log_desc)

        if not edge_index_list:
            continue

        # Construct comprehensive provenance node features:
        # [Entity Type (8) | Outgoing Syscalls (23) | Incoming Syscalls (23) | In-Degree (1) | Out-Degree (1)] = 56 dims
        type_onehot = torch.zeros((len(node_list), len(NODE_TYPES)), dtype=torch.float)
        for idx, node_id in enumerate(node_list):
            ntype = sub_digraph.nodes[node_id].get('type', 'a')
            t_idx = NODE_TYPE_TO_IDX.get(ntype, 0)
            type_onehot[idx, t_idx] = 1.0

        in_deg = in_event_counts.sum(dim=-1, keepdim=True)
        out_deg = out_event_counts.sum(dim=-1, keepdim=True)

        x = torch.cat([type_onehot, out_event_counts, in_event_counts, in_deg, out_deg], dim=-1)

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.zeros((len(edge_attr_list), len(EDGE_TYPES)), dtype=torch.float)
        for i, etype_idx in enumerate(edge_attr_list):
            edge_attr[i, etype_idx] = 1.0

        y = torch.tensor([label], dtype=torch.long)

        pyg_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )
        # Store metadata attributes directly on Data object
        pyg_data.graph_id = graph_id
        pyg_data.subgraph_id = sub_idx
        pyg_data.node_map = node_list  # local idx -> original entity ID
        pyg_data.node_to_log = node_to_log  # local idx -> list of human-readable log strings

        subgraphs.append(pyg_data)

    return subgraphs


def build_streamspot_dataset(
    tar_path: str = 'data/sbustreamspot-data/all.tar.gz',
    output_path: str = 'data/processed_subgraphs.pt',
    num_benign_graphs: int = 50,
    num_attack_graphs: int = 50,
    min_nodes: int = 15,
    max_nodes: int = 200,
    seed: int = 42
) -> List[Data]:
    """
    Parses StreamSpot audit logs from all.tar.gz, constructs provenance graphs,
    and extracts Louvain subgraphs for benign and attack scenarios.
    Preserves node_id -> log_line lookup table throughout.
    """
    print("=" * 60)
    print("STREAMSPOT PROVENANCE GRAPH INGESTION & SUBGRAPH EXTRACTION")
    print("Theoretical Basis: Wu et al. (ProvX 2025), FDUDSDE/MAGIC (2024)")
    print("=" * 60)

    # Select balanced scenarios:
    # 0-99: YouTube, 100-199: GMail, 200-299: VGame, 400-499: Download, 500-599: CNN
    # 300-399: Attack scenarios (Drive-by-download)
    benign_gids = set()
    per_scenario = num_benign_graphs // 5
    for start in [0, 100, 200, 400, 500]:
        benign_gids.update(range(start, start + per_scenario))
    
    attack_gids = set(range(300, 300 + num_attack_graphs))
    target_gids = benign_gids.union(attack_gids)

    print(f"Target graphs: {len(benign_gids)} Benign scenarios, {len(attack_gids)} Attack scenarios.")
    print(f"Streaming from: {tar_path}")

    current_gid = -1
    current_graph = nx.DiGraph()
    all_subgraphs: List[Data] = []
    processed_count = 0

    with tarfile.open(tar_path, 'r:gz') as tar:
        f = tar.extractfile('all.tsv')
        line_num = 0
        pbar = tqdm(total=len(target_gids), desc="Processing Provenance Graphs")

        for line_bytes in f:
            line_num += 1
            parts = line_bytes.decode('utf-8').strip().split('\t')
            gid = int(parts[5])

            if gid > max(target_gids) and current_gid in target_gids:
                # Process the last active graph
                lbl = 1 if (300 <= current_gid <= 399) else 0
                subs = extract_louvain_subgraphs(
                    current_graph, current_gid, lbl, min_nodes, max_nodes, random_seed=seed
                )
                all_subgraphs.extend(subs)
                pbar.update(1)
                break

            if gid != current_gid:
                if current_gid in target_gids:
                    lbl = 1 if (300 <= current_gid <= 399) else 0
                    subs = extract_louvain_subgraphs(
                        current_graph, current_gid, lbl, min_nodes, max_nodes, random_seed=seed
                    )
                    all_subgraphs.extend(subs)
                    pbar.update(1)
                    processed_count += 1

                current_gid = gid
                current_graph = nx.DiGraph()

            if gid in target_gids:
                src, stype, dst, dtype, etype = parts[0], parts[1], parts[2], parts[3], parts[4]
                s_name = NODE_TYPE_MAP.get(stype, stype)
                d_name = NODE_TYPE_MAP.get(dtype, dtype)
                e_name = EDGE_TYPE_MAP.get(etype, etype)
                log_desc = f"[LogLine #{line_num}] {s_name}(id:{src}) --{e_name}--> {d_name}(id:{dst})"

                current_graph.add_node(src, type=stype)
                current_graph.add_node(dst, type=dtype)
                current_graph.add_edge(src, dst, type=etype, desc=log_desc, line_num=line_num)

        pbar.close()

    benign_count = sum(1 for d in all_subgraphs if d.y.item() == 0)
    attack_count = sum(1 for d in all_subgraphs if d.y.item() == 1)

    print("\nDataset Extraction Summary:")
    print(f"Total Subgraphs Extracted: {len(all_subgraphs)}")
    print(f"  - Benign Subgraphs (y=0):   {benign_count}")
    print(f"  - Malicious Subgraphs (y=1): {attack_count}")

    if all_subgraphs:
        avg_nodes = sum(d.num_nodes for d in all_subgraphs) / len(all_subgraphs)
        avg_edges = sum(d.num_edges for d in all_subgraphs) / len(all_subgraphs)
        print(f"  - Avg Nodes per Subgraph:  {avg_nodes:.1f}")
        print(f"  - Avg Edges per Subgraph:  {avg_edges:.1f}")

        # Verify mapping preservation on sample
        sample = all_subgraphs[0]
        sample_logs = sample.node_to_log.get(0, [])
        print(f"\n[Verification] Sample node 0 log entries preserved: {len(sample_logs)}")
        if sample_logs:
            print(f"  Sample entry: {sample_logs[0]}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(all_subgraphs, output_path)
    print(f"Saved dataset to: {output_path}")

    return all_subgraphs


if __name__ == '__main__':
    build_streamspot_dataset(
        num_benign_graphs=40,
        num_attack_graphs=40,
        min_nodes=15,
        max_nodes=150,
        output_path='data/processed_subgraphs.pt'
    )
