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
import zipfile
import io
import json
import random
import pickle
from typing import Dict, List, Tuple, Any, Optional

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

# Cadets entity and interaction mappings
CADETS_NODE_TYPES = ['process', 'file', 'netflow', 'memory', 'pipe', 'ipc']
CADETS_NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(CADETS_NODE_TYPES)}

CADETS_EDGE_TYPES = [
    'EVENT_CLOSE', 'EVENT_CREATE_OBJECT', 'EVENT_CONNECT', 'EVENT_ACCEPT',
    'EVENT_WRITE', 'EVENT_READ', 'EVENT_OPEN', 'EVENT_EXIT', 'EVENT_FORK',
    'EVENT_EXECUTE', 'EVENT_MMAP', 'EVENT_MODIFY_PROCESS', 'EVENT_LSEEK',
    'EVENT_CHANGE_PRINCIPAL', 'EVENT_SENDTO', 'EVENT_RECVMSG', 'EVENT_RECVFROM',
    'EVENT_UNLINK', 'EVENT_RENAME', 'EVENT_FCNTL', 'EVENT_MODIFY_FILE_ATTRIBUTES',
    'EVENT_LINK', 'EVENT_SIGNAL', 'EVENT_SENDMSG', 'EVENT_FLOWS_TO', 'EVENT_TRUNCATE',
    'EVENT_OTHER'
]
CADETS_EDGE_TYPE_TO_IDX = {t: i for i, t in enumerate(CADETS_EDGE_TYPES)}


class DGLNativeUnpickler(pickle.Unpickler):
    """
    Direct in-memory unpickler for DGL graphs serialized by MAGIC/DGL.
    Deserializes PyTorch tensor storages directly without needing DGL C++ FFI installed.
    """
    def find_class(self, module, name):
        if module.startswith('torch'):
            return super().find_class(module, name)
        if module == 'dgl._ffi.object' and name == '_new_object':
            return lambda cls: cls.__new__(cls)
        if module == 'dgl.frame' and name == 'Scheme':
            class Scheme:
                @classmethod
                def _reconstruct_scheme(cls, shape, dtype_str):
                    s = cls()
                    s.shape = shape
                    s.dtype_str = dtype_str
                    return s
            return Scheme

        class GenericDummy:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
            def __setstate__(self, state):
                self.state = state
            def __call__(self, *args, **kwargs):
                return GenericDummy(*args, **kwargs)
        return GenericDummy


def load_dgl_graph_tensors(zip_path: str, pkl_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loads src, dst, node_type, and edge_type tensors from a zipped DGL graph."""
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(pkl_name)
    u = DGLNativeUnpickler(io.BytesIO(raw))
    obj = u.load()
    st = obj.state
    src = st['_graph'].state.state[2][0]
    dst = st['_graph'].state.state[2][1]
    node_type = st['_node_frames'][0].state['_columns']['type'].state['storage']
    edge_type = st['_edge_frames'][0].state['_columns']['type'].state['storage']
    return src, dst, node_type, edge_type


def build_cadets_subgraph(
    node_list: List[int],
    src_all: torch.Tensor,
    dst_all: torch.Tensor,
    etype_all: torch.Tensor,
    ntype_all: torch.Tensor,
    label: int,
    graph_id: int,
    subgraph_id: int,
    names_map: Dict[str, str],
    types_map: Dict[str, str],
    uuid_map: Dict[int, str],
    mal_nodes_set: set,
    ubc_netflow_map: Dict[str, str] = None
) -> Optional[Data]:
    """Constructs a single PyG Data subgraph with full log-grounding attributes."""
    if len(node_list) < 10:
        return None

    node_set = set(node_list)
    node_to_local = {node_id: idx for idx, node_id in enumerate(node_list)}
    N = len(node_list)

    # Filter internal edges
    node_tensor = torch.tensor(node_list, dtype=torch.long)
    mask_nodes = torch.zeros(ntype_all.size(0), dtype=torch.bool)
    mask_nodes[node_tensor] = True

    # Vectorized edge selection
    edge_mask = mask_nodes[src_all] & mask_nodes[dst_all]
    src_sub = src_all[edge_mask]
    dst_sub = dst_all[edge_mask]
    etype_sub = etype_all[edge_mask]

    E = src_sub.size(0)
    if E < 8:
        return None

    edge_index_list = []
    edge_attr_list = []
    out_event_counts = torch.zeros((N, len(CADETS_EDGE_TYPES)), dtype=torch.float)
    in_event_counts = torch.zeros((N, len(CADETS_EDGE_TYPES)), dtype=torch.float)

    node_to_log: Dict[int, List[str]] = {i: [] for i in range(N)}
    raw_attributes: Dict[int, Dict[str, Any]] = {}

    # Pre-resolve raw attributes for nodes
    for local_idx, orig_id in enumerate(node_list):
        uuid = uuid_map.get(orig_id, f"node_{orig_id}")
        t_code = int(ntype_all[orig_id].item()) if orig_id < ntype_all.size(0) else 0
        t_name = CADETS_NODE_TYPES[min(t_code, len(CADETS_NODE_TYPES) - 1)]

        # Human-readable entity name
        entity_name = names_map.get(uuid, types_map.get(uuid, f"{t_name}_{orig_id}"))
        if ubc_netflow_map and uuid in ubc_netflow_map:
            entity_name = ubc_netflow_map[uuid]

        is_mal = orig_id in mal_nodes_set
        raw_attributes[local_idx] = {
            'orig_id': orig_id,
            'uuid': uuid,
            'type': t_name,
            'name': entity_name,
            'is_malicious': is_mal
        }

    for i in range(E):
        u = int(src_sub[i].item())
        v = int(dst_sub[i].item())
        e = int(etype_sub[i].item())
        e_idx = min(e, len(CADETS_EDGE_TYPES) - 1)

        src_local = node_to_local[u]
        dst_local = node_to_local[v]

        edge_index_list.append((src_local, dst_local))
        edge_attr_list.append(e_idx)

        out_event_counts[src_local, e_idx] += 1.0
        in_event_counts[dst_local, e_idx] += 1.0

        e_name = CADETS_EDGE_TYPES[e_idx]
        s_desc = raw_attributes[src_local]['name']
        s_type = raw_attributes[src_local]['type']
        d_desc = raw_attributes[dst_local]['name']
        d_type = raw_attributes[dst_local]['type']

        log_str = f"[Cadets Event] {s_type}({s_desc}) --{e_name}--> {d_type}({d_desc})"
        node_to_log[src_local].append(log_str)
        node_to_log[dst_local].append(log_str)

    # Construct node features: [Type (6) | Outgoing Events (27) | Incoming Events (27) | In-Deg (1) | Out-Deg (1)] = 62 dims
    type_onehot = torch.zeros((N, len(CADETS_NODE_TYPES)), dtype=torch.float)
    for local_idx, orig_id in enumerate(node_list):
        t_code = int(ntype_all[orig_id].item()) if orig_id < ntype_all.size(0) else 0
        type_onehot[local_idx, min(t_code, len(CADETS_NODE_TYPES) - 1)] = 1.0

    in_deg = in_event_counts.sum(dim=-1, keepdim=True)
    out_deg = out_event_counts.sum(dim=-1, keepdim=True)
    x = torch.cat([type_onehot, out_event_counts, in_event_counts, in_deg, out_deg], dim=-1)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.zeros((len(edge_attr_list), len(CADETS_EDGE_TYPES)), dtype=torch.float)
    for i, e_idx in enumerate(edge_attr_list):
        edge_attr[i, e_idx] = 1.0

    y = torch.tensor([label], dtype=torch.long)

    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )
    pyg_data.graph_id = graph_id
    pyg_data.subgraph_id = subgraph_id
    pyg_data.node_map = [raw_attributes[i]['uuid'] for i in range(N)]
    pyg_data.node_to_log = node_to_log
    pyg_data.raw_attributes = raw_attributes

    return pyg_data


def build_cadets_dataset(
    graphs_zip_path: str = 'data/MAGIC/data/cadets/graphs.zip',
    names_json_path: str = 'data/ProHunter/dataset/darpa_cadets/names.json',
    types_json_path: str = 'data/ProHunter/dataset/darpa_cadets/types.json',
    ubc_zip_path: str = 'data/ubc_ground_truth.zip',
    output_path: str = 'data/darpa_cadets/processed_subgraphs.pt',
    num_malicious: int = 200,
    num_benign: int = 200,
    min_nodes: int = 15,
    max_nodes: int = 150,
    seed: int = 42
) -> List[Data]:
    """
    Extracts cohesive provenance subgraphs from DARPA TC E3 Cadets (benign and attack).
    Preserves exact node-ID -> raw attribute and log-grounding lookup tables.
    """
    print("=" * 70)
    print("DARPA TC E3 CADETS PROVENANCE GRAPH INGESTION & SUBGRAPH EXTRACTION")
    print("Theoretical Basis: ProvX (Wu et al. 2025), MAGIC (Jia et al. USENIX Sec '24)")
    print("=" * 70)

    random.seed(seed)
    torch.manual_seed(seed)

    # 1. Load ground truth metadata
    print("Loading Cadets metadata and ground truth labels...")
    with zipfile.ZipFile(graphs_zip_path) as z:
        meta = json.loads(z.read('metadata.json'))
    mal_nodes_set = set(meta['malicious'][0])
    uuid_map = {idx: u for idx, u in zip(meta['malicious'][0], meta['malicious'][1])}
    print(f"Total Cadets ground-truth malicious nodes: {len(mal_nodes_set)}")

    # 2. Load entity names and types
    print(f"Loading entity name mapping from {names_json_path}...")
    with open(names_json_path, 'r') as f:
        names_map = json.load(f)
    print(f"Total entity names resolved: {len(names_map)}")

    print(f"Loading entity type mapping from {types_json_path}...")
    with open(types_json_path, 'r') as f:
        types_map = json.load(f)

    # 3. Load UBC netflow mappings if available
    ubc_netflow_map = {}
    if os.path.exists(ubc_zip_path):
        with zipfile.ZipFile(ubc_zip_path) as z:
            for fname in z.namelist():
                if 'E3-CADETS' in fname and fname.endswith('.csv'):
                    content = z.read(fname).decode('utf-8')
                    for line in content.strip().split('\n'):
                        if not line.strip(): continue
                        parts = line.strip().split(',', 2)
                        u = parts[0].strip()
                        if len(parts) > 1 and 'netflow' in parts[1]:
                            try:
                                d = eval(parts[1])
                                ubc_netflow_map[u] = d.get('netflow', parts[1])
                            except Exception:
                                pass
        print(f"UBC netflow mappings loaded: {len(ubc_netflow_map)}")

    # 4. Load attack graph (test0)
    print("\nLoading attack provenance graph (test0)...")
    src_test, dst_test, ntype_test, etype_test = load_dgl_graph_tensors(graphs_zip_path, 'test0.pkl')
    print(f"Test graph: {ntype_test.size(0)} nodes, {src_test.size(0)} edges.")

    # 5. Load benign training graph (train0)
    print("Loading benign training provenance graph (train0)...")
    src_train, dst_train, ntype_train, etype_train = load_dgl_graph_tensors(graphs_zip_path, 'train0.pkl')
    print(f"Train graph: {ntype_train.size(0)} nodes, {src_train.size(0)} edges.")

    all_subgraphs: List[Data] = []

    # Fast adjacency builder for BFS subgraph sampling
    def build_fast_adj(src_t, dst_t, limit=300000):
        from collections import defaultdict
        adj = defaultdict(list)
        s_list = src_t[:limit].tolist()
        d_list = dst_t[:limit].tolist()
        for s, d in zip(s_list, d_list):
            adj[s].append(d)
            adj[d].append(s)
        return adj

    def sample_ego_subgraph(adj_dict, seed_n, min_n=15, max_n=140):
        from collections import deque
        visited = {seed_n}
        q = deque([seed_n])
        while q and len(visited) < max_n:
            curr = q.popleft()
            for nbr in adj_dict.get(curr, []):
                if nbr not in visited:
                    visited.add(nbr)
                    q.append(nbr)
                    if len(visited) >= max_n:
                        break
        if len(visited) < min_n:
            return None
        return list(visited)

    print("\nExtracting Malicious Subgraphs around ground-truth attack entities (y=1)...")
    adj_test = build_fast_adj(src_test, dst_test, limit=src_test.size(0))

    # Prioritize seeds with known named entities (/tmp/minions, /tmp/main, etc.)
    named_mal_seeds = [n for n in mal_nodes_set if n in uuid_map and uuid_map[n] in names_map]
    other_mal_seeds = [n for n in mal_nodes_set if n not in named_mal_seeds]
    random.shuffle(named_mal_seeds)
    random.shuffle(other_mal_seeds)
    candidate_mal_seeds = named_mal_seeds * 5 + other_mal_seeds

    mal_subgraphs_count = 0
    sub_id = 0
    pbar_mal = tqdm(total=num_malicious, desc="Malicious Subgraphs")

    used_seed_sets = set()
    for seed_node in candidate_mal_seeds:
        if mal_subgraphs_count >= num_malicious:
            break
        nodes = sample_ego_subgraph(adj_test, seed_node, min_nodes, max_nodes)
        if not nodes:
            continue
        # Verify contains at least 1 malicious node
        if not any(n in mal_nodes_set for n in nodes):
            continue

        # Prevent duplicate identical subgraphs
        froz = frozenset(nodes)
        if froz in used_seed_sets:
            continue
        used_seed_sets.add(froz)

        pyg_data = build_cadets_subgraph(
            nodes, src_test, dst_test, etype_test, ntype_test,
            label=1, graph_id=0, subgraph_id=sub_id,
            names_map=names_map, types_map=types_map, uuid_map=uuid_map,
            mal_nodes_set=mal_nodes_set, ubc_netflow_map=ubc_netflow_map
        )
        if pyg_data is not None:
            all_subgraphs.append(pyg_data)
            mal_subgraphs_count += 1
            sub_id += 1
            pbar_mal.update(1)
    pbar_mal.close()

    print(f"\nExtracting Benign Subgraphs (y=0) from clean training provenance logs...")
    adj_train = build_fast_adj(src_train, dst_train, limit=src_train.size(0))
    train_nodes = list(adj_train.keys())
    random.shuffle(train_nodes)

    benign_subgraphs_count = 0
    pbar_ben = tqdm(total=num_benign, desc="Benign Subgraphs")
    used_ben_sets = set()

    for seed_node in train_nodes:
        if benign_subgraphs_count >= num_benign:
            break
        nodes = sample_ego_subgraph(adj_train, seed_node, min_nodes, max_nodes)
        if not nodes:
            continue
        froz = frozenset(nodes)
        if froz in used_ben_sets:
            continue
        used_ben_sets.add(froz)

        pyg_data = build_cadets_subgraph(
            nodes, src_train, dst_train, etype_train, ntype_train,
            label=0, graph_id=1, subgraph_id=sub_id,
            names_map=names_map, types_map=types_map, uuid_map={},
            mal_nodes_set=set(), ubc_netflow_map=ubc_netflow_map
        )
        if pyg_data is not None:
            all_subgraphs.append(pyg_data)
            benign_subgraphs_count += 1
            sub_id += 1
            pbar_ben.update(1)
    pbar_ben.close()

    # Summary and verification
    benign_count = sum(1 for d in all_subgraphs if d.y.item() == 0)
    attack_count = sum(1 for d in all_subgraphs if d.y.item() == 1)
    avg_nodes = sum(d.num_nodes for d in all_subgraphs) / len(all_subgraphs)
    avg_edges = sum(d.num_edges for d in all_subgraphs) / len(all_subgraphs)

    print("\n" + "=" * 70)
    print("DARPA TC E3 CADETS SUBGRAPH EXTRACTION REPORT")
    print("=" * 70)
    print(f"Total Subgraphs Extracted: {len(all_subgraphs)}")
    print(f"  - Benign Subgraphs (y=0):    {benign_count}")
    print(f"  - Malicious Subgraphs (y=1):  {attack_count}")
    print(f"  - Average Nodes / Subgraph:  {avg_nodes:.1f}")
    print(f"  - Average Edges / Subgraph:  {avg_edges:.1f}")
    print(f"  - Feature Dimension:         {all_subgraphs[0].x.size(1)}")

    # Sample audit log verification
    sample_mal = next(d for d in all_subgraphs if d.y.item() == 1)
    for nid, logs in sample_mal.node_to_log.items():
        if logs and sample_mal.raw_attributes[nid]['is_malicious']:
            print(f"\n[Verification] Sample Grounded Attack Log (Node {sample_mal.raw_attributes[nid]['name']}):")
            print(f"  --> {logs[0]}")
            break

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(all_subgraphs, output_path)
    print(f"\nSaved CADETS dataset to: {output_path}")
    print("=" * 70)

    return all_subgraphs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Provenance Graph Ingestion & Subgraph Extraction')
    parser.add_argument('--dataset', type=str, default='streamspot', choices=['streamspot', 'darpa_cadets'])
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.dataset == 'streamspot':
        out_p = args.output_path or 'data/processed_subgraphs.pt'
        build_streamspot_dataset(
            num_benign_graphs=40,
            num_attack_graphs=40,
            min_nodes=15,
            max_nodes=150,
            output_path=out_p,
            seed=args.seed
        )
    elif args.dataset == 'darpa_cadets':
        out_p = args.output_path or 'data/darpa_cadets/processed_subgraphs.pt'
        build_cadets_dataset(
            output_path=out_p,
            num_malicious=200,
            num_benign=200,
            min_nodes=15,
            max_nodes=150,
            seed=args.seed
        )

