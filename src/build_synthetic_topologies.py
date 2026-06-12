#!/usr/bin/env python3
"""
Generate Extreme LUR Synthetic Tree Topologies.

Design:
- Left side: deep chain with hard region (extreme LUR pattern)
- Right side: multi-level branching tree (~6500 leaves)
  - Build merged base first (9000 edges)
  - Expand 2000 leaf edges into 2-edge chains for raw (11000 edges)
  - This ensures Merge(Raw) == Merged and same leaf count
- Output edges in BFS order so Tree class edge ordering matches
- Add 'is_selected' column marking edges in selected local configurations

Outputs:
  combined: synthetic_topologies_combined.xlsx
    - 3 raw edge sheets

Usage:
  python src/build_synthetic_topologies.py [--out_dir DIR]
"""

import argparse
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
import networkx as nx

# ═══════════════════════════════════════
# Configuration
# ═══════════════════════════════════════
SEED_BASE = 20260205

L_LEFT = 500
A_NS = 663000.0
LEFT_HARD_TARGETS = [100, 500, 1000]

RIGHT_MERGED_EDGES = 9000
RIGHT_RAW_EDGES = 11000
RIGHT_EXPAND = RIGHT_RAW_EDGES - RIGHT_MERGED_EDGES  # 2000 edges to expand
RIGHT_TARGET_LEAVES = 6500  # target right-side leaves

TARGET_NONZERO_RATIOS = [0.10]

ZETA_MAP = {
    100: 0.01,
    500: 0.05,
    1000: 0.10,
}


def case_sheet_name(hard_k_target: int, target_ratio: float) -> str:
    zeta_pct = int(round(ZETA_MAP[hard_k_target] * 100))
    alpha_pct = int(round(target_ratio * 100))
    return f"synthetic_zeta{zeta_pct}pct_alpha{alpha_pct}"


def legacy_case_sheet_name(city_idx: int, hard_k_target: int, target_ratio: float) -> str:
    return f"City{city_idx}_leftK{hard_k_target}_r{int(target_ratio*100):02d}"


# ═══════════════════════════════════════
# Label builders
# ═══════════════════════════════════════
def core_label():
    return "32-19999-21F/CORE-01"

def b_label(i: int) -> str:
    return f"32-{20000+i:05d}-21F/B{i:04d}-01"

def l_label(i: int) -> str:
    return f"32-{30000+i:05d}-15F/L{i:04d}-01"

def top_a_label():
    return "32-40001-11F/TOP-A-01"

def top_b_label():
    return "32-40002-11F/TOP-B-01"

def rsp_label(j: int) -> str:
    return f"32-{50000+j:05d}-17F/RSP{j:04d}-01"

def rl_label(k: int) -> str:
    return f"32-{60000+k:05d}-15F/RL{k:06d}-01"

def rmid_label(m: int) -> str:
    return f"32-{70000+m:05d}-16F/RM{m:06d}-01"


# ═══════════════════════════════════════
# Sampling
# ═══════════════════════════════════════
def sample_asym_value(rng: np.random.Generator, nonzero_prob: float,
                      mag_low: int = 100_000, mag_high: int = 800_000) -> float:
    if rng.random() >= nonzero_prob:
        return 0.0
    mag = float(rng.integers(mag_low, mag_high + 1))
    sgn = -1.0 if rng.random() < 0.5 else 1.0
    return sgn * mag


# ═══════════════════════════════════════
# Left region (same logic as v2)
# ═══════════════════════════════════════
def hard_value(edge_type: str, depth: int) -> float:
    if edge_type == "BACKBONE":
        sign = -1.0 if ((depth - 1) % 2 == 0) else 1.0
        return sign * A_NS
    return 0.0


def build_left_region(G: nx.DiGraph, rng: np.random.Generator,
                      alpha_normal: float, hard_k_target: int):
    """
    Build left region into DiGraph G. Returns hard edge set and stats.

    Left structure:
      CORE → B1 → B2 → ... → B500
      Each B(i) also has leaf L(i) (for i=1..499)
      B500 → TOP-A, B500 → TOP-B
    """
    hard_k_target = min(max(int(hard_k_target), 0), 2 * L_LEFT - 1)

    if hard_k_target in (1000, 2000):
        K = hard_k_target // 2
        backbone_max = K
        leaf_max = K
    elif hard_k_target == 2999:
        backbone_max = L_LEFT
        leaf_max = L_LEFT - 1
    else:
        K = hard_k_target // 2
        backbone_max = K + (hard_k_target - 2 * K)
        leaf_max = K

    CORE = core_label()
    B = [b_label(i) for i in range(1, L_LEFT + 1)]

    hard_edges = set()  # (parent, child) pairs that are hard
    hard_real = 0

    # CORE → B1
    depth = 1
    if depth <= backbone_max:
        x = hard_value("BACKBONE", depth)
        hard_edges.add((CORE, B[0]))
        hard_real += 1
    else:
        x = sample_asym_value(rng, alpha_normal)
    G.add_edge(CORE, B[0], x_true=x)

    # B(i) → B(i+1) and B(i) → L(i+1)  for i=0..498
    for i in range(L_LEFT - 1):
        bb_depth = i + 2  # depth of backbone edge B(i+1) → B(i+2)
        lf_depth = i + 1  # depth of leaf edge B(i+1) → L(i+1)

        # Backbone edge
        if bb_depth <= backbone_max:
            x_bb = hard_value("BACKBONE", bb_depth)
            hard_edges.add((B[i], B[i + 1]))
            hard_real += 1
        else:
            x_bb = sample_asym_value(rng, alpha_normal)
        G.add_edge(B[i], B[i + 1], x_true=x_bb)

        # Leaf edge
        leaf_node = l_label(i + 1)
        if lf_depth <= leaf_max:
            x_lf = 0.0
            hard_edges.add((B[i], leaf_node))
            hard_real += 1
        else:
            x_lf = sample_asym_value(rng, alpha_normal)
        G.add_edge(B[i], leaf_node, x_true=x_lf)

    # B500 → TOP-A, TOP-B
    G.add_edge(B[-1], top_a_label(), x_true=0.0)
    G.add_edge(B[-1], top_b_label(), x_true=0.0)

    stats = {
        "hard_real": hard_real,
        "backbone_max": backbone_max,
        "leaf_max": leaf_max,
        "left_edges": G.number_of_edges(),
    }
    return hard_edges, stats


# ═══════════════════════════════════════
# Right region: multi-level branching
# ═══════════════════════════════════════
def build_right_merged(G: nx.DiGraph, rng: np.random.Generator,
                       alpha: float, attach_root: str, n_edges: int,
                       target_leaves: int):
    """
    Build right side as a multi-level branching tree (merged version).

    Strategy:
    1. Compute n_internal = n_edges - target_leaves
    2. Build internal tree using BFS-level expansion (each node gets 2-5 children)
    3. Attach leaf nodes: at least 1 per internal node, rest distributed randomly

    Every internal node ends up with ≥2 children → no degree-2 nodes → merge-safe.
    """
    n_internal = n_edges - target_leaves

    # ── Step 1: Build internal skeleton ──
    internal_nodes = [attach_root]
    internal_edges = []  # (parent, child)
    queue = deque([attach_root])
    spine_counter = 0

    while len(internal_edges) < n_internal and queue:
        parent = queue.popleft()
        remaining = n_internal - len(internal_edges)
        if remaining <= 0:
            break

        # Branching factor: 2-5, but cap to remaining budget
        n_children = int(rng.integers(2, 6))
        n_children = min(n_children, remaining)

        # If this is the last parent and we need ≥2 children, ensure that
        if n_children < 2 and remaining >= 2:
            n_children = 2

        for _ in range(n_children):
            if len(internal_edges) >= n_internal:
                break
            spine_counter += 1
            child = rsp_label(spine_counter)
            internal_edges.append((parent, child))
            internal_nodes.append(child)
            queue.append(child)

    # Add internal edges to graph
    for p, c in internal_edges:
        val = sample_asym_value(rng, alpha)
        G.add_edge(p, c, x_true=val)

    # ── Step 2: Attach leaf nodes ──
    # First: ensure each internal node gets at least 1 leaf (→ degree ≥ 3)
    leaf_counter = 0
    leaf_edges = []

    for node in internal_nodes:
        leaf_counter += 1
        leaf = rl_label(leaf_counter)
        leaf_edges.append((node, leaf))

    # Count children per internal node
    child_count = {}
    for p, c in internal_edges:
        child_count[p] = child_count.get(p, 0) + 1

    # Nodes with 0 internal children (skeleton leaves) need ≥1 more leaf
    # to bring total children ≥ 2
    skeleton_leaves = [n for n in internal_nodes if child_count.get(n, 0) == 0]
    for node in skeleton_leaves:
        leaf_counter += 1
        leaf = rl_label(leaf_counter)
        leaf_edges.append((node, leaf))

    # Distribute remaining leaves randomly
    remaining_leaves = target_leaves - len(leaf_edges)
    if remaining_leaves > 0:
        parents = [internal_nodes[int(rng.integers(0, len(internal_nodes)))]
                   for _ in range(remaining_leaves)]
        for p in parents:
            leaf_counter += 1
            leaf = rl_label(leaf_counter)
            leaf_edges.append((p, leaf))

    # If we overshot (more leaf edges than target), trim from end
    if len(leaf_edges) > target_leaves:
        leaf_edges = leaf_edges[:target_leaves]

    # Add leaf edges to graph
    for p, c in leaf_edges:
        val = sample_asym_value(rng, alpha)
        G.add_edge(p, c, x_true=val)

    actual_right_edges = len(internal_edges) + len(leaf_edges)
    actual_leaves = len(leaf_edges)

    return {
        "n_internal": len(internal_edges),
        "n_leaf": actual_leaves,
        "total": actual_right_edges,
        "target": n_edges,
    }


def expand_for_raw(G_merged: nx.DiGraph, rng: np.random.Generator,
                   alpha: float, n_expand: int) -> nx.DiGraph:
    """
    Create raw version by expanding n_expand leaf edges into 2-edge chains.

    For each selected leaf edge (parent → leaf):
      Remove it, insert (parent → mid → leaf)
      mid is a degree-2 node that Tree.reduce() will merge back.

    Returns a new DiGraph (raw version).
    """
    G_raw = G_merged.copy()

    # Find RIGHT-SIDE leaf edges only (RL-labeled nodes).
    # Avoids expanding left-side hard edges (L-nodes, TOP-nodes, etc.)
    leaves = [n for n in G_merged.nodes()
              if G_merged.out_degree(n) == 0 and "/RL" in str(n)]
    leaf_edges = []
    for leaf in leaves:
        preds = list(G_merged.predecessors(leaf))
        if preds:
            leaf_edges.append((preds[0], leaf))

    # Randomly select n_expand leaf edges to expand
    n_expand = min(n_expand, len(leaf_edges))
    selected_indices = rng.choice(len(leaf_edges), size=n_expand, replace=False)
    selected_edges = [leaf_edges[i] for i in selected_indices]

    mid_counter = 0
    for parent, leaf in selected_edges:
        # Get original edge data
        edge_data = G_raw.edges[parent, leaf]
        original_value = edge_data.get('x_true', 0.0)

        # Remove original edge
        G_raw.remove_edge(parent, leaf)

        # Insert intermediate node
        mid_counter += 1
        mid = rmid_label(mid_counter)

        # Split value: mid inherits original x_true, leaf gets a new random value
        G_raw.add_edge(parent, mid, x_true=original_value)
        G_raw.add_edge(mid, leaf, x_true=sample_asym_value(rng, alpha))

    return G_raw


# ═══════════════════════════════════════
# BFS-ordered output
# ═══════════════════════════════════════
def graph_to_bfs_dataframe(G: nx.DiGraph, source: str,
                           hard_edges: set) -> pd.DataFrame:
    """
    Convert DiGraph to DataFrame with edges in BFS order.
    Adds is_selected column based on hard_edges set.
    """
    bfs_edges = list(nx.bfs_edges(G, source=source))

    rows = []
    for i, (parent, child) in enumerate(bfs_edges):
        x_true = G.edges[parent, child].get('x_true', 0.0)
        is_selected = 1 if (parent, child) in hard_edges else 0
        rows.append({
            'edge_id': i,
            'parent_label': parent,
            'child_label': child,
            'x_true': x_true,
            'is_selected': is_selected,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════
# Main generation
# ═══════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str,
                        default="../data/synthetic_topologies")
    args = parser.parse_args()

    out_dir = (Path(__file__).resolve().parent / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_combined = out_dir / "synthetic_topologies_combined.xlsx"
    with pd.ExcelWriter(raw_combined, engine="openpyxl") as raw_writer:

        for city_idx, hard_k_target in enumerate(LEFT_HARD_TARGETS, start=1):
            city_tag = f"City{city_idx}_leftK{hard_k_target}"

            for target_ratio in TARGET_NONZERO_RATIOS:
                # Seed per case
                seed = (SEED_BASE
                        + hard_k_target * 7
                        + int(round(target_ratio * 1000)) * 13
                        + city_idx * 101)
                rng = np.random.default_rng(seed)

                # ── Calculate alpha ──
                left_total = 2 * L_LEFT + 1  # 1001
                hard_backbone = hard_k_target // 2

                # Merged alpha
                total_merged = left_total + RIGHT_MERGED_EDGES
                target_nz_merged = int(total_merged * target_ratio)
                needed_merged = target_nz_merged - hard_backbone
                left_nonhard = left_total - hard_k_target
                nonhard_merged = left_nonhard + RIGHT_MERGED_EDGES
                alpha_merged = max(0.0, min(1.0, needed_merged / nonhard_merged))

                # Raw alpha
                total_raw = left_total + RIGHT_RAW_EDGES
                target_nz_raw = int(total_raw * target_ratio)
                needed_raw = target_nz_raw - hard_backbone
                nonhard_raw = left_nonhard + RIGHT_RAW_EDGES
                alpha_raw = max(0.0, min(1.0, needed_raw / nonhard_raw))

                # ── Build merged graph ──
                G_merged = nx.DiGraph()
                CORE = core_label()

                hard_edges, left_stats = build_left_region(
                    G_merged, rng, alpha_merged, hard_k_target
                )

                attach = l_label(1)  # Right side attaches to L1
                right_stats = build_right_merged(
                    G_merged, rng, alpha_merged, attach,
                    n_edges=RIGHT_MERGED_EDGES,
                    target_leaves=RIGHT_TARGET_LEAVES
                )

                # ── Build raw graph by expanding merged ──
                G_raw = expand_for_raw(G_merged, rng, alpha_raw, n_expand=RIGHT_EXPAND)

                # ── Convert to DataFrames in BFS order ──
                df_merged = graph_to_bfs_dataframe(G_merged, CORE, hard_edges)
                df_raw = graph_to_bfs_dataframe(G_raw, CORE, hard_edges)

                # ── Count statistics ──
                merged_arr = df_merged['x_true'].values
                raw_arr = df_raw['x_true'].values

                merged_nz = int(np.sum(np.abs(merged_arr) > 1e-12))
                raw_nz = int(np.sum(np.abs(raw_arr) > 1e-12))

                # Count leaves
                merged_leaves = sum(1 for n in G_merged.nodes()
                                    if G_merged.out_degree(n) == 0)
                raw_leaves = sum(1 for n in G_raw.nodes()
                                 if G_raw.out_degree(n) == 0)

                sheet = case_sheet_name(hard_k_target, target_ratio)[:31]
                legacy_sheet = legacy_case_sheet_name(city_idx, hard_k_target, target_ratio)[:31]

                # ── Write raw sheet to combined excel ──
                df_raw.to_excel(raw_writer, index=False, sheet_name=sheet)

                print(f"  {sheet}: merged={len(df_merged)} edges "
                      f"({merged_leaves} leaves), "
                      f"raw={len(df_raw)} edges ({raw_leaves} leaves), "
                      f"hard={left_stats['hard_real']}")

    print("\nDone.")
    print(f"Output: {out_dir}")
    print(f"Combined raw:    {raw_combined.name}")


if __name__ == "__main__":
    main()
