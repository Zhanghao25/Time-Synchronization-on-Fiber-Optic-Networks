#!/usr/bin/env python3
"""
Tree Structure Module for Hierarchical Network Analysis

This module provides the Tree class that handles tree construction, 
path reduction, and matrix generation for hierarchical network regression.

Key Features:
- Tree construction from networkx graphs
- Optional path merging for dimension reduction
- Matrix representation generation
- Confidence-based node grouping
- Edge analysis utilities
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Set, Tuple, Union, Optional, Any
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class Tree:
    """
    Enhanced Tree class for hierarchical graph operations and linear system representation.
    
    This class combines the functionality of the original Tree and HierarchicalTree classes,
    providing comprehensive tree-based network analysis capabilities.
    
    Attributes:
        tree (nx.DiGraph): The processed tree structure
        source (int): Root node identifier
        leaves (List[int]): List of leaf node identifiers
        parent_dict (Dict[int, Optional[int]]): Parent mapping for each node
        edges (List[Tuple[int, int]]): List of tree edges in BFS order
        edge2idx (Dict[Tuple[int, int], int]): Edge to index mapping
        merged (bool): Whether path merging was applied
    """
    
    def __init__(self, g: nx.Graph, source: int, merge: bool = True) -> None:
        """
        Initialize Tree from networkx graph.
        
        Args:
            g (nx.Graph): Input graph (must be tree-like)
            source (int): Root node for tree construction
            merge (bool): Whether to apply path merging for dimension reduction
            
        Raises:
            ValueError: If input graph is not a valid tree structure
        """
        # Validate tree structure
        if not nx.is_tree(g):
            raise ValueError("Input graph must be a tree structure")
        
        if source not in g.nodes():
            raise ValueError(f"Source node {source} not found in graph")
        
        # Construct directed tree from source
        tree = nx.dfs_tree(g, source=source).to_directed()
        
        # Preserve edge attributes (values)
        for edge in tree.edges():
            if g.has_edge(*edge):
                tree.edges[edge].update({"value": g.get_edge_data(*edge)["value"]})
            else:
                # Handle undirected edges
                u, v = edge
                tree.edges[edge].update({"value": g.get_edge_data(v, u)["value"]})
        
        self.tree = tree
        self.source = source
        self.merged = merge
        
        # Apply path merging if requested
        if merge:
            self.reduce()
        
        # Initialize tree properties
        self._initialize_tree_properties()
        
        # Validate final structure
        self._validate_tree_structure()
    
    def _initialize_tree_properties(self) -> None:
        """Initialize tree structural properties and mappings."""
        # Identify leaf nodes
        self.leaves = [n for n in self.tree.nodes() if self.tree.out_degree(n) == 0]
        
        # Build parent mapping
        self.parent_dict = {self.source: None}
        for node in self.tree.nodes():
            if node != self.source:
                predecessors = list(self.tree.predecessors(node))
                if predecessors:
                    self.parent_dict[node] = predecessors[0]
                else:
                    raise ValueError(f"Node {node} has no parent (disconnected from source)")
        
        # Generate edge ordering using BFS
        self.edges = list(nx.bfs_edges(self.tree, source=self.source))
        self.edge2idx = {edge: idx for idx, edge in enumerate(self.edges)}
    
    def _validate_tree_structure(self) -> None:
        """Validate final tree structure integrity."""
        if self.source in self.leaves:
            raise ValueError("Source node cannot be a leaf node")
        
        if len(self.leaves) == 0:
            raise ValueError("Tree must have at least one leaf node")
        
        # Verify connectivity
        if not nx.is_weakly_connected(self.tree):
            raise ValueError("Tree structure is not connected")

    def find_cut_paths(self) -> List[Tuple[int, ...]]:
        """
        Identify linear paths suitable for reduction.
        
        Returns:
            List[Tuple[int, ...]]: List of node sequences representing cut paths
        """
        leaves = [n for n in self.tree.nodes() if self.tree.out_degree(n) == 0]
        parent_dict = {self.source: None}
        
        # Build parent mapping
        for node in self.tree.nodes():
            if node != self.source:
                parent = next(self.tree.predecessors(node), None)
                if parent is not None:
                    parent_dict[node] = parent
                
        cut_paths = []
        
        for leaf in leaves:
            cut_path = []
            node = leaf
            last_visit = None
            
            while node is not None:
                # Check if node is in a linear segment (degree 2)
                if (self.tree.in_degree(node) == 1 and 
                    self.tree.out_degree(node) == 1):
                    
                    if not cut_path:
                        cut_path.append(last_visit)
                    cut_path.append(node)
                else:
                    # End of linear segment
                    if cut_path:
                        # Add the parent node to complete the path
                        parent = next(self.tree.predecessors(cut_path[-1]), None)
                        if parent is not None:
                            cut_path.append(parent)
                        cut_path.reverse()
                        cut_paths.append(tuple(cut_path))
                        cut_path = []
                
                last_visit = node
                node = parent_dict.get(node)
        
        return list(set(cut_paths))

    def cut_path(self, path: Tuple[int, ...]) -> None:
        """
        Reduce a linear path by removing intermediate nodes.
        
        Args:
            path (Tuple[int, ...]): Sequence of nodes to be reduced
        """
        if len(path) < 3:
            return  # Nothing to cut
        
        # Calculate total path weight
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        total_value = sum(
            self.tree.get_edge_data(*edge).get("value", 0) 
            for edge in path_edges
        )
        
        # Remove intermediate nodes
        for node in path[1:-1]:
            if node in self.tree:
                self.tree.remove_node(node)
        
        # Add direct edge with accumulated weight
        if not self.tree.has_edge(path[0], path[-1]):
            self.tree.add_edge(path[0], path[-1], value=total_value)

    def reduce(self) -> None:
        """
        Simplify tree structure by cutting all identified linear paths.
        """
        max_iterations = 100  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            cut_paths = self.find_cut_paths()
            if not cut_paths:
                break
                
            for path in cut_paths:
                self.cut_path(path)
            
            iteration += 1
        
        if iteration >= max_iterations:
            warnings.warn("Tree reduction reached maximum iterations limit")

    def get_Ax(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate matrix representation (A, x) for linear system Ax = b.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - A: Incidence matrix (n_leaves × n_edges)
                - x: Edge weight vector (n_edges,)
        """
        # Build incidence matrix
        incidence_matrix = []
        
        for leaf in self.leaves:
            # Construct path from leaf to root
            path_nodes = []
            current_node = leaf
            
            while current_node is not None:
                path_nodes.append(current_node)
                current_node = self.parent_dict.get(current_node)
            
            path_nodes.reverse()  # Root to leaf order
            
            # Convert to edges
            path_edges = [(path_nodes[i], path_nodes[i+1]) 
                         for i in range(len(path_nodes)-1)]
            
            # Create incidence vector
            incidence_vector = np.zeros(len(self.edges))
            for edge in path_edges:
                if edge in self.edge2idx:
                    incidence_vector[self.edge2idx[edge]] = 1
            
            incidence_matrix.append(incidence_vector)

        # Construct matrices
        A = np.asarray(incidence_matrix)
        
        # Extract edge weights
        edge_value_dict = nx.get_edge_attributes(self.tree, "value")
        x = np.array([edge_value_dict.get(edge, 0) for edge in self.edges])
        
        return A, x

    def find_edges_for_node(self, node: int) -> List[int]:
        """
        Find all edge indices connected to a specific node.
        
        Args:
            node (int): Target node identifier
            
        Returns:
            List[int]: List of edge indices connected to the node
        """
        edge_indices = []
        for idx, edge in enumerate(self.edges):
            if node in edge:
                edge_indices.append(idx)
        return edge_indices
    
    def get_predecessor_edges(self, nodes: Union[int, List[int]]) -> List[int]:
        """
        Get predecessor edge indices for given nodes.
        
        Args:
            nodes (Union[int, List[int]]): Node(s) to analyze
            
        Returns:
            List[int]: List of predecessor edge indices
        """
        if isinstance(nodes, int):
            nodes = [nodes]
        
        edge_indices = []
        for node in nodes:
            predecessors = list(self.tree.predecessors(node))
            for predecessor in predecessors:
                edge = (predecessor, node)
                if edge in self.edge2idx:
                    edge_indices.append(self.edge2idx[edge])
        
        return edge_indices

    def find_binary_edges(self, metric: List[int], neg_idx: List[int]) -> List[int]:
        """
        Identify binary constraint edges based on node connectivity patterns.
        
        Args:
            metric (List[int]): Edge indices to analyze
            neg_idx (List[int]): Negative/constraint edge indices
            
        Returns:
            List[int]: List of binary edge indices
        """
        binary_edges = []
        
        for edge_index in metric:
            if edge_index in neg_idx:
                node1, node2 = self.edges[edge_index]
                
                # Count constraint edges for each endpoint
                edges_node1 = self.find_edges_for_node(node1)
                edges_node2 = self.find_edges_for_node(node2)
                
                count_node1 = sum(1 for edge in edges_node1 if edge in neg_idx)
                count_node2 = sum(1 for edge in edges_node2 if edge in neg_idx)
                
                # Binary criterion: at least one endpoint has ≥2 constraint edges
                if count_node1 >= 2 or count_node2 >= 2:
                    binary_edges.append(edge_index)
        
        return binary_edges

    def restrictive(self, edge: int, binary: List[int]) -> bool:
        """
        Check if an edge has restrictive properties for high confidence classification.
        
        Args:
            edge (int): Edge index to analyze
            binary (List[int]): Binary edges list
            
        Returns:
            bool: True if edge is restrictive (high confidence)
        """
        node1, node2 = self.edges[edge]
        
        predecessor_edges = self.get_predecessor_edges([node1])
        
        if edge not in predecessor_edges:
            parent = node1
            child = node2
        else:
            parent = node2
            child = node1
        
        predecessor_edges = [i for i in self.find_edges_for_node(parent) if i != edge]
        postorder_edges = [i for i in self.find_edges_for_node(child) if i != edge]
        
        new_binary_ls = []
        
        for i in predecessor_edges:
            upnode = [node for node in self.edges[i] if node != parent][0]
            up_edge = self.find_edges_for_node(upnode)
            count = 0
            for j in up_edge:
                if j in binary:
                    count += 1
                if count >= 2:
                    new_binary_ls.append(edge)
                    break
        
        if not new_binary_ls:
            for i in postorder_edges:
                downnode = [node for node in self.edges[i] if node != child][0]
                down_edge = self.find_edges_for_node(downnode)
                count1 = 0
                for j in down_edge:
                    if j in binary:
                        count1 += 1
                    if count1 >= 2:
                        new_binary_ls.append(edge)
                        break
        
        return len(list(set(new_binary_ls))) > 0

    def find_subhigh_nodes(self, metric: List[int], neg_idx: List[int], 
                          true_nonzero_indices: List[int]) -> Dict[str, Any]:
        """
        Enhanced confidence grouping for all predicted non-zero edges.
        
        Confidence-Based Node Classification:
        - Subhigh Nodes: Nodes connected to two or more predicted non-zero edges
        - High Nodes: Nodes that are one-hop neighbors of subhigh nodes and connected to exactly one predicted non-zero edge  
        - Ultra-high Nodes: All remaining predicted non-zero edges not classified as subhigh or high
        
        Args:
            metric (List[int]): Edge indices to analyze
            neg_idx (List[int]): Predicted non-zero edge indices
            true_nonzero_indices (List[int]): Ground truth non-zero edge indices
            
        Returns:
            Dict containing detailed confidence grouping results
        """
        # Convert to sets for efficient operations
        true_nonzero_set = set(true_nonzero_indices)
        neg_idx_set = set(neg_idx)
        
        # Step 1: Find subhigh nodes (nodes connected to two or more predicted non-zero edges)
        binary_edges = []
        for edge_index in metric:
            if edge_index in neg_idx_set:
                node1, node2 = self.edges[edge_index]
                edges_node1 = self.find_edges_for_node(node1)
                edges_node2 = self.find_edges_for_node(node2)
                
                count_node1 = sum(1 for edge in edges_node1 if edge in neg_idx_set)
                count_node2 = sum(1 for edge in edges_node2 if edge in neg_idx_set)
                
                if count_node1 >= 2 or count_node2 >= 2:
                    binary_edges.append(edge_index)
        
        # Step 2: Identify subhigh nodes (nodes connected to binary edges)
        subhigh_nodes = set()
        subhigh_edges = binary_edges.copy()
        
        for edge_index in binary_edges:
            node1, node2 = self.edges[edge_index]
            edges_node1 = self.find_edges_for_node(node1)
            edges_node2 = self.find_edges_for_node(node2)
            
            count_node1 = sum(1 for edge in edges_node1 if edge in neg_idx_set)
            count_node2 = sum(1 for edge in edges_node2 if edge in neg_idx_set)
            
            if count_node1 >= 2:
                subhigh_nodes.add(node1)
            if count_node2 >= 2:
                subhigh_nodes.add(node2)
        
        # Step 3: Classify remaining edges into high and ultra-high confidence
        nonbinary_edges = [edge for edge in neg_idx if edge not in binary_edges]
        
        # High confidence edges: restrictive edges
        high_edges = []
        for edge in nonbinary_edges:
            if self.restrictive(edge, binary_edges):
                high_edges.append(edge)
        
        # Ultra-high confidence edges: non-restrictive edges
        ultrahigh_edges = [edge for edge in nonbinary_edges if edge not in high_edges]
        
        # Step 4: Identify high nodes (one-hop neighbors of subhigh nodes with exactly one predicted non-zero edge)
        high_nodes = set()
        for node in subhigh_nodes:
            # Iterate over all edges connected to the node
            for edge_index in self.find_edges_for_node(node):
                # Find the other endpoint (neighbor) of the edge
                n1, n2 = self.edges[edge_index]
                neighbor = n2 if n1 == node else n1
                
                # Count how many edges of the neighbor are in neg_idx
                neighbor_edges = self.find_edges_for_node(neighbor)
                neg_count = sum(1 for e in neighbor_edges if e in neg_idx)
                
                # If there is exactly one such edge and the neighbor is not a leaf node, add to high_nodes
                if neg_count == 1 and neighbor not in self.leaves:
                    high_nodes.add(neighbor)

        # Step 5: Identify ultra-high nodes (nodes connected to ultra-high confidence edges)
        ultrahigh_nodes = set()
        for edge_idx in ultrahigh_edges:
            node1, node2 = self.edges[edge_idx]
            ultrahigh_nodes.add(node1)
            ultrahigh_nodes.add(node2)
        
        # Step 6: Calculate confidence statistics
        def calculate_confidence(edge_list: List[int]) -> float:
            if not edge_list:
                return 0.0
            correct_predictions = sum(1 for edge in edge_list if edge in true_nonzero_set)
            return correct_predictions / len(edge_list)
        
        confidence_stats = {
            'subhigh_confidence': calculate_confidence(subhigh_edges),
            'high_confidence': calculate_confidence(high_edges),
            'ultrahigh_confidence': calculate_confidence(ultrahigh_edges),
            'overall_confidence': calculate_confidence(neg_idx)
        }
        
        # Step 7: Count statistics
        group_counts = {
            'subhigh_nodes_count': len(subhigh_nodes),
            'subhigh_edges_count': len(subhigh_edges),
            'high_nodes_count': len(high_nodes),
            'high_edges_count': len(high_edges),
            'ultrahigh_nodes_count': len(ultrahigh_nodes),
            'ultrahigh_edges_count': len(ultrahigh_edges),
            'total_predicted_edges': len(neg_idx)
        }
        
        # Comprehensive results
        results = {
            'subhigh_nodes': subhigh_nodes,
            'subhigh_edges': subhigh_edges,
            'high_nodes': high_nodes,
            'high_edges': high_edges,
            'ultrahigh_nodes': ultrahigh_nodes,
            'ultrahigh_edges': ultrahigh_edges,
            'confidence_stats': confidence_stats,
            'group_counts': group_counts,
            'binary_edges': binary_edges
        }
        
        return results


def graph_from_pandas(file_path: Union[str, Path], 
                     columns: List[int] = [1, 2]) -> Tuple[nx.Graph, Optional[int]]:
    """
    Create networkx graph from Excel file with automatic source detection (topology only).
    
    Args:
        file_path (Union[str, Path]): Path to Excel file
        columns (List[int]): Column indices [source, target]; edge value set to 0.
        
    Returns:
        Tuple[nx.Graph, Optional[int]]: (graph object, source node)
    """
    try:
        df = pd.read_excel(file_path, usecols=columns)
        if df.empty:
            raise ValueError("Excel file contains no data")
        df.columns = ['source', 'target']
        df['value'] = 0
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot find file: {file_path}")
    except Exception as e:
        raise ValueError(f"Cannot read file {file_path}: {e}")
    
    # Encode node labels to integers
    label_encoder = LabelEncoder()
    all_nodes = pd.concat([df['source'], df['target']])
    label_encoder.fit(all_nodes)
    
    df['source'] = label_encoder.transform(df['source'])
    df['target'] = label_encoder.transform(df['target'])
    
    # Create weighted graph
    graph = nx.from_pandas_edgelist(df, edge_attr="value")
    
    # Identify source node (appears in source column but not target column)
    source_candidates = set(df["source"]) - set(df["target"])
    
    if len(source_candidates) == 1:
        source = source_candidates.pop()
    elif len(source_candidates) == 0:
        warnings.warn("No unique source node found (graph may be cyclic)")
        source = None
    else:
        warnings.warn(f"Multiple source candidates found: {source_candidates}")
        source = min(source_candidates)  # Choose minimum as default
    
    return graph, source


def get_all_edge_indices_above(tree: Tree, nodes: Union[int, List[int]]) -> List[int]:
    """
    Get all edge indices on paths from given nodes to the root.
    
    Args:
        tree (Tree): Tree object
        nodes (Union[int, List[int]]): Node(s) to trace from
        
    Returns:
        List[int]: List of edge indices on paths to root
    """
    if isinstance(nodes, int):
        nodes = [nodes]
    
    edge_indices_above = []
    
    for node in nodes:
        current_node = node
        
        # Traverse path to root
        while current_node is not None:
            parent_node = tree.parent_dict.get(current_node)
            
            if parent_node is None:
                break  # Reached root
            
            # Find edge from parent to current node
            edge = (parent_node, current_node)
            edge_index = tree.edge2idx.get(edge)
            
            if edge_index is not None:
                edge_indices_above.append(edge_index)
            
            current_node = parent_node

    return edge_indices_above