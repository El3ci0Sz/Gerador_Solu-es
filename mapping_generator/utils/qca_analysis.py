import networkx as nx
import logging
from typing import Dict, Any, Tuple
import math

logger = logging.getLogger(__name__)

class QcaValidator:
    """
    Centralizes validation logic for QCA graphs.
    Ensures generated graphs meet physical and logical constraints.
    """

    @staticmethod
    def validate(graph: nx.DiGraph, border_nodes: set) -> Tuple[bool, list]:
        """
        Runs a comprehensive validation suite on the graph.

        Args:
            graph (nx.DiGraph): The QCA placement graph.
            border_nodes (set): Set of coordinate tuples representing valid border positions.

        Returns:
            Tuple[bool, list]: (IsValid, List of error messages).
        """
        errors = []

        if not nx.is_directed_acyclic_graph(graph):
            errors.append("Graph contains cycles (not a DAG).")

        if not nx.is_weakly_connected(graph):
            errors.append("Graph is disconnected (multiple components).")

        # Validate Inputs (Must have in-degree 0)
        inputs = [n for n, d in graph.nodes(data=True) if d.get('type') == 'input']
        if not inputs:
            errors.append("No input nodes found.")
        for inp in inputs:
            if graph.in_degree(inp) > 0:
                errors.append(f"Input node {inp} has incoming edges.")

        # Validate Outputs (Should ideally be on border)
        outputs = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        if not outputs:
            errors.append("No output nodes found.")
        
        # Check node types
        for n, data in graph.nodes(data=True):
            if 'type' not in data:
                errors.append(f"Node {n} missing 'type' attribute.")

        return len(errors) == 0, errors

class QcaMetrics:
    """
    Calculates quality metrics for QCA graphs to allow ranking and analysis.
    """

    @staticmethod
    def calculate_all(graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Computes all available metrics for a given graph.

        Returns:
            Dict containing metrics like routing_overhead, compactness, balance_score, etc.
        """
        try:
            return {
                'node_count': graph.number_of_nodes(),
                'edge_count': graph.number_of_edges(),
                'routing_overhead': QcaMetrics.routing_overhead(graph),
                'balance_score': QcaMetrics.balance_score(graph),
                'diameter': QcaMetrics.diameter(graph),
                'density': nx.density(graph),
                'critical_path': QcaMetrics.critical_path_length(graph)
            }
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            return {}

    @staticmethod
    def routing_overhead(graph: nx.DiGraph) -> float:
        """
        Calculates the ratio of routing nodes to logic operations.
        Lower is better.
        """
        ops = sum(1 for _, d in graph.nodes(data=True) if d.get('type') in ['operation', 'convergence'])
        routing = sum(1 for _, d in graph.nodes(data=True) if d.get('type') in ['routing', 'buffer'])
        
        if ops == 0: return 0.0
        return routing / ops

    @staticmethod
    def balance_score(graph: nx.DiGraph) -> float:
        """
        Quantifies how balanced the graph is (0.0 to 1.0).
        1.0 means perfectly balanced (all paths to an endpoint have equal length).
        """
        # Calculate ASAP levels
        try:
            levels = {}
            for node in nx.topological_sort(graph):
                preds = list(graph.predecessors(node))
                if not preds:
                    levels[node] = 0
                else:
                    levels[node] = max(levels[p] for p in preds) + 1
            
            # Check deviation at merge points
            deviations = []
            for node in graph.nodes():
                preds = list(graph.predecessors(node))
                if len(preds) > 1:
                    pred_levels = [levels[p] for p in preds]
                    deviations.append(max(pred_levels) - min(pred_levels))
            
            if not deviations: return 1.0
            
            avg_deviation = sum(deviations) / len(deviations)
            # Normalize: 0 deviation -> score 1.0. Large deviation -> score closer to 0.
            return 1.0 / (1.0 + avg_deviation)
            
        except Exception:
            return 0.0

    @staticmethod
    def diameter(graph: nx.DiGraph) -> int:
        """
        Calculates the longest shortest path in the graph.
        """
        try:
            return nx.dag_longest_path_length(graph)
        except Exception:
            return -1

    @staticmethod
    def critical_path_length(graph: nx.DiGraph) -> int:
        """
        Returns the maximum logical depth.
        """
        try:
            return nx.dag_longest_path_length(graph)
        except Exception:
            return 0
