# mapping_generator/utils/graph_processor.py

import networkx as nx
from .mapping import Mapping

class GraphProcessor:
    """Provides utility functions for processing and validating mappings."""

    def __init__(self, mapping: Mapping):
        """Initializes the processor with a mapping object."""
        self.mapping = mapping
        self.dfg = self._build_dfg_from_mapping()

    def _build_dfg_from_mapping(self) -> nx.DiGraph:
        """Constructs a NetworkX DiGraph from the mapping's routing information."""
        dfg = nx.DiGraph()
        nodes = list(self.mapping.placement.keys())
        dfg.add_nodes_from(nodes)
        
        for (source_id, dest_id) in self.mapping.routing.keys():
            if source_id in nodes and dest_id in nodes:
                dfg.add_edge(source_id, dest_id)
        return dfg

    def _calculate_levels(self):
        """Calculates the topological level of each node in the DFG.

        The level of a node is the length of the longest path from an input
        node to it. This is stored as a 'level' attribute on each node.
        """
        for node in nx.topological_sort(self.dfg):
            level = 0
            predecessors = list(self.dfg.predecessors(node))
            if predecessors:
                level = max(self.dfg.nodes[p].get('level', -1) for p in predecessors) + 1
            self.dfg.nodes[node]['level'] = level

    def _is_balanced(self) -> bool:
        """Checks if the DFG is balanced.

        A DFG is considered balanced if all predecessors of any given node
        reside at the same topological level.

        Returns:
            bool: True if the graph is balanced, False otherwise.
        """
        for node in self.dfg.nodes():
            predecessors = list(self.dfg.predecessors(node))
            if len(predecessors) > 1:
                # Get the level of the first predecessor
                first_level = self.dfg.nodes[predecessors[0]].get('level')
                if first_level is None: return False 
                
                # Check if all other predecessors have the same level
                for p in predecessors[1:]:
                    if self.dfg.nodes[p].get('level') != first_level:
                        return False
        return True

    def is_valid(self) -> bool:
        """Performs a full validation of the DFG.

        A valid DFG must be:
        1. Weakly connected.
        2. A Directed Acyclic Graph (DAG).
        3. Balanced (all predecessors of a node are at the same level).

        Returns:
            bool: True if all validation checks pass, False otherwise.
        """
        if self.dfg.number_of_nodes() < 2:
            return False

        # Standard checks
        if not nx.is_weakly_connected(self.dfg):
            return False
        if not nx.is_directed_acyclic_graph(self.dfg):
            return False
        
        # Custom validation checks from the original project
        self._calculate_levels()
        if not self._is_balanced():
            return False
            
        return True
