import networkx as nx
import logging
from typing import TYPE_CHECKING, Tuple, Optional
from .base import BaseGrammarRule

if TYPE_CHECKING:
    from ..QcaGrammarGenerator import QcaGrammarGenerator

logger = logging.getLogger(__name__)


class TreeRule(BaseGrammarRule):
    """
    Tree growth rule: Expands the graph by adding a new node and 
    a path to it from a leaf node.
    
    Features:
    - Simpler and faster than other rules.
    - Always creates 1 new operation.
    - Maintains tree structure (no reconvergence).
    """
    
    def __init__(self, max_path_length: int = 15):
        """
        Initializes the tree rule.
        
        Args:
            max_path_length: Maximum allowed path length to the new node.
        """
        super().__init__(max_path_length=max_path_length)
        self.max_path_length = max_path_length
    
    def get_rule_type(self) -> str:
        return "tree"
    
    def can_apply(self, generator: 'QcaGrammarGenerator', start_node: Tuple[int, int]) -> bool:
        """
        Checks if a valid path is available.
        Conditions:
        1. start_node must be a leaf.
        2. At least one node must be available in the architecture.
        """
        if generator.placement_graph.out_degree(start_node) > 0:
            logger.debug(f"TreeRule: {start_node} is not a leaf.")
            return False
        
        available_nodes = set(generator.qca_arch_graph.nodes()) - generator.used_nodes
        if len(available_nodes) == 0:
            logger.debug("TreeRule: No nodes available in architecture.")
            return False
        
        return True
    
    def apply(self, generator: 'QcaGrammarGenerator', start_node: Tuple[int, int]) -> bool:
        """
        Applies the tree rule: finds path to new node and adds to graph.
        """
        if not self.can_apply(generator, start_node):
            return False
        
        path = generator.find_shortest_path_to_new_node(start_node)
        
        if not path:
            logger.debug(f"TreeRule: No path found from {start_node}.")
            return False
        
        path_length = len(path) - 1
        if path_length > self.max_path_length:
            logger.debug(f"TreeRule: Path too long ({path_length} > {self.max_path_length}).")
            return False
        
        try:
            generator.used_nodes.update(path)
            nx.add_path(generator.placement_graph, path)
            
            # Mark new node as operation
            new_node = path[-1]
            generator.placement_graph.nodes[new_node]['type'] = 'operation'
            
            # Mark intermediates as routing
            for node in path[1:-1]:
                if 'type' not in generator.placement_graph.nodes[node]:
                    generator.placement_graph.nodes[node]['type'] = 'routing'
            
            self._increment_counter()
            logger.debug(f"TreeRule: Applied. Path: {start_node} -> {new_node} ({path_length} hops).")
            return True
            
        except Exception as e:
            logger.error(f"TreeRule: Error applying rule: {e}", exc_info=True)
            return False
    
    def estimate_cost(self, generator: 'QcaGrammarGenerator', start_node: Tuple[int, int]) -> Optional[int]:
        """Estimates number of nodes that will be added."""
        if not self.can_apply(generator, start_node):
            return None
        return self.max_path_length // 2
