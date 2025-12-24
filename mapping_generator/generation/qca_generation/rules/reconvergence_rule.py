import random
import networkx as nx
import logging
from typing import TYPE_CHECKING, Tuple, Optional, List
from .base import BaseGrammarRule

if TYPE_CHECKING:
    from ..QcaGrammarGenerator import QcaGrammarGenerator

logger = logging.getLogger(__name__)


class ReconvergenceRule(BaseGrammarRule):
    """
    Reconvergence rule: Splits a start node into 'k' disjoint paths 
    that converge to a new destination node.
    
    Features:
    - Creates more complex structures than TreeRule.
    - Increases graph parallelism.
    - Requires 'k' disjoint paths (no shared intermediate nodes).
    """
    
    def __init__(self, k_range: Tuple[int, int] = (2, 2), max_path_length: int = 15):
        """
        Initializes the reconvergence rule.
        
        Args:
            k_range: Tuple (min_k, max_k) defining number of disjoint paths.
            max_path_length: Maximum allowed length for each path.
        """
        super().__init__(k_range=k_range, max_path_length=max_path_length)
        self.k_range = k_range
        self.max_path_length = max_path_length
    
    def get_rule_type(self) -> str:
        return "reconvergence"
    
    def can_apply(self, generator: 'QcaGrammarGenerator', start_node: Tuple[int, int]) -> bool:
        """Checks if the rule can potentially be applied."""
        if start_node not in generator.placement_graph.nodes():
            return False
        
        available_nodes = set(generator.qca_arch_graph.nodes()) - generator.used_nodes
        min_nodes_needed = self.k_range[0] + 1
        
        if len(available_nodes) < min_nodes_needed:
            logger.debug(f"ReconvergenceRule: Not enough nodes available ({len(available_nodes)} < {min_nodes_needed})")
            return False
        
        return True

    def apply(self, generator: 'QcaGrammarGenerator', start_node: Tuple[int, int]) -> bool:
        """
        Applies the reconvergence rule.
        
        Process:
        1. Choose random 'k' in range.
        2. Iterate over potential target nodes.
        3. Try to find 'k' disjoint paths.
        4. Update graph with paths and types.
        """
        if not self.can_apply(generator, start_node):
            return False
        
        k = random.randint(self.k_range[0], self.k_range[1])
        target_pool = list(generator.qca_arch_graph.nodes())
        random.shuffle(target_pool)

        for target_node in target_pool:
            if target_node not in generator.used_nodes:
                paths = self._find_disjoint_paths(generator, start_node, target_node, k)
                
                if paths:
                    try:
                        for path in paths:
                            generator.used_nodes.update(path)
                            nx.add_path(generator.placement_graph, path)
                        
                        generator.placement_graph.nodes[target_node]['type'] = 'operation'
                        
                        for path in paths:
                            for node in path[1:-1]:
                                if 'type' not in generator.placement_graph.nodes[node]:
                                    generator.placement_graph.nodes[node]['type'] = 'routing'
                        
                        self._increment_counter()
                        logger.debug(f"ReconvergenceRule: Applied. {k} paths from {start_node} -> {target_node}")
                        return True
                        
                    except Exception as e:
                        logger.error(f"ReconvergenceRule: Error applying: {e}", exc_info=True)
                        return False
        
        logger.debug(f"ReconvergenceRule: No viable target found for {start_node}")
        return False

    def _find_disjoint_paths(self, generator: 'QcaGrammarGenerator', source: Tuple[int, int], target: Tuple[int, int], k: int) -> Optional[List[List[Tuple[int, int]]]]:
        """Finds 'k' paths of equal length that are disjoint at intermediate nodes."""
        try:
            temp_graph = generator.qca_arch_graph.copy()
            nodes_to_avoid = generator.used_nodes - {source}
            temp_graph.remove_nodes_from(nodes_to_avoid)

            all_paths = list(nx.all_shortest_paths(temp_graph, source=source, target=target))
            
            if not all_paths: return None
            
            path_length = len(all_paths[0]) - 1
            if path_length > self.max_path_length: return None
            if len(all_paths) < k: return None

            random.shuffle(all_paths)
            selected_paths = []
            claimed_intermediate = set()

            for path in all_paths:
                intermediate_nodes = set(path[1:-1])
                if claimed_intermediate.isdisjoint(intermediate_nodes):
                    selected_paths.append(path)
                    claimed_intermediate.update(intermediate_nodes)
                    if len(selected_paths) == k:
                        return selected_paths
            
            return None
                        
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        except Exception as e:
            logger.error(f"ReconvergenceRule: Unexpected error: {e}", exc_info=True)
            return None
    
    def estimate_cost(self, generator: 'QcaGrammarGenerator', start_node: Tuple[int, int]) -> Optional[int]:
        """Estimates node cost."""
        if not self.can_apply(generator, start_node):
            return None
        
        avg_k = (self.k_range[0] + self.k_range[1]) / 2
        avg_path_length = self.max_path_length / 2
        return int(avg_k * avg_path_length) + 1
