import random
import networkx as nx
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class Grammar:
    """
    Implements the grammar-based procedural generation of a DFG on an architecture graph.
    """

    def __init__(self, architecture_graph: nx.DiGraph, border_nodes: set, grid_dim: tuple, 
                 target_size: int, recipe: dict, k_range: tuple, max_path_length: int, 
                 no_extend_io: bool, allow_partial_recipe: bool = False):
        self.arch_graph = architecture_graph
        self.border_nodes = border_nodes
        self.grid_dim = grid_dim
        self.target_size = target_size
        self.recipe = recipe
        self.k_range = k_range
        self.max_path_length = max_path_length
        self.no_extend_io = no_extend_io
        self.allow_partial_recipe = allow_partial_recipe
        
        self.placement_graph = nx.DiGraph()
        self.used_nodes = set()
        self.step_counter = 0
        self.reconvergences_created = 0
        self.convergences_created = 0

    def generate(self, growth_timeout: int = 200) -> nx.DiGraph | None:
        """
        Executes the main generation loop.
        """
        if not list(self.arch_graph.nodes()): return None
        available_nodes = list(set(self.arch_graph.nodes()) - self.used_nodes)
        if not available_nodes: return None
        
        start_node = random.choice(available_nodes)
        self.placement_graph.add_node(start_node)
        self.used_nodes.add(start_node)
        
        steps = 0
        while len(self.used_nodes) < self.target_size and steps < growth_timeout:
            if not self._apply_pattern(): 
                budget = self.target_size - len(self.used_nodes)
                if budget > 0 and not self._tree_rule(budget):
                    break
            steps += 1
            
        recipe_fulfilled = (
            self.reconvergences_created >= self.recipe.get('reconvergence', 0) and
            self.convergences_created >= self.recipe.get('convergence', 0)
        )
        
        if len(self.used_nodes) >= self.target_size:
            if recipe_fulfilled:
                return self.placement_graph
            
            elif self.allow_partial_recipe:
                logger.warning(
                    f"Partial recipe accepted: "
                    f"Rec={self.reconvergences_created}/{self.recipe.get('reconvergence', 0)}, "
                    f"Conv={self.convergences_created}/{self.recipe.get('convergence', 0)}"
                )
                return self.placement_graph
            
            else:
                return None
            
        return None

    def _apply_pattern(self) -> bool:
        """
        Selects and applies a grammar rule based on remaining budget and recipe needs.
        Prioritizes recipe fulfillment over random growth.
        """
        budget = self.target_size - len(self.used_nodes)
        if budget <= 0: return False
        
        recipe_fulfilled = (
            self.reconvergences_created >= self.recipe.get('reconvergence', 0) and
            self.convergences_created >= self.recipe.get('convergence', 0)
        )
        
        if not recipe_fulfilled:
            potential_rules = []
            
            if self.reconvergences_created < self.recipe.get('reconvergence', 0):
                potential_rules.append(self._reconvergence_rule)
            
            if self.convergences_created < self.recipe.get('convergence', 0):
                potential_rules.append(self._convergence_rule)
            
            random.shuffle(potential_rules)
            for rule in potential_rules:
                if rule(budget): return True
        
        if self._tree_rule(budget): return True
        
        return False

    def _tree_rule(self, budget: int) -> bool:
        """
        Expands the graph using a simple tree structure (1 to k).
        """
        if not self.used_nodes or budget < 1: return False
        potential_start_nodes = list(self.used_nodes)
        random.shuffle(potential_start_nodes)
        
        for start_node in potential_start_nodes[:10]:
            k_max_allowed = min(self.k_range[1], budget)
            if k_max_allowed < 1: continue
            k = random.randint(1, k_max_allowed)
            free_nodes = list(set(self.arch_graph.nodes()) - self.used_nodes - {start_node})
            random.shuffle(free_nodes)
            paths, newly_claimed = [], set()
            for target_node in free_nodes:
                if len(paths) >= k: break
                path = self._find_shortest_path(start_node, target_node, newly_claimed)
                if path:
                    new_nodes = set(path) - self.used_nodes
                    if len(newly_claimed.union(new_nodes)) <= budget:
                        paths.append(path)
                        newly_claimed.update(new_nodes)
            if paths:
                self._add_paths_to_placement(paths, f"Tree (k={len(paths)})")
                return True
        return False

    def _convergence_rule(self, budget: int) -> bool:
        """
        Creates a convergence (multiple sources merging into one target).
        """
        min_cost = self.k_range[0]
        if not self.used_nodes or budget < min_cost: return False
        potential_targets = list(self.used_nodes)
        random.shuffle(potential_targets)
        
        for target_node in potential_targets[:5]:
            k_max_allowed = min(self.k_range[1], budget)
            if k_max_allowed < self.k_range[0]: continue
            k = random.randint(self.k_range[0], k_max_allowed)
            source_pool = list(set(self.arch_graph.nodes()) - self.used_nodes - {target_node})
            random.shuffle(source_pool)
            paths, newly_claimed = [], set()
            for source_node in source_pool:
                if len(paths) >= k: break
                path = self._find_shortest_path(source_node, target_node, newly_claimed)
                if path:
                    new_nodes = set(path) - self.used_nodes
                    if len(newly_claimed.union(new_nodes)) <= budget:
                        paths.append(path)
                        newly_claimed.update(new_nodes)
            if len(paths) >= self.k_range[0]:
                self._add_paths_to_placement(paths, f"Convergence (k={len(paths)})")
                self.convergences_created += 1
                return True
        return False

    def _reconvergence_rule(self, budget: int) -> bool:
        """
        Creates a reconvergence (split and merge).
        """
        min_cost = self.k_range[0] + 1
        if not self.used_nodes or budget < min_cost: return False
        k_max_allowed = min(self.k_range[1], budget - 1)
        if k_max_allowed < self.k_range[0]: return False
        k = random.randint(self.k_range[0], k_max_allowed)
        potential_start_nodes = list(self.used_nodes)
        random.shuffle(potential_start_nodes)
        
        for start_node in potential_start_nodes[:5]:
            target_pool = list(set(self.arch_graph.nodes()) - self.used_nodes - {start_node})
            random.shuffle(target_pool)
            for target_node in target_pool[:10]:
                paths_found = self._find_balanced_disjoint_paths(start_node, target_node, k, budget)
                if paths_found:
                    self._add_paths_to_placement(paths_found, f"Reconvergence (k={len(paths_found)})")
                    self.reconvergences_created += 1
                    return True
        return False
        
    def _find_shortest_path(self, source, target, extra_nodes_to_avoid=None) -> list | None:
        """
        Finds the shortest path between source and target considering used nodes.
        """
        try:
            nodes_to_avoid = self.used_nodes.copy()
            if extra_nodes_to_avoid: nodes_to_avoid.update(extra_nodes_to_avoid)
            nodes_to_avoid -= {source, target}
            subgraph = self.arch_graph.copy()
            subgraph.remove_nodes_from(nodes_to_avoid)
            path = nx.shortest_path(subgraph, source=source, target=target)
            return path if path and (len(path) - 1 <= self.max_path_length) else None
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _find_balanced_disjoint_paths(self, source, target, k, budget) -> list | None:
        """
        Finds k disjoint paths between source and target for reconvergence.
        """
        try:
            subgraph = self.arch_graph.copy()
            subgraph.remove_nodes_from(self.used_nodes - {source})
            if target not in subgraph: return None

            all_paths = list(nx.all_shortest_paths(subgraph, source=source, target=target))
            if not all_paths or len(all_paths[0]) > self.max_path_length: return None
            if len(all_paths) < k: return None

            random.shuffle(all_paths)
            selected_paths = []
            claimed_intermediate = set()
            newly_claimed_total = set()

            for path in all_paths:
                intermediate_nodes = set(path[1:-1])
                if claimed_intermediate.isdisjoint(intermediate_nodes):
                    path_new_nodes = (intermediate_nodes | {target}) - self.used_nodes
                    if len(newly_claimed_total.union(path_new_nodes)) <= budget:
                        selected_paths.append(path)
                        claimed_intermediate.update(intermediate_nodes)
                        newly_claimed_total.update(path_new_nodes)
                        if len(selected_paths) == k:
                            return selected_paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return None

    def _add_paths_to_placement(self, paths: list, rule_name: str):
        """
        Adds the generated paths to the main placement graph.
        """
        if not paths: return
        for path in paths:
            self.used_nodes.update(path)
            nx.add_path(self.placement_graph, path)
