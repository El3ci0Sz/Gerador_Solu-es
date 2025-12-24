import random
import networkx as nx
import logging
from typing import Optional, List, Tuple, Set

from ...architectures.qca import QCA
from .rules.tree_rule import TreeRule
from .rules.reconvergence_rule import ReconvergenceRule

logger = logging.getLogger(__name__)

class QcaGrammarGenerator:
    """
    Procedural QCA mapping generator using Stochastic Grammars.
    
    OPTIMIZED VERSION:
    - Uses A* Pathfinding (no graph copying) for high performance.
    - Supports virtual crossovers (wire crossing) to resolve congestion.
    - Robust Input/Output protection.
    """
    
    def __init__(self, qca_architecture: QCA, num_inputs: int, num_derivations: int, 
                 routing_factor: float = 2.5, strict_balance: bool = False):
        self.qca_architecture = qca_architecture
        self.num_inputs = num_inputs
        self.num_derivations = num_derivations
        self.routing_factor = routing_factor
        self.strict_balance = strict_balance
        
        self.placement_graph = nx.DiGraph()
        self.used_nodes: Set[Tuple[int, int]] = set()
        
        self.rules = [TreeRule(), ReconvergenceRule(k_range=(2, 2))]
        self.qca_arch_graph = None
        self.qca_border_nodes = None

        self.COST_FREE = 1
        self.COST_CROSSOVER = 15      
        self.COST_BLOCKED = 999999   
        
        if self.strict_balance:
            self.MAX_PATH_LENGTH = 10
            self.MAX_GRID_EXPANSIONS = 3
        else:
            self.MAX_PATH_LENGTH = 80 
            self.MAX_GRID_EXPANSIONS = 2
            
        self.MAX_MERGE_CANDIDATES = 20

    def generate(self) -> Optional[nx.DiGraph]:
        """Orchestrates the generation pipeline."""
        try:
            self.qca_arch_graph = self.qca_architecture.get_graph()
            self.qca_border_nodes = self.qca_architecture.get_border_nodes()
            self.placement_graph.clear()
            self.used_nodes.clear()

            # 1. Seed Inputs
            if not self._seed_input_nodes():
                return None
            
            # 2. Grow
            for _ in range(self.num_derivations):
                self._apply_growth_rule()
            
            if self.placement_graph.number_of_nodes() <= self.num_inputs:
                return None

            # 3. Merge
            if not self._merge_trees():
                return None

            # 4. Route Outputs to Border
            self._ensure_all_outputs_on_border()

            # 5. Balancing
            if self.strict_balance:
                self._balance_graph_strict()
            
            # 6. Final Validation & Marking
            if self.placement_graph.number_of_nodes() > 0:
                if self._validate_final_graph():
                    self._mark_graph_outputs()
                    
                    self._log_stats()
                    return self.placement_graph
            
            return None
            
        except Exception as e:
            logger.error(f"Error in generator: {e}", exc_info=True)
            return None

    def _mark_graph_outputs(self):
        """
        Identifies leaf nodes (out_degree == 0) and marks them as 'output'.
        This makes the grid visualization much clearer.
        """
        for node in self.placement_graph.nodes():
            if self.placement_graph.out_degree(node) == 0:
                current_type = self.placement_graph.nodes[node].get('type')
                
                if current_type != 'input':
                    self.placement_graph.nodes[node]['type'] = 'output'
                    self.placement_graph.nodes[node]['name'] = f"OUT_{node}"

    def _seed_input_nodes(self) -> bool:
        """Seeds input nodes on border."""
        available = list(self.qca_border_nodes)
        if self.num_inputs > len(available): return False
        
        input_nodes = random.sample(available, self.num_inputs)
        for node in input_nodes:
            self.placement_graph.add_node(node, type='input', name=f"in_{node}")
            self.used_nodes.add(node)
        return True

    def _apply_growth_rule(self) -> bool:
        """Applies random growth rule."""
        leaf_nodes = [n for n, d in self.placement_graph.out_degree() if d == 0]
        if not leaf_nodes: return False
        
        start_node = random.choice(leaf_nodes)
        rule = random.choice(self.rules)
        
        # Prefer tree growth initially to spread out
        if len(self.used_nodes) < 10:
            rule = next((r for r in self.rules if isinstance(r, TreeRule)), rule)
            
        try:
            return rule.apply(self, start_node)
        except Exception:
            return False

    def _merge_trees(self) -> bool:
        """Connects disconnected components using optimized A* pathfinding."""
        expansions_left = self.MAX_GRID_EXPANSIONS
        
        while len(list(nx.weakly_connected_components(self.placement_graph))) > 1:
            components = list(nx.weakly_connected_components(self.placement_graph))
            
            candidates = self._find_closest_pairs(components, limit=self.MAX_MERGE_CANDIDATES)
            
            merged = False
            for _, _, n1, n2, _ in candidates:
                type1 = self.placement_graph.nodes[n1].get('type')
                type2 = self.placement_graph.nodes[n2].get('type')
                
                if type1 == 'input' and type2 == 'input': continue
                
                if type1 == 'input':
                    src, dst = n1, n2
                elif type2 == 'input':
                    src, dst = n2, n1
                else:
                    src, dst = (n1, n2) if random.random() < 0.5 else (n2, n1)
                
                path = self._find_optimized_path(src, dst)
                
                if path:
                    self._realize_path(path)
                    merged = True
                    break 
            
            if not merged:
                if expansions_left > 0:
                    self._expand_grid()
                    expansions_left -= 1
                else:
                    return False
                
        return True

    def _find_optimized_path(self, source, target) -> Optional[List[Tuple]]:
        """
        Finds path using A* without copying the graph (Performance Fix).
        Uses a dynamic weight function to handle congestion.
        """
        
        def weight_func(u, v, d):
            if v in self.used_nodes:
                node_type = self.placement_graph.nodes[v].get('type', 'unknown')
                if node_type in ['routing', 'crossover']:
                    return self.COST_CROSSOVER
                elif node_type in ['input', 'output', 'operation', 'convergence']:
                    return self.COST_BLOCKED
            return self.COST_FREE

        def heuristic(u, v):
            return abs(u[0] - v[0]) + abs(u[1] - v[1])

        try:
            path = nx.astar_path(
                self.qca_arch_graph, 
                source, 
                target, 
                heuristic=heuristic, 
                weight=weight_func
            )
            
            total_cost = 0
            for i in range(len(path) - 1):
                total_cost += weight_func(path[i], path[i+1], {})
            
            if total_cost >= self.COST_BLOCKED:
                return None
            if len(path) - 1 > self.MAX_PATH_LENGTH:
                return None
                
            return path
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _realize_path(self, path: List[Tuple]):
        """Adds path to graph and updates node types."""
        nx.add_path(self.placement_graph, path)
        self.used_nodes.update(path)
        
        for node in path[1:-1]:
            current_type = self.placement_graph.nodes[node].get('type')
            
            if current_type == 'routing':
                # Convert to crossover if heavily used
                self.placement_graph.nodes[node]['type'] = 'crossover'
                self.placement_graph.nodes[node]['name'] = f"cross_{node}"
            elif not current_type:
                # New routing
                self.placement_graph.nodes[node]['type'] = 'routing'
                self.placement_graph.nodes[node]['name'] = f"rout_{node}"

    def _find_closest_pairs(self, components, limit=10):
        """Finds candidate merge pairs."""
        candidates = []
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                comp1 = list(components[i])
                comp2 = list(components[j])
                
                # Sample for performance
                nodes1 = random.sample(comp1, min(len(comp1), 10))
                nodes2 = random.sample(comp2, min(len(comp2), 10))
                
                for n1 in nodes1:
                    for n2 in nodes2:
                        dist = abs(n1[0]-n2[0]) + abs(n1[1]-n2[1])
                        # Loose filter
                        if dist < self.MAX_PATH_LENGTH * 2:
                            candidates.append((components[i], components[j], n1, n2, dist))
        
        candidates.sort(key=lambda x: x[4])
        return candidates[:limit]

    def _ensure_all_outputs_on_border(self):
        """Routes outputs to border."""
        outs = [n for n, d in self.placement_graph.out_degree() if d == 0]
        for out in outs:
            if out not in self.qca_border_nodes:
                avail = [b for b in self.qca_border_nodes if b not in self.used_nodes]
                if not avail: continue
                
                # Simple closest check
                best_b = min(avail, key=lambda b: abs(out[0]-b[0]) + abs(out[1]-b[1]))
                path = self._find_optimized_path(out, best_b)
                if path: self._realize_path(path)

    def _expand_grid(self):
        """Expands grid size."""
        current = self.qca_architecture.dim[0] * self.qca_architecture.dim[1]
        self.qca_architecture.expand_grid(int(current * 1.5))
        self.qca_arch_graph = self.qca_architecture.get_graph()
        self.qca_border_nodes = self.qca_architecture.get_border_nodes()

    def _validate_final_graph(self) -> bool:
        if not nx.is_directed_acyclic_graph(self.placement_graph): return False
        if not nx.is_weakly_connected(self.placement_graph): return False
        for n in self.placement_graph.nodes():
            if self.placement_graph.nodes[n].get('type') == 'input':
                if self.placement_graph.in_degree(n) > 0: return False
        return True

    def _balance_graph_strict(self):
        """Basic ASAP leveling."""
        try:
            for n in nx.topological_sort(self.placement_graph):
                preds = list(self.placement_graph.predecessors(n))
                if not preds: self.placement_graph.nodes[n]['level'] = 0
                else: self.placement_graph.nodes[n]['level'] = max(self.placement_graph.nodes[p].get('level', 0) for p in preds) + 1
        except Exception: pass

    def _log_stats(self):
        nodes = self.placement_graph.number_of_nodes()
        logger.debug(f"Generated successfully. Nodes: {nodes}")

    def find_shortest_path_to_new_node(self, source):
        """Used by TreeRule."""
        # For simple growth, we still use the optimized pathfinder but with free-space preference
        targets = []
        r, c = source
        for i in range(max(0, r-6), min(self.qca_architecture.dim[0], r+7)):
            for j in range(max(0, c-6), min(self.qca_architecture.dim[1], c+7)):
                if (i,j) not in self.used_nodes: targets.append((i,j))
        
        random.shuffle(targets)
        for t in targets[:5]:
            path = self._find_optimized_path(source, t)
            if path: return path
        return None
