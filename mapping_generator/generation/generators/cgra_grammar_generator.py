import random
import logging
import networkx as nx
from typing import List, Tuple, Optional

from ...architectures.cgra import CgraArch
from ...utils.file_saver import FileSaver, OutputPathManager
from ...utils.mapping import Mapping
from ...utils.graph_processor import GraphProcessor
from ..grammar import Grammar
from ..strategies.base import DifficultyStrategy
from ..strategies.recipes import generate_recipes

logger = logging.getLogger(__name__)

class CgraGrammarGenerator:
    """
    CGRA Graph Generator using procedural grammar with difficulty strategies.
    """
    
    def __init__(self, strategy: DifficultyStrategy, k_target: int,
                 arch_sizes: List[Tuple[int, int]], cgra_params: dict,
                 graph_range: Tuple[int, int], k_range: Tuple[int, int],
                 no_extend_io: bool, max_path_length: int,
                 fixed_ii: Optional[int], retries_multiplier: int,
                 file_saver: FileSaver, allow_partial_recipe: bool = False):
        
        self.strategy = strategy
        self.k_target = k_target
        self.arch_sizes = arch_sizes
        self.cgra_params = cgra_params
        self.graph_range = graph_range
        self.k_range = k_range
        self.no_extend_io = no_extend_io
        self.max_path_length = max_path_length
        self.fixed_ii = fixed_ii
        self.retries_multiplier = retries_multiplier
        self.file_saver = file_saver
        self.allow_partial_recipe = allow_partial_recipe
        
        self.graphs_generated = 0
    
    def generate(self) -> bool:
        """
        Main generation entry point.
        """
        logger.info(
            f"CGRA Grammar: Strategy={self.strategy.get_strategy_name()}, "
            f"Target={self.k_target}, FlexibleRecipe={self.allow_partial_recipe}"
        )
        
        self.graphs_generated = self._generation_loop()
        
        if self.graphs_generated < self.k_target:
            fallback = self.strategy.get_fallback_strategy()
            
            if fallback:
                logger.warning(
                    f"Generated only {self.graphs_generated}/{self.k_target}. "
                    f"Activating fallback: {fallback.get_strategy_name()}"
                )
                
                original_strategy = self.strategy
                self.strategy = fallback
                
                remaining = self.k_target - self.graphs_generated
                fallback_generated = self._generation_loop(
                    k_to_generate=remaining,
                    is_fallback=True
                )
                
                self.strategy = original_strategy
                self.graphs_generated += fallback_generated
        
        logger.info(
            f"Generation finished: {self.graphs_generated}/{self.k_target} graphs. "
            f"Rate: {self.graphs_generated/self.k_target*100:.1f}%"
        )
        
        stats = self.strategy.get_statistics()
        logger.info(f"Strategy statistics: {stats}")
        
        return self.graphs_generated > 0
    
    def _generation_loop(self, k_to_generate: Optional[int] = None, 
                        is_fallback: bool = False) -> int:
        """
        Internal loop to generate k graphs with retries.
        """
        if k_to_generate is None:
            k_to_generate = self.k_target
        
        saved_count = 0
        total_attempts = 0
        max_attempts = k_to_generate * self.retries_multiplier
        consecutive_failures = 0
        max_consecutive_failures = 300
        
        while saved_count < k_to_generate and total_attempts < max_attempts:
            if consecutive_failures >= max_consecutive_failures:
                logger.warning(
                    f"Stopping after {max_consecutive_failures} consecutive failures."
                )
                break
            
            total_attempts += 1
            
            try:
                difficulty, recipe = self.strategy.select_difficulty(
                    graph_size=self.graph_range[0],
                    k_range=self.k_range
                )
            except Exception as e:
                logger.error(f"Error selecting difficulty: {e}")
                consecutive_failures += 1
                continue
            
            placement_graph = self._attempt_generation(recipe)
            
            if placement_graph:
                saved_count += 1
                consecutive_failures = 0
                self.strategy.on_success(difficulty)
                
                self._save_graph(
                    graph=placement_graph,
                    index=self.graphs_generated + saved_count,
                    difficulty=difficulty,
                    is_fallback=is_fallback
                )
                
                if saved_count % 10 == 0:
                    logger.info(f"Progress: {saved_count}/{k_to_generate}")
            else:
                consecutive_failures += 1
                self.strategy.on_failure(difficulty)
        
        logger.info(
            f"Loop finished: {saved_count}/{k_to_generate} graphs "
            f"in {total_attempts} attempts. "
            f"Success rate: {saved_count/max(total_attempts, 1)*100:.2f}%"
        )
        
        return saved_count
    
    def _attempt_generation(self, recipe: dict) -> Optional[nx.DiGraph]:
        """
        Tries to generate a single valid graph for a given recipe.
        """
        try:
            target_nodes = self.graph_range[0]
            arch_size = random.choice(self.arch_sizes)
            final_ii = self._calculate_ii(target_nodes, arch_size)
            
            cgra_architecture = CgraArch(
                dimensions=arch_size,
                interconnect_bits=self.cgra_params['bits'],
                ii=final_ii
            )
            
            grammar = Grammar(
                architecture_graph=cgra_architecture.get_graph(),
                border_nodes=cgra_architecture.get_border_nodes(),
                grid_dim=arch_size,
                target_size=target_nodes,
                recipe=recipe,
                k_range=self.k_range,
                max_path_length=self.max_path_length,
                no_extend_io=self.no_extend_io,
                allow_partial_recipe=self.allow_partial_recipe
            )
            
            placement_graph = grammar.generate()
            
            if not placement_graph:
                return None
            
            for i, node_coord in enumerate(list(placement_graph.nodes())):
                placement_graph.nodes[node_coord]['name'] = f'temp_op_{i}'
            
            temp_mapping = Mapping(placement_graph.number_of_nodes())
            temp_mapping.placement = {
                data['name']: node 
                for node, data in placement_graph.nodes(data=True)
            }
            
            for u, v in placement_graph.edges():
                u_name = placement_graph.nodes[u].get('name')
                v_name = placement_graph.nodes[v].get('name')
                if u_name and v_name:
                    temp_mapping.routing[(u_name, v_name)] = []
            
            if GraphProcessor(temp_mapping).is_valid():
                return placement_graph
            else:
                return None
                
        except Exception as e:
            return None
    
    def _calculate_ii(self, num_nodes: int, arch_size: Tuple[int, int]) -> int:
        """
        Calculates the minimal Initiation Interval (II).
        """
        if self.fixed_ii is not None:
            return self.fixed_ii
        
        rows, cols = arch_size
        total_pes = rows * cols
        
        if total_pes == 0:
            return 1
        
        from math import ceil
        return int(ceil(num_nodes / total_pes))
    
    def _save_graph(self, graph: nx.DiGraph, index: int, 
                   difficulty: int, is_fallback: bool):
        """
        Saves the generated graph to disk.
        """
        for i, node_coord in enumerate(list(graph.nodes())):
            graph.nodes[node_coord]['name'] = f'add{i}'
            graph.nodes[node_coord]['opcode'] = 'add'
        
        arch_size = random.choice(self.arch_sizes)
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        final_ii = self._calculate_ii(num_nodes, arch_size)
        bits = self.cgra_params['bits']
        interconnect_name = OutputPathManager.get_interconnect_name(bits)
        
        all_recipes = generate_recipes(difficulty if isinstance(difficulty, int) else 10)
        recipe = all_recipes.get(difficulty) if isinstance(difficulty, int) else None
        
        strategy_name = self.strategy.get_strategy_name()
        
        subdirs = OutputPathManager.build_subdirs(
            tec_name="CGRA",
            gen_mode="grammar",
            difficulty=strategy_name,
            interconnect=interconnect_name,
            arch_size=arch_size,
            num_nodes=num_nodes
        )
        
        filename = OutputPathManager.build_filename(
            tec_name="CGRA",
            arch_size=arch_size,
            num_nodes=num_nodes,
            num_edges=num_edges,
            difficulty=difficulty,
            index=index,
            is_fallback=is_fallback
        )
        
        metadata = OutputPathManager.build_metadata(
            tec_name="CGRA",
            num_nodes=num_nodes,
            num_edges=num_edges,
            arch_size=arch_size,
            gen_mode="grammar",
            difficulty=difficulty,
            recipe=recipe,
            ii=final_ii,
            bits=bits,
            interconnect_name=interconnect_name
        )
        
        self.file_saver.save_graph(graph, filename, metadata, subdirs)
