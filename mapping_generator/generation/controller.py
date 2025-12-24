import logging
import random
from typing import Optional, List, Tuple
from math import ceil
import os

from .generators.cgra_grammar_generator import CgraGrammarGenerator
from .generators.cgra_random_generator import CgraRandomGenerator
from .qca_generation.QcaMultiClusterGenerator import QcaMultiClusterGenerator
from .qca_generation.QcaGrammarGenerator import QcaGrammarGenerator

from ..architectures.qca import QCA
from .strategies import SystematicStrategy, RandomStrategy
from ..utils.file_saver import FileSaver, OutputPathManager
from .strategies.recipes import generate_recipes
from ..utils.qca_analysis import QcaValidator, QcaMetrics
from ..utils.visualizer import GraphVisualizer

logger = logging.getLogger(__name__)

class QcaGeneratorWithSave:
    """
    Internal wrapper to coordinate QCA generation, validation, metrics calculation, and file saving.
    """
    def __init__(self, k_target, arch_sizes, qca_arch, num_inputs,
                 num_derivations, routing_factor, retries_multiplier, file_saver,
                 balanced=True, visualize=False):
        """
        Initialize QCA generator wrapper.
        """
        self.k_target = k_target
        self.arch_sizes = arch_sizes
        self.qca_arch_type = qca_arch
        self.num_inputs = num_inputs
        self.num_derivations = num_derivations
        self.routing_factor = routing_factor
        self.retries_multiplier = retries_multiplier
        self.file_saver = file_saver
        self.visualize = visualize
        
        if balanced:
            self.strategy = 'multicluster'
            self.strategy_label = 'Balanced (MultiCluster)'
            logger.info("QCA Mode: BALANCED - MultiCluster (100% clock valid)")
        else:
            self.strategy = 'grammar'
            self.strategy_label = 'Unbalanced (Grammar)'
            logger.warning("QCA Mode: UNBALANCED - Grammar")
            logger.warning("Note: Graphs may have unequal path latencies (unbalanced).")
        
        self.graphs_generated = 0

    def generate(self) -> bool:
        """Executes generation loop with integrated Validation and Metrics calculation."""
        logger.info(f"Starting {self.strategy_label} generation")
        logger.info(f"Target: {self.k_target} graphs")
        
        saved_count = 0
        total_attempts = 0
        max_attempts = self.k_target * self.retries_multiplier
        
        generated_count = 0
        
        while saved_count < self.k_target and total_attempts < max_attempts:
            total_attempts += 1
            
            if total_attempts % 50 == 0:
                logger.info(f"Progress: {saved_count}/{self.k_target} saved, attempt {total_attempts}/{max_attempts}")
            
            try:
                initial_arch_size = random.choice(self.arch_sizes)
                qca_architecture = QCA(dimensions=initial_arch_size, arch_type=self.qca_arch_type)
                
                generator = self._get_generator_instance(qca_architecture)
                placement_graph = generator.generate()
                
                if not placement_graph:
                    continue
                
                if placement_graph.number_of_nodes() == 0:
                    continue
                
                final_arch_size = qca_architecture.dim 

                generated_count += 1
                
                if generated_count <= 5:
                    logger.debug(
                        f"Generated graph #{generated_count}: "
                        f"{placement_graph.number_of_nodes()} nodes, "
                        f"{placement_graph.number_of_edges()} edges "
                        f"(Grid: {final_arch_size[0]}x{final_arch_size[1]})"
                    )
                
                try:
                    is_valid, errors = QcaValidator.validate(
                        placement_graph, 
                        qca_architecture.get_border_nodes()
                    )
                    if not is_valid and generated_count <= 3:
                        logger.debug(f"Validation failed: {errors[:2]}")
                except Exception as e:
                    logger.error(f"Validation exception: {e}")
                
                try:
                    metrics = QcaMetrics.calculate_all(placement_graph)
                except Exception:
                    metrics = {}
                
                try:
                    saved_count += 1
                    self._save_graph(placement_graph, saved_count, final_arch_size, metrics)
                except Exception as e:
                    logger.error(f"Save exception: {e}", exc_info=True)
                    save_failed += 1
                    saved_count -= 1
                    continue

            except Exception as e:
                logger.debug(f"Outer exception at attempt {total_attempts}: {e}")
                continue
        
        self.graphs_generated = saved_count
        
        logger.info(f"{self.strategy_label} Generation Complete:")
        logger.info(f"Saved: {saved_count}/{self.k_target}")
        logger.info(f"Attempts: {total_attempts}")
        
        return saved_count > 0
    
    def _get_generator_instance(self, qca_architecture):
        if self.strategy == 'multicluster':
            return QcaMultiClusterGenerator(
                qca_architecture=qca_architecture,
                num_inputs=self.num_inputs,
                target_depth=self.num_derivations
            )
        elif self.strategy == 'grammar':
            return QcaGrammarGenerator(
                qca_architecture=qca_architecture,
                num_inputs=self.num_inputs,
                num_derivations=self.num_derivations,
                routing_factor=self.routing_factor,
                strict_balance=False
            )
        else:
            raise ValueError(f"Invalid QCA strategy: {self.strategy}")

    def _save_graph(self, graph, index, arch_size, metrics):
        """Saves the graph. arch_size must be the FINAL grid dimensions."""
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        difficulty_str = f"i{self.num_inputs}d{self.num_derivations}"
        
        filename = OutputPathManager.build_filename(
            tec_name='QCA',
            arch_size=arch_size,
            num_nodes=num_nodes,
            num_edges=num_edges,
            difficulty=difficulty_str,
            index=index
        )
        
        subdirs = OutputPathManager.build_subdirs(
            tec_name='QCA',
            gen_mode=self.strategy,
            arch_size=arch_size,
            num_nodes=num_nodes,
            qca_arch_type=self.qca_arch_type
        )
        
        metadata = {
            'tec_name': 'QCA',
            'tec': 'qca',
            'technology': 'QCA',
            'gen_mode': f"qca_{self.strategy}",
            'mode': self.strategy,
            'qca_strategy': self.strategy,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'arch_size': list(arch_size),
            'qca_arch_type': self.qca_arch_type,
            'difficulty': difficulty_str,
            'num_inputs': self.num_inputs,
            'num_derivations': self.num_derivations,
            'metrics': metrics
        }
        
        try:
            paths = self.file_saver.save_graph(graph, filename, metadata, subdirs)
            
            if self.visualize:
                full_dir = os.path.join(self.file_saver.output_dir, subdirs)
                base_path = os.path.join(full_dir, filename)
                
                grid_path = f"{base_path}.grid"
                GraphVisualizer.save_placement_grid(graph, arch_size, grid_path)
                
                if not self.file_saver.no_images:
                    phys_dot_path = f"{base_path}.phys.dot"
                    phys_png_path = f"{base_path}.phys.png"
                    GraphVisualizer.generate_physical_dot(graph, arch_size, phys_dot_path)
                    try:
                        os.system(f"neato -n2 -Tpng {phys_dot_path} -o {phys_png_path}")
                    except Exception:
                        pass
            
            if paths and 'json' in paths:
                logger.info(f"Saved #{index} | Grid:{arch_size[0]}x{arch_size[1]} | Nodes:{num_nodes}")
            else:
                logger.warning(f"Failed to save {self.strategy} graph #{index}")
            return paths
        except Exception as e:
            logger.error(f"Error saving graph #{index}: {e}", exc_info=True)
            raise


class GenerationTask:
    def __init__(self, tec: str, gen_mode: str, k: int, output_dir: str = 'results', 
                 no_images: bool = False, **kwargs):
        self.tec = tec
        self.gen_mode = gen_mode
        self.k = k
        self.output_dir = output_dir
        self.no_images = no_images
        self.params = kwargs
        self.file_saver = FileSaver(output_dir, no_images)
        self.generator = None
        self._validate_configuration()
    
    def _validate_configuration(self):
        if self.tec not in ['cgra', 'qca']:
            raise ValueError(f"Invalid technology '{self.tec}'")
        if self.gen_mode not in ['grammar', 'random']:
            raise ValueError(f"Invalid mode '{self.gen_mode}'")
    
    def run(self) -> bool:
        try:
            self.generator = self._create_generator()
            if not self.generator:
                return False
            
            logger.info(f"Starting generation: {self.tec.upper()} + {self.gen_mode}, Target: {self.k} graphs")
            return self.generator.generate()
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            return False
            
    def _create_generator(self):
        if self.tec == 'cgra':
            if self.gen_mode == 'grammar':
                return self._create_cgra_grammar_generator()
            elif self.gen_mode == 'random':
                return self._create_cgra_random_generator()
        elif self.tec == 'qca':
            if self.gen_mode == 'grammar':
                return self._create_qca_generator()
        raise ValueError(f"Unsupported combination: {self.tec} + {self.gen_mode}")

    def _create_cgra_grammar_generator(self):
        strategy = self._create_difficulty_strategy()
        return CgraGrammarGenerator(
            strategy=strategy,
            k_target=self.k,
            arch_sizes=self.params.get('arch_sizes', [(4, 4)]),
            cgra_params=self.params.get('cgra_params', {'bits': '1000'}),
            graph_range=self.params.get('graph_range', (10, 10)),
            k_range=self.params.get('k_range', (2, 3)),
            no_extend_io=self.params.get('no_extend_io', False),
            max_path_length=self.params.get('max_path_length', 15),
            fixed_ii=self.params.get('ii', None),
            retries_multiplier=self.params.get('retries_multiplier', 150),
            file_saver=self.file_saver,
            allow_partial_recipe=self.params.get('flexible_recipe', False)
        )

    def _create_cgra_random_generator(self):
        return CgraRandomGenerator(
            k_target=self.k,
            arch_sizes=self.params.get('arch_sizes', [(4, 4)]),
            cgra_params=self.params.get('cgra_params', {'bits': '1000'}),
            graph_range=self.params.get('graph_range', (10, 10)),
            alpha=self.params.get('alpha', 0.3),
            fixed_ii=self.params.get('ii', None),
            retries_multiplier=self.params.get('retries_multiplier', 150),
            file_saver=self.file_saver
        )

    def _create_qca_generator(self):
        arch_sizes = self.params.get('arch_sizes', [(4, 4)])
        qca_arch = self.params.get('qca_arch', 'U')
        num_inputs = self.params.get('num_inputs', 3)
        num_derivations = self.params.get('num_derivations', 10)
        routing_factor = self.params.get('routing_factor', 2.5)
        retries_multiplier = self.params.get('retries_multiplier', 150)
        visualize = self.params.get('visualize', False)
        
        balanced = self.params.get('balanced', True)
        if self.params.get('unbalanced', False):
            balanced = False
            
        return QcaGeneratorWithSave(
            k_target=self.k,
            arch_sizes=arch_sizes,
            qca_arch=qca_arch,
            num_inputs=num_inputs,
            num_derivations=num_derivations,
            routing_factor=routing_factor,
            retries_multiplier=retries_multiplier,
            file_saver=self.file_saver,
            balanced=balanced,
            visualize=visualize
        )

    def _create_difficulty_strategy(self):
        strategy_name = self.params.get('strategy', 'systematic')
        if strategy_name == 'systematic':
            return SystematicStrategy(difficulty=self.params.get('difficulty', 1))
        elif strategy_name == 'random':
            diff_range = self.params.get('difficulty_range', (1, 10))
            return RandomStrategy(difficulty_range=tuple(diff_range), adaptive=self.params.get('adaptive', True))
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

def get_ii(num_nodes: int, arch_size: tuple, fixed_ii: Optional[int] = None) -> int:
    if fixed_ii is not None:
        return fixed_ii
    rows, cols = arch_size
    total_pes = rows * cols
    if total_pes == 0: return 1
    return int(ceil(num_nodes / total_pes))
