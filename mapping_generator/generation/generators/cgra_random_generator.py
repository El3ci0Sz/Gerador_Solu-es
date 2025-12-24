# mapping_generator/generation/generators/cgra_random_generator.py

"""
Gerador de grafos CGRA usando geração aleatória (sem gramática).

Responsabilidades:
- Geração aleatória de DFGs
- Validação de conectividade e DAG
- Salvamento com metadados apropriados
"""

import random
import logging
import networkx as nx
from typing import List, Tuple, Optional
from math import ceil

from ...architectures.cgra import CgraArch
from ...utils.file_saver import FileSaver, OutputPathManager
from ...utils.mapping import Mapping
from ..random_cgra_generator import RandomCgraGenerator

logger = logging.getLogger(__name__)


class CgraRandomGenerator:
    """
    Gerador de grafos CGRA usando geração aleatória (DFG random).
    
    NÃO usa estratégias de dificuldade.
    """
    
    def __init__(self, k_target: int, arch_sizes: List[Tuple[int, int]],
                 cgra_params: dict, graph_range: Tuple[int, int],
                 alpha: float, fixed_ii: Optional[int],
                 retries_multiplier: int, file_saver: FileSaver):
        """
        Inicializa gerador CGRA + random.
        
        Args:
            k_target: Número de grafos a gerar
            arch_sizes: Lista de tamanhos [(rows, cols), ...]
            cgra_params: Dict com 'bits'
            graph_range: (min_nodes, max_nodes)
            alpha: Probabilidade de adicionar arestas extras
            fixed_ii: II fixo ou None
            retries_multiplier: Multiplicador para max tentativas
            file_saver: Instância de FileSaver
        """
        self.k_target = k_target
        self.arch_sizes = arch_sizes
        self.cgra_params = cgra_params
        self.graph_range = graph_range
        self.alpha = alpha
        self.fixed_ii = fixed_ii
        self.retries_multiplier = retries_multiplier
        self.file_saver = file_saver
        
        self.graphs_generated = 0
    
    def generate(self) -> bool:
        """
        Executa loop de geração aleatória.
        
        Returns:
            True se gerou pelo menos algum grafo
        """
        logger.info(
            f"CGRA Random: Target={self.k_target}, "
            f"SizeRange={self.graph_range}, Alpha={self.alpha}"
        )
        
        saved_count = 0
        total_attempts = 0
        max_attempts = self.k_target * self.retries_multiplier
        
        while saved_count < self.k_target and total_attempts < max_attempts:
            total_attempts += 1
            
            try:
                # Tamanho aleatório dentro do range
                num_nodes = random.randint(self.graph_range[0], self.graph_range[1])
                arch_size = random.choice(self.arch_sizes)
                final_ii = self._calculate_ii(num_nodes, arch_size)
                
                # Gerar mapeamento aleatório
                generator = RandomCgraGenerator(
                    dfg_size=num_nodes,
                    II=final_ii,
                    cgra_dim=arch_size,
                    bits=self.cgra_params['bits'],
                    alpha=self.alpha
                )
                
                mapping_obj = generator.generate_mapping()
                
                if mapping_obj:
                    # Converter para grafo NetworkX
                    final_graph = self._build_graph_from_mapping(mapping_obj)
                    
                    saved_count += 1
                    self._save_graph(final_graph, saved_count, num_nodes, arch_size)
                    
                    if saved_count % 10 == 0:
                        logger.info(f"Progresso: {saved_count}/{self.k_target}")
                        
            except Exception as e:
                logger.debug(f"Erro durante geração aleatória: {e}")
                continue
        
        self.graphs_generated = saved_count
        
        logger.info(
            f"Geração Random concluída: {saved_count}/{self.k_target} grafos "
            f"em {total_attempts} tentativas. "
            f"Taxa: {saved_count/max(total_attempts, 1)*100:.2f}%"
        )
        
        return saved_count > 0
    
    def _calculate_ii(self, num_nodes: int, arch_size: Tuple[int, int]) -> int:
        """Calcula II baseado em nodes e arch size."""
        if self.fixed_ii is not None:
            return self.fixed_ii
        
        rows, cols = arch_size
        total_pes = rows * cols
        
        if total_pes == 0:
            return 1
        
        return int(ceil(num_nodes / total_pes))
    
    def _build_graph_from_mapping(self, mapping_obj: Mapping) -> nx.DiGraph:
        """
        Converte Mapping para grafo NetworkX.
        
        Args:
            mapping_obj: Objeto Mapping com placement e routing
        
        Returns:
            Grafo NetworkX
        """
        final_graph = nx.DiGraph()
        node_map = {}
        
        # Adicionar nós
        for node_id, pos in mapping_obj.placement.items():
            final_graph.add_node(tuple(pos))
            node_map[node_id] = tuple(pos)
        
        # Adicionar arestas
        for (src_id, dst_id) in mapping_obj.routing.keys():
            if src_id in node_map and dst_id in node_map:
                final_graph.add_edge(node_map[src_id], node_map[dst_id])
        
        return final_graph
    
    def _save_graph(self, graph: nx.DiGraph, index: int, 
                   num_nodes: int, arch_size: Tuple[int, int]):
        """
        Salva grafo random usando FileSaver.
        
        Args:
            graph: Grafo a salvar
            index: Índice sequencial
            num_nodes: Número de nós (usado para path)
            arch_size: Tamanho da arquitetura
        """
        # Renomear nós
        for i, node_coord in enumerate(list(graph.nodes())):
            graph.nodes[node_coord]['name'] = f'rnd{i}'
            graph.nodes[node_coord]['opcode'] = 'rnd'
        
        num_edges = graph.number_of_edges()
        final_ii = self._calculate_ii(num_nodes, arch_size)
        bits = self.cgra_params['bits']
        interconnect_name = OutputPathManager.get_interconnect_name(bits)
        
        subdirs = OutputPathManager.build_subdirs(
            tec_name="CGRA",
            gen_mode="random",
            interconnect=interconnect_name,
            arch_size=arch_size,
            num_nodes=num_nodes
        )
        
        filename = OutputPathManager.build_filename(
            tec_name="CGRA",
            arch_size=arch_size,
            num_nodes=num_nodes,
            num_edges=num_edges,
            difficulty="NA",  # Random não tem dificuldade
            index=index
        )
        
        metadata = OutputPathManager.build_metadata(
            tec_name="CGRA",
            num_nodes=num_nodes,
            num_edges=num_edges,
            arch_size=arch_size,
            gen_mode="random",
            alpha=self.alpha,
            ii=final_ii,
            bits=bits,
            interconnect_name=interconnect_name
        )
        
        self.file_saver.save_graph(graph, filename, metadata, subdirs)
