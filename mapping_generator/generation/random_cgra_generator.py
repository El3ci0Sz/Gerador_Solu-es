# mapping_generator/generation/random_cgra_generator.py

import random
import networkx as nx
from typing import Optional
from ..architectures.cgra import CgraArch
from ..utils.mapping import Mapping
from ..utils.graph_processor import GraphProcessor

class RandomCgraGenerator:
    """
    Generates a random valid mapping of a DFG to a CGRA.
    Ensures that generated graphs are valid DAGs.
    """

    def __init__(self, dfg_size: int, II: int, cgra_dim: tuple, bits: str, 
                 alpha: float = 0.5, alpha2: float = 0.4):
        """
        Initializes the random generator.

        Args:
            dfg_size (int): The number of nodes in the DFG to generate.
            II (int): The initiation interval for the mapping.
            cgra_dim (tuple): The (rows, cols) dimensions of the CGRA.
            bits (str): The 4-bit string defining the CGRA interconnect.
            alpha (float): Probability of adding extra connections during routing.
            alpha2 (float): Probability of removing existing connections during routing.
        """
        self.dfg_size = dfg_size
        self.II = II
        self.cgra_dim = cgra_dim
        self.bits = bits
        self.alpha = alpha
        self.alpha2 = alpha2
        
        cgra_architecture = CgraArch(self.cgra_dim, self.bits, self.II)
        self.cgra_graph = cgra_architecture.get_graph()

    def generate_mapping(self, max_attempts: int = 20000) -> Optional[Mapping]:
        """
        Attempts to generate a valid (placement + routing) mapping.
        
        Args:
            max_attempts (int): Max number of tries before giving up.

        Returns:
            Optional[Mapping]: A valid Mapping object or None if generation fails.
        """
        for _ in range(max_attempts):
            mapping = Mapping(self.dfg_size)
            self._perform_placement(mapping)
            self._perform_routing(mapping)
            
            if GraphProcessor(mapping).is_valid():
                return mapping
        
        return None

    def _perform_placement(self, mapping: Mapping):
        """Performs a random placement of DFG nodes onto the CGRA grid."""
        node_ids = [f'op_{i}' for i in range(self.dfg_size)]
        
        available_pes = [(r, c, t) for r in range(self.cgra_dim[0]) 
                                   for c in range(self.cgra_dim[1]) 
                                   for t in range(self.II)]
        
        if len(available_pes) < self.dfg_size:
            raise ValueError("Not enough PEs in the CGRA to place the DFG.")
            
        placed_pes = random.sample(available_pes, self.dfg_size)

        for i, node_id in enumerate(node_ids):
            mapping.placement[node_id] = placed_pes[i]

    def _perform_routing(self, mapping: Mapping):
        """
        Generates random routing using a 3-phase logic:
        1. Add Edges randomly based on alpha.
        2. Prune Edges randomly based on alpha2.
        3. Ensure Connectivity between disconnected components.
        """
        node_ids = list(mapping.placement.keys())
        temp_dfg = nx.DiGraph()
        temp_dfg.add_nodes_from(node_ids)
        
        created_edges = set()
        for source_id in node_ids:
            for dest_id in node_ids:
                if source_id != dest_id and (source_id, dest_id) not in created_edges and (dest_id, source_id) not in created_edges:
                    if random.random() < self.alpha:
                        u, v = (source_id, dest_id) if random.random() < 0.5 else (dest_id, source_id)
                        created_edges.add((u, v))
        
        if self.alpha2 > 0:
            for u, v in list(created_edges):
                if random.random() < self.alpha2:
                    created_edges.remove((u, v))

        temp_dfg.add_edges_from(created_edges)
        
        components = list(nx.weakly_connected_components(temp_dfg))
        if len(components) > 1:
            for i in range(len(components) - 1):
                comp1 = list(components[i])
                comp2 = list(components[i+1])
                
                u = random.choice(comp1)
                v = random.choice(comp2)

                source_id, dest_id = (u, v) if random.random() < 0.5 else (v, u)
                created_edges.add((source_id, dest_id))
        
        mapping.routing = {edge: [] for edge in created_edges}
