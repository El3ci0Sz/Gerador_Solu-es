# mapping_generator/architectures/qca.py

import networkx as nx
import logging
from typing import Optional, Tuple, Set

logger = logging.getLogger(__name__)

class QCA:
    """Generates connectivity graphs for QCA architectures."""
    
    def __init__(self, dimensions: Optional[tuple] = None, arch_type: str = 'U', dim: Optional[tuple] = None):
        """
        Initializes the QCA architecture generator.

        Args:
            dimensions (tuple): The (rows, cols) dimensions.
            arch_type (str): The clock zone scheme ('U', 'T', 'R').
            dim (tuple, optional): Alias for dimensions for backward compatibility.
        """
        if dimensions is None and dim is not None:
            dimensions = dim
        if dimensions is None:
            raise ValueError("Dimensions must be provided via 'dimensions' or 'dim'.")
            
        self.dim = dimensions
        self.arch_type = arch_type.upper()
        self.USE_CLOCK_TILE = [[1, 2, 3, 4], [4, 3, 2, 1], [3, 4, 1, 2], [2, 1, 4, 3]]
        self.RES_CLOCK_TILE = [[4, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2]]
        self._graph = None

    def get_graph(self) -> nx.DiGraph:
        """Returns the generated QCA connectivity graph, creating it if necessary."""
        if self._graph is None:
            self._graph = self._generate_connectivity_graph()
        return self._graph

    def expand_grid(self, required_cells: int):
        """
        Doubles the grid dimensions if the current size is insufficient.
        
        Args:
            required_cells (int): The minimum number of cells required.
        """
        current_cells = self.dim[0] * self.dim[1]
        if current_cells >= required_cells:
            return
            
        rows, cols = self.dim
        if rows <= cols:
            rows *= 2
        else:
            cols *= 2
            
        logger.info(f"Expanding QCA grid from {self.dim[0]}x{self.dim[1]} to {rows}x{cols}.")
        self.dim = (rows, cols)
        self._graph = None

    def get_border_nodes(self) -> Set[Tuple[int, int]]:
        """
        Identifies nodes located on the edges of the grid.

        Returns:
            Set[Tuple[int, int]]: A set of coordinates (row, col) for border nodes.
        """
        rows, cols = self.dim
        borders = set()
        if rows < 2 or cols < 2:
            return set([(r, c) for r in range(rows) for c in range(cols)])
            
        for r in range(rows):
            borders.add((r, 0))
            borders.add((r, cols - 1))
        for c in range(cols):
            borders.add((0, c))
            borders.add((rows - 1, c))
        return borders

    def _generate_connectivity_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        all_nodes = [(r, c) for r in range(self.dim[0]) for c in range(self.dim[1])]
        
        for node in all_nodes:
            valid_neighbors = set()
            if self.arch_type == "U":
                valid_neighbors = self._get_neighbors_by_clock_flow(node, self._get_use_clock_zone)
            elif self.arch_type == "T":
                valid_neighbors = self._get_neighbors_2ddwave(node)
            elif self.arch_type == "R": 
                valid_neighbors = self._get_neighbors_by_clock_flow(node, self._get_res_clock_zone)
            else:
                raise ValueError(f"Unknown architecture type: {self.arch_type}")
                
            for neighbor in valid_neighbors:
                graph.add_edge(node, neighbor)
        return graph

    def _is_valid_node(self, node: tuple) -> bool:
        r, c = node
        return 0 <= r < self.dim[0] and 0 <= c < self.dim[1]
    
    def _get_use_clock_zone(self, node: tuple) -> int:
        r, c = node
        return self.USE_CLOCK_TILE[r % 4][c % 4]

    def _get_res_clock_zone(self, node: tuple) -> int:
        r, c = node
        return self.RES_CLOCK_TILE[r % 4][c % 4]
    
    def _get_neighbors_by_clock_flow(self, node: tuple, clock_zone_func) -> set:
        r, c = node
        source_zone = clock_zone_func(node)
        target_zone = (source_zone % 4) + 1
        potential_neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        valid_neighbors = set()
        
        for neighbor in potential_neighbors:
            if self._is_valid_node(neighbor) and clock_zone_func(neighbor) == target_zone:
                valid_neighbors.add(neighbor)
        return valid_neighbors

    def _get_neighbors_2ddwave(self, node: tuple) -> set:
        r, c = node
        potential_neighbors = [(r, c + 1), (r + 1, c)]
        return {n for n in potential_neighbors if self._is_valid_node(n)}
