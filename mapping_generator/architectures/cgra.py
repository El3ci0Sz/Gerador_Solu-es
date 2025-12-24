# mapping_generator/architectures/cgra.py

from typing import Optional, Set, Tuple
import networkx as nx

class CgraArch:
    """Generates the connectivity graph for a CGRA architecture.
    
    Builds a directed graph representing all possible connections in a CGRA 
    for a given size, Initiation Interval (II), and interconnection scheme.
    """

    def __init__(self, dimensions: Optional[tuple] = None, interconnect_bits: str = '1000', ii: int = 1, dim: Optional[tuple] = None):
        """
        Initializes the CGRA architecture.

        Args:
            dimensions (tuple): The (rows, cols) dimensions of the CGRA grid.
            interconnect_bits (str): A 4-bit string "mdht" representing:
                                     'm': mesh (Von Neumann neighbors)
                                     'd': diagonal (Moore neighbors)
                                     'h': one-hop (knight moves)
                                     't': toroidal connections
            ii (int): The Initiation Interval, representing the time dimension.
            dim (tuple, optional): Alias for dimensions.
        """
        if dimensions is None and dim is not None:
            dimensions = dim
        if dimensions is None:
            raise ValueError("Dimensions must be provided via 'dimensions' or 'dim'.")
            
        if len(interconnect_bits) != 4 or not all(c in '01' for c in interconnect_bits):
            raise ValueError(f"interconnect_bits must be a 4-character binary string, got '{interconnect_bits}'")
        
        self.rows, self.cols = dimensions
        self.bits = interconnect_bits
        self.ii = ii
        self.graph = self._create_base_grid_with_ii()
        self._add_interconnections()

    def get_graph(self) -> nx.DiGraph:
        """Returns the generated CGRA connectivity graph."""
        return self.graph
        
    def get_border_nodes(self) -> Set[Tuple[int, int, int]]:
        """
        Returns a set of all nodes on the physical border of the CGRA.
        
        Returns:
            Set[Tuple[int, int, int]]: Set of (row, col, time) coordinates.
        """
        borders = set()
        for t in range(self.ii):
            for r in range(self.rows):
                for c in range(self.cols):
                    if r == 0 or r == self.rows - 1 or c == 0 or c == self.cols - 1:
                        borders.add((r, c, t))
        return borders

    def _create_base_grid_with_ii(self) -> nx.DiGraph:
        """Creates a grid graph with nodes representing PEs at time steps."""
        graph = nx.DiGraph()
        for t in range(self.ii):
            for r in range(self.rows):
                for c in range(self.cols):
                    graph.add_node((r, c, t))
        return graph

    def _add_interconnections(self):
        """
        Adds all configured connections for every node in the graph.
        
        For II > 1, connections cross time step boundaries (t -> t+1).
        For II = 1, connections are within the same time step (t=0).
        """
        mesh, diagonal, one_hop, toroidal = [bool(int(b)) for b in self.bits]
        
        for t in range(self.ii):
            for r in range(self.rows):
                for c in range(self.cols):
                    source_node = (r, c, t)
                    target_t = t if self.ii == 1 else (t + 1) % self.ii
                    
                    # Temporal connection (same PE, next cycle) if II > 1
                    if self.ii > 1:
                        self.graph.add_edge(source_node, (r, c, target_t))
                    
                    potential_neighbors = []
                    if mesh:
                        potential_neighbors.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
                    if diagonal:
                        potential_neighbors.extend([(r+1, c+1), (r+1, c-1), (r-1, c+1), (r-1, c-1)])
                    if one_hop:
                        potential_neighbors.extend([(r+2, c), (r-2, c), (r, c+2), (r, c-2)])
                    
                    # Add standard physical connections
                    for nr, nc in potential_neighbors:
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            self.graph.add_edge(source_node, (nr, nc, target_t))

                    # Add toroidal connections
                    if toroidal:
                        tor_neighbors = [
                            ((r + 1) % self.rows, c), 
                            ((r - 1 + self.rows) % self.rows, c), 
                            (r, (c + 1) % self.cols), 
                            (r, (c - 1 + self.cols) % self.cols)
                        ]
                        for nr, nc in tor_neighbors:
                            target_node = (nr, nc, target_t)
                            if not self.graph.has_edge(source_node, target_node):
                                self.graph.add_edge(source_node, target_node)
