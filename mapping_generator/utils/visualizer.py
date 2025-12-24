import os
import logging
import networkx as nx
from collections import defaultdict

logger = logging.getLogger(__name__)

class GraphVisualizer:
    """Utility class for exporting graphs to DOT (logical/physical) and ASCII Grid formats."""

    @staticmethod
    def _write_dot_file(graph: nx.DiGraph, dot_filename: str):
        """Internal helper to write a graph to a .dot file with ranked layout."""
        levels = defaultdict(list)
        try:
            for node in nx.topological_sort(graph):
                level = 0
                for pred in graph.predecessors(node):
                    level = max(level, graph.nodes[pred].get('level', -1) + 1)
                graph.nodes[node]['level'] = level
                levels[level].append(node)
        except nx.NetworkXUnfeasible:
            logger.warning("Graph contains cycles; cannot calculate levels for alignment.")
            levels.clear()

        with open(dot_filename, "w", encoding="utf-8") as f:
            f.write("strict digraph {\n")
            f.write("    rankdir=TB;\n")
            for node, data in graph.nodes(data=True):
                node_name = data.get('name', str(node))
                opcode = data.get('opcode', 'op')
                f.write(f'    "{node_name}" [opcode={opcode}];\n')
            f.write("\n")
            for src, dst in sorted(graph.edges()):
                source_name = graph.nodes[src].get('name', str(src))
                dest_name = graph.nodes[dst].get('name', str(dst))
                f.write(f'    "{source_name}" -> "{dest_name}";\n')
            if levels:
                for level in sorted(levels.keys()):
                    nodes_in_level = " ".join([f'"{graph.nodes[n].get("name", str(n))}"' for n in levels[level]])
                    if len(levels[level]) > 1:
                        f.write(f"    {{ rank = same; {nodes_in_level} }}\n")
            f.write("}\n")

    @staticmethod
    def generate_custom_dot_and_image(graph: nx.DiGraph, dot_filename: str, output_image_filename: str):
        """Generates a logical .dot file and renders a PNG image."""
        if not graph or not graph.nodes:
            logger.warning("Graph is empty, no image to generate.")
            return
        
        try:
            GraphVisualizer._write_dot_file(graph, dot_filename)
        except Exception as e:
            logger.error(f"Error writing custom .dot file: {e}")
            return
        
        try:
            base_name = os.path.splitext(output_image_filename)[0]
            command = f"dot -Tpng {dot_filename} -o {base_name}.png"
            os.system(command)
        except Exception as e:
            logger.debug(f"Error generating image with Graphviz: {e}")

    @staticmethod
    def generate_dot_file_only(graph: nx.DiGraph, dot_filename: str):
        """Generates a .dot file without rendering an image."""
        if not graph or not graph.nodes:
            logger.warning("Graph is empty, no .dot file to generate.")
            return
        
        try:
            GraphVisualizer._write_dot_file(graph, dot_filename)
        except Exception as e:
            logger.error(f"Error writing .dot file: {e}")

    @staticmethod
    def generate_physical_dot(graph: nx.DiGraph, dimensions: tuple, filename: str):
        """
        Generates a DOT file where nodes are pinned to their physical (r, c) coordinates.
        This allows visualization of the actual physical layout on the grid.
        
        Args:
            graph: The NetworkX graph containing 'type' and coordinates.
            dimensions: (rows, cols) of the architecture.
            filename: Output path for the .dot file.
        """
        rows, cols = dimensions
        
        # CORREÇÃO: Aumentar escala para evitar sobreposição. 
        # 72 pontos = 1 polegada. Isso dá espaço suficiente para os nós.
        scale = 75.0 
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("graph PhysicalLayout {\n")
            # Usar 'neato' que respeita o atributo 'pos'
            f.write('    layout=neato;\n')
            # Definir tamanho do nó fixo para ficar bonito no grid
            f.write('    node [shape=rect, style=filled, fixedsize=true, width=0.8, height=0.8, fontsize=10];\n')
            f.write('    splines=false;\n') # Linhas retas para parecer fios
            
            for node, data in graph.nodes(data=True):
                if isinstance(node, tuple) and len(node) == 2:
                    r, c = node
                else:
                    continue
                
                color = "white"
                fontcolor = "black"
                node_type = data.get('type', 'unknown')
                label = ""
                
                # Cores mais distintas e labels curtos
                if node_type == 'input': 
                    color = "#2ECC71" # Verde
                    label = "IN"
                elif node_type == 'output': 
                    color = "#E74C3C" # Vermelho
                    label = "OUT"
                elif node_type == 'operation': 
                    color = "#3498DB" # Azul
                    label = "OP"
                    fontcolor = "white"
                elif node_type == 'routing': 
                    color = "#ECF0F1" # Cinza claro
                    label = "+"
                elif node_type == 'crossover':
                    color = "#9B59B6" # Roxo
                    label = "X"
                    fontcolor = "white"
                elif node_type == 'buffer': 
                    color = "#F1C40F" # Amarelo
                    label = "B"
                elif node_type == 'convergence': 
                    color = "#E67E22" # Laranja
                    label = "Conv"
                
                # Coordenada detalhada para debug visual se quiser
                # label += f"\n({r},{c})"
                
                # Graphviz usa plano Cartesiano (Y cresce para cima). 
                # Matriz usa Y cresce para baixo. Invertemos Y (-r).
                pos_str = f"{c*scale},{-r*scale}!"
                
                f.write(f'    "{node}" [pos="{pos_str}", label="{label}", fillcolor="{color}", fontcolor="{fontcolor}"];\n')
            
            f.write("\n")
            for u, v in graph.edges():
                if isinstance(u, tuple) and isinstance(v, tuple):
                    # Arestas simples cinzas
                    f.write(f'    "{u}" -- "{v}" [color="#7F8C8D", penwidth=2.0];\n')
                    
            f.write("}\n")
    @staticmethod
    def save_placement_grid(graph: nx.DiGraph, dimensions: tuple, filename: str):
        """
        Gera o grid ASCII com identificação clara de Input (I) e Output (OUT).
        """
        rows, cols = dimensions
        grid = [[" . " for _ in range(cols)] for _ in range(rows)]
        
        for node, data in graph.nodes(data=True):
            if isinstance(node, tuple) and len(node) == 2:
                r, c = node
                if 0 <= r < rows and 0 <= c < cols:
                    ntype = data.get('type', 'unknown')
                    symb = " ? "
                    
                    # Símbolos claros de 3 caracteres
                    if ntype == 'input': symb = " I "
                    elif ntype == 'output': symb = "OUT" # Novo símbolo
                    elif ntype == 'operation': symb = "[O]"
                    elif ntype == 'convergence': symb = "<C>"
                    elif ntype == 'routing': symb = " + "
                    elif ntype == 'crossover': symb = " X "
                    elif ntype == 'buffer': symb = " b "
                    
                    grid[r][c] = symb

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Dimensions: {rows}x{cols}\n")
            f.write(f"Legend: I=Input, OUT=Output, [O]=Op, <C>=Convergence, +=Routing, X=Crossover, .=Empty\n\n")
            
            f.write("    " + "".join(f"{c:^3}" for c in range(cols)) + "\n")
            f.write("    " + "---" * cols + "\n")
            
            for r in range(rows):
                line = "".join(grid[r])
                f.write(f"{r:2} |{line}|\n")
            f.write("    " + "---" * cols + "\n")
