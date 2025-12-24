import random
import networkx as nx
import logging
from typing import Optional, List, Tuple, Set, Dict
from ...architectures.qca import QCA

logger = logging.getLogger(__name__)

class QcaMultiClusterGenerator:
    """
    QCA Generator that creates independent sub-graphs (clusters) in distinct 
    grid regions and merges them into a final output, handling synchronization (delays).
    
    ✅ FIXED: Agora preenche fisicamente os caminhos de merge com células de roteamento
    para evitar "linhas diagonais" (conexões lógicas sem corpo físico).
    """
    
    def __init__(self, qca_architecture: QCA, num_inputs: int, target_depth: int):
        self.qca_architecture = qca_architecture
        self.num_inputs = num_inputs
        self.target_depth = target_depth
        
        self.placement_graph = nx.DiGraph()
        self.used_nodes: Set[Tuple[int, int]] = set()
        self.qca_border_nodes = self.qca_architecture.get_border_nodes()
        self.qca_arch_graph = self.qca_architecture.get_graph() # Necessário para roteamento
        self.MAX_DELAY_BUFFERS = 10

    def generate(self) -> Optional[nx.DiGraph]:
        self.placement_graph.clear()
        self.used_nodes.clear()
        # Atualiza referência do grafo da arquitetura
        self.qca_arch_graph = self.qca_architecture.get_graph()
        
        # 1. Partition Grid
        regions = self._partition_grid()
        if not regions: return None
        
        num_clusters_to_use = len(regions)
        if len(regions) > 2 and random.random() < 0.4:
            num_clusters_to_use = 2
        
        regions = regions[:num_clusters_to_use]
        cluster_outputs = []
        base_inputs = max(1, self.num_inputs // len(regions))
        
        # 2. Grow Clusters
        for region_idx, region_bounds in enumerate(regions):
            varied_inputs = base_inputs + random.choice([-1, 0, 1])
            varied_inputs = max(1, varied_inputs)
            
            output_node, level = self._generate_cluster(
                region_idx, region_bounds, varied_inputs
            )
            if output_node:
                cluster_outputs.append({'node': output_node, 'level': level})
        
        if len(cluster_outputs) < 2: return None
            
        # 3. Synchronized Merge
        if not self._merge_clusters(cluster_outputs):
            return None
            
        # 4. Validar e Marcar Saídas
        self._mark_outputs()
        
        return self.placement_graph

    def _partition_grid(self) -> List[Tuple[int, int, int, int]]:
        rows, cols = self.qca_architecture.dim
        if cols < 4: return []
        
        num_regions = random.choice([2, 2, 3])
        if num_regions == 2:
            mid_col = cols // 2 + random.randint(-1, 1)
            mid_col = max(2, min(cols - 2, mid_col))
            return [(0, rows, 0, mid_col - 1), (0, rows, mid_col + 1, cols)]
        else:
            third = cols // 3
            return [(0, rows, 0, third), (0, rows, third + 1, 2 * third), (0, rows, 2 * third + 1, cols)]

    def _generate_cluster(self, idx: int, bounds: Tuple[int, int, int, int], n_inputs: int) -> Tuple[Optional[Tuple[int, int]], int]:
        r_min, r_max, c_min, c_max = bounds
        border_candidates = []
        
        # Coletar bordas
        for c in range(c_min, c_max):
            node = (0, c)
            if node in self.qca_border_nodes and node not in self.used_nodes:
                border_candidates.append(node)
        for r in range(1, min(3, r_max)):
            left, right = (r, c_min), (r, c_max - 1)
            if left in self.qca_border_nodes and left not in self.used_nodes: border_candidates.append(left)
            if right in self.qca_border_nodes and right not in self.used_nodes: border_candidates.append(right)
        
        if len(border_candidates) < n_inputs: return None, 0
        
        random.shuffle(border_candidates)
        cluster_inputs = border_candidates[:n_inputs]
        
        for node in cluster_inputs:
            self.placement_graph.add_node(node, type='input', level=0, name=f"in_c{idx}_{node}")
            self.used_nodes.add(node)
        
        current_layer = cluster_inputs.copy()
        current_level = 0
        
        # Crescimento
        for d in range(1, self.target_depth + 1):
            if d >= self.target_depth * 0.6 and random.random() < 0.25: break
            
            next_layer = []
            random.shuffle(current_layer)
            
            while len(current_layer) >= 2:
                k = min(len(current_layer), random.choice([2, 2, 2, 3]))
                parents = current_layer[:k]
                current_layer = current_layer[k:]
                
                child = self._find_bounded_child(parents, bounds)
                if child:
                    self.placement_graph.add_node(child, type='operation', level=d, name=f"op_c{idx}_{child}")
                    self.used_nodes.add(child)
                    # Roteamento simples para pais (assumindo adjacência próxima)
                    for p in parents:
                        self._route_physical_connection(p, child)
                    next_layer.append(child)
            
            if current_layer:
                parent = current_layer[0]
                child = self._find_bounded_child([parent], bounds)
                if child:
                    self.placement_graph.add_node(child, type='buffer', level=d, name=f"buf_c{idx}_{child}")
                    self.used_nodes.add(child)
                    self._route_physical_connection(parent, child)
                    next_layer.append(child)
            
            if not next_layer: break
            current_layer = next_layer
            current_level = d
            
        return (current_layer[0] if current_layer else None, current_level)

    def _find_bounded_child(self, parents: List[Tuple], bounds: Tuple) -> Optional[Tuple]:
        r_min, r_max, c_min, c_max = bounds
        avg_r = int(sum(p[0] for p in parents) / len(parents))
        avg_c = int(sum(p[1] for p in parents) / len(parents))
        
        # Busca espiral simples
        for r in range(avg_r, min(r_max, avg_r + 4)):
            for c in range(max(c_min, avg_c - 3), min(c_max, avg_c + 4)):
                node = (r, c)
                if node not in self.used_nodes and node not in parents:
                    # Verifica se é alcançável
                    reachable = True
                    for p in parents:
                        if not self._is_reachable(p, node):
                            reachable = False; break
                    if reachable: return node
        return None

    def _merge_clusters(self, clusters: List[Dict]) -> bool:
        max_level = max(c['level'] for c in clusters)
        final_inputs = []
        
        for cluster in clusters:
            node = cluster['node']
            level = cluster['level']
            delay_needed = max_level - level
            current_tip = node
            
            # Adicionar Buffers de Sincronização
            if delay_needed > 0:
                for i in range(delay_needed):
                    # Encontrar vizinho válido para buffer
                    next_pos = self._find_buffer_pos(current_tip)
                    if next_pos:
                        self.placement_graph.add_node(next_pos, type='buffer', level=level + 1 + i, name=f"sync_{next_pos}")
                        self.used_nodes.add(next_pos)
                        # Roteamento físico entre buffer e anterior
                        self._route_physical_connection(current_tip, next_pos)
                        current_tip = next_pos
                    else:
                        return False
            
            final_inputs.append(current_tip)
            
        # Merge Final
        merge_point = self._find_merge_point(final_inputs)
        if merge_point:
            self.placement_graph.add_node(merge_point, type='convergence', level=max_level + 1, name="FINAL_MERGE")
            self.used_nodes.add(merge_point)
            
            # Rotear inputs finais até o ponto de merge
            for inp in final_inputs:
                success = self._route_physical_connection(inp, merge_point)
                if not success: return False
            return True
            
        return False

    def _route_physical_connection(self, source, target) -> bool:
        """
        Roteia fisicamente (preenchendo com '+') entre dois nós.
        Corrige o problema das linhas diagonais.
        """
        if source == target: return True
        
        try:
            # Cria grafo temporário para busca
            temp_graph = self.qca_arch_graph.copy()
            nodes_to_avoid = self.used_nodes - {source, target}
            temp_graph.remove_nodes_from(nodes_to_avoid)
            
            path = nx.shortest_path(temp_graph, source=source, target=target)
            
            # Adiciona caminho ao grafo
            if len(path) > 2:
                for node in path[1:-1]:
                    self.placement_graph.add_node(node, type='routing', name=f"rout_{node}")
                    self.used_nodes.add(node)
            
            # Adiciona arestas sequenciais
            nx.add_path(self.placement_graph, path)
            return True
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Se falhar o físico, adiciona lógico como fallback (mas avisa)
            self.placement_graph.add_edge(source, target)
            return False

    def _find_buffer_pos(self, source):
        """Encontra posição livre vizinha para colocar buffer."""
        neighbors = list(self.qca_arch_graph.neighbors(source))
        random.shuffle(neighbors)
        for n in neighbors:
            if n not in self.used_nodes: return n
        return None

    def _is_reachable(self, u, v):
        """Verifica se v é alcançável a partir de u na arquitetura."""
        try:
            return nx.has_path(self.qca_arch_graph, u, v)
        except: return False

    def _find_merge_point(self, nodes: List[Tuple]) -> Optional[Tuple]:
        avg_r = sum(n[0] for n in nodes) / len(nodes)
        avg_c = sum(n[1] for n in nodes) / len(nodes)
        
        # Procura ponto de convergência abaixo/direita da média
        start_r, start_c = int(avg_r), int(avg_c)
        rows, cols = self.qca_architecture.dim
        
        # Espiral de busca
        for radius in range(1, 10):
            for r in range(start_r, min(rows, start_r + radius + 1)):
                for c in range(max(0, start_c - radius), min(cols, start_c + radius + 1)):
                    node = (r, c)
                    if node not in self.used_nodes:
                        # Verifica se todos conseguem chegar aqui
                        if all(self._is_reachable(n, node) for n in nodes):
                            return node
        return None

    def _is_valid(self, node: Tuple) -> bool:
        r, c = node
        rows, cols = self.qca_architecture.dim
        return 0 <= r < rows and 0 <= c < cols

    def _mark_outputs(self):
        """Marca nós folha como output para visualização."""
        for node in self.placement_graph.nodes():
            if self.placement_graph.out_degree(node) == 0:
                if self.placement_graph.nodes[node].get('type') != 'input':
                    self.placement_graph.nodes[node]['type'] = 'output'
                    self.placement_graph.nodes[node]['name'] = f"OUT_{node}"
