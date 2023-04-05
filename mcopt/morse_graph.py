from __future__ import annotations
from typing import Set, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike
import networkx as nx
import hypernetx as hnx
import matplotlib

from mcopt.mm_space import (
  MetricProbabilityNetwork, 
  MetricProbabilityHypernetwork,
  Coupling
)

Colors = ArrayLike | Dict[int, float]

class MorseGraph(nx.Graph):
  critical_nodes: Set[int]
  
  def __init__(self, critical_nodes : Set[int]):
    super().__init__()
    self.critical_nodes = critical_nodes

  def sample(self, rate: float|int, mode: str = 'step'):
    graph = MorseGraph(self.critical_nodes)
    
    visited = set()
    
    def dfs(start, node, length=0):
      if node in visited and node not in self.critical_nodes:
        return
      
      visited.add(node)
      
      for n in self.neighbors(node):
        if n in visited:
          continue
        
        if n in self.critical_nodes:
          graph.add_node(n, **self.nodes(data=True)[n])

          assert graph.has_node(start)
          graph.add_edge(start, n)
          
          continue
        
        if mode == 'step':
          new_length = length + 1
        if mode == 'geo':
          new_length = length + np.linalg.norm(self.nodes(data='pos2')[n] - self.nodes(data='pos2')[node])
          
        if new_length > rate:
          graph.add_node(n, **self.nodes(data=True)[n])
          
          assert graph.has_node(start)
          graph.add_edge(start, n)
          
          dfs(n, n)
        else:
          dfs(start, n, new_length)
          
    for crit in self.critical_nodes:
      graph.add_node(crit, **self.nodes(data=True)[crit])
      dfs(crit, crit)
      
    assert nx.is_connected(graph)
    assert all(graph.has_node(n) for n in self.critical_nodes)
    
    return graph

  def node_color_by_position(self) -> Dict[int, float]:
    return {n : np.linalg.norm(pos) for n, pos in self.nodes(data='pos2')}

  def node_color_by_coupling(
    self, 
    src_node_color: Dict[int, float],
    coupling: Coupling
  ) -> Dict[int, float]:
    colors = {}
    
    for n in self.nodes:
      i = coupling.dest_rev_map[n]
      src_i = coupling[:, i].argmax()
      
      if (np.isclose(coupling[src_i, i], 0)):
        colors[n] = np.nan
      else:
        src = coupling.src_map[src_i]
        colors[n] = src_node_color[src]
    
    return colors

  def draw(
    self,
    ax: matplotlib.axes.Axes,
    cmap='viridis',
    node_color: Optional[Colors] = None,
    node_size: int = 40,
    critical_scale: int = 3,
    **kwargs,
  ):
    if not node_color:
      node_color = self.node_color_by_position()
      
    if type(node_color) is dict:
      node_color = np.array([node_color[n] for n in self.nodes()])
    
    node_size = np.array([
      node_size * critical_scale if n in self.critical_nodes else node_size
      for n in self.nodes()
    ])
    
    # Allow for nodes that should be given a "bad color" to have `nan` values.
    vmin = np.nanmin(node_color)
    vmax = np.nanmax(node_color)
    
    nx.draw(
      self,
      ax=ax,
      cmap = cmap,
      pos = self.nodes(data='pos2'),
      node_color = node_color,
      node_size = node_size,
      
      vmin=vmin,
      vmax=vmax,
      alpha=[1],
      **kwargs
    )
  
  def to_hyper(self, mode: str = 'simple'):
    edges = {}

    next_entity = max(self.nodes()) + 1

    if mode == 'simple':
      for edge in self.edges:
        edges[next_entity] = [edge[0], edge[1]]
        next_entity += 1
    else:
      raise ValueError(f'Unrecognized mode {mode}')
    
    hyper_graph = MorseHypergraph(self.critical_nodes, edges)
    
    for n in hyper_graph.nodes:
      hyper_graph.nodes[n].__dict__.update(self.nodes(data=True)[n])
    
    return hyper_graph
  
  def to_mpn(self, hist: str = 'uniform', dist: str = 'step') -> MetricProbabilityNetwork:
    X = np.array(self.nodes())
    X.sort()
    
    if hist == 'uniform':
      measure = np.ones(X.shape[0]) / X.shape[0]
    elif hist == 'degree':
      degs = np.array([self.degree(n) for n in X])
      
      measure = degs / degs.sum()
    else:
      raise ValueError(f'Unrecognized histogram type {hist}')
    
    metric = np.zeros(shape=(X.shape[0], X.shape[0]), dtype=float)
    
    if dist == 'step':
      lens = dict(nx.all_pairs_shortest_path_length(self))
      
      for u_i, u in enumerate(X):
        for v_i, v in enumerate(X):
          metric[u_i, v_i] = lens[u][v]
    elif dist == 'geo':
      lens = dict(nx.all_pairs_dijkstra_path_length(
        self,
        weight=lambda u, v, _: np.linalg.norm(self.nodes(data='pos2')[u] - self.nodes(data='pos2')[v])
      ))
      
      for u_i, u in enumerate(X):
        for v_i, v in enumerate(X):
          metric[u_i, v_i] = lens[u][v]
    elif dist == 'adj':
      for u_i, u in enumerate(X):
        for v_i, v in enumerate(X):
          metric[u_i, v_i] = int(v in self.adj[u])
    else:
      raise ValueError(f'Unrecognized distance type {dist}')
    
    return MetricProbabilityNetwork(X, measure, metric)
  
  @staticmethod
  def attribute_cost_matrix(src: MorseGraph, dest: MorseGraph) -> np.ndarray:
    X = list(src.nodes())
    X.sort()
    
    Y = list(dest.nodes())
    Y.sort()
    
    X_attrs = list(src.nodes(data='pos2')[n] for n in X)
    Y_attrs = list(dest.nodes(data='pos2')[n] for n in Y)
    
    M = np.zeros((len(X), len(Y)), dtype=float)
    
    for u_i, u in enumerate(X):
      for v_i, v in enumerate(Y):
        M[u_i, v_i] = np.linalg.norm(X_attrs[u_i] - Y_attrs[v_i])
    
    return M
  
class MorseHypergraph(hnx.Hypergraph):
  critical_nodes: Set[int]
  
  def __init__(self, critical_nodes: Set[int], *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.critical_nodes = critical_nodes
    
  def node_color_by_position(self) -> Dict[int, float]:
    return {n : np.linalg.norm(data.pos2) for n, data in self.nodes.elements.items()}
  
  def node_color_by_coupling(
    self,
    src_node_color : Dict[int, float],
    coupling: Coupling
  ):
    colors = {}
    
    for n in self.nodes:
      i = coupling.dest_rev_map[n]
      src_i = coupling[:, i].argmax()
      
      if (np.isclose(coupling[src_i, i], 0)):
        colors[n] = np.nan
      else:
        src = coupling.src_map[src_i]
        colors[n] = src_node_color[src]
    
    return colors
  
  def edge_color_by_node(self, node_color: Dict[int, float]) -> Dict[int, float]:
    return {
      e : sum([node_color[n] for n in nodes]) / len(nodes) 
      for e, nodes in self.incidence_dict.items()
    }
    
  def edge_color_by_coupling(
    self,
    src_edge_color: Dict[int, float],
    coupling: Coupling
  ) -> Dict[int, float]:
    colors = {}
    
    for e in self.incidence_dict.keys():
      i = coupling.dest_rev_map[e]
      src_i = coupling[:, i].argmax()
      
      if (np.isclose(coupling[src_i, i], 0)):
        colors[e] = np.nan
      else:
        src = coupling.src_map[src_i]
        colors[e] = src_edge_color[src]
    
    return   
  
  def draw(
    self,
    ax: matplotlib.axes.Axes,
    cmap='viridis',
    node_color: Colors = None,
    edge_color: Colors = None,
    edge_alpha: float = 0.5,
    node_size: int = 40,
    critical_scale: int = 3,
    **kwargs,
  ):
    if isinstance(cmap, str):
      cmap = matplotlib.colormaps[cmap]
      
    critical_scale = critical_scale / 3 * 1.3  
    
    node_radius = node_size / 80
    node_radius = {
      n : node_radius * critical_scale if n in self.critical_nodes else node_radius
      for n in self.nodes
    }
    
    if not node_color:
      node_color = self.node_color_by_position()

    if type(node_color) is dict:
      node_color_map = node_color
      node_color = np.array([node_color[n] for n in self.nodes])
    else:
      node_color_map = {i : v for i, v in enumerate(node_color)}
      node_color = np.asarray(node_color)
      
    node_color /= node_color.max()
    node_facecolor = [cmap(v) for v in node_color]
    
    if not edge_color:
      edge_color = self.edge_color_by_node(node_color_map)
      
    if type(edge_color) is dict:
      edge_color = np.array([edge_color[e] for e in self.edges])
    else:
      edge_color = np.asarray(edge_color)

    edge_color /= edge_color.max()
    edge_facecolor = [cmap(v, alpha=edge_alpha) for v in edge_color]
    
    hnx.rubber_band.draw(
      self,
      ax=ax,
      pos = {n : data.pos2 for n, data in self.nodes.elements.items()},
      node_radius = node_radius,
      with_color = False,
      with_edge_labels = False,
      with_node_labels = False,
      nodes_kwargs = {
        'facecolors': node_facecolor,
        'edgecolors': 'black'
      },
      edges_kwargs = {
        'edgecolors': (0, 0, 0, 0),
        'facecolors': edge_facecolor,
        'dr': 0.1
      },
      **kwargs
    )
    
  def to_mph(
    self,
    node_hist: str = 'uniform',
    edge_hist: str = 'uniform',
    dist: str = 'jaccard_index',
  ) -> MetricProbabilityHypernetwork:
    node_space = np.asarray(self.nodes)
    node_space.sort()
    
    edge_space = np.asarray(self.edges)
    edge_space.sort()
    
    if node_hist == 'uniform':
      node_measure = np.ones(shape = len(node_space)) / len(node_space)
    else:
      raise ValueError(f'Unrecognized node histogram type {node_hist}')
    
    if edge_hist == 'uniform':
      edge_measure = np.ones(shape = len(edge_space)) / len(edge_space)
    else:
      raise ValueError(f'Unrecognized edge histogram type {edge_hist}')
    
    if dist == 'incidence':
      metric = self.incidence_matrix().astype(float).todense()
    elif dist == 'intersection_size' or dist == 'union_size' or dist == 'jaccard_index':
      metric = np.zeros(shape = (len(node_space), len(edge_space)), dtype=float)
      edge_map = {e: i for i, e in enumerate(edge_space)}
      
      lg = self._line_graph()
      dual = self.dual()
            
      ldist = nx.floyd_warshall_numpy(lg ,weight=dist)
      
      for i, n in enumerate(node_space):
        idxs = [edge_map[e] for e in dual.incidence_dict[n]]
        shortest_target_dists = ldist[idxs,:].min(axis=0)
        metric[i,:] = shortest_target_dists
    else:
      raise ValueError(f'Unrecognized distance type {dist}')
    
    return MetricProbabilityHypernetwork(
      node_space,
      edge_space,
      node_measure,
      edge_measure,
      metric
    )
  
  def _line_graph(self) -> nx.Graph:
    hgraph_dict = self.incidence_dict
    line_graph = nx.Graph()

    node_list = list(hgraph_dict.keys())
    node_list.sort() # sort the node by id
    # Add nodes
    [line_graph.add_node(edge) for edge in node_list]

    # For all pairs of edges (e1, e2), add edges such that
    # intersection(e1, e2) is not empty
    s = 1
    for node_idx_1, node1 in enumerate(node_list):
      for node_idx_2, node2 in enumerate(node_list[node_idx_1 + 1:]):
        vertices1 = hgraph_dict[node1]
        vertices2 = hgraph_dict[node2]
        if len(vertices1) > 0 or len(vertices2) > 0:
          # Compute the intersection size
          intersection_size = len(set(vertices1) & set(vertices2))
          union_size = len(set(vertices1) | set(vertices2))
          jaccard_index = intersection_size / union_size
          if intersection_size >= s:
            line_graph.add_edge(node1, node2, intersection_size=1/intersection_size, jaccard_index=1/jaccard_index)
    # line_graph = nx.readwrite.json_graph.node_link_data(line_graph)
    return line_graph