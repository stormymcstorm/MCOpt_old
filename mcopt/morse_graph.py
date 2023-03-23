"""
Representation and logic for working with Morse Graphs
"""
from __future__ import annotations
from typing import Set, Dict, Optional, Any

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import networkx as nx
import hypernetx as hnx
import matplotlib

from .morse_complex import MorseComplex
from .ot.mm import MetricProbabilityNetwork, MetricMeasureHypernetwork, Coupling

def _make_point_map(
  separatrices_points: pd.DataFrame,
  critical_points: pd.DataFrame
):
  critical_cells = set(critical_points['CellId'])
  
  separatrices_points = separatrices_points.sort_values(by=['Points_0', 'Points_1'])
  
  nodes = {}
  cell_map = {}
  point_map = {}
  
  next_node = 0
  
  for id, data in separatrices_points.iterrows():
    assert id not in point_map
    
    cell_id = data['CellId']
    is_crit = data['ttkMaskScalarField'] == 0
    
    if is_crit and cell_id in cell_map:
      node = cell_map[cell_id]
      nodes[node]['point_ids'].append(id)
      
      point_map[id] = node
      continue
    
    elif is_crit:
      assert(cell_id in critical_cells)
      
      cell_map[cell_id] = next_node
      
    x, y = data['Points_0'], data['Points_1']
    point_map[id] = next_node
    nodes[next_node] = {
      'pos2': np.array([x, y]),
      'point_ids': [id],
      'is_critical': is_crit
    }
    
    next_node += 1
    
  critical_nodes = set(cell_map.values())
  
  return nodes, point_map, critical_nodes

class MorseGraph(nx.Graph):
  critical_nodes: Set[int]
  
  @staticmethod
  def from_complex(complex : MorseComplex) -> MorseGraph:
    separatrices_points = complex.separatrices_point_data
    separatrices_cells = complex.separatrices_cell_data
    critical_points = complex.critical_points_point_data
    
    nodes, point_map, critical_nodes = _make_point_map(separatrices_points, critical_points)
    
    graph = MorseGraph(critical_nodes)
    graph.add_nodes_from(nodes.items())
    
    for _, cell_data in separatrices_cells.iterrows():
      graph.add_edge(
        point_map[cell_data['Point Index 0']],
        point_map[cell_data['Point Index 1']],
      )
    
    assert nx.is_connected(graph), "MorseGraph should be connected"
    
    return graph
  
  @staticmethod
  def attribute_cost_matrix(
    src: MorseGraph,
    dest: MorseGraph
  ) -> np.ndarray:
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
  
  def __init__(self, critical_nodes : Set[int]):
    super().__init__()
    self.critical_nodes = critical_nodes
    
  def color_by_position(self) -> Dict[int, float]:
    return {n : np.linalg.norm(pos) for n, pos in self.nodes(data='pos2')}
  
  def color_by_coupling(
    self,
    src_colors : Dict[int, float],
    coupling: Coupling
  ) -> Dict[int, float]:
    colors = {}
    
    for n in self.nodes():
      i = coupling.dest_rev_map[n]
      src_i = coupling[:, i].argmax()
      
      if (np.isclose(coupling[src_i, i], 0)):
        colors[n] = np.nan
      else:
        src = coupling.src_map[src_i]
        colors[n] = src_colors[src]
    
    return colors
  
  def draw(
    self,
    ax,
    critical_scale = 3,
    node_color: Optional[Dict[int, float] | ArrayLike] = None,
    node_size = 10,
    **kwargs
  ):
    kwargs.setdefault('cmap', 'viridis')
    
    if not node_color:
      node_color = self.color_by_position()
      
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
      ax = ax,
      pos = self.nodes(data='pos2'),
      node_size = node_size,
      node_color = node_color,
      
      vmin=vmin,
      vmax=vmax,
      alpha=[1],
      **kwargs
    )
    
  def sample(self, rate, mode='step') -> MorseGraph:
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

  def to_mp(self, dist='path_length', measure='uniform') -> MetricProbabilityNetwork:
    X = np.array(self.nodes())
    X.sort()
    
    d = np.zeros(shape=(X.shape[0], X.shape[0]), dtype=float)
    
    if dist == 'path_length':
      lens = dict(nx.all_pairs_shortest_path_length(self))
      
      for u_i, u in enumerate(X):
        for v_i, v in enumerate(X):
          d[u_i, v_i] = lens[u][v]
    elif dist == 'geo_dist':
      lens = dict(nx.all_pairs_dijkstra_path_length(
        self,
        weight=lambda u, v, _: np.linalg.norm(self.nodes(data='pos2')[u] - self.nodes(data='pos2')[v])
      ))
      
      for u_i, u in enumerate(X):
        for v_i, v in enumerate(X):
          d[u_i, v_i] = lens[u][v]
    elif dist == 'adj':
      for u_i, u in enumerate(X):
        for v_i, v in enumerate(X):
          d[u_i, v_i] =  int(v in self.adj[u])
    else:
      raise ValueError(f'Unrecognized distance metric {dist}')
    
    if measure == 'uniform':
      mu = np.ones(X.shape[0])/X.shape[0]
    elif measure == 'degree':
      degs = np.array([self.degree(n) for n in X]) 
      
      mu = degs / degs.sum()
    else:
      raise ValueError(f'Unrecognized measure {measure}')
    
    return MetricProbabilitySpace(X, d, mu)
  
class MorseHypergraph(hnx.Hypergraph):
  critical_nodes: Set[int]
  node_data: Dict[int, Dict[str, Any]]
  
  @staticmethod
  def from_graph(graph: MorseGraph, construction = 'simple'):
    node_data = dict(graph.nodes(data=True))
    critical_nodes = graph.critical_nodes
    
    edges = []
    
    if construction == 'simple':
      for edge in graph.edges():
        edges.append((edge[0], edge[1]))
    elif construction == 'neighbors':
      for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        neighbors.append(node)
        edges.append(neighbors)
    else:
      raise ValueError(f'Unrecognized construction {construction}')
    
    return MorseHypergraph(edges, critical_nodes, node_data)
  
  def __init__(self, edges, critical_nodes, node_data):
    super().__init__(edges, static=True)
    self.critical_nodes = critical_nodes
    self.node_data = node_data
    
  def node_color_by_position(self) -> Dict[int, float]:
    return {n : np.linalg.norm(data['pos2']) for n, data in self.node_data.items()}
  
  def edge_color_by_nodes(self, node_color: Dict[int, float]) -> Dict[str, float]:
    return {
      e : sum([node_color[n] for n in nodes]) / len(nodes) 
      for e, nodes in self.incidence_dict.items()
    }
    
  def draw(self,
    ax,
    critical_scale = 1.3,
    node_color: Optional[Dict[int, float] | ArrayLike] = None,
    edge_color: Optional[Dict[int, float] | ArrayLike] = None,
    node_size = 10,
    cmap = None,
  ):
    if cmap is None:
      cmap = 'viridis'
      
    if isinstance(cmap, str):
      cmap = matplotlib.colormaps[cmap]
    
    node_radius = node_size / 20
    node_radius = {
      n: node_radius * critical_scale if n in self.critical_nodes else node_radius
      for n in self.nodes
    }
    
    if node_color is None:
      node_color = self.node_color_by_position()
      
    if type(node_color) is dict:
      node_color_lookup = node_color
      node_color = np.array([node_color[n] for n in self.nodes])
    else:
      node_color_lookup = {i : v for i, v in enumerate(node_color)}
      node_color = np.asarray(node_color)
      
    node_color /= node_color.max()
    node_facecolors = [cmap(v) for v in node_color]
    
    if edge_color is None:
      edge_color = self.edge_color_by_nodes(node_color_lookup)
      
    if type(edge_color) is dict:
      edge_color = np.array([edge_color[e] for e in self.edges])
    else:
      edge_color = np.asarray(edge_color)  
    
    edge_color /= edge_color.max()
    edge_facecolors = [cmap(v, alpha=0.5) for v in edge_color]
    
    hnx.rubber_band.draw(
      self,
      ax=ax,
      pos = {n : data['pos2'] for n, data in self.node_data.items()},
      node_radius=node_radius,
      with_color=False,
      with_edge_labels=False,
      with_node_labels=False,
      nodes_kwargs={
        'facecolors': node_facecolors,
        'edgecolors': 'black', 
      },
      edges_kwargs={
        'edgecolors': (0, 0, 0, 0),
        'facecolors': edge_facecolors,
        'dr': 0.1
      }
    )
  
  def to_mph(self, dist='', node_measure='uniform', edge_measure='uniform') -> MetricMeasureHypernetwork:
    pass