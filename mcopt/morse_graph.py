from __future__ import annotations
from typing import  Tuple, Dict, Set, Optional
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib as mpl
from .morse_complex import MorseSmaleComplex
from .opt import MeasureNetwork


def color_by_position(graph : MorseGraph) -> dict:
  """
  Constructs a `node_color` array where the color of a node determined it's position.
  
  Args:
    graph (MorseGraph): The graph to generate a node coloring for.
  """
  return {n : np.linalg.norm(pos) for n, pos in graph.nodes(data='pos2')}

def color_by_attr(graph : MorseGraph, attr : str) -> dict:
  """
  Constructs a `node_color` array where the color of a node is determined by the 
  value of it's `attr` attribute.
  
  Args:
    graph (MorseGraph): The graph to generate a node coloring for.
    attr (str): The attribute to base the coloring off of.
  """
  return {n : int(v) for n, v in graph.nodes(data=attr)}

def color_by_component(graph : MorseGraph) -> dict:
  """
  Constructs a `node_color` array  where the color of a node is determined by
  the connected component it resides in, i.e. all nodes in the same component 
  will have the same color. Works best when combined with a qualitative color 
  map [see](https://matplotlib.org/stable/tutorials/colors/colormaps.html).
  
  This coloring is particularly useful for debugging.
  
  Args:
    graph (MorseGraph): The graph to generate a node coloring for.
  """
  vals = {}
  
  for i, comp in enumerate(nx.connected_components(graph)):
    for n in comp:
      vals[n] = i
  
  return vals
    
# Takes in the raw vtk data and produces a mapping of the points which is easier
# to work with.
def _make_point_map(separatrices_points : pd.DataFrame, critical_points : pd.DataFrame):
  critical_cells = set(critical_points['CellId'])
  
  min_x, max_x = float('inf'), -float('inf')
  min_y, max_y = float('inf'), -float('inf')
  
  for _, data in separatrices_points[['Points_0', 'Points_1']].iterrows():
    x, y = data['Points_0'], data['Points_1']
    
    min_x = min(min_x, x)
    max_x = max(max_x, x)
    min_y = min(min_y, y)
    max_y = max(max_y, y)
  
  separatrices_points = separatrices_points.sort_values(by=['Points_0', 'Points_1'])
  
  nodes = {}
  cell_map = {}
  point_map = {}
  
  next_node = 0
  
  for id, data in separatrices_points.iterrows():
    assert id not in point_map
    
    cell_id = data['CellId']
    # Not sure exactly why, but this is only way to tell if a point is a critical point
    is_crit = data['ttkMaskScalarField'] == 0
    
    if is_crit and cell_id in cell_map:
      # We have seen this critical point before
      node = cell_map[cell_id]
      nodes[node]['point_ids'].append(id)
      
      point_map[id] = node
      continue
    elif is_crit:
      # This is a new critical point
      if cell_id not in critical_cells:
        raise ValueError(f'Expected point {id}\'s cell {cell_id} to be in critical cells:\n{critical_cells}')
        
      cell_map[cell_id] = next_node
    
    x, y = data['Points_0'], data['Points_1']
      
    point_map[id] = next_node
    nodes[next_node] = {
      'pos2': np.array([data['Points_0'], data['Points_1']]),
      'point_ids': [id],
      'is_critical': is_crit,
      'on_boundary': x == min_x or x == max_x or y == min_y or y == max_y,
    }
    next_node += 1
    
  critical_nodes = set(cell_map.values())
    
  return nodes, point_map, critical_nodes
  
class MorseGraph(nx.Graph):
  @staticmethod
  def from_complex(complex: MorseSmaleComplex):
    """
    Creates a Morse Graph from a Morse Smale Complex.
    
    Args:
      complex (MorseSmaleComplex): A Morse Smale Complex.
      
    Returns:
      MorseGraph: A Morse Graph.
    """
    return MorseGraph.from_csvs(
      complex.separatrices_cell_data, 
      complex.separatrices_point_data, 
      complex.critical_points_point_data
    )
  
  @staticmethod
  def from_csvs(separatrices_cells : pd.DataFrame, separatrices_points : pd.DataFrame, critical_points : pd.DataFrame):
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
  
  critical_nodes : Set[int]
    
  def __init__(self, critical_nodes):
    super().__init__()
    self.critical_nodes = critical_nodes
          
  def draw(
    self, 
    ax = None, 
    crit_scale = 3,
    **kwargs
  ):        
    kwargs.setdefault('node_size', 10)
    kwargs.setdefault('cmap', 'viridis')
    
    if 'node_color' not in kwargs:
      kwargs['node_color'] = color_by_position(self)
      
    if type(kwargs['node_color']) is dict:
      node_color = kwargs['node_color']
      kwargs['node_color'] = np.array([node_color[n] for n in self.nodes()])
      
    n_size = kwargs.pop('node_size', 10)
    
    nx.draw(
      self, 
      ax = ax,
      pos = self.nodes(data = 'pos2'),
      node_size = np.array([n_size * crit_scale if n in self.critical_nodes else n_size for n in self.nodes()]),
      **kwargs,
    )
    
  def sample(self, min_length, mode='step') -> MorseGraph:
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
        if mode == 'geo_dist':
          new_length = length + np.linalg.norm(self.nodes(data=True)[n]['pos2'] - self.nodes(data=True)[node]['pos2'])
        
        if new_length > min_length:
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
  
  def to_measure_network(self, weight='path_length', hist='unif') -> MeasureNetwork:
    X = np.array(self.nodes())
    X.sort()
    
    if weight == 'path_length':
      lens = dict(nx.all_pairs_shortest_path_length(self))
      
      W = np.zeros((X.shape[0], X.shape[0]), dtype=float)
      
      for u_i, u in enumerate(X):
        for v_i, v in enumerate(X):
           W[u_i,v_i] = lens[u][v]
    elif weight == 'geo_dist':
      lens = dict(nx.all_pairs_dijkstra_path_length(
        self, 
        weight=lambda u, v, _: np.linalg.norm(self.nodes(data='pos2')[u] - self.nodes(data='pos2')[v])
      ))
      
      W = np.zeros((X.shape[0], X.shape[0]), dtype=float)
      
      for u_i, u in enumerate(X):
        for v_i, v in enumerate(X):
           W[u_i,v_i] = lens[u][v]
    elif weight == 'adj':
      W = np.zeros((X.shape[0], X.shape[0]), dtype=float)
      
      for u_i, u in enumerate(X):
        for v_i, v in enumerate(X):
           W[u_i,v_i] = int(v in self.adj[u])
    else:
      raise ValueError(f'weight mode not supported {weight}')
    
    if hist == 'unif':
      mu = np.ones(X.shape[0])/len(X)
    elif hist == 'deg':
      mu = np.array([self.degree(n) for n in X]) / sum([self.degree(n) for n in X])
    else:
      raise ValueError(f'hist mode not supported {hist}')
    
    return X, W, mu

def attribute_cost_mat(
  X_graph: MorseGraph, 
  Y_graph : MorseGraph, 
  attr: str = 'pos2'
) -> np.ndarray:
  X = list(X_graph.nodes())
  X.sort()
  
  Y = list(Y_graph.nodes())
  Y.sort()
  
  if attr == 'pos2':
    X_attrs = list(X_graph.nodes(data='pos2')[n] for n in X)
    Y_attrs = list(Y_graph.nodes(data='pos2')[n] for n in Y)
  else:
    raise ValueError(f'attribute mode not supported {attr}')
  
  M = np.zeros((len(X), len(Y)), dtype=float)
  
  for u_i, u in enumerate(X):
    for v_i, v in enumerate(Y):
      M[u_i, v_i] = np.linalg.norm(X_attrs[u_i] - Y_attrs[v_i])
  
  return M