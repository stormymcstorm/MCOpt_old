from __future__ import annotations
import networkx as nx
import numpy as np
import pandas as pd
from .morse_complex import MorseSmaleComplex
from typing import Set,List


def _make_point_map(separatrices_points : pd.DataFrame, critical_points : pd.DataFrame):
  critical_cells = set(critical_points['CellId'])
  
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
      
    point_map[id] = next_node
    nodes[next_node] = {
      'pos2': np.array([data['Points_0'], data['Points_1']]),
      'point_ids': [id],
    }
    next_node += 1
    
  return nodes, point_map
  
def position_colors(graph : MorseGraph) -> np.ndarray:
  dists = np.array([np.linalg.norm(pos) for _, pos in graph.nodes(data='pos2')])
  
  return np.interp(dists, (dists.min(), dists.max()), (0, 1))

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
    graph = MorseGraph()
    
    nodes, point_map = _make_point_map(separatrices_points, critical_points)
    
    graph.add_nodes_from(nodes.items())
      
    for _, cell_data in separatrices_cells.iterrows():
      graph.add_edge(
        point_map[cell_data['Point Index 0']],
        point_map[cell_data['Point Index 1']],
      )
      
    assert nx.is_connected(graph), "MorseGraph should be connected" 
          
    return graph
  
  critical_points : Set[int]
  
  def __init__(self):
    super().__init__()
          
  def draw(self, ax = None, **kwargs):        
    kwargs.setdefault('node_size', 10)
    kwargs.setdefault('cmap', 'viridis')
    
    if 'node_color' not in kwargs:
      kwargs['node_color'] = position_colors(self)
      
    nx.draw(
      self, 
      ax = ax,
      pos = self.nodes(data = 'pos2'),
      **kwargs,
    )
    
  def simplify(self, min_length = 0) -> MorseGraph:
    new_graph = MorseGraph()
    
    # TODO: What should I call this sort of note? 
    srcs = set(filter(lambda n: self.degree(n) != 2, self.nodes))
    
    print(srcs)
    
    
    return new_graph
  