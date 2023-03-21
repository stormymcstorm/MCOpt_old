"""
Representation and logic for working with Morse Graphs
"""
from __future__ import annotations
from typing import Set, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import networkx as nx

from .morse_complex import MorseComplex

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
  
  def __init__(self, critical_nodes : Set[int]):
    super().__init__()
    self.critical_nodes = critical_nodes
    
  def color_by_position(self) -> Dict[int, float]:
    return {n : np.linalg.norm(pos) for n, pos in self.nodes(data='pos2')}
  
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