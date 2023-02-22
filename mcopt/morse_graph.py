from __future__ import annotations
import networkx as nx
import numpy as np
import pandas as pd
from .morse_complex import MorseSmaleComplex
from functools import cache

class MorseGraph(nx.Graph):
  @staticmethod
  def from_complex(complex: MorseSmaleComplex):
    return MorseGraph.from_csvs(
      complex.separatrices_cell_data, 
      complex.separatrices_point_data, 
      complex.critical_points_point_data
    )
  
  @staticmethod
  def from_csvs(separatrices_cells : pd.DataFrame, separatrices_points : pd.DataFrame, critical_points : pd.DataFrame):
    graph = MorseGraph()
    
    critical_cells = set(critical_points['CellId'])
    cell_map = {}
    
    def map_point(point_id):
      cell_id = separatrices_points.loc[point_id, 'CellId']
      
      if cell_id in cell_map:
        return cell_map[cell_id]
      elif cell_id in critical_cells:
        cell_map[cell_id] = point_id
      
      return point_id
    
    for point_id, point_data in separatrices_points.iterrows():
      graph.add_node(
        map_point(point_id),
        pos2 = np.array([point_data['Points_0'], point_data['Points_1']]),
        cell_id = point_data['CellId']
      )
      
    for cell_id, cell_data in separatrices_cells.iterrows():
      graph.add_edge(
        map_point(cell_data['Point Index 0']),
        map_point(cell_data['Point Index 1']),
      )
      
    return graph
  
  def __init__(self):
    super().__init__()
          
  def draw(self, ax = None, **kwargs):        
    kwargs.setdefault('node_size', 10)
      
    nx.draw(
      self, 
      ax = ax,
      pos = self.nodes(data = 'pos2'),
      **kwargs,
    )
    
  def simplify(self, min_length) -> MorseGraph:
    new_graph = MorseGraph()
    
    # TODO: What should I call this sort of note? 
    # srcs = set(filter(lambda n: self.degree(n) > 2, self.nodes))
    
    # print(srcs)
    
    
    return new_graph
  