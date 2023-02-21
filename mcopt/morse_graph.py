import networkx as nx
import pandas as pd
from .morse_complex import MorseSmaleComplex

class MorseGraph(nx.Graph):
  @staticmethod
  def from_complex(complex: MorseSmaleComplex):
    return MorseGraph(complex.point_data, complex.cell_data)
  
  def __init__(self, point_data: pd.DataFrame, cell_data: pd.DataFrame):
    super().__init__()
    
    for point_id, data in point_data.iterrows():
      self.add_node(
        point_id,
        x = data['Points_0'],
        y = data['Points_1'],
        z = data['Points_2'],
        cell_id = data['CellId'],
      )
    
    for cell_id, data in cell_data[cell_data['Cell Type'] == 3].iterrows():
      self.add_edge(
        data['Point Index 0'],
        data['Point Index 1'],
        cell_id = cell_id,
        n_boundary_points = data['NumberOfCriticalPointsOnBoundary'],
        source_id = data['SourceId'],
        dest_id = data['DestinationId'],
      )
      
  def draw(self, ax = None):
    G = self
    
    pos = {n : (d['y'], d['x']) for n, d in G.nodes(data = True)}
    
    nx.draw(
      G, 
      ax = ax,
      node_size = 10,
      pos = pos,
    )