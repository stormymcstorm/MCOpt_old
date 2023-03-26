"""
Utilities for working with Morse Complexes
"""

from typing import List, Optional
from functools import cached_property

import vtk
import pandas as pd
import numpy as np
import networkx as nx

import mcopt.pipeline.ttk as ttk_util
import mcopt.pipeline.vtk as vtk_util
from mcopt.morse_graph import MorseGraph

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

class MorseSmaleComplex:
  critical_points: vtk.vtkAlgorithm
  separatrices: vtk.vtkAlgorithm
  segmentation: vtk.vtkAlgorithm
  
  @staticmethod
  def create(
    input: vtk.vtkAlgorithmOutput,
    persistence_threshold: float = 0,
    field_name: Optional[str] = None
  ):
    morse_complex = ttk_util.MorseSmaleComplex(input, persistence_threshold, field_name)
    
    critical_points = vtk.vtkPassThrough()
    critical_points.SetInputConnection(morse_complex.GetOutputPort(0))
    
    separatrices = vtk.vtkPassThrough()
    separatrices.SetInputConnection(morse_complex.GetOutputPort(1))
    
    segmentation = vtk.vtkPassThrough()
    segmentation.SetInputConnection(morse_complex.GetOutputPort(3))
    
    return MorseSmaleComplex(
      critical_points,
      separatrices,
      segmentation
    )
  
  def __init__(
    self,
    critical_points: vtk.vtkAlgorithm,
    separatrices: vtk.vtkAlgorithm,
    segmentation: vtk.vtkAlgorithm
  ):
    self.critical_points = critical_points
    self.separatrices = separatrices
    self.segmentation = segmentation

  @cached_property
  def critical_points_point_data(self) -> pd.DataFrame:
    return vtk_util.PolyPointDataToDataFrame(self.critical_points.GetOutput())
  
  @cached_property
  def separatrices_point_data(self) -> pd.DataFrame:
    return vtk_util.PolyPointDataToDataFrame(self.separatrices.GetOutput())
  
  @cached_property
  def separatrices_cell_data(self) -> pd.DataFrame:
    return vtk_util.PolyCellDataToDataFrame(self.separatrices.GetOutput())

  def to_graph(self) -> MorseGraph:
    separatrices_points = self.separatrices_point_data
    separatrices_cells = self.separatrices_cell_data
    critical_points = self.critical_points_point_data
    
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

class MorseComplex(MorseSmaleComplex):
  critical_points: vtk.vtkAlgorithm
  separatrices: vtk.vtkAlgorithm
  segmentation: vtk.vtkAlgorithm
  
  @staticmethod
  def create(
    input: vtk.vtkAlgorithmOutput,
    persistence_threshold: float = 0,
    ascending: bool = True,
    field_name: Optional[str] = None
  ):
    morse_complex = ttk_util.MorseComplex(input, persistence_threshold, ascending, field_name)
    
    critical_points = vtk.vtkPassThrough()
    critical_points.SetInputConnection(morse_complex.GetOutputPort(0))
    
    separatrices = vtk.vtkPassThrough()
    separatrices.SetInputConnection(morse_complex.GetOutputPort(1))
    
    segmentation = vtk.vtkPassThrough()
    segmentation.SetInputConnection(morse_complex.GetOutputPort(3))
    
    return MorseComplex(
      critical_points,
      separatrices,
      segmentation
    )
  
  def __init__(
    self,
    critical_points: vtk.vtkAlgorithm,
    separatrices: vtk.vtkAlgorithm,
    segmentation: vtk.vtkAlgorithm
  ):
    super().__init__(
      critical_points,
      separatrices,
      segmentation
    )

class Complex:
  name: str
  frames: List[MorseSmaleComplex]
  
  def __init__(self, name, frames):
    self.name = name
    self.frames = frames
