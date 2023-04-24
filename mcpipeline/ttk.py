"""
Utilities for working with TTK
"""

from functools import cached_property

import vtk # pylint: disable=import-error
import numpy as np
import pandas as pd
import networkx as nx

import mcpipeline.vtk as vtk_util
from mcopt import MorseGraph

try:
  import topologytoolkit as ttk
except ImportError:
  ttk = None
  
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
      
    x, y, z = data['Points_0'], data['Points_1'], data['Points_2']
    point_map[id] = next_node
    nodes[next_node] = {
      'pos2': np.array([x, y]),
      'pos3': np.array([x,y,z]),
      'point_ids': [id],
      'is_critical': is_crit
    }
    
    next_node += 1
    
  critical_nodes = set(cell_map.values())
  
  return nodes, point_map, critical_nodes
  
class MorseComplex:
  @staticmethod
  def create(
    input: vtk.vtkAlgorithm,
    persistence_threshold: float,
    ascending: bool = True,
    scalar_field: str | None = None
  ):
    if ttk is None:
      raise RuntimeError(f'ttk is required for MorseComplex creation')
    
    tetra = vtk.vtkDataSetTriangleFilter() # type: ignore
    tetra.SetInputConnection(input)
    
    persistence_diagram = ttk.ttkPersistenceDiagram() # type: ignore
    persistence_diagram.SetInputConnection(tetra.GetOutputPort())
    if scalar_field is None:
      persistence_diagram.SetInputArrayToProcess(
        0, 0, 0, 0, vtk.vtkDataSetAttributes.SCALARS # type: ignore
      )
    else:
      persistence_diagram.SetInputArrayToProcess(
        0, 0, 0, 0, scalar_field
      )
    persistence_diagram.Update()
    
    critical_pairs = vtk.vtkThreshold() # type: ignore
    critical_pairs.SetInputConnection(persistence_diagram.GetOutputPort())
    critical_pairs.SetInputArrayToProcess(
      0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "PairIdentifier" # type: ignore
    )
    critical_pairs.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN) # type: ignore
    critical_pairs.SetLowerThreshold(-0.1)
    
    persistent_pairs = vtk.vtkThreshold() # type: ignore
    persistent_pairs.SetInputConnection(critical_pairs.GetOutputPort())
    persistent_pairs.SetInputArrayToProcess(
      0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Persistence" # type: ignore
    )
    persistent_pairs.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN) # type: ignore
    persistent_pairs.SetLowerThreshold(persistence_threshold)
    
    simplification = ttk.ttkTopologicalSimplification() # type: ignore
    simplification.SetInputConnection(0, tetra.GetOutputPort())
    if scalar_field is None:
      simplification.SetInputArrayToProcess(
        0, 0, 0, 0, vtk.vtkDataSetAttributes.SCALARS # type: ignore
      )
    else:
      simplification.SetInputArrayToProcess(
        0, 0, 0, 0, scalar_field
      )
    simplification.SetInputConnection(1, persistent_pairs.GetOutputPort())
    
    complex = ttk.ttkMorseSmaleComplex() # type: ignore
    complex.SetInputConnection(simplification.GetOutputPort())
    if scalar_field is None:
      complex.SetInputArrayToProcess(
        0, 0, 0, 0, vtk.vtkDataSetAttributes.SCALARS # type: ignore
      )
    else:
      complex.SetInputArrayToProcess(
        0, 0, 0, 0, scalar_field
      )
      
    complex.SetComputeCriticalPoints(True)
    complex.SetComputeAscendingSeparatrices1(False)
    complex.SetComputeAscendingSeparatrices2(False)
    
    complex.SetComputeAscendingSegmentation(ascending)
    complex.SetComputeDescendingSegmentation(not ascending)
    complex.SetComputeDescendingSeparatrices1(ascending)
    complex.SetComputeDescendingSeparatrices2(not ascending)
    
    complex.Update()
      
    critical_points = vtk.vtkPassThrough()
    critical_points.SetInputConnection(complex.GetOutputPort(0))
    
    separatrices = vtk.vtkPassThrough()
    separatrices.SetInputConnection(complex.GetOutputPort(1))
    
    segmentation = vtk.vtkPassThrough()
    segmentation.SetInputConnection(complex.GetOutputPort(3))
    
    return MorseComplex(critical_points, separatrices, segmentation)
  
  critical_points: vtk.vtkAlgorithm
  separatrices: vtk.vtkAlgorithm
  segmentation: vtk.vtkAlgorithm
  
  def __init__(
    self,
    critical_points: vtk.vtkAlgorithm,
    separatrices: vtk.vtkAlgorithm,
    segmentation: vtk.vtkAlgorithm,
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