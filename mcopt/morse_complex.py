from functools import cached_property
import pandas as pd
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
# import topologytoolkit as ttk
from importlib import import_module

try:
  topologytoolkit = import_module('topologytoolkit')
except ImportError:
  topologytoolkit = None

globals()['ttk'] = topologytoolkit


def _make_complex(
  input : vtk.vtkAlgorithmOutput, 
  field_name, 
  persistence_threshold,
):
  if ttk is None:
    raise ImportError('topologytoolkit required for Morse Complex Computation')
  
  persistence_diagram = ttk.ttkPersistenceDiagram()
  persistence_diagram.SetInputConnection(input)
  persistence_diagram.SetInputArrayToProcess(0,0,0,0, field_name)
  
  critical_pairs = vtk.vtkThreshold()
  critical_pairs.SetInputConnection(persistence_diagram.GetOutputPort())
  critical_pairs.SetInputArrayToProcess(
    0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "PairIdentifier"
  )
  critical_pairs.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
  critical_pairs.SetLowerThreshold(-0.1)
  
  persistence_pairs = vtk.vtkThreshold()
  persistence_pairs.SetInputConnection(critical_pairs.GetOutputPort())
  persistence_pairs.SetInputArrayToProcess(
    0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Persistence"
  )
  persistence_pairs.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
  persistence_pairs.SetLowerThreshold(persistence_threshold)
  
  simplification = ttk.ttkTopologicalSimplification()
  simplification.SetInputConnection(0, input)
  simplification.SetInputArrayToProcess(0,0,0,0,field_name)
  simplification.SetInputConnection(1, persistence_pairs.GetOutputPort())
  
  complex = ttk.ttkMorseSmaleComplex()
  complex.SetInputConnection(simplification.GetOutputPort())
  complex.SetInputArrayToProcess(0,0,0,0,field_name)

  return complex

def _cell_data(poly):
  adapter = dsa.WrapDataObject(poly)
    
  cell_data = {}
  
  for k in adapter.CellData.keys():
    cell_data[k] = adapter.CellData[k]
    
  cells = pd.DataFrame(cell_data)
  
  cells.index.names = ['Cell Id']
  cells['Cell Type'] = pd.Series(dtype="Int64")
  
  id_list = vtk.vtkIdList()

  for cell_id in range(cells.shape[0]):
    poly.GetCellPoints(cell_id, id_list)
    
    for i in range(id_list.GetNumberOfIds()):
      k = 'Point Index ' + str(i)
      
      if k not in cells:
        cells[k] = pd.Series(dtype="Int64")
      
      cells.at[cell_id, k] = id_list.GetId(i)
      
    cells.at[cell_id, 'Cell Type'] = poly.GetCellType(cell_id)
  
  return cells

def _point_data(poly):
  adapter = dsa.WrapDataObject(poly)
    
  point_data = {}

  for k in adapter.PointData.keys():
    point_data[k] = adapter.PointData[k]
    
  points = pd.DataFrame(point_data)
  points.index.names = ['Point ID']
  points['Points_0'] = pd.Series(dtype="Float64")
  points['Points_1'] = pd.Series(dtype="Float64")
  points['Points_2'] = pd.Series(dtype="Float64")
  
  for point_id in range(points.shape[0]):
    x, y, z = poly.GetPoint(point_id)
    
    points.at[point_id, 'Points_0'] = x
    points.at[point_id, 'Points_1'] = y
    points.at[point_id, 'Points_2'] = z
  
  return points

class MorseSmaleComplex:
  @staticmethod
  def create(
    input : vtk.vtkAlgorithmOutput, 
    field_name='data', 
    persistence_threshold=0.05
  ):
    complex = _make_complex(input, field_name, persistence_threshold)
    
    complex.Update()
    
    return MorseSmaleComplex(
      complex.GetOutputPort(0), 
      complex.GetOutputPort(1), 
      complex.GetOutputPort(3), 
      _save = complex # Prevent deletion of complex
    )
  
  critical_points : vtk.vtkAlgorithmOutput
  separatrices : vtk.vtkAlgorithmOutput
  segmentation : vtk.vtkAlgorithmOutput
  
  def __init__(
    self,
    critical_points : vtk.vtkAlgorithmOutput,
    separatrices : vtk.vtkAlgorithmOutput,
    segmentation : vtk.vtkAlgorithmOutput,
    _save = None
  ):
    self._save = _save
    self.critical_points = critical_points
    self.separatrices = separatrices
    self.segmentation = segmentation   
    
  @property
  def critical_points_data(self):
    return self.critical_points.GetProducer().GetOutput(self.critical_points.GetIndex())
  
  @property
  def separatrices_data(self):
    return self.separatrices.GetProducer().GetOutput(self.separatrices.GetIndex())
  
  @property
  def segmentation_data(self):
    return self.segmentation.GetProducer().GetOutput(self.segmentation.GetIndex())

  # Tries to mimic the output of exporting from paraview
  @cached_property
  def separatrices_cell_data(self) -> pd.DataFrame:
    return _cell_data(self.separatrices_data)
    
  
  # Tries to mimic the output of exporting from paraview
  @cached_property
  def separatrices_point_data(self) -> pd.DataFrame:
    return _point_data(self.separatrices_data)
    

  @cached_property
  def critical_points_point_data(self) -> pd.DataFrame:
    return _point_data(self.critical_points_data)

class MorseComplex(MorseSmaleComplex):
  @staticmethod
  def create(
    input : vtk.vtkAlgorithmOutput, 
    field_name='data', 
    persistence_threshold=0.05,
    ascending=True
  ):
    complex = _make_complex(input, field_name, persistence_threshold)
    
    complex.SetComputeCriticalPoints(True)
    complex.SetComputeAscendingSeparatrices1(False)
    complex.SetComputeAscendingSeparatrices2(False)

    complex.SetComputeAscendingSegmentation(ascending)
    complex.SetComputeDescendingSegmentation(not ascending)
    complex.SetComputeDescendingSeparatrices1(ascending)
    complex.SetComputeDescendingSeparatrices2(not ascending)
        
    return MorseComplex(
      complex.GetOutputPort(0), 
      complex.GetOutputPort(1), 
      complex.GetOutputPort(3), 
      _save = complex # Prevent deletion of complex
    )
  
  def __init__(
    self,
    critical_points : vtk.vtkAlgorithmOutput,
    separatrices : vtk.vtkAlgorithmOutput,
    segmentation : vtk.vtkAlgorithmOutput,
    _save = None
  ):
    super().__init__(critical_points, separatrices, segmentation, _save = _save)
    