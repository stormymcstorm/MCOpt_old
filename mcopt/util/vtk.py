"""
Utilities for computations with VTK
"""

from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.numpy_interface import dataset_adapter as dsa

def Warp(
  input: vtk.vtkAlgorithmOutput,
  scale_factor: float = 1,
) -> vtk.vtkAlgorithm:
  warp = vtk.vtkWarpScalar()
  warp.SetInputConnection(input)
  warp.SetScaleFactor(scale_factor)
  
  return warp

def PlaneSource(scalars : ArrayLike) -> vtk.vtkAlgorithm:
  arr = np.asarray(scalars)
  
  assert(arr.ndim == 2)
  
  plane = vtk.vtkPlaneSource()
  plane.SetResolution(scalars.shape[0] - 1, scalars.shape[1] - 1)
  plane.SetOrigin([0, 0, 0])
  plane.SetPoint1([scalars.shape[0], 0, 0])
  plane.SetPoint2([0, scalars.shape[0], 0])
  plane.Update()
  
  data = numpy_to_vtk(scalars.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
  data.SetName('data')
  
  plane.GetOutput().GetPointData().SetScalars(data)
  
  return plane
  
def ReadVTP(
  file_name: str
) -> vtk.vtkAlgorithm:
  reader = vtk.vtkXMLPolyDataReader()
  reader.SetFileName(file_name)
  reader.Update()
  
  return reader

def ReadVTU(
  file_name: str
) -> vtk.vtkAlgorithm:
  reader = vtk.vtkXMLUnstructuredGridReader()
  reader.SetFileName(file_name)
  reader.Update()
  
  return reader

def ReadVTI(
  file_name: str
) -> vtk.vtkAlgorithm:
  reader = vtk.vtkXMLImageDataReader()
  reader.SetFileName(file_name)
  reader.Update()
  
  return reader

def WriteVTP(
  input: vtk.vtkAlgorithmOutput,
  file_name: str
):
  writer = vtk.vtkXMLPolyDataWriter()
  writer.SetInputConnection(input)
  writer.SetFileName(file_name)
  writer.Write()
  
def WriteVTU(
  input: vtk.vtkAlgorithmOutput,
  file_name: str
):
  writer = vtk.vtkXMLUnstructuredGridWriter()
  writer.SetInputConnection(input)
  writer.SetFileName(file_name)
  writer.Write()
  
def WriteVTI(
  input: vtk.vtkAlgorithmOutput,
  file_name: str
):
  writer = vtk.vtkXMLImageDataWriter()
  writer.SetInputConnection(input)
  writer.SetFileName(file_name)
  writer.Write()
  
def PolyCellDataToDataFrame(poly: vtk.vtkPolyData) -> pd.DataFrame:
  adapter = dsa.WrapDataObject(poly)
  
  cells = pd.DataFrame(dict(adapter.CellData))
  
  cells.index.names = ['Cell Id']
  cells['Cell Type'] = pd.Series(dtype='Int64')
  
  id_list = vtk.vtkIdList()
  
  for cell_id in range(cells.shape[0]):
    poly.GetCellPoints(cell_id, id_list)
    
    for i in range(id_list.GetNumberOfIds()):
      k = f'Point Index {i}'
      
      if k not in cells:
        cells[k] = pd.Series(dtype='Int64')
        
      cells.at[cell_id, k] = id_list.GetId(i)
    
    cells.at[cell_id, 'Cell Type'] = poly.GetCellType(cell_id)
  
  return cells

def PolyPointDataToDataFrame(poly: vtk.vtkPolyData) -> pd.DataFrame:
  adapter = dsa.WrapDataObject(poly)
  
  points = pd.DataFrame(dict(adapter.PointData))
  
  points.index.names = ['Point ID']
  points['Points_0'] = pd.Series(dtype='Float64')
  points['Points_1'] = pd.Series(dtype='Float64')
  points['Points_2'] = pd.Series(dtype='Float64')
  
  for point_id in range(points.shape[0]):
    x, y, z = poly.GetPoint(point_id)
    
    points.at[point_id, 'Points_0'] = x
    points.at[point_id, 'Points_1'] = y
    points.at[point_id, 'Points_2'] = z
  
  return points
  