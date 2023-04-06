"""
Utilities for working with VTK
"""

import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.numpy_interface import dataset_adapter as dsa

import numpy as np
import pandas as pd

from mcopt.pipeline.util import file_ext

FILE_TYPE_READERS = {
  'vtk': vtk.vtkStructuredGridReader,
  'vts': vtk.vtkXMLStructuredGridReader,
  'vtu': vtk.vtkXMLUnstructuredGridReader,
  'vtp': vtk.vtkXMLPolyDataReader,
  'vti': vtk.vtkXMLImageDataReader,
}

DATA_TYPE_WRITERS = {
  vtk.vtkStructuredGrid: [vtk.vtkXMLStructuredGridWriter, 'vts'],
  vtk.vtkUnstructuredGrid: [vtk.vtkXMLUnstructuredGridWriter, 'vtu'],
  vtk.vtkPolyData: [vtk.vtkXMLPolyDataWriter, 'vtp'],
  vtk.vtkImageData: [vtk.vtkXMLImageDataWriter, 'vti'],
}

def Read(file_name : str):
  ext = file_ext(file_name)
  
  if ext in FILE_TYPE_READERS:
    reader = FILE_TYPE_READERS[ext]()
  else:
    raise ValueError(f'Unrecognized file type {ext}')
  
  reader.SetFileName(file_name)
  reader.Update()
  return reader

def Write(input: vtk.vtkAlgorithmOutput, file_name: str):
  output_ty = type(input.GetProducer().GetOutput(input.GetIndex()))
  
  if output_ty in DATA_TYPE_WRITERS:
    writer = DATA_TYPE_WRITERS[output_ty][0]()
  else:
    raise ValueError(f'Unsupported data type {output_ty}')
  
  
  if file_ext(file_name) is None:
    file_name = file_name + '.' + DATA_TYPE_WRITERS[output_ty][1]
  
  writer.SetInputConnection(input)
  writer.SetFileName(file_name)
  writer.Write()
  
def BoxClip(
  input: vtk.vtkAlgorithmOutput,
  xmin: float,
  xmax: float,
  ymin: float,
  ymax: float,
  zmin: float,
  zmax: float 
):
  box_clip = vtk.vtkBoxClipDataSet()
  box_clip.SetInputConnection(input)
  box_clip.SetBoxClip(xmin, xmax, ymin, ymax, zmin, zmax)
  
  return box_clip

def Warp(
  input: vtk.vtkAlgorithmOutput,
  scale_factor : float = 1
):
  warp = vtk.vtkWarpScalar()
  warp.SetInputConnection(input)
  warp.SetScaleFactor(scale_factor)
  
  return warp

FILTERS = {
  'box_clip': BoxClip,
  'warp': Warp,
}

def PlaneSource(scalars) -> vtk.vtkAlgorithm:  
  assert(scalars.ndim == 2)
  
  plane = vtk.vtkPlaneSource()
  plane.SetResolution(scalars.shape[0] - 1, scalars.shape[1] - 1)
  plane.SetOrigin([0, 0, 0])
  plane.SetPoint1([scalars.shape[0], 0, 0])
  plane.SetPoint2([0, scalars.shape[0], 0])
  plane.Update()
  
  data= numpy_to_vtk(scalars.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
  data.SetName('data')
  
  plane.GetOutput().GetPointData().SetScalars(data)
  
  return plane
 
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