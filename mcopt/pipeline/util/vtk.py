"""
Utilities for working with VTK
"""

from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkIOLegacy import (
  vtkStructuredGridReader
)
from vtkmodules.vtkIOXML import (
  vtkXMLStructuredGridReader,
  vtkXMLStructuredGridWriter,
  vtkXMLUnstructuredGridReader,
  vtkXMLUnstructuredGridWriter,
  vtkXMLPolyDataReader,
  vtkXMLPolyDataWriter,
  vtkXMLImageDataReader,
  vtkXMLImageDataWriter,
)
from vtkmodules.vtkCommonDataModel import (
  vtkStructuredGrid,
  vtkUnstructuredGrid,
  vtkPolyData,
  vtkImageData
)
from vtkmodules.vtkCommonCore import vtkIdList, VTK_DOUBLE
from vtkmodules.vtkCommonExecutionModel import (
  vtkAlgorithm,
  vtkAlgorithmOutput,
)
from vtkmodules.vtkFiltersGeneral import (
  vtkBoxClipDataSet,
  vtkWarpScalar
)
from vtkmodules.vtkFiltersSources import (
  vtkPlaneSource
)

import numpy as np
import pandas as pd

from mcopt.pipeline.util import file_ext

FILE_TYPE_READERS = {
  'vtk': vtkStructuredGridReader,
  'vts': vtkXMLStructuredGridReader,
  'vtu': vtkXMLUnstructuredGridReader,
  'vtp': vtkXMLPolyDataReader,
  'vti': vtkXMLImageDataReader,
}

DATA_TYPE_WRITERS = {
  vtkStructuredGrid: [vtkXMLStructuredGridWriter, 'vts'],
  vtkUnstructuredGrid: [vtkXMLUnstructuredGridWriter, 'vtu'],
  vtkPolyData: [vtkXMLPolyDataWriter, 'vtp'],
  vtkImageData: [vtkXMLImageDataWriter, 'vti'],
}

def Read(file_name : str) -> vtkAlgorithm:
  ext = file_ext(file_name)
  
  if ext in FILE_TYPE_READERS:
    reader = FILE_TYPE_READERS[ext]()
  else:
    raise ValueError(f'Unrecognized file type {ext}')
  
  reader.SetFileName(file_name)
  reader.Update()
  return reader

def Write(input: vtkAlgorithmOutput, file_name: str):
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
  input: vtkAlgorithmOutput,
  xmin: float,
  xmax: float,
  ymin: float,
  ymax: float,
  zmin: float,
  zmax: float 
) -> vtkAlgorithm:
  box_clip = vtkBoxClipDataSet()
  box_clip.SetInputConnection(input)
  box_clip.SetBoxClip(xmin, xmax, ymin, ymax, zmin, zmax)
  
  return box_clip

def Warp(
  input: vtkAlgorithmOutput,
  scale_factor : float = 1
) -> vtkAlgorithm:
  warp = vtkWarpScalar()
  warp.SetInputConnection(input)
  warp.SetScaleFactor(scale_factor)
  
  return warp

FILTERS = {
  'box_clip': BoxClip,
  'warp': Warp,
}

def PlaneSource(scalars: np.ndarray) -> vtkAlgorithm:  
  assert(scalars.ndim == 2)
  
  plane = vtkPlaneSource()
  plane.SetResolution(scalars.shape[0] - 1, scalars.shape[1] - 1)
  plane.SetOrigin([0, 0, 0])
  plane.SetPoint1([scalars.shape[0], 0, 0])
  plane.SetPoint2([0, scalars.shape[0], 0])
  plane.Update()
  
  data= numpy_to_vtk(scalars.ravel(), deep=True, array_type=VTK_DOUBLE)
  data.SetName('data')
  
  plane.GetOutput().GetPointData().SetScalars(data)
  
  return plane
 
def PolyCellDataToDataFrame(poly: vtkPolyData) -> pd.DataFrame:
  adapter = dsa.WrapDataObject(poly)
  
  cells = pd.DataFrame(dict(adapter.CellData)) # type: ignore
  
  cells.index.names = ['Cell Id']
  cells['Cell Type'] = pd.Series(dtype='Int64')
  
  id_list = vtkIdList()
  
  for cell_id in range(cells.shape[0]):
    poly.GetCellPoints(cell_id, id_list)
    
    for i in range(id_list.GetNumberOfIds()):
      k = f'Point Index {i}'
      
      if k not in cells:
        cells[k] = pd.Series(dtype='Int64')
        
      cells.at[cell_id, k] = id_list.GetId(i)
    
    cells.at[cell_id, 'Cell Type'] = poly.GetCellType(cell_id)
  
  return cells

def PolyPointDataToDataFrame(poly: vtkPolyData) -> pd.DataFrame:
  adapter = dsa.WrapDataObject(poly)
  
  points = pd.DataFrame(dict(adapter.PointData)) # type: ignore
  
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