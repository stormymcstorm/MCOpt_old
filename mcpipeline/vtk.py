"""
Utilities for working with VTK
"""

from abc import ABC, abstractmethod
import os
import json

import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa
import numpy as np
import pandas as pd

FILE_TYPE_READERS = {
  '.vtk': vtk.vtkStructuredGridReader,
  '.vts': vtk.vtkXMLStructuredGridReader,
  '.vtu': vtk.vtkXMLUnstructuredGridReader,
  '.vtp': vtk.vtkXMLPolyDataReader,
  '.vti': vtk.vtkXMLImageDataReader,
}

DATA_TYPE_WRITERS = {
  vtk.vtkStructuredGrid: vtk.vtkXMLStructuredGridWriter,
  vtk.vtkUnstructuredGrid: vtk.vtkXMLUnstructuredGridWriter,
  vtk.vtkPolyData: vtk.vtkXMLPolyDataWriter,
  vtk.vtkImageData: vtk.vtkXMLImageDataWriter,
}

DATA_TYPE_EXT = {
  vtk.vtkStructuredGrid: '.vts',
  vtk.vtkUnstructuredGrid: '.vtu',
  vtk.vtkPolyData: '.vtp',
  vtk.vtkImageData: '.vti',
}

def Read(file_name: str) -> vtk.vtkAlgorithm:
  _, ext = os.path.splitext(file_name)
  
  if ext in FILE_TYPE_READERS:
    reader = FILE_TYPE_READERS[ext]()
  else:
    raise ValueError(f'Unrecognized vtk file type for {file_name}: {ext}')
  
  reader.SetFileName(file_name)
  reader.Update()
  return reader

def Write(input: vtk.vtkAlgorithmOutput, file_name: str) -> str:
  output_ty = type(input.GetProducer().GetOutput(input.GetIndex()))
  
  if output_ty in DATA_TYPE_WRITERS:
    writer = DATA_TYPE_WRITERS[output_ty]()
  else:
    raise ValueError(f'Unsupported data type: {output_ty}')
  
  if os.path.splitext(file_name)[1] == '':
    file_name += DATA_TYPE_EXT[output_ty]
    
  writer.SetInputConnection(input)
  writer.SetFileName(file_name)
  writer.Write()  
  
  return file_name

class VTKFilter(ABC):
  filter_ty: str
  
  def __init__(self, args):
    super().__init__()
    
    self.args = args
    
  def __eq__(self, other: object) -> bool:
    if isinstance(other, VTKFilter):
      return self.filter_ty == other.filter_ty and self.args == other.args
    
    return False
  
  @abstractmethod
  def __call__(self, input: vtk.vtkAlgorithmOutput) -> vtk.vtkAlgorithm:
    raise NotImplementedError()
  
class WarpFilter(VTKFilter):
  filter_ty = 'warp'
  
  scale_factor: float
  
  def __init__(self, scale_factor: float) -> None:
    self.scale_factor = scale_factor
    
    super().__init__(args=dict(scale_factor=scale_factor))
    
  def __call__(self, input: vtk.vtkAlgorithmOutput) -> vtk.vtkAlgorithm:
    warp = vtk.vtkWarpScalar()
    warp.SetInputConnection(input)
    warp.SetScaleFactor(self.scale_factor)
    
    return warp
  
# class Tetrahedralize(VTKFilter):
#   filter_ty = 'tetra'
  
#   def __init__(self):
#     super().__init__(args={})
    
#   def __call__(self, input: vtk.vtkAlgorithmOutput) -> vtk.vtkAlgorithm:
#     tetra = vtk.vtkDataSetTriangleFilter() # type: ignore
#     tetra.SetInputConnection(input)
    
#     return tetra
  
class BoxClipFilter(VTKFilter):
  filter_ty = 'boxclip'
  
  xmin: float
  xmax: float
  ymin: float
  ymax: float
  zmin: float
  zmax: float
  
  def __init__(
    self,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float 
  ):
    self.xmin = xmin
    self.xmax = xmax
    self.ymin = ymin
    self.ymax = ymax
    self.zmin = zmin
    self.zmax = zmax
    
    super().__init__(args = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax))
    
  def __call__(self, input: vtk.vtkAlgorithmOutput) -> vtk.vtkAlgorithm:
    box_clip = vtk.vtkBoxClipDataSet()
    box_clip.SetInputConnection(input)
    box_clip.SetBoxClip(
      self.xmin, 
      self.xmax, 
      self.ymin, 
      self.ymax, 
      self.zmin, 
      self.zmax
    )
    
    return box_clip
  
class ImageClipFilter(VTKFilter):
  filter_ty = 'image_clip'
  
  xmin: int
  xmax: int
  ymin: int
  ymax: int
  zmin: int
  zmax: int
  
  def __init__(
    self,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    zmin: int,
    zmax: int 
  ):
    self.xmin = xmin
    self.xmax = xmax
    self.ymin = ymin
    self.ymax = ymax
    self.zmin = zmin
    self.zmax = zmax
    
    super().__init__(args = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax))
    
  def __call__(self, input: vtk.vtkAlgorithmOutput) -> vtk.vtkAlgorithm:
    clip = vtk.vtkImageClip()
    clip.SetInputConnection(input)
    clip.SetOutputWholeExtent(
      self.xmin, 
      self.xmax, 
      self.ymin, 
      self.ymax, 
      self.zmin, 
      self.zmax
    )
    clip.ClipDataOn()
    
    return clip

class TranslateFilter(VTKFilter):
  filter_ty = 'translate'
  
  dx: float
  dy: float
  dz: float
  
  def __init__(self,  dx: float = 0, dy: float = 0, dz: float = 0):
    self.dx = dx
    self.dy = dy
    self.dz = dz
    
    super().__init__(args = dict(dx = dx, dy = dy, dz = dz))
    
  def __call__(self, input: vtk.vtkAlgorithmOutput) -> vtk.vtkAlgorithm:
    filter = vtk.vtkTransformFilter()
    filter.SetInputConnection(input)
    
    trans = vtk.vtkTransform()
    trans.Translate(self.dx, self.dy, self.dz)
    
    filter.SetTransform(trans)
    
    return filter

class VTKFilterEncoder(json.JSONEncoder):
  def default(self, o):
    if isinstance(o, VTKFilter):
      return {'__vtk_filter__': o.filter_ty, 'args': o.args}
    
    return json.JSONEncoder.default(self, o)

VTK_FILTERS = [WarpFilter, BoxClipFilter, TranslateFilter, ImageClipFilter]

def VTKFilterDecoder(dct):
  if '__vtk_filter__' in dct:
    filter_ty = dct['__vtk_filter__']
    args = dct['args']
    
    for fltr in VTK_FILTERS:
      if fltr.filter_ty == filter_ty:
        return fltr(**args)
      
    raise ValueError(f'Unrecognized VTKFilter type: {filter_ty}')
  
  return dct

def PlaneSource(scalars: np.ndarray) -> vtk.vtkAlgorithm:  
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
  
  cells = pd.DataFrame(dict(adapter.CellData)) # type: ignore
  
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