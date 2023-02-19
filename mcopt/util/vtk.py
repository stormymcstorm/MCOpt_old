"""
Utilities for interacting with VTK
"""

import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.numpy_interface import dataset_adapter as dsa

def Tetrahedralize(input: vtk.vtkAlgorithmOutput):
  tetra = vtk.vtkDataSetTriangleFilter()
  tetra.SetInputConnection(0, input)
  
  return tetra

def Warp(input: vtk.vtkAlgorithmOutput, scale_factor: float):
  warp = vtk.vtkWarpScalar()
  warp.SetInputConnection(input)
  warp.SetScaleFactor(scale_factor)

  return warp
  
def Plane(scalars: np.ndarray, arr_name = 'data'):
  assert(scalars.ndim == 2)
  
  plane = vtk.vtkPlaneSource()
  plane.SetResolution(scalars.shape[0] - 1, scalars.shape[1] - 1)
  plane.SetOrigin([0, 0, 0])
  plane.SetPoint1([scalars.shape[0], 0, 0])
  plane.SetPoint2([0, scalars.shape[0], 0])
  plane.Update()
  
  data = numpy_to_vtk(scalars.ravel(), deep = True, array_type=vtk.VTK_DOUBLE)
  data.SetName(arr_name)
  
  plane.GetOutput().GetPointData().SetScalars(data)

  return plane

def get_cell_dataframe(poly : vtk.vtkPolyData):
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
