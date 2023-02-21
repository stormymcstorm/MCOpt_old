"""
Utilities for interacting with VTK
"""

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

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

