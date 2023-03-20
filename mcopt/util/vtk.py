"""
Utilities for computations with VTK
"""

from numpy.typing import ArrayLike
import numpy as np

import vtk
from vtk.util.numpy_support import numpy_to_vtk

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
  
  return reader

def ReadVTU(
  file_name: str
) -> vtk.vtkAlgorithm:
  reader = vtk.vtkXMLUnstructuredGridReader()
  reader.SetFileName(file_name)
  
  return reader

def ReadVTI(
  file_name: str
) -> vtk.vtkAlgorithm:
  reader = vtk.vtkXMLImageDataReader()
  reader.SetFileName(file_name)
  
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