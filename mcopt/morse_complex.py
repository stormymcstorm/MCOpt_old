"""
Representation and logic for working with Morse Complexes
"""

from __future__ import annotations
import os

import vtk

import mcopt.util.ttk as ttk_util
import mcopt.util.vtk as vtk_util

def _read_complex(dir_name):
  crit_file = os.path.join(dir_name, 'critical_points.vtp')
  if not os.path.exists(crit_file):
    raise ValueError('Cannot find critical_points.vtp')
  
  critical_points = vtk_util.ReadVTP(crit_file)
  
  sep_file = os.path.join(dir_name, 'separatrices.vtp')
  if not os.path.exists(sep_file):
    raise ValueError('Cannot find separatrices.vtp')
  
  separatrices = vtk_util.ReadVTP(sep_file)
  
  seg_file = os.path.join(dir_name, 'segmentation')
  
  if os.path.exists(f'{seg_file}.vtu'):
    segmentation = vtk_util.ReadVTU(f'{seg_file}.vtu')
  elif os.path.exists(f'{seg_file}.vti'):
    segmentation = vtk_util.ReadVTI(f'{seg_file}.vti')
  else:
    raise ValueError('Cannot find segmentation file')
  
  return (critical_points, separatrices, segmentation)

def _validate_morse_complex_data(
  critical_points : vtk.vtkAlgorithm,
  separatrices : vtk.vtkAlgorithm,
  segmentation : vtk.vtkAlgorithm,
):
  pass

class MorseSmaleComplex:
  critical_points : vtk.vtkAlgorithm
  separatrices : vtk.vtkAlgorithm
  segmentation : vtk.vtkAlgorithm
  
  @staticmethod
  def read(dir_name):
    (critical_points, separatrices, segmentation) = _read_complex(dir_name)
    
    return MorseSmaleComplex(
      critical_points, 
      separatrices, 
      segmentation,
    )
  
  @staticmethod
  def create(
    input: vtk.vtkAlgorithmOutput,
    persistence_threshold : float = 0
  ) -> MorseSmaleComplex:
    complex = ttk_util.MorseSmaleComplex(input, persistence_threshold)
    
    critical_points = vtk.vtkPassThrough()
    critical_points.SetInputConnection(complex.GetOutputPort(0))
    
    separatrices = vtk.vtkPassThrough()
    separatrices.SetInputConnection(complex.GetOutputPort(1))
    
    segmentation = vtk.vtkPassThrough()
    segmentation.SetInputConnection(complex.GetOutputPort(3))
    
    return MorseSmaleComplex(
      critical_points,
      separatrices,
      segmentation
    )
  
  def __init__(
    self,
    critical_points : vtk.vtkAlgorithm,
    separatrices : vtk.vtkAlgorithm,
    segmentation : vtk.vtkAlgorithm,
  ):
    _validate_morse_complex_data(
      critical_points, separatrices, segmentation
    )
    
    self.critical_points = critical_points
    self.separatrices = separatrices
    self.segmentation = segmentation
        
  def write(self, dir_name):
    os.makedirs(dir_name, exist_ok=True)
    
    vtk_util.WriteVTP(
      self.critical_points.GetOutputPort(), 
      os.path.join(dir_name, 'critical_points.vtp')
    )
    vtk_util.WriteVTP(
      self.separatrices.GetOutputPort(), 
      os.path.join(dir_name, 'separatrices.vtp')
    )
    
    if isinstance(self.segmentation.GetOutput(), vtk.vtkUnstructuredGrid):
      vtk_util.WriteVTU(
        self.segmentation.GetOutputPort(), 
        os.path.join(dir_name, 'segmentation.vtu')
      )
    elif isinstance(self.segmentation.GetOutput(), vtk.vtkImageData):
      vtk_util.WriteVTI(
        self.segmentation.GetOutputPort(), 
        os.path.join(dir_name, 'segmentation.vti')
      )
  
class MorseComplex(MorseSmaleComplex):
  @staticmethod
  def read(dir_name):
    (critical_points, separatrices, segmentation) = _read_complex(dir_name)
    
    return MorseComplex(
      critical_points, 
      separatrices, 
      segmentation,
    )
  
  @staticmethod
  def create(
    input: vtk.vtkAlgorithmOutput,
    persistence_threshold: float = 0,
    ascending: bool = True,
  ):
    complex = ttk_util.MorseSmaleComplex(input, persistence_threshold)

    complex.SetComputeCriticalPoints(True)
    complex.SetComputeAscendingSeparatrices1(False)
    complex.SetComputeAscendingSeparatrices2(False)
    
    complex.SetComputeAscendingSegmentation(ascending)
    complex.SetComputeDescendingSegmentation(not ascending)
    complex.SetComputeDescendingSeparatrices1(ascending)
    complex.SetComputeDescendingSeparatrices2(not ascending)
    
    critical_points = vtk.vtkPassThrough()
    critical_points.SetInputConnection(complex.GetOutputPort(0))
    
    separatrices = vtk.vtkPassThrough()
    separatrices.SetInputConnection(complex.GetOutputPort(1))
    
    segmentation = vtk.vtkPassThrough()
    segmentation.SetInputConnection(complex.GetOutputPort(3))
    
    return MorseComplex(
      critical_points,
      separatrices,
      segmentation
    )
  
  def __init__(
    self, 
    critical_points: vtk.vtkAlgorithm, 
    separatrices: vtk.vtkAlgorithm, 
    segmentation: vtk.vtkAlgorithm,
  ):
    super().__init__(critical_points, separatrices, segmentation)