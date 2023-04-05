"""
Utilities for computing Morse Complexes
"""

from typing import Optional
from importlib import import_module

import vtk

try:
  import topologytoolkit as ttk
except ImportError:
  ttk = None

def MorseSmaleComplex(
  input: vtk.vtkAlgorithmOutput,
  persistence_threshold: float = 0,
  field_name: Optional[str] = None
):
  tetra = vtk.vtkDataSetTriangleFilter()
  tetra.SetInputConnection(input)
  
  persistence_diagram = ttk.ttkPersistenceDiagram()
  persistence_diagram.SetInputConnection(tetra.GetOutputPort())
  if field_name is None:
    persistence_diagram.SetInputArrayToProcess(
      0, 0, 0, 0, vtk.vtkDataSetAttributes.SCALARS
    )
  else:
    persistence_diagram.SetInputArrayToProcess(
      0, 0, 0, 0, field_name
    )
  persistence_diagram.Update()
  
  critical_pairs = vtk.vtkThreshold()
  critical_pairs.SetInputConnection(persistence_diagram.GetOutputPort())
  critical_pairs.SetInputArrayToProcess(
    0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "PairIdentifier"
  )
  critical_pairs.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
  critical_pairs.SetLowerThreshold(-0.1)
  
  persistent_pairs = vtk.vtkThreshold()
  persistent_pairs.SetInputConnection(critical_pairs.GetOutputPort())
  persistent_pairs.SetInputArrayToProcess(
    0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Persistence"
  )
  persistent_pairs.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
  persistent_pairs.SetLowerThreshold(persistence_threshold)
  
  simplification = ttk.ttkTopologicalSimplification()
  simplification.SetInputConnection(0, tetra.GetOutputPort())
  if field_name is None:
    simplification.SetInputArrayToProcess(
      0, 0, 0, 0, vtk.vtkDataSetAttributes.SCALARS
    )
  else:
    simplification.SetInputArrayToProcess(
      0, 0, 0, 0, field_name
    )
  simplification.SetInputConnection(1, persistent_pairs.GetOutputPort())
  
  complex = ttk.ttkMorseSmaleComplex()
  complex.SetInputConnection(simplification.GetOutputPort())
  if field_name is None:
    complex.SetInputArrayToProcess(
      0, 0, 0, 0, vtk.vtkDataSetAttributes.SCALARS
    )
  else:
    complex.SetInputArrayToProcess(
      0, 0, 0, 0, field_name
    )
    
  return complex

def MorseComplex(
  input: vtk.vtkAlgorithmOutput,
  persistence_threshold: float = 0,
  ascending: bool = True,
  field_name: Optional[str] = None,
):
  complex = MorseSmaleComplex(input, persistence_threshold, field_name)
  
  complex.SetComputeCriticalPoints(True)
  complex.SetComputeAscendingSeparatrices1(False)
  complex.SetComputeAscendingSeparatrices2(False)
  
  complex.SetComputeAscendingSegmentation(ascending)
  complex.SetComputeDescendingSegmentation(not ascending)
  complex.SetComputeDescendingSeparatrices1(ascending)
  complex.SetComputeDescendingSeparatrices2(not ascending)

  return complex