"""
Utilities for computing Morse Complexes
"""

from typing import Optional
from importlib import import_module

import vtk # pylint: disable=import-error

try:
  import topologytoolkit as ttk
except ImportError:
  ttk = None

def MorseSmaleComplex(
  input: vtk.vtkAlgorithmOutput,  # type: ignore
  persistence_threshold: float = 0,
  field_name: Optional[str] = None
) -> vtk.vtkAlgorithm: # type: ignore
  tetra = vtk.vtkDataSetTriangleFilter() # type: ignore
  tetra.SetInputConnection(input)
  
  persistence_diagram = ttk.ttkPersistenceDiagram() # type: ignore
  persistence_diagram.SetInputConnection(tetra.GetOutputPort())
  if field_name is None:
    persistence_diagram.SetInputArrayToProcess(
      0, 0, 0, 0, vtk.vtkDataSetAttributes.SCALARS # type: ignore
    )
  else:
    persistence_diagram.SetInputArrayToProcess(
      0, 0, 0, 0, field_name
    )
  persistence_diagram.Update()
  
  critical_pairs = vtk.vtkThreshold() # type: ignore
  critical_pairs.SetInputConnection(persistence_diagram.GetOutputPort())
  critical_pairs.SetInputArrayToProcess(
    0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "PairIdentifier" # type: ignore
  )
  critical_pairs.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN) # type: ignore
  critical_pairs.SetLowerThreshold(-0.1)
  
  persistent_pairs = vtk.vtkThreshold() # type: ignore
  persistent_pairs.SetInputConnection(critical_pairs.GetOutputPort())
  persistent_pairs.SetInputArrayToProcess(
    0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Persistence" # type: ignore
  )
  persistent_pairs.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN) # type: ignore
  persistent_pairs.SetLowerThreshold(persistence_threshold)
  
  simplification = ttk.ttkTopologicalSimplification() # type: ignore
  simplification.SetInputConnection(0, tetra.GetOutputPort())
  if field_name is None:
    simplification.SetInputArrayToProcess(
      0, 0, 0, 0, vtk.vtkDataSetAttributes.SCALARS # type: ignore
    )
  else:
    simplification.SetInputArrayToProcess(
      0, 0, 0, 0, field_name
    )
  simplification.SetInputConnection(1, persistent_pairs.GetOutputPort())
  
  complex = ttk.ttkMorseSmaleComplex() # type: ignore
  complex.SetInputConnection(simplification.GetOutputPort())
  if field_name is None:
    complex.SetInputArrayToProcess(
      0, 0, 0, 0, vtk.vtkDataSetAttributes.SCALARS # type: ignore
    )
  else:
    complex.SetInputArrayToProcess(
      0, 0, 0, 0, field_name
    )
    
  return complex

def MorseComplex(
  input: vtk.vtkAlgorithmOutput, # type: ignore
  persistence_threshold: float = 0,
  ascending: bool = True,
  field_name: Optional[str] = None,
) -> vtk.vtkAlgorithm: # type: ignore
  complex = MorseSmaleComplex(input, persistence_threshold, field_name)
  
  complex.SetComputeCriticalPoints(True)
  complex.SetComputeAscendingSeparatrices1(False)
  complex.SetComputeAscendingSeparatrices2(False)
  
  complex.SetComputeAscendingSegmentation(ascending)
  complex.SetComputeDescendingSegmentation(not ascending)
  complex.SetComputeDescendingSeparatrices1(ascending)
  complex.SetComputeDescendingSeparatrices2(not ascending)

  return complex