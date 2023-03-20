"""
Utilities for computations with TTK
"""

from importlib import import_module

import vtk

try:
  ttk = import_module('topologytoolkit')
except ImportError:
  ttk = None
  
def MorseSmaleComplex(
  input: vtk.vtkAlgorithmOutput,
  persistence_threshold : float = 0
) -> vtk.vtkAlgorithm:
  if ttk is None:
    raise ImportError('topologytoolkit required for MorseComplex Computation')
  
  tetra = vtk.vtkDataSetTriangleFilter()
  tetra.SetInputConnection(0, input)
  
  persistence_diagram = ttk.ttkPersistenceDiagram()
  persistence_diagram.SetInputConnection(tetra.GetOutputPort())
  persistence_diagram.SetInputArrayToProcess(
    0, 0, 0, 0, vtk.vtkDataSetAttributes.SCALARS
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
  simplification.SetInputArrayToProcess(
    0, 0, 0, 0, vtk.vtkDataSetAttributes.SCALARS
  )
  simplification.SetInputConnection(1, persistent_pairs.GetOutputPort())
  
  complex = ttk.ttkMorseSmaleComplex()
  complex.SetInputConnection(simplification.GetOutputPort())
  complex.SetInputArrayToProcess(
    0, 0, 0, 0, vtk.vtkDataSetAttributes.SCALARS
  )
  
  return complex