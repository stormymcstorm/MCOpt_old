from importlib import import_module

import vtk


try:
  topologytoolkit = import_module('topology_toolkit')
except ImportError:
  topologytoolkit = None
  
globals()['ttk'] = topologytoolkit

def _make_MorseComplex(
  input: vtk.vtkAlgorithmOutput,
  persistence_threshold,
):
  if ttk is None:
    raise ImportError('topologytoolkit required for Morse Complex Computation')
  
  persistence_diagram = ttk.ttkPersistenceDiagram()
  persistence_diagram.SetInputConnection(input)
  persistence_diagram.SetInputArrayToProcess(0,0,0,0,vtk.vtkDataSetAttributes.SCALARS)