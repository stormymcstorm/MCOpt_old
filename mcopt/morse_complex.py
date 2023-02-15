import topologytoolkit as ttk
import vtk

def MorseSmaleComplex(
  input : vtk.vtkAlgorithm, 
  fieldname='data', 
  persistence_threshold=0.05
):
  persistence_diagram = ttk.ttkPersistenceDiagram()
  persistence_diagram.SetInputConnection(input.GetOutputPort())
  persistence_diagram.SetInputArrayToProcess(0,0,0,0, fieldname)

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
    0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "PairIdentifier"
  )
  persistent_pairs.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
  persistent_pairs.SetLowerThreshold(persistence_threshold)

  simplification = ttk.ttkTopologicalSimplification()
  simplification.SetInputConnection(0, input.GetOutputPort())
  simplification.SetInputArrayToProcess(0,0,0,0,fieldname)
  simplification.SetInputConnection(1, persistent_pairs.GetOutputPort())

  mc = ttk.ttkMorseSmaleComplex()
  mc.SetInputConnection(simplification.GetOutputPort())
  mc.SetInputArrayToProcess(0,0,0,0, fieldname)

  return mc

