import topologytoolkit as ttk
import vtk

def MorseSmaleComplex(
  input : vtk.vtkAlgorithm, 
  field_name='data', 
  persistence_threshold=0.05
) -> ttk.ttkMorseSmaleComplex :
  persistence_diagram = ttk.ttkPersistenceDiagram()
  persistence_diagram.SetInputConnection(input.GetOutputPort())
  persistence_diagram.SetInputArrayToProcess(0,0,0,0, field_name)

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
  simplification.SetInputConnection(0, input.GetOutputPort())
  simplification.SetInputArrayToProcess(0,0,0,0,field_name)
  simplification.SetInputConnection(1, persistent_pairs.GetOutputPort())

  mc = ttk.ttkMorseSmaleComplex()
  mc.SetInputConnection(simplification.GetOutputPort())
  mc.SetInputArrayToProcess(0,0,0,0, field_name)

  return mc

def MorseComplex(
  input : vtk.vtkAlgorithm, 
  field_name='data', 
  persistence_threshold=0.05,
  ascending=True,
):
  mc = MorseSmaleComplex(input, field_name=field_name, persistence_threshold=persistence_threshold)

  mc.SetComputeCriticalPoints(True)
  mc.SetComputeAscendingSeparatrices1(False)
  mc.SetComputeAscendingSeparatrices2(False)

  mc.SetComputeAscendingSegmentation(ascending)
  mc.SetComputeDescendingSegmentation(not ascending)
  mc.SetComputeDescendingSeparatrices1(ascending)
  mc.SetComputeDescendingSeparatrices2(not ascending)

  return mc