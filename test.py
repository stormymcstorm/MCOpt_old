import vtk
import topologytoolkit as ttk

if __name__ == '__main__':
  reader = vtk.vtkXMLImageDataReader()
  reader.SetFileName('datasets/wind/wind1.vti')
  reader.Update()
  
  tetra = vtk.vtkDataSetTriangleFilter()
  tetra.SetInputConnection(0, reader.GetOutputPort())
  
  warp = vtk.vtkWarpScalar()
  warp.SetInputConnection(tetra.GetOutputPort())
  warp.SetScaleFactor(1)
  
  persistence_diagram = ttk.ttkPersistenceDiagram()
  persistence_diagram.SetInputConnection(warp.GetOutputPort())
  persistence_diagram.SetInputArrayToProcess(0,0,0,0,vtk.vtkDataSetAttributes.SCALARS)
  
  