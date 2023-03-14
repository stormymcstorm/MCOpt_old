import vtk
import pandas as pd
import os
from ..morse_complex import MorseSmaleComplex

def save_vtu(filename : str, output : vtk.vtkAlgorithmOutput):
  writer = vtk.vtkXMLUnstructuredGridWriter()
  writer.SetInputConnection(output)
  writer.SetFileName(filename)
  writer.Write()
  
def load_vtu(filename : str) -> vtk.vtkAlgorithm:
  reader = vtk.vtkXMLUnstructuredGridReader()
  reader.SetFileName(filename)
  reader.Update()
  return reader
  
def save_vtp(filename : str, output : vtk.vtkAlgorithmOutput):
  writer = vtk.vtkXMLPolyDataWriter()
  writer.SetInputConnection(output)
  writer.SetFileName(filename)
  writer.Write()
  
def load_vtp(filename : str) -> vtk.vtkAlgorithm:
  reader = vtk.vtkXMLPolyDataReader()
  reader.SetFileName(filename)
  reader.Update()
  return reader

def load_vti(filename : str) -> vtk.vtkAlgorithm:
  reader = vtk.vtkXMLImageDataReader()
  reader.SetFileName(filename)
  reader.Update()
  return reader
  
def save_csv(filename : str, output : pd.DataFrame):
  output.to_csv(filename)

def save_complex(complex: MorseSmaleComplex, path : str):
  os.makedirs(path, exist_ok=True)
  
  save_vtp(os.path.join(path, f'critical_points.vtp'), complex.critical_points)
  save_vtp(os.path.join(path, f'separatrices.vtp'), complex.separatrices)
  save_vtu(os.path.join(path, f'segmentation.vtu'), complex.segmentation)
  
  save_csv(os.path.join(path, f'critical_points_point_data.csv'), 
            complex.critical_points_point_data)
  save_csv(os.path.join(path, f'separatrices_point_data.csv'), 
            complex.separatrices_point_data)
  save_csv(os.path.join(path, f'separatrices_cell_data.csv'), 
            complex.separatrices_cell_data)
  
def load_complex(path : str) -> MorseSmaleComplex:
  crit = load_vtp(os.path.join(path, f'critical_points.vtp'))
  sep = load_vtp(os.path.join(path, f'separatrices.vtp'))
  seg = load_vtu(os.path.join(path, f'segmentation.vtu'))

  return MorseSmaleComplex(
    crit.GetOutputPort(), 
    sep.GetOutputPort(), 
    seg.GetOutputPort(),
    _save = (crit, sep, seg)
  )