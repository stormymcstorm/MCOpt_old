import vtk
import pandas as pd
import os
from ..morse_complex import MorseSmaleComplex

def _save_vtu(filename : str, output : vtk.vtkAlgorithmOutput):
  writer = vtk.vtkXMLUnstructuredGridWriter()
  writer.SetInputConnection(output)
  writer.SetFileName(filename)
  writer.Write()
  
def _load_vtu(filename : str) -> vtk.vtkAlgorithm:
  reader = vtk.vtkXMLUnstructuredGridReader()
  reader.SetFileName(filename)
  reader.Update()
  return reader
  
def _save_vtp(filename : str, output : vtk.vtkAlgorithmOutput):
  writer = vtk.vtkXMLPolyDataWriter()
  writer.SetInputConnection(output)
  writer.SetFileName(filename)
  writer.Write()
  
def _load_vtp(filename : str) -> vtk.vtkAlgorithm:
  reader = vtk.vtkXMLPolyDataReader()
  reader.SetFileName(filename)
  reader.Update()
  return reader
  
def _save_csv(filename : str, output : pd.DataFrame):
  output.to_csv(filename)

def save_complex(complex: MorseSmaleComplex, path : str):
  os.makedirs(path, exist_ok=True)
  
  _save_vtp(os.path.join(path, f'critical_points.vtp'), complex.critical_points)
  _save_vtp(os.path.join(path, f'separatrices.vtp'), complex.separatrices)
  _save_vtu(os.path.join(path, f'segmentation.vtu'), complex.segmentation)
  
  _save_csv(os.path.join(path, f'critical_points_point_data.csv'), 
            complex.critical_points_point_data)
  _save_csv(os.path.join(path, f'separatrices_point_data.csv'), 
            complex.separatrices_point_data)
  _save_csv(os.path.join(path, f'separatrices_cell_data.csv'), 
            complex.separatrices_cell_data)
  
def load_complex(path : str) -> MorseSmaleComplex:
  crit = _load_vtp(os.path.join(path, f'critical_points.vtp'))
  sep = _load_vtp(os.path.join(path, f'separatrices.vtp'))
  seg = _load_vtu(os.path.join(path, f'segmentation.vtu'))

  return MorseSmaleComplex(
    crit.GetOutputPort(), 
    sep.GetOutputPort(), 
    seg.GetOutputPort(),
    _save = (crit, sep, seg)
  )