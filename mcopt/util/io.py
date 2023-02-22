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

def save_complex(complex: MorseSmaleComplex, data_dir : str, suffix : str):
  _save_vtp(os.path.join(data_dir, f'critical_points{suffix}.vtp'), complex.critical_points)
  _save_vtp(os.path.join(data_dir, f'separatrices{suffix}.vtp'), complex.separatrices)
  _save_vtu(os.path.join(data_dir, f'segmentation{suffix}.vtu'), complex.segmentation)
  
  _save_csv(os.path.join(data_dir, f'critical_points_point_data{suffix}.csv'), 
            complex.critical_points_point_data)
  _save_csv(os.path.join(data_dir, f'separatrices_point_data{suffix}.csv'), 
            complex.separatrices_point_data)
  _save_csv(os.path.join(data_dir, f'separatrices_cell_data{suffix}.csv'), 
            complex.separatrices_cell_data)
  
def load_complex(data_dir : str, suffix : str) -> MorseSmaleComplex:
  crit = _load_vtp(os.path.join(data_dir, f'critical_points{suffix}.vtp'))
  sep = _load_vtp(os.path.join(data_dir, f'separatrices{suffix}.vtp'))
  seg = _load_vtu(os.path.join(data_dir, f'segmentation{suffix}.vtu'))

  return MorseSmaleComplex(
    crit.GetOutputPort(), 
    sep.GetOutputPort(), 
    seg.GetOutputPort(),
    _save = (crit, sep, seg)
  )