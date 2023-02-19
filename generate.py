import os
import vtk
import numpy as np
import pandas as pd

from mcopt.util.data import (
  Sinusoidal, 
  Combine, 
  Distance, 
  GaussianNoise,
  Smooth,
)
from mcopt.util.vtk import (
  Tetrahedralize,
  Warp,
  Plane,
  get_cell_dataframe,
)
from mcopt.morse_complex import (MorseSmaleComplex, MorseComplex)

DATA_DIR = 'data'

def make_dir(new_dir):
  if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def save_data(output, name, type='vtu'):
  if type == 'vtu':
    writer = vtk.vtkXMLUnstructuredGridWriter()
  elif type == 'vtp':
    writer = vtk.vtkXMLPolyDataWriter()
  else:
    raise "Unknown data type: " + type

  writer.SetInputConnection(output)
  writer.SetFileName(os.path.join(DATA_DIR, name + "." + type))
  writer.Write()

def gen_terrain(data: np.ndarray, scale_factor=50):
  plane = Plane(data)
  tetra = Tetrahedralize(plane.GetOutputPort())

  return Warp(tetra.GetOutputPort(), scale_factor)

if __name__ == '__main__':
  make_dir(DATA_DIR)

  rng = np.random.default_rng(42)

  data1 = Combine([
    Sinusoidal(npeaks=3),
    # Highest peak in the center
    -Distance(),
  ])

  data1 += Smooth(GaussianNoise(rng=rng) * 0.1)
  terrain1 = gen_terrain(data1)
  save_data(terrain1.GetOutputPort(), "terrain1", type="vtu")

  complex1 = MorseComplex(terrain1, persistence_threshold=0.1)
  save_data(complex1.GetOutputPort(0), "critical_points1", type="vtp")
  save_data(complex1.GetOutputPort(1), "separatrices1", type="vtp")
  save_data(complex1.GetOutputPort(3), "segmentation1", type="vtu")
  
  morse_graph_data1 = get_cell_dataframe(complex1.GetOutput(1))
  morse_graph_data1.to_csv('data/morse_graph1.csv')

  data2 = data1 + Smooth(GaussianNoise(rng=rng) * 0.2)
  terrain2 = gen_terrain(data2)
  save_data(terrain2.GetOutputPort(), "terrain2", type="vtu")

  complex2 = MorseComplex(terrain2, persistence_threshold=0.1)
  save_data(complex2.GetOutputPort(0), "critical_points2", type="vtp")
  save_data(complex2.GetOutputPort(1), "separatrices2", type="vtp")
  save_data(complex2.GetOutputPort(3), "segmentation2", type="vtu")
  
  morse_graph_data2 = get_cell_dataframe(complex2.GetOutput(1))
  morse_graph_data2.to_csv('data/morse_graph2.csv')

