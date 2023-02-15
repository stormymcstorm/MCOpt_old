import os
import vtk
import numpy as np

from mcopt.util.data import (
  Sinusoidal, 
  Combine, 
  Distance, 
  GaussianNoise, 
  Smooth, 
  array_to_plane,
  Tetrahedralize,
  Warp
)
from mcopt.morse_complex import MorseSmaleComplex

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

if __name__ == '__main__':
  make_dir(DATA_DIR)

  rng = np.random.default_rng(42)

  terrain = Combine([
    Sinusoidal(),
    # Highest peak in the center
    -Distance(),
    Smooth(GaussianNoise(rng=rng) * 0.25),
  ])

  plane = array_to_plane(terrain)

  tetra = Tetrahedralize(plane)

  warp = Warp(tetra, 50)

  morse_smale_complex = MorseSmaleComplex(warp)

  save_data(morse_smale_complex.GetOutputPort(1), "separatrices", type="vtp")

  save_data(morse_smale_complex.GetOutputPort(3), "segmentation", type="vtu")
