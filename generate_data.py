#!/usr/bin/env python

import os
import vtk
import numpy as np
import pandas as pd

from mcopt.util.data import (
  Sinusoidal, 
  Combine, 
  Distance, 
  GaussianNoise,
  Gaussian,
  Smooth,
)
from mcopt.util.vtk import (
  Tetrahedralize,
  Warp,
  Plane,
)
from mcopt.util.io import save_complex
from mcopt.morse_complex import (MorseSmaleComplex, MorseComplex)

DATA_DIR = 'data'

def gen_complex(data: np.ndarray, scale_factor=50, persistence_threshold=0.1):
  plane = Plane(data)
  tetra = Tetrahedralize(plane.GetOutputPort())
  
  warp = Warp(tetra.GetOutputPort(), scale_factor)

  return MorseComplex.create(warp.GetOutputPort(), persistence_threshold=persistence_threshold)

def gen_dataset(name, data: np.ndarray, scale_factor=50, persistence_threshold=0):
  complex = gen_complex(data, scale_factor, persistence_threshold)
  save_complex(complex, os.path.join(DATA_DIR, name))
  

if __name__ == '__main__':
  out_dir = os.path.join(os.getcwd(), DATA_DIR)
  
  rng = np.random.default_rng(42)

  # Sinusoidal Datasets

  sinusoidal_data = Combine([
    Sinusoidal(npeaks=3),
    # Highest peak in the center
    -Distance(),
  ])

  sinusoidal_data += Smooth(GaussianNoise(rng=rng) * 0.1)
  
  gen_dataset('sinusoidal', sinusoidal_data, scale_factor=50, persistence_threshold=0.1)
  
  sinusoidal_noisy_data1 = sinusoidal_data + Smooth(GaussianNoise(rng=rng) * 0.2)
  
  gen_dataset('sinusoidal_noisy', sinusoidal_noisy_data1, scale_factor=50, persistence_threshold=0.1)
  
  
  
  