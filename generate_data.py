#!/usr/bin/env python

import os
import vtk
import numpy as np
import pandas as pd

from mcopt.util.data import (
  Sinusoidal, 
  Combine, 
  Distance, 
  Gaussian,
  GaussianNoise,
  Smooth,
)
from mcopt.util.vtk import (
  Tetrahedralize,
  Warp,
  Plane,
)
from mcopt.util.io import save_complex
from mcopt.morse_complex import (MorseSmaleComplex, MorseComplex)

DATA_DIR = 'gen_data'

def gen_dataset(name, data: np.ndarray, scale_factor=50, persistence_threshold=0):
  plane = Plane(data)
  tetra = Tetrahedralize(plane.GetOutputPort())
  
  warp = Warp(tetra.GetOutputPort(), scale_factor)
  
  complex = MorseComplex.create(warp.GetOutputPort(), persistence_threshold=persistence_threshold)
  save_complex(complex, os.path.join(DATA_DIR, name))
  
  return complex
  

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
  
  # Complex Datasets
  
  complex_data = sum([
    Gaussian((17     ,      17), shape=(102, 102), sigma=9)* 0.5,
    Gaussian((17     , 17 + 34), shape=(102, 102), sigma=9),
    Gaussian((17     , 17 + 68), shape=(102, 102), sigma=15),
    Gaussian((17 + 34,      17), shape=(102, 102), sigma=9),
    Gaussian((17 + 34, 17 + 34), shape=(102, 102), sigma=12) * -0.75,
    Gaussian((17 + 34, 17 + 68), shape=(102, 102), sigma=6),
    Gaussian((17 + 68,      17), shape=(102, 102), sigma=9) * 0.75,
    Gaussian((17 + 68, 17 + 34), shape=(102, 102), sigma=3) * 0.5,
    Gaussian((17 + 68, 17 + 68), shape=(102, 102), sigma=15) * -0.5,
  ])
  
  complex_data += Gaussian((17 + 68 + 8, 17 + 34 + 8), shape=(102, 102), sigma=3) * 0.4
  complex_data += Gaussian((17 + 68 + 8, 17 + 34 - 8), shape=(102, 102), sigma=3) * 0.6
  
  complex_data += Smooth(GaussianNoise(rng=rng, shape=(102, 102)) * 0.1)
    
  gen_dataset('complex', complex_data, scale_factor=50, persistence_threshold=0.2)
  
  complex_noisy_data1 = complex_data + Smooth(GaussianNoise(rng=rng, shape=(102, 102)) * 0.2)
  
  gen_dataset('complex_noisy', complex_noisy_data1, scale_factor=50, persistence_threshold=0.2)
  
  