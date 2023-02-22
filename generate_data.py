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

if __name__ == '__main__':
  out_dir = os.path.join(os.getcwd(), DATA_DIR)
  
  rng = np.random.default_rng(42)

  data1 = Combine([
    Sinusoidal(npeaks=3),
    # Highest peak in the center
    -Distance(),
  ])

  data1 += Smooth(GaussianNoise(rng=rng) * 0.1)

  complex1 = gen_complex(data1, scale_factor=50, persistence_threshold=0.1)
  save_complex(complex1, os.path.join(out_dir, 'sinusoidal'))

  data2 = data1 + Smooth(GaussianNoise(rng=rng) * 0.2)
  
  complex2 = gen_complex(data2, scale_factor=50, persistence_threshold=0.1)
  save_complex(complex2, os.path.join(out_dir, 'sinusoidal_noisy'))
  