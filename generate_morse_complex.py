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

def make_dir(new_dir):
  if not os.path.exists(new_dir):
    os.makedirs(new_dir)

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

  complex1 = MorseComplex.create(terrain1.GetOutputPort(), persistence_threshold=0.1)
  save_complex(complex1, DATA_DIR, '1')

  data2 = data1 + Smooth(GaussianNoise(rng=rng) * 0.2)
  terrain2 = gen_terrain(data2)

  complex2 = MorseComplex.create(terrain2.GetOutputPort(), persistence_threshold=0.1)
  save_complex(complex2, DATA_DIR, '2')
  