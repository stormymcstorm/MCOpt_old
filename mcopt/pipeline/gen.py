"""
Utilities for dataset generation
"""

import numpy as np
from skimage import filters

def Sinusoidal(shape, npeaks=3) -> np.ndarray:
  assert(len(shape) == 2)
  
  def height(x, y):
    return 0.5 * (np.sin(np.pi * (2 * x * npeaks / shape[0] - 0.5)) 
                  + np.sin(np.pi * (2 * y * npeaks / shape[1] - 0.5)))
    
  return np.fromfunction(height, shape=shape, dtype=float)

def Distance(shape) -> np.ndarray:
  assert(len(shape) == 2)
  
  def height(x, y):
    return np.sqrt((x / shape[0] - 0.5) ** 2 + (y / shape[1] - 0.5) ** 2)
  
  return np.fromfunction(height, shape=shape, dtype=float)

def GaussianNoise(shape, random_state=None) -> np.ndarray:
  assert(len(shape) == 2)
  
  rng = np.random.default_rng(random_state)
  
  return rng.normal(size=shape)

def Smooth(arr : np.ndarray, sigma=1) -> np.ndarray:
  assert(arr.ndim == 2)
  
  return filters.gaussian(arr, sigma=sigma)

def Noise(shape, scale=1, **kwargs) -> np.ndarray:
  return Smooth(GaussianNoise(shape, **kwargs) * scale)

GEN_FUNCTIONS = {
  'sinusoidal': Sinusoidal,
  'distance': Distance,
  'noise': Noise
}