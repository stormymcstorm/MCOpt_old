"""
Utilities for dataset generation
"""

from typing import Sequence, Any

import numpy as np
from skimage import filters

def Sinusoidal(shape : Sequence[int], npeaks: int = 3) -> np.ndarray:
  assert(len(shape) == 2)
  
  def height(x, y):
    return 0.5 * (np.sin(np.pi * (2 * x * npeaks / shape[0] - 0.5)) 
                  + np.sin(np.pi * (2 * y * npeaks / shape[1] - 0.5)))
    
  return np.fromfunction(height, shape=shape, dtype=float)

def Distance(shape: Sequence[int]) -> np.ndarray:
  assert(len(shape) == 2)
  
  def height(x, y):
    return np.sqrt((x / shape[0] - 0.5) ** 2 + (y / shape[1] - 0.5) ** 2)
  
  return np.fromfunction(height, shape=shape, dtype=float)

def Normal(shape: Sequence[int], center : Sequence[int], sigma = 1) -> np.ndarray:
  assert(len(shape) == 2)
  assert(len(center) == 2)
  
  if not hasattr(sigma, "__len__"):
    sigma = [sigma, sigma]
    
  assert(len(sigma) == 2)
  
  sigma_x, sigma_y = sigma
  
  def gauss(x, y):
    return 1/(2 * np.pi * sigma_x * sigma_y) * np.exp(-(x - center[0])**2 / (2 * sigma_x ** 2) -(y - center[1])**2 / (2 * sigma_y ** 2))
  
  return np.fromfunction(gauss, shape=shape)

def GaussianNoise(shape: Sequence[int], random_state: Any = None) -> np.ndarray:
  assert(len(shape) == 2)
  
  rng = np.random.default_rng(random_state)
  
  return rng.normal(size=shape)

def Smooth(arr : np.ndarray, sigma: int = 1) -> np.ndarray:
  assert(arr.ndim == 2)
  
  return filters.gaussian(arr, sigma=sigma)

def Noise(shape: Sequence[int], scale:float = 1, **kwargs) -> np.ndarray:
  return Smooth(GaussianNoise(shape, **kwargs) * scale)

GEN_FUNCTIONS = {
  'sinusoidal': Sinusoidal,
  'distance': Distance,
  'normal': Normal,
  'noise': Noise
}