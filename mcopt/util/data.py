"""
Utilities for generating datasets
"""

from typing import List
import numpy as np
from skimage import filters

# TODO: better docs

def Sinusoidal(shape = (100, 100), npeaks=3) -> np.ndarray:
  """Generates terrain with a sinusodial pattern

  Args:
    shape (tuple, optional): A 2d array describing width and height of the terrain. Defaults to (100, 100).
    npeaks (int, optional): The number of peaks. Defaults to 3.

  Returns:
    np.ndarray: The generated dataset
  """
  assert (len(shape) == 2)

  def height(x,y):
    return 0.5 * (np.sin(np.pi * (2 * x * npeaks / shape[0] - 0.5)) 
                  + np.sin(np.pi * (2 * y * npeaks / shape[1] - 0.5)))

  return np.fromfunction(height, shape=shape, dtype=float)

def Distance(shape = (100, 100)) -> np.ndarray:
  assert (len(shape) == 2)

  def height(x,y):
    return np.sqrt((x / shape[0] - 0.5)**2 + (y / shape[1] - 0.5)**2)

  return np.fromfunction(height, shape=shape, dtype=float)


def Gaussian(center, shape = (100, 100), sigma=1) -> np.ndarray:
  assert (len(center) == 2)
  assert (len(shape) == 2)

  arr = np.zeros(shape=shape, dtype=float)

  arr[center] = 1
  
  arr = filters.gaussian(arr, sigma=sigma)

  return arr * 1/arr.max()

def GaussianNoise(shape = (100, 100), rng = None) -> np.ndarray:
  assert (len(shape) == 2)

  if rng is None:
    rng = np.random.default_rng()
  
  return rng.normal(size=shape)

def Smooth(arr : np.ndarray, sigma=1) -> np.ndarray:
  assert (arr.ndim == 2)

  return filters.gaussian(arr, sigma=sigma)

def Combine(layers : List[np.ndarray], weights = None) -> np.ndarray:
  if weights is None:
    weights = np.ones(len(layers), dtype=float) / len(layers)

  return np.array(layers).T.dot(weights).T
