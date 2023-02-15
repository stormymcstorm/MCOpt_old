from typing import List
import numpy as np
from skimage import filters
import vtk
from vtk.util.numpy_support import numpy_to_vtk

def array_to_plane(arr: np.ndarray, field_name="data") -> vtk.vtkPlaneSource :
  assert(arr.ndim == 2)

  plane = vtk.vtkPlaneSource()
  plane.SetResolution(arr.shape[0] - 1, arr.shape[1] - 1)
  plane.SetOrigin([0, 0, 0])
  plane.SetPoint1([arr.shape[0], 0, 0])
  plane.SetPoint2([0, arr.shape[0], 0])
  plane.Update()

  scalars = numpy_to_vtk(arr.ravel(), deep = True, array_type=vtk.VTK_DOUBLE)

  scalars.SetName(field_name)

  plane.GetOutput().GetPointData().SetScalars(scalars)

  return plane

# TODO: better docs

def Sinusoidal(shape = (100, 100), npeaks=3) -> np.ndarray:
  """
  Generates terrain with a sinusoidal pattern
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

  return filters.gaussian(arr, sigma=sigma)
  
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

def Tetrahedralize(input: vtk.vtkAlgorithm, field_name="data"):
  tetra = vtk.vtkDataSetTriangleFilter()
  tetra.SetInputConnection(input.GetOutputPort())
  tetra.SetInputArrayToProcess(0,0,0,0,field_name)

  return tetra

def Warp(input: vtk.vtkAlgorithm, scale_factor, field_name="data"):
  warp = vtk.vtkWarpScalar()
  warp.SetInputConnection(input.GetOutputPort())
  warp.SetInputArrayToProcess(0,0,0,0,field_name)
  warp.SetScaleFactor(scale_factor)

  return warp
