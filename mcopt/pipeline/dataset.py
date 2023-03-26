"""
Utilities for working with datasets
"""

from typing import List

import vtk

class Dataset:
  name: str
  frames: List[vtk.vtkAlgorithm]
  
  def __init__(self, name, frames):
    self.name = name
    self.frames = frames
  