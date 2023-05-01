"""

"""

from typing import Iterable
import os 

import numpy as np

from mcpipeline import Pipeline, GenDatasetTarget
from mcpipeline import vtk
from mcpipeline import gen

__all__ = ['pipeline']

pipeline = Pipeline(
  os.path.join(os.path.dirname(__file__), '__pipeline_cache__')
)
class Sinusoidal(GenDatasetTarget):
  def generate(self) -> Iterable[np.ndarray]:
    shape = (100, 100)
    
    initial = gen.Sinusoidal(shape=shape, npeaks=3) * 0.5
    initial += gen.Distance(shape=shape) * -0.5
    
    frame0 = initial + gen.Noise(shape=shape, scale=0.1, random_state=42)
    
    yield frame0
    
    frame1 = initial + gen.Noise(shape=shape, scale=0.25, random_state=42)
    
    yield frame1
    
sinusoidal_dataset = pipeline.add_gen_dataset(
  name = 'sinusoidal',
  cls = Sinusoidal,
  filters = [
    vtk.WarpFilter(scale_factor=50)
  ]
)

sinusoidal_complex = pipeline.add_complex(
  name = 'sinusoidal',
  dataset = sinusoidal_dataset,
  persistence_threshold = 0.1
)