
import numpy as np
from numpy.typing import ArrayLike

Space = ArrayLike
Metric = ArrayLike
Measure = ArrayLike

class MetricMeasureNetwork:
  space: np.ndarray
  measure: np.ndarray
  metric: np.ndarray
  
  def __init__(
    self,
    space: Space,
    measure: Measure,
    metric: Metric,
  ):
    self.space = np.asarray(space)
    assert self.space.ndim == 1
    
    self.measure = np.asarray(measure)
    assert self.measure.ndim == 1
    assert len(self.measure) == len(self.space)
    
    self.metric = np.asarray(metric)
    assert self.metric.ndim == 2
    assert self.metric.shape[0] == len(self.space)
    assert self.metric.shape[1] == len(self.space)
    
class MetricProbabilityNetwork(MetricMeasureNetwork):
  space: np.ndarray
  measure: np.ndarray
  metric: np.ndarray
  
  def __init__(
    self,
    space: Space,
    measure: Measure,
    metric: Metric
  ):
    super().__init__(space, measure, metric)
    
    assert np.isclose(self.measure.sum(), 1), "Measure must sum to one."
   
class Coupling(np.ndarray):  
  def __new__(cls, raw: np.ndarray, X: Space, Y: Space):
    obj = np.asarray(raw, dtype=float).view(cls)    
    obj.src_map = {i : n for i, n in enumerate(X)}
    obj.src_rev_map = {n : i for i, n in enumerate(X)}
    obj.dest_map = {i : n for i, n in enumerate(X)}
    obj.dest_rev_map = {n : i for i, n in enumerate(Y)}
    
    return obj
  
  def __array_finalize__(self, obj):
    if obj is None: return
    
    self.src_map = getattr(obj, 'src_map', None)
    self.src_map = getattr(obj, 'src_rev_map', None)
    self.dest_map = getattr(obj, 'dest_map', None)
    self.dest_map = getattr(obj, 'dest_rev_map', None)