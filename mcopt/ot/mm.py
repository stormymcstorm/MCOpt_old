
from typing import TypeVar, Callable

from numpy.typing import ArrayLike
import numpy as np

T = TypeVar('T')

Space = ArrayLike
Metric = Callable[[T, T], float] | ArrayLike
Measure = Callable[[T], float] | ArrayLike

class MetricMeasureSpace:
  space: np.ndarray
  metric: np.ndarray
  measure: np.ndarray
  
  def __init__(self, space : Space, metric : Metric, measure : Measure):
    self.space = np.asarray(space)
    assert space.ndim == 1
    
    if callable(metric):
      def f(i, j):
        return metric(self.space[i], self.space[j])
      
      self.metric = np.fromfunction(f, shape=(len(self.space), len(self.space)), dtype=float)
    else:
      self.metric = np.asarray(metric)
      
      assert self.metric.ndim == 2
      assert self.metric.shape[0] == len(self.space)
      assert self.metric.shape[1] == len(self.space)
    
    if callable(measure):
      def f(i):
        return measure(self.space[i])
      
      self.measure = np.fromfunction(f, shape=len(self.space), dtype=float)
    else:
      self.measure = np.asarray(measure)
      
      assert self.measure.ndim == 1
      assert len(self.measure) == len(self.space)

class MetricProbabilitySpace(MetricMeasureSpace):
  def __init__(self, space: Space, metric : Metric, measure : Measure):
    super().__init__(space, metric, measure)
    
    assert np.isclose(measure.sum(), 1), "Measure must sum to one"
    
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