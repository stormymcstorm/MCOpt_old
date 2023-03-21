
from typing import TypeVar, Callable

import numpy.typing as npt
import numpy as np

T = TypeVar('T')

Space = npt.ArrayLike
Metric = Callable[[T, T], float] | npt.ArrayLike
Measure = Callable[[T], float] | npt.ArrayLike

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