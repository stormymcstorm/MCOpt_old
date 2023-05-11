"""
Definitions for Metric Measure Networks
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
  'Space', 
  'Metric', 
  'Measure', 
  'MetricMeasureNetwork', 
  'MetricProbabilityNetwork',
  'Coupling'
]

Space = ArrayLike
Metric = ArrayLike
Measure = ArrayLike

class MetricMeasureNetwork:
  """
  A triple (V, p, W) where V is the space of nodes, p is a discrete measure on V,
  and W is a metric on V.
  
  Parameters
  ----------
  space: arraylike, shape (ns)
    The space of nodes
  measure: arraylike, shape (ns)
    The dirac measure for each nodes in the same order as `space`.
  metric: arraylike, shape (ns, ns)
    A metric cost matrix for each pair nodes in the same order as `space`.
  """
  
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
  """
  A `MetricMeasureNetwork` in which the measure of the space is `1`.
  
  Parameters
  ----------
  space: arraylike, shape (ns)
    The space of nodes
  measure: arraylike, shape (ns)
    The dirac probability measure for each nodes in the same order as `space`.
  metric: arraylike, shape (ns, ns)
    A metric cost matrix for each pair nodes in the same order as `space`.
  """
  
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
   
class Coupling:
  """
  A coupling matrix. Includes references to the source and destination spaces.
  
  Parameters
  ----------
  raw: arraylike, shape (ns, nr)
    The raw coupling matrix
  X: arraylike, shape (ns)
    The source space
  Y: arraylike, shape (nr)
    The destination space
  """
  
  def __init__(self, raw: np.ndarray, X: Space, Y: Space):
    self.raw = raw
    self.src_space = X 
    self.src_map = {i : n for i, n in enumerate(X)}
    self.src_rev_map = {n : i for i, n in enumerate(X)}
    self.dest_space = Y
    self.dest_map = {i : n for i, n in enumerate(X)}
    self.dest_rev_map = {n : i for i, n in enumerate(Y)}
    
  def __array__(self):
    return self.raw
    
  def reverse(self) -> Coupling:
    return Coupling(self.raw.T, self.dest_space, self.src_space)