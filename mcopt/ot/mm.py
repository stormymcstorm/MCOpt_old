
from typing import TypeVar, Callable

from numpy.typing import ArrayLike
import numpy as np

T = TypeVar('T')

Space = ArrayLike
Metric = ArrayLike
Measure = ArrayLike

class MetricMeasureSpace:
  space: np.ndarray
  metric: np.ndarray
  measure: np.ndarray
  
  def __init__(self, space : Space, metric : Metric, measure : Measure):
    self.space = np.asarray(space)
    assert space.ndim == 1
    
    self.metric = np.asarray(metric)
    
    assert self.metric.ndim == 2
    assert self.metric.shape[0] == len(self.space)
    assert self.metric.shape[1] == len(self.space)
    
    self.measure = np.asarray(measure)
    
    assert self.measure.ndim == 1
    assert len(self.measure) == len(self.space)

class MetricProbabilitySpace(MetricMeasureSpace):
  def __init__(self, space: Space, metric : Metric, measure : Measure):
    super().__init__(space, metric, measure)
    
    assert np.isclose(measure.sum(), 1), "Measure must sum to one"

MetricMeasureNetwork = MetricMeasureSpace
MetricProbabilityNetwork = MetricProbabilitySpace

class MetricMeasureHypernetwork:
  node_space: Space
  node_measure: Measure
  edge_space: Space
  edge_measure: Measure
  metric: Metric
  
  def __init__(
    self, 
    node_space: Space,
    node_measure: Measure,
    edge_space: Space,
    edge_measure: Measure,
    metric: Metric,
  ):
    self.node_space = np.asarray(node_space)
    assert node_space.ndim == 1
    
    self.node_measure = np.asarray(node_measure)
    
    assert self.node_measure.ndim == 1
    assert len(self.node_measure) == len(self.node_space)
    
    self.edge_space = np.asarray(edge_space)
    assert edge_space.ndim == 1
    
    self.edge_measure = np.asarray(edge_measure)
    
    assert self.edge_measure.ndim == 1
    assert len(self.edge_measure) == len(self.edge_measure)
    
    self.metric = np.asarray(metric)
    
    assert self.metric.ndim == 2
    assert self.metric.shape[0] == len(self.node_space)
    assert self.metric.shape[1] == len(self.edge_space)

  def to_bipartite(self) -> MetricMeasureNetwork:
    # space (n + m)
    space = np.append(self.node_space, self.edge_space)
    
    # metric (n + m, n + m)
    def omega(w, z):
      if w < len(self.node_space) and z >= len(self.node_space):
        x = w
        y = z - len(self.node_space)
      elif w >= len(self.node_space) and z < len(self.node_space):
        x = z
        y = w - len(self.node_space)
      else:
        return 0   
        
      return self.metric[x, y]
      
      pass
    
    metric = np.fromfunction(omega, (space.shape[0], space.shape[0]), dtype=float)
    
    def mu(w):
      mu1 = self.node_measure[w] if w < len(self.node_space) else 0
      mu2 = self.edge_space[w] if w >= len(self.node_space) else 0
      
      return 0.5 * (mu1 + mu2)

    measure = np.fromfunction(mu, len(space), dtype=float)
    
    return MetricMeasureNetwork(space, metric, measure)

class MetricProbabilityHypernetwork(MetricMeasureHypernetwork):
  node_space: Space
  node_measure: Measure
  edge_space: Space
  edge_measure: Measure
  metric: Metric
  
  def __init__(
    self, 
    node_space: Space,
    node_measure: Measure,
    edge_space: Space,
    edge_measure: Measure,
    metric: Metric,             
  ):
    super().__init__(node_space, node_measure, edge_space, edge_measure, metric)
    
    assert np.isclose(node_measure.sum(), 1), "node_measure must sum to one"
    assert np.isclose(edge_measure.sum(), 1), "edge_measure must sum to one"

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