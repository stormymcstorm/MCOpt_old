"""
Target for couplings
"""

from __future__ import annotations
from typing import Dict, Callable, Tuple, TypedDict, TypeVar
import os
import pickle

import numpy as np
from mcopt import MetricMeasureNetwork, Coupling


from mcpipeline.entity import CacheableEntity
from mcpipeline.target import CacheableTarget, Rule
from mcpipeline.targets.mm_network import MMNetwork
from mcpipeline.util import ProgressFactory

__all__ = ['Couplings', 'CouplingsConf', 'CouplingsRule', 'CouplingsTarget']

class Couplings(CacheableEntity):
  couplings: np.ndarray
  distances: np.ndarray
  index_map: Dict[int, int]
  
  @staticmethod
  def load(cache_path: str, progress: ProgressFactory) -> Couplings:
    couplings = np.load(
      os.path.join(cache_path, 'couplings.npy'), 
      allow_pickle=True
    )
    
    distances = np.load(
      os.path.join(cache_path, 'distances.npy'), 
    )
    
    index_file_path = os.path.join(cache_path, 'index.pickle')
    with open(index_file_path, 'rb') as index_file:
      index_map = pickle.load(index_file)
    
    return Couplings(couplings, distances, index_map)
  
  def __init__(
    self,
    couplings: np.ndarray,
    distances: np.ndarray,
    index_map: Dict[int, int],
  ) -> None:
    super().__init__()
    
    self.couplings = couplings
    self.distances = distances
    self.index_map = index_map
    
  def save(self, cache_path: str, progress: ProgressFactory):
    np.save(
      file = os.path.join(cache_path, 'couplings.npy'),
      allow_pickle=True,
      arr = self.couplings
    )
    
    np.save(
      file = os.path.join(cache_path, 'distances.npy'),
      arr = self.distances
    )
    
    index_file_path = os.path.join(cache_path, 'index.pickle')
    with open(index_file_path, 'wb') as index_file:
      pickle.dump(self.index_map, index_file)

class CouplingsConf(TypedDict):
  num_random_iter: int | None
  random_state: int | None
  
Conf = TypeVar('Conf', bound=CouplingsConf)

class CouplingsRule(Rule[Conf, Couplings]):
  distance_only: bool
  
  def __init__(self, distance_only: bool = False) -> None:
    self.distance_only = distance_only
  
  def run(
    self,
    ot: Callable[[int, MetricMeasureNetwork, int, MetricMeasureNetwork], Tuple[Coupling, float]],
    type: str,
    network: MMNetwork,
    num_random_iter: int | None,
    random_state: int | None,
    progress: ProgressFactory,
  ) -> Couplings:
    shape=(len(network.frames), len(network.frames))
    
    couplings = np.full(
      shape=shape,
      fill_value=None,
      dtype=object
    )
    distances = np.zeros(shape=shape, dtype=float)
    index_map = {}
    
    rng = np.random.default_rng(random_state)
    
    with progress(
      total = len(network.frames) ** 2 // 2 * (1 if num_random_iter is None else num_random_iter),
      desc = f'running {type}'
    ) as prog:
      for i, (t, src) in enumerate(network.frames.items()):
        index_map[t] = i

        distances[i, i] = 0
        if not self.distance_only: couplings[i, i] = np.identity(len(src.space), dtype=float)
        prog.update()
        
        for (j, (s, dest)), _ in zip(enumerate(network.frames.items()), range(i)):
          if num_random_iter is not None:
            min_dist = float('inf')
            min_coupling = None
            
            for _ in range(num_random_iter):
              c, d = ot(t, src, s, dest, random_G0=True, random_state=rng)
              
              if d < min_dist:
                min_dist = d
                min_coupling = c
              
              prog.update() 
            
            if not self.distance_only: 
              couplings[i, j] = min_coupling
              couplings[j, i] = min_coupling.reverse()
              
            distances[i, j] = min_dist
            distances[j, i] = min_dist
          else:
            c, d = ot(t, src, s, dest, random_G0=False)
            
            if not self.distance_only: 
              couplings[i, j] = c
              couplings[j, i] = c.reverse()
              
            distances[i, j] = d
            distances[j, i] = d
            prog.update()
    
    return Couplings(couplings, distances, index_map)
      
class CouplingsTarget(CacheableTarget[Conf, Couplings]):
  @staticmethod
  def entity_type() -> type[Couplings]:
    return Couplings
  
  @staticmethod
  def target_type() -> str:
    return 'couplings'