"""
Experiment for MDS
"""

from __future__ import annotations
from typing import Dict, List, TypedDict
import os
import pickle
import json

import numpy as np
from sklearn.manifold import MDS

from mcpipeline.entity import CacheableEntity
from mcpipeline.target import CacheableTarget, Rule
from mcpipeline.targets.combine import CombinedGraphs, CombineGraphsTarget
from mcpipeline.targets.couplings import Couplings, CouplingsTarget
from mcpipeline.util import ProgressFactory

__all__ = ['MDSTransforms', 'MDSTarget']

class MDSTransforms(CacheableEntity):
  @staticmethod
  def load(cache_path: str, progress: ProgressFactory) -> CacheableEntity:
    cache_file_path = os.path.join(cache_path, 'mds.json')
    
    with open(cache_file_path, 'r') as cache_file:
      contents = json.load(cache_file)

    if 'n_classes' not in contents:
      raise RuntimeError(f'Invalid mds cache {cache_file_path}')
    
    n_classes = contents['n_classes']
    
    transform = np.load(
      file = os.path.join(cache_path, 'transform.npy')
    )
    
    class_transforms = []
    index_maps = []
    
    for i in range(n_classes):
      class_transforms.append(np.load(
        file = os.path.join(cache_path, f'class_transform{i}.npy')
      ))
      
      index_file_path = os.path.join(cache_path, f'index{i}.pickle')
      with open(index_file_path, 'rb') as index_file:
        index_maps.append(pickle.load(index_file))
    
    return MDSTransforms(transform, class_transforms, index_maps)
  
  transform: np.ndarray
  class_transforms: List[np.ndarray]
  index_maps: List[Dict[int, int]]
  
  def __init__(
    self, 
    transform: np.ndarray,
    class_transforms: List[np.ndarray], 
    index_maps: List[Dict[int, int]]
  ):
    super().__init__()
    
    self.transform = transform
    self.class_transforms = class_transforms
    self.index_maps = index_maps
    
  def save(self, cache_path: str, progress: ProgressFactory):
    cache_file_path = os.path.join(cache_path, 'mds.json')
    
    with open(cache_file_path, 'w') as cache_file:
      contents = {
        'n_classes': len(self.class_transforms)
      }
      
      json.dump(contents, cache_file)
    
    np.save(
      file=os.path.join(cache_path, 'transform.npy'),
      arr=self.transform
    )
    
    for i, trans in enumerate(self.class_transforms):
      np.save(
        file=os.path.join(cache_path, f'class_transform{i}.npy'),
        arr=trans
      )
      
      index_file_path = os.path.join(cache_path, f'index{i}.pickle')
      with open(index_file_path, 'wb') as index_file:
        pickle.dump(self.index_maps[i], index_file)

class MDSConf(TypedDict):
  random_state: int | None
  n_init: int

class MDSRule(Rule[MDSConf, MDSTransforms]):
  def __call__(
    self, 
    graphs: CombinedGraphs,
    couplings: Couplings,
    random_state: int | None,
    n_init: int,
    progress: ProgressFactory, 
  ) -> Couplings:
    mds = MDS(random_state=random_state, n_init=n_init, dissimilarity='precomputed')
    
    transform = mds.fit_transform(couplings.distances)
    
    class_transforms = []
    index_maps = []
    for o_i, graph in enumerate(graphs.originals):
      trans = np.zeros(shape = (len(graph.frames), 2))
      
      index_map = {}
      for t_i, (t, i) in enumerate(graphs.index_map[o_i].items()):
        trans[t_i] = transform[i]
        index_map[t] = t_i
      
      class_transforms.append(trans)
      index_maps.append(index_map)

    return MDSTransforms(transform, class_transforms, index_maps)
  
class MDSTarget(CacheableTarget[MDSConf, MDSTransforms]):
  @staticmethod
  def target_type() -> str:
    return 'mds'
  
  @staticmethod
  def entity_type() -> type[MDSTransforms]:
    return MDSTransforms
  
  def __init__(
    self, 
    name: str,
    cache_path: str,
    graphs: CombinedGraphs,
    couplings: CouplingsTarget,
    n_init: int = 10,
    random_state: int | None = None,
    display_name: str | None = None,
    desc: str | None = None,
    **kwargs,
  ):
    super().__init__(
      name = name,
      cache_path = cache_path,
      rule = MDSRule(),
      conf = {
        'n_init': n_init,
        'random_state': random_state
      },
      depends = [graphs, couplings],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )