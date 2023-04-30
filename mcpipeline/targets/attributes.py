"""
Target for Morse Graph attributes
"""

from __future__ import annotations
from typing import TypedDict, Dict
import os
import pickle

import numpy as np
from mcopt import MetricMeasureNetwork

from mcpipeline.entity import CacheableEntity
from mcpipeline.target import CacheableTarget, Rule
from mcpipeline.targets.graph import Graph, GraphTarget
from mcpipeline.util import ProgressFactory

from mcopt import MorseGraph

__all__ = ['Attributes', 'AttributesTarget']

class Attributes(CacheableEntity):
  @staticmethod
  def load(cache_path: str, progress: ProgressFactory) -> CacheableEntity:
    attrs = np.load(os.path.join(cache_path, 'attributes.npy'), allow_pickle=True)
    
    index_file_path = os.path.join(cache_path, 'index.pickle')
    with open(index_file_path, 'rb') as index_file:
      index_map = pickle.load(index_file)
    
    return Attributes(attrs, index_map)
  
  attrs: np.ndarray
  index_map: Dict[int, int]
  
  def __init__(self, attrs: np.ndarray, index_map: Dict[int, int]):
    self.attrs = attrs
    self.index_map = index_map
    
  def save(self, cache_path: str, progress: ProgressFactory):
    np.save(
      file = os.path.join(cache_path, 'attributes.npy'),
      allow_pickle=True,
      arr = self.attrs
    )
    
    index_file_path = os.path.join(cache_path, 'index.pickle')
    with open(index_file_path, 'wb') as index_file:
      pickle.dump(self.index_map, index_file)
    
    
class AttributesConf(TypedDict):
  normalize: bool

class AttributesRule(Rule[AttributesConf, Attributes]):
  def __call__(
    self, 
    graph: Graph, 
    normalize: bool,
    progress: ProgressFactory
  ) -> Attributes:
    attrs = np.full(shape = (len(graph.frames), len(graph.frames)), fill_value=None, dtype=object)
    
    index_map = {}
    
    with progress(
      total = len(graph.frames) ** 2 // 2,
      desc='constructing attributes',
    ) as prog:
      for i, (t, src) in enumerate(graph.frames.items()):
        index_map[t] = i
        
        for j, dest in enumerate(graph.frames.values()):
          if attrs[i, j] is not None:
            continue
          
          if i == j:
            attrs[i, j] = np.zeros(1)
            attrs[j, i] = np.zeros(1)
            prog.update()
            continue
          
          attrs[i, j] = MorseGraph.attribute_cost_matrix(src, dest)
          attrs[j, i] = attrs[i, j].T
          prog.update()
    
    if normalize:
      max = np.max([A.max() for A in attrs.ravel()])
      
      if max != 0:
        attrs /= max
    
    return Attributes(attrs, index_map)
  
class AttributesTarget(CacheableTarget[AttributesConf, Attributes]):
  @staticmethod
  def target_type() -> str:
    return 'attributes'
  
  @staticmethod
  def entity_type() -> type[Attributes]:
    return Attributes
  
  def __init__(
    self, 
    name: str,
    cache_path: str,
    graph: GraphTarget,
    normalize: bool = True,
    display_name: str | None = None,
    desc: str | None = None,
    **kwargs,
  ):
    if display_name is None:
      display_name = graph.display_name
    
    if desc is None:
      desc = graph.desc
    
    super().__init__(
      name = name,
      cache_path = cache_path,
      rule = AttributesRule(),
      conf = {
        'normalize': normalize,
      },
      depends = [graph],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )