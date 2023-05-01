"""
Target for Morse Graphs
"""

from __future__ import annotations
from typing import TypedDict, Dict
import os
import json
import pickle

import numpy as np
from mcopt import MorseGraph

from mcpipeline.entity import CacheableEntity
from mcpipeline.target import CacheableTarget, Rule
from mcpipeline.targets.complex import Complex, ComplexTarget
from mcpipeline.util import ProgressFactory

__all__ = ['Graph', 'GraphTarget']

class Graph(CacheableEntity):
  @staticmethod
  def load(cache_path: str, progress: ProgressFactory) -> CacheableEntity:
    cache_file_path = os.path.join(cache_path, 'graph.json')
    
    with open(cache_file_path, 'r') as cache_file:
      contents = json.load(cache_file)

    if 'name' not in contents or 'file_paths' not in contents:
      raise RuntimeError(f'Invalid graph cache {cache_file_path}')
    
    name = contents['name']
    file_paths = contents['file_paths']
    
    frames = {}
    
    for t, file_path in progress(file_paths.items(), desc='reading graphs', unit='Graphs'):
      with open(file_path, 'rb') as file:
        frames[int(t)] = pickle.load(file)
      
    return Graph(name, frames)
  
  name: str
  frames: Dict[int, MorseGraph] 
  
  def __init__(self, name: str, frames: Dict[int, MorseGraph]) -> None:
    self.name = name
    self.frames = frames
    
  def save(self, cache_path: str, progress: ProgressFactory):
    file_paths = {}
    
    for t, frame in progress(
      self.frames.items(), desc='writing graphs', unit='Graphs'
    ):
      file_path = os.path.join(cache_path, f'{self.name}{t}.pickle')
      
      file_paths[t] = file_path
      
      with open(file_path, 'wb') as file:
        pickle.dump(frame, file)
    
    cache_file_path = os.path.join(cache_path, 'graph.json')
    
    with open(cache_file_path, 'w') as cache_file:
      contents = {
        'name': self.name,
        'file_paths': file_paths
      }
      
      json.dump(contents, cache_file)

class GraphConf(TypedDict):
  sample_rate: int | None
  
class GraphRule(Rule[GraphConf, Graph]):
  name: str
  
  def __init__(self, name: str):
    super().__init__()
    
    self.name = name
  
  def __call__(
    self, 
    complex: Complex,
    sample_rate: int | None,
    progress: ProgressFactory
  ) -> Graph:
    frames = {}
    
    for t, complex_frame in progress(
      complex.frames.items(), desc='constructing graphs', unit='Graphs'
    ):
      frames[t] = complex_frame.to_graph()
      
      if sample_rate is not None:
        frames[t] = frames[t].sample(sample_rate)
        
    return Graph(self.name, frames)
  
class GraphTarget(CacheableTarget[GraphConf, Graph]):
  @staticmethod
  def target_type() -> str:
    return 'graph'
  
  @staticmethod
  def entity_type() -> type[Graph]:
    return Graph
  
  def __init__(
    self, 
    name: str,
    cache_path: str,
    complex: ComplexTarget,
    sample_rate: int | None = None,
    display_name: str | None = None,
    desc: str | None = None,
    **kwargs,
  ):
    if display_name is None:
      display_name = complex.display_name
    
    if desc is None:
      desc = complex.desc
    
    super().__init__(
      name = name,
      cache_path = cache_path,
      rule = GraphRule(name),
      conf = {
        'sample_rate': sample_rate
      },
      depends = [complex],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )

