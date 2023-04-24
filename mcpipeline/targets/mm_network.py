"""
Target for Metric Measure Networks
"""

from __future__ import annotations
from typing import TypedDict, Dict
import os
import json
import pickle

import numpy as np
from mcopt import MetricMeasureNetwork

from mcpipeline.entity import CacheableEntity
from mcpipeline.target import CacheableTarget, Rule
from mcpipeline.targets.graph import Graph, GraphTarget
from mcpipeline.util import ProgressFactory

__all__ = ['MMNetwork', 'MMNetworkTarget']

class MMNetwork(CacheableEntity):
  
  @staticmethod
  def load(cache_path: str, progress: ProgressFactory) -> CacheableEntity:
    cache_file_path = os.path.join(cache_path, 'mmnetwork.json')
    
    with open(cache_file_path, 'r') as cache_file:
      contents = json.load(cache_file)

    if 'time_steps' not in contents:
      raise RuntimeError(f'Invalid mmnetwork cache {cache_file_path}')
    
    time_steps = contents['time_steps']
    
    frames = {}
    
    for t in progress(
      time_steps, desc='reading networks', unit='Networks'
    ):
      space = np.load(os.path.join(cache_path, f'space{t}.npy'))
      measure = np.load(os.path.join(cache_path, f'measure{t}.npy'))
      metric = np.load(os.path.join(cache_path, f'metric{t}.npy'))
      
      frames[int(t)] = MetricMeasureNetwork(space, measure, metric)
      
    return MMNetwork(frames)
  
  frames: Dict[int, MetricMeasureNetwork]
  def __init__(self, frames: Dict[int, MetricMeasureNetwork]) -> None:
    self.frames = frames
    
  def save(self, cache_path: str, progress: ProgressFactory):
    cache_file_path = os.path.join(cache_path, 'mmnetwork.json')
    
    with open(cache_file_path, 'w') as cache_file:
      contents = {
        'time_steps': list(self.frames.keys())
      }
      
      json.dump(contents, cache_file)
      
    for t, frame in progress(
      self.frames.items(), desc='writing networks', unit='Networks',
    ):
      np.save(
        file=os.path.join(cache_path, f'space{t}.npy'),
        arr=frame.space
      )
      
      np.save(
        file=os.path.join(cache_path, f'measure{t}.npy'),
        arr=frame.measure
      )
      
      np.save(
        file=os.path.join(cache_path, f'metric{t}.npy'),
        arr=frame.metric
      )
      
class MMNetworkConf(TypedDict):
  dist: str
  hist: str
  normalize: bool
  
class MMNetworkRule(Rule[MMNetworkConf, MMNetwork]):
  def __call__(
    self, 
    graph: Graph,
    dist: str,
    hist: str,
    normalize: bool,
    progress: ProgressFactory,
  ) -> MMNetwork:
    frames = {}
    
    max = float('-inf')
    
    for t, graph_frame in progress(
      graph.frames.items(), desc='constructing networks', unit='Networks' 
    ):
      frames[t] = graph_frame.to_mpn(hist=hist, dist=dist)
      
      metric_max = frames[t].metric.max()
      
      if metric_max > max:
        max = metric_max
        
    if normalize:
      for frame in frames.values():
        frame.metric /= max
        
        assert(frame.metric.max() <= 1)
        assert(frame.metric.min() >= 0)
        
    return MMNetwork(frames)
  
class MMNetworkTarget(CacheableTarget[MMNetworkConf, MMNetwork]):
  @staticmethod
  def target_type() -> str:
    return 'mmnetwork'
  
  @staticmethod
  def entity_type() -> type[MMNetwork]:
    return MMNetwork
      
  def __init__(
    self, 
    name: str,
    cache_path: str,
    graph: GraphTarget,
    dist: str,
    hist: str,
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
      rule = MMNetworkRule(),
      conf = {
        'dist': dist,
        'hist': hist,
        'normalize': normalize
      },
      depends = [graph],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )  
  
   