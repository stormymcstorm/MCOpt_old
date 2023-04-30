"""
Target for combination of entities
"""

from __future__ import annotations
from typing import TypedDict, Dict, List, Tuple
import os
import json
import pickle

import numpy as np
from mcopt import MorseGraph, MetricMeasureNetwork

from mcpipeline.entity import Entity
from mcpipeline.target import Target, Rule
from mcpipeline.targets.graph import Graph, GraphTarget
from mcpipeline.targets.mm_network import MMNetwork, MMNetworkTarget
from mcpipeline.util import ProgressFactory

__all__ = ['CombinedGraphs', 'CombineGraphsTarget', 'CombinedNetworks', 'CombineNetworksTarget']

class CombinedGraphs(Entity):
  originals: List[Graph]
  frames: Dict[int, MorseGraph]
  index_map: Dict[int, Dict[int, int]]
  rev_index_map: Dict[int, Tuple[int, int]]
  
  def __init__(self, originals: List[Graph]) -> None:
    self.originals = originals
    
    index_map = {}
    rev_index_map = {}
    frames = []
    for o_i, original in enumerate(originals):
      index_map[o_i] = {}
      
      for t, graph in original.frames.items():
        i = len(frames)
        
        frames.append(graph)
        
        index_map[o_i][t] = i
        rev_index_map[i] = (o_i, t)
        
    self.index_map = index_map
    self.rev_index_map = rev_index_map
    self.frames = {i: frame for i, frame in enumerate(frames)}

class CombineGraphsRule(Rule[TypedDict, CombinedGraphs]):
  def __call__(self, *originals: List[Graph], progress: ProgressFactory) -> CombinedNetworks:
    return CombinedGraphs(originals)
  
class CombineGraphsTarget(Target[TypedDict, CombinedGraphs]):
  @staticmethod
  def target_type() -> str:
    return 'graph'
  
  @staticmethod
  def entity_type() -> type[Graph]:
    return CombinedGraphs
  
  def __init__(
    self, 
    name: str,
    cache_path: str,
    graphs: List[GraphTarget],
    display_name: str | None = None,
    desc: str | None = None,
    **kwargs,
  ):
    super().__init__(
      name = name,
      cache_path = cache_path,
      rule = CombineGraphsRule(),
      conf = {},
      depends = graphs,
      display_name = display_name,
      desc = desc,
      **kwargs,
    )
    
class CombinedNetworks(Entity):
  originals: List[MMNetwork]
  frames: Dict[int, MetricMeasureNetwork]
  index_map: Dict[int, Tuple[int, int]]
  
  def __init__(self, originals: List[MMNetwork]) -> None:
    self.originals = originals
    
    index_map = {}
    frames = []
    for o_i, original in enumerate(originals):
      for t, graph in original.frames.items():
        i = len(frames)
        
        frames.append(graph)
        
        index_map[i] = (o_i, t)
        
    self.index_map = index_map
    self.frames = {i: frame for i, frame in enumerate(frames)}

class CombineNetworksRule(Rule[TypedDict, CombinedNetworks]):
  def __call__(self, *originals, progress: ProgressFactory) -> CombinedNetworks:
    return CombinedNetworks(originals)
  
class CombineNetworksTarget(Target[TypedDict, CombinedNetworks]):
  @staticmethod
  def target_type() -> str:
    return 'mmnetwork'
  
  @staticmethod
  def entity_type() -> type[Graph]:
    return CombinedNetworks
  
  def __init__(
    self, 
    name: str,
    cache_path: str,
    networks: List[MMNetworkTarget],
    display_name: str | None = None,
    desc: str | None = None,
    **kwargs,
  ):
    super().__init__(
      name = name,
      cache_path = cache_path,
      rule = CombineNetworksRule(),
      conf = {},
      depends = networks,
      display_name = display_name,
      desc = desc,
      **kwargs,
    )