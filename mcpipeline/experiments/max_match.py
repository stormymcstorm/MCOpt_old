"""
Experiment to calculate max match distances with various m values
"""

from __future__ import annotations
from typing import TypedDict, Dict, List
import os

import numpy as np
from mcopt import MorseGraph, ot

from mcpipeline.entity import CacheableEntity
from mcpipeline.target import CacheableTarget, Rule
from mcpipeline.targets.mm_network import MMNetwork, MMNetworkTarget
from mcpipeline.targets.graph import Graph, GraphTarget
from mcpipeline.targets.attributes import Attributes, AttributesTarget
from mcpipeline.util import ProgressFactory

__all__ = ['MaxMatch', 'MaxMatchPfGWTarget', 'MaxMatchPWTarget', 'MaxMatchPGWTarget']

def _max_match(src, dest, coupling):
  max_match_dist = 0
  
  c = np.asarray(coupling)
    
  for dest_n in dest.nodes:
    dest_i = coupling.dest_rev_map[dest_n]
    src_i = c[:, dest_i].argmax()
    
    if np.isclose(c[src_i, dest_i], 0):
      continue
    
    src_n = coupling.src_map[src_i]
    
    dest_pos = dest.nodes(data='pos2')[dest_n]
    src_pos = src.nodes(data='pos2')[src_n]
    
    match_dist = np.linalg.norm(dest_pos - src_pos)
    
    if match_dist > max_match_dist:
      max_match_dist = match_dist
      
  return max_match_dist

class MaxMatch(CacheableEntity):
  @staticmethod
  def load(cache_path: str, progress: ProgressFactory) -> CacheableEntity:
    results = np.load(os.path.join(cache_path, 'results.npy'))
    
    ms = np.load(os.path.join(cache_path, 'ms.npy'))
    
    return MaxMatch(ms, results)
  
  results: np.ndarray
  
  def __init__(self, ms: List[float], results: np.ndarray):
    self.results = results
    self.ms = ms
    
  def save(self, cache_path: str, progress: ProgressFactory):
    np.save(
      file = os.path.join(cache_path, 'results.npy'),
      arr = self.results
    )
    
    np.save(
      file = os.path.join(cache_path, 'ms.npy'),
      arr = self.ms
    )
    
class MaxMatchConf(TypedDict):
  ms: List[float]
  src_t: int
  num_random_iter: int | None
  random_state: int | None
  
class MaxMatchPfGWConf(MaxMatchConf):
  alpha: float
  
class MaxMatchPfGWRule(Rule[MaxMatchPfGWConf, MaxMatch]):
  def __call__(
    self, 
    network: MMNetwork,
    graph: Graph,
    attributes: Attributes,
    ms: List[float],
    src_t: int,
    alpha: float,
    num_random_iter: int | None,
    random_state: int | None,
    progress: ProgressFactory
  ) -> MaxMatch:
    src_graph = graph.frames[src_t]
    dest_graphs = graph.frames.copy()
    dest_graphs.pop(src_t)
    
    src_net = network.frames[src_t]
    dest_nets = network.frames.copy()
    dest_nets.pop(src_t)
    
    results = np.zeros(shape=(len(ms), len(dest_graphs)))
    
    rng = np.random.default_rng(random_state)
    
    with progress(
      total = len(ms) * len(dest_graphs) * (1 if num_random_iter is None else num_random_iter),
      desc = 'calculating max match results'
    ) as prog:
      for dest_i, (t, dest_graph) in enumerate(dest_graphs.items()):
        dest_net = dest_nets[t]
        
        M = attributes.attrs[attributes.index_map[src_t], attributes.index_map[t]]
        
        assert(M.shape[0] == src_graph.number_of_nodes())
        assert(M.shape[1] == dest_graph.number_of_nodes())
        
        for m_i, m in enumerate(ms):
          # print(f'm = {m}')
          
          try:
            if num_random_iter is not None:
              min_dist = float('inf')
              min_coupling = None
              
              for _ in range(num_random_iter):
                if np.isclose(m, 1):
                  c, d = ot.fGW(src_net, dest_net, M=M, alpha=alpha, random_G0=True, random_state=rng)
                else:
                  c, d = ot.pfGW(src_net, dest_net, m=m, M=M, alpha=alpha, random_G0=True, random_state=rng)
                
                if d < min_dist:
                  min_dist = d
                  min_coupling = c
                  
                prog.update()
              
              coupling = min_coupling
            else:
              if np.isclose(m, 1):
                coupling, _ = ot.fGW(src_net, dest_net, M=M, alpha=alpha, random_G0=False)
              else:
                coupling, _ = ot.pfGW(src_net, dest_net, m=m, M=M, alpha=alpha, random_G0=False)
              
              prog.update()
          except(ValueError):
            print(f'Failed for m={m}')
            raise
          
          results[m_i, dest_i] = _max_match(src_graph, dest_graph, coupling)

    return MaxMatch(ms, results)
  
class MaxMatchPfGWTarget(CacheableTarget[MaxMatchPfGWConf, MaxMatch]):
  @staticmethod
  def target_type() -> str:
    return 'max_match'
  
  @staticmethod
  def entity_type() -> type[MaxMatch]:
    return MaxMatch
  
  def __init__(
    self, 
    name: str,
    cache_path: str,
    network: MMNetworkTarget,
    graph: GraphTarget,
    attributes: AttributesTarget,
    ms: List[float],
    src_t: int,
    alpha: float = 0.5,
    num_random_iter: int | None = None,
    random_state: int | None = None,
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
      rule = MaxMatchPfGWRule(),
      conf = {
        'ms': ms,
        'num_random_iter': num_random_iter,
        'random_state': random_state,
        'src_t': src_t,
        'alpha': alpha
      },
      depends = [network, graph, attributes],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )
    
class MaxMatchPWRule(Rule[MaxMatchConf, MaxMatch]):
  def __call__(
    self, 
    network: MMNetwork,
    graph: Graph,
    attributes: Attributes,
    ms: List[float],
    src_t: int,
    num_random_iter: int | None,
    random_state: int | None,
    progress: ProgressFactory
  ) -> MaxMatch:
    src_graph = graph.frames[src_t]
    dest_graphs = graph.frames.copy()
    dest_graphs.pop(src_t)
    
    src_net = network.frames[src_t]
    dest_nets = network.frames.copy()
    dest_nets.pop(src_t)
    
    results = np.zeros(shape=(len(ms), len(dest_graphs)))
    
    rng = np.random.default_rng(random_state)
    
    with progress(
      total = len(ms) * len(dest_graphs) * (1 if num_random_iter is None else num_random_iter),
      desc = 'calculating max match results'
    ) as prog:
      for dest_i, (t, dest_graph) in enumerate(dest_graphs.items()):
        dest_net = dest_nets[t]
        
        M = attributes.attrs[attributes.index_map[src_t], attributes.index_map[t]]
        
        assert(M.shape[0] == src_graph.number_of_nodes())
        assert(M.shape[1] == dest_graph.number_of_nodes())
        
        for m_i, m in enumerate(ms):
          # print(f'm = {m}')
          
          try:
            if num_random_iter is not None:
              min_dist = float('inf')
              min_coupling = None
              
              for _ in range(num_random_iter):
                if np.isclose(m, 1):
                  c, d = ot.Wasserstein(src_net, dest_net, M=M, random_G0=True, random_state=rng)
                else:
                  c, d = ot.pWasserstein(src_net, dest_net, m=m, M=M, random_G0=True, random_state=rng)
                
                if d < min_dist:
                  min_dist = d
                  min_coupling = c
                  
                prog.update()
              
              coupling = min_coupling
            else:
              if np.isclose(m, 1):
                coupling, _ = ot.Wasserstein(src_net, dest_net, M=M, random_G0=False)
              else:
                coupling, _ = ot.pWasserstein(src_net, dest_net, m=m, M=M, random_G0=False)
              
              prog.update()
          except (ValueError):
            print(f'Failed for m={m}')
            raise
          
          results[m_i, dest_i] = _max_match(src_graph, dest_graph, coupling)

    return MaxMatch(ms, results)

class MaxMatchPWTarget(CacheableTarget[MaxMatchPfGWConf, MaxMatch]):
  @staticmethod
  def target_type() -> str:
    return 'max_match'
  
  @staticmethod
  def entity_type() -> type[MaxMatch]:
    return MaxMatch
  
  def __init__(
    self, 
    name: str,
    cache_path: str,
    network: MMNetworkTarget,
    graph: GraphTarget,
    attributes: AttributesTarget,
    ms: List[float],
    src_t: int,
    num_random_iter: int | None = None,
    random_state: int | None = None,
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
      rule = MaxMatchPWRule(),
      conf = {
        'ms': ms,
        'num_random_iter': num_random_iter,
        'random_state': random_state,
        'src_t': src_t,
      },
      depends = [network, graph, attributes],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )
    
class MaxMatchPGWRule(Rule[MaxMatchConf, MaxMatch]):
  def __call__(
    self, 
    network: MMNetwork,
    graph: Graph,
    ms: List[float],
    src_t: int,
    num_random_iter: int | None,
    random_state: int | None,
    progress: ProgressFactory
  ) -> MaxMatch:
    src_graph = graph.frames[src_t]
    dest_graphs = graph.frames.copy()
    dest_graphs.pop(src_t)
    
    src_net = network.frames[src_t]
    dest_nets = network.frames.copy()
    dest_nets.pop(src_t)
    
    results = np.zeros(shape=(len(ms), len(dest_graphs)))
    
    rng = np.random.default_rng(random_state)
    
    with progress(
      total = len(ms) * len(dest_graphs) * (1 if num_random_iter is None else num_random_iter),
      desc = 'calculating max match results'
    ) as prog:
      for dest_i, (t, dest_graph) in enumerate(dest_graphs.items()):
        dest_net = dest_nets[t]
        
        for m_i, m in enumerate(ms):
          # print(f'm = {m}')
          try:
            if np.isclose(m, 1):
              coupling, d = ot.GW(src_net, dest_net, random_G0=False)
            else:
              coupling, d = ot.pGW(src_net, dest_net, m=m, random_G0=False)
            
            prog.update()
            
            results[m_i, dest_i] = d
          except (ValueError):
            print(f'Failed for m = {m}')
            raise

    return MaxMatch(ms, results)

class MaxMatchPGWTarget(CacheableTarget[MaxMatchPfGWConf, MaxMatch]):
  @staticmethod
  def target_type() -> str:
    return 'max_match'
  
  @staticmethod
  def entity_type() -> type[MaxMatch]:
    return MaxMatch
  
  def __init__(
    self, 
    name: str,
    cache_path: str,
    network: MMNetworkTarget,
    graph: GraphTarget,
    ms: List[float],
    src_t: int,
    num_random_iter: int | None = None,
    random_state: int | None = None,
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
      rule = MaxMatchPGWRule(),
      conf = {
        'ms': ms,
        'num_random_iter': num_random_iter,
        'random_state': random_state,
        'src_t': src_t,
      },
      depends = [network, graph],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )
