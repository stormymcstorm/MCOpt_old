"""
Experiment to calculate gw couplings
"""

from __future__ import annotations
from typing import TypedDict, Dict, List
import os

import numpy as np
from mcopt import ot

from mcpipeline.target import Rule
from mcpipeline.targets.mm_network import MMNetwork, MMNetworkTarget
from mcpipeline.targets.couplings import Couplings, CouplingsConf, CouplingsRule, CouplingsTarget
from mcpipeline.util import ProgressFactory

__all__ = ['GWTarget']

class GWRule(CouplingsRule[CouplingsConf]):
  def __init__(self, distance_only: bool = False) -> None:
    super().__init__(distance_only)
  
  def __call__(
    self, 
    network: MMNetwork,
    num_random_iter: int | None,
    random_state: int | None,
    progress: ProgressFactory, 
  ) -> Couplings:
    def gw(src_t, src, dest_t, dest, **kwargs):
      return ot.GW(src, dest, **kwargs)
    
    return self.run(
      ot = gw,
      type = 'GW',
      network = network,
      num_random_iter=num_random_iter,
      random_state=random_state,
      progress=progress
    )
  
class GWTarget(CouplingsTarget[CouplingsConf]):
  def __init__(
    self, 
    name: str,
    cache_path: str,
    network: MMNetworkTarget,
    distance_only: bool = False,
    num_random_iter: int | None = None,
    random_state: int | None = None,
    display_name: str | None = None,
    desc: str | None = None,
    **kwargs,
  ):
    if display_name is None:
      display_name = network.display_name
    
    if desc is None:
      desc = network.desc
    
    super().__init__(
      name = name,
      cache_path = cache_path,
      rule = GWRule(distance_only=distance_only),
      conf = {
        'num_random_iter': num_random_iter,
        'random_state': random_state,
      },
      depends = [network],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )