"""
Experiment to calculate pgw couplings
"""

from __future__ import annotations
from typing import TypedDict, Dict, List
import os

import numpy as np
from mcopt import ot

from mcpipeline.target import Rule
from mcpipeline.targets.mm_network import MMNetwork, MMNetworkTarget
from mcpipeline.targets.attributes import Attributes, AttributesTarget
from mcpipeline.targets.couplings import Couplings, CouplingsConf, CouplingsRule, CouplingsTarget
from mcpipeline.util import ProgressFactory

__all__ = ['PGWTarget']

class PGWConf(CouplingsConf):
  m: float

class PGWRule(CouplingsRule[PGWConf]):
  def __call__(
    self, 
    network: MMNetwork,
    m: float,
    num_random_iter: int | None,
    random_state: int | None,
    progress: ProgressFactory, 
  ) -> Couplings:
    def pfgw(src_t, src, dest_t, dest, **kwargs):
      return ot.pGW(src, dest, m = m, **kwargs)
    
    return self.run(
      ot = pfgw,
      type = 'pGW',
      network = network,
      num_random_iter=num_random_iter,
      random_state=random_state,
      progress=progress
    )
  
class PGWTarget(CouplingsTarget[CouplingsConf]):
  def __init__(
    self, 
    name: str,
    cache_path: str,
    network: MMNetworkTarget,
    m: float,
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
      rule = PGWRule(),
      conf = {
        'm': m,
        'num_random_iter': num_random_iter,
        'random_state': random_state,
      },
      depends = [network],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )