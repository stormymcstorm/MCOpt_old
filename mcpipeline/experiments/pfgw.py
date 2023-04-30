"""
Experiment to calculate pfgw couplings
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

__all__ = ['PfGWTarget']

class PfGWConf(CouplingsConf):
  m: float

class PfGWRule(CouplingsRule[PfGWConf]):
  def __call__(
    self, 
    network: MMNetwork,
    attributes: Attributes,
    m: float,
    num_random_iter: int | None,
    random_state: int | None,
    progress: ProgressFactory, 
  ) -> Couplings:
    def pfgw(src_t, src, dest_t, dest, **kwargs):
      M = attributes.attrs[
        attributes.index_map[src_t], attributes.index_map[dest_t]
      ]
      
      return ot.pfGW(src, dest, m = m, M=M, **kwargs)
    
    return self.run(
      ot = pfgw,
      type = 'pfGW',
      network = network,
      num_random_iter=num_random_iter,
      random_state=random_state,
      progress=progress
    )
  
class PfGWTarget(CouplingsTarget[CouplingsConf]):
  def __init__(
    self, 
    name: str,
    cache_path: str,
    network: MMNetworkTarget,
    attributes: AttributesTarget,
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
      rule = PfGWRule(),
      conf = {
        'm': m,
        'num_random_iter': num_random_iter,
        'random_state': random_state,
      },
      depends = [network, attributes],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )