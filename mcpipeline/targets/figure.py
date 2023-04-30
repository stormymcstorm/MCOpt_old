"""
Target for figures
"""

from __future__ import annotations
from typing import Dict, TypedDict
import os

import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt

from mcpipeline.entity import Entity
from mcpipeline.target import Target, Conf
from mcpipeline.util import ProgressFactory

__all__ = ['Figure', 'FigureConf', 'FigureTarget']

class Figure(Entity):
  figs: Dict[int, matplotlib.figure.Figure]
  output_path: str
  output_fmt: str
  savefig_kwargs: dict
  
  def __init__(
    self, 
    figs: Dict[int, matplotlib.figure.Figure],
    output_path: str,
    output_fmt: str,
    savefig_kwargs: dict | None = None
  ):
    super().__init__()
    
    self.figs = figs
    self.output_path = output_path
    self.output_fmt = output_fmt
    self.savefig_kwargs = savefig_kwargs if savefig_kwargs is not None else {}
    
  def save(self, cache_path: str, progress: ProgressFactory):
    os.makedirs(self.output_path, exist_ok=True)
    
    for t, fig in progress(
      self.figs.items(),
      desc = 'writing figures',
    ):
      fig.savefig(
        fname=os.path.join(self.output_path, self.output_fmt.format(t = t)),
        **self.savefig_kwargs
      )
      
class FigureConf(TypedDict):
  output_path: str
  output_fmt: str
  savefig_kwargs: dict | None
      
class FigureTarget(Target[Conf, Figure]):
  @staticmethod
  def target_type() -> str:
    return 'figure'
  
  @staticmethod
  def entity_type() -> type[Figure]:
    return Figure