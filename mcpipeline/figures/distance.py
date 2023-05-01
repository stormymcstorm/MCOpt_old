"""
Figure to display distance matrix
"""

from __future__ import annotations
from typing import TypedDict, List

import numpy as np
import matplotlib.pyplot as plt
from seaborn import heatmap

from mcpipeline.target import Rule
from mcpipeline.targets.figure import Figure, FigureTarget
from mcpipeline.targets.graph import Graph, GraphTarget
from mcpipeline.targets.couplings import Couplings, CouplingsTarget
from mcpipeline.figures.graphs import GraphsFigureConf
from mcpipeline.util import ProgressFactory

class DistanceFigureConf(GraphsFigureConf):
  cmap: str
  pass

class DistanceFigureRule(Rule[DistanceFigureConf, Figure]):
  def __call__(
    self,
    couplings: Couplings,
    cmap: str,
    output_path: str,
    output_fmt: str,
    savefig_kwargs: dict | None,
    progress: ProgressFactory,
  ) -> Figure:
    fig, ax = plt.subplots()
    
    ax.matshow(
      couplings.distances,
      cmap = cmap,
    )
    
    return Figure({0: fig}, output_path, output_fmt, savefig_kwargs)

class DistanceFigureTarget(FigureTarget[DistanceFigureConf]):
  def __init__(
    self, 
    name: str,
    cache_path: str,
    couplings: CouplingsTarget,
    cmap: str = 'Blues',
    output_path: str | None = None,
    output_fmt: str | None = None,
    savefig_kwargs: dict | None = None,
    display_name: str | None = None,
    desc: str | None = None,
    **kwargs,
  ):
    if display_name is None:
      display_name = couplings.display_name
    
    if desc is None:
      desc = couplings.desc
      
    if output_path is None:
      output_path = cache_path
    
    if output_fmt is None:
      output_fmt = name + '.png'
    
    super().__init__(
      name = name,
      cache_path = cache_path,
      rule = DistanceFigureRule(),
      conf = {
        'cmap': cmap,
        'output_path': output_path,
        'output_fmt': output_fmt,
        'savefig_kwargs': savefig_kwargs,
      },
      depends = [couplings],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )