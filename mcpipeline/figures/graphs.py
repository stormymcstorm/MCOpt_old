"""
Figure to display graphs
"""

from __future__ import annotations
from typing import TypedDict

import numpy as np
import matplotlib.pyplot as plt

from mcpipeline.target import Rule
from mcpipeline.targets.figure import Figure, FigureConf, FigureTarget
from mcpipeline.targets.graph import Graph, GraphTarget
from mcpipeline.util import ProgressFactory

__all__ = ['GraphsFigureTarget', 'GraphsFigureConf']

class GraphsFigureConf(FigureConf):
  cmap: str
  rotation: float
  node_size: int

class GraphsFigureRule(Rule[GraphsFigureConf, Figure]):
  def __call__(
    self, 
    graph: Graph,
    cmap: str,
    rotation: float,
    node_size: int,
    output_path: str,
    output_fmt: str,
    savefig_kwargs: dict | None,
    progress: ProgressFactory,
  ) -> Figure:
    figs = {}
    
    for t, graph_frame in progress(
      graph.frames.items(),
      desc = 'drawing graphs',
    ):
      fig, ax = plt.subplots()
      
      graph_frame.draw(
        ax = ax,
        cmap = cmap,
        rotation = rotation,
        node_size=node_size
      )
      
      figs[t] = fig
    
    return Figure(figs, output_path, output_fmt, savefig_kwargs)
  
class GraphsFigureTarget(FigureTarget[GraphsFigureConf]):
  def __init__(
    self, 
    name: str,
    cache_path: str,
    graph: GraphTarget,
    cmap: str = 'black',
    rotation: float = 0,
    node_size: int = 40,
    output_path: str | None = None,
    output_fmt: str | None = None,
    savefig_kwargs: dict | None = None,
    display_name: str | None = None,
    desc: str | None = None,
    **kwargs,
  ):
    if display_name is None:
      display_name = graph.display_name
    
    if desc is None:
      desc = graph.desc
      
    if output_path is None:
      output_path = cache_path
    
    if output_fmt is None:
      output_fmt = name + '.{t:04d}.png'
    
    super().__init__(
      name = name,
      cache_path = cache_path,
      rule = GraphsFigureRule(),
      conf = {
        'cmap': cmap,
        'rotation': rotation,
        'node_size': node_size,
        'output_path': output_path,
        'output_fmt': output_fmt,
        'savefig_kwargs': savefig_kwargs,
      },
      depends = [graph],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )