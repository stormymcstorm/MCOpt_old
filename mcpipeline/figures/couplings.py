"""
Figure to display graphs couplings
"""

from __future__ import annotations
from typing import TypedDict, List

import numpy as np
import matplotlib.pyplot as plt

from mcpipeline.target import Rule
from mcpipeline.targets.figure import Figure, FigureTarget
from mcpipeline.targets.graph import Graph, GraphTarget
from mcpipeline.targets.couplings import Couplings, CouplingsTarget
from mcpipeline.figures.graphs import GraphsFigureConf
from mcpipeline.util import ProgressFactory

__all__ = ['CouplingsFigureTarget', 'CouplingsCombinedFigureTarget']

class CouplingsFigureConf(GraphsFigureConf):
  src_t: int

class CouplingsFigureRule(Rule[CouplingsFigureConf, Figure]):
  def __call__(
    self, 
    graph: Graph,
    couplings: Couplings,
    src_t: int,
    cmap: str,
    rotation: float,
    node_size: int,
    output_path: str,
    output_fmt: str,
    savefig_kwargs: dict | None,
    progress: ProgressFactory,
  ) -> Figure:
    figs = {}
    
    src = graph.frames[src_t]
    dests = graph.frames.copy()
    dests.pop(src_t)
    
    src_node_color = src.node_color_by_position()
    figs[src_t], src_ax = plt.subplots()
    src.draw(
      ax = src_ax,
      cmap = cmap,
      rotation = rotation,
      node_color=src_node_color,
      node_size=node_size
    )
    
    for t, dest in progress(
      dests.items(),
      desc = 'drawing graphs',
    ):
      fig, ax = plt.subplots()
      
      coupling = couplings.couplings[
        couplings.index_map[src_t], couplings.index_map[t]
      ]
      
      dest.draw(
        ax = ax,
        cmap = cmap,
        rotation = rotation,
        node_color = dest.node_color_by_coupling(src_node_color, coupling),
        node_size=node_size
      )
      
      figs[t] = fig
    
    return Figure(figs, output_path, output_fmt, savefig_kwargs)
    
class CouplingsFigureTarget(FigureTarget[CouplingsFigureConf]):
  def __init__(
    self, 
    name: str,
    cache_path: str,
    graph: GraphTarget,
    couplings: CouplingsTarget,
    src_t: int,
    cmap: str = 'cool',
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
      rule = CouplingsFigureRule(),
      conf = {
        'src_t': src_t,
        'cmap': cmap,
        'rotation': rotation,
        'node_size': node_size,
        'output_path': output_path,
        'output_fmt': output_fmt,
        'savefig_kwargs': savefig_kwargs,
      },
      depends = [graph, couplings],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )

class CouplingsCombinedFigureConf(GraphsFigureConf):
  src_t: int
  nrows: int
  ncols: int
  figsize: List[int] | None
  fontsize: float
   
class CouplingsCombinedFigureRule(Rule[CouplingsCombinedFigureConf, Figure]):
  def __call__(
    self, 
    graph: Graph,
    couplings: Couplings,
    src_t: int,
    nrows: int,
    ncols: int,
    figsize: List[int] | None,
    fontsize: float,
    cmap: str,
    rotation: float,
    node_size: int,
    output_path: str,
    output_fmt: str,
    savefig_kwargs: dict | None,
    progress: ProgressFactory,
  ) -> Figure:
    src = graph.frames[src_t]
    dests = graph.frames.copy()
    dests.pop(src_t)
    
    assert(ncols * (nrows - 1)) >= len(dests)
    
    if figsize is None:
      figsize = (ncols * 12, nrows * 12)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    for ax in axes.ravel():
      ax.set_axis_off()
    
    src_node_color = src.node_color_by_position()
    src_ax = axes[0, ncols//2]
    src.draw(
      ax = src_ax,
      cmap = cmap,
      rotation = rotation,
      node_color=src_node_color,
      node_size=node_size
    )
    src_ax.set_title(f't = {src_t}', y=0, pad=-20, fontsize=fontsize)
    
    for ax, (t, dest) in progress(
      zip(axes.ravel()[ncols:], dests.items()),
      desc = 'drawing graphs',
    ):      
      coupling = couplings.couplings[
        couplings.index_map[src_t], couplings.index_map[t]
      ]
      
      dest.draw(
        ax = ax,
        cmap = cmap,
        rotation = rotation,
        node_color = dest.node_color_by_coupling(src_node_color, coupling),
        node_size=node_size
      )
      
      ax.set_title(f't = {t}', y=0, pad=-20, fontsize=fontsize)
    
    return Figure({0: fig}, output_path, output_fmt, savefig_kwargs)
  
class CouplingsCombinedFigureTarget(FigureTarget[CouplingsCombinedFigureConf]):
  def __init__(
    self, 
    name: str,
    cache_path: str,
    graph: GraphTarget,
    couplings: CouplingsTarget,
    src_t: int,
    nrows: int,
    ncols: int,
    cmap: str = 'cool',
    rotation: float = 0,
    node_size: int = 40,
    figsize: List[int] | None = None,
    fontsize: float = 40,
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
      output_fmt = name + '.png'
    
    super().__init__(
      name = name,
      cache_path = cache_path,
      rule = CouplingsCombinedFigureRule(),
      conf = {
        'src_t': src_t,
        'nrows': nrows,
        'ncols': ncols,
        'figsize': figsize,
        'fontsize': fontsize,
        'cmap': cmap,
        'rotation': rotation,
        'node_size': node_size,
        'output_path': output_path,
        'output_fmt': output_fmt,
        'savefig_kwargs': savefig_kwargs,
      },
      depends = [graph, couplings],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )
