"""
Figure to display max match results
"""

from __future__ import annotations
from typing import TypedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mcpipeline.target import Rule
from mcpipeline.targets.figure import Figure, FigureConf, FigureTarget
from mcpipeline.experiments.max_match import MaxMatch
from mcpipeline.util import ProgressFactory

__all__ = ['MaxMatchFigureTarget', 'MaxMatchCombinedFigureTarget']

class MaxMatchFigureConf(TypedDict):
  m: float

class MaxMatchFigureRule(Rule[MaxMatchFigureConf, Figure]):
  def __call__(
    self, 
    max_match: MaxMatch,
    m: float,
    output_path: str,
    output_fmt: str,
    savefig_kwargs: dict | None,
    progress: ProgressFactory,
  ) -> Figure:
    fig, ax = plt.subplots()
  
    x = np.asarray(max_match.ms)
    y = np.nanmax(max_match.results, axis=1)
    
    ax.plot(x, y)
    
    ymin, _ = ax.get_ybound()
    xmin, _ = ax.get_xbound()
    
    i = np.abs(x - m).argmin()
    
    ax.add_line(Line2D([x[i], x[i]], [ymin, y[i]], color='grey', linestyle='--'))
    ax.add_line(Line2D([xmin, x[i]], [y[i], y[i]], color='grey', linestyle='--'))
    ax.plot(x[i], y[i], marker='o')

    ax.set_xticks([x.min(), x[i], x.max()])
    ax.set_yticks([y.min(), y[i], y.max()])
    ax.tick_params('both', labelsize=16)
    
    return Figure({0 : fig}, output_path, output_fmt, savefig_kwargs)
  
class MaxMatchFigureTarget(FigureTarget[MaxMatchFigureConf]):
  def __init__(
    self, 
    name: str,
    cache_path: str,
    max_match,
    m: float,
    output_path: str | None = None,
    output_fmt: str | None = None,
    savefig_kwargs: dict | None = None,
    display_name: str | None = None,
    desc: str | None = None,
    **kwargs,
  ):
    if display_name is None:
      display_name = max_match.display_name
    
    if desc is None:
      desc = max_match.desc
      
    if output_path is None:
      output_path = cache_path
    
    if output_fmt is None:
      output_fmt = name + '.png'
    
    super().__init__(
      name = name,
      cache_path = cache_path,
      rule = MaxMatchFigureRule(),
      conf = {
        'm': m,
        'output_path': output_path,
        'output_fmt': output_fmt,
        'savefig_kwargs': savefig_kwargs,
      },
      depends = [max_match],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )


class MaxMatchCombinedFigureRule(Rule[MaxMatchFigureConf, Figure]):
  def __call__(
    self, 
    max_match: MaxMatch,
    m: float,
    output_path: str,
    output_fmt: str,
    savefig_kwargs: dict | None,
    progress: ProgressFactory,
  ) -> Figure:
    fig, ax = plt.subplots()
  
    x = np.asarray(max_match.ms)
    
    for mm in max_match.results.T:
      ax.plot(x, mm)
    
    ymin, _ = ax.get_ybound()
    xmin, _ = ax.get_xbound()
    
    i = np.abs(x - m).argmin()
    
    ax.axvline(m, color='grey', linestyle='--')

    ax.set_xlabel('m', fontsize=10)
    ax.set_ylabel('Max Match Distance', fontsize=10)
    
    return Figure({0 : fig}, output_path, output_fmt, savefig_kwargs)
  
class MaxMatchCombinedFigureTarget(FigureTarget[MaxMatchFigureConf]):
  def __init__(
    self, 
    name: str,
    cache_path: str,
    max_match,
    m: float,
    output_path: str | None = None,
    output_fmt: str | None = None,
    savefig_kwargs: dict | None = None,
    display_name: str | None = None,
    desc: str | None = None,
    **kwargs,
  ):
    if display_name is None:
      display_name = max_match.display_name
    
    if desc is None:
      desc = max_match.desc
      
    if output_path is None:
      output_path = cache_path
    
    if output_fmt is None:
      output_fmt = name + '.png'
    
    super().__init__(
      name = name,
      cache_path = cache_path,
      rule = MaxMatchCombinedFigureRule(),
      conf = {
        'm': m,
        'output_path': output_path,
        'output_fmt': output_fmt,
        'savefig_kwargs': savefig_kwargs,
      },
      depends = [max_match],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )

