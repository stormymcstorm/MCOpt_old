"""
Utilities that are common across experiments
"""

from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import seaborn
import numpy as np

from mcopt import MorseGraph, Coupling

def draw_graphs(
  src: MorseGraph,
  dests: Dict[int, MorseGraph],
  width: int,
  height: int,
  cmap: str | None = None,
  fontsize: int = 40,
  src_title: str = 'Source',
  dest_title_fmt: str = '{t}',
  couplings: Dict[int, Coupling] | None = None,
  figsize = None,
  **kwargs
):
  assert(width * (height - 1) >= len(dests))
  
  if figsize is None:
    figsize = (width * 12, height * 12)
    
  if cmap is None:
    if couplings is None:
      cmap = 'black'
    else:
      cmap = 'cool'
    
  fig, axes = plt.subplots(height, width, figsize=figsize)
  
  for ax in axes.ravel():
    ax.set_axis_off()
    
  src_node_color = src.node_color_by_position()
  src.draw(
    ax = axes[0, width//2],
    cmap = cmap,
    node_color = src_node_color,
    **kwargs
  )
  axes[0, width//2].set_title(src_title, fontsize=fontsize)
  
  for ax, (t, dest) in zip(axes.ravel()[width:], dests.items()):
    node_color = None
    
    if couplings is not None:
      node_color = dest.node_color_by_coupling(src_node_color, couplings[t])
    
    dest.draw(
      ax = ax,
      cmap = cmap,
      node_color=node_color,
      **kwargs
    )
    ax.set_title(dest_title_fmt.format(t = t), fontsize=fontsize)
  
  return fig

def plot_max_match_results(
  max_match_results,
  m,
):
  fig, ax = plt.subplots()
  
  x = max_match_results.ms
  y = max_match_results.results.max(axis=1)
  
  ax.plot(x, y)
  
  ymin, _ = ax.get_ybound()
  xmin, _ = ax.get_xbound()
  
  i = np.abs(x - m).argmin()
  
  ax.add_line(Line2D([x[i], x[i]], [ymin, y[i]], color='grey', linestyle='--'))
  ax.add_line(Line2D([xmin, x[i]], [y[i], y[i]], color='grey', linestyle='--'))
  ax.plot(x[i], y[i], marker='o')

  ax.set_xlabel('m', fontsize=10)
  ax.set_ylabel('Max Match Distance', fontsize=10)
  
  ax.set_xticks([x.min(), x[i], x.max()])
  ax.set_yticks([y.min(), y[i], y.max()])
  
  return fig

def plot_distance_heatmap(
  distances,
  fontsize=80,
  cmap = 'Blues',
  figsize = None,
):
  keys = list(distances.keys())
  keys.sort()
  
  data = np.array([[distances[i] for i in keys]])
  
  if figsize is None:
    figsize = (len(keys) * 15, 20)
    
  fig, ax = plt.subplots(figsize = figsize)
  
  seaborn.heatmap(
    data,
    ax = ax,
    annot = True,
    vmax = data.max(),
    vmin = data.min(),
    square = True,
    cmap = cmap,
    robust = False,
    cbar = False,
    yticklabels = False,
    annot_kws=dict(
      fontsize = 60
    )
  )
  
  ax.tick_params(axis='x', length=20, width=5, pad=10)
  ax.set_xticklabels([f'{i}' for i in keys], fontsize=60)
  
  ax.set_xlabel('Time Step', fontsize=100)
  
  cbar = fig.colorbar(
    cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=data.min(), vmax=data.max())),
    location = 'bottom',
    pad = 0.3,
  )
  
  cbar.ax.tick_params(length = 20, width = 5, pad = 10, labelsize=60)
  
  return fig