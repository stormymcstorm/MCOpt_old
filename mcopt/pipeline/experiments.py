import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from mcopt import MorseGraph, ot
from mcopt.pipeline.graph import Graph
import mcopt.pipeline.util as util

def draw_graph(
  graph: Graph, 
  out: str,
  cmap='cool',
  node_size=40,
  critical_scale=3,
  fontsize=40,
  title=None, 
  ax_prefix=None,
  **_rest
):
  if not ax_prefix:
    ax_prefix = graph.name
  
  fig = util.draw_graphs(
    graph.frames,
    ax_prefix,
    cmap=cmap,
    node_size=node_size,
    critical_scale=critical_scale,
    fontsize=fontsize,
    title=title,
  )
  
  fig.savefig(out, bbox_inches='tight')
  
def tune_m(
  graph: Graph, 
  out: str,
  hist: str = None,
  dist: str = None,
  m_start: float = 0.9,
  num_ms: int = 5,
  title=None,
  fontsize=40,
  figsize=None,
  detailed=True,
  **_rest,
):
  if not hist:
    hist = 'degree'
    
  if not dist:
    dist = 'geo'
        
  m_step = (1 - m_start) / num_ms
  ms = np.array([m_start + m_step * i for i in range(num_ms + 1)])
    
  src_graph = graph.frames[0]
  
  dest_graphs = graph.frames[1:]
  
  results = np.zeros(shape = (len(ms), len(dest_graphs)))
  
  src_net = src_graph.to_mpn(hist=hist, dist=dist)
  
  for dest_i, dest_graph in enumerate(tqdm(dest_graphs, desc='Running fpGW for each graph', leave=False)):
    M = MorseGraph.attribute_cost_matrix(src_graph, dest_graph)
    
    dest_net = dest_graph.to_mpn(hist=hist, dist=dist)
    
    for m_i, m in enumerate(tqdm(ms, desc='Running for each m', leave=False)):
      if np.isclose(m, 1):
        coupling, _ = ot.fGW(src_net, dest_net, M)
      else:
        coupling, _ = ot.fpGW(src_net, dest_net, m, M)
          
      max_match_dist = float('-inf')
      
      for dest_n in dest_graph.nodes:
        i = coupling.dest_rev_map[dest_n]
        src_i = coupling[:, i].argmax()
        
        if np.isclose(coupling[src_i, i], 0):
          pass
        
        src_n = coupling.src_map[src_i]
        dest_pos = dest_graph.nodes(data='pos2')[dest_n]
        src_pos = src_graph.nodes(data='pos2')[src_n]
        
        match_dist = np.linalg.norm(dest_pos - src_pos)
        
        if match_dist > max_match_dist:
          max_match_dist = match_dist
      
      results[m_i, dest_i] = max_match_dist
      
  fig_summary, ax_summary = plt.subplots()
  
  x = ms
  y = results.max(axis=1)
  
  ax_summary.plot(x, y)
  ax_summary.set_xticks(np.round(ms, decimals=2))
  ax_summary.tick_params(axis='x', labelrotation=45, labelright=True)

  ax_summary.set_xlabel('m')
  ax_summary.set_ylabel('Max match distance')
  
  if title:
    fig_summary.suptitle(title, fontsize=fontsize)
    
  fig_summary.savefig(os.path.join(out, 'summary.png'), bbox_inches='tight')
  
  if not detailed:
    return
  
  if figsize is None:
    w, h = util.layout_like_square(len(dest_graphs))
  else:
    w, h = figsize
  
  assert (w * h >= len(dest_graphs))
  fig, axes = plt.subplots(h, w, figsize=(w * 12, h * 12))
  
  for dest_i, ax in zip(range(len(dest_graphs)), axes.ravel()):
    x = ms
    y = results[:, dest_i]
    
    ax.plot(x, y)
    ax.set_xticks(np.round(ms, decimals=2))
    ax.tick_params(axis='x', labelrotation=45, labelright=True)
    ax.set_xlabel('m')
    ax.set_ylabel('Max match distance')
    ax.set_title(f'{graph.name} {dest_i + 2}', fontsize=fontsize//2)
    
  if len(dest_graph) > len(axes.ravel()):
    for ax in axes.ravel()[len(dest_graphs):]:
      ax.set_axis_off()
    
  if title:
    fig.suptitle(title, fontsize=fontsize)
    
  fig.savefig(os.path.join(out, 'detailed.png'), bbox_inches='tight')
    
    
  