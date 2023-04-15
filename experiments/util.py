from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tqdm.autonotebook import tqdm, trange

from mcopt import MorseGraph, ot, Coupling


def draw_graphs(
  src : MorseGraph, 
  dests : Dict[int, MorseGraph], 
  width : int, 
  height : int, cmap='cool',
  fontsize=40,
  src_title: str = 'Source',
  dest_title_fmt: str = '{i}',
  couplings = None,
):
  assert(width * (height - 1) >= len(dests))
  
  fig, axes = plt.subplots(height, width, figsize=(width * 12, height * 12))
  
  for ax in axes.ravel():
    ax.set_axis_off()
    
  src_node_color = src.node_color_by_position()
  src.draw(
    ax=axes[0, width//2],
    cmap=cmap,
    node_color = src_node_color
  )
  axes[0, width//2].set_title(src_title, fontsize=fontsize)
  
  for ax, (i, dest) in zip(axes.ravel()[width:], dests.items()):
    node_color = None
    
    if couplings is not None:
      coupling = couplings[i]
      node_color = dest.node_color_by_coupling(src_node_color, coupling)
    
    dest.draw(
      ax=ax,
      cmap=cmap,
      node_color=node_color
    )
    ax.set_title(dest_title_fmt.format(i = i), fontsize = fontsize)
  
  return fig

def compare_couplings(
  src: MorseGraph, dest: MorseGraph,
  couplings: Dict[str, Coupling],
  width : int, 
  height : int, 
  cmap='cool',
  fontsize=40,
  src_title: str = 'Source',
  dest_title_fmt: str = '{i}',
):
  assert(width * (height - 1) >= len(couplings))
  
  fig, axes = plt.subplots(height, width, figsize=(width * 12, height * 12))
  
  for ax in axes.ravel():
    ax.set_axis_off()
    
  src_node_color = src.node_color_by_position()
  src.draw(
    ax=axes[0, width//2],
    cmap=cmap,
    node_color = src_node_color
  )
  axes[0, width//2].set_title(src_title, fontsize=fontsize)
  
  for ax, (title, coupling) in zip(axes.ravel()[width:], couplings.items()):
    dest.draw(
      ax=ax,
      cmap=cmap,
      node_color=dest.node_color_by_coupling(src_node_color, coupling)
    )
    ax.set_title(title, fontsize = fontsize)
  
  return fig
  

def run_gw(
  src : MorseGraph, dests : Dict[int, MorseGraph],
  hist = 'degree',
  dist = 'geo',
  random_G0 = True,
  random_state = None,
  num_random_iter = 10,
):
  couplings = {}
  src_net = src.to_mpn(hist=hist, dist=dist)
  
  progress = tqdm(total=len(dests) * (num_random_iter if random_G0 else 1), desc="Running GW for each destination", leave=False)
  
  for (i, dest) in dests.items():
    dest_net = dest.to_mpn(hist=hist, dist=dist)
    
    if random_G0:
      min_dist = float('inf')
      min_coupling = None
      
      for _ in range(num_random_iter):
        coupling, d = ot.GW(src_net, dest_net, random_G0=random_G0, random_state=random_state)
        
        if d < min_dist:
          min_dist = d
          min_coupling = coupling
          
        progress.update(1)
          
      couplings[i] = min_coupling
    else:
      couplings[i], _ = ot.GW(src_net, dest_net)
      
      progress.update(1)
  
  progress.close()
  return couplings

def run_fgw(
  src : MorseGraph, dests : Dict[int, MorseGraph],
  alpha: float = 0.5,
  hist = 'degree',
  dist = 'geo',
  random_G0 = False,
  random_state = None,
  num_random_iter = 10,
):
  couplings = {}
  src_net = src.to_mpn(hist=hist, dist=dist)
  
  progress = tqdm(total=len(dests) * (num_random_iter if random_G0 else 1), desc="Running fGW for each destination", leave=False)
  
  for (i, dest) in dests.items():
    M = MorseGraph.attribute_cost_matrix(src, dest)
    dest_net = dest.to_mpn(hist=hist, dist=dist)
    
    if random_G0:
      min_dist = float('inf')
      min_coupling = None
      
      for _ in range(num_random_iter):
        coupling, d = ot.fGW(src_net, dest_net, M, alpha=alpha, random_G0=random_G0, random_state=random_state)
        
        if d < min_dist:
          min_dist = d
          min_coupling = coupling
          
        progress.update(1)
          
      couplings[i] = min_coupling
    else:
      couplings[i], _ = ot.fGW(src_net, dest_net, M, alpha=alpha, random_G0=random_G0, random_state=random_state)
      
      progress.update(1)
  
  progress.close()
  
  return couplings

def run_pfgw(
  src : MorseGraph, dests : Dict[int, MorseGraph],
  ms,
  alpha: float = 0.5,
  hist = 'degree',
  dist = 'geo',
  random_G0 = False,
  random_state = None,
  num_random_iter = 10,
):
  couplings = {}
  src_net = src.to_mpn(hist=hist, dist=dist)
  
  if not hasattr(ms, '__len__'):
    ms = np.repeat(ms, len(dests))
  
  progress = tqdm(total=len(dests) * (num_random_iter if random_G0 else 1), desc="Running fGW for each destination", leave=True)
  
  for (i, dest), m in zip(dests.items(), ms):
    M = MorseGraph.attribute_cost_matrix(src, dest)
    dest_net = dest.to_mpn(hist=hist, dist=dist)
    
    if random_G0:
      min_dist = float('inf')
      min_coupling = None
      
      for _ in range(num_random_iter):
        if np.isclose(m, 1):
          coupling, d = ot.fGW(src_net, dest_net, M, alpha=alpha, random_G0=True, random_state=random_state)
        else:
          coupling, d = ot.fpGW(src_net, dest_net, m, M, alpha=alpha, random_G0=True, random_state=random_state)
        
        if d < min_dist:
          min_dist = d
          min_coupling = coupling
          
        progress.update(1)
          
      couplings[i] = min_coupling
    else:
      if np.isclose(m, 1):
        couplings[i], _ = ot.fGW(src_net, dest_net, M, alpha=alpha, random_G0=False)
      else:
        couplings[i], _ = ot.fpGW(src_net, dest_net, m, M, alpha=alpha, random_G0=False)
      
      progress.update(1)
  
  progress.close()
  
  return couplings

def _max_match(src, dest, coupling):
  max_match_dist = 0
    
  for dest_n in dest.nodes:
    dest_i = coupling.dest_rev_map[dest_n]
    src_i = coupling[:, dest_i].argmax()
    
    if np.isclose(coupling[src_i, dest_i], 0):
      continue
    
    src_n = coupling.src_map[src_i]
    
    dest_pos = dest.nodes(data='pos2')[dest_n]
    src_pos = src.nodes(data='pos2')[src_n]
    
    match_dist = np.linalg.norm(dest_pos - src_pos)
    
    if match_dist > max_match_dist:
      max_match_dist = match_dist
      
  return max_match_dist

def calculate_max_match_distances(
  src, dests, ms,
  alpha: float = 0.5,
  hist = 'degree',
  dist = 'geo',
  random_G0 = False,
  random_state = None,
  num_random_iter = 10,
):
  results = np.zeros(shape=(len(ms), len(dests)))
  
  src_net = src.to_mpn(hist=hist, dist=dist)
  
  progress = tqdm(total = len(dests.values()) * len(ms) * (num_random_iter if random_G0 else 1), desc="Running pfGW for each (destination, m)", leave=False)
  
  for (res_i, dest) in enumerate(dests.values()):
    M = MorseGraph.attribute_cost_matrix(src, dest)
    dest_net = dest.to_mpn(hist=hist, dist=dist)
    
    for m_i, m in enumerate(ms):
      
      if random_G0:
        min_dist = float('inf')
        min_coupling = None
        
        for _ in range(num_random_iter):
          if np.isclose(m, 1):
            c, d = ot.fGW(src_net, dest_net, M, alpha=alpha, random_G0=True, random_state=random_state)
          else:
            c, d = ot.fpGW(src_net, dest_net, m, M, alpha=alpha, random_G0=True, random_state=random_state)
          
          if d < min_dist:
            min_dist = d
            min_coupling = c
            
          progress.update(1)
        
        coupling = min_coupling
      else:
        if np.isclose(m, 1):
          coupling, _ = ot.fGW(src_net, dest_net, M, alpha=alpha)
        else:
          coupling, _ = ot.fpGW(src_net, dest_net, m, M, alpha=alpha)
          
        progress.update(1)
    
      results[m_i, res_i] = _max_match(src, dest, coupling)
  
  progress.close()
  return results

def plot_max_match_results(
  max_match_results,
  ms,
):
  x = np.asarray(ms)
  y = max_match_results.max(axis = 1)
  
  opt = []
  
  ds = np.diff(y) / np.diff(ms)
  
  i = 0
  while i < len(ds):
    if np.isclose(ds[i], 0):
      while i < len(ds) and np.isclose(ds[i], 0):
        i += 1
      
      opt.append((x[i], y[i]))
    
    i += 1
    
  fig, ax = plt.subplots()
  
  ax.plot(x, y)
  
  ymin, _ = ax.get_ybound()
  xmin, _ = ax.get_xbound()
    
  for (vx, vy) in opt:
    ax.add_line(Line2D([vx, vx], [ymin, vy], color='grey', linestyle='--'))
    ax.add_line(Line2D([xmin, vx], [vy, vy], color='grey', linestyle='--'))
    ax.plot(vx, vy, marker='o')
    
    
  x_ticks, y_ticks = zip(*opt)
  
  ax.set_xticks(x_ticks)
  ax.set_yticks(y_ticks)
  
  ax.set_xlabel('m', fontsize=20)
  ax.set_ylabel('maximum match distance', fontsize=20)
  
  return fig, opt

def tune_m(
  Ls,
  src,
  dests,
  m_step: float = 0.001,
  alpha: float = 0.5,
  hist = 'degree',
  dist = 'geo',
):
  results = np.zeros(shape=(len(Ls), len(dests)))
  src_net = src.to_mpn(hist=hist, dist=dist)
  
  progress = tqdm(total = len(dests.values()) * len(Ls), desc="Running pfGW for each (destination, m)", leave=False)
  
  for res_i, dest in enumerate(dests.values()):
    M = MorseGraph.attribute_cost_matrix(src, dest)
    dest_net = dest.to_mpn(hist=hist, dist=dist)
    
    for L_i, (m_star, L_star) in enumerate(Ls):
      m = m_star
      
      while m + m_step <= 1.0:
        m_next = m + m_step
        
        if np.isclose(m_next, 1):
          coupling, _ = ot.fGW(src_net, dest_net, M, alpha=alpha)
        else:
          coupling, _ = ot.fpGW(src_net, dest_net, m_next, M, alpha=alpha)
        
        L = _max_match(src, dest, coupling)
        
        if L > L_star:
          break
        else:
          m = m_next
      
      results[L_i, res_i] = m
      progress.update()
  
  progress.close()
  return results
