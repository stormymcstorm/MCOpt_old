import os
import math

import matplotlib.pyplot as plt

def file_ext(name):
  res = os.path.basename(name).rsplit(os.path.extsep, maxsplit=1)

  if len(res) < 2:
    return None
  
  return res[1]

def sort_files(files):
  files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
  
  return files


def layout_like_square(area, max_ratio=5/2):  
  min_diff = float('inf')
  result = None
  
  for i in range(1, int(area**0.5) + 1):
    if area % i == 0:
      w = i
      h = area//w
      
      diff = abs(w - h)
      
      if diff < min_diff:
        min_diff = diff
        result = (w, h)
  
  w, h = result
  
  if h > w:
    w, h = h, w
    
  while w / h > max_ratio:
    w -= 1
    h = math.ceil(area / w)
    
  return w, h

def draw_graphs(
  graphs,
  ax_prefix: str,
  cmap='cool',
  node_size=40,
  critical_scale=3,
  fontsize=40,
  title=None, 
):
  w, h = layout_like_square(len(graphs))
  
  assert (w * h >= len(graphs))
  fig, axes = plt.subplots(h, w, figsize=(w * 12, h * 12))
  
  for ax in axes.ravel():
    ax.set_axis_off()
  
  for i, (frame, ax) in enumerate(zip(graphs, axes.ravel())):
    frame.draw(
      ax=ax,
      cmap=cmap,
      node_size=node_size,
      critical_scale=critical_scale
    )
    ax.set_title(f'{ax_prefix} {i + 1}', fontsize=fontsize//2)
    
  if title:
    fig.suptitle(title, fontsize=fontsize)
    
  return fig