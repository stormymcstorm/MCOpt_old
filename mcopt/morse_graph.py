from typing import Set, Dict

import numpy as np
import networkx as nx

class MorseGraph(nx.Graph):
  critical_nodes: Set[int]
  
  def __init__(self, critical_nodes):
    super().__init__()
    self.critical_nodes = critical_nodes

  def sample(self, rate, mode='step'):
    graph = MorseGraph(self.critical_nodes)
    
    visited = set()
    
    def dfs(start, node, length=0):
      if node in visited and node not in self.critical_nodes:
        return
      
      visited.add(node)
      
      for n in self.neighbors(node):
        if n in visited:
          continue
        
        if n in self.critical_nodes:
          graph.add_node(n, **self.nodes(data=True)[n])

          assert graph.has_node(start)
          graph.add_edge(start, n)
          
          continue
        
        if mode == 'step':
          new_length = length + 1
        if mode == 'geo':
          new_length = length + np.linalg.norm(self.nodes(data='pos2')[n] - self.nodes(data='pos2')[node])
          
        if new_length > rate:
          graph.add_node(n, **self.nodes(data=True)[n])
          
          assert graph.has_node(start)
          graph.add_edge(start, n)
          
          dfs(n, n)
        else:
          dfs(start, n, new_length)
          
    for crit in self.critical_nodes:
      graph.add_node(crit, **self.nodes(data=True)[crit])
      dfs(crit, crit)
      
    assert nx.is_connected(graph)
    assert all(graph.has_node(n) for n in self.critical_nodes)
    
    return graph

  def node_color_by_position(self) -> Dict[int, float]:
    return {n : np.linalg.norm(pos) for n, pos in self.nodes(data='pos2')}

  def draw(
    self,
    ax,
    cmap='viridis',
    node_color = None,
    node_size=40,
    critical_scale = 3,
    **kwargs,
  ):
    if not node_color:
      node_color = self.node_color_by_position()
      
    if type(node_color) is dict:
      node_color = np.array([node_color[n] for n in self.nodes()])
    
    node_size = np.array([
      node_size * critical_scale if n in self.critical_nodes else node_size
      for n in self.nodes()
    ])
    
    # Allow for nodes that should be given a "bad color" to have `nan` values.
    vmin = np.nanmin(node_color)
    vmax = np.nanmax(node_color)
    
    nx.draw(
      self,
      ax=ax,
      cmap = cmap,
      pos = self.nodes(data='pos2'),
      node_color = node_color,
      node_size = node_size,
      
      vmin=vmin,
      vmax=vmax,
      alpha=[1],
      **kwargs
    )
  