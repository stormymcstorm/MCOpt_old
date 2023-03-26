from typing import Set, Dict

import numpy as np
import networkx as nx
import hypernetx as hnx
import matplotlib

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
  
  def to_hyper(self, mode='simple'):
    edges = {}

    next_entity = max(self.nodes()) + 1

    if mode == 'simple':
      for edge in self.edges:
        edges[next_entity] = [edge[0], edge[1]]
        next_entity += 1
    else:
      raise ValueError(f'Unrecognized mode {mode}')
    
    hyper_graph = MorseHyperGraph(self.critical_nodes, edges)
    
    for n in hyper_graph.nodes:
      hyper_graph.nodes[n].__dict__.update(self.nodes(data=True)[n])
    
    return hyper_graph
  
class MorseHyperGraph(hnx.Hypergraph):
  critical_nodes: Set[int]
  
  def __init__(self, critical_nodes, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.critical_nodes = critical_nodes
    
  def node_color_by_position(self) -> Dict[int, float]:
    return {n : np.linalg.norm(data.pos2) for n, data in self.nodes.elements.items()}
  
  def edge_color_by_node(self, node_color: Dict[int, float]) -> Dict[int, float]:
    return {
      e : sum([node_color[n] for n in nodes]) / len(nodes) 
      for e, nodes in self.incidence_dict.items()
    }
    
  def draw(
    self,
    ax,
    cmap='viridis',
    node_color = None,
    edge_color = None,
    edge_alpha = 0.5,
    node_size=40,
    critical_scale = 3,
    **kwargs,
  ):
    if isinstance(cmap, str):
      cmap = matplotlib.colormaps[cmap]
      
    critical_scale = critical_scale / 3 * 1.3  
    
    node_radius = node_size / 80
    node_radius = {
      n : node_radius * critical_scale if n in self.critical_nodes else node_radius
      for n in self.nodes
    }
    
    if not node_color:
      node_color = self.node_color_by_position()

    if type(node_color) is dict:
      node_color_map = node_color
      node_color = np.array([node_color[n] for n in self.nodes])
    else:
      node_color_map = {i : v for i, v in enumerate(node_color)}
      node_color = np.asarray(node_color)
      
    node_color /= node_color.max()
    node_facecolor = [cmap(v) for v in node_color]
    
    if not edge_color:
      edge_color = self.edge_color_by_node(node_color_map)
      
    if type(edge_color) is dict:
      edge_color = np.array([edge_color[e] for e in self.edges])
    else:
      edge_color = np.asarray(edge_color)

    edge_color /= edge_color.max()
    edge_facecolor = [cmap(v, alpha=edge_alpha) for v in edge_color]
    
    hnx.rubber_band.draw(
      self,
      ax=ax,
      pos = {n : data.pos2 for n, data in self.nodes.elements.items()},
      node_radius = node_radius,
      with_color = False,
      with_edge_labels = False,
      with_node_labels = False,
      nodes_kwargs = {
        'facecolors': node_facecolor,
        'edgecolors': 'black'
      },
      edges_kwargs = {
        'edgecolors': (0, 0, 0, 0),
        'facecolors': edge_facecolor,
        'dr': 0.1
      },
      **kwargs
    )