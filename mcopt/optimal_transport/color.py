
from typing import Dict, TypeVar
import numpy as np
from .ot import MeasureNetwork

N = TypeVar('N')

def color_transfer(
  src_network : MeasureNetwork, 
  dst_network : MeasureNetwork,
  coupling : np.ndarray, # (# src_nodes, # dst_nodes)
  src_color : Dict[N, float]
) -> Dict[N, float]:
  X, _, _ = src_network
  Y, _, _ = dst_network
  
  src_node_rev_map = {i : n for i, n in enumerate(X)}
  
  dst_color = {}
  
  for i, n in enumerate(Y):
    src_i = coupling[:,i].argmax()
    src = src_node_rev_map[src_i]
    
    dst_color[n] = src_color[src]
  
  return dst_color

def color_transfer_expected_color(
  src_network : MeasureNetwork, 
  dst_network : MeasureNetwork,
  coupling : np.ndarray, # (# src_nodes, # dst_nodes)
  src_color : Dict[N, float]
) -> Dict[N, float]:
  """
  Computes the color transfer from the nodes in `src_network` to the nodes in 
  `dst_network` as the "expected" color.
  
  Specifically, if `src_color` is thought of as a function from a source node to
  it's color, and `X_n` is a random variable representing the probability of 
  matching a node `n` in `dst_network` with each node in `src_network`, then
  `dst_color = E[src_color(X_n)]`.
  
  Args:
    src_network (MeasureNetwork): The source network.
    dst_network (MeasureNetwork): The destination network.
    coupling (np.ndarray): The coupling matrix produced from optimal transport.
    src_color (Dict[N, float]): A mapping from nodes in `src_network` to their
    color.
    
  Returns:
    Dict[N, float]: A mapping from nodes in `dst_network` to their color.
  """
  
  X, _, _ = src_network
  Y, _, _ = dst_network
  
  if coupling.shape != (len(X), len(Y)):
    raise ValueError(f"Expected coupling to have shape ({len(X)}, {len(Y)})")
  
  dst_color = {}
  
  src_color_arr = np.array([src_color[n] for n in X])
  
  for i, n in enumerate(Y):
    # Get the distribution for the coupling between n and X[j] for each j
    couples = coupling[:,i]
    
    assert couples.shape[0] == len(src_color_arr)
    
    dst_color[n] = src_color_arr.dot(couples)
  
  return dst_color