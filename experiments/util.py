
from mcopt.morse_graph import (
  MorseGraph,
  color_by_position,
)

from mcopt.opt import (
  MeasureNetwork,
  color_transfer,
  random_gamma_init,
)

def find_good_init(
  f,
  src_net: MeasureNetwork,
  dest_net: MeasureNetwork,
  random_state = None,
  n_iter = 1,
):
  coupling = None
  min_dist = float('inf')
  
  for _ in range(n_iter):
    _, _, p = src_net
    _, _, q = dest_net
    
    G0 = random_gamma_init(p, q, random_state)
    
    c, dist = f(G0)
    
    if dist < min_dist:
      min_dist = dist
      coupling = c
  
  return coupling
  

# Execute `run` function for each profile
# `run` is expected to be of form `(MeasureNetwork, MeasureNetwork, profile, G0) -> (coupling, dist)`
def param_comp(
  profiles, 
  run, 
  src : MorseGraph, 
  dest : MorseGraph, 
  src_node_color,
  random_state = None,
  n_iter = 1,
):    
  for profile in profiles:
    weight = profile.get('weight', None)
    hist = profile.get('hist', None)
    
    src_net = src.to_measure_network(weight=weight, hist=hist)
    dest_net = dest.to_measure_network(weight=weight, hist=hist)
    
    # coupling = None
    # min_dist = float('inf')
    
    # for _ in range(n_iter):
    #   _, _, p = src_net
    #   _, _, q = dest_net
      
    #   G0 = random_gamma_init(p, q, random_state)
      
    #   c, dist = run(src_net, dest_net, profile, G0)
      
    #   if dist < min_dist:
    #     min_dist = dist
    #     coupling = c
    
    def f(G0):
      return run(src_net, dest_net, profile, G0)
      
    
    coupling = find_good_init(f, src_net, dest_net, random_state=random_state, n_iter=n_iter)
    
    # coupling, _ = run(src_net, dest_net, profile, G0=None)
    
    dest_node_color = color_transfer(
      src_net,
      dest_net,
      coupling,
      src_node_color
    )
    
    profile['coupling'] = coupling
    profile['node_color'] = dest_node_color

    