
from mcopt.morse_graph import (
  MorseGraph,
  color_by_position,
)

from mcopt.opt import (
  color_transfer,
  random_gamma_init,
)

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
    
    coupling = None
    min_dist = float('inf')
    
    for _ in range(n_iter):
      _, _, p = src_net
      _, _, q = dest_net
      
      G0 = random_gamma_init(p, q, random_state)
      
      c, dist = run(src_net, dest_net, profile, G0)
      
      if dist < min_dist:
        min_dist = dist
        coupling = c
    
    # coupling, _ = run(src_net, dest_net, profile, G0=None)
    
    dest_node_color = color_transfer(
      src_net,
      dest_net,
      coupling,
      src_node_color
    )
    
    profile['coupling'] = coupling
    profile['node_color'] = dest_node_color

    