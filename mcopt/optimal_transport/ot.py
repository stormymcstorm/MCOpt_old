
from typing import Tuple
import numpy as np
from .gwd import GromovWassersteinDistance
from ot.optim import cg

MeasureNetwork = Tuple[np.ndarray, np.ndarray, np.ndarray]

def optimal_transport(X_net : MeasureNetwork, Y_net : MeasureNetwork, dist_type = 'gw'):
  X, W_x, mu_x = X_net
  Y, W_y, mu_y = Y_net
  
  if dist_type == 'gw':
    dist_func = GromovWassersteinDistance(W_x, mu_x, W_y, mu_y)
  else:
    raise ValueError(f'Unknown distance type {dist_type}')
  
  coupling = cg(mu_x, mu_y, 0, 1, dist_func, dist_func.grad)
  
  coupling_map = {u_x : {v_y : coupling[i][j] for j, v_y in enumerate(Y)} for i, u_x in enumerate(X)}
    
  return coupling, coupling_map