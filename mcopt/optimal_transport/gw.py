from .measure_network import MeasureNetwork
import ot 

def gromov_wasserstein(X_net, Y_net, **kwargs):
  X, W_x, mu_x = X_net
  Y, W_y, mu_y = Y_net
  
  return ot.gromov.gromov_wasserstein(W_x, W_y, mu_x, mu_y, **kwargs)

def fused_gromov_wasserstein(M, X_net, Y_net, **kwargs):
  X, W_x, mu_x = X_net
  Y, W_y, mu_y = Y_net
  
  return ot.gromov.fused_gromov_wasserstein(M, W_x, W_y, mu_x, mu_y, **kwargs)
