from .measure_network import MeasureNetwork
import ot 

def GW(
  X_net : MeasureNetwork, 
  Y_net : MeasureNetwork, 
  **kwargs
):
  X, W_x, mu_x = X_net
  Y, W_y, mu_y = Y_net
  
  return ot.gromov.gromov_wasserstein(W_x, W_y, mu_x, mu_y, **kwargs)

def fGW(
  X_net : MeasureNetwork, 
  Y_net : MeasureNetwork, 
  M,
  **kwargs
):
  X, W_x, mu_x = X_net
  Y, W_y, mu_y = Y_net
  
  return ot.gromov.fused_gromov_wasserstein(M, W_x, W_y, mu_x, mu_y, **kwargs)

def pGW(
  X_net : MeasureNetwork, 
  Y_net : MeasureNetwork, 
  m,
  **kwargs
):
  X, W_x, mu_x = X_net
  Y, W_y, mu_y = Y_net
  
  return ot.partial.partial_gromov_wasserstein(W_x, W_y, mu_x, mu_y, m, **kwargs)