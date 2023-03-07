from .measure_network import MeasureNetwork
import ot 
import numpy as np
from .optim import cg, pcg, NonConvergenceError

def init_matrix(C1, C2, p, q, loss_fun='square_loss'):
  
  if loss_fun == 'square_loss':
    def f1(a):
      return a**2
    
    def f2(b):
      return b**2
    
    def h1(a):
      return a
    
    def h2(b):
      return 2*b
  
  constC1 = np.dot(
    np.dot(f1(C1), np.reshape(p, (-1, 1))),
    np.ones((1, len(q))),
  )
  constC2 = np.dot(
    np.ones((len(p), 1)),
    np.dot(np.reshape(q, (1, -1)), f2(C2).T)
  )
  
  constC = constC1 + constC2
  hC1 = h1(C1)
  hC2 = h2(C2)
  
  return constC, hC1, hC2

def tensor_product(constC, hC1, hC2, T):
  A = -np.dot(np.dot(hC1, T), hC2.T)
  tens = constC + A
  
  return tens

def gwloss(constC, hC1, hC2, T):
  tens = tensor_product(constC, hC1, hC2, T)
  
  return np.sum(tens * T)

def gwgrad(constC, hC1, hC2, T):
  return 2*tensor_product(constC, hC1, hC2, T)
  
  
# CHECKME: does this assume square loss
def gwgrad_partial(C1, C2, T):
  cC1 = np.dot(C1 ** 2 / 2, np.dot(T, np.ones(C2.shape[0]).reshape(-1, 1)))
  cC2 = np.dot(np.dot(np.ones(C1.shape[0]).reshape(1, -1), T), C2 ** 2 / 2)
  constC = cC1 + cC2
  A = -np.dot(C1, T).dot(C2.T)
  tens = constC + A
  return tens * 2

def gwloss_partial(C1, C2, T):
  g = gwgrad_partial(C1, C2, T) * 0.5
  return np.sum(g * T)

def GW(
  X_net : MeasureNetwork, 
  Y_net : MeasureNetwork, 
  M : np.ndarray = 0,
  alpha : float = 1,
  armijo=True,
  log = False,
  verbose=False,
  **kwargs
):
  
  X, C1, p = X_net
  Y, C2, q = Y_net
  
  constC, hC1, hC2 = init_matrix(C1, C2, p, q)
  
  G0 = p[:,None] * q[None,:]
  
  def f(G):
    return gwloss(constC, hC1, hC2, G)
  
  def df(G):
    return gwgrad(constC, hC1, hC2, G)
  
  try:
    out = cg(
      p, q, M, alpha, f, df, G0, 
      log=log, verbose=verbose, armijo=armijo, C1=C1, C2=C2, constC=constC, 
      **kwargs
    )
  except (NonConvergenceError):
    if armijo:
      if verbose:
        print('Fail to converge. Turning off armijo research. Using closed form.')
        
      out = cg(
        p, q, M, alpha, f, df, G0, 
        log=log, verbose=verbose, armijo=False, C1=C1, C2=C2, constC=constC, 
        **kwargs
      )
    else:
      raise
    
  if log:
    res, log = out
    
    dist = gwloss(constC, hC1, hC2, res)
    log['gw_dist'] = dist
    
    return res, dist, log
  else:
    res = out
  
    dist = gwloss(constC, hC1, hC2, res)
    
    return res, dist
  
def fGW(
  X_net : MeasureNetwork,
  Y_net : MeasureNetwork,
  M : np.ndarray,
  alpha : float = 0.5,
  log = False,
  **kwargs
):
  return GW(X_net, Y_net, M=M, alpha=alpha, log=log)

def pGW(
  X_net : MeasureNetwork, 
  Y_net : MeasureNetwork, 
  m,
  M : np.ndarray = 0,
  alpha : float = 1,
  armijo=True,
  log = False,
  verbose=False,
  **kwargs
):
  X, C1, p = X_net
  Y, C2, q = Y_net
  
  G0 = np.outer(p, q)
  
  if np.sum(G0) > m:
    G0 *= (m / np.sum(G0))
  
  cC1 = np.dot(C1 ** 2 / 2, np.dot(G0, np.ones(C2.shape[0]).reshape(-1, 1)))
  cC2 = np.dot(np.dot(np.ones(C1.shape[0]).reshape(1, -1), G0), C2 ** 2 / 2)
  constC = cC1 + cC2
  
  def f(G):
    return gwloss_partial(C1, C2, G)
  
  def df(G):
    return gwgrad_partial(C1, C2, G)
  
  try:
    out = pcg(
      p, q, M, alpha, m, f, df, 
      log=log, verbose=verbose, armijo=armijo, C1=C1, C2=C2, constC=constC, 
      **kwargs
    )
  except (NonConvergenceError):
    if armijo:
      if verbose:
        print('Fail to converge. Turning off armijo research. Using closed form.')
        
      out = pcg(
        p, q, M, alpha, m, f, df, 
        log=log, verbose=verbose, armijo=False, C1=C1, C2=C2, constC=constC, 
        **kwargs
      )
    else:
      raise
    
  if log:
    res, log = out
    
    dist = gwloss_partial(C1, C2, res)
    log['gw_dist'] = dist
    
    return res, dist, log
  else:
    res = out
  
    dist = gwloss_partial(C1, C2, res)
    
    return res, dist

# def GW(
#   X_net : MeasureNetwork, 
#   Y_net : MeasureNetwork, 
#   **kwargs
# ):
#   X, W_x, mu_x = X_net
#   Y, W_y, mu_y = Y_net
  
#   return ot.gromov.gromov_wasserstein(W_x, W_y, mu_x, mu_y, **kwargs)

# def fGW(
#   X_net : MeasureNetwork, 
#   Y_net : MeasureNetwork, 
#   M,
#   **kwargs
# ):
#   X, W_x, mu_x = X_net
#   Y, W_y, mu_y = Y_net
  
#   return ot.gromov.fused_gromov_wasserstein(M, W_x, W_y, mu_x, mu_y, **kwargs)

def pGW_old(
  X_net : MeasureNetwork, 
  Y_net : MeasureNetwork, 
  m,
  **kwargs
):
  X, W_x, mu_x = X_net
  Y, W_y, mu_y = Y_net
  
  return ot.partial.partial_gromov_wasserstein(W_x, W_y, mu_x, mu_y, m, **kwargs)