"""
Implementation of GW optimal transport
"""

from typing import Optional, Tuple

import numpy as np
import torch
from scipy import stats
from scipy.sparse import random
from unbalancedgw import log_ugw_sinkhorn

from mcopt.ot.mm import MetricProbabilitySpace, MetricMeasureSpace, Coupling
from mcopt.ot.bregman import sinkhorn_scaling
from mcopt.ot.optim import cg, pcg, NonConvergenceError

def make_random_G0(mu, nu, random_state=None, **kwargs):
  rvs = stats.beta(1e-1, 1e-1).rvs
  S = random(len(mu), len(nu), density=1, data_rvs=rvs, random_state=random_state)
  
  return sinkhorn_scaling(mu, nu, S.A, **kwargs)

def _gw_init_matrix(d_X, d_Y, mu, nu, loss_fun):
  if loss_fun == 'square_loss':
    def f1(a):
      return a**2
    
    def f2(b):
      return b**2
    
    def h1(a):
      return a
    
    def h2(b):
      return 2 * b
  else:
    raise ValueError(f'Unrecognized loss_fun: {loss_fun}')
  
  constC1 = np.dot(
    np.dot(f1(d_X), np.reshape(mu, (-1, 1))),
    np.ones((1, len(nu)))
  )
  
  constC2 = np.dot(
    np.ones((len(mu), 1)),
    np.dot(np.reshape(nu, (1, -1)), f2(d_Y).T)
  )
  
  constC = constC1 + constC2
  hC1 = h1(d_X)
  hC2 = h2(d_Y)
  
  return constC, hC1, hC2

def _gw_tensor_product(constC, hC1, hC2, T):
  A = -np.dot(np.dot(hC1, T), hC2.T)
  tens = constC + A
  
  return tens

def _gw_loss(constC, hC1, hC2, T):
  tens = _gw_tensor_product(constC, hC1, hC2, T)
  
  return np.sum(tens * T)

def _gw_grad(constC, hC1, hC2, T):
  return 2*_gw_tensor_product(constC, hC1, hC2, T)

def _pgw_grad(C1, C2, T):
  cC1 = np.dot(C1 ** 2 / 2, np.dot(T, np.ones(C2.shape[0]).reshape(-1, 1)))
  cC2 = np.dot(np.dot(np.ones(C1.shape[0]).reshape(1, -1), T), C2 ** 2 / 2)
  constC = cC1 + cC2
  A = -np.dot(C1, T).dot(C2.T)
  tens = constC + A
  return tens * 2

def _pgw_loss(C1, C2, T):
  g = _pgw_grad(C1, C2, T) * 0.5
  return np.sum(g * T)

def GW(
  X: MetricProbabilitySpace,
  Y: MetricProbabilitySpace,
  M: np.ndarray = 0,
  alpha: float = 1,
  G0 : Optional[np.ndarray] = None,
  loss_fun: str = 'square_loss',
  armijo=True,
  log=False,
  verbose=False,
  random_G0: bool = False,
  random_state=None,
  num_rand_iter: int = 10,
  **kwargs
) -> Tuple[np.ndarray, float]:
  if random_G0:
    assert G0 is None
    
    min_dist = float('inf')
    coupling = None
    
    assert num_rand_iter > 0
    
    for _ in range(num_rand_iter):
      mu = X.measure
      nu = Y.measure
      
      G0 = make_random_G0(mu, nu, random_state=random_state)
      
      c, dist = GW(
        X, Y, M, alpha, 
        G0=G0, 
        armijo=armijo, 
        log=log, 
        verbose=verbose, 
        random_G0=False, 
        **kwargs
      )
      
      if dist < min_dist:
        coupling = c
        min_dist = dist
    
    return coupling, min_dist
  
  d_X = X.metric
  d_Y = Y.metric
  
  mu = X.measure
  nu = Y.measure
  
  constC, hC1, hC2 = _gw_init_matrix(d_X, d_Y, mu, nu, loss_fun)
  
  def f(G):
    return _gw_loss(constC, hC1, hC2, G)
  
  def df(G):
    return _gw_grad(constC, hC1, hC2, G)
  
  try:
    out = cg(
      mu, nu, M, alpha, f, df,
      G0=G0, log=log, verbose=verbose, armijo=armijo, C1=d_X, C2=d_Y, constC=constC,
      **kwargs
    )
  except NonConvergenceError:
    if armijo:
      if verbose:
        print('Fail to converge. Turning off armijo research. Using closed form.')
        
      out = cg(
        mu, nu, M, alpha, f, df,
        G0=G0, log=log, verbose=verbose, armijo=False, C1=d_X, C2=d_Y, constC=constC,
        **kwargs
      )
    else:
      raise
    
  if log:
    raw_coupling, log = out
    
    dist = _gw_loss(constC, hC1, hC2, raw_coupling)
    log['gw_dist']=dist
    
    coupling = Coupling(raw_coupling, X.space, Y.space)
    
    return coupling, dist, log
  
  raw_coupling = out
  dist = _gw_loss(constC, hC1, hC2, raw_coupling)
  
  coupling = Coupling(raw_coupling, X.space, Y.space)
  
  return coupling, dist

def fGW(
  X: MetricProbabilitySpace,
  Y: MetricProbabilitySpace,
  M : np.ndarray,
  alpha: float = 0.5,
  **kwargs
) -> Tuple[np.ndarray, float]:
  return GW(X, Y, M, alpha=alpha, **kwargs)

def pGW(
  X: MetricProbabilitySpace,
  Y: MetricProbabilitySpace,
  m: float,
  M: np.ndarray = 0,
  alpha: float = 1,
  G0 : Optional[np.ndarray] = None,
  armijo=True,
  log=False,
  verbose=False,
  random_G0: bool = False,
  random_state=None,
  num_rand_iter: int = 10,
  **kwargs
) -> Tuple[np.ndarray, float]:
  if random_G0:
    assert G0 is None
    
    min_dist = float('inf')
    coupling = None
    
    assert num_rand_iter > 0
    
    for _ in range(num_rand_iter):
      mu = X.measure
      nu = Y.measure
      
      G0 = make_random_G0(mu, nu, random_state=random_state)
      
      c, dist = pGW(
        X, Y, m, M, alpha, 
        G0=G0, 
        armijo=armijo, 
        log=log, 
        verbose=verbose, 
        random_G0=False, 
        **kwargs
      )
      
      if dist < min_dist:
        coupling = c
        min_dist = dist
    
    return coupling, min_dist
  
  d_X = X.metric
  d_Y = Y.metric
  
  mu = X.measure
  nu = Y.measure
  
  G0 = np.outer(mu, nu)
  
  if np.sum(G0) > m:
    G0 *= (m / np.sum(G0))
  
  cC1 = np.dot(d_X ** 2 / 2, np.dot(G0, np.ones(d_Y.shape[0]).reshape(-1, 1)))
  cC2 = np.dot(np.dot(np.ones(d_X.shape[0]).reshape(1, -1), G0), d_Y ** 2 / 2)
  constC = cC1 + cC2
  
  def f(G):
    return _pgw_loss(d_X, d_Y, G)
  
  def df(G):
    return _pgw_grad(d_X, d_Y, G)
  
  try:
    out = pcg(
      mu, nu, M, alpha, m, f, df,
      G0=G0, log=log, verbose=verbose, armijo=armijo, C1=d_X, C2=d_Y, constC=constC,
      **kwargs
    )
  except NonConvergenceError:
    if armijo:
      if verbose:
        print('Fail to converge. Turning off armijo research. Using closed form.')
        
      out = pcg(
        mu, nu, M, alpha, m, f, df,
        G0=G0, log=log, verbose=verbose, armijo=False, C1=d_X, C2=d_Y, constC=constC,
        **kwargs
      )
    else:
      raise
    
  if log:
    raw_coupling, log = out
    
    dist = _pgw_loss(d_X, d_Y, raw_coupling)
    log['gw_dist']=dist
    
    coupling = Coupling(raw_coupling, X.space, Y.space)
    
    return coupling, dist, log
  
  raw_coupling = out
  dist = _pgw_loss(d_X, d_Y, raw_coupling)
  
  coupling = Coupling(raw_coupling, X.space, Y.space)
  
  return coupling, dist

def fpGW(
  X: MetricProbabilitySpace,
  Y: MetricProbabilitySpace,
  m: float,
  M: np.ndarray,
  alpha: float = 0.5,
  **kwargs
):
  return pGW(X, Y, m, M, alpha=alpha, **kwargs)

def uGW(
  X: MetricMeasureSpace,
  Y: MetricMeasureSpace,
  G0 : Optional[np.ndarray] = None,
  **kwargs
):
  d_X = torch.from_numpy(X.metric)
  d_Y = torch.from_numpy(Y.metric)
  
  mu = torch.from_numpy(X.measure)
  nu = torch.from_numpy(Y.measure)
  
  if G0 is not None:
    G0 = torch.from_numpy(G0)
  
  raw_coupling = log_ugw_sinkhorn(
    mu, d_X, nu, d_Y, init=G0, **kwargs
  )
  
  return Coupling(raw_coupling, X.space, Y.space)