"""
Implementation of GW optimal transport
"""

from typing import Optional, Tuple

import numpy as np
from scipy import stats
from scipy.sparse import random
from ot.optim import emd

from mcopt.mm_space import (
  MetricProbabilityNetwork, 
  MetricMeasureNetwork,
  MetricProbabilityHypernetwork, 
  Coupling
)
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
  
  fd_X = f1(d_X)
  fd_Y = f2(d_Y)
  
  constC1 = np.dot(
    np.dot(fd_X, mu.reshape(-1, 1)),
    np.ones(fd_Y.shape[0]).reshape(1, -1)
  )
  # constC1 = np.dot(
  #   np.dot(f1(d_X), np.reshape(mu, (-1, 1))),
  #   np.ones((1, len(nu)))
  # )
  
  constC2 = np.dot(
    np.ones(fd_X.shape[0]).reshape(-1, 1),
    np.dot(nu.reshape(1, -1), fd_Y.T)
  )
  
  # constC2 = np.dot(
  #   np.ones((len(mu), 1)),
  #   np.dot(np.reshape(nu, (1, -1)), f2(d_Y).T)
  # )
  
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

def _ugw_compute_local_cost(pi, a, dx, b, dy, eps, rho, rho2, complete_cost=True):
  distxy = torch.einsum(
    "ij,kj->ik", dx, torch.einsum("kl,jl->kj", dy, pi)
  )
  kl_pi = torch.sum(
    pi * (pi / (a[:, None] * b[None, :]) + 1e-10).log()
  )
  if not complete_cost:
    return - 2 * distxy + eps * kl_pi

  mu, nu = torch.sum(pi, dim=1), torch.sum(pi, dim=0)
  distxx = torch.einsum("ij,j->i", dx ** 2, mu)
  distyy = torch.einsum("kl,l->k", dy ** 2, nu)

  lcost = (distxx[:, None] + distyy[None, :] - 2 * distxy) + eps * kl_pi

  if rho < float("Inf"):
    lcost = (
      lcost
      + rho
      * torch.sum(mu * (mu / a + 1e-10).log())
    )
  if rho2 < float("Inf"):
    lcost = (
      lcost
      + rho2
      * torch.sum(nu * (nu / b + 1e-10).log())
    )
  return lcost

def _ugw_log_translate_potential(u, v, lcost, a, b, mass, eps, rho, rho2):
  c1 = (
    -torch.cat((u, v), 0) / (mass * rho)
    + torch.cat((a, b), 0).log()
  ).logsumexp(dim=0) - torch.log(2 * torch.ones([1]))
  c2 = (
    (
      a.log()[:, None]
      + b.log()[None, :]
      + (
        (u[:, None] + v[None, :] - lcost)
        / (mass * eps)
      )
    ).logsumexp(dim=1).logsumexp(dim=0)
  )
  z = (0.5 * mass * eps) / (
    2.0 + 0.5 * (eps / rho) + 0.5 * (eps / rho2))
  k = z * (c1 - c2)
  return u + k, v + k

def _ugw_aprox_softmin(cost, a, b, mass, eps, rho, rho2):
  tau = 1.0 / (1.0 + eps / rho)
  tau2 = 1.0 / (1.0 + eps / rho2)

  def s_y(g):
    return (
      -mass
      * tau2
      * eps
      * (
        (g / (mass * eps) + b.log())[None, :]
        - cost / (mass * eps)
      ).logsumexp(dim=1)
    )

  def s_x(f):
    return (
      -mass
      * tau
      * eps
      * (
              (f / (mass * eps) + a.log())[:, None]
              - cost / (mass * eps)
      ).logsumexp(dim=0)
    )

  return s_x, s_y

def _ugw_log_sinkhorn(lcost, f, g, a, b, mass, eps, rho, rho2, nits_sinkhorn, tol_sinkhorn):
  # Initialize potentials by finding best translation
  if f is None or g is None:
    f, g = torch.zeros_like(a), torch.zeros_like(b)
  f, g = _ugw_log_translate_potential(f, g, lcost, a, b, mass, eps, rho, rho2)

  # perform Sinkhorn algorithm in LSE form
  s_x, s_y = _ugw_aprox_softmin(lcost, a, b, mass, eps, rho, rho2)
  for j in range(nits_sinkhorn):
    f_prev = f.clone()
    g = s_x(f)
    f = s_y(g)
    if (f - f_prev).abs().max().item() < tol_sinkhorn:
        break
  logpi = (
    (
      (f[:, None] + g[None, :] - lcost)
      / (mass * eps)
    )
    + a.log()[:, None]
    + b.log()[None, :]
  )
  return f, g, logpi

def GW(
  X: MetricProbabilityNetwork,
  Y: MetricProbabilityNetwork,
  M: np.ndarray = 0,
  alpha: float = 1,
  G0 : Optional[np.ndarray] = None,
  loss_fun: str = 'square_loss',
  armijo=True,
  log=False,
  verbose=False,
  random_G0: bool = False,
  random_state=None,
  **kwargs
) -> Tuple[np.ndarray, float]:
  d_X = X.metric
  d_Y = Y.metric
  
  mu = X.measure
  nu = Y.measure
  
  if random_G0:
    assert G0 is None
    G0 = make_random_G0(mu, nu, random_state=random_state)
  
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
  X: MetricProbabilityNetwork,
  Y: MetricProbabilityNetwork,
  M : np.ndarray,
  alpha: float = 0.5,
  **kwargs
) -> Tuple[np.ndarray, float]:
  return GW(X, Y, M, alpha=alpha, **kwargs)

def pGW(
  X: MetricProbabilityNetwork,
  Y: MetricProbabilityNetwork,
  m: float,
  M: np.ndarray = 0,
  alpha: float = 1,
  G0 : Optional[np.ndarray] = None,
  armijo=True,
  log=False,
  verbose=False,
  random_G0: bool = False,
  random_state=None,
  **kwargs
) -> Tuple[np.ndarray, float]:
  d_X = X.metric
  d_Y = Y.metric
  
  mu = X.measure
  nu = Y.measure
  
  if random_G0:
    assert G0 is None
    G0 = make_random_G0(mu, nu, random_state=random_state)
  elif G0 is None:
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
  X: MetricProbabilityNetwork,
  Y: MetricProbabilityNetwork,
  m: float,
  M: np.ndarray,
  alpha: float = 0.5,
  **kwargs
):
  return pGW(X, Y, m, M, alpha=alpha, **kwargs)

def cGW(
  X: MetricProbabilityHypernetwork,
  Y: MetricProbabilityHypernetwork,
  G0_nodes : Optional[np.ndarray] = None,
  G0_edges : Optional[np.ndarray] = None,
  loss_fun: str = 'square_loss',
  log=False,
  verbose=False,
  numItermax: int = 200,
  numItermaxEmd: int = 100000,
  random_G0: bool = False,
  random_state=None,
  **kwargs
):
  d_X = X.metric
  d_Y = Y.metric
  
  mu_X = X.node_measure
  mu_Y = Y.node_measure
  nu_X = X.edge_measure
  nu_Y = Y.edge_measure
  
  if random_G0:
    assert G0_nodes is None
    assert G0_edges is None
    
    G0_nodes = make_random_G0(mu_X, mu_Y, random_state=random_state)
    G0_edges = make_random_G0(nu_X, nu_Y, random_state=random_state)
  
  constC_nodes, hC1_nodes, hC2_nodes = _gw_init_matrix(d_X, d_Y, nu_X, nu_Y, loss_fun)
  constC_edges, hC1_edges, hC2_edges = _gw_init_matrix(d_X.T, d_Y.T, mu_X, mu_Y, loss_fun)
 
  if log:
    log = {'cost': []}
    
  if G0_nodes is None:
    G0_nodes = np.outer(mu_X, mu_Y)
  else:
    assert G0_nodes.ndim == 2
    assert G0_nodes.shape[0] == mu_X.shape[0]
    assert G0_nodes.shape[1] == mu_Y.shape[0]
    
  if G0_edges is None:
    G0_edges = np.outer(nu_X, nu_Y)
  else:
    assert G0_edges.ndim == 2
    assert G0_edges.shape[0] == nu_X.shape[0]
    assert G0_edges.shape[1] == nu_Y.shape[0]
  
  G_nodes = G0_nodes
  G_edges = G0_edges
  
  cost = np.inf
  
  it = 0
  
  if verbose:
    print('{:5s}|{:12s}'.format('It.', 'Loss') + '\n' + '-' * 32)
    print('{:5d}|{:8e}'.format(it, cost))
  
  loop = True
  while loop:
    it += 1
    Gn_old = G_nodes
    Ge_old = G_edges
    cost_old = cost
    
    M = constC_nodes - np.dot(hC1_nodes, G_edges).dot(hC2_nodes.T)
    G_nodes = emd(mu_X, mu_Y, M, numItermax=numItermaxEmd, log=False)
    
    M = constC_edges - np.dot(hC1_edges, G_nodes).dot(hC2_edges.T)
    G_edges = emd(nu_X, nu_Y, M, numItermax=numItermaxEmd, log=False)
    
    cost = np.sum(M * G_edges)
    
    if log:
      log['cost'].append(cost)
      
    if verbose:
      print('{:5s}|{:12s}'.format('It.', 'Loss') 
            + '\n' + '-' * 32)
      print('{:5d}|{:8e}'.format(it, cost))
      
    if it >= numItermax:
      break
    
    delta = np.linalg.norm(G_nodes - Gn_old) + np.linalg.norm(G_edges - Ge_old)
    
    if delta < 1e-16 or np.abs(cost_old - cost) < 1e-7:
      break
  
  node_coupling = Coupling(G_nodes, X.node_space, Y.node_space)
  edge_coupling = Coupling(G_edges, X.edge_space, Y.edge_space)
  dist = cost
    
  if log:
    return node_coupling, edge_coupling, dist, log
  else:
    return node_coupling, edge_coupling, dist
  
def uGW(
  X: MetricProbabilityNetwork,
  Y: MetricProbabilityNetwork,
  G0: Optional[np.ndarray] = None,
  rho: float = float('inf'),
  rho2: Optional[float] = None,
  eps: float = 1,
  random_G0: bool = False,
  random_state = None,
  numItermax: int = 200,
):
  if rho2 is None:
    rho2 = rho
    
  d_X = X.metric
  d_Y = Y.metric
  
  mu = X.measure
  nu = Y.measure
  
  if random_G0:
    assert G0 is None
    G0 = make_random_G0(mu, nu, random_state=random_state)
  elif G0 is None:
    G0 = np.outer(mu, nu)
    
  d_X = torch.from_numpy(d_X)
  d_Y = torch.from_numpy(d_Y)
  mu = torch.from_numpy(mu)
  nu = torch.from_numpy(nu)
  G0 = torch.from_numpy(G0)
  
  logpi = (G0 + 1e-30).log()
  logpi_prev = torch.zeros_like(logpi)
  lcost = float('inf')
  
  up, vp = None, None
  for _ in range(numItermax):
    logpi_prev = logpi.clone()
    
    lcost = _ugw_compute_local_cost(
      logpi.exp(),
      mu,
      d_X,
      nu,
      d_Y,
      eps,
      rho,
      rho2
    )
    
    logmp = logpi.logsumexp(dim=(0,1))
    
    up, vp, logpi = _ugw_log_sinkhorn(
      lcost, up, vp, mu, nu, logmp.exp() + 1e-10, eps, rho, rho2,
      numItermax, 1e-6
    )
    
    if torch.any(torch.isnan(logpi)):
      raise Exception(
        f'Solver got NaN plan with params (eps, rho, rho2) '
        f' = {eps, rho, rho2}. Try increasing argument eps.'
      )
      
    logpi = (
      0.5 * (logmp - logpi.logsumexp(dim=(0, 1)))
      + logpi
    )
    
    if (logpi - logpi_prev).abs().max().item() < 1e-6:
      break
  
  dist = lcost
  raw_coupling = logpi.exp().numpy()
  
  coupling = Coupling(raw_coupling, X.space, Y.space)
  
  return coupling, dist
  
    