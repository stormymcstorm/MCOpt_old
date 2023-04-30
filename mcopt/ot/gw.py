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
  Coupling
)
from mcopt.ot.bregman import sinkhorn_scaling
from mcopt.ot.optim import NonConvergenceError
from mcopt.ot._fgw import gw_lp, fgw_lp, fgw_partial_lp

__all__ = ['GW', 'fGW', 'pfGW', 'pGW', 'Wasserstein', 'pWasserstein']

def make_random_G0(mu, nu, random_state=None, **kwargs):
  rvs = stats.beta(1e-1, 1e-1).rvs
  S = random(len(mu), len(nu), density=1, data_rvs=rvs, random_state=random_state)
  
  return sinkhorn_scaling(mu, nu, S.A, **kwargs)

def GW(
  X: MetricProbabilityNetwork,
  Y: MetricProbabilityNetwork,
  G0 : Optional[np.ndarray] = None,
  armijo=True,
  random_G0: bool = False,
  random_state=None,
) -> Tuple[np.ndarray, float]:
  d_X = X.metric
  d_Y = Y.metric
  
  mu = X.measure
  nu = Y.measure
  
  if random_G0:
    assert G0 is None
    G0 = make_random_G0(mu, nu, random_state=random_state)
  
  try:
    dist, raw_coupling = gw_lp(
      C1=d_X, 
      C2=d_Y,
      p = mu,
      q = nu,
      armijo=armijo
    )
  except(NonConvergenceError):
    dist, raw_coupling = gw_lp(
      C1=d_X, 
      C2=d_Y,
      p = mu,
      q = nu,
      armijo=False
    )
  
  coupling = Coupling(raw_coupling, X.space, Y.space)
  
  return coupling, dist ** 0.5

def fGW(
  X: MetricProbabilityNetwork,
  Y: MetricProbabilityNetwork,
  M: np.ndarray,
  alpha: float = 0.5,
  G0 : Optional[np.ndarray] = None,
  armijo=True,
  random_G0: bool = False,
  random_state=None,
) -> Tuple[np.ndarray, float]:
  d_X = X.metric
  d_Y = Y.metric
  
  mu = X.measure
  nu = Y.measure
  
  if random_G0:
    assert G0 is None
    G0 = make_random_G0(mu, nu, random_state=random_state)
  
  try:
    dist, raw_coupling = fgw_lp(
      M = np.square(M),
      C1=d_X, 
      C2=d_Y,
      p = mu,
      q = nu,
      alpha = alpha,
      armijo=armijo
    )
  except(NonConvergenceError):
    dist, raw_coupling = fgw_lp(
      M = np.square(M),
      C1=d_X, 
      C2=d_Y,
      p = mu,
      q = nu,
      alpha=alpha,
      armijo=False
    )
  
  coupling = Coupling(raw_coupling, X.space, Y.space)
  
  return coupling, dist ** 0.5
  # return coupling, dist

def Wasserstein(
  X: MetricProbabilityNetwork,
  Y: MetricProbabilityNetwork,
  M: np.ndarray,
  G0 : Optional[np.ndarray] = None,
  armijo=True,
  random_G0: bool = False,
  random_state=None,
) -> Tuple[np.ndarray, float]:
  d_X = X.metric
  d_Y = Y.metric
  
  mu = X.measure
  nu = Y.measure
  
  if random_G0:
    assert G0 is None
    G0 = make_random_G0(mu, nu, random_state=random_state)
  
  try:
    dist, raw_coupling = fgw_lp(
      M = np.square(M),
      C1=d_X, 
      C2=d_Y,
      p = mu,
      q = nu,
      alpha = 0,
      armijo=armijo
    )
  except(NonConvergenceError):
    dist, raw_coupling = fgw_lp(
      M = np.square(M),
      C1=d_X, 
      C2=d_Y,
      p = mu,
      q = nu,
      alpha=0,
      armijo=False
    )
  
  coupling = Coupling(raw_coupling, X.space, Y.space)
  
  return coupling, dist ** 0.5

def pfGW(
  X: MetricProbabilityNetwork,
  Y: MetricProbabilityNetwork,
  m: float,
  M: np.ndarray,
  alpha: float = 0.5,
  G0 : Optional[np.ndarray] = None,
  armijo=True,
  random_G0: bool = False,
  random_state=None,
) -> Tuple[np.ndarray, float]:
  d_X = X.metric
  d_Y = Y.metric
  
  mu = X.measure
  nu = Y.measure
  
  if random_G0:
    assert G0 is None
    G0 = make_random_G0(mu, nu, random_state=random_state)
  
  try:
    dist, raw_coupling = fgw_partial_lp(
      M = np.square(M),
      C1=d_X, 
      C2=d_Y,
      p = mu,
      q = nu,
      m = m,
      alpha = alpha,
      armijo=armijo,
    )
  except(NonConvergenceError):
    dist, raw_coupling = fgw_partial_lp(
      M = np.square(M),
      C1=d_X, 
      C2=d_Y,
      p = mu,
      q = nu,
      m = m,
      alpha=alpha,
      armijo=False,
      nb_dummies = 25,
    )
  
  coupling = Coupling(raw_coupling, X.space, Y.space)
  
  return coupling, dist ** 0.5

def pGW(
  X: MetricProbabilityNetwork,
  Y: MetricProbabilityNetwork,
  m: float,
  G0 : Optional[np.ndarray] = None,
  armijo=True,
  random_G0: bool = False,
  random_state=None,
) -> Tuple[np.ndarray, float]:
  d_X = X.metric
  d_Y = Y.metric
  
  mu = X.measure
  nu = Y.measure
  
  if random_G0:
    assert G0 is None
    G0 = make_random_G0(mu, nu, random_state=random_state)
  
  try:
    dist, raw_coupling = fgw_partial_lp(
      M = 0,
      C1=d_X, 
      C2=d_Y,
      p = mu,
      q = nu,
      m = m,
      alpha = 1,
      armijo=armijo,
       nb_dummies = 25,
    )
  except(NonConvergenceError):
    dist, raw_coupling = fgw_partial_lp(
      M = 0,
      C1=d_X, 
      C2=d_Y,
      p = mu,
      q = nu,
      m = m,
      alpha=1,
      armijo=False,
      nb_dummies = 25,
    )
  
  coupling = Coupling(raw_coupling, X.space, Y.space)
  
  return coupling, dist ** 0.5

def pWasserstein(
  X: MetricProbabilityNetwork,
  Y: MetricProbabilityNetwork,
  m: float,
  M: np.ndarray,
  G0 : Optional[np.ndarray] = None,
  armijo=True,
  random_G0: bool = False,
  random_state=None,
) -> Tuple[np.ndarray, float]:
  d_X = X.metric
  d_Y = Y.metric
  
  mu = X.measure
  nu = Y.measure
  
  if random_G0:
    assert G0 is None
    G0 = make_random_G0(mu, nu, random_state=random_state)
  
  try:
    dist, raw_coupling = fgw_partial_lp(
      M = np.square(M),
      C1=d_X, 
      C2=d_Y,
      p = mu,
      q = nu,
      m = m,
      alpha = 0,
      armijo=armijo
    )
  except(NonConvergenceError):
    dist, raw_coupling = fgw_partial_lp(
      M = np.square(M),
      C1=d_X, 
      C2=d_Y,
      p = mu,
      q = nu,
      m = m,
      alpha=0,
      armijo=False
    )
  
  coupling = Coupling(raw_coupling, X.space, Y.space)
  
  return coupling, dist ** 0.5
