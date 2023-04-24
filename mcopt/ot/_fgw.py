# -*- coding: utf-8 -*-

"""
From https://github.com/tdavislab/GWMT/blob/main/Tracking/FGW.py
"""

import numpy as np
import mcopt.ot.optim as optim
# from utils import dist,reshaper
# from bregman import sinkhorn_scaling
from scipy import stats
from scipy.sparse import random

class StopError(Exception):
    pass

def init_matrix(C1,C2,p,q,loss_fun='square_loss'):
  """ Return loss matrices and tensors for Gromov-Wasserstein fast computation
  Returns the value of \mathcal{L}(C1,C2) \otimes T with the selected loss
  function as the loss function of Gromow-Wasserstein discrepancy.
  The matrices are computed as described in Proposition 1 in [1]
  Where :
      * C1 : Metric cost matrix in the source space
      * C2 : Metric cost matrix in the target space
      * T : A coupling between those two spaces
  The square-loss function L(a,b)=(1/2)*|a-b|^2 is read as :
      L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
          * f1(a)=(a^2)
          * f2(b)=(b^2)
          * h1(a)=a
          * h2(b)=2b
  Parameters
  ----------
  C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
  C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
  T :  ndarray, shape (ns, nt)
        Coupling between source and target spaces
  p : ndarray, shape (ns,)
  Returns
  -------
  constC : ndarray, shape (ns, nt)
          Constant C matrix in Eq. (6)
  hC1 : ndarray, shape (ns, ns)
          h1(C1) matrix in Eq. (6)
  hC2 : ndarray, shape (nt, nt)
          h2(C) matrix in Eq. (6)
  References
  ----------
  .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
  "Gromov-Wasserstein averaging of kernel and distance matrices."
  International Conference on Machine Learning (ICML). 2016.
  """
          
  if loss_fun == 'square_loss':
    def f1(a):
      return a**2 

    def f2(b):
      return b**2

    def h1(a):
      return a

    def h2(b):
      return 2*b

  constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
                    np.ones(len(q)).reshape(1, -1))
  constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
                    np.dot(q.reshape(1, -1), f2(C2).T))
  constC=constC1+constC2
  hC1 = h1(C1)
  hC2 = h2(C2)

  return constC,hC1,hC2

def tensor_product(constC,hC1,hC2,T):
  """ Return the tensor for Gromov-Wasserstein fast computation
  The tensor is computed as described in Proposition 1 Eq. (6) in [1].
  Parameters
  ----------
  constC : ndarray, shape (ns, nt)
          Constant C matrix in Eq. (6)
  hC1 : ndarray, shape (ns, ns)
          h1(C1) matrix in Eq. (6)
  hC2 : ndarray, shape (nt, nt)
          h2(C) matrix in Eq. (6)
  Returns
  -------
  tens : ndarray, shape (ns, nt)
          \mathcal{L}(C1,C2) \otimes T tensor-matrix multiplication result
  References
  ----------
  .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
  "Gromov-Wasserstein averaging of kernel and distance matrices."
  International Conference on Machine Learning (ICML). 2016.
  """
  
  A=-np.dot(hC1, T).dot(hC2.T)
  tens = constC+A

  return tens

def gwloss(constC,hC1,hC2,T):
  """ Return the Loss for Gromov-Wasserstein
  The loss is computed as described in Proposition 1 Eq. (6) in [1].
  Parameters
  ----------
  constC : ndarray, shape (ns, nt)
          Constant C matrix in Eq. (6)
  hC1 : ndarray, shape (ns, ns)
          h1(C1) matrix in Eq. (6)
  hC2 : ndarray, shape (nt, nt)
          h2(C) matrix in Eq. (6)
  T : ndarray, shape (ns, nt)
          Current value of transport matrix T
  Returns
  -------
  loss : float
          Gromov Wasserstein loss
  References
  ----------
  .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
  "Gromov-Wasserstein averaging of kernel and distance matrices."
  International Conference on Machine Learning (ICML). 2016.
  """

  tens=tensor_product(constC,hC1,hC2,T) 
            
  return np.sum(tens*T) 


def gwggrad(constC,hC1,hC2,T):
  """ Return the gradient for Gromov-Wasserstein
  The gradient is computed as described in Proposition 2 in [1].
  Parameters
  ----------
  constC : ndarray, shape (ns, nt)
          Constant C matrix in Eq. (6)
  hC1 : ndarray, shape (ns, ns)
          h1(C1) matrix in Eq. (6)
  hC2 : ndarray, shape (nt, nt)
          h2(C) matrix in Eq. (6)
  T : ndarray, shape (ns, nt)
          Current value of transport matrix T
  Returns
  -------
  grad : ndarray, shape (ns, nt)
          Gromov Wasserstein gradient
  References
  ----------
  .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
  "Gromov-Wasserstein averaging of kernel and distance matrices."
  International Conference on Machine Learning (ICML). 2016.
  """
        
  return 2*tensor_product(constC,hC1,hC2,T) 

def gw_lp(C1,C2,p,q,loss_fun='square_loss',alpha=1,armijo=True,**kwargs): 
  """
  Returns the gromov-wasserstein transport between (C1,p) and (C2,q)
  The function solves the following optimization problem:
  .. math::
      \GW_Dist = \min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
  Where :
      C1 : Metric cost matrix in the source space
      C2 : Metric cost matrix in the target space
      p  : distribution in the source space
      q  : distribution in the target space
      L  : loss function to account for the misfit between the similarity matrices
      H  : entropy
  Parameters
  ----------
  C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
  C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
  p :  ndarray, shape (ns,)
        distribution in the source space
  q :  ndarray, shape (nt,)
        distribution in the target space
  loss_fun :  string
      loss function used for the solver
  max_iter : int, optional
      Max number of iterations
  tol : float, optional
      Stop threshold on error (>0)
  verbose : bool, optional
      Print information along iterations
  log : bool, optional
      record log if True
  amijo : bool, optional
      If True the step of the line-search is found via an amijo research. Else closed form is used.
      If there is convergence issues use False.
  **kwargs : dict
      parameters can be directly pased to the ot.optim.cg solver
  Returns
  -------
  T : ndarray, shape (ns, nt)
      coupling between the two spaces that minimizes :
          \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
  log : dict
      convergence information and loss
  References
  ----------
  .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
      "Gromov-Wasserstein averaging of kernel and distance matrices."
      International Conference on Machine Learning (ICML). 2016.
  .. [2] Mémoli, Facundo. Gromov–Wasserstein distances and the
      metric approach to object matching. Foundations of computational
      mathematics 11.4 (2011): 417-487.
  """

  constC,hC1,hC2=init_matrix(C1,C2,p,q,loss_fun)
  M=np.zeros((C1.shape[0],C2.shape[0]))
  
  G0=p[:,None]*q[None,:]
  
  def f(G):
    return gwloss(constC,hC1,hC2,G)
  def df(G):
    return gwggrad(constC,hC1,hC2,G)
  
  G = optim.cg(p,q,M,alpha,f,df,G0,armijo=armijo,constC=constC,C1=C1,C2=C2,**kwargs)

  return f(G), G
    
def fgw_lp(M,C1,C2,p,q,loss_fun='square_loss',alpha=1,armijo=True,G0=None,**kwargs): 
  """
  Computes the FGW distance between two graphs see [3]
  .. math::
      \gamma = arg\min_\gamma (1-\alpha)*<\gamma,M>_F + alpha* \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
      s.t. \gamma 1 = p
            \gamma^T 1= q
            \gamma\geq 0
  where :
  - M is the (ns,nt) metric cost matrix
  - :math:`f` is the regularization term ( and df is its gradient)
  - a and b are source and target weights (sum to 1)
  The algorithm used for solving the problem is conditional gradient as discussed in  [1]_
  Parameters
  ----------
  M  : ndarray, shape (ns, nt)
        Metric cost matrix between features across domains
  C1 : ndarray, shape (ns, ns)
        Metric cost matrix respresentative of the structure in the source space
  C2 : ndarray, shape (nt, nt)
        Metric cost matrix espresentative of the structure in the target space
  p :  ndarray, shape (ns,)
        distribution in the source space
  q :  ndarray, shape (nt,)
        distribution in the target space
  loss_fun :  string,optionnal
      loss function used for the solver 
  max_iter : int, optional
      Max number of iterations
  tol : float, optional
      Stop threshold on error (>0)
  verbose : bool, optional
      Print information along iterations
  log : bool, optional
      record log if True
  amijo : bool, optional
      If True the steps of the line-search is found via an amijo research. Else closed form is used.
      If there is convergence issues use False.
  **kwargs : dict
      parameters can be directly pased to the ot.optim.cg solver
  Returns
  -------
  gamma : (ns x nt) ndarray
      Optimal transportation matrix for the given parameters
  log : dict
      log dictionary return only if log==True in parameters
  References
  ----------
  .. [3] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
        and Courty Nicolas
      "Optimal Transport for structured data with application on graphs"
      International Conference on Machine Learning (ICML). 2019.
  """

  if G0 is None:
    G0=p[:,None]*q[None,:]

  constC,hC1,hC2=init_matrix(C1,C2,p,q,loss_fun)
  
  def f(G):
    return gwloss(constC,hC1,hC2,G)

  def df(G):
    return gwggrad(constC,hC1,hC2,G)
  
  G = optim.cg(
    a = p,
    b = q,
    M = (1 - alpha) * M,
    reg = alpha,
    f = f,
    df = df,
    G0 = G0,
    armijo=armijo,
    C1=C1,
    C2=C2,
    constC=constC,
    **kwargs
  )

  return f(G), G


def fgw_partial_lp(M,C1,C2,p,q,m,alpha=1.0,armijo=True,G0=None,**kwargs):
  def df(T):
    """Compute the GW gradient. Note: we can not use the trick in [12]_  as
    the marginals may not sum to 1.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    numpy.array of shape (n_p+nb_dummies, n_u)
        gradient

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    cC1 = np.dot(C1 ** 2 / 2, np.dot(T, np.ones(C2.shape[0]).reshape(-1, 1)))
    cC2 = np.dot(np.dot(np.ones(C1.shape[0]).reshape(1, -1), T), C2 ** 2 / 2)
    constC = cC1 + cC2
    A = -np.dot(C1, T).dot(C2.T)
    tens = constC + A
    return tens * 2

  def f(T):
    """Compute the GW loss.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    GW loss
    """
    g = df(T) * 0.5
    return np.sum(g * T)
  
  G = optim.partial_cg(
    a = p, b = q, M = (1 - alpha) * M, reg = alpha, m = m, f = f, df = df, 
    G0=G0, armijo=armijo, C1=C1, C2=C2, **kwargs
  )
  
  return f(G), G