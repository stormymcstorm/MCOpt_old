"""
Optimization Solvers
"""

from typing import Callable, Optional, Tuple
from numpy.typing import ArrayLike
import numpy as np

from ot.lp import emd
from ot.optim import solve_linesearch

class NonConvergenceError(Exception):
  """
  Used to indicate a failure to converge
  """
  pass

def cg(
  a: np.ndarray,
  b: np.ndarray,
  M: np.ndarray,
  reg: float,
  f: Callable[[ArrayLike], float],
  df: Callable[[ArrayLike], float],
  G0: Optional[np.ndarray] = None,
  numItermax: int = 200,
  numItermaxEmd: int = 100000,
  stopThr: float = 1e-9,
  stopThr2: float = 1e-9,
  verbose: bool = False,
  log: bool = False,
  **kwargs,
) -> np.ndarray | Tuple[np.ndarray, dict]:
  """
  Solve the general regularized OT problem with conditional gradient.

  See [POT](https://pythonot.github.io/gen_modules/ot.optim.html#ot.optim.cg) for
  more details.

  Parameters
  ----------
  a : array-like, shape (ns,)
      samples weights in the source domain
  b : array-like, shape (nt,)
      samples in the target domain
  M : array-like, shape (ns, nt)
      loss matrix
  reg : float
      Regularization term >0
  G0 :  array-like, shape (ns,nt), optional
      initial guess (default is indep joint density)
  numItermax : int, optional
      Max number of iterations
  numItermaxEmd : int, optional
      Max number of iterations for emd
  stopThr : float, optional
      Stop threshold on the relative variation (>0)
  stopThr2 : float, optional
      Stop threshold on the absolute variation (>0)
  verbose : bool, optional
      Print information along iterations
  log : bool, optional
      record log if True
  **kwargs : dict
            Parameters for linesearch
            
  Returns
  -------
  gamma : (ns x nt) ndarray
      Optimal transportation matrix for the given parameters
  log : dict
      log dictionary return only if log==True in parameters
  """
  
  if log:
    log = {'loss': [], 'delta_fval': []}
    
  if G0 is None:
    G = np.outer(a, b)
  else:
    assert G0.ndim == 2
    assert G0.shape[0] == a.shape[0]
    assert G0.shape[1] == b.shape[0]
    
    G = G0
    
  def cost(G):
    return np.sum(M * G) + reg * f(G)
  
  f_val = cost(G)
  if log:
    log['loss'].append(f_val)
  
  it = 0
  
  if verbose:
    print('{:5s}|{:12s}|{:8s}'.format('It.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
    print('{:5d}|{:8e}|{:8e}'.format(it, f_val, 0))
    
  loop = True
  while loop:
    it += 1
    old_fval = f_val
    
    Mi = M + reg * df(G)
    Mi += Mi.min()
    
    if log:
      Gc, logemd = emd(a, b, Mi, numItermax=numItermaxEmd, log=True)
    else:
      Gc = emd(a, b, Mi, numItermax=numItermaxEmd, log=False)
    
    deltaG = Gc - G
    
    alpha, fc, f_val = solve_linesearch(
      cost, G, deltaG, Mi, f_val, reg=reg, M=M, Gc=Gc,
      alpha_min=0, alpha_max=1, **kwargs
    )
    
    if alpha is None or np.isnan(alpha):
      raise NonConvergenceError('Failed to converge on alpha')

    G = G + alpha * deltaG
    
    delta_fval = (f_val - old_fval)
    
    if log:
      log['loss'].append(f_val)
      log['delta_fval'].append(delta_fval)
      
    if verbose:
      print('{:5s}|{:12s}|{:8s}'.format('It.', 'Loss', 'Delta loss') 
            + '\n' + '-' * 32)
      print('{:5d}|{:8e}|{:8e}|{:5e}'.format(it, f_val, delta_fval, alpha))
    
    if it >= numItermax:
      break
   
    abs_delta = abs(delta_fval)
    rel_delta = abs_delta / abs(f_val)
    
    if rel_delta < stopThr or abs_delta < stopThr2:
      break
    
  if log:
    log.update(logemd)
    return G, log
  
  return G
  
def pcg(
  a: np.ndarray,
  b: np.ndarray,
  M: np.ndarray,
  reg: float,
  m: float,
  f: Callable[[ArrayLike], float],
  df: Callable[[ArrayLike], float],
  nb_dummies=1,
  G0: Optional[np.ndarray] = None,
  numItermax: int = 200,
  numItermaxEmd: int = 100000,
  stopThr: float = 1e-9,
  stopThr2: float = 1e-9,
  verbose: bool = False,
  log: bool = False,
  **kwargs,
) -> np.ndarray | Tuple[np.ndarray, dict]:
  """
  Solve the general regularized OT problem with conditional gradient.

  See [POT](https://pythonot.github.io/gen_modules/ot.optim.html#ot.optim.cg) for
  more details.

  Parameters
  ----------
  a : array-like, shape (ns,)
      samples weights in the source domain
  b : array-like, shape (nt,)
      samples in the target domain
  M : array-like, shape (ns, nt)
      loss matrix
  reg : float
      Regularization term >0
  G0 :  array-like, shape (ns,nt), optional
      initial guess (default is indep joint density)
  numItermax : int, optional
      Max number of iterations
  numItermaxEmd : int, optional
      Max number of iterations for emd
  stopThr : float, optional
      Stop threshold on the relative variation (>0)
  stopThr2 : float, optional
      Stop threshold on the absolute variation (>0)
  verbose : bool, optional
      Print information along iterations
  log : bool, optional
      record log if True
  **kwargs : dict
            Parameters for linesearch
            
  Returns
  -------
  gamma : (ns x nt) ndarray
      Optimal transportation matrix for the given parameters
  log : dict
      log dictionary return only if log==True in parameters
  """
  
  if m is None:
    m = np.min((np.sum(a), np.sum(b)))
  elif m < 0:
    raise ValueError("Problem infeasible. Parameter m should be greater"
                     " than 0.")
  elif m >  np.min((np.sum(a), np.sum(b))):
    raise ValueError("Problem infeasible. Parameter m should lower or"
                     " equal than min(|a|_1, |b|_1).")
  
  dim_G_extended = (len(a) + nb_dummies, len(b) + nb_dummies)
  b_extended = np.append(b, [(np.sum(a) - m) / nb_dummies] * nb_dummies)
  a_extended = np.append(a, [(np.sum(b) - m) / nb_dummies] * nb_dummies)
  
  if log:
    log = {'loss': [],'delta_fval': []}
    
  if G0 is None:
    G = np.outer(a, b)
  else:
    G = G0
    
  if np.sum(G) > m:
    G *= (m / np.sum(G))
    
  def cost(G):
    return np.sum(M * G) + reg * f(G)
  
  f_val = cost(G)
  if log:
    log['loss'].append(f_val)
    
  it = 0
    
  if verbose:
    print('{:5s}|{:12s}|{:8s}'.format(
      'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
    print('{:5d}|{:8e}|{:8e}'.format(it, f_val, 0))
    
  while True:
    it += 1
    old_fval = f_val
    G_prev = G.copy()
    
    Mi = M + reg * df(G)
    Mi += Mi.min()
    
    M_emd = np.zeros(dim_G_extended)
    M_emd[:len(a), :len(b)] = Mi
    M_emd[-nb_dummies:, -nb_dummies:] = np.max(Mi) * 1e5
    M_emd = np.asarray(M_emd, dtype=np.float64)

    # solve linear program
    Gcemd, logemd = emd(
      a_extended, 
      b_extended, 
      M_emd, 
      numItermax=numItermaxEmd, 
      log=True
    )
    
    if logemd['warning'] is not None:
      raise ValueError("Error in the EMD resolution: try to increase the"
                       " number of dummy points")
      
    if nb_dummies > 0 and np.any(Gcemd[-nb_dummies:, -nb_dummies:] > 1e-16):
      raise ValueError("Solution from EMD is illegal: G[-nb_dummies:, -nb_dummies] > 0!")

    Gc = Gcemd[:len(a), :len(b)]

    deltaG = Gc - G 

    alpha, fc, f_val = solve_linesearch(
      cost, G, deltaG, Mi, f_val, reg=reg, M=M, Gc=Gc,
      alpha_min=0,alpha_max=1, **kwargs
    )

    if alpha is None or np.isnan(alpha) :
      raise NonConvergenceError('Failed to find Alpha')

    G = G + alpha * deltaG 

    delta_fval = (f_val - old_fval)
    
    if log:
      log['loss'].append(f_val)
      log['delta_fval'].append(delta_fval)

    if verbose:
      if it % 20 == 0:
        print('{:5s}|{:12s}|{:8s}'.format(
          'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
        print('{:5d}|{:8e}|{:8e}|{:5e}'.format(it, f_val, delta_fval,alpha))

    # test convergence
    if it >= numItermax:
      break
        

    if abs(np.linalg.norm(G - G_prev)) < stopThr:
      break
    
    abs_delta_fval = abs(delta_fval)
    relative_delta_fval = abs_delta_fval / abs(f_val)
    if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
      break

  if log:
    log.update(logemd)
    return G, log
  else:
    return G
