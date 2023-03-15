"""
Optimization algorithms for optimal transport. 

Adapted from [POT](https://github.com/PythonOT/POT/blob/master/ot/optim.py)
"""

import numpy as np
from ot.lp import emd
from ot.optim import solve_linesearch

class NonConvergenceError(Exception):
  """Used to indicate failed convergence
  """
  pass

def cg(
  a, b, M, reg, f, df, G0=None, numItermax=200, numItermaxEmd=100000,
  stopThr=1e-9, stopThr2=1e-9, verbose=False, log=False, **kwargs
):
  """Adapted from https://pythonot.github.io/gen_modules/ot.optim.html#ot.optim.cg

  Raises:
    NonConvergenceError: Raise if optimization does not converge
  """
  if log:
    log = {'loss': [],'delta_fval': []}
    
  if G0 is None:
    G = np.outer(a, b)
  else:
    G = G0
    
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
    
  loop = True
  while loop:
    it += 1
    old_fval = f_val
    #G=xt
    # problem linearization
    Mi = M + reg * df(G) #Gradient(xt)
    # set M positive
    Mi += Mi.min()

    # solve linear program
    Gc, logemd = emd(a, b, Mi, numItermax=numItermaxEmd, log=True) #st

    deltaG = Gc - G #dt

    # argmin_alpha f(xt+alpha dt)
    alpha, fc, f_val = solve_linesearch(
      cost, G, deltaG, Mi, f_val, reg=reg, M=M, Gc=Gc,
      alpha_min=0,alpha_max=1, **kwargs
    )

    if alpha is None or np.isnan(alpha) :
      raise NonConvergenceError('Failed to find Alpha')
    else:
      G = G + alpha * deltaG #xt+1=xt +alpha dt

    # test convergence
    if it >= numItermax:
      loop = False
        
    delta_fval = (f_val - old_fval)

    abs_delta_fval = abs(delta_fval)
    relative_delta_fval = abs_delta_fval / abs(f_val)
    if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
      loop = False

    if log:
      log['loss'].append(f_val)
      log['delta_fval'].append(delta_fval)

    if verbose:
      if it % 20 == 0:
        print('{:5s}|{:12s}|{:8s}'.format(
          'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
        print('{:5d}|{:8e}|{:8e}|{:5e}'.format(it, f_val, delta_fval,alpha))

  if log:
    log.update(logemd)
    return G, log
  else:
    return G

def pcg(
  a, b, M, reg, m, f, df, G0=None, nb_dummies=1, numItermax=200, numItermaxEmd=100000,
  stopThr=1e-9, stopThr2=1e-9, verbose=False, log=False, **kwargs
):
  
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
    
  loop = True
  while loop:
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

    deltaG = Gc - G #dt

    # argmin_alpha f(xt+alpha dt)
    alpha, fc, f_val = solve_linesearch(
      cost, G, deltaG, Mi, f_val, reg=reg, M=M, Gc=Gc,
      alpha_min=0,alpha_max=1, **kwargs
    )

    if alpha is None or np.isnan(alpha) :
      raise NonConvergenceError('Failed to find Alpha')
    else:
      G = G + alpha * deltaG #xt+1=xt +alpha dt

    # test convergence
    if it >= numItermax:
      loop = False
        
    delta_fval = (f_val - old_fval)

    # CHECKME: other code compares G to Gprev
    if abs(np.linalg.norm(G - G_prev)) < stopThr:
      loop = 0
    
    abs_delta_fval = abs(delta_fval)
    relative_delta_fval = abs_delta_fval / abs(f_val)
    if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
      loop = False

    if log:
      log['loss'].append(f_val)
      log['delta_fval'].append(delta_fval)

    if verbose:
      if it % 20 == 0:
        print('{:5s}|{:12s}|{:8s}'.format(
          'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
        print('{:5d}|{:8e}|{:8e}|{:5e}'.format(it, f_val, delta_fval,alpha))

  if log:
    log.update(logemd)
    return G, log
  else:
    return G