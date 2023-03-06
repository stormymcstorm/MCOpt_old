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
  loop = 1
  
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
      loop = 0
        
    delta_fval = (f_val - old_fval)

    abs_delta_fval = abs(delta_fval)
    relative_delta_fval = abs_delta_fval / abs(f_val)
    if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
      loop = 0

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

# def pcg(
#   a, b, M, reg, f, df, G0=None, numItermax=200, numItermaxEmd=100000,
#   stopThr=1e-9, stopThr2=1e-9, verbose=False, log=False, **kwargs
# ):
  
#   pass