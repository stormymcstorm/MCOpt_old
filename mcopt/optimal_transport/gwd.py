
from typing import Callable
import numpy as np

def init_matrix(C1, C2, T, p, q):
  def f1(a):
    return (a**2) / 2.0

  def f2(b):
    return (b**2) / 2.0

  def h1(a):
    return a

  def h2(b):
    return b

  constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
                    np.ones(len(q)).reshape(1, -1))
  constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
                    np.dot(q.reshape(1, -1), f2(C2).T))
  constC = constC1 + constC2
  hC1 = h1(C1)
  hC2 = h2(C2)

  return constC, hC1, hC2
  
def tensor_product(constC, hC1, hC2, T):
  A = -np.dot(hC1, T).dot(hC2.T)
  tens = constC + A
  return tens

def frobenius(A,B):
  return np.trace(np.matmul(np.transpose(A),B))

class GromovWassersteinDistance(Callable):
  def __init__(self, W_x, mu_x, W_y, mu_y):
    G0 = mu_x[:, None] * mu_y[None, :]
    
    constC, hC1, hC2 = init_matrix(W_x, W_y, G0, mu_x, mu_y)
    constCt, hC1t, hC2t = init_matrix(W_x.T, W_y.T, G0, mu_x, mu_y)
    
    self.constC = constC
    self.hC1 = hC1
    self.hC2 = hC2
    self.constCt = constCt
    self.hC1t = hC1t
    self.hC2t = hC2t
  
  def __call__(self, T):
    tens = (1/2.0)*tensor_product(self.constC, self.hC1, self.hC2, T) \
            + (1/2.0)*tensor_product(self.constCt, self.hC1t, self.hC2t, T)
    return frobenius(tens,T)
  
  def grad(self, T):
    return tensor_product(self.constC, self.hC1, self.hC2, T) \
             + tensor_product(self.constCt, self.hC1t, self.hC2t, T)