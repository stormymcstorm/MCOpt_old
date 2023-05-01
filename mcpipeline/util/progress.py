"""

"""

from __future__ import annotations
from typing import Generic, TypeVar, Iterable, Iterator

from tqdm import tqdm


T = TypeVar("T")

class ProgressFactory:
  show: bool
  tqdm_kwargs: dict
  curr: ProgressBar | None
  
  def __init__(self, show: bool, **kwargs):
    self.show = show
    self.tqdm_kwargs = kwargs
    self.curr = None
    
  def __call__(self, iterable: Iterable[T] | None = None, **kwargs) -> ProgressBar[T]:
    if self.show:
      self.curr = ProgressBar(t = tqdm(iterable = iterable, **self.tqdm_kwargs, **kwargs))
    else:
      self.curr = ProgressBar(iterable=iterable)
    
    return self.curr
    

class ProgressBar(Generic[T]):
  _t: tqdm | None
  _iter: Iterable[T] | None
  _total: int
  
  def __init__(self, t: tqdm | None = None, iterable: Iterable[T] | None = None):
    self._t = t
    self._iter = iterable
    self._total = 0
    
  def __enter__(self) -> ProgressBar:
    return self
  
  def __exit__(self, *args, **kwargs):
    if self._t is not None:
      self._t.__exit__(*args, **kwargs)
      
  def __iter__(self) -> Iterator[T]:
    if self._t is not None:
      return self._t.__iter__()
    
    assert (self._iter is not None)
    
    return self._iter.__iter__()
  
  @property
  def total(self) -> int:
    if self._t is not None:
      return self._t.total if self._t.total is not None else 0
    
    return self._total
  
  @total.setter
  def total(self, val):
    if self._t is not None:
      self._t.total = val
    else:
      self._total = val
  
  def write(self, msg: str):
    if self._t is not None:
      self._t.write(msg)
    else:
      print(msg)    
  
  def update(self, n: float = 1):
    if self._t is not None:
      self._t.update(n)
  
  def report_hook(self, b: int, bsize: int, tsize: int):
    if self._t is not None:
      self._t.total = tsize
      
      self._t.update(b * bsize - self._t.n)