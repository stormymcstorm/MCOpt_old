
from __future__ import annotations
from typing import Optional, Dict

from tqdm.autonotebook import tqdm

class ProgressBar:
  
  _t: Optional[tqdm]
  
  def __init__(self, t: Optional[tqdm]):
    self._t = t
  
  def __enter__(self) -> ProgressBar:
    if self._t is not None:
      self._t.__enter__()
    
    return self
  
  def __exit__(self, *args, **kwargs):
    if self._t is not None:
      self._t.__exit__(*args, **kwargs)
      
  def update(self, *args, **kwargs):
    if self._t is not None:
      return self._t.update(*args, **kwargs)
  
  def report_hook(self, b : int, bsize: int, tsize: int) -> object:
    if self._t is not None:
      if tsize is not None:
        self._t.total = tsize
      
      return self._t.update(b * bsize - self._t.n)
    
    return None
    
class ProgressFactory:
  show: bool
  tqdm_kwargs: Dict
  
  def __init__(self, show: bool = False, **kwargs):
    self.show = show
    self.tqdm_kwargs = kwargs
    
  def __call__(self, *args, **kwargs) -> ProgressBar:
    if self.show:
      return ProgressBar(tqdm(*args, **kwargs, **self.tqdm_kwargs))
    else:
      return ProgressBar(None)