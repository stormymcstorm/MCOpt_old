"""
TODO
"""
from __future__ import annotations

from typing import (
  Generic, 
  TypeVar, 
  TypedDict,
  Dict, 
  Optional,
  Callable,
  cast,
)
from abc import ABC, abstractmethod, abstractproperty
import os
import json

from mcopt.pipeline.progress import ProgressFactory

Conf = TypeVar("Conf")
Output = TypeVar("Output")

class Target(ABC, Generic[Conf, Output]):
  """
  An abstract class for targets in the pipeline.
  """
  
  name: str
  conf: Conf
  cache_path: Optional[str]
  
  _output: Optional[Output]
  progress: ProgressFactory
  silent: bool
  
  
  @staticmethod
  @abstractmethod
  def _validate_conf(name: str, conf: Dict) -> Conf:
    raise NotImplementedError()
  
  def __init__(
    self, 
    name: str, 
    conf: Dict,
    cache_path: Optional[str],
    progress: Optional[ProgressFactory] = None, 
    silent: bool = True,
  ):
    self.conf = self._validate_conf(name, conf)
    self.name = name
    self.cache_path = cache_path
    
    self._output = None
    
    self.progress = progress if progress is not None else ProgressFactory(show = False)

    self.silent = silent
  
  def _config_changed(self, cache_path: str, conf: Conf) -> bool:
    conf_path = os.path.join(cache_path, 'config.json')
    
    if os.path.exists(conf_path):
      with open(conf_path, 'r') as file:
        contents = json.load(file)
        
        return contents != conf
    
    return True
  
  def _save_config(self, cache_path: str, conf: Conf):
    os.makedirs(cache_path, exist_ok=True)
    
    conf_path = os.path.join(cache_path, 'config.json')
    with open(conf_path, 'w') as config_file:
      json.dump(conf, config_file)
  
  @abstractproperty
  def target_name(self) -> str:
    raise NotImplementedError()
  
  @classmethod
  @abstractmethod
  def _load(cls, *args, **kwargs) -> Output:
    raise NotImplementedError()
  
  @classmethod
  @abstractmethod
  def _save(cls, output: Output):
    raise NotImplementedError()
  
  @classmethod
  @abstractmethod
  def _make(cls, *args, **kwargs) -> Output:
    raise NotImplementedError()
  
  def generate(self, *args, **kwargs) -> Output:
    if not self.silent:
      print(f'> Generating {self.name} {self.target_name}')
    
    if self.cache_path is not None \
      and not self._config_changed(self.cache_path, self.conf):
      if not self.silent:
        print(f'  config unchanged, loading {self.target_name}')
        
      return self._load(*args, **kwargs)
    
    output = self._make(*args, **kwargs)
    
    if self.cache_path is not None:
      print(f' saving to {self.cache_path}')
      self._save(output)
      self._save_config(self.cache_path, self.conf)
      
    return output
  
  def get(self, *args, **kwargs) -> Output:
    if self._output is None:
      self._output = self.generate(*args, **kwargs)
    
    return self._output