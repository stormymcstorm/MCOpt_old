"""

"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
  Generic, 
  TypeVar,
  TypedDict,
  List,
  Any,
  cast
)
import os
import json
import textwrap

from mcpipeline.entity import Entity, CacheableEntity
from mcpipeline.util.progress import ProgressFactory
from mcpipeline.util.logger import Logger

__all__ = ['Out', 'Conf', 'Target', 'Rule']

Out = TypeVar('Out', bound=Entity)
Conf = TypeVar('Conf', bound=TypedDict)

class Rule(ABC, Generic[Conf, Out]):
  @abstractmethod
  def __call__(self, *args: Any, progress: ProgressFactory, **kwds: Conf) -> Out:
    raise NotImplementedError()

class Target(ABC, Generic[Conf, Out]):
  @staticmethod
  @abstractmethod
  def target_type() -> str:
    raise NotImplementedError()
  
  @staticmethod
  @abstractmethod
  def entity_type() -> type[Out]:
    raise NotImplementedError()
  
  name: str
  cache_path: str
  rule: Rule[Conf, Out]
  conf: Conf
  depends: List[Target]
  
  display_name: str
  desc: str
  
  _out: Out | None = None
  
  def __init__(
    self,
    name: str,
    cache_path: str,
    rule: Rule[Conf, Out],
    conf: Conf,
    depends: List[Target] | None = None,
    display_name: str | None = None,
    desc: str | None = None,
  ):
    self.name = name
    self.cache_path = cache_path
    self.rule = rule
    self.conf = conf
    self.depends = [] if depends is None else depends
    
    self.display_name = name if display_name is None else display_name
    self.desc = '' if desc is None else desc
  
  def build(
    self,
    use_cache: bool = True,
    silent = False,
    show_progress = True,
  ) -> Out:
    if self._out is not None:
      return self._out
    
    logger = Logger(silent, self.target_type(), self.name)
    progress = ProgressFactory(
      show = show_progress,
      leave = False,
    )
    
    logger.outer(f'building')
    
    os.makedirs(self.cache_path, exist_ok=True)
    
    inputs = [
      dep.build(
        use_cache = use_cache,
        silent = silent,
        show_progress=show_progress
      ) 
      for dep in self.depends
    ]
    
    self._out = self.rule(*inputs, progress=progress, **self.conf)
    logger.inner(f'saving to {self.cache_path}')
    self._out.save(self.cache_path, progress)
    self._save_config()
    
    return self._out
  
  def display_description(self) -> str:
    return \
f'''
# {self.display_name}

{textwrap.dedent(self.desc)} 
'''

  def changed(self) -> bool:
    return True

  def _save_config(self):
    conf_path = os.path.join(self.cache_path, 'config.json')
    
    with open(conf_path, 'w') as conf_file:
      json.dump(self.conf, conf_file, cls=self._conf_encoder())
      
  def _conf_encoder(self) -> type[json.JSONEncoder]:
    return json.JSONEncoder
  
  def _conf_decoder(self):
    return (lambda x : x)
  
CacheableOut = TypeVar('CacheableOut', bound=CacheableEntity)
  
class CacheableTarget(Target[Conf, CacheableOut]):
  _changed: bool
  
  def __init__(
    self, 
    name: str, 
    cache_path: str, 
    rule: Rule[Conf, CacheableOut], 
    conf: Conf, 
    **kwargs
  ):
    super().__init__(
      name = name,
      cache_path = cache_path,
      rule = rule, 
      conf = conf, 
      **kwargs
    )
    
    self._changed = False
  
  def build(self, use_cache: bool = True, silent=False, show_progress=True) -> CacheableOut:
    if self._out is not None:
      return self._out
    
    logger = Logger(silent, self.target_type(), self.name)
    progress = ProgressFactory(
      show = show_progress,
      leave = False,
    )
    
    if use_cache and not self.changed():
      logger.outer('config unchanged, loading')
      
      self._out = self.entity_type().load(self.cache_path, progress) # type: ignore
      
      return self._out # type: ignore
    
    self._out = super().build(use_cache, silent, show_progress)
    
    return self._out
  
  def changed(self) -> bool:
    self._changed = self._config_changed() or self._changed
    
    return self._changed or any([dep.changed() for dep in self.depends])
  
  def _config_changed(self) -> bool:
    conf_path = os.path.join(self.cache_path, 'config.json')
        
    if os.path.exists(conf_path):
      with open(conf_path, 'r') as conf_file:
        contents = json.load(conf_file, object_hook=self._conf_decoder())
        
        return contents != self.conf
    
    return True
  
  