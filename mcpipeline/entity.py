"""

"""

from __future__ import annotations
from abc import ABC, abstractmethod

from mcpipeline.util.progress import ProgressFactory
from mcpipeline.util.logger import Logger

class Entity:
  @abstractmethod
  def save(self, cache_path: str, progress: ProgressFactory):
    raise NotImplementedError()
  
class CacheableEntity(ABC, Entity):
  @staticmethod
  @abstractmethod
  def load(
    cache_path: str,
    progress: ProgressFactory
  ) -> CacheableEntity:
    raise NotImplementedError()