"""

"""

from __future__ import annotations
from typing import List
import os
import json

from mcpipeline.entity import CacheableEntity
from mcpipeline.target import CacheableTarget, Conf
from mcpipeline.util import Logger, ProgressFactory

__all__ = ['FilePathGroup', 'FilePathGroupTarget']

class FilePathGroup(CacheableEntity):
  file_paths: List[str]
  
  @staticmethod
  def load(cache_path: str, progress: ProgressFactory) -> FilePathGroup:
    cache_file_path = os.path.join(cache_path, 'filegroup.json')
    
    with open(cache_file_path, 'r') as cache_file:
      contents = json.load(cache_file)
      
      if 'file_paths' not in contents:
        raise RuntimeError(f'Invalid filegroup cache {cache_file_path}')
      
      return FilePathGroup(contents['file_paths'])
  
  def __init__(
    self,
    file_paths: List[str]
  ):
    super().__init__()
      
    self.file_paths = file_paths
    
  def save(self, cache_path: str, progress: ProgressFactory):
    cache_file_path = os.path.join(cache_path, 'filegroup.json')
    
    with open(cache_file_path, 'w') as cache_file:
      contents = {
        'file_paths': self.file_paths
      }
      
      json.dump(contents, cache_file)
    
class FilePathGroupTarget(CacheableTarget[Conf, FilePathGroup]):
  @staticmethod
  def entity_type() -> type[FilePathGroup]:
    return FilePathGroup