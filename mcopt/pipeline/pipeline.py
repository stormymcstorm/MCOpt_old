"""
Logic for loading and generating pipeline targets
"""

from typing import (
  Dict, 
  Optional
)
import os
from os import PathLike
import json

from tqdm.autonotebook import tqdm

from mcopt.pipeline.progress import ProgressFactory
from mcopt.pipeline.download import Download, DownloadTarget

class Pipeline:
  """
  Utility for managing datasets/graph/complexes/etc.
  
  Parameters
  ----------
  
  """
  
  _downloads: Dict[str, DownloadTarget]
  
  def __init__(
    self, 
    config_path: PathLike | str,
    use_cache: bool = True,
    silent: bool = False,
    show_progress: bool = False,
  ):
    with open(config_path, 'r') as config_file:
      config = json.load(config_file)
      
    root = os.path.dirname(config_path)
    cache = 'pipeline_out' if 'cache' not in config else config['cache']
    
    progress = ProgressFactory(
      show = show_progress and not silent, 
      leave = False
    )
    
    self._downloads = {}
    if 'downloads' in config:
      downloads_cache = os.path.join(root, cache, 'downloads') if use_cache else None

      for name, conf in config['downloads'].items():
        self._downloads[name] = DownloadTarget(
          name, 
          conf, 
          downloads_cache, 
          progress=progress,
          silent=silent
        )
  
  def download(self, name: str) -> Download:
    if name not in self._downloads:
      raise ValueError(f'Unrecognized download {name}')

    target = self._downloads[name]
    
    return target.get()
  
  def generate_all(self):
    for target in self._downloads.values():
      target.generate()

