"""
Download target definition
"""

from typing import (
  Dict, 
  TypedDict,
  IO,
  Optional,
  cast
)
from urllib.parse import urlparse
from urllib.request import urlretrieve
import os
import tempfile

from mcopt.pipeline.target import Target
from mcopt.pipeline.progress import ProgressFactory

class DownloadConf(TypedDict):
  url: str
  file_name: str
  
class Download:
  file: IO
  
  def __init__(self, file: IO) -> None:
    self.file = file
  
class DownloadTarget(Target[DownloadConf, Download]):
  """
  A target representing some downloaded asset
  """
    
  @staticmethod
  def _validate_conf(name: str, conf: Dict) -> DownloadConf:
    if 'url' not in conf:
      raise ValueError(f'Download target {name} must have url')
    
    url = conf['url']

    if 'file_name' not in conf:
      url_parts = urlparse(url)
      
      conf['file_name'] = os.path.basename(url_parts.path)    
    
    return cast(DownloadConf, conf)
  
  def __init__(
    self, 
    name: str, 
    conf: Dict,
    cache_path: Optional[str],
    progress: Optional[ProgressFactory] = None,
    silent: bool = True,
  ):
    super().__init__(name, conf, cache_path, progress=progress, silent=silent)
  
  @property
  def target_name(self) -> str:
    return "download"
    
  def _load(self) -> Download:
    assert(self.cache_path is not None)
    
    file_path = os.path.join(self.cache_path, self.conf['file_name'])
    
    return Download(open(file_path, 'r'))
  
  def _save(self, download: Download):
    assert(self.cache_path is not None)
    
    file_path = os.path.join(self.cache_path, self.conf['file_name'])
    
    if download.file.name == file_path:
      return
        
    os.makedirs(self.cache_path, exist_ok=True)
    
    with open(file_path, 'w') as file:
      file.write(download.file.read())
  
  def _make(self) -> Download:
    if self.cache_path is not None:
      os.makedirs(self.cache_path, exist_ok=True)
      
      file_path = os.path.join(self.cache_path, self.conf['file_name'])
    else:
      file_path = tempfile.mktemp()
    
    with self.progress(
      desc=f'Downloading {self.name}',
      unit='B',
      unit_scale = True,
      unit_divisor = 1024,
    ) as prog:
      _, res = urlretrieve(self.conf['url'], file_path, reporthook=prog.report_hook)
    
    return Download(open(file_path, 'r'))
  
  def generate(self) -> Download:
    return super().generate()
  
  def get(self) -> Download:
    return super().get()