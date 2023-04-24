"""
Target for downloading files
"""

from typing import TypedDict

import os
import urllib.request
import urllib.parse

from mcpipeline.targets.filegroup import FilePathGroup, FilePathGroupTarget
from mcpipeline.target import Rule
from mcpipeline.util import ProgressFactory

__all__ = ['DownloadTarget']

class DownloadConf(TypedDict):
  url: str
  out_file_path: str
  timeout: float | None
  
class DownloadRule(Rule[DownloadConf, FilePathGroup]):
  def __call__(
    self,
    url: str, 
    out_file_path: str, 
    timeout: float | None,
    progress: ProgressFactory,
  ) -> FilePathGroup:
    with urllib.request.urlopen(url, timeout=timeout) as res:
      headers = res.info()
      
      with progress(
        desc=f'Downloading {os.path.basename(out_file_path)}', 
        unit='B', 
        unit_scale=True,
        unit_divisor=1024
      ) as prog:
        with open(out_file_path, 'wb') as out_file:
          blocksize = 1024 * 8
          size = -1
          read = 0
          blocknum = 0
          
          if 'content-length' in headers:
            size = int(headers['content-length'])
          
          prog.report_hook(blocknum, blocksize, size)
          
          while True:
            block = res.read(blocksize)
            if not block:
              break
            
            read += len(block)
            out_file.write(block)
            
            blocknum += 1
            prog.report_hook(blocknum, blocksize, size)
            
        if size >= 0 and read < size:
          raise RuntimeError(f'download incomplete: got only {read} out of {size} bytes')
        
    return FilePathGroup([out_file_path])

class DownloadTarget(FilePathGroupTarget[DownloadConf]):
  @staticmethod
  def target_type() -> str:
    return 'download'
  
  def __init__(
    self,
    url: str,
    cache_path: str,
    timeout: float | None = None,
    file_name: str | None = None,
    **kwargs,
  ):
    if file_name is not None:
      out_file_path = os.path.join(cache_path, file_name)
    else:
      url_parts = urllib.parse.urlparse(url)
      
      out_file_path = os.path.join(cache_path, os.path.basename(url_parts.path))
    
    super().__init__(
      cache_path=cache_path,
      rule=DownloadRule(),
      conf={
        'url': url,
        'timeout': timeout,
        'out_file_path': out_file_path
      },
      **kwargs
    )