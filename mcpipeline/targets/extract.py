"""
Target for extracting files
"""

from typing import TypedDict
import os
import zipfile
from fnmatch import fnmatch

from mcpipeline.target import Rule
from mcpipeline.targets.filegroup import FilePathGroup, FilePathGroupTarget
from mcpipeline.util import ProgressFactory

__all__ = ['ExtractTarget', 'ExtractZipTarget']

class ExtractConf(TypedDict):
  output_path: str
  pattern: str
  
class ExtractTarget(FilePathGroupTarget[ExtractConf]):
  @staticmethod
  def target_type() -> str:
    return 'extract'
  
  def __init__(
    self, 
    name: str, 
    cache_path: str,
    rule: Rule[ExtractConf, FilePathGroup],
    to_extract: FilePathGroupTarget,
    pattern: str = '*',
    **kwargs
  ):
    super().__init__(
      name = name, 
      cache_path = cache_path, 
      rule = rule, 
      conf = {
        'output_path': cache_path,
        'pattern': pattern
      }, 
      depends = [to_extract],
      **kwargs
    )
  
class ExtractZipRule(Rule[ExtractConf, FilePathGroup]):
  def __call__(
    self, 
    zips: FilePathGroup, 
    output_path: str,
    pattern: str,
    progress: ProgressFactory, 
  ) -> FilePathGroup:
    out_files = []
    
    with progress(
      desc=f'Extracting zip files',
      unit = 'Files',  
    ) as prog:
      for zip_file_path in zips.file_paths:
        out_dir, _ = os.path.splitext(os.path.basename(zip_file_path))
        out_dir = os.path.join(output_path, out_dir)
        
        os.makedirs(out_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_file_path) as zip_file:
          members = zip_file.infolist()
          
          prog.total += len(members)
          
          for member in members:
            if member.is_dir():
              zip_file.extract(member, out_dir)
              prog.update()
            
            if not member.is_dir and not fnmatch(member.filename, pattern):
              prog.update()
              continue
            
            out_file_path = zip_file.extract(member, out_dir)
            prog.update()
            
            out_files.append(out_file_path)
          
    return FilePathGroup(out_files)

class ExtractZipTarget(ExtractTarget):
  def __init__(
    self, 
    name: str, 
    cache_path: str,
    zips: FilePathGroupTarget,
    pattern: str = '*',
    display_name: str | None = None,
    desc: str | None = None,
    **kwargs
  ):
    if display_name is None:
      display_name = zips.display_name
    
    if desc is None:
      desc = zips.desc
    
    super().__init__(
      name = name, 
      cache_path = cache_path, 
      rule = ExtractZipRule(), 
      to_extract = zips,
      pattern = pattern,
      display_name = display_name,
      desc = desc,
      **kwargs
    )