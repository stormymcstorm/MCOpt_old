
from typing import Optional, List
import os
import math

def file_ext(name : str) -> Optional[str]:
  res = os.path.basename(name).rsplit(os.path.extsep, maxsplit=1)

  if len(res) < 2:
    return None
  
  return res[1]

def sort_files(files: List[str]) -> List[str]:
  files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
  
  return files