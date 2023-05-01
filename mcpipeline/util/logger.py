"""

"""

from mcpipeline.util.progress import ProgressFactory

class Logger:
  silent: bool
  type: str
  name: str
  progress: ProgressFactory
  
  def __init__(
    self,
    silent: bool,
    type: str,
    name: str,
    progress: ProgressFactory
  ):
    self.silent = silent
    self.type = type
    self.name = name
    self.progress = progress
    
  def outer(self, msg: str):
    if not self.silent:
      if self.progress.curr is not None:
        self.progress.curr.write(f'> [{self.type}:{self.name}] {msg}')
      else:
        print(f'> [{self.type}:{self.name}] {msg}')
  
  def inner(self, msg: str):
    if not self.silent:
      
      if self.progress.curr is not None:
        self.progress.curr.write(f'  {msg}')
      else:
        print(f'  {msg}')