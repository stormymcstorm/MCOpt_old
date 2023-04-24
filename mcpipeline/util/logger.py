"""

"""

class Logger:
  silent: bool
  type: str
  name: str
  
  def __init__(
    self,
    silent: bool,
    type: str,
    name: str,
  ):
    self.silent = silent
    self.type = type
    self.name = name
    
  def outer(self, msg: str):
    if not self.silent:
      print(f'> [{self.type}:{self.name}] {msg}')
  
  def inner(self, msg: str):
    if not self.silent:
      print(f'  {msg}')