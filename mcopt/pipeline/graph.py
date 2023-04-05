"""
Utilities for working with Morse Graphs
"""
from typing import List

from mcopt.morse_graph import MorseGraph

class Graph:
  name: str
  frames : List[MorseGraph]
  
  def __init__(self, name: str, frames : List[MorseGraph]):
    self.name = name
    self.frames = frames