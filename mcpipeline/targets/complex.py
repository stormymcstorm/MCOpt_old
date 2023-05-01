"""
Target for Morse Complexes
"""

from __future__ import annotations
from typing import TypedDict, Dict, List
import os
import json

from mcpipeline.entity import CacheableEntity
from mcpipeline.target import CacheableTarget, Rule
from mcpipeline.targets.dataset import Dataset, DatasetTarget
from mcpipeline.util import ProgressFactory

import mcpipeline.ttk as ttk
import mcpipeline.vtk as vtk_util

__all__ = ['Complex', 'ComplexTarget']

class Complex(CacheableEntity):
  frames: Dict[int, ttk.MorseComplex]
  
  @staticmethod
  def load(cache_path: str, progress: ProgressFactory) -> Complex:
    cache_file_path = os.path.join(cache_path, 'complex.json')
    
    with open(cache_file_path, 'r') as cache_file:
      contents = json.load(cache_file)
      
    if 'time_steps' not in contents:
      raise RuntimeError(f'Invalid complex cache {cache_file_path}')
    
    time_steps = contents['time_steps']
    frames = {}
    
    for t in progress(time_steps, desc='reading morse complexes', unit='Complexes'):
      t = int(t)
      
      critical_points = vtk_util.Read(os.path.join(cache_path, f'critical_points{t}.vtp'))

      separatrices = vtk_util.Read(os.path.join(cache_path, f'separatrices{t}.vtp'))

      segmentation = vtk_util.Read(os.path.join(cache_path, f'segmentation{t}.vtu'))
      
      complex = ttk.MorseComplex(critical_points, separatrices, segmentation)
      
      frames[t] = complex
    
    return Complex(frames)
  
  def __init__(self, frames: Dict[int, ttk.MorseComplex]) -> None:
    self.frames = frames
    
  def save(self, cache_path: str, progress: ProgressFactory):
    cache_file_path = os.path.join(cache_path, 'complex.json')
    
    for t, frame in progress(
      self.frames.items(), desc='writing morse complexes', unit='Complexes'
    ):
      vtk_util.Write(
        frame.critical_points.GetOutputPort(),
        os.path.join(cache_path, f'critical_points{t}.vtp')
      )
      
      vtk_util.Write(
        frame.separatrices.GetOutputPort(),
        os.path.join(cache_path, f'separatrices{t}.vtp')
      )
      
      vtk_util.Write(
        frame.segmentation.GetOutputPort(),
        os.path.join(cache_path, f'segmentation{t}.vtu')
      )
    
    with open(cache_file_path, 'w') as cache_file:
      contents = {
        'time_steps': list(self.frames.keys())
      }
      
      json.dump(contents, cache_file)

class ComplexConf(TypedDict):
  persistence_threshold: float
  scalar_field: str | None
  ascending: bool
  
class ComplexRule(Rule[ComplexConf, Complex]):
  def __call__(
    self, 
    dataset: Dataset,
    persistence_threshold: float,
    scalar_field: str | None,
    ascending: bool,
    progress: ProgressFactory,
  ) -> Complex:
    frames = {}
    
    for t, dataset_frame in progress(
      dataset.frames.items(), desc='computing morse complexes', unit='Complexes'
    ):
      frames[t] = ttk.MorseComplex.create(
        dataset_frame.GetOutputPort(),
        persistence_threshold=persistence_threshold,
        ascending=ascending,
        scalar_field=scalar_field
      )
    
    return Complex(frames)
  
class ComplexTarget(CacheableTarget[ComplexConf, Complex]):
  @staticmethod
  def target_type() -> str:
    return 'complex'
    
  @staticmethod
  def entity_type() -> type[Complex]:
    return Complex
  
  def __init__(
    self, 
    name: str,
    cache_path: str,
    dataset: Dataset,
    persistence_threshold: float,
    ascending: bool = True,
    scalar_field: str | None = None,
    display_name: str | None = None,
    desc: str | None = None,
    **kwargs,
  ):
    if display_name is None:
      display_name = dataset.display_name
    
    if desc is None:
      desc = dataset.desc
      
    super().__init__(
      name = name, 
      cache_path = cache_path,
      rule = ComplexRule(),
      conf = {
        'persistence_threshold': persistence_threshold,
        'ascending': ascending,
        'scalar_field': scalar_field,
      },
      depends = [dataset],
      display_name = display_name,
      desc = desc,
      **kwargs
    )