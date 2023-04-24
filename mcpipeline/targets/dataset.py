"""
Target for datasets
"""

from __future__ import annotations
from typing import TypedDict, Dict, List, Iterable
from abc import abstractmethod
import os
import json

import vtk
import numpy as np

from mcpipeline.entity import CacheableEntity
from mcpipeline.target import CacheableTarget, Rule, Conf
from mcpipeline.targets.filegroup import FilePathGroupTarget, FilePathGroup
from mcpipeline.util import ProgressFactory
import mcpipeline.vtk as vtk_util
from mcpipeline.vtk import VTKFilter

__all__ = ['Dataset', 'DatasetTarget', 'LoadDatasetTarget', 'GenDatasetTarget']

class Dataset(CacheableEntity):
  name: str
  frames: Dict[int, vtk.vtkAlgorithm]
  
  @staticmethod
  def load(cache_path: str, progress: ProgressFactory) -> Dataset:
    cache_file_path = os.path.join(cache_path, 'dataset.json')
    
    with open(cache_file_path, 'r') as cache_file:
      contents = json.load(cache_file)

    if 'name' not in contents or 'file_paths' not in contents:
      raise RuntimeError(f'Invalid dataset cache {cache_file_path}')
    
    name = contents['name']
    file_paths = contents['file_paths']
    
    frames = {}
    
    for t, file_path in progress(file_paths.items(), desc='reading dataset frames', unit='Frame'):
      frames[int(t)] = vtk_util.Read(file_path)
      
    return Dataset(name, frames)  
    
  def __init__(self, name: str, frames: Dict[int, vtk.vtkAlgorithm]):
    self.name = name
    self.frames = frames

  def save(self, cache_path: str, progress: ProgressFactory):
    file_paths = {}
    for t, frame in progress(self.frames.items(), desc='writing dataset frames', unit='Frame'):
      file_name = os.path.join(cache_path, f'{self.name}{t}')
      
      file_paths[t] = vtk_util.Write(frame.GetOutputPort(), file_name)
    
    cache_file_path = os.path.join(cache_path, 'dataset.json')
    
    with open(cache_file_path, 'w') as cache_file:
      contents = {
        'name': self.name,
        'file_paths': file_paths
      }
      
      json.dump(contents, cache_file)

class DatasetConf(TypedDict):
  filters: List[VTKFilter]

class DatasetTarget(CacheableTarget[Conf, Dataset]):
  @staticmethod
  def target_type() -> str:
    return 'dataset'
  
  @staticmethod
  def entity_type() -> type[Dataset]:
    return Dataset
  
  def _conf_encoder(self) -> type[json.SONEncoder]:
    return vtk_util.VTKFilterEncoder
  
  def _conf_decoder(self):
    return vtk_util.VTKFilterDecoder

class LoadDatasetConf(DatasetConf):
  time_steps: List[int] | None

class LoadDatasetRule(Rule[LoadDatasetConf, Dataset]):
  name: str
  
  def __init__(self, name: str):
    self.name = name
  
  def __call__(
    self, 
    files: FilePathGroup, 
    time_steps: List[int] | None,
    filters: List[VTKFilter],
    progress: ProgressFactory,
  ) -> Dataset:
    frames = {}
    
    for file_path in progress(files.file_paths, desc='reading dataset files'):
      base = os.path.basename(file_path)
      
      num_part = ''.join(filter(str.isdigit, base))
      
      if num_part == '':
        raise ValueError(f'Failed to determine frame timestep in {base}')
      
      t = int(num_part)
      
      if time_steps is None or t in time_steps:
        frames[t] = vtk_util.Read(file_path)
    
    if len(filters) != 0:
      with progress(
        total = len(filters) * len(frames), 
        desc='applying filters',
        unit='Filters'
      ) as prog:
        for t, frame in frames.items():
          for f in filters:
            frame = f(frame.GetOutputPort())
            prog.update()
          
          frame.Update()
          frames[t] = frame   
      
    return Dataset(self.name, frames)

class LoadDatasetTarget(DatasetTarget[LoadDatasetConf]):
  def __init__(
    self, 
    name: str,
    cache_path: str,
    files: FilePathGroupTarget,
    filters: List[VTKFilter] | None = None,
    time_steps: List[int] | None = None,
    display_name: str | None = None,
    desc: str | None = None,
    **kwargs,
  ):
    if display_name is None:
      display_name = files.display_name
    
    if desc is None:
      desc = files.desc
    
    super().__init__(
      name = name,
      cache_path = cache_path,
      rule = LoadDatasetRule(name),
      conf = {
        'filters': filters if filters is not None else [],
        'time_steps': time_steps
      },
      depends = [files],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )

class GenDatasetConf(DatasetConf):
  n_frames: int | None

class GenDatasetRule(Rule[GenDatasetConf, Dataset]):
  name: str
  data: Iterable[np.ndarray]
  
  def __init__(self, name: str, data: Iterable[np.ndarray]):
    super().__init__()
    
    self.name = name
    self.data = data
    
  def __call__(
    self, 
    n_frames: int | None,
    filters: List[VTKFilter], 
    progress: ProgressFactory
  ) -> Dataset:
    frames = {}
    
    if n_frames is not None:
      for t, data in progress(
        zip(range(n_frames), self.data), desc='creating dataset frames', unit='Frames'
      ):
        frames[t] = vtk_util.PlaneSource(data)
    else:
      for t, data in progress(
        enumerate(self.data), desc='creating dataset frames', unit='Frames'
      ):
        frames[t] = vtk_util.PlaneSource(data)
        
    if len(filters) != 0:
      with progress(
        total = len(filters) * len(frames), 
        desc='applying filters',
        unit='Filters'
      ) as prog:
        for t, frame in frames.items():
          for f in filters:
            frame = f(frame.GetOutputPort())
            prog.update()
          
          frame.Update()
          frames[t] = frame 
    
    return Dataset(self.name, frames)
  
class GenDatasetTarget(DatasetTarget[GenDatasetConf]):
  def __init__(
    self, 
    name: str,
    cache_path: str,
    n_frames: int | None = None,
    filters: List[VTKFilter] | None = None,
    **kwargs,
  ):
    super().__init__(
      name = name,
      cache_path = cache_path,
      rule = GenDatasetRule(name, self.generate()),
      conf = {
        'filters': filters if filters is not None else [],
        'n_frames': n_frames
      },
      **kwargs,
    )
    
  @abstractmethod
  def generate(self) -> Iterable[np.ndarray]:
    raise NotImplementedError()
