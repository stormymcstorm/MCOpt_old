"""

"""

from typing import (
  Dict,
  List,
  TypeVar
)
import os

from mcpipeline.entity import Entity
from mcpipeline.target import Target, CacheableTarget, Out, Conf
from mcpipeline.targets import *
from mcpipeline.experiments import *

__all__ = ['Pipeline']

T = TypeVar('T', bound=Target)

class Pipeline:
  cache_path: str
  
  _downloads: Dict[str, DownloadTarget]
  _targets: Dict[str, Dict[str, Target]]
  
  def __init__(
    self,
    cache_path: str
  ):
    self.cache_path = cache_path
    
    self._targets = {}
    
  def add_target(
    self,
    target_cls: type[T],
    name: str,
    *args,
    **kwargs, 
  ) -> T:
    ty = target_cls.target_type()
    
    if ty not in self._targets:
      self._targets[ty] = {}
    
    if name in self._targets[ty]:
      raise ValueError(f'{ty} target with name {name} already exists')
    
    if issubclass(target_cls, CacheableTarget):
      target = target_cls(
        *args, 
        name = name,
        cache_path = os.path.join(self.cache_path, ty, name),
        **kwargs
      )
    else:
      target = target_cls(
        *args, 
        name = name,
        **kwargs
      )
    
    self._targets[ty][name] = target
    
    return target
    
  def get_target(
    self,
    target_cls: type[T],
    name: str,
  ) -> T:
    ty = target_cls.target_type()
    
    if ty not in self._targets or name not in self._targets[ty]:
      raise ValueError(f'Unrecognized {ty} target {name}')
    
    return self._targets[ty][name] # type: ignore
  
  def build_all(
    self,
    **kwargs
  ):
    for cat in self._targets.values():
      for target in cat.values():
        target.build(**kwargs)
    
  def add_download(
    self,
    name: str,
    url: str,
    **kwargs
  ) -> DownloadTarget:
    return self.add_target(
      DownloadTarget, 
      name = name,
      url = url,
      **kwargs
    )
    
  def download(self, name: str) -> DownloadTarget:
    return self.get_target(
      DownloadTarget,
      name,
    )
  
  def add_extract_zip(
    self,
    name: str,
    zips: FilePathGroupTarget,
    pattern: str = '*',
    **kwargs
  ) -> ExtractTarget:
    return self.add_target(
      ExtractZipTarget,
      name = name,
      zips = zips,
      pattern = pattern,
      **kwargs
    )
    
  def extract(self, name: str) -> ExtractTarget:
    return self.get_target(
      ExtractTarget,
      name,
    )
    
  def add_load_dataset(
    self,
    name: str,
    files: FilePathGroupTarget,
    **kwargs
  ) -> DatasetTarget:
    return self.add_target(
      LoadDatasetTarget,
      name = name,
      files = files,
      **kwargs
    )
    
  def add_gen_dataset(
    self,
    name: str,
    cls: type[GenDatasetTarget],
    **kwargs
  ) -> DatasetTarget:
    return self.add_target(
      cls,
      name = name,
      **kwargs
    )
    
  def dataset(self, name: str) -> DatasetTarget:
    return self.get_target(
      DatasetTarget,
      name
    )
    
  def add_complex(
    self, 
    name: str,
    dataset: DatasetTarget,
    persistence_threshold: float,
    **kwargs,
  ) -> ComplexTarget:
    return self.add_target(
      ComplexTarget,
      name = name,
      dataset = dataset,
      persistence_threshold = persistence_threshold,
      **kwargs
    )
    
  def complex(self, name: str) -> ComplexTarget:
    return self.get_target(
      ComplexTarget,
      name
    )
  
  def add_graph(
    self,
    name: str,
    complex: ComplexTarget,
    **kwargs,
  ) -> GraphTarget:
    return self.add_target(
      GraphTarget,
      name = name,
      complex = complex,
      **kwargs
    )
    
  def graph(self, name: str) -> GraphTarget:
    return self.get_target(
      GraphTarget,
      name
    )
  
  def add_mm_network(
    self,
    name: str,
    graph: GraphTarget,
    dist: str,
    hist: str,
    **kwargs
  ) -> MMNetworkTarget:
    return self.add_target(
      MMNetworkTarget,
      name = name,
      graph = graph,
      dist = dist,
      hist = hist,
      **kwargs
    )
    
  def mm_network(self, name: str) -> MMNetworkTarget:
    return self.get_target(
      MMNetworkTarget,
      name
    )
  
  def add_attributes(
    self,
    name: str,
    graph: GraphTarget,
    **kwargs
  ) -> AttributesTarget:
    return self.add_target(
      AttributesTarget,
      name = name,
      graph = graph,
      **kwargs
    )
  
  def attributes(self, name: str) -> AttributesTarget:
    return self.get_target(
      AttributesTarget,
      name
    )
  
  def add_max_match_pfgw(
    self, 
    name: str,
    network: MMNetworkTarget,
    graph: GraphTarget,
    attributes: AttributesTarget,
    ms: List[float],
    src_t: int,
    **kwargs
  ) -> MaxMatchPfGWTarget:
    return self.add_target(
      MaxMatchPfGWTarget,
      name = name, 
      network = network,
      graph = graph,
      attributes = attributes,
      ms = ms,
      src_t = src_t,
      **kwargs
    )
    
  def add_max_match_wasserstein(
    self, 
    name: str,
    network: MMNetworkTarget,
    graph: GraphTarget,
    attributes: AttributesTarget,
    ms: List[float],
    src_t: int,
    **kwargs
  ) -> MaxMatchWassersteinTarget:
    return self.add_target(
      MaxMatchWassersteinTarget,
      name = name, 
      network = network,
      graph = graph,
      attributes = attributes,
      ms = ms,
      src_t = src_t,
      **kwargs
    )
    
  def max_match(self, name: str) -> MaxMatch:
    return self.get_target(
      MaxMatchPfGWTarget,
      name
    )