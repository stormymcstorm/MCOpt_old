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
from mcpipeline.figures import *
from mcpipeline.util import ProgressFactory

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
    
    target = target_cls(
      *args, 
      name = name,
      cache_path = os.path.join(self.cache_path, ty, name),
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
    show_progress : bool = True,
    **kwargs
  ):
    progress = ProgressFactory(show_progress, leave=False)
    
    with progress(total = 0, desc='Building targets') as prog:
      for cat in self._targets.values():
        prog.total += len(cat.values())
        
        for target in cat.values():
          target.ensure_built(_progress=progress,**kwargs)
          
          prog.update()
    
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
  
  def add_combine_graphs(
    self,
    name: str,
    graphs: List[GraphTarget],
    **kwargs,
  ) -> CombineGraphsTarget:
    return self.add_target(
      CombineGraphsTarget,
      name = name,
      graphs = graphs,
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
  
  def add_combine_mm_network(
    self,
    name: str,
    networks: List[MMNetworkTarget],
    **kwargs,
  ) -> CombineNetworksTarget:
    return self.add_target(
      CombineNetworksTarget,
      name = name,
      networks = networks,
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
    
  def add_max_match_pw(
    self, 
    name: str,
    network: MMNetworkTarget,
    graph: GraphTarget,
    attributes: AttributesTarget,
    ms: List[float],
    src_t: int,
    **kwargs
  ) -> MaxMatchPWTarget:
    return self.add_target(
      MaxMatchPWTarget,
      name = name, 
      network = network,
      graph = graph,
      attributes = attributes,
      ms = ms,
      src_t = src_t,
      **kwargs
    )
  
  def add_max_match_pgw(
    self, 
    name: str,
    network: MMNetworkTarget,
    graph: GraphTarget,
    ms: List[float],
    src_t: int,
    **kwargs
  ) -> MaxMatchPGWTarget:
    return self.add_target(
      MaxMatchPGWTarget,
      name = name, 
      network = network,
      graph = graph,
      ms = ms,
      src_t = src_t,
      **kwargs
    )
  
  def max_match(self, name: str) -> MaxMatch:
    return self.get_target(
      MaxMatchPfGWTarget,
      name
    )
    
  def add_gw(
    self,
    name: str,
    network: MMNetworkTarget,
    **kwargs
  ) -> GWTarget:
    return self.add_target(
      GWTarget,
      name = name,
      network = network,
      **kwargs
    )
  
  def add_fgw(
    self,
    name: str,
    network: MMNetworkTarget,
    attributes: AttributesTarget,
    **kwargs
  ) -> FGWTarget:
    return self.add_target(
      FGWTarget,
      name = name,
      network = network,
      attributes = attributes,
      **kwargs
    )
  
  def add_wasserstein(
    self,
    name: str,
    network: MMNetworkTarget,
    attributes: AttributesTarget,
    **kwargs
  ) -> WassersteinTarget:
    return self.add_target(
      WassersteinTarget,
      name = name,
      network = network,
      attributes = attributes,
      **kwargs
    )
  
  def add_pfgw(
    self,
    name: str,
    network: MMNetworkTarget,
    attributes: AttributesTarget,
    m: float,
    **kwargs
  ) -> PfGWTarget:
    return self.add_target(
      PfGWTarget,
      name = name,
      network = network,
      attributes = attributes,
      m = m,
      **kwargs
    )
    
  def add_pw(
    self,
    name: str,
    network: MMNetworkTarget,
    attributes: AttributesTarget,
    m: float,
    **kwargs
  ) -> PWTarget:
    return self.add_target(
      PWTarget,
      name = name,
      network = network,
      attributes = attributes,
      m = m,
      **kwargs
    )
    
  def add_pgw(
    self,
    name: str,
    network: MMNetworkTarget,
    m: float,
    **kwargs
  ) -> PGWTarget:
    return self.add_target(
      PGWTarget,
      name = name,
      network = network,
      m = m,
      **kwargs
    )
  
  def couplings(self, name: str) -> CouplingsTarget:
    return self.get_target(
      CouplingsTarget,
      name
    )
    
  def add_mds(
    self,
    name: str,
    graphs: CombinedGraphs,
    couplings: CouplingsTarget,
    **kwargs
  ) -> MDSTarget:
    return self.add_target(
      MDSTarget,
      name = name,
      graphs = graphs,
      couplings = couplings,
      **kwargs
    )
    
  def mds(self, name: str) -> MDSTarget:
    return self.get_target(
      MDSTarget,
      name
    )
  
  def add_knearest_neighbors(
    self,
    name: str,
    graphs: CombineGraphsTarget,
    couplings: CouplingsTarget,
    **kwargs
  ) -> KNearestNeighborsTarget:
    return self.add_target(
      KNearestNeighborsTarget,
      name = name,
      graphs = graphs,
      couplings = couplings,
      **kwargs
    )
    
  def classification(self, name: str) -> ClassificationTarget:
    return self.get_target(
      ClassificationTarget,
      name
    )
  
  def add_graphs_figure(
    self,
    name: str,
    graph: GraphTarget,
    **kwargs
  ) -> GraphsFigureTarget:
    return self.add_target(
      GraphsFigureTarget,
      name = name,
      graph = graph,
      **kwargs
    )
    
  def add_couplings_figure(
    self,
    name: str,
    graph: GraphTarget,
    couplings: CouplingsTarget,
    src_t: int,
    **kwargs
  ) -> CouplingsFigureTarget:
    return self.add_target(
      CouplingsFigureTarget,
      name = name,
      graph = graph,
      couplings = couplings,
      src_t = src_t,
      **kwargs
    )
    
  def add_couplings_combined_figure(
    self,
    name: str,
    graph: GraphTarget,
    couplings: CouplingsTarget,
    src_t: int,
    nrows: int,
    ncols: int,
    **kwargs
  ) -> CouplingsCombinedFigureTarget:
    return self.add_target(
      CouplingsCombinedFigureTarget,
      name = name,
      graph = graph,
      couplings = couplings,
      src_t = src_t,
      nrows = nrows,
      ncols = ncols,
      **kwargs
    )
    
  def add_max_match_figure(
    self,
    name: str,
    max_match,
    m: float,
    **kwargs
  ) -> CouplingsFigureTarget:
    return self.add_target(
      MaxMatchFigureTarget,
      name = name,
      max_match = max_match,
      m = m,
      **kwargs
    )
    
  def add_max_match_combined_figure(
    self,
    name: str,
    max_match,
    m: float,
    **kwargs
  ) -> MaxMatchCombinedFigureTarget:
    return self.add_target(
      MaxMatchCombinedFigureTarget,
      name = name,
      max_match = max_match,
      m = m,
      **kwargs
    )
    
  def add_classification_figure(
    self,
    name: str,
    mds: MDSTransforms,
    classification: Classification,
    labels: List[str],
    **kwargs
  ) -> ClassificationFigureTarget:
    return self.add_target(
      ClassificationFigureTarget,
      name = name,
      mds = mds,
      classification = classification,
      labels = labels,
      **kwargs
    )
  
  def add_distance_figure(
    self,
    name: str,
    couplings: CouplingsTarget,
    **kwargs
  ):
    return self.add_target(
      DistanceFigureTarget,
      name = name,
      couplings = couplings,
      **kwargs
    )
