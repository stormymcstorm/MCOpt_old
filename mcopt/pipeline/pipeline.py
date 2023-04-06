"""
Logic for loading and generating pipeline targets
"""

from __future__ import annotations
from typing import (
  Dict,
  Generic,
  Iterable,
  Iterator,
  List,
  Literal, 
  Optional,
  IO,
  Sequence,
  Tuple,
  TypeVar,
  TypedDict,
  cast
)
from abc import ABC, abstractmethod
import os
from os import PathLike
import json
import zipfile
import tempfile
from glob import glob
from urllib.request import urlretrieve
from urllib.parse import urlparse

import numpy as np
import networkx as nx
from tqdm.autonotebook import tqdm
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm

from mcopt.pipeline.util import sort_files, file_ext
from mcopt.pipeline.complex import MorseComplex
import mcopt.pipeline.util.vtk as vtk_util
import mcopt.pipeline.util.gen as gen_util
from mcopt import MorseGraph

class Pipeline:
  """
  Utility for managing datasets/graph/complexes/etc.
  
  Parameters
  ----------
  
  """
  
  progress: ProgressFactory
  
  _downloads: Dict[str, DownloadTarget]
  _datasets: Dict[str, DatasetTarget]
  _complexes: Dict[str, ComplexTarget]
  _graphs: Dict[str, GraphTarget]
  _silent: bool
  
  def __init__(
    self, 
    config_path: PathLike | str,
    use_cache: bool = True,
    silent: bool = False,
    show_progress: bool = False,
  ):
    with open(config_path, 'r') as config_file:
      config = json.load(config_file)
      
    root = os.path.dirname(config_path)
    cache = 'pipeline_out' if 'cache' not in config else config['cache']
    
    show_progress = show_progress and not silent
    
    self.progress = ProgressFactory(show_progress, leave=False)
    self._silent = silent
    
    self._downloads = {}
    if 'downloads' in config:
      for name, conf in config['downloads'].items():
        downloads_cache = os.path.join(root, cache, 'downloads', name) if use_cache else None
        
        self._downloads[name] = DownloadTarget(
          name, 
          conf, 
          self,
          downloads_cache
        )
      
    self._datasets = {}
    if 'datasets' in config:
      for name, conf in config['datasets'].items():
        datasets_cache = os.path.join(root, cache, 'datasets', name) if use_cache else None
        
        self._datasets[name] = DatasetTarget(
          name,
          conf,
          self,
          datasets_cache
        )
        
    self._complexes = {}
    if 'complexes' in config:
      for name, conf in config['complexes'].items():
        complexes_cache = os.path.join(root, cache, 'complexes', name) if use_cache else None
        
        self._complexes[name] = ComplexTarget(
          name,
          conf,
          self, 
          complexes_cache
        )
        
    self._graphs = {}
    if 'graphs' in config:
      for name, conf in config['graphs'].items():
        graphs_cache = os.path.join(root, cache, 'graphs', name) if use_cache else None
        
        self._graphs[name] = GraphTarget(
          name,
          conf,
          self,
          graphs_cache
        )

  def generate_all(self):
    for target in self._downloads.values():
      target.get()
    
    for target in self._datasets.values():
      target.get()
      
    for target in self._complexes.values():
      target.get()
      
    for target in self._graphs.values():
      target.get()
  
  def download(self, name: str, idxs: Optional[slice] = None) -> Download:
    if name not in self._downloads:
      raise ValueError(f'Unrecognized download {name}')
    
    return self._downloads[name].get(idxs)
  
  def dataset(self, name: str, idxs: Optional[slice] = None) -> Dataset:
    if name not in self._datasets:
      raise ValueError(f'Unrecognized dataset {name}')
    
    return self._datasets[name].get(idxs)
  
  def complex(self, name: str, idxs: Optional[slice] = None) -> Complex:
    if name not in self._complexes:
      raise ValueError(f'Unrecognized complex {name}')
    
    return self._complexes[name].get(idxs)
  
  def graph(self, name: str, idxs: Optional[slice] = None) -> Graph:
    if name not in self._graphs:
      raise ValueError(f'Unrecognized graph {name}')
    
    return self._graphs[name].get(idxs)
  
  def _urlretrieve(self, url: str, out: str, desc=''):
    with self.progress(
      desc=desc,
      unit='B',
      unit_scale=True,
      unit_divisor=1024,
    ) as prog:
      urlretrieve(url, out, reporthook=prog.report_hook)
      
  def log(self, msg):
    if not self._silent:
      print(msg)
      
class ProgressFactory:
  show: bool
  tqdm_kwargs: dict
  
  def __init__(self, show: bool, **kwargs):
    self.show = show
    self.tqdm_kwargs = kwargs
    
  def __call__(self, *args, **kwargs) -> ProgressBar:
    if self.show:
      return ProgressBar(t = tqdm(*args, **self.tqdm_kwargs, **kwargs))
    else:
      return ProgressBar(iter=args[0])

T = TypeVar("T")

class ProgressBar(Generic[T]):
  _t: Optional[tqdm]
  _iter: Optional[Iterable[T]]
  
  def __init__(self, t: Optional[tqdm] = None, iter: Optional[Iterable[T]] = None):
    self._t = t
    self._iter = iter
    
  def __enter__(self) -> ProgressBar:
    return self
  
  def __exit__(self, *args, **kwargs):
    if self._t is not None:
      self._t.__exit__(*args, **kwargs)
      
  def __iter__(self) -> Iterator[T]:
    if self._t is not None:
      return self._t.__iter__()
    
    assert (self._iter is not None)
    
    return self._iter.__iter__()
  
  def report_hook(self, b: int, bsize: int, tsize: int):
    if self._t is not None:
      self._t.total = tsize
      
      self._t.update(b * bsize - self._t.n)

EntityFrame = TypeVar('EntityFrame')

class Entity(Generic[EntityFrame]):
  name: str
  frames: Dict[int, EntityFrame]
  
  def __init__(self, name: str, frames: Dict[int, EntityFrame]):
    self.name = name
    self.frames = frames
    
class Download(Entity[str]):
  file: IO
  _tempdir: tempfile.TemporaryDirectory
  
  def __init__(self, name: str, file: IO, frame_pattern: str = '*'):    
    zip = zipfile.ZipFile(file.name)
    
    tempdir = tempfile.TemporaryDirectory()
    
    zip.extractall(tempdir.name)
    
    frame_list = glob(os.path.join(tempdir.name, frame_pattern))
    frame_list = sort_files(frame_list)
    
    frames = {}
    for i, frame in enumerate(frame_list):
      frames[i] = frame
    
    super().__init__(name, frames) 
    
    self.file = file
    self._tempdir = tempdir
    
class Dataset(Entity[vtkAlgorithm]):
  pass

class Complex(Entity[MorseComplex]):
  pass

class Graph(Entity[MorseGraph]):
  pass

TargetEntity = TypeVar('TargetEntity', bound=Entity)
TargetConf = TypeVar('TargetConf', bound=TypedDict)

class Target(ABC, Generic[TargetConf, TargetEntity]):
  target_name: str = "unknown"
  
  name: str
  pipeline: Pipeline
  cache_path: Optional[str]
  conf: TargetConf
  
  _entity: Optional[TargetEntity]
  
  @staticmethod
  @abstractmethod
  def _validate_config(name: str, conf: Dict) -> TargetConf:
    raise NotImplementedError()
  
  def __init__(
    self,
    name: str,
    conf: Dict,
    pipeline: Pipeline,
    cache_path: Optional[str],
  ):
    self.name = name
    self.pipeline = pipeline
    self.cache_path = cache_path
    self.conf = self._validate_config(name, conf)
    
    self._entity = None
  
  @classmethod
  @abstractmethod
  def _load(cls, *args, **kwargs) -> TargetEntity:
    raise NotImplementedError()
  
  @classmethod
  @abstractmethod
  def _save(cls):
    raise NotImplementedError()
  
  @classmethod
  @abstractmethod
  def _make(cls, *args, **kwargs) -> TargetEntity:
    raise NotImplementedError()
  
  def _config_changed(self):
    if self.cache_path is None:
      return True
    
    conf_path = os.path.join(self.cache_path, 'config.json')
    if os.path.exists(conf_path):
      with open(conf_path, 'r') as config_file:
        contents = json.load(config_file)
        
        return contents != self.conf
    
    return True
  
  def _save_config(self):
    assert(self.cache_path is not None)
    os.makedirs(self.cache_path, exist_ok=True)
    
    conf_path = os.path.join(self.cache_path, 'config.json')
    with open(conf_path, 'w') as config_file:
      json.dump(self.conf, config_file)
  
  def _get(self) -> TargetEntity:
    if self._entity is not None:
      return self._entity
    
    self.pipeline.log(f'> Generating {self.name} {self.target_name}')
    
    if not self._config_changed():
      self.pipeline.log(f'  config unchanged, loading {self.target_name}')
      self._entity = self._load()
      self.pipeline.log(f'  loaded {len(self._entity.frames)} frames')
      
      return self._entity
    
    self._entity = self._make()
    
    self.pipeline.log(f'  generated {len(self._entity.frames)} frames')
    
    if self.cache_path is not None:
      self.pipeline.log(f'  saving {self.target_name} to {self.cache_path}')
      self._save()
      self._save_config()
      
    return self._entity
  
  def get(self, idxs: Optional[slice] = None) -> TargetEntity:
    entity = self._get()
    
    if idxs is not None:
      frames = {}
      
      for i, frame in entity.frames.items()[idxs]:
        frames[i] = frame
      
      entity = TargetEntity(entity.name, frames)
      
    return entity

class DownloadConf(TypedDict):
  url: str
  file_name: str
  frame_pattern: str

class DownloadTarget(Target[DownloadConf, Download]):
  target_name = "download"
  
  @staticmethod
  def _validate_config(name: str, conf: Dict) -> DownloadConf:
    target = f'Download {name}'
    
    if 'url' not in conf:
      raise ValueError(f'{target}: url field required')
    
    if not isinstance(conf['url'], str):
      raise ValueError(f'{target}: url must be a string')
    
    if 'file_name' not in conf:
      url_parts = urlparse(conf['url'])
      
      conf['file_name'] = os.path.basename(url_parts.path)
      
    file_name = conf['file_name']
    ext = file_ext(file_name)
    
    if ext is not None and ext != 'zip':
      raise ValueError(f'{target}: file must be a zip file')
    
    if 'frame_pattern' not in conf:
      conf['frame_pattern'] = '*'
    
    return cast(DownloadConf, conf)
  
  def _load(self) -> Download:
    assert(self.cache_path is not None)
    
    file_path = os.path.join(self.cache_path, self.conf['file_name'])
    
    return Download(
      self.name,
      open(file_path, 'r'),
      self.conf['frame_pattern']
    )
  
  def _save(self):
    assert(self.cache_path is not None)
    assert(self._entity is not None)
    
    os.makedirs(self.cache_path, exist_ok=True)
    
    file_path = os.path.join(self.cache_path, self.conf['file_name'])
    
    if file_path == self._entity.file.name:
      return
    
    with open(file_path, 'w') as file:
      file.write(self._entity.file.read())
  
  def _make(self) -> Download:
    if self.cache_path is not None:
      os.makedirs(self.cache_path, exist_ok=True)
      
      file_path = os.path.join(self.cache_path, self.conf['file_name'])
      file = open(file_path, 'w')
    else:
      file = tempfile.TemporaryFile()
      
      
    file_name = self.conf['file_name']
    self.pipeline._urlretrieve(
      self.conf['url'], 
      file.name,
      desc=f'downloading {file_name}'
    )
    
    return Download(self.name, file, self.conf['frame_pattern'])

class DatasetFilterConf(TypedDict):
  type: str
  args: dict
  
class DatasetLayersConf(TypedDict):
  type: str
  weight: float
  args: dict

class DatasetLoadConf(TypedDict):
  type: Literal['load']
  download: str
  filters: List[DatasetFilterConf]
  
class DatasetGenConf(TypedDict):
  type: Literal['gen']
  shape: Sequence[int]
  layers: List[DatasetLayersConf]
  filters: List[DatasetFilterConf]
  
DatasetConf = DatasetLoadConf | DatasetGenConf

class DatasetTarget(Target[DatasetConf, Dataset]):
  target_name = "dataset"
  
  @staticmethod
  def _validate_filters(name: str, conf: Dict):
    target = f'Dataset {name}'
    
    if 'type' not in conf:
      raise ValueError(f'{target}: filters must have type field')
    
    if not isinstance(conf['type'], str):
      raise ValueError(f'{target}: filter type must be a string')
    
    ty = conf['type']
    
    if ty not in vtk_util.FILTERS:
      raise ValueError(f'{target}: unrecognized filter type {ty}')
    
    if 'args' not in conf:
      conf['args'] = {}
      
  @staticmethod
  def _validate_load_config(name: str, conf: Dict) -> DatasetLoadConf:
    assert(conf['type'] == 'load')
    
    target = f'Dataset {name}'
    
    if 'download' not in conf:
      raise ValueError(f'{target}: load dataset must have download field')
    
    if not isinstance(conf['download'], str):
      raise ValueError(f'{target}: download field must be a string')
    
    return cast(DatasetLoadConf, conf)
  
  @staticmethod
  def _validate_layers(name: str, conf: Dict):
    target = f'Dataset {name}'
    
    if 'type' not in conf:
      raise ValueError(f'{target}: layers must have type field')
    
    if not isinstance(conf['type'], str):
      raise ValueError(f'{target}: layer type must be a string')
    
    ty = conf['type']
    
    if ty not in gen_util.GEN_FUNCTIONS:
      raise ValueError(f'{target}: unrecognized layer type {ty}')
    
    if 'args' not in conf:
      conf['args'] = {}
      
    if 'weight' not in conf:
      conf['weight'] = 1
  
  @staticmethod
  def _validate_gen_config(name: str, conf: Dict) -> DatasetGenConf:
    assert(conf['type'] == 'gen')
    
    target = f'Dataset {name}'
    
    if 'shape' not in conf:
      raise ValueError(f'{target}: gen dataset must have shape field')
    
    if 'layers' not in conf:
      raise ValueError(f'{target}: gen dataset must have layers field')
    
    for layer_conf in conf['layers']:
      DatasetTarget._validate_layers(name, layer_conf)
    
    return cast(DatasetGenConf, conf)
  
  @staticmethod
  def _validate_config(name: str, conf: Dict) -> DatasetConf:
    target = f'Dataset {name}'
    
    if 'type' not in conf:
      raise ValueError(f'{target}: type field required')
    
    if not isinstance(conf['type'], str):
      raise ValueError(f'{target}: type field must be a string')
    
    if 'filters' in conf:
      for filter_conf in conf['filters']:
        DatasetTarget._validate_filters(name, filter_conf)
    else:
      conf['filters'] = []
    
    ty = conf['type']
    
    if ty == 'load':
      return DatasetTarget._validate_load_config(name, conf)
    elif ty == 'gen':
      return DatasetTarget._validate_gen_config(name, conf)
    else:
      raise ValueError(f'{target}: unrecognized type {ty}')
    
  def _load(self) -> Dataset:
    assert(self.cache_path is not None)
    
    frame_files = glob(os.path.join(self.cache_path, f'{self.name}*'))
    frame_files = sort_files(frame_files)
    assert(len(frame_files) > 0)
    
    frames = {}
    for i, frame_file in enumerate(self.pipeline.progress(frame_files, desc='read dataset frames')):
      frames[i] = vtk_util.Read(frame_file)
      
    return Dataset(self.name, frames)
  
  def _save(self):
    assert(self.cache_path is not None)
    assert(self._entity is not None)
    os.makedirs(self.cache_path, exist_ok=True)
    
    for i, frame in self.pipeline.progress(self._entity.frames.items(), desc='writing dataset frames'): 
      vtk_util.Write(
        frame.GetOutputPort(), 
        os.path.join(self.cache_path, f'{self._entity.name}{i}')
      )
      
  def _make_load(self) -> Dict[int, vtkAlgorithm]:
    conf = cast(DatasetLoadConf, self.conf)
    
    download = self.pipeline.download(conf['download'])
    
    frames = {}
    for i, frame_file in self.pipeline.progress(download.frames.items(), desc='read dataset frames'):
      frames[i] = vtk_util.Read(frame_file)
    
    return frames
  
  def _make_gen(self) -> Dict[int, vtkAlgorithm]:
    conf = cast(DatasetGenConf, self.conf)
    
    shape = conf['shape'] 
    data = np.zeros(shape)
    
    for layer_conf in conf['layers']:
      args = layer_conf['args']
      weight = layer_conf['weight']
      
      ty = layer_conf['type']
      
      data += gen_util.GEN_FUNCTIONS[ty](shape=shape, **args) * weight

    return {0: vtk_util.PlaneSource(data)}
  
  def _make(self) -> Dataset:
    if self.conf['type'] == 'load':
      frames = self._make_load()
    elif self.conf['type'] == 'gen':
      frames = self._make_gen()
    else:
      raise NotImplementedError()
    
    for i, frame in self.pipeline.progress(frames.items(), desc='applying filters'):
      for filter_conf in self.conf['filters']:
        ty = filter_conf['type']
        args = filter_conf['args']
        
        frame = vtk_util.FILTERS[ty](frame.GetOutputPort(), **args)
        
      frames[i] = frame
    
    return Dataset(self.name, frames)

class ComplexConf(TypedDict):
  dataset: str
  persistence_threshold: float
  field_name: Optional[str]

class ComplexTarget(Target[ComplexConf, Complex]):
  target_name = "complex"
  
  @staticmethod
  def _validate_config(name: str, conf: Dict) -> ComplexConf:
    target = f'Complex {name}'
    
    if 'dataset' not in conf:
      raise ValueError(f'{target}: dataset field required')
    
    if not isinstance(conf['dataset'], str):
      raise ValueError(f'{target}: dataset field must be a string')
    
    if 'persistence_threshold' not in conf:
      raise ValueError(f'{target}: persistence_threshold field required')
    
    if not isinstance(conf['persistence_threshold'], float):
      raise ValueError(f'{target}: persistence_threshold field must be a float')
    
    if 'field_name' not in conf:
      conf['field_name'] = None
    
    return cast(ComplexConf, conf)
  
  def _load(self) -> Complex:
    assert(self.cache_path is not None)
    
    crit_files = glob(os.path.join(self.cache_path, f'critical_points*'))
    crit_files = sort_files(crit_files)
    assert(len(crit_files) > 0)
    
    sep_files = glob(os.path.join(self.cache_path, f'separatrices*'))
    sep_files = sort_files(sep_files)
    assert(len(sep_files) > 0)
    
    seg_files = glob(os.path.join(self.cache_path, f'segmentation*'))
    seg_files = sort_files(seg_files)
    assert(len(seg_files) > 0)
    
    assert(len(crit_files) == len(seg_files))
    assert(len(seg_files) == len(sep_files))
    
    frames = {}
    for i, (crit_file, sep_file, seg_file) in \
      enumerate(zip(crit_files, sep_files, self.pipeline.progress(seg_files, desc='reading complex frames'))):
      critical_points = vtk_util.Read(crit_file)
      separatrices = vtk_util.Read(sep_file)
      segmentation = vtk_util.Read(seg_file)
      
      morse_complex = MorseComplex(critical_points, separatrices, segmentation)
      
      frames[i] = morse_complex
    
    return Complex(self.name, frames)
  
  def _save(self):
    assert(self.cache_path is not None)
    assert(self._entity is not None)
    os.makedirs(self.cache_path, exist_ok=True)
    
    for i, frame in self.pipeline.progress(self._entity.frames.items(), desc='writing dataset frames'):
      vtk_util.Write(
        frame.critical_points.GetOutputPort(),
        os.path.join(self.cache_path, f'critical_points{i}')
      )
      
      vtk_util.Write(
        frame.separatrices.GetOutputPort(),
        os.path.join(self.cache_path, f'separatrices{i}')
      )
      
      vtk_util.Write(
        frame.segmentation.GetOutputPort(),
        os.path.join(self.cache_path, f'segmentation{i}')
      )
      
  def _make(self) -> Complex:
    dataset = self.pipeline.dataset(self.conf['dataset'])
    persistence_threshold = self.conf['persistence_threshold']
    field_name = self.conf['field_name']
    
    frames = {}
    for i, frame in self.pipeline.progress(dataset.frames.items(), desc='generating complex'):
      frames[i] = MorseComplex.create(
        frame.GetOutputPort(),
        persistence_threshold=persistence_threshold,
        field_name=field_name
      )
    
    return Complex(self.name, frames)
  
class GraphConf(TypedDict):
  complex: str
  sample_rate: int
  sample_mode: str
  
class GraphTarget(Target[GraphConf, Graph]):
  target_name = 'graph'
  
  def __init__(
    self, 
    name: str, 
    conf: Dict, 
    pipeline: Pipeline, 
    cache_path: Optional[str]
  ):
    super().__init__(name, conf, pipeline, cache_path)
  
  @staticmethod
  def _validate_config(name: str, conf: Dict) -> GraphConf:
    target = f'Graph {name}'
    
    if 'complex' not in conf:
      raise ValueError(f'{target}: complex field required')
    
    if not isinstance(conf['complex'], str):
      raise ValueError(f'{target}: complex field must be a string')
    
    if 'sample_rate' not in conf:
      raise ValueError(f'{target}: sample_rate field required')
    
    if not isinstance(conf['sample_rate'], int):
      raise ValueError(f'{target}: sample_rate field must be an int')
    
    if 'sample_mode' not in conf:
      conf['sample_mode'] = 'step'
    
    return cast(GraphConf, conf)
  
  def _load(self) -> Graph:
    assert(self.cache_path is not None)
    
    graph_files = glob(os.path.join(self.cache_path, f'{self.name}*.json'))
    graph_files = sort_files(graph_files)
    
    frames = {}
    for i, graph_file in enumerate(self.pipeline.progress(graph_files, desc='read graph frames')):
      with open(graph_file, 'r') as file:
        data = json.load(file)
        critical_nodes = set(data['critical_nodes'])
        
        graph = nx.from_dict_of_dicts(
          data['edges'],
          create_using=MorseGraph(critical_nodes)
        )
        
        nodes = {}
        for n, node_data in data['nodes'].items():
          node_data['pos2'] = np.array(node_data['pos2'])
          
          nodes[n] = node_data
          
        graph.add_nodes_from(nodes)
        
        frames[i] = graph
      
    return Graph(self.name, frames)
  
  def _save(self):
    assert(self.cache_path is not None)
    assert(self._entity is not None)
    os.makedirs(self.cache_path, exist_ok=True)
    
    for i, frame in self.pipeline.progress(self._entity.frames.items(), desc='writing graph frames'):
      with open(os.path.join(self.cache_path, f'{self._entity.name}{i}.json'), 'w') as file:
        data = {}
        data['critical_nodes'] = list(frame.critical_nodes)
        data['nodes'] = {}
        
        for n, node_data in frame.nodes(data=True):
          data['nodes'][n] = {
            'pos2': list(node_data['pos2']),
          }
        
        data['edges'] = nx.to_dict_of_dicts(frame)
        
        json.dump(data, file)
  
  def _make(self) -> Graph:
    complex = self.pipeline.complex(self.conf['complex'])
    sample_rate = self.conf['sample_rate']
    sample_mode = self.conf['sample_mode']
    
    frames = {}
    for i, frame in self.pipeline.progress(complex.frames.items(), desc='generating graph'):
      frames[i] = frame.to_graph().sample(sample_rate, sample_mode)
      
    return Graph(self.name, frames) 
      