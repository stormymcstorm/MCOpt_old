"""
Dataset target definition
"""

from typing_extensions import NotRequired
from typing import (
  Literal, 
  Dict, 
  TypedDict,
  List,
  Any,
  IO,
  Optional,
  Callable,
  cast
)

from mcopt.pipeline.target import Target
from mcopt.pipeline.download import DownloadTarget
from mcopt.pipeline.progress import ProgressFactory

class FilterConf(TypedDict):
  pass

class DatasetLoadConf(TypedDict):
  type: Literal['load']
  frames: str | List[str]
  download: str
  filters: NotRequired[List[FilterConf]]

class LayerConf(TypedDict):
  type: str
  args: Dict[str, Any]
  weight: float

class DatasetGenConf(TypedDict):
  type: Literal['gen']
  layers: List[LayerConf]
  filters: NotRequired[List[FilterConf]]

DatasetConf = DatasetLoadConf | DatasetGenConf

class Dataset:
  pass

class DatasetTarget(Target[DatasetConf, Dataset]):
  """
  A target representing a dataset
  """
  
  @staticmethod
  def _validate_filter_conf(name: str, conf: Dict):
    # TODO
    raise NotImplementedError()
  
  @staticmethod
  def _validate_load_conf(name: str, conf: Dict) -> DatasetLoadConf:
    assert(conf['type'] == 'load')
    
    target = f'Dataset {name}'
    
    if 'download' not in conf:
      raise ValueError(f'{target}: must have download field')
    
    if not isinstance(conf['download'], str):
      raise ValueError(f'{target}: download must be a string')
    
    if 'frames' not in conf:
      raise ValueError(f'{target}: must frames field')
    
    if not isinstance(conf['frames'], str) and not isinstance(conf['frames'], list):
      raise ValueError(f'{target}: frames must be a string or a list of strings')
    
    return cast(DatasetLoadConf, conf)
  
  @staticmethod
  def _validate_gen_conf(name: str, conf: Dict) -> DatasetGenConf:
    # TODO
    raise NotImplementedError()
  
  @staticmethod
  def _validate_conf(name: str, conf: Dict) -> DatasetConf:
    target = f'Dataset {name}'
    
    if 'type' not in conf:
      raise ValueError(f'{target}: must have type field')
    
    if not isinstance(conf['type'], str):
      raise ValueError(f'{target}: type field must be a string')
    
    if 'filters' in conf:
      if not isinstance(conf['filters'], list):
        raise ValueError(f'{target}: filters field must be a list')
      
      for filter_conf in conf['filters']:
        DatasetTarget._validate_filter_conf(name, filter_conf)
    
    ty = conf['type']
    
    if ty == 'load':
      return DatasetTarget._validate_load_conf(name, conf)
    elif ty == 'gen':
      return DatasetTarget._validate_gen_conf(name, conf)
    else:
      raise ValueError(f'Dataset target {name} has unrecognized type: {ty}')

  def __init__(
    self, 
    name: str, 
    conf: Dict,
    cache_path: Optional[str],
    progress: Optional[ProgressFactory] = None,
    silent: bool = True,
  ):
    super().__init__(name, conf, cache_path, progress=progress, silent=silent)
  
  @property
  def target_name(self) -> str:
    return "dataset"
  
  def _load(self, downloads: Dict[str, DownloadTarget]) -> Dataset:
    raise NotImplementedError()
  
  def _save(self, dataset: Dataset):
    pass
  
  def _make(self, downloads: Dict[str, DownloadTarget]) -> Dataset:
    raise NotImplementedError()
  
  def generate(self, downloads: Dict[str, DownloadTarget]) -> Dataset:
    return super().generate(downloads)
  
  def get(self, downloads: Dict[str, DownloadTarget]) -> Dataset:
    return super().get(downloads)