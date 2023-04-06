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
from mcopt.pipeline.progress import ProgressFactory

class FilterConf(TypedDict):
  pass

class DatasetLoadConf(TypedDict):
  type: Literal['load']
  frames: str
  src: NotRequired[str]
  download: NotRequired[str]
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
  pass