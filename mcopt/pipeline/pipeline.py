"""
Logic for loading and generating datasets/complexes
"""

from typing import Optional, List
import os
import json
import tempfile
import zipfile
from glob import glob
import urllib.request
import urllib.parse

from tqdm import tqdm
import numpy as np

from mcopt.pipeline.util import file_ext, sort_files
from mcopt.pipeline.dataset import Dataset
from mcopt.pipeline.complex import Complex, MorseComplex
import mcopt.pipeline.vtk as vtk_util
import mcopt.pipeline.gen as gen_util

def conf_changed(file_path, current):
  if os.path.exists(file_path):
    file = open(file_path)
    contents = json.load(file)
    file.close()
    
    return contents != current
    
  return True
  
class Pipeline:
  def __init__(
    self,
    config_path : str,
    use_cache=True
  ):
    self._root = os.path.dirname(config_path)
    
    with open(config_path) as config_file:
      self._config = json.load(config_file)
    
    self._datasets = {}
    self._complexes = {}
    
    self._use_cache = use_cache
    
    if 'cache' in self._config:
      self._cache_path = os.path.join(self._root, self._config['cache'])
    else:
      self._cache_path = os.path.join(self._root, 'pipeline_out')
      
    self._dataset_cache_path = os.path.join(self._cache_path, 'datasets')  
    os.makedirs(self._dataset_cache_path, exist_ok=True)
    
    self._complex_cache_path = os.path.join(self._cache_path, 'complexes')
    os.makedirs(self._complex_cache_path, exist_ok=True)
    
    self._download_cache_path = os.path.join(self._cache_path, 'downloads')
    os.makedirs(self._download_cache_path, exist_ok=True)
      
  def _read_dataset(self, dir, name):
    frame_files = glob(os.path.join(dir, f'{name}*'))
    frame_files = sort_files(frame_files)
    assert(len(frame_files) > 0)
    
    frames = []
    for frame_file in tqdm(frame_files, desc='read dataset frames', leave=False):
      frames.append(vtk_util.Read(frame_file))
      
    return Dataset(name, frames)
  
  def _write_dataset(self, dir, dataset):
    os.makedirs(dir, exist_ok=True)
    
    for i, frame in enumerate(tqdm(dataset.frames, desc='writing dataset frames', leave=False)):
      vtk_util.Write(frame.GetOutputPort(), os.path.join(dir, f'{dataset.name}{i}'))

  def _make_load_frames(self, conf):
    assert conf['type'] == 'load'
    
    if 'src' not in conf:
      raise ValueError('load datasets must have a \'src\' field')
    
    src = conf['src']
    
    if src.startswith('http'):
      filename = os.path.join(self._download_cache_path, os.path.basename(urllib.parse.urlparse(src).path))
      src, _ = urllib.request.urlretrieve(src, filename)
    
    src = os.path.join(self._root, src)
    src_ext = file_ext(src)
    
    tempdir = None
    if src_ext == 'zip':
      tempdir = tempfile.TemporaryDirectory()
      
      with zipfile.ZipFile(src) as src_zip:
        src_zip.extractall(tempdir.name)
      
      src = tempdir.name
      src_ext = None
      
    if src_ext is None and not os.path.isdir(src):
      raise ValueError(f'expected {src} to be a directory')
     
    frame_files = []
    
    if 'frames' not in conf:
      frame_files = [src]
    elif isinstance(conf['frames'], str):
      frame_files = glob(os.path.join(src, conf['frames']))
      
      if len(frame_files) == 0:
        raise ValueError('Glob %s does not match any files' % (conf['frames']))
    
      frame_files = sort_files(frame_files)
    else:
      # TODO: probably need to make these relative to the root
      frame_files = conf['frames']
      
    frames = []
    
    for frame_file in tqdm(frame_files, desc='reading dataset frames', leave=False):
      frames.append(vtk_util.Read(frame_file))
      
    if tempdir is not None:
      tempdir.cleanup()
    
    return frames
    
  def _make_gen_frames(self, conf):
    assert conf['type'] == 'gen'
    
    shape = conf['shape']
    layers = conf['layers']
    
    data = np.zeros(shape)
    
    for layer in layers:
      args = layer['args'] if 'args' in layer else {}
      weight = layer['weight'] if 'weight' in layer else 1
      
      ty = layer['type']
      
      if ty not in gen_util.GEN_FUNCTIONS:
        raise ValueError(f'Unrecognized layer type {ty}')
      
      data += gen_util.GEN_FUNCTIONS[ty](shape=shape, **args) * weight

    return [
      vtk_util.PlaneSource(data)
    ]
    
  def _load_dataset(self, name):
    if name in self._datasets:
      return
    
    if name not in self._config['datasets']:
      raise ValueError(f'Unrecognized dataset {name}')
    
    conf = self._config['datasets'][name]
    out = os.path.join(self._dataset_cache_path, name)
    
    print(f'> Loading {name} dataset')
    
    cached_conf_file_path = os.path.join(out, 'gen_config.json')
    
    if self._use_cache and not conf_changed(cached_conf_file_path, conf):
      print(f'  config unchanged, reading dataset')
      dataset = self._read_dataset(out, name)
      
      self._datasets[name] = dataset
      return
    
    ty = conf['type']
    
    if ty == 'load':
      frames = self._make_load_frames(conf)
    elif ty == 'gen':
      frames = self._make_gen_frames(conf)
    else:
      raise ValueError(f'Unrecognized dataset type {ty}')

    if 'filters' in conf:
      filters = conf['filters']
      
      for i, frame in enumerate(tqdm(frames, desc='applying filters', leave=False)):
        for filter in filters:
          ty = filter['type']
          args = filter['args'] if 'args' in filter else {}
          
          if ty in vtk_util.FILTERS:
            frame = vtk_util.FILTERS[ty](frame.GetOutputPort(), **args)
            frame.Update()
          else:
            raise ValueError(f'Unrecognized filter: {ty}')

        frames[i] = frame
        
    dataset = Dataset(name, frames)
    
    self._write_dataset(out, dataset)
    
    conf_file = open(cached_conf_file_path, mode='w')
    json.dump(conf, conf_file)
    
    self._datasets[name] = dataset
  
  def _read_complex(self, dir, name):
    crit_files = glob(os.path.join(dir, 'critical_points*'))
    crit_files = sort_files(crit_files)
    assert(len(crit_files) > 0)
    
    seg_files = glob(os.path.join(dir, 'segmentation*'))
    seg_files = sort_files(seg_files)
    assert(len(seg_files) > 0)
    
    sep_files = glob(os.path.join(dir, 'separatrices*'))
    sep_files = sort_files(sep_files)
    assert(len(sep_files) > 0)
    
    assert(len(crit_files) == len(seg_files))
    assert(len(seg_files) == len(sep_files))
    
    frames = []
    
    for crit_file, sep_file, seg_file in zip(crit_files, sep_files, tqdm(seg_files, desc='reading complex frames', leave=False)):
      critical_points = vtk_util.Read(crit_file)
      separatrices = vtk_util.Read(sep_file)
      segmentation = vtk_util.Read(seg_file)
      
      morse_complex = MorseComplex(critical_points, separatrices, segmentation)
      
      frames.append(morse_complex)
    
    return Complex(name, frames)
  
  def _write_complex(self, dir, complex):
    os.makedirs(dir, exist_ok=True)
    
    for i, frame in enumerate(tqdm(complex.frames, desc='writing complex frames', leave=False)):
      vtk_util.Write(
        frame.critical_points.GetOutputPort(),
        os.path.join(dir, f'critical_points{i}')
      )
      vtk_util.Write(
        frame.separatrices.GetOutputPort(),
        os.path.join(dir, f'separatrices{i}')
      )
      vtk_util.Write(
        frame.segmentation.GetOutputPort(),
        os.path.join(dir, f'segmentation{i}')
      )
  
  def _load_complex(self, name):
    if name in self._complexes:
      return
    
    if name not in self._config['complexes']:
      raise ValueError(f'Unrecognized complex {name}')
    
    conf = self._config['complexes'][name]
    out = os.path.join(self._complex_cache_path, name)
    
    print(f'> Loading {name} complex')
    
    cached_conf_file_path = os.path.join(out, 'gen_config.json')

    if self._use_cache and not conf_changed(cached_conf_file_path, conf):
      print(f'  config unchanged, reading complex')
      
      complex = self._read_complex(out, name)
      self._complexes[name] = complex
      return
    
    dataset = self.dataset(conf['dataset']) 
    persistence_threshold = conf['persistence_threshold']
    field_name = conf['field_name'] if 'field_name' in conf else None
    
    frames = []
    
    for frame in tqdm(dataset.frames, desc='generating complex', leave=False) :
      complex_frame = MorseComplex.create(
        frame.GetOutputPort(), 
        persistence_threshold=persistence_threshold,
        field_name=field_name
      )
      frames.append(complex_frame)

    complex = Complex(name, frames)
    self._write_complex(out, complex)
    
    conf_file = open(cached_conf_file_path, mode='w')
    json.dump(conf, conf_file)
    
    self._complexes[name] = complex
      
  def dataset(self, name) -> Dataset:
    self._load_dataset(name)
    assert name in self._datasets
    
    return self._datasets[name]
  
  def complex(self, name) -> Complex:
    self._load_complex(name)
    assert name in self._complexes
    
    return self._complexes[name]
    
  def load_all(self):
    self.load_datasets()
    self.load_complexes()
      
  def load_datasets(self):
    for name in self._config['datasets'].keys():
      self._load_dataset(name)
      
  def load_complexes(self):
    for name in self._config['complexes'].keys():
      self._load_complex(name)
      