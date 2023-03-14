#!/usr/bin/env python

import os
import vtk
import numpy as np
import pandas as pd
import json

from mcopt.util.data import (
  Sinusoidal, 
  Combine, 
  Distance, 
  Gaussian,
  GaussianNoise,
  Smooth,
)
from mcopt.util.vtk import (
  Tetrahedralize,
  Warp,
  Plane,
)
from mcopt.util.io import save_complex, load_vti, save_vtp, save_vtu
from mcopt.morse_complex import (MorseSmaleComplex, MorseComplex)


  

ROOT = os.path.dirname(__file__)
CONFIG = os.path.join(ROOT, "gen_config.json")
GEN_DATA = os.path.join(ROOT, "gen_data")  

def complex_from_arr(arr, scale_factor, persistence_threshold):
  plane = Plane(arr)
  tetra = Tetrahedralize(plane.GetOutputPort())
  warp = Warp(tetra.GetOutputPort(), scale_factor = scale_factor)
  
  return MorseComplex.create(
    warp.GetOutputPort(), 
    persistence_threshold=persistence_threshold,
    field_name='data'
  )

# def gen_complex(dataset):
#   assert(dataset['type'] == 'gen')
  
#   rng = np.random.default_rng(dataset['random']) if 'random' in dataset else None
  
#   shape = dataset['shape']
#   data = np.zeros(shape)
      
#   for layer in dataset['layers']:
#     layer_type = layer['type'].lower()
#     layer_args = layer['args'] if 'args' in layer else {}
    
#     if layer_type == 'sinusoidal':
#       layer_data = Sinusoidal(shape = shape, **layer_args)
#     elif layer_type == 'distance':
#       layer_data = Distance(shape=shape, **layer_args)
#     elif layer_type == 'gaussian':
#       layer_data = Gaussian(shape=shape, **layer_args)
#     else:
#       raise ValueError(f'Unrecognized layer type: {layer_type}')
    
#     layer_weight = layer['weight'] if 'weight' in layer else 1
#     data += layer_data * layer_weight
    
#   if 'noise' in dataset:
#     data += Smooth(GaussianNoise(shape=dataset['shape'], rng=rng) * dataset['noise'])
    
#   return data, complex_from_arr(
#     data,
#     dataset['scale'],
#     dataset['persistence_threshold']
#   )

# def add_noise_complex(dataset, from_data):
#   assert(dataset['type'] == 'add_noise')
  
#   rng = np.random.default_rng(dataset['random']) if 'random' in dataset else None
  
#   data = from_data + Smooth(GaussianNoise(shape=from_data.shape, rng=rng) * dataset['noise'])
  
#   return complex_from_arr(
#     data,
#     dataset['scale'],
#     dataset['persistence_threshold']
#   )

# def from_data_file_complex(dataset):
#   assert(dataset['type'] == 'from_data_file')
  
#   data = load_vti(os.path.join(ROOT, dataset['path']))
  
#   tetra = Tetrahedralize(data.GetOutputPort())
#   warp = Warp(tetra.GetOutputPort(), dataset['scale']) 
  
#   return MorseComplex.create(
#     warp.GetOutputPort(),
#     persistence_threshold=dataset['persistence_threshold'],
#     field_name='ImageScalars'
#   )
  
#   pass

# def make_complexes(config):
#   complexes = {}
  
#   gen_datasets = []
#   add_noise_datasets = []
#   from_data_file_datasets = []
  
#   for dataset in config['datasets']:
#     type = dataset['type']
    
#     if type == 'gen':
#       gen_datasets.append(dataset)
#     elif type == 'add_noise':
#       add_noise_datasets.append(dataset)
#     elif type == 'from_data_file':
#       from_data_file_datasets.append(dataset)
#     else:
#       raise ValueError(f'Unrecognized dataset type: {type}')
    
#   for dataset in gen_datasets:
#     name = dataset['name']
#     data, complex = gen_complex(dataset)
    
#     complexes[name] = {
#       'complex': complex,
#       'data': data
#     }
    
#   for dataset in from_data_file_datasets:
#     name = dataset['name']
#     complex = from_data_file_complex(dataset)
    
#     complexes[name] = {
#       'complex': complex
#     }
    
#   for dataset in add_noise_datasets:
#     name = dataset['name']
#     fro = dataset['from']
    
#     if fro not in complexes:
#       raise ValueError(f'Could not find dataset {fro}')
#     if 'data' not in complexes[fro]:
#       raise ValueError(f'Cannot access raw data for {fro}')
    
#     data = complexes[fro]['data']
    
#     complex = add_noise_complex(dataset, data)
    
#     complexes[name] = {
#       'complex': complex
#     }
  
#   return complexes

def Noise(scale=1, random_state = None, **kwargs):
  rng = np.random.default_rng(random_state)
  
  return Smooth(GaussianNoise(rng=rng, **kwargs) * scale)

GEN_FUNCTIONS = {
  'sinusoidal': Sinusoidal,
  'distance': Distance,
  'gaussian': Gaussian,
  'noise': Noise,
}

def make_gen_data(dataset):
  shape = dataset['shape']
  data = np.zeros(shape)
  
  for layer in dataset['layers']:
    type = layer['layer']
    
    if type not in GEN_FUNCTIONS:
      raise ValueError(f'Unrecognized layer type: {type}')
    
    fn = GEN_FUNCTIONS[type]
    args = {} if 'args' not in layer else layer['args']
    weight = 1 if 'weight' not in layer else layer['weight']
    
    data += fn(shape=shape, **args) * weight
  
  return 'data', Plane(data)

def make_load_data(dataset):
  path =  dataset['path']
  abs_path = os.path.join(ROOT, path)
  
  res = path.rsplit(os.path.extsep, maxsplit=1)
  if (len(res) < 2):
    raise ValueError(f'path requires file extension {path}')
  
  ext = res[1]
  
  if ext == 'vti':
    return 'ImageScalars', load_vti(abs_path)
  else:
    raise ValueError(f'Unrecognized file type {ext}')
  
def make_complexes(config):
  complexes = {}
  
  for name, dataset in config['datasets'].items():
    type = dataset['type']
    
    if type == 'gen':
      field_name, data = make_gen_data(dataset)
    elif type == 'load':
      field_name, data = make_load_data(dataset)
    else:
      raise ValueError(f'Unrecognized dataset type {type}')

    scale_factor = dataset['scale_factor']
    persistence_threshold = dataset['persistence_threshold']

    tetra = Tetrahedralize(data.GetOutputPort())
    warp = Warp(tetra.GetOutputPort(), scale_factor = scale_factor)

    complexes[name] = MorseComplex.create(
      warp.GetOutputPort(), 
      persistence_threshold=persistence_threshold,
      field_name=field_name
    )
    
  return complexes    

if __name__ == '__main__':  
  config_file = open(CONFIG, 'rb')
  config = json.load(config_file)
  config_file.close()
  
  complexes = make_complexes(config)
  
  for name, complex in complexes.items():
    save_complex(complex, os.path.join(GEN_DATA, name))