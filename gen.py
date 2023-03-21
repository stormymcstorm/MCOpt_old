#!/usr/bin/env python

"""
Script for generating dataset complexes
"""

import os
import json

import numpy as np

import mcopt.util.gen as gen_util
import mcopt.util.vtk as vtk_util
from mcopt import MorseComplex

ROOT = os.path.dirname(__file__)
CONFIG = os.path.join(ROOT, "gen.config.json")
COMPLEXES_OUT = os.path.join(ROOT, "gen_complexes")
FIGURES_OUT = os.path.join(ROOT, "gen_figures")

def Noise(scale=1, **kwargs):
  return gen_util.Smooth(gen_util.GaussianNoise(**kwargs) * scale)

GEN_COMPLEX_FUNCTIONS = {
  'sinusoidal': gen_util.Sinusoidal,
  'distance': gen_util.Distance,
  'noise': Noise,
}

def generate_complexes(complex_config):
  complexes={}
  
  for name, conf in complex_config.items():
    ty = conf['type']
    
    print(f'Generating {name} complex')
    
    if ty == 'gen':
      shape = conf['shape']
      layers = conf['layers']
      
      data = np.zeros(shape)
      
      for layer in layers:
        args = layer['args'] if 'args' in layer else {}
        weight = layer['weight'] if 'weight' in layer else 1
        
        layer_ty = layer['layer']
        
        if layer_ty not in GEN_COMPLEX_FUNCTIONS:
          raise ValueError(f'Unrecognized layer {layer_ty}')
        
        data += GEN_COMPLEX_FUNCTIONS[layer_ty](shape=shape, **args) * weight
      
      plane = vtk_util.PlaneSource(data)
      
      
      if 'warp' in conf:
        warp = vtk_util.Warp(plane.GetOutputPort(), conf['warp'])
        source = warp.GetOutputPort()
      else:
        source = plane.GetOutputPort()
    elif ty == 'load':
      path = conf['path']
      abs_path = os.path.join(ROOT, path)
      
      res = path.rsplit(os.path.extsep, maxsplit=1)
      if (len(res) < 2):
        raise ValueError(f'path requires file extension: {path}')
      
      ext = res[1]
      if ext == 'vti':
        image_data = vtk_util.ReadVTI(abs_path)
        source = image_data.GetOutputPort()
      else:
        raise ValueError(f'Unrecognized file type {ext}')
    else:
      raise ValueError(f'Unrecognized complex type {ty}')
    
    persistence_threshold = conf['persistence_threshold']
    
    complex = MorseComplex.create(source, persistence_threshold)
    complex.write(os.path.join(COMPLEXES_OUT, name))
    complexes[name] = complex
  
  return complexes

if __name__ == '__main__':
  config_file = open(CONFIG)
  config = json.load(config_file)
  config_file.close()
  
  complexes = generate_complexes(config['complexes'])