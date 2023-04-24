"""

"""

from typing import Iterable
import os 

import numpy as np
from scipy.ndimage import rotate

from mcpipeline import Pipeline, GenDatasetTarget
from mcpipeline import vtk
from mcpipeline import gen

__all__ = ['make_pipeline']

################################################################################
# Gaussian
################################################################################

def gaussian(pipeline: Pipeline):
  class BinaryGaussianSimple(GenDatasetTarget):
    def generate(self) -> Iterable[np.ndarray]:
      shape = (100, 100)
      rng = np.random.default_rng(42)
      
      initial = gen.Normal(shape=shape, center=(70, 50), sigma=(5)) * 100
      initial += gen.Normal(shape=shape, center=(30, 50), sigma=(5)) * 100
      
      num_frames = 10
      for i in range(num_frames):
        angle = 360 // num_frames * i
        
        frame = rotate(initial, angle, reshape=False)
        frame += gen.Distance(shape=shape) * 1.5
        frame += gen.Noise(shape=shape, scale=0.05, random_state=rng)
        
        yield frame
    
  class BinaryGaussianComplex(GenDatasetTarget):
    def generate(self) -> Iterable[np.ndarray]:
      shape = (100, 100)
      rng = np.random.default_rng(42)
      
      while True:
        frame = gen.Normal(shape=shape, center=rng.uniform(30, 70, size=2), sigma=5) * 100      
        frame += gen.Normal(shape=shape, center=rng.uniform(30, 70, size=2), sigma=5) * 100      
        
        frame += gen.Distance(shape=shape) * 1.5
        frame += gen.Noise(shape=shape, scale=0.05, random_state=rng)
        
        yield frame
    
  bingaus_simple_dataset = pipeline.add_gen_dataset(
    name = 'binary_gaussian_simple',
    display_name = 'Binary Gaussian Simple',
    desc='''
    A toy example in which 2 gaussian functions with $\sigma = 5$ are placed
    in the center.
    
    There are 10 frames which each rotate the gaussian's roughly 16 degrees.
    ''',
    cls = BinaryGaussianSimple,
    filters = [
      vtk.WarpFilter(scale_factor=50)
    ]
  )
  
  bingaus_complex_dataset = pipeline.add_gen_dataset(
    name = 'binary_gaussian_complex',
    display_name = 'Binary Gaussian Complex',
    desc='''
    A toy example in which 2 gaussian functions with $\sigma = 5$ are randomly
    placed.
    
    There are 10 frames which each placement is randomized.
    ''',
    cls = BinaryGaussianComplex,
    n_frames = 10,
    filters = [
      vtk.WarpFilter(scale_factor=50)
    ]
  )
  
  bingaus_simple_complex = pipeline.add_complex(
    name = 'binary_gaussian_simple',
    dataset = bingaus_simple_dataset,
    persistence_threshold=0.1
  )
  
  bingaus_complex_complex = pipeline.add_complex(
    name = 'binary_gaussian_complex',
    dataset = bingaus_complex_dataset,
    persistence_threshold=0.1
  )
  
  bingaus_simple_graph = pipeline.add_graph(
    name = 'binary_gaussian_simple',
    complex = bingaus_simple_complex,
    sample_rate = 5,
  )
  
  bingaus_complex_graph = pipeline.add_graph(
    name = 'binary_gaussian_complex',
    complex = bingaus_complex_complex,
    sample_rate = 5,
  )
  
  bingaus_simple_network = pipeline.add_mm_network(
    name = 'binary_gaussian_simple',
    graph = bingaus_simple_graph,
    dist = 'geo',
    hist = 'degree'
  )
  
  bingaus_complex_network = pipeline.add_mm_network(
    name = 'binary_gaussian_complex',
    graph = bingaus_complex_graph,
    dist = 'geo',
    hist = 'degree'
  )
  
  class TrinaryGaussianSimple(GenDatasetTarget):
    def generate(self) -> Iterable[np.ndarray]:
      shape = (100, 100)
      rng = np.random.default_rng(42)
      
      initial = gen.Normal(shape=shape, center=(70, 50), sigma=5) * 100
      initial += gen.Normal(shape=shape, center=(40, 67), sigma=5) * 100
      initial += gen.Normal(shape=shape, center=(40, 32), sigma=5) * 100
      
      
      num_frames = 10
      for i in range(num_frames):
        angle = 360 // num_frames * i
        
        frame = rotate(initial, angle, reshape=False)
        frame += gen.Distance(shape=shape) * 1.5
        frame += gen.Noise(shape=shape, scale=0.05, random_state=rng)
        
        yield frame
        
  class TrinaryGaussianComplex(GenDatasetTarget):
    def generate(self) -> Iterable[np.ndarray]:
      shape = (100, 100)
      rng = np.random.default_rng(42)
      
      while True:
        frame = gen.Normal(shape=shape, center=rng.uniform(30, 70, size=2), sigma=5) * 100      
        frame += gen.Normal(shape=shape, center=rng.uniform(30, 70, size=2), sigma=5) * 100      
        frame += gen.Normal(shape=shape, center=rng.uniform(30, 70, size=2), sigma=5) * 100      
        
        frame += gen.Distance(shape=shape) * 1.5
        frame += gen.Noise(shape=shape, scale=0.05, random_state=rng)
        
        yield frame
    
  trigaus_simple_dataset = pipeline.add_gen_dataset(
    name = 'trinary_gaussian_simple',
    display_name = 'Trinary Gaussian Simple',
    desc='''
    A toy example in which 3 gaussian functions with $\sigma = 5$ are placed
    in the center.
    
    There are 10 frames which each rotate the gaussian's roughly 16 degrees.
    ''',
    cls = TrinaryGaussianSimple,
    filters = [
      vtk.WarpFilter(scale_factor=50)
    ]
  )
  
  trigaus_complex_dataset = pipeline.add_gen_dataset(
    name = 'trinary_gaussian_complex',
    display_name = 'Trinary Gaussian Complex',
    desc='''
    A toy example in which 3 gaussian functions with $\sigma = 5$ are randomly
    placed.
    
    There are 10 frames which each placement is randomized.
    ''',
    cls = TrinaryGaussianComplex,
    n_frames = 10,
    filters = [
      vtk.WarpFilter(scale_factor=50)
    ]
  )
  
  trigaus_simple_complex = pipeline.add_complex(
    name = 'trinary_gaussian_simple',
    dataset = trigaus_simple_dataset,
    persistence_threshold=0.1
  )
  
  trigaus_complex_complex = pipeline.add_complex(
    name = 'trinary_gaussian_complex',
    dataset = trigaus_complex_dataset,
    persistence_threshold=0.1
  )
  
  trigaus_simple_graph = pipeline.add_graph(
    name = 'trinary_gaussian_simple',
    complex = trigaus_simple_complex,
    sample_rate = 5,
  )
  
  trigaus_complex_graph = pipeline.add_graph(
    name = 'trinary_gaussian_complex',
    complex = trigaus_complex_complex,
    sample_rate = 5,
  )
  
  trigaus_simple_network = pipeline.add_mm_network(
    name = 'trinary_gaussian_simple',
    graph = trigaus_simple_graph,
    dist = 'geo',
    hist = 'degree'
  )
  
  trigaus_complex_network = pipeline.add_mm_network(
    name = 'trinary_gaussian_complex',
    graph = trigaus_complex_graph,
    dist = 'geo',
    hist = 'degree'
  )

################################################################################
# Heated Cylinder
################################################################################

def heated_cylinder(pipeline: Pipeline):
  download = pipeline.add_download(
    name = 'heated_cylinder',
    display_name = 'Heated Cylinder',
    desc = '''
    TODO
    ''',
    url='https://github.com/stormymcstorm/MCOpt/releases/download/v0.5.0/heatedcylinder-800-899.zip'
  )
  
  extract = pipeline.add_extract_zip(
    name = 'heated_cylinder',
    zips = download,
    pattern = 'data_*.vtp'
  )
  
  dataset = pipeline.add_load_dataset(
    name = 'heated_cylinder',
    files = extract,
    time_steps = list(range(800, 899, 10))
  )
  
  complex = pipeline.add_complex(
    name = 'heated_cylinder',
    dataset = dataset,
    persistence_threshold = 0.15,
    scalar_field = 'velocityMagnitude'
  )
  
  graph = pipeline.add_graph(
    name = 'heated_cylinder',
    complex = complex,
    sample_rate = 30,
  )
  
  network = pipeline.add_mm_network(
    name = 'heated_cylinder',
    graph = graph,
    dist = 'geo',
    hist = 'degree',
    normalize = True
  )
  
  attributes = pipeline.add_attributes(
    name = 'heated_cylinder',
    graph = graph,
    normalize = True
  )
  
  m_start = 0.5
  m_end = 1
  num_ms = 40
  
  ms = [m_start + i * (m_end - m_start) / num_ms for i in range(num_ms)] + [m_end]
  
  max_match_pfgw = pipeline.add_max_match_pfgw(
    name = 'heated_cylinder_pfgw_max_match',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 800
  )
  
  max_match_wasserstein = pipeline.add_max_match_wasserstein(
    name = 'heated_cylinder_wasserstein_max_match',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 800
  )

################################################################################
# Navier Stokes
################################################################################

def navier_stokes(pipeline: Pipeline):
  download = pipeline.add_download(
    name = 'navier_stokes',
    display_name = 'Navier Stokes',
    desc = '''
    TODO
    ''',
    url='https://github.com/stormymcstorm/MCOpt/releases/download/v0.5.0/Navier-Stokes.zip'
  )
  
  extract = pipeline.add_extract_zip(
    name = 'navier_stokes',
    zips = download,
    pattern = 'speed*.vti'
  )
  
  dataset = pipeline.add_load_dataset(
    name = 'navier_stokes',
    files = extract,
  )
  
  complex = pipeline.add_complex(
    name = 'navier_stokes',
    dataset = dataset,
    persistence_threshold = 0.1,
  )
  
  graph = pipeline.add_graph(
    name = 'navier_stokes',
    complex = complex,
    sample_rate = 10,
  )
  
  network = pipeline.add_mm_network(
    name = 'navier_stokes',
    graph = graph,
    dist = 'geo',
    hist = 'degree',
    normalize = True
  )
  
  attributes = pipeline.add_attributes(
    name = 'navier_stokes',
    graph = graph,
    normalize = True
  )
  
  m_start = 0.5
  m_end = 1
  num_ms = 40
  
  ms = [m_start + i * (m_end - m_start) / num_ms for i in range(num_ms)] + [m_end]
  
  max_match_pfgw = pipeline.add_max_match_pfgw(
    name = 'navier_stokes_pfgw_max_match',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 1
  )
  
  max_match_wasserstein = pipeline.add_max_match_wasserstein(
    name = 'navier_stokes_wasserstein_max_match',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 1
  )

################################################################################
# Red Sea
################################################################################

def red_sea(pipeline: Pipeline):
  download = pipeline.add_download(
    name = 'red_sea',
    display_name = 'Red Sea',
    desc = '''
    TODO
    ''',
    url='https://github.com/stormymcstorm/MCOpt/releases/download/v0.5.0/redSea.zip'
  )
  
  extract = pipeline.add_extract_zip(
    name = 'red_sea',
    zips = download,
    pattern = 'redSeaVelocity*.vti'
  )
  
  dataset = pipeline.add_load_dataset(
    name = 'red_sea',
    files = extract,
  )
  
  complex = pipeline.add_complex(
    name = 'red_sea',
    dataset = dataset,
    persistence_threshold = 0.01,
  )
  
  graph = pipeline.add_graph(
    name = 'red_sea',
    complex = complex,
    sample_rate = 10,
  )
  
  network = pipeline.add_mm_network(
    name = 'red_sea',
    graph = graph,
    dist = 'geo',
    hist = 'degree',
    normalize = True
  )
  
  attributes = pipeline.add_attributes(
    name = 'red_sea',
    graph = graph,
    normalize = True
  )
  
  m_start = 0.5
  m_end = 1
  num_ms = 40
  
  ms = [m_start + i * (m_end - m_start) / num_ms for i in range(num_ms)] + [m_end]
  
  max_match_pfgw = pipeline.add_max_match_pfgw(
    name = 'red_sea_pfgw_max_match',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 1
  )
  
  max_match_wasserstein = pipeline.add_max_match_wasserstein(
    name = 'red_sea_wasserstein_max_match',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 1
  )

################################################################################
# Sinusoidal
################################################################################

def sinusoidal(pipeline: Pipeline):
  class Sinusoidal(GenDatasetTarget):
    def generate(self) -> Iterable[np.ndarray]:
      shape = (100, 100)
      
      initial = gen.Sinusoidal(shape=shape, npeaks=3) * 0.5
      initial += gen.Distance(shape=shape) * -0.5
      
      frame0 = initial + gen.Noise(shape=shape, scale=0.1, random_state=42)
      
      yield frame0
      
      frame1 = initial + gen.Noise(shape=shape, scale=0.25, random_state=42)
      
      yield frame1
  
  sinusoidal_dataset = pipeline.add_gen_dataset(
    name = 'sinusoidal',
    display_name = 'Sinusoidal',
    desc = '''
    A toy example in which the scalar field resembles a mountainous landscape.
    
    ## Frames
    * Frame 0: A small amount of noise
    * Frame 1: More noise
    ''',
    cls = Sinusoidal,
    filters = [
      vtk.WarpFilter(scale_factor=50)
    ]
  )

  sinusoidal_complex = pipeline.add_complex(
    name = 'sinusoidal',
    dataset = sinusoidal_dataset,
    persistence_threshold = 0.1
  )

################################################################################
# Tangaroa
################################################################################
  
def tangaroa(pipeline: Pipeline):
  download = pipeline.add_download(
    name = 'tangaroa',
    display_name = 'Tangaroa',
    desc = '''
    TODO
    ''',
    url='https://github.com/stormymcstorm/MCOpt/releases/download/v0.5.0/tangaroa-dataset-50-200.zip'
  )
  
  extract = pipeline.add_extract_zip(
    name = 'tangaroa',
    zips = download,
    pattern = 'data_*.vti'
  )
  
  dataset = pipeline.add_load_dataset(
    name = 'tangaroa',
    files = extract,
    time_steps = list(range(51, 200, 10)),
    filters = [
      vtk.BoxClipFilter(
        xmin = -0.25,
        xmax = 0.75,
        ymin = -0.3,
        ymax = 0.3,
        zmin = -0.5,
        zmax = -0.49
      ),
      vtk.TranslateFilter(
        dx = 0.25,
        dy = 0.3,
      )
    ],
  )
  
  complex = pipeline.add_complex(
    name = 'tangaroa',
    dataset = dataset,
    persistence_threshold = 0.1,
    scalar_field = 'velocityMagnitude'
  )
  
  graph = pipeline.add_graph(
    name = 'tangaroa',
    complex = complex,
    sample_rate = 10,
  )
  
  network = pipeline.add_mm_network(
    name = 'tangaroa',
    graph = graph,
    dist = 'geo',
    hist = 'degree',
    normalize = True
  )
  
  attributes = pipeline.add_attributes(
    name = 'tangaroa',
    graph = graph,
    normalize = True
  )
  
  m_start = 0.5
  m_end = 1
  num_ms = 40
  
  ms = [m_start + i * (m_end - m_start) / num_ms for i in range(num_ms)] + [m_end]
  
  max_match_pfgw = pipeline.add_max_match_pfgw(
    name = 'tangaroa_pfgw_max_match',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 51
  )
  
  max_match_wasserstein = pipeline.add_max_match_wasserstein(
    name = 'tangaroa_wasserstein_max_match',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 51
  )

################################################################################
# Tropopause
################################################################################

def tropopause(pipeline: Pipeline):
  download = pipeline.add_download(
    name = 'tropopause',
    display_name = 'Tangaroa',
    desc = '''
    TODO
    ''',
    url='https://github.com/stormymcstorm/MCOpt/releases/download/v0.5.0/tropoause-VISContest.zip'
  )
  
  extract = pipeline.add_extract_zip(
    name = 'tropopause',
    zips = download,
    pattern = 'Tropoause-VISContest_*.vtk'
  )
  
  dataset = pipeline.add_load_dataset(
    name = 'tropopause',
    files = extract,
    time_steps = list(range(0, 31, 3)),
    filters = [
      vtk.TranslateFilter(
        dx = 180,
        dy = 87,
      )
    ],
  )
  
  complex = pipeline.add_complex(
    name = 'tropopause',
    dataset = dataset,
    persistence_threshold = 2.0,
    scalar_field = 'trop_1'
  )
  
  graph = pipeline.add_graph(
    name = 'tropopause',
    complex = complex,
    sample_rate = 20,
  )
  
  network = pipeline.add_mm_network(
    name = 'tropopause',
    graph = graph,
    dist = 'geo',
    hist = 'degree',
    normalize = True
  )
  
  attributes = pipeline.add_attributes(
    name = 'tropopause',
    graph = graph,
    normalize = True
  )
  
  m_start = 0.5
  m_end = 1
  num_ms = 40
  
  ms = [m_start + i * (m_end - m_start) / num_ms for i in range(num_ms)] + [m_end]
  
  max_match_pfgw = pipeline.add_max_match_pfgw(
    name = 'tropopause_pfgw_max_match',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 0
  )
  
  max_match_wasserstein = pipeline.add_max_match_wasserstein(
    name = 'tropopause_wasserstein_max_match',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 0
  )

################################################################################
# Wind Dataset
################################################################################

def wind(pipeline: Pipeline):
  download = pipeline.add_download(
    name = 'wind',
    display_name = 'Wind',
    desc = '''
    A dataset of 15 vector fields from a wind dataset of the IRI/LDEO Climate Data
    Library.
    
    Originally obtained from [IRI](http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/)
    and preprocessed before packaging and inclusion in [MCOpt](https://github.com/stormymcstorm/MCOpt).
    
    Please see [Uncertainty Visualization of 2D Morse Complex Ensembles Using Statistical Summary Maps](https://www.sci.utah.edu/~beiwang/publications/Uncertain_MSC_BeiWang_2020.pdf) 
    section 7.1 for a description of the preprocessing.
    ''',
    url = 'https://github.com/stormymcstorm/MCOpt/releases/download/v0.5.0/wind.zip',
  )
  
  extract = pipeline.add_extract_zip(
    name = 'wind',
    zips = download,
    pattern = 'Tropoause-VISContest_*.vtk'
  )
  
  dataset = pipeline.add_load_dataset(
    name = 'wind',
    files = extract,
  )
  
  complex = pipeline.add_complex(
    name = 'wind',
    dataset = dataset,
    persistence_threshold = 2.0,
  )
  
  graph = pipeline.add_graph(
    name = 'wind',
    complex = complex,
    sample_rate = 10,
  )
  
  network = pipeline.add_mm_network(
    name = 'wind',
    graph = graph,
    dist = 'geo',
    hist = 'degree',
    normalize = True
  )
  
  attributes = pipeline.add_attributes(
    name = 'wind',
    graph = graph,
    normalize = True
  )
  
  m_start = 0.5
  m_end = 1
  num_ms = 40
  
  ms = [m_start + i * (m_end - m_start) / num_ms for i in range(num_ms)] + [m_end]
  
  max_match_pfgw = pipeline.add_max_match_pfgw(
    name = 'wind_pfgw_max_match',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 1
  )
  
  max_match_wasserstein = pipeline.add_max_match_wasserstein(
    name = 'wind_wasserstein_max_match',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 1
  )

################################################################################
# Make Pipeline
################################################################################

def make_pipeline():
  pipeline = Pipeline(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), '__pipeline_cache__')
  )
  
  gaussian(pipeline)
  heated_cylinder(pipeline)
  navier_stokes(pipeline)
  red_sea(pipeline)
  sinusoidal(pipeline)
  tangaroa(pipeline)
  tropopause(pipeline)
  wind(pipeline)
  
  return pipeline

if __name__ == '__main__':
  pipeline = make_pipeline()
  
  pipeline.build_all()