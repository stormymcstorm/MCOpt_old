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

savefig_kwargs = dict(
  dpi = 300,
  bbox_inches = 'tight',
)

savefig_kwargs_combined = dict(
  dpi = 96,
  bbox_inches = 'tight',
)

n_neighbors = 5

################################################################################
# Gaussian
################################################################################

def gaussian(pipeline: Pipeline, fig_path: str):
  n_frames = 100
  
  class BinaryGaussianSimple(GenDatasetTarget):
    def generate(self) -> Iterable[np.ndarray]:
      shape = (100, 100)
      rng = np.random.default_rng(42)
      
      initial = gen.Normal(shape=shape, center=(70, 50), sigma=(5)) * 100
      initial += gen.Normal(shape=shape, center=(30, 50), sigma=(5)) * 100
      
      for i in range(n_frames):
        angle = 360 / n_frames * i
        
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
    
    There are 100 frames which each rotate the gaussian's roughly 3 degrees.
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
    
    There are 100 frames which each placement is randomized.
    ''',
    cls = BinaryGaussianComplex,
    n_frames = n_frames,
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
      
      
      for i in range(n_frames):
        angle = 360 / n_frames * i
        
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
    
    There are 100 frames which each rotate the gaussian's roughly 3 degrees.
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
    
    There are 100 frames which each placement is randomized.
    ''',
    cls = TrinaryGaussianComplex,
    n_frames = n_frames,
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

  # SIMPLE

  simple_graphs = pipeline.add_combine_graphs(
    name = 'gaussian_simple',
    graphs = [bingaus_simple_graph, trigaus_simple_graph]
  )
  
  simple_networks = pipeline.add_combine_mm_network(
    name = 'gaussian_simple',
    networks = [bingaus_simple_network, trigaus_simple_network]
  )

  simple_attributes = pipeline.add_attributes(
    name = 'gaussian_simple',
    graph = simple_graphs,
    normalize = True,
  )
  
  simple_gw = pipeline.add_gw(
    name = 'gaussian_simple_gw',
    network = simple_networks,
    num_random_iter = 3,
    random_state = 42,
  )
  
  simple_gw_mds = pipeline.add_mds(
    name = 'gaussian_simple_gw',
    graphs = simple_graphs,
    couplings = simple_gw,
    random_state = 42
  )
  
  simple_gw_classes = pipeline.add_knearest_neighbors(
    name = 'gaussian_simple_gw',
    graphs = simple_graphs,
    couplings = simple_gw,
    random_state = 42,
    n_neighbors = n_neighbors,
  )
  
  pipeline.add_classification_figure(
    name = 'gaussian_simple_gw_classification',
    mds = simple_gw_mds,
    classification = simple_gw_classes,
    labels=[
      bingaus_simple_graph.display_name,  
      trigaus_simple_graph.display_name,  
    ],
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_classification_figure(
    name = 'gaussian_simple_gw_classification_hulls',
    mds = simple_gw_mds,
    classification = simple_gw_classes,
    labels=[
      bingaus_simple_graph.display_name,  
      trigaus_simple_graph.display_name,  
    ],
    class_hulls = True,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  simple_wasserstein = pipeline.add_wasserstein(
    name = 'gaussian_simple_wasserstein',
    network = simple_networks,
    attributes=simple_attributes,
  )
  
  simple_wasserstein_mds = pipeline.add_mds(
    name = 'gaussian_simple_wasserstein',
    graphs = simple_graphs,
    couplings = simple_wasserstein,
    random_state = 42
  )
  
  simple_wasserstein_classes = pipeline.add_knearest_neighbors(
    name = 'gaussian_simple_wasserstein',
    graphs = simple_graphs,
    couplings = simple_wasserstein,
    random_state = 42,
    n_neighbors = n_neighbors,
  )
  
  pipeline.add_classification_figure(
    name = 'gaussian_simple_wasserstein_classification',
    mds = simple_wasserstein_mds,
    classification = simple_wasserstein_classes,
    labels=[
      bingaus_simple_graph.display_name,  
      trigaus_simple_graph.display_name,  
    ],
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_classification_figure(
    name = 'gaussian_simple_wasserstein_classification_hulls',
    mds = simple_wasserstein_mds,
    classification = simple_wasserstein_classes,
    labels=[
      bingaus_simple_graph.display_name,  
      trigaus_simple_graph.display_name,  
    ],
    class_hulls = True,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  simple_fgw = pipeline.add_fgw(
    name = 'gaussian_simple_fgw',
    network = simple_networks,
    attributes=simple_attributes,
  )
  
  simple_fgw_mds = pipeline.add_mds(
    name = 'gaussian_simple_fgw',
    graphs = simple_graphs,
    couplings = simple_fgw,
    random_state = 42
  )
  
  simple_fgw_classes = pipeline.add_knearest_neighbors(
    name = 'gaussian_simple_fgw',
    graphs = simple_graphs,
    couplings = simple_fgw,
    random_state = 42,
    n_neighbors = n_neighbors,
  )

  pipeline.add_classification_figure(
    name = 'gaussian_simple_fgw_classification',
    mds = simple_fgw_mds,
    classification = simple_fgw_classes,
    labels=[
      bingaus_simple_graph.display_name,  
      trigaus_simple_graph.display_name,  
    ],
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_classification_figure(
    name = 'gaussian_simple_fgw_classification_hulls',
    mds = simple_fgw_mds,
    classification = simple_fgw_classes,
    labels=[
      bingaus_simple_graph.display_name,  
      trigaus_simple_graph.display_name,  
    ],
    class_hulls = True,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )

  # COMPLEX

  complex_graphs = pipeline.add_combine_graphs(
    name = 'gaussian_complex',
    graphs = [bingaus_complex_graph, trigaus_complex_graph]
  )
  
  complex_networks = pipeline.add_combine_mm_network(
    name = 'gaussian_complex',
    networks = [bingaus_complex_network, trigaus_complex_network]
  )

  complex_attributes = pipeline.add_attributes(
    name = 'gaussian_complex',
    graph = complex_graphs,
    normalize = True,
  )
  
  complex_gw = pipeline.add_gw(
    name = 'gaussian_complex_gw',
    network = complex_networks,
    num_random_iter = 3,
    random_state = 42,
  )
  
  complex_gw_mds = pipeline.add_mds(
    name = 'gaussian_complex_gw',
    graphs = complex_graphs,
    couplings = complex_gw,
    random_state = 42
  )
  
  complex_gw_classes = pipeline.add_knearest_neighbors(
    name = 'gaussian_complex_gw',
    graphs = complex_graphs,
    couplings = complex_gw,
    random_state = 42,
    n_neighbors = n_neighbors,
  )
  
  pipeline.add_classification_figure(
    name = 'gaussian_complex_gw_classification',
    mds = complex_gw_mds,
    classification = complex_gw_classes,
    labels=[
      bingaus_complex_graph.display_name,  
      trigaus_complex_graph.display_name,  
    ],
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_classification_figure(
    name = 'gaussian_complex_gw_classification_hulls',
    mds = complex_gw_mds,
    classification = complex_gw_classes,
    labels=[
      bingaus_complex_graph.display_name,  
      trigaus_complex_graph.display_name,  
    ],
    class_hulls = True,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  complex_wasserstein = pipeline.add_wasserstein(
    name = 'gaussian_complex_wasserstein',
    network = complex_networks,
    attributes=complex_attributes,
  )
  
  complex_wasserstein_mds = pipeline.add_mds(
    name = 'gaussian_complex_wasserstein',
    graphs = complex_graphs,
    couplings = complex_wasserstein,
    random_state = 42
  )
  
  complex_wasserstein_classes = pipeline.add_knearest_neighbors(
    name = 'gaussian_complex_wasserstein',
    graphs = complex_graphs,
    couplings = complex_wasserstein,
    random_state = 42,
    n_neighbors = n_neighbors,
  )
  
  pipeline.add_classification_figure(
    name = 'gaussian_complex_wasserstein_classification',
    mds = complex_wasserstein_mds,
    classification = complex_wasserstein_classes,
    labels=[
      bingaus_complex_graph.display_name,  
      trigaus_complex_graph.display_name,  
    ],
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_classification_figure(
    name = 'gaussian_complex_wasserstein_classification_hulls',
    mds = complex_wasserstein_mds,
    classification = complex_wasserstein_classes,
    labels=[
      bingaus_complex_graph.display_name,  
      trigaus_complex_graph.display_name,  
    ],
    class_hulls = True,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  complex_fgw = pipeline.add_fgw(
    name = 'gaussian_complex_fgw',
    network = complex_networks,
    attributes=complex_attributes,
  )
  
  complex_fgw_mds = pipeline.add_mds(
    name = 'gaussian_complex_fgw',
    graphs = complex_graphs,
    couplings = complex_fgw,
    random_state = 42
  )

  complex_fgw_classes = pipeline.add_knearest_neighbors(
    name = 'gaussian_complex_fgw',
    graphs = complex_graphs,
    couplings = complex_fgw,
    random_state = 42,
    n_neighbors = n_neighbors,
  )
  
  pipeline.add_classification_figure(
    name = 'gaussian_complex_fgw_classification',
    mds = complex_fgw_mds,
    classification = complex_fgw_classes,
    labels=[
      bingaus_complex_graph.display_name,  
      trigaus_complex_graph.display_name,  
    ],
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_classification_figure(
    name = 'gaussian_complex_fgw_classification_hulls',
    mds = complex_fgw_mds,
    classification = complex_fgw_classes,
    labels=[
      bingaus_complex_graph.display_name,  
      trigaus_complex_graph.display_name,  
    ],
    class_hulls = True,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
################################################################################
# Heated Cylinder
################################################################################

def heated_cylinder(pipeline: Pipeline, fig_path: str):
  m_pfgw = 0.95
  m_pw = 0.9
  m_pgw = 0.9
  
  nrows = 3
  ncols = 5
  figsize = [ncols * 12, nrows * 6]
  rotation = -90
  
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
    time_steps = [800] + list(range(809, 900, 10))
  )
  
  complex = pipeline.add_complex(
    name = 'heated_cylinder',
    dataset = dataset,
    persistence_threshold = 0.093,
    # persistence_threshold = 0.0927,
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
  
  pipeline.add_graphs_figure(
    name = 'heated_cylinder',
    graph = graph,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  # MAX MATCH
  m_start = 0.5
  m_end = 1
  num_ms = 40
  
  ms = [m_start + i * (m_end - m_start) / num_ms for i in range(num_ms)] + [m_end]
  
  max_match_pfgw = pipeline.add_max_match_pfgw(
    name = 'heated_cylinder_max_match_pfgw',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 800,
    num_random_iter = 1,
    random_state = 42,
  )
  
  max_match_pw = pipeline.add_max_match_pw(
    name = 'heated_cylinder_max_match_pw',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 800
  )
  
  m_start = 0.75
  m_end = 1
  num_ms = 20
  
  ms = [m_start + i * (m_end - m_start) / num_ms for i in range(num_ms)] + [m_end]
  
  max_match_pgw = pipeline.add_max_match_pgw(
    name = 'heated_cylinder_max_match_pgw',
    network = network,
    graph = graph,
    ms = ms,
    src_t = 800
  )
  
  pipeline.add_max_match_figure(
    name = 'heated_cylinder_max_match_pfgw',
    max_match = max_match_pfgw,
    m = m_pfgw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_max_match_combined_figure(
    name = 'heated_cylinder_max_match_all_pfgw',
    max_match = max_match_pfgw,
    m = m_pfgw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_max_match_figure(
    name = 'heated_cylinder_max_match_pw',
    max_match = max_match_pw,
    m = m_pw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs_combined
  )

  pipeline.add_max_match_figure(
    name = 'heated_cylinder_max_match_pgw',
    max_match = max_match_pgw,
    m = m_pgw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs_combined
  )

  # GW
  
  gw = pipeline.add_gw(
    name = 'heated_cylinder_gw',
    network = network,
    num_random_iter = 5,
    random_state = 42
  )
  
  pipeline.add_couplings_figure(
    name = 'heated_cylinder_gw',
    graph = graph,
    couplings = gw,
    src_t = 800,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'heated_cylinder_gw_combined',
    graph = graph,
    couplings = gw,
    src_t = 800,
    rotation = rotation,
    figsize = figsize,
    nrows = nrows,
    ncols = ncols,
    output_path = fig_path,
    output_fmt = 'heated_cylinder_gw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # fGW
  
  fgw = pipeline.add_fgw(
    name = 'heated_cylinder_fgw',
    network = network,
    attributes = attributes
  )
  
  pipeline.add_couplings_figure(
    name = 'heated_cylinder_fgw',
    graph = graph,
    couplings = fgw,
    src_t = 800,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'heated_cylinder_fgw_combined',
    graph = graph,
    couplings = fgw,
    src_t = 800,
    rotation = rotation,
    figsize = figsize,
    nrows = nrows,
    ncols = ncols,
    output_path = fig_path,
    output_fmt = 'heated_cylinder_fgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # Wasserstein
  
  wasserstein = pipeline.add_wasserstein(
    name = 'heated_cylinder_w',
    network = network,
    attributes = attributes
  )
  
  pipeline.add_couplings_figure(
    name = 'heated_cylinder_w',
    graph = graph,
    couplings = wasserstein,
    src_t = 800,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'heated_cylinder_w_combined',
    graph = graph,
    couplings = wasserstein,
    src_t = 800,
    rotation = rotation,
    figsize = figsize,
    nrows = nrows,
    ncols = ncols,
    output_path = fig_path,
    output_fmt = 'heated_cylinder_w.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pfGW
  
  pfgw = pipeline.add_pfgw(
    name = 'heated_cylinder_pfgw',
    network = network,
    attributes = attributes,
    m = m_pfgw
  )
  
  pipeline.add_couplings_figure(
    name = 'heated_cylinder_pfgw',
    graph = graph,
    couplings = pfgw,
    src_t = 800,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'heated_cylinder_pfgw_combined',
    graph = graph,
    couplings = pfgw,
    src_t = 800,
    rotation = rotation,
    figsize = figsize,
    nrows = nrows,
    ncols = ncols,
    output_path = fig_path,
    output_fmt = 'heated_cylinder_pfgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pW
  
  pw = pipeline.add_pfgw(
    name = 'heated_cylinder_pw',
    network = network,
    attributes = attributes,
    m = m_pw
  )
  
  pipeline.add_couplings_figure(
    name = 'heated_cylinder_pw',
    graph = graph,
    couplings = pw,
    src_t = 800,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'heated_cylinder_pw_combined',
    graph = graph,
    couplings = pw,
    src_t = 800,
    rotation = rotation,
    figsize = figsize,
    nrows = nrows,
    ncols = ncols,
    output_path = fig_path,
    output_fmt = 'heated_cylinder_pw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pGW
  
  pgw = pipeline.add_pgw(
    name = 'heated_cylinder_pgw',
    network = network,
    m = m_pgw
  )
  
  pipeline.add_couplings_figure(
    name = 'heated_cylinder_pgw',
    graph = graph,
    couplings = pgw,
    src_t = 800,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'heated_cylinder_pgw_combined',
    graph = graph,
    couplings = pgw,
    src_t = 800,
    rotation = rotation,
    figsize = figsize,
    nrows = nrows,
    ncols = ncols,
    output_path = fig_path,
    output_fmt = 'heated_cylinder_pgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

################################################################################
# Navier Stokes
################################################################################

def navier_stokes(pipeline: Pipeline, fig_path: str):
  m_pfgw = 0.8625
  m_pw = 0.925
  m_pgw = 0.925
  
  nrows = 3
  ncols = 3
  figsize = None
  rotation = 0
  
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
    # persistence_threshold = 0.143,
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
  
  pipeline.add_graphs_figure(
    name = 'navier_stokes',
    graph = graph,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  # MAX MATCH
  m_start = 0.5
  m_end = 1
  num_ms = 40
  
  ms = [m_start + i * (m_end - m_start) / num_ms for i in range(num_ms)] + [m_end]
  
  max_match_pfgw = pipeline.add_max_match_pfgw(
    name = 'navier_stokes_max_match_pfgw',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 1
  )
  
  max_match_pw = pipeline.add_max_match_pw(
    name = 'navier_stokes_max_match_pw',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 1
  )
  
  max_match_pgw = pipeline.add_max_match_pgw(
    name = 'navier_stokes_max_match_pgw',
    network = network,
    graph = graph,
    ms = ms,
    src_t = 1
  )
  
  pipeline.add_max_match_figure(
    name = 'navier_stokes_max_match_pfgw',
    max_match = max_match_pfgw,
    m = m_pfgw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs,
  )
  
  pipeline.add_max_match_figure(
    name = 'navier_stokes_max_match_pw',
    max_match = max_match_pw,
    m = m_pw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs_combined,
  )
  
  pipeline.add_max_match_figure(
    name = 'navier_stokes_max_match_pgw',
    max_match = max_match_pgw,
    m = m_pw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs_combined,
  )

  # GW
  
  gw = pipeline.add_gw(
    name = 'navier_stokes_gw',
    network = network,
    num_random_iter = 5,
    random_state = 42
  )
  
  pipeline.add_couplings_figure(
    name = 'navier_stokes_gw',
    graph = graph,
    couplings = gw,
    src_t = 1,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'navier_stokes_gw_combined',
    graph = graph,
    couplings = gw,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'navier_stokes_gw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # fGW
  
  fgw = pipeline.add_fgw(
    name = 'navier_stokes_fgw',
    network = network,
    attributes = attributes
  )
  
  pipeline.add_couplings_figure(
    name = 'navier_stokes_fgw',
    graph = graph,
    couplings = fgw,
    src_t = 1,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'navier_stokes_fgw_combined',
    graph = graph,
    couplings = fgw,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'navier_stokes_fgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # Wasserstein
  
  wasserstein = pipeline.add_wasserstein(
    name = 'navier_stokes_w',
    network = network,
    attributes = attributes
  )
  
  pipeline.add_couplings_figure(
    name = 'navier_stokes_w',
    graph = graph,
    couplings = wasserstein,
    src_t = 1,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'navier_stokes_w_combined',
    graph = graph,
    couplings = wasserstein,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'navier_stokes_w.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pfGW
  
  pfgw = pipeline.add_pfgw(
    name = 'navier_stokes_pfgw',
    network = network,
    attributes = attributes,
    m = m_pfgw
  )
  
  pipeline.add_couplings_figure(
    name = 'navier_stokes_pfgw',
    graph = graph,
    couplings = pfgw,
    src_t = 1,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'navier_stokes_pfgw_combined',
    graph = graph,
    couplings = pfgw,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'navier_stokes_pfgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pW
  
  pw = pipeline.add_pfgw(
    name = 'navier_stokes_pw',
    network = network,
    attributes = attributes,
    m = m_pw
  )
  
  pipeline.add_couplings_figure(
    name = 'navier_stokes_pw',
    graph = graph,
    couplings = pw,
    src_t = 1,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'navier_stokes_pw_combined',
    graph = graph,
    couplings = pw,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'navier_stokes_pw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pGW
  
  pgw = pipeline.add_pgw(
    name = 'navier_stokes_pgw',
    network = network,
    m = m_pgw
  )
  
  pipeline.add_couplings_figure(
    name = 'navier_stokes_pgw',
    graph = graph,
    couplings = pgw,
    src_t = 1,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'navier_stokes_pgw_combined',
    graph = graph,
    couplings = pgw,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'navier_stokes_pgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

################################################################################
# Red Sea
################################################################################

def red_sea(pipeline: Pipeline, fig_path: str):
  m_pfgw = 0.7875
  m_pw = 0.8125
  m_pgw = 0.8125
  
  nrows = 3
  ncols = 5
  figsize = None
  rotation = 0
  
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
  
  pipeline.add_graphs_figure(
    name = 'red_sea',
    graph = graph,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  # MAX MATCH
  m_start = 0.5
  m_end = 1
  num_ms = 40
  
  ms = [m_start + i * (m_end - m_start) / num_ms for i in range(num_ms)] + [m_end]
  
  max_match_pfgw = pipeline.add_max_match_pfgw(
    name = 'red_sea_max_match_pfgw',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 1
  )
  
  max_match_pw = pipeline.add_max_match_pw(
    name = 'red_sea_max_match_pw',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 1
  )
  
  max_match_pgw = pipeline.add_max_match_pgw(
    name = 'red_sea_max_match_pgw',
    network = network,
    graph = graph,
    ms = ms,
    src_t = 1
  )
  
  pipeline.add_max_match_figure(
    name = 'red_sea_max_match_pfgw',
    max_match = max_match_pfgw,
    m = m_pfgw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs,
  )
  
  pipeline.add_max_match_figure(
    name = 'red_sea_max_match_pw',
    max_match = max_match_pw,
    m = m_pw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs,
  )

  pipeline.add_max_match_figure(
    name = 'red_sea_max_match_pgw',
    max_match = max_match_pgw,
    m = m_pw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs,
  )

  # GW
  
  gw = pipeline.add_gw(
    name = 'red_sea_gw',
    network = network,
    num_random_iter = 5,
    random_state = 42
  )
  
  pipeline.add_couplings_figure(
    name = 'red_sea_gw',
    graph = graph,
    couplings = gw,
    src_t = 1,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'red_sea_gw_combined',
    graph = graph,
    couplings = gw,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'red_sea_gw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # fGW
  
  fgw = pipeline.add_fgw(
    name = 'red_sea_fgw',
    network = network,
    attributes = attributes
  )
  
  pipeline.add_couplings_figure(
    name = 'red_sea_fgw',
    graph = graph,
    couplings = fgw,
    src_t = 1,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'red_sea_fgw_combined',
    graph = graph,
    couplings = fgw,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'red_sea_fgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # Wasserstein
  
  wasserstein = pipeline.add_wasserstein(
    name = 'red_sea_w',
    network = network,
    attributes = attributes
  )
  
  pipeline.add_couplings_figure(
    name = 'red_sea_w',
    graph = graph,
    couplings = wasserstein,
    src_t = 1,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'red_sea_w_combined',
    graph = graph,
    couplings = wasserstein,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'red_sea_w.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pfGW
  
  pfgw = pipeline.add_pfgw(
    name = 'red_sea_pfgw',
    network = network,
    attributes = attributes,
    m = m_pfgw
  )
  
  pipeline.add_couplings_figure(
    name = 'red_sea_pfgw',
    graph = graph,
    couplings = pfgw,
    src_t = 1,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'red_sea_pfgw_combined',
    graph = graph,
    couplings = pfgw,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'red_sea_pfgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pW
  
  pw = pipeline.add_pfgw(
    name = 'red_sea_pw',
    network = network,
    attributes = attributes,
    m = m_pw
  )
  
  pipeline.add_couplings_figure(
    name = 'red_sea_pw',
    graph = graph,
    couplings = pw,
    src_t = 1,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'red_sea_pw_combined',
    graph = graph,
    couplings = pw,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'red_sea_pw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pGW
  
  pgw = pipeline.add_pgw(
    name = 'red_sea_pgw',
    network = network,
    m = m_pgw
  )
  
  pipeline.add_couplings_figure(
    name = 'red_sea_pgw',
    graph = graph,
    couplings = pgw,
    src_t = 1,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'red_sea_pgw_combined',
    graph = graph,
    couplings = pgw,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'red_sea_pgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

################################################################################
# Sinusoidal
################################################################################

def sinusoidal(pipeline: Pipeline, fig_path: str):
  class Sinusoidal(GenDatasetTarget):
    def generate(self) -> Iterable[np.ndarray]:
      shape = (100, 100)
      
      initial = gen.Sinusoidal(shape=shape, npeaks=3) * 0.5
      initial += gen.Distance(shape=shape) * -0.5
      
      frame0 = initial + gen.Noise(shape=shape, scale=0.1, random_state=42)
      
      yield frame0
      
      frame1 = initial + gen.Noise(shape=shape, scale=0.25, random_state=42)
      
      yield frame1
  
  dataset = pipeline.add_gen_dataset(
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

  complex = pipeline.add_complex(
    name = 'sinusoidal',
    dataset = dataset,
    persistence_threshold = 0.1
  )
  
  graph = pipeline.add_graph(
    name = 'sinusoidal',
    complex = complex,
    sample_rate = 10,
  )
  
  network = pipeline.add_mm_network(
    name = 'sinusoidal',
    graph = graph,
    dist = 'geo',
    hist = 'degree',
    normalize = True
  )
  
  attributes = pipeline.add_attributes(
    name = 'sinusoidal',
    graph = graph
  )
  
  pipeline.add_graphs_figure(
    name = 'sinusoidal',
    graph = graph,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  # MAX MATCH
  m_pfgw = 0.8625
  m_pgw = 0.8625
  m_pw = 0.8625
  
  m_start = 0.5
  m_end = 1
  num_ms = 40
  
  ms = [m_start + i * (m_end - m_start) / num_ms for i in range(num_ms)] + [m_end]

  max_match_pfgw = pipeline.add_max_match_pfgw(
    name = 'sinusoidal_pfgw',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 0
  )
  
  max_match_pw = pipeline.add_max_match_pw(
    name = 'sinusoidal_pw',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 0
  )
  
  max_match_pgw = pipeline.add_max_match_pgw(
    name = 'sinusoidal_pgw',
    network = network,
    graph = graph,
    ms = ms,
    src_t = 0
  )
  
  pipeline.add_max_match_figure(
    name = 'sinusoidal_max_match_pfgw',
    max_match = max_match_pfgw,
    m = m_pfgw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_max_match_figure(
    name = 'sinusoidal_max_match_pw',
    max_match = max_match_pw,
    m = m_pw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_max_match_figure(
    name = 'sinusoidal_max_match_pgw',
    max_match = max_match_pgw,
    m = m_pgw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  # GW
  
  gw = pipeline.add_gw(
    name = 'sinusoidal_gw',
    network = network,
    num_random_iter = 10,
    random_state = 42
  )
  
  pipeline.add_couplings_figure(
    name = 'sinusoidal_gw',
    graph = graph,
    couplings = gw,
    src_t = 0,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  # fGW
  
  fgw = pipeline.add_fgw(
    name = 'sinusoidal_fgw',
    network = network,
    attributes = attributes
  )
  
  pipeline.add_couplings_figure(
    name = 'sinusoidal_fgw',
    graph = graph,
    couplings = fgw,
    src_t = 0,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  # Wasserstein
  
  wasserstein = pipeline.add_wasserstein(
    name = 'sinusoidal_wasserstein',
    network = network,
    attributes = attributes
  )
  
  pipeline.add_couplings_figure(
    name = 'sinusoidal_wasserstein',
    graph = graph,
    couplings = wasserstein,
    src_t = 0,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  # pfGW
  
  pfgw = pipeline.add_pfgw(
    name = 'sinusoidal_pfgw',
    network = network,
    attributes = attributes,
    m = m_pfgw,
  )
  
  pipeline.add_couplings_figure(
    name = 'sinusoidal_pfgw',
    graph = graph,
    couplings = pfgw,
    src_t = 0,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  # pGW
  
  pgw = pipeline.add_pgw(
    name = 'sinusoidal_pgw',
    network = network,
    m = m_pgw,
    num_random_iter = 10,
    random_state = 42,
  )
  
  pipeline.add_couplings_figure(
    name = 'sinusoidal_pgw',
    graph = graph,
    couplings = pgw,
    src_t = 0,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  # pW

  pw = pipeline.add_pw(
    name = 'sinusoidal_pw',
    network = network,
    attributes = attributes,
    m = m_pw,
  )
  
  pipeline.add_couplings_figure(
    name = 'sinusoidal_pw',
    graph = graph,
    couplings = pw,
    src_t = 0,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
################################################################################
# Tangaroa
################################################################################
  
def tangaroa(pipeline: Pipeline, fig_path: str):
  m_pfgw = 0.82
  # m_pfgw = 0.7625
  m_pw = 0.71
  m_pgw = 0.71
  
  nrows = 3
  ncols = 7
  figsize = None
  rotation = 0
  
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
    time_steps = list(range(51, 51 + 15)),
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
  
  pipeline.add_graphs_figure(
    name = 'tangaroa',
    graph = graph,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  # MAX MATCH
  m_start = 0.5
  m_end = 1
  num_ms = 40
  
  ms = [m_start + i * (m_end - m_start) / num_ms for i in range(num_ms)] + [m_end]
  
  max_match_pfgw = pipeline.add_max_match_pfgw(
    name = 'tangaroa_max_match_pfgw',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 51
  )
  
  max_match_pw = pipeline.add_max_match_pw(
    name = 'tangaroa_max_match_pw',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 51
  )
  
  max_match_pgw = pipeline.add_max_match_pgw(
    name = 'tangaroa_max_match_pgw',
    network = network,
    graph = graph,
    ms = ms,
    src_t = 51
  )
  
  pipeline.add_max_match_figure(
    name = 'tangaroa_max_match_pfgw',
    max_match = max_match_pfgw,
    m = m_pfgw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs,
  )
  
  pipeline.add_max_match_figure(
    name = 'tangaroa_max_match_pw',
    max_match = max_match_pw,
    m = m_pw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs,
  )
  
  pipeline.add_max_match_figure(
    name = 'tangaroa_max_match_pgw',
    max_match = max_match_pgw,
    m = m_pgw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs,
  )

  # GW
  
  gw = pipeline.add_gw(
    name = 'tangaroa_gw',
    network = network,
    num_random_iter = 5,
    random_state = 42
  )
  
  pipeline.add_couplings_figure(
    name = 'tangaroa_gw',
    graph = graph,
    couplings = gw,
    src_t = 51,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'tangaroa_gw_combined',
    graph = graph,
    couplings = gw,
    src_t = 51,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'tangaroa_gw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # fGW
  
  fgw = pipeline.add_fgw(
    name = 'tangaroa_fgw',
    network = network,
    attributes = attributes
  )
  
  pipeline.add_couplings_figure(
    name = 'tangaroa_fgw',
    graph = graph,
    couplings = fgw,
    src_t = 51,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'tangaroa_fgw_combined',
    graph = graph,
    couplings = fgw,
    src_t = 51,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'tangaroa_fgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # Wasserstein
  
  wasserstein = pipeline.add_wasserstein(
    name = 'tangaroa_w',
    network = network,
    attributes = attributes
  )
  
  pipeline.add_couplings_figure(
    name = 'tangaroa_w',
    graph = graph,
    couplings = wasserstein,
    src_t = 51,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'tangaroa_w_combined',
    graph = graph,
    couplings = wasserstein,
    src_t = 51,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'tangaroa_w.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pfGW
  
  pfgw = pipeline.add_pfgw(
    name = 'tangaroa_pfgw',
    network = network,
    attributes = attributes,
    m = m_pfgw
  )
  
  pipeline.add_couplings_figure(
    name = 'tangaroa_pfgw',
    graph = graph,
    couplings = pfgw,
    src_t = 51,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'tangaroa_pfgw_combined',
    graph = graph,
    couplings = pfgw,
    src_t = 51,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'tangaroa_pfgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pW
  
  pw = pipeline.add_pfgw(
    name = 'tangaroa_pw',
    network = network,
    attributes = attributes,
    m = m_pw
  )
  
  pipeline.add_couplings_figure(
    name = 'tangaroa_pw',
    graph = graph,
    couplings = pw,
    src_t = 51,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'tangaroa_pw_combined',
    graph = graph,
    couplings = pw,
    src_t = 51,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'tangaroa_pw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pGW
  
  pgw = pipeline.add_pgw(
    name = 'tangaroa_pgw',
    network = network,
    m = m_pgw
  )
  
  pipeline.add_couplings_figure(
    name = 'tangaroa_pgw',
    graph = graph,
    couplings = pgw,
    src_t = 51,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'tangaroa_pgw_combined',
    graph = graph,
    couplings = pgw,
    src_t = 51,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'tangaroa_pgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

################################################################################
# Tropopause
################################################################################

def tropopause(pipeline: Pipeline, fig_path: str):
  m_pfgw = 0.85
  m_pw = 0.75
  m_pgw = 0.75
  
  nrows = 3
  ncols = 5
  figsize = None
  rotation = 0
  
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
  
  pipeline.add_graphs_figure(
    name = 'tropopause',
    graph = graph,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  # MAX MATCH
  m_start = 0.5
  m_end = 1
  num_ms = 40
  
  ms = [m_start + i * (m_end - m_start) / num_ms for i in range(num_ms)] + [m_end]
  
  max_match_pfgw = pipeline.add_max_match_pfgw(
    name = 'tropopause_max_match_pfgw',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 0
  )
  
  max_match_pw = pipeline.add_max_match_pw(
    name = 'tropopause_max_match_pw',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 0
  )
  
  max_match_pgw = pipeline.add_max_match_pgw(
    name = 'tropopause_max_match_pgw',
    network = network,
    graph = graph,
    ms = ms,
    src_t = 0
  )
  
  pipeline.add_max_match_figure(
    name = 'tropopause_max_match_pfgw',
    max_match = max_match_pfgw,
    m = m_pfgw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs,
  )
  
  pipeline.add_max_match_figure(
    name = 'tropopause_max_match_pw',
    max_match = max_match_pw,
    m = m_pw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs,
  )
  
  pipeline.add_max_match_figure(
    name = 'tropopause_max_match_pgw',
    max_match = max_match_pgw,
    m = m_pw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs,
  )

  # GW
  
  gw = pipeline.add_gw(
    name = 'tropopause_gw',
    network = network,
    num_random_iter = 5,
    random_state = 42
  )
  
  pipeline.add_couplings_figure(
    name = 'tropopause_gw',
    graph = graph,
    couplings = gw,
    src_t = 0,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'tropopause_gw_combined',
    graph = graph,
    couplings = gw,
    src_t = 0,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'tropopause_gw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # fGW
  
  fgw = pipeline.add_fgw(
    name = 'tropopause_fgw',
    network = network,
    attributes = attributes
  )
  
  pipeline.add_couplings_figure(
    name = 'tropopause_fgw',
    graph = graph,
    couplings = fgw,
    src_t = 0,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'tropopause_fgw_combined',
    graph = graph,
    couplings = fgw,
    src_t = 0,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'tropopause_fgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # Wasserstein
  
  wasserstein = pipeline.add_wasserstein(
    name = 'tropopause_w',
    network = network,
    attributes = attributes
  )
  
  pipeline.add_couplings_figure(
    name = 'tropopause_w',
    graph = graph,
    couplings = wasserstein,
    src_t = 0,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'tropopause_w_combined',
    graph = graph,
    couplings = wasserstein,
    src_t = 0,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'tropopause_w.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pfGW
  
  pfgw = pipeline.add_pfgw(
    name = 'tropopause_pfgw',
    network = network,
    attributes = attributes,
    m = m_pfgw
  )
  
  pipeline.add_couplings_figure(
    name = 'tropopause_pfgw',
    graph = graph,
    couplings = pfgw,
    src_t = 0,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'tropopause_pfgw_combined',
    graph = graph,
    couplings = pfgw,
    src_t = 0,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'tropopause_pfgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pW
  
  pw = pipeline.add_pfgw(
    name = 'tropopause_pw',
    network = network,
    attributes = attributes,
    m = m_pw
  )
  
  pipeline.add_couplings_figure(
    name = 'tropopause_pw',
    graph = graph,
    couplings = pw,
    src_t = 0,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'tropopause_pw_combined',
    graph = graph,
    couplings = pw,
    src_t = 0,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'tropopause_pw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pGW
  
  pgw = pipeline.add_pgw(
    name = 'tropopause_pgw',
    network = network,
    m = m_pgw
  )
  
  pipeline.add_couplings_figure(
    name = 'tropopause_pgw',
    graph = graph,
    couplings = pgw,
    src_t = 0,
    rotation = rotation,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'tropopause_pgw_combined',
    graph = graph,
    couplings = pgw,
    src_t = 0,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    output_fmt = 'tropopause_pgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

################################################################################
# Wind Dataset
################################################################################

def wind(pipeline: Pipeline, fig_path: str):
  m_pfgw = 0.8875
  m_pw = 0.85
  m_pgw = 0.85
  
  nrows = 3
  ncols = 7
  figsize = None
  rotation = 0
  node_size = 80
  
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
    persistence_threshold = 4.824,
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
  
  pipeline.add_graphs_figure(
    name = 'wind',
    graph = graph,
    rotation = rotation,
    node_size = node_size,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  # MAX MATCH
  m_start = 0.5
  m_end = 1
  num_ms = 40
  
  ms = [m_start + i * (m_end - m_start) / num_ms for i in range(num_ms)] + [m_end]
  
  max_match_pfgw = pipeline.add_max_match_pfgw(
    name = 'wind_max_match_pfgw',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 1
  )
  
  max_match_pw = pipeline.add_max_match_pw(
    name = 'wind_max_match_pw',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 1
  )
  
  max_match_pgw = pipeline.add_max_match_pgw(
    name = 'wind_max_match_pgw',
    network = network,
    graph = graph,
    ms = ms,
    src_t = 1
  )
  
  pipeline.add_max_match_figure(
    name = 'wind_max_match_pfgw',
    max_match = max_match_pfgw,
    m = m_pfgw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs,
  )
  
  pipeline.add_max_match_figure(
    name = 'wind_max_match_pw',
    max_match = max_match_pw,
    m = m_pw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs,
  )
  
  pipeline.add_max_match_figure(
    name = 'wind_max_match_pgw',
    max_match = max_match_pgw,
    m = m_pw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs,
  )

  # GW
  
  gw = pipeline.add_gw(
    name = 'wind_gw',
    network = network,
    num_random_iter = 5,
    random_state = 42
  )
  
  pipeline.add_couplings_figure(
    name = 'wind_gw',
    graph = graph,
    couplings = gw,
    src_t = 1,
    rotation = rotation,
    node_size = node_size,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'wind_gw_combined',
    graph = graph,
    couplings = gw,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    node_size = node_size,
    output_path = fig_path,
    output_fmt = 'wind_gw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # fGW
  
  fgw = pipeline.add_fgw(
    name = 'wind_fgw',
    network = network,
    attributes = attributes
  )
  
  pipeline.add_couplings_figure(
    name = 'wind_fgw',
    graph = graph,
    couplings = fgw,
    src_t = 1,
    rotation = rotation,
    node_size = node_size,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'wind_fgw_combined',
    graph = graph,
    couplings = fgw,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    node_size = node_size,
    output_path = fig_path,
    output_fmt = 'wind_fgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # Wasserstein
  
  wasserstein = pipeline.add_wasserstein(
    name = 'wind_w',
    network = network,
    attributes = attributes
  )
  
  pipeline.add_couplings_figure(
    name = 'wind_w',
    graph = graph,
    couplings = wasserstein,
    src_t = 1,
    rotation = rotation,
    node_size = node_size,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'wind_w_combined',
    graph = graph,
    couplings = wasserstein,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    node_size = node_size,
    output_path = fig_path,
    output_fmt = 'wind_w.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pfGW
  
  pfgw = pipeline.add_pfgw(
    name = 'wind_pfgw',
    network = network,
    attributes = attributes,
    m = m_pfgw
  )
  
  pipeline.add_couplings_figure(
    name = 'wind_pfgw',
    graph = graph,
    couplings = pfgw,
    src_t = 1,
    rotation = rotation,
    node_size = node_size,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'wind_pfgw_combined',
    graph = graph,
    couplings = pfgw,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    output_path = fig_path,
    node_size = node_size,
    output_fmt = 'wind_pfgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pW
  
  pw = pipeline.add_pfgw(
    name = 'wind_pw',
    network = network,
    attributes = attributes,
    m = m_pw
  )
  
  pipeline.add_couplings_figure(
    name = 'wind_pw',
    graph = graph,
    couplings = pw,
    src_t = 1,
    rotation = rotation,
    node_size = node_size,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'wind_pw_combined',
    graph = graph,
    couplings = pw,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    node_size = node_size,
    output_path = fig_path,
    output_fmt = 'wind_pw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

  # pGW
  
  pgw = pipeline.add_pgw(
    name = 'wind_pgw',
    network = network,
    m = m_pgw
  )
  
  pipeline.add_couplings_figure(
    name = 'wind_pgw',
    graph = graph,
    couplings = pgw,
    src_t = 1,
    rotation = rotation,
    node_size = node_size,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'wind_pgw_combined',
    graph = graph,
    couplings = pgw,
    src_t = 1,
    nrows = nrows,
    ncols = ncols,
    figsize = figsize,
    node_size = node_size,
    output_path = fig_path,
    output_fmt = 'wind_pgw.png',
    savefig_kwargs = savefig_kwargs_combined
  )

################################################################################
# Vortex Street
################################################################################

def vortex_street(pipeline: Pipeline, fig_path: str):
  m_pfgw = 0.9125
  m_pw = 0.95
  m_pgw = 0.76
  
  nrows = 3
  ncols = 4
  figsize = [12 * ncols, 4 * nrows]
  rotation = 0
  node_size = 20
  
  couplings_kwargs = dict(
    src_t = 0,
    rotation = rotation,
    node_size = node_size,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  combined_couplings_kwargs = dict(
    src_t = 0,
    nrows = nrows,
    ncols = ncols,
    rotation = rotation,
    node_size = node_size,
    figsize = figsize,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs_combined
  )
  
  download = pipeline.add_download(
    name = 'vortex_street',
    display_name = 'Vortex Street',
    desc = '''
    TODO
    ''',
    url = 'https://github.com/stormymcstorm/MCOpt/releases/download/v0.5.0/VortexStreet.zip',
  )
  
  extract = pipeline.add_extract_zip(
    name = 'vortex_street',
    zips = download,
    pattern = 'monoMesh_*.vti'
  )
  
  dataset = pipeline.add_load_dataset(
    name = 'vortex_street',
    files = extract,
    time_steps = [0, 19, 37, 56, 75, 94, 112, 131, 150],
    filters = [
      vtk.ImageClipFilter(
        xmin = 0,
        xmax = 200,
        ymin = 0,
        ymax = 50,
        zmin = 0,
        zmax = 0,
      ),
    ]
  )
  
  complex = pipeline.add_complex(
    name = 'vortex_street',
    dataset = dataset,
    persistence_threshold = 0.031,
    # persistence_threshold = 0.029,
    scalar_field = 'speed'
  )
  
  graph = pipeline.add_graph(
    name = 'vortex_street',
    complex = complex,
    sample_rate = 15,
  )
  
  network = pipeline.add_mm_network(
    name = 'vortex_street',
    graph = graph,
    dist = 'geo',
    hist = 'degree',
    normalize = True
  )
  
  attributes = pipeline.add_attributes(
    name = 'vortex_street',
    graph = graph,
    normalize = True
  )
  
  pipeline.add_graphs_figure(
    name = 'vortex_street',
    graph = graph,
    rotation = rotation,
    node_size = node_size,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )
  
  # MAX MATCH
  m_start = 0.5
  m_end = 1
  num_ms = 40
  
  ms = [m_start + i * (m_end - m_start) / num_ms for i in range(num_ms)] + [m_end]
  
  max_match_pfgw = pipeline.add_max_match_pfgw(
    name = 'vortex_street_max_match_pfgw',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 0
  )
  
  max_match_pw = pipeline.add_max_match_pw(
    name = 'vortex_street_max_match_pw',
    network = network,
    graph = graph,
    attributes = attributes,
    ms = ms,
    src_t = 0
  )
  
  max_match_pgw = pipeline.add_max_match_pgw(
    name = 'vortex_street_max_match_pgw',
    network = network,
    graph = graph,
    ms = ms,
    src_t = 0
  )
  
  pipeline.add_max_match_figure(
    name = 'vortex_street_max_match_pfgw',
    max_match = max_match_pfgw,
    m = m_pfgw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs,
  )
  
  pipeline.add_max_match_figure(
    name = 'vortex_street_max_match_pw',
    max_match = max_match_pw,
    m = m_pw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs,
  )
  
  pipeline.add_max_match_figure(
    name = 'vortex_street_max_match_pgw',
    max_match = max_match_pgw,
    m = m_pgw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs_combined
  )
  
  # GW
  
  gw = pipeline.add_gw(
    name = 'vortex_street_gw',
    network = network,
    num_random_iter = 5,
    random_state = 42
  )
  
  pipeline.add_couplings_figure(
    name = 'vortex_street_gw',
    graph = graph,
    couplings = gw,
    **couplings_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'vortex_street_gw_combined',
    graph = graph,
    couplings = gw,
    output_fmt = 'vortex_street_gw.png',
    **combined_couplings_kwargs
  )

  pipeline.add_distance_figure(
    name = 'vortex_street_gw_distances',
    couplings = gw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )

  # fGW
  
  fgw = pipeline.add_fgw(
    name = 'vortex_street_fgw',
    network = network,
    attributes = attributes
  )
  
  pipeline.add_couplings_figure(
    name = 'vortex_street_fgw',
    graph = graph,
    couplings = fgw,
    **couplings_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'vortex_street_fgw_combined',
    graph = graph,
    couplings = fgw,
    output_fmt = 'vortex_street_fgw.png',
    **combined_couplings_kwargs,
  )

  pipeline.add_distance_figure(
    name = 'vortex_street_fgw_distances',
    couplings = fgw,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )

  # Wasserstein
  
  wasserstein = pipeline.add_wasserstein(
    name = 'vortex_street_w',
    network = network,
    attributes = attributes
  )
  
  pipeline.add_couplings_figure(
    name = 'vortex_street_w',
    graph = graph,
    couplings = wasserstein,
    **couplings_kwargs,
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'vortex_street_w_combined',
    graph = graph,
    couplings = wasserstein,
    output_fmt = 'vortex_street_w.png',
    **combined_couplings_kwargs,
  )

  pipeline.add_distance_figure(
    name = 'vortex_street_w_distances',
    couplings = wasserstein,
    output_path = fig_path,
    savefig_kwargs = savefig_kwargs
  )

  # pfGW
  
  pfgw = pipeline.add_pfgw(
    name = 'vortex_street_pfgw',
    network = network,
    attributes = attributes,
    m = m_pfgw
  )
  
  pipeline.add_couplings_figure(
    name = 'vortex_street_pfgw',
    graph = graph,
    couplings = pfgw,
    **couplings_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'vortex_street_pfgw_combined',
    graph = graph,
    couplings = pfgw,
    output_fmt = 'vortex_street_pfgw.png',
    **combined_couplings_kwargs
  )

  # pW
  
  pw = pipeline.add_pfgw(
    name = 'vortex_street_pw',
    network = network,
    attributes = attributes,
    m = m_pw
  )
  
  pipeline.add_couplings_figure(
    name = 'vortex_street_pw',
    graph = graph,
    couplings = pw,
    **couplings_kwargs
  )
  
  pipeline.add_couplings_combined_figure(
    name = 'vortex_street_pw_combined',
    graph = graph,
    couplings = pw,
    output_fmt = 'vortex_street_pw.png',
    **combined_couplings_kwargs
  )


  ### FULL ##
  
  # dataset_full = pipeline.add_load_dataset(
  #   name = 'vortex_street_full',
  #   files = extract,
  # )
  
  # complex_full = pipeline.add_complex(
  #   name = 'vortex_street_full',
  #   dataset = dataset_full,
  #   persistence_threshold = 0.031,
  #   # persistence_threshold = 0.029,
  #   scalar_field = 'speed'
  # )
  
  # graph_full = pipeline.add_graph(
  #   name = 'vortex_street_full',
  #   complex = complex_full,
  #   sample_rate = 15,
  # )
  
  # network_full = pipeline.add_mm_network(
  #   name = 'vortex_street_full',
  #   graph = graph_full,
  #   dist = 'geo',
  #   hist = 'degree',
  #   normalize = True
  # )
  
  # attributes_full = pipeline.add_attributes(
  #   name = 'vortex_street_full',
  #   graph = graph_full,
  #   normalize = True
  # )
  
  # GW full
  
  # gw_full = pipeline.add_gw(
  #   name = 'vortex_street_full_gw',
  #   network = network_full,
  #   distance_only = True,
  # )
  
  # pipeline.add_distance_figure(
  #   name = 'vortex_street_full_gw_distances',
  #   couplings = gw_full,
  #   output_path = fig_path,
  #   savefig_kwargs = savefig_kwargs
  # )
  
  # fGW full
  
  # fgw_full = pipeline.add_fgw(
  #   name = 'vortex_street_full_fgw',
  #   network = network_full,
  #   attributes = attributes_full,
  #   distance_only = True,
  # )
  
  # # Wasserstein Full
  
  # wasserstein_full = pipeline.add_fgw(
  #   name = 'vortex_street_full_w',
  #   network = network_full,
  #   attributes = attributes_full,
  #   distance_only = True,
  # )
  
  


################################################################################
# Make Pipeline
################################################################################

def make_pipeline():
  cache_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '__pipeline_cache__')
  fig_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'figures')
  
  pipeline = Pipeline(cache_path)
  
  gaussian(pipeline, fig_path)
  heated_cylinder(pipeline, fig_path)
  navier_stokes(pipeline, fig_path)
  red_sea(pipeline, fig_path)
  sinusoidal(pipeline, fig_path)
  tangaroa(pipeline, fig_path)
  # tropopause(pipeline, fig_path)
  wind(pipeline, fig_path)
  vortex_street(pipeline, fig_path)
  
  return pipeline

if __name__ == '__main__':
  pipeline = make_pipeline()
  
  pipeline.build_all()