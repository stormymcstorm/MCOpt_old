"""
Figure to display classifications
"""

from __future__ import annotations
from typing import TypedDict, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from mcpipeline.target import Rule
from mcpipeline.targets.figure import Figure, FigureConf, FigureTarget
from mcpipeline.targets.graph import Graph, GraphTarget
from mcpipeline.targets.couplings import Couplings, CouplingsTarget
from mcpipeline.experiments.classification import Classification, ClassificationTarget
from mcpipeline.experiments.mds import MDSTransforms, MDSTarget
from mcpipeline.util import ProgressFactory

class ClassificationFigureConf(FigureConf):
  labels: List[str]
  class_hulls: bool

class ClassificationFigureRule(Rule[ClassificationFigureConf, Figure]):
  def __call__(
    self,
    mds: MDSTransforms,
    classification: Classification,
    labels: List[str],
    class_hulls: bool,
    output_path: str,
    output_fmt: str,
    savefig_kwargs: dict | None,
    progress: ProgressFactory,
  ) -> Figure:
    fig, ax = plt.subplots()
    
    ax.tick_params(
      which='both',      
      bottom=False,
      top=False,
      left=False,
      right=False,       
      labelbottom=False,
      labeltop=False,
      labelleft=False,
      labelright=False,
    )
    
    for i, class_transform in enumerate(mds.class_transforms):
      coll = ax.scatter(
        class_transform[:, 0],
        class_transform[:, 1],
        # label = labels[i]
      )
      
      if class_hulls:
        pred = np.argwhere(classification.classifications == i).ravel()
        
        hull = ConvexHull(mds.transform[pred])
        hull_points = mds.transform[pred][hull.vertices]
        
        ax.fill(
          hull_points[:, 0],
          hull_points[:, 1],
          c = coll.get_facecolor(),
          alpha = 0.10
        )
    
    # ax.legend()
    
    return Figure({0: fig}, output_path, output_fmt, savefig_kwargs)

class ClassificationFigureTarget(FigureTarget):
  def __init__(
    self, 
    name: str,
    cache_path: str,
    mds: MDSTransforms,
    classification: Classification,
    labels: List[str],
    class_hulls: bool = False,
    output_path: str | None = None,
    output_fmt: str | None = None,
    savefig_kwargs: dict | None = None,
    display_name: str | None = None,
    desc: str | None = None,
    **kwargs,
  ):
    if output_path is None:
      output_path = cache_path
    
    if output_fmt is None:
      output_fmt = name + '.png'
    
    super().__init__(
      name = name,
      cache_path = cache_path,
      rule = ClassificationFigureRule(),
      conf = {
        'labels': labels,
        'class_hulls': class_hulls,
        'output_path': output_path,
        'output_fmt': output_fmt,
        'savefig_kwargs': savefig_kwargs,
      },
      depends = [mds, classification],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )