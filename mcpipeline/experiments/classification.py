"""
Experiment for KNearestNeighbor
"""

from __future__ import annotations
from typing import Dict, List, TypedDict
import os
import pickle
import json

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics

from mcpipeline.entity import Entity
from mcpipeline.target import Target, Rule, Conf
from mcpipeline.targets.combine import CombinedGraphs, CombineGraphsTarget
from mcpipeline.targets.couplings import Couplings, CouplingsTarget
from mcpipeline.util import ProgressFactory

__all__ = ['Classification', 'ClassificationTarget', 'KNearestNeighborsTarget']

class Classification(Entity):
  classifications: np.ndarray
  score: float
  f1_scores: float
  
  def __init__(
    self, 
    classifications: np.ndarray, 
    score: float,
    f1_scores: float,
  ):
    self.classifications = classifications
    self.score = score
    self.f1_scores = f1_scores
    
  def save(self, cache_path: str, progress: ProgressFactory):
    with open(os.path.join(cache_path, 'score'), 'w') as score_file:
      score_file.write(str(self.score))
      
    with open(os.path.join(cache_path, 'f1_scores.json'), 'w') as score_file:
      json.dump({
        'f1_scores': list(self.f1_scores)
      }, score_file)
    
class ClassificationConf(TypedDict):
  train_size: float | int
  random_state: int | None

class ClassificationTarget(Target[Conf, Classification]):
  @staticmethod
  def target_type() -> str:
    return 'classification'
  
  @staticmethod
  def entity_type() -> type[Classification]:
    return Classification

class KNearestNeighborsConf(ClassificationConf):
  n_neighbors: int

class KNearestNeighborsRule(Rule[KNearestNeighborsConf, Classification]):
  def __call__(
    self, 
    graphs: CombinedGraphs,
    couplings: Couplings, 
    train_size: float | int,
    random_state: int | None,
    n_neighbors: int,
    progress: ProgressFactory
  ) -> Classification:
    # indexes = [instances_bi, instances_tri]
    indexes = [
      np.asarray(list(graphs.index_map[i].values()))
      for i in graphs.index_map.keys()
    ]
    
    train_indexes = []
    test_indexes = []
    
    for indices in indexes:
      # instance #'s in binary gaussian
      train_indices, test_indices = train_test_split(
        indices, train_size=train_size, random_state=random_state
      )
      
      train_indexes.append(train_indices)
      test_indexes.append(test_indices)
    
    train_indices = np.hstack(train_indexes)
    test_indices = np.hstack(test_indexes)
    
    nearest_neighbors = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
    
    X = couplings.distances[:, train_indices]
    Y_true = np.zeros(shape=X.shape[0])
    
    for c, indices in enumerate(indexes):
      Y_true[indices] = c

    X_train = X[train_indices, :]
    Y_train = Y_true[train_indices]
    
    X_test = X[test_indices, :]
    Y_test = Y_true[test_indices]
    
    nearest_neighbors.fit(X_train, Y_train)
    
    score = nearest_neighbors.score(X_test, Y_test)
    f1_scores = sklearn.metrics.f1_score(
      Y_test, nearest_neighbors.predict(X_test), 
      average=None,
      labels=list(graphs.index_map.keys()),
    )
    
    classifications = nearest_neighbors.predict(X)
    
    return Classification(classifications, score, f1_scores)

class KNearestNeighborsTarget(ClassificationTarget[KNearestNeighborsConf]):
  def __init__(
    self, 
    name: str,
    cache_path: str,
    graphs: CombineGraphsTarget,
    couplings: CouplingsTarget, 
    train_size: float | int = 0.8,
    random_state: int | None = None,
    n_neighbors: int = 3,
    display_name: str | None = None,
    desc: str | None = None,
    **kwargs,
  ):
    if display_name is None:
      display_name = graphs.display_name
    
    if desc is None:
      desc = graphs.desc
    
    super().__init__(
      name = name,
      cache_path = cache_path,
      rule = KNearestNeighborsRule(),
      conf = {
        'train_size': train_size,
        'random_state': random_state,
        'n_neighbors': n_neighbors,
      },
      depends = [graphs, couplings],
      display_name = display_name,
      desc = desc,
      **kwargs,
    )