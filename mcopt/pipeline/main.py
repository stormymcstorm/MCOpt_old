import os
import argparse
import json

from mcopt.pipeline.pipeline import Pipeline

parser = argparse.ArgumentParser()

parser.add_argument(
  "--config", 
  dest='config',
  help='The config file to use',
  default = './pipeline.config.json',
  type=str,
)

parser.add_argument(
  '--no-cache',
  dest='use_cache',
  help='Whether or not to use cache if possible',
  default=True,
  action='store_false'
)

def main():
  args = parser.parse_args()
  
  config_path = args.config
  
  pipeline = Pipeline(config_path, use_cache=args.use_cache)
  pipeline.load_all()
  