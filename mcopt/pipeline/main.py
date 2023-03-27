import os
import argparse
import json

from mcopt.pipeline.pipeline import Pipeline

parser = argparse.ArgumentParser(
  formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

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

parser.add_argument(
  '--run',
  dest='run',
  help='the experiment to run',
)


def main():
  args = parser.parse_args()
  
  config_path = args.config
  
  pipeline = Pipeline(config_path, use_cache=args.use_cache)
  
  if args.run is None:
    pipeline.generate_all()
    return
  
  target = args.run
  
  pipeline.run(target)
  
  