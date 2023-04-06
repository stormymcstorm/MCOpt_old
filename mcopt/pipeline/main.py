"""
Main method for mcpipeline
"""

import argparse

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

def main():
  args = parser.parse_args()
  
  pipeline = Pipeline(
    args.config,
    use_cache=args.use_cache,
    show_progress=True,
  )
  
  pipeline.generate_all()