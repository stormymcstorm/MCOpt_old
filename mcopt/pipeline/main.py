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

subparsers = parser.add_subparsers(dest='subcommand')

parser_dataset = subparsers.add_parser('dataset')
parser_dataset.add_argument('name')

parser_complex = subparsers.add_parser('complex')
parser_complex.add_argument('name')

parser_graph = subparsers.add_parser('graph')
parser_graph.add_argument('name')

parser_figure = subparsers.add_parser('figure')
parser_figure.add_argument('name')

parser_targets = subparsers.add_parser('targets')

def main():
  args = parser.parse_args()
  
  pipeline = Pipeline(
    args.config,
    use_cache=args.use_cache,
    show_progress=True,
  )
  
  subcommand = args.subcommand  
  
  if subcommand is None:
    pipeline.generate_all()
  elif subcommand == 'targets':
    targets = pipeline.targets()
    
    for ty, names in targets.items():
      print(f'{ty}:')
      
      for name in names:
        print(f'\t{name}')
  elif subcommand == 'dataset':
    pipeline.dataset(args.name)
  elif subcommand == 'complex':
    pipeline.complex(args.name)
  elif subcommand == 'graph':
    pipeline.graph(args.name)
  elif subcommand == 'figure':
    pipeline.figure(args.name)