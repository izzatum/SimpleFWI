"""Run a small public experiment: python -m simplefwi.fwi2d --output run."""
import argparse
import json
from pathlib import Path

from .experiments import run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,help='SI JSON experiment; default: small Camembert')
    parser.add_argument('--output',type=Path,default=Path('fwi2d-run'),help='new output directory')
    parser.add_argument('--resume',type=Path,help='accepted checkpoint with matching experiment identity')
    args = parser.parse_args(argv)
    config = None if args.config is None else json.loads(args.config.read_text())
    run_experiment(config,output=args.output,
                   base_path='.' if args.config is None else args.config.resolve().parent,
                   resume=args.resume)
    print(args.output/'manifest.json')


if __name__ == '__main__':
    main()
