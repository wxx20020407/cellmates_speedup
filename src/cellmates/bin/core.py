import argparse
import logging
from cellmates.inference.pipeline import run_inference_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Estimate distance matrix and build phylogenetic tree")
    parser.add_argument('--input', '-i', required=True, help="Path to input AnnData file")
    parser.add_argument('--output', '-o', default=None, help="Path to output directory")
    parser.add_argument('--n-states', '-s', type=int, default=7)
    parser.add_argument('--max-iter', '-m', type=int, default=30)
    parser.add_argument('--verbose', '-v', type=int, default=0)
    parser.add_argument('--num-processors', '-p', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--jc-correction', action='store_true')
    parser.add_argument('--rtol', '-t', type=float, default=1e-3)
    parser.add_argument('--learn-obs-params', action='store_true')
    parser.add_argument('--numpy', action='store_true')
    parser.add_argument('--use-copynumbers', action='store_true')
    parser.add_argument('--tau', type=float, default=50.0)
    parser.add_argument('--save-diagnostics', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    run_inference_pipeline(**vars(args))

if __name__ == "__main__":
    main()
