import argparse
import logging
from cellmates.inference.pipeline import run_inference_pipeline
from multiprocessing import set_start_method

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
    parser.add_argument('--tau', type=float, default=5.0)
    parser.add_argument('--save-diagnostics', action='store_true')
    parser.add_argument('--predict-cn', action='store_true')
    parser.add_argument('--layer-name', type=str, default=None)
    parser.add_argument('--jitter', type=float, default=0.1)
    parser.add_argument('--profile-hmm', action='store_true', help='Enable low/high-level HMM timing logs')
    parser.add_argument('--profile-log-path', type=str, default=None, help='Path to timing log file')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.num_processors > 1:
        method = 'spawn'
        print(f"Using {method} method for multiprocessing with {args.num_processors} processors.")
        set_start_method(method, force=True)

    run_inference_pipeline(**vars(args))

if __name__ == "__main__":
    main()
