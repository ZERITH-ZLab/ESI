import argparse
import json
from .compute import compute


def main():
    parser = argparse.ArgumentParser(description='Compute robust state/action statistics; no images or training.')
    parser.add_argument('--index', required=True)
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(compute(args.index, args.base_checkpoint, args.output)), flush=True)


if __name__ == '__main__':
    main()
