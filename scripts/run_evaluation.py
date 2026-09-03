#!/usr/bin/env python
"""Run benchmark evaluation and generate metrics comparison."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate MASCV against baselines")
    parser.add_argument("--benchmark", default="data/benchmarks/scifact", help="Benchmark dataset path")
    args = parser.parse_args()

    print(f"Evaluation runner initialized on benchmark: {args.benchmark}")


if __name__ == "__main__":
    main()
