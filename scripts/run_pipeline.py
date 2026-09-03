#!/usr/bin/env python
"""Execution script to run MASCV pipeline on a specified research paper."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Run MASCV claim verification pipeline")
    parser.add_argument("--paper", required=True, help="Path to research paper PDF")
    parser.add_argument("--config", default="config/default_config.yaml", help="Configuration file")
    args = parser.parse_args()

    print(f"Pipeline runner initialized for paper: {args.paper}")
    # Pipeline execution will be wired here once implementation is complete.


if __name__ == "__main__":
    main()
