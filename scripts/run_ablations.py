#!/usr/bin/env python
"""Execute ablation matrix across the 8 defined system configurations."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run MASCV ablation studies")
    parser.add_argument("--config", default="config/evaluation/ablation_studies.yaml")
    args = parser.parse_args()

    print("Ablation study matrix runner initialized.")


if __name__ == "__main__":
    main()
