#!/usr/bin/env python
"""Export verification results into PDF, Markdown, or LaTeX format."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Export scientific report")
    parser.add_argument("--state-file", required=True, help="Serialized InvestigationState file")
    parser.add_argument("--format", choices=["md", "latex", "json"], default="md")
    args = parser.parse_args()

    print(f"Exporting report in {args.format} format from {args.state_file}...")


if __name__ == "__main__":
    main()
