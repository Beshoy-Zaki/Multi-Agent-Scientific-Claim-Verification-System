"""Command line interface entrypoint for MASCV."""

import argparse
import sys


def main() -> None:
    """CLI execution entrypoint.
    
    Coordinates command parsing and execution of the multi-agent claim verification pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Agent Scientific Claim Verification System (MASCV)"
    )
    parser.add_argument("--paper", type=str, help="Path to the target research paper (PDF or text)")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="outputs/reports", help="Directory to save generated reports")
    
    args = parser.parse_args()
    print("MASCV CLI initialized. Use pipeline modules or UI to execute analysis.")


if __name__ == "__main__":
    main()
