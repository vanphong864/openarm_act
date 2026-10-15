#!/usr/bin/env python3
"""Plot loss logs from an existing FlowPolicy training directory."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openarm_flow.plotting import plot_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(plot_losses(args.directory))
