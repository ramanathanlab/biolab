"""Command-line interface for generating benchmark reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from biolab.reporting.reporter import generate_combined_report


def main():
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Generate benchmark reports from run directories.'
    )
    parser.add_argument(
        '--run_dirs',
        nargs='+',
        type=Path,
        required=True,
        help='One or more run directories (each with config.yaml and *.metrics).',
    )
    parser.add_argument(
        '--out',
        default='reports',
        type=Path,
        help='Output directory for the combined summary.',
    )
    parser.add_argument(
        '--reporter',
        choices=['html', 'dash'],
        default='html',
        help='Choose the output reporter type.',
    )
    args = parser.parse_args()

    generate_combined_report(args.run_dirs, args.out, reporter=args.reporter)


if __name__ == '__main__':
    main()
