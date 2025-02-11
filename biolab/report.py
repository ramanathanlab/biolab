"""Command-line interface for generating benchmark reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from biolab.reporting import generate_aggregated_csv
from biolab.reporting import report_aggregated_metrics


def generate_report(
    run_dirs: list[Path], aggregated_csv: Path, output_dir: Path, reporter: str
) -> None:
    """
    Generate the specified report given run directories or an existing aggregated CSV.

    Parameters
    ----------
    run_dirs : list[Path]
        List of run directories.
    aggregated_csv : Path
        Path to aggregated CSV (will be created if it doesn't exist).
    output_dir : Path
        Output directory for the combined summary.
    reporter : str
        'html' or 'dash'.
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if run_dirs:
        # Generate or overwrite the aggregated CSV from the run directories
        if not aggregated_csv or not aggregated_csv.exists():
            aggregated_csv = output_dir / 'all_results.csv'
        df_agg = generate_aggregated_csv(run_dirs, aggregated_csv)
    else:
        # If run_dirs not provided, we must have aggregated_csv
        df_agg = pd.read_csv(aggregated_csv)

    # Generate the final report
    report_aggregated_metrics(df_agg, output_dir, reporter)


def main():
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Generate benchmark reports from run directories.'
    )
    parser.add_argument(
        '--run-dirs',
        nargs='+',
        type=Path,
        help=(
            'One or more run directories (each with config.yaml and *.metrics). '
            'Either provide this or --aggregated-csv.'
        ),
    )
    parser.add_argument(
        '--aggregated-csv',
        type=Path,
        help=(
            'Path to an aggregated CSV. If it exists, data is loaded from it. '
            'If it doesn’t exist but run-dirs were provided, it will be created. '
            'One of --run-dirs or --aggregated-csv must be provided.'
        ),
    )
    parser.add_argument(
        '--out',
        default=Path('reports'),
        type=Path,
        help='Output directory for the combined summary and any generated reports.',
    )
    parser.add_argument(
        '--reporter',
        choices=['html', 'dash'],
        default='html',
        help='Reporter type, supported: html, dash. Default: html.',
    )
    args = parser.parse_args()

    if not args.run_dirs and not args.aggregated_csv:
        parser.error('Either --run-dirs or --aggregated-csv must be provided.')

    generate_report(
        run_dirs=args.run_dirs,
        aggregated_csv=args.aggregated_csv,
        output_dir=args.out,
        reporter=args.reporter,
    )


if __name__ == '__main__':
    main()
