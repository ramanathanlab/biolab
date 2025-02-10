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
    """Generate the specified report given input (run directories or aggregated CSV).

    Parameters
    ----------
    run_dirs : list[Path]
        List of run directories.
    aggregated_csv : Path
        Path to aggregated CSV. (This will be made if it doesn't already exist.)
    output_dir : Path
        Output directory for the combined summary.
    reporter : str
        Choose the output reporter type.

    Returns
    -------
    None
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if run_dirs:
        # Generate the aggregated CSV from the run directories
        if not aggregated_csv or not aggregated_csv.exists():
            aggregated_csv = output_dir / 'all_results.csv'
        df_agg = generate_aggregated_csv(run_dirs, aggregated_csv)

    elif aggregated_csv:
        # Read the aggregated CSV
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
            'Either provide this or the aggregated CSV.'
        ),
    )
    parser.add_argument(
        '--aggregated-csv',
        type=Path,
        help=(
            'Path to aggregated CSV. If this is provided and exists, Data will be'
            " loaded from it instead of regenerating from the input dirs.If it doesn't"
            ' exist, it will be created ( this arg name or `<OUT>/all-results.csv`).'
        ),
    )
    parser.add_argument(
        '--out',
        default=Path('reports'),
        type=Path,
        help='Output directory for the combined summary and other report files.',
    )
    parser.add_argument(
        '--reporter',
        choices=['html', 'dash'],
        default='html',
        help='Reporter type, supported: html, dash. Default: html.',
    )
    args = parser.parse_args()

    if not args.run_dirs and not args.aggregated_csv:
        parser.error('Either `--run-dirs` or `--aggregated-csv` must be provided.')

    generate_report(
        run_dirs=args.run_dirs,
        aggregated_csv=args.aggregated_csv,
        output_dir=args.out,
        reporter=args.reporter,
    )


if __name__ == '__main__':
    main()
