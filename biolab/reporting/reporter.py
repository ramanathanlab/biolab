"""General reporting functions, used by specific reporters to display results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from biolab.api.logging import logger
from biolab.reporting.aggregator import combine_scores_and_aggregate
from biolab.reporting.parsers import parse_run_directory


def generate_aggregated_csv(run_dirs: list[Path], output_csv: Path) -> pd.DataFrame:
    """
    Generate an aggregated CSV from multiple run directories.

    Parameters
    ----------
    run_dirs : list[Path]
        List of directories containing run data
        (each with config.yaml and *.metrics files).
    output_csv : Path
        Path to write the aggregated CSV file.

    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame.
    """
    df_all = []
    for rd in run_dirs:
        logger.info(f'Parsing run directory: {rd}')
        df_run = parse_run_directory(rd)
        if not df_run.empty:
            df_all.append(df_run)

    if not df_all:
        raise ValueError('No data found in any run directories.')

    df_combined_raw = pd.concat(df_all, ignore_index=True)
    df_agg = combine_scores_and_aggregate(df_combined_raw)
    df_agg.to_csv(output_csv, index=False)
    print(f'Aggregated CSV written to: {output_csv}')
    return df_agg


def report_aggregated_metrics(
    aggregated_data: pd.DataFrame,
    output_dir: Path,
    reporter: str = 'html',
) -> None:
    """
    Generate an aggregated CSV and hand off to a downstream reporter.

    Parameters
    ----------
    run_dirs : list[Path]
        List of run directories.
    output_dir : Path
        Output directory for the aggregated CSV and final report.
    output_csv : str, optional
        Filename for the aggregated CSV.
    reporter : str, optional
        The reporter type to use. Options: 'html', 'dash', etc.
    """
    if reporter.lower() == 'html':
        from biolab.reporting.reporters.html_reporter import generate_html_report

        html_path = output_dir / 'all_results.html'
        generate_html_report(aggregated_data, html_path)
    elif reporter.lower() == 'dash':
        from biolab.reporting.reporters.dash_reporter import serve_dash_report

        # This function doesn't need output directory, but can take it for now
        # (for api consistency)
        serve_dash_report(aggregated_data, output_dir)
    else:
        raise ValueError(f'Unknown reporter type: {reporter}')
