"""General reporting functions, used by specific reporters to display results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from biolab.api.logging import logger  # assuming your project has this
from biolab.reporting.aggregator import combine_scores_and_aggregate
from biolab.reporting.parsers import parse_run_directory


def generate_aggregated_csv(run_dirs: list[Path], output_csv: Path) -> pd.DataFrame:
    """Generate an aggregated CSV from multiple run directories.

    Parameters
    ----------
    run_dirs : list of Path
        Each directory has config.yaml and *.metrics files.
    output_csv : Path
        Where to save the combined results.

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
    """Generate the final report (HTML or Dash) from the aggregated data.

    Parameters
    ----------
    aggregated_data : pd.DataFrame
        Already-aggregated DataFrame.
    output_dir : Path
        Output directory.
    reporter : str, optional
        The type of reporter to use, 'html' or 'dash'.
    """
    if reporter.lower() == 'html':
        from biolab.reporting.reporters.html_reporter import generate_html_report

        html_path = output_dir / 'all_results.html'
        generate_html_report(aggregated_data, html_path)
    elif reporter.lower() == 'dash':
        from biolab.reporting.reporters.dash_reporter import serve_dash_report

        serve_dash_report(aggregated_data, output_dir)
    else:
        raise ValueError(f'Unknown reporter type: {reporter}')
