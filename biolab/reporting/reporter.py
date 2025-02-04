# biolab/reporting/reporter.py

from pathlib import Path
import pandas as pd
from biolab.reporting.parsers import parse_run_directory
from biolab.reporting.aggregator import combine_scores_and_aggregate

def generate_aggregated_csv(run_dirs: list[Path], output_csv: Path) -> pd.DataFrame:
    """
    Generate an aggregated CSV from multiple run directories.

    Parameters
    ----------
    run_dirs : list[Path]
        List of directories containing run data (each with config.yaml and *.metrics files).
    output_csv : Path
        Path to write the aggregated CSV file.
    
    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame.
    """
    df_all = []
    for rd in run_dirs:
        df_run = parse_run_directory(rd)
        if not df_run.empty:
            df_all.append(df_run)
    
    if not df_all:
        raise ValueError("No data found in any run directories.")
    
    df_combined_raw = pd.concat(df_all, ignore_index=True)
    df_agg = combine_scores_and_aggregate(df_combined_raw)
    df_agg.to_csv(output_csv, index=False)
    print(f"Aggregated CSV written to: {output_csv}")
    return df_agg

def generate_combined_report(
    run_dirs: list[Path],
    output_dir: Path,
    output_csv: str = "all_results.csv",
    reporter: str = "html"
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
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / output_csv
    df_agg = generate_aggregated_csv(run_dirs, csv_path)
    
    if reporter.lower() == "html":
        from biolab.reporting.reporters.html_reporter import generate_html_report
        html_path = output_dir / "all_results.html"
        generate_html_report(csv_path, html_path)
    elif reporter.lower() == "dash":
        from biolab.reporting.reporters.dash_reporter import serve_dash_report
        serve_dash_report(csv_path)
    else:
        raise ValueError(f"Unknown reporter type: {reporter}")
