"""Basic HTML reporting for aggregated results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from biolab.reporting.aggregator import compute_win_rates
from biolab.reporting.aggregator import make_pivot_for_metric


def generate_html_report(
    aggregated_data: pd.DataFrame, output_html: Path, *args
) -> None:
    """Generate an HTML report from the aggregated DataFrame.

    Includes:
      - mapping from display_name → output_dir
      - pivot tables per metric
      - win rates (both union & intersection)
      - the full aggregated table.

    Parameters
    ----------
    aggregated_data : pd.DataFrame
        The aggregated DataFrame (with means, stds, etc.).
    output_html : Path
        Path to write the final HTML report.
    """
    # Unique mapping
    df_mapping = aggregated_data[['display_name', 'output_dir']].drop_duplicates()
    # Unique metrics
    metrics = sorted(aggregated_data['metric_name'].unique().tolist())

    df_win = compute_win_rates(aggregated_data)

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write('<html><body>\n')
        f.write('<h1>Biolab Combined HTML Report</h1>\n')

        # Model mapping
        f.write('<h2>Model Mapping (Display Name -> Output Directory)</h2>\n')
        f.write(df_mapping.to_html(index=False))
        f.write('<hr/>\n')

        # One pivot table per metric
        for metric in metrics:
            f.write(f'<h2>Metric: {metric}</h2>\n')
            pivot = make_pivot_for_metric(aggregated_data, metric)
            f.write(pivot.to_html(na_rep='–'))
            f.write('<hr/>\n')

        # Win rates
        f.write('<h2>Model Win Rates</h2>\n')
        f.write(df_win.to_html(index=False))
        f.write('<hr/>\n')

        # Full aggregated table
        f.write('<h2>Full Aggregated Table</h2>\n')
        f.write(aggregated_data.to_html(index=False, na_rep='–'))
        f.write('<hr/>\n')
        f.write('</body></html>\n')

    print(f'HTML report written to: {output_html}')
