"""Generate a report from the benchmark results."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from biolab.reporting import REPORTERS
from biolab.reporting.utils import discover_results


def generate_report(input_dirs: list[Path], output_format: str, **kwargs) -> None:
    """Generate a report from the benchmark results."""
    results = discover_results(input_dirs)

    reporter_class = REPORTERS.get(output_format)
    if not reporter_class:
        print(f'Unsupported output format: {output_format}')
        return

    reporter = reporter_class(results)
    reporter.generate_report(**kwargs)


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate reports from benchmark results.')
    subparsers = parser.add_subparsers(
        dest='format',
        required=True,
        help='Output format. Call {format} --help for more info.',
    )

    # Console format subparser
    console_parser = subparsers.add_parser('console', help='Generate a console report')

    console_parser.add_argument(
        '--input',
        type=Path,
        required=True,
        nargs='+',
        help='Path(s) to the input directory(s)',
    )
    console_parser.add_argument(
        '--group-models-by',
        type=str,
        choices=['model_input', 'model_encoding', 'none'],
        default='none',
        help='Attribute to group models by in the report',
    )
    console_parser.add_argument(
        '--show-charts',
        action='store_true',
        help='Enable bar charts to visualize model performance per task',
    )

    # Markdown format subparser
    markdown_parser = subparsers.add_parser(
        'markdown', help='Generate a markdown report'
    )

    markdown_parser.add_argument(
        '--input',
        type=Path,
        required=True,
        nargs='+',
        help='Path(s) to the input directory(s)',
    )
    markdown_parser.add_argument(
        '--output-file',
        type=Path,
        required=True,
        help='Path to the output markdown file',
    )

    # PDF format subparser
    pdf_parser = subparsers.add_parser('pdf', help='Generate a PDF report')

    pdf_parser.add_argument(
        '--input',
        type=Path,
        required=True,
        nargs='+',
        help='Path(s) to the input directory(s)',
    )
    pdf_parser.add_argument(
        '--output-file',
        type=Path,
        required=True,
        help='Path to the output PDF file',
    )

    args = parser.parse_args()

    # Prepare keyword arguments for generate_report
    kwargs = vars(args).copy()
    input_dirs = kwargs.pop('input')
    output_format = kwargs.pop('format')

    generate_report(input_dirs, output_format, **kwargs)
