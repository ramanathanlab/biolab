"""Reporters for generating and displaying benchmark results."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from biolab import model_registry
from biolab import task_registry
from biolab.api.logging import logger
from biolab.api.metric import Metric


# Reporter Interface
class Reporter(ABC):
    """Abstract base class for reporters."""

    def __init__(self, results: dict[str, dict[str, Metric]]):
        self.results = results
        # Collect all unique tasks
        self.tasks = sorted({task for model in results.values() for task in model})
        # Gather model and task metadata
        self.model_metadata = self._gather_model_metadata()
        self.task_metadata = self._gather_task_metadata()

    @abstractmethod
    def generate_report(self, *args, **kwargs):
        """Generate the report."""
        pass

    def _gather_model_metadata(self) -> dict[str, dict[str, Any]]:
        """Gather metadata for each model from the model_registry."""
        metadata = {}
        for unique_model_name in self.results.keys():
            model_name, _ = unique_model_name.split(' (', 1)
            model_info = model_registry.get(model_name)
            if model_info:
                metadata[unique_model_name] = {
                    'model_input': getattr(
                        model_info['class'], 'model_input', 'Unknown'
                    ),
                    'model_encoding': getattr(
                        model_info['class'], 'model_encoding', 'Unknown'
                    ),
                }
            else:
                metadata[unique_model_name] = {
                    'model_input': 'Unknown',
                    'model_encoding': 'Unknown',
                }
        return metadata

    def _gather_task_metadata(self) -> dict[str, dict[str, Any]]:
        """Gather metadata for each task from the task_registry."""
        metadata = {}
        for task_name in self.tasks:
            task_info = task_registry.get(task_name)
            if task_info:
                metadata[task_name] = {
                    'resolution': getattr(task_info['class'], 'resolution', 'Unknown'),
                }
            else:
                metadata[task_name] = {
                    'resolution': 'Unknown',
                }
        return metadata

    def _prepare_model_entries(
        self, group_models_by: str = 'none'
    ) -> list[tuple[str, str, str, dict[str, Metric], dict[str, Any]]]:
        """Prepare and sort model entries for reporting."""
        model_entries = []
        for unique_model_name, model_tasks in self.results.items():
            # Split the model name and identifier
            model_name, model_id = unique_model_name.split(' (', 1)
            model_id = '(' + model_id  # Add back the opening parenthesis
            metadata = self.model_metadata.get(unique_model_name, {})
            model_entries.append(
                (model_name, model_id, unique_model_name, model_tasks, metadata)
            )

        # Determine the sorting and grouping key
        if group_models_by in {'model_input', 'model_encoding'}:
            # Ensure the grouping key exists in metadata
            def sort_key(x):
                return (
                    x[4].get(group_models_by, 'Unknown'),  # Grouping attribute
                    x[0],  # Model name
                    x[1],  # Model ID
                )
        else:
            # Default sorting
            def sort_key(x):
                return (x[0], x[1])

        # Sort the model entries based on the grouping key
        model_entries.sort(key=sort_key)
        return model_entries


# Reporter Implementations


class ConsoleReporter(Reporter):
    """Generates a console report using Rich."""

    def _truncate_left(self, text: str, max_length: int) -> str:
        """Truncate text from the left to fit within max_length."""
        if len(text) <= max_length:
            return text
        return '…' + text[-(max_length - 1) :]

    def generate_report(self, group_models_by: str = 'none', show_charts: bool = False):
        """Generate and display the console report."""
        console = Console(theme=Theme({'title': 'bold underline'}))

        # Prepare sorted model entries
        model_entries = self._prepare_model_entries(group_models_by)

        # Create table with styles
        table = Table(title='Benchmark Results', title_justify='left', style='dim')
        table.header_style = 'bold magenta'
        table.row_styles = ['none', 'dim']

        # TODO: parameterize? or make dynamic?
        model_column_width = 30

        # Add columns to the table
        table.add_column('Model', width=model_column_width, no_wrap=False)
        # TODO: parameterize? or make dynamic?
        max_task_name_length = 20
        for task in self.tasks:
            task_display = (
                task
                if len(task) <= max_task_name_length
                else f'{task[:max_task_name_length - 1]}…'
            )
            table.add_column(task_display, justify='center')

        # Find the best score for each task
        best_scores = {}
        for task in self.tasks:
            scores = [
                model_tasks[task].test_acc
                for model_tasks in self.results.values()
                if task in model_tasks and model_tasks[task].test_acc is not None
            ]
            best_scores[task] = max(scores) if scores else None

        # Group the model entries based on group_models_by
        if group_models_by in {'model_input', 'model_encoding'}:
            group_function = lambda x: x[4].get(group_models_by, 'Unknown')  # noqa: E731
        else:
            group_function = lambda x: None  # noqa: E731

        model_entries_grouped = groupby(model_entries, key=group_function)

        # Add rows to the table
        for group_value, group in model_entries_grouped:
            if group_models_by != 'none':
                # Add a separator row or header
                group_title = (
                    f"{group_models_by.replace('_', ' ').title()}: {group_value}"
                )
                table.add_section()
                table.add_row(
                    Text(group_title, style='bold underline'), *[''] * (len(self.tasks))
                )

            for model_entry in group:
                model_name, model_id, _unique_model_name, model_tasks, _metadata = (
                    model_entry
                )

                # Truncate the model_id from the left
                truncated_model_id = self._truncate_left(
                    model_id, model_column_width - len(model_name) - 1
                )

                # Combine model name and truncated identifier
                model_display = f'{model_name} {truncated_model_id}'

                row = [model_display]
                for task in self.tasks:
                    metric = model_tasks.get(task)
                    if metric and metric.test_acc is not None:
                        score_value = metric.test_acc
                        score = f'{score_value:.3f}'
                        if score_value == best_scores[task]:
                            # Highlight the best score
                            score = Text(score, style='bold green')
                    else:
                        score = Text('N/A', style='italic red')
                    row.append(score)
                table.add_row(*row)

        # Render the table to console
        console.print(table)

        # If show_charts is enabled, display bar charts per task
        if show_charts:
            self._display_bar_charts(console, model_entries, best_scores)

    def _display_bar_charts(
        self, console: Console, model_entries: list, best_scores: dict
    ):
        """Display bar charts for each task."""
        # Prepare data for bar charts
        task_scores = defaultdict(list)
        for model_entry in model_entries:
            model_name, model_id, unique_model_name, model_tasks, metadata = model_entry
            display_name = f'{model_name} {self._truncate_left(model_id, 15)}'
            for task in self.tasks:
                metric = model_tasks.get(task)
                if metric and metric.test_acc is not None:
                    score_value = metric.test_acc
                    task_scores[task].append((display_name, score_value))
                else:
                    task_scores[task].append((display_name, None))

        # Display bar charts per task
        for task in self.tasks:
            console.print(f'\n[bold]Task: {task}[/bold]')
            # Get scores and labels
            scores = task_scores[task]
            # Find maximum score for normalization
            max_score = max(
                (score for _, score in scores if score is not None),
                default=1.0,
            )

            # Create table for bar chart
            chart_table = Table(show_header=False)
            chart_table.add_column('Model')
            chart_table.add_column('Score', justify='right')
            chart_table.add_column('Bar')

            for display_name, score in scores:
                if score is not None:
                    # Normalize score to bar length
                    # TODO: parameterize? or make dynamic?
                    bar_length = 20
                    normalized_length = int((score / max_score) * bar_length)
                    bar = '█' * normalized_length
                    bar_text = Text(bar)
                    if score == best_scores[task]:
                        bar_text.stylize('bold green')
                    else:
                        bar_text.stylize('cyan')
                    score_text = f'{score:.3f}'
                    chart_table.add_row(display_name, score_text, bar_text)
                else:
                    chart_table.add_row(display_name, 'N/A', '')

            console.print(chart_table)


class MarkdownReporter(Reporter):
    """Generates a markdown report."""

    def generate_report(self, output_file: Path):
        """Generate and save the markdown report."""
        model_entries = self._prepare_model_entries()

        with open(output_file, 'w') as f:
            # Write the header row
            headers = ['Model', *self.tasks]
            f.write('| ' + ' | '.join(headers) + ' |\n')
            # Write the separator row
            f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
            # Write the data rows
            for model_name, model_id, _unique_model_name, model_tasks in model_entries:
                # Truncate or format model_id as needed
                truncated_model_id = self._truncate_left(
                    model_id, 30 - len(model_name) - 1
                )
                model_display = f'{model_name} {truncated_model_id}'

                row = [model_display]
                for task in self.tasks:
                    metric = model_tasks.get(task)
                    if metric and metric.test_acc is not None:
                        row.append(f'{metric.test_acc:.3f}')
                    else:
                        row.append('N/A')
                f.write('| ' + ' | '.join(row) + ' |\n')
        logger.info(f'Markdown report generated at {output_file}')


class PDFReporter(Reporter):
    """Generates a PDF report."""

    def generate_report(self, output_file: Path = Path('report.pdf')):
        """Generate and save the PDF report."""
        # TODO: this is o1 generated placeholder
        from weasyprint import HTML

        # Generate HTML content
        html_content = '<html><head><title>Benchmark Results</title></head><body>'
        html_content += '<h1>Benchmark Results</h1>'
        html_content += (
            '<table border="1" cellpadding="4" cellspacing="0"><tr><th>Model</th>'
        )
        for task in self.tasks:
            html_content += f'<th>{task}</th>'
        html_content += '</tr>'
        for unique_model_name, model_tasks in self.results.items():
            html_content += f'<tr><td>{unique_model_name}</td>'
            for task in self.tasks:
                metric = model_tasks.get(task)
                if metric and metric.test_acc is not None:
                    html_content += f'<td>{metric.test_acc:.3f}</td>'
                else:
                    html_content += '<td>N/A</td>'
            html_content += '</tr>'
        html_content += '</table></body></html>'

        # Convert HTML to PDF
        HTML(string=html_content).write_pdf(output_file)
        logger.info(f'PDF report generated at {output_file}')