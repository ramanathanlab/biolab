"""Utilities for result reporting."""

from __future__ import annotations

from pathlib import Path

from biolab.api.metric import MetricCollection
from biolab.evaluate import EvalConfig
from biolab.metrics import metric_registry


def discover_results(input_dirs: list[Path]) -> dict[str, dict[str, MetricCollection]]:
    """Discover results from the benchmark runs."""
    results = {}
    for input_dir in input_dirs:
        # Load config.yaml from input_dir
        config_path = input_dir / 'config.yaml'
        if not config_path.exists():
            print(f'No config.yaml found in {input_dir}')
            continue
        try:
            config = EvalConfig.from_yaml(config_path)
        except Exception as e:
            print(f'Failed to load config from {config_path}: {e}')
            continue
        # Extract the model configuration
        lm_config = config.lm_config
        # Try to get a unique identifier for the model
        model_name = getattr(lm_config, 'name', 'UnknownModel')
        # Hash the configuration to get a unique identifier
        model_id = str(hash(str(config)))
        unique_model_name = f'{model_name} ({model_id})'

        # Now process the report files in input_dir
        for report_file in input_dir.glob('*.metrics'):
            filename = report_file.stem  # Filename without extension
            # Filename format: MODELNAME_TASKNAME.metrics
            try:
                _, task_name = filename.split('_', 1)
            except ValueError:
                print(f'Invalid filename format: {filename}')
                continue
            # Load the metric from the report file
            metric_collection = MetricCollection()
            metric_collection.load(report_file, metric_registry)

            results.setdefault(unique_model_name, {})[task_name] = metric_collection

    return results
