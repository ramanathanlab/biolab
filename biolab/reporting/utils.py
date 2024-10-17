"""Utilities for result reporting."""

from __future__ import annotations

from pathlib import Path

from biolab.api.metric import Metric
from biolab.evaluate import EvalConfig


def discover_results(input_dirs: list[Path]) -> dict[str, dict[str, Metric]]:
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
        model_id = None
        # Try different possible fields for model id
        for key in ['pretrained_model_name_or_path', 'weights_path']:
            if hasattr(lm_config, key):
                model_id = getattr(lm_config, key)
                break
        if model_id is None:
            # If no specific identifier is found, we can use a hash of the lm_config
            model_id = str(hash(str(lm_config)))
        unique_model_name = f'{model_name} ({model_id})'

        # Now process the report files in input_dir
        for report_file in input_dir.glob('*.report'):
            filename = report_file.stem  # Filename without extension
            # Filename format: MODELNAME_TASKNAME.report
            try:
                _, task_name = filename.split('_', 1)
            except ValueError:
                print(f'Invalid filename format: {filename}')
                continue
            # Load the metric from the report file
            metric = Metric()
            metric.load(report_file)
            results.setdefault(unique_model_name, {})[task_name] = metric
    return results
