"""Parsing model and metric information from configuration and metric files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_config_yaml(config_path: Path) -> dict[str, str]:
    """Parse the configuration YAML file and extract model information.

    Parameters
    ----------
    config_path : Path
        Path to the config.yaml file.

    Returns
    -------
    dict[str, str]
        Fields such as:
          - model_id
          - display_name
          - model_name
          - output_dir
    """
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    lm_cfg = config_data.get('lm_config', {})

    # Generate a unique model_id by hashing the eval configuration
    config_str = json.dumps(config_data, sort_keys=True)
    model_id = hashlib.md5(config_str.encode('utf-8')).hexdigest()

    # Use configuration name for first part of display if available
    model_name = lm_cfg.get('name', 'UnknownModel')

    # Retrieve the output directory and take final components for display
    output_dir_raw = Path(config_data.get('output_dir', 'UnknownOutputDir'))
    output_dir_short = output_dir_raw.name

    # Create a unique display name for the model
    display_name = f'{model_name} {output_dir_short}'

    return {
        'model_id': model_id,
        'display_name': display_name,
        'model_name': model_name,
        'output_dir': config_data.get('output_dir', 'UnknownOutputDir'),
    }


def parse_metric_file(metric_file: Path, model_info: dict[str, str]) -> pd.DataFrame:
    """Parse a metric file (.metrics) and return a DataFrame with metric details.

    Parameters
    ----------
    metric_file : Path
        Path to a .metrics file.
    model_info : dict[str, str]
        Dictionary containing model metadata from the config file
        (e.g. 'model_id', 'display_name', 'output_dir').

    Returns
    -------
    pd.DataFrame
        Columns include:
          - model_id
          - display_name
          - output_dir
          - task_name
          - metric_name
          - is_higher_better
          - train_scores / test_scores / train_mean / test_mean
    """
    filename = metric_file.name
    # e.g. "Something_CaLM-Meltome.metrics" → "CaLM-Meltome"
    parts = filename.replace('.metrics', '').split('_', maxsplit=1)
    task_name = parts[1] if len(parts) == 2 else 'UnknownTask'  # noqa: PLR2004

    with open(metric_file) as f:
        metric_entries = json.load(f)

    rows = []
    for entry in metric_entries:
        class_name = entry['class_name']
        train_scores = entry.get('train_scores') or []
        test_scores = entry.get('test_scores') or []
        train_mean = float(np.mean(train_scores)) if train_scores else None
        test_mean = float(np.mean(test_scores)) if test_scores else None

        rows.append(
            {
                'model_id': model_info['model_id'],
                'display_name': model_info['display_name'],
                'output_dir': model_info.get('output_dir', 'UnknownOutput'),
                'task_name': task_name,
                'metric_name': class_name,
                'is_higher_better': entry.get('is_higher_better', True),
                'train_scores': train_scores,
                'test_scores': test_scores,
                'train_mean': train_mean,
                'test_mean': test_mean,
            }
        )
    return pd.DataFrame(rows)


def parse_run_directory(run_dir: Path) -> pd.DataFrame:
    """Parse a run directory to extract the configuration and all *.metrics files.

    Parameters
    ----------
    run_dir : Path
        Path to the run directory (must contain config.yaml and *.metrics files).

    Returns
    -------
    pd.DataFrame
        DataFrame containing parsed information from the run.
    """
    config_path = run_dir / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f'config.yaml not found in {run_dir}')

    model_info = parse_config_yaml(config_path)
    metric_files = list(run_dir.glob('*.metrics'))

    df_list = [parse_metric_file(mf, model_info) for mf in metric_files]
    if df_list:
        combined = pd.concat(df_list, ignore_index=True)
    else:
        columns = [
            'model_id',
            'display_name',
            'output_dir',
            'task_name',
            'metric_name',
            'is_higher_better',
            'train_scores',
            'train_mean',
            'test_scores',
            'test_mean',
        ]
        combined = pd.DataFrame(columns=columns)

    return combined
