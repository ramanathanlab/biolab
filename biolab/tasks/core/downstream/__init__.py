"""Core downstream modeling module."""

from __future__ import annotations

from .classification import sklearn_mlp
from .classification import sklearn_multilabel_mlp
from .classification import sklearn_svc
from .regression import sklearn_svr

# TODO: make downstream task runners protocols for typing
task_map = {
    'classification': sklearn_svc,
    'regression': sklearn_svr,
    'multi-label-classification': sklearn_multilabel_mlp,
}
