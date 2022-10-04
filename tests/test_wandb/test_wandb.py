"""Test wandb logging."""
from application.t2d.generate_features_and_write_to_disk import log_to_wandb

# pylint: disable=missing-function-docstring


def test_log_to_wandb():
    log_to_wandb("test", "x", "y")
