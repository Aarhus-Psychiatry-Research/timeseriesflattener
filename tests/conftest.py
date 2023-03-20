# pylint: disable-all

import pytest


def pytest_addoption(parser):  # noqa
    parser.addoption(
        "--skiphuggingface",
        action="store_true",
        default=False,
        help="run tests that use huggingface models",
    )


def pytest_configure(config):  # noqa
    config.addinivalue_line(
        "markers",
        "huggingface: mark test as using huggingface models",
    )


def pytest_collection_modifyitems(config, items):  # noqa
    if config.getoption("--skiphuggingface"):
        # --skiphuggingface given in cli: skip huggingface tests
        skip_hf = pytest.mark.skip(reason="remove --skiphuggingface option to run")
        for item in items:
            if "huggingface" in item.keywords:
                item.add_marker(skip_hf)
        return
