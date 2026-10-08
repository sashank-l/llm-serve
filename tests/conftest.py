import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--model-path",
        action="store",
        default="/home/surya/models/qwen2.5-1.5b",
        help="Path to model directory",
    )
