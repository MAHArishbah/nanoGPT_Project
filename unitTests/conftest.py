"""Shared pytest setup: make the project root importable and fix the random seed for every test."""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def seed():
    """Every test starts from the same random state, so random weights and inputs are reproducible."""
    torch.manual_seed(0)
