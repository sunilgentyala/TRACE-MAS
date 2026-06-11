"""Shared fixtures for TRACE-MAS test suite."""

import sys
import os

# Allow importing from src/ without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
