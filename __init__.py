"""Machine-learning pipeline for the Network Intrusion Detection System.

This package is UI-independent: nothing here imports Streamlit, so every
module can be reused from the training CLI (``train.py``), the tests, or any
other caller. Import submodules explicitly, e.g. ``from src.preprocessing
import clean_flows``.
"""
