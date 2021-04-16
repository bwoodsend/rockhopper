# -*- coding: utf-8 -*-
"""
"""

from ._version import __version__, __version_info__
from ._ragged_array import RaggedArray, ragged_array


def _PyInstaller_hook():
    """Tell PyInstaller where to find ``hook-rockhopper.py``."""
    import os
    return [os.path.dirname(__file__)]
