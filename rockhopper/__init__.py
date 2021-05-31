# -*- coding: utf-8 -*-
"""
"""


class RequestMeError(NotImplementedError):
    """Raised for features which I'd like to add but haven't."""

    def __str__(self):
        return self.args[0] + \
            " If you genuinely need this (i.e. you didn't land here by " \
            "accident) then let me know by raising an issue at " \
            "https://github.com/bwoodsend/rockhopper/issues/new"


from ._version import __version__, __version_info__
from ._ragged_array import RaggedArray, ragged_array


def _PyInstaller_hook():
    """Tell PyInstaller where to find ``hook-rockhopper.py``."""
    import os
    return [os.path.dirname(__file__)]
