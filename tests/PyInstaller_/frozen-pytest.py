# -*- coding: utf-8 -*-
"""
Freeze pytest.main() with rockhopper included.
"""
import sys
import pytest
import rockhopper

pytest.main(sys.argv[1:] + ["--no-cov"])
