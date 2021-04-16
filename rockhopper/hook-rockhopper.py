# -*- coding: utf-8 -*-
"""
Hook for PyInstaller.
"""

from rockhopper._ragged_array import slug

# Put the cslug binary and its types json in a `rockhopper/_slugs` directory.
datas = [(str(slug.path), "rockhopper/_slugs"),
         (str(slug.types_map.json_path), "rockhopper/_slugs")]
