import sys
import os
import runpy


def test_PyInstaller_hook():
    if getattr(sys, "frozen", False):
        from rockhopper._ragged_array import slug
        assert slug.path.exists()
        assert slug.types_map.json_path.exists()

    else:
        from rockhopper import _PyInstaller_hook
        hook_dir, = _PyInstaller_hook()
        assert os.path.isdir(hook_dir)
        hook = os.path.join(hook_dir, "hook-rockhopper.py")
        assert os.path.isfile(hook)

        namespace = runpy.run_path(hook)
        assert len(namespace["datas"]) == 2
