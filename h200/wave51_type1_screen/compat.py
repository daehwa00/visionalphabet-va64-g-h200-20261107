"""Launch a Python entry point with support for images using Python 3.10."""
import runpy
import sys
import typing

if not hasattr(typing, "override"):
    import typing_extensions

    typing.override = typing_extensions.override

entry = sys.argv.pop(1)
runpy.run_path(entry, run_name="__main__")
