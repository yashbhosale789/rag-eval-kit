"""Load the user's adapter from a "module:attr" string.

The adapter is any object with:
    version: str                         # what you changed; part of every result key
    corpus_id: str = ""                  # optional; recorded, not checked
    retrieve(query) -> list[dict]        # ranked; each {"text", "source", "page"?}
    answer(query) -> dict                # {"answer", "contexts": [str]}; generation only
"""

import importlib
import inspect
import os
import sys


def load_adapter(spec: str) -> object:
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise SystemExit(f"bad adapter {spec!r}: use module:attr, e.g. my_rag:Adapter")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())  # like uvicorn: adapters live in the user's project
    obj = getattr(importlib.import_module(module_name), attr)
    return obj() if inspect.isclass(obj) else obj
