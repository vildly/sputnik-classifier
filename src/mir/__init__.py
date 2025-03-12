# src/mir/__init__.py
# Importing the entire `main` module, so you can access functions, variables, and classes via `main`
from . import main  # ✅ Allows you to reference `main.function_name`, `main.app`, etc.
# Add other sources here if needed to be accessed outside the scope of the module!

# Explicitly importing `app` from `main`, so `app` can be directly used when importing `mir`
# from .main import app  # ✅ Allows `from mir import app` without needing `mir.main.app`

# Fixes any Ruff or Pyright warnings
# due to not using the import
__all__ = ["main"]
