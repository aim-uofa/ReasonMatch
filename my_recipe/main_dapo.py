"""Backward-compatible alias for the public DCRL training entry point.

Use ``python -m my_recipe.main_dcrl`` for new runs.
"""

from .main_dcrl import TaskRunner, main, run_ppo

__all__ = ["TaskRunner", "main", "run_ppo"]


if __name__ == "__main__":
    main()
