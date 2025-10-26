"""
Expose CLI entry point for tests:
    from watermark_remover.cli import main
"""
try:
    from .wmr import main  # noqa: F401
except Exception as _e:  # keep import-time safe; tests will call only when available
    def main(*args, **kwargs):  # type: ignore
        raise RuntimeError("CLI entry point unavailable") from _e

__all__ = ["main", "wmr"]
