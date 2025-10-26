"""
Expose CLI entry point for tests:
    from watermark_remover.cli import main
"""
try:
    from .main import main  # noqa: F401
except Exception as _e:  # keep import-time safe; tests will call only when available
    _import_error = _e  # Capture exception for later use
    def main(*args, **kwargs):  # type: ignore
        raise RuntimeError("CLI entry point unavailable") from _import_error

__all__ = ["main"]
