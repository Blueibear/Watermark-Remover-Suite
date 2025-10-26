"""Expose backend utilities within the package namespace."""

from __future__ import annotations

import backend.build as _build
import backend.generate_samples as _generate_samples
import backend.publish_release as _publish_release
import backend.run_benchmarks as _run_benchmarks
import backend.sign_release as _sign_release
import backend.verify_release as _verify_release

# Re-export the backend modules so callers can access their CLI entry points.
build = _build
generate_samples = _generate_samples
publish_release = _publish_release
run_benchmarks = _run_benchmarks
sign_release = _sign_release
verify_release = _verify_release

__all__ = [
    "build",
    "generate_samples",
    "publish_release",
    "run_benchmarks",
    "sign_release",
    "verify_release",
]
