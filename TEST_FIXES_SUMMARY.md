# Test Failures Triage & Fixes Summary

## Overview
Fixed 7 test failures/blockers to enable green CI runs on CPU baseline tests.

## Failures Classification & Fixes

### 1. **Collection Blocker** ✅ FIXED
- **Issue**: `backend/verify_release.py` had `sys.exit(1)` at module level when `GITHUB_TOKEN` missing
- **Category**: Test Infrastructure
- **Fix**: Moved validation into `main()` function instead of module-level execution
- **Files**: `backend/verify_release.py`

### 2. **Import/Packaging Issues** ✅ FIXED
- **Issue**: `test_gui.py` - PyQt5 not installed, `QObject` undefined in fallback
- **Category**: Import/packaging
- **Fix**: Added stub classes for all PyQt5 imports when library not available
- **Files**: `ui/main_window.py`

### 3. **Env/Tooling** ✅ MARKED
- **Issue**: `test_signing.py` - Windows `signtool.exe` not available on Linux
- **Category**: Env/tooling (platform-specific)
- **Fix**: Marked with `@pytest.mark.windows_only` to skip on non-Windows CI
- **Files**: `tests/test_signing.py`, `pytest.ini`

### 4. **Network/External Dependencies** ✅ MARKED
- **Issue**: `test_publish.py` - Attempts real git push (HTTP 403)
- **Category**: Network/integration
- **Fix**: Marked with `@pytest.mark.integration` to exclude from default CI
- **Files**: `tests/test_publish.py`, `pytest.ini`

### 5. **API Mismatch** ✅ FIXED
- **Issue**: `test_verify.py` (2 tests) - Missing `_hash_file()` and `parse_args()` functions
- **Category**: Implementation doesn't match test expectations
- **Fix**: Added missing API functions as aliases/adapters
- **Files**: `backend/verify_release.py`

### 6. **Test Infrastructure** ✅ FIXED
- **Issue**: `test_config_logging.py` - Log file not created (Windows `%VAR%` syntax on Linux)
- **Category**: Platform quirks (path expansion)
- **Fix**: Updated `_resolve_log_path()` to handle both `%VAR%` (Windows) and `$VAR` (Unix) env variable syntax
- **Files**: `core/logger.py`

## Changes Made

### Core Fixes
1. **backend/verify_release.py**
   - Removed module-level `sys.exit(1)` blocking imports
   - Added `_hash_file()` alias for `sha256()`
   - Added `parse_args()` for test compatibility
   - Made `main()` accept optional args and return `(code, messages)` tuple

2. **ui/main_window.py**
   - Added stub classes for PyQt5 imports when not available
   - Stubs: `QObject`, `QRunnable`, `Qt`, `QThreadPool`, `pyqtSignal`

3. **core/logger.py**
   - Added Windows-style `%VAR%` environment variable expansion
   - Now works cross-platform with both Windows and Unix path syntaxes

### Test Infrastructure
4. **pytest.ini** (NEW)
   - Added markers: `gpu`, `integration`, `slow`, `windows_only`
   - Configured default pytest behavior

5. **tests/test_signing.py**
   - Marked with `@pytest.mark.windows_only` and `@pytest.mark.integration`

6. **tests/test_publish.py**
   - Marked with `@pytest.mark.integration`

### CI/CD
7. **.github/workflows/ci.yml** (NEW)
   - Matrix: Ubuntu + Windows, Python 3.11 + 3.12
   - Installs ffmpeg on both platforms
   - Runs CPU baseline tests only: `-m "not gpu and not integration and not slow and not windows_only"`

8. **pyproject.toml**
   - Added `[project.optional-dependencies] test` section
   - Includes: pytest, PyYAML, moviepy, GitPython, requests

## Test Results

### Before Fixes
- 6-7 test failures
- 1 collection blocker (couldn't run tests at all without GITHUB_TOKEN)

### After Fixes
- ✅ **25 tests PASSED** (CPU baseline)
- ⏭️ **2 tests SKIPPED** (GUI tests - PyQt5/Tkinter not installed, properly handled)
- ⏭️ **2 tests DESELECTED** (integration tests - excluded by markers)
- ⚠️ **37 warnings** (moviepy video frame reading - expected, not errors)

### Test Command
```bash
pytest -v -m "not gpu and not integration and not slow and not windows_only"
```

## Remaining Work (Optional)

### For Full Coverage
- **GPU tests**: Require model weights, should run on nightly workflow with GPU runners
- **Integration tests**: Need mocking or separate workflow with credentials
- **Windows-only tests**: Need Windows runner with signtool installed

### Recommendations
1. Keep PR focused on CPU baseline tests (current state)
2. Add GPU/integration tests in follow-up PR with proper infrastructure
3. Consider mocking external dependencies (git push, signtool) for broader coverage

## CI Workflow

The new CI workflow:
- Runs on push to `main` and `claude/**` branches
- Runs on pull requests to `main`
- Tests on Ubuntu + Windows with Python 3.11 and 3.12
- Installs ffmpeg (required for video tests)
- Only runs CPU baseline unit tests (fast, no external dependencies)
- Uploads test artifacts on failure for debugging

## Conclusion

All CPU baseline tests now pass cleanly. The test suite is ready for CI integration without fighting external dependencies, platform-specific tools, or network requirements.
