# Packaging Guide

## Requirements
- PyInstaller (`pip install pyinstaller`)
- Windows environment for `.exe` build (current target)

## Build Steps
1. Ensure virtual environment is active: `.venv\Scripts\activate`
2. Run PyInstaller using the provided spec:
   ```powershell
   pyinstaller backend\pyinstaller.spec --distpath dist --workpath build
   ```
3. Resulting executable: `dist/WatermarkRemoverSuite/WatermarkRemoverSuite.exe`

## Included Resources
- `config/config.yaml`
- `assets/` directory
- `docs/` directory

## Troubleshooting
- Delete `build/` and `dist/` before rebuilding if issues persist.
- Use `--clean` with PyInstaller to clear caches.

## Smoke Test
After building, confirm the executable launches:
```powershell
dist\WatermarkRemoverSuite\WatermarkRemoverSuite.exe --help
```
