# Windows Installer (Phase 12 Placeholder)

This phase uses a simple Inno Setup script to package the PyInstaller output.

## Requirements
- Inno Setup 6 installed.
- PyInstaller build located at `dist/WatermarkRemoverSuite`.

## Script
`installers/watermark_remover.iss` copies the PyInstaller output into `{app}` and generates:
- `installers/build/WatermarkRemoverSuite_Setup.exe`
- `installers/build/SHA256SUMS.txt` (placeholder hash)

## Build Steps
1. Ensure `dist/WatermarkRemoverSuite` exists (Phase 10 output).
2. Run Inno Setup (GUI or command line):
   ```powershell
   iscc installers\watermark_remover.iss
   ```
3. Verify the installer and checksum file in `installers/build`.

## Notes
- Signing of the installer is handled in Phase 11 (placeholder).
- Replace placeholder artifacts with real builds during finalization.
