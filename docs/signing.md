# Code Signing Workflow (Placeholder)

Phase 11 requires a signed executable. Actual signing is environment-specific; this project provides a mock workflow.

## Environment Variables
- `CODESIGN_PFX_PATH` – path to `.pfx` certificate (placeholder).
- `CODESIGN_PFX_PASSWORD` – certificate password (placeholder).

## Mock Signing Script
```powershell
python backend\sign_release.py --input dist\WatermarkRemoverSuite\WatermarkRemoverSuite.exe `
    --output dist\WatermarkRemoverSuite_signed.exe `
    --log signature_verification.log
```
- Copies the executable to the target path.
- Logs actions and environment variables (masking the password) to `signature_verification.log`.

## Verification
Record a SHA-256 hash for traceability:
```powershell
Get-FileHash dist\WatermarkRemoverSuite_signed.exe -Algorithm SHA256 >> signature_verification.log
```

## Notes
- Replace the mock script with real signing (e.g., `signtool.exe`) once certificates are available.
- Keep sensitive secrets out of source control.
