# Post-Release Verification (Mock)

## Requirements
- `dist/WatermarkRemoverSuite_signed.exe`
- `installers/build/WatermarkRemoverSuite_Setup.exe`
- `installers/build/SHA256SUMS.txt`

## Command
```powershell
python backend\verify_release.py `
    --artifacts dist\WatermarkRemoverSuite_signed.exe installers\build\WatermarkRemoverSuite_Setup.exe `
    --checksums installers\build\SHA256SUMS.txt `
    --log verification_reports\hash_verification.log
```

- Computes SHA-256 for each artifact.
- Compares against checksum file when present.
- Appends results to `verification_reports/hash_verification.log` and returns exit code `0` on success.

## Notes
- Replace this script with real download checks when integrating with GitHub.
- Ensure that verification runs in CI after publishing artifacts.
