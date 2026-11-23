<#
.SYNOPSIS
  Runs the frontend/backend contract tests (no .env is read or written).
#>
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repo
try {
    python -m unittest tests.test_frontend_backend_contract -v
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
