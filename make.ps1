<#
.SYNOPSIS
    Windows shim for the Makefile. GNU make is not installed by default on Windows,
    and the build machine for this project is a Windows box, so every Makefile
    target is mirrored here verbatim.

.EXAMPLE
    ./make.ps1 validate
    ./make.ps1 check
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'install', 'new-task', 'gold-patches', 'validate', 'verify-gold',
        'catalogue', 'smoke', 'smoke-gym', 'test', 'lint', 'fmt', 'check', 'freshclone', 'clean')]
    [string]$Target = 'help',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Invoke-Step {
    param([string]$Label, [scriptblock]$Body)
    Write-Host "==> $Label" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) { throw "$Label failed (exit $LASTEXITCODE)" }
}

$uv = 'uv'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $candidate = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314\Scripts\uv.exe'
    if (Test-Path $candidate) { $uv = $candidate }
    else { throw "uv not found on PATH. Install it: pip install uv" }
}

switch ($Target) {
    'help' {
        Write-Host "rl-coding-agents targets:" -ForegroundColor Green
        @(
            @('install', 'Create the venv and install everything (core + dev)'),
            @('new-task', 'Scaffold a blank task: ./make.ps1 new-task my_id --category bug_fix'),
            @('gold-patches', 'Regenerate every solution/gold.patch from solution/fixed/'),
            @('validate', 'Schema-validate every task in benchmark/tasks/**'),
            @('verify-gold', 'Prove every gold patch fixes its task; untouched repo fails'),
            @('catalogue', 'Regenerate benchmark/catalogue.md'),
            @('smoke', 'Run ONE task end-to-end and print the reward'),
            @('smoke-gym', 'Same, over real NeMo Gym HTTP servers (Linux/macOS)'),
            @('test', 'Unit tests'),
            @('lint', 'Ruff lint + format check'),
            @('fmt', 'Auto-fix lint and format'),
            @('check', 'Everything CI runs'),
            @('freshclone', 'The five-step acceptance check (spec 0.3)'),
            @('clean', 'Remove caches and sandbox scratch')
        ) | ForEach-Object { "  {0,-12} {1}" -f $_[0], $_[1] }
    }
    'install' { Invoke-Step 'uv sync' { & $uv sync --extra dev } }
    'new-task' { Invoke-Step 'scaffold task' { & $uv run python benchmark/tools/new_task.py @Rest } }
    'gold-patches' { Invoke-Step 'build gold patches' { & $uv run python benchmark/tools/make_gold_patch.py --all @Rest } }
    'validate' { Invoke-Step 'validate tasks' { & $uv run python benchmark/schema/validate_tasks.py --all @Rest } }
    'verify-gold' { Invoke-Step 'verify gold patches' { & $uv run python scripts/verify_gold_patches.py --split all @Rest } }
    'catalogue' { Invoke-Step 'build catalogue' { & $uv run python benchmark/tools/build_catalogue.py @Rest } }
    'smoke' { Invoke-Step 'smoke one task' { & $uv run python scripts/smoke_one_task.py --limit 1 @Rest } }
    'smoke-gym' { Invoke-Step 'smoke via NeMo Gym' { & $uv run python scripts/smoke_one_task.py --limit 1 --transport gym @Rest } }
    'test' { Invoke-Step 'pytest' { & $uv run pytest @Rest } }
    'lint' {
        Invoke-Step 'ruff check' { & $uv run ruff check . }
        Invoke-Step 'ruff format --check' { & $uv run ruff format --check . }
    }
    'fmt' {
        Invoke-Step 'ruff check --fix' { & $uv run ruff check --fix . }
        Invoke-Step 'ruff format' { & $uv run ruff format . }
    }
    'check' {
        & $PSCommandPath validate
        & $PSCommandPath verify-gold
        & $PSCommandPath test
        & $PSCommandPath lint
    }
    'freshclone' {
        Invoke-Step 'uv sync' { & $uv sync --extra dev }
        & $PSCommandPath validate
        & $PSCommandPath verify-gold
        & $PSCommandPath test
        & $PSCommandPath smoke
    }
    'clean' {
        Get-ChildItem -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        '.pytest_cache', '.ruff_cache', '.sandbox' | ForEach-Object {
            Remove-Item $_ -Recurse -Force -ErrorAction SilentlyContinue
        }
        Write-Host 'clean' -ForegroundColor Green
    }
}
