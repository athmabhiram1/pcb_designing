param(
    [switch]$NoBackup
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcDir = Join-Path $repoRoot "plugin"

if (-not (Test-Path $srcDir)) {
    throw "Source plugin folder not found: $srcDir"
}

$targets = @(
    "$env:APPDATA\KiCad\9.0\scripting\plugins\ai_pcb_assistant",
    "$env:APPDATA\kicad\9.0\scripting\plugins\ai_pcb_assistant"
)

$targets = $targets |
    Where-Object { $_ -and $_.Trim().Length -gt 0 } |
    ForEach-Object { [System.IO.Path]::GetFullPath($_) } |
    Sort-Object -Unique

$filesToCopy = @(
    "__init__.py",
    "plugin.py",
    "pcbnew_action.py",
    "metadata.json"
)

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

foreach ($target in $targets) {
    $parent = Split-Path -Parent $target
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    if ((Test-Path $target) -and (-not $NoBackup)) {
        $backup = "$target.backup_$stamp"
        Rename-Item -Path $target -NewName (Split-Path -Leaf $backup)
    }

    if (-not (Test-Path $target)) {
        New-Item -ItemType Directory -Path $target -Force | Out-Null
    }

    foreach ($file in $filesToCopy) {
        $src = Join-Path $srcDir $file
        if (Test-Path $src) {
            Copy-Item -Force $src -Destination (Join-Path $target $file)
        }
    }

    $srcResources = Join-Path $srcDir "resources"
    if (Test-Path $srcResources) {
        Copy-Item -Recurse -Force $srcResources -Destination (Join-Path $target "resources")
    }

    $pycache = Join-Path $target "__pycache__"
    if (Test-Path $pycache) {
        Remove-Item -Recurse -Force $pycache
    }

    Write-Host "Deployed plugin to:" $target
}

Write-Host "Done. Restart KiCad completely before testing."
