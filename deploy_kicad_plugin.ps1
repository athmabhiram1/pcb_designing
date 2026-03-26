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
    "$env:APPDATA\kicad\9.0\scripting\plugins\ai_pcb_assistant",
    "$env:APPDATA\KiCad\9.0\3rdparty\plugins\ai_pcb_assistant",
    "$env:APPDATA\kicad\9.0\3rdparty\plugins\ai_pcb_assistant"
)

$targets = $targets |
    Where-Object { $_ -and $_.Trim().Length -gt 0 } |
    ForEach-Object { [System.IO.Path]::GetFullPath($_) } |
    Sort-Object -Unique

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

    Get-ChildItem -Path $srcDir -Force | ForEach-Object {
        $name = $_.Name
        if ($name -eq "__pycache__") {
            return
        }
        Copy-Item -Path $_.FullName -Destination (Join-Path $target $name) -Recurse -Force
    }

    $pycache = Join-Path $target "__pycache__"
    if (Test-Path $pycache) {
        Remove-Item -Recurse -Force $pycache
    }

    Write-Host "Deployed plugin to:" $target
}

Write-Host "Done. Restart KiCad completely before testing."
