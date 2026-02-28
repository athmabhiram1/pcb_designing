param(
    [string]$OutputRoot = "dist/pcm"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pluginDir = Join-Path $repoRoot "plugin"

if (-not (Test-Path $pluginDir)) {
    throw "Plugin directory not found: $pluginDir"
}

$stageRoot = Join-Path $repoRoot ".pcm_stage"
if (Test-Path $stageRoot) {
    Remove-Item -Recurse -Force $stageRoot
}

New-Item -ItemType Directory -Path $stageRoot | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stageRoot "plugins") | Out-Null

$metadataPath = Join-Path $pluginDir "metadata.json"
Copy-Item $metadataPath -Destination (Join-Path $stageRoot "metadata.json") -Force

$metadata = Get-Content -Raw -Path $metadataPath | ConvertFrom-Json
$version = $metadata.versions[0].version
if (-not $version) {
    throw "Could not determine version from metadata.json"
}

$pluginFiles = @(
    "__init__.py",
    "pcbnew_action.py",
    "plugin.py"
)

foreach ($file in $pluginFiles) {
    $src = Join-Path $pluginDir $file
    if (-not (Test-Path $src)) {
        throw "Required plugin file missing: $src"
    }
    Copy-Item $src -Destination (Join-Path $stageRoot "plugins\$file") -Force
}

$resourcesDir = Join-Path $pluginDir "resources"
if (Test-Path $resourcesDir) {
    Copy-Item -Recurse -Force $resourcesDir -Destination (Join-Path $stageRoot "resources")
}

$outputDir = Join-Path $repoRoot $OutputRoot
$versionDir = Join-Path $outputDir ("v" + $version)
New-Item -ItemType Directory -Path $versionDir -Force | Out-Null

$zipName = "ai-pcb-assistant-pcm-v$version.zip"
$zipPath = Join-Path $versionDir $zipName
if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}

Compress-Archive -Path (Join-Path $stageRoot "*") -DestinationPath $zipPath
Write-Host "PCM package created:" $zipPath
