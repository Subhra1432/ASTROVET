[CmdletBinding()]
param([int]$Port = 8080)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataRoot = 'D:\Final Year Projet\ASTROVET_data\artifacts\models\planet_disposition_prototype'
$SslCheckpoint = Join-Path $DataRoot 'ssl_encoder_trainhosts_v1.npz'
$FusionModel = Join-Path $DataRoot 'ssl_physics_fusion\astrovet_ssl_physics_fusion_research.joblib'
foreach ($RequiredPath in @($SslCheckpoint, $FusionModel)) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "Required trained artifact is missing: $RequiredPath"
    }
}
$env:ASTROVET_SSL_CHECKPOINT = $SslCheckpoint
$env:ASTROVET_FUSION_MODEL = $FusionModel
$env:PYTHONPATH = (Join-Path $ProjectRoot 'src') + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" })
Set-Location -LiteralPath $ProjectRoot
Write-Host "Starting ASTROVET with its real-data SSL encoder and selected-TOI research fusion head."
Write-Host "Scores are for human review and are not population-wide planet probabilities."
Write-Host "Dashboard: http://127.0.0.1:$Port/"
& py -m astrovet.cli ui --host 127.0.0.1 --port $Port
