[CmdletBinding()]
param([int]$Port = 8082)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SslCheckpoint = 'D:\Final Year Projet\ASTROVET_data\artifacts\models\ssl_encoder_all_qlp_trainhosts_20261030.npz'
$FusionModel = 'D:\Final Year Projet\ASTROVET_data\artifacts\models\planet_disposition_prototype\ssl_physics_fusion_all_qlp_20261030\astrovet_ssl_physics_fusion_research.joblib'

foreach ($RequiredPath in @($SslCheckpoint, $FusionModel)) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "Required trained artifact is missing: $RequiredPath"
    }
}

$env:ASTROVET_SSL_CHECKPOINT = $SslCheckpoint
$env:ASTROVET_FUSION_MODEL = $FusionModel
$env:PYTHONPATH = (Join-Path $ProjectRoot 'src') + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" })
Set-Location -LiteralPath $ProjectRoot

Write-Host 'Starting ASTROVET with the 11,511-file QLP-trained encoder and its matching vetting head.'
Write-Host 'Scores are for human review and are not population-wide planet probabilities.'
Write-Host "Dashboard: http://127.0.0.1:$Port/"
& py -m astrovet.cli ui --host 127.0.0.1 --port $Port
