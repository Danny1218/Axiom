param(
    [string]$TaskId = "noisy_affine_thermometer",
    [string]$TaskJson = "benchmarks/copilot_symbolic_robustness_ambiguity_stress_tasks.json",
    [string]$ExpertUrl = "http://127.0.0.1:8000",
    [string]$ExpertModel = "onyx-qwen-production-v1",
    [string]$ExpertApiKey = "",
    [double]$Timeout = 45,
    [int]$MaxTokens = 64,
    [int]$Repeats = 3,
    [string]$RequestCaptureDir = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$priorUrl = [Environment]::GetEnvironmentVariable("AXIOM_EXPERT_URL", "Process")
$priorModel = [Environment]::GetEnvironmentVariable("AXIOM_EXPERT_MODEL", "Process")
$priorApiKey = [Environment]::GetEnvironmentVariable("AXIOM_EXPERT_API_KEY", "Process")
$priorCaptureDir = [Environment]::GetEnvironmentVariable("AXIOM_ONYX_REQUEST_CAPTURE_DIR", "Process")

$env:AXIOM_EXPERT_URL = $ExpertUrl
$env:AXIOM_EXPERT_MODEL = $ExpertModel
if ([string]::IsNullOrWhiteSpace($ExpertApiKey)) {
    Remove-Item Env:AXIOM_EXPERT_API_KEY -ErrorAction SilentlyContinue
}
else {
    $env:AXIOM_EXPERT_API_KEY = $ExpertApiKey
}
if ([string]::IsNullOrWhiteSpace($RequestCaptureDir)) {
    Remove-Item Env:AXIOM_ONYX_REQUEST_CAPTURE_DIR -ErrorAction SilentlyContinue
}
else {
    $env:AXIOM_ONYX_REQUEST_CAPTURE_DIR = $RequestCaptureDir
}

try {
    $cmd = @(
        "python",
        "scripts/profile_onyx_task_latency.py",
        "--task-id", $TaskId,
        "--task-json", $TaskJson,
        "--timeout", ("{0}" -f $Timeout),
        "--max-tokens", ("{0}" -f $MaxTokens),
        "--repeats", ("{0}" -f $Repeats)
    )
    Write-Host ("==> Running: {0}" -f ($cmd -join " ")) -ForegroundColor Cyan
    & $cmd[0] $cmd[1..($cmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Latency profiling run failed with exit code $LASTEXITCODE."
    }
}
finally {
    if ([string]::IsNullOrWhiteSpace($priorUrl)) {
        Remove-Item Env:AXIOM_EXPERT_URL -ErrorAction SilentlyContinue
    }
    else {
        $env:AXIOM_EXPERT_URL = $priorUrl
    }
    if ([string]::IsNullOrWhiteSpace($priorModel)) {
        Remove-Item Env:AXIOM_EXPERT_MODEL -ErrorAction SilentlyContinue
    }
    else {
        $env:AXIOM_EXPERT_MODEL = $priorModel
    }
    if ([string]::IsNullOrWhiteSpace($priorApiKey)) {
        Remove-Item Env:AXIOM_EXPERT_API_KEY -ErrorAction SilentlyContinue
    }
    else {
        $env:AXIOM_EXPERT_API_KEY = $priorApiKey
    }
    if ([string]::IsNullOrWhiteSpace($priorCaptureDir)) {
        Remove-Item Env:AXIOM_ONYX_REQUEST_CAPTURE_DIR -ErrorAction SilentlyContinue
    }
    else {
        $env:AXIOM_ONYX_REQUEST_CAPTURE_DIR = $priorCaptureDir
    }
}
