param(
    [string]$TaskId = "noisy_affine_thermometer",
    [string]$TaskJson = "benchmarks/copilot_symbolic_robustness_ambiguity_stress_tasks.json",
    [string]$ExpertUrl = "http://127.0.0.1:8000",
    [string]$ExpertModel = "onyx-qwen-production-v1",
    [string]$ExpertApiKey = "",
    [int]$Repeats = 3,
    [string]$RequestCaptureDir = "",
    [string]$OutDir = "debug_onyx_latency_sweeps"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Get-JsonDoc {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try {
        return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json)
    }
    catch {
        return $null
    }
}

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
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    foreach ($Timeout in @(45, 60, 90, 120)) {
        foreach ($MaxTokens in @(16, 32, 64)) {
            $jsonOut = Join-Path $OutDir ("{0}_timeout{1}_maxtokens{2}.json" -f $TaskId, $Timeout, $MaxTokens)
            $cmd = @(
                "python",
                "scripts/profile_onyx_task_latency.py",
                "--task-id", $TaskId,
                "--task-json", $TaskJson,
                "--timeout", ("{0}" -f $Timeout),
                "--max-tokens", ("{0}" -f $MaxTokens),
                "--repeats", ("{0}" -f $Repeats),
                "--json-out", $jsonOut
            )
            Write-Host ("==> Running: {0}" -f ($cmd -join " ")) -ForegroundColor Cyan
            & $cmd[0] $cmd[1..($cmd.Length - 1)]
            if ($LASTEXITCODE -ne 0) {
                throw "Latency sweep run failed with exit code $LASTEXITCODE for timeout=$Timeout max_tokens=$MaxTokens."
            }
            $doc = Get-JsonDoc -Path $jsonOut
            if ($null -eq $doc) {
                throw "Failed to read sweep JSON at '$jsonOut'."
            }
            $summary = $doc.summary
            $attempts = @($doc.attempts)
            $compileCount = @($attempts | Where-Object { $_.compile_ok -eq $true }).Count
            $metricCount = @($attempts | Where-Object { $_.metric_ok -eq $true }).Count
            Write-Host (
                "SWEEP timeout={0} max_tokens={1} repeats={2} success={3} timeout_count={4} compile_ok={5} metric_ok={6} mean_elapsed={7} json={8}" -f `
                $Timeout,
                $MaxTokens,
                $summary.repeats,
                $summary.success_count,
                $summary.timeout_count,
                $compileCount,
                $metricCount,
                $summary.mean_elapsed,
                $jsonOut
            )
        }
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
