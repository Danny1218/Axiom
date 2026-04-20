param(
    [string]$Backend = "benchmark-dispatch",
    [string]$ExpertUrl = "http://127.0.0.1:8000",
    [string]$ExpertModel = "onyx-qwen-production-v1",
    [string]$ExpertApiKey = "",
    [double]$Timeout = 45,
    [double]$Temperature = 0,
    [int]$MaxTokens = 96,
    [int]$MaxIterations = 10,
    [string]$TaskJson = "benchmarks/copilot_symbolic_robustness_ambiguity_stress_tasks.json",
    [string]$OutJson = "benchmark_symbolic_suite_robustness_ambiguity_stress.json",
    [string]$TaskId = "",
    [string]$RequestCaptureDir = "",
    [string]$ConfigJson = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$splat = @{
    Backend = $Backend
    ExpertUrl = $ExpertUrl
    ExpertModel = $ExpertModel
    ExpertApiKey = $ExpertApiKey
    Temperature = $Temperature
    MaxIterations = $MaxIterations
    TaskJson = $TaskJson
    OutJson = $OutJson
    TaskId = $TaskId
    RequestCaptureDir = $RequestCaptureDir
}
if ($PSBoundParameters.ContainsKey("Timeout")) {
    $splat.Timeout = $Timeout
}
if ($PSBoundParameters.ContainsKey("MaxTokens")) {
    $splat.MaxTokens = $MaxTokens
}
if ($PSBoundParameters.ContainsKey("ConfigJson")) {
    $splat.ConfigJson = $ConfigJson
}

& (Join-Path $PSScriptRoot "smoke_copilot_next_milestone_compare.ps1") @splat
