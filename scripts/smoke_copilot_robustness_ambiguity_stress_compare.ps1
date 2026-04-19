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
    [string]$RequestCaptureDir = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

& (Join-Path $PSScriptRoot "smoke_copilot_next_milestone_compare.ps1") `
    -Backend $Backend `
    -ExpertUrl $ExpertUrl `
    -ExpertModel $ExpertModel `
    -ExpertApiKey $ExpertApiKey `
    -Timeout $Timeout `
    -Temperature $Temperature `
    -MaxTokens $MaxTokens `
    -MaxIterations $MaxIterations `
    -TaskJson $TaskJson `
    -OutJson $OutJson `
    -TaskId $TaskId `
    -RequestCaptureDir $RequestCaptureDir
