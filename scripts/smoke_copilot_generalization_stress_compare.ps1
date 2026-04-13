param(
    [string]$Backend = "benchmark-dispatch",
    [string]$ExpertUrl = "http://127.0.0.1:8000",
    [string]$ExpertModel = "onyx-qwen-production-v1",
    [string]$ExpertApiKey = "",
    [double]$Temperature = 0,
    [int]$MaxIterations = 10,
    [string]$TaskJson = "benchmarks/copilot_symbolic_generalization_stress_tasks.json",
    [string]$OutJson = "benchmark_symbolic_suite_generalization_stress.json"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

& (Join-Path $PSScriptRoot "smoke_copilot_next_milestone_compare.ps1") `
    -Backend $Backend `
    -ExpertUrl $ExpertUrl `
    -ExpertModel $ExpertModel `
    -ExpertApiKey $ExpertApiKey `
    -Temperature $Temperature `
    -MaxIterations $MaxIterations `
    -TaskJson $TaskJson `
    -OutJson $OutJson
