param(
    [string]$Backend = "benchmark-dispatch",
    [string]$ExpertUrl = "http://127.0.0.1:8000",
    [string]$ExpertModel = "onyx-qwen-production-v1",
    [string]$ExpertApiKey = "",
    [double]$Timeout = 45,
    [double]$Temperature = 0,
    [int]$MaxTokens = 96,
    [int]$MaxIterations = 10,
    [string]$TaskJson = "benchmarks/copilot_symbolic_next_milestone_tasks.json",
    [string]$OutJson = "benchmark_symbolic_suite_next_milestone.json"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function _Read-JsonDoc {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try {
        return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json)
    }
    catch {
        return $null
    }
}

function _Get-PropValue {
    param(
        [AllowNull()]
        [Parameter(Mandatory = $true)]
        $Object,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )
    if ($null -eq $Object) { return $null }
    $prop = $Object.PSObject.Properties[$Name]
    if ($null -eq $prop) { return $null }
    return $prop.Value
}

function _Text {
    param($Value)
    if ([string]::IsNullOrWhiteSpace([string]$Value)) { return "n/a" }
    return [string]$Value
}

function _BoolText {
    param($Value)
    if ($null -eq $Value) { return "n/a" }
    return ([bool]$Value).ToString().ToLowerInvariant()
}

function _Arm-Summary {
    param(
        [Parameter(Mandatory = $true)]
        $Task,
        [Parameter(Mandatory = $true)]
        [string]$ArmName
    )
    $arm = _Get-PropValue -Object $Task -Name $ArmName
    if ($null -eq $arm) { return $null }
    return [pscustomobject]@{
        CompileOk   = _Get-PropValue -Object $arm -Name "compile_ok"
        MetricOk    = _Get-PropValue -Object $arm -Name "metric_ok"
        BackendKind = _Text (_Get-PropValue -Object $arm -Name "backend_kind")
    }
}

function _Arm-Pass {
    param($Summary)
    if ($null -eq $Summary) { return $false }
    return ([bool]$Summary.CompileOk) -and ([bool]$Summary.MetricOk)
}

$cmd = @(
    "axiom", "copilot-benchmark",
    "--backend", $Backend,
    "--timeout", ("{0}" -f $Timeout),
    "--task-json", $TaskJson,
    "--max-iterations", ("{0}" -f $MaxIterations),
    "--temperature", ("{0}" -f $Temperature),
    "--max-tokens", ("{0}" -f $MaxTokens),
    "--out", $OutJson
)
if ($Backend -ne "benchmark-dispatch") {
    $cmd += @("--expert-url", $ExpertUrl, "--expert-model", $ExpertModel)
    if (-not [string]::IsNullOrWhiteSpace($ExpertApiKey)) {
        $cmd += @("--expert-api-key", $ExpertApiKey)
    }
}

Write-Host ("==> Running: {0}" -f ($cmd -join " ")) -ForegroundColor Cyan
& $cmd[0] $cmd[1..($cmd.Length - 1)]
if ($LASTEXITCODE -ne 0) {
    throw "copilot-benchmark comparison run failed with exit code $LASTEXITCODE."
}

$doc = _Read-JsonDoc -Path $OutJson
if ($null -eq $doc) {
    throw "Failed to read or parse benchmark output JSON at '$OutJson'."
}

$runOptions = _Get-PropValue -Object $doc -Name "run_options"
$draftEnabled = _Get-PropValue -Object $runOptions -Name "draft"
$searchEnabled = _Get-PropValue -Object $runOptions -Name "search"
if (-not [bool]$draftEnabled -or -not [bool]$searchEnabled) {
    throw "Expected benchmark output with both draft and search arms in '$OutJson'."
}

$tasks = _Get-PropValue -Object $doc -Name "tasks"
if ($null -eq $tasks -or @($tasks).Count -eq 0) {
    throw "Benchmark output contained no tasks."
}

$regressionCount = 0
foreach ($task in @($tasks)) {
    $taskId = _Text (_Get-PropValue -Object $task -Name "task_id")
    $draft = _Arm-Summary -Task $task -ArmName "draft_only"
    $search = _Arm-Summary -Task $task -ArmName "search"
    if ($null -eq $draft -or $null -eq $search) {
        throw "Benchmark output task '$taskId' is missing draft or search results."
    }
    $draftPass = _Arm-Pass -Summary $draft
    $searchPass = _Arm-Pass -Summary $search
    $regressed = $draftPass -and (-not $searchPass)
    if ($regressed) {
        $regressionCount++
    }
    Write-Host (
        "TASK {0}: draft compile_ok={1} metric_ok={2} backend_kind={3}; search compile_ok={4} metric_ok={5} backend_kind={6}; search_regressed={7}" -f `
        $taskId,
        (_BoolText $draft.CompileOk),
        (_BoolText $draft.MetricOk),
        (_Text $draft.BackendKind),
        (_BoolText $search.CompileOk),
        (_BoolText $search.MetricOk),
        (_Text $search.BackendKind),
        (_BoolText $regressed)
    )
}

$draftSummary = _Get-PropValue -Object $doc -Name "draft_summary"
$searchSummary = _Get-PropValue -Object $doc -Name "search_summary"

Write-Host ""
Write-Host (
    "SUMMARY: task_count={0}; draft compile_ok_count={1} metric_ok_count={2}; search compile_ok_count={3} metric_ok_count={4}; regressions={5}; out={6}" -f `
    (_Text (_Get-PropValue -Object $draftSummary -Name "task_count")),
    (_Text (_Get-PropValue -Object $draftSummary -Name "compile_ok_count")),
    (_Text (_Get-PropValue -Object $draftSummary -Name "metric_ok_count")),
    (_Text (_Get-PropValue -Object $searchSummary -Name "compile_ok_count")),
    (_Text (_Get-PropValue -Object $searchSummary -Name "metric_ok_count")),
    $regressionCount,
    $OutJson
)

if ($regressionCount -gt 0) {
    throw "Search regressed relative to draft on $regressionCount task(s)."
}
