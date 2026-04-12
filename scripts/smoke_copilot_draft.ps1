param(
    [string]$Backend = "onyx-qwen",
    [string]$ExpertUrl = "http://127.0.0.1:8000",
    [string]$ExpertModel = "onyx-qwen-production-v1",
    [string]$ExpertApiKey = "sk-morph-b2b-test",
    [string]$TaskJson = "benchmarks/copilot_symbolic_and_generalization_tasks.json",
    [string]$OutJson = "benchmark_symbolic_snapshot.json",
    [string]$CompareJson = ""
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
        [Parameter(Mandatory = $true)]
        $Object,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )
    if ($null -eq $Object) { return $null }
    $p = $Object.PSObject.Properties[$Name]
    if ($null -eq $p) { return $null }
    return $p.Value
}

function _Get-NegMse {
    param($Metrics)
    if ($null -eq $Metrics) { return $null }
    $prop = $Metrics.PSObject.Properties["neg_mse"]
    if ($null -eq $prop) { return $null }
    try { return [double]$prop.Value } catch { return $null }
}

function _BoolText {
    param($Value)
    if ($null -eq $Value) { return "n/a" }
    return ([bool]$Value).ToString().ToLowerInvariant()
}

function _Text {
    param($Value)
    if ([string]::IsNullOrWhiteSpace([string]$Value)) { return "n/a" }
    return [string]$Value
}

function _Task-Summary {
    param($Task)
    if ($null -eq $Task) { return $null }
    $draft = _Get-PropValue -Object $Task -Name "draft_only"
    if ($null -eq $draft) { return $null }
    $eval = _Get-PropValue -Object $draft -Name "evaluation"
    $metrics = _Get-PropValue -Object $eval -Name "metrics"
    return [pscustomobject]@{
        TaskId      = _Text (_Get-PropValue -Object $Task -Name "task_id")
        Title       = _Text (_Get-PropValue -Object $Task -Name "title")
        CompileOk   = _Get-PropValue -Object $draft -Name "compile_ok"
        MetricOk    = _Get-PropValue -Object $draft -Name "metric_ok"
        BackendKind = _Text (_Get-PropValue -Object $draft -Name "backend_kind")
        BackendName = _Text (_Get-PropValue -Object $draft -Name "producing_backend_name")
        NegMse      = _Get-NegMse -Metrics $metrics
    }
}

function _Index-Tasks {
    param($Doc)
    $map = @{}
    if ($null -eq $Doc) { return $map }
    $tasks = _Get-PropValue -Object $Doc -Name "tasks"
    if ($null -eq $tasks) { return $map }
    foreach ($task in @($tasks)) {
        $summary = _Task-Summary -Task $task
        if ($null -eq $summary) { continue }
        $map[$summary.TaskId] = $summary
    }
    return $map
}

function _Task-ChangeText {
    param($Before, $After)
    if ($null -eq $Before) { return "change=baseline_missing" }
    $parts = @()
    if (_BoolText $Before.CompileOk -ne _BoolText $After.CompileOk) {
        $parts += ("compile_ok:{0}->{1}" -f (_BoolText $Before.CompileOk), (_BoolText $After.CompileOk))
    }
    if (_BoolText $Before.MetricOk -ne _BoolText $After.MetricOk) {
        $parts += ("metric_ok:{0}->{1}" -f (_BoolText $Before.MetricOk), (_BoolText $After.MetricOk))
    }
    if (_Text $Before.BackendKind -ne _Text $After.BackendKind) {
        $parts += ("backend_kind:{0}->{1}" -f (_Text $Before.BackendKind), (_Text $After.BackendKind))
    }
    if ($parts.Count -eq 0) { return "change=none" }
    return "change=" + ($parts -join ",")
}

$baselinePath = if ([string]::IsNullOrWhiteSpace($CompareJson)) { $OutJson } else { $CompareJson }
$baselineDoc = _Read-JsonDoc -Path $baselinePath
$baselineTasks = _Index-Tasks -Doc $baselineDoc

$cmd = @(
    "axiom", "copilot-benchmark",
    "--backend", $Backend,
    "--expert-url", $ExpertUrl,
    "--expert-model", $ExpertModel,
    "--draft-only",
    "--task-json", $TaskJson,
    "--out", $OutJson
)
if (-not [string]::IsNullOrWhiteSpace($ExpertApiKey)) {
    $cmd += @("--expert-api-key", $ExpertApiKey)
}

Write-Host ("==> Running: {0}" -f ($cmd -join " ")) -ForegroundColor Cyan
& $cmd[0] $cmd[1..($cmd.Length - 1)]
if ($LASTEXITCODE -ne 0) {
    throw "copilot-benchmark draft-only run failed with exit code $LASTEXITCODE."
}

$doc = _Read-JsonDoc -Path $OutJson
if ($null -eq $doc) {
    throw "Failed to read or parse benchmark output JSON at '$OutJson'."
}

$runOptions = _Get-PropValue -Object $doc -Name "run_options"
$draftEnabled = _Get-PropValue -Object $runOptions -Name "draft"
$searchEnabled = _Get-PropValue -Object $runOptions -Name "search"
if (-not [bool]$draftEnabled -or [bool]$searchEnabled) {
    throw "Expected draft-only benchmark output in '$OutJson'."
}

$tasks = _Get-PropValue -Object $doc -Name "tasks"
if ($null -eq $tasks -or @($tasks).Count -eq 0) {
    throw "Benchmark output contained no tasks."
}

$changedCount = 0
foreach ($task in @($tasks)) {
    $summary = _Task-Summary -Task $task
    if ($null -eq $summary) { continue }
    $change = _Task-ChangeText -Before $baselineTasks[$summary.TaskId] -After $summary
    if ($change -ne "change=none" -and $change -ne "change=baseline_missing") {
        $changedCount++
    }
    $negMseText = if ($null -eq $summary.NegMse) { "n/a" } else { "{0}" -f $summary.NegMse }
    Write-Host (
        "TASK {0}: compile_ok={1} metric_ok={2} backend_kind={3} backend={4} neg_mse={5} {6}" -f `
        $summary.TaskId,
        (_BoolText $summary.CompileOk),
        (_BoolText $summary.MetricOk),
        (_Text $summary.BackendKind),
        (_Text $summary.BackendName),
        $negMseText,
        $change
    )
}

$draftSummary = _Get-PropValue -Object $doc -Name "draft_summary"
$taskCount = _Get-PropValue -Object $draftSummary -Name "task_count"
$compileOkCount = _Get-PropValue -Object $draftSummary -Name "compile_ok_count"
$metricOkCount = _Get-PropValue -Object $draftSummary -Name "metric_ok_count"

Write-Host ""
Write-Host (
    "DRAFT SUMMARY: task_count={0} compile_ok_count={1} metric_ok_count={2} out={3}" -f `
    (_Text $taskCount), (_Text $compileOkCount), (_Text $metricOkCount), $OutJson
)

if ($null -ne $baselineDoc) {
    $baseSummary = _Get-PropValue -Object $baselineDoc -Name "draft_summary"
    $baseCompile = _Get-PropValue -Object $baseSummary -Name "compile_ok_count"
    $baseMetric = _Get-PropValue -Object $baseSummary -Name "metric_ok_count"
    Write-Host (
        "COMPARE SUMMARY: compile_ok_count={0}->{1} metric_ok_count={2}->{3} changed_tasks={4} baseline={5}" -f `
        (_Text $baseCompile), (_Text $compileOkCount), (_Text $baseMetric), (_Text $metricOkCount), $changedCount, $baselinePath
    )
}
else {
    Write-Host ("COMPARE SUMMARY: baseline_missing path={0}" -f $baselinePath)
}
