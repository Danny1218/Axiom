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
    [string]$OutJson = "benchmark_symbolic_suite_next_milestone.json",
    [string]$TaskId = "",
    [string]$RequestCaptureDir = "",
    [string]$ConfigJson = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot "_smoke_common.ps1")

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

function _Print-LatestFailureMetadata {
    param([string]$CaptureDir)
    if ([string]::IsNullOrWhiteSpace($CaptureDir) -or -not (Test-Path -LiteralPath $CaptureDir)) { return }
    $latest = Get-ChildItem -LiteralPath $CaptureDir -Filter *.json -File | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    if ($null -eq $latest) { return }
    $doc = _Read-JsonDoc -Path $latest.FullName
    if ($null -eq $doc) { return }
    Write-Host (
        "FAILURE METADATA: task_id={0}; failure_kind={1}; exception_class={2}; exception_message={3}; payload_sha256={4}; request_capture_path={5}" -f `
        (_Text (_Get-PropValue -Object $doc -Name "benchmark_task_id")),
        (_Text (_Get-PropValue -Object $doc -Name "failure_kind")),
        (_Text (_Get-PropValue -Object $doc -Name "exception_class")),
        (_Text (_Get-PropValue -Object $doc -Name "exception_message")),
        (_Text (_Get-PropValue -Object $doc -Name "payload_sha256")),
        $latest.FullName
    ) -ForegroundColor Yellow
}

if (-not [string]::IsNullOrWhiteSpace($ConfigJson)) {
    $cfgPath = $ConfigJson
    if (-not (Test-Path -LiteralPath $cfgPath)) {
        throw "ConfigJson not found: '$cfgPath'."
    }
    $cfgDoc = _Read-JsonDoc -Path $cfgPath
    if ($null -eq $cfgDoc) {
        throw "Failed to read or parse ConfigJson at '$cfgPath'."
    }
    $cfgBlock = _Get-PropValue -Object $cfgDoc -Name "config"
    if ($null -eq $cfgBlock) {
        throw "ConfigJson missing top-level 'config' object: '$cfgPath'."
    }
    $tFrom = _Get-PropValue -Object $cfgBlock -Name "timeout"
    $mtFrom = _Get-PropValue -Object $cfgBlock -Name "max_tokens"
    if (-not $PSBoundParameters.ContainsKey("Timeout") -and $null -ne $tFrom) {
        $Timeout = [double]$tFrom
    }
    if (-not $PSBoundParameters.ContainsKey("MaxTokens") -and $null -ne $mtFrom) {
        $MaxTokens = [int]$mtFrom
    }
    Write-Host (
        "CONFIG_JSON: path={0} timeout={1} max_tokens={2} (explicit -Timeout/-MaxTokens override file when provided)" -f `
        $cfgPath, $Timeout, $MaxTokens
    ) -ForegroundColor DarkGray
}

$taskJsonPath = $TaskJson
$tempTaskJsonPath = $null
if (-not [string]::IsNullOrWhiteSpace($TaskId)) {
    $taskDoc = _Read-JsonDoc -Path $TaskJson
    if ($null -eq $taskDoc) {
        throw "Failed to read or parse benchmark task JSON at '$TaskJson'."
    }
    $taskList = @(_Get-PropValue -Object $taskDoc -Name "tasks")
    $filtered = @($taskList | Where-Object { (_Text (_Get-PropValue -Object $_ -Name "id")) -eq $TaskId })
    if ($filtered.Count -ne 1) {
        throw "Task filter '$TaskId' matched $($filtered.Count) tasks in '$TaskJson'."
    }
    $tempTaskJsonPath = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), ".json")
    $filteredJson = [pscustomobject]@{
        schema_version = _Get-PropValue -Object $taskDoc -Name "schema_version"
        tasks = $filtered
    } | ConvertTo-Json -Depth 32
    [System.IO.File]::WriteAllText($tempTaskJsonPath, $filteredJson, [System.Text.UTF8Encoding]::new($false))
    $taskJsonPath = $tempTaskJsonPath
}

$captureDir = if (-not [string]::IsNullOrWhiteSpace($RequestCaptureDir)) { $RequestCaptureDir } else { $env:AXIOM_ONYX_REQUEST_CAPTURE_DIR }
$priorCaptureDir = [Environment]::GetEnvironmentVariable("AXIOM_ONYX_REQUEST_CAPTURE_DIR", "Process")
if (-not [string]::IsNullOrWhiteSpace($RequestCaptureDir)) {
    $env:AXIOM_ONYX_REQUEST_CAPTURE_DIR = $RequestCaptureDir
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
    "--task-json", $taskJsonPath,
    "--max-iterations", ("{0}" -f $MaxIterations),
    "--temperature", ("{0}" -f $Temperature),
    "--max-tokens", ("{0}" -f $MaxTokens),
    "--out", $OutJson
)
if ($Backend -eq "benchmark-dispatch") {
    $cmd += "--gate"
}
if ($Backend -ne "benchmark-dispatch") {
    $ExpertApiKey = Resolve-ExpertApiKey -Explicit $ExpertApiKey
    $cmd += @("--expert-url", $ExpertUrl, "--expert-model", $ExpertModel)
    $cmdList = [System.Collections.Generic.List[string]]::new()
    $cmdList.AddRange([string[]]$cmd)
    Append-ExpertApiKeyArgs -Command $cmdList -ExpertApiKey $ExpertApiKey
    $cmd = $cmdList.ToArray()
}

Write-Host ("==> Running: {0}" -f (Format-RedactedCommand -Command $cmd)) -ForegroundColor Cyan
try {
    & $cmd[0] $cmd[1..($cmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        _Print-LatestFailureMetadata -CaptureDir $captureDir
        throw "copilot-benchmark comparison run failed with exit code $LASTEXITCODE."
    }
}
finally {
    if ($null -ne $tempTaskJsonPath -and (Test-Path -LiteralPath $tempTaskJsonPath)) {
        Remove-Item -LiteralPath $tempTaskJsonPath -Force -ErrorAction SilentlyContinue
    }
    if (-not [string]::IsNullOrWhiteSpace($RequestCaptureDir)) {
        if ([string]::IsNullOrWhiteSpace($priorCaptureDir)) {
            Remove-Item Env:AXIOM_ONYX_REQUEST_CAPTURE_DIR -ErrorAction SilentlyContinue
        }
        else {
            $env:AXIOM_ONYX_REQUEST_CAPTURE_DIR = $priorCaptureDir
        }
    }
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
