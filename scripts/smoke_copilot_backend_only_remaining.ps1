$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Backend = "onyx-qwen"
$ExpertUrl = "http://127.0.0.1:8000"
$ExpertModel = "onyx-qwen-production-v1"
$ExpertApiKey = "sk-morph-b2b-test"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$NegMseMinDefault = -1e-9

function _Get-NegMse {
    param([Parameter(Mandatory = $true)]$Metrics)
    if ($null -eq $Metrics) { return $null }
    if ($Metrics -is [System.Collections.IDictionary]) {
        if ($Metrics.Contains("neg_mse")) {
            try { return [double]$Metrics["neg_mse"] } catch { return $null }
        }
        return $null
    }
    if ($Metrics -is [System.Array] -or $Metrics -is [System.Collections.IEnumerable] -and -not ($Metrics -is [string])) {
        foreach ($item in $Metrics) {
            if ($null -eq $item) { continue }
            $nameValue = $item.PSObject.Properties["name"]
            if ($null -eq $nameValue) { $nameValue = $item.PSObject.Properties["Name"] }
            $valueValue = $item.PSObject.Properties["value"]
            if ($null -eq $valueValue) { $valueValue = $item.PSObject.Properties["Value"] }
            if ($null -eq $nameValue -or $null -eq $valueValue) { continue }
            if ([string]$nameValue.Value -eq "neg_mse") {
                try { return [double]$valueValue.Value } catch { return $null }
            }
        }
        return $null
    }
    $prop = $Metrics.PSObject.Properties["neg_mse"]
    if ($null -ne $prop) {
        try { return [double]$prop.Value } catch { return $null }
    }
    return $null
}

function _Get-PropValue {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string]$Name
    )
    if ($null -eq $Object) { return $null }
    if ($Object -is [System.Collections.IDictionary]) {
        if ($Object.Contains($Name)) { return $Object[$Name] }
        return $null
    }
    $p = $Object.PSObject.Properties[$Name]
    if ($null -eq $p) { return $null }
    return $p.Value
}

function _Get-RunReportFields {
    param([Parameter(Mandatory = $true)]$Doc)
    $fields = @{
        Converged = _Get-PropValue -Object $Doc -Name "converged"
        ConvergenceReason = _Get-PropValue -Object $Doc -Name "convergence_reason"
        BestEvaluation = _Get-PropValue -Object $Doc -Name "best_evaluation"
        FinalEvaluation = _Get-PropValue -Object $Doc -Name "final_evaluation"
        FinalValidation = _Get-PropValue -Object $Doc -Name "final_validation"
    }
    $fields["Readable"] = (
        ($null -ne $fields["Converged"]) -or
        ($null -ne $fields["ConvergenceReason"]) -or
        ($null -ne $fields["BestEvaluation"]) -or
        ($null -ne $fields["FinalEvaluation"]) -or
        ($null -ne $fields["FinalValidation"])
    )
    return $fields
}

function _Read-JsonDoc {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try { return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json) } catch { return $null }
}

function _BackendKind {
    param([string]$BackendName)
    if ([string]::IsNullOrWhiteSpace($BackendName)) { return "unknown" }
    if ($BackendName -like "*fast_path*") { return "deterministic_fast_path" }
    return "expert_backend"
}

$steps = @(
    @{
        Name = "copilot-search quadratic_with_cross_term"
        Type = "search"
        ReportPath = "debug_backend_only_quadratic_with_cross_term/search_report_cli.json"
        IterationsPath = "debug_backend_only_quadratic_with_cross_term/iterations.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-search",
            "--backend", $Backend,
            "--goal", "Write .ax so y = a * b + a + 1.0.",
            "--examples-json", "benchmarks/fixtures/backend_only_harder/quadratic_with_cross_term.json",
            "--iterations", "10",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--expert-api-key", $ExpertApiKey,
            "--artifact-dir", "debug_backend_only_quadratic_with_cross_term",
            "--report-out", "debug_backend_only_quadratic_with_cross_term/search_report_cli.json",
            "--out", "debug_backend_only_quadratic_with_cross_term/best.ax"
        )
    },
    @{
        Name = "copilot-run quadratic_with_cross_term"
        Type = "run"
        ReportPath = "showcase_backend_only_quadratic_with_cross_term/pipeline_summary.json"
        IterationsPath = "showcase_backend_only_quadratic_with_cross_term/iterations.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-run",
            "--backend", $Backend,
            "--goal", "Write .ax so y = a * b + a + 1.0.",
            "--examples-json", "benchmarks/fixtures/backend_only_harder/quadratic_with_cross_term.json",
            "--iterations", "10",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--expert-api-key", $ExpertApiKey,
            "--artifact-dir", "showcase_backend_only_quadratic_with_cross_term",
            "--summary-out", "showcase_backend_only_quadratic_with_cross_term/pipeline_summary.json",
            "--out", "showcase_backend_only_quadratic_with_cross_term.ax"
        )
    },
    @{
        Name = "copilot-search nested_piecewise"
        Type = "search"
        ReportPath = "debug_backend_only_nested_piecewise/search_report_cli.json"
        IterationsPath = "debug_backend_only_nested_piecewise/iterations.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-search",
            "--backend", $Backend,
            "--goal", "Write .ax so if x < 0 then y = 0.0 else if x < 1 then y = x else y = 1.0.",
            "--examples-json", "benchmarks/fixtures/backend_only_harder/nested_piecewise.json",
            "--iterations", "10",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--expert-api-key", $ExpertApiKey,
            "--artifact-dir", "debug_backend_only_nested_piecewise",
            "--report-out", "debug_backend_only_nested_piecewise/search_report_cli.json",
            "--out", "debug_backend_only_nested_piecewise/best.ax"
        )
    },
    @{
        Name = "copilot-run nested_piecewise"
        Type = "run"
        ReportPath = "showcase_backend_only_nested_piecewise/pipeline_summary.json"
        IterationsPath = "showcase_backend_only_nested_piecewise/iterations.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-run",
            "--backend", $Backend,
            "--goal", "Write .ax so if x < 0 then y = 0.0 else if x < 1 then y = x else y = 1.0.",
            "--examples-json", "benchmarks/fixtures/backend_only_harder/nested_piecewise.json",
            "--iterations", "10",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--expert-api-key", $ExpertApiKey,
            "--artifact-dir", "showcase_backend_only_nested_piecewise",
            "--summary-out", "showcase_backend_only_nested_piecewise/pipeline_summary.json",
            "--out", "showcase_backend_only_nested_piecewise.ax"
        )
    }
)

$processPassCount = 0
$processFailCount = 0
$qualityPassCount = 0

foreach ($step in $steps) {
    Write-Host "==> Running: $($step.Name)" -ForegroundColor Cyan
    & $step.Command[0] $step.Command[1..($step.Command.Length - 1)]
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) { $processPassCount++ } else { $processFailCount++ }

    $qualityOk = $true
    $converged = $null
    $convergenceReason = $null
    $negMse = $null
    $backendName = "unknown"
    $why = @()
    if ($exitCode -ne 0) { $why += ("process_exit={0}" -f $exitCode) }

    $reportExists = Test-Path -LiteralPath $step.ReportPath
    $doc = _Read-JsonDoc -Path $step.ReportPath
    if ($null -eq $doc) {
        $qualityOk = $false
        if ($reportExists) {
            $why += "invalid_report_json"
        }
        else {
            $why += "missing_or_invalid_report"
        }
    }
    else {
        if ($step.Type -eq "search") {
            $finalReport = _Get-PropValue -Object $doc -Name "final_report"
            $finalSuccess = _Get-PropValue -Object $finalReport -Name "success"
            if (($null -ne $finalSuccess) -and (-not [bool]$finalSuccess)) {
                $qualityOk = $false
                $why += "final_report.success=false"
            }
            $converged = _Get-PropValue -Object $doc -Name "converged"
            $convergenceReason = _Get-PropValue -Object $doc -Name "convergence_reason"
            $finalMetrics = _Get-PropValue -Object $finalReport -Name "metrics"
            $negMse = _Get-NegMse -Metrics $finalMetrics
        }
        elseif ($step.Type -eq "run") {
            $runReport = _Get-RunReportFields -Doc $doc
            if (-not [bool]$runReport["Readable"]) {
                $qualityOk = $false
                $why += "invalid_pipeline_summary_shape"
            }
            $converged = $runReport["Converged"]
            $convergenceReason = $runReport["ConvergenceReason"]
            $finalValidation = $runReport["FinalValidation"]
            $bestEvaluation = $runReport["BestEvaluation"]
            $finalEvaluation = $runReport["FinalEvaluation"]
            $finalValidationOk = _Get-PropValue -Object $finalValidation -Name "success"
            if (($null -ne $finalValidationOk) -and (-not [bool]$finalValidationOk)) {
                $qualityOk = $false
                $why += "final_validation.success=false"
            }
            $finalEvalMetrics = _Get-PropValue -Object $finalEvaluation -Name "metrics"
            $bestEvalMetrics = _Get-PropValue -Object $bestEvaluation -Name "metrics"
            $negMse = _Get-NegMse -Metrics $finalEvalMetrics
            if ($null -eq $negMse) { $negMse = _Get-NegMse -Metrics $bestEvalMetrics }
        }

        if (($null -ne $converged) -and (-not [bool]$converged)) {
            $qualityOk = $false
            $why += "converged=false"
        }
        if ($null -eq $negMse) {
            $qualityOk = $false
            $why += "neg_mse_missing"
        }
        elseif ($negMse -lt [double]$step.NegMseMin) {
            $qualityOk = $false
            $why += ("neg_mse<{0}" -f $step.NegMseMin)
        }
    }

    $iterDoc = _Read-JsonDoc -Path $step.IterationsPath
    if ($null -ne $iterDoc) {
        $iterations = _Get-PropValue -Object $iterDoc -Name "iterations"
        if ($iterations -is [System.Array] -and $iterations.Length -gt 0) {
            $first = $iterations[0]
            $producing = _Get-PropValue -Object $first -Name "producing_expert"
            $bn = _Get-PropValue -Object $producing -Name "backend_name"
            if (-not [string]::IsNullOrWhiteSpace([string]$bn)) { $backendName = [string]$bn }
        }
    }
    $backendKind = _BackendKind -BackendName $backendName
    if ($backendKind -ne "expert_backend") {
        $qualityOk = $false
        $why += "non_expert_backend_used"
    }

    if ($qualityOk) { $qualityPassCount++ }

    $status = if ($qualityOk) { "PASS" } else { "FAIL" }
    $convText = if ($null -eq $converged) { "n/a" } else { "$converged" }
    $convReasonText = if ([string]::IsNullOrWhiteSpace([string]$convergenceReason)) { "n/a" } else { [string]$convergenceReason }
    $metricText = if ($null -eq $negMse) { "n/a" } else { "{0}" -f $negMse }
    $extra = if ($why.Count -gt 0) { " reason=" + ($why -join ",") } else { "" }
    Write-Host ("STEP {0}: {1} converged={2} convergence_reason={3} neg_mse={4} backend_kind={5}{6}" -f $step.Name, $status, $convText, $convReasonText, $metricText, $backendKind, $extra)
}

Write-Host ""
if ($qualityPassCount -eq $steps.Count) {
    Write-Host "SMOKE SUMMARY: PASS ($qualityPassCount/$($steps.Count) quality checks passed; process_ok=$processPassCount process_fail=$processFailCount)" -ForegroundColor Green
}
else {
    Write-Host "SMOKE SUMMARY: FAIL ($qualityPassCount/$($steps.Count) quality checks passed; process_ok=$processPassCount process_fail=$processFailCount)" -ForegroundColor Red
    exit 1
}
