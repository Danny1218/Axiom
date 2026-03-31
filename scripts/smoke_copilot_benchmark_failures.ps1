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
    param(
        [Parameter(Mandatory = $true)]
        $Metrics
    )
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

function _BackendKind {
    param([string]$BackendName)
    if ([string]::IsNullOrWhiteSpace($BackendName)) { return "unknown" }
    if ($BackendName -like "*fast_path*") { return "deterministic_fast_path" }
    return "expert_backend"
}

$steps = @(
    @{
        Name = "copilot-search three_input_affine"
        Type = "search"
        ReportPath = "debug_benchmark_failures_three_input_affine/search_report_cli.json"
        IterationsPath = "debug_benchmark_failures_three_input_affine/iterations.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-search",
            "--backend", $Backend,
            "--goal", "Write a valid Axiom .ax program in this repo's DSL that computes score = 0.5 * a + 0.3 * b + 0.2 * c.",
            "--examples-json", "examples/three_input_affine.json",
            "--iterations", "10",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--expert-api-key", $ExpertApiKey,
            "--artifact-dir", "debug_benchmark_failures_three_input_affine",
            "--report-out", "debug_benchmark_failures_three_input_affine/search_report_cli.json",
            "--out", "debug_benchmark_failures_three_input_affine/best.ax"
        )
    },
    @{
        Name = "copilot-run three_input_affine"
        Type = "run"
        ReportPath = "showcase_benchmark_failures_three_input_affine/pipeline_summary.json"
        IterationsPath = "showcase_benchmark_failures_three_input_affine/iterations.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-run",
            "--backend", $Backend,
            "--goal", "Write a valid Axiom .ax program in this repo's DSL that computes score = 0.5 * a + 0.3 * b + 0.2 * c.",
            "--examples-json", "examples/three_input_affine.json",
            "--iterations", "10",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--expert-api-key", $ExpertApiKey,
            "--artifact-dir", "showcase_benchmark_failures_three_input_affine",
            "--summary-out", "showcase_benchmark_failures_three_input_affine/pipeline_summary.json",
            "--out", "showcase_benchmark_failures_three_input_affine.ax"
        )
    },
    @{
        Name = "copilot-search minmax_blend"
        Type = "search"
        ReportPath = "debug_benchmark_failures_minmax_blend/search_report_cli.json"
        IterationsPath = "debug_benchmark_failures_minmax_blend/iterations.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-search",
            "--backend", $Backend,
            "--goal", "Write a valid Axiom .ax program in this repo's DSL that computes score = max(0.0, min(a + b, 1.0));",
            "--examples-json", "examples/minmax_blend.json",
            "--iterations", "10",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--expert-api-key", $ExpertApiKey,
            "--artifact-dir", "debug_benchmark_failures_minmax_blend",
            "--report-out", "debug_benchmark_failures_minmax_blend/search_report_cli.json",
            "--out", "debug_benchmark_failures_minmax_blend/best.ax"
        )
    },
    @{
        Name = "copilot-run minmax_blend"
        Type = "run"
        ReportPath = "showcase_benchmark_failures_minmax_blend/pipeline_summary.json"
        IterationsPath = "showcase_benchmark_failures_minmax_blend/iterations.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-run",
            "--backend", $Backend,
            "--goal", "Write a valid Axiom .ax program in this repo's DSL that computes score = max(0.0, min(a + b, 1.0));",
            "--examples-json", "examples/minmax_blend.json",
            "--iterations", "10",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--expert-api-key", $ExpertApiKey,
            "--artifact-dir", "showcase_benchmark_failures_minmax_blend",
            "--summary-out", "showcase_benchmark_failures_minmax_blend/pipeline_summary.json",
            "--out", "showcase_benchmark_failures_minmax_blend.ax"
        )
    }
)

$processPassCount = 0
$qualityPassCount = 0
$failedStep = $null

try {
    foreach ($step in $steps) {
        Write-Host "==> Running: $($step.Name)" -ForegroundColor Cyan
        & $step.Command[0] $step.Command[1..($step.Command.Length - 1)]
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $($step.Name)"
        }
        $processPassCount++

        $qualityOk = $true
        $converged = $null
        $negMse = $null
        $backendName = "unknown"
        $backendKind = "unknown"
        $why = @()

        $doc = _Read-JsonDoc -Path $step.ReportPath
        if ($null -eq $doc) {
            $qualityOk = $false
            $why += "missing_or_invalid_report"
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
                $finalMetrics = _Get-PropValue -Object $finalReport -Name "metrics"
                $negMse = _Get-NegMse -Metrics $finalMetrics
            }
            elseif ($step.Type -eq "run") {
                $converged = _Get-PropValue -Object $doc -Name "converged"
                $finalValidation = _Get-PropValue -Object $doc -Name "final_validation"
                $finalValidationOk = _Get-PropValue -Object $finalValidation -Name "success"
                if (($null -ne $finalValidationOk) -and (-not [bool]$finalValidationOk)) {
                    $qualityOk = $false
                    $why += "final_validation.success=false"
                }
                $finalEvaluation = _Get-PropValue -Object $doc -Name "final_evaluation"
                $bestEvaluation = _Get-PropValue -Object $doc -Name "best_evaluation"
                $finalEvalMetrics = _Get-PropValue -Object $finalEvaluation -Name "metrics"
                $bestEvalMetrics = _Get-PropValue -Object $bestEvaluation -Name "metrics"
                $negMse = _Get-NegMse -Metrics $finalEvalMetrics
                if ($null -eq $negMse) {
                    $negMse = _Get-NegMse -Metrics $bestEvalMetrics
                }
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
                if (-not [string]::IsNullOrWhiteSpace([string]$bn)) {
                    $backendName = [string]$bn
                }
            }
        }
        $backendKind = _BackendKind -BackendName $backendName
        if ($backendKind -eq "deterministic_fast_path") {
            $qualityOk = $false
            $why += "deterministic_fast_path_used"
        }

        if ($qualityOk) {
            $qualityPassCount++
        }

        $status = if ($qualityOk) { "PASS" } else { "FAIL" }
        $convText = if ($null -eq $converged) { "n/a" } else { "$converged" }
        $metricText = if ($null -eq $negMse) { "n/a" } else { "{0}" -f $negMse }
        $extra = if ($why.Count -gt 0) { " reason=" + ($why -join ",") } else { "" }
        Write-Host ("STEP {0}: {1} converged={2} neg_mse={3} backend_kind={4}{5}" -f $step.Name, $status, $convText, $metricText, $backendKind, $extra)
    }
}
catch {
    $failedStep = if ($step -and $step.Name) { $step.Name } else { "unknown step" }
    Write-Host ""
    Write-Host "SMOKE SUMMARY: FAIL (process error; $processPassCount/$($steps.Count) commands passed)" -ForegroundColor Red
    Write-Host "Failed step: $failedStep" -ForegroundColor Red
    throw
}

Write-Host ""
if ($qualityPassCount -eq $steps.Count) {
    Write-Host "SMOKE SUMMARY: PASS ($qualityPassCount/$($steps.Count) quality checks passed)" -ForegroundColor Green
}
else {
    Write-Host "SMOKE SUMMARY: FAIL ($qualityPassCount/$($steps.Count) quality checks passed)" -ForegroundColor Red
    exit 1
}
