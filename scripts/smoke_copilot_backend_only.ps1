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

$stepResults = @()

$steps = @(
    @{
        Name = "copilot-search quadratic_single_input"
        Type = "search"
        ReportPath = "debug_backend_only_quadratic/search_report_cli.json"
        IterationsPath = "debug_backend_only_quadratic/iterations.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-search",
            "--backend", $Backend,
            "--goal", "Write a valid Axiom .ax program in this repo's DSL that computes y = x * x + 1.0;",
            "--examples-json", "examples/quadratic_single_input.json",
            "--iterations", "10",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--expert-api-key", $ExpertApiKey,
            "--artifact-dir", "debug_backend_only_quadratic",
            "--report-out", "debug_backend_only_quadratic/search_report_cli.json",
            "--out", "debug_backend_only_quadratic/best.ax"
        )
    },
    @{
        Name = "copilot-run quadratic_single_input"
        Type = "run"
        ReportPath = "showcase_backend_only_quadratic/pipeline_summary.json"
        IterationsPath = "showcase_backend_only_quadratic/iterations.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-run",
            "--backend", $Backend,
            "--goal", "Write a valid Axiom .ax program in this repo's DSL that computes y = x * x + 1.0;",
            "--examples-json", "examples/quadratic_single_input.json",
            "--iterations", "10",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--expert-api-key", $ExpertApiKey,
            "--artifact-dir", "showcase_backend_only_quadratic",
            "--summary-out", "showcase_backend_only_quadratic/pipeline_summary.json",
            "--out", "showcase_backend_only_quadratic.ax"
        )
    },
    @{
        Name = "copilot-search max_of_two"
        Type = "search"
        ReportPath = "debug_backend_only_max_of_two/search_report_cli.json"
        IterationsPath = "debug_backend_only_max_of_two/iterations.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-search",
            "--backend", $Backend,
            "--goal", "Write a valid Axiom .ax program in this repo's DSL that computes score = max(a, b);",
            "--examples-json", "examples/max_of_two.json",
            "--iterations", "10",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--expert-api-key", $ExpertApiKey,
            "--artifact-dir", "debug_backend_only_max_of_two",
            "--report-out", "debug_backend_only_max_of_two/search_report_cli.json",
            "--out", "debug_backend_only_max_of_two/best.ax"
        )
    },
    @{
        Name = "copilot-run max_of_two"
        Type = "run"
        ReportPath = "showcase_backend_only_max_of_two/pipeline_summary.json"
        IterationsPath = "showcase_backend_only_max_of_two/iterations.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-run",
            "--backend", $Backend,
            "--goal", "Write a valid Axiom .ax program in this repo's DSL that computes score = max(a, b);",
            "--examples-json", "examples/max_of_two.json",
            "--iterations", "10",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--expert-api-key", $ExpertApiKey,
            "--artifact-dir", "showcase_backend_only_max_of_two",
            "--summary-out", "showcase_backend_only_max_of_two/pipeline_summary.json",
            "--out", "showcase_backend_only_max_of_two.ax"
        )
    },
    @{
        Name = "copilot-search minmax_blend"
        Type = "search"
        ReportPath = "debug_backend_only_minmax_blend/search_report_cli.json"
        IterationsPath = "debug_backend_only_minmax_blend/iterations.json"
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
            "--artifact-dir", "debug_backend_only_minmax_blend",
            "--report-out", "debug_backend_only_minmax_blend/search_report_cli.json",
            "--out", "debug_backend_only_minmax_blend/best.ax"
        )
    },
    @{
        Name = "copilot-run minmax_blend"
        Type = "run"
        ReportPath = "showcase_backend_only_minmax_blend/pipeline_summary.json"
        IterationsPath = "showcase_backend_only_minmax_blend/iterations.json"
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
            "--artifact-dir", "showcase_backend_only_minmax_blend",
            "--summary-out", "showcase_backend_only_minmax_blend/pipeline_summary.json",
            "--out", "showcase_backend_only_minmax_blend.ax"
        )
    },
    @{
        Name = "copilot-search quadratic_with_cross_term"
        Type = "search"
        ReportPath = "debug_backend_only_quadratic_with_cross_term/search_report_cli.json"
        IterationsPath = "debug_backend_only_quadratic_with_cross_term/iterations.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-search",
            "--backend", $Backend,
            "--goal", "Write a valid Axiom .ax program in this repo's DSL that computes y = a * b + a + 1.0;",
            "--examples-json", "examples/quadratic_with_cross_term.json",
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
            "--goal", "Write a valid Axiom .ax program in this repo's DSL that computes y = a * b + a + 1.0;",
            "--examples-json", "examples/quadratic_with_cross_term.json",
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
            "--goal", "Write a valid Axiom .ax program in this repo's DSL so if x < 0 then y = 0.0 else if x < 1 then y = x else y = 1.0;",
            "--examples-json", "examples/nested_piecewise.json",
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
            "--goal", "Write a valid Axiom .ax program in this repo's DSL so if x < 0 then y = 0.0 else if x < 1 then y = x else y = 1.0;",
            "--examples-json", "examples/nested_piecewise.json",
            "--iterations", "10",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--expert-api-key", $ExpertApiKey,
            "--artifact-dir", "showcase_backend_only_nested_piecewise",
            "--summary-out", "showcase_backend_only_nested_piecewise/pipeline_summary.json",
            "--out", "showcase_backend_only_nested_piecewise.ax"
        )
    },
    @{
        Name = "copilot-search three_way_maxmin"
        Type = "search"
        ReportPath = "debug_backend_only_three_way_maxmin/search_report_cli.json"
        IterationsPath = "debug_backend_only_three_way_maxmin/iterations.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-search",
            "--backend", $Backend,
            "--goal", "Write a valid Axiom .ax program in this repo's DSL that computes score = max(min(a, b), c);",
            "--examples-json", "examples/three_way_maxmin.json",
            "--iterations", "10",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--expert-api-key", $ExpertApiKey,
            "--artifact-dir", "debug_backend_only_three_way_maxmin",
            "--report-out", "debug_backend_only_three_way_maxmin/search_report_cli.json",
            "--out", "debug_backend_only_three_way_maxmin/best.ax"
        )
    },
    @{
        Name = "copilot-run three_way_maxmin"
        Type = "run"
        ReportPath = "showcase_backend_only_three_way_maxmin/pipeline_summary.json"
        IterationsPath = "showcase_backend_only_three_way_maxmin/iterations.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-run",
            "--backend", $Backend,
            "--goal", "Write a valid Axiom .ax program in this repo's DSL that computes score = max(min(a, b), c);",
            "--examples-json", "examples/three_way_maxmin.json",
            "--iterations", "10",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--expert-api-key", $ExpertApiKey,
            "--artifact-dir", "showcase_backend_only_three_way_maxmin",
            "--summary-out", "showcase_backend_only_three_way_maxmin/pipeline_summary.json",
            "--out", "showcase_backend_only_three_way_maxmin.ax"
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
        Write-Host ("STEP {0}: {1} converged={2} neg_mse={3} backend={4} backend_kind={5}{6}" -f $step.Name, $status, $convText, $metricText, $backendName, $backendKind, $extra)

        $stepResults += @{
            Name = $step.Name
            Status = $status
            Converged = $convText
            NegMse = $metricText
            Backend = $backendName
            BackendKind = $backendKind
            Reason = ($why -join ",")
        }
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
