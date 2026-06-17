$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot "_smoke_common.ps1")

$Backend = "onyx-qwen"
$ExpertUrl = "http://127.0.0.1:8000"
$ExpertModel = "onyx-qwen-production-v1"
$ExpertApiKey = Resolve-ExpertApiKey

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

$stepResults = @()

$steps = @(
    @{
        Name = "copilot-search piecewise_threshold"
        Type = "search"
        ReportPath = "debug_piecewise_threshold/search_report_cli.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-search",
            "--backend", $Backend,
            "--goal", "Write a valid Axiom .ax program in this repo's DSL that computes y = x when x > 0, otherwise y = 0.0.",
            "--examples-json", "examples/piecewise_threshold.json",
            "--iterations", "8",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--artifact-dir", "debug_piecewise_threshold",
            "--report-out", "debug_piecewise_threshold/search_report_cli.json",
            "--out", "debug_piecewise_threshold/best.ax"
        )
    },
    @{
        Name = "copilot-run piecewise_threshold"
        Type = "run"
        ReportPath = "showcase_piecewise_threshold/pipeline_summary.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-run",
            "--backend", $Backend,
            "--goal", "Write a valid Axiom .ax program in this repo's DSL that computes y = x when x > 0, otherwise y = 0.0.",
            "--examples-json", "examples/piecewise_threshold.json",
            "--iterations", "8",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--artifact-dir", "showcase_piecewise_threshold",
            "--summary-out", "showcase_piecewise_threshold/pipeline_summary.json",
            "--out", "showcase_piecewise_threshold.ax"
        )
    },
    @{
        Name = "copilot-search three_input_affine"
        Type = "search"
        ReportPath = "debug_three_input_affine/search_report_cli.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-search",
            "--backend", $Backend,
            "--goal", "Write a valid Axiom .ax program in this repo's DSL that computes score = 0.5 * a + 0.3 * b + 0.2 * c.",
            "--examples-json", "examples/three_input_affine.json",
            "--iterations", "8",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--artifact-dir", "debug_three_input_affine",
            "--report-out", "debug_three_input_affine/search_report_cli.json",
            "--out", "debug_three_input_affine/best.ax"
        )
    },
    @{
        Name = "copilot-run three_input_affine"
        Type = "run"
        ReportPath = "showcase_three_input_affine/pipeline_summary.json"
        NegMseMin = $NegMseMinDefault
        Command = @(
            "axiom", "copilot-run",
            "--backend", $Backend,
            "--goal", "Write a valid Axiom .ax program in this repo's DSL that computes score = 0.5 * a + 0.3 * b + 0.2 * c.",
            "--examples-json", "examples/three_input_affine.json",
            "--iterations", "8",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--artifact-dir", "showcase_three_input_affine",
            "--summary-out", "showcase_three_input_affine/pipeline_summary.json",
            "--out", "showcase_three_input_affine.ax"
        )
    }
)

$processPassCount = 0
$qualityPassCount = 0
$failedStep = $null

try {
    foreach ($step in $steps) {
        $cmd = [System.Collections.Generic.List[string]]::new()
        $cmd.AddRange([string[]]$step.Command)
        Append-ExpertApiKeyArgs -Command $cmd -ExpertApiKey $ExpertApiKey
        Write-Host ("==> Running: {0} ({1})" -f $step.Name, (Format-RedactedCommand -Command $cmd.ToArray())) -ForegroundColor Cyan
        & $cmd[0] $cmd[1..($cmd.Count - 1)]
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $($step.Name)"
        }
        $processPassCount++

        $qualityOk = $true
        $converged = $null
        $negMse = $null
        $why = @()

        if (-not (Test-Path -LiteralPath $step.ReportPath)) {
            $qualityOk = $false
            $why += "missing_report"
        }
        else {
            try {
                $doc = Get-Content -LiteralPath $step.ReportPath -Raw | ConvertFrom-Json
            }
            catch {
                $qualityOk = $false
                $why += "invalid_json"
                $doc = $null
            }

            if ($null -ne $doc) {
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
                    if (($null -ne $converged) -and (-not [bool]$converged)) {
                        $qualityOk = $false
                        $why += "converged=false"
                    }

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

                if ($null -ne $negMse -and ($negMse -lt [double]$step.NegMseMin)) {
                    $qualityOk = $false
                    $why += ("neg_mse<{0}" -f $step.NegMseMin)
                }
            }
        }

        if ($qualityOk) {
            $qualityPassCount++
        }

        $status = if ($qualityOk) { "PASS" } else { "FAIL" }
        $convText = if ($null -eq $converged) { "n/a" } else { "$converged" }
        $metricText = if ($null -eq $negMse) { "n/a" } else { "{0}" -f $negMse }
        $extra = if ($why.Count -gt 0) { " reason=" + ($why -join ",") } else { "" }
        Write-Host ("STEP {0}: {1} converged={2} neg_mse={3}{4}" -f $step.Name, $status, $convText, $metricText, $extra)

        $stepResults += @{
            Name = $step.Name
            Status = $status
            Converged = $convText
            NegMse = $metricText
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
