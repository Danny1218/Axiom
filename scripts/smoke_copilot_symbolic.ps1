$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot "_smoke_common.ps1")

$Backend = "onyx-qwen"
$ExpertUrl = "http://127.0.0.1:8000"
$ExpertModel = "onyx-qwen-production-v1"
$ExpertApiKey = Resolve-ExpertApiKey

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$steps = @(
    @{
        Name = "copilot-search double_x"
        Command = @(
            "axiom", "copilot-search",
            "--backend", $Backend,
            "--goal", "Compute y as double of x.",
            "--examples-json", "examples/double_x.json",
            "--iterations", "6",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--artifact-dir", "debug_double_x",
            "--report-out", "debug_double_x/search_report_cli.json",
            "--out", "debug_double_x/best.ax"
        )
    },
    @{
        Name = "copilot-run double_x"
        Command = @(
            "axiom", "copilot-run",
            "--backend", $Backend,
            "--goal", "Compute y as double of x.",
            "--examples-json", "examples/double_x.json",
            "--iterations", "6",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--artifact-dir", "showcase_double_x",
            "--summary-out", "showcase_double_x/pipeline_summary.json",
            "--out", "showcase_double_x.ax"
        )
    },
    @{
        Name = "copilot-search risk_score"
        Command = @(
            "axiom", "copilot-search",
            "--backend", $Backend,
            "--goal", "Compute risk_score = max(0.0, min(1.0, 0.7 * risk_a + 0.3 * risk_b));",
            "--examples-json", "examples/risk_score_v3.json",
            "--iterations", "8",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--artifact-dir", "debug_risk_score",
            "--report-out", "debug_risk_score/search_report_cli.json",
            "--out", "debug_risk_score/best.ax"
        )
    },
    @{
        Name = "copilot-run risk_score"
        Command = @(
            "axiom", "copilot-run",
            "--backend", $Backend,
            "--goal", "Compute risk_score = max(0.0, min(1.0, 0.7 * risk_a + 0.3 * risk_b));",
            "--examples-json", "examples/risk_score_v3.json",
            "--iterations", "8",
            "--expert-url", $ExpertUrl,
            "--expert-model", $ExpertModel,
            "--artifact-dir", "showcase_risk_score",
            "--summary-out", "showcase_risk_score/pipeline_summary.json",
            "--out", "showcase_risk_score.ax"
        )
    }
)

$passCount = 0
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
        $passCount++
    }
}
catch {
    $failedStep = if ($step -and $step.Name) { $step.Name } else { "unknown step" }
    Write-Host ""
    Write-Host "SMOKE SUMMARY: FAIL ($passCount/$($steps.Count) passed)" -ForegroundColor Red
    Write-Host "Failed step: $failedStep" -ForegroundColor Red
    throw
}

Write-Host ""
Write-Host "SMOKE SUMMARY: PASS ($passCount/$($steps.Count) passed)" -ForegroundColor Green
