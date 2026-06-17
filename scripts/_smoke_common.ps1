# Shared helpers for copilot smoke scripts (API key resolution + safe command echo).

function Resolve-ExpertApiKey {
    param(
        [string]$Explicit = "",
        [string]$KeyFile = ""
    )
    if (-not [string]::IsNullOrWhiteSpace($Explicit)) {
        return $Explicit.Trim()
    }
    if (-not [string]::IsNullOrWhiteSpace($KeyFile)) {
        $path = $KeyFile.Trim()
        if (-not (Test-Path -LiteralPath $path)) {
            throw "expert-api-key-file not found: $path"
        }
        $text = (Get-Content -LiteralPath $path -Raw).Trim()
        if ([string]::IsNullOrWhiteSpace($text)) {
            throw "expert-api-key-file is empty: $path"
        }
        return $text
    }
    $envKey = [string]$env:AXIOM_EXPERT_API_KEY
    if (-not [string]::IsNullOrWhiteSpace($envKey)) {
        return $envKey.Trim()
    }
    return ""
}

function Format-RedactedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Command
    )
    $parts = @()
    $skipNext = $false
    foreach ($item in $Command) {
        if ($skipNext) {
            $parts += "<redacted>"
            $skipNext = $false
            continue
        }
        if ($item -eq "--expert-api-key") {
            $parts += $item
            $skipNext = $true
            continue
        }
        $parts += $item
    }
    return ($parts -join " ")
}

function Append-ExpertApiKeyArgs {
    param(
        [Parameter(Mandatory = $true)]
        [System.Collections.Generic.List[string]]$Command,
        [string]$ExpertApiKey
    )
    if (-not [string]::IsNullOrWhiteSpace($ExpertApiKey)) {
        $Command.Add("--expert-api-key") | Out-Null
        $Command.Add($ExpertApiKey) | Out-Null
    }
}
