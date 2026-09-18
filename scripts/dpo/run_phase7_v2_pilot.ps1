[CmdletBinding()]
param(
    [string]$Tasks = "data/dpo/pilot_v2/tasks.jsonl",
    [string]$OutputRoot = "data/dpo/pilot_v2/generated_shards",
    [int]$StartOffset = 0,
    [int]$EndOffset = 50,
    [int]$ShardSize = 5,
    [double]$RequestIntervalSeconds = 2.0,
    [int]$MaxRetries = 2,
    [int]$RetryDelaySeconds = 30,
    [string]$ApiKeyEnv = "MODEL_API_KEY",
    [string]$Endpoint = "",
    [string]$GenerationModel = "",
    [string]$JudgeModel = "",
    [ValidateSet("chat_completions", "responses")]
    [string]$WireApi = "responses",
    [switch]$Stream,
    [ValidateSet("pilot", "full")]
    [string]$BudgetProfile = "full",
    [string]$Python = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ($ShardSize -le 0) { throw "ShardSize must be positive" }
if ($StartOffset -lt 0 -or $EndOffset -le $StartOffset) { throw "invalid offset range" }
if ($RequestIntervalSeconds -lt 0) { throw "RequestIntervalSeconds must be non-negative" }
if ($MaxRetries -lt 0) { throw "MaxRetries must be non-negative" }

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot

if (-not $Python) {
    $VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    $Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
}

$Generator = Join-Path $ProjectRoot "scripts\dpo\generate_phase7_v2_pilot.py"
if (-not (Test-Path $Generator)) { throw "generator not found: $Generator" }
if (-not (Test-Path $Tasks)) { throw "tasks not found: $Tasks" }
if (-not [Environment]::GetEnvironmentVariable($ApiKeyEnv)) {
    throw "environment variable $ApiKeyEnv is not set"
}

function Get-ShardResult {
    param([string]$OutputDir)
    $ManifestPath = Join-Path $OutputDir "manifest.json"
    $StylePath = Join-Path $OutputDir "style_report.json"
    if (-not (Test-Path $ManifestPath) -or -not (Test-Path $StylePath)) { return $null }
    $Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
    $Style = Get-Content $StylePath -Raw | ConvertFrom-Json
    [pscustomobject]@{
        Selected = [int]$Manifest.counts.pairs_selected
        Quarantined = [int]$Manifest.counts.quarantined
        StyleStatus = [string]$Style.status
        StyleFailures = @($Style.failures)
    }
}

function Invoke-Shard {
    param([int]$Offset, [int]$Limit, [string]$OutputDir)
    if (Test-Path $OutputDir) { throw "refusing to overwrite existing output: $OutputDir" }

    $Arguments = @(
        $Generator,
        "--tasks", $Tasks,
        "--output-dir", $OutputDir,
        "--offset", $Offset,
        "--limit", $Limit,
        "--request-interval-seconds", $RequestIntervalSeconds
    )
    if ($Endpoint) { $Arguments += @("--endpoint", $Endpoint) }
    if ($GenerationModel) { $Arguments += @("--generation-model", $GenerationModel) }
    if ($JudgeModel) { $Arguments += @("--judge-model", $JudgeModel) }
    $Arguments += @("--wire-api", $WireApi)
    if ($Stream) { $Arguments += "--stream" } else { $Arguments += "--no-stream" }
    $Arguments += @("--budget-profile", $BudgetProfile)

    Write-Host ("[RUN] offset={0} limit={1} interval={2}s output={3}" -f $Offset, $Limit, $RequestIntervalSeconds, $OutputDir)
    if ($DryRun) { return $null }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "generator failed for offset=$Offset limit=$Limit exit=$LASTEXITCODE"
    }
    return Get-ShardResult -OutputDir $OutputDir
}

$ShardIndex = 0
$Completed = 0
$Skipped = 0

for ($Offset = $StartOffset; $Offset -lt $EndOffset; $Offset += $ShardSize) {
    $Limit = [Math]::Min($ShardSize, $EndOffset - $Offset)
    $ShardIndex += 1
    $Success = $false

    for ($Attempt = 0; $Attempt -le $MaxRetries; $Attempt++) {
        $Suffix = if ($Attempt -eq 0) { "" } else { "_retry$Attempt" }
        $OutputDir = Join-Path $OutputRoot ("shard_{0:D4}_{1:D4}{2}" -f $Offset, ($Offset + $Limit - 1), $Suffix)

        if (Test-Path $OutputDir) {
            $Existing = Get-ShardResult -OutputDir $OutputDir
            if ($Existing -and $Existing.Selected -eq $Limit) {
                if ($Existing.StyleStatus -ne "passed") {
                    Write-Warning ("[STYLE] shard-level report not passed for {0}: {1}; global gate is evaluated after merging" -f $OutputDir, ($Existing.StyleFailures -join ", "))
                }
                Write-Host ("[SKIP] already complete: {0}" -f $OutputDir)
                $Success = $true
                $Skipped += 1
                break
            }
            if ($Attempt -eq $MaxRetries) {
                throw ("shard remains incomplete after retries: {0}" -f $OutputDir)
            }
            continue
        }

        try {
            $Result = Invoke-Shard -Offset $Offset -Limit $Limit -OutputDir $OutputDir
            if ($DryRun) { $Success = $true; break }
            if ($null -eq $Result) { throw "shard produced no result" }
            if ($Result.Selected -lt $Limit) {
                throw ("shard incomplete: selected={0} expected={1} quarantine={2}" -f $Result.Selected, $Limit, $Result.Quarantined)
            }
            if ($Result.StyleStatus -ne "passed") {
                Write-Warning ("[STYLE] shard-level report not passed: {0}; global gate is evaluated after merging" -f ($Result.StyleFailures -join ", "))
            }
            Write-Host ("[OK] selected={0}/{1} quarantined={2}" -f $Result.Selected, $Limit, $Result.Quarantined)
            $Completed += 1
            $Success = $true
            break
        }
        catch {
            Write-Warning $_.Exception.Message
            if ($Attempt -eq $MaxRetries) { throw }
            Write-Host ("[WAIT] retrying shard after {0}s" -f $RetryDelaySeconds)
            if (-not $DryRun) { Start-Sleep -Seconds $RetryDelaySeconds }
        }
    }

    if (-not $Success) { throw "shard failed without success: offset=$Offset" }
}

Write-Host ("[DONE] shards={0} completed={1} skipped={2} output_root={3}" -f $ShardIndex, $Completed, $Skipped, $OutputRoot)
