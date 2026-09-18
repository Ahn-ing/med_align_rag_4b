param(
    [int]$StartOffset = 0,
    [int]$EndOffset = 900,
    [int]$ShardSize = 100,
    [string]$Endpoint,
    [string]$GenerationModel,
    [string]$JudgeModel
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$generator = Join-Path $PSScriptRoot "generate_phase7_pilot.py"
$tasksArg = "data/dpo/full_v1/remaining/tasks.jsonl"
$tasksPath = Join-Path $root $tasksArg

if (-not (Test-Path -LiteralPath $tasksPath)) { throw "任务文件不存在: $tasksPath" }
if ([string]::IsNullOrWhiteSpace($env:MODEL_API_KEY)) {
    throw "环境变量 MODEL_API_KEY 未设置；请在当前终端设置后再运行"
}
if ($StartOffset -lt 0 -or $EndOffset -gt 900 -or $EndOffset -le $StartOffset) {
    throw "StartOffset/EndOffset 必须位于 0..900 且 EndOffset > StartOffset"
}
if ($ShardSize -le 0) { throw "ShardSize 必须为正数" }

$extraArgs = @()
if ($Endpoint) { $extraArgs += @("--endpoint", $Endpoint) }
if ($GenerationModel) { $extraArgs += @("--generation-model", $GenerationModel) }
if ($JudgeModel) { $extraArgs += @("--judge-model", $JudgeModel) }

for ($offset = $StartOffset; $offset -lt $EndOffset; $offset += $ShardSize) {
    $limit = [Math]::Min($ShardSize, $EndOffset - $offset)
    $end = $offset + $limit - 1
    $output = Join-Path $root ("data\dpo\full_v1\shards\shard_{0:D4}_{1:D4}" -f $offset, $end)
    $manifest = Join-Path $output "manifest.json"
    if (Test-Path -LiteralPath $manifest) {
        Write-Host "SKIP completed shard $offset-$end"
        continue
    }
    if (Test-Path -LiteralPath $output) {
        throw "分片目录存在但没有完整 manifest，请人工检查: $output"
    }
    Write-Host "GENERATE shard $offset-$end"
    & $python $generator `
        --tasks $tasksArg `
        --output-dir ("data/dpo/full_v1/shards/shard_{0:D4}_{1:D4}" -f $offset, $end) `
        --budget-profile full `
        --offset $offset `
        --limit $limit `
        @extraArgs
    if ($LASTEXITCODE -ne 0) { throw "Phase 7 shard $offset-$end 失败，退出码: $LASTEXITCODE" }
}