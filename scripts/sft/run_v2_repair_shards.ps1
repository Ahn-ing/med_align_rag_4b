param(
    [string]$ApiKeyEnv = "MODEL_API_KEY",
    [string]$Endpoint = "https://api.deepseek.com",
    [string]$Model = "deepseek-flash",
    [int]$BatchSize = 50
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$tasks = Join-Path $root "data\sft\reviewed\v2_repair\repair_tasks.jsonl"
$taskManifest = Join-Path $root "data\sft\reviewed\v2_repair\repair_tasks.manifest.json"
$outputDir = Join-Path $root "data\sft\reviewed\v2_repair\generated"
$tokenizer = Join-Path $root "data\external\tokenizers\Qwen3-4B"
$generator = Join-Path $root "scripts\sft\generate_phase6_v2_samples_with_api.py"

if (-not $env:MODEL_API_KEY -and -not [Environment]::GetEnvironmentVariable($ApiKeyEnv)) {
    throw "环境变量 $ApiKeyEnv 未设置"
}
if (-not (Test-Path -LiteralPath $python)) { throw "Python 不存在: $python" }
if (-not (Test-Path -LiteralPath $tasks)) { throw "任务文件不存在: $tasks" }
if (-not (Test-Path -LiteralPath $taskManifest)) { throw "任务 manifest 不存在: $taskManifest" }
if (-not (Test-Path -LiteralPath $tokenizer)) { throw "Tokenizer 不存在: $tokenizer" }
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$taskCount = (Get-Content -Raw -LiteralPath $taskManifest | ConvertFrom-Json).tasks
if ($BatchSize -le 0) { throw "BatchSize 必须为正数" }

for ($offset = 0; $offset -lt $taskCount; $offset += $BatchSize) {
    $output = Join-Path $outputDir ("repair_shard_{0:D4}.jsonl" -f $offset)
    if (Test-Path -LiteralPath $output) {
        Write-Host "SKIP $output"
        continue
    }
    Write-Host "GENERATE offset=$offset output=$output"
    & $python $generator --tasks $tasks --output $output --endpoint $Endpoint --model $Model --tokenizer-path $tokenizer --api-key-env $ApiKeyEnv --offset $offset --limit $BatchSize --max-attempts 3
    if ($LASTEXITCODE -ne 0) {
        throw "修复分片生成失败: offset=$offset"
    }
}

Write-Host "ALL_REPAIR_SHARDS_COMPLETE"
