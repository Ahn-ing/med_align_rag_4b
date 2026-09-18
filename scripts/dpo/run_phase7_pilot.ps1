$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$generator = Join-Path $PSScriptRoot "generate_phase7_pilot.py"

if ([string]::IsNullOrWhiteSpace($env:MODEL_API_KEY)) {
    throw "环境变量 MODEL_API_KEY 未设置；请在当前终端设置后再运行"
}
if (-not (Test-Path -LiteralPath $python)) { throw "Python 不存在: $python" }

& $python $generator
if ($LASTEXITCODE -ne 0) { throw "Phase 7 API pilot 失败，退出码: $LASTEXITCODE" }