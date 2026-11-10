$ErrorActionPreference = 'Stop'
$modelDir = Join-Path $PSScriptRoot '.models'
$null = New-Item -ItemType Directory -Path $modelDir -Force
$base = 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/5641a7880f40ebf4035d05e60c5f9b7a9c272c84'
$assets = @(
    @{ Local='model.onnx'; Remote='onnx/model_quint8_avx2.onnx'; Hash='b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21' },
    @{ Local='tokenizer.json'; Remote='tokenizer.json'; Hash='be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037' }
)
foreach ($asset in $assets) {
    $path = Join-Path $modelDir $asset.Local
    if ((Test-Path -LiteralPath $path) -and ((Get-FileHash -LiteralPath $path).Hash -eq $asset.Hash)) { continue }
    $part = $path + '.part'
    $remote = $asset.Remote
    curl.exe -fL --show-error --max-time 600 "$base/${remote}?download=true" -o $part
    if ($LASTEXITCODE -ne 0 -or (Get-FileHash -LiteralPath $part).Hash -ne $asset.Hash) {
        throw "Could not verify $remote. Run fetch_model.ps1 again."
    }
    Move-Item -LiteralPath $part -Destination $path -Force
}
