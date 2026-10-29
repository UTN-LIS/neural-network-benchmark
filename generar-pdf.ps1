[CmdletBinding()]
param(
    [string]$Html = "Explicacion-Proyecto-Prebenchmark.html",
    [string]$Pdf  = "Explicacion-Proyecto-Prebenchmark.pdf"
)

$ErrorActionPreference = "Stop"

$root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$htmlPath = if ([IO.Path]::IsPathRooted($Html)) { $Html } else { Join-Path $root $Html }
$pdfPath  = if ([IO.Path]::IsPathRooted($Pdf))  { $Pdf }  else { Join-Path $root $Pdf }

if (-not (Test-Path -LiteralPath $htmlPath)) {
    Write-Error "No se encontro el HTML: $htmlPath"
    exit 1
}

$browser = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LocalAppData\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe"
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

if (-not $browser) {
    Write-Error "No se encontro Chrome ni Edge."
    exit 1
}

$uri = ([Uri]$htmlPath).AbsoluteUri
Write-Host "Navegador: $browser"
Write-Host "HTML     : $htmlPath"
Write-Host "PDF      : $pdfPath"

& $browser --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$pdfPath" "$uri" 2>$null
Start-Sleep -Seconds 2

if (Test-Path -LiteralPath $pdfPath) {
    $f = Get-Item -LiteralPath $pdfPath
    Write-Host ("PDF generado: {0} ({1} KB) {2}" -f $f.Name, [math]::Round($f.Length / 1KB, 1), $f.LastWriteTime)
} else {
    Write-Error "No se genero el PDF."
    exit 1
}
