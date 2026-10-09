# Cleanup files created today under the current project recursively
$root = Get-Location
$tod = Get-ChildItem -LiteralPath . -File -Recurse | Where-Object { $_.CreationTime.Date -eq (Get-Date).Date }

if (-not $tod -or $tod.Count -eq 0) {
    Write-Host "No files created today under $root"
    exit 0
}

Write-Host "The following files created today will be deleted:" -ForegroundColor Yellow
$tod | Select-Object FullName, CreationTime | Format-Table -AutoSize

foreach ($f in $tod) {
    try {
        Remove-Item -LiteralPath $f.FullName -Force -ErrorAction Stop
        Write-Host "Deleted: $($f.FullName)"
    } catch {
        Write-Warning "Failed to delete: $($f.FullName) - $($_.Exception.Message)"
    }
}

Write-Host "Cleanup complete. Deleted $($tod.Count) files." -ForegroundColor Green
