$checkInterval = 30

Write-Host "[BlueStacks Watchdog] Started - checking every ${checkInterval}s"

while ($true) {
    $hung = Get-Process -Name "HD-Player" -ErrorAction SilentlyContinue |
        Where-Object { -not $_.Responding }

    foreach ($proc in $hung) {
        $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.Id)" `
            -ErrorAction SilentlyContinue).CommandLine
        $instance = if ($cmdLine -match '--instance\s+(\S+)') { $Matches[1] } else { "unknown" }

        $ts = Get-Date -Format 'HH:mm:ss'
        Write-Host "[$ts] Killing hung BlueStacks: PID=$($proc.Id) Instance=$instance"
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }

    Start-Sleep -Seconds $checkInterval
}
