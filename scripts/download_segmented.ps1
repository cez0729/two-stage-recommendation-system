param(
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$Destination,
    [Parameter(Mandatory = $true)][long]$TotalBytes,
    [int]$ChunkSizeMb = 2,
    [int]$MaxParallel = 24
)

$ErrorActionPreference = "Stop"
$destinationPath = [IO.Path]::GetFullPath($Destination)
$partDirectory = "$destinationPath.parts"
[IO.Directory]::CreateDirectory($partDirectory) | Out-Null
$chunkBytes = $ChunkSizeMb * 1MB
$chunks = @()

for ($start = 0L; $start -lt $TotalBytes; $start += $chunkBytes) {
    $end = [Math]::Min($start + $chunkBytes - 1, $TotalBytes - 1)
    $partPath = Join-Path $partDirectory ("{0:D6}.part" -f ($start / $chunkBytes))
    $chunks += [PSCustomObject]@{
        Start = $start
        End = $end
        Length = $end - $start + 1
        Path = $partPath
        Process = $null
        Attempts = 0
    }
}

$pending = [Collections.Generic.Queue[object]]::new()
foreach ($chunk in $chunks) {
    if ((Test-Path -LiteralPath $chunk.Path) -and (Get-Item $chunk.Path).Length -eq $chunk.Length) {
        continue
    }
    $pending.Enqueue($chunk)
}

$active = [Collections.Generic.List[object]]::new()
while ($pending.Count -gt 0 -or $active.Count -gt 0) {
    while ($pending.Count -gt 0 -and $active.Count -lt $MaxParallel) {
        $chunk = $pending.Dequeue()
        $chunk.Attempts += 1
        $range = "$($chunk.Start)-$($chunk.End)"
        $arguments = @(
            "-L", "--silent", "--show-error", "--fail", "--retry", "5",
            "--connect-timeout", "30", "--range", $range, "--output", $chunk.Path, $Url
        )
        $chunk.Process = Start-Process curl.exe -ArgumentList $arguments -PassThru -WindowStyle Hidden
        $active.Add($chunk)
    }

    Start-Sleep -Milliseconds 500
    foreach ($chunk in @($active)) {
        if (-not $chunk.Process.HasExited) { continue }
        $chunk.Process.WaitForExit()
        $null = $active.Remove($chunk)
        $actual = if (Test-Path -LiteralPath $chunk.Path) {
            (Get-Item -LiteralPath $chunk.Path).Length
        } else { 0 }
        if ($chunk.Process.ExitCode -ne 0 -or $actual -ne $chunk.Length) {
            if ($chunk.Attempts -ge 5) {
                throw "Range $($chunk.Start)-$($chunk.End) failed after 5 attempts"
            }
            $pending.Enqueue($chunk)
        }
    }
}

$output = [IO.File]::Open($destinationPath, [IO.FileMode]::Create, [IO.FileAccess]::Write)
try {
    foreach ($chunk in $chunks) {
        $input = [IO.File]::OpenRead($chunk.Path)
        try { $input.CopyTo($output) } finally { $input.Dispose() }
    }
} finally {
    $output.Dispose()
}

$result = Get-Item -LiteralPath $destinationPath
if ($result.Length -ne $TotalBytes) {
    throw "Combined file expected $TotalBytes bytes, got $($result.Length)"
}
Write-Output "Downloaded $($result.FullName) ($($result.Length) bytes)"
