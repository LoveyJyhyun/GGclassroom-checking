# Loads .env into the environment, then runs the app (or any passed command).
# Anchored to this script's own folder, so it works from any working directory.
Set-Location $PSScriptRoot

Get-Content .env | Where-Object { $_ -match '^\s*[^#=\s]+=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2
    [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), 'Process')
}

# Re-downloading the OAuth client gives it a new filename; fall back to whatever is here.
if (-not $env:GOOGLE_CLIENT_SECRETS -or -not (Test-Path $env:GOOGLE_CLIENT_SECRETS)) {
    $found = Get-ChildItem client_secret*.json | Where-Object { $_.Name -notmatch 'desktop' } | Select-Object -First 1
    if ($found) { $env:GOOGLE_CLIENT_SECRETS = $found.Name }
}
if (-not $env:GOOGLE_CLIENT_SECRETS -or -not (Test-Path $env:GOOGLE_CLIENT_SECRETS)) {
    Write-Error "No OAuth client file found. Put the Web application client JSON in $PSScriptRoot."
    exit 1
}
# Absolute, so a child process that changes directory still finds it.
$env:GOOGLE_CLIENT_SECRETS = (Resolve-Path $env:GOOGLE_CLIENT_SECRETS).Path
Write-Output "OAuth client: $env:GOOGLE_CLIENT_SECRETS"

if ($args) { & .venv\Scripts\python.exe @args } else { & .venv\Scripts\python.exe app.py }
