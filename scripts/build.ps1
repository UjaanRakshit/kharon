# Windows build wrapper: set up the MSVC env (vcvars) so nvcc finds cl.exe,
# put MSYS make on PATH, then run make from the project root. Pass make args
# through, e.g.  scripts\build.ps1 tests   or   scripts\build.ps1 DEBUG=1 clean
$ErrorActionPreference = "Stop"
$vcbase = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$vcvars = "$vcbase\VC\Auxiliary\Build\vcvars64.bat"
$msvcver = (Get-ChildItem "$vcbase\VC\Tools\MSVC" | Sort-Object Name -Descending | Select-Object -First 1).Name
$ccbin  = "$vcbase\VC\Tools\MSVC\$msvcver\bin\Hostx64\x64"
$msys   = "C:\msys64\usr\bin"
$proj   = Split-Path $PSScriptRoot -Parent
Set-Location $proj
$makeargs = $args -join ' '
cmd /c "`"$vcvars`" >nul 2>&1 && set `"PATH=%PATH%;$msys`" && make CCBIN=`"$ccbin`" $makeargs"
exit $LASTEXITCODE
