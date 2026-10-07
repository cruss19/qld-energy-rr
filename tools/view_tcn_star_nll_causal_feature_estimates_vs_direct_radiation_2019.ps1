param(
    [ValidateRange(1, 300)]
    [int]$RefreshSeconds = 2,
    [switch]$Once,
    [string]$CandidateDir = (Join-Path $PSScriptRoot '..\training_output\runs\TCN_starNLL_causal_feature_estimates\development\validation_2019\seed_42'),
    [Parameter(Mandatory = $true)]
    [string]$ComparatorDir,
    [string]$CandidateLogDir = (Join-Path $PSScriptRoot '..\training_output\logs\TCN_starNLL_causal_feature_estimates_validate_2019_seed42')
)

$ErrorActionPreference = 'Stop'
$candidateName = 'TCN_starNLL_causal_feature_estimates'
$comparatorName = 'TCN_starNLL_direct_radiation'
$candidateDir = [System.IO.Path]::GetFullPath($CandidateDir)
$comparatorDir = [System.IO.Path]::GetFullPath($ComparatorDir)
$candidateLogDir = [System.IO.Path]::GetFullPath($CandidateLogDir)
$candidateProgress = Join-Path $candidateDir 'live_progress.json'
$candidateEpochs = Join-Path $candidateDir 'epoch_log.csv'
$candidateComplete = Join-Path $candidateDir 'completed_run.json'
$candidateErrorLog = Join-Path $candidateLogDir 'stderr.log'
$comparatorEpochs = Join-Path $comparatorDir 'epoch_log.csv'
$comparatorComplete = Join-Path $comparatorDir 'completed_run.json'

$Host.UI.RawUI.WindowTitle = 'Causal feature estimates vs direct radiation | 2019 | seed 42'

Write-Verbose "Candidate run directory: $candidateDir"
Write-Verbose "Comparator run directory: $comparatorDir"

function Read-CsvSafely([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return @() }
    try { return @(Import-Csv -LiteralPath $Path) } catch { return @() }
}

function Read-JsonSafely([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try { return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json } catch { return $null }
}

function Format-Duration([double]$Seconds) {
    if ($Seconds -lt 0 -or [double]::IsNaN($Seconds) -or [double]::IsInfinity($Seconds)) {
        return 'measuring...'
    }
    return [TimeSpan]::FromSeconds([Math]::Ceiling($Seconds)).ToString('hh\:mm\:ss')
}

function Write-SignedDelta([double]$Value, [string]$Format, [string]$Suffix = '') {
    $colour = if ($Value -lt 0) { 'Green' } elseif ($Value -gt 0) { 'Red' } else { 'White' }
    Write-Host (($Format -f $Value) + $Suffix) -ForegroundColor $colour -NoNewline
}

function Write-Bar([double]$Numerator, [double]$Denominator, [int]$Width = 38) {
    $ratio = if ($Denominator -gt 0) {
        [Math]::Max(0.0, [Math]::Min(1.0, $Numerator / $Denominator))
    } else { 0.0 }
    $filled = [Math]::Min($Width, [int][Math]::Floor($ratio * $Width))
    Write-Host '[' -NoNewline
    if ($filled -gt 0) { Write-Host ('#' * $filled) -ForegroundColor Green -NoNewline }
    if ($filled -lt $Width) { Write-Host ('-' * ($Width - $filled)) -ForegroundColor DarkGray -NoNewline }
    Write-Host ('] {0,6:P1}' -f $ratio) -NoNewline
}

function Show-LatestCompleted($CandidateRows, $ComparatorRows) {
    Write-Host 'LATEST COMPLETED EPOCH' -ForegroundColor Yellow
    if ($CandidateRows.Count -eq 0) {
        Write-Host '  No extension epoch has completed yet.' -ForegroundColor DarkGray
        return
    }
    $last = $CandidateRows[-1]
    $match = @($ComparatorRows | Where-Object { [int]$_.epoch -eq [int]$last.epoch }) | Select-Object -First 1
    Write-Host ('  Extension epoch {0}' -f [int]$last.epoch) -ForegroundColor Cyan
    Write-Host ('    train: NLL {0:F6} | MAE {1:F3} MW | RMSE {2:F3} MW' -f
        [double]$last.training_nll, [double]$last.training_mae_mw,
        [double]$last.training_rmse_mw)
    Write-Host ('    valid: NLL {0:F6} | MAE {1:F3} MW | RMSE {2:F3} MW' -f
        [double]$last.validation_nll, [double]$last.validation_mae_mw,
        [double]$last.validation_rmse_mw)
    Write-Host ('    LR used {0:G6} -> next {1:G6} | best epoch {2} | bad epochs {3}' -f
        [double]$last.learning_rate_used, [double]$last.learning_rate_next,
        [int]$last.best_epoch, [int]$last.epochs_no_improve)
    if ($null -eq $match) {
        Write-Host '  No same-epoch comp-model row is available.' -ForegroundColor DarkGray
        return
    }
    Write-Host ('  Direct-radiation comp epoch {0}' -f [int]$match.epoch) -ForegroundColor Yellow
    Write-Host ('    valid: NLL {0:F6} | MAE {1:F3} MW | RMSE {2:F3} MW' -f
        [double]$match.validation_nll, [double]$match.validation_mae_mw,
        [double]$match.validation_rmse_mw)
    Write-Host '  Signed extension-minus-comp deltas: NLL ' -NoNewline
    Write-SignedDelta ([double]$last.validation_nll - [double]$match.validation_nll) '{0:+0.000000;-0.000000;0.000000}'
    Write-Host ' | MAE ' -NoNewline
    Write-SignedDelta ([double]$last.validation_mae_mw - [double]$match.validation_mae_mw) '{0:+0.000;-0.000;0.000}' ' MW'
    Write-Host ' | RMSE ' -NoNewline
    Write-SignedDelta ([double]$last.validation_rmse_mw - [double]$match.validation_rmse_mw) '{0:+0.000;-0.000;0.000}' ' MW'
    Write-Host ''
}

function Show-BestComparison($CandidateRows, $ComparatorRows) {
    Write-Host 'BEST-CHECKPOINT COMPARISON (minimum validation NLL)' -ForegroundColor Yellow
    if ($CandidateRows.Count -eq 0 -or $ComparatorRows.Count -eq 0) {
        Write-Host '  Waiting for completed extension and comp-model rows.' -ForegroundColor DarkGray
        return
    }
    $candidateBest = $CandidateRows | Sort-Object { [double]$_.validation_nll } | Select-Object -First 1
    $comparatorBest = $ComparatorRows | Sort-Object { [double]$_.validation_nll } | Select-Object -First 1
    Write-Host ('  Extension:        epoch {0,2} | NLL {1:F6} | MAE {2:F3} MW | RMSE {3:F3} MW' -f
        [int]$candidateBest.epoch, [double]$candidateBest.validation_nll,
        [double]$candidateBest.validation_mae_mw, [double]$candidateBest.validation_rmse_mw) -ForegroundColor Cyan
    Write-Host ('  Direct radiation: epoch {0,2} | NLL {1:F6} | MAE {2:F3} MW | RMSE {3:F3} MW' -f
        [int]$comparatorBest.epoch, [double]$comparatorBest.validation_nll,
        [double]$comparatorBest.validation_mae_mw, [double]$comparatorBest.validation_rmse_mw) -ForegroundColor Yellow
    Write-Host '  Signed best deltas: NLL ' -NoNewline
    Write-SignedDelta ([double]$candidateBest.validation_nll - [double]$comparatorBest.validation_nll) '{0:+0.000000;-0.000000;0.000000}'
    Write-Host ' | MAE ' -NoNewline
    Write-SignedDelta ([double]$candidateBest.validation_mae_mw - [double]$comparatorBest.validation_mae_mw) '{0:+0.000;-0.000;0.000}' ' MW'
    Write-Host ' | RMSE ' -NoNewline
    Write-SignedDelta ([double]$candidateBest.validation_rmse_mw - [double]$comparatorBest.validation_rmse_mw) '{0:+0.000;-0.000;0.000}' ' MW'
    Write-Host ''
}

function Show-EpochTable($CandidateRows, $ComparatorRows) {
    Write-Host 'RECENT MATCHED EPOCHS (extension minus direct-radiation comp)' -ForegroundColor Yellow
    if ($CandidateRows.Count -eq 0) {
        Write-Host '  Waiting for completed extension epochs.' -ForegroundColor DarkGray
        return
    }
    Write-Host '  Ep   Ext NLL    Comp NLL      dNLL    Ext MAE   Comp MAE     dMAE' -ForegroundColor DarkGray
    foreach ($row in @($CandidateRows | Select-Object -Last 8)) {
        $match = @($ComparatorRows | Where-Object { [int]$_.epoch -eq [int]$row.epoch }) | Select-Object -First 1
        if ($null -eq $match) { continue }
        $deltaNll = [double]$row.validation_nll - [double]$match.validation_nll
        $deltaMae = [double]$row.validation_mae_mw - [double]$match.validation_mae_mw
        $colour = if ($deltaNll -lt 0) { 'Green' } elseif ($deltaNll -gt 0) { 'Red' } else { 'White' }
        Write-Host ('  {0,2}   {1,8:F6}   {2,8:F6}  {3,9:+0.000000;-0.000000;0.000000}   {4,7:F3}   {5,8:F3}  {6,8:+0.000;-0.000;0.000}' -f
            [int]$row.epoch, [double]$row.validation_nll, [double]$match.validation_nll,
            $deltaNll, [double]$row.validation_mae_mw, [double]$match.validation_mae_mw,
            $deltaMae) -ForegroundColor $colour
    }
}

if (-not (Test-Path -LiteralPath $comparatorEpochs)) {
    throw "Exact 2019 seed-42 comp-model epoch log is missing: $comparatorEpochs"
}
if (-not (Test-Path -LiteralPath $comparatorComplete)) {
    throw "Exact 2019 seed-42 comp model is not complete: $comparatorComplete"
}

while ($true) {
    if (-not $Once) { Clear-Host }
    Write-Host 'TCN_starNLL_causal_feature_estimates vs TCN_starNLL_direct_radiation' -ForegroundColor Cyan
    Write-Host 'Train 2015-2018 | validate 2019 | seed 42 | lower is better' -ForegroundColor White
    Write-Host 'Extension: upstream causal demand/radiation estimates plus aligned estimate-SD channels' -ForegroundColor DarkGray
    Write-Host ('Updated: {0}' -f (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')) -ForegroundColor DarkGray
    Write-Host ('Candidate:  {0}' -f $candidateName) -ForegroundColor Cyan
    Write-Host ('Comp model: {0}' -f $comparatorName) -ForegroundColor Yellow
    Write-Host ''

    $progress = Read-JsonSafely $candidateProgress
    $candidateRows = Read-CsvSafely $candidateEpochs
    $comparatorRows = Read-CsvSafely $comparatorEpochs
    Write-Host 'LIVE PROGRESS' -ForegroundColor Yellow
    if ($null -eq $progress) {
        if (Test-Path -LiteralPath $candidateComplete) {
            Write-Host '  COMPLETE' -ForegroundColor Green
        } else {
            Write-Host '  Waiting for live progress...' -ForegroundColor DarkGray
        }
    } else {
        $age = ((Get-Date) - (Get-Item -LiteralPath $candidateProgress).LastWriteTime).TotalSeconds
        $health = if (Test-Path -LiteralPath $candidateComplete) {
            'COMPLETE'
        } elseif ($age -le 120) {
            'ACTIVE'
        } else {
            'NO RECENT PROGRESS UPDATE'
        }
        $healthColour = if ($health -eq 'ACTIVE' -or $health -eq 'COMPLETE') { 'Green' } else { 'Red' }
        Write-Host ('  Status {0} | progress age {1:F0}s' -f $health, $age) -ForegroundColor $healthColour
        Write-Host ('  Epoch {0}/50 | phase {1} | batch {2}/{3}' -f
            [int]$progress.epoch, $progress.phase,
            [int]$progress.current_batch, [int]$progress.total_batches) -ForegroundColor Cyan
        Write-Host '  Phase  ' -NoNewline
        Write-Bar ([double]$progress.current_batch) ([double]$progress.total_batches)
        Write-Host (' | ETA {0}' -f (Format-Duration ([double]$progress.eta_seconds)))
        $epochNumerator = if ($progress.status -eq 'epoch_complete') {
            [double]$progress.epoch
        } else {
            [Math]::Max(0.0, [double]$progress.epoch - 1.0)
        }
        Write-Host '  Epochs ' -NoNewline
        Write-Bar $epochNumerator 50
        Write-Host ' | maximum ceiling; early stopping may finish sooner'
        Write-Host ('  Phase running: NLL {0:F6} | MAE {1:F3} MW' -f
            [double]$progress.running_nll, [double]$progress.running_mae_mw) -ForegroundColor Cyan
        if ($progress.status -eq 'epoch_complete') {
            Write-Host ('  Completed train: NLL {0:F6} | MAE {1:F3} MW' -f
                [double]$progress.training_nll, [double]$progress.training_mae_mw)
            Write-Host ('  Completed valid: NLL {0:F6} | MAE {1:F3} MW' -f
                [double]$progress.validation_nll, [double]$progress.validation_mae_mw)
        }
    }

    Write-Host ''
    Show-LatestCompleted $candidateRows $comparatorRows
    Write-Host ''
    Show-BestComparison $candidateRows $comparatorRows
    Write-Host ''
    Show-EpochTable $candidateRows $comparatorRows

    if ((Test-Path -LiteralPath $candidateErrorLog) -and
        (Get-Item -LiteralPath $candidateErrorLog).Length -gt 0) {
        Write-Host ''
        Write-Host 'LATEST STDERR' -ForegroundColor Red
        Get-Content -LiteralPath $candidateErrorLog -Tail 6 |
            ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    }

    Write-Host ''
    Write-Host 'Green signed delta = extension is better; red = direct-radiation comp model is better.' -ForegroundColor DarkGray
    Write-Host 'Read-only viewer. Closing it does not affect training.' -ForegroundColor DarkGray
    if ($Once) { break }
    Start-Sleep -Seconds $RefreshSeconds
}
