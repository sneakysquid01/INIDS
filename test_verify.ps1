
$files = @(
    "web_app/templates/404.html",
    "web_app/templates/about.html",
    "web_app/templates/actions.html",
    "web_app/templates/alerts.html",
    "web_app/templates/allowlist.html",
    "web_app/templates/batch.html",
    "web_app/templates/capture.html",
    "web_app/templates/dashboard_main.html",
    "web_app/templates/detection.html",
    "web_app/templates/engines.html",
    "web_app/templates/error.html",
    "web_app/templates/health.html",
    "web_app/templates/home.html",
    "web_app/templates/index_main.html",
    "web_app/templates/index.html",
    "web_app/templates/investigate.html",
    "web_app/templates/learn.html",
    "web_app/templates/models.html",
    "web_app/templates/monitor.html",
    "web_app/templates/policy.html",
    "web_app/templates/predict.html",
    "web_app/templates/realtime.html",
    "web_app/templates/respond.html"
)
$failures = @()
foreach ($file in $files) {
    if (-not (Test-Path $file)) { $failures += "Missing: $file"; continue }
    $content = Get-Content $file -Raw
    if ($content -notmatch "\{% extends `"base.html`" %\}") { $failures += "($file) missing base" }
    if ($content -notmatch "\{% block content %\}") { $failures += "($file) missing content block" }
    if ($content -notmatch "\{% block page_title %\}") { $failures += "($file) missing title block" }
    if ($content -match "<html" -or $content -match "<body") { $failures += "($file) has html/body" }
    if ($content -match "\{\{%" -or $content -match "%\}\}") { $failures += "($file) has bad syntax" }
}
if ($failures.Count -eq 0) { "all passed" } else { $failures }

