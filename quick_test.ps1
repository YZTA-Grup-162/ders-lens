

Write-Host "AttentionPulse Kurulum Kontrol Ediliyor..." -ForegroundColor Green

# Konteyner durumlarini kontrol et
Write-Host "`nKonteyner Durumlari:" -ForegroundColor Yellow
docker ps --format "table {{.Names}}\t{{.Status}}"

# Backend test
Write-Host "`nBackend Test:" -ForegroundColor Yellow
try {
    $backend = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5
    Write-Host "Backend: CALISIYOR (Status: $($backend.StatusCode))" -ForegroundColor Green
} catch {
    Write-Host "Backend: HATA" -ForegroundColor Red
}

# Frontend test  
Write-Host "`nFrontend Test:" -ForegroundColor Yellow
try {
    $frontend = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 5
    Write-Host "Frontend: CALISIYOR (Status: $($frontend.StatusCode))" -ForegroundColor Green
} catch {
    Write-Host "Frontend: HATA" -ForegroundColor Red
}

Write-Host "`nKurulum Basarili!" -ForegroundColor Green
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "Backend:  http://localhost:8000" -ForegroundColor Cyan
