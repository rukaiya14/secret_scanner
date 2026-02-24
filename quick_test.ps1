# Quick Test Script for Deployment Blocker

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "Testing Universal Deployment Blocker" -ForegroundColor Yellow
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""

# Step 1: Create test directory
Write-Host "Step 1: Creating test repository..." -ForegroundColor Green
$testDir = "C:\Users\Aliasgar Ghadiali\Downloads\test-blocker"
if (Test-Path $testDir) {
    Remove-Item $testDir -Recurse -Force
}
New-Item -ItemType Directory -Path $testDir | Out-Null
Set-Location $testDir

# Step 2: Initialize git
Write-Host "Step 2: Initializing Git..." -ForegroundColor Green
git init | Out-Null
git config user.name "Test User"
git config user.email "test@example.com"

# Step 3: Create file with secrets
Write-Host "Step 3: Creating file with fake secrets..." -ForegroundColor Green
@"
# Test Configuration

AWS_ACCESS_KEY_ID=AKIAI44QH8DHBEX7AMPLE
AWS_SECRET_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
ADMIN_EMAIL=john.doe@company.com
PHONE=555-123-4567
"@ | Out-File -FilePath "config.txt" -Encoding UTF8

# Step 4: Commit
Write-Host "Step 4: Committing file..." -ForegroundColor Green
git add config.txt
git commit -m "Add config with secrets" | Out-Null

# Step 5: Try to push (will be blocked)
Write-Host "Step 5: Attempting to push (should be BLOCKED)..." -ForegroundColor Green
Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Red
Write-Host ("=" * 69) -ForegroundColor Red

git remote add origin https://github.com/test/test-repo.git
git push -u origin main 2>&1

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host ""

# Step 6: Clean up
Write-Host "Cleaning up test directory..." -ForegroundColor Green
Set-Location "C:\Users\Aliasgar Ghadiali\Downloads\Major Project"
Remove-Item $testDir -Recurse -Force

Write-Host ""
Write-Host "Test Complete!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Expected Result:" -ForegroundColor Cyan
Write-Host "  - Scanner should have detected 4 secrets" -ForegroundColor White
Write-Host "  - Push should have been BLOCKED" -ForegroundColor White
Write-Host "  - Error message should show detected secrets" -ForegroundColor White
Write-Host ""
