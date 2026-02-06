# ==============================================================================
# Fix PostgreSQL Password Issue
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "`n==================================================================" -ForegroundColor Cyan
Write-Host "POSTGRESQL PASSWORD FIX" -ForegroundColor Cyan
Write-Host "==================================================================`n" -ForegroundColor Cyan

# Step 1: Find all .env files
Write-Host "Step 1: Searching for .env files...`n" -ForegroundColor Yellow

$envFiles = @(
    ".env",
    "../.env",
    "../infrastructure/.env",
    ".env.local",
    ".env.development"
)

$foundFiles = @()

foreach ($file in $envFiles) {
    if (Test-Path $file) {
        $fullPath = (Resolve-Path $file).Path
        $foundFiles += $fullPath
        Write-Host "✅ Found: $fullPath" -ForegroundColor Green
        
        # Check password in this file
        $content = Get-Content $file -Raw
        if ($content -match 'POSTGRES_PASSWORD=(.+)') {
            $password = $matches[1].Trim().Trim('"').Trim("'")
            Write-Host "   POSTGRES_PASSWORD length: $($password.Length)" -ForegroundColor White
            
            if ($password.Length -eq 8) {
                Write-Host "   ⚠️  THIS FILE HAS THE 8-CHARACTER PASSWORD!" -ForegroundColor Red
            }
        }
        Write-Host ""
    }
}

if ($foundFiles.Count -eq 0) {
    Write-Host "❌ No .env files found!" -ForegroundColor Red
    exit 1
}

# Step 2: Check system environment variable
Write-Host "`nStep 2: Checking system environment variable...`n" -ForegroundColor Yellow

$systemPassword = $env:POSTGRES_PASSWORD
if ($systemPassword) {
    Write-Host "✅ POSTGRES_PASSWORD is set in environment" -ForegroundColor Green
    Write-Host "   Length: $($systemPassword.Length)" -ForegroundColor White
    
    if ($systemPassword.Length -eq 8) {
        Write-Host "   ⚠️  THE SYSTEM ENVIRONMENT VARIABLE HAS 8 CHARACTERS!" -ForegroundColor Red
        Write-Host "   This will override your .env file!" -ForegroundColor Red
        
        $unset = Read-Host "`nDo you want to unset it? (yes/no)"
        if ($unset -eq 'yes') {
            $env:POSTGRES_PASSWORD = $null
            Write-Host "✅ Unset for current session" -ForegroundColor Green
            Write-Host "⚠️  Note: This only affects the current PowerShell session" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "ℹ️  No system environment variable set" -ForegroundColor Blue
}

# Step 3: Check Docker container
Write-Host "`nStep 3: Checking Docker container password...`n" -ForegroundColor Yellow

$container = docker ps --filter "name=postgres" --format "{{.Names}}" | Select-Object -First 1

if ($container) {
    Write-Host "✅ Found PostgreSQL container: $container" -ForegroundColor Green
    
    # Try to connect with devpass
    Write-Host "`nTesting connection with 'devpass' (7 chars)..." -ForegroundColor White
    $result = docker exec $container psql -U admin -d appdb -c "SELECT 1;" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Connection successful! Container accepts 'devpass'" -ForegroundColor Green
    } else {
        Write-Host "❌ Connection failed with 'devpass'" -ForegroundColor Red
        
        $reset = Read-Host "`nDo you want to reset the container password to 'devpass'? (yes/no)"
        if ($reset -eq 'yes') {
            Write-Host "`nAttempting to reset password..." -ForegroundColor Yellow
            
            # Try with postgres user first
            docker exec $container psql -U postgres -c "ALTER USER admin WITH PASSWORD 'devpass';" 2>&1
            
            if ($LASTEXITCODE -ne 0) {
                # Try with admin user
                docker exec $container psql -U admin -d appdb -c "ALTER USER admin WITH PASSWORD 'devpass';" 2>&1
            }
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ Password reset to 'devpass'" -ForegroundColor Green
            } else {
                Write-Host "❌ Could not reset password" -ForegroundColor Red
                Write-Host "   You may need to recreate the container" -ForegroundColor Yellow
            }
        }
    }
} else {
    Write-Host "❌ PostgreSQL container not running" -ForegroundColor Red
}

# Step 4: Fix .env files
Write-Host "`nStep 4: Fix .env files...`n" -ForegroundColor Yellow

$mainEnvFile = ".env"
if (-not (Test-Path $mainEnvFile)) {
    $mainEnvFile = "../.env"
}

if (Test-Path $mainEnvFile) {
    $content = Get-Content $mainEnvFile -Raw
    
    if ($content -match 'POSTGRES_PASSWORD=(.+)') {
        $currentPassword = $matches[1].Trim().Trim('"').Trim("'")
        
        if ($currentPassword.Length -ne 7 -or $currentPassword -ne 'devpass') {
            Write-Host "Current password in $mainEnvFile : $('*' * $currentPassword.Length) chars" -ForegroundColor White
            
            $fix = Read-Host "Do you want to change it to 'devpass'? (yes/no)"
            if ($fix -eq 'yes') {
                $newContent = $content -replace 'POSTGRES_PASSWORD=.+', 'POSTGRES_PASSWORD=devpass'
                Set-Content -Path $mainEnvFile -Value $newContent
                Write-Host "✅ Updated $mainEnvFile" -ForegroundColor Green
            }
        } else {
            Write-Host "✅ $mainEnvFile already has 'devpass'" -ForegroundColor Green
        }
    }
}

# Step 5: Check ai-service .env
Write-Host "`nStep 5: Check ai-service .env...`n" -ForegroundColor Yellow

$aiServiceEnv = "ai-service\.env"
if (Test-Path $aiServiceEnv) {
    $content = Get-Content $aiServiceEnv -Raw
    
    if ($content -match 'POSTGRES_PASSWORD=(.+)') {
        $currentPassword = $matches[1].Trim().Trim('"').Trim("'")
        Write-Host "ai-service\.env password: $('*' * $currentPassword.Length) chars" -ForegroundColor White
        
        if ($currentPassword.Length -eq 8) {
            Write-Host "⚠️  THIS IS LIKELY YOUR PROBLEM!" -ForegroundColor Red
            
            $fix = Read-Host "Change to 'devpass'? (yes/no)"
            if ($fix -eq 'yes') {
                $newContent = $content -replace 'POSTGRES_PASSWORD=.+', 'POSTGRES_PASSWORD=devpass'
                Set-Content -Path $aiServiceEnv -Value $newContent
                Write-Host "✅ Updated ai-service\.env" -ForegroundColor Green
            }
        }
    }
} else {
    Write-Host "ℹ️  ai-service\.env not found" -ForegroundColor Blue
}

# Summary
Write-Host "`n==================================================================" -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "==================================================================`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Restart your terminal/IDE to clear any cached environment variables"
Write-Host "2. Run the diagnostic: poetry run python find_password_source.py"
Write-Host "3. Restart your FastAPI app: poetry run uvicorn app.main:app --reload"
Write-Host ""
Write-Host "If the problem persists:" -ForegroundColor Yellow
Write-Host "- Check app/config.py for env_file path"
Write-Host "- Search for any hardcoded 8-character passwords in your code"
Write-Host "- Check if Pydantic is loading from a different source"
Write-Host ""