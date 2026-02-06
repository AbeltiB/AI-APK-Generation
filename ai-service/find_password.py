"""
Find where the wrong password is being loaded from
"""
import os
from pathlib import Path

print("\n" + "=" * 70)
print("PASSWORD SOURCE DIAGNOSTIC")
print("=" * 70 + "\n")

# Check all possible .env files
env_files = [
    ".env",
    "../.env",
    "../../.env",
    ".env.local",
    ".env.development",
    ".env.prod",
    ".env.production",
    "../infrastructure/.env",
]

print("Checking all possible .env files:\n")

for env_file in env_files:
    path = Path(env_file)
    if path.exists():
        print(f"✅ Found: {path.absolute()}")
        
        # Read the file
        content = path.read_text()
        
        # Find POSTGRES_PASSWORD
        for line in content.split('\n'):
            if 'POSTGRES_PASSWORD' in line and not line.strip().startswith('#'):
                # Mask the password but show length
                if '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"').strip("'")
                    print(f"   {key}={'*' * len(value)} (length: {len(value)})")
                    
                    if len(value) == 8:
                        print(f"   ⚠️  THIS IS THE 8-CHARACTER PASSWORD!")
                        print(f"   File: {path.absolute()}")
        print()
    else:
        print(f"❌ Not found: {env_file}")

print("\n" + "=" * 70)
print("ENVIRONMENT VARIABLES (Current Process)")
print("=" * 70 + "\n")

password = os.getenv('POSTGRES_PASSWORD', '')
print(f"POSTGRES_PASSWORD = {'*' * len(password)} (length: {len(password)})")

if len(password) == 8:
    print("\n⚠️  The environment variable is 8 characters!")
    print("This might be set in:")
    print("  1. System environment variables")
    print("  2. PowerShell profile")
    print("  3. Docker environment")
    print("  4. Another .env file that's being loaded")

print("\n" + "=" * 70)
print("CHECKING YOUR SETTINGS MODULE")
print("=" * 70 + "\n")

try:
    from app.config import settings
    
    password = settings.postgres_password
    print(f"settings.postgres_password = {'*' * len(password)} (length: {len(password)})")
    
    if len(password) == 8:
        print("\n⚠️  Settings module is loading an 8-character password!")
        print("\nYour Settings class is likely:")
        print("  1. Loading from a different .env file")
        print("  2. Using env_file parameter pointing to wrong file")
        print("  3. Getting overridden by system environment variable")
        
    # Check if settings has env_file configured
    if hasattr(settings, 'Config'):
        config = settings.Config
        if hasattr(config, 'env_file'):
            print(f"\nSettings.Config.env_file = {config.env_file}")
            
except ImportError as e:
    print(f"❌ Could not import settings: {e}")

print("\n" + "=" * 70)
print("SOLUTION")
print("=" * 70 + "\n")

print("If the 8-character password is in a different .env file:")
print("  1. Update that file to use 'devpass' (7 chars)")
print("  2. Or delete that file and use only one .env file")
print()
print("If it's a system environment variable:")
print("  PowerShell: $env:POSTGRES_PASSWORD = 'devpass'")
print("  Or restart your terminal/IDE")
print()
print("If it's in docker-compose:")
print("  Check docker-compose files for POSTGRES_PASSWORD")
print()