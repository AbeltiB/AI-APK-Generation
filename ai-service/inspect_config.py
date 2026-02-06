"""
Inspect how your config module loads environment variables
"""
import sys
import os
from pathlib import Path

print("\n" + "=" * 70)
print("CONFIG MODULE INSPECTION")
print("=" * 70 + "\n")

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

try:
    # Try to import and inspect the config
    from app import config
    
    print("✅ Successfully imported app.config\n")
    
    # List all attributes
    print("Available attributes:")
    for attr in dir(config):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    print("\n" + "=" * 70)
    print("SETTINGS OBJECT")
    print("=" * 70 + "\n")
    
    if hasattr(config, 'settings'):
        settings = config.settings
        
        print(f"Type: {type(settings)}")
        print(f"\nDatabase Configuration:")
        
        attrs_to_check = [
            'postgres_host',
            'postgres_port',
            'postgres_db',
            'postgres_user',
            'postgres_password',
            'database_url',
        ]
        
        for attr in attrs_to_check:
            if hasattr(settings, attr):
                value = getattr(settings, attr)
                if 'password' in attr.lower():
                    display = f"{'*' * len(str(value))} (length: {len(str(value))})"
                else:
                    display = value
                print(f"  {attr:30s} = {display}")
        
        # Check Config class
        print(f"\n" + "=" * 70)
        print("SETTINGS CONFIG")
        print("=" * 70 + "\n")
        
        if hasattr(settings, '__class__'):
            cls = settings.__class__
            print(f"Class: {cls.__name__}")
            
            if hasattr(cls, 'Config'):
                config_cls = cls.Config
                print(f"\nConfig class attributes:")
                for attr in dir(config_cls):
                    if not attr.startswith('_'):
                        value = getattr(config_cls, attr, None)
                        print(f"  {attr:30s} = {value}")
        
        # Check if it's using Pydantic
        print(f"\n" + "=" * 70)
        print("PYDANTIC INFO")
        print("=" * 70 + "\n")
        
        try:
            from pydantic import BaseSettings
            if isinstance(settings, BaseSettings):
                print("✅ Using Pydantic BaseSettings")
                
                # Try to see where values came from
                if hasattr(settings, 'model_fields'):
                    print("\nModel fields:")
                    for name, field in settings.model_fields.items():
                        if 'postgres' in name.lower() or 'database' in name.lower():
                            print(f"  {name}: {field}")
        except ImportError:
            print("ℹ️  Not using Pydantic BaseSettings")
            
    else:
        print("❌ No 'settings' object found in config module")
        print("\nTry importing directly:")
        print("  from app.config import Settings")
        
except ImportError as e:
    print(f"❌ Could not import app.config: {e}")
    print("\nMake sure you're in the correct directory (ai-service)")
    print("Current directory:", Path.cwd())
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70 + "\n")

print("1. Run this script from ai-service directory:")
print("   cd ai-service")
print("   poetry run python find_password_source.py")
print()
print("2. Check your app/config.py or app/settings.py file")
print()
print("3. Most likely issues:")
print("   - env_file pointing to wrong path")
print("   - Multiple .env files with different passwords")
print("   - System environment variable overriding .env")
print()