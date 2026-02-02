
import sys
import os

# Add the project root to python path
sys.path.append(os.getcwd())

try:
    print("Attempting to import app.services.generation.blockly_generator...")
    from app.services.generation import blockly_generator
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)
