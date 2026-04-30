#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set the PYTHONPATH environment variable for the subprocess
env = os.environ.copy()
env['PYTHONPATH'] = str(project_root) + os.pathsep + env.get('PYTHONPATH', '')

# Change to project root directory
os.chdir(project_root)

# Now run the Streamlit app with the environment
import subprocess
subprocess.run(["streamlit", "run", "UI/main.py"], env=env)