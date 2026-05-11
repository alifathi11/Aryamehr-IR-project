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

import subprocess
subprocess.run(["python3", "Logic/preprocess.py"], env=env)
subprocess.run(["python3", "Logic/indexer/index.py"], env=env)
subprocess.run(["python3", "Logic/indexer/document_lengths_index.py"], env=env)
subprocess.run(["python3", "Logic/indexer/metadata_index.py"], env=env)
subprocess.run(["python3", "Logic/indexer/tiered_index.py"], env=env)
subprocess.run(["python3", "Logic/LSH.py"], env=env)