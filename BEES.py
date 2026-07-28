#!/usr/bin/env python3
r"""
Executable wrapper for BEES (Biochemical Engine for Enzymatic kinetic modelS)


For linux users:
Make sure this file is executable: chmod +x BEES.py

To run BEES from anywhere, add to your PATH (in ~/.bashrc or ~/.zshrc):
    export PATH="$PATH:$HOME/BEES"

Then you can run BEES like this:
cd ~/BEES
./BEES.py -i ~/BEES/projects/minimal/input.yml 


"""

import sys
import os
import re
import argparse

# Load .env.bees BEFORE importing bees modules so that class-level env vars
# (e.g. CATPRED_PYTHON in CatPredEstimator) are resolved at class-definition time.
_repo_root = os.path.dirname(os.path.abspath(__file__))
_env_bees = os.path.join(_repo_root, ".env.bees")
if os.path.isfile(_env_bees):
    with open(_env_bees) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
            if m:
                key, val = m.group(1), m.group(2).strip()
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                os.environ[key] = val

# Import necessary modules from BEES
import bees.common as common
from bees.main import BEES


BEES_PATH = common.BEES_PATH   

def _warn_if_not_bees_env() -> None:
    """
    Best-effort UX hint for local users.

    IMPORTANT: do not hard-exit at import time (breaks `import BEES` and tooling).
    """
    if "bees_env" not in sys.executable:
        print("Warning: it looks like you're not running inside the 'bees_env' environment.")

def parse_and_load_input() -> dict:
    parser = argparse.ArgumentParser(description="BEES")
    parser.add_argument("-i", "--input_file", type=str, help="Path to input YAML")
    parser.add_argument("-p", "--project", type=str, help="Project name")
    parser.add_argument("-v", "--verbose", type=int, choices=[10, 20, 30, 40, 50], help="Log level")
    parser.add_argument("-o", "--output_directory", type=str, help="Output directory")
    args = parser.parse_args()

    default_input_path = os.path.join(BEES_PATH, "projects", "minimal", "input.yml")
    input_path = args.input_file or default_input_path

    input_data = common.read_yaml_file(input_path)
    input_data.setdefault("settings", {})

    if args.project:
        input_data["project"] = args.project
    if args.verbose is not None:
        input_data["settings"]["verbose"] = args.verbose
    if args.output_directory:
        input_data["settings"]["output_directory"] = args.output_directory

    if not input_data.get("project"):
        raise ValueError("Project name is required!")

    return input_data

def main():
    """
    the main BEES excutable function
    """
    _warn_if_not_bees_env()
    input_data = parse_and_load_input()
    bees_instance = BEES(input_data=input_data)
    results = bees_instance.execute()
    
    # Print summary of results
    if results and results.get('success'):
        print(f"\n{'='*60}")
        print("BEES Execution Summary:")
        print(f"  Project: {results['project']}")
        print(f"  Reactions Generated: {results['n_reactions']}")
        print(f"  Execution Time: {results['execution_time']}")
        if results.get('summary_path'):
            print(f"  Summary: {results['summary_path']}")
        print(f"{'='*60}\n")
    else:
        print("\nBEES execution completed with warnings or errors.")
        if results and results.get('message'):
            print(f"  {results['message']}\n")
    
    return results


if __name__ == "__main__":
    main()
