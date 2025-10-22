import subprocess

template = """#!/bin/bash

#SBATCH --job-name="NCG"

# number of tasks
#SBATCH -n1

# number of cpus per task
#SBATCH --cpus-per-task=8

# memory per CPU
#SBATCH --mem=0

# (4 * 24(h) * 60(min) = 5760)
#SBATCH --time=5760

# Email when done or failed
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=raphael.leu@unibe.ch

module purge
module load Workspace_Home

source .venv/bin/activate
{command}
"""

# Read commands from file
with open("experiments.txt") as f:
    commands = [line.strip() for line in f if line.strip()]

for i, cmd in enumerate(commands, start=1):
    # Fill in the placeholders
    job_script = template.format(command=cmd)

    # Write to a temporary sbatch file
    script_name = f"tmp{i}.sh"
    with open(script_name, "w") as f:
        f.write(job_script)

    # Submit with sbatch
    subprocess.run(["sbatch", script_name])

    # Optional: remove the temp file after submission
    # os.remove(script_name)