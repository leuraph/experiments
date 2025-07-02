import os
import itertools
import subprocess
import argparse

# Define parameter lists
theta_values = [0.5]
fudge_values = [1.0, 0.1, 0.01]
delay_values = [10]
miniter_values = [5, 10]
delay_increase_values = [10]
tau_values = [1.01]

# Generate all combinations of parameters
combinations = list(itertools.product(
    theta_values,
    fudge_values,
    delay_values,
    miniter_values,
    delay_increase_values,
    tau_values))

# Directory for generated scripts
output_dir = "sbatch_scripts"
os.makedirs(output_dir, exist_ok=True)

# Template for the SBATCH script
sbatch_template = """#!/bin/bash

#SBATCH --job-name="{job_name}"

# number of tasks
#SBATCH -n 1

# number of cpus per task
#SBATCH --cpus-per-task=8

# (4 * 24(h) * 60(min) = 2880)
#SBATCH --time=5760

# Email when job is done or failed
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=raphael.leu@unibe.ch

module purge
module load Workspace_Home

source .venv/bin/activate
python experiment_01.py --theta {theta} --fudge {fudge} --initial_delay {delay} --miniter {miniter} --tau {tau} --delay_increase {delay_increase}
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Submit jobs with optional debug mode.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode (do not submit jobs, just generate scripts).')
args = parser.parse_args()

# Generate and submit scripts
for i, (theta, fudge, delay, miniter, delay_increase, tau) in enumerate(combinations):
    # Create a unique job name
    job_name = f"exp-01_theta-{theta}_fudge-{fudge}_delay-{delay}_miniter-{miniter}_delay_increase-{delay_increase}_tau-{tau}"
    
    # Generate the script content
    sbatch_content = sbatch_template.format(
        job_name=job_name,
        theta=theta,
        fudge=fudge,
        delay=delay,
        miniter=miniter,
        delay_increase=delay_increase,
        tau=tau
    )
    
    # Save the script to a file
    script_path = os.path.join(output_dir, f"submit_{i}.sl")
    with open(script_path, "w") as f:
        f.write(sbatch_content)
    
    # Submit the job if not in debug mode
    if not args.debug:
        subprocess.run(["sbatch", script_path])
    else:
        print("----------------------------------------")
        print("The following would haven been sbatched:")
        print("----------------------------------------")
        print(sbatch_content)
