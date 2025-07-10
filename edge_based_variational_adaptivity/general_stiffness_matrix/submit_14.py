import os
import itertools
import subprocess
import argparse

# Define parameter lists
theta_values = [0.5]
arioli_fudge_values = [1.0, 0.1]
tail_fudge_values = [0.1, 0.01]
miniter_values = [10]
initial_delay_values = [10]
delay_increase_values = [10]
tau_values = [1.01]

# Generate all combinations of parameters
combinations = list(itertools.product(
    theta_values,
    arioli_fudge_values,
    tail_fudge_values,
    miniter_values,
    initial_delay_values,
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

# (4 * 24(h) * 60(min) = 5760)
#SBATCH --time=5760

# Email when job is done or failed
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=raphael.leu@unibe.ch

module purge
module load Workspace_Home

source .venv/bin/activate
python experiment_14.py \
--theta {theta} \
--fudge_arioli {fudge_arioli} \
--fudge_tail {fudge_tail} \
--miniter {miniter} \
--initial_delay {initial_delay} \
--delay_increase {delay_increase} \
--tau {tau}
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Submit jobs with optional debug mode.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode (do not submit jobs, just generate scripts).')
args = parser.parse_args()

# Generate and submit scripts
for i, (theta, fudge_arioli, fudge_tail, miniter, initial_delay, delay_increase, tau) in enumerate(combinations):
    # Create a unique job name
    job_name = (
    f"exp-14_"
    f"theta-{theta}_"
    f"fudge_arioli-{fudge_arioli}_"
    f"fudge_tail-{fudge_tail}"
    f"miniter-{miniter}"
    f"initial_delay-{initial_delay}"
    f"delay_increase-{delay_increase}"
    f"tau-{tau}")

    # Generate the script content
    sbatch_content = sbatch_template.format(
        job_name=job_name,
        theta=theta,
        fudge_arioli=fudge_arioli,
        fudge_tail=fudge_tail,
        miniter=miniter,
        initial_delay=initial_delay,
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
