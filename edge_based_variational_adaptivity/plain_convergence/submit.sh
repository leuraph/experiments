#!/bin/bash

#SBATCH --job-name="exp-02_theta-0.1"

# number of tasks
#SBATCH -n 1

# number of cpus per task
#SBATCH --cpus-per-task=8

# (2 * 24(h) * 60(min) = 2880)
#SBATCH --time=2880

# Email when job is done or failed
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=raphael.leu@unibe.ch

module purge
module load Workspace_Home

source .venv/bin/activate
python experiment_02.py --theta 0.1