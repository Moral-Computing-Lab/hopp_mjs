#!/bin/bash
#SBATCH --partition=mnl-all               # Specify the partition
#SBATCH --array=1-64%15                   # Job array range
#SBATCH --ntasks=1                        # Number of CPUs per task
#SBATCH --cpus-per-task=7                 # Number of CPUs per task
#SBATCH --job-name=MJS                    # Job name
#SBATCH --exclude=babbage,ada             # Exclude specific nodes
#SBATCH --output=/srv/lab/fmri/mft/fhopp_diss/analysis/signature/code/logs/glm/sub-%a.out
#SBATCH --time=01:00:00                   # Maximum time limit
#SBATCH --requeue

echo "Running job on: $SLURM_JOB_NODELIST"

/home/fhopp/.conda/envs/mjs/bin/python -u /srv/lab/fmri/mft/fhopp_diss/analysis/signature/code/glm/03_glm_trial.py -subject $(printf "%02d" $SLURM_ARRAY_TASK_ID)