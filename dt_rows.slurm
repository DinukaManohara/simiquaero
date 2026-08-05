#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Tue Aug 04 2026 00:49:46 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=sapphire

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="dt_rows"

# The project ID which this job should run under:
#SBATCH --account="punim2258"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# The amount of memory in megabytes per node:
#SBATCH --mem=65536

# Use this email address:
#SBATCH --mail-user=bdezoysa@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-12:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load GCCcore/11.3.0
module load CUDA/12.4.1
module load UCX-CUDA/1.16.0-CUDA-12.4.1
module load cuDNN/9.6.0.74-CUDA-12.4.1

# The job command(s):
./run_dt_rows.sh

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
