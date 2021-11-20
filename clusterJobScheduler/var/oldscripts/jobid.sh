#!/bin/bash -l 

#
# Sample TaskProlog script that will print a batch job's
# job ID and node list to the job's stdout
#

if [ X"$SLURM_STEP_ID" = "X" -a X"$SLURM_PROCID" = "X"0 ]
then
  echo "print =========================================="
  echo "print SLURM_JOB_ID = $SLURM_JOB_ID"
  echo "print SLURM_NODELIST = $SLURM_NODELIST"
  echo "print =========================================="
fi




