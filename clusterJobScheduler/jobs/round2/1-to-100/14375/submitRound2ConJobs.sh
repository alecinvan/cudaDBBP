##########################################################
#
#  This is a script to throw second round training jobs 
#  to cuda cluster in batch.
#
#  Calling my job description script which has applied
#  Monte Carlo algorithm to run the cuda code in cluster.
#  Each job is equipped with different parameter set.
#
#  Author: Sa Li
#  Date: 2/14/2013
#  PlentyofFish Media Inc.
##########################################################

#!/bin/bash -l

date

echo "srun --gres=gpu:1 -w cuda06 14375_round2ContinueJob_file.sh &  "
srun --gres=gpu:1 -w cuda06 14375_round2ContinueJob_file.sh &

date
