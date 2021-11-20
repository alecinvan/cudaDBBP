##########################################################
#
#  This is a script to send jobs to cuda cluster in batch
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

#echo "srun --gres=gpu:1 ./jobs/jobs.sh & - first"
#srun --gres=gpu:1 jobs.sh &


for (( c=1; c<=10 ; c++)) 
do
   echo "srun --gres=gpu:1 ./jobs/jobs.sh &  "
   srun --gres=gpu:1 ./jobs/jobs.sh &
#  srun --gres=gpu:1 -w cuda01 ./jobs/jobs.sh &
done


