##########################################################
#
#  This is a script to send first round training jobs to 
#  cuda cluster in batch
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
#echo "srun --gres=gpu:1 round1jobs.sh & - first"
#srun --gres=gpu:1 jobs.sh &


for (( c=1; c<=5 ; c++))
do
   echo "srun --gres=gpu:1 round1jobs.sh &  "
   srun --gres=gpu:1 ./round1jobs_mry_file.sh &
#   sleep 3
#   srun --gres=gpu:1 ./round1jobs_db.sh &
#   srun --gres=gpu:1 -w cuda10 round1jobs.sh &
   sleep 3
done


date

