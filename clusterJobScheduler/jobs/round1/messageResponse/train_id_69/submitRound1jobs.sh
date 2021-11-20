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

for (( c=1; c<=10 ; c++))
do
   echo "srun --gres=gpu:1 id_69_job.sh &  "
   srun --gres=gpu:1 ./id_69_job.sh &

   sleep 3
done


date

