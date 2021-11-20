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

##############test about passing arguments to job scripts#####
#JOBID=$1
#OUTCOMEID=$2
#echo $JOBID
#echo $OUTCOMEID
#./round2ContinueJob.sh $JOBID $OUTCOMEID &


for (( c=1; c<=10 ; c++))
do
   echo "srun --gres=gpu:1 25012_round2RandomJobs.sh &  "
   srun --gres=gpu:1 25012_round2RandomJobs.sh &
   sleep 3
done


date
