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
#echo "srun --gres=gpu:1 round1jobs.sh & - first"
#srun --gres=gpu:1 jobs.sh &

##############test about passing arguments to job scripts#####
#JOBID=$1
#OUTCOMEID=$2
#echo $JOBID
#echo $OUTCOMEID
#./round2ContinueJob.sh $JOBID $OUTCOMEID &




echo "srun --gres=gpu:1 Cont_5.sh &  "
srun --gres=gpu:1 Cont_5.sh &

for (( c=1; c<=20 ; c++))
do
   echo "srun --gres=gpu:1 Rand_5.sh &  "
   srun --gres=gpu:1 Rand_5.sh &
##############  srun --gres=gpu:1 -w cuda01 round1jobs.sh &
   sleep 1
done


date
