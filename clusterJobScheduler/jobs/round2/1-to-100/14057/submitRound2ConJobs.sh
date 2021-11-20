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




#echo "srun --gres=gpu:1 14082_round2ContinueJob_file.sh &  "
srun --gres=gpu:1 14057_round2ContinueJob_file.sh &
#sleep 3
#srun --gres=gpu:1 round2ContinueJob_db.sh &

date
