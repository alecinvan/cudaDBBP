#!/bin/bash

##############################################################
#                       \\\\ ////                            #
#                      \\  - -  //                           #
#                          @ @                               #
#                  ---oOOo-( )-oOOo---                       #
#                 ||        =        ||                      #
#                          |||                               #
##############################################################
#     submit jobs to cluster in Monte Carlo distribution     #
#     the learning rate u is generated exponentially         #
#                                                            #
#     Author:  Sa Li                                         #
#     Date:    04/01/2013                                    #
#                                                            #
##############################################################


echo -e "== one 1st-round job starts =="

OUTCOMEDIR="/cluster/clusterPipeline/MonteCarlo/OUTCOME"
CUDADBBPDIR="/cluster/clusterPipeline/MonteCarlo/CUDADBBP_MEM_OPT_MOMENTUM"
TRAINDATA="/cluster/projects/cuda_neural_net/train_data_set/data/csv/regressionNN/usersTimeOnSite_binary_train_dataset.dat"
TESTDATA="/cluster/projects/cuda_neural_net/train_data_set/data/csv/regressionNN/usersTimeOnSite_binary_test_dataset.dat"
TRAINEDWEIGHTSTR="/TrainedWeights_100_epochs.dat"
TRAINEDWEIGHTLASTEPOCH="/TrainedWeights_100_epochs.dat"

STAGE=1
SLURMDEVICEID=0              #always fix as 0
LASTJOBID=0                  #set last round slurm job id as 0 always
LASTOUTCOMEID=0
LASTWEIGHTPATH="null"

MEAN=0.470229
MEAN_TEST=0.470285


function parameter_fixed ()
{
       INPUT=57
       OUTPUT=1
       EPOCH=101

#       echo $INPUT
#       echo $OUTPUT
#       echo $EPOCH

}

function parameter_setup ()
{
       FHIDDEN=$(($RANDOM%30 + 100))       #newrand=$[ ( $RANDOM % ( $[ $highest - $lowest ] + 1 ) ) + $lowest ]
       SHIDDEN=$(($RANDOM%20 + 60))

       ALFA=1
       read U BETA <<< $(/cluster/clusterPipeline/MonteCarlo/CUDADBBP_MEM_OPT_MOMENTUM/clusterJobScheduler/jobs/expRandomG/expRandomGen)
}


function submit_job_node()
{

       echo `hostname` - $CUDA_VISIBLE_DEVICES

       if [  -d $OUTCOMEDIR/`hostname` ] ; then
               echo -e "Directory $OUTCOMEDIR/`hostname` exists"
       else
               mkdir $OUTCOMEDIR/`hostname`
       fi

       device=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n")
       for id in $device
       do

               parameter_setup

               if [ -d $OUTCOMEDIR/`hostname`/$id ] ; then
                           echo -e "Directory $OUTCOMEDIR/`hostname`/$id exists"
               else
                           mkdir $OUTCOMEDIR/`hostname`/$id
               fi

               if [ -d $OUTCOMEDIR/`hostname`/$id/$SLURM_JOB_ID ] ; then
                           echo -e "Directory $OUTCOMEDIR/`hostname`/$id/$SLURM_JOB_ID exists"
               else
                           mkdir $OUTCOMEDIR/`hostname`/$id/$SLURM_JOB_ID
               fi

                echo -e "\n$CUDADBBPDIR/bin/regressionNN/regression_id_88 $id $U $ALFA $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $BETA $NLEN $MEAN_TEST $LASTJOBID $LASTOUTCOMEID $LASTWEIGHTPATH \n"
                $CUDADBBPDIR/bin/regressionNN/regression_id_88 $SLURMDEVICEID $U $ALFA $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $BETA $MEAN $MEAN_TEST $LASTJOBID $LASTOUTCOMEID $LASTWEIGHTPATH  2>&1 | tee $OUTCOMEDIR/`hostname`/$id/$SLURM_JOB_ID/output
#               sleep 3
      done

}



           #################
           ## main branch ##
           #################

parameter_fixed
submit_job_node



