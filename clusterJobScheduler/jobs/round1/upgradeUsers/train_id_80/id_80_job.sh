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
CUDADBBPDIR="/cluster/clusterPipeline/MonteCarlo/CUDADBBP_memory_opt"
TRAINDATA="/cluster/projects/cuda_neural_net/train_data_set/data/csv/userupgrade/combine/userData_important_train.dat"
TESTDATA="/cluster/projects/cuda_neural_net/train_data_set/data/csv/userupgrade/combine/userData_important_test.dat"
TRAINEDWEIGHTSTR="/TrainedWeights_100_epochs.dat"
TRAINEDWEIGHTLASTEPOCH="/TrainedWeights_100_epochs.dat"
PLEN=125432
NLEN=126135
PLENTEST=41990
NLENTEST=41865
STAGE=1
SLURMDEVICEID=0              #always fix as 0
LASTJOBID=0                  #set last round slurm job id as 0 always
LASTOUTCOMEID=0
LASTWEIGHTPATH="null"

function parameter_fixed ()
{
       INPUT=26
       OUTPUT=1
       EPOCH=101

#       echo $INPUT
#       echo $OUTPUT
#       echo $EPOCH

}

function parameter_setup ()
{
       FHIDDEN=$(($RANDOM%15 + 45))       #newrand=$[ ( $RANDOM % ( $[ $highest - $lowest ] + 1 ) ) + $lowest ]
       SHIDDEN=$(($RANDOM%10 + 30))

       read U ALFA <<< $(/cluster/clusterPipeline/MonteCarlo/CUDADBBP_memory_opt/clusterJobScheduler/jobs/expRandomG/expRandomGen)
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

                echo -e "\n$CUDADBBPDIR/bin/upgradeUser/upgradeUser_id_80 $id $U $ALFA $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $PLEN $NLEN $PLENTEST $NLENTEST $LASTJOBID $LASTOUTCOMEID $LASTWEIGHTPATH\n"
                $CUDADBBPDIR/bin/upgradeUser/upgradeUser_id_80 $SLURMDEVICEID $U $ALFA $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $PLEN $NLEN $PLENTEST $NLENTEST $LASTJOBID $LASTOUTCOMEID $LASTWEIGHTPATH 2>&1 | tee $OUTCOMEDIR/`hostname`/$id/$SLURM_JOB_ID/output
#               sleep 3
      done

}



           #################
           ## main branch ##
           #################

parameter_fixed
submit_job_node



