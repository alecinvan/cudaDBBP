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
CUDADBBPDIR="/cluster/clusterPipeline/MonteCarlo/CUDADBBP"
#TRAINDATA="/cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/couples1day_dec_2012/couples1day_dec_2012_var_cid.dat"
#TESTDATA="/cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/couples1day_dec_2012/couples1day_dec_2012_var_cid.dat"
TRAINDATA="/cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/couples_1_to_100_train/couples_1_to_100_train.dat"
TESTDATA="/cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/couples_1_to_100_test/couples_1_to_100_test.dat"
TRAINEDWEIGHTSTR="/TrainedWeights_120_epochs.dat"
TRAINEDWEIGHTLASTEPOCH="/TrainedWeights_120_epochs.dat"
PLEN=1687100
NLEN=1687100
PLENTEST=4217
NLENTEST=421800
STAGE=1
SLURMDEVICEID=0              #always fix as 0
LASTJOBID=0                  #set last round slurm job id as 0 always
LASTOUTCOMEID=0
LASTWEIGHTPATH="null"

function parameter_fixed ()
{
       INPUT=68
       OUTPUT=1
       EPOCH=121

#       echo $INPUT
#       echo $OUTPUT
#       echo $EPOCH

}

function parameter_setup ()
{
       FHIDDEN=$(($RANDOM%71 + 100))       #newrand=$[ ( $RANDOM % ( $[ $highest - $lowest ] + 1 ) ) + $lowest ]
       SHIDDEN=$(($RANDOM%71 + 70))

       v=$[100+(RANDOM%100)]$[1000+(RANDOM%1000)]
       k=$[100+(RANDOM%100)]$[1000+(RANDOM%1000)]
#       U=0.00${v:1:2}${v:4:3}
#       ALFA=$[RANDOM%3+2].${k:1:2}
       read U ALFA <<< $(../expRandomG/expRandomGen)

#       echo $FHIDDEN
#       echo $SHIDDEN
#       echo $U
#       echo $ALFA

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

#######            echo $SLURM_JOB_ID
#######            echo $PLEN
#######            echo $NLEN
#######            echo $STAGE
#######            echo $SLURMDEVICEID

                echo -e "\n$CUDADBBPDIR/bin/cuda_bp_cluster.file $id $U $ALFA $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $PLEN $NLEN $PLENTEST $NLENTEST $LASTJOBID $LASTOUTCOMEID $LASTWEIGHTPATH\n"
                $CUDADBBPDIR/bin/cuda_bp_cluster.file $SLURMDEVICEID $U $ALFA $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $PLEN $NLEN $PLENTEST $NLENTEST $LASTJOBID $LASTOUTCOMEID $LASTWEIGHTPATH 2>&1 | tee $OUTCOMEDIR/`hostname`/$id/$SLURM_JOB_ID/output
#               sleep 3
      done

}



           #################
           ## main branch ##
           #################

parameter_fixed
#parameter_setup
#for (( c =1 ; c <=3; c++))
#do
          submit_job_node
#done


