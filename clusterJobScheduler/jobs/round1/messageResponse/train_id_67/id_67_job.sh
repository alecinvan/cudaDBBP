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
TRAINDATA="/cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/mbyf/reply100k-mbyf/reply100k-mbyf-training-byid-50.dat"
TESTDATA="/cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/mbyf/reply100k-mbyf/reply100k-mbyf-testing-byid-50.dat"
TRAINEDWEIGHTSTR="/TrainedWeights_160_epochs.dat"
TRAINEDWEIGHTLASTEPOCH="/TrainedWeights_160_epochs.dat"
PLEN=155659
NLEN=311318
PLENTEST=84328
NLENTEST=311227
STAGE=1
SLURMDEVICEID=0              #always fix as 0
LASTJOBID=0                  #set last round slurm job id as 0 always
LASTOUTCOMEID=0
LASTWEIGHTPATH="null"

function parameter_fixed ()
{
       INPUT=108
       OUTPUT=1
       EPOCH=161

#       echo $INPUT
#       echo $OUTPUT
#       echo $EPOCH

}

function parameter_setup ()
{
       FHIDDEN=$(($RANDOM%101 + 160))       #newrand=$[ ( $RANDOM % ( $[ $highest - $lowest ] + 1 ) ) + $lowest ]
       SHIDDEN=$(($RANDOM%101 + 115))

#       v=$[100+(RANDOM%100)]$[1000+(RANDOM%1000)]
#       k=$[100+(RANDOM%100)]$[1000+(RANDOM%1000)]
#       U=0.00${v:1:2}${v:4:3}
#       ALFA=$[RANDOM%3+2].${k:1:2}
       read U ALFA <<< $(/cluster/clusterPipeline/MonteCarlo/CUDADBBP_memory_opt/clusterJobScheduler/jobs/expRandomG/expRandomGen)

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

                echo -e "\n$CUDADBBPDIR/bin/messageResponse/messageResponse_67 $id $U $ALFA $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $PLEN $NLEN $PLENTEST $NLENTEST $LASTJOBID $LASTOUTCOMEID $LASTWEIGHTPATH\n"
                $CUDADBBPDIR/bin/messageResponse/messageResponse_67 $SLURMDEVICEID $U $ALFA $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $PLEN $NLEN $PLENTEST $NLENTEST $LASTJOBID $LASTOUTCOMEID $LASTWEIGHTPATH 2>&1 | tee $OUTCOMEDIR/`hostname`/$id/$SLURM_JOB_ID/output
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


