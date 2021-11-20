#!/bin/bash

date
#########################################################
#                       \\\\ ////                       #
#                      \\  - -  //                      #
#                          @ @                          #
#                  ---oOOo-( )-oOOo---                  #
#                 ||        =        ||                 #
#########################################################
#                                                       #
#   submit jobs to cluster in Monte Carlo distribution  #
#   the learning rate u is generated exponentially      #
#                                                       #
#   Author: Sa Li                                       #
#   Date:   4/5/2013                                    #
#                                                       #
#########################################################


echo -e "== one 2nd-round job starts =="


#****************************
# specifying the directories
#****************************
OUTCOMEDIR="/cluster/clusterPipeline/MonteCarlo/OUTCOME"
CUDADBBPDIR="/cluster/clusterPipeline/MonteCarlo/CUDADBBP"
TRAINDATA="/cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/couples1day_dec_2012/couples1day_dec_2012_var_cid.dat"
TESTDATA="/cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/couples1day_dec_2012/couples1day_dec_2012_var_cid.dat"

#**********************************
# specifying the weight file names
#**********************************
TRAINEDWEIGHTSTR="/TrainedWeights_160_epochs.dat"
TRAINEDWEIGHTLASTEPOCH="/TrainedWeights_160_epochs.dat"

#********************************************
# specifying the number of positive/negative
# result lines in training set
#********************************************
PLEN=1223677
NLEN=1223678


#*******************************
# specifying the training stage
#*******************************
STAGE=2

#***************************************
# default the slurm deviceid = 0 always,
# and slurm will automatically assign
# the jobs to different gpu
#***************************************
SLURMDEVICEID=0

#******************************************
# the parameters inherited from round1
#******************************************
LASTJOBID=0
LASTWEIGHTPATH="null"



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#  function to fix some parameter values  #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
JOBID=$1
OUTCOMEID=$2

read INPUT FHIDDEN SHIDDEN U1 ALFA1 N_EPOCH <<< $(./readLastRoundParam $JOBID $OUTCOMEID)

echo $INPUT
echo $FHIDDEN
echo $SHIDDEN
echo $U1
echo $ALFA1
echo $N_EPOCH

function parameter_fixed ()
{
#       INPUT=68
       OUTPUT=1
       EPOCH=100

#       echo $INPUT
#       echo $OUTPUT
#       echo $EPOCH

}

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#  function to randomize some parameters #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
function parameter_setup ()
{
        read U ALFA <<< $(../expRandomG/expRandomGen)

#        read FHIDDEN SHIDDEN LASTJOBID LASTWEIGHTPATH  <<< $(./readLastRoundParam)

#        echo $U
#        echo $ALFA

#        FHIDDEN
#        SHIDDEN
#        LASTJOBID
#        LASTWEIGHTPATH

}

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#  function to throw single job #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
function submit_job_node()
{

       echo `hostname` - $CUDA_VISIBLE_DEVICES

       if [  -d $OUTCOMEDIR/`hostname` ] ; then
               echo -e "Directory $OUTCOMEDIR/`hostname` exists"
       elseqq
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

                echo -e "\n$CUDADBBPDIR/bin/cuda_bp_cluster $id $U $ALFA $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $PLEN $NLEN $LASTJOBID $LASTWEIGHTPATH\n"
               $CUDADBBPDIR/bin/cuda_bp_cluster $SLURMDEVICEID $U $ALFA $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $PLEN $NLEN $LASTJOBID $LASTWEIGHTPATH 2>&1 | tee $OUTCOMEDIR/`hostname`/$id/$SLURM_JOB_ID/output
#               sleep 3
      done

}



           #################
           ## main branch ##
           #################

parameter_fixed
#for (( c =1 ; c <=3; c++))
#do
          submit_job_node
#done





