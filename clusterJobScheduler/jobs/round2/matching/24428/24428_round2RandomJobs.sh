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
#   SUBMIT JOBS TO CUDA CLUSTER IN MONTE-CARLO          #
#   DISTRIBUTION THE LEARNING RATE u IS GENERATED       #
#   EXPONENTIALLY                                       #
#                                                       #
#   AUTHOR: SA LI                                       #
#   DATE:   4/5/2013                                    #
#                                                       #
#########################################################


echo -e "== one 4th-round job starts =="


#****************************
# specifying the directories
#****************************
OUTCOMEDIR="/cluster/clusterPipeline/MonteCarlo/OUTCOME"
CUDADBBPDIR="/cluster/clusterPipeline/MonteCarlo/CUDADBBP_memory_opt"
SLURMROUND2="/cluster/clusterPipeline/MonteCarlo/CUDADBBP_memory_opt/clusterJobScheduler/jobs/round2"
SLURMJOBS="/cluster/clusterPipeline/MonteCarlo/CUDADBBP_memory_opt/clusterJobScheduler/jobs"
TRAINDATA="/cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/matching/matching_10_24_2013/matching_10_24_2013_train.dat"
TESTDATA="/cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/matching/matching_10_24_2013/matching_10_24_2013_test.dat"


#********************************************
# specifying the number of positive/negative
# result lines in training set
#********************************************
PLEN=4354799
NLEN=4354800

PLENTEST=1088700
NLENTEST=1088700


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


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#  function to fix some parameter values  #
#  need to changed every group launch     #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
JOBID=6693
OUTCOMEID=24428

#**********************************************
# the parameters inherited from previous round
#**********************************************
LASTJOBID=$JOBID
LASTOUTCOMEID=$OUTCOMEID


function read_parameters ()
{

    read INPUT FHIDDEN SHIDDEN U1 ALFA1 LASTWEIGHTPATH N_EPOCH <<< $($SLURMROUND2/readLastRoundParam/readLastRoundParam $JOBID $OUTCOMEID)

#     echo $INPUT
#     echo $FHIDDEN
#     echo $SHIDDEN
#     echo $U1
#     echo $ALFA1
#     echo $LASTWEIGHTPATH
#     echo $N_EPOCH

}

#******************************************
# some parameters can be fixed
#******************************************
function parameter_fixed ()
{

       OUTPUT=1
       EPOCH=101

}

#**********************************
# specifying the weight file names
#**********************************
function weight_file_name ()
{

       parameter_fixed
       let NEWEPOCH=N_EPOCH+EPOCH

       TRAINEDWEIGHTSTR="/TrainedWeights_"$NEWEPOCH"_epochs.dat"
       TRAINEDWEIGHTLASTEPOCH="/TrainedWeights_"$N_EPOCH"_epochs.dat"
}


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#  function to randomize some parameters #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
function parameter_setup ()
{
        read U ALFA <<< $($SLURMJOBS/expRandomG/expRandomGen2)

}




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#  function to throw single job #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
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

         #      print_arguments

               echo -e "\n$CUDADBBPDIR/bin/matching/matching_id_78 $id $U $ALFA $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $PLEN $NLEN $PLENTEST $NLENTEST $LASTJOBID $LASTOUTCOMEID $LASTWEIGHTPATH\n"
               $CUDADBBPDIR/bin/matching/matching_id_78 $SLURMDEVICEID $U $ALFA $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $PLEN $NLEN $PLENTEST $NLENTEST $LASTJOBID $LASTOUTCOMEID $LASTWEIGHTPATH 2>&1 | tee $OUTCOMEDIR/`hostname`/$id/$SLURM_JOB_ID/output
               sleep 3
      done

}


function print_arguments ()
{
         echo $id
         echo $U
         echo $ALFA
         echo $INPUT
         echo $FHIDDEN
         echo $SHIDDEN
         echo $OUTPUT
         echo $EPOCH
         echo $TRAINDATA
         echo $TESTDATA
         echo $OUTCOMEDIR
         echo $id
         echo $SLURM_JOB_ID
         echo $STAGE
         echo $TRAINEDWEIGHTSTR
         echo $TRAINEDWEIGHTLASTEPOCH
         echo $PLEN
         echo $NLEN
         echo $LASTJOBID
         echo $LASTWEIGHTPATH

}



           #################
           ## MAIN BRANCH ##
           #################

read_parameters
weight_file_name
submit_job_node






