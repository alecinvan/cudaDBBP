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
CUDADBBPDIR="/cluster/clusterPipeline/MonteCarlo/CUDADBBP_memory_opt"
SLURMROUND2="/cluster/clusterPipeline/MonteCarlo/CUDADBBP_memory_opt/clusterJobScheduler/jobs/round2"
SLURMJOBS="/cluster/clusterPipeline/MonteCarlo/CUDADBBP_memory_opt/clusterJobScheduler/jobs"
TRAINDATA="/cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/couples_1_to_100_train/couples_1_to_100_train.dat"
TESTDATA="/cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/couples_1_to_100_test/couples_1_to_100_test.dat"


#********************************************
# specifying the number of positive/negative
# result lines in training set
#********************************************
PLEN=1687100
NLEN=1687100

PLENTEST=4217
NLENTEST=421800


#*******************************
# specifying the training stage
#*******************************
STAGE=3

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
JOBID=4624
OUTCOMEID=15153

#******************************************
# the parameters inherited from round1
#******************************************
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
       EPOCH=100

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
        read U ALFA <<< $($SLURMJOBS/expRandomG/expRandomGen)

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

               echo -e "\n$CUDADBBPDIR/bin/cuda_nn_mry_opt_file $id $U $ALFA $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $PLEN $NLEN $PLENTEST $NLENTEST $LASTJOBID $LASTOUTCOMEID $LASTWEIGHTPATH\n"
               $CUDADBBPDIR/bin/cuda_nn_mry_opt_file $SLURMDEVICEID $U $ALFA $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $PLEN $NLEN $PLENTEST $NLENTEST $LASTJOBID $LASTOUTCOMEID $LASTWEIGHTPATH 2>&1 | tee $OUTCOMEDIR/`hostname`/$id/$SLURM_JOB_ID/output
#               sleep 3
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
           ## main branch ##
           #################

read_parameters
weight_file_name
#for (( c =1 ; c <=3; c++))
#do
          submit_job_node
#done





