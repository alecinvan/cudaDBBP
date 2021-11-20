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


echo -e "== one 2nd-round job starts (same parameter set) =="


#****************************
# specifying the directories
#****************************
OUTCOMEDIR="/cluster/clusterPipeline/MonteCarlo/OUTCOME"
CUDADBBPDIR="/cluster/clusterPipeline/MonteCarlo/CUDADBBP"
TRAINDATA="/cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/couples1day_dec_2012/couples1day_dec_2012_var_cid.dat"
TESTDATA="/cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/couples1day_dec_2012/couples1day_dec_2012_var_cid.dat"


#********************************************
# specifying the number of positive/negative
# result lines in training set
#********************************************
PLEN=1223677
NLEN=1223678


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
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
JOBID=1014
OUTCOMEID=6583

#******************************************
# the parameters inherited from round1
#******************************************
LASTJOBID=$JOBID


function read_parameters ()
{

     read INPUT FHIDDEN SHIDDEN U1 ALFA1 LASTWEIGHTPATH N_EPOCH <<< $(../readLastRoundParam/readLastRoundParam $JOBID $OUTCOMEID)

#     echo $INPUT
#     echo $FHIDDEN
#     echo $SHIDDEN
#     echo $U1
#     echo $ALFA1
#     echo $LASTWEIGHTPATH
#     echo $N_EPOCH

}


#******************************************
# the parameters can be fixed
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
        read U ALFA <<< $(../expRandomG/expRandomGen)
    #    echo $U
    #    echo $ALFA
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
       for id1 in $device
       do

               if [ -d $OUTCOMEDIR/`hostname`/$id1 ] ; then
                           echo -e "Directory $OUTCOMEDIR/`hostname`/$id1 exists"
               else
                           mkdir $OUTCOMEDIR/`hostname`/$id1
               fi

               if [ -d $OUTCOMEDIR/`hostname`/$id1/$SLURM_JOB_ID ] ; then
                           echo -e "Directory $OUTCOMEDIR/`hostname`/$id1/$SLURM_JOB_ID exists"
               else
                           mkdir $OUTCOMEDIR/`hostname`/$id1/$SLURM_JOB_ID
               fi

               echo -e "\n$CUDADBBPDIR/bin/cuda_bp_cluster_round2 $id1 $U1 $ALFA1 $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id1 $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $PLEN $NLEN $LASTJOBID $LASTWEIGHTPATH\n"
               $CUDADBBPDIR/bin/cuda_bp_cluster_round2 $SLURMDEVICEID $U1 $ALFA1 $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id1 $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $PLEN $NLEN $LASTJOBID $LASTWEIGHTPATH 2>&1 | tee $OUTCOMEDIR/`hostname`/$id1/$SLURM_JOB_ID/output
       done

}



           #################
           ## main branch ##
           #################

read_parameters
weight_file_name
submit_job_node






