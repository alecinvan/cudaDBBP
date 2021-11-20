#!/bin/bash

OUTCOMEDIR="/cluster/clusterPipeline/MonteCarlo/OUTCOME"
CUDADBBPDIR="/cluster/clusterPipeline/MonteCarlo/CUDADBBP"
TRAINDATA="/cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/couples1day_dec_2012/couples1day_dec_2012_var_cid.dat"
TESTDATA="/cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/couples1day_dec_2012/couples1day_dec_2012_var_cid.dat"
TRAINEDWEIGHTSTR="/TrainedWeight.dat"
TRAINEDWEIGHTLASTEPOCH="/TrainedWeight.dat"
PLEN=1223677
NLEN=1223678
STAGE=1
SLURMDEVICEID=0   #always fix as 0

function parameter_fixed ()
{
       INPUT=68
       OUTPUT=1
       EPOCH=100

#       FHIDDEN=150
#       SHIDDEN=100
#       U=0.00003
#       ALFA=2
}

function parameter_setup ()
{
       FHIDDEN=$(($RANDOM%21 + 140))       #newrand=$[ ( $RANDOM % ( $[ $highest - $lowest ] + 1 ) ) + $lowest ]

               echo -e "Directory $OUTCOMEDIR/`hostname` exists"
       SHIDDEN=$(($RANDOM%21+90))
       v=$[100+(RANDOM%100)]$[1000+(RANDOM%1000)]
       U=0.00${v:1:2}${v:4:3}
       ALFA=$[RANDOM%2+2].${v:1:2}
#       echo $FHIDDEN
#       echo $SHIDDEN
#       echo $v
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

#            echo $SLURM_JOB_ID
#            echo $PLEN
#            echo $NLEN
#            echo $STAGE
#            echo $SLURMDEVICEID
               echo -e "\n$CUDADBBPDIR/bin/cuda_bp_cluster $id $U $ALFA $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $PLEN $NLEN\n"
               $CUDADBBPDIR/bin/cuda_bp_cluster $SLURMDEVICEID $U $ALFA $INPUT $FHIDDEN $SHIDDEN $OUTPUT $EPOCH $TRAINDATA $TESTDATA $OUTCOMEDIR `hostname` $id $SLURM_JOB_ID $STAGE $TRAINEDWEIGHTSTR $TRAINEDWEIGHTLASTEPOCH $PLEN $NLEN 2>&1 | tee $OUTCOMEDIR/`hostname`/$id/$SLURM_JOB_ID/output
#               sleep 3
      done

}



#########main branch########

parameter_fixed
#parameter_setup
#for (( c =1 ; c <=3; c++))
#do
          submit_job_node
#done
