#!/bin/bash

date
echo "----"
MAINDIR="/cluster/clusterPipeline/MonteCarlo/OUTCOME/*"


for node in $MAINDIR
do
   # if [ -d "$node" ]
   # then
         for device in $node/*
         do
                for job in $device/*/
                do
                       #echo $job
                       if [ -f "$job/weightMatrix/TrainedWeight_100_epochs.dat" ]
                       then
                                 mv  $job/weightMatrix/TrainedWeight_100_epochs.dat $job/weightMatrix/TrainedWeights_100_epochs.dat
                       fi
                done
         done

done

echo "----"
date
