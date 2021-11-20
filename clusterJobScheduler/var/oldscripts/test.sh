#!/bin/bash -l


#echo `hostname` - $CUDA_VISIBLE_DEVICES
#echo "Command: ./bin/cuda_bp_cluster $CUDA_VISIBLE_DEVICES 0.00003 1.2 66 150 100 1 100 /cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/couples1day/train_after_oversampling_var_cid.dat /cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/couples1day/train_after_oversampling_var_cid.dat /cluster/clusterPipeline/MonteCarlo/OUTCOME/cuda04/1 /TrainedWeight.dat /TrainedWeight.dat"
#ls -l /dev/nv*
#./bin/cuda_bp_cluster $CUDA_VISIBLE_DEVICES 0.00003 1.2 66 150 100 1 100 /cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/couples1day/train_after_oversampling_var_cid.dat /cluster/projects/cuda_neural_net/train_data_set/data/final/trainingdate/couples1day/train_after_oversampling_var_cid.dat /cluster/clusterPipeline/MonteCarlo/OUTCOME/cuda04/1 /TrainedWeight.dat /TrainedWeight.dat
#sleep 30


FHIDDEN=$(($RANDOM%61 + 120))
SHIDDEN=$(($RANDOM%61+70))

v=$[100 + (RANDOM % 100)]$[1000 + (RANDOM % 1000)]
m=0.${v:1:2}${v:4:3}
U=0.00${v:1:2}${v:4:3}
ALFA=$[RANDOM%3+1].${v:1:2}

echo $FHIDDEN
echo $SHIDDEN
#echo $v
echo $m
echo $U
echo $ALFA

