############################
# COMPILE CUDA APPLICATION #
# AUTHOR: Sa Li            #
# DATE:   6/20/2013        #
############################


# THIS MAKEFILE USES NVCC AS THE COMPILER 
CUDA_COMPILER=nvcc


# PASS CUDAFLAGS TO COMPILER
CUDAFLAGS=-arch sm_13


# PASS OPENMPFLAGS TO COMPILER TO PARALLEL
# CPU THREADS WITH FLAGS: 
#             -Xcompiler 
#             -fopenmp 
#             -lgomp
OPENMPFLAGS=-Xcompiler -fopenmp -lgomp


# PASS CUBLASFLAG TO COMPILER TO USE CUBLAS
# API FOR REPLACING MY KERNELS WITH FLAG:
#             -lcublas
CUBLASFLAG=-lcublas


# PASS CUDARANDFLAG TO COMPILER TO GENERATE 
# RANDOM NUMBER INSIDE GPU DEVICES  
# WITH FLAGS:
#             -lcurand 
CUDARANDFLAG=-lcurand


# PASS PSQLFLAG TO COMPILER TO CONNECT 
# POSTGRESQL DATABASE WITH FLAG:
#             -lpq
PSQLFLAG=-lpq


# PASS OUTPUTFLAG TO COMPILER:
#             -o
OUTPUTFLAG=-o



# TELL THE COMPILER WHERE THE SOURCE CUDA 
# CODE IS:
SOURCE=./src/cuda_bp.cu

# TELL THE COMPILER WHAT IS THE OBJECT 
# FILE:
EXECUTABLE=./bin/cuda_bp_cluster



cuda_bp_cluster_round2: $(SOURCE)
#	$(CUDA_COMPILER) $(CUDAFLAGS) $(OPENMPFLAGS) $(OUTPUTFLAG) $(EXECUTABLE) $(SOURCE) $(PSQLFLAG) $(CUBLASFLAG) $(CUDARANDFLAG) 
	$(CUDA_COMPILER) $(CUDAFLAGS) $(OUTPUTFLAG) $(EXECUTABLE) $(SOURCE) $(PSQLFLAG) $(CUBLASFLAG) $(CUDARANDFLAG)

clean:



