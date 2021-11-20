# This makefile uses nvcc as the compiler 
CUDA_COMPILER=nvcc

# Pass CUDAFLAGS to compiler
CUDAFLAGS=-arch sm_13

# Use openMP to parallel CPU threads
# with flags: 
#             -Xcompiler 
#             -fopenmp 
#             -lgomp
OPENMPFLAGS=-Xcompiler -fopenmp -lgomp

# Apply cublas API and CUDA rand for 
# replace my kernels with flag
#             -lcublas
CUBLASFLAG=-lcublas 

# Generate random number inside GPUs 
# with flags:
#             -lcurand 
CUDARANDFLAG=-lcurand

# Connect PostgreSQL with flag
#             -lpq
PSQLFLAG=-lpq

# Output flag
#             -o
OUTPUTFLAG=-o 


# Tell the compiler where the source 
# cuda code is
SOURCE=./src/cuda_bp.cu

# Tell the compiler what is the 
# output file
EXECUTABLE=./bin/cuda_bp_cluster



cuda_bp_cluster_round2: $(SOURCE)
	$(CUDA_COMPILER) $(CUDAFLAGS) $(OPENMPFLAG) $(OUTPUTFLAG) $(EXECUTABLE) $(SOURCE) $(PSQLFLAG) $(CUBLASFLAG) $(CUDARANDFLAG) 


clean:
