
#ifndef    _CUDA_KERNEL_FUNCTIONS_H_
#define    _CUDA_KERNEL_FUNCTIONS_H_

                 //////////////////
                 // cuda kernels //
                 //////////////////

__device__ float sigmoidActivFuncComput(float arrayElement, float alfa);
__device__ float outputLayerDelta(float desOutValue, float actualOutValue, float alfa);

__global__ void hiddenLayerOutputCalculation(float* A, float* B, int N, float alfa);
__global__ void outputLayerOutputCalculation(float* A, float* B, int N, float alfa);
__global__ void deltaOutputLayerCalculation(float* C, float* A, float* B, int N, float alfa);
__global__ void deltaHiddenLayerCalculation(float* D, float* A, float* B, float* C, int wA, int wB, float alfa);
__global__ void layerWeightUpdate(float* C, float* A, float* B, int wA, int wB, float u);

void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) ;
void GPU_fill_rand(float *A, int rows_A, int cols_A) ;



#endif
