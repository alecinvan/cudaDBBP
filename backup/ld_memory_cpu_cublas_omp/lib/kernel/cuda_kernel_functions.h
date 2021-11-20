/*******************************
 *  The GPU kernels
 *  Author: Sa Li
*******************************/

#ifndef    _CUDA_KERNEL_FUNCTIONS_H_
#define    _CUDA_KERNEL_FUNCTIONS_H_

////////////////////////////////////////
#include "../help/cuda_help_functions.h"
#include "cuda_kernel_define.h"
#include "cuda_kernel_functions_list.h"
////////////////////////////////////////

#define CHECK_BANK_CONFLICTS  0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) CUT_BANK_CHECKER(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) CUT_BANK_CHECKER(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif



/**************************************************************************/


__device__ float sigmoidActivFuncComput(float arrayElement, float alfa)
{
         return (float) ( 1 / ( 1 + exp( -alfa * arrayElement) ) ) ;
}

__device__ float outputLayerDelta(float desOutValue, float actualOutValue, float alfa)
{
         return (float) alfa * actualOutValue * (desOutValue - actualOutValue) * (1 - actualOutValue) ;
}


__global__ void hiddenLayerOutputCalculation(float* A, float* B, int N, float alfa)
{
         int tx = blockIdx.x * blockDim.x + threadIdx.x ;
         if (tx < N)
                      A[tx] = sigmoidActivFuncComput(B[tx],alfa) ;
         A[0] = 1 ;
}

__global__ void outputLayerOutputCalculation(float* A, float* B, int N, float alfa)
{
         int tx = blockIdx.x * blockDim.x + threadIdx.x ;
         if (tx < N)
                      A[tx] = sigmoidActivFuncComput(B[tx],alfa) ;
         A[0] = 0 ;
}

__global__ void deltaOutputLayerCalculation(float* C, float* A, float* B, int N, float alfa)
{
         int tx = blockIdx.x * blockDim.x + threadIdx.x;
         if (tx < N )
                     C[tx] = outputLayerDelta(A[tx],B[tx],alfa) ;
}

__global__ void deltaHiddenLayerCalculation(float* D, float* A, float* B, float* C, int wA, int wB, float alfa)
{
         int tx = blockIdx.x * blockDim.x + threadIdx.x ;
         int ty = blockIdx.y * blockDim.y + threadIdx.y ;

         float value = 0 ;
         for (int i = 0; i < wA ; ++i)
         {
                   float elementA = A[ty * wA + i] ;
                   float elementB = B[i * wB + tx] ;
                   value += elementA * elementB ;
         }
         D[ty * wB + tx] = value * alfa * C[ty * wB + tx] * (1 - C[ty *wB + tx]) ;
         D[0] = 0 ;
}

__global__ void layerWeightUpdate(float* C, float* A, float* B, int wA, int wB, float u)
{
         int tx = blockIdx.x * blockDim.x + threadIdx.x ;
         int ty = blockIdx.y * blockDim.y + threadIdx.y ;

         float value = 0 ;
         for (int i = 0 ; i < wA ; ++i)
         {
                   float elementA = A[ty * wA + i] ;
                   float elementB = B[i * wB + tx] ;
                   value += elementA * elementB ;
         }
         C[ty * wB + tx] = C[ty * wB + tx] + u*value ;
}

__global__ void matrixMul(float* C, float* A, float* B, int wA, int wB)
{
         int tx = blockIdx.x * blockDim.x + threadIdx.x ;
         int ty = blockIdx.y * blockDim.y + threadIdx.y ;

         float value = 0 ;
         for (int i = 0 ; i < wA ; ++i)
         {
                   float elementA = A[ty * wA + i] ;
                   float elementB = B[i * wB + tx] ;
                   value += elementA * elementB ;
         }
         C[ty * wB + tx] = value ;
}

__global__ void matrixMulT(float* C, float* A, float* B, int wA, int wB, float u)
{
         int tx = blockIdx.x * blockDim.x + threadIdx.x ;
         int ty = blockIdx.y * blockDim.y + threadIdx.y ;

         float value = 0 ;
         for (int i = 0 ; i < wA ; ++i)
         {
                   float elementA = A[ty * wA + i] ;
                   float elementB = B[i * wB + tx] ;
                   value += elementA * elementB ;
         }
         C[ty * wB + tx] = value*u ;
}


__global__ void matrixAdd(float* C, float* A, float* B, int N, float u)
{

        int i = blockDim.x * blockIdx.x + threadIdx.x;

        if (i < N)
                  C[i] = A[i] + u*B[i];

}



                       /*************************************************************
                         Multiply the arrays A and B on GPU and save the result in C
                         C(m,n) = A(m,k) * B(k,n)
                        ************************************************************/
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n)
{
         int lda=m,ldb=k,ldc=m;
         const float alf = 1;
         const float bet = 0;
         const float *alpha = &alf;
         const float *beta = &bet;

                     // Create a handle for CUBLAS
         cublasHandle_t handle;
         cublasCreate(&handle);

                     // Do the actual multiplication
         cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

                     // Destroy the handle
         cublasDestroy(handle);
}


void GPU_fill_rand(float *A, int rows_A, int cols_A)
{
                     // Create a pseudo-random number generator
          curandGenerator_t prng;
          curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

                     // Set the seed for the random number generator using the system clock
          curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

                     // Fill the array with random numbers on the device
          curandGenerateUniform(prng, A, rows_A * cols_A);
}






#endif
