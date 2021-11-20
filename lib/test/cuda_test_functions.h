/***************************
 *  THE GPU TEST FUNCTIONS
 *  AUTHOR: SA LI
***************************/

#ifndef    _CUDA_TEST_FUNCTIONS_H_
#define    _CUDA_TEST_FUNCTIONS_H_

/////////////////////////////////////////////
#include "../kernel/cuda_kernel_functions.h"
#include "./cuda_test_define.h"
#include "./cuda_test_functions_list.h"
/////////////////////////////////////////////

using namespace std ;

/******************************************************/

void randomInit(float* data, int size)
{
         for (int i = 0; i < size ; ++i)
              // data[i] = rand() / (float)RAND_MAX ;
                data[i] = 0.3 ;
}

//void arrayMultiplication(int argc, char** argv)
void arrayMultiplication()
{
        srand(2006) ;
        unsigned int size_A = WA * HA ;
        unsigned int mem_size_A = sizeof(float) * size_A ;
        float* h_A = (float*) malloc(mem_size_A) ;
        unsigned int size_B = WB * HB ;
        unsigned int mem_size_B = sizeof(float) * size_B ;
        float* h_B = (float*) malloc(mem_size_B) ;
        randomInit(h_A, size_A) ;
        randomInit(h_B, size_B) ;

        std::cout << "gpu multiplication testing ..." << endl ;
        std::cout << endl << endl << "Matrix A" << endl ;
        print_matrix(h_A, size_A, WA) ;
        std::cout << endl << endl << "Matrix B" << endl ;
        print_matrix(h_B, size_B, WB) ;

        float* d_A;
        float* d_B;
        checkCudaErrors(cudaMalloc((void**) &d_A, mem_size_A)) ;
        checkCudaErrors(cudaMalloc((void**) &d_B, mem_size_B)) ;

        checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice)) ;
        checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice)) ;
        unsigned int size_C = WC * HC ;
        unsigned int mem_size_C = sizeof(float) * size_C ;
        float* h_C = (float*) malloc(mem_size_C) ;

        float* d_C ;
        checkCudaErrors(cudaMalloc((void**) &d_C, mem_size_C)) ;

        dim3 threads(BLOCK_SIZE, BLOCK_SIZE) ;
        dim3 grid(WC / threads.x + (WC%threads.x == 0? 0:1), HC / threads.y + (HC%threads.y == 0? 0:1)) ;

        int nIter = 30 ;
        StopWatchInterface * timer_matrixMul;

                     // Start Timing
        sdkCreateTimer(&timer_matrixMul);
        sdkStartTimer(&timer_matrixMul);
        for (int j = 0; j < nIter; j++) {
                //  matrixMulT<<< grid, threads >>>(d_C, d_A, d_B, WA, WB, atof(argv[2]) ) ;
                  matrixMulT<<< grid, threads >>>(d_C, d_A, d_B, WA, WB, 1.0 ) ;
        }
        getLastCudaError("CUDA matrixMul Kernel execution failed");
        cudaDeviceSynchronize();
                      // stop and destroy timer
        sdkStopTimer(&timer_matrixMul);
        double dSeconds = sdkGetTimerValue(&timer_matrixMul)/((double)nIter * 1000.0);
        double dNumOps = 2.0 * (double)WA * (double)HA * (double)WB;
        double gflops = 1.0e-9 * dNumOps/dSeconds;

        checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost)) ;
        std::cout << endl << endl << "Matrix C" << endl ;
        print_matrix(h_C, size_C, WC) ;
        printf("\n") ;

        printf("> CUDA matrixMul %.4f GFlop/s, Time = %.5f s, Size = %.0f Ops, ",
                                gflops, dSeconds, dNumOps);
        printf("NumDevsUsed = %d, Workgroup = %u\n", 1, threads.x * threads.y);
        sdkDeleteTimer(&timer_matrixMul);

        std::cout << endl << endl << "=======" << endl ;
        free(h_A);
        free(h_B);
        free(h_C);
        checkCudaErrors(cudaFree(d_A));
        checkCudaErrors(cudaFree(d_B));
        checkCudaErrors(cudaFree(d_C));

}


#endif
