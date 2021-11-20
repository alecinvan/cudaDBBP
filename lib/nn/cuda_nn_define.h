/**************************************
 * neural net parameter declaration
 * Author: Sa Li
**************************************/


#ifndef          _CUDA_NN_DEFINE_H_
#define          _CUDA_NN_DEFINE_H_


#define          WRITE_FILE_EPOCHS                  20
#define          PRINT_PREDICT_EPOCHS               20

#define          INPUT_P_N_LENGTH
#ifndef          INPUT_P_N_LENGTH
#define          P_Len                              1223677
#define          N_Len                              1223678
#endif


                 /*++++++++++++++++++++
                 +   training stage   +
                 +++++++++++++++++++++*/

#define          STAGE_OF_TRAINING           // define macro of training

#ifdef           START_MODE
#define          START_TRAINING
#endif

#ifdef           CONTINUT_MODE
#define          CONTINUE_TRAINING
#endif

#ifdef           PREDICTING
#define          STAGE_OF_PREDICTING        // define macro of predicting
#endif

#ifdef           DEBUG_MODE
#define          STAGE_OF_DEBUGGING
#endif

#define          READ_WGT_FROM_FILE
#ifndef          READ_WGT_FROM_FILE
#define          READ_WGT_FROM_DB
#endif

#define          ACUAL_PREDICTION

#ifndef          LOAD_DATA
#define          LOAD_DATA_TO_MEMORY
#endif


//#define          CUBLAS_LIB_MODE
#ifndef          CUBLAS_LIB_MODE
#define          CUDA_KERNEL_MODE
#endif


#ifdef           PRINT
#define          PRINT_MEASURE_MATRIX
#endif


#ifdef           OMP_MODE
#define          OMP_LOOP_MODE
#endif

#endif

