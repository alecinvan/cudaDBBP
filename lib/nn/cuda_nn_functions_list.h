#ifndef _CUDA_NN_FUNCTIONS_LIST_H_
#define _CUDA_NN_FUCNTIONS_LIST_H_

              ////////////////////////////
              // neural network functions
              ////////////////////////////

void   global_variable_define(unsigned int N_input, unsigned int N_1hidden,
                              unsigned int N_2hidden, unsigned int N_output) ;
void   connectionWeightInit(unsigned int inputSize, unsigned int firstLayerSize,
                            unsigned int secondLayerSize, unsigned int outputSize) ;
void   arrayAssign(ifstream&input, unsigned int inputSize,
                   unsigned int outputSize, string s) ;
void   arrayClear(float* inputArray, unsigned int inputSize) ;
void   writeTwoLayerWeightsInFile(ofstream&output, unsigned int inputSize, unsigned int firstHidLayerSize,
                                  unsigned int secHidLayerSize, unsigned int outputSize) ;
void   writeParameter(ofstream & output, float u, float alfa, float beta, unsigned int inputSize,
                      unsigned int firstHidLayerSize, unsigned int secHidLayerSize,
                      unsigned int outputSize, unsigned epochs) ;
void   loadTwoLayerWeightsInArray(ifstream&input, unsigned int inputSize, unsigned int firstHidLayerSize,
                                  unsigned int secHidLayerSize, unsigned int outputSize) ;
void   writeParameterIntoFile(string path, string paraFileString, float u, float alfa, float beta,
                              unsigned int N_input, unsigned int N_1hidden, unsigned int N_2hidden,
                              unsigned int N_output, unsigned int epochs) ;
void   writeWeightsIntoFile(string path, string weightFileString, unsigned int N_input,
                            unsigned int N_1hidden, unsigned int N_2hidden, unsigned int N_output) ;
void   readWeightsFromFile(string path, string weightFileString, unsigned int N_input,
                           unsigned int N_1hidden, unsigned int N_2hidden, unsigned int N_output) ;
string readWeightsIntoString(string path, string weightFileString);
string writeWeightMatrixIntoString(int inputSize, int firstHidLayerSize, int secHidLayerSize, int outputSize);
void   convertStringToArray(string weightString, int inputSize, int firstHidLayerSize, int secHidLayerSize, int outputSize) ;
void   fMeasureCompute(float desOutValue, float actualOutValue) ;
void   fMeasureTestCompute(float desOutValue, float actualOutValue) ;
float  accuracyCompute(int tn, int fp, int tp, int fn) ;
void   printMeasureMatrix(string path, string resultFileString,
                          int tp, int tn, int fp, int fn, int len)  ;
void   printPredictMatrix(unsigned int tp, unsigned int tn, unsigned int fp,
                          unsigned int fn, unsigned int len) ;
void   printTimeStamp(float diff_time, unsigned int k, unsigned int Epochs) ;
void   accuratePredictCount(float desOutValue, float actualOutValue) ;
void   accuratePredictTestCount(float desOutValue, float actualOutValue) ;
float  sumsSquareCalculation(float desOutValue, float actualOutValue) ;
float  absoluteError(float desOutValue, float actualOutValue) ;
float  standardDeviation(float residual, float mae, int len) ;
float  rootMeanSqareError(float mae, float sd) ;

#endif
