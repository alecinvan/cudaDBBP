#ifndef  _CUDA_TRAINING_LIST_H_
#define  _CUDA_TRAINING_LIST_H_

                  /////////////////////////////////
                  // neural net training functions
                  /////////////////////////////////
void         gpuNeuralNetworkBPtrain(PGconn *conn, string path, string lastRoundWeightPath, string lastRoundWeightFileString,
                                     string inputfile, string testfile, unsigned int slurm_job_id, float u, float alfa, float beta,
                                     float mean, float mean_test, unsigned int N_input, unsigned int N_1hidden, unsigned int N_2hidden,
                                     unsigned int N_output, unsigned int epochs, unsigned int stage) ;

void         gpuMomentumTermInit(unsigned int N_input, unsigned int N_1hidden, unsigned int N_2hidden, unsigned int N_output);

void         gpuMomentumTermUpdate(unsigned int N_input, unsigned int N_1hidden, unsigned int N_2hidden, unsigned int N_output) ;

void         gpuVectorMemoryAssign(unsigned int inputSize, unsigned int firstLayerSize,
                                   unsigned int secondLayerSize, unsigned int outputSize) ;

void         gpuVectorMemoryClean(void) ;

void         cpuWeightsToGpu(PGconn *conn, string path, string weightFileString,
                             unsigned int inputSize, unsigned int firstHidLayerSize,
                             unsigned int secHidLayerSize, unsigned int outputSize) ;

void         loadDatafileToMemory(string inputfile) ;

void         gpuFeedforwardCompute(float alfa, unsigned int N_input, unsigned int N_1hidden,
                                   unsigned int N_2hidden, unsigned int N_output) ;

void         gpuBackPropagationCompute(float u, float alfa, float beta, unsigned int N_input,
                                       unsigned int N_1hidden, unsigned int N_2hidden, unsigned int N_output) ;

void         predictByTrainedWeights(string path, string resultFileString, string inputfile, float alfa, unsigned int N_input,
                                     unsigned int N_1hidden, unsigned int N_2hidden, unsigned int N_output, unsigned int tn,
                                     unsigned int fp, unsigned int tp, unsigned int fn) ;

void         printPredictions(string path, string resultFileString, string inputfile, float alfa, unsigned int N_input,
                              unsigned int N_1hidden, unsigned int N_2hidden, unsigned int N_output) ;

void         printTestPredictions(string path, string resultFileString, string inputfile, float alfa, unsigned int N_input,
                              unsigned int N_1hidden, unsigned int N_2hidden, unsigned int N_output) ;

void         measureRegressionNN(string inputfile, float alfa, unsigned int N_input, unsigned int N_1hidden,
                                 unsigned int N_2hidden, unsigned int N_output, float mean) ;

void         measureTestRegressionNN(string inputfile, float alfa, unsigned int N_input, unsigned int N_1hidden,
                                     unsigned int N_2hidden, unsigned int N_output, float mean_test) ;

void         measureByRegressionNNWeight(string inputfile, float alfa, unsigned int N_input, unsigned int N_1hidden,
                                         unsigned int N_2hidden, unsigned int N_output, float mean) ;

void         measureTestByRegressionNNWeight(string inputfile, float alfa, unsigned int N_input, unsigned int N_1hidden,
                                             unsigned int N_2hidden, unsigned int N_output, float mean_test) ;

#endif

