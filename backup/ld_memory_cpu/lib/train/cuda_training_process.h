/*********************************
 * The body of nerual net BP
 * training on GPU
 * Author: Sa Li
*********************************/

#ifndef  _CUDA_TRAINING_PROCESS_H_
#define  _CUDA_TRAINING_PROCESS_H_

////////////////////////////////////
#include "../db/cuda_db_populate.h"
#include "./cuda_training_list.h"
////////////////////////////////////

using namespace std;


/*************************************************************/

void gpuVectorMemoryAssign(unsigned int inputSize, unsigned int firstLayerSize,
                           unsigned int secondLayerSize, unsigned int outputSize)
{
             checkCudaErrors(cudaMalloc((void**)&W_1_d, (inputSize+1) *(firstLayerSize+1)* sizeof(float)));
	     checkCudaErrors(cudaMalloc((void**)&W_2_d, (firstLayerSize+1) * (secondLayerSize+1) * sizeof(float)));
	     checkCudaErrors(cudaMalloc((void**)&W_3_d, (secondLayerSize+1) * (outputSize+1) * sizeof(float)));
             checkCudaErrors(cudaMalloc((void**)&X_in_d, (inputSize+1) * sizeof(float)));
	     checkCudaErrors(cudaMalloc((void**)&Y_dout_d, (outputSize+1) * sizeof(float)));
	     checkCudaErrors(cudaMalloc((void**)&Y_out_d, (outputSize+1) * sizeof(float)));
	     checkCudaErrors(cudaMalloc((void**)&delta_o_d, (outputSize+1) * sizeof(float)));
	     checkCudaErrors(cudaMalloc((void**)&delta_h_d, (secondLayerSize+1) * sizeof(float)));
	     checkCudaErrors(cudaMalloc((void**)&delta_i_d, (firstLayerSize+1) * sizeof(float)));
	     checkCudaErrors(cudaMalloc((void**)&y_1_d, (firstLayerSize+1) * sizeof(float)));
	     checkCudaErrors(cudaMalloc((void**)&y_2_d, (secondLayerSize+1) * sizeof(float)));
	     checkCudaErrors(cudaMalloc((void**)&y_3_d, (outputSize+1) * sizeof(float)));
	     checkCudaErrors(cudaMalloc((void**)&x_1_d, (firstLayerSize+1) * sizeof(float)));
	     checkCudaErrors(cudaMalloc((void**)&x_2_d, (secondLayerSize+1) * sizeof(float)));
}

void gpuVectorMemoryClean(void)
{
             checkCudaErrors(cudaFree(W_1_d));
	     checkCudaErrors(cudaFree(W_2_d));
	     checkCudaErrors(cudaFree(W_3_d));
	     checkCudaErrors(cudaFree(X_in_d));
	     checkCudaErrors(cudaFree(Y_dout_d));
             checkCudaErrors(cudaFree(Y_out_d));
	     checkCudaErrors(cudaFree(delta_o_d));
	     checkCudaErrors(cudaFree(delta_h_d));
             checkCudaErrors(cudaFree(delta_i_d));
	     checkCudaErrors(cudaFree(y_1_d));
	     checkCudaErrors(cudaFree(y_2_d));
             checkCudaErrors(cudaFree(y_3_d));
	     checkCudaErrors(cudaFree(x_1_d));
	     checkCudaErrors(cudaFree(x_2_d));
}


void cpuWeightsToGpu(PGconn *conn, string lastRoundWeightPath, string lastRoundWeightFileString, unsigned int inputSize, unsigned int firstHidLayerSize,
                     unsigned int secHidLayerSize, unsigned int outputSize, unsigned int stage)
{                                                                                               // fullpath, TRAINEDWEIGHTLASTEPOCH

             if ( stage < 2 )
             {
                            connectionWeightInit(inputSize+1, firstHidLayerSize+1, secHidLayerSize+1, outputSize+1) ;
             }
             else
             {
#ifdef                      READ_WGT_FROM_FILE
                            readWeightsFromFile(lastRoundWeightPath, lastRoundWeightFileString, inputSize, firstHidLayerSize, secHidLayerSize, outputSize) ; 
#endif

#ifdef                      READ_WGT_FROM_DB
                            string weightMatrixString ;
                            weightMatrixString = fetchWeightString(conn, integerToCharArray(lastjobid), integerToCharArray(lastoutcomeid)) ;
                            convertStringToArray(weightMatrixString, inputSize+1, firstHidLayerSize+1, secHidLayerSize+1, outputSize+1) ;
#endif

             }

             checkCudaErrors(cudaMemcpy( W_1_d, W_1_h, (inputSize+1)*(firstHidLayerSize+1)*sizeof(float), cudaMemcpyHostToDevice) );
	     checkCudaErrors(cudaMemcpy( W_2_d, W_2_h, (firstHidLayerSize+1)*(secHidLayerSize+1)*sizeof(float), cudaMemcpyHostToDevice)) ;
             checkCudaErrors(cudaMemcpy( W_3_d, W_3_h, (secHidLayerSize+1)*(outputSize+1)*sizeof(float), cudaMemcpyHostToDevice));

}

void gpuFeedforwardCompute(float alfa, unsigned int N_input, unsigned int N_1hidden, unsigned int N_2hidden, unsigned int N_output)
{

	     arrayClear(y_1_h, N_1hidden) ;
	     arrayClear(y_2_h, N_2hidden) ;
             arrayClear(y_3_h, N_output) ;
             arrayClear(x_1_h, N_1hidden) ;
             arrayClear(x_2_h, N_2hidden) ;

             checkCudaErrors(cudaMemcpy(y_1_d, y_1_h, (N_1hidden+1)*sizeof(float), cudaMemcpyHostToDevice));
                                                                                       // copy y_1  to gpu
	     checkCudaErrors(cudaMemcpy(x_1_d, x_1_h, (N_1hidden+1)*sizeof(float), cudaMemcpyHostToDevice));
                                                                                       // copy x_1  to gpu
	     checkCudaErrors(cudaMemcpy(y_2_d, y_2_h, (N_2hidden+1)*sizeof(float), cudaMemcpyHostToDevice));
                                                                                       // copy y_2  to gpu
	     checkCudaErrors(cudaMemcpy(x_2_d, x_2_h, (N_2hidden+1)*sizeof(float), cudaMemcpyHostToDevice));
                                                                                       // copy x_2  to gpu
	     checkCudaErrors(cudaMemcpy(y_3_d, y_3_h, (N_output+1)*sizeof(float), cudaMemcpyHostToDevice));
                                                                                       // copy y_3  to gpu
	     checkCudaErrors(cudaMemcpy(X_in_d, X_in_h, (N_input+1)*sizeof(float), cudaMemcpyHostToDevice));
                                                                                       // copy X_in to gpu

#ifdef       STAGE_OF_DEBUGGING
	     std::cout << std::endl  << "==========" << std::endl << "X_in: " << std::endl ;
	     print_matrix(X_in_h, N_input+1, N_input+1) ;
	     checkCudaErrors(cudaMemcpy( W_1_h, W_1_d, (N_input+1)*(N_1hidden+1)*sizeof(float), cudaMemcpyDeviceToHost)) ;
	     std::cout << std::endl <<  "W_1: " << std::endl ;
             print_matrix(W_1_h, (N_input+1)*(N_1hidden+1), N_1hidden+1) ;
             pause_keyboard() ;
#endif


             dim3 threadArray(BLOCK_SIZE, BLOCK_SIZE) ;
	     dim3 y_1_grid( (N_1hidden+1) / threadArray.x + ((N_1hidden+1)%threadArray.x == 0? 0:1), 1/threadArray.y + (1%threadArray.y == 0? 0:1)) ;
                                                                                   // WA=N_input, HA=1, WB=N_1hidden, HB=N_input, WC=N_1hidden, HC=1
	     matrixMul<<< y_1_grid, threadArray >>>(y_1_d, X_in_d, W_1_d, N_input+1, N_1hidden+1);
                                                                  // first layer feedforward  WA, WB

#ifdef       STAGE_OF_DEBUGGING
	     checkCudaErrors(cudaMemcpy( y_1_h, y_1_d, (N_1hidden+1)*sizeof(float), cudaMemcpyDeviceToHost)) ;
                                                                                        // copy y_1 back to cpu
	     std::cout << std::endl << "y_1: " << std::endl;
             print_matrix(y_1_h, N_1hidden+1, N_1hidden+1) ;
             pause_keyboard() ;
#endif


             dim3 outputThreadArray(256, 1) ;
	     dim3 x_1_grid( (N_1hidden+1)/outputThreadArray.x + ((N_1hidden+1)%outputThreadArray.x == 0? 0:1), 1);
	     hiddenLayerOutputCalculation<<< x_1_grid, outputThreadArray >>>(x_1_d, y_1_d, N_1hidden+1, alfa);

#ifdef       STAGE_OF_DEBUGGING
	     checkCudaErrors(cudaMemcpy( x_1_h, x_1_d, (N_1hidden+1)*sizeof(float), cudaMemcpyDeviceToHost));
                                                                                          // copy x_1 back to cpu
	     checkCudaErrors(cudaMemcpy( W_2_h, W_2_d, (N_1hidden+1)*(N_2hidden+1)*sizeof(float), cudaMemcpyDeviceToHost)) ;
	     std::cout << std::endl << "x_1: " << std::endl ;
             print_matrix(x_1_h, N_1hidden+1, N_1hidden+1) ;
	     std::cout << std::endl << "W_2: " << std::endl ;
             print_matrix(W_2_h, (N_1hidden+1)*(N_2hidden+1), N_2hidden+1) ;
             pause_keyboard() ;
#endif


             dim3 y_2_grid( (N_2hidden+1) / threadArray.x + ((N_2hidden+1)%threadArray.x == 0? 0:1), 1/threadArray.y + (1%threadArray.y == 0? 0:1)) ;
                                                                              // WA=N_1hidden, HA=1, WB=N_2hidden, HB=N_1hidden, WC=WB=N_2hidden, HC=HA=1
	     matrixMul<<< y_2_grid, threadArray >>>(y_2_d, x_1_d, W_2_d, N_1hidden+1, N_2hidden+1);
                                       // second layer feedforward, (float* C, float* A, float* B, int wA, int wB)

#ifdef       STAGE_OF_DEBUGGING
	     checkCudaErrors(cudaMemcpy( y_2_h, y_2_d, (N_2hidden+1)*sizeof(float), cudaMemcpyDeviceToHost)) ;
                                                                                         // copy y_2 back to cpu
             std::cout <<  "y_2: " << std::endl ;
             print_matrix(y_2_h, N_2hidden+1, N_2hidden+1) ;
             pause_keyboard() ;
#endif



             dim3 x_2_grid( (N_2hidden+1)/outputThreadArray.x + ((N_2hidden+1)%outputThreadArray.x == 0? 0:1), 1);
	     hiddenLayerOutputCalculation<<< x_2_grid, outputThreadArray >>>(x_2_d, y_2_d, N_2hidden+1, alfa);

#ifdef       STAGE_OF_DEBUGGING
	     checkCudaErrors(cudaMemcpy( x_2_h, x_2_d, (N_2hidden+1)*sizeof(float), cudaMemcpyDeviceToHost));
                                                                                        // copy x_2 back to cpu
	     checkCudaErrors(cudaMemcpy( W_3_h, W_3_d, (N_2hidden+1)*(N_output+1)*sizeof(float), cudaMemcpyDeviceToHost) );
	     std::cout << std::endl << "x_2: " << std::endl;
             print_matrix(x_2_h, N_2hidden+1, N_2hidden+1) ;
	     std::cout << endl << "W_3: " << std::endl ;
             print_matrix(W_3_h, (N_2hidden+1)*(N_output+1), N_output+1) ;
             pause_keyboard() ;
#endif





             dim3 y_3_grid( (N_output+1) / threadArray.x + ((N_output+1)%threadArray.x == 0? 0:1), 1/threadArray.y + (1%threadArray.y == 0? 0:1)) ;
                                                                                    // WA=N_2hidden, HA=1, WB=N_output, HB=N_2hidden, WC=N_output, HC=1
             matrixMul<<< y_3_grid, threadArray >>>(y_3_d, x_2_d, W_3_d, N_2hidden+1, N_output+1);
                                                                                     // output layer feedforward

#ifdef       STAGE_OF_DEBUGGING
             pause_keyboard() ;
	     checkCudaErrors(cudaMemcpy( y_3_h, y_3_d, (N_output+1)*sizeof(float), cudaMemcpyDeviceToHost));
                                                                                     // copy Y_out back to cpu
             std::cout <<  "y_3: " << std::endl;
             print_matrix(y_3_h, N_output+1, N_output+1) ;
#endif



             dim3 Y_out_grid( (N_output+1)/outputThreadArray.x + ((N_output+1)%outputThreadArray.x == 0? 0:1), 1);
             outputLayerOutputCalculation<<< Y_out_grid, outputThreadArray >>>(Y_out_d, y_3_d, N_output+1, alfa);

	     checkCudaErrors(cudaMemcpy( Y_out_h, Y_out_d, (N_output+1)*sizeof(float), cudaMemcpyDeviceToHost));
                                                                                              // copy Y_out back to cpu

#ifdef       STAGE_OF_DEBUGGING
             std::cout <<  "Y_out: " << std::endl;
             print_matrix(Y_out_h, N_output+1, N_output+1) ;
             std::cout <<  "Y_dout_h: " << std::endl ;
             print_matrix(Y_dout_h, N_output+1, N_output+1) ;
             std::cout << "~~~~~~~~~" << std::endl;
             pause_keyboard() ;
#endif

}

void gpuBackPropagationCompute(float u, float alfa, unsigned int N_input, unsigned int N_1hidden, unsigned int N_2hidden, unsigned int N_output)
{

            checkCudaErrors(cudaMemcpy(Y_dout_d, Y_dout_h, (N_output+1)*sizeof(float), cudaMemcpyHostToDevice));
                                                                                             // copy Y_dout to gpu


            dim3 deltaThreadArray(256, 1) ;
	    dim3 deltaOutputGrid( (N_output+1)/deltaThreadArray.x + ((N_output+1)%deltaThreadArray.x == 0? 0:1), 1);
	    deltaOutputLayerCalculation<<< deltaOutputGrid, deltaThreadArray >>>(delta_o_d, Y_dout_d, Y_out_d, N_output+1, alfa) ;
                                                                                                      // get delta on output later

#ifdef      STAGE_OF_DEBUGGING
	    checkCudaErrors(cudaMemcpy( delta_o_h, delta_o_d, (N_output+1)*sizeof(float), cudaMemcpyDeviceToHost));
                                                                                           // copy delta_o back to cpu
	    checkCudaErrors(cudaMemcpy( x_2_h, x_2_d, (N_2hidden+1)*sizeof(float), cudaMemcpyDeviceToHost)); 
                                                                                           // copy y_2_d back to cpu
	    std::cout << std::endl << "x_2_h: " << std::endl ;
            print_matrix(x_2_h, N_2hidden+1, 1) ;
            std::cout << std::endl << "delta_o_h: " << std::endl ;
            print_matrix(delta_o_h, N_output+1, N_output+1) ;
            pause_keyboard() ;
#endif




            dim3 threadArray(BLOCK_SIZE, BLOCK_SIZE) ;
	    dim3 w_3_grid((N_output+1) / threadArray.x + ((N_output+1)%threadArray.x == 0? 0:1), (N_2hidden+1) / threadArray.y + ((N_2hidden+1) %threadArray.y == 0? 0:1)) ;
                                                                                                  // WA=1, HA=N_2hidden+1, WB=N_output+1, HB=1, WC=N_output+1, HC=N_2hidden+1
	    layerWeightUpdate<<< w_3_grid, threadArray >>>(W_3_d, x_2_d, delta_o_d, 1, N_output+1, u) ;





            dim3 delta2hiddenGrid( 1/threadArray.x + (1%threadArray.x == 0? 0:1) , (N_2hidden+1) / threadArray.y + ((N_2hidden+1) %threadArray.y == 0? 0:1) ) ;
                                                                                    // WA=N_output+1, HA=N_2hidden+1, WB=1, HB=N_output+1, WC=1, HC=N_2hidden+1
	    deltaHiddenLayerCalculation<<< delta2hiddenGrid, threadArray >>>(delta_h_d, W_3_d, delta_o_d, x_2_d, N_output+1, 1, alfa) ;

#ifdef      STAGE_OF_DEBUGGING
            pause_keyboard() ;
	    checkCudaErrors(cudaMemcpy( delta_h_h, delta_h_d, (N_2hidden+1)*sizeof(float), cudaMemcpyDeviceToHost));
                                                                                        // copy delta_h_d back to cpu
	    std::cout << std::endl << "delta_h_h: " << std::endl ;
            print_matrix(delta_h_h, N_2hidden+1, 1) ;
            pause_keyboard() ;
#endif



            dim3 w_2_grid( (N_2hidden+1) / threadArray.x + ((N_2hidden+1) %threadArray.x == 0? 0:1), (N_1hidden+1) / threadArray.y + ((N_1hidden+1)%threadArray.y == 0? 0:1)) ;
                                                                                                    // WA=1, HA=N_1hidden+1, WB=N_2hidden+1, HB=1, WC=N_2hidden+1, HC=N_1hidden+1
	    layerWeightUpdate<<< w_2_grid, threadArray>>>(W_2_d, x_1_d, delta_h_d, 1, N_2hidden+1, u) ;
                                                                                                //WA, WB



            dim3 delta1hiddenGrid(1/threadArray.x + (1%threadArray.x == 0? 0:1), (N_1hidden+1) / threadArray.y + ((N_1hidden+1)%threadArray.y == 0? 0:1)) ;
                                                                                      // WA=N_2hidden+1, HA=N_1hidden+1, WB=1, HB=N_2hidden+1, WC=1, HC=N_1hidden+1
	    deltaHiddenLayerCalculation<<< delta1hiddenGrid, threadArray >>>(delta_i_d, W_2_d, delta_h_d, x_1_d, N_2hidden+1, 1, alfa) ;

#ifdef      STAGE_OF_DEBUGGING
	    checkCudaErrors(cudaMemcpy( delta_i_h, delta_i_d, (N_1hidden+1)*sizeof(float), cudaMemcpyDeviceToHost)) ;
                                                                                              // copy delta_h_d back to cpu
	    std::cout << std::endl << "delta_i_h: " << std::endl ;
            print_matrix(delta_i_h, N_1hidden+1, 1) ;
            pause_keyboard() ;
#endif




	    dim3 w_1_grid( (N_1hidden+1) / threadArray.x + ((N_1hidden+1) %threadArray.x == 0? 0:1), (N_input+1) / threadArray.y + ((N_input+1)%threadArray.y == 0? 0:1)) ;
                                                                                                    // WA=1, HA=N_input+1, WB=N_1hidden+1, HB=1, WC=N_1hidden+1, HC=N_input+1
	    layerWeightUpdate<<< w_1_grid, threadArray >>>(W_1_d, X_in_d, delta_i_d, 1, N_1hidden+1, u) ;


}

void gpuNeuralNetworkBPtrain(PGconn *conn, string path, string lastRoundWeightPath, string lastRoundWeightFileString, string inputfile,
                             string testfile, unsigned int slurm_job_id, float u, float alfa, unsigned int N_input, unsigned int N_1hidden,
                             unsigned int N_2hidden, unsigned int N_output, unsigned int epochs, unsigned int stage)
{
            ifstream     fileInput ;
            gpuVectorMemoryAssign(N_input, N_1hidden, N_2hidden, N_output) ; // cudaMalloc the memory in device
            cpuWeightsToGpu(conn, lastRoundWeightPath, lastRoundWeightFileString, N_input, N_1hidden, N_2hidden, N_output, stage) ;  // It is been switched to Start training or continue training in this function
            time_t start, end ;

            unsigned int Epochs = epochs ;
	    unsigned int ten_epoch_count =  1 ;
	    unsigned int write_file_epoch_count = 0 ;

#ifdef      PRINT_PREDICTION_COUNT
            unsigned int print_predictions_count = 0 ;
#endif


#ifdef      LOAD_DATA_TO_MEMORY
            fileInput.open(inputfile.c_str()) ;
            openInputFilesFailed(fileInput) ;
            unsigned int row_idx = 1 ;
            while (!fileInput.eof())  {
                                   string s ;
                                   getline(fileInput, s) ;
                                   dataLine[row_idx] = s ;
                                   row_idx++ ;
            }
            fileInput.close();
#endif


#ifdef      ACUAL_PREDICTION
	    for (int k=0; k< Epochs; k++){
				   if  (ten_epoch_count == 1 )
		         	                           start = time (0) ;

                                                         /*************************\
                                                         | training the whole file |
                                                         \*************************/
                                   for (unsigned int map_idx=1; map_idx <= dataLine.size()-1; map_idx++)
                                   {
                                                           arrayAssign(N_input, N_output, dataLine[map_idx]) ;
		                                           gpuFeedforwardCompute(alfa, N_input, N_1hidden, N_2hidden, N_output) ;
                                                           gpuBackPropagationCompute(u, alfa, N_input, N_1hidden, N_2hidden, N_output) ;
                                   }

				   ten_epoch_count++;
				   if (ten_epoch_count == 10)
				   {
                                 		           end = time (0) ;
							   printTimeStamp(difftime(end,start), k, Epochs) ;
							   ten_epoch_count = 1 ;
				   }
				   write_file_epoch_count++ ;
				   if ( k%WRITE_FILE_EPOCHS == 0 )
				   {
                                                                               /*&&&&&&&&&&&&&&&&\
                                                                               |  START TRAINING |
                                                                               \&&&&&&&&&&&&&&&&*/
                                            if ( stage < 2 )
                                            {

	                                                    writeWeightsIntoFile(path, result_files(k, "/TrainedWeights_", "_epochs.dat"),          // write weights into files
                                                                                 N_input, N_1hidden, N_2hidden, N_output) ;


                                                            printPredictions(path, result_files(k, "/TrainedWeights_", "_epochs.log"),              // write results log into files
                                                                             inputfile, alfa, N_input, N_1hidden, N_2hidden, N_output) ;

                                                            printTestPredictions(path, result_files(k, "/TrainedWeights_", "_epochs.test.log"),              // write results log into files
                                                                                 testfile, alfa, N_input, N_1hidden, N_2hidden, N_output) ;

                                                            job_outcome_table(slurm_job_id, k, TP, TN, FP, FN, TP_TEST, TN_TEST, FP_TEST, FN_TEST,
                                                                              P_Len, N_Len, P_Len_test, N_Len_test, path, "", lastjobid) ;      // populate job_outcome table
                                                            job_outcome.job_outcome_id = insertJobOutcomeTable(conn, integerToCharArray(job_outcome.slurm_job_id), integerToCharArray(job_outcome.n_epoch),
                                                                                                               integerToCharArray(job_outcome.true_pos), floatToCharArray(job_outcome.true_pos_percentage),
                                                                                                               integerToCharArray(job_outcome.true_neg), floatToCharArray(job_outcome.true_neg_percentage),
                                                                                                               integerToCharArray(job_outcome.false_pos), floatToCharArray(job_outcome.false_pos_percentage),
                                                                                                               integerToCharArray(job_outcome.false_neg), floatToCharArray(job_outcome.false_neg_percentage),
                                                                                                               floatToCharArray(job_outcome.test_true_pos_per), floatToCharArray(job_outcome.test_true_neg_per),
                                                                                                               floatToCharArray(job_outcome.test_false_pos_per), floatToCharArray(job_outcome.test_false_neg_per),
                                                                                                               job_outcome.result_path.c_str(), job_outcome.note.c_str(),
                                                                                                               integerToCharArray(job_outcome.last_round_job_id)) ;


#ifdef                                                       READ_WGT_FROM_FILE
                                                             string weightStr = readWeightsIntoString(path, result_files(k, "/TrainedWeights_", "_epochs.dat")) ;        //populate weight_matrix table
#endif

#ifdef                                                       READ_WGT_FROM_DB
                                                             string weightStr = writeWeightMatrixIntoString(N_input+1, N_1hidden+1, N_2hidden+1, N_output+1) ;                  // write matrix into string
#endif


                                                             weight_matrix_table(job_outcome.job_outcome_id, path, result_files(k, "/TrainedWeights_", "_epochs.dat"), weightStr);
                                                             insertWeightMatrixTable(conn, integerToCharArray(weight_matrix.job_outcome_id), weight_matrix.path.c_str(),
                                                                                     weight_matrix.name.c_str(), weight_matrix.matrix.c_str()) ;
                                             }
                                             else
                                             {
                                                                                           /*%%%%%%%%%%%%%%%%%%%\
                                                                                           |  CONTINUE TRAINING |
                                                                                           \%%%%%%%%%%%%%%%%%%%*/

                                                             writeWeightsIntoFile(path, result_files(k+parse_file_string(inputfilename.trainedweightlastepoch), "/TrainedWeights_", "_epochs.dat"),
                                                                                  N_input, N_1hidden, N_2hidden, N_output) ;


                                                             printPredictions(path, result_files(k+parse_file_string(inputfilename.trainedweightlastepoch), "/TrainedWeights_", "_epochs.log"),
                                                                              inputfile, alfa, N_input, N_1hidden, N_2hidden, N_output) ;

                                                             printTestPredictions(path, result_files(k+parse_file_string(inputfilename.trainedweightlastepoch), "/TrainedWeights_", "_epochs.log"),
                                                                                  testfile, alfa, N_input, N_1hidden, N_2hidden, N_output) ;

                                                             job_outcome_table(slurm_job_id, k+parse_file_string(inputfilename.trainedweightlastepoch), TP, TN, FP, FN, TP_TEST, TN_TEST, FP_TEST, FN_TEST,
                                                                               P_Len, N_Len, P_Len_test, N_Len_test, path, "", lastjobid) ;                                // populate job_outcome table
                                                             job_outcome.job_outcome_id = insertJobOutcomeTable(conn, integerToCharArray(job_outcome.slurm_job_id), integerToCharArray(job_outcome.n_epoch),
                                                                                                                integerToCharArray(job_outcome.true_pos), floatToCharArray(job_outcome.true_pos_percentage),
                                                                                                                integerToCharArray(job_outcome.true_neg), floatToCharArray(job_outcome.true_neg_percentage),
                                                                                                                integerToCharArray(job_outcome.false_pos), floatToCharArray(job_outcome.false_pos_percentage),
                                                                                                                integerToCharArray(job_outcome.false_neg), floatToCharArray(job_outcome.false_neg_percentage),
                                                                                                                floatToCharArray(job_outcome.test_true_pos_per), floatToCharArray(job_outcome.test_true_neg_per),
                                                                                                                floatToCharArray(job_outcome.test_false_pos_per), floatToCharArray(job_outcome.test_false_neg_per),
                                                                                                                job_outcome.result_path.c_str(), job_outcome.note.c_str(),
                                                                                                                integerToCharArray(job_outcome.last_round_job_id)) ;

#ifdef                                                       READ_WGT_FROM_FILE
                                                             string weightStr = readWeightsIntoString(path, result_files(k+parse_file_string(inputfilename.trainedweightlastepoch),       // populate weight_matrix table
                                                                                                      "/TrainedWeights_", "_epochs.dat")) ;
#endif

#ifdef                                                       READ_WGT_FROM_DB
                                                             string weightStr = writeWeightMatrixIntoString(N_input+1, N_1hidden+1, N_2hidden+1, N_output+1) ;
#endif

                                                             weight_matrix_table(job_outcome.job_outcome_id, path, result_files(k+parse_file_string(inputfilename.trainedweightlastepoch),
                                                                                 "/TrainedWeights_", "_epochs.dat"), weightStr);

                                                             insertWeightMatrixTable(conn, integerToCharArray(weight_matrix.job_outcome_id), weight_matrix.path.c_str(),
                                                                                     weight_matrix.name.c_str(), weight_matrix.matrix.c_str()) ;
                                              }

				           write_file_epoch_count = 1 ;
				   }

	    }
#endif

}

void printPredictions(string path, string resultFileString, string inputfile,
                      float alfa, unsigned int N_input, unsigned int N_1hidden,
                      unsigned int N_2hidden, unsigned int N_output)
{
           ifstream testFileInput ;
           testFileInput.open(inputfile.c_str());
           openInputFilesFailed(testFileInput);
           TN = FP = TP = FN = 0 ;

           length = 1 ;
           while (!testFileInput.eof())  {
                               string s ;
                               getline(testFileInput, s) ;
                               arrayAssign(N_input, N_output, s) ;
                               gpuFeedforwardCompute(alfa, N_input, N_1hidden, N_2hidden, N_output) ;
                               fMeasureCompute(Y_dout_h[1], Y_out_h[1]) ;
                               length++ ;
           }
           testFileInput.close() ;

          /*
             printMeasureMatrix(path, resultFileString, TP, TN, FP, FN, length) ;
          */
}

void printTestPredictions(string path, string resultFileString, string inputfile,
                      float alfa, unsigned int N_input, unsigned int N_1hidden,
                      unsigned int N_2hidden, unsigned int N_output)
{
           ifstream testFileInput ;
           testFileInput.open(inputfile.c_str());
           openInputFilesFailed(testFileInput);
           TN_TEST = FP_TEST = TP_TEST = FN_TEST = 0 ;

           length = 1 ;
           while (!testFileInput.eof())  {
                               string s ;
                               getline(testFileInput, s) ;
                               arrayAssign(N_input, N_output, s) ;
                               gpuFeedforwardCompute(alfa, N_input, N_1hidden, N_2hidden, N_output) ;
                               fMeasureTestCompute(Y_dout_h[1], Y_out_h[1]) ;
                               length++ ;
           }
           testFileInput.close() ;

           /*
               printMeasureMatrix(path, resultFileString, TP_TEST, TN_TEST, FP_TEST, FN_TEST, length) ;
           */
}


void predictByTrainedWeights(string path, string resultFileString, string inputfile,
                             float alfa, unsigned int N_input, unsigned int N_1hidden,
                             unsigned int N_2hidden, unsigned int N_output, unsigned int tn,
                             unsigned int fp, unsigned int tp, unsigned int fn)
{
           ifstream  newFileInput ;
           gpuVectorMemoryAssign(N_input, N_1hidden, N_2hidden, N_output) ;
	   checkCudaErrors(cudaMemcpy( W_1_d, W_1_h, (N_input+1)*(N_1hidden+1)*sizeof(float), cudaMemcpyHostToDevice));
	   checkCudaErrors(cudaMemcpy( W_2_d, W_2_h, (N_1hidden+1)*(N_2hidden+1)*sizeof(float), cudaMemcpyHostToDevice));
	   checkCudaErrors(cudaMemcpy( W_3_d, W_3_h, (N_2hidden+1)*(N_output+1)*sizeof(float), cudaMemcpyHostToDevice));
           TN = FP = TP = FN = 0 ;

           newFileInput.open(inputfile.c_str()) ;
           openInputFilesFailed(newFileInput);
           length = 1 ;
	   while (!newFileInput.eof())  {
	 	               string s ;
                               getline(newFileInput, s) ;
                               arrayAssign(N_input, N_output, s) ;
                               gpuFeedforwardCompute(alfa, N_input, N_1hidden, N_2hidden, N_output) ;
                    	       fMeasureCompute(Y_dout_h[1], Y_out_h[1]) ;
                               length++ ;
	   }
           newFileInput.close() ;

          /*
             printMeasureMatrix(path, resultFileString, TP,  TN, FP, FN, length) ;
          */
}


#endif
