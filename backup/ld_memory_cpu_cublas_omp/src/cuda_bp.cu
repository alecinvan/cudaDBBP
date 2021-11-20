/*###############################################################
#  cuda_bp.cu : version 0.3.4                                   #
#  Copyright 2012, Plentyoffish Media Inc. All rights reserved  #  
#                                                               #
#  This file contains confidential intellectual property of     #
#  Plentyoffish Media Inc.                                      #
#                                                               #
#  Author: Sa Li                                                #
#  Date:   2013-4-4                                             #
#                                                               #     
#  Content: The body of statistical predictive model by using   #
#           multilayer neural network trained by BP algorithm.  # 
#                                                               #
#  Description: This C++/OpenCL code is written in linux cuda   #
#               parallel computing platform, run on GPUs and    #
#               connect to postgresql database. The job invokes #
#               the mutiple training stages based on arguments  #
#               specified in command-line.                      #
#                                                               #
#  Compile:                                                     #  
#               make Makefile                                   #
################################################################*/


//#include "../lib/train/cuda_training_process.h"
#include "/cluster/clusterPipeline/MonteCarlo/CUDADBBP/lib/train/cuda_training_process.h"

#ifdef __cplusplus
extern "C"
#endif

using namespace std ;


           /*====================
           #    main program    #
           =====================*/

int main(int argc, char** argv)
{
           timestamp() ;
                           
                                  /*+++++++++++++++++++++++++++++++++++++
                                  |  taking arguments from commandline  |
                                  |  training_data_id need to changed   |
                                  |  here while applying a new data set |
                                  +++++++++++++++++++++++++++++++++++++*/
           slurm_job_nn_parameters_tables(argc, argv, 6, 1, 0);  
        
                           
                                      /*++++++++++++++++++++++++++
                                      |       gpu testing        |
                                      | deviceID is set 0 always |
                                      | in slurm cluster run     |
                                      ++++++++++++++++++++++++++*/
           checkCudaErrors(cudaSetDevice(deviceID)) ; 
           deviceTest(argc, argv) ;
           arrayMultiplication() ;


                               /*+++++++++++++++++++++++++++++++++++++++++
                               |  write job and parameters into database |
                               |  populate two tables:                   |
                               |          - nn_parameters                |
                               |          - slurm_job                    |
                               +++++++++++++++++++++++++++++++++++++++++*/
           PGconn      *conn   =          NULL ;
           conn                =   connectDB(readDBconfToString(DB_CONF)) ;

           populate_parameter_job_tables(conn);       

                        
  
                                               /*++++++++++++++++++++++++++++++++++++
                                               |        job training process        |
                                               ++++++++++++++++++++++++++++++++++++*/    
       
                                                       /* cuda_nn_functions::global_variable_define to define host matrices 
                                                        */                                                                                 
           global_variable_define(nn_parameters.n_input,  nn_parameters.n_1hidden,  nn_parameters.n_2hidden,  nn_parameters.n_output) ;
   
                                                       /* cuda_help_functions::create_weight_dir creates path nn_parameters.file_path 
                                                          to store weight matrices 
                                                       */
           create_weight_dir(nn_parameters.file_path) ;   

                                               /*################### 
                                               #   training stage  #
                                               ###################*/

#ifdef     STAGE_OF_TRAINING
 
                                                       /* cuda_nn_functions::writeParameterIntoFile write parameters into run.log 
                                                        */                                                                         
           writeParameterIntoFile(nn_parameters.file_path, PARAMETERSTR, nn_parameters.u, nn_parameters.alfa, nn_parameters.n_input, 
                                  nn_parameters.n_1hidden, nn_parameters.n_2hidden, nn_parameters.n_output, nn_parameters.epochs) ;

                                                       /* cuda_training_process::gpuNeuralNetworkBPtrain starts to training, write
                                                          weight matrices and results into files and database every 20 epochs 
                                                        */
           gpuNeuralNetworkBPtrain(conn, nn_parameters.file_path, slurm_job.last_round_weight_path, inputfilename.trainedweightlastepoch, 
                                   inputfilename.inputfilestr, inputfilename.testsetstr, slurm_job.slurm_job_id, nn_parameters.u, nn_parameters.alfa, 
                                   nn_parameters.n_input, nn_parameters.n_1hidden, nn_parameters.n_2hidden, nn_parameters.n_output, 
                                   nn_parameters.epochs, slurm_job.stage) ;
                                                       
                                                       /* cuda_nn_functions::writeWeightsIntoFile writes weights into .dat file 
                                                        */
//           writeWeightsIntoFile(nn_parameters.file_path, inputfilename.trainedweightstr, nn_parameters.n_input, nn_parameters.n_1hidden,   
//                                nn_parameters.n_2hidden, nn_parameters.n_output) ;

           gpuVectorMemoryClean() ;

#endif


                                              /*##################
                                              #   predict stage  #
                                              ##################*/
#ifdef     STAGE_OF_PREDICTING                  

                                                       /* cuda_nn_functions::readWeightsIntoFile from last .dat file from last round
                                                        */

#ifdef              READ_WGT_FROM_FILE
                    readWeightsFromFile(nn_parameters.file_path, inputfilename.trainedweightlastepoch, nn_parameters.n_input, nn_parameters.n_1hidden, 
                                        nn_parameters.n_2hidden, nn_parameters.n_output) ;
#endif

#ifdef              READ_WGT_FROM_DB
                    string weightMatrixString ;
                    weightMatrixString = fetchWeightString(conn, integerToCharArray(lastjobid), integerToCharArray(lastoutcomeid)) ;
                    convertStringToArray(weightMatrixString, nn_parameters.n_input, nn_parameters.n_1hidden, nn_parameters.n_2hidden, nn_parameters.n_output) ;
#endif


                                                       /* cuda_training_process::predictByTrainedWeights predicts the test set
                                                        */
           predictByTrainedWeights(nn_parameters.file_path, RESULTSTR+static_cast<ostringstream*>(&(ostringstream()<<slurm_job.stage))->str(),
                                   inputfilename.inputfilestr, nn_parameters.alfa, nn_parameters.n_input, nn_parameters.n_1hidden,
                                   nn_parameters.n_2hidden, nn_parameters.n_output, TN, FP, TP, FN) ; 

           predictByTrainedWeights(nn_parameters.file_path, RESULTSTR+static_cast<ostringstream*>(&(ostringstream()<<slurm_job.stage))->str(), 
                                   inputfilename.testsetstr, nn_parameters.alfa, nn_parameters.n_input, nn_parameters.n_1hidden, 
                                   nn_parameters.n_2hidden, nn_parameters.n_output, TN_TEST, FP_TEST, TP_TEST, FN_TEST) ;
           

                                                       /* populate last epoch results int job_outcome table, write last epoch as 
                                                          nn_parameters.epochs
                                                        */  
           job_outcome_table(slurm_job.slurm_job_id, nn_parameters.epochs, TP, TN, FP, FN, TP_TEST, TN_TEST, FP_TEST, FN_TEST, P_Len, N_Len, 
                             P_Len_test, N_Len_test, nn_parameters.file_path, "", lastjobid) ;      
           job_outcome.job_outcome_id = insertJobOutcomeTable(conn, integerToCharArray(job_outcome.slurm_job_id), integerToCharArray(job_outcome.n_epoch),
                                                              integerToCharArray(job_outcome.true_pos), floatToCharArray(job_outcome.true_pos_percentage),
                                                              integerToCharArray(job_outcome.true_neg), floatToCharArray(job_outcome.true_neg_percentage),
                                                              integerToCharArray(job_outcome.false_pos), floatToCharArray(job_outcome.false_pos_percentage),
                                                              integerToCharArray(job_outcome.false_neg), floatToCharArray(job_outcome.false_neg_percentage),
                                                              floatToCharArray(job_outcome.test_true_pos_per), floatToCharArray(job_outcome.test_true_neg_per),
                                                              floatToCharArray(job_outcome.test_false_pos_per), floatToCharArray(job_outcome.test_false_neg_per),
                                                              job_outcome.result_path.c_str(), job_outcome.note.c_str(), 
                                                              integerToCharArray(job_outcome.last_round_job_id) ) ;                                                             
                                 

                                                       /* populate into weight_matrix table, really need?                                                           
                                                        */          
           weight_matrix_table(job_outcome.job_outcome_id, nn_parameters.file_path, inputfilename.trainedweightstr, weightMatrixString);
           insertWeightMatrixTable(conn, integerToCharArray(weight_matrix.job_outcome_id), weight_matrix.path.c_str(), weight_matrix.name.c_str(), 
                                   weight_matrix.matrix.c_str()) ;
                                                        
                                                
           gpuVectorMemoryClean() ;

#endif

           timestamp();
           closeConn(conn) ;
           return 0 ;

}






