/*###############################################################
#  CUDA_BP.CU:   VERSION 0.5.2                                  #
#  COPYRIGHT 2015, SIMUMIND TECHNOLOGY INC. ALL RIGHTS RESERVED #  
#                                                               #
#  THIS FILE CONSTAINS CONFIDENTIAL INTELLECTUAL PROPERTY OF    #
#  SIMUMIND TECHNOLOGY INC.                                     #
#                                                               #
#  AUTHOR: SA LI                                                #
#  DATE:   2015-4-4                                             #
#                                                               #     
#  CONTENT: THE BODY OF STATISTICAL PREDICTIVE MODEL BY USING   #
#           MULTILAYER NEURAL NETWORK TRAINED BY BP ALGORITHM.  # 
#                                                               #
#  DESCRIPTION:                                                 #
#           THIS C++/OpenCL CODE IS WRITTEN IN LINUX CUDA       #
#           PARALLEL COMPUTING PLATFORM, RUN ON GPUs AND        #
#           CONNECT TO POSTGRESQL DATABASE. THE JOB INVOKES     #
#           THE MULTIPLE TRAINING STAGES BASED ON ARGUMENTS     #
#           SPECIFIED IN COMMAND-LINE.                          #
#                                                               #
#  COMPILE:                                                     #  
#           make Makefile                                       #
################################################################*/


#include "/cluster/clusterPipeline/MonteCarlo/CUDADBBP_MEM_OPT_MOMENTUM/lib/train/cuda_training_process.h"

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
           slurm_job_nn_parameters_tables(argc, argv, 88, 1, 0);
        
                           
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

                        
  
                                               /*++++++++++++++++++++++++++++++++++++++++++++
                                               |  cuda_nn_functions::global_variable_define |
                                               |  to define host matrices                   |
                                               ++++++++++++++++++++++++++++++++++++++++++++*/                                                                                     
           global_variable_define(nn_parameters.n_input,  
                                  nn_parameters.n_1hidden,  
                                  nn_parameters.n_2hidden,  
                                  nn_parameters.n_output 
                                  );
   

                                               /*++++++++++++++++++++++++++++++++++++++++++++
                                               |   cuda_help_functions::create_weight_dir   |
                                               |   creates path nn_parameters.file_path     |
                                               |   to store weight matrices                 |
                                               ++++++++++++++++++++++++++++++++++++++++++++*/                                                       
           create_weight_dir(nn_parameters.file_path) ;   



                                               /*################### 
                                               #                   #
                                               #   training stage  #
                                               #                   #
                                               ###################*/

#ifdef     STAGE_OF_TRAINING
 
                                               /*++++++++++++++++++++++++++++++++++++++++++++
                                               |  cuda_nn_functions::writeParameterIntoFile |
                                               |  write parameters into run.log             |
                                               ++++++++++++++++++++++++++++++++++++++++++++*/                                                                         
           writeParameterIntoFile(nn_parameters.file_path, 
                                  PARAMETERSTR, 
                                  nn_parameters.u, 
                                  nn_parameters.alfa,
                                  nn_parameters.beta, 
                                  nn_parameters.n_input, 
                                  nn_parameters.n_1hidden, 
                                  nn_parameters.n_2hidden, 
                                  nn_parameters.n_output, 
                                  nn_parameters.epochs
                                  ) ;

                                               /*++++++++++++++++++++++++++++++++++++++++++++++++
                                               | cuda_training_process::gpuNeuralNetworkBPtrain |
                                               | starts to training, write weight matrices and  |
                                               | results into files and database every 20 epochs|
                                               ++++++++++++++++++++++++++++++++++++++++++++++++*/                                                         
           gpuNeuralNetworkBPtrain(conn, 
                                   nn_parameters.file_path, 
                                   slurm_job.last_round_weight_path, 
                                   inputfilename.trainedweightlastepoch, 
                                   inputfilename.inputfilestr, 
                                   inputfilename.testsetstr, 
                                   slurm_job.slurm_job_id, 
                                   nn_parameters.u, 
                                   nn_parameters.alfa,
                                   nn_parameters.beta,
                                   nn_parameters.mean,
                                   nn_parameters.mean_test, 
                                   nn_parameters.n_input, 
                                   nn_parameters.n_1hidden, 
                                   nn_parameters.n_2hidden, 
                                   nn_parameters.n_output, 
                                   nn_parameters.epochs, 
                                   slurm_job.stage
                                   ) ;
                                                       
                                               /*+++++++++++++++++++++++++++++++++++++++++++++        
                                               |   cuda_nn_functions::writeWeightsIntoFile   |
                                               |   writes weights into .dat file             |
                                               +++++++++++++++++++++++++++++++++++++++++++++*/                                                        
/*
           writeWeightsIntoFile(nn_parameters.file_path, 
                                inputfilename.trainedweightstr, 
                                nn_parameters.n_input, 
                                nn_parameters.n_1hidden,   
                                nn_parameters.n_2hidden, 
                                nn_parameters.n_output
                                ) ;
*/

           gpuVectorMemoryClean() ;

#endif


                                              /*##################
                                              #                  # 
                                              #   predict stage  #
                                              #                  #
                                              ##################*/
#ifdef     STAGE_OF_PREDICTING                  

                                                /*++++++++++++++++++++++++++++++++++++++++++
                                                |  cuda_nn_functions::readWeightsIntoFile  |
                                                |  from last .dat file from last round     |
                                                ++++++++++++++++++++++++++++++++++++++++++*/
#ifdef     READ_WGT_FROM_FILE
           readWeightsFromFile(nn_parameters.file_path, 
                               inputfilename.trainedweightlastepoch, 
                               nn_parameters.n_input, 
                               nn_parameters.n_1hidden, 
                               nn_parameters.n_2hidden, 
                               nn_parameters.n_output
                               ) ;
#endif
#ifdef     READ_WGT_FROM_DB
           string weightMatrixString ;
           weightMatrixString = fetchWeightString(conn, 
                                                  integerToCharArray(lastjobid),                                               
                                                  integerToCharArray(lastoutcomeid)
                                                  ) ;
           convertStringToArray(weightMatrixString, 
                                nn_parameters.n_input, 
                                nn_parameters.n_1hidden, 
                                nn_parameters.n_2hidden, 
                                nn_parameters.n_output
                                ) ;
#endif

                                                /*++++++++++++++++++++++++++++++++++++++++++++++++
                                                | cuda_training_process::predictByTrainedWeights |
                                                | predicts the test set                          |
                                                ++++++++++++++++++++++++++++++++++++++++++++++++*/
           measureByRegressionNNWeight(inputfilename.inputfilestr, 
                                       nn_parameters.alfa, 
                                       nn_parameters.n_input, 
                                       nn_parameters.n_1hidden,
                                       nn_parameters.n_2hidden, 
                                       nn_parameters.n_output, 
                                       nn_parameters.mean
                                   ) ; 

           measureTestByRegressionNNWeight(inputfilename.testsetstr, 
                                           nn_parameters.alfa, 
                                           nn_parameters.n_input, 
                                           nn_parameters.n_1hidden, 
                                           nn_parameters.n_2hidden, 
                                           nn_parameters.n_output, 
                                           nn_parameters.mean_test
                                   ) ;
           
                                                 
                                                 /*++++++++++++++++++++++++++++++++++++++++++++++++++
                                                 |    populate last epoch results into job_outcome  |
                                                 |  table, write last epoch as nn_parameters.epochs |
                                                 ++++++++++++++++++++++++++++++++++++++++++++++++++*/  
           job_outcome_table(slurm_job.slurm_job_id, 
                             nn_parameters.epochs, 
                             N_ACCURATE,
                             N_ACCURATE_TEST,
                             SS_TOT,
                             SS_REG,
                             SS_RES,
                             R_SQUARED,
                             SS_TOT_TEST,
                             SS_REG_TEST,
                             SS_RES_TEST,
                             R_SQUARED_TEST,
                             F_RATIO,
                             F_RATIO_TEST, 
                             MSPE,
                             SD,
                             RMSE,
                             MSPE_TEST,
                             SD_TEST,
                             RMSE_TEST,
                             nn_parameters.file_path,                              
                             lastjobid
                             ) ;      
           job_outcome.job_outcome_id = insertJobOutcomeTable(conn, 
                                                              integerToCharArray(job_outcome.slurm_job_id),
                                                              integerToCharArray(job_outcome.n_epoch),
                                                              integerToCharArray(job_outcome.num_accurate),
                                                              integerToCharArray(job_outcome.num_accurate_test),
                                                              floatToCharArray(job_outcome.ss_tot),
                                                              floatToCharArray(job_outcome.ss_reg),
                                                              floatToCharArray(job_outcome.ss_res),
                                                              floatToCharArray(job_outcome.r_squared),
                                                              floatToCharArray(job_outcome.ss_tot_test),
                                                              floatToCharArray(job_outcome.ss_reg_test),
                                                              floatToCharArray(job_outcome.ss_res_test),
                                                              floatToCharArray(job_outcome.r_squared_test),
                                                              floatToCharArray(job_outcome.f_ratio),
                                                              floatToCharArray(job_outcome.f_ratio_test),
                                                              floatToCharArray(job_outcome.mspe),
                                                              floatToCharArray(job_outcome.sd),
                                                              floatToCharArray(job_outcome.rmse),
                                                              floatToCharArray(job_outcome.mspe_test),
                                                              floatToCharArray(job_outcome.sd_test),
                                                              floatToCharArray(job_outcome.rmse_test),
                                                              job_outcome.result_path.c_str(),                                                                
                                                              integerToCharArray(job_outcome.last_round_job_id) 
                                                              ) ;                                                             
                                 

                                                  /*++++++++++++++++++++++++++++++++++++++++++++++++++
                                                  |  populate into weight_matrix table, really need? |                                                          
                                                  ++++++++++++++++++++++++++++++++++++++++++++++++++*/          
           weight_matrix_table(job_outcome.job_outcome_id, 
                               nn_parameters.file_path, 
                               inputfilename.trainedweightstr, 
                               weightMatrixString
                               );
           insertWeightMatrixTable(conn, 
                                   integerToCharArray(weight_matrix.job_outcome_id), 
                                   weight_matrix.path.c_str(), 
                                   weight_matrix.name.c_str(), 
                                   weight_matrix.matrix.c_str()
                                   ) ;
                                                        
                                                
           gpuVectorMemoryClean() ;

#endif

           timestamp();
           closeConn(conn) ;
           return 0 ;

}






