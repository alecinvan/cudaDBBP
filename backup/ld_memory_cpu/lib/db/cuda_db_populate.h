/*#############################################
 # functions related to postgresql to populate
 # cluster information into cudaDB
 #
 # Author: Sa Li
##############################################*/

#ifndef _CUDA_DB_POPULATE_H_
#define _CUDA_DB_POPULATE_H_

/////////////////////////////////////
#include "./cuda_db_insert.h"
#include "./cuda_db_populate_list.h"
/////////////////////////////////////


void slurm_job_nn_parameters_tables(int argc, char** argv, int training_data_id, int gpu_status_id, int job_outcome_id)
{
           //    for (unsigned int ii = 1; ii <= 17 ; ii++)  cout << "argv[" << ii <<"]" << argv[ii] << endl;
           deviceID                               =  atoi(argv[1])    ;
           nn_parameters.u                        =  atof(argv[2])    ;
           nn_parameters.alfa                     =  atof(argv[3])    ;
           nn_parameters.n_input                  =  atoi(argv[4])    ;
           nn_parameters.n_1hidden                =  atoi(argv[5])    ;
           nn_parameters.n_2hidden                =  atoi(argv[6])    ;
           nn_parameters.n_output                 =  atoi(argv[7])    ;
           nn_parameters.epochs                   =  atoi(argv[8])    ;
           inputfilename.inputfilestr             =  argv[9]          ;
           inputfilename.testsetstr               =  argv[10]         ;
           nn_parameters.file_path                =  argv[11]         ;
           nn_parameters.file_path                =  nn_parameters.file_path+FORWARDSLASH+argv[12]+FORWARDSLASH+argv[13]+FORWARDSLASH+argv[14]+WEIGHTDIRSTR ;
           slurm_job.node                         =  argv[12]         ;
           slurm_job.gpu                          =  atoi(argv[13])   ;
           slurm_job.job                          =  atoi(argv[14])   ;
           slurm_job.stage                        =  atoi(argv[15])   ;
           inputfilename.trainedweightstr         =  argv[16]         ;
           inputfilename.trainedweightlastepoch   =  argv[17]         ;
           P_Len                                  =  atoi(argv[18])   ;
           N_Len                                  =  atoi(argv[19])   ;
           P_Len_test                             =  atoi(argv[20])   ;
           N_Len_test                             =  atoi(argv[21])   ;
           lastjobid                              =  atoi(argv[22])   ;
           lastoutcomeid                          =  atoi(argv[23])   ;
           slurm_job.last_round_weight_path       =  argv[24]         ;
           slurm_job.training_data_id             =  training_data_id ;
           slurm_job.gpu_status_id                =  gpu_status_id    ;
           slurm_job.job_outcome_id               =  job_outcome_id   ;  // this is not exactly same as the job_outcome_id from job_outcome table
           slurm_job.screen_output_path           =  argv[11]         ;
           slurm_job.screen_output_path           =  slurm_job.screen_output_path+FORWARDSLASH+argv[12]+FORWARDSLASH+argv[13]+FORWARDSLASH+argv[14] ;
           slurm_job.log_path                     =  nn_parameters.file_path ;
}

void job_outcome_table(int slurm_job_id, int n_epoch, int tp, int tn, int fp, int fn, int tp_test, int tn_test, int fp_test, int fn_test,
                       int p_len, int n_len, int p_len_test, int n_len_test, string result_path, string note, int last_round_job_id)
{
           job_outcome.slurm_job_id               = slurm_job_id ;
           job_outcome.n_epoch                    = n_epoch ;
           job_outcome.true_pos                   = tp ;
           job_outcome.true_pos_percentage        = (float) tp/p_len * 100 ;
           job_outcome.true_neg                   = tn ;
           job_outcome.true_neg_percentage        = (float) tn/n_len * 100 ;
           job_outcome.false_pos                  = fp ;
           job_outcome.false_pos_percentage       = (float) fp/n_len * 100 ;
           job_outcome.false_neg                  = fn ;
           job_outcome.false_neg_percentage       = (float) fn/p_len * 100 ;
           job_outcome.test_true_pos_per          = (float) tp_test/p_len_test * 100 ;
           job_outcome.test_true_neg_per          = (float) tn_test/n_len_test * 100 ;
           job_outcome.test_false_pos_per         = (float) fp_test/n_len_test * 100 ;
           job_outcome.test_false_neg_per         = (float) fn_test/p_len_test * 100 ;
           job_outcome.result_path                = result_path ;
           job_outcome.note                       = note ;
           job_outcome.last_round_job_id          = last_round_job_id ;
}

void weight_matrix_table(int job_outcome_id, string path, string name, string matrix)
{
           weight_matrix.job_outcome_id           = job_outcome_id ;
           weight_matrix.path                     = path ;
           weight_matrix.name                     = name ;
           weight_matrix.matrix                   = matrix ;
}

void populate_parameter_job_tables(PGconn *conn)
{

           if (conn != NULL)
           {
                         nn_parameters.nn_parameter_id  = insertNNparameterTable(conn, integerToCharArray(nn_parameters.n_input), integerToCharArray(nn_parameters.n_1hidden),
                                                                                 integerToCharArray(nn_parameters.n_2hidden), floatToCharArray(nn_parameters.u),
                                                                                 floatToCharArray(nn_parameters.alfa), integerToCharArray(nn_parameters.epochs),
                                                                                 nn_parameters.file_path.c_str()) ;

                         slurm_job.nn_parameter_id      =  nn_parameters.nn_parameter_id   ;

                         slurm_job.slurm_job_id         =  insertSlurmJobTable(conn, integerToCharArray(slurm_job.training_data_id), integerToCharArray(nn_parameters.nn_parameter_id),
                                                                               integerToCharArray(slurm_job.gpu_status_id), integerToCharArray(slurm_job.job_outcome_id), integerToCharArray(slurm_job.stage),
                                                                               integerToCharArray(slurm_job.job), slurm_job.node.c_str(), integerToCharArray(slurm_job.gpu),
                                                                               slurm_job.screen_output_path.c_str(), slurm_job.log_path.c_str(), slurm_job.last_round_weight_path.c_str()) ;
           }

}


#endif
