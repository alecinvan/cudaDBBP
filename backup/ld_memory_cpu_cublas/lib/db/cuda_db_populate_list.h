#ifndef _CUDA_DB_POPULATE_LIST_H_
#define _CUDA_DB_POPULATE_LIST_H_

#include <stdio.h>


void slurm_job_nn_parameters_tables(int argc, char** argv, int training_data_id, int gpu_status_id) ;

void populate_parameter_job_tables(PGconn *conn);

void weight_matrix_table(int job_outcome_id, string path, string name, string matrix);

void job_outcome_table(int slurm_job_id, int n_epoch, int tp, int tn, int fp, int fn, int tp_test, int tn_test, int fp_test, int fn_test,
                       int p_len, int n_len, int p_len_test, int n_len_test, string result_path, string note, int last_round_job_id);

#endif
