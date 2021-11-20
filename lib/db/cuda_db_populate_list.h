 #ifndef _CUDA_DB_POPULATE_LIST_H_
#define _CUDA_DB_POPULATE_LIST_H_

#include <stdio.h>


void slurm_job_nn_parameters_tables(int argc, char** argv, int training_data_id, int gpu_status_id) ;

void populate_parameter_job_tables(PGconn *conn);

void weight_matrix_table(int job_outcome_id, string path, string name, string matrix);

void job_outcome_table(int slurm_job_id, int n_epoch, int num_accurate, int num_accurate_test, float ss_tot, float ss_reg, float ss_res, float r_squared,
                       float ss_tot_test, float ss_reg_test, float ss_res_test, float r_squared_test, float f_ratio, float f_ratio_test, float mspe,
                       float sd, float rmse, float mspe_test, float sd_test, float rmse_test, string result_path, int last_round_job_id);


#endif
