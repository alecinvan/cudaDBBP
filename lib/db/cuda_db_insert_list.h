#ifndef _CUDA_DB_INSERT_LIST_H_
#define _CUDA_DB_INSERT_LIST_H_

int insertNNparameterTable(PGconn *conn, const char *n_input, const char *n_1hidden, const char *n_2hidden,
                            const char *u, const char *alfa, const char* epochs, const char* file_out_path, const char* beta, const char* mean, const char* mean_test);

int insertSlurmJobTable(PGconn *conn, const char *training_data_id, const char *nn_parameter_id, const char *gpu_status_id, const char *stage,
                         const char *job, const char *node, const char *gpu, const char *screen_output_path, const char *log_path,
                         const char *last_round_weight_path);

int insertJobOutcomeTable(PGconn *conn, const char *slurm_job_id, const char *n_epoch, const char *num_accurate, const char *num_accurate_test,
                          const char *ss_tot, const char *ss_reg, const char *ss_res, const char *r_squared, const char *ss_tot_test, const char *ss_reg_test,
                          const char *ss_res_test, const char *r_squared_test, const char *f_ratio, const char *f_ratio_test, const char *mspe,
                          const char *sd, const char *rmse, const char *mspe_test, const char *sd_test, const char *rmse_test, const char *result_path, const char *last_round_job_id) ;

void insertWeightMatrixTable(PGconn *conn, const char *job_outcome_id, const char *path, const char *name, const char *matrix);

string fetchWeightString(PGconn *conn, const char *job_id, const char *outcome_id);

#endif
