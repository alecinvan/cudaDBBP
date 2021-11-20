#ifndef _CUDA_DB_INSERT_LIST_H_
#define _CUDA_DB_INSERT_LIST_H_

int insertNNparameterTable(PGconn *conn, const char *n_input, const char *n_1hidden, const char *n_2hidden,
                            const char *u, const char *alfa, const char* epochs, const char* file_out_path);

int insertSlurmJobTable(PGconn *conn, const char *training_data_id, const char *nn_parameter_id, const char *gpu_status_id, const char *stage,
                         const char *job, const char *node, const char *gpu, const char *screen_output_path, const char *log_path, 
                         const char *last_round_weight_path);

int insertJobOutcomeTable(PGconn *conn, const char *slurm_job_id, const char *n_epoch, const char *true_pos, const char *true_pos_percentage,
                          const char *true_neg, const char *true_neg_percentage, const char *false_pos, const char *false_pos_percentage,
                          const char *false_neg, const char *false_neg_percentage, const char *test_true_pos_per, const char *test_true_neg_per,
                          const char *test_false_pos_per, const char *test_false_neg_per, const char *result_path, const char *note,
                          const char *last_round_job_id);

void insertWeightMatrixTable(PGconn *conn, const char *job_outcome_id, const char *path, const char *name, const char *matrix);

string fetchWeightString(PGconn *conn, const char *job_id, const char *outcome_id);

#endif
