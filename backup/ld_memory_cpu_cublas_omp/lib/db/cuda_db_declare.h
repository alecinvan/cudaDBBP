#ifndef _CUDA_DB_DECLARE_H_
#define _CUDA_DB_DECLARE_H_


struct job
{
     int         slurm_job_id;
     int         training_data_id;
     int         nn_parameter_id;
     int         gpu_status_id;
     int         job_outcome_id;
     int         stage;
     int         job;
     int         gpu;
     string      node;
     string      screen_output_path;
     string      log_path;
     string      last_round_weight_path;  // database was changed name
     string      note;
} slurm_job;



struct parameter
{
     int         nn_parameter_id;
     int         n_input;
     int         n_1hidden;
     int         n_2hidden;
     int         n_3hidden;
     int         n_output;
     float       u;
     float       alfa;
     int         epochs;
     string      file_path;
} nn_parameters;


struct outcome
{
     int         job_outcome_id;
     int         slurm_job_id;
     int         n_epoch;
     int         true_pos;
     float       true_pos_percentage;
     int         true_neg;
     float       true_neg_percentage;
     int         false_pos;
     float       false_pos_percentage;
     int         false_neg;
     float       false_neg_percentage;
     float       test_true_pos_per;
     float       test_true_neg_per;
     float       test_false_pos_per;
     float       test_false_neg_per;
     string      result_path;
     string      note;
     int         last_round_job_id;     // database was changed name and attribute
} job_outcome;


struct status
{
     int         gpu_status_id;
     char        gpu_status[5];
} gpu_status;


struct weight
{
     int         weight_matrix_id;
     int         job_outcome_id;
     string      path;
     string      name;
     string      matrix;
} weight_matrix;


struct binary_encode
{
     int         encode_id;
     char        not_to_say[30];
     char        no[30];
     char        yes[30];
     char        all[30];
} datekids_encode, datesmokers_encode, havekids_encode, smoke_encode, wantkids_encode;


struct file_name_string
{
     string      inputfilestr;
     string      testsetstr;
     string      trainedweightstr;
     string      trainedweightlastepoch;
} inputfilename;


#endif
