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
     float       beta;
     float       mean;
     float       mean_test;
} nn_parameters;


struct outcome
{
     int         job_outcome_id;
     int         slurm_job_id;
     int         n_epoch;
     int         true_pos;
     float       sensitivity;
     int         true_neg;
     float       specificity;
     int         false_pos;
     float       pos_predictive;
     int         false_neg;
     float       neg_predictive;
     float       sensitivity_on_test;
     float       specificity_on_test;
     float       pos_pred_on_test;
     float       neg_pred_on_test;
     string      result_path;
     string      note;
     int         last_round_job_id;     // database was changed name and attribute
     float       ss_tot          ;    //  total sum of squares
     float       ss_reg          ;    //  explained sum of squares
     float       ss_res          ;    //  residual sum of squares
     float       r_squared       ;    //  measure how well data points fit a statistical model
     float       ss_tot_test     ;
     float       ss_reg_test     ;
     float       ss_res_test     ;
     float       r_squared_test  ;
     int         num_accurate;
     int         num_accurate_test;
     float       f_ratio;
     float       p_value;
     float       f_ratio_test;
     float       p_value_test;
     float       mspe;
     float       sd;
     float       rmse;
     float       mspe_test;
     float       sd_test;
     float       rmse_test;
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
