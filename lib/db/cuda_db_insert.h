/*#########################################
 # FUNCTIONS RELATED TO POSTGRESQL INSERT
 # AUTHOR: SA LI
#########################################*/

#ifndef _CUDA_DB_INSERT_H_
#define _CUDA_DB_INSERT_H_

//================================
#include "./cuda_db_connect.h"
#include "./cuda_db_declare.h"
#include "./cuda_db_insert_list.h"
//================================


int insertNNparameterTable(PGconn *conn, const char *n_input, const char *n_1hidden, const char *n_2hidden, const char *u, const char *alfa,
                           const char* epochs, const char* file_path, const char* beta, const char* mean, const char* mean_test)
{
          int id ;
          std::string sSQL;
          sSQL.append("INSERT INTO nn_parameters (n_input, n_1hidden, n_2hidden, u, alfa, epochs, file_path, beta, mean, mean_test) VALUES('");
          sSQL.append(n_input);
          sSQL.append("', '");
          sSQL.append(n_1hidden);
          sSQL.append("', '");
          sSQL.append(n_2hidden);
          sSQL.append("', '");
          sSQL.append(u);
          sSQL.append("', '");
          sSQL.append(alfa);
          sSQL.append("', '");
          sSQL.append(epochs);
          sSQL.append("', '");
          sSQL.append(file_path);
          sSQL.append("', '");
          sSQL.append(beta);
          sSQL.append("', '");
          sSQL.append(mean);
          sSQL.append("', '");
          sSQL.append(mean_test);
          sSQL.append("') RETURNING nn_parameter_id");

          PGresult *res = PQexec(conn, sSQL.c_str());
          if (PQresultStatus(res) != PGRES_TUPLES_OK)
          {
                 cout << "Insert nn_parameters record failed\n";
                 PQclear(res);
                 closeConn(conn);
          }
          cout << "Insert nn_parameters record - OK\n";
          id = atoi(PQgetvalue(res, 0, 0)) ;
          PQclear(res);
          return id;
}


int insertSlurmJobTable(PGconn *conn, const char *training_data_id, const char *nn_parameter_id, const char *gpu_status_id, const char *job_outcome_id,
                        const char *stage, const char *job, const char *node, const char *gpu, const char *screen_output_path, const char *log_path,
                        const char *last_round_weight_path)
{
          int id;
          std::string sSQL;
          sSQL.append("INSERT INTO slurm_job (training_data_id, nn_parameter_id, gpu_status_id, job_outcome_id, stage, job, node, gpu, screen_output_path, log_path, last_round_weight_path, time_stamp) VALUES ('");
          sSQL.append(training_data_id);
          sSQL.append("', '");
          sSQL.append(nn_parameter_id);
          sSQL.append("', '");
          sSQL.append(gpu_status_id);
          sSQL.append("', '");
          sSQL.append(job_outcome_id);
          sSQL.append("', '");
          sSQL.append(stage);
          sSQL.append("', '");
          sSQL.append(job);
          sSQL.append("', '");
          sSQL.append(node);
          sSQL.append("', '");
          sSQL.append(gpu);
          sSQL.append("', '");
          sSQL.append(screen_output_path);
          sSQL.append("', '");
          sSQL.append(log_path);
          sSQL.append("', '");
          sSQL.append(last_round_weight_path);
          sSQL.append("', ");
          sSQL.append("CURRENT");
          sSQL.append("_TIMESTAMP");
          sSQL.append(") RETURNING slurm_job_id");

          PGresult *res = PQexec(conn, sSQL.c_str());
          if (PQresultStatus(res) != PGRES_TUPLES_OK)
          {
                 cout << "Insert slurm_job record failed\n";
                 PQclear(res);
                 closeConn(conn);
          }
          cout << "Insert slurm_job record - OK\n";
          id = atoi(PQgetvalue(res, 0, 0)) ;
          PQclear(res);
          return id;
}


int insertJobOutcomeTable(PGconn *conn, const char *slurm_job_id, const char *n_epoch, const char *num_accurate, const char *num_accurate_test,
                          const char *ss_tot, const char *ss_reg, const char *ss_res, const char *r_squared, const char *ss_tot_test, const char *ss_reg_test,
                          const char *ss_res_test, const char *r_squared_test, const char *f_ratio, const char *f_ratio_test, const char *mspe,
                          const char *sd, const char *rmse, const char *mspe_test, const char *sd_test, const char *rmse_test, const char *result_path, const char *last_round_job_id)
{
          int id;
          std::string sSQL;
          sSQL.append("INSERT INTO job_outcome (slurm_job_id, n_epoch, num_accurate, num_accurate_test, ss_tot, ss_reg, ss_res, r_squared, ss_tot_test, ss_reg_test, ss_res_test, r_squared_test, f_ratio, f_ratio_test, mspe, sd, rmse, mspe_test, sd_test, rmse_test, result_path, last_round_job_id, time_stamp) VALUES('");
          sSQL.append(slurm_job_id);
          sSQL.append("', '");
          sSQL.append(n_epoch);
          sSQL.append("', '");
          sSQL.append(num_accurate);
          sSQL.append("', '");
          sSQL.append(num_accurate_test);
          sSQL.append("', '");
          sSQL.append(ss_tot);
          sSQL.append("', '");
          sSQL.append(ss_reg);
          sSQL.append("', '");
          sSQL.append(ss_res);
          sSQL.append("', '");
          sSQL.append(r_squared);
          sSQL.append("', '");
          sSQL.append(ss_tot_test);
          sSQL.append("', '");
          sSQL.append(ss_reg_test);
          sSQL.append("', '");
          sSQL.append(ss_res_test);
          sSQL.append("', '");
          sSQL.append(r_squared_test);
          sSQL.append("', '");
          sSQL.append(f_ratio);
          sSQL.append("', '");
          sSQL.append(f_ratio_test);
          sSQL.append("', '");
          sSQL.append(mspe);
          sSQL.append("', '");
          sSQL.append(sd);
          sSQL.append("', '");
          sSQL.append(rmse);
          sSQL.append("', '");
          sSQL.append(mspe_test);
          sSQL.append("', '");
          sSQL.append(sd_test);
          sSQL.append("', '");
          sSQL.append(rmse_test);
          sSQL.append("', '");
          sSQL.append(result_path);
          sSQL.append("', '");
          sSQL.append(last_round_job_id);
          sSQL.append("', ");
          sSQL.append("CURRENT");
          sSQL.append("_TIMESTAMP");
          sSQL.append(") RETURNING job_outcome_id");

          PGresult *res = PQexec(conn, sSQL.c_str());
          if (PQresultStatus(res) != PGRES_TUPLES_OK)
          {
                 cout << "Insert job_outcome record failed\n";
                 PQclear(res);
                 closeConn(conn);
          }
          cout << "Insert job_outcome record - OK\n";
          id = atoi(PQgetvalue(res, 0, 0)) ;
          PQclear(res);
          return id;
}


void insertWeightMatrixTable(PGconn *conn, const char *job_outcome_id, const char *path, const char *name, const char *matrix)
{
          std::string sSQL;
          sSQL.append("INSERT INTO weight_matrix (job_outcome_id, path, name, matrix) VALUES('");
          sSQL.append(job_outcome_id);
          sSQL.append("', '");
          sSQL.append(path);
          sSQL.append("', '");
          sSQL.append(name);
          sSQL.append("', '");
          sSQL.append(matrix);
          sSQL.append("')");
          string weight_matrix = "weight_matrix" ;
          string insert = "Insert" ;
          executeCommand(conn, sSQL, weight_matrix, insert);
}


int InsertAdminGuys(PGconn *conn, const char * name)
{
          std::string sSQL;
          sSQL.append("INSERT INTO adminguys (name) VALUES ('");
          sSQL.append(name);
          sSQL.append("') RETURNING id");

          cout << sSQL << endl;

          PGresult *res = PQexec(conn, sSQL.c_str());

          if (PQresultStatus(res) != PGRES_TUPLES_OK)
          {
                 cout << "Insert adminguys record failed\n";
                 PQclear(res);
                 closeConn(conn);
          }
          cout << "Insert adminguys record - OK\n";

          int id  =  atoi(PQgetvalue(res, 0, 0)) ;

          PQclear(res);
          return id;
}


string fetchWeightString(PGconn *conn, const char *job_id, const char *outcome_id)
{
           //int nFields;
           string weightString ;
           std::string sSQL;
           sSQL.append("select weight_matrix.matrix from slurm_job inner join job_outcome on (slurm_job.slurm_job_id=job_outcome.slurm_job_id) inner join weight_matrix on (job_outcome.job_outcome_id=weight_matrix.job_outcome_id) where slurm_job.slurm_job_id=");
           sSQL.append(job_id);
           sSQL.append(" AND job_outcome.job_outcome_id=");
           sSQL.append(outcome_id);


           PGresult *res = PQexec(conn, sSQL.c_str());
           //nFields = PQnfields(res);
           if (PQresultStatus(res) != PGRES_TUPLES_OK)
           {
                     cout << "Fetch weight matrix string failed\n";
                     PQclear(res);
                     closeConn(conn);
           }

           //cout << PQgetvalue(res, 0, 0);
           weightString = PQgetvalue(res, 0, 0);
           PQclear(res);

           return weightString ; 
}





#endif
