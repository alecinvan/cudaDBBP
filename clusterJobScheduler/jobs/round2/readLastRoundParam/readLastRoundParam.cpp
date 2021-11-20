/*********************************************
  This is a program to get corresponding
  parameters from postgresql, nn_parameters
  TABLE, by taking the
    - slurm_job_id
    - job_outcome_id
  from commandline.

  Author: Sa Li
  Date: 4/9/2013

*********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include "/usr/include/postgresql/libpq-fe.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <sstream>
#include <string.h>
#include <map>
#include <time.h>

using namespace std ;


class postgresDBApp
{
public:
           postgresDBApp() {};
           ~postgresDBApp() {};

           void closeConn(PGconn *conn);
           PGconn *connectDB();
           void executeQuery(PGconn *conn, string pQuerySQL, string tablename, string action);
           void fetchJobParameters(PGconn *conn, const char *job_id, const char *outcome_id);
           char *integerToCharArray(int num);
           char *floatToCharArray(double value);

private:

} ;

void postgresDBApp::closeConn(PGconn *conn)
{
           PQfinish(conn) ;
           getchar();
           exit(1) ;
}


PGconn *postgresDBApp::connectDB()
{
           PGconn *conn = NULL ;
           conn = PQconnectdb("dbname=cudaDB host=10.100.78.10 user=sali password=Lixintong8$");
           conn = PQconnectdb("dbname=cudaDB host=pof-cm01.pof.local user=sali password=Lixintong8$");

           if (PQstatus(conn) != CONNECTION_OK)
           {
                  cout << "Connection to database failed" << endl ;
                  closeConn(conn) ;
           }

//           cout << "Connection to database - OK" << endl ;
           return conn ;
}



void postgresDBApp::executeQuery(PGconn *conn, string pQuerySQL, string tablename, string action)
{
           PGresult *res = PQexec(conn, pQuerySQL.c_str());


           if (PQresultStatus(res) != PGRES_COMMAND_OK)
           {
                    cout << action << " " << tablename << " table failed" << endl;
                    PQclear(res);
                    closeConn(conn);
           }

           cout << action << " " << tablename << " table - OK" << endl;
           PQclear(res);

}

void postgresDBApp::fetchJobParameters(PGconn *conn, const char *job_id, const char *outcome_id)
{
           int nFields;

           std::string sSQL;
           sSQL.append("select nn_parameters.n_input, nn_parameters.n_1hidden, nn_parameters.n_2hidden, nn_parameters.u, nn_parameters.alfa, job_outcome.result_path, job_outcome.n_epoch from slurm_job inner join job_outcome on (slurm_job.slurm_job_id=job_outcome.slurm_job_id) inner join nn_parameters on (slurm_job.nn_parameter_id=nn_parameters.nn_parameter_id) where slurm_job.slurm_job_id="); 
           sSQL.append(job_id);
           sSQL.append(" AND job_outcome.job_outcome_id=");
           sSQL.append(outcome_id);

                      // Fetch rows from nn_parameter table by running join query

           PGresult *res = PQexec(conn, sSQL.c_str());
           nFields = PQnfields(res);
           if (PQresultStatus(res) != PGRES_TUPLES_OK)
           {
                     cout << "Fetch job parameters failed\n";
                     PQclear(res);
                     closeConn(conn);
           }

           for (int i = 0; i < PQntuples(res); i++)
           {
                    for (int j = 0; j < nFields; j++)
                                 cout <<  PQgetvalue(res, i, j)  << " ";
           }
           PQclear(res);


}



char* postgresDBApp::integerToCharArray(int num)
{
        char *arr = new char[20];
        sprintf(arr, "%d", num);
        return arr ;
}

char* postgresDBApp::floatToCharArray(double value)
{
        char *arr = new char[50];
        sprintf(arr, "%f", value);
        return arr ;
}



int main(int argc, char** argv)
{

          int job_id = atoi(argv[1]);
          int outcome_id = atoi(argv[2]);

          postgresDBApp         cudaDB ;

          PGconn      *conn   =   NULL ;
          conn                =   cudaDB.connectDB();
          if (conn != NULL)
          {
                    cudaDB.fetchJobParameters(conn, cudaDB.integerToCharArray(job_id), cudaDB.integerToCharArray(outcome_id)) ;
          }


//          cout << "Disonnect to database - OK" << endl ;
          cudaDB.closeConn(conn) ;

          return 1;

}
