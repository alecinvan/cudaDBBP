/*********************************************
  This is a program to get weight from cudaDB

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

#define   DB_CONF                "/cluster/clusterPipeline/MonteCarlo/CUDADBBP/lib/db/cuda_db_connect.conf"
#define   WEIGHTFILE             "./weightMatrix.dat"
#define   N_input                68
#define   N_1hidden              154
#define   N_2hidden              101
#define   N_output               1



float*           W_1_h        ;
float*           W_2_h        ;
float*           W_3_h        ;



using namespace std ;



class postgresDBApp
{

public:
           postgresDBApp() {};
           ~postgresDBApp() {};

           void closeConn(PGconn *conn);
           PGconn *connectDB(string connStr);
           void openInputFilesFailed(ifstream & input);
           void openOutputFilesFailed(ofstream & output);
           void readWeightsFromFile(string weightFileString, int Ninput, int N1hidden, int N2hidden, int Noutput);
           void print_matrix(float *matrix, unsigned int size, unsigned int width);
           void executeQuery(PGconn *conn, string pQuerySQL, string tablename, string action);
           string fetchWeightString(PGconn *conn, string weightFileString, const char *job_id, const char *outcome_id);
           char *integerToCharArray(int num);
           char *floatToCharArray(double value);
           string readDBconf(string db_conf);

private:

} ;

void postgresDBApp::openInputFilesFailed(ifstream & input)
{
                      if (input.fail()) {  cout << "Error opening the input files" << endl;     exit(1); }
}



void postgresDBApp::openOutputFilesFailed(ofstream & output)
{
                      if (output.fail())        {    cout << "Error opening the output files" << endl;  exit(1); }
}


void postgresDBApp::print_matrix(float *matrix, unsigned int size, unsigned int width)
{
       for (unsigned int i = 0; i < size ; i++)
       {
                   printf("%f ", matrix[i]);
                   if(((i+1) % width) == 0)     cout << endl ;
       }
       cout << endl ;
}

void postgresDBApp::readWeightsFromFile(string weightFileString, int Ninput, int N1hidden, int N2hidden, int Noutput)
{
                ifstream      weightFileInput ;
                weightFileInput.open(weightFileString.c_str());
                openInputFilesFailed(weightFileInput);
                for(int i=0; i < Ninput*N1hidden; i++)                    weightFileInput >> W_1_h[i] ;
                for(int i=0; i < N1hidden*N2hidden; i++)                  weightFileInput >> W_2_h[i] ;
                for(int i=0; i <= N2hidden*Noutput; i++)                  weightFileInput >> W_3_h[i] ;
                weightFileInput.close();
}

void postgresDBApp::closeConn(PGconn *conn)
{
           PQfinish(conn) ;
           getchar();
           exit(1) ;
}

PGconn *postgresDBApp::connectDB(string connStr)
{
           PGconn *conn = NULL ;
           conn = PQconnectdb(connStr.c_str());

           if (PQstatus(conn) != CONNECTION_OK)
           {
                  cout << "Connection to database failed" << endl ;
                  closeConn(conn) ;
           }
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

string postgresDBApp::fetchWeightString(PGconn *conn, string weightFileString, const char *job_id, const char *outcome_id)
{
           int nFields;

           std::string sSQL;
           sSQL.append("select weight_matrix.matrix from slurm_job inner join job_outcome on (slurm_job.slurm_job_id=job_outcome.slurm_job_id) inner join weight_matrix on (job_outcome.job_outcome_id=weight_matrix.job_outcome_id) where slurm_job.slurm_job_id="); 
           sSQL.append(job_id);
           sSQL.append(" AND job_outcome.job_outcome_id=");
           sSQL.append(outcome_id);


           PGresult *res = PQexec(conn, sSQL.c_str());
           nFields = PQnfields(res);
           if (PQresultStatus(res) != PGRES_TUPLES_OK)
           {
                     cout << "Fetch weight matrix string failed\n";
                     PQclear(res);
                     closeConn(conn);
           }
       //    cout << PQgetvalue(res, 0, 0);
           string N; 
           N = PQgetvalue(res, 0, 0);
        //   cout << N ;  
/*
           ofstream    weightOutput;
           weightOutput.open(weightFileString.c_str());
           openOutputFilesFailed(weightOutput);
           weightOutput << PQgetvalue(res, 0, 0);
           weightOutput.close() ;
*/  
          PQclear(res);
          return N;

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


string postgresDBApp::readDBconf(string db_conf)
{
          ifstream   cuda_db_conf;
          cuda_db_conf.open(db_conf.c_str());
          openInputFilesFailed(cuda_db_conf);
          string s, db_conn_str;
          while(!cuda_db_conf.eof()) {
                           getline(cuda_db_conf,s) ;
                           db_conn_str = db_conn_str + s + " " ;
          }

          cuda_db_conf.close() ;

          return db_conn_str ;
}


int main(int argc, char** argv)
{

          int job_id = atoi(argv[1]);
          int outcome_id = atoi(argv[2]);

          W_1_h       =    new float [(N_input+1)*(N_1hidden+1)] ;
          W_2_h       =    new float [(N_1hidden+1)*(N_2hidden+1)] ;
          W_3_h       =    new float [(N_2hidden+1)*(N_output+1)] ;

          postgresDBApp         cudaDB ;


          PGconn      *conn   =   NULL ;
          conn                =   cudaDB.connectDB(cudaDB.readDBconf(DB_CONF));

          if (conn != NULL)
          {
                   cout <<   cudaDB.fetchWeightString(conn, WEIGHTFILE, cudaDB.integerToCharArray(job_id), cudaDB.integerToCharArray(outcome_id)) ;

          }
          cudaDB.closeConn(conn) ;



          return 0;

}
