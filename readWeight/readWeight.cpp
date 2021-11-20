/*********************************************
  This is a program to read Weight string 
  from postgresql, by taking the
     - job_outcome_id
  from commandline.

  Author: Sa Li
  Date: 4/20/2013

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

float*           W_1_h         ;
float*           W_2_h         ;
float*           W_3_h         ;


class postgresDBApp
{
public:
           postgresDBApp() {};
           ~postgresDBApp() {};

           void closeConn(PGconn *conn);
           PGconn *connectDB();
           void executeQuery(PGconn *conn, string pQuerySQL, string tablename, string action);
           string fetchWeight(PGconn *conn, const char *outcome_id);
           void convertStringToArray(string weightString, int inputSize, int firstHidLayerSize, int secHidLayerSize, int outputSize);
           void print_matrix(float *matrix, unsigned int size, unsigned int width);
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
           cout << "Connection to database - OK" << endl ;
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

string postgresDBApp::fetchWeight(PGconn *conn, const char *outcome_id)
{
           int nFields;

           std::string sSQL;
           sSQL.append("select matrix from weight_matrix where job_outcome_id="); 
           sSQL.append(outcome_id);

                      // Fetch rows from nn_parameter table by running join query

           PGresult *res = PQexec(conn, sSQL.c_str());
           //nFields = PQnfields(res);
           if (PQresultStatus(res) != PGRES_TUPLES_OK)
           {
                     cout << "Fetch job parameters failed\n";
                     PQclear(res);
                     closeConn(conn);
           }

/*
           for (int i = 0; i < PQntuples(res); i++)
           {
                    for (int j = 0; j < nFields; j++)
                                 cout <<  PQgetvalue(res, i, j)  << " ";
           }
*/
           string weightString = PQgetvalue(res, 0, 0);
           PQclear(res);

           return weightString ;

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


void postgresDBApp::print_matrix(float *matrix, unsigned int size, unsigned int width)
{
       for (unsigned int i = 0; i < size ; i++)
       {
                   printf("%f ", matrix[i]);
                   if(((i+1) % width) == 0)     cout << endl ;
       }
       cout << endl ;
}

void postgresDBApp::convertStringToArray(string weightString, int inputSize, int firstHidLayerSize, int secHidLayerSize, int outputSize)
{
                istringstream iss(weightString, istringstream::in) ;
                int i=0 ;
                while (iss) {
                                         string val ;
                                         if (!getline(iss, val, ' '))           break;
                                         stringstream valss(val) ;
                                         char values[20] ;
                                         valss >> values ;
                                         if ( i < inputSize*firstHidLayerSize )
                                         {
                                                  W_1_h[i] = atof(values) ;
                                         }
                                         else if ( i < (inputSize*firstHidLayerSize + firstHidLayerSize*secHidLayerSize))
                                         {
                                                  W_2_h[i-inputSize*firstHidLayerSize] = atof(values) ;
                                         }
                                         else
                                         {
                                                  W_3_h[i-inputSize*firstHidLayerSize-firstHidLayerSize*secHidLayerSize] = atof(values) ;
                                         }
                                         i++ ;

                }
}



int main(int argc, char** argv)
{

          int outcome_id = atoi(argv[1]);
          int N_input    = atoi(argv[2]);
          int N_1hidden  = atoi(argv[3]);
          int N_2hidden  = atoi(argv[4]);
          int N_output   = 1 ;

          W_1_h       =    new float [(N_input+1)*(N_1hidden+1)] ;
          W_2_h       =    new float [(N_1hidden+1)*(N_2hidden+1)] ;
          W_3_h       =    new float [(N_2hidden+1)*(N_output+1)] ;

          string weightMatrixString ;

          postgresDBApp         cudaDB ;

          PGconn      *conn   =   NULL ;
          conn                =   cudaDB.connectDB();

          if (conn != NULL)
          {
                   weightMatrixString = cudaDB.fetchWeight(conn, cudaDB.integerToCharArray(outcome_id)) ;
//                   cout << "here.." << endl;
          }

          cudaDB.closeConn(conn) ;
          cout << weightMatrixString << endl ;

//          cudaDB.convertStringToArray(weightMatrixString, N_input, N_1hidden, N_2hidden, N_output) ;
//          cudaDB.print_matrix(W_1_h, (N_input+1)*(N_1hidden+1), N_1hidden+1) ;
//          cudaDB.print_matrix(W_2_h, (N_1hidden+1)*(N_2hidden+1), N_2hidden+1) ;
//          cudaDB.print_matrix(W_3_h, (N_2hidden+1)*(N_output+1), N_output+1) ;

          return 0;

}
