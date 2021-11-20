/*#############################################
#  functions related to postgresql to populate
#  cluster information into cudaDB
#
#  Author: Sa Li
##############################################*/

#ifndef _CUDA_DB_CONNECT_H_
#define _CUDA_DB_CONNECT_H_

/////////////////////////////////////
#include "../nn/cuda_nn_functions.h"
#include "./cuda_db_connect_list.h"
#include "./cuda_db_connect_define.h"
/////////////////////////////////////



void closeConn(PGconn *conn)
{
           PQfinish(conn) ;
//           getchar();
           exit(1) ;
}

string readDBconfToString(string dbConf)
{
          ifstream   cuda_db_conf;
          cuda_db_conf.open(dbConf.c_str());
          openInputFilesFailed(cuda_db_conf);
          string s, db_conn_str;
          while(!cuda_db_conf.eof()) {
                           getline(cuda_db_conf,s) ;
                           db_conn_str = db_conn_str + s + " " ;
          }
          cuda_db_conf.close() ;

          return db_conn_str ;
}


PGconn *connectDB(string connStr)
{
           PGconn *conn = NULL ;
       //    conn = PQconnectdb("dbname=cudaDB host=pof-cm01.pof.local user=sali password=Lixintong8$");
           conn = PQconnectdb(connStr.c_str());
           if (PQstatus(conn) != CONNECTION_OK)
           {
                  cout << "Connection to database failed" << endl ;
                  closeConn(conn) ;
           }

           cout << "Connection to database - OK" << endl ;
           return conn ;
}

void executeCommand(PGconn *conn, string pQuerySQL, string tablename, string action)
{
           PGresult *res = PQexec(conn, pQuerySQL.c_str());

           if (PQresultStatus(res) != PGRES_COMMAND_OK)
           {
                    cout << action << " " << tablename << " table failed" << endl;
                    PQclear(res);
                    closeConn(conn);
           }

           PQclear(res);

}

void executeQuery(PGconn *conn, string pQuerySQL, string tablename, string action)
{
           PGresult *res = PQexec(conn, pQuerySQL.c_str());

           if (PQresultStatus(res) != PGRES_TUPLES_OK)
           {
                    cout << action << " " << tablename << " table failed" << endl;
                    PQclear(res);
                    closeConn(conn);
           }

           PQclear(res);

}




#endif
