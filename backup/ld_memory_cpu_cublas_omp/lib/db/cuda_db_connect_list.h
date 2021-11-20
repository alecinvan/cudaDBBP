#ifndef _CUDA_DB_CONNECT_LIST_H_
#define _CUDA_DB_CONNECT_LIST_H_
/////////////////////////////////////////////////////

void closeConn(PGconn *conn) ;
PGconn *connectDB(string connStr) ;
string readDBconfToString(string dbConf) ;
void executeCommand(PGconn *conn, string pQuerySQL, string tablename, string action);
void executeQuery(PGconn *conn, string pQuerySQL, string tablename, string action);

/////////////////////////////////////////////////////


#endif
