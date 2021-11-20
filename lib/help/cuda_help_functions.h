/************************************
 *  some general functions in linux
 *
 *  Author:  Sa Li
 *  Date:    4/4/2013
***********************************/

#ifndef  _CUDA_HELP_FUNCTIONS_H_
#define  _CUDA_HELP_FUNCTIONS_H_

//////////////////////////////////////
#include "../sdk/cuda_sdk_functions.h"
#include "./cuda_help_define.h"
#include "./cuda_help_functions_list.h"
//////////////////////////////////////

using namespace std ;


char *integerToCharArray(int num)
{
        char *arr = new char[10];
        sprintf(arr, "%d", num);
        return arr ;
}


char *floatToCharArray(float value)
{
        char *arr = new char[20];
        sprintf(arr, "%f", value);
        return arr ;
}


string floatToString(float value)
{
      std::ostringstream buff;
      buff<<value;
      return buff.str();

}

float randomRealNumber(int low, int high)
{
        return ((float) rand()/RAND_MAX)*(high-low)+low ;
}

void openInputFilesFailed(ifstream & input)
{
       if (input.fail()) {
                  cout << "Error opening the input files" << endl ;
                  exit(1) ;
       }
}

void openOutputFilesFailed(ofstream & output)
{
       if (output.fail()){
                  cout << "Error opening the output files"  << endl ;
                  exit(1) ;
       }
}

void pause_keyboard(void)
{
       int  keyboardinput = getchar() ;
       if (keyboardinput != 0 )
       cout << "Please press ENTER ..." << endl ;
}

void print_matrix(float *matrix, unsigned int size, unsigned int width)
{
       for (unsigned int i = 0; i < size ; i++)
       {
                   printf("%f ", matrix[i]);
                   if(((i+1) % width) == 0)     cout << endl ;
       }
       cout << endl ;
}

void timestamp(void)
{
       time_t ltime;
       ltime = time(NULL) ;
       printf("%s", asctime(localtime(&ltime) ) ) ;
       cout << "----" << endl << endl;
}

string result_files(unsigned int count, string prefx, string posfx)
{
       string result ;

       char numstr[21] ;
       sprintf(numstr, "%d", count);

       result = prefx + numstr + posfx ;
       return result ;
}

string working_dir(void)
{
       char *path = NULL ;
       path = getcwd(NULL, 0) ;
       return path ;
}

int parse_file_string(string fileString)
{
       istringstream ss(fileString) ;
       int lastEpochs;
       while (!ss.eof())
       {
                string x ;
                getline(ss, x, '_') ;
                const char *p ;
                p = x.c_str() ;
                if (isdigit(p[0]))
                {
                       stringstream valss(x) ;
                       valss >> lastEpochs ;
                }
       }
       return lastEpochs ;
}

bool dir_exist(const char* pzPath)
{
       if (pzPath == NULL) return false ;
       DIR *pDir ;
       bool bExists = false ;

       pDir = opendir(pzPath) ;
       if(pDir != NULL)
       {
             bExists = true ;
             (void) closedir (pDir) ;
       }
       return bExists ;
}

void create_weight_dir(string path)
{
       if (!dir_exist(path.c_str()))
              mkdir(path.c_str(), 755) ;

}





#endif

