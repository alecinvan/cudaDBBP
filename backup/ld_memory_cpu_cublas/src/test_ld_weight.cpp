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

#define   WEIGHTFILE             "./weightMatrix.dat"
#define   N_input                68
#define   N_1hidden              154
#define   N_2hidden              101
#define   N_output               1



float*           W_1_h        ;
float*           W_2_h        ;
float*           W_3_h        ;



using namespace std ;



void openInputFilesFailed(ifstream & input)
{
               if (input.fail()) {  cout << "Error opening the input files" << endl;     exit(1); }
}



void openOutputFilesFailed(ofstream & output)
{
               if (output.fail())        {    cout << "Error opening the output files" << endl;  exit(1); }
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


void global_variable_define(unsigned int n_input, unsigned int n_1hidden, unsigned int n_2hidden, unsigned int n_output)
{

                W_1_h       =    new float [(n_input+1)*(n_1hidden+1)] ;
                W_2_h       =    new float [(n_1hidden+1)*(n_2hidden+1)] ;
                W_3_h       =    new float [(n_2hidden+1)*(n_output+1)] ;

}


void loadTwoLayerWeightsInArray(ifstream&input, int inputSize, int firstHidLayerSize, int secHidLayerSize, int outputSize)
{
                for(unsigned int i=0; i < inputSize*firstHidLayerSize; i++)                    input >> W_1_h[i] ;
                for(unsigned int i=0; i < firstHidLayerSize*secHidLayerSize; i++)              input >> W_2_h[i] ;
                for(unsigned int i=0; i <= secHidLayerSize*outputSize; i++)                    input >> W_3_h[i] ;
}


void readWeightsFromFile(string weightFileString, int n_input, int n_1hidden, int n_2hidden, int n_output)
{
                ifstream      weightFileInput ;
                weightFileInput.open(weightFileString.c_str());
                openInputFilesFailed(weightFileInput);
                loadTwoLayerWeightsInArray(weightFileInput, n_input+1, n_1hidden+1, n_2hidden+1, n_output+1) ;
                weightFileInput.close();
}


char* integerToCharArray(int num)
{
        char *arr = new char[20];
        sprintf(arr, "%d", num);
        return arr ;
}


char* floatToCharArray(float value)
{
        char *arr = new char[50];
        sprintf(arr, "%f", value);
        return arr ;
}

string floatToString(float value)
{
      std::ostringstream buff;
      buff<<value;
      return buff.str();

}



string writeWeightsIntoString(int inputSize, int firstHidLayerSize, int secHidLayerSize, int outputSize)
{

                string w_matrix ;
                for(int i=0; i < inputSize*firstHidLayerSize; i++)
                {
                            w_matrix = w_matrix + floatToString(W_2_h[i]) + " " ;
                }
 //               w_matrix = w_matrix + "\n" ;
                for(int i=0; i < firstHidLayerSize*secHidLayerSize; i++)
                {
                            w_matrix = w_matrix + floatToString(W_2_h[i]) + " " ;
                }
 //                w_matrix = w_matrix + "\n" ;
                for(int i=0; i < secHidLayerSize*outputSize; i++)
                {
                            w_matrix = w_matrix + floatToString(W_3_h[i]) + " " ;
                }
                return w_matrix ;
}



void stringToArray(string weightString, int inputSize, int firstHidLayerSize, int secHidLayerSize, int outputSize)
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

          global_variable_define(N_input,  N_1hidden,  N_2hidden,  N_output) ;
          readWeightsFromFile(WEIGHTFILE, N_input,  N_1hidden,  N_2hidden,  N_output) ;
          string weightString  =  writeWeightsIntoString(N_input, N_1hidden, N_2hidden, N_output) ;
          stringToArray(weightString, N_input,  N_1hidden,  N_2hidden,  N_output);
          print_matrix(W_1_h, (N_input+1)*(N_1hidden+1), N_1hidden+1) ;
          print_matrix(W_2_h, (N_1hidden+1)*(N_2hidden+1), N_2hidden+1) ;
          print_matrix(W_3_h, (N_2hidden+1)*(N_output+1), N_output+1) ;


          return 0;

}
