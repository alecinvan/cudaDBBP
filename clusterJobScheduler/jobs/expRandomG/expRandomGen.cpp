
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>

using namespace std ;

float randomRealNumber(float low, float high)
{
        return ((float) rand()/RAND_MAX)*(high-low)+low ;
}


float u(float a, float b)
{
         return a * pow(10, b) ;
}

int main(int argc, char *argv[])
{

            srand((unsigned)time(NULL));

     //       cout <<  randomRealNumber(1, 10)* pow(10,randomRealNumber(-3,-8)) <<  " " << randomRealNumber(1.5,4.5) << endl ;
           cout <<  randomRealNumber(1, 10)* pow(10,randomRealNumber(-1,-9)) <<  " " << randomRealNumber(0.0,0.2) << endl ;  // beta is between 0 ~ 1
}


