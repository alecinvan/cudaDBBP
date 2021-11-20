#ifndef  _CUDA_HELP_FUNCTIONS_LIST_H_
#define  _CUDA_HELP_FUNCTIONS_LIST_H_

/////////////////////////////////////////

char    *integerToCharArray(int num);
char    *floatToCharArray(float value);
string  floatToString(float value);
float   randomRealNumber(int low, int high) ;
void    openInputFilesFailed(ifstream & input) ;
void    openOutputFilesFailed(ofstream & output) ;
void    pause_keyboard(void) ;
void    print_matrix(float *matrix, unsigned int size, unsigned int width) ;
void    timestamp(void) ;
string  result_files(unsigned int count, string prefx, string posfx) ;
string  working_dir(void) ;
int     parse_file_string(string fileString) ;
bool    dir_exist(const char* pzPath) ;
void    create_weight_dir(string path) ;

/////////////////////////////////////////

#endif
