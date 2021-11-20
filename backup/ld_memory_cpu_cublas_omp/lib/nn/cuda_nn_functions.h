/**************************************
 * neural net related functions
 * Author: Sa Li
**************************************/


#ifndef _CUDA_NN_FUNCTIONS_H_
#define _CUDA_NN_FUCNTIONS_H_

/////////////////////////////////////////
#include "../test/cuda_test_functions.h"
#include "./cuda_nn_define.h"
#include "./cuda_nn_functions_list.h"
#include "./cuda_nn_declare.h"
/////////////////////////////////////////


using namespace std;


/*********************************************************************/
void global_variable_define(unsigned int N_input, unsigned int N_1hidden,
                            unsigned int N_2hidden, unsigned int N_output)
{

                W_1_h       =    new float [(N_input+1)*(N_1hidden+1)] ;
                W_2_h       =    new float [(N_1hidden+1)*(N_2hidden+1)] ;
                W_3_h       =    new float [(N_2hidden+1)*(N_output+1)] ;
                X_in_h      =    new float [N_input+1]  ;
                Y_dout_h    =    new float [N_output+1] ;
                Y_out_h     =    new float [N_output+1] ;
                delta_o_h   =    new float [N_output+1] ;
                delta_h_h   =    new float [N_2hidden+1] ;
                delta_i_h   =    new float [N_1hidden+1] ;
                y_1_h       =    new float [N_1hidden+1] ;
                y_2_h       =    new float [N_2hidden+1] ;
                y_3_h       =    new float [N_output+1] ;
                x_1_h       =    new float [N_1hidden+1] ;
                x_2_h       =    new float [N_2hidden+1] ;

}

void connectionWeightInit(unsigned int inputSize, unsigned int firstLayerSize,
                          unsigned int secondLayerSize, unsigned int outputSize)
{
                srand((unsigned)(time(NULL)));
                for (unsigned int i=0; i< inputSize*firstLayerSize; i++)                  W_1_h[i] = randomRealNumber(-1, 1)  ;
		for (unsigned int i=0; i< inputSize; i++)                                 W_1_h[i*firstLayerSize] =  0 ;
	        for (unsigned int i=0; i< firstLayerSize*secondLayerSize; i++)            W_2_h[i] =  randomRealNumber(-1, 1) ;
                for (unsigned int i=0; i< firstLayerSize; i++)                            W_2_h[i*secondLayerSize] =  0 ;
		for (unsigned int i=0; i< secondLayerSize*outputSize; i++)                W_3_h[i] =  randomRealNumber(-1, 1) ;
                for (unsigned int i=0; i< secondLayerSize; i++)                           W_3_h[i*outputSize] =  0 ;

}


void arrayAssign(unsigned int inputSize, unsigned int outputSize, string s)
{


                X_in_h[0] = BIAS, x_1_h[0] = BIAS, x_2_h[0] = BIAS, Y_dout_h[0] = 0 ;
                float value ;
                unsigned int idx = 1 ;
                istringstream iss(s, istringstream::in) ;
                while(iss) {
                                   string val ;
                                   if (!getline(iss, val, ' ')) break ;
                                   stringstream valss(val) ;
                                   valss >> value ;
                                   X_in_h[idx] = value ;
                                   idx++ ;
                }
                Y_dout_h[outputSize] =  X_in_h[inputSize+1] ;

}


void arrayClear(float* inputArray, unsigned int inputsize)
{
	        for (unsigned int i=1; i<= inputsize; i++)                       inputArray[i] = 0;

}



void writeTwoLayerWeightsInFile(ofstream & output, unsigned int inputSize,
                                unsigned int firstHidLayerSize, unsigned int secHidLayerSize,
                                unsigned int outputSize)
{
	        for(unsigned int i=0; i < inputSize*firstHidLayerSize; i++)
	        {
	                    output << W_1_h[i] << " " ;
                            if(((i + 1) % firstHidLayerSize ) == 0)              output << endl ;
                }
		output << endl ;
		for(unsigned int i=0; i < firstHidLayerSize*secHidLayerSize; i++)
		{
		            output << W_2_h[i] << " " ;
                            if(((i + 1) % secHidLayerSize) == 0)                 output << endl ;
                }
		output << endl ;
                for(unsigned int i=0; i < secHidLayerSize*outputSize; i++)
	        {
	                    output << W_3_h[i] << " " ;
                            if(((i + 1) % outputSize) == 0)                      output << endl ;
                }
	        output << endl ;
}

void writeParameter(ofstream & output, float u, float alfa,
                    unsigned int inputSize, unsigned int firstHidLayerSize,
                    unsigned int secHidLayerSize, unsigned int outputSize,
                    unsigned epochs)
{
                output << "u = "                   <<  u                   << endl ;
                output << "alfa = "                <<  alfa                << endl ;
                output << "N_input = "             <<  inputSize           << endl ;
	        output << "N_1hidden = "           <<  firstHidLayerSize   << endl ;
	        output << "N_2hidden = "           <<  secHidLayerSize     << endl ;
                output << "N_output = "            <<  outputSize          << endl ;
		output << "Epochs = "              <<  epochs              << endl ;
		output << "BIAS = "                <<  BIAS                << endl ;
		output << "cuda BLOCK_SIZE = "     <<  BLOCK_SIZE          << endl ;
		output << "write_file_epochs = "   <<  WRITE_FILE_EPOCHS   << endl ;
}

void loadTwoLayerWeightsInArray(ifstream&input, unsigned int inputSize,
                                unsigned int firstHidLayerSize, unsigned int secHidLayerSize,
                                unsigned int outputSize)
{
                for(unsigned int i=0; i < inputSize*firstHidLayerSize; i++)                    input >> W_1_h[i] ;
		for(unsigned int i=0; i < firstHidLayerSize*secHidLayerSize; i++)              input >> W_2_h[i] ;
		for(unsigned int i=0; i <= secHidLayerSize*outputSize; i++)                    input >> W_3_h[i] ;
}

void writeParameterIntoFile(string path, string paraFileString, float u, float alfa,
                            unsigned int N_input, unsigned int N_1hidden, unsigned int N_2hidden,
                            unsigned int N_output, unsigned int epochs)
{
	        ofstream      parameterOutput;

          	string writeParaFileString = path + paraFileString ;
	        parameterOutput.open(writeParaFileString.c_str());
                openOutputFilesFailed(parameterOutput);
		writeParameter(parameterOutput, u, alfa, N_input, N_1hidden, N_2hidden, N_output, epochs)  ;
                parameterOutput.close();
}

void writeWeightsIntoFile(string path, string weightFileString, unsigned int N_input,
                          unsigned int N_1hidden, unsigned int N_2hidden, unsigned int N_output)
{
	        ofstream      weightOutput;

		string writeWeightFileString = path + weightFileString ;
                cudaMemcpy( W_1_h, W_1_d, (N_input+1)*(N_1hidden+1)*sizeof(float), cudaMemcpyDeviceToHost);
	        cudaMemcpy( W_2_h, W_2_d, (N_1hidden+1)*(N_2hidden+1)*sizeof(float), cudaMemcpyDeviceToHost);
	        cudaMemcpy( W_3_h, W_3_d, (N_2hidden+1)*(N_output+1)*sizeof(float), cudaMemcpyDeviceToHost);

	        weightOutput.open(writeWeightFileString.c_str());
                openOutputFilesFailed(weightOutput);
		writeTwoLayerWeightsInFile(weightOutput, N_input+1, N_1hidden+1, N_2hidden+1, N_output+1) ;
                weightOutput.close();
}

void readWeightsFromFile(string path, string weightFileString, unsigned int N_input,
                         unsigned int N_1hidden, unsigned int N_2hidden, unsigned int N_output)
{
	        ifstream      weightFileInput ;
                string readWeightFileString = path + weightFileString ;
	        weightFileInput.open(readWeightFileString.c_str());
                openInputFilesFailed(weightFileInput);
	 	loadTwoLayerWeightsInArray(weightFileInput, N_input+1, N_1hidden+1, N_2hidden+1, N_output+1) ;
                weightFileInput.close();
}

string readWeightsIntoString(string path, string weightFileString)
{
                ifstream     weightFileInput ;
                string readWeightFileString = path + weightFileString ;
                weightFileInput.open(readWeightFileString.c_str());
                openInputFilesFailed(weightFileInput);
                string str, s;
                while (!weightFileInput.eof()) {
                               getline(weightFileInput, s) ;
                               str = str + s + "\n" ;
                }
                weightFileInput.close() ;
                return str ;
}

string writeWeightMatrixIntoString(int inputSize, int firstHidLayerSize, int secHidLayerSize, int outputSize)
{

                string w_matrix ;
                for(int i=0; i < inputSize*firstHidLayerSize; i++)
                {
                            w_matrix = w_matrix + floatToString(W_1_h[i]) + " " ;
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


void convertStringToArray(string weightString, int inputSize, int firstHidLayerSize, int secHidLayerSize, int outputSize)
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



void fMeasureCompute(float desOutValue, float actualOutValue)
{
            	switch ( int(desOutValue) ) {
	        case 0:
	                if ( abs(actualOutValue-desOutValue) > 0.6 )
	                	                                          FP++ ;
			else if ( abs(actualOutValue-desOutValue) < 0.4 )
				                                          TN++ ;
			break;
	        case 1:
			if ( abs(actualOutValue-desOutValue) > 0.6  )
			                                                  FN++ ;
			else if ( abs(actualOutValue-desOutValue) < 0.4 )
			                                                  TP++ ;
			break;
		default:
		        break;
		}
}

void   fMeasureTestCompute(float desOutValue, float actualOutValue)
{
                switch ( int(desOutValue) ) {
                case 0:
                        if ( abs(actualOutValue-desOutValue) > 0.6 )
                                                                          FP_TEST++ ;
                        else if ( abs(actualOutValue-desOutValue) < 0.4 )
                                                                          TN_TEST++ ;
                        break;
                case 1:
                        if ( abs(actualOutValue-desOutValue) > 0.6  )
                                                                          FN_TEST++ ;
                        else if ( abs(actualOutValue-desOutValue) < 0.4 )
                                                                          TP_TEST++ ;
                        break;
                default:
                        break;
                }
}


float accuracyCompute(unsigned int tn, unsigned int fp, unsigned int tp, unsigned int fn)
{
		return  (float)  (( tp + tn ) / (tp + tn + fp + fn )) ;
}


void printMeasureMatrix(string path, string resultFileString,
                        unsigned int tp, unsigned int tn, unsigned int fp,
                        unsigned int fn, unsigned int len)
{

                ofstream      resultOutput ;
                string   writeResultFileString  =  path  +  resultFileString ;
                resultOutput.open(writeResultFileString.c_str()) ;
                openOutputFilesFailed(resultOutput) ;

                cout << "*************************************************************" <<  endl ;
                cout << "@ number of predicting couple as couple:          " << tp <<  ",   percentage  = " << (float) tp/P_Len * 100 << "%" <<  endl ;
		cout << "@ number of predicting non-couple as non-couple:  " << tn <<  ",   percentage = " << (float) tn/N_Len  * 100 << "%"  <<  endl ;
		cout << "@ number of predicting non-couple as couple:      " << fp <<  ",   percentage = " << (float) fp/N_Len  * 100 << "%" << endl ;
		cout << "@ number of predicting couple as non-couple:      " << fn <<  ",   percentage = " << (float) fn/P_Len  * 100 << "%" << endl ;
                cout << "@ number of predicting ambiguous:                 " << len - tp - tn - fp - fn - 1 << ",   percentage = " << ((float) len -tp - tn -fp -fn -1 ) / len  * 100 << "%" << endl << endl ;

                resultOutput << "- Number of predicting couple as couple:          " << tp <<  ",   percentage  = " << (float) tp/P_Len * 100  <<   "%" <<  endl ;
                resultOutput << "- Number of predicting non-couple as non-couple:  " << tn <<  ",   percentage = " << (float) tn/N_Len  * 100 <<   "%" << endl ;  ;
                resultOutput << "- Number of predicting non-couple as couple:      " << fp <<  ",   percentage = " << (float) fp/N_Len  * 100 <<   "%" << endl ;
                resultOutput << "- Number of predicting couple as non-couple:      " << fn <<  ",   percentage = " << (float) fn/P_Len  * 100 <<   "%" << endl ;
                resultOutput << "- Number of predicting ambiguous:                 " << len  - tp - tn - fp - fn - 1 << ",   percentage = " << ((float) len -tp - tn -fp -fn -1 ) /len  * 100 <<  "%" << endl << endl ;

                resultOutput.close() ;

}

void printPredictMatrix(unsigned int tp, unsigned int tn, unsigned int fp,
                        unsigned int fn, unsigned int len)
{

                cout << "*************************************************************" <<  endl ;
                cout << "*            Accuracy of prediction on test set             *" << endl ; 
                cout << "*************************************************************" << endl; 
                cout << "number of predicting couple as couple:          " << tp <<  ",   percentage  = " << (float) tp/P_Len_test * 100 << "%" <<  endl ;
                cout << "number of predicting non-couple as non-couple:  " << tn <<  ",   percentage = " << (float) tn/N_Len_test  * 100 << "%"  <<  endl ;
                cout << "number of predicting non-couple as couple:      " << fp <<  ",   percentage = " << (float) fp/N_Len_test  * 100 << "%" << endl ;
                cout << "number of predicting couple as non-couple:      " << fn <<  ",   percentage = " << (float) fn/P_Len_test  * 100 << "%" << endl ;
                cout << "number of predicting ambiguous:                 " << len - tp - tn - fp - fn - 1 << ",   percentage = " << ((float) len -tp - tn -fp -fn -1 ) / len  * 100 << "%" << endl << endl ;

}

void printTimeStamp(float diff_time, unsigned int k, unsigned int Epochs)
{
	        cout << "Each epoch takes about:        " << diff_time /300  << " minutes"  << endl;
		cout << "Approximate time left:         " << (diff_time /300) * (Epochs - k ) /60 << " hours"  << endl << endl << "~~~~~~~~~~~~~~~" << endl << endl; 
}




#endif

