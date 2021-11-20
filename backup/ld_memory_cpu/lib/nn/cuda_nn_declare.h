#ifndef _CUDA_NN_DECLARE_H_
#define _CUDA_NN_DECLARE_H_

               ////////////////////////////
               // Declare global variables
               ////////////////////////////

float*           W_1_h         ;
float*           W_2_h         ;
float*           W_3_h         ;
float*           X_in_h        ;
float*           Y_dout_h      ;
float*           Y_out_h       ;
float*           delta_o_h     ;
float*           delta_h_h     ;
float*           delta_i_h     ;
float*           y_1_h         ;
float*           y_2_h         ;
float*           y_3_h         ;
float*           x_1_h         ;
float*           x_2_h         ;
float*           W_1_d         ;
float*           W_2_d         ;
float*           W_3_d         ;
float*           X_in_d        ;
float*           Y_dout_d      ;
float*           Y_out_d       ;
float*           delta_o_d     ;
float*           delta_h_d     ;
float*           delta_i_d     ;
float*           y_1_d         ;
float*           y_2_d         ;
float*           y_3_d         ;
float*           x_1_d         ;
float*           x_2_d         ;
unsigned int     TN            ;
unsigned int     FP            ;
unsigned int     TP            ;
unsigned int     FN            ;
unsigned int     TN_TEST       ;
unsigned int     FP_TEST       ;
unsigned int     TP_TEST       ;
unsigned int     FN_TEST       ;
unsigned int     P_Len         ;
unsigned int     N_Len         ;
unsigned int     P_Len_test    ;
unsigned int     N_Len_test    ;
unsigned int     length        ;
int              lastjobid     ;
int              lastoutcomeid ;
map<int, string>   dataLine    ;


#endif
