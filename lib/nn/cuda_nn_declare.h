#ifndef _CUDA_NN_DECLARE_H_
#define _CUDA_NN_DECLARE_H_

               ////////////////////////////
               // Declare global variables
               ////////////////////////////

float*           W_1_h           ;
float*           W_2_h           ;
float*           W_3_h           ;
float*           X_in_h          ;
float*           Y_dout_h        ;
float*           Y_out_h         ;
float*           delta_o_h       ;
float*           delta_h_h       ;
float*           delta_i_h       ;
float*           y_1_h           ;
float*           y_2_h           ;
float*           y_3_h           ;
float*           x_1_h           ;
float*           x_2_h           ;
float*           W_1_d           ;
float*           W_2_d           ;
float*           W_3_d           ;
float*           delta_W_1_d     ;
float*           delta_W_2_d     ;
float*           delta_W_3_d     ;
float*           Y_dout_d        ;
float*           Y_out_d         ;
float*           delta_o_d       ;
float*           delta_h_d       ;
float*           delta_i_d       ;
float*           delta_o_d_past  ;
float*           delta_h_d_past  ;
float*           delta_i_d_past  ;
float*           y_1_d           ;
float*           y_2_d           ;
float*           y_3_d           ;
float*           X_in_d          ;
float*           x_1_d           ;
float*           x_2_d           ;
float*           X_in_d_past     ;
float*           x_1_d_past      ;
float*           x_2_d_past      ;
unsigned int     TN              ;
unsigned int     FP              ;
unsigned int     TP              ;
unsigned int     FN              ;
unsigned int     TN_TEST         ;
unsigned int     FP_TEST         ;
unsigned int     TP_TEST         ;
unsigned int     FN_TEST         ;
unsigned int     N_ACCURATE      ;
unsigned int     N_ACCURATE_TEST ;
unsigned int     P_Len           ;
unsigned int     N_Len           ;
unsigned int     P_Len_test      ;
unsigned int     N_Len_test      ;
unsigned int     length          ;
int              lastjobid       ;
int              lastoutcomeid   ;

float            SS_TOT          ;    //  total sum of squares
float            SS_REG          ;    //  explained sum of squares
float            SS_RES          ;    //  residual sum of squares
float            R_SQUARED       ;    //  measure how well data points fit a statistical model
float            SS_TOT_TEST     ;
float            SS_REG_TEST     ;
float            SS_RES_TEST     ;
float            R_SQUARED_TEST  ;

float            F_RATIO         ;
float            P_VALUE         ;
float            F_RATIO_TEST    ;
float            P_VALUE_TEST    ;

int              DF_TOT          ;   // degree of freedom for total, n-1
int              DF_REG          ;   // degree of freedom for model, p-1, p=N_input
int              DF_RES          ;   // degree of error term (residual), n-p
int              DF_TOT_TEST     ;
int              DF_REG_TEST     ;
int              DF_RES_TEST     ;

float            MS_TOT          ;   // mean square error for each term, MeanSq = SumSq / DF
float            MS_REG          ;
float            MS_RES          ;
float            MS_TOT_TEST     ;
float            MS_REG_TEST     ;
float            MS_RES_TEST     ;

float            AE              ;     // absolute error
float            MAE             ;     // mean absolute error, u
float            MAPE            ;     // mean absolute percentage error
float            MSPE            ;     // mean squared percentage error
float            SD              ;     // standard deviation
float            RMSE            ;     // rms error

float            AE_TEST         ;     // absolute error
float            MAE_TEST        ;     // mean absolute error, u
float            MAPE_TEST       ;     // mean absolute percentage error
float            MSPE_TEST       ;     // mean squared percentage error
float            SD_TEST         ;     // standard deviation
float            RMSE_TEST       ;     // rms error


map<int, string>   dataLine      ;


#endif
