/*
 * BROCCOLI: An open source multi-platform software for parallel analysis of fMRI data on many core CPUs and GPUS
 * Copyright (C) <2013>  Anders Eklund, andek034@gmail.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "mex.h"
#include "help_functions.cpp"
#include "broccoli_lib.h"

void cleanUp()
{
    //cudaDeviceReset();
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //-----------------------
    // Input
    
    double		    *h_fMRI_Volumes_double, *h_T1_Volume_double, *h_MNI_Volume_double, *h_MNI_Brain_Mask_double;
    float           *h_fMRI_Volumes, *h_T1_Volume, *h_MNI_Volume, *h_MNI_Brain_Mask; 

    double          *h_Quadrature_Filter_1_Real_double, *h_Quadrature_Filter_2_Real_double, *h_Quadrature_Filter_3_Real_double, *h_Quadrature_Filter_1_Imag_double, *h_Quadrature_Filter_2_Imag_double, *h_Quadrature_Filter_3_Imag_double;
    float           *h_Quadrature_Filter_1_Real, *h_Quadrature_Filter_2_Real, *h_Quadrature_Filter_3_Real, *h_Quadrature_Filter_1_Imag, *h_Quadrature_Filter_2_Imag, *h_Quadrature_Filter_3_Imag;
    
    int             IMAGE_REGISTRATION_FILTER_SIZE, NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION, COARSEST_SCALE_T1_MNI, COARSEST_SCALE_EPI_T1, MM_T1_Z_CUT, MM_EPI_Z_CUT, NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION;
    int             NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_RIGID = 6;
    int             NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_AFFINE = 12;
    
    double          *h_T1_MNI_Registration_Parameters_double, *h_EPI_T1_Registration_Parameters_double, *h_Motion_Parameters_double, *h_EPI_MNI_Registration_Parameters_double;
    float           *h_T1_MNI_Registration_Parameters, *h_EPI_T1_Registration_Parameters, *h_Motion_Parameters, *h_EPI_MNI_Registration_Parameters;
    
    double          *h_Motion_Corrected_fMRI_Volumes_double;
    float           *h_Motion_Corrected_fMRI_Volumes;
    
    double          *h_Smoothing_Filter_X_double, *h_Smoothing_Filter_Y_double, *h_Smoothing_Filter_Z_double;
    float           *h_Smoothing_Filter_X, *h_Smoothing_Filter_Y, *h_Smoothing_Filter_Z;
    int             SMOOTHING_FILTER_LENGTH;
    float           EPI_SMOOTHING_AMOUNT, AR_SMOOTHING_AMOUNT;
    
    double          *h_Smoothed_fMRI_Volumes_double;
    float           *h_Smoothed_fMRI_Volumes;    
    
    double		    *h_X_GLM_double, *h_xtxxt_GLM_double, *h_Contrasts_double, *h_ctxtxc_GLM_double;
    float           *h_Mask, *h_X_GLM, *h_xtxxt_GLM, *h_Contrasts, *h_ctxtxc_GLM;  
    
    double          *h_AR1_Estimates_double, *h_AR2_Estimates_double, *h_AR3_Estimates_double, *h_AR4_Estimates_double;
    float           *h_AR1_Estimates, *h_AR2_Estimates, *h_AR3_Estimates, *h_AR4_Estimates;
        
    int             EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, T1_DATA_H, T1_DATA_W, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D; 
                
    float           EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z;                
    
    int             NUMBER_OF_REGRESSORS, NUMBER_OF_CONTRASTS, BETA_SPACE;
    
    int             OPENCL_PLATFORM, OPENCL_DEVICE;
    
    int             NUMBER_OF_DIMENSIONS;
    
    //-----------------------
    // Output
    
    double     		*h_Beta_Volumes_double, *h_Residuals_double, *h_Residual_Variances_double, *h_Statistical_Maps_double;
    float           *h_Beta_Volumes, *h_Residuals, *h_Residual_Variances, *h_Statistical_Maps;
    
    //---------------------
    
    /* Check the number of input and output arguments. */
    if(nrhs<31)
    {
        mexErrMsgTxt("Too few input arguments.");
    }
    if(nrhs>31)
    {
        mexErrMsgTxt("Too many input arguments.");
    }
    if(nlhs<14)
    {
        mexErrMsgTxt("Too few output arguments.");
    }
    if(nlhs>14)
    {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Input arguments */
    
    // The data
    h_fMRI_Volumes_double =  (double*)mxGetData(prhs[0]);
    h_T1_Volume_double =  (double*)mxGetData(prhs[1]);
    h_MNI_Volume_double =  (double*)mxGetData(prhs[2]);
    h_MNI_Brain_Mask_double = (double*)mxGetData(prhs[3]);
    EPI_VOXEL_SIZE_X = (float)mxGetScalar(prhs[4]);
    EPI_VOXEL_SIZE_Y = (float)mxGetScalar(prhs[5]);
    EPI_VOXEL_SIZE_Z = (float)mxGetScalar(prhs[6]);    
    T1_VOXEL_SIZE_X = (float)mxGetScalar(prhs[7]);
    T1_VOXEL_SIZE_Y = (float)mxGetScalar(prhs[8]);
    T1_VOXEL_SIZE_Z = (float)mxGetScalar(prhs[9]);
    MNI_VOXEL_SIZE_X = (float)mxGetScalar(prhs[10]);
    MNI_VOXEL_SIZE_Y = (float)mxGetScalar(prhs[11]);
    MNI_VOXEL_SIZE_Z = (float)mxGetScalar(prhs[12]);    
    
    h_Quadrature_Filter_1_Real_double =  (double*)mxGetPr(prhs[13]);
    h_Quadrature_Filter_1_Imag_double =  (double*)mxGetPi(prhs[13]);
    h_Quadrature_Filter_2_Real_double =  (double*)mxGetPr(prhs[14]);
    h_Quadrature_Filter_2_Imag_double =  (double*)mxGetPi(prhs[14]);
    h_Quadrature_Filter_3_Real_double =  (double*)mxGetPr(prhs[15]);
    h_Quadrature_Filter_3_Imag_double =  (double*)mxGetPi(prhs[15]);
    NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION  = (int)mxGetScalar(prhs[16]);    
    COARSEST_SCALE_T1_MNI  = (int)mxGetScalar(prhs[17]);
    COARSEST_SCALE_EPI_T1  = (int)mxGetScalar(prhs[18]);
    MM_T1_Z_CUT  = (int)mxGetScalar(prhs[19]);
    MM_EPI_Z_CUT  = (int)mxGetScalar(prhs[20]);
    
    NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION  = (int)mxGetScalar(prhs[21]);
    
    //h_Smoothing_Filter_X_double = (double*)mxGetData(prhs[22]);
    //h_Smoothing_Filter_Y_double = (double*)mxGetData(prhs[23]);
    //h_Smoothing_Filter_Z_double = (double*)mxGetData(prhs[24]);
    EPI_SMOOTHING_AMOUNT = (float)mxGetScalar(prhs[22]);
    AR_SMOOTHING_AMOUNT = (float)mxGetScalar(prhs[23]);
    
    h_X_GLM_double =  (double*)mxGetData(prhs[24]);
    h_xtxxt_GLM_double =  (double*)mxGetData(prhs[25]);
    h_Contrasts_double = (double*)mxGetData(prhs[26]);
    h_ctxtxc_GLM_double = (double*)mxGetData(prhs[27]);
    BETA_SPACE = (int)mxGetScalar(prhs[28]);
    
    OPENCL_PLATFORM  = (int)mxGetScalar(prhs[29]);
    OPENCL_DEVICE = (int)mxGetScalar(prhs[30]);
    
    const int *ARRAY_DIMENSIONS_EPI = mxGetDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_T1 = mxGetDimensions(prhs[1]);
    const int *ARRAY_DIMENSIONS_MNI = mxGetDimensions(prhs[2]);
    const int *ARRAY_DIMENSIONS_QUADRATURE_FILTER = mxGetDimensions(prhs[13]);        
    //const int *ARRAY_DIMENSIONS_SMOOTHING_FILTER = mxGetDimensions(prhs[22]);        
    const int *ARRAY_DIMENSIONS_GLM = mxGetDimensions(prhs[25]);
    const int *ARRAY_DIMENSIONS_CONTRAST = mxGetDimensions(prhs[28]);        
    
    EPI_DATA_H = ARRAY_DIMENSIONS_EPI[0];
    EPI_DATA_W = ARRAY_DIMENSIONS_EPI[1];
    EPI_DATA_D = ARRAY_DIMENSIONS_EPI[2];
    EPI_DATA_T = ARRAY_DIMENSIONS_EPI[3];
    
    T1_DATA_H = ARRAY_DIMENSIONS_T1[0];
    T1_DATA_W = ARRAY_DIMENSIONS_T1[1];
    T1_DATA_D = ARRAY_DIMENSIONS_T1[2];
    
    MNI_DATA_H = ARRAY_DIMENSIONS_MNI[0];
    MNI_DATA_W = ARRAY_DIMENSIONS_MNI[1];
    MNI_DATA_D = ARRAY_DIMENSIONS_MNI[2];
        
    IMAGE_REGISTRATION_FILTER_SIZE = ARRAY_DIMENSIONS_QUADRATURE_FILTER[0];                   
    SMOOTHING_FILTER_LENGTH = 9;
    
    NUMBER_OF_REGRESSORS = ARRAY_DIMENSIONS_GLM[1];
    NUMBER_OF_CONTRASTS = ARRAY_DIMENSIONS_CONTRAST[1];
    
    int EPI_DATA_SIZE = EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);
    int T1_DATA_SIZE = T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float);
    int MNI_DATA_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float);
    
    int EPI_VOLUME_SIZE = EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);
    
    int QUADRATURE_FILTER_SIZE = IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float);
    
    int T1_MNI_PARAMETERS_SIZE = NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_AFFINE * sizeof(float);
    int EPI_T1_PARAMETERS_SIZE = NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_RIGID * sizeof(float);
    int EPI_MNI_PARAMETERS_SIZE = NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_AFFINE * sizeof(float);
    int MOTION_PARAMETERS_SIZE = NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_RIGID * EPI_DATA_T * sizeof(float);
        
    int SMOOTHING_FILTER_SIZE = SMOOTHING_FILTER_LENGTH * sizeof(float);    
    
    int GLM_SIZE = EPI_DATA_T * NUMBER_OF_REGRESSORS * sizeof(float);
    int CONTRAST_SIZE = NUMBER_OF_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float);
    int CONTRAST_SCALAR_SIZE = NUMBER_OF_CONTRASTS * sizeof(float);
    
    int BETA_DATA_SIZE, STATISTICAL_MAPS_DATA_SIZE, RESIDUAL_VARIANCES_DATA_SIZE;
    
    if (BETA_SPACE == MNI)
    {
        BETA_DATA_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_REGRESSORS * sizeof(float);        
        STATISTICAL_MAPS_DATA_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float);    
        RESIDUAL_VARIANCES_DATA_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float);    
    }
    else if (BETA_SPACE == EPI)
    {
        BETA_DATA_SIZE = EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_REGRESSORS * sizeof(float);
        STATISTICAL_MAPS_DATA_SIZE = EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float);           
        RESIDUAL_VARIANCES_DATA_SIZE = EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);    
    }        
    
    int RESIDUAL_DATA_SIZE = EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);
    
    mexPrintf("fMRI data size : %i x %i x %i x %i \n", EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
    mexPrintf("T1 data size : %i x %i x %i \n", T1_DATA_W, T1_DATA_H, T1_DATA_D);
    mexPrintf("MNI data size : %i x %i x %i \n", MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    mexPrintf("Number of regressors : %i \n",  NUMBER_OF_REGRESSORS);
    mexPrintf("Number of contrasts : %i \n",  NUMBER_OF_CONTRASTS);
    
    //-------------------------------------------------
    // Output to Matlab
    
    // Create pointer for volumes to Matlab        
    NUMBER_OF_DIMENSIONS = 4;
    int ARRAY_DIMENSIONS_OUT_BETA[4];
    if (BETA_SPACE == MNI)
    {
        ARRAY_DIMENSIONS_OUT_BETA[0] = MNI_DATA_H;
        ARRAY_DIMENSIONS_OUT_BETA[1] = MNI_DATA_W;
        ARRAY_DIMENSIONS_OUT_BETA[2] = MNI_DATA_D;
    }
    else if (BETA_SPACE == EPI)
    {
        ARRAY_DIMENSIONS_OUT_BETA[0] = EPI_DATA_H;
        ARRAY_DIMENSIONS_OUT_BETA[1] = EPI_DATA_W;
        ARRAY_DIMENSIONS_OUT_BETA[2] = EPI_DATA_D;    
    }
    
    ARRAY_DIMENSIONS_OUT_BETA[3] = NUMBER_OF_REGRESSORS;
    
    plhs[0] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_BETA,mxDOUBLE_CLASS, mxREAL);
    h_Beta_Volumes_double = mxGetPr(plhs[0]);          
    
    NUMBER_OF_DIMENSIONS = 4;
    int ARRAY_DIMENSIONS_OUT_RESIDUALS[4];
    ARRAY_DIMENSIONS_OUT_RESIDUALS[0] = EPI_DATA_H;
    ARRAY_DIMENSIONS_OUT_RESIDUALS[1] = EPI_DATA_W;
    ARRAY_DIMENSIONS_OUT_RESIDUALS[2] = EPI_DATA_D;
    ARRAY_DIMENSIONS_OUT_RESIDUALS[3] = EPI_DATA_T;
    
    plhs[1] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_RESIDUALS,mxDOUBLE_CLASS, mxREAL);
    h_Residuals_double = mxGetPr(plhs[1]);          
        
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES[3];
    if (BETA_SPACE == MNI)
    {        
        ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES[0] = MNI_DATA_H;
        ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES[1] = MNI_DATA_W;
        ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES[2] = MNI_DATA_D;
    }
    else if (BETA_SPACE == EPI)
    {
        ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES[0] = EPI_DATA_H;
        ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES[1] = EPI_DATA_W;
        ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES[2] = EPI_DATA_D;
    }
    
    plhs[2] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES,mxDOUBLE_CLASS, mxREAL);
    h_Residual_Variances_double = mxGetPr(plhs[2]);          
    
    NUMBER_OF_DIMENSIONS = 4;
    int ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[4];
    if (BETA_SPACE == MNI)
    {
        ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[0] = MNI_DATA_H;
        ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[1] = MNI_DATA_W;
        ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[2] = MNI_DATA_D;        
    }
    else if (BETA_SPACE == EPI)
    {
        ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[0] = EPI_DATA_H;
        ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[1] = EPI_DATA_W;
        ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[2] = EPI_DATA_D;
    }
    
    ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[3] = NUMBER_OF_CONTRASTS;
    
    plhs[3] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS,mxDOUBLE_CLASS, mxREAL);
    h_Statistical_Maps_double = mxGetPr(plhs[3]);          
    
    NUMBER_OF_DIMENSIONS = 2;
    int ARRAY_DIMENSIONS_OUT_T1_MNI_PARAMETERS[2];
    ARRAY_DIMENSIONS_OUT_T1_MNI_PARAMETERS[0] = 1;
    ARRAY_DIMENSIONS_OUT_T1_MNI_PARAMETERS[1] = NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_AFFINE;
    
    plhs[4] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_T1_MNI_PARAMETERS,mxDOUBLE_CLASS, mxREAL);
    h_T1_MNI_Registration_Parameters_double = mxGetPr(plhs[4]);  
    
    NUMBER_OF_DIMENSIONS = 2;
    int ARRAY_DIMENSIONS_OUT_EPI_T1_PARAMETERS[2];
    ARRAY_DIMENSIONS_OUT_EPI_T1_PARAMETERS[0] = 1;
    ARRAY_DIMENSIONS_OUT_EPI_T1_PARAMETERS[1] = NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_RIGID;
    
    plhs[5] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_EPI_T1_PARAMETERS,mxDOUBLE_CLASS, mxREAL);
    h_EPI_T1_Registration_Parameters_double = mxGetPr(plhs[5]);  
    
    NUMBER_OF_DIMENSIONS = 2;
    int ARRAY_DIMENSIONS_OUT_EPI_MNI_PARAMETERS[2];
    ARRAY_DIMENSIONS_OUT_EPI_MNI_PARAMETERS[0] = 1;
    ARRAY_DIMENSIONS_OUT_EPI_MNI_PARAMETERS[1] = NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_AFFINE;
    
    plhs[6] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_EPI_MNI_PARAMETERS,mxDOUBLE_CLASS, mxREAL);
    h_EPI_MNI_Registration_Parameters_double = mxGetPr(plhs[6]);  
    
    
    NUMBER_OF_DIMENSIONS = 2;
    int ARRAY_DIMENSIONS_OUT_MOTION_PARAMETERS[2];
    ARRAY_DIMENSIONS_OUT_MOTION_PARAMETERS[0] = EPI_DATA_T;
    ARRAY_DIMENSIONS_OUT_MOTION_PARAMETERS[1] = NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_RIGID;    
    
    plhs[7] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_MOTION_PARAMETERS,mxDOUBLE_CLASS, mxREAL);
    h_Motion_Parameters_double = mxGetPr(plhs[7]);  
    
    NUMBER_OF_DIMENSIONS = 4;
    int ARRAY_DIMENSIONS_OUT_MOTION_CORRECTED_FMRI_VOLUMES[4];
    ARRAY_DIMENSIONS_OUT_MOTION_CORRECTED_FMRI_VOLUMES[0] = EPI_DATA_H;
    ARRAY_DIMENSIONS_OUT_MOTION_CORRECTED_FMRI_VOLUMES[1] = EPI_DATA_W;
    ARRAY_DIMENSIONS_OUT_MOTION_CORRECTED_FMRI_VOLUMES[2] = EPI_DATA_D;
    ARRAY_DIMENSIONS_OUT_MOTION_CORRECTED_FMRI_VOLUMES[3] = EPI_DATA_T;
    
    plhs[8] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_MOTION_CORRECTED_FMRI_VOLUMES,mxDOUBLE_CLASS, mxREAL);
    h_Motion_Corrected_fMRI_Volumes_double = mxGetPr(plhs[8]);          
    
    NUMBER_OF_DIMENSIONS = 4;
    int ARRAY_DIMENSIONS_OUT_SMOOTHED_FMRI_VOLUMES[4];
    ARRAY_DIMENSIONS_OUT_SMOOTHED_FMRI_VOLUMES[0] = EPI_DATA_H;
    ARRAY_DIMENSIONS_OUT_SMOOTHED_FMRI_VOLUMES[1] = EPI_DATA_W;
    ARRAY_DIMENSIONS_OUT_SMOOTHED_FMRI_VOLUMES[2] = EPI_DATA_D;
    ARRAY_DIMENSIONS_OUT_SMOOTHED_FMRI_VOLUMES[3] = EPI_DATA_T;
    
    plhs[9] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_SMOOTHED_FMRI_VOLUMES,mxDOUBLE_CLASS, mxREAL);
    h_Smoothed_fMRI_Volumes_double = mxGetPr(plhs[9]);          
    
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_AR_ESTIMATES[3];
    ARRAY_DIMENSIONS_OUT_AR_ESTIMATES[0] = EPI_DATA_H;
    ARRAY_DIMENSIONS_OUT_AR_ESTIMATES[1] = EPI_DATA_W;
    ARRAY_DIMENSIONS_OUT_AR_ESTIMATES[2] = EPI_DATA_D;
    
    plhs[10] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_AR_ESTIMATES,mxDOUBLE_CLASS, mxREAL);
    h_AR1_Estimates_double = mxGetPr(plhs[10]);          
    
    plhs[11] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_AR_ESTIMATES,mxDOUBLE_CLASS, mxREAL);
    h_AR2_Estimates_double = mxGetPr(plhs[11]);          
    
    plhs[12] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_AR_ESTIMATES,mxDOUBLE_CLASS, mxREAL);
    h_AR3_Estimates_double = mxGetPr(plhs[12]);          
    
    plhs[13] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_AR_ESTIMATES,mxDOUBLE_CLASS, mxREAL);
    h_AR4_Estimates_double = mxGetPr(plhs[13]);  
    
    // ------------------------------------------------
    
    // Allocate memory on the host
    h_fMRI_Volumes                      = (float *)mxMalloc(EPI_DATA_SIZE);
    h_T1_Volume                         = (float *)mxMalloc(T1_DATA_SIZE);
    h_MNI_Volume                        = (float *)mxMalloc(MNI_DATA_SIZE);
    h_MNI_Brain_Mask                    = (float *)mxMalloc(MNI_DATA_SIZE);
    
    h_Quadrature_Filter_1_Real          = (float *)mxMalloc(QUADRATURE_FILTER_SIZE);
    h_Quadrature_Filter_1_Imag          = (float *)mxMalloc(QUADRATURE_FILTER_SIZE);
    h_Quadrature_Filter_2_Real          = (float *)mxMalloc(QUADRATURE_FILTER_SIZE);
    h_Quadrature_Filter_2_Imag          = (float *)mxMalloc(QUADRATURE_FILTER_SIZE);    
    h_Quadrature_Filter_3_Real          = (float *)mxMalloc(QUADRATURE_FILTER_SIZE);
    h_Quadrature_Filter_3_Imag          = (float *)mxMalloc(QUADRATURE_FILTER_SIZE);    
    
    h_Motion_Corrected_fMRI_Volumes     = (float *)mxMalloc(EPI_DATA_SIZE);
    
    h_T1_MNI_Registration_Parameters    = (float *)mxMalloc(T1_MNI_PARAMETERS_SIZE);
    h_EPI_T1_Registration_Parameters    = (float *)mxMalloc(EPI_T1_PARAMETERS_SIZE);
    h_EPI_MNI_Registration_Parameters    = (float *)mxMalloc(EPI_MNI_PARAMETERS_SIZE);
    h_Motion_Parameters                 = (float *)mxMalloc(MOTION_PARAMETERS_SIZE);
    
    //h_Smoothing_Filter_X                = (float *)mxMalloc(SMOOTHING_FILTER_SIZE);
    //h_Smoothing_Filter_Y                = (float *)mxMalloc(SMOOTHING_FILTER_SIZE);
    //h_Smoothing_Filter_Z                = (float *)mxMalloc(SMOOTHING_FILTER_SIZE);
    
    h_Smoothed_fMRI_Volumes             = (float *)mxMalloc(EPI_DATA_SIZE);
    
    h_X_GLM                             = (float *)mxMalloc(GLM_SIZE);
    h_xtxxt_GLM                         = (float *)mxMalloc(GLM_SIZE);
    h_Contrasts                         = (float *)mxMalloc(CONTRAST_SIZE);
    h_ctxtxc_GLM                        = (float *)mxMalloc(CONTRAST_SCALAR_SIZE);
    
    h_Beta_Volumes                      = (float *)mxMalloc(BETA_DATA_SIZE);
    h_Residuals                         = (float *)mxMalloc(RESIDUAL_DATA_SIZE);
    h_Residual_Variances                = (float *)mxMalloc(RESIDUAL_VARIANCES_DATA_SIZE);
    h_Statistical_Maps                  = (float *)mxMalloc(STATISTICAL_MAPS_DATA_SIZE);
        
    h_AR1_Estimates                     = (float *)mxMalloc(EPI_VOLUME_SIZE);
    h_AR2_Estimates                     = (float *)mxMalloc(EPI_VOLUME_SIZE);
    h_AR3_Estimates                     = (float *)mxMalloc(EPI_VOLUME_SIZE);
    h_AR4_Estimates                     = (float *)mxMalloc(EPI_VOLUME_SIZE);
    
    // Reorder and cast data
    pack_double2float_volumes(h_fMRI_Volumes, h_fMRI_Volumes_double, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
    pack_double2float_volume(h_T1_Volume, h_T1_Volume_double, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    pack_double2float_volume(h_MNI_Volume, h_MNI_Volume_double, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    pack_double2float_volume(h_MNI_Brain_Mask, h_MNI_Brain_Mask_double, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    
    pack_double2float_volume(h_Quadrature_Filter_1_Real, h_Quadrature_Filter_1_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_1_Imag, h_Quadrature_Filter_1_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_2_Real, h_Quadrature_Filter_2_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_2_Imag, h_Quadrature_Filter_2_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_3_Real, h_Quadrature_Filter_3_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_3_Imag, h_Quadrature_Filter_3_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);

    //pack_double2float(h_Smoothing_Filter_X, h_Smoothing_Filter_X_double, SMOOTHING_FILTER_LENGTH);
    //pack_double2float(h_Smoothing_Filter_Y, h_Smoothing_Filter_Y_double, SMOOTHING_FILTER_LENGTH);
    //pack_double2float(h_Smoothing_Filter_Z, h_Smoothing_Filter_Z_double, SMOOTHING_FILTER_LENGTH);
    
    pack_double2float(h_X_GLM, h_X_GLM_double, NUMBER_OF_REGRESSORS * EPI_DATA_T);
    pack_double2float(h_xtxxt_GLM, h_xtxxt_GLM_double, NUMBER_OF_REGRESSORS * EPI_DATA_T);    
    pack_double2float(h_Contrasts, h_Contrasts_double, NUMBER_OF_REGRESSORS * NUMBER_OF_CONTRASTS);
    pack_double2float(h_ctxtxc_GLM, h_ctxtxc_GLM_double, NUMBER_OF_CONTRASTS);
       
    //------------------------
    
    BROCCOLI_LIB BROCCOLI(OPENCL_PLATFORM, OPENCL_DEVICE);
    
    BROCCOLI.SetInputfMRIVolumes(h_fMRI_Volumes);
    BROCCOLI.SetInputT1Volume(h_T1_Volume);
    BROCCOLI.SetInputMNIVolume(h_MNI_Volume);
    BROCCOLI.SetInputMNIBrainMask(h_MNI_Brain_Mask);
    BROCCOLI.SetEPIWidth(EPI_DATA_W);
    BROCCOLI.SetEPIHeight(EPI_DATA_H);
    BROCCOLI.SetEPIDepth(EPI_DATA_D);
    BROCCOLI.SetEPITimepoints(EPI_DATA_T);     
    BROCCOLI.SetT1Width(T1_DATA_W);
    BROCCOLI.SetT1Height(T1_DATA_H);
    BROCCOLI.SetT1Depth(T1_DATA_D);
    BROCCOLI.SetMNIWidth(MNI_DATA_W);
    BROCCOLI.SetMNIHeight(MNI_DATA_H);
    BROCCOLI.SetMNIDepth(MNI_DATA_D);
    BROCCOLI.SetEPIVoxelSizeX(EPI_VOXEL_SIZE_X);
    BROCCOLI.SetEPIVoxelSizeY(EPI_VOXEL_SIZE_Y);
    BROCCOLI.SetEPIVoxelSizeZ(EPI_VOXEL_SIZE_Z);       
    BROCCOLI.SetT1VoxelSizeX(T1_VOXEL_SIZE_X);
    BROCCOLI.SetT1VoxelSizeY(T1_VOXEL_SIZE_Y);
    BROCCOLI.SetT1VoxelSizeZ(T1_VOXEL_SIZE_Z);   
    BROCCOLI.SetMNIVoxelSizeX(MNI_VOXEL_SIZE_X);
    BROCCOLI.SetMNIVoxelSizeY(MNI_VOXEL_SIZE_Y);
    BROCCOLI.SetMNIVoxelSizeZ(MNI_VOXEL_SIZE_Z); 
    BROCCOLI.SetInterpolationMode(LINEAR);
    BROCCOLI.SetImageRegistrationFilterSize(IMAGE_REGISTRATION_FILTER_SIZE);
    BROCCOLI.SetParametricImageRegistrationFilters(h_Quadrature_Filter_1_Real, h_Quadrature_Filter_1_Imag, h_Quadrature_Filter_2_Real, h_Quadrature_Filter_2_Imag, h_Quadrature_Filter_3_Real, h_Quadrature_Filter_3_Imag);
    BROCCOLI.SetNumberOfIterationsForParametricImageRegistration(NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION);    
    BROCCOLI.SetNumberOfIterationsForMotionCorrection(NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION);    
    BROCCOLI.SetCoarsestScaleT1MNI(COARSEST_SCALE_T1_MNI);
    BROCCOLI.SetCoarsestScaleEPIT1(COARSEST_SCALE_EPI_T1);
    BROCCOLI.SetMMT1ZCUT(MM_T1_Z_CUT);   
    BROCCOLI.SetMMEPIZCUT(MM_EPI_Z_CUT);   
    BROCCOLI.SetOutputT1MNIRegistrationParameters(h_T1_MNI_Registration_Parameters);
    BROCCOLI.SetOutputEPIT1RegistrationParameters(h_EPI_T1_Registration_Parameters);
    BROCCOLI.SetOutputEPIMNIRegistrationParameters(h_EPI_MNI_Registration_Parameters);
    BROCCOLI.SetOutputMotionCorrectedfMRIVolumes(h_Motion_Corrected_fMRI_Volumes);
    BROCCOLI.SetOutputMotionParameters(h_Motion_Parameters);
    //BROCCOLI.SetSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z);
    BROCCOLI.SetEPISmoothingAmount(EPI_SMOOTHING_AMOUNT);
    BROCCOLI.SetARSmoothingAmount(AR_SMOOTHING_AMOUNT);
    BROCCOLI.SetOutputSmoothedfMRIVolumes(h_Smoothed_fMRI_Volumes);
    BROCCOLI.SetNumberOfGLMRegressors(NUMBER_OF_REGRESSORS);
    BROCCOLI.SetNumberOfContrasts(NUMBER_OF_CONTRASTS);    
    BROCCOLI.SetDesignMatrix(h_X_GLM, h_xtxxt_GLM);
    BROCCOLI.SetContrasts(h_Contrasts);
    BROCCOLI.SetGLMScalars(h_ctxtxc_GLM);
    BROCCOLI.SetOutputBetaVolumes(h_Beta_Volumes);
    BROCCOLI.SetOutputResiduals(h_Residuals);
    BROCCOLI.SetOutputResidualVariances(h_Residual_Variances);
    BROCCOLI.SetOutputStatisticalMaps(h_Statistical_Maps);
    BROCCOLI.SetOutputAREstimates(h_AR1_Estimates, h_AR2_Estimates, h_AR3_Estimates, h_AR4_Estimates);
    BROCCOLI.SetBetaSpace(BETA_SPACE);
    
    
     /*
     * Error checking     
     */

    int getPlatformIDsError = BROCCOLI.GetOpenCLPlatformIDsError();
	int getDeviceIDsError = BROCCOLI.GetOpenCLDeviceIDsError();		
	int createContextError = BROCCOLI.GetOpenCLCreateContextError();
	int getContextInfoError = BROCCOLI.GetOpenCLContextInfoError();
	int createCommandQueueError = BROCCOLI.GetOpenCLCreateCommandQueueError();
	int createProgramError = BROCCOLI.GetOpenCLCreateProgramError();
	int buildProgramError = BROCCOLI.GetOpenCLBuildProgramError();
	int getProgramBuildInfoError = BROCCOLI.GetOpenCLProgramBuildInfoError();
          
    mexPrintf("Get platform IDs error is %d \n",getPlatformIDsError);
    mexPrintf("Get device IDs error is %d \n",getDeviceIDsError);
    mexPrintf("Create context error is %d \n",createContextError);
    mexPrintf("Get create context info error is %d \n",getContextInfoError);
    mexPrintf("Create command queue error is %d \n",createCommandQueueError);
    mexPrintf("Create program error is %d \n",createProgramError);
    mexPrintf("Build program error is %d \n",buildProgramError);
    mexPrintf("Get program build info error is %d \n",getProgramBuildInfoError);
    
    int* createKernelErrors = BROCCOLI.GetOpenCLCreateKernelErrors();
    for (int i = 0; i < 34; i++)
    {
        if (createKernelErrors[i] != 0)
        {
            mexPrintf("Create kernel error %i is %d \n",i,createKernelErrors[i]);
        }
    }
    
    int* createBufferErrors = BROCCOLI.GetOpenCLCreateBufferErrors();
    for (int i = 0; i < 30; i++)
    {
        if (createBufferErrors[i] != 0)
        {
            mexPrintf("Create buffer error %i is %d \n",i,createBufferErrors[i]);
        }
    }
        
    int* runKernelErrors = BROCCOLI.GetOpenCLRunKernelErrors();
    for (int i = 0; i < 20; i++)
    {
        if (runKernelErrors[i] != 0)
        {
            mexPrintf("Run kernel error %i is %d \n",i,runKernelErrors[i]);
        }
    } 
    
    mexPrintf("Build info \n \n %s \n", BROCCOLI.GetOpenCLBuildInfoChar());     
    
    if ( (getPlatformIDsError + getDeviceIDsError + createContextError + getContextInfoError + createCommandQueueError + createProgramError + buildProgramError + getProgramBuildInfoError) == 0)
    { 
        BROCCOLI.PerformFirstLevelAnalysisWrapper();
    }
    else
    {
        mexPrintf("OPENCL error detected, aborting \n");
    }    
    
    unpack_float2double(h_T1_MNI_Registration_Parameters_double, h_T1_MNI_Registration_Parameters, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_AFFINE);
    unpack_float2double(h_EPI_T1_Registration_Parameters_double, h_EPI_T1_Registration_Parameters, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_RIGID);
    unpack_float2double(h_EPI_MNI_Registration_Parameters_double, h_EPI_MNI_Registration_Parameters, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_AFFINE);
    unpack_float2double(h_Motion_Parameters_double, h_Motion_Parameters, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_RIGID * EPI_DATA_T);
    unpack_float2double_volumes(h_Motion_Corrected_fMRI_Volumes_double, h_Motion_Corrected_fMRI_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
    unpack_float2double_volumes(h_Smoothed_fMRI_Volumes_double, h_Smoothed_fMRI_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
    
    if (BETA_SPACE == MNI)
    {
        unpack_float2double_volumes(h_Beta_Volumes_double, h_Beta_Volumes, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_REGRESSORS);  
        unpack_float2double_volumes(h_Statistical_Maps_double, h_Statistical_Maps, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS);                
        unpack_float2double_volume(h_Residual_Variances_double, h_Residual_Variances, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    }
    else if (BETA_SPACE == EPI)
    {
        unpack_float2double_volumes(h_Beta_Volumes_double, h_Beta_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_REGRESSORS);
        unpack_float2double_volumes(h_Statistical_Maps_double, h_Statistical_Maps, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS);        
        unpack_float2double_volume(h_Residual_Variances_double, h_Residual_Variances, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
    }        
    
    unpack_float2double_volumes(h_Residuals_double, h_Residuals, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);        
    
    unpack_float2double_volume(h_AR1_Estimates_double, h_AR1_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
    unpack_float2double_volume(h_AR2_Estimates_double, h_AR2_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
    unpack_float2double_volume(h_AR3_Estimates_double, h_AR3_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
    unpack_float2double_volume(h_AR4_Estimates_double, h_AR4_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
    
    // Free all the allocated memory on the host
    
    mxFree(h_fMRI_Volumes);
    mxFree(h_T1_Volume);
    mxFree(h_MNI_Volume);
    mxFree(h_MNI_Brain_Mask);
    
    mxFree(h_Quadrature_Filter_1_Real);
    mxFree(h_Quadrature_Filter_1_Imag);
    mxFree(h_Quadrature_Filter_2_Real);
    mxFree(h_Quadrature_Filter_2_Imag);    
    mxFree(h_Quadrature_Filter_3_Real);
    mxFree(h_Quadrature_Filter_3_Imag);    
    
    mxFree(h_Motion_Corrected_fMRI_Volumes);
    
    mxFree(h_T1_MNI_Registration_Parameters);
    mxFree(h_EPI_T1_Registration_Parameters);
    mxFree(h_EPI_MNI_Registration_Parameters);
    mxFree(h_Motion_Parameters);
    
    //mxFree(h_Smoothing_Filter_X);
    //mxFree(h_Smoothing_Filter_Y);
    //mxFree(h_Smoothing_Filter_Z);
    
    mxFree(h_Smoothed_fMRI_Volumes);
    
    mxFree(h_X_GLM);
    mxFree(h_xtxxt_GLM);
    mxFree(h_Contrasts);
    mxFree(h_ctxtxc_GLM);
    
    mxFree(h_Beta_Volumes);
    mxFree(h_Residuals);
    mxFree(h_Residual_Variances);
    mxFree(h_Statistical_Maps);
   
    mxFree(h_AR1_Estimates);
    mxFree(h_AR2_Estimates);
    mxFree(h_AR3_Estimates);
    mxFree(h_AR4_Estimates);
    
    //mexAtExit(cleanUp);
    
    return;
}



