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
    // Input pointers
    
    double		    *h_fMRI_Volumes_double, *h_Mask_double, *h_X_GLM_double, *h_xtxxt_GLM_double, *h_Contrasts_double, *h_ctxtxc_GLM_double;
    float           *h_fMRI_Volumes, *h_Mask, *h_X_GLM, *h_xtxxt_GLM, *h_Contrasts, *h_ctxtxc_GLM;        
    
    //-----------------------
    // Output pointers        
    
    double     		*h_Beta_Volumes_double, *h_Residuals_double, *h_Residual_Variances_double, *h_Statistical_Maps_double;
    float           *h_Beta_Volumes, *h_Residuals, *h_Residual_Variances, *h_Statistical_Maps;
    
    //---------------------
    
    /* Check the number of input and output arguments. */
    if(nrhs<6)
    {
        mexErrMsgTxt("Too few input arguments.");
    }
    if(nrhs>6)
    {
        mexErrMsgTxt("Too many input arguments.");
    }
    if(nlhs<4)
    {
        mexErrMsgTxt("Too few output arguments.");
    }
    if(nlhs>4)
    {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Input arguments */
    
    // The data
    h_fMRI_Volumes_double =  (double*)mxGetData(prhs[0]);
    h_Mask_double =  (double*)mxGetData(prhs[1]);
    h_X_GLM_double =  (double*)mxGetData(prhs[2]);
    h_xtxxt_GLM_double =  (double*)mxGetData(prhs[3]);
    h_Contrasts_double = (double*)mxGetData(prhs[4]);
    h_ctxtxc_GLM_double = (double*)mxGetData(prhs[5]);
    
    int NUMBER_OF_DIMENSIONS = mxGetNumberOfDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_DATA = mxGetDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_GLM = mxGetDimensions(prhs[2]);
    const int *ARRAY_DIMENSIONS_CONTRAST = mxGetDimensions(prhs[5]);
    
    int DATA_H, DATA_W, DATA_D, DATA_T, NUMBER_OF_REGRESSORS, NUMBER_OF_CONTRASTS;
    
    DATA_H = ARRAY_DIMENSIONS_DATA[0];
    DATA_W = ARRAY_DIMENSIONS_DATA[1];
    DATA_D = ARRAY_DIMENSIONS_DATA[2];
    DATA_T = ARRAY_DIMENSIONS_DATA[3];
    
    NUMBER_OF_REGRESSORS = ARRAY_DIMENSIONS_GLM[1];
    NUMBER_OF_CONTRASTS = ARRAY_DIMENSIONS_CONTRAST[1];
            
    int DATA_SIZE = DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float);
    int VOLUME_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);
    int GLM_SIZE = DATA_T * NUMBER_OF_REGRESSORS * sizeof(float);
    int CONTRAST_SIZE = NUMBER_OF_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float);
    int CONTRAST_SCALAR_SIZE = NUMBER_OF_CONTRASTS * sizeof(float);
    int BETA_SIZE = DATA_W * DATA_H * DATA_D * NUMBER_OF_REGRESSORS * sizeof(float);
    int STATISTICAL_MAPS_SIZE = DATA_W * DATA_H * DATA_D * NUMBER_OF_CONTRASTS * sizeof(float);
            
    mexPrintf("Data size : %i x %i x %i x %i \n",  DATA_W, DATA_H, DATA_D, DATA_T);
    mexPrintf("Number of regressors : %i \n",  NUMBER_OF_REGRESSORS);
    mexPrintf("Number of contrasts : %i \n",  NUMBER_OF_CONTRASTS);
    
    //-------------------------------------------------
    // Output to Matlab
    
    // Create pointer for volumes to Matlab        
    NUMBER_OF_DIMENSIONS = 4;
    int ARRAY_DIMENSIONS_OUT_BETA[4];
    ARRAY_DIMENSIONS_OUT_BETA[0] = DATA_H;
    ARRAY_DIMENSIONS_OUT_BETA[1] = DATA_W;
    ARRAY_DIMENSIONS_OUT_BETA[2] = DATA_D;
    ARRAY_DIMENSIONS_OUT_BETA[3] = NUMBER_OF_REGRESSORS;
    
    plhs[0] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_BETA,mxDOUBLE_CLASS, mxREAL);
    h_Beta_Volumes_double = mxGetPr(plhs[0]);          
    
    NUMBER_OF_DIMENSIONS = 4;
    int ARRAY_DIMENSIONS_OUT_RESIDUALS[4];
    ARRAY_DIMENSIONS_OUT_RESIDUALS[0] = DATA_H;
    ARRAY_DIMENSIONS_OUT_RESIDUALS[1] = DATA_W;
    ARRAY_DIMENSIONS_OUT_RESIDUALS[2] = DATA_D;
    ARRAY_DIMENSIONS_OUT_RESIDUALS[3] = DATA_T;
    
    plhs[1] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_RESIDUALS,mxDOUBLE_CLASS, mxREAL);
    h_Residuals_double = mxGetPr(plhs[1]);          
        
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES[3];
    ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES[0] = DATA_H;
    ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES[1] = DATA_W;
    ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES[2] = DATA_D;
    
    plhs[2] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES,mxDOUBLE_CLASS, mxREAL);
    h_Residual_Variances_double = mxGetPr(plhs[2]);          
    
    NUMBER_OF_DIMENSIONS = 4;
    int ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[4];
    ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[0] = DATA_H;
    ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[1] = DATA_W;
    ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[2] = DATA_D;
    ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[3] = NUMBER_OF_CONTRASTS;
    
    plhs[3] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS,mxDOUBLE_CLASS, mxREAL);
    h_Statistical_Maps_double = mxGetPr(plhs[3]);          
    
    
    // ------------------------------------------------
    
    // Allocate memory on the host
    h_fMRI_Volumes                 = (float *)mxMalloc(DATA_SIZE);
    h_Mask                         = (float *)mxMalloc(VOLUME_SIZE);
    h_X_GLM                        = (float *)mxMalloc(GLM_SIZE);
    h_xtxxt_GLM                    = (float *)mxMalloc(GLM_SIZE);
    h_Contrasts                    = (float *)mxMalloc(CONTRAST_SIZE);
    h_ctxtxc_GLM                   = (float *)mxMalloc(CONTRAST_SCALAR_SIZE);
    
    h_Beta_Volumes                 = (float *)mxMalloc(BETA_SIZE);
    h_Residuals                    = (float *)mxMalloc(DATA_SIZE);
    h_Residual_Variances           = (float *)mxMalloc(VOLUME_SIZE);
    h_Statistical_Maps             = (float *)mxMalloc(STATISTICAL_MAPS_SIZE);
        
    // Reorder and cast data
    pack_double2float_volumes(h_fMRI_Volumes, h_fMRI_Volumes_double, DATA_W, DATA_H, DATA_D, DATA_T);
    pack_double2float_volume(h_Mask, h_Mask_double, DATA_W, DATA_H, DATA_D);
    pack_double2float(h_X_GLM, h_X_GLM_double, NUMBER_OF_REGRESSORS * DATA_T);
    pack_double2float(h_xtxxt_GLM, h_xtxxt_GLM_double, NUMBER_OF_REGRESSORS * DATA_T);    
    pack_double2float(h_Contrasts, h_Contrasts_double, NUMBER_OF_REGRESSORS * NUMBER_OF_CONTRASTS);
    pack_double2float(h_ctxtxc_GLM, h_ctxtxc_GLM_double, NUMBER_OF_CONTRASTS);
       
    //------------------------
    
    BROCCOLI_LIB BROCCOLI;
    
    BROCCOLI.SetEPIWidth(DATA_W);
    BROCCOLI.SetEPIHeight(DATA_H);
    BROCCOLI.SetEPIDepth(DATA_D);
    BROCCOLI.SetEPITimepoints(DATA_T);   
    BROCCOLI.SetNumberOfRegressors(NUMBER_OF_REGRESSORS);
    BROCCOLI.SetNumberOfContrasts(NUMBER_OF_CONTRASTS);    
    BROCCOLI.SetInputfMRIVolumes(h_fMRI_Volumes);
    BROCCOLI.SetMask(h_Mask);
    BROCCOLI.SetDesignMatrix(h_X_GLM, h_xtxxt_GLM);
    BROCCOLI.SetContrasts(h_Contrasts);
    BROCCOLI.SetGLMScalars(h_ctxtxc_GLM);
    BROCCOLI.SetOutputBetaVolumes(h_Beta_Volumes);
    BROCCOLI.SetOutputResiduals(h_Residuals);
    BROCCOLI.SetOutputResidualVariances(h_Residual_Variances);
    BROCCOLI.SetOutputStatisticalMaps(h_Statistical_Maps);
    
    mexPrintf("Device info \n \n %s \n", BROCCOLI.GetOpenCLDeviceInfoChar());
    mexPrintf("Build info \n \n %s \n", BROCCOLI.GetOpenCLBuildInfoChar());
            
    BROCCOLI.PerformGLMWrapper();
    
    int error = BROCCOLI.GetOpenCLError();
    mexPrintf("Error is %d \n",error);
    
    int createKernelError = BROCCOLI.GetOpenCLCreateKernelError();
    mexPrintf("Create kernel error is %d \n",createKernelError);
    
    double GLM_time = BROCCOLI.GetProcessingTimeConvolution();
    //mexPrintf("GLM time is %f ms \n",GLM_time/1000000.0);
    mexPrintf("GLM time is %f ms \n",GLM_time);
    
    unpack_float2double_volumes(h_Beta_Volumes_double, h_Beta_Volumes, DATA_W, DATA_H, DATA_D, NUMBER_OF_REGRESSORS);
    unpack_float2double_volumes(h_Residuals_double, h_Residuals, DATA_W, DATA_H, DATA_D, DATA_T);
    unpack_float2double_volume(h_Residual_Variances_double, h_Residual_Variances, DATA_W, DATA_H, DATA_D);
    unpack_float2double_volumes(h_Statistical_Maps_double, h_Statistical_Maps, DATA_W, DATA_H, DATA_D, NUMBER_OF_CONTRASTS);
        
    // Free all the allocated memory on the host
    mxFree(h_fMRI_Volumes);
    mxFree(h_Mask);
    mxFree(h_X_GLM);
    mxFree(h_xtxxt_GLM);    
    mxFree(h_Contrasts);
    mxFree(h_ctxtxc_GLM);
    
    mxFree(h_Beta_Volumes);
    mxFree(h_Residuals);
    mxFree(h_Residual_Variances);
    mxFree(h_Statistical_Maps);
    
    //mexAtExit(cleanUp);
    
    return;
}


