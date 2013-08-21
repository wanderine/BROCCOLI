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
    
    double		    *h_fMRI_Volumes_double, *h_X_GLM_double, *h_xtxxt_GLM_double, *h_Contrasts_double, *h_ctxtxc_GLM_double;
    float           *h_fMRI_Volumes, *h_X_GLM, *h_xtxxt_GLM, *h_Contrasts, *h_ctxtxc_GLM;  
    
    float           EPI_SMOOTHING_AMOUNT, AR_SMOOTHING_AMOUNT;
    float           EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z;
    
    int             NUMBER_OF_PERMUTATIONS, OPENCL_PLATFORM, OPENCL_DEVICE;
    
    //-----------------------
    // Output pointers        
    
    double     		*h_Beta_Volumes_double, *h_Residuals_double, *h_Residual_Variances_double, *h_Statistical_Maps_double;
    double          *h_AR1_Estimates_double, *h_AR2_Estimates_double, *h_AR3_Estimates_double, *h_AR4_Estimates_double;
    double          *h_Permutation_Distribution_double;
    float           *h_Beta_Volumes, *h_Residuals, *h_Residual_Variances, *h_Statistical_Maps;
    float           *h_AR1_Estimates, *h_AR2_Estimates, *h_AR3_Estimates, *h_AR4_Estimates;
    float           *h_Permutation_Distribution;
    double          *h_Detrended_fMRI_Volumes_double, *h_Whitened_fMRI_Volumes_double, *h_Permuted_fMRI_Volumes_double;
    float           *h_Detrended_fMRI_Volumes, *h_Whitened_fMRI_Volumes, *h_Permuted_fMRI_Volumes;
    
    //---------------------
    
    /* Check the number of input and output arguments. */
    if(nrhs<13)
    {
        mexErrMsgTxt("Too few input arguments.");
    }
    if(nrhs>13)
    {
        mexErrMsgTxt("Too many input arguments.");
    }
    if(nlhs<12)
    {
        mexErrMsgTxt("Too few output arguments.");
    }
    if(nlhs>12)
    {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Input arguments */
    
    // The data
    h_fMRI_Volumes_double =  (double*)mxGetData(prhs[0]);
    h_X_GLM_double =  (double*)mxGetData(prhs[1]);
    h_xtxxt_GLM_double =  (double*)mxGetData(prhs[2]);
    h_Contrasts_double = (double*)mxGetData(prhs[3]);
    h_ctxtxc_GLM_double = (double*)mxGetData(prhs[4]);
    EPI_SMOOTHING_AMOUNT = (float)mxGetScalar(prhs[5]);
    AR_SMOOTHING_AMOUNT = (float)mxGetScalar(prhs[6]);
    EPI_VOXEL_SIZE_X = (float)mxGetScalar(prhs[7]);
    EPI_VOXEL_SIZE_Y = (float)mxGetScalar(prhs[8]);
    EPI_VOXEL_SIZE_Z = (float)mxGetScalar(prhs[9]);
    NUMBER_OF_PERMUTATIONS = (int)mxGetScalar(prhs[10]);
    OPENCL_PLATFORM  = (int)mxGetScalar(prhs[11]);
    OPENCL_DEVICE  = (int)mxGetScalar(prhs[12]);
    
    int NUMBER_OF_DIMENSIONS = mxGetNumberOfDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_DATA = mxGetDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_GLM = mxGetDimensions(prhs[1]);
    const int *ARRAY_DIMENSIONS_CONTRAST = mxGetDimensions(prhs[4]);
    
    int DATA_H, DATA_W, DATA_D, DATA_T, NUMBER_OF_GLM_REGRESSORS, NUMBER_OF_CONTRASTS;
    
    DATA_H = ARRAY_DIMENSIONS_DATA[0];
    DATA_W = ARRAY_DIMENSIONS_DATA[1];
    DATA_D = ARRAY_DIMENSIONS_DATA[2];
    DATA_T = ARRAY_DIMENSIONS_DATA[3];
    
    NUMBER_OF_GLM_REGRESSORS = ARRAY_DIMENSIONS_GLM[1];
    NUMBER_OF_CONTRASTS = ARRAY_DIMENSIONS_CONTRAST[1];
                
    int DATA_SIZE = DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float);
    int VOLUME_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);
    int GLM_SIZE = DATA_T * NUMBER_OF_GLM_REGRESSORS * sizeof(float);
    int CONTRAST_SIZE = NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float);
    int CONTRAST_SCALAR_SIZE = NUMBER_OF_CONTRASTS * sizeof(float);
    int BETA_SIZE = DATA_W * DATA_H * DATA_D * NUMBER_OF_GLM_REGRESSORS * sizeof(float);
    int STATISTICAL_MAPS_SIZE = DATA_W * DATA_H * DATA_D * NUMBER_OF_CONTRASTS * sizeof(float);
    int DISTRIBUTION_SIZE = NUMBER_OF_PERMUTATIONS * sizeof(float);
    
    mexPrintf("Data size : %i x %i x %i x %i \n",  DATA_W, DATA_H, DATA_D, DATA_T);
    mexPrintf("Number of regressors : %i \n",  NUMBER_OF_GLM_REGRESSORS);
    mexPrintf("Number of contrasts : %i \n",  NUMBER_OF_CONTRASTS);
    mexPrintf("Number of permutations : %i \n",  NUMBER_OF_PERMUTATIONS);
    
    //-------------------------------------------------
    // Output to Matlab
    
    // Create pointer for volumes to Matlab        
    NUMBER_OF_DIMENSIONS = 4;
    int ARRAY_DIMENSIONS_OUT_BETA[4];
    ARRAY_DIMENSIONS_OUT_BETA[0] = DATA_H;
    ARRAY_DIMENSIONS_OUT_BETA[1] = DATA_W;
    ARRAY_DIMENSIONS_OUT_BETA[2] = DATA_D;
    ARRAY_DIMENSIONS_OUT_BETA[3] = NUMBER_OF_GLM_REGRESSORS;
    
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
    
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_AR_ESTIMATES[3];
    ARRAY_DIMENSIONS_OUT_AR_ESTIMATES[0] = DATA_H;
    ARRAY_DIMENSIONS_OUT_AR_ESTIMATES[1] = DATA_W;
    ARRAY_DIMENSIONS_OUT_AR_ESTIMATES[2] = DATA_D;
    
    plhs[4] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_AR_ESTIMATES,mxDOUBLE_CLASS, mxREAL);
    h_AR1_Estimates_double = mxGetPr(plhs[4]);          
    
    plhs[5] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_AR_ESTIMATES,mxDOUBLE_CLASS, mxREAL);
    h_AR2_Estimates_double = mxGetPr(plhs[5]);          
    
    plhs[6] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_AR_ESTIMATES,mxDOUBLE_CLASS, mxREAL);
    h_AR3_Estimates_double = mxGetPr(plhs[6]);          
    
    plhs[7] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_AR_ESTIMATES,mxDOUBLE_CLASS, mxREAL);
    h_AR4_Estimates_double = mxGetPr(plhs[7]);          
    
    NUMBER_OF_DIMENSIONS = 2;
    int ARRAY_DIMENSIONS_OUT_DISTRIBUTION[2];
    ARRAY_DIMENSIONS_OUT_DISTRIBUTION[0] = NUMBER_OF_PERMUTATIONS;
    ARRAY_DIMENSIONS_OUT_DISTRIBUTION[1] = 1;
    
    plhs[8] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_DISTRIBUTION,mxDOUBLE_CLASS, mxREAL);
    h_Permutation_Distribution_double = mxGetPr(plhs[8]);          
    
    NUMBER_OF_DIMENSIONS = 4;
    int ARRAY_DIMENSIONS_OUT_DETRENDED[4];
    ARRAY_DIMENSIONS_OUT_DETRENDED[0] = DATA_H;
    ARRAY_DIMENSIONS_OUT_DETRENDED[1] = DATA_W;
    ARRAY_DIMENSIONS_OUT_DETRENDED[2] = DATA_D;
    ARRAY_DIMENSIONS_OUT_DETRENDED[3] = DATA_T;
    
    plhs[9] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_DETRENDED,mxDOUBLE_CLASS, mxREAL);
    h_Detrended_fMRI_Volumes_double = mxGetPr(plhs[9]);          
    
    plhs[10] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_DETRENDED,mxDOUBLE_CLASS, mxREAL);
    h_Whitened_fMRI_Volumes_double = mxGetPr(plhs[10]);          
    
    plhs[11] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_DETRENDED,mxDOUBLE_CLASS, mxREAL);
    h_Permuted_fMRI_Volumes_double = mxGetPr(plhs[11]);          
    
    // ------------------------------------------------
    
    // Allocate memory on the host
    h_fMRI_Volumes                 = (float *)mxMalloc(DATA_SIZE);
    h_Detrended_fMRI_Volumes       = (float *)mxMalloc(DATA_SIZE);
    h_Whitened_fMRI_Volumes        = (float *)mxMalloc(DATA_SIZE);
    h_Permuted_fMRI_Volumes        = (float *)mxMalloc(DATA_SIZE);
    
    h_X_GLM                        = (float *)mxMalloc(GLM_SIZE);
    h_xtxxt_GLM                    = (float *)mxMalloc(GLM_SIZE);
    h_Contrasts                    = (float *)mxMalloc(CONTRAST_SIZE);
    h_ctxtxc_GLM                   = (float *)mxMalloc(CONTRAST_SCALAR_SIZE);
    
    h_Beta_Volumes                 = (float *)mxMalloc(BETA_SIZE);
    h_Residuals                    = (float *)mxMalloc(DATA_SIZE);
    h_Residual_Variances           = (float *)mxMalloc(VOLUME_SIZE);
    h_Statistical_Maps             = (float *)mxMalloc(STATISTICAL_MAPS_SIZE);
    
    h_AR1_Estimates                = (float *)mxMalloc(VOLUME_SIZE);
    h_AR2_Estimates                = (float *)mxMalloc(VOLUME_SIZE);
    h_AR3_Estimates                = (float *)mxMalloc(VOLUME_SIZE);
    h_AR4_Estimates                = (float *)mxMalloc(VOLUME_SIZE);
    
    h_Permutation_Distribution     = (float *)mxMalloc(DISTRIBUTION_SIZE);
    
    
        
    // Reorder and cast data
    pack_double2float_volumes(h_fMRI_Volumes, h_fMRI_Volumes_double, DATA_W, DATA_H, DATA_D, DATA_T);
    pack_double2float(h_X_GLM, h_X_GLM_double, NUMBER_OF_GLM_REGRESSORS * DATA_T);
    pack_double2float(h_xtxxt_GLM, h_xtxxt_GLM_double, NUMBER_OF_GLM_REGRESSORS * DATA_T);    
    pack_double2float(h_Contrasts, h_Contrasts_double, NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS);
    pack_double2float(h_ctxtxc_GLM, h_ctxtxc_GLM_double, NUMBER_OF_CONTRASTS);
    
    //------------------------
    
    BROCCOLI_LIB BROCCOLI(OPENCL_PLATFORM,OPENCL_DEVICE);
    
    BROCCOLI.SetEPIWidth(DATA_W);
    BROCCOLI.SetEPIHeight(DATA_H);
    BROCCOLI.SetEPIDepth(DATA_D);
    BROCCOLI.SetEPITimepoints(DATA_T);   
    BROCCOLI.SetNumberOfGLMRegressors(NUMBER_OF_GLM_REGRESSORS);
    BROCCOLI.SetNumberOfContrasts(NUMBER_OF_CONTRASTS);    
    BROCCOLI.SetInputfMRIVolumes(h_fMRI_Volumes);
    BROCCOLI.SetDesignMatrix(h_X_GLM, h_xtxxt_GLM);
    BROCCOLI.SetContrasts(h_Contrasts);
    BROCCOLI.SetGLMScalars(h_ctxtxc_GLM);
    BROCCOLI.SetNumberOfPermutations(NUMBER_OF_PERMUTATIONS);
    BROCCOLI.SetEPIVoxelSizeX(EPI_VOXEL_SIZE_X);
    BROCCOLI.SetEPIVoxelSizeY(EPI_VOXEL_SIZE_Y);
    BROCCOLI.SetEPIVoxelSizeZ(EPI_VOXEL_SIZE_Z);  
    BROCCOLI.SetEPISmoothingAmount(EPI_SMOOTHING_AMOUNT);
    BROCCOLI.SetARSmoothingAmount(AR_SMOOTHING_AMOUNT);
    BROCCOLI.SetOutputBetaVolumes(h_Beta_Volumes);
    BROCCOLI.SetOutputResiduals(h_Residuals);
    BROCCOLI.SetOutputResidualVariances(h_Residual_Variances);
    BROCCOLI.SetOutputStatisticalMaps(h_Statistical_Maps);
    BROCCOLI.SetOutputAREstimates(h_AR1_Estimates, h_AR2_Estimates, h_AR3_Estimates, h_AR4_Estimates);
    BROCCOLI.SetOutputPermutationDistribution(h_Permutation_Distribution);
    BROCCOLI.SetOutputDetrendedfMRIVolumes(h_Detrended_fMRI_Volumes);
    BROCCOLI.SetOutputWhitenedfMRIVolumes(h_Whitened_fMRI_Volumes);
    BROCCOLI.SetOutputPermutedfMRIVolumes(h_Permuted_fMRI_Volumes);
    
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
    for (int i = 0; i < 39; i++)
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
        BROCCOLI.CalculatePermutationTestThresholdFirstLevelWrapper();
    }
    else
    {
        mexPrintf("OPENCL error detected, aborting \n");
    }
    
    
    unpack_float2double_volumes(h_Beta_Volumes_double, h_Beta_Volumes, DATA_W, DATA_H, DATA_D, NUMBER_OF_GLM_REGRESSORS);
    unpack_float2double_volumes(h_Residuals_double, h_Residuals, DATA_W, DATA_H, DATA_D, DATA_T);
    unpack_float2double_volume(h_Residual_Variances_double, h_Residual_Variances, DATA_W, DATA_H, DATA_D);
    unpack_float2double_volumes(h_Statistical_Maps_double, h_Statistical_Maps, DATA_W, DATA_H, DATA_D, NUMBER_OF_CONTRASTS);
    unpack_float2double_volume(h_AR1_Estimates_double, h_AR1_Estimates, DATA_W, DATA_H, DATA_D);
    unpack_float2double_volume(h_AR2_Estimates_double, h_AR2_Estimates, DATA_W, DATA_H, DATA_D);
    unpack_float2double_volume(h_AR3_Estimates_double, h_AR3_Estimates, DATA_W, DATA_H, DATA_D);
    unpack_float2double_volume(h_AR4_Estimates_double, h_AR4_Estimates, DATA_W, DATA_H, DATA_D);
    unpack_float2double(h_Permutation_Distribution_double, h_Permutation_Distribution, NUMBER_OF_PERMUTATIONS);        
    unpack_float2double_volumes(h_Detrended_fMRI_Volumes_double, h_Detrended_fMRI_Volumes, DATA_W, DATA_H, DATA_D, DATA_T);
    unpack_float2double_volumes(h_Whitened_fMRI_Volumes_double, h_Whitened_fMRI_Volumes, DATA_W, DATA_H, DATA_D, DATA_T);
    unpack_float2double_volumes(h_Permuted_fMRI_Volumes_double, h_Permuted_fMRI_Volumes, DATA_W, DATA_H, DATA_D, DATA_T);
    
    // Free all the allocated memory on the host
    mxFree(h_fMRI_Volumes);
    mxFree(h_Detrended_fMRI_Volumes);
    mxFree(h_Whitened_fMRI_Volumes);
    mxFree(h_Permuted_fMRI_Volumes);
    
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
    
    mxFree(h_Permutation_Distribution);
    
    return;
}


