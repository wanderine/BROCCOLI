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


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //-----------------------
    // Input
    
    double		    *h_First_Level_Results_double, *h_MNI_Brain_Mask_double;
    float           *h_First_Level_Results, *h_MNI_Brain_Mask; 

    unsigned short int        *h_Permutation_Matrix;
    
    double		    *h_X_GLM_double, *h_xtxxt_GLM_double, *h_Contrasts_double, *h_ctxtxc_GLM_double;
    float           *h_X_GLM, *h_xtxxt_GLM, *h_Contrasts, *h_ctxtxc_GLM;  
            
    int             MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_SUBJECTS, NUMBER_OF_PERMUTATIONS; 
                    
    int             NUMBER_OF_GLM_REGRESSORS, NUMBER_OF_CONTRASTS, INFERENCE_MODE; 
    float           CLUSTER_DEFINING_THRESHOLD;
    
    int             OPENCL_PLATFORM, OPENCL_DEVICE;
    
    int             NUMBER_OF_DIMENSIONS;
        
    //-----------------------
    // Output
    
    double          *h_Permutation_Distribution_double;
    double     		*h_Beta_Volumes_double, *h_Residuals_double, *h_Residual_Variances_double, *h_Statistical_Maps_double;    
    int             *h_Cluster_Indices, *h_Cluster_Indices_Out;
    double          *h_Permuted_First_Level_Results_double;
    float           *h_Permutation_Distribution;
    float           *h_Beta_Volumes, *h_Residuals, *h_Residual_Variances, *h_Statistical_Maps;        
    float           *h_Permuted_First_Level_Results;
    
    //---------------------
    
    /* Check the number of input and output arguments. */
    if(nrhs<12)
    {
        mexErrMsgTxt("Too few input arguments.");
    }
    if(nrhs>12)
    {
        mexErrMsgTxt("Too many input arguments.");
    }
    if(nlhs<7)
    {
        mexErrMsgTxt("Too few output arguments.");
    }
    if(nlhs>7)
    {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Input arguments */
    
    // The data
    h_First_Level_Results_double =  (double*)mxGetData(prhs[0]);
    h_MNI_Brain_Mask_double = (double*)mxGetData(prhs[1]);
          
    h_X_GLM_double =  (double*)mxGetData(prhs[2]);    
    h_xtxxt_GLM_double =  (double*)mxGetData(prhs[3]);
    h_Contrasts_double = (double*)mxGetData(prhs[4]);
    h_ctxtxc_GLM_double = (double*)mxGetData(prhs[5]);    
    
    h_Permutation_Matrix = (unsigned short int*)mxGetData(prhs[6]);
    NUMBER_OF_PERMUTATIONS = (int)mxGetScalar(prhs[7]);        
    INFERENCE_MODE = (int)mxGetScalar(prhs[8]);        
    CLUSTER_DEFINING_THRESHOLD = (float)mxGetScalar(prhs[9]);    
    
    OPENCL_PLATFORM  = (int)mxGetScalar(prhs[10]);
    OPENCL_DEVICE = (int)mxGetScalar(prhs[11]);
    
    const int *ARRAY_DIMENSIONS_FIRST_LEVEL_RESULTS = mxGetDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_MNI = mxGetDimensions(prhs[1]);
    const int *ARRAY_DIMENSIONS_GLM = mxGetDimensions(prhs[2]);    
    const int *ARRAY_DIMENSIONS_CONTRAST = mxGetDimensions(prhs[5]);        
      
    NUMBER_OF_SUBJECTS = ARRAY_DIMENSIONS_FIRST_LEVEL_RESULTS[3];
    
    MNI_DATA_H = ARRAY_DIMENSIONS_MNI[0];
    MNI_DATA_W = ARRAY_DIMENSIONS_MNI[1];
    MNI_DATA_D = ARRAY_DIMENSIONS_MNI[2];
            
    NUMBER_OF_GLM_REGRESSORS = ARRAY_DIMENSIONS_GLM[1];
    NUMBER_OF_CONTRASTS = ARRAY_DIMENSIONS_CONTRAST[1];
    
    int FIRST_LEVEL_RESULTS_DATA_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float);
    int MNI_DATA_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float);
    int MNI_DATA_SIZE_INT = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int);
        
    int GLM_SIZE = NUMBER_OF_SUBJECTS * NUMBER_OF_GLM_REGRESSORS * sizeof(float);
    int CONTRAST_SIZE = NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float);
    int CONTRAST_MATRIX_SIZE = NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float);
    int DESIGN_MATRIX_SIZE = NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float);
            
    int BETA_DATA_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_GLM_REGRESSORS * sizeof(float);        
    int STATISTICAL_MAPS_DATA_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float);    
    int RESIDUAL_VARIANCES_DATA_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float);    
    
    int RESIDUAL_DATA_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float);
    
    int NULL_DISTRIBUTION_SIZE = NUMBER_OF_PERMUTATIONS * sizeof(float);
    
    mexPrintf("First level results data size : %i x %i x %i \n", MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    mexPrintf("Number of subjects: %i \n",NUMBER_OF_SUBJECTS);
    mexPrintf("Number of GLM regressors : %i \n",  NUMBER_OF_GLM_REGRESSORS);
    mexPrintf("Number of contrasts : %i \n",  NUMBER_OF_CONTRASTS);
    mexPrintf("Number of permutations : %i \n",  NUMBER_OF_PERMUTATIONS);
    
    //-------------------------------------------------
    // Output to Matlab
    
    // Create pointer for volumes to Matlab        
    NUMBER_OF_DIMENSIONS = 4;
    int ARRAY_DIMENSIONS_OUT_BETA[4];
    ARRAY_DIMENSIONS_OUT_BETA[0] = MNI_DATA_H;
    ARRAY_DIMENSIONS_OUT_BETA[1] = MNI_DATA_W;
    ARRAY_DIMENSIONS_OUT_BETA[2] = MNI_DATA_D; 
    ARRAY_DIMENSIONS_OUT_BETA[3] = NUMBER_OF_GLM_REGRESSORS;
    
    plhs[0] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_BETA,mxDOUBLE_CLASS, mxREAL);
    h_Beta_Volumes_double = mxGetPr(plhs[0]);          
    
    NUMBER_OF_DIMENSIONS = 4;
    int ARRAY_DIMENSIONS_OUT_RESIDUALS[4];
    ARRAY_DIMENSIONS_OUT_RESIDUALS[0] = MNI_DATA_H;
    ARRAY_DIMENSIONS_OUT_RESIDUALS[1] = MNI_DATA_W;
    ARRAY_DIMENSIONS_OUT_RESIDUALS[2] = MNI_DATA_D;
    ARRAY_DIMENSIONS_OUT_RESIDUALS[3] = NUMBER_OF_SUBJECTS;
    
    plhs[1] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_RESIDUALS,mxDOUBLE_CLASS, mxREAL);
    h_Residuals_double = mxGetPr(plhs[1]);          
        
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES[3];
    ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES[0] = MNI_DATA_H;
    ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES[1] = MNI_DATA_W;
    ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES[2] = MNI_DATA_D;
    
    plhs[2] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_RESIDUAL_VARIANCES,mxDOUBLE_CLASS, mxREAL);
    h_Residual_Variances_double = mxGetPr(plhs[2]);          
    
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[3];
    ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[0] = MNI_DATA_H;
    ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[1] = MNI_DATA_W;
    ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[2] = MNI_DATA_D;            
    
    plhs[3] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS,mxDOUBLE_CLASS, mxREAL);
    h_Statistical_Maps_double = mxGetPr(plhs[3]);          
                  
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_CLUSTER_INDICES[3];
    ARRAY_DIMENSIONS_OUT_CLUSTER_INDICES[0] = MNI_DATA_H;
    ARRAY_DIMENSIONS_OUT_CLUSTER_INDICES[1] = MNI_DATA_W;
    ARRAY_DIMENSIONS_OUT_CLUSTER_INDICES[2] = MNI_DATA_D;
    
    plhs[4] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_CLUSTER_INDICES,mxINT32_CLASS, mxREAL);
    h_Cluster_Indices_Out = (int*)mxGetData(plhs[4]);   
    
    NUMBER_OF_DIMENSIONS = 2;
    int ARRAY_DIMENSIONS_OUT_DISTRIBUTION[2];
    ARRAY_DIMENSIONS_OUT_DISTRIBUTION[0] = NUMBER_OF_PERMUTATIONS;
    ARRAY_DIMENSIONS_OUT_DISTRIBUTION[1] = 1;
    
    plhs[5] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_DISTRIBUTION,mxDOUBLE_CLASS, mxREAL);
    h_Permutation_Distribution_double = mxGetPr(plhs[5]); 
    
    NUMBER_OF_DIMENSIONS = 4;
    int ARRAY_DIMENSIONS_OUT_PERMUTED[4];
    ARRAY_DIMENSIONS_OUT_PERMUTED[0] = MNI_DATA_H;
    ARRAY_DIMENSIONS_OUT_PERMUTED[1] = MNI_DATA_W;
    ARRAY_DIMENSIONS_OUT_PERMUTED[2] = MNI_DATA_D; 
    ARRAY_DIMENSIONS_OUT_PERMUTED[3] = NUMBER_OF_SUBJECTS;
    
    plhs[6] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_PERMUTED,mxDOUBLE_CLASS, mxREAL);
    h_Permuted_First_Level_Results_double = mxGetPr(plhs[6]);  
    
    // ------------------------------------------------
    
    // Allocate memory on the host
    h_First_Level_Results               = (float *)mxMalloc(FIRST_LEVEL_RESULTS_DATA_SIZE);
    h_Permuted_First_Level_Results      = (float *)mxMalloc(FIRST_LEVEL_RESULTS_DATA_SIZE);
    
    h_MNI_Brain_Mask                    = (float *)mxMalloc(MNI_DATA_SIZE);
    
    h_Cluster_Indices                   = (int *)mxMalloc(MNI_DATA_SIZE_INT);
            
    h_X_GLM                        = (float *)mxMalloc(GLM_SIZE);
    h_xtxxt_GLM                    = (float *)mxMalloc(GLM_SIZE);
    h_Contrasts                    = (float *)mxMalloc(CONTRAST_SIZE);
    h_ctxtxc_GLM                   = (float *)mxMalloc(CONTRAST_MATRIX_SIZE);
         
    h_Beta_Volumes                      = (float *)mxMalloc(BETA_DATA_SIZE);
    h_Residuals                         = (float *)mxMalloc(RESIDUAL_DATA_SIZE);
    h_Residual_Variances                = (float *)mxMalloc(RESIDUAL_VARIANCES_DATA_SIZE);
    h_Statistical_Maps                  = (float *)mxMalloc(STATISTICAL_MAPS_DATA_SIZE);
                
    h_Permutation_Distribution          = (float *)mxMalloc(NULL_DISTRIBUTION_SIZE);
    
    // Reorder and cast data
    pack_double2float_volumes(h_First_Level_Results, h_First_Level_Results_double, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_SUBJECTS);
    pack_double2float_volume(h_MNI_Brain_Mask, h_MNI_Brain_Mask_double, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    
    pack_double2float(h_X_GLM, h_X_GLM_double, NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_SUBJECTS);
    pack_double2float(h_xtxxt_GLM, h_xtxxt_GLM_double, NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_SUBJECTS);    
    //pack_double2float(h_Contrasts, h_Contrasts_double, NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS);    
    pack_double2float_image(h_Contrasts, h_Contrasts_double, NUMBER_OF_GLM_REGRESSORS, NUMBER_OF_CONTRASTS);        
    pack_double2float(h_ctxtxc_GLM, h_ctxtxc_GLM_double, NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS);       
    
    //------------------------
    
    
    BROCCOLI_LIB BROCCOLI(OPENCL_PLATFORM, OPENCL_DEVICE);
        
    // Something went wrong...
    if (BROCCOLI.GetOpenCLInitiated() == 0)
    {    
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
    
        // Print create kernel errors
        int* createKernelErrors = BROCCOLI.GetOpenCLCreateKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createKernelErrors[i] != 0)
            {
                mexPrintf("Create kernel error %i is %d \n",i,createKernelErrors[i]);
            }
        }                
        
        mexPrintf("OPENCL initialization failed, aborting \n");        
    }    
    else if (BROCCOLI.GetOpenCLInitiated() == 1)
    {     
        BROCCOLI.SetInputFirstLevelResults(h_First_Level_Results);        
        BROCCOLI.SetInputMNIBrainMask(h_MNI_Brain_Mask);
        BROCCOLI.SetMNIWidth(MNI_DATA_W);
        BROCCOLI.SetMNIHeight(MNI_DATA_H);
        BROCCOLI.SetMNIDepth(MNI_DATA_D);                
        BROCCOLI.SetStatisticalTest(1); // F-test
        BROCCOLI.SetInferenceMode(INFERENCE_MODE);
        BROCCOLI.SetClusterDefiningThreshold(CLUSTER_DEFINING_THRESHOLD);
        BROCCOLI.SetNumberOfSubjects(NUMBER_OF_SUBJECTS);
        BROCCOLI.SetNumberOfPermutations(NUMBER_OF_PERMUTATIONS);
        BROCCOLI.SetNumberOfGLMRegressors(NUMBER_OF_GLM_REGRESSORS);
        BROCCOLI.SetNumberOfContrasts(NUMBER_OF_CONTRASTS);    
        BROCCOLI.SetDesignMatrix(h_X_GLM, h_xtxxt_GLM);
        BROCCOLI.SetContrasts(h_Contrasts);
        BROCCOLI.SetGLMScalars(h_ctxtxc_GLM);
        BROCCOLI.SetPermutationMatrix(h_Permutation_Matrix);        
        BROCCOLI.SetOutputBetaVolumes(h_Beta_Volumes);        
        BROCCOLI.SetOutputResiduals(h_Residuals);        
        BROCCOLI.SetOutputResidualVariances(h_Residual_Variances);        
        BROCCOLI.SetOutputStatisticalMaps(h_Statistical_Maps);        
        BROCCOLI.SetOutputClusterIndices(h_Cluster_Indices);
        BROCCOLI.SetOutputPermutationDistribution(h_Permutation_Distribution);
        BROCCOLI.SetOutputPermutedFirstLevelResults(h_Permuted_First_Level_Results);       

        BROCCOLI.PerformGLMFTestSecondLevelPermutationWrapper();
                
        // Print create buffer errors
        int* createBufferErrors = BROCCOLI.GetOpenCLCreateBufferErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createBufferErrors[i] != 0)
            {
                mexPrintf("Create buffer error %i is %d \n",i,createBufferErrors[i]);
            }
        }
        
        // Print run kernel errors
        int* runKernelErrors = BROCCOLI.GetOpenCLRunKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (runKernelErrors[i] != 0)
            {
                mexPrintf("Run kernel error %i is %d \n",i,runKernelErrors[i]);
            }
        } 
    }
    
    // Print build info
    mexPrintf("Build info \n \n %s \n", BROCCOLI.GetOpenCLBuildInfoChar());     
          
    // Unpack results to Matlab
    unpack_float2double_volumes(h_Permuted_First_Level_Results_double, h_Permuted_First_Level_Results, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_SUBJECTS);  
        
    unpack_int2int_volume(h_Cluster_Indices_Out, h_Cluster_Indices, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    
    unpack_float2double_volumes(h_Beta_Volumes_double, h_Beta_Volumes, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_GLM_REGRESSORS);  
    unpack_float2double_volume(h_Statistical_Maps_double, h_Statistical_Maps, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);                
    unpack_float2double_volume(h_Residual_Variances_double, h_Residual_Variances, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    
    unpack_float2double_volumes(h_Residuals_double, h_Residuals, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_SUBJECTS);        
        
    unpack_float2double(h_Permutation_Distribution_double, h_Permutation_Distribution, NUMBER_OF_PERMUTATIONS);  
            
    // Free all the allocated memory on the host
        
    mxFree(h_First_Level_Results);
    mxFree(h_Permuted_First_Level_Results);
    
    mxFree(h_MNI_Brain_Mask);
    
    mxFree(h_Cluster_Indices);
    
    mxFree(h_X_GLM);
    mxFree(h_xtxxt_GLM);
    mxFree(h_Contrasts);
    mxFree(h_ctxtxc_GLM);    
            
    mxFree(h_Beta_Volumes);
    mxFree(h_Residuals);
    mxFree(h_Residual_Variances);
    mxFree(h_Statistical_Maps);
          
    mxFree(h_Permutation_Distribution);
    
    return;
}



