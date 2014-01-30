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
    
    double		    *h_Data_double, *h_MNI_Brain_Mask_double;
    float           *h_Data, *h_MNI_Brain_Mask; 
            
    int             MNI_DATA_W, MNI_DATA_H, MNI_DATA_D; 
                    
    float           CLUSTER_DEFINING_THRESHOLD;
    
    int             OPENCL_PLATFORM, OPENCL_DEVICE;
    
    int             NUMBER_OF_DIMENSIONS;
        
    int             ALGORITHM;
    
    //-----------------------
    // Output
    
    int             *h_Cluster_Indices, *h_Cluster_Indices_Out, *h_Largest_Cluster;
    
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
    if(nlhs<2)
    {
        mexErrMsgTxt("Too few output arguments.");
    }
    if(nlhs>2)
    {
        mexErrMsgTxt("Too many output arguments.");
    }            
    
    // The data
    h_Data_double =  (double*)mxGetData(prhs[0]);
    h_MNI_Brain_Mask_double = (double*)mxGetData(prhs[1]);          
    CLUSTER_DEFINING_THRESHOLD = (float)mxGetScalar(prhs[2]);        
    OPENCL_PLATFORM  = (int)mxGetScalar(prhs[3]);
    OPENCL_DEVICE = (int)mxGetScalar(prhs[4]);
    ALGORITHM = (int)mxGetScalar(prhs[5]);
    
    const int *ARRAY_DIMENSIONS_DATA = mxGetDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_MNI = mxGetDimensions(prhs[1]);
          
    MNI_DATA_H = ARRAY_DIMENSIONS_MNI[0];
    MNI_DATA_W = ARRAY_DIMENSIONS_MNI[1];
    MNI_DATA_D = ARRAY_DIMENSIONS_MNI[2];
            
    
    int DATA_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float);
    int MNI_DATA_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float);
    int MNI_DATA_SIZE_INT = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int);
            
    mexPrintf("Data size : %i x %i x %i \n", MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    
    //-------------------------------------------------
    // Output to Matlab
    
    // Create pointer for volumes to Matlab        
                  
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_CLUSTER_INDICES[3];
    ARRAY_DIMENSIONS_OUT_CLUSTER_INDICES[0] = MNI_DATA_H;
    ARRAY_DIMENSIONS_OUT_CLUSTER_INDICES[1] = MNI_DATA_W;
    ARRAY_DIMENSIONS_OUT_CLUSTER_INDICES[2] = MNI_DATA_D;
    
    plhs[0] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_CLUSTER_INDICES,mxINT32_CLASS, mxREAL);
    h_Cluster_Indices_Out = (int*)mxGetData(plhs[0]);           
    
    NUMBER_OF_DIMENSIONS = 2;
    int ARRAY_DIMENSIONS_OUT_LARGEST_CLUSTER[2];
    ARRAY_DIMENSIONS_OUT_LARGEST_CLUSTER[0] = 1;
    ARRAY_DIMENSIONS_OUT_LARGEST_CLUSTER[1] = 1;
    
    plhs[1] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_LARGEST_CLUSTER,mxINT32_CLASS, mxREAL);
    h_Largest_Cluster = (int*)mxGetData(plhs[1]);           
    
    // ------------------------------------------------
    
    // Allocate memory on the host
    h_Data                              = (float *)mxMalloc(DATA_SIZE);
    h_MNI_Brain_Mask                    = (float *)mxMalloc(MNI_DATA_SIZE);    
    h_Cluster_Indices                   = (int *)mxMalloc(MNI_DATA_SIZE_INT);
    
    // Reorder and cast data
    pack_double2float_volume(h_Data, h_Data_double, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    pack_double2float_volume(h_MNI_Brain_Mask, h_MNI_Brain_Mask_double, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
           
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
        
        BROCCOLI.SetInputFirstLevelResults(h_Data);        
        BROCCOLI.SetInputMNIBrainMask(h_MNI_Brain_Mask);        
        BROCCOLI.SetMNIWidth(MNI_DATA_W);
        BROCCOLI.SetMNIHeight(MNI_DATA_H);
        BROCCOLI.SetMNIDepth(MNI_DATA_D);                
        BROCCOLI.SetClusterDefiningThreshold(CLUSTER_DEFINING_THRESHOLD);
        BROCCOLI.SetOutputClusterIndices(h_Cluster_Indices);
        BROCCOLI.SetOutputLargestCluster(h_Largest_Cluster);

        if (ALGORITHM == 1)
        {
            BROCCOLI.ClusterizeOpenCLWrapper();
        }
        else if (ALGORITHM == 2)
        {
            BROCCOLI.ClusterizeOpenCLWrapper2();
        }
        else if (ALGORITHM == 3)
        {
            BROCCOLI.ClusterizeOpenCLWrapper3();
        }
                
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
    unpack_int2int_volume(h_Cluster_Indices_Out, h_Cluster_Indices, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    
    // Free all the allocated memory on the host            
    mxFree(h_Data);
    mxFree(h_MNI_Brain_Mask);   
    mxFree(h_Cluster_Indices);    

    
    return;
}



