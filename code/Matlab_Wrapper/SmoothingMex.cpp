/*
 * BROCCOLI: Software for Fast fMRI Analysis on Many-Core CPUs and GPUs
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
#include <stdlib.h> 

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //-----------------------
    // Input pointers
    
    double		    *h_fMRI_Volumes_double;
    float           *h_fMRI_Volumes;
    
    float           *h_Certainty;
        
    float           EPI_SMOOTHING_AMOUNT, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, SMOOTHING_TYPE;
    
    int             OPENCL_PLATFORM, OPENCL_DEVICE;
        
    const char*     BROCCOLI_LOCATION;
    
    //-----------------------
    // Output pointers
    
    double     		*h_Filter_Response_double;
        
    //---------------------
    
    /* Check the number of input and output arguments. */
    if(nrhs<8)
    {
        mexErrMsgTxt("Too few input arguments.");
    }
    if(nrhs>8)
    {
        mexErrMsgTxt("Too many input arguments.");
    }
    if(nlhs<1)
    {
        mexErrMsgTxt("Too few output arguments.");
    }
    if(nlhs>1)
    {
        mexErrMsgTxt("Too many output arguments.");
    }
        
    /* Input arguments */
    
    // The data
    h_fMRI_Volumes_double =  (double*)mxGetData(prhs[0]);
    EPI_VOXEL_SIZE_X = (float)mxGetScalar(prhs[1]);
    EPI_VOXEL_SIZE_Y = (float)mxGetScalar(prhs[2]);
    EPI_VOXEL_SIZE_Z = (float)mxGetScalar(prhs[3]);
    EPI_SMOOTHING_AMOUNT = (float)mxGetScalar(prhs[4]);
    OPENCL_PLATFORM  = (int)mxGetScalar(prhs[5]);
    OPENCL_DEVICE  = (int)mxGetScalar(prhs[6]);
    BROCCOLI_LOCATION  = mxArrayToString(prhs[7]);
    
    int NUMBER_OF_DIMENSIONS = mxGetNumberOfDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_DATA = mxGetDimensions(prhs[0]);
    
    int DATA_H, DATA_W, DATA_D, DATA_T, FILTER_LENGTH;
    
    DATA_H = ARRAY_DIMENSIONS_DATA[0];
    DATA_W = ARRAY_DIMENSIONS_DATA[1];
    DATA_D = ARRAY_DIMENSIONS_DATA[2];
    DATA_T = ARRAY_DIMENSIONS_DATA[3];
    
    size_t			allocatedHostMemory = 0;
    bool			AUTO_MASK = false;
    
    int DATA_SIZE = DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float);
    int VOLUME_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);
            
    mexPrintf("Data size : %i x %i x %i x %i \n",  DATA_W, DATA_H, DATA_D, DATA_T);
    mexPrintf("Voxel size : %f x %f x %f mm \n",  EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
        
    //-------------------------------------------------
    // Output to Matlab
    
    // Create pointer for volumes to Matlab
    
    int ARRAY_DIMENSIONS_OUT[4];
    ARRAY_DIMENSIONS_OUT[0] = DATA_H;
    ARRAY_DIMENSIONS_OUT[1] = DATA_W;
    ARRAY_DIMENSIONS_OUT[2] = DATA_D;
    ARRAY_DIMENSIONS_OUT[3] = DATA_T;
    
    plhs[0] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT,mxDOUBLE_CLASS, mxREAL);
    h_Filter_Response_double = mxGetPr(plhs[0]);          
    
    // ------------------------------------------------
    
    // Allocate memory on the host
    h_fMRI_Volumes                 = (float *)mxMalloc(DATA_SIZE);    
    h_Certainty                 = (float *)mxMalloc(VOLUME_SIZE);
    
    allocatedHostMemory += (size_t)DATA_SIZE;
    allocatedHostMemory += (size_t)VOLUME_SIZE;
    
    for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
    {
        h_Certainty[i] = 1.0f;
    }
    
    // Pack data (reorder from y,x,z to x,y,z and cast from double to float)
    pack_double2float_volumes(h_fMRI_Volumes, h_fMRI_Volumes_double, DATA_W, DATA_H, DATA_D, DATA_T);
    
    //------------------------
            
    BROCCOLI_LIB BROCCOLI(OPENCL_PLATFORM,OPENCL_DEVICE,BROCCOLI_LOCATION); 
        
    // Print build info to file (always)
	std::vector<std::string> buildInfo = BROCCOLI.GetOpenCLBuildInfo();
	std::vector<std::string> kernelFileNames = BROCCOLI.GetKernelFileNames();

	std::string buildInfoPath;
	buildInfoPath.append(BROCCOLI_LOCATION);
	buildInfoPath.append("compiled/Kernels/");
    
    FILE *fp = NULL;
    
    for (int k = 0; k < BROCCOLI.GetNumberOfKernelFiles(); k++)
	{
		std::string temp = buildInfoPath;
		temp.append("buildInfo_");
		temp.append(BROCCOLI.GetOpenCLPlatformName());
		temp.append("_");	
		temp.append(BROCCOLI.GetOpenCLDeviceName());
		temp.append("_");	
		std::string name = kernelFileNames[k];
		// Remove "kernel" and ".cpp" from kernel filename
		name = name.substr(0,name.size()-4);
		name = name.substr(6,name.size());
		temp.append(name);
		temp.append(".txt");
		fp = fopen(temp.c_str(),"w");
		if (fp == NULL)
		{     
		    mexPrintf("Could not open %s for writing ! \n",temp.c_str());
		}
		else
		{	
			if (buildInfo[k].c_str() != NULL)
			{
			    int error = fputs(buildInfo[k].c_str(),fp);
			    if (error == EOF)
			    {
			        mexPrintf("Could not write to %s ! \n",temp.c_str());
			    }
			}
			fclose(fp);
		}
	}
                
    // Something went wrong...
    if (!BROCCOLI.GetOpenCLInitiated())
    { 
        mexPrintf("Initialization error is \"%s\" \n",BROCCOLI.GetOpenCLInitializationError().c_str());
		mexPrintf("OpenCL error is \"%s\" \n",BROCCOLI.GetOpenCLError());

        // Print create kernel errors
        int* createKernelErrors = BROCCOLI.GetOpenCLCreateKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createKernelErrors[i] != 0)
            {
                mexPrintf("Create kernel error for kernel '%s' is '%s' \n",BROCCOLI.GetOpenCLKernelName(i),BROCCOLI.GetOpenCLErrorMessage(createKernelErrors[i]));
            }
        }                        
                
        mexPrintf("OpenCL initialization failed, aborting! \nSee buildInfo* for output of OpenCL compilation!\n");         
    }
    else
    {    
        // Set all necessary pointers and values
        BROCCOLI.SetInputfMRIVolumes(h_fMRI_Volumes);
		BROCCOLI.SetAutoMask(AUTO_MASK);
		BROCCOLI.SetInputCertainty(h_Certainty);

        BROCCOLI.SetEPISmoothingAmount(EPI_SMOOTHING_AMOUNT);
		BROCCOLI.SetAllocatedHostMemory(allocatedHostMemory);

        BROCCOLI.SetEPIWidth(DATA_W);
        BROCCOLI.SetEPIHeight(DATA_H);
        BROCCOLI.SetEPIDepth(DATA_D);
        BROCCOLI.SetEPITimepoints(DATA_T);  

        BROCCOLI.SetEPIVoxelSizeX(EPI_VOXEL_SIZE_X);
        BROCCOLI.SetEPIVoxelSizeY(EPI_VOXEL_SIZE_Y);
        BROCCOLI.SetEPIVoxelSizeZ(EPI_VOXEL_SIZE_Z); 
                                
        BROCCOLI.PerformSmoothingNormalizedHostWrapper(); 
                
        // Print create buffer errors
        int* createBufferErrors = BROCCOLI.GetOpenCLCreateBufferErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createBufferErrors[i] != 0)
            {
                mexPrintf("Create buffer error %i is %s \n",i,BROCCOLI.GetOpenCLErrorMessage(createBufferErrors[i]));
            }
        }
        
        // Print create kernel errors
        int* createKernelErrors = BROCCOLI.GetOpenCLCreateKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createKernelErrors[i] != 0)
            {
                mexPrintf("Create kernel error for kernel '%s' is '%s' \n",BROCCOLI.GetOpenCLKernelName(i),BROCCOLI.GetOpenCLErrorMessage(createKernelErrors[i]));
            }
        } 

        // Print run kernel errors
        int* runKernelErrors = BROCCOLI.GetOpenCLRunKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (runKernelErrors[i] != 0)
            {
                mexPrintf("Run kernel error for kernel '%s' is '%s' \n",BROCCOLI.GetOpenCLKernelName(i),BROCCOLI.GetOpenCLErrorMessage(runKernelErrors[i]));
            }
        } 
    }
        
    // Change from floats back to doubles
    unpack_float2double_volumes(h_Filter_Response_double, h_fMRI_Volumes, DATA_W, DATA_H, DATA_D, DATA_T);
                
    // Free all the allocated memory on the host
    mxFree(h_fMRI_Volumes);
    mxFree(h_Certainty);
        
    return;
}


