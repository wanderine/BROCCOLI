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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //-----------------------
    // Input pointers
    
    double		    *h_fMRI_Volumes_double, *h_Quadrature_Filter_1_Real_double, *h_Quadrature_Filter_2_Real_double, *h_Quadrature_Filter_3_Real_double, *h_Quadrature_Filter_1_Imag_double, *h_Quadrature_Filter_2_Imag_double, *h_Quadrature_Filter_3_Imag_double;
    float           *h_fMRI_Volumes, *h_Quadrature_Filter_1_Real, *h_Quadrature_Filter_2_Real, *h_Quadrature_Filter_3_Real, *h_Quadrature_Filter_1_Imag, *h_Quadrature_Filter_2_Imag, *h_Quadrature_Filter_3_Imag;
    const char*     BROCCOLI_LOCATION;
    int             MOTION_CORRECTION_FILTER_SIZE, NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION;
    int             OPENCL_PLATFORM,OPENCL_DEVICE;
    float           EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z;
    
    //-----------------------
    // Output pointers        
    
    double     		*h_Motion_Corrected_fMRI_Volumes_double, *h_Motion_Parameters_double;
    float           *h_Motion_Parameters;
    
    //---------------------
    
    /* Check the number of input and output arguments. */
    if(nrhs<11)
    {
        mexErrMsgTxt("Too few input arguments.");
    }
    if(nrhs>11)
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
    
    /* Input arguments */
    
    // The data
    h_fMRI_Volumes_double =  (double*)mxGetData(prhs[0]);
    EPI_VOXEL_SIZE_X = (float)mxGetScalar(prhs[1]);
    EPI_VOXEL_SIZE_Y = (float)mxGetScalar(prhs[2]);
    EPI_VOXEL_SIZE_Z = (float)mxGetScalar(prhs[3]);
    h_Quadrature_Filter_1_Real_double =  (double*)mxGetPr(prhs[4]);
    h_Quadrature_Filter_1_Imag_double =  (double*)mxGetPi(prhs[4]);
    h_Quadrature_Filter_2_Real_double =  (double*)mxGetPr(prhs[5]);
    h_Quadrature_Filter_2_Imag_double =  (double*)mxGetPi(prhs[5]);
    h_Quadrature_Filter_3_Real_double =  (double*)mxGetPr(prhs[6]);
    h_Quadrature_Filter_3_Imag_double =  (double*)mxGetPi(prhs[6]);
    NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION  = (int)mxGetScalar(prhs[7]);
    OPENCL_PLATFORM  = (int)mxGetScalar(prhs[8]);
    OPENCL_DEVICE  = (int)mxGetScalar(prhs[9]);
    BROCCOLI_LOCATION  = mxArrayToString(prhs[10]);
    
    int NUMBER_OF_DIMENSIONS = mxGetNumberOfDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_DATA = mxGetDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_FILTER = mxGetDimensions(prhs[4]);
    
    int DATA_H, DATA_W, DATA_D, DATA_T, NUMBER_OF_MOTION_CORRECTION_PARAMETERS;
    NUMBER_OF_MOTION_CORRECTION_PARAMETERS = 6;
    
    DATA_H = ARRAY_DIMENSIONS_DATA[0];
    DATA_W = ARRAY_DIMENSIONS_DATA[1];
    DATA_D = ARRAY_DIMENSIONS_DATA[2];
    DATA_T = ARRAY_DIMENSIONS_DATA[3];
                
    MOTION_CORRECTION_FILTER_SIZE = ARRAY_DIMENSIONS_FILTER[0];
            
    int DATA_SIZE = DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float);
    int MOTION_PARAMETERS_SIZE = NUMBER_OF_MOTION_CORRECTION_PARAMETERS * DATA_T * sizeof(float);
    int FILTER_SIZE = MOTION_CORRECTION_FILTER_SIZE * MOTION_CORRECTION_FILTER_SIZE * MOTION_CORRECTION_FILTER_SIZE * sizeof(float);
    int VOLUME_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);
    
    mexPrintf("Data size : %i x %i x %i x %i \n",  DATA_W, DATA_H, DATA_D, DATA_T);
    mexPrintf("Voxel size : %f x %f x %f mm \n",  EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
    
    //-------------------------------------------------
    // Output to Matlab
    
    // Create pointer for volumes to Matlab        
    NUMBER_OF_DIMENSIONS = 4;
    int ARRAY_DIMENSIONS_OUT_MOTION_CORRECTED_FMRI_VOLUMES[4];
    ARRAY_DIMENSIONS_OUT_MOTION_CORRECTED_FMRI_VOLUMES[0] = DATA_H;
    ARRAY_DIMENSIONS_OUT_MOTION_CORRECTED_FMRI_VOLUMES[1] = DATA_W;
    ARRAY_DIMENSIONS_OUT_MOTION_CORRECTED_FMRI_VOLUMES[2] = DATA_D;
    ARRAY_DIMENSIONS_OUT_MOTION_CORRECTED_FMRI_VOLUMES[3] = DATA_T;
    
    plhs[0] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_MOTION_CORRECTED_FMRI_VOLUMES,mxDOUBLE_CLASS, mxREAL);
    h_Motion_Corrected_fMRI_Volumes_double = mxGetPr(plhs[0]);          
    
    NUMBER_OF_DIMENSIONS = 2;
    int ARRAY_DIMENSIONS_OUT_MOTION_PARAMETERS[2];
    ARRAY_DIMENSIONS_OUT_MOTION_PARAMETERS[0] = DATA_T;
    ARRAY_DIMENSIONS_OUT_MOTION_PARAMETERS[1] = NUMBER_OF_MOTION_CORRECTION_PARAMETERS;    
    
    plhs[1] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_MOTION_PARAMETERS,mxDOUBLE_CLASS, mxREAL);
    h_Motion_Parameters_double = mxGetPr(plhs[1]);          
        
    // ------------------------------------------------
    
    // Allocate memory on the host
    h_fMRI_Volumes                         = (float *)mxMalloc(DATA_SIZE);
    h_Quadrature_Filter_1_Real             = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_1_Imag             = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_2_Real             = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_2_Imag             = (float *)mxMalloc(FILTER_SIZE);    
    h_Quadrature_Filter_3_Real             = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_3_Imag             = (float *)mxMalloc(FILTER_SIZE);    
    h_Motion_Parameters                    = (float *)mxMalloc(MOTION_PARAMETERS_SIZE);    
    
    // Pack data (reorder from y,x,z to x,y,z and cast from double to float)
    pack_double2float_volumes(h_fMRI_Volumes, h_fMRI_Volumes_double, DATA_W, DATA_H, DATA_D, DATA_T);
    pack_double2float_volume(h_Quadrature_Filter_1_Real, h_Quadrature_Filter_1_Real_double, MOTION_CORRECTION_FILTER_SIZE, MOTION_CORRECTION_FILTER_SIZE, MOTION_CORRECTION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_1_Imag, h_Quadrature_Filter_1_Imag_double, MOTION_CORRECTION_FILTER_SIZE, MOTION_CORRECTION_FILTER_SIZE, MOTION_CORRECTION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_2_Real, h_Quadrature_Filter_2_Real_double, MOTION_CORRECTION_FILTER_SIZE, MOTION_CORRECTION_FILTER_SIZE, MOTION_CORRECTION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_2_Imag, h_Quadrature_Filter_2_Imag_double, MOTION_CORRECTION_FILTER_SIZE, MOTION_CORRECTION_FILTER_SIZE, MOTION_CORRECTION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_3_Real, h_Quadrature_Filter_3_Real_double, MOTION_CORRECTION_FILTER_SIZE, MOTION_CORRECTION_FILTER_SIZE, MOTION_CORRECTION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_3_Imag, h_Quadrature_Filter_3_Imag_double, MOTION_CORRECTION_FILTER_SIZE, MOTION_CORRECTION_FILTER_SIZE, MOTION_CORRECTION_FILTER_SIZE);

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
        BROCCOLI.SetEPIWidth(DATA_W);
        BROCCOLI.SetEPIHeight(DATA_H);
        BROCCOLI.SetEPIDepth(DATA_D);
        BROCCOLI.SetEPITimepoints(DATA_T);   
        
        BROCCOLI.SetEPIVoxelSizeX(EPI_VOXEL_SIZE_X);
        BROCCOLI.SetEPIVoxelSizeY(EPI_VOXEL_SIZE_Y);
        BROCCOLI.SetEPIVoxelSizeZ(EPI_VOXEL_SIZE_Z);
        
        BROCCOLI.SetInputfMRIVolumes(h_fMRI_Volumes);
        BROCCOLI.SetImageRegistrationFilterSize(MOTION_CORRECTION_FILTER_SIZE);
        BROCCOLI.SetLinearImageRegistrationFilters(h_Quadrature_Filter_1_Real, h_Quadrature_Filter_1_Imag, h_Quadrature_Filter_2_Real, h_Quadrature_Filter_2_Imag, h_Quadrature_Filter_3_Real, h_Quadrature_Filter_3_Imag);
        BROCCOLI.SetNumberOfIterationsForMotionCorrection(NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION);
        BROCCOLI.SetOutputMotionParameters(h_Motion_Parameters);
             
        mexPrintf("Running motion correction \n");
        BROCCOLI.PerformMotionCorrectionWrapper();        
    
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
    
    // Unpack results to Matlab
    unpack_float2double_volumes(h_Motion_Corrected_fMRI_Volumes_double, h_fMRI_Volumes, DATA_W, DATA_H, DATA_D, DATA_T);
    unpack_float2double(h_Motion_Parameters_double, h_Motion_Parameters, NUMBER_OF_MOTION_CORRECTION_PARAMETERS * DATA_T);
    
    // Free all the allocated memory on the host
    mxFree(h_fMRI_Volumes);
    mxFree(h_Quadrature_Filter_1_Real);
    mxFree(h_Quadrature_Filter_1_Imag);
    mxFree(h_Quadrature_Filter_2_Real);
    mxFree(h_Quadrature_Filter_2_Imag);
    mxFree(h_Quadrature_Filter_3_Real);
    mxFree(h_Quadrature_Filter_3_Imag);
    mxFree(h_Motion_Parameters);        
    
    return;
}


