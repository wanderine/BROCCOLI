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
#include <math.h>

float round_( float d )
{
    return floor( d + 0.5f );
}

void cleanUp()
{
    //cudaDeviceReset();
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //-----------------------
    // Input pointers
    
    double          *h_Quadrature_Filter_1_Parametric_Registration_Real_double, *h_Quadrature_Filter_2_Parametric_Registration_Real_double, *h_Quadrature_Filter_3_Parametric_Registration_Real_double, *h_Quadrature_Filter_1_Parametric_Registration_Imag_double, *h_Quadrature_Filter_2_Parametric_Registration_Imag_double, *h_Quadrature_Filter_3_Parametric_Registration_Imag_double;
    float           *h_Quadrature_Filter_1_Parametric_Registration_Real, *h_Quadrature_Filter_2_Parametric_Registration_Real, *h_Quadrature_Filter_3_Parametric_Registration_Real, *h_Quadrature_Filter_1_Parametric_Registration_Imag, *h_Quadrature_Filter_2_Parametric_Registration_Imag, *h_Quadrature_Filter_3_Parametric_Registration_Imag;
    
    double		    *h_T1_Volume_double, *h_EPI_Volume_double;
    float           *h_T1_Volume, *h_EPI_Volume;
    int             IMAGE_REGISTRATION_FILTER_SIZE, NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION, COARSEST_SCALE, MM_EPI_Z_CUT;
    int             OPENCL_PLATFORM, OPENCL_DEVICE;
       
    float2          *h_Quadrature_Filter_1_Parametric_Registration, *h_Quadrature_Filter_2_Parametric_Registration, *h_Quadrature_Filter_3_Parametric_Registration;
    float2          *h_Quadrature_Filter_Response_1, *h_Quadrature_Filter_Response_2, *h_Quadrature_Filter_Response_3;    
    
    float           T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z;
    float           EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z;
    
    //-----------------------
    // Output pointers        
    
    double     		*h_Aligned_EPI_Volume_double, *h_Interpolated_EPI_Volume_double, *h_Registration_Parameters_double;
    double          *h_Quadrature_Filter_Response_1_Real_double, *h_Quadrature_Filter_Response_1_Imag_double;
    double          *h_Quadrature_Filter_Response_2_Real_double, *h_Quadrature_Filter_Response_2_Imag_double;
    double          *h_Quadrature_Filter_Response_3_Real_double, *h_Quadrature_Filter_Response_3_Imag_double;
    double          *h_Phase_Differences_double, *h_Phase_Certainties_double, *h_Phase_Gradients_double;
    float           *h_Quadrature_Filter_Response_1_Real, *h_Quadrature_Filter_Response_1_Imag;
    float           *h_Quadrature_Filter_Response_2_Real, *h_Quadrature_Filter_Response_2_Imag;
    float           *h_Quadrature_Filter_Response_3_Real, *h_Quadrature_Filter_Response_3_Imag;  
    float           *h_Phase_Differences, *h_Phase_Certainties, *h_Phase_Gradients;
    float           *h_Aligned_EPI_Volume, *h_Interpolated_EPI_Volume, *h_Registration_Parameters;
    
    //---------------------
    
    /* Check the number of input and output arguments. */
    if(nrhs<16)
    {
        mexErrMsgTxt("Too few input arguments.");
    }
    if(nrhs>16)
    {
        mexErrMsgTxt("Too many input arguments.");
    }
    if(nlhs<9)
    {
        mexErrMsgTxt("Too few output arguments.");
    }
    if(nlhs>9)
    {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Input arguments */
    
    // The data
    h_EPI_Volume_double =  (double*)mxGetData(prhs[0]);
    h_T1_Volume_double =  (double*)mxGetData(prhs[1]);
    EPI_VOXEL_SIZE_X = (float)mxGetScalar(prhs[2]);
    EPI_VOXEL_SIZE_Y = (float)mxGetScalar(prhs[3]);
    EPI_VOXEL_SIZE_Z = (float)mxGetScalar(prhs[4]);
    T1_VOXEL_SIZE_X = (float)mxGetScalar(prhs[5]);
    T1_VOXEL_SIZE_Y = (float)mxGetScalar(prhs[6]);
    T1_VOXEL_SIZE_Z = (float)mxGetScalar(prhs[7]);    
    h_Quadrature_Filter_1_Parametric_Registration_Real_double =  (double*)mxGetPr(prhs[8]);
    h_Quadrature_Filter_1_Parametric_Registration_Imag_double =  (double*)mxGetPi(prhs[8]);
    h_Quadrature_Filter_2_Parametric_Registration_Real_double =  (double*)mxGetPr(prhs[9]);
    h_Quadrature_Filter_2_Parametric_Registration_Imag_double =  (double*)mxGetPi(prhs[9]);
    h_Quadrature_Filter_3_Parametric_Registration_Real_double =  (double*)mxGetPr(prhs[10]);
    h_Quadrature_Filter_3_Parametric_Registration_Imag_double =  (double*)mxGetPi(prhs[10]);
    NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION  = (int)mxGetScalar(prhs[11]);
    COARSEST_SCALE  = (int)mxGetScalar(prhs[12]);
    MM_EPI_Z_CUT  = (int)mxGetScalar(prhs[13]);
    OPENCL_PLATFORM  = (int)mxGetScalar(prhs[14]);
    OPENCL_DEVICE  = (int)mxGetScalar(prhs[15]);
    
    int NUMBER_OF_DIMENSIONS = mxGetNumberOfDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_EPI = mxGetDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_T1 = mxGetDimensions(prhs[1]);
    const int *ARRAY_DIMENSIONS_FILTER = mxGetDimensions(prhs[8]);
    
    int T1_DATA_H, T1_DATA_W, T1_DATA_D, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS;
    NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS = 6;
    
    EPI_DATA_H = ARRAY_DIMENSIONS_EPI[0];
    EPI_DATA_W = ARRAY_DIMENSIONS_EPI[1];
    EPI_DATA_D = ARRAY_DIMENSIONS_EPI[2];
    
    T1_DATA_H = ARRAY_DIMENSIONS_T1[0];
    T1_DATA_W = ARRAY_DIMENSIONS_T1[1];
    T1_DATA_D = ARRAY_DIMENSIONS_T1[2];
        
    IMAGE_REGISTRATION_FILTER_SIZE = ARRAY_DIMENSIONS_FILTER[0];
            
   	int EPI_DATA_W_INTERPOLATED = (int)round_((float)EPI_DATA_W * EPI_VOXEL_SIZE_X / T1_VOXEL_SIZE_X);
	int EPI_DATA_H_INTERPOLATED = (int)round_((float)EPI_DATA_H * EPI_VOXEL_SIZE_Y / T1_VOXEL_SIZE_Y);
	int EPI_DATA_D_INTERPOLATED = (int)round_((float)EPI_DATA_D * EPI_VOXEL_SIZE_Z / T1_VOXEL_SIZE_Z);

    int IMAGE_REGISTRATION_PARAMETERS_SIZE = NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float);
    int FILTER_SIZE = IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float);
    int FILTER_SIZE2 = IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float2);
    int T1_VOLUME_SIZE = T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float);
    int EPI_VOLUME_SIZE = EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);
    int INTERPOLATED_EPI_VOLUME_SIZE = EPI_DATA_W_INTERPOLATED * EPI_DATA_H_INTERPOLATED * EPI_DATA_D_INTERPOLATED * sizeof(float);
    
    mexPrintf("EPI size : %i x %i x %i \n",  EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
    mexPrintf("EPI interpolated size : %i x %i x %i \n",  EPI_DATA_W_INTERPOLATED, EPI_DATA_H_INTERPOLATED, EPI_DATA_D_INTERPOLATED);
    mexPrintf("Filter size : %i x %i x %i \n",  IMAGE_REGISTRATION_FILTER_SIZE,IMAGE_REGISTRATION_FILTER_SIZE,IMAGE_REGISTRATION_FILTER_SIZE);
    mexPrintf("Number of iterations : %i \n",  NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION);
    
    //-------------------------------------------------
    // Output to Matlab
    
    // Create pointer for volumes to Matlab        
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_ALIGNED_EPI_VOLUME[3];
    ARRAY_DIMENSIONS_OUT_ALIGNED_EPI_VOLUME[0] = T1_DATA_H;
    ARRAY_DIMENSIONS_OUT_ALIGNED_EPI_VOLUME[1] = T1_DATA_W;
    ARRAY_DIMENSIONS_OUT_ALIGNED_EPI_VOLUME[2] = T1_DATA_D;
    
    plhs[0] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_ALIGNED_EPI_VOLUME,mxDOUBLE_CLASS, mxREAL);
    h_Aligned_EPI_Volume_double = mxGetPr(plhs[0]);          
             
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_INTERPOLATED_EPI_VOLUME[3];
    ARRAY_DIMENSIONS_OUT_INTERPOLATED_EPI_VOLUME[0] = EPI_DATA_H_INTERPOLATED;
    ARRAY_DIMENSIONS_OUT_INTERPOLATED_EPI_VOLUME[1] = EPI_DATA_W_INTERPOLATED;
    ARRAY_DIMENSIONS_OUT_INTERPOLATED_EPI_VOLUME[2] = EPI_DATA_D_INTERPOLATED;
    
    //plhs[1] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_INTERPOLATED_T1_VOLUME,mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_ALIGNED_EPI_VOLUME,mxDOUBLE_CLASS, mxREAL);
    h_Interpolated_EPI_Volume_double = mxGetPr(plhs[1]);          
        
    NUMBER_OF_DIMENSIONS = 2;
    int ARRAY_DIMENSIONS_OUT_IMAGE_REGISTRATION_PARAMETERS[2];
    ARRAY_DIMENSIONS_OUT_IMAGE_REGISTRATION_PARAMETERS[0] = 1;
    ARRAY_DIMENSIONS_OUT_IMAGE_REGISTRATION_PARAMETERS[1] = NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS;    
    
    plhs[2] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_IMAGE_REGISTRATION_PARAMETERS,mxDOUBLE_CLASS, mxREAL);
    h_Registration_Parameters_double = mxGetPr(plhs[2]);          
    
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_FILTER_RESPONSE[3];
    ARRAY_DIMENSIONS_OUT_FILTER_RESPONSE[0] = T1_DATA_H;
    ARRAY_DIMENSIONS_OUT_FILTER_RESPONSE[1] = T1_DATA_W;    
    ARRAY_DIMENSIONS_OUT_FILTER_RESPONSE[2] = T1_DATA_D;  
    
    plhs[3] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_FILTER_RESPONSE,mxDOUBLE_CLASS, mxCOMPLEX);
    h_Quadrature_Filter_Response_1_Real_double = mxGetPr(plhs[3]);          
    h_Quadrature_Filter_Response_1_Imag_double = mxGetPi(plhs[3]);          
    
    plhs[4] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_FILTER_RESPONSE,mxDOUBLE_CLASS, mxCOMPLEX);
    h_Quadrature_Filter_Response_2_Real_double = mxGetPr(plhs[4]);          
    h_Quadrature_Filter_Response_2_Imag_double = mxGetPi(plhs[4]);          
        
    plhs[5] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_FILTER_RESPONSE,mxDOUBLE_CLASS, mxCOMPLEX);
    h_Quadrature_Filter_Response_3_Real_double = mxGetPr(plhs[5]);          
    h_Quadrature_Filter_Response_3_Imag_double = mxGetPi(plhs[5]);          
                
    plhs[6] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_FILTER_RESPONSE,mxDOUBLE_CLASS, mxREAL);
    h_Phase_Differences_double = mxGetPr(plhs[6]);          
    
    plhs[7] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_FILTER_RESPONSE,mxDOUBLE_CLASS, mxREAL);
    h_Phase_Certainties_double = mxGetPr(plhs[7]);          
    
    plhs[8] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_FILTER_RESPONSE,mxDOUBLE_CLASS, mxREAL);
    h_Phase_Gradients_double = mxGetPr(plhs[8]);          
    
    
    // ------------------------------------------------
    
    // Allocate memory on the host
    h_EPI_Volume                                        = (float *)mxMalloc(EPI_VOLUME_SIZE);
    h_T1_Volume                                         = (float *)mxMalloc(T1_VOLUME_SIZE);
    h_Interpolated_EPI_Volume                           = (float *)mxMalloc(T1_VOLUME_SIZE);
    h_Quadrature_Filter_1_Parametric_Registration_Real  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_2_Parametric_Registration_Real  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_3_Parametric_Registration_Real  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_1_Parametric_Registration_Imag  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_2_Parametric_Registration_Imag  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_3_Parametric_Registration_Imag  = (float *)mxMalloc(FILTER_SIZE);    
    h_Quadrature_Filter_Response_1                      = (float2 *)mxMalloc(T1_VOLUME_SIZE*2);
    h_Quadrature_Filter_Response_2                      = (float2 *)mxMalloc(T1_VOLUME_SIZE*2);
    h_Quadrature_Filter_Response_3                      = (float2 *)mxMalloc(T1_VOLUME_SIZE*2);
    h_Phase_Differences                                 = (float *)mxMalloc(T1_VOLUME_SIZE);    
    h_Phase_Certainties                                 = (float *)mxMalloc(T1_VOLUME_SIZE);  
    h_Phase_Gradients                                   = (float *)mxMalloc(T1_VOLUME_SIZE);  
    h_Aligned_EPI_Volume                                = (float *)mxMalloc(T1_VOLUME_SIZE);
    h_Registration_Parameters                           = (float *)mxMalloc(IMAGE_REGISTRATION_PARAMETERS_SIZE);

    
    // Reorder and cast data
    pack_double2float_volume(h_EPI_Volume, h_EPI_Volume_double, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
    pack_double2float_volume(h_T1_Volume, h_T1_Volume_double, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    
    //pack_c2c_volume(h_Quadrature_Filter_1_Parametric_Registration, h_Quadrature_Filter_1_Parametric_Registration_Real_double, h_Quadrature_Filter_1_Parametric_Registration_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    //pack_c2c_volume(h_Quadrature_Filter_2_Parametric_Registration, h_Quadrature_Filter_2_Parametric_Registration_Real_double, h_Quadrature_Filter_2_Parametric_Registration_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    //pack_c2c_volume(h_Quadrature_Filter_3_Parametric_Registration, h_Quadrature_Filter_3_Parametric_Registration_Real_double, h_Quadrature_Filter_3_Parametric_Registration_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    
    pack_double2float_volume(h_Quadrature_Filter_1_Parametric_Registration_Real, h_Quadrature_Filter_1_Parametric_Registration_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_1_Parametric_Registration_Imag, h_Quadrature_Filter_1_Parametric_Registration_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_2_Parametric_Registration_Real, h_Quadrature_Filter_2_Parametric_Registration_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_2_Parametric_Registration_Imag, h_Quadrature_Filter_2_Parametric_Registration_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_3_Parametric_Registration_Real, h_Quadrature_Filter_3_Parametric_Registration_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_3_Parametric_Registration_Imag, h_Quadrature_Filter_3_Parametric_Registration_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    
    
    //------------------------
    
    BROCCOLI_LIB BROCCOLI(OPENCL_PLATFORM, OPENCL_DEVICE);

    BROCCOLI.SetEPIWidth(EPI_DATA_W);
    BROCCOLI.SetEPIHeight(EPI_DATA_H);
    BROCCOLI.SetEPIDepth(EPI_DATA_D);    
    BROCCOLI.SetEPIVoxelSizeX(EPI_VOXEL_SIZE_X);
    BROCCOLI.SetEPIVoxelSizeY(EPI_VOXEL_SIZE_Y);
    BROCCOLI.SetEPIVoxelSizeZ(EPI_VOXEL_SIZE_Z);                
    BROCCOLI.SetT1Width(T1_DATA_W);
    BROCCOLI.SetT1Height(T1_DATA_H);
    BROCCOLI.SetT1Depth(T1_DATA_D);
    BROCCOLI.SetT1VoxelSizeX(T1_VOXEL_SIZE_X);
    BROCCOLI.SetT1VoxelSizeY(T1_VOXEL_SIZE_Y);
    BROCCOLI.SetT1VoxelSizeZ(T1_VOXEL_SIZE_Z);   
    BROCCOLI.SetInputEPIVolume(h_EPI_Volume);
    BROCCOLI.SetInputT1Volume(h_T1_Volume);
    BROCCOLI.SetInterpolationMode(LINEAR);
    BROCCOLI.SetImageRegistrationFilterSize(IMAGE_REGISTRATION_FILTER_SIZE);
    //BROCCOLI.SetParametricImageRegistrationFilters(h_Quadrature_Filter_1_Parametric_Registration, h_Quadrature_Filter_2_Parametric_Registration, h_Quadrature_Filter_3_Parametric_Registration);
    BROCCOLI.SetParametricImageRegistrationFilters(h_Quadrature_Filter_1_Parametric_Registration_Real, h_Quadrature_Filter_1_Parametric_Registration_Imag, h_Quadrature_Filter_2_Parametric_Registration_Real, h_Quadrature_Filter_2_Parametric_Registration_Imag, h_Quadrature_Filter_3_Parametric_Registration_Real, h_Quadrature_Filter_3_Parametric_Registration_Imag);
    BROCCOLI.SetNumberOfIterationsForParametricImageRegistration(NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION);
    BROCCOLI.SetCoarsestScaleEPIT1(COARSEST_SCALE);
    BROCCOLI.SetMMEPIZCUT(MM_EPI_Z_CUT);   
    BROCCOLI.SetOutputAlignedEPIVolume(h_Aligned_EPI_Volume);
    BROCCOLI.SetOutputInterpolatedEPIVolume(h_Interpolated_EPI_Volume);
    BROCCOLI.SetOutputEPIT1RegistrationParameters(h_Registration_Parameters);
    //BROCCOLI.SetOutputQuadratureFilterResponses(h_Quadrature_Filter_Response_1, h_Quadrature_Filter_Response_2, h_Quadrature_Filter_Response_3);
    BROCCOLI.SetOutputPhaseDifferences(h_Phase_Differences);
    BROCCOLI.SetOutputPhaseCertainties(h_Phase_Certainties);
    BROCCOLI.SetOutputPhaseGradients(h_Phase_Gradients);       
    
    
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
    for (int i = 0; i < 46; i++)
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
        BROCCOLI.PerformRegistrationEPIT1Wrapper();
    }
    else
    {
        mexPrintf("OPENCL error detected, aborting \n");
    }   
    
    
    double convolution_time = BROCCOLI.GetProcessingTimeConvolution();
    mexPrintf("Convolution time is %f ms \n",convolution_time/1000000.0);

    unpack_float2double_volume(h_Aligned_EPI_Volume_double, h_Aligned_EPI_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    unpack_float2double_volume(h_Interpolated_EPI_Volume_double, h_Interpolated_EPI_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    unpack_float2double(h_Registration_Parameters_double, h_Registration_Parameters, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS);
    
    /*
    unpack_float2double_volume(h_Quadrature_Filter_Response_1_Real_double, h_Quadrature_Filter_Response_1_Real, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    unpack_float2double_volume(h_Quadrature_Filter_Response_1_Imag_double, h_Quadrature_Filter_Response_1_Imag, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    unpack_float2double_volume(h_Quadrature_Filter_Response_2_Real_double, h_Quadrature_Filter_Response_2_Real, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    unpack_float2double_volume(h_Quadrature_Filter_Response_2_Imag_double, h_Quadrature_Filter_Response_2_Imag, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    unpack_float2double_volume(h_Quadrature_Filter_Response_3_Real_double, h_Quadrature_Filter_Response_3_Real, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    unpack_float2double_volume(h_Quadrature_Filter_Response_3_Imag_double, h_Quadrature_Filter_Response_3_Imag, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    */
    
    unpack_c2c_volume(h_Quadrature_Filter_Response_1_Real_double, h_Quadrature_Filter_Response_1_Imag_double, h_Quadrature_Filter_Response_1, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    unpack_c2c_volume(h_Quadrature_Filter_Response_2_Real_double, h_Quadrature_Filter_Response_2_Imag_double, h_Quadrature_Filter_Response_2, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    unpack_c2c_volume(h_Quadrature_Filter_Response_3_Real_double, h_Quadrature_Filter_Response_3_Imag_double, h_Quadrature_Filter_Response_3, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    
    unpack_float2double_volume(h_Phase_Differences_double, h_Phase_Differences, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    unpack_float2double_volume(h_Phase_Certainties_double, h_Phase_Certainties, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    unpack_float2double_volume(h_Phase_Gradients_double, h_Phase_Gradients, T1_DATA_W, T1_DATA_H, T1_DATA_D);    

    
    // Free all the allocated memory on the host
    mxFree(h_EPI_Volume);    
    mxFree(h_T1_Volume);
    mxFree(h_Interpolated_EPI_Volume);
    
    //mxFree(h_Quadrature_Filter_1_Parametric_Registration);
    //mxFree(h_Quadrature_Filter_2_Parametric_Registration);
    //mxFree(h_Quadrature_Filter_3_Parametric_Registration);
    
    mxFree(h_Quadrature_Filter_1_Parametric_Registration_Real);
    mxFree(h_Quadrature_Filter_2_Parametric_Registration_Real);
    mxFree(h_Quadrature_Filter_3_Parametric_Registration_Real);
    mxFree(h_Quadrature_Filter_1_Parametric_Registration_Imag);
    mxFree(h_Quadrature_Filter_2_Parametric_Registration_Imag);
    mxFree(h_Quadrature_Filter_3_Parametric_Registration_Imag);
    
    mxFree(h_Quadrature_Filter_Response_1);
    mxFree(h_Quadrature_Filter_Response_2);
    mxFree(h_Quadrature_Filter_Response_3);
    mxFree(h_Phase_Differences);
    mxFree(h_Phase_Certainties);
    mxFree(h_Phase_Gradients);
    mxFree(h_Aligned_EPI_Volume); 
    mxFree(h_Registration_Parameters);

    
    //mexAtExit(cleanUp);
    
    return;
}


