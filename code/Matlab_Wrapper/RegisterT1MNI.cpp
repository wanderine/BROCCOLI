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
    
    double		    *h_T1_Volume_double, *h_MNI_Volume_double, *h_Quadrature_Filter_1_Real_double, *h_Quadrature_Filter_2_Real_double, *h_Quadrature_Filter_3_Real_double, *h_Quadrature_Filter_1_Imag_double, *h_Quadrature_Filter_2_Imag_double, *h_Quadrature_Filter_3_Imag_double;
    float           *h_T1_Volume, *h_MNI_Volume, *h_Quadrature_Filter_1_Real, *h_Quadrature_Filter_2_Real, *h_Quadrature_Filter_3_Real, *h_Quadrature_Filter_1_Imag, *h_Quadrature_Filter_2_Imag, *h_Quadrature_Filter_3_Imag;
    int             IMAGE_REGISTRATION_FILTER_SIZE, NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION, MM_T1_Z_CUT;
    
    float           T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z;
    float           MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z;
    
    //-----------------------
    // Output pointers        
    
    double     		*h_Aligned_T1_Volume_double, *h_Interpolated_T1_Volume_double, *h_Registration_Parameters_double;
    double          *h_Quadrature_Filter_Response_1_Real_double, *h_Quadrature_Filter_Response_1_Imag_double;
    double          *h_Quadrature_Filter_Response_2_Real_double, *h_Quadrature_Filter_Response_2_Imag_double;
    double          *h_Quadrature_Filter_Response_3_Real_double, *h_Quadrature_Filter_Response_3_Imag_double;
    double          *h_Phase_Differences_double, *h_Phase_Certainties_double, *h_Phase_Gradients_double;
    double          *h_Downsampled_Volume_double;
    float           *h_Quadrature_Filter_Response_1_Real, *h_Quadrature_Filter_Response_1_Imag;
    float           *h_Quadrature_Filter_Response_2_Real, *h_Quadrature_Filter_Response_2_Imag;
    float           *h_Quadrature_Filter_Response_3_Real, *h_Quadrature_Filter_Response_3_Imag;  
    float           *h_Phase_Differences, *h_Phase_Certainties, *h_Phase_Gradients;
    float           *h_Aligned_T1_Volume, *h_Interpolated_T1_Volume, *h_Registration_Parameters;
    float           *h_Downsampled_Volume;
    
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
    if(nlhs<10)
    {
        mexErrMsgTxt("Too few output arguments.");
    }
    if(nlhs>10)
    {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Input arguments */
    
    // The data
    h_T1_Volume_double =  (double*)mxGetData(prhs[0]);
    h_MNI_Volume_double =  (double*)mxGetData(prhs[1]);
    T1_VOXEL_SIZE_X = (float)mxGetScalar(prhs[2]);
    T1_VOXEL_SIZE_Y = (float)mxGetScalar(prhs[3]);
    T1_VOXEL_SIZE_Z = (float)mxGetScalar(prhs[4]);
    MNI_VOXEL_SIZE_X = (float)mxGetScalar(prhs[5]);
    MNI_VOXEL_SIZE_Y = (float)mxGetScalar(prhs[6]);
    MNI_VOXEL_SIZE_Z = (float)mxGetScalar(prhs[7]);    
    h_Quadrature_Filter_1_Real_double =  (double*)mxGetPr(prhs[8]);
    h_Quadrature_Filter_1_Imag_double =  (double*)mxGetPi(prhs[8]);
    h_Quadrature_Filter_2_Real_double =  (double*)mxGetPr(prhs[9]);
    h_Quadrature_Filter_2_Imag_double =  (double*)mxGetPi(prhs[9]);
    h_Quadrature_Filter_3_Real_double =  (double*)mxGetPr(prhs[10]);
    h_Quadrature_Filter_3_Imag_double =  (double*)mxGetPi(prhs[10]);
    NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION  = (int)mxGetScalar(prhs[11]);
    MM_T1_Z_CUT  = (int)mxGetScalar(prhs[12]);
    
    int NUMBER_OF_DIMENSIONS = mxGetNumberOfDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_T1 = mxGetDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_MNI = mxGetDimensions(prhs[1]);
    const int *ARRAY_DIMENSIONS_FILTER = mxGetDimensions(prhs[8]);
    
    int T1_DATA_H, T1_DATA_W, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS;
    NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS = 12;
    
    T1_DATA_H = ARRAY_DIMENSIONS_T1[0];
    T1_DATA_W = ARRAY_DIMENSIONS_T1[1];
    T1_DATA_D = ARRAY_DIMENSIONS_T1[2];
    
    MNI_DATA_H = ARRAY_DIMENSIONS_MNI[0];
    MNI_DATA_W = ARRAY_DIMENSIONS_MNI[1];
    MNI_DATA_D = ARRAY_DIMENSIONS_MNI[2];
    
    int NUMBER_OF_SCALES = 4;
    int DOWNSAMPLED_DATA_W = (int)round_((float)MNI_DATA_W/(float)NUMBER_OF_SCALES);
	int DOWNSAMPLED_DATA_H = (int)round_((float)MNI_DATA_H/(float)NUMBER_OF_SCALES);
	int DOWNSAMPLED_DATA_D = (int)round_((float)MNI_DATA_D/(float)NUMBER_OF_SCALES);
    
    IMAGE_REGISTRATION_FILTER_SIZE = ARRAY_DIMENSIONS_FILTER[0];
            
   	int T1_DATA_W_INTERPOLATED = (int)round_((float)T1_DATA_W * T1_VOXEL_SIZE_X / MNI_VOXEL_SIZE_X);
	int T1_DATA_H_INTERPOLATED = (int)round_((float)T1_DATA_H * T1_VOXEL_SIZE_Y / MNI_VOXEL_SIZE_Y);
	int T1_DATA_D_INTERPOLATED = (int)round_((float)T1_DATA_D * T1_VOXEL_SIZE_Z / MNI_VOXEL_SIZE_Z);

    int IMAGE_REGISTRATION_PARAMETERS_SIZE = NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float);
    int FILTER_SIZE = IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float);
    int T1_VOLUME_SIZE = T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float);
    int MNI_VOLUME_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float);
    int INTERPOLATED_T1_VOLUME_SIZE = T1_DATA_W_INTERPOLATED * T1_DATA_H_INTERPOLATED * T1_DATA_D_INTERPOLATED * sizeof(float);
    int DOWNSAMPLED_VOLUME_SIZE = DOWNSAMPLED_DATA_W * DOWNSAMPLED_DATA_H * DOWNSAMPLED_DATA_D * sizeof(float);
    
    mexPrintf("T1 size : %i x %i x %i \n",  T1_DATA_W, T1_DATA_H, T1_DATA_D);
    mexPrintf("T1 interpolated size : %i x %i x %i \n",  T1_DATA_W_INTERPOLATED, T1_DATA_H_INTERPOLATED, T1_DATA_D_INTERPOLATED);
    mexPrintf("Filter size : %i x %i x %i \n",  IMAGE_REGISTRATION_FILTER_SIZE,IMAGE_REGISTRATION_FILTER_SIZE,IMAGE_REGISTRATION_FILTER_SIZE);
    mexPrintf("Number of iterations : %i \n",  NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION);
    
    //-------------------------------------------------
    // Output to Matlab
    
    // Create pointer for volumes to Matlab        
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_ALIGNED_T1_VOLUME[3];
    ARRAY_DIMENSIONS_OUT_ALIGNED_T1_VOLUME[0] = MNI_DATA_H;
    ARRAY_DIMENSIONS_OUT_ALIGNED_T1_VOLUME[1] = MNI_DATA_W;
    ARRAY_DIMENSIONS_OUT_ALIGNED_T1_VOLUME[2] = MNI_DATA_D;
    
    plhs[0] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_ALIGNED_T1_VOLUME,mxDOUBLE_CLASS, mxREAL);
    h_Aligned_T1_Volume_double = mxGetPr(plhs[0]);          
    
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_INTERPOLATED_T1_VOLUME[3];
    ARRAY_DIMENSIONS_OUT_INTERPOLATED_T1_VOLUME[0] = T1_DATA_H_INTERPOLATED;
    ARRAY_DIMENSIONS_OUT_INTERPOLATED_T1_VOLUME[1] = T1_DATA_W_INTERPOLATED;
    ARRAY_DIMENSIONS_OUT_INTERPOLATED_T1_VOLUME[2] = T1_DATA_D_INTERPOLATED;
    
    plhs[1] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_INTERPOLATED_T1_VOLUME,mxDOUBLE_CLASS, mxREAL);
    h_Interpolated_T1_Volume_double = mxGetPr(plhs[1]);          
        
    NUMBER_OF_DIMENSIONS = 2;
    int ARRAY_DIMENSIONS_OUT_IMAGE_REGISTRATION_PARAMETERS[2];
    ARRAY_DIMENSIONS_OUT_IMAGE_REGISTRATION_PARAMETERS[0] = 1;
    ARRAY_DIMENSIONS_OUT_IMAGE_REGISTRATION_PARAMETERS[1] = NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS;    
    
    plhs[2] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_IMAGE_REGISTRATION_PARAMETERS,mxDOUBLE_CLASS, mxREAL);
    h_Registration_Parameters_double = mxGetPr(plhs[2]);          
    
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_FILTER_RESPONSE[3];
    ARRAY_DIMENSIONS_OUT_FILTER_RESPONSE[0] = MNI_DATA_H;
    ARRAY_DIMENSIONS_OUT_FILTER_RESPONSE[1] = MNI_DATA_W;    
    ARRAY_DIMENSIONS_OUT_FILTER_RESPONSE[2] = MNI_DATA_D;  
    
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
    
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_DOWNSAMPLED[3];
    ARRAY_DIMENSIONS_OUT_DOWNSAMPLED[0] = DOWNSAMPLED_DATA_H;
    ARRAY_DIMENSIONS_OUT_DOWNSAMPLED[1] = DOWNSAMPLED_DATA_W;    
    ARRAY_DIMENSIONS_OUT_DOWNSAMPLED[2] = DOWNSAMPLED_DATA_D;  
    
    plhs[9] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_DOWNSAMPLED,mxDOUBLE_CLASS, mxREAL);
    h_Downsampled_Volume_double = mxGetPr(plhs[9]);     
    
    // ------------------------------------------------
    
    // Allocate memory on the host
    h_T1_Volume                            = (float *)mxMalloc(T1_VOLUME_SIZE);
    h_MNI_Volume                           = (float *)mxMalloc(MNI_VOLUME_SIZE);
    h_Interpolated_T1_Volume               = (float *)mxMalloc(INTERPOLATED_T1_VOLUME_SIZE);
    h_Downsampled_Volume                   = (float *)mxMalloc(DOWNSAMPLED_VOLUME_SIZE);
    h_Quadrature_Filter_1_Real             = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_1_Imag             = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_2_Real             = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_2_Imag             = (float *)mxMalloc(FILTER_SIZE);    
    h_Quadrature_Filter_3_Real             = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_3_Imag             = (float *)mxMalloc(FILTER_SIZE);    
    h_Quadrature_Filter_Response_1_Real    = (float *)mxMalloc(MNI_VOLUME_SIZE);
    h_Quadrature_Filter_Response_1_Imag    = (float *)mxMalloc(MNI_VOLUME_SIZE);
    h_Quadrature_Filter_Response_2_Real    = (float *)mxMalloc(MNI_VOLUME_SIZE);
    h_Quadrature_Filter_Response_2_Imag    = (float *)mxMalloc(MNI_VOLUME_SIZE);
    h_Quadrature_Filter_Response_3_Real    = (float *)mxMalloc(MNI_VOLUME_SIZE);
    h_Quadrature_Filter_Response_3_Imag    = (float *)mxMalloc(MNI_VOLUME_SIZE);    
    h_Phase_Differences                    = (float *)mxMalloc(MNI_VOLUME_SIZE);    
    h_Phase_Certainties                    = (float *)mxMalloc(MNI_VOLUME_SIZE);  
    h_Phase_Gradients                      = (float *)mxMalloc(MNI_VOLUME_SIZE);  
    h_Aligned_T1_Volume                    = (float *)mxMalloc(MNI_VOLUME_SIZE);
    h_Registration_Parameters              = (float *)mxMalloc(IMAGE_REGISTRATION_PARAMETERS_SIZE);
    
    // Reorder and cast data
    pack_double2float_volume(h_T1_Volume, h_T1_Volume_double, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    pack_double2float_volume(h_MNI_Volume, h_MNI_Volume_double, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    pack_double2float_volume(h_Quadrature_Filter_1_Real, h_Quadrature_Filter_1_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_1_Imag, h_Quadrature_Filter_1_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_2_Real, h_Quadrature_Filter_2_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_2_Imag, h_Quadrature_Filter_2_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_3_Real, h_Quadrature_Filter_3_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_3_Imag, h_Quadrature_Filter_3_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);

    //------------------------
    
    BROCCOLI_LIB BROCCOLI;
    
    BROCCOLI.SetT1Width(T1_DATA_W);
    BROCCOLI.SetT1Height(T1_DATA_H);
    BROCCOLI.SetT1Depth(T1_DATA_D);
    BROCCOLI.SetT1VoxelSizeX(T1_VOXEL_SIZE_X);
    BROCCOLI.SetT1VoxelSizeY(T1_VOXEL_SIZE_Y);
    BROCCOLI.SetT1VoxelSizeZ(T1_VOXEL_SIZE_Z);   
    BROCCOLI.SetMNIWidth(MNI_DATA_W);
    BROCCOLI.SetMNIHeight(MNI_DATA_H);
    BROCCOLI.SetMNIDepth(MNI_DATA_D);    
    BROCCOLI.SetMNIVoxelSizeX(MNI_VOXEL_SIZE_X);
    BROCCOLI.SetMNIVoxelSizeY(MNI_VOXEL_SIZE_Y);
    BROCCOLI.SetMNIVoxelSizeZ(MNI_VOXEL_SIZE_Z);                
    BROCCOLI.SetInputT1Volume(h_T1_Volume);
    BROCCOLI.SetInputMNIVolume(h_MNI_Volume);
    BROCCOLI.SetImageRegistrationFilterSize(IMAGE_REGISTRATION_FILTER_SIZE);
    BROCCOLI.SetImageRegistrationFilters(h_Quadrature_Filter_1_Real, h_Quadrature_Filter_1_Imag, h_Quadrature_Filter_2_Real, h_Quadrature_Filter_2_Imag, h_Quadrature_Filter_3_Real, h_Quadrature_Filter_3_Imag);
    BROCCOLI.SetNumberOfIterationsForImageRegistration(NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION);
    BROCCOLI.SetMMT1ZCUT(MM_T1_Z_CUT);   
    BROCCOLI.SetOutputAlignedT1Volume(h_Aligned_T1_Volume);
    BROCCOLI.SetOutputInterpolatedT1Volume(h_Interpolated_T1_Volume);
    BROCCOLI.SetOutputDownsampledVolume(h_Downsampled_Volume);
    BROCCOLI.SetOutputT1MNIRegistrationParameters(h_Registration_Parameters);
    BROCCOLI.SetOutputQuadratureFilterResponses(h_Quadrature_Filter_Response_1_Real, h_Quadrature_Filter_Response_1_Imag, h_Quadrature_Filter_Response_2_Real, h_Quadrature_Filter_Response_2_Imag, h_Quadrature_Filter_Response_3_Real, h_Quadrature_Filter_Response_3_Imag);
    BROCCOLI.SetOutputPhaseDifferences(h_Phase_Differences);
    BROCCOLI.SetOutputPhaseCertainties(h_Phase_Certainties);
    BROCCOLI.SetOutputPhaseGradients(h_Phase_Gradients);
    
    mexPrintf("Device info \n \n %s \n", BROCCOLI.GetOpenCLDeviceInfoChar());
    mexPrintf("Build info \n \n %s \n", BROCCOLI.GetOpenCLBuildInfoChar());
            
    //BROCCOLI.ChangeT1VolumeResolutionAndSizeWrapper();
    BROCCOLI.PerformRegistrationT1MNIWrapper();
    
    int error = BROCCOLI.GetOpenCLError();
    mexPrintf("Error is %d \n",error);
    
    int createKernelError = BROCCOLI.GetOpenCLCreateKernelError();
    mexPrintf("Create kernel error is %d \n",createKernelError);
    
    double convolution_time = BROCCOLI.GetProcessingTimeConvolution();
    mexPrintf("Convolution time is %f ms \n",convolution_time/1000000.0);
    
    int* createBufferErrors = BROCCOLI.GetOpenCLCreateBufferErrors();
    for (int i = 0; i < 30; i++)
    {
        mexPrintf("Create buffer error %i is %d \n",i,createBufferErrors[i]);
    }
        
    int* runKernelErrors = BROCCOLI.GetOpenCLRunKernelErrors();
    for (int i = 0; i < 13; i++)
    {
        mexPrintf("Run kernel error %i is %d \n",i,runKernelErrors[i]);
    } 
    
    unpack_float2double_volume(h_Aligned_T1_Volume_double, h_Aligned_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    unpack_float2double_volume(h_Interpolated_T1_Volume_double, h_Interpolated_T1_Volume, T1_DATA_W_INTERPOLATED, T1_DATA_H_INTERPOLATED, T1_DATA_D_INTERPOLATED);
    unpack_float2double(h_Registration_Parameters_double, h_Registration_Parameters, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS);
    unpack_float2double_volume(h_Quadrature_Filter_Response_1_Real_double, h_Quadrature_Filter_Response_1_Real, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    unpack_float2double_volume(h_Quadrature_Filter_Response_1_Imag_double, h_Quadrature_Filter_Response_1_Imag, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    unpack_float2double_volume(h_Quadrature_Filter_Response_2_Real_double, h_Quadrature_Filter_Response_2_Real, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    unpack_float2double_volume(h_Quadrature_Filter_Response_2_Imag_double, h_Quadrature_Filter_Response_2_Imag, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    unpack_float2double_volume(h_Quadrature_Filter_Response_3_Real_double, h_Quadrature_Filter_Response_3_Real, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    unpack_float2double_volume(h_Quadrature_Filter_Response_3_Imag_double, h_Quadrature_Filter_Response_3_Imag, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    unpack_float2double_volume(h_Phase_Differences_double, h_Phase_Differences, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    unpack_float2double_volume(h_Phase_Certainties_double, h_Phase_Certainties, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    unpack_float2double_volume(h_Phase_Gradients_double, h_Phase_Gradients, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);    
    unpack_float2double_volume(h_Downsampled_Volume_double, h_Downsampled_Volume, DOWNSAMPLED_DATA_W, DOWNSAMPLED_DATA_H, DOWNSAMPLED_DATA_D);    
    
    // Free all the allocated memory on the host
    mxFree(h_T1_Volume);
    mxFree(h_Interpolated_T1_Volume);
    mxFree(h_Downsampled_Volume);
    mxFree(h_Quadrature_Filter_1_Real);
    mxFree(h_Quadrature_Filter_1_Imag);
    mxFree(h_Quadrature_Filter_2_Real);
    mxFree(h_Quadrature_Filter_2_Imag);
    mxFree(h_Quadrature_Filter_3_Real);
    mxFree(h_Quadrature_Filter_3_Imag);
    mxFree(h_Quadrature_Filter_Response_1_Real);
    mxFree(h_Quadrature_Filter_Response_1_Imag);
    mxFree(h_Quadrature_Filter_Response_2_Real);
    mxFree(h_Quadrature_Filter_Response_2_Imag);
    mxFree(h_Quadrature_Filter_Response_3_Real);
    mxFree(h_Quadrature_Filter_Response_3_Imag);
    mxFree(h_Phase_Differences);
    mxFree(h_Phase_Certainties);
    mxFree(h_Phase_Gradients);
    mxFree(h_Aligned_T1_Volume); 
    mxFree(h_Registration_Parameters);
    
    //mexAtExit(cleanUp);
    
    return;
}


