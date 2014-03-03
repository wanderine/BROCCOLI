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


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //-----------------------
    // Input pointers and other stuff
    
    double		    *h_T1_Volume_double, *h_MNI_Volume_double, *h_MNI_Brain_Volume_double, *h_MNI_Brain_Mask_double;
    double          *h_Quadrature_Filter_1_Parametric_Registration_Real_double, *h_Quadrature_Filter_2_Parametric_Registration_Real_double, *h_Quadrature_Filter_3_Parametric_Registration_Real_double, *h_Quadrature_Filter_1_Parametric_Registration_Imag_double, *h_Quadrature_Filter_2_Parametric_Registration_Imag_double, *h_Quadrature_Filter_3_Parametric_Registration_Imag_double;
    double          *h_Quadrature_Filter_1_NonParametric_Registration_Real_double, *h_Quadrature_Filter_2_NonParametric_Registration_Real_double, *h_Quadrature_Filter_3_NonParametric_Registration_Real_double, *h_Quadrature_Filter_1_NonParametric_Registration_Imag_double, *h_Quadrature_Filter_2_NonParametric_Registration_Imag_double, *h_Quadrature_Filter_3_NonParametric_Registration_Imag_double;
    double          *h_Quadrature_Filter_4_NonParametric_Registration_Real_double, *h_Quadrature_Filter_5_NonParametric_Registration_Real_double, *h_Quadrature_Filter_6_NonParametric_Registration_Real_double, *h_Quadrature_Filter_4_NonParametric_Registration_Imag_double, *h_Quadrature_Filter_5_NonParametric_Registration_Imag_double, *h_Quadrature_Filter_6_NonParametric_Registration_Imag_double;
    float           *h_T1_Volume, *h_MNI_Volume, *h_MNI_Brain_Volume, *h_MNI_Brain_Mask;
    float           *h_Quadrature_Filter_1_Parametric_Registration_Real, *h_Quadrature_Filter_2_Parametric_Registration_Real, *h_Quadrature_Filter_3_Parametric_Registration_Real, *h_Quadrature_Filter_1_Parametric_Registration_Imag, *h_Quadrature_Filter_2_Parametric_Registration_Imag, *h_Quadrature_Filter_3_Parametric_Registration_Imag;
    float           *h_Quadrature_Filter_1_NonParametric_Registration_Real, *h_Quadrature_Filter_2_NonParametric_Registration_Real, *h_Quadrature_Filter_3_NonParametric_Registration_Real, *h_Quadrature_Filter_1_NonParametric_Registration_Imag, *h_Quadrature_Filter_2_NonParametric_Registration_Imag, *h_Quadrature_Filter_3_NonParametric_Registration_Imag;
    float           *h_Quadrature_Filter_4_NonParametric_Registration_Real, *h_Quadrature_Filter_5_NonParametric_Registration_Real, *h_Quadrature_Filter_6_NonParametric_Registration_Real, *h_Quadrature_Filter_4_NonParametric_Registration_Imag, *h_Quadrature_Filter_5_NonParametric_Registration_Imag, *h_Quadrature_Filter_6_NonParametric_Registration_Imag;
    int             IMAGE_REGISTRATION_FILTER_SIZE, NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION, NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION, COARSEST_SCALE, MM_T1_Z_CUT;
    int             OPENCL_PLATFORM, OPENCL_DEVICE;
    
    cl_float2          *h_Quadrature_Filter_1_Parametric_Registration, *h_Quadrature_Filter_2_Parametric_Registration, *h_Quadrature_Filter_3_Parametric_Registration;
    cl_float2          *h_Quadrature_Filter_1_NonParametric_Registration, *h_Quadrature_Filter_2_NonParametric_Registration, *h_Quadrature_Filter_3_NonParametric_Registration, *h_Quadrature_Filter_4_NonParametric_Registration, *h_Quadrature_Filter_5_NonParametric_Registration, *h_Quadrature_Filter_6_NonParametric_Registration;
    cl_float2          *h_Quadrature_Filter_Response_1, *h_Quadrature_Filter_Response_2, *h_Quadrature_Filter_Response_3, *h_Quadrature_Filter_Response_4, *h_Quadrature_Filter_Response_5, *h_Quadrature_Filter_Response_6;
    
    double          *h_Projection_Tensor_1_double, *h_Projection_Tensor_2_double, *h_Projection_Tensor_3_double, *h_Projection_Tensor_4_double, *h_Projection_Tensor_5_double, *h_Projection_Tensor_6_double;
    float           *h_Projection_Tensor_1, *h_Projection_Tensor_2, *h_Projection_Tensor_3, *h_Projection_Tensor_4, *h_Projection_Tensor_5, *h_Projection_Tensor_6;
    
    
    double          *h_Filter_Directions_X_double, *h_Filter_Directions_Y_double, *h_Filter_Directions_Z_double;
    float           *h_Filter_Directions_X, *h_Filter_Directions_Y, *h_Filter_Directions_Z;
    
    int             T1_DATA_H, T1_DATA_W, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS;
    
    float           T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z;
    float           MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z;
    
    //-----------------------
    // Output pointers        
    
    double     		*h_Aligned_T1_Volume_double, *h_Aligned_T1_Volume_NonParametric_double, *h_Interpolated_T1_Volume_double, *h_Skullstripped_T1_Volume_double, *h_Registration_Parameters_double;    
    
    double          *h_Displacement_Field_X_double, *h_Displacement_Field_Y_double, *h_Displacement_Field_Z_double;
    float           *h_Displacement_Field_X, *h_Displacement_Field_Y, *h_Displacement_Field_Z;

    float           *h_Aligned_T1_Volume, *h_Aligned_T1_Volume_NonParametric, *h_Interpolated_T1_Volume, *h_Registration_Parameters;
    
    //---------------------
    
    /* Check the number of input and output arguments. */
    if(nrhs<32)
    {
        mexErrMsgTxt("Too few input arguments.");
    }
    if(nrhs>32)
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
    h_T1_Volume_double =  (double*)mxGetData(prhs[0]);
    h_MNI_Volume_double =  (double*)mxGetData(prhs[1]);
    T1_VOXEL_SIZE_X = (float)mxGetScalar(prhs[2]);
    T1_VOXEL_SIZE_Y = (float)mxGetScalar(prhs[3]);
    T1_VOXEL_SIZE_Z = (float)mxGetScalar(prhs[4]);
    MNI_VOXEL_SIZE_X = (float)mxGetScalar(prhs[5]);
    MNI_VOXEL_SIZE_Y = (float)mxGetScalar(prhs[6]);
    MNI_VOXEL_SIZE_Z = (float)mxGetScalar(prhs[7]);    
    h_Quadrature_Filter_1_Parametric_Registration_Real_double =  (double*)mxGetPr(prhs[8]);
    h_Quadrature_Filter_1_Parametric_Registration_Imag_double =  (double*)mxGetPi(prhs[8]);
    h_Quadrature_Filter_2_Parametric_Registration_Real_double =  (double*)mxGetPr(prhs[9]);
    h_Quadrature_Filter_2_Parametric_Registration_Imag_double =  (double*)mxGetPi(prhs[9]);
    h_Quadrature_Filter_3_Parametric_Registration_Real_double =  (double*)mxGetPr(prhs[10]);
    h_Quadrature_Filter_3_Parametric_Registration_Imag_double =  (double*)mxGetPi(prhs[10]);
    h_Quadrature_Filter_1_NonParametric_Registration_Real_double =  (double*)mxGetPr(prhs[11]);
    h_Quadrature_Filter_1_NonParametric_Registration_Imag_double =  (double*)mxGetPi(prhs[11]);
    h_Quadrature_Filter_2_NonParametric_Registration_Real_double =  (double*)mxGetPr(prhs[12]);
    h_Quadrature_Filter_2_NonParametric_Registration_Imag_double =  (double*)mxGetPi(prhs[12]);
    h_Quadrature_Filter_3_NonParametric_Registration_Real_double =  (double*)mxGetPr(prhs[13]);
    h_Quadrature_Filter_3_NonParametric_Registration_Imag_double =  (double*)mxGetPi(prhs[13]);
    h_Quadrature_Filter_4_NonParametric_Registration_Real_double =  (double*)mxGetPr(prhs[14]);
    h_Quadrature_Filter_4_NonParametric_Registration_Imag_double =  (double*)mxGetPi(prhs[14]);
    h_Quadrature_Filter_5_NonParametric_Registration_Real_double =  (double*)mxGetPr(prhs[15]);
    h_Quadrature_Filter_5_NonParametric_Registration_Imag_double =  (double*)mxGetPi(prhs[15]);
    h_Quadrature_Filter_6_NonParametric_Registration_Real_double =  (double*)mxGetPr(prhs[16]);
    h_Quadrature_Filter_6_NonParametric_Registration_Imag_double =  (double*)mxGetPi(prhs[16]);     
    h_Projection_Tensor_1_double = (double*)mxGetPr(prhs[17]);
    h_Projection_Tensor_2_double = (double*)mxGetPr(prhs[18]);
    h_Projection_Tensor_3_double = (double*)mxGetPr(prhs[19]);
    h_Projection_Tensor_4_double = (double*)mxGetPr(prhs[20]);
    h_Projection_Tensor_5_double = (double*)mxGetPr(prhs[21]);
    h_Projection_Tensor_6_double = (double*)mxGetPr(prhs[22]);
    h_Filter_Directions_X_double = (double*)mxGetPr(prhs[23]);
    h_Filter_Directions_Y_double = (double*)mxGetPr(prhs[24]);
    h_Filter_Directions_Z_double = (double*)mxGetPr(prhs[25]);
    NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION  = (int)mxGetScalar(prhs[26]);
    NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION  = (int)mxGetScalar(prhs[27]);
    COARSEST_SCALE  = (int)mxGetScalar(prhs[28]);
    MM_T1_Z_CUT  = (int)mxGetScalar(prhs[29]);
    OPENCL_PLATFORM  = (int)mxGetScalar(prhs[30]);
    OPENCL_DEVICE  = (int)mxGetScalar(prhs[31]);
    
    int NUMBER_OF_DIMENSIONS = mxGetNumberOfDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_T1 = mxGetDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_MNI = mxGetDimensions(prhs[1]);
    const int *ARRAY_DIMENSIONS_FILTER = mxGetDimensions(prhs[8]);
    
    
    NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS = 12;
    
    T1_DATA_H = ARRAY_DIMENSIONS_T1[0];
    T1_DATA_W = ARRAY_DIMENSIONS_T1[1];
    T1_DATA_D = ARRAY_DIMENSIONS_T1[2];
    
    MNI_DATA_H = ARRAY_DIMENSIONS_MNI[0];
    MNI_DATA_W = ARRAY_DIMENSIONS_MNI[1];
    MNI_DATA_D = ARRAY_DIMENSIONS_MNI[2];
    
    int DOWNSAMPLED_DATA_W = (int)round_((float)MNI_DATA_W/(float)COARSEST_SCALE);
	int DOWNSAMPLED_DATA_H = (int)round_((float)MNI_DATA_H/(float)COARSEST_SCALE);
	int DOWNSAMPLED_DATA_D = (int)round_((float)MNI_DATA_D/(float)COARSEST_SCALE);
    
    IMAGE_REGISTRATION_FILTER_SIZE = ARRAY_DIMENSIONS_FILTER[0];
            
   	int T1_DATA_W_INTERPOLATED = (int)round_((float)T1_DATA_W * T1_VOXEL_SIZE_X / MNI_VOXEL_SIZE_X);
	int T1_DATA_H_INTERPOLATED = (int)round_((float)T1_DATA_H * T1_VOXEL_SIZE_Y / MNI_VOXEL_SIZE_Y);
	int T1_DATA_D_INTERPOLATED = (int)round_((float)T1_DATA_D * T1_VOXEL_SIZE_Z / MNI_VOXEL_SIZE_Z);

    int IMAGE_REGISTRATION_PARAMETERS_SIZE = NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS * sizeof(float);
    int FILTER_SIZE = IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float);
    int FILTER_SIZE2 = IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(cl_float2);
    int T1_VOLUME_SIZE = T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float);
    int MNI_VOLUME_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float);
    int MNI2_VOLUME_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(cl_float2);
    int INTERPOLATED_T1_VOLUME_SIZE = T1_DATA_W_INTERPOLATED * T1_DATA_H_INTERPOLATED * T1_DATA_D_INTERPOLATED * sizeof(float);
    int DOWNSAMPLED_VOLUME_SIZE = DOWNSAMPLED_DATA_W * DOWNSAMPLED_DATA_H * DOWNSAMPLED_DATA_D * sizeof(float);
    int PROJECTION_TENSOR_SIZE = NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION * sizeof(float);
    int FILTER_DIRECTIONS_SIZE = NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION * sizeof(float);
    
    mexPrintf("Source volume size : %i x %i x %i \n",  T1_DATA_W, T1_DATA_H, T1_DATA_D);
    //mexPrintf("T1 interpolated size : %i x %i x %i \n",  T1_DATA_W_INTERPOLATED, T1_DATA_H_INTERPOLATED, T1_DATA_D_INTERPOLATED);
    mexPrintf("Reference volume size : %i x %i x %i \n",  MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    //mexPrintf("Filter size : %i x %i x %i \n",  IMAGE_REGISTRATION_FILTER_SIZE,IMAGE_REGISTRATION_FILTER_SIZE,IMAGE_REGISTRATION_FILTER_SIZE);
    
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
    
    plhs[1] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_ALIGNED_T1_VOLUME,mxDOUBLE_CLASS, mxREAL);
    h_Aligned_T1_Volume_NonParametric_double = mxGetPr(plhs[1]);       
              
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_INTERPOLATED_T1_VOLUME[3];
    ARRAY_DIMENSIONS_OUT_INTERPOLATED_T1_VOLUME[0] = T1_DATA_H_INTERPOLATED;
    ARRAY_DIMENSIONS_OUT_INTERPOLATED_T1_VOLUME[1] = T1_DATA_W_INTERPOLATED;
    ARRAY_DIMENSIONS_OUT_INTERPOLATED_T1_VOLUME[2] = T1_DATA_D_INTERPOLATED;
    
    plhs[2] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_ALIGNED_T1_VOLUME,mxDOUBLE_CLASS, mxREAL);
    h_Interpolated_T1_Volume_double = mxGetPr(plhs[2]);          
        
    NUMBER_OF_DIMENSIONS = 2;
    int ARRAY_DIMENSIONS_OUT_IMAGE_REGISTRATION_PARAMETERS[2];
    ARRAY_DIMENSIONS_OUT_IMAGE_REGISTRATION_PARAMETERS[0] = 1;
    ARRAY_DIMENSIONS_OUT_IMAGE_REGISTRATION_PARAMETERS[1] = NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS;    
    
    plhs[3] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_IMAGE_REGISTRATION_PARAMETERS,mxDOUBLE_CLASS, mxREAL);
    h_Registration_Parameters_double = mxGetPr(plhs[3]);          
    
    NUMBER_OF_DIMENSIONS = 3;
    int ARRAY_DIMENSIONS_OUT_DISPLACEMENT[3];
    ARRAY_DIMENSIONS_OUT_DISPLACEMENT[0] = MNI_DATA_H;
    ARRAY_DIMENSIONS_OUT_DISPLACEMENT[1] = MNI_DATA_W;    
    ARRAY_DIMENSIONS_OUT_DISPLACEMENT[2] = MNI_DATA_D;  
    
    plhs[4] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_DISPLACEMENT,mxDOUBLE_CLASS, mxREAL);
    h_Displacement_Field_X_double = mxGetPr(plhs[4]);          
   
    plhs[5] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_DISPLACEMENT,mxDOUBLE_CLASS, mxREAL);
    h_Displacement_Field_Y_double = mxGetPr(plhs[5]);          
    
    plhs[6] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_DISPLACEMENT,mxDOUBLE_CLASS, mxREAL);
    h_Displacement_Field_Z_double = mxGetPr(plhs[6]);                  
    
    // ------------------------------------------------
    
    // Allocate memory on the host
    h_T1_Volume                                         = (float *)mxMalloc(T1_VOLUME_SIZE);
    h_MNI_Volume                                        = (float *)mxMalloc(MNI_VOLUME_SIZE);
    h_Interpolated_T1_Volume                            = (float *)mxMalloc(MNI_VOLUME_SIZE);
           
    h_Quadrature_Filter_1_Parametric_Registration_Real  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_2_Parametric_Registration_Real  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_3_Parametric_Registration_Real  = (float *)mxMalloc(FILTER_SIZE);    
    h_Quadrature_Filter_1_Parametric_Registration_Imag  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_2_Parametric_Registration_Imag  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_3_Parametric_Registration_Imag  = (float *)mxMalloc(FILTER_SIZE);    
    
    h_Quadrature_Filter_1_NonParametric_Registration_Real  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_2_NonParametric_Registration_Real  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_3_NonParametric_Registration_Real  = (float *)mxMalloc(FILTER_SIZE);    
    h_Quadrature_Filter_4_NonParametric_Registration_Real  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_5_NonParametric_Registration_Real  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_6_NonParametric_Registration_Real  = (float *)mxMalloc(FILTER_SIZE);        
    h_Quadrature_Filter_1_NonParametric_Registration_Imag  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_2_NonParametric_Registration_Imag  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_3_NonParametric_Registration_Imag  = (float *)mxMalloc(FILTER_SIZE);    
    h_Quadrature_Filter_4_NonParametric_Registration_Imag  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_5_NonParametric_Registration_Imag  = (float *)mxMalloc(FILTER_SIZE);
    h_Quadrature_Filter_6_NonParametric_Registration_Imag  = (float *)mxMalloc(FILTER_SIZE);            
               
    h_Displacement_Field_X                              = (float*)mxMalloc(MNI_VOLUME_SIZE);
    h_Displacement_Field_Y                              = (float*)mxMalloc(MNI_VOLUME_SIZE);
    h_Displacement_Field_Z                              = (float*)mxMalloc(MNI_VOLUME_SIZE);
    
    h_Projection_Tensor_1                               = (float*)mxMalloc(PROJECTION_TENSOR_SIZE);
    h_Projection_Tensor_2                               = (float*)mxMalloc(PROJECTION_TENSOR_SIZE);
    h_Projection_Tensor_3                               = (float*)mxMalloc(PROJECTION_TENSOR_SIZE);
    h_Projection_Tensor_4                               = (float*)mxMalloc(PROJECTION_TENSOR_SIZE);
    h_Projection_Tensor_5                               = (float*)mxMalloc(PROJECTION_TENSOR_SIZE);
    h_Projection_Tensor_6                               = (float*)mxMalloc(PROJECTION_TENSOR_SIZE);
    
    h_Filter_Directions_X                               = (float*)mxMalloc(FILTER_DIRECTIONS_SIZE);
    h_Filter_Directions_Y                               = (float*)mxMalloc(FILTER_DIRECTIONS_SIZE);
    h_Filter_Directions_Z                               = (float*)mxMalloc(FILTER_DIRECTIONS_SIZE);
    
    h_Aligned_T1_Volume                                 = (float *)mxMalloc(MNI_VOLUME_SIZE);
    h_Aligned_T1_Volume_NonParametric                   = (float *)mxMalloc(MNI_VOLUME_SIZE);
    h_Registration_Parameters                           = (float *)mxMalloc(IMAGE_REGISTRATION_PARAMETERS_SIZE);
    
    
    // Pack data (reorder from y,x,z to x,y,z and cast from double to float)
    pack_double2float_volume(h_T1_Volume, h_T1_Volume_double, T1_DATA_W, T1_DATA_H, T1_DATA_D);
    pack_double2float_volume(h_MNI_Volume, h_MNI_Volume_double, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    
    pack_double2float_volume(h_Quadrature_Filter_1_Parametric_Registration_Real, h_Quadrature_Filter_1_Parametric_Registration_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_1_Parametric_Registration_Imag, h_Quadrature_Filter_1_Parametric_Registration_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_2_Parametric_Registration_Real, h_Quadrature_Filter_2_Parametric_Registration_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_2_Parametric_Registration_Imag, h_Quadrature_Filter_2_Parametric_Registration_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_3_Parametric_Registration_Real, h_Quadrature_Filter_3_Parametric_Registration_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_3_Parametric_Registration_Imag, h_Quadrature_Filter_3_Parametric_Registration_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    
    pack_double2float_volume(h_Quadrature_Filter_1_NonParametric_Registration_Real, h_Quadrature_Filter_1_NonParametric_Registration_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_1_NonParametric_Registration_Imag, h_Quadrature_Filter_1_NonParametric_Registration_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_2_NonParametric_Registration_Real, h_Quadrature_Filter_2_NonParametric_Registration_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_2_NonParametric_Registration_Imag, h_Quadrature_Filter_2_NonParametric_Registration_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_3_NonParametric_Registration_Real, h_Quadrature_Filter_3_NonParametric_Registration_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_3_NonParametric_Registration_Imag, h_Quadrature_Filter_3_NonParametric_Registration_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_4_NonParametric_Registration_Real, h_Quadrature_Filter_4_NonParametric_Registration_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_4_NonParametric_Registration_Imag, h_Quadrature_Filter_4_NonParametric_Registration_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_5_NonParametric_Registration_Real, h_Quadrature_Filter_5_NonParametric_Registration_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_5_NonParametric_Registration_Imag, h_Quadrature_Filter_5_NonParametric_Registration_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_6_NonParametric_Registration_Real, h_Quadrature_Filter_6_NonParametric_Registration_Real_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    pack_double2float_volume(h_Quadrature_Filter_6_NonParametric_Registration_Imag, h_Quadrature_Filter_6_NonParametric_Registration_Imag_double, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE, IMAGE_REGISTRATION_FILTER_SIZE);
    
    pack_double2float(h_Projection_Tensor_1, h_Projection_Tensor_1_double, NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION);
    pack_double2float(h_Projection_Tensor_2, h_Projection_Tensor_2_double, NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION);
    pack_double2float(h_Projection_Tensor_3, h_Projection_Tensor_3_double, NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION);
    pack_double2float(h_Projection_Tensor_4, h_Projection_Tensor_4_double, NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION);
    pack_double2float(h_Projection_Tensor_5, h_Projection_Tensor_5_double, NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION);
    pack_double2float(h_Projection_Tensor_6, h_Projection_Tensor_6_double, NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION);
    
    pack_double2float(h_Filter_Directions_X, h_Filter_Directions_X_double, NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION);
    pack_double2float(h_Filter_Directions_Y, h_Filter_Directions_Y_double, NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION);
    pack_double2float(h_Filter_Directions_Z, h_Filter_Directions_Z_double, NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION);
        
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
        BROCCOLI.SetInputT1Volume(h_T1_Volume);
        BROCCOLI.SetInputMNIBrainVolume(h_MNI_Volume);
        
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
        
        BROCCOLI.SetInterpolationMode(LINEAR);
        
        BROCCOLI.SetNumberOfIterationsForParametricImageRegistration(NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION);
        BROCCOLI.SetNumberOfIterationsForNonParametricImageRegistration(NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION);
        BROCCOLI.SetImageRegistrationFilterSize(IMAGE_REGISTRATION_FILTER_SIZE);    
        
        BROCCOLI.SetParametricImageRegistrationFilters(h_Quadrature_Filter_1_Parametric_Registration_Real, h_Quadrature_Filter_1_Parametric_Registration_Imag, h_Quadrature_Filter_2_Parametric_Registration_Real, h_Quadrature_Filter_2_Parametric_Registration_Imag, h_Quadrature_Filter_3_Parametric_Registration_Real, h_Quadrature_Filter_3_Parametric_Registration_Imag);
        BROCCOLI.SetNonParametricImageRegistrationFilters(h_Quadrature_Filter_1_NonParametric_Registration_Real, h_Quadrature_Filter_1_NonParametric_Registration_Imag, h_Quadrature_Filter_2_NonParametric_Registration_Real, h_Quadrature_Filter_2_NonParametric_Registration_Imag, h_Quadrature_Filter_3_NonParametric_Registration_Real, h_Quadrature_Filter_3_NonParametric_Registration_Imag, h_Quadrature_Filter_4_NonParametric_Registration_Real, h_Quadrature_Filter_4_NonParametric_Registration_Imag, h_Quadrature_Filter_5_NonParametric_Registration_Real, h_Quadrature_Filter_5_NonParametric_Registration_Imag, h_Quadrature_Filter_6_NonParametric_Registration_Real, h_Quadrature_Filter_6_NonParametric_Registration_Imag);    
        BROCCOLI.SetProjectionTensorMatrixFirstFilter(h_Projection_Tensor_1[0],h_Projection_Tensor_1[1],h_Projection_Tensor_1[2],h_Projection_Tensor_1[3],h_Projection_Tensor_1[4],h_Projection_Tensor_1[5]);
        BROCCOLI.SetProjectionTensorMatrixSecondFilter(h_Projection_Tensor_2[0],h_Projection_Tensor_2[1],h_Projection_Tensor_2[2],h_Projection_Tensor_2[3],h_Projection_Tensor_2[4],h_Projection_Tensor_2[5]);
        BROCCOLI.SetProjectionTensorMatrixThirdFilter(h_Projection_Tensor_3[0],h_Projection_Tensor_3[1],h_Projection_Tensor_3[2],h_Projection_Tensor_3[3],h_Projection_Tensor_3[4],h_Projection_Tensor_3[5]);
        BROCCOLI.SetProjectionTensorMatrixFourthFilter(h_Projection_Tensor_4[0],h_Projection_Tensor_4[1],h_Projection_Tensor_4[2],h_Projection_Tensor_4[3],h_Projection_Tensor_4[4],h_Projection_Tensor_4[5]);
        BROCCOLI.SetProjectionTensorMatrixFifthFilter(h_Projection_Tensor_5[0],h_Projection_Tensor_5[1],h_Projection_Tensor_5[2],h_Projection_Tensor_5[3],h_Projection_Tensor_5[4],h_Projection_Tensor_5[5]);
        BROCCOLI.SetProjectionTensorMatrixSixthFilter(h_Projection_Tensor_6[0],h_Projection_Tensor_6[1],h_Projection_Tensor_6[2],h_Projection_Tensor_6[3],h_Projection_Tensor_6[4],h_Projection_Tensor_6[5]);
        BROCCOLI.SetFilterDirections(h_Filter_Directions_X, h_Filter_Directions_Y, h_Filter_Directions_Z);
        BROCCOLI.SetCoarsestScaleT1MNI(COARSEST_SCALE);
        BROCCOLI.SetMMT1ZCUT(MM_T1_Z_CUT);   
        BROCCOLI.SetOutputAlignedT1Volume(h_Aligned_T1_Volume);
        BROCCOLI.SetOutputAlignedT1VolumeNonParametric(h_Aligned_T1_Volume_NonParametric);
        BROCCOLI.SetOutputInterpolatedT1Volume(h_Interpolated_T1_Volume);
        BROCCOLI.SetOutputT1MNIRegistrationParameters(h_Registration_Parameters);
        BROCCOLI.SetOutputQuadratureFilterResponses(h_Quadrature_Filter_Response_1, h_Quadrature_Filter_Response_2, h_Quadrature_Filter_Response_3, h_Quadrature_Filter_Response_4, h_Quadrature_Filter_Response_5, h_Quadrature_Filter_Response_6);
        BROCCOLI.SetOutputDisplacementField(h_Displacement_Field_X,h_Displacement_Field_Y,h_Displacement_Field_Z);
                            
        BROCCOLI.PerformRegistrationTwoVolumesWrapper();     
        
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
        
    unpack_float2double_volume(h_Aligned_T1_Volume_double, h_Aligned_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    unpack_float2double_volume(h_Aligned_T1_Volume_NonParametric_double, h_Aligned_T1_Volume_NonParametric, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    unpack_float2double_volume(h_Interpolated_T1_Volume_double, h_Interpolated_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    unpack_float2double(h_Registration_Parameters_double, h_Registration_Parameters, NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS);
            
    unpack_float2double_volume(h_Displacement_Field_X_double, h_Displacement_Field_X, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    unpack_float2double_volume(h_Displacement_Field_Y_double, h_Displacement_Field_Y, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    unpack_float2double_volume(h_Displacement_Field_Z_double, h_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
        
    // Free all the allocated memory on the host
    mxFree(h_T1_Volume);
    mxFree(h_MNI_Volume);
    mxFree(h_Interpolated_T1_Volume);
        
    mxFree(h_Quadrature_Filter_1_Parametric_Registration_Real);
    mxFree(h_Quadrature_Filter_2_Parametric_Registration_Real);
    mxFree(h_Quadrature_Filter_3_Parametric_Registration_Real);
    mxFree(h_Quadrature_Filter_1_Parametric_Registration_Imag);
    mxFree(h_Quadrature_Filter_2_Parametric_Registration_Imag);
    mxFree(h_Quadrature_Filter_3_Parametric_Registration_Imag);
    
    mxFree(h_Quadrature_Filter_1_NonParametric_Registration_Real);
    mxFree(h_Quadrature_Filter_2_NonParametric_Registration_Real);
    mxFree(h_Quadrature_Filter_3_NonParametric_Registration_Real);
    mxFree(h_Quadrature_Filter_4_NonParametric_Registration_Real);
    mxFree(h_Quadrature_Filter_5_NonParametric_Registration_Real);
    mxFree(h_Quadrature_Filter_6_NonParametric_Registration_Real);
    
    mxFree(h_Quadrature_Filter_1_NonParametric_Registration_Imag);
    mxFree(h_Quadrature_Filter_2_NonParametric_Registration_Imag);
    mxFree(h_Quadrature_Filter_3_NonParametric_Registration_Imag);
    mxFree(h_Quadrature_Filter_4_NonParametric_Registration_Imag);
    mxFree(h_Quadrature_Filter_5_NonParametric_Registration_Imag);
    mxFree(h_Quadrature_Filter_6_NonParametric_Registration_Imag);
              
    mxFree(h_Displacement_Field_X);
    mxFree(h_Displacement_Field_Y);
    mxFree(h_Displacement_Field_Z);
    
    mxFree(h_Projection_Tensor_1);
    mxFree(h_Projection_Tensor_2);
    mxFree(h_Projection_Tensor_3);
    mxFree(h_Projection_Tensor_4);
    mxFree(h_Projection_Tensor_5);
    mxFree(h_Projection_Tensor_6);
    
    mxFree(h_Filter_Directions_X);
    mxFree(h_Filter_Directions_Y);
    mxFree(h_Filter_Directions_Z);
    
    mxFree(h_Aligned_T1_Volume); 
    mxFree(h_Aligned_T1_Volume_NonParametric); 
    mxFree(h_Registration_Parameters);
    
    return;
}


