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
    
    double		    *h_fMRI_Volumes_double, *h_Quadrature_Filter_1_Real_double, *h_Quadrature_Filter_2_Real_double, *h_Quadrature_Filter_3_Real_double, *h_Quadrature_Filter_1_Imag_double, *h_Quadrature_Filter_2_Imag_double, *h_Quadrature_Filter_3_Imag_double;
    float           *h_fMRI_Volumes, *h_Quadrature_Filter_1_Real, *h_Quadrature_Filter_2_Real, *h_Quadrature_Filter_3_Real, *h_Quadrature_Filter_1_Imag, *h_Quadrature_Filter_2_Imag, *h_Quadrature_Filter_3_Imag;
    int             NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION;
    
    //-----------------------
    // Output pointers        
    
    double     		*h_Motion_Corrected_fMRI_Volumes_double, *h_Motion_Parameters_double;
    float           *h_Motion_Corrected_fMRI_Volumes, *h_Motion_Parameters;
    
    //---------------------
    
    /* Check the number of input and output arguments. */
    if(nrhs<5)
    {
        mexErrMsgTxt("Too few input arguments.");
    }
    if(nrhs>5)
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
    h_Quadrature_Filter_1_Real_double =  (double*)mxGetPr(prhs[1]);
    h_Quadrature_Filter_1_Imag_double =  (double*)mxGetPi(prhs[1]);
    h_Quadrature_Filter_2_Real_double =  (double*)mxGetPr(prhs[2]);
    h_Quadrature_Filter_2_Imag_double =  (double*)mxGetPi(prhs[2]);
    h_Quadrature_Filter_3_Real_double =  (double*)mxGetPr(prhs[3]);
    h_Quadrature_Filter_3_Imag_double =  (double*)mxGetPi(prhs[3]);
    NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION  = (int)mxGetScalar(prhs[4]);
    
    int NUMBER_OF_DIMENSIONS = mxGetNumberOfDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_DATA = mxGetDimensions(prhs[0]);
    
    int DATA_H, DATA_W, DATA_D, DATA_T, NUMBER_OF_MOTION_CORRECTION_PARAMETERS;
    NUMBER_OF_MOTION_CORRECTION_PARAMETERS = 12;
    
    DATA_H = ARRAY_DIMENSIONS_DATA[0];
    DATA_W = ARRAY_DIMENSIONS_DATA[1];
    DATA_D = ARRAY_DIMENSIONS_DATA[2];
    DATA_T = ARRAY_DIMENSIONS_DATA[3];
                
    int DATA_SIZE = DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float);
    int MOTION_PARAMETERS_SIZE = NUMBER_OF_MOTION_CORRECTION_PARAMETERS * DATA_T * sizeof(float);
    
    mexPrintf("Data size : %i x %i x %i x %i \n",  DATA_W, DATA_H, DATA_D, DATA_T);
    mexPrintf("Filter size : %i x %i x %i \n",  MOTION_CORRECTION_FILTER_SIZE,MOTION_CORRECTION_FILTER_SIZE,MOTION_CORRECTION_FILTER_SIZE);
    mexPrintf("Number of iterations : %i \n",  NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION);
    
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
    h_Motion_Corrected_fMRI_Volumes        = (float *)mxMalloc(DATA_SIZE);
    h_Motion_Parameters                    = (float *)mxMalloc(MOTION_PARAMETERS_SIZE);
    
    // Reorder and cast data
    pack_double2float_volumes(h_fMRI_Volumes, h_fMRI_Volumes_double, DATA_W, DATA_H, DATA_D, DATA_T);
    
       
    //------------------------
    
    BROCCOLI_LIB BROCCOLI;
    
    BROCCOLI.SetWidth(DATA_W);
    BROCCOLI.SetHeight(DATA_H);
    BROCCOLI.SetDepth(DATA_D);
    BROCCOLI.SetTimepoints(DATA_T);   
    BROCCOLI.SetGlobalAndLocalWorkSizes();
    BROCCOLI.SetInputfMRIVolumes(h_fMRI_Volumes);
    BROCCOLI.SetMotionCorrectionFilters(h_Quadrature_Filter_1_Real, h_Quadrature_Filter_1_Imag, h_Quadrature_Filter_2_Real, h_Quadrature_Filter_2_Imag, h_Quadrature_Filter_3_Real, h_Quadrature_Filter_3_Imag);
    BROCCOLI.SetOutputData(h_Motion_Corrected_fMRI_Volumes);
    BROCCOLI.SetOutputMotionParameters(h_Motion_Parameters);
    
    mexPrintf("Device info \n \n %s \n", BROCCOLI.GetOpenCLDeviceInfoChar());
    mexPrintf("Build info \n \n %s \n", BROCCOLI.GetOpenCLBuildInfoChar());
            
    BROCCOLI.PerformMotionCorrectionTest();
    
    int error = BROCCOLI.GetOpenCLError();
    mexPrintf("Error is %d \n",error);
    
    int kernel_error = BROCCOLI.GetOpenCLKernelError();
    mexPrintf("Kernel error is %d \n",kernel_error);
    
    unpack_float2double_volumes(h_Motion_Corrected_fMRI_Volumes_double, h_Motion_Corrected_fMRI_Volumes, DATA_W, DATA_H, DATA_D, DATA_T);
    unpack_float2double(h_Motion_Parameters_double, h_Motion_Parameters, NUMBER_OF_MOTION_CORRECTION_PARAMETERS * DATA_T);
        
    // Free all the allocated memory on the host
    mxFree(h_fMRI_Volumes);
    mxFree(h_Motion_Corrected_fMRI_Volumes); 
    mxFree(h_Motion_Parameters);
    
    //mexAtExit(cleanUp);
    
    return;
}


