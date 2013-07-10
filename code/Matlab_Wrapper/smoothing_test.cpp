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
    
    double		    *h_Data_double;
    float           *h_Data;
    
    double	        *h_Filter_double;
    float		    *h_Filter;       
    
    //-----------------------
    // Output pointers
    
    double     		*h_Filter_Response_double;
    float    		*h_Filter_Response;
    
    double          *convolution_time;
    
    //---------------------
    
    /* Check the number of input and output arguments. */
    if(nrhs<2)
    {
        mexErrMsgTxt("Too few input arguments.");
    }
    if(nrhs>2)
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
    h_Data_double =  (double*)mxGetData(prhs[0]);
    h_Filter_double = (double*)mxGetData(prhs[1]);
    
    int NUMBER_OF_DIMENSIONS = mxGetNumberOfDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_DATA = mxGetDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_FILTER = mxGetDimensions(prhs[1]);
    
    int DATA_H, DATA_W, DATA_D, DATA_T, FILTER_LENGTH;
    
    DATA_H = ARRAY_DIMENSIONS_DATA[0];
    DATA_W = ARRAY_DIMENSIONS_DATA[1];
    DATA_D = ARRAY_DIMENSIONS_DATA[2];
    DATA_T = ARRAY_DIMENSIONS_DATA[3];
    
    FILTER_LENGTH = ARRAY_DIMENSIONS_FILTER[0];
            
    int DATA_SIZE = DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float);
    int FILTER_SIZE = FILTER_LENGTH * sizeof(float);
            
    mexPrintf("Data size : %i x %i x %i x %i \n",  DATA_W, DATA_H, DATA_D, DATA_T);
    
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
    h_Data                         = (float *)mxMalloc(DATA_SIZE);
    h_Filter                       = (float *)mxMalloc(FILTER_SIZE);
    h_Filter_Response              = (float *)mxMalloc(DATA_SIZE);
    
    // Reorder and cast data
    pack_double2float_volumes(h_Data, h_Data_double, DATA_W, DATA_H, DATA_D, DATA_T);
    pack_double2float(h_Filter, h_Filter_double, FILTER_LENGTH);
    
    //------------------------
    
    BROCCOLI_LIB BROCCOLI;
    
    BROCCOLI.SetWidth(DATA_W);
    BROCCOLI.SetHeight(DATA_H);
    BROCCOLI.SetDepth(DATA_D);
    BROCCOLI.SetTimepoints(DATA_T);
    BROCCOLI.SetInputData(h_Data);
    BROCCOLI.SetOutputData(h_Filter_Response);
    //BROCCOLI.AllocateMemory();
    //char* log = BROCCOLI.OpenCLInitiate();
    
    int err = BROCCOLI.OpenCLInitiate();
    
    //mexPrintf("%s\n", log);
    
    mexPrintf("Error is %i \n",err);
    
    BROCCOLI.AddVolumes();
    
    unpack_float2double_volumes(h_Filter_Response_double, h_Filter_Response, DATA_W, DATA_H, DATA_D, DATA_T);
        
    
    
    // Free all the allocated memory on the host
    mxFree(h_Data);
    mxFree(h_Filter);
    mxFree(h_Filter_Response);
    
    //mexAtExit(cleanUp);
    
    return;
}


