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
#include "broccoli_lib.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Check the number of input and output arguments. */
    if(nrhs>0)
    {
        mexErrMsgTxt("Too many input arguments.");
    }
    if(nlhs>0)
    {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    BROCCOLI_LIB BROCCOLI;
    
    BROCCOLI.GetOpenCLInfo();
    
    mexPrintf("Device info \n \n %s \n", BROCCOLI.GetOpenCLDeviceInfoChar());
            
    return;
}


