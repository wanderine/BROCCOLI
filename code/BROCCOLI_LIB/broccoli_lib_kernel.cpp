/*
	BROCCOLI: An Open Source Multi-Platform Software for Parallel Analysis of fMRI Data on Many-Core CPUs and GPUs
    Copyright (C) <2013>  Anders Eklund, andek034@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "broccoli_lib.h"
#include <opencl.h>


// Help functions
int Calculate3DIndex(int x, int y, int z, int DATA_W, int DATA_H)
{
	return x + y * DATA_W + z * DATA_W * DATA_H;
}

int Calculate4DIndex(int x, int y, int z, int t, int DATA_W, int DATA_H, int DATA_D)
{
	// Return a 3D index if t = -1, to make it possible to run the separable convolution functions for
	// a 4D dataset or 3D dataset
	if (t == -1)
	{
		return x + y * DATA_W + z * DATA_W * DATA_H;
	}
	else
	{
		return x + y * DATA_W + z * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D;
	}
}

void GetParameterIndices(int& i, int& j, int parameter)
{
	switch(parameter)
	{
		case 0:
			i = 0; j = 0;
			break;

		case 1:
			i = 3; j = 0;
			break;

		case 2:
			i = 4; j = 0;
			break;

		case 3:
			i = 5; j = 0;
			break;

		case 4:
			i = 3; j = 3;
			break;

		case 5:
			i = 4; j = 3;
			break;

		case 6:
			i = 5; j = 3;
			break;

		case 7:
			i = 4; j = 4;
			break;

		case 8:
			i = 5; j = 4;
			break;

		case 9:
			i = 5; j = 5;
			break;

		case 10:
			i = 1; j = 1;
			break;

		case 11:
			i = 6; j = 1;
			break;

		case 12:
			i = 7; j = 1;
			break;

		case 13:
			i = 8; j = 1;
			break;

		case 14:
			i = 6; j = 6;
			break;

		case 15:
			i = 7; j = 6;
			break;

		case 16:
			i = 8; j = 6;
			break;

		case 17:
			i = 7; j = 7;
			break;

		case 18:
			i = 8; j = 7;
			break;

		case 19:
			i = 8; j = 8;
			break;

		case 20:
			i = 2; j = 2;
			break;

		case 21:
			i = 9; j = 2;
			break;

		case 22:
			i = 10; j = 2;
			break;

		case 23:
			i = 11; j = 2;
			break;

		case 24:
			i = 9; j = 9;
			break;

		case 25:
			i = 10; j = 9;
			break;

		case 26:
			i = 11; j = 9;
			break;

		case 27:
			i = 10; j = 10;
			break;

		case 28:
			i = 11; j = 10;
			break;

		case 29:
			i = 11; j = 11;
			break;

		default:
			i = 0; j = 0;
			break;
	}
}

__kernel void AddVectors(__global const float *a, __global const float *b, __global float *c)
{
	int id = get_global_id(0);

	c[id] = a[id] + b[id];
}

// Some useful parameters

typedef struct DATA_PARAMETERS
{
    int DATA_W;
    int DATA_H;
	int DATA_D;
	int DATA_T;
    int xBlockDifference;
	int yBlockDifference;
	int zBlockDifference;
} DATA_PARAMETERS;


// Convolution functions

// Separable 3D convolution

__kernel void SeparableConvolutionRows(__global float *Filter_Response, __global const float* Volume, __constant float *c_Smoothing_Filter_Y, int t, __constant struct DATA_PARAMETERS* DATA)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_local_size(2) * get_group_id(2) * 4 + get_local_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	//volatile int x = blockIdx.x * blockDim.x + tIdx.x;
	//volatile int y = blockIdx.y * blockDim.y + tIdx.y;
	//volatile int z = blockIdx.z * blockDim.z * 4 + tIdx.z;

    if (x >= (DATA->DATA_W + DATA->xBlockDifference) || y >= (DATA->DATA_H + DATA->yBlockDifference) || z >= (DATA->DATA_D + DATA->zBlockDifference))
        return;

	// 8 * 8 * 32 valid filter responses = 2048
	__local float l_Volume[8][16][32];

	// Reset local memory

	l_Volume[tIdx.z][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 2][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 4][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 6][tIdx.y][tIdx.x] = 0.0f;

	l_Volume[tIdx.z][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x] = 0.0f;

	// Read data into shared memory

	// Upper apron + first half main data

	if ( (x < DATA->DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA->DATA_H) && (z < DATA->DATA->DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y][tIdx.x] = Volume[Calculate_4D_Index(x,y - 4,z,t,DATA->DATA_W, DATA->DATA_H, DATA->DATA->DATA_D)];
	}

	if ( (x < DATA->DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA->DATA_H) && ((z + 2) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y][tIdx.x] = Volume[Calculate_4D_Index(x,y - 4,z + 2,t,DATA->DATA_W, DATA->DATA_H, DATA->DATA->DATA_D)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y][tIdx.x] = Volume[Calculate_4D_Index(x,y - 4,z + 4,t,DATA->DATA_W, DATA->DATA_H, DATA->DATA_D)];
	}

	if ( (x < DATA->DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA->DATA_H) && ((z + 6) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y][tIdx.x] = Volume[Calculate_4D_Index(x,y - 4,z + 6,t,DATA->DATA_W, DATA->DATA_H, DATA->DATA_D)];
	}

	// Second half main data + lower apron

	if ( (x < DATA->DATA_W) && ((y + 4) < DATA->DATA_H) && (z < DATA->DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 8][tIdx.x] = Volume[Calculate_4D_Index(x,y + 4,z,t,DATA->DATA_W, DATA->DATA_H, DATA->DATA_D)];
	}

	if ( (x < DATA->DATA_W) && ((y + 4) < DATA->DATA_H) && ((z + 2) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x] = Volume[Calculate_4D_Index(x,y + 4,z + 2,t,DATA->DATA_W, DATA->DATA_H, DATA->DATA_D)];
	}

	if ( (x < DATA->DATA_W) && ((y + 4) < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x] = Volume[Calculate_4D_Index(x,y + 4,z + 4,t,DATA->DATA_W, DATA->DATA_H, DATA->DATA_D)];
	}

	if ( (x < DATA->DATA_W) && ((y + 4) < DATA->DATA_H) && ((z + 6) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x] = Volume[Calculate_4D_Index(x,y + 4,z + 6,t,DATA->DATA_W, DATA->DATA_H, DATA->DATA_D)];
	}

	// Make sure all threads have written to local memory
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Only threads within the volume do the convolution
	if ( (x < DATA->DATA_W) && (y < DATA->DATA_H) && (z < DATA->DATA_D) )
	{
	    float sum = 0.0f;

		sum += l_Volume[tIdx.z][tIdx.y + 0][tIdx.x] * c_Smoothing_Filter_Y[8];
		sum += l_Volume[tIdx.z][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Y[7];
		sum += l_Volume[tIdx.z][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Y[6];
		sum += l_Volume[tIdx.z][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Y[5];
		sum += l_Volume[tIdx.z][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Y[4];
		sum += l_Volume[tIdx.z][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Y[3];
		sum += l_Volume[tIdx.z][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Y[2];
		sum += l_Volume[tIdx.z][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Y[1];
		sum += l_Volume[tIdx.z][tIdx.y + 8][tIdx.x] * c_Smoothing_Filter_Y[0];

		Filter_Response[Calculate_3D_Index(x,y,z,DATA->DATA_W, DATA->DATA_H)] = sum;
	}

	if ( (x < DATA->DATA_W) && (y < DATA->DATA_H) && ((z + 2) < DATA->DATA_D) )
	{
	    float sum = 0.0f;

		sum += l_Volume[tIdx.z + 2][tIdx.y + 0][tIdx.x] * c_Smoothing_Filter_Y[8];
		sum += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Y[7];
		sum += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Y[6];
		sum += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Y[5];
		sum += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Y[4];
		sum += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Y[3];
		sum += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Y[2];
		sum += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Y[1];
		sum += l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x] * c_Smoothing_Filter_Y[0];

		Filter_Response[Calculate_3D_Index(x,y,z + 2,DATA->DATA_W, DATA->DATA_H)] = sum;
	}

	if ( (x < DATA->DATA_W) && (y < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
	    float sum = 0.0f;

		sum += l_Volume[tIdx.z + 4][tIdx.y + 0][tIdx.x] * c_Smoothing_Filter_Y[8];
		sum += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Y[7];
		sum += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Y[6];
		sum += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Y[5];
		sum += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Y[4];
		sum += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Y[3];
		sum += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Y[2];
		sum += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Y[1];
		sum += l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x] * c_Smoothing_Filter_Y[0];

		Filter_Response[Calculate_3D_Index(x,y,z + 4,DATA->DATA_W, DATA->DATA_H)] = sum;
	}

	if ( (x < DATA->DATA_W) && (y < DATA->DATA_H) && ((z + 6) < DATA->DATA_D) )
	{
	    float sum = 0.0f;

		sum += l_Volume[tIdx.z + 6][tIdx.y + 0][tIdx.x] * c_Smoothing_Filter_Y[8];
		sum += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Y[7];
		sum += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Y[6];
		sum += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Y[5];
		sum += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Y[4];
		sum += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Y[3];
		sum += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Y[2];
		sum += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Y[1];
		sum += l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x] * c_Smoothing_Filter_Y[0];

		Filter_Response[Calculate_3D_Index(x,y,z + 6,DATA->DATA_W, DATA->DATA_H)] = sum;
	}
}

__kernel void SeparableConvolutionColumns(__global float *Filter_Response, __global const float* Volume, __constant float *c_Smoothing_Filter_X, int t, __constant struct DATA_PARAMETERS *DATA)
{
	int x = get_local_size(0) * get_group_id(0) / 32 * 24 + get_local_id(0);;
	int y = get_local_size(1) * get_group_id(1) * 2 + get_local_id(2);
	int z = get_local_size(2) * get_group_id(2) * 2 + get_local_id(2);  

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	//volatile int x = blockIdx.x * blockDim.x / 32 * 24 + tIdx.x;
	//volatile int y = blockIdx.y * blockDim.y * 2 + tIdx.y;
	//volatile int z = blockIdx.z * blockDim.z * 4 + tIdx.z;

    if (x >= (DATA->DATA_W + DATA->xBlockDifference) || y >= (DATA->DATA_H + DATA->yBlockDifference) || z >= (DATA->DATA_D + DATA->zBlockDifference))
        return;

	// 8 * 8 * 32 valid filter responses = 2048
	__local float l_Volume[8][16][32];

	// Reset shared memory
	l_Volume[tIdx.z][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 2][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 4][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 6][tIdx.y][tIdx.x] = 0.0f;

	l_Volume[tIdx.z][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x] = 0.0f;

	// Read data into shared memory

	if ( ((x - 4) >= 0) && ((x - 4) < DATA->DATA_W) && (y < DATA->DATA_H) && (z < DATA->DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y,z,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA->DATA_W) && (y < DATA->DATA_H) && ((z + 2) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y,z + 2,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA->DATA_W) && (y < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y,z + 4,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA->DATA_W) && (y < DATA->DATA_H) && ((z + 6) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y,z + 6,DATA->DATA_W, DATA->DATA_H)];
	}



	if ( ((x - 4) >= 0) && ((x - 4) < DATA->DATA_W) && ((y + 8) < DATA->DATA_H) && (z < DATA->DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 8][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y + 8,z,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA->DATA_W) && ((y + 8) < DATA->DATA_H) && ((z + 2) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y + 8,z + 2,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA->DATA_W) && ((y + 8) < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y + 8,z + 4,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA->DATA_W) && ((y + 8) < DATA->DATA_H) && ((z + 6) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y + 8,z + 6,DATA->DATA_W, DATA->DATA_H)];
	}

	// Make sure all threads have written to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Only threads within the volume do the convolution
	if (get_local_id(1) < 24)
	{
		if ( (x < DATA->DATA_W) && (y < DATA->DATA_H) && (z < DATA->DATA_D) )
		{
		    float sum = 0.0f;

			sum += l_Volume[tIdx.z][tIdx.y][tIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += l_Volume[tIdx.z][tIdx.y][tIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += l_Volume[tIdx.z][tIdx.y][tIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += l_Volume[tIdx.z][tIdx.y][tIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += l_Volume[tIdx.z][tIdx.y][tIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += l_Volume[tIdx.z][tIdx.y][tIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += l_Volume[tIdx.z][tIdx.y][tIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += l_Volume[tIdx.z][tIdx.y][tIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += l_Volume[tIdx.z][tIdx.y][tIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Response[Calculate_3D_Index(x,y,z,DATA->DATA_W, DATA->DATA_H)] = sum;
		}

		if ( (x < DATA->DATA_W) && (y < DATA->DATA_H) && ((z + 2) < DATA->DATA_D) )
		{
		    float sum = 0.0f;

			sum += l_Volume[tIdx.z + 2][tIdx.y][tIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += l_Volume[tIdx.z + 2][tIdx.y][tIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += l_Volume[tIdx.z + 2][tIdx.y][tIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += l_Volume[tIdx.z + 2][tIdx.y][tIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += l_Volume[tIdx.z + 2][tIdx.y][tIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += l_Volume[tIdx.z + 2][tIdx.y][tIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += l_Volume[tIdx.z + 2][tIdx.y][tIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += l_Volume[tIdx.z + 2][tIdx.y][tIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += l_Volume[tIdx.z + 2][tIdx.y][tIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Response[Calculate_3D_Index(x,y,z + 2,DATA->DATA_W, DATA->DATA_H)] = sum;
		}

		if ( (x < DATA->DATA_W) && (y < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
		{
		    float sum = 0.0f;

			sum += l_Volume[tIdx.z + 4][tIdx.y][tIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += l_Volume[tIdx.z + 4][tIdx.y][tIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += l_Volume[tIdx.z + 4][tIdx.y][tIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += l_Volume[tIdx.z + 4][tIdx.y][tIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += l_Volume[tIdx.z + 4][tIdx.y][tIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += l_Volume[tIdx.z + 4][tIdx.y][tIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += l_Volume[tIdx.z + 4][tIdx.y][tIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += l_Volume[tIdx.z + 4][tIdx.y][tIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += l_Volume[tIdx.z + 4][tIdx.y][tIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Response[Calculate_3D_Index(x,y,z + 4,DATA->DATA_W, DATA->DATA_H)] = sum;
		}

		if ( (x < DATA->DATA_W) && (y < DATA->DATA_H) && ((z + 6) < DATA->DATA_D) )
		{
		    float sum = 0.0f;

			sum += l_Volume[tIdx.z + 6][tIdx.y][tIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += l_Volume[tIdx.z + 6][tIdx.y][tIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += l_Volume[tIdx.z + 6][tIdx.y][tIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += l_Volume[tIdx.z + 6][tIdx.y][tIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += l_Volume[tIdx.z + 6][tIdx.y][tIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += l_Volume[tIdx.z + 6][tIdx.y][tIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += l_Volume[tIdx.z + 6][tIdx.y][tIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += l_Volume[tIdx.z + 6][tIdx.y][tIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += l_Volume[tIdx.z + 6][tIdx.y][tIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Response[Calculate_3D_Index(x,y,z + 6,DATA->DATA_W, DATA->DATA_H)] = sum;
		}

		if ( (x < DATA->DATA_W) && ((y + 8) < DATA->DATA_H) && (z < DATA->DATA_D) )
		{
		    float sum = 0.0f;

			sum += l_Volume[tIdx.z][tIdx.y + 8][tIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += l_Volume[tIdx.z][tIdx.y + 8][tIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += l_Volume[tIdx.z][tIdx.y + 8][tIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += l_Volume[tIdx.z][tIdx.y + 8][tIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += l_Volume[tIdx.z][tIdx.y + 8][tIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += l_Volume[tIdx.z][tIdx.y + 8][tIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += l_Volume[tIdx.z][tIdx.y + 8][tIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += l_Volume[tIdx.z][tIdx.y + 8][tIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += l_Volume[tIdx.z][tIdx.y + 8][tIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Response[Calculate_3D_Index(x,y + 8,z,DATA->DATA_W, DATA->DATA_H)] = sum;
		}

		if ( (x < DATA->DATA_W) && ((y + 8) < DATA->DATA_H) && ((z + 2) < DATA->DATA_D) )
		{
		    float sum = 0.0f;

			sum += l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Response[Calculate_3D_Index(x,y + 8,z + 2,DATA->DATA_W, DATA->DATA_H)] = sum;
		}

		if ( (x < DATA->DATA_W) && ((y + 8) < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
		{
		    float sum = 0.0f;

			sum += l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Response[Calculate_3D_Index(x,y + 8,z + 4,DATA->DATA_W, DATA->DATA_H)] = sum;
		}

		if ( (x < DATA->DATA_W) && ((y + 8) < DATA->DATA_H) && ((z + 6) < DATA->DATA_D) )
		{
		    float sum = 0.0f;

			sum += l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Response[Calculate_3D_Index(x,y + 8,z + 6,DATA->DATA_W, DATA->DATA_H)] = sum;
		}
	}
}

__kernel void SeparableConvolutionRods(__global float *Filter_Response, __global const float* Volume, __constant float *c_Smoothing_Filter_Z, int t, __constant struct DATA_PARAMETERS *DATA)
{
	int x = get_global_id(0);
	int y = get_local_size(1) * get_group_id(1) * 4 + get_local_id(1); 
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	//volatile int x = blockIdx.x * blockDim.x + tIdx.x;
	//volatile int y = blockIdx.y * blockDim.y * 4 + tIdx.y;
	//volatile int z = blockIdx.z * blockDim.z + tIdx.z;

    if (x >= (DATA->DATA_W + DATA->xBlockDifference) || y >= (DATA->DATA_H + DATA->yBlockDifference) || z >= (DATA->DATA_D + DATA->zBlockDifference))
        return;

	// 8 * 8 * 32 valid filter responses = 2048
	__local float l_Volume[16][8][32];

	// Reset shared memory
	l_Volume[tIdx.z][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z][tIdx.y + 2][tIdx.x] = 0.0f;
	l_Volume[tIdx.z][tIdx.y + 4][tIdx.x] = 0.0f;
	l_Volume[tIdx.z][tIdx.y + 6][tIdx.x] = 0.0f;

	l_Volume[tIdx.z + 8][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 8][tIdx.y + 2][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 8][tIdx.y + 4][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 8][tIdx.y + 6][tIdx.x] = 0.0f;

    
	// Read data into shared memory

	// Above apron + first half main data

	if ( (x < DATA->DATA_W) && (y < DATA->DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y][tIdx.x] = Volume[Calculate_3D_Index(x,y,z - 4,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( (x < DATA->DATA_W) && ((y + 2) < DATA->DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 2][tIdx.x] = Volume[Calculate_3D_Index(x,y + 2,z - 4,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( (x < DATA->DATA_W) && ((y + 4) < DATA->DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 4][tIdx.x] = Volume[Calculate_3D_Index(x,y + 4,z - 4,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( (x < DATA->DATA_W) && ((y + 6) < DATA->DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 6][tIdx.x] = Volume[Calculate_3D_Index(x,y + 6,z - 4,DATA->DATA_W, DATA->DATA_H)];
	}

	// Second half main data + below apron

	if ( (x < DATA->DATA_W) && (y < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y][tIdx.x] = Volume[Calculate_3D_Index(x,y,z + 4,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( (x < DATA->DATA_W) && ((y + 2) < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y + 2][tIdx.x] = Volume[Calculate_3D_Index(x,y + 2,z + 4,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( (x < DATA->DATA_W) && ((y + 4) < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y + 4][tIdx.x] = Volume[Calculate_3D_Index(x,y + 4,z + 4,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( (x < DATA->DATA_W) && ((y + 6) < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y + 6][tIdx.x] = Volume[Calculate_3D_Index(x,y + 6,z + 4,DATA->DATA_W, DATA->DATA_H)];
	}

	// Make sure all threads have written to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Only threads within the volume do the convolution
	if ( (x < DATA->DATA_W) && (y < DATA->DATA_H) && (z < DATA->DATA_D) )
	{
	    float sum = 0.0f;

		sum += l_Volume[tIdx.z + 0][tIdx.y][tIdx.x] * c_Smoothing_Filter_Z[8];
		sum += l_Volume[tIdx.z + 1][tIdx.y][tIdx.x] * c_Smoothing_Filter_Z[7];
		sum += l_Volume[tIdx.z + 2][tIdx.y][tIdx.x] * c_Smoothing_Filter_Z[6];
		sum += l_Volume[tIdx.z + 3][tIdx.y][tIdx.x] * c_Smoothing_Filter_Z[5];
		sum += l_Volume[tIdx.z + 4][tIdx.y][tIdx.x] * c_Smoothing_Filter_Z[4];
		sum += l_Volume[tIdx.z + 5][tIdx.y][tIdx.x] * c_Smoothing_Filter_Z[3];
		sum += l_Volume[tIdx.z + 6][tIdx.y][tIdx.x] * c_Smoothing_Filter_Z[2];
		sum += l_Volume[tIdx.z + 7][tIdx.y][tIdx.x] * c_Smoothing_Filter_Z[1];
		sum += l_Volume[tIdx.z + 8][tIdx.y][tIdx.x] * c_Smoothing_Filter_Z[0];

		Filter_Response[Calculate_4D_Index(x,y,z,t,DATA->DATA_W,DATA->DATA_H,DATA->DATA_D)] = sum;
	}

	if ( (x < DATA->DATA_W) && ((y + 2) < DATA->DATA_H) && (z < DATA->DATA_D) )
	{
	    float sum = 0.0f;

		sum += l_Volume[tIdx.z + 0][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Z[8];
		sum += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Z[7];
		sum += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Z[6];
		sum += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Z[5];
		sum += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Z[4];
		sum += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Z[3];
		sum += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Z[2];
		sum += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Z[1];
		sum += l_Volume[tIdx.z + 8][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Z[0];

		Filter_Response[Calculate_4D_Index(x,y + 2,z,t,DATA->DATA_W,DATA->DATA_H,DATA->DATA_D)] = sum;
	}

	if ( (x < DATA->DATA_W) && ((y + 4) < DATA->DATA_H) && (z < DATA->DATA_D) )
	{
	    float sum = 0.0f;

		sum += l_Volume[tIdx.z + 0][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Z[8];
		sum += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Z[7];
		sum += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Z[6];
		sum += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Z[5];
		sum += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Z[4];
		sum += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Z[3];
		sum += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Z[2];
		sum += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Z[1];
		sum += l_Volume[tIdx.z + 8][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Z[0];

		Filter_Response[Calculate_4D_Index(x,y + 4,z,t,DATA->DATA_W,DATA->DATA_H,DATA->DATA_D)] = sum;
	}

	if ( (x < DATA->DATA_W) && ((y + 6) < DATA->DATA_H) && (z < DATA->DATA_D) )
	{
	    float sum = 0.0f;

		sum += l_Volume[tIdx.z + 0][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Z[8];
		sum += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Z[7];
		sum += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Z[6];
		sum += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Z[5];
		sum += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Z[4];
		sum += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Z[3];
		sum += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Z[2];
		sum += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Z[1];
		sum += l_Volume[tIdx.z + 8][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Z[0];

		Filter_Response[Calculate_4D_Index(x,y + 6,z,t,DATA->DATA_W,DATA->DATA_H,DATA->DATA_D)] = sum;
	}
}

// Non-separable 3D convolution

__kernel void NonseparableConvolution3DComplex(__global float2 *Filter_Response_1, __global float2 *Filter_Response_2, __global float2 *Filter_Response_3, __global const float* Volume, __constant float2 *c_Quadrature_Filter_1, __constant float2 *c_Quadrature_Filter_2, __constant float2 *c_Quadrature_Filter_3, __constant struct DATA_PARAMETERS *DATA)
{   
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

    if (x >= (DATA_W + xBlockDifference) || y >= (DATA_H + yBlockDifference) || z >= (DATA_D + zBlockDifference))
        return;
	
	__local float l_Volume[16][16][16];    // z, y, x

	/* 27 blocks in local memory, filter response is calculated for block 14, 8 x 8 x 8 threads per block
	
	Top layer, 16 x 16 x 4
		
	1 2 3
	4 5 6
	7 8 9

	Middle layer, 16 x 16 x 8

	10 11 12
	13 14 15
	16 17 18

	Bottom layer, 16 x 16 x 4

	19 20 21
	22 23 24
	25 26 27	
	
	*/

	// Read data into local memory
	
	// Top layer

	// Block 1 (4 x 4 x 4)
	if ((tIdx.x < 4) && (tIdx.y < 4) && (tIdx.z < 4))
	{
		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			l_Volume[tIdx.z][tIdx.y][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y - 4,z - 4,DATA_W, DATA_H)];
		}
		else
		{
			l_Volume[tIdx.z][tIdx.y][tIdx.x] = 0.0f;
		}
	}
	
	// Block 2 (8 x 4 x 4)
	if ((tIdx.x < 8) && (tIdx.y < 4) && (tIdx.z < 4))
	{		
		if ( (x < DATA_W) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			l_Volume[tIdx.z][tIdx.y][tIdx.x + 4] = Volume[Calculate_3D_Index(x,y - 4,z - 4,DATA_W, DATA_H)];
		}
		else
		{
			l_Volume[tIdx.z][tIdx.y][tIdx.x + 4] = 0.0f;
		}
	}

	// Block 3 (4 x 4 x 4)
	if ((tIdx.x < 4) && (tIdx.y < 4) && (tIdx.z < 4))
	{		
		if ( ((x + 8) < DATA_W) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			l_Volume[tIdx.z][tIdx.y][tIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y - 4,z - 4,DATA_W, DATA_H)];
		}
		else
		{
			l_Volume[tIdx.z][tIdx.y][tIdx.x + 12] = 0.0f;
		}
	}

	
	// Block 4 (4 x 8 x 4)
	if ((tIdx.x < 4) && (tIdx.y < 8) && (tIdx.z < 4))
	{		
		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && (y < DATA_H) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			l_Volume[tIdx.z][tIdx.y + 4][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y,z - 4,DATA_W, DATA_H)];
		}
		else
		{
			l_Volume[tIdx.z][tIdx.y + 4][tIdx.x] = 0.0f;
		}
	}

	
	// Block 5 (8 x 8 x 4)
	if ((tIdx.x < 8) && (tIdx.y < 8) && (tIdx.z < 4))
	{		
		if ( (x < DATA_W) && (y < DATA_H) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			l_Volume[tIdx.z][tIdx.y + 4][tIdx.x + 4] = Volume[Calculate_3D_Index(x,y,z - 4,DATA_W, DATA_H)];
		}
		else
		{
			l_Volume[tIdx.z][tIdx.y + 4][tIdx.x + 4] = 0.0f;
		}
	}

	
	// Block 6 (4 x 8 x 4)
	if ((tIdx.x < 4) && (tIdx.y < 8) && (tIdx.z < 4))
	{		
		if ( ((x + 8) < DATA_W) && (y < DATA_H) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			l_Volume[tIdx.z][tIdx.y + 4][tIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y,z - 4,DATA_W, DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z][tIdx.y + 4][tIdx.x + 12] = 0.0f;
		}
	}

	
	// Block 7 (4 x 4 x 4)
	if ((tIdx.x < 4) && (tIdx.y < 4) && (tIdx.z < 4))
	{		

		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y + 8) < DATA_H) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			l_Volume[tIdx.z][tIdx.y + 12][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y + 8,z - 4,DATA_W, DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z][tIdx.y + 12][tIdx.x] = 0.0f;
		}
	}

	
	// Block 8 (8 x 4 x 4)
	if ((tIdx.x < 8) && (tIdx.y < 4) && (tIdx.z < 4))
	{		
		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			l_Volume[tIdx.z][tIdx.y + 12][tIdx.x + 4] = Volume[Calculate_3D_Index(x,y + 8,z - 4,DATA_W, DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z][tIdx.y + 12][tIdx.x + 4] = 0.0f;
		}
	}

	
	
	// Block 9 (4 x 4 x 4)
	if ((tIdx.x < 4) && (tIdx.y < 4) && (tIdx.z < 4))
	{		
		if ( ((x + 8) < DATA_W) && ((y + 8) < DATA_H) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			l_Volume[tIdx.z][tIdx.y + 12][tIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y + 8,z - 4,DATA_W, DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z][tIdx.y + 12][tIdx.x + 12] = 0.0f;
		}
	}
	

	// Middle layer

	
	// Block 10 (4 x 4 x 8)
	if ((tIdx.x < 4) && (tIdx.y < 4) && (tIdx.z < 8))
	{		
		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && (z < DATA_D) )
		{
			l_Volume[tIdx.z + 4][tIdx.y][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y - 4,z,DATA_W, DATA_H)];
		}
		else
		{
			l_Volume[tIdx.z + 4][tIdx.y][tIdx.x] = 0.0f;
		}
	}

	
	// Block 11 (8 x 4 x 8)
	if ((tIdx.x < 8) && (tIdx.y < 4) && (tIdx.z < 8))
	{		

		if ( (x < DATA_W) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && (z < DATA_D) )
		{
			l_Volume[tIdx.z + 4][tIdx.y][tIdx.x + 4] = Volume[Calculate_3D_Index(x,y - 4,z,DATA_W, DATA_H)];
		}
		else
		{
			l_Volume[tIdx.z + 4][tIdx.y][tIdx.x + 4] = 0.0f;
		}
	}

	
	
	// Block 12 (4 x 4 x 8)
	if ((tIdx.x < 4) && (tIdx.y < 4) && (tIdx.z < 8))
	{		
		if ( ((x  + 8) < DATA_W) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && (z < DATA_D) )
		{
			l_Volume[tIdx.z + 4][tIdx.y][tIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y - 4,z,DATA_W, DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z + 4][tIdx.y][tIdx.x + 12] = 0.0f;
		}
	}
	
	
	// Block 13 (4 x 8 x 8)
	if ((tIdx.x < 4) && (tIdx.y < 8) && (tIdx.z < 8))
	{		
		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && (y < DATA_H) && (z < DATA_D) )
		{
			l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y,z,DATA_W, DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x] = 0.0f;
		}
	
	}

	
	// Block 14, main data (8 x 8 x 8)	
	if ( (x < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4] = Volume[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)];			
	}
	else
	{
		l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4] = 0.0f;			
	}

	
	// Block 15 (4 x 8 x 8)
	if ((tIdx.x < 4) && (tIdx.y < 8) && (tIdx.z < 8))
	{		
		if ( ((x + 8) < DATA_W) && (y < DATA_H) && (z < DATA_D) )
		{
			l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y,z,DATA_W, DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 12] = 0.0f;
		}		
	}

	
	// Block 16 (4 x 4 x 8)
	if ((tIdx.x < 4) && (tIdx.y < 4) && (tIdx.z < 8))
	{		
		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y + 8) < DATA_H) && (z < DATA_D) )
		{
			l_Volume[tIdx.z + 4][tIdx.y + 12][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y + 8,z,DATA_W, DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z + 4][tIdx.y + 12][tIdx.x] = 0.0f;
		}
	}

	
	// Block 17 (8 x 4 x 8)
	if ((tIdx.x < 8) && (tIdx.y < 4) && (tIdx.z < 8))
	{		
		if ( (x < DATA_W) && ((y + 8) < DATA_H) && (z < DATA_D) )
		{
			l_Volume[tIdx.z + 4][tIdx.y + 12][tIdx.x + 4] = Volume[Calculate_3D_Index(x,y + 8,z,DATA_W, DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z + 4][tIdx.y + 12][tIdx.x + 4] = 0.0f;
		}
	}

	
	// Block 18 (4 x 4 x 8)
	if ((tIdx.x < 4) && (tIdx.y < 4) && (tIdx.z < 8))
	{		
		if ( ((x + 8) < DATA_W) && ((y + 8) < DATA_H) && (z < DATA_D) )
		{
			l_Volume[tIdx.z + 4][tIdx.y + 12][tIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y + 8,z,DATA_W, DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z + 4][tIdx.y + 12][tIdx.x + 12] = 0.0f;
		}
	}


	// Bottom layer

	
	// Block 19 (4 x 4 x 4)
	if ((tIdx.x < 4) && (tIdx.y < 4) && (tIdx.z < 4))
	{		
		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z + 8) < DATA_D) )
		{
			l_Volume[tIdx.z + 12][tIdx.y][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y - 4,z + 8,DATA_W, DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z + 12][tIdx.y][tIdx.x] = 0.0f;
		}
	}

	
	// Block 20 (8 x 4 x 4)
	if ((tIdx.x < 8) && (tIdx.y < 4) && (tIdx.z < 4))
	{	
		if ( (x < DATA_W) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z + 8) < DATA_D) )
		{
			l_Volume[tIdx.z + 12][tIdx.y][tIdx.x + 4] = Volume[Calculate_3D_Index(x,y - 4,z + 8,DATA_W, DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z + 12][tIdx.y][tIdx.x + 4] = 0.0f;
		}
	}

	
	// Block 21 (4 x 4 x 4)
	if ((tIdx.x < 4) && (tIdx.y < 4) && (tIdx.z < 4))
	{		
		if ( ((x + 8) < DATA_W) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z + 8) < DATA_D) )
		{
			l_Volume[tIdx.z + 12][tIdx.y][tIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y - 4,z + 8,DATA_W, DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z + 12][tIdx.y][tIdx.x + 12] = 0.0f;
		}
	}

	
	// Block 22 (4 x 8 x 4)
	if ((tIdx.x < 4) && (tIdx.y < 8) && (tIdx.z < 4))
	{		
		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && (y < DATA_H) && ((z + 8) < DATA_D) )
		{
			l_Volume[tIdx.z + 12][tIdx.y + 4][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y,z + 8,DATA_W,DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z + 12][tIdx.y + 4][tIdx.x] = 0.0f;
		}
	}

	
	// Block 23 (8 x 8 x 4)
	if ((tIdx.x < 8) && (tIdx.y < 8) && (tIdx.z < 4))
	{		
		if ( (x < DATA_W) && (y < DATA_H) && ((z + 8) < DATA_D) )
		{
			l_Volume[tIdx.z + 12][tIdx.y + 4][tIdx.x + 4] = Volume[Calculate_3D_Index(x,y,z + 8,DATA_W,DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z + 12][tIdx.y + 4][tIdx.x + 4] = 0.0f;
		}
	}

	
	// Block 24 (4 x 8 x 4)
	if ((tIdx.x < 4) && (tIdx.y < 8) && (tIdx.z < 4))
	{		
		if ( ((x + 8) < DATA_W) && (y < DATA_H) && ((z + 8) < DATA_D) )
		{
			l_Volume[tIdx.z + 12][tIdx.y + 4][tIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y,z + 8,DATA_W,DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z + 12][tIdx.y + 4][tIdx.x + 12] = 0.0f;
		}
	}

	
	// Block 25 (4 x 4 x 4)
	if ((tIdx.x < 4) && (tIdx.y < 4) && (tIdx.z < 4))
	{		
		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y + 8) < DATA_H) && ((z + 8) < DATA_D) )
		{
			l_Volume[tIdx.z + 12][tIdx.y + 12][tIdx.x] = Volume[Calculate_3D_Index(x - 4,y + 8,z + 8,DATA_W,DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z + 12][tIdx.y + 12][tIdx.x] = 0.0f;
		}
	}

	
	// Block 26 (8 x 4 x 4)
	if ((tIdx.x < 8) && (tIdx.y < 4) && (tIdx.z < 4))
	{		
		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 8) < DATA_D) )
		{
			l_Volume[tIdx.z + 12][tIdx.y + 12][tIdx.x + 4] = Volume[Calculate_3D_Index(x,y + 8,z + 8,DATA_W,DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z + 12][tIdx.y + 12][tIdx.x + 4] = 0.0f;
		}
	}

	
	// Block 27 (4 x 4 x 4)
	if ((tIdx.x < 4) && (tIdx.y < 4) && (tIdx.z < 4))
	{		
		if ( ((x + 8) < DATA_W) && ((y + 8) < DATA_H) && ((z + 8) < DATA_D) )
		{
			l_Volume[tIdx.z + 12][tIdx.y + 12][tIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y + 8,z + 8,DATA_W,DATA_H)];
		}	
		else
		{
			l_Volume[tIdx.z + 12][tIdx.y + 12][tIdx.x + 12] = 0.0f;
		}
	}

	// Make sure all threads have written to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Calculate filter responses for block 14
	
	// Only threads within the volume do the convolution
	if ( (x < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
			float2 sum1, sum2, sum3;
			sum1.x = 0.0f; sum1.y = 0.0f; sum2.x = 0.0f; sum2.y = 0.0f; sum3.x = 0.0f; sum3.y = 0.0f;

            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_1[6][6][6].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_1[6][6][6].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_2[6][6][6].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_2[6][6][6].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_3[6][6][6].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_3[6][6][6].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_1[5][6][6].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_1[5][6][6].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_2[5][6][6].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_2[5][6][6].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_3[5][6][6].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_3[5][6][6].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_1[4][6][6].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_1[4][6][6].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_2[4][6][6].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_2[4][6][6].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_3[4][6][6].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_3[4][6][6].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_1[3][6][6].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_1[3][6][6].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_2[3][6][6].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_2[3][6][6].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_3[3][6][6].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_3[3][6][6].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_1[2][6][6].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_1[2][6][6].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_2[2][6][6].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_2[2][6][6].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_3[2][6][6].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_3[2][6][6].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_1[1][6][6].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_1[1][6][6].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_2[1][6][6].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_2[1][6][6].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_3[1][6][6].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_3[1][6][6].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_1[0][6][6].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_1[0][6][6].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_2[0][6][6].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_2[0][6][6].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_3[0][6][6].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 1] * c_Quadrature_Filter_3[0][6][6].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_1[6][5][6].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_1[6][5][6].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_2[6][5][6].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_2[6][5][6].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_3[6][5][6].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_3[6][5][6].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_1[5][5][6].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_1[5][5][6].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_2[5][5][6].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_2[5][5][6].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_3[5][5][6].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_3[5][5][6].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_1[4][5][6].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_1[4][5][6].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_2[4][5][6].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_2[4][5][6].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_3[4][5][6].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_3[4][5][6].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_1[3][5][6].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_1[3][5][6].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_2[3][5][6].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_2[3][5][6].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_3[3][5][6].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_3[3][5][6].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_1[2][5][6].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_1[2][5][6].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_2[2][5][6].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_2[2][5][6].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_3[2][5][6].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_3[2][5][6].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_1[1][5][6].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_1[1][5][6].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_2[1][5][6].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_2[1][5][6].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_3[1][5][6].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_3[1][5][6].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_1[0][5][6].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_1[0][5][6].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_2[0][5][6].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_2[0][5][6].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_3[0][5][6].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 1] * c_Quadrature_Filter_3[0][5][6].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_1[6][4][6].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_1[6][4][6].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_2[6][4][6].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_2[6][4][6].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_3[6][4][6].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_3[6][4][6].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_1[5][4][6].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_1[5][4][6].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_2[5][4][6].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_2[5][4][6].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_3[5][4][6].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_3[5][4][6].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_1[4][4][6].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_1[4][4][6].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_2[4][4][6].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_2[4][4][6].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_3[4][4][6].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_3[4][4][6].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_1[3][4][6].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_1[3][4][6].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_2[3][4][6].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_2[3][4][6].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_3[3][4][6].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_3[3][4][6].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_1[2][4][6].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_1[2][4][6].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_2[2][4][6].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_2[2][4][6].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_3[2][4][6].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_3[2][4][6].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_1[1][4][6].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_1[1][4][6].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_2[1][4][6].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_2[1][4][6].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_3[1][4][6].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_3[1][4][6].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_1[0][4][6].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_1[0][4][6].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_2[0][4][6].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_2[0][4][6].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_3[0][4][6].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 1] * c_Quadrature_Filter_3[0][4][6].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_1[6][3][6].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_1[6][3][6].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_2[6][3][6].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_2[6][3][6].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_3[6][3][6].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_3[6][3][6].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_1[5][3][6].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_1[5][3][6].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_2[5][3][6].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_2[5][3][6].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_3[5][3][6].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_3[5][3][6].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_1[4][3][6].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_1[4][3][6].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_2[4][3][6].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_2[4][3][6].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_3[4][3][6].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_3[4][3][6].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_1[3][3][6].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_1[3][3][6].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_2[3][3][6].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_2[3][3][6].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_3[3][3][6].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_3[3][3][6].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_1[2][3][6].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_1[2][3][6].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_2[2][3][6].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_2[2][3][6].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_3[2][3][6].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_3[2][3][6].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_1[1][3][6].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_1[1][3][6].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_2[1][3][6].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_2[1][3][6].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_3[1][3][6].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_3[1][3][6].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_1[0][3][6].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_1[0][3][6].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_2[0][3][6].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_2[0][3][6].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_3[0][3][6].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 1] * c_Quadrature_Filter_3[0][3][6].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_1[6][2][6].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_1[6][2][6].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_2[6][2][6].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_2[6][2][6].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_3[6][2][6].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_3[6][2][6].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_1[5][2][6].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_1[5][2][6].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_2[5][2][6].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_2[5][2][6].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_3[5][2][6].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_3[5][2][6].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_1[4][2][6].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_1[4][2][6].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_2[4][2][6].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_2[4][2][6].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_3[4][2][6].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_3[4][2][6].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_1[3][2][6].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_1[3][2][6].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_2[3][2][6].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_2[3][2][6].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_3[3][2][6].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_3[3][2][6].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_1[2][2][6].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_1[2][2][6].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_2[2][2][6].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_2[2][2][6].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_3[2][2][6].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_3[2][2][6].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_1[1][2][6].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_1[1][2][6].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_2[1][2][6].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_2[1][2][6].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_3[1][2][6].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_3[1][2][6].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_1[0][2][6].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_1[0][2][6].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_2[0][2][6].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_2[0][2][6].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_3[0][2][6].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 1] * c_Quadrature_Filter_3[0][2][6].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_1[6][1][6].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_1[6][1][6].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_2[6][1][6].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_2[6][1][6].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_3[6][1][6].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_3[6][1][6].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_1[5][1][6].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_1[5][1][6].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_2[5][1][6].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_2[5][1][6].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_3[5][1][6].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_3[5][1][6].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_1[4][1][6].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_1[4][1][6].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_2[4][1][6].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_2[4][1][6].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_3[4][1][6].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_3[4][1][6].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_1[3][1][6].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_1[3][1][6].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_2[3][1][6].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_2[3][1][6].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_3[3][1][6].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_3[3][1][6].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_1[2][1][6].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_1[2][1][6].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_2[2][1][6].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_2[2][1][6].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_3[2][1][6].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_3[2][1][6].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_1[1][1][6].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_1[1][1][6].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_2[1][1][6].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_2[1][1][6].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_3[1][1][6].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_3[1][1][6].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_1[0][1][6].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_1[0][1][6].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_2[0][1][6].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_2[0][1][6].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_3[0][1][6].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 1] * c_Quadrature_Filter_3[0][1][6].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_1[6][0][6].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_1[6][0][6].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_2[6][0][6].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_2[6][0][6].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_3[6][0][6].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_3[6][0][6].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_1[5][0][6].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_1[5][0][6].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_2[5][0][6].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_2[5][0][6].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_3[5][0][6].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_3[5][0][6].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_1[4][0][6].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_1[4][0][6].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_2[4][0][6].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_2[4][0][6].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_3[4][0][6].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_3[4][0][6].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_1[3][0][6].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_1[3][0][6].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_2[3][0][6].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_2[3][0][6].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_3[3][0][6].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_3[3][0][6].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_1[2][0][6].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_1[2][0][6].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_2[2][0][6].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_2[2][0][6].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_3[2][0][6].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_3[2][0][6].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_1[1][0][6].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_1[1][0][6].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_2[1][0][6].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_2[1][0][6].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_3[1][0][6].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_3[1][0][6].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_1[0][0][6].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_1[0][0][6].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_2[0][0][6].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_2[0][0][6].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_3[0][0][6].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 1] * c_Quadrature_Filter_3[0][0][6].y;

            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_1[6][6][5].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_1[6][6][5].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_2[6][6][5].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_2[6][6][5].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_3[6][6][5].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_3[6][6][5].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_1[5][6][5].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_1[5][6][5].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_2[5][6][5].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_2[5][6][5].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_3[5][6][5].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_3[5][6][5].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_1[4][6][5].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_1[4][6][5].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_2[4][6][5].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_2[4][6][5].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_3[4][6][5].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_3[4][6][5].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_1[3][6][5].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_1[3][6][5].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_2[3][6][5].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_2[3][6][5].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_3[3][6][5].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_3[3][6][5].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_1[2][6][5].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_1[2][6][5].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_2[2][6][5].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_2[2][6][5].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_3[2][6][5].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_3[2][6][5].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_1[1][6][5].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_1[1][6][5].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_2[1][6][5].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_2[1][6][5].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_3[1][6][5].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_3[1][6][5].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_1[0][6][5].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_1[0][6][5].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_2[0][6][5].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_2[0][6][5].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_3[0][6][5].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 2] * c_Quadrature_Filter_3[0][6][5].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_1[6][5][5].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_1[6][5][5].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_2[6][5][5].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_2[6][5][5].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_3[6][5][5].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_3[6][5][5].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_1[5][5][5].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_1[5][5][5].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_2[5][5][5].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_2[5][5][5].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_3[5][5][5].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_3[5][5][5].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_1[4][5][5].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_1[4][5][5].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_2[4][5][5].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_2[4][5][5].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_3[4][5][5].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_3[4][5][5].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_1[3][5][5].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_1[3][5][5].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_2[3][5][5].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_2[3][5][5].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_3[3][5][5].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_3[3][5][5].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_1[2][5][5].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_1[2][5][5].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_2[2][5][5].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_2[2][5][5].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_3[2][5][5].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_3[2][5][5].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_1[1][5][5].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_1[1][5][5].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_2[1][5][5].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_2[1][5][5].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_3[1][5][5].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_3[1][5][5].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_1[0][5][5].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_1[0][5][5].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_2[0][5][5].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_2[0][5][5].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_3[0][5][5].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 2] * c_Quadrature_Filter_3[0][5][5].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_1[6][4][5].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_1[6][4][5].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_2[6][4][5].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_2[6][4][5].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_3[6][4][5].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_3[6][4][5].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_1[5][4][5].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_1[5][4][5].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_2[5][4][5].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_2[5][4][5].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_3[5][4][5].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_3[5][4][5].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_1[4][4][5].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_1[4][4][5].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_2[4][4][5].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_2[4][4][5].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_3[4][4][5].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_3[4][4][5].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_1[3][4][5].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_1[3][4][5].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_2[3][4][5].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_2[3][4][5].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_3[3][4][5].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_3[3][4][5].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_1[2][4][5].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_1[2][4][5].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_2[2][4][5].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_2[2][4][5].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_3[2][4][5].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_3[2][4][5].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_1[1][4][5].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_1[1][4][5].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_2[1][4][5].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_2[1][4][5].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_3[1][4][5].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_3[1][4][5].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_1[0][4][5].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_1[0][4][5].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_2[0][4][5].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_2[0][4][5].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_3[0][4][5].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 2] * c_Quadrature_Filter_3[0][4][5].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_1[6][3][5].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_1[6][3][5].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_2[6][3][5].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_2[6][3][5].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_3[6][3][5].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_3[6][3][5].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_1[5][3][5].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_1[5][3][5].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_2[5][3][5].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_2[5][3][5].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_3[5][3][5].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_3[5][3][5].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_1[4][3][5].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_1[4][3][5].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_2[4][3][5].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_2[4][3][5].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_3[4][3][5].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_3[4][3][5].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_1[3][3][5].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_1[3][3][5].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_2[3][3][5].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_2[3][3][5].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_3[3][3][5].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_3[3][3][5].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_1[2][3][5].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_1[2][3][5].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_2[2][3][5].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_2[2][3][5].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_3[2][3][5].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_3[2][3][5].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_1[1][3][5].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_1[1][3][5].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_2[1][3][5].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_2[1][3][5].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_3[1][3][5].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_3[1][3][5].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_1[0][3][5].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_1[0][3][5].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_2[0][3][5].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_2[0][3][5].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_3[0][3][5].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 2] * c_Quadrature_Filter_3[0][3][5].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_1[6][2][5].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_1[6][2][5].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_2[6][2][5].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_2[6][2][5].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_3[6][2][5].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_3[6][2][5].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_1[5][2][5].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_1[5][2][5].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_2[5][2][5].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_2[5][2][5].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_3[5][2][5].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_3[5][2][5].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_1[4][2][5].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_1[4][2][5].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_2[4][2][5].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_2[4][2][5].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_3[4][2][5].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_3[4][2][5].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_1[3][2][5].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_1[3][2][5].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_2[3][2][5].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_2[3][2][5].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_3[3][2][5].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_3[3][2][5].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_1[2][2][5].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_1[2][2][5].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_2[2][2][5].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_2[2][2][5].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_3[2][2][5].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_3[2][2][5].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_1[1][2][5].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_1[1][2][5].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_2[1][2][5].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_2[1][2][5].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_3[1][2][5].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_3[1][2][5].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_1[0][2][5].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_1[0][2][5].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_2[0][2][5].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_2[0][2][5].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_3[0][2][5].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 2] * c_Quadrature_Filter_3[0][2][5].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_1[6][1][5].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_1[6][1][5].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_2[6][1][5].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_2[6][1][5].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_3[6][1][5].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_3[6][1][5].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_1[5][1][5].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_1[5][1][5].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_2[5][1][5].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_2[5][1][5].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_3[5][1][5].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_3[5][1][5].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_1[4][1][5].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_1[4][1][5].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_2[4][1][5].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_2[4][1][5].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_3[4][1][5].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_3[4][1][5].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_1[3][1][5].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_1[3][1][5].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_2[3][1][5].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_2[3][1][5].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_3[3][1][5].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_3[3][1][5].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_1[2][1][5].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_1[2][1][5].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_2[2][1][5].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_2[2][1][5].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_3[2][1][5].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_3[2][1][5].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_1[1][1][5].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_1[1][1][5].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_2[1][1][5].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_2[1][1][5].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_3[1][1][5].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_3[1][1][5].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_1[0][1][5].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_1[0][1][5].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_2[0][1][5].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_2[0][1][5].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_3[0][1][5].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 2] * c_Quadrature_Filter_3[0][1][5].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_1[6][0][5].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_1[6][0][5].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_2[6][0][5].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_2[6][0][5].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_3[6][0][5].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_3[6][0][5].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_1[5][0][5].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_1[5][0][5].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_2[5][0][5].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_2[5][0][5].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_3[5][0][5].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_3[5][0][5].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_1[4][0][5].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_1[4][0][5].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_2[4][0][5].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_2[4][0][5].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_3[4][0][5].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_3[4][0][5].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_1[3][0][5].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_1[3][0][5].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_2[3][0][5].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_2[3][0][5].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_3[3][0][5].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_3[3][0][5].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_1[2][0][5].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_1[2][0][5].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_2[2][0][5].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_2[2][0][5].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_3[2][0][5].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_3[2][0][5].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_1[1][0][5].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_1[1][0][5].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_2[1][0][5].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_2[1][0][5].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_3[1][0][5].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_3[1][0][5].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_1[0][0][5].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_1[0][0][5].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_2[0][0][5].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_2[0][0][5].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_3[0][0][5].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 2] * c_Quadrature_Filter_3[0][0][5].y;


            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_1[6][6][4].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_1[6][6][4].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_2[6][6][4].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_2[6][6][4].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_3[6][6][4].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_3[6][6][4].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_1[5][6][4].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_1[5][6][4].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_2[5][6][4].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_2[5][6][4].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_3[5][6][4].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_3[5][6][4].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_1[4][6][4].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_1[4][6][4].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_2[4][6][4].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_2[4][6][4].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_3[4][6][4].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_3[4][6][4].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_1[3][6][4].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_1[3][6][4].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_2[3][6][4].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_2[3][6][4].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_3[3][6][4].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_3[3][6][4].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_1[2][6][4].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_1[2][6][4].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_2[2][6][4].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_2[2][6][4].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_3[2][6][4].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_3[2][6][4].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_1[1][6][4].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_1[1][6][4].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_2[1][6][4].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_2[1][6][4].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_3[1][6][4].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_3[1][6][4].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_1[0][6][4].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_1[0][6][4].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_2[0][6][4].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_2[0][6][4].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_3[0][6][4].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 3] * c_Quadrature_Filter_3[0][6][4].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_1[6][5][4].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_1[6][5][4].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_2[6][5][4].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_2[6][5][4].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_3[6][5][4].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_3[6][5][4].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_1[5][5][4].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_1[5][5][4].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_2[5][5][4].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_2[5][5][4].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_3[5][5][4].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_3[5][5][4].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_1[4][5][4].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_1[4][5][4].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_2[4][5][4].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_2[4][5][4].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_3[4][5][4].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_3[4][5][4].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_1[3][5][4].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_1[3][5][4].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_2[3][5][4].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_2[3][5][4].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_3[3][5][4].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_3[3][5][4].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_1[2][5][4].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_1[2][5][4].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_2[2][5][4].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_2[2][5][4].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_3[2][5][4].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_3[2][5][4].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_1[1][5][4].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_1[1][5][4].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_2[1][5][4].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_2[1][5][4].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_3[1][5][4].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_3[1][5][4].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_1[0][5][4].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_1[0][5][4].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_2[0][5][4].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_2[0][5][4].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_3[0][5][4].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 3] * c_Quadrature_Filter_3[0][5][4].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_1[6][4][4].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_1[6][4][4].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_2[6][4][4].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_2[6][4][4].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_3[6][4][4].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_3[6][4][4].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_1[5][4][4].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_1[5][4][4].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_2[5][4][4].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_2[5][4][4].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_3[5][4][4].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_3[5][4][4].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_1[4][4][4].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_1[4][4][4].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_2[4][4][4].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_2[4][4][4].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_3[4][4][4].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_3[4][4][4].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_1[3][4][4].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_1[3][4][4].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_2[3][4][4].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_2[3][4][4].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_3[3][4][4].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_3[3][4][4].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_1[2][4][4].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_1[2][4][4].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_2[2][4][4].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_2[2][4][4].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_3[2][4][4].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_3[2][4][4].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_1[1][4][4].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_1[1][4][4].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_2[1][4][4].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_2[1][4][4].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_3[1][4][4].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_3[1][4][4].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_1[0][4][4].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_1[0][4][4].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_2[0][4][4].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_2[0][4][4].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_3[0][4][4].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 3] * c_Quadrature_Filter_3[0][4][4].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_1[6][3][4].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_1[6][3][4].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_2[6][3][4].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_2[6][3][4].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_3[6][3][4].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_3[6][3][4].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_1[5][3][4].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_1[5][3][4].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_2[5][3][4].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_2[5][3][4].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_3[5][3][4].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_3[5][3][4].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_1[4][3][4].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_1[4][3][4].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_2[4][3][4].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_2[4][3][4].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_3[4][3][4].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_3[4][3][4].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_1[3][3][4].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_1[3][3][4].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_2[3][3][4].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_2[3][3][4].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_3[3][3][4].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_3[3][3][4].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_1[2][3][4].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_1[2][3][4].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_2[2][3][4].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_2[2][3][4].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_3[2][3][4].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_3[2][3][4].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_1[1][3][4].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_1[1][3][4].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_2[1][3][4].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_2[1][3][4].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_3[1][3][4].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_3[1][3][4].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_1[0][3][4].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_1[0][3][4].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_2[0][3][4].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_2[0][3][4].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_3[0][3][4].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 3] * c_Quadrature_Filter_3[0][3][4].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_1[6][2][4].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_1[6][2][4].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_2[6][2][4].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_2[6][2][4].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_3[6][2][4].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_3[6][2][4].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_1[5][2][4].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_1[5][2][4].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_2[5][2][4].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_2[5][2][4].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_3[5][2][4].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_3[5][2][4].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_1[4][2][4].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_1[4][2][4].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_2[4][2][4].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_2[4][2][4].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_3[4][2][4].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_3[4][2][4].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_1[3][2][4].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_1[3][2][4].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_2[3][2][4].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_2[3][2][4].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_3[3][2][4].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_3[3][2][4].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_1[2][2][4].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_1[2][2][4].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_2[2][2][4].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_2[2][2][4].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_3[2][2][4].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_3[2][2][4].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_1[1][2][4].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_1[1][2][4].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_2[1][2][4].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_2[1][2][4].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_3[1][2][4].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_3[1][2][4].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_1[0][2][4].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_1[0][2][4].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_2[0][2][4].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_2[0][2][4].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_3[0][2][4].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 3] * c_Quadrature_Filter_3[0][2][4].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_1[6][1][4].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_1[6][1][4].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_2[6][1][4].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_2[6][1][4].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_3[6][1][4].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_3[6][1][4].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_1[5][1][4].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_1[5][1][4].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_2[5][1][4].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_2[5][1][4].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_3[5][1][4].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_3[5][1][4].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_1[4][1][4].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_1[4][1][4].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_2[4][1][4].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_2[4][1][4].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_3[4][1][4].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_3[4][1][4].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_1[3][1][4].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_1[3][1][4].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_2[3][1][4].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_2[3][1][4].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_3[3][1][4].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_3[3][1][4].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_1[2][1][4].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_1[2][1][4].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_2[2][1][4].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_2[2][1][4].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_3[2][1][4].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_3[2][1][4].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_1[1][1][4].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_1[1][1][4].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_2[1][1][4].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_2[1][1][4].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_3[1][1][4].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_3[1][1][4].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_1[0][1][4].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_1[0][1][4].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_2[0][1][4].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_2[0][1][4].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_3[0][1][4].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 3] * c_Quadrature_Filter_3[0][1][4].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_1[6][0][4].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_1[6][0][4].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_2[6][0][4].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_2[6][0][4].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_3[6][0][4].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_3[6][0][4].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_1[5][0][4].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_1[5][0][4].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_2[5][0][4].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_2[5][0][4].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_3[5][0][4].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_3[5][0][4].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_1[4][0][4].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_1[4][0][4].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_2[4][0][4].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_2[4][0][4].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_3[4][0][4].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_3[4][0][4].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_1[3][0][4].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_1[3][0][4].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_2[3][0][4].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_2[3][0][4].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_3[3][0][4].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_3[3][0][4].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_1[2][0][4].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_1[2][0][4].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_2[2][0][4].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_2[2][0][4].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_3[2][0][4].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_3[2][0][4].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_1[1][0][4].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_1[1][0][4].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_2[1][0][4].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_2[1][0][4].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_3[1][0][4].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_3[1][0][4].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_1[0][0][4].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_1[0][0][4].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_2[0][0][4].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_2[0][0][4].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_3[0][0][4].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 3] * c_Quadrature_Filter_3[0][0][4].y;

            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_1[6][6][3].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_1[6][6][3].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_2[6][6][3].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_2[6][6][3].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_3[6][6][3].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_3[6][6][3].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_1[5][6][3].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_1[5][6][3].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_2[5][6][3].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_2[5][6][3].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_3[5][6][3].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_3[5][6][3].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_1[4][6][3].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_1[4][6][3].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_2[4][6][3].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_2[4][6][3].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_3[4][6][3].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_3[4][6][3].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_1[3][6][3].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_1[3][6][3].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_2[3][6][3].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_2[3][6][3].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_3[3][6][3].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_3[3][6][3].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_1[2][6][3].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_1[2][6][3].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_2[2][6][3].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_2[2][6][3].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_3[2][6][3].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_3[2][6][3].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_1[1][6][3].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_1[1][6][3].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_2[1][6][3].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_2[1][6][3].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_3[1][6][3].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_3[1][6][3].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_1[0][6][3].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_1[0][6][3].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_2[0][6][3].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_2[0][6][3].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_3[0][6][3].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 4] * c_Quadrature_Filter_3[0][6][3].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_1[6][5][3].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_1[6][5][3].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_2[6][5][3].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_2[6][5][3].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_3[6][5][3].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_3[6][5][3].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_1[5][5][3].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_1[5][5][3].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_2[5][5][3].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_2[5][5][3].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_3[5][5][3].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_3[5][5][3].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_1[4][5][3].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_1[4][5][3].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_2[4][5][3].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_2[4][5][3].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_3[4][5][3].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_3[4][5][3].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_1[3][5][3].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_1[3][5][3].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_2[3][5][3].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_2[3][5][3].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_3[3][5][3].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_3[3][5][3].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_1[2][5][3].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_1[2][5][3].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_2[2][5][3].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_2[2][5][3].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_3[2][5][3].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_3[2][5][3].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_1[1][5][3].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_1[1][5][3].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_2[1][5][3].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_2[1][5][3].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_3[1][5][3].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_3[1][5][3].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_1[0][5][3].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_1[0][5][3].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_2[0][5][3].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_2[0][5][3].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_3[0][5][3].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 4] * c_Quadrature_Filter_3[0][5][3].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_1[6][4][3].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_1[6][4][3].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_2[6][4][3].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_2[6][4][3].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_3[6][4][3].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_3[6][4][3].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_1[5][4][3].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_1[5][4][3].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_2[5][4][3].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_2[5][4][3].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_3[5][4][3].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_3[5][4][3].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_1[4][4][3].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_1[4][4][3].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_2[4][4][3].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_2[4][4][3].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_3[4][4][3].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_3[4][4][3].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_1[3][4][3].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_1[3][4][3].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_2[3][4][3].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_2[3][4][3].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_3[3][4][3].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_3[3][4][3].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_1[2][4][3].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_1[2][4][3].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_2[2][4][3].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_2[2][4][3].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_3[2][4][3].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_3[2][4][3].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_1[1][4][3].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_1[1][4][3].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_2[1][4][3].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_2[1][4][3].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_3[1][4][3].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_3[1][4][3].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_1[0][4][3].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_1[0][4][3].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_2[0][4][3].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_2[0][4][3].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_3[0][4][3].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 4] * c_Quadrature_Filter_3[0][4][3].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_1[6][3][3].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_1[6][3][3].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_2[6][3][3].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_2[6][3][3].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_3[6][3][3].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_3[6][3][3].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_1[5][3][3].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_1[5][3][3].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_2[5][3][3].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_2[5][3][3].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_3[5][3][3].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_3[5][3][3].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_1[4][3][3].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_1[4][3][3].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_2[4][3][3].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_2[4][3][3].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_3[4][3][3].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_3[4][3][3].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_1[3][3][3].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_1[3][3][3].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_2[3][3][3].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_2[3][3][3].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_3[3][3][3].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_3[3][3][3].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_1[2][3][3].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_1[2][3][3].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_2[2][3][3].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_2[2][3][3].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_3[2][3][3].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_3[2][3][3].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_1[1][3][3].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_1[1][3][3].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_2[1][3][3].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_2[1][3][3].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_3[1][3][3].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_3[1][3][3].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_1[0][3][3].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_1[0][3][3].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_2[0][3][3].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_2[0][3][3].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_3[0][3][3].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 4] * c_Quadrature_Filter_3[0][3][3].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_1[6][2][3].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_1[6][2][3].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_2[6][2][3].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_2[6][2][3].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_3[6][2][3].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_3[6][2][3].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_1[5][2][3].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_1[5][2][3].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_2[5][2][3].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_2[5][2][3].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_3[5][2][3].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_3[5][2][3].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_1[4][2][3].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_1[4][2][3].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_2[4][2][3].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_2[4][2][3].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_3[4][2][3].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_3[4][2][3].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_1[3][2][3].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_1[3][2][3].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_2[3][2][3].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_2[3][2][3].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_3[3][2][3].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_3[3][2][3].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_1[2][2][3].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_1[2][2][3].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_2[2][2][3].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_2[2][2][3].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_3[2][2][3].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_3[2][2][3].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_1[1][2][3].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_1[1][2][3].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_2[1][2][3].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_2[1][2][3].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_3[1][2][3].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_3[1][2][3].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_1[0][2][3].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_1[0][2][3].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_2[0][2][3].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_2[0][2][3].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_3[0][2][3].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 4] * c_Quadrature_Filter_3[0][2][3].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_1[6][1][3].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_1[6][1][3].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_2[6][1][3].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_2[6][1][3].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_3[6][1][3].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_3[6][1][3].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_1[5][1][3].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_1[5][1][3].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_2[5][1][3].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_2[5][1][3].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_3[5][1][3].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_3[5][1][3].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_1[4][1][3].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_1[4][1][3].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_2[4][1][3].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_2[4][1][3].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_3[4][1][3].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_3[4][1][3].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_1[3][1][3].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_1[3][1][3].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_2[3][1][3].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_2[3][1][3].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_3[3][1][3].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_3[3][1][3].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_1[2][1][3].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_1[2][1][3].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_2[2][1][3].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_2[2][1][3].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_3[2][1][3].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_3[2][1][3].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_1[1][1][3].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_1[1][1][3].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_2[1][1][3].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_2[1][1][3].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_3[1][1][3].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_3[1][1][3].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_1[0][1][3].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_1[0][1][3].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_2[0][1][3].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_2[0][1][3].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_3[0][1][3].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 4] * c_Quadrature_Filter_3[0][1][3].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_1[6][0][3].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_1[6][0][3].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_2[6][0][3].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_2[6][0][3].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_3[6][0][3].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_3[6][0][3].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_1[5][0][3].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_1[5][0][3].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_2[5][0][3].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_2[5][0][3].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_3[5][0][3].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_3[5][0][3].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_1[4][0][3].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_1[4][0][3].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_2[4][0][3].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_2[4][0][3].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_3[4][0][3].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_3[4][0][3].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_1[3][0][3].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_1[3][0][3].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_2[3][0][3].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_2[3][0][3].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_3[3][0][3].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_3[3][0][3].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_1[2][0][3].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_1[2][0][3].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_2[2][0][3].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_2[2][0][3].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_3[2][0][3].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_3[2][0][3].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_1[1][0][3].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_1[1][0][3].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_2[1][0][3].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_2[1][0][3].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_3[1][0][3].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_3[1][0][3].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_1[0][0][3].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_1[0][0][3].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_2[0][0][3].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_2[0][0][3].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_3[0][0][3].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 4] * c_Quadrature_Filter_3[0][0][3].y;

            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_1[6][6][2].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_1[6][6][2].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_2[6][6][2].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_2[6][6][2].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_3[6][6][2].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_3[6][6][2].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_1[5][6][2].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_1[5][6][2].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_2[5][6][2].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_2[5][6][2].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_3[5][6][2].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_3[5][6][2].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_1[4][6][2].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_1[4][6][2].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_2[4][6][2].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_2[4][6][2].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_3[4][6][2].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_3[4][6][2].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_1[3][6][2].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_1[3][6][2].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_2[3][6][2].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_2[3][6][2].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_3[3][6][2].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_3[3][6][2].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_1[2][6][2].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_1[2][6][2].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_2[2][6][2].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_2[2][6][2].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_3[2][6][2].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_3[2][6][2].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_1[1][6][2].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_1[1][6][2].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_2[1][6][2].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_2[1][6][2].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_3[1][6][2].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_3[1][6][2].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_1[0][6][2].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_1[0][6][2].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_2[0][6][2].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_2[0][6][2].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_3[0][6][2].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 5] * c_Quadrature_Filter_3[0][6][2].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_1[6][5][2].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_1[6][5][2].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_2[6][5][2].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_2[6][5][2].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_3[6][5][2].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_3[6][5][2].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_1[5][5][2].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_1[5][5][2].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_2[5][5][2].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_2[5][5][2].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_3[5][5][2].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_3[5][5][2].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_1[4][5][2].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_1[4][5][2].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_2[4][5][2].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_2[4][5][2].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_3[4][5][2].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_3[4][5][2].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_1[3][5][2].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_1[3][5][2].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_2[3][5][2].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_2[3][5][2].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_3[3][5][2].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_3[3][5][2].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_1[2][5][2].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_1[2][5][2].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_2[2][5][2].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_2[2][5][2].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_3[2][5][2].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_3[2][5][2].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_1[1][5][2].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_1[1][5][2].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_2[1][5][2].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_2[1][5][2].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_3[1][5][2].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_3[1][5][2].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_1[0][5][2].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_1[0][5][2].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_2[0][5][2].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_2[0][5][2].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_3[0][5][2].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 5] * c_Quadrature_Filter_3[0][5][2].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_1[6][4][2].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_1[6][4][2].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_2[6][4][2].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_2[6][4][2].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_3[6][4][2].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_3[6][4][2].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_1[5][4][2].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_1[5][4][2].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_2[5][4][2].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_2[5][4][2].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_3[5][4][2].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_3[5][4][2].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_1[4][4][2].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_1[4][4][2].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_2[4][4][2].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_2[4][4][2].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_3[4][4][2].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_3[4][4][2].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_1[3][4][2].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_1[3][4][2].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_2[3][4][2].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_2[3][4][2].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_3[3][4][2].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_3[3][4][2].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_1[2][4][2].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_1[2][4][2].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_2[2][4][2].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_2[2][4][2].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_3[2][4][2].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_3[2][4][2].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_1[1][4][2].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_1[1][4][2].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_2[1][4][2].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_2[1][4][2].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_3[1][4][2].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_3[1][4][2].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_1[0][4][2].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_1[0][4][2].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_2[0][4][2].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_2[0][4][2].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_3[0][4][2].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 5] * c_Quadrature_Filter_3[0][4][2].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_1[6][3][2].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_1[6][3][2].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_2[6][3][2].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_2[6][3][2].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_3[6][3][2].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_3[6][3][2].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_1[5][3][2].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_1[5][3][2].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_2[5][3][2].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_2[5][3][2].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_3[5][3][2].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_3[5][3][2].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_1[4][3][2].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_1[4][3][2].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_2[4][3][2].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_2[4][3][2].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_3[4][3][2].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_3[4][3][2].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_1[3][3][2].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_1[3][3][2].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_2[3][3][2].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_2[3][3][2].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_3[3][3][2].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_3[3][3][2].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_1[2][3][2].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_1[2][3][2].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_2[2][3][2].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_2[2][3][2].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_3[2][3][2].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_3[2][3][2].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_1[1][3][2].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_1[1][3][2].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_2[1][3][2].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_2[1][3][2].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_3[1][3][2].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_3[1][3][2].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_1[0][3][2].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_1[0][3][2].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_2[0][3][2].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_2[0][3][2].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_3[0][3][2].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 5] * c_Quadrature_Filter_3[0][3][2].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_1[6][2][2].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_1[6][2][2].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_2[6][2][2].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_2[6][2][2].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_3[6][2][2].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_3[6][2][2].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_1[5][2][2].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_1[5][2][2].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_2[5][2][2].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_2[5][2][2].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_3[5][2][2].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_3[5][2][2].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_1[4][2][2].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_1[4][2][2].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_2[4][2][2].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_2[4][2][2].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_3[4][2][2].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_3[4][2][2].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_1[3][2][2].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_1[3][2][2].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_2[3][2][2].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_2[3][2][2].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_3[3][2][2].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_3[3][2][2].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_1[2][2][2].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_1[2][2][2].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_2[2][2][2].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_2[2][2][2].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_3[2][2][2].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_3[2][2][2].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_1[1][2][2].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_1[1][2][2].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_2[1][2][2].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_2[1][2][2].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_3[1][2][2].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_3[1][2][2].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_1[0][2][2].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_1[0][2][2].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_2[0][2][2].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_2[0][2][2].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_3[0][2][2].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 5] * c_Quadrature_Filter_3[0][2][2].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_1[6][1][2].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_1[6][1][2].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_2[6][1][2].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_2[6][1][2].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_3[6][1][2].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_3[6][1][2].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_1[5][1][2].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_1[5][1][2].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_2[5][1][2].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_2[5][1][2].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_3[5][1][2].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_3[5][1][2].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_1[4][1][2].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_1[4][1][2].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_2[4][1][2].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_2[4][1][2].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_3[4][1][2].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_3[4][1][2].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_1[3][1][2].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_1[3][1][2].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_2[3][1][2].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_2[3][1][2].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_3[3][1][2].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_3[3][1][2].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_1[2][1][2].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_1[2][1][2].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_2[2][1][2].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_2[2][1][2].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_3[2][1][2].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_3[2][1][2].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_1[1][1][2].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_1[1][1][2].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_2[1][1][2].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_2[1][1][2].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_3[1][1][2].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_3[1][1][2].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_1[0][1][2].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_1[0][1][2].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_2[0][1][2].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_2[0][1][2].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_3[0][1][2].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 5] * c_Quadrature_Filter_3[0][1][2].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_1[6][0][2].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_1[6][0][2].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_2[6][0][2].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_2[6][0][2].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_3[6][0][2].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_3[6][0][2].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_1[5][0][2].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_1[5][0][2].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_2[5][0][2].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_2[5][0][2].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_3[5][0][2].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_3[5][0][2].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_1[4][0][2].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_1[4][0][2].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_2[4][0][2].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_2[4][0][2].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_3[4][0][2].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_3[4][0][2].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_1[3][0][2].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_1[3][0][2].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_2[3][0][2].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_2[3][0][2].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_3[3][0][2].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_3[3][0][2].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_1[2][0][2].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_1[2][0][2].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_2[2][0][2].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_2[2][0][2].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_3[2][0][2].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_3[2][0][2].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_1[1][0][2].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_1[1][0][2].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_2[1][0][2].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_2[1][0][2].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_3[1][0][2].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_3[1][0][2].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_1[0][0][2].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_1[0][0][2].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_2[0][0][2].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_2[0][0][2].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_3[0][0][2].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 5] * c_Quadrature_Filter_3[0][0][2].y;

            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_1[6][6][1].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_1[6][6][1].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_2[6][6][1].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_2[6][6][1].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_3[6][6][1].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_3[6][6][1].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_1[5][6][1].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_1[5][6][1].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_2[5][6][1].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_2[5][6][1].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_3[5][6][1].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_3[5][6][1].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_1[4][6][1].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_1[4][6][1].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_2[4][6][1].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_2[4][6][1].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_3[4][6][1].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_3[4][6][1].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_1[3][6][1].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_1[3][6][1].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_2[3][6][1].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_2[3][6][1].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_3[3][6][1].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_3[3][6][1].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_1[2][6][1].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_1[2][6][1].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_2[2][6][1].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_2[2][6][1].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_3[2][6][1].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_3[2][6][1].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_1[1][6][1].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_1[1][6][1].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_2[1][6][1].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_2[1][6][1].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_3[1][6][1].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_3[1][6][1].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_1[0][6][1].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_1[0][6][1].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_2[0][6][1].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_2[0][6][1].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_3[0][6][1].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 6] * c_Quadrature_Filter_3[0][6][1].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_1[6][5][1].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_1[6][5][1].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_2[6][5][1].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_2[6][5][1].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_3[6][5][1].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_3[6][5][1].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_1[5][5][1].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_1[5][5][1].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_2[5][5][1].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_2[5][5][1].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_3[5][5][1].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_3[5][5][1].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_1[4][5][1].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_1[4][5][1].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_2[4][5][1].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_2[4][5][1].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_3[4][5][1].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_3[4][5][1].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_1[3][5][1].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_1[3][5][1].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_2[3][5][1].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_2[3][5][1].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_3[3][5][1].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_3[3][5][1].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_1[2][5][1].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_1[2][5][1].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_2[2][5][1].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_2[2][5][1].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_3[2][5][1].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_3[2][5][1].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_1[1][5][1].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_1[1][5][1].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_2[1][5][1].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_2[1][5][1].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_3[1][5][1].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_3[1][5][1].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_1[0][5][1].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_1[0][5][1].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_2[0][5][1].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_2[0][5][1].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_3[0][5][1].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 6] * c_Quadrature_Filter_3[0][5][1].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_1[6][4][1].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_1[6][4][1].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_2[6][4][1].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_2[6][4][1].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_3[6][4][1].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_3[6][4][1].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_1[5][4][1].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_1[5][4][1].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_2[5][4][1].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_2[5][4][1].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_3[5][4][1].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_3[5][4][1].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_1[4][4][1].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_1[4][4][1].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_2[4][4][1].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_2[4][4][1].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_3[4][4][1].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_3[4][4][1].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_1[3][4][1].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_1[3][4][1].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_2[3][4][1].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_2[3][4][1].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_3[3][4][1].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_3[3][4][1].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_1[2][4][1].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_1[2][4][1].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_2[2][4][1].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_2[2][4][1].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_3[2][4][1].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_3[2][4][1].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_1[1][4][1].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_1[1][4][1].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_2[1][4][1].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_2[1][4][1].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_3[1][4][1].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_3[1][4][1].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_1[0][4][1].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_1[0][4][1].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_2[0][4][1].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_2[0][4][1].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_3[0][4][1].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 6] * c_Quadrature_Filter_3[0][4][1].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_1[6][3][1].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_1[6][3][1].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_2[6][3][1].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_2[6][3][1].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_3[6][3][1].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_3[6][3][1].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_1[5][3][1].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_1[5][3][1].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_2[5][3][1].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_2[5][3][1].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_3[5][3][1].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_3[5][3][1].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_1[4][3][1].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_1[4][3][1].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_2[4][3][1].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_2[4][3][1].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_3[4][3][1].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_3[4][3][1].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_1[3][3][1].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_1[3][3][1].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_2[3][3][1].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_2[3][3][1].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_3[3][3][1].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_3[3][3][1].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_1[2][3][1].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_1[2][3][1].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_2[2][3][1].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_2[2][3][1].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_3[2][3][1].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_3[2][3][1].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_1[1][3][1].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_1[1][3][1].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_2[1][3][1].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_2[1][3][1].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_3[1][3][1].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_3[1][3][1].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_1[0][3][1].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_1[0][3][1].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_2[0][3][1].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_2[0][3][1].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_3[0][3][1].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 6] * c_Quadrature_Filter_3[0][3][1].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_1[6][2][1].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_1[6][2][1].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_2[6][2][1].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_2[6][2][1].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_3[6][2][1].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_3[6][2][1].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_1[5][2][1].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_1[5][2][1].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_2[5][2][1].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_2[5][2][1].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_3[5][2][1].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_3[5][2][1].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_1[4][2][1].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_1[4][2][1].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_2[4][2][1].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_2[4][2][1].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_3[4][2][1].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_3[4][2][1].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_1[3][2][1].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_1[3][2][1].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_2[3][2][1].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_2[3][2][1].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_3[3][2][1].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_3[3][2][1].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_1[2][2][1].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_1[2][2][1].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_2[2][2][1].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_2[2][2][1].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_3[2][2][1].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_3[2][2][1].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_1[1][2][1].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_1[1][2][1].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_2[1][2][1].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_2[1][2][1].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_3[1][2][1].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_3[1][2][1].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_1[0][2][1].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_1[0][2][1].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_2[0][2][1].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_2[0][2][1].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_3[0][2][1].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 6] * c_Quadrature_Filter_3[0][2][1].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_1[6][1][1].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_1[6][1][1].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_2[6][1][1].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_2[6][1][1].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_3[6][1][1].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_3[6][1][1].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_1[5][1][1].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_1[5][1][1].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_2[5][1][1].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_2[5][1][1].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_3[5][1][1].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_3[5][1][1].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_1[4][1][1].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_1[4][1][1].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_2[4][1][1].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_2[4][1][1].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_3[4][1][1].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_3[4][1][1].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_1[3][1][1].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_1[3][1][1].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_2[3][1][1].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_2[3][1][1].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_3[3][1][1].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_3[3][1][1].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_1[2][1][1].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_1[2][1][1].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_2[2][1][1].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_2[2][1][1].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_3[2][1][1].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_3[2][1][1].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_1[1][1][1].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_1[1][1][1].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_2[1][1][1].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_2[1][1][1].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_3[1][1][1].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_3[1][1][1].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_1[0][1][1].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_1[0][1][1].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_2[0][1][1].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_2[0][1][1].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_3[0][1][1].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 6] * c_Quadrature_Filter_3[0][1][1].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_1[6][0][1].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_1[6][0][1].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_2[6][0][1].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_2[6][0][1].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_3[6][0][1].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_3[6][0][1].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_1[5][0][1].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_1[5][0][1].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_2[5][0][1].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_2[5][0][1].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_3[5][0][1].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_3[5][0][1].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_1[4][0][1].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_1[4][0][1].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_2[4][0][1].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_2[4][0][1].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_3[4][0][1].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_3[4][0][1].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_1[3][0][1].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_1[3][0][1].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_2[3][0][1].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_2[3][0][1].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_3[3][0][1].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_3[3][0][1].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_1[2][0][1].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_1[2][0][1].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_2[2][0][1].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_2[2][0][1].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_3[2][0][1].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_3[2][0][1].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_1[1][0][1].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_1[1][0][1].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_2[1][0][1].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_2[1][0][1].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_3[1][0][1].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_3[1][0][1].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_1[0][0][1].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_1[0][0][1].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_2[0][0][1].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_2[0][0][1].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_3[0][0][1].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 6] * c_Quadrature_Filter_3[0][0][1].y;

            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_1[6][6][0].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_1[6][6][0].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_2[6][6][0].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_2[6][6][0].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_3[6][6][0].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_3[6][6][0].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_1[5][6][0].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_1[5][6][0].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_2[5][6][0].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_2[5][6][0].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_3[5][6][0].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_3[5][6][0].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_1[4][6][0].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_1[4][6][0].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_2[4][6][0].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_2[4][6][0].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_3[4][6][0].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_3[4][6][0].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_1[3][6][0].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_1[3][6][0].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_2[3][6][0].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_2[3][6][0].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_3[3][6][0].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_3[3][6][0].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_1[2][6][0].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_1[2][6][0].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_2[2][6][0].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_2[2][6][0].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_3[2][6][0].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_3[2][6][0].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_1[1][6][0].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_1[1][6][0].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_2[1][6][0].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_2[1][6][0].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_3[1][6][0].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_3[1][6][0].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_1[0][6][0].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_1[0][6][0].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_2[0][6][0].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_2[0][6][0].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_3[0][6][0].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x + 7] * c_Quadrature_Filter_3[0][6][0].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_1[6][5][0].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_1[6][5][0].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_2[6][5][0].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_2[6][5][0].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_3[6][5][0].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_3[6][5][0].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_1[5][5][0].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_1[5][5][0].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_2[5][5][0].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_2[5][5][0].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_3[5][5][0].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_3[5][5][0].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_1[4][5][0].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_1[4][5][0].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_2[4][5][0].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_2[4][5][0].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_3[4][5][0].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_3[4][5][0].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_1[3][5][0].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_1[3][5][0].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_2[3][5][0].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_2[3][5][0].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_3[3][5][0].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_3[3][5][0].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_1[2][5][0].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_1[2][5][0].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_2[2][5][0].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_2[2][5][0].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_3[2][5][0].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_3[2][5][0].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_1[1][5][0].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_1[1][5][0].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_2[1][5][0].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_2[1][5][0].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_3[1][5][0].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_3[1][5][0].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_1[0][5][0].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_1[0][5][0].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_2[0][5][0].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_2[0][5][0].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_3[0][5][0].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x + 7] * c_Quadrature_Filter_3[0][5][0].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_1[6][4][0].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_1[6][4][0].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_2[6][4][0].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_2[6][4][0].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_3[6][4][0].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_3[6][4][0].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_1[5][4][0].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_1[5][4][0].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_2[5][4][0].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_2[5][4][0].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_3[5][4][0].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_3[5][4][0].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_1[4][4][0].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_1[4][4][0].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_2[4][4][0].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_2[4][4][0].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_3[4][4][0].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_3[4][4][0].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_1[3][4][0].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_1[3][4][0].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_2[3][4][0].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_2[3][4][0].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_3[3][4][0].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_3[3][4][0].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_1[2][4][0].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_1[2][4][0].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_2[2][4][0].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_2[2][4][0].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_3[2][4][0].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_3[2][4][0].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_1[1][4][0].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_1[1][4][0].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_2[1][4][0].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_2[1][4][0].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_3[1][4][0].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_3[1][4][0].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_1[0][4][0].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_1[0][4][0].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_2[0][4][0].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_2[0][4][0].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_3[0][4][0].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x + 7] * c_Quadrature_Filter_3[0][4][0].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_1[6][3][0].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_1[6][3][0].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_2[6][3][0].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_2[6][3][0].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_3[6][3][0].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_3[6][3][0].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_1[5][3][0].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_1[5][3][0].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_2[5][3][0].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_2[5][3][0].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_3[5][3][0].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_3[5][3][0].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_1[4][3][0].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_1[4][3][0].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_2[4][3][0].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_2[4][3][0].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_3[4][3][0].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_3[4][3][0].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_1[3][3][0].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_1[3][3][0].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_2[3][3][0].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_2[3][3][0].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_3[3][3][0].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_3[3][3][0].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_1[2][3][0].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_1[2][3][0].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_2[2][3][0].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_2[2][3][0].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_3[2][3][0].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_3[2][3][0].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_1[1][3][0].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_1[1][3][0].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_2[1][3][0].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_2[1][3][0].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_3[1][3][0].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_3[1][3][0].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_1[0][3][0].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_1[0][3][0].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_2[0][3][0].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_2[0][3][0].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_3[0][3][0].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x + 7] * c_Quadrature_Filter_3[0][3][0].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_1[6][2][0].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_1[6][2][0].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_2[6][2][0].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_2[6][2][0].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_3[6][2][0].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_3[6][2][0].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_1[5][2][0].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_1[5][2][0].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_2[5][2][0].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_2[5][2][0].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_3[5][2][0].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_3[5][2][0].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_1[4][2][0].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_1[4][2][0].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_2[4][2][0].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_2[4][2][0].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_3[4][2][0].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_3[4][2][0].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_1[3][2][0].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_1[3][2][0].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_2[3][2][0].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_2[3][2][0].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_3[3][2][0].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_3[3][2][0].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_1[2][2][0].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_1[2][2][0].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_2[2][2][0].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_2[2][2][0].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_3[2][2][0].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_3[2][2][0].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_1[1][2][0].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_1[1][2][0].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_2[1][2][0].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_2[1][2][0].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_3[1][2][0].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_3[1][2][0].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_1[0][2][0].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_1[0][2][0].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_2[0][2][0].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_2[0][2][0].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_3[0][2][0].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x + 7] * c_Quadrature_Filter_3[0][2][0].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_1[6][1][0].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_1[6][1][0].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_2[6][1][0].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_2[6][1][0].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_3[6][1][0].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_3[6][1][0].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_1[5][1][0].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_1[5][1][0].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_2[5][1][0].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_2[5][1][0].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_3[5][1][0].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_3[5][1][0].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_1[4][1][0].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_1[4][1][0].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_2[4][1][0].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_2[4][1][0].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_3[4][1][0].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_3[4][1][0].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_1[3][1][0].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_1[3][1][0].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_2[3][1][0].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_2[3][1][0].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_3[3][1][0].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_3[3][1][0].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_1[2][1][0].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_1[2][1][0].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_2[2][1][0].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_2[2][1][0].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_3[2][1][0].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_3[2][1][0].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_1[1][1][0].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_1[1][1][0].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_2[1][1][0].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_2[1][1][0].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_3[1][1][0].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_3[1][1][0].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_1[0][1][0].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_1[0][1][0].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_2[0][1][0].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_2[0][1][0].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_3[0][1][0].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x + 7] * c_Quadrature_Filter_3[0][1][0].y;
            sum1.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_1[6][0][0].x;
            sum1.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_1[6][0][0].y;
            sum2.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_2[6][0][0].x;
            sum2.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_2[6][0][0].y;
            sum3.x += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_3[6][0][0].x;
            sum3.y += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_3[6][0][0].y;
            sum1.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_1[5][0][0].x;
            sum1.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_1[5][0][0].y;
            sum2.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_2[5][0][0].x;
            sum2.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_2[5][0][0].y;
            sum3.x += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_3[5][0][0].x;
            sum3.y += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_3[5][0][0].y;
            sum1.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_1[4][0][0].x;
            sum1.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_1[4][0][0].y;
            sum2.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_2[4][0][0].x;
            sum2.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_2[4][0][0].y;
            sum3.x += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_3[4][0][0].x;
            sum3.y += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_3[4][0][0].y;
            sum1.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_1[3][0][0].x;
            sum1.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_1[3][0][0].y;
            sum2.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_2[3][0][0].x;
            sum2.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_2[3][0][0].y;
            sum3.x += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_3[3][0][0].x;
            sum3.y += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_3[3][0][0].y;
            sum1.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_1[2][0][0].x;
            sum1.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_1[2][0][0].y;
            sum2.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_2[2][0][0].x;
            sum2.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_2[2][0][0].y;
            sum3.x += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_3[2][0][0].x;
            sum3.y += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_3[2][0][0].y;
            sum1.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_1[1][0][0].x;
            sum1.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_1[1][0][0].y;
            sum2.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_2[1][0][0].x;
            sum2.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_2[1][0][0].y;
            sum3.x += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_3[1][0][0].x;
            sum3.y += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_3[1][0][0].y;
            sum1.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_1[0][0][0].x;
            sum1.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_1[0][0][0].y;
            sum2.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_2[0][0][0].x;
            sum2.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_2[0][0][0].y;
            sum3.x += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_3[0][0][0].x;
            sum3.y += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x + 7] * c_Quadrature_Filter_3[0][0][0].y;


			Filter_Response_1[Calculate_3D_Index(x,y,z,DATA_W,DATA_H)] = sum1;
			Filter_Response_2[Calculate_3D_Index(x,y,z,DATA_W,DATA_H)] = sum2;
			Filter_Response_3[Calculate_3D_Index(x,y,z,DATA_W,DATA_H)] = sum3;
	}	
}

// Functions for motion correction

__kernel void CalculatePhaseDifferencesAndCertainties(__global float* Phase_Differences, __global float* Certainties, __global const float2* q11, __global const float2* q21, int DATA_W, int DATA_H, int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	
	if ( (x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D))
		return;

	int idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

	float2 complex_product;
	float a, b, c, d, phase_difference;

	// q1 = a + i * b
	// q2 = c + i * d
	a = q11[idx].x;
	b = q11[idx].y;
	c = q21[idx].x;
	d = q21[idx].y;

	// phase difference = arg (q1 * (complex conjugate of q2))
	complex_product.x = a * c + b * d;
	complex_product.y = b * c - a * d;

	phase_difference = atan2f(complex_product.y, complex_product.x);

	complex_product.x = a * c - b * d;
  	complex_product.y = b * c + a * d;

	c = __cosf( phase_difference * 0.5f );
	Phase_Differences[idx] = phase_difference;
	Certainties[idx] = sqrtf(complex_product.x * complex_product.x + complex_product.y * complex_product.y) * c * c;
}

__kernel void CalculatePhaseGradientsX_(__global float* Phase_Gradients, __global const float2* q11, __global const float2* q21, int DATA_W, int DATA_H, int DATA_D)
{
	int x = get_global_id(0);
	int y = get_glboal_id(1);
	int z = get_global_id(2);
	
	if ( (x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D))
			return;

	float2 total_complex_product;
	float a, b, c, d;
	int idx_minus_1, idx_plus_1, idx;

	idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

	// X
	idx_minus_1 = Calculate_3D_Index(x - 1, y, z, DATA_W, DATA_H);
	idx_plus_1 = Calculate_3D_Index(x + 1, y, z, DATA_W, DATA_H);

	total_complex_product.x = 0.0f;
	total_complex_product.y = 0.0f;

	a = q11[idx_plus_1].x;
	b = q11[idx_plus_1].y;
	c = q11[idx].x;
	d = q11[idx].y;

	total_complex_product.x += a * c + b * d;
	total_complex_product.y += b * c - a * d;

	a = c;
	b = d;
	c = q11[idx_minus_1].x;
	d = q11[idx_minus_1].y;

	total_complex_product.x += a * c + b * d;
	total_complex_product.y += b * c - a * d;

	a = q21[idx_plus_1].x;
	b = q21[idx_plus_1].y;
	c = q21[idx].x;
	d = q21[idx].y;

	total_complex_product.x += a * c + b * d;
	total_complex_product.y += b * c - a * d;

	a = c;
	b = d;
	c = q21[idx_minus_1].x;
	d = q21[idx_minus_1].y;

	total_complex_product.x += a * c + b * d;
	total_complex_product.y += b * c - a * d;

	Phase_Gradients[idx] = atan2f(total_complex_product.y, total_complex_product.x);
}

__kernel void CalculatePhaseGradientsY_(__global float* Phase_Gradients, __global const float2* q12, __global const float2* q22, int DATA_W, int DATA_H, int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	
	if ( (x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D))
			return;

	float2 total_complex_product;
	float a, b, c, d;
	int idx_minus_1, idx_plus_1, idx;

	idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

	// Y

	idx_plus_1 =  Calculate_3D_Index(x, y + 1, z, DATA_W, DATA_H);
	idx_minus_1 =  Calculate_3D_Index(x, y - 1, z, DATA_W, DATA_H);

	total_complex_product.x = 0.0f;
	total_complex_product.y = 0.0f;

	a = q12[idx_plus_1].x;
	b = q12[idx_plus_1].y;
	c = q12[idx].x;
	d = q12[idx].y;

	total_complex_product.x += a * c + b * d;
	total_complex_product.y += b * c - a * d;

	a = c;
	b = d;
	c = q12[idx_minus_1].x;
	d = q12[idx_minus_1].y;

	total_complex_product.x += a * c + b * d;
	total_complex_product.y += b * c - a * d;

	a = q22[idx_plus_1].x;
	b = q22[idx_plus_1].y;
	c = q22[idx].x;
	d = q22[idx].y;
	
	total_complex_product.x += a * c + b * d;
	total_complex_product.y += b * c - a * d;

	a = c;
	b = d;
	c = q22[idx_minus_1].x;
	d = q22[idx_minus_1].y;

	total_complex_product.x += a * c + b * d;
	total_complex_product.y += b * c - a * d;

	Phase_Gradients[idx] = atan2f(total_complex_product.y, total_complex_product.x);
}

__kernel void CalculatePhaseGradientsZ_(__global float* Phase_Gradients, __global const float2* q13, __global const float2* q23, int DATA_W, int DATA_H, int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	
	if ( (x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D))
			return;

	float2 total_complex_product;
	float a, b, c, d;
	int idx_minus_1, idx_plus_1, idx;

	idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

	// Z

	idx_plus_1 = Calculate_3D_Index(x, y, z + 1, DATA_W, DATA_H);
	idx_minus_1 = Calculate_3D_Index(x, y, z - 1, DATA_W, DATA_H);

	total_complex_product.x = 0.0f;
	total_complex_product.y = 0.0f;

	a = q13[idx_plus_1].x;
	b = q13[idx_plus_1].y;
	c = q13[idx].x;
	d = q13[idx].y;

	total_complex_product.x += a * c + b * d;
	total_complex_product.y += b * c - a * d;

	a = c;
	b = d;
	c = q13[idx_minus_1].x;
	d = q13[idx_minus_1].y;

	total_complex_product.x += a * c + b * d;
	total_complex_product.y += b * c - a * d;

	a = q23[idx_plus_1].x;
	b = q23[idx_plus_1].y;
	c = q23[idx].x;
	d = q23[idx].y;

	total_complex_product.x += a * c + b * d;
	total_complex_product.y += b * c - a * d;

	a = c;
	b = d;
	c = q23[idx_minus_1].x;
	d = q23[idx_minus_1].y;

	total_complex_product.x += a * c + b * d;
	total_complex_product.y += b * c - a * d;

	Phase_Gradients[idx] = atan2f(total_complex_product.y, total_complex_product.x);
}

__kernel void CalculatePhaseGradientsX(__global float* Phase_Gradients, __global const float2* q11, __global const float2* q21, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (((x >= (FILTER_SIZE - 1)/2) && (x < DATA_W - (FILTER_SIZE - 1)/2)) && ((y >= (FILTER_SIZE - 1)/2) && (y < DATA_H - (FILTER_SIZE - 1)/2)) && ((z >= (FILTER_SIZE - 1)/2) && (z < DATA_D - (FILTER_SIZE - 1)/2)))
	{
		float2 total_complex_product;
		float a, b, c, d;
		int idx_minus_1, idx_plus_1, idx;

		idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

		// X
		idx_minus_1 = Calculate_3D_Index(x - 1, y, z, DATA_W, DATA_H);
		idx_plus_1 = Calculate_3D_Index(x + 1, y, z, DATA_W, DATA_H);

		total_complex_product.x = 0.0f;
		total_complex_product.y = 0.0f;

		a = q11[idx_plus_1].x;
		b = q11[idx_plus_1].y;
		c = q11[idx].x;
		d = q11[idx].y;

		total_complex_product.x += a * c + b * d;
		total_complex_product.y += b * c - a * d;

		a = c;
		b = d;
		c = q11[idx_minus_1].x;
		d = q11[idx_minus_1].y;

		total_complex_product.x += a * c + b * d;
		total_complex_product.y += b * c - a * d;

		a = q21[idx_plus_1].x;
		b = q21[idx_plus_1].y;
		c = q21[idx].x;
		d = q21[idx].y;

		total_complex_product.x += a * c + b * d;
		total_complex_product.y += b * c - a * d;

		a = c;
		b = d;
		c = q21[idx_minus_1].x;
		d = q21[idx_minus_1].y;

		total_complex_product.x += a * c + b * d;
		total_complex_product.y += b * c - a * d;

		Phase_Gradients[idx] = atan2f(total_complex_product.y, total_complex_product.x);
	}
}

__kernel void CalculatePhaseGradientsY(__global float* Phase_Gradients, __global const float2* q12, __global const float2* q22, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (((x >= (FILTER_SIZE - 1)/2) && (x < DATA_W - (FILTER_SIZE - 1)/2)) && ((y >= (FILTER_SIZE - 1)/2) && (y < DATA_H - (FILTER_SIZE - 1)/2)) && ((z >= (FILTER_SIZE - 1)/2) && (z < DATA_D - (FILTER_SIZE - 1)/2)))
	{
		float2 total_complex_product;
		float a, b, c, d;
		int idx_minus_1, idx_plus_1, idx;

		idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

		// Y

		idx_plus_1 =  Calculate_3D_Index(x, y + 1, z, DATA_W, DATA_H);
		idx_minus_1 =  Calculate_3D_Index(x, y - 1, z, DATA_W, DATA_H);

		total_complex_product.x = 0.0f;
		total_complex_product.y = 0.0f;

		a = q12[idx_plus_1].x;
		b = q12[idx_plus_1].y;
		c = q12[idx].x;
		d = q12[idx].y;

		total_complex_product.x += a * c + b * d;
		total_complex_product.y += b * c - a * d;

		a = c;
		b = d;
		c = q12[idx_minus_1].x;
		d = q12[idx_minus_1].y;

		total_complex_product.x += a * c + b * d;
		total_complex_product.y += b * c - a * d;

		a = q22[idx_plus_1].x;
		b = q22[idx_plus_1].y;
		c = q22[idx].x;
		d = q22[idx].y;

		total_complex_product.x += a * c + b * d;
		total_complex_product.y += b * c - a * d;

		a = c;
		b = d;
		c = q22[idx_minus_1].x;
		d = q22[idx_minus_1].y;

		total_complex_product.x += a * c + b * d;
		total_complex_product.y += b * c - a * d;

		Phase_Gradients[idx] = atan2f(total_complex_product.y, total_complex_product.x);
	}
}


__kernel void CalculatePhaseGradientsZ(__global float* Phase_Gradients, __global const float2* q13, __global const float2* q23, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (((x >= (FILTER_SIZE - 1)/2) && (x < DATA_W - (FILTER_SIZE - 1)/2)) && ((y >= (FILTER_SIZE - 1)/2) && (y < DATA_H - (FILTER_SIZE - 1)/2)) && ((z >= (FILTER_SIZE - 1)/2) && (z < DATA_D - (FILTER_SIZE - 1)/2)))
	{
		float2 total_complex_product;
		float a, b, c, d;
		int idx_minus_1, idx_plus_1, idx;

		idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

		// Z

		idx_plus_1 = Calculate_3D_Index(x, y, z + 1, DATA_W, DATA_H);
		idx_minus_1 = Calculate_3D_Index(x, y, z - 1, DATA_W, DATA_H);

		total_complex_product.x = 0.0f;
		total_complex_product.y = 0.0f;

		a = q13[idx_plus_1].x;
		b = q13[idx_plus_1].y;
		c = q13[idx].x;
		d = q13[idx].y;

		total_complex_product.x += a * c + b * d;
		total_complex_product.y += b * c - a * d;

		a = c;
		b = d;
		c = q13[idx_minus_1].x;
		d = q13[idx_minus_1].y;

		total_complex_product.x += a * c + b * d;
		total_complex_product.y += b * c - a * d;

		a = q23[idx_plus_1].x;
		b = q23[idx_plus_1].y;
		c = q23[idx].x;
		d = q23[idx].y;

		total_complex_product.x += a * c + b * d;
		total_complex_product.y += b * c - a * d;

		a = c;
		b = d;
		c = q23[idx_minus_1].x;
		d = q23[idx_minus_1].y;

		total_complex_product.x += a * c + b * d;
		total_complex_product.y += b * c - a * d;

		Phase_Gradients[idx] = atan2f(total_complex_product.y, total_complex_product.x);
	}
}


// dimBlock.x = DATA_H; dimBlock.y = 1; dimBlock.z = 1;
// dimGrid.x = DATA_D; dimGrid.y = 1;

__kernel void CalculateAMatrixAndHVector2DValuesX(__global float* A_matrix_2D_values, __global float* h_vector_2D_values, __global const float* Phase_Differences, __global const float* Phase_Gradients, __global const float* Certainties, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)

{
	int y = get_local_id(0);
	int z = get_group_id(0); 
				
	//volatile int y = blockIdx.x * blockDim.x + threadIdx.x;
	//volatile int z = blockIdx.y * blockDim.y + threadIdx.y;

	if (((y >= (FILTER_SIZE - 1)/2) && (y < DATA_H - (FILTER_SIZE - 1)/2)) && ((z >= (FILTER_SIZE - 1)/2) && (z < DATA_D - (FILTER_SIZE - 1)/2)))
	{
		float yf, zf;
		int matrix_element_idx, vector_element_idx;
		float A_matrix_2D_value[10], h_vector_2D_value[4];

    	yf = (float)y - ((float)DATA_H - 1.0f) * 0.5f;
		zf = (float)z - ((float)DATA_D - 1.0f) * 0.5f;

		// X

		A_matrix_2D_value[0] = 0.0f;
		A_matrix_2D_value[1] = 0.0f;
		A_matrix_2D_value[2] = 0.0f;
		A_matrix_2D_value[3] = 0.0f;
		A_matrix_2D_value[4] = 0.0f;
		A_matrix_2D_value[5] = 0.0f;
		A_matrix_2D_value[6] = 0.0f;
		A_matrix_2D_value[7] = 0.0f;
		A_matrix_2D_value[8] = 0.0f;
		A_matrix_2D_value[9] = 0.0f;

		h_vector_2D_value[0] = 0.0f;
		h_vector_2D_value[1] = 0.0f;
		h_vector_2D_value[2] = 0.0f;
		h_vector_2D_value[3] = 0.0f;

		for (int x = (FILTER_SIZE - 1)/2; x < (DATA_W - (FILTER_SIZE - 1)/2); x++)
		{
			float xf = (float)x - ((float)DATA_W - 1.0f) * 0.5f;
			int idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

			float phase_difference = Phase_Differences[idx];
			float phase_gradient = Phase_Gradients[idx];
			float certainty = Certainties[idx];
			float c_pg_pg = certainty * phase_gradient * phase_gradient;
			float c_pg_pd = certainty * phase_gradient * phase_difference;

			A_matrix_2D_value[0] += c_pg_pg;
			A_matrix_2D_value[1] += xf * c_pg_pg;
			A_matrix_2D_value[2] += yf * c_pg_pg;
			A_matrix_2D_value[3] += zf * c_pg_pg;
			A_matrix_2D_value[4] += xf * xf * c_pg_pg;
			A_matrix_2D_value[5] += xf * yf * c_pg_pg;
			A_matrix_2D_value[6] += xf * zf * c_pg_pg;
			A_matrix_2D_value[7] += yf * yf * c_pg_pg;
			A_matrix_2D_value[8] += yf * zf * c_pg_pg;
			A_matrix_2D_value[9] += zf * zf * c_pg_pg;

			h_vector_2D_value[0] += c_pg_pd;
			h_vector_2D_value[1] += xf * c_pg_pd;
			h_vector_2D_value[2] += yf * c_pg_pd;
			h_vector_2D_value[3] += zf * c_pg_pd;
		}

		matrix_element_idx = y + z * DATA_H;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[0];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[1];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[2];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[3];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[4];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[5];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[6];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[7];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[8];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[9];

		vector_element_idx = y + z * DATA_H;
		h_vector_2D_values[vector_element_idx] = h_vector_2D_value[0];
		vector_element_idx += 3 * DATA_H * DATA_D;
		h_vector_2D_values[vector_element_idx] = h_vector_2D_value[1];
		vector_element_idx += DATA_H * DATA_D;
		h_vector_2D_values[vector_element_idx] = h_vector_2D_value[2];
		vector_element_idx += DATA_H * DATA_D;
		h_vector_2D_values[vector_element_idx] = h_vector_2D_value[3];
	}
}

__kernel void CalculateAMatrixAndHVector2DValuesX_(__global float* A_matrix_2D_values, __global float* h_vector_2D_values, __global const float* Phase_Differences, __global const float* Phase_Gradients, __global const float* Certainties, int DATA_W, int DATA_H, int DATA_D)

{
	int y = get_local_id(0);
	int z = get_group_id(0);

	//volatile int y = blockIdx.x * blockDim.x + threadIdx.x;
	//volatile int z = blockIdx.y * blockDim.y + threadIdx.y;

	if ( (y >= DATA_H) || (z >= DATA_D))
			return;

	float yf, zf;
	int matrix_element_idx, vector_element_idx;
	float A_matrix_2D_value[10], h_vector_2D_value[4];

    yf = (float)y - ((float)DATA_H - 1.0f) * 0.5f + 3.0f;
	zf = (float)z - ((float)DATA_D - 1.0f) * 0.5f + 3.0f;

	// X
	
	A_matrix_2D_value[0] = 0.0f;
	A_matrix_2D_value[1] = 0.0f;
	A_matrix_2D_value[2] = 0.0f;
	A_matrix_2D_value[3] = 0.0f;
	A_matrix_2D_value[4] = 0.0f;
	A_matrix_2D_value[5] = 0.0f;
	A_matrix_2D_value[6] = 0.0f;
	A_matrix_2D_value[7] = 0.0f;
	A_matrix_2D_value[8] = 0.0f;
	A_matrix_2D_value[9] = 0.0f;

	h_vector_2D_value[0] = 0.0f;
	h_vector_2D_value[1] = 0.0f;
	h_vector_2D_value[2] = 0.0f;
	h_vector_2D_value[3] = 0.0f;

	for (int x = 0; x < DATA_W; x++)
	{
		float xf = (float)x - ((float)DATA_W - 1.0f) * 0.5f + 3.0f;
		int idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);
		
		float phase_difference = Phase_Differences[idx];
		float phase_gradient = Phase_Gradients[idx];
		float certainty = Certainties[idx];
		float c_pg_pg = certainty * phase_gradient * phase_gradient;
		float c_pg_pd = certainty * phase_gradient * phase_difference;

		A_matrix_2D_value[0] += c_pg_pg;
		A_matrix_2D_value[1] += xf * c_pg_pg;
		A_matrix_2D_value[2] += yf * c_pg_pg;
		A_matrix_2D_value[3] += zf * c_pg_pg;
		A_matrix_2D_value[4] += xf * xf * c_pg_pg;
		A_matrix_2D_value[5] += xf * yf * c_pg_pg;
		A_matrix_2D_value[6] += xf * zf * c_pg_pg;
		A_matrix_2D_value[7] += yf * yf * c_pg_pg;
		A_matrix_2D_value[8] += yf * zf * c_pg_pg;
		A_matrix_2D_value[9] += zf * zf * c_pg_pg;

		h_vector_2D_value[0] += c_pg_pd;
		h_vector_2D_value[1] += xf * c_pg_pd;
		h_vector_2D_value[2] += yf * c_pg_pd;
		h_vector_2D_value[3] += zf * c_pg_pd;
	}

	matrix_element_idx = y + z * DATA_H;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[0];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[1];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[2];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[3];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[4];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[5];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[6];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[7];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[8];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[9];

	vector_element_idx = y + z * DATA_H;
	h_vector_2D_values[vector_element_idx] = h_vector_2D_value[0];
	vector_element_idx += 3 * DATA_H * DATA_D;
	h_vector_2D_values[vector_element_idx] = h_vector_2D_value[1];
	vector_element_idx += DATA_H * DATA_D;
	h_vector_2D_values[vector_element_idx] = h_vector_2D_value[2];
	vector_element_idx += DATA_H * DATA_D;
	h_vector_2D_values[vector_element_idx] = h_vector_2D_value[3];
}

__kernel void CalculateAMatrixAndHVector2DValuesY(__global float* A_matrix_2D_values, __global float* h_vector_2D_values, __glocal const float* Phase_Differences, __global const float* Phase_Gradients, __global const float* Certainties, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)
{
	int y = get_local_id(0);
	int z = get_group_id(0);

	if (((y >= (FILTER_SIZE - 1)/2) && (y < DATA_H - (FILTER_SIZE - 1)/2)) && ((z >= (FILTER_SIZE - 1)/2) && (z < DATA_D - (FILTER_SIZE - 1)/2)))
	{
		float yf, zf;
		int matrix_element_idx, vector_element_idx;
		float A_matrix_2D_value[10], h_vector_2D_value[4];

    	yf = (float)y - ((float)DATA_H - 1.0f) * 0.5f;
		zf = (float)z - ((float)DATA_D - 1.0f) * 0.5f;

		// Y

		A_matrix_2D_value[0] = 0.0f;
		A_matrix_2D_value[1] = 0.0f;
		A_matrix_2D_value[2] = 0.0f;
		A_matrix_2D_value[3] = 0.0f;
		A_matrix_2D_value[4] = 0.0f;
		A_matrix_2D_value[5] = 0.0f;
		A_matrix_2D_value[6] = 0.0f;
		A_matrix_2D_value[7] = 0.0f;
		A_matrix_2D_value[8] = 0.0f;
		A_matrix_2D_value[9] = 0.0f;

		h_vector_2D_value[0] = 0.0f;
		h_vector_2D_value[1] = 0.0f;
		h_vector_2D_value[2] = 0.0f;
		h_vector_2D_value[3] = 0.0f;

		for (int x = (FILTER_SIZE - 1)/2; x < (DATA_W - (FILTER_SIZE - 1)/2); x++)
		{
			float xf = (float)x - ((float)DATA_W - 1.0f) * 0.5f;
			int idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

			float phase_difference = Phase_Differences[idx];
			float phase_gradient = Phase_Gradients[idx];
			float certainty = Certainties[idx];
			float c_pg_pg = certainty * phase_gradient * phase_gradient;
			float c_pg_pd = certainty * phase_gradient * phase_difference;

			A_matrix_2D_value[0] += c_pg_pg;
			A_matrix_2D_value[1] += xf * c_pg_pg;
			A_matrix_2D_value[2] += yf * c_pg_pg;
			A_matrix_2D_value[3] += zf * c_pg_pg;
			A_matrix_2D_value[4] += xf * xf * c_pg_pg;
			A_matrix_2D_value[5] += xf * yf * c_pg_pg;
			A_matrix_2D_value[6] += xf * zf * c_pg_pg;
			A_matrix_2D_value[7] += yf * yf * c_pg_pg;
			A_matrix_2D_value[8] += yf * zf * c_pg_pg;
			A_matrix_2D_value[9] += zf * zf * c_pg_pg;

			h_vector_2D_value[0] += c_pg_pd;
			h_vector_2D_value[1] += xf * c_pg_pd;
			h_vector_2D_value[2] += yf * c_pg_pd;
			h_vector_2D_value[3] += zf * c_pg_pd;
		}

		matrix_element_idx = y + z * DATA_H + 10 * DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[0];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[1];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[2];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[3];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[4];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[5];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[6];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[7];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[8];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[9];

		vector_element_idx = y + z * DATA_H + DATA_H * DATA_D;
		h_vector_2D_values[vector_element_idx] = h_vector_2D_value[0];
		vector_element_idx += 5 * DATA_H * DATA_D;
		h_vector_2D_values[vector_element_idx] = h_vector_2D_value[1];
		vector_element_idx += DATA_H * DATA_D;
		h_vector_2D_values[vector_element_idx] = h_vector_2D_value[2];
		vector_element_idx += DATA_H * DATA_D;
		h_vector_2D_values[vector_element_idx] = h_vector_2D_value[3];
	}
}


__kernel void CalculateAMatrixAndHVector2DValuesY_(__global float* A_matrix_2D_values, __global float* h_vector_2D_values, __global const float* Phase_Differences, __global const float* Phase_Gradients, __global const float* Certainties, int DATA_W, int DATA_H, int DATA_D)
{
	int y = get_local_id(0);
	int z = get_group_id(0);

	if ( (y >= DATA_H) || (z >= DATA_D))
			return;

	float yf, zf;
	int matrix_element_idx, vector_element_idx;
	float A_matrix_2D_value[10], h_vector_2D_value[4];

    yf = (float)y - ((float)DATA_H - 1.0f) * 0.5f + 3.0f;
	zf = (float)z - ((float)DATA_D - 1.0f) * 0.5f + 3.0f;

	// Y

	A_matrix_2D_value[0] = 0.0f;
	A_matrix_2D_value[1] = 0.0f;
	A_matrix_2D_value[2] = 0.0f;
	A_matrix_2D_value[3] = 0.0f;
	A_matrix_2D_value[4] = 0.0f;
	A_matrix_2D_value[5] = 0.0f;
	A_matrix_2D_value[6] = 0.0f;
	A_matrix_2D_value[7] = 0.0f;
	A_matrix_2D_value[8] = 0.0f;
	A_matrix_2D_value[9] = 0.0f;

	h_vector_2D_value[0] = 0.0f;
	h_vector_2D_value[1] = 0.0f;
	h_vector_2D_value[2] = 0.0f;
	h_vector_2D_value[3] = 0.0f;

	for (int x = 0; x < DATA_W; x++)
	{
		float xf = (float)x - ((float)DATA_W - 1.0f) * 0.5f + 3.0f;
		int idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

		float phase_difference = Phase_Differences[idx];
		float phase_gradient = Phase_Gradients[idx];
		float certainty = Certainties[idx];
		float c_pg_pg = certainty * phase_gradient * phase_gradient;
		float c_pg_pd = certainty * phase_gradient * phase_difference;

		A_matrix_2D_value[0] += c_pg_pg;
		A_matrix_2D_value[1] += xf * c_pg_pg;
		A_matrix_2D_value[2] += yf * c_pg_pg;
		A_matrix_2D_value[3] += zf * c_pg_pg;
		A_matrix_2D_value[4] += xf * xf * c_pg_pg;
		A_matrix_2D_value[5] += xf * yf * c_pg_pg;
		A_matrix_2D_value[6] += xf * zf * c_pg_pg;
		A_matrix_2D_value[7] += yf * yf * c_pg_pg;
		A_matrix_2D_value[8] += yf * zf * c_pg_pg;
		A_matrix_2D_value[9] += zf * zf * c_pg_pg;

		h_vector_2D_value[0] += c_pg_pd;
		h_vector_2D_value[1] += xf * c_pg_pd;
		h_vector_2D_value[2] += yf * c_pg_pd;
		h_vector_2D_value[3] += zf * c_pg_pd;
	}

	matrix_element_idx = y + z * DATA_H + 10 * DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[0];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[1];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[2];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[3];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[4];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[5];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[6];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[7];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[8];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[9];

	vector_element_idx = y + z * DATA_H + DATA_H * DATA_D;
	h_vector_2D_values[vector_element_idx] = h_vector_2D_value[0];
	vector_element_idx += 5 * DATA_H * DATA_D;
	h_vector_2D_values[vector_element_idx] = h_vector_2D_value[1];
	vector_element_idx += DATA_H * DATA_D;
	h_vector_2D_values[vector_element_idx] = h_vector_2D_value[2];
	vector_element_idx += DATA_H * DATA_D;
	h_vector_2D_values[vector_element_idx] = h_vector_2D_value[3];
}


__kernel void CalculateAMatrixAndHVector2DValues_Z(__global float* A_matrix_2D_values, __global float* h_vector_2D_values, __global const float* Phase_Differences, __global const float* Phase_Gradients, __global const float* Certainties, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)

{
	int y = get_local_id(0);
	int z = get_group_id(0);

	//volatile int y = blockIdx.x * blockDim.x + threadIdx.x;
	//volatile int z = blockIdx.y * blockDim.y + threadIdx.y;

	if (((y >= (FILTER_SIZE - 1)/2) && (y < DATA_H - (FILTER_SIZE - 1)/2)) && ((z >= (FILTER_SIZE - 1)/2) && (z < DATA_D - (FILTER_SIZE - 1)/2)))
	{
	    float yf, zf;
		int matrix_element_idx, vector_element_idx;
		float A_matrix_2D_value[10], h_vector_2D_value[4];

    	yf = (float)y - ((float)DATA_H - 1.0f) * 0.5f;
		zf = (float)z - ((float)DATA_D - 1.0f) * 0.5f;

		// Z

		A_matrix_2D_value[0] = 0.0f;
		A_matrix_2D_value[1] = 0.0f;
		A_matrix_2D_value[2] = 0.0f;
		A_matrix_2D_value[3] = 0.0f;
		A_matrix_2D_value[4] = 0.0f;
		A_matrix_2D_value[5] = 0.0f;
		A_matrix_2D_value[6] = 0.0f;
		A_matrix_2D_value[7] = 0.0f;
		A_matrix_2D_value[8] = 0.0f;
		A_matrix_2D_value[9] = 0.0f;

		h_vector_2D_value[0] = 0.0f;
		h_vector_2D_value[1] = 0.0f;
		h_vector_2D_value[2] = 0.0f;
		h_vector_2D_value[3] = 0.0f;

		for (int x = (FILTER_SIZE - 1)/2; x < (DATA_W - (FILTER_SIZE - 1)/2); x++)
		{
			float xf = (float)x - ((float)DATA_W - 1.0f) * 0.5f;
			int idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

			float phase_difference = Phase_Differences[idx];
			float phase_gradient = Phase_Gradients[idx];
			float certainty = Certainties[idx];
			float c_pg_pg = certainty * phase_gradient * phase_gradient;
			float c_pg_pd = certainty * phase_gradient * phase_difference;

			A_matrix_2D_value[0] += c_pg_pg;
			A_matrix_2D_value[1] += xf * c_pg_pg;
			A_matrix_2D_value[2] += yf * c_pg_pg;
			A_matrix_2D_value[3] += zf * c_pg_pg;
			A_matrix_2D_value[4] += xf * xf * c_pg_pg;
			A_matrix_2D_value[5] += xf * yf * c_pg_pg;
			A_matrix_2D_value[6] += xf * zf * c_pg_pg;
			A_matrix_2D_value[7] += yf * yf * c_pg_pg;
			A_matrix_2D_value[8] += yf * zf * c_pg_pg;
			A_matrix_2D_value[9] += zf * zf * c_pg_pg;

			h_vector_2D_value[0] += c_pg_pd;
			h_vector_2D_value[1] += xf * c_pg_pd;
			h_vector_2D_value[2] += yf * c_pg_pd;
			h_vector_2D_value[3] += zf * c_pg_pd;
		}


		matrix_element_idx = y + z * DATA_H + 20 * DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[0];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[1];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[2];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[3];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[4];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[5];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[6];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[7];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[8];
		matrix_element_idx += DATA_H * DATA_D;
		A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[9];

		vector_element_idx = y + z * DATA_H + 2 * DATA_H * DATA_D;
		h_vector_2D_values[vector_element_idx] = h_vector_2D_value[0];
		vector_element_idx += 7 * DATA_H * DATA_D;
		h_vector_2D_values[vector_element_idx] = h_vector_2D_value[1];
		vector_element_idx += DATA_H * DATA_D;
		h_vector_2D_values[vector_element_idx] = h_vector_2D_value[2];
		vector_element_idx += DATA_H * DATA_D;
		h_vector_2D_values[vector_element_idx] = h_vector_2D_value[3];
	}
}



__kernel void	Calculate_A_matrix_and_h_vector_2D_values_Z_(__global float* A_matrix_2D_values, __global float* h_vector_2D_values, __global const float* Phase_Differences, __global const float* Phase_Gradients, __global const float* Certainties, int DATA_W, int DATA_H, int DATA_D)
{
	int y = get_local_id(0);
	int z = get_group_id(0);

	if ( (y >= DATA_H) || (z >= DATA_D))
			return;

    float yf, zf;
	int matrix_element_idx, vector_element_idx;
	float A_matrix_2D_value[10], h_vector_2D_value[4];

   	yf = (float)y - ((float)DATA_H - 1.0f) * 0.5f + 3.0f;
	zf = (float)z - ((float)DATA_D - 1.0f) * 0.5f + 3.0f;

	// Z

	A_matrix_2D_value[0] = 0.0f;
	A_matrix_2D_value[1] = 0.0f;
	A_matrix_2D_value[2] = 0.0f;
	A_matrix_2D_value[3] = 0.0f;
	A_matrix_2D_value[4] = 0.0f;
	A_matrix_2D_value[5] = 0.0f;
	A_matrix_2D_value[6] = 0.0f;
	A_matrix_2D_value[7] = 0.0f;
	A_matrix_2D_value[8] = 0.0f;
	A_matrix_2D_value[9] = 0.0f;

	h_vector_2D_value[0] = 0.0f;
	h_vector_2D_value[1] = 0.0f;
	h_vector_2D_value[2] = 0.0f;
	h_vector_2D_value[3] = 0.0f;

	for (int x = 0; x < DATA_W; x++)
	{
		float xf = (float)x - ((float)DATA_W - 1.0f) * 0.5f + 3.0f;
		int idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

		float phase_difference = Phase_Differences[idx];
		float phase_gradient = Phase_Gradients[idx];
		float certainty = Certainties[idx];
		float c_pg_pg = certainty * phase_gradient * phase_gradient;
		float c_pg_pd = certainty * phase_gradient * phase_difference;

		A_matrix_2D_value[0] += c_pg_pg;
		A_matrix_2D_value[1] += xf * c_pg_pg;
		A_matrix_2D_value[2] += yf * c_pg_pg;
		A_matrix_2D_value[3] += zf * c_pg_pg;
		A_matrix_2D_value[4] += xf * xf * c_pg_pg;
		A_matrix_2D_value[5] += xf * yf * c_pg_pg;
		A_matrix_2D_value[6] += xf * zf * c_pg_pg;
		A_matrix_2D_value[7] += yf * yf * c_pg_pg;
		A_matrix_2D_value[8] += yf * zf * c_pg_pg;
		A_matrix_2D_value[9] += zf * zf * c_pg_pg;

		h_vector_2D_value[0] += c_pg_pd;
		h_vector_2D_value[1] += xf * c_pg_pd;
		h_vector_2D_value[2] += yf * c_pg_pd;
		h_vector_2D_value[3] += zf * c_pg_pd;
	}

	matrix_element_idx = y + z * DATA_H + 20 * DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[0];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[1];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[2];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[3];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[4];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[5];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[6];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[7];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[8];
	matrix_element_idx += DATA_H * DATA_D;
	A_matrix_2D_values[matrix_element_idx] = A_matrix_2D_value[9];

	vector_element_idx = y + z * DATA_H + 2 * DATA_H * DATA_D;
	h_vector_2D_values[vector_element_idx] = h_vector_2D_value[0];
	vector_element_idx += 7 * DATA_H * DATA_D;
	h_vector_2D_values[vector_element_idx] = h_vector_2D_value[1];
	vector_element_idx += DATA_H * DATA_D;
	h_vector_2D_values[vector_element_idx] = h_vector_2D_value[2];
	vector_element_idx += DATA_H * DATA_D;
	h_vector_2D_values[vector_element_idx] = h_vector_2D_value[3];
}


// dimBlock.x = DATA_D; dimBlock.y = 1; dimBlock.z = 1;
// dimGrid.x = NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS; dimGrid.y = 1;

__kernel void CalculateAMatrix1DValues(__global float* A_matrix_1D_values, __global const float* A_matrix_2D_values, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)
{
	int z = get_local_id(0);
	
	if (z >= (FILTER_SIZE - 1)/2 && z < (DATA_D - (FILTER_SIZE - 1)/2))
	{
		int A_matrix_element   = blockIdx.x; // 144 element (12 x 12 matrix) (30 that are != 0)
		int matrix_element_idx = z + A_matrix_element * DATA_D;
		int idx;
		float matrix_1D_value = 0.0f;

		idx = z * DATA_H + A_matrix_element * DATA_H * DATA_D;
		// Sum over all y positions
		for (int y = (FILTER_SIZE - 1)/2; y < (DATA_H - (FILTER_SIZE - 1)/2); y++)
		{
			matrix_1D_value += A_matrix_2D_values[idx + y];
		}

		A_matrix_1D_values[matrix_element_idx] = matrix_1D_value;
	}
}

__kernel void CalculateAMatrix1DValues_(__global float* A_matrix_1D_values, __global const float* A_matrix_2D_values, int DATA_W, int DATA_H, int DATA_D)
{
	int z = get_local_id(0);

	if (z >= DATA_D)
		return;

	int A_matrix_element   = blockIdx.x; // 144 element (12 x 12 matrix) (30 that are != 0)
	int matrix_element_idx = z + A_matrix_element * DATA_D;
	int idx;
	float matrix_1D_value = 0.0f;

	idx = z * DATA_H + A_matrix_element * DATA_H * DATA_D;
	// Sum over all y positions
	for (int y = 0; y < DATA_H; y++)
	{
		matrix_1D_value += A_matrix_2D_values[idx + y];
	}

	A_matrix_1D_values[matrix_element_idx] = matrix_1D_value;
}

// dimBlock.x = NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS; dimBlock.y = 1; dimBlock.z = 1;
// dimGrid.x = 1; dimGrid.y = 1;

__kernel void Calculate_A_matrix(__global float* A_matrix, __global const float* A_matrix_1D_values, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)
{
	int A_matrix_element = get_local_id(0);
	int idx, i, j;

	float matrix_value = 0.0f;

	idx = A_matrix_element * DATA_D;

	// Sum over all z positions
	//#pragma unroll 128
	for (int z = (FILTER_SIZE - 1)/2; z < (DATA_D - (FILTER_SIZE - 1)/2); z++)
	{
		matrix_value += A_matrix_1D_values[idx + z];
	}

	Get_Parameter_Indices_Filter(i,j,A_matrix_element);
	A_matrix_element = i + j * NUMBER_OF_MOTION_CORRECTION_PARAMETERS;

	A_matrix[A_matrix_element] = matrix_value;
}

__kernel void Calculate_A_matrix_(__global float* A_matrix, __global const float* A_matrix_1D_values, int DATA_W, int DATA_H, int DATA_D)
{
	int A_matrix_element = get_local_id(0);
	int idx, i, j;

	float matrix_value = 0.0f;

	idx = A_matrix_element * DATA_D;

	// Sum over all z positions
	//#pragma unroll 128
	for (int z = 0; z < DATA_D; z++)
	{
		matrix_value += A_matrix_1D_values[idx + z];
	}

	Get_Parameter_Indices_Filter(i,j,A_matrix_element);
	A_matrix_element = i + j * NUMBER_OF_MOTION_CORRECTION_PARAMETERS;

	A_matrix[A_matrix_element] = matrix_value;
}

__kernel void Reset_A_matrix(__global float* A_matrix)
{
	int i = get_local_id(0);

	A_matrix[i] = 0.0f;
}

__kernel void Reset_h_vector(__global float* h_vector)
{
	int i = get_local_id(0);

	h_vector[i] = 0.0f;
}


// dimBlock.x = DATA_D; dimBlock.y = 1; dimBlock.z = 1;
// dimGrid.x = NUMBER_OF_PARAMETERS; dimGrid.y = 1;

__kernel void Calculate_h_vector_1D_values(__global float* h_vector_1D_values, __global const float* h_vector_2D_values, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)
{
	int z = get_local_id(0);

	if (z >= (FILTER_SIZE - 1)/2 && z < (DATA_D - (FILTER_SIZE - 1)/2))
	{
		int h_vector_element   = blockIdx.x; // 12 parameters
		int vector_element_idx = z + h_vector_element * DATA_D;
		int idx;

		float vector_1D_value = 0.0f;

		idx = z * DATA_H + h_vector_element * DATA_H * DATA_D;
		// Sum over all y positions
		for (int y = (FILTER_SIZE - 1)/2; y < (DATA_H - (FILTER_SIZE - 1)/2); y++)
		{
			vector_1D_value += h_vector_2D_values[idx + y];
		}

		h_vector_1D_values[vector_element_idx] = vector_1D_value;
	}
}

__kernel void Calculate_h_vector_1D_values_(__global float* h_vector_1D_values, __global const float* h_vector_2D_values, int DATA_W, int DATA_H, int DATA_D)
{
	int z = get_local_id(0);

	if (z >= DATA_D)
		return;

	int h_vector_element   = blockIdx.x; // 12 parameters
	int vector_element_idx = z + h_vector_element * DATA_D;
	int idx;

	float vector_1D_value = 0.0f;

	idx = z * DATA_H + h_vector_element * DATA_H * DATA_D;
	// Sum over all y positions
	for (int y = 0; y < DATA_H; y++)
	{
		vector_1D_value += h_vector_2D_values[idx + y];
	}

	h_vector_1D_values[vector_element_idx] = vector_1D_value;
}

// dimBlock.x = NUMBER_OF_PARAMETERS; dimBlock.y = 1; dimBlock.z = 1;
// dimGrid.x = 1; dimGrid.y = 1;

__kernel void Calculate_h_vector(__global float* h_vector, __global const float* h_vector_1D_values, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)
{
	int h_vector_element = get_local_id(0);
	int idx;

	float vector_value = 0.0f;
	idx = h_vector_element * DATA_D;

	// Sum over all z positions
	for (int z = (FILTER_SIZE - 1)/2; z < (DATA_D - (FILTER_SIZE - 1)/2); z++)
	{
		vector_value += h_vector_1D_values[idx + z];
	}

	h_vector[h_vector_element] = vector_value;
}

__kernel void Calculate_h_vector_(__global float* h_vector, __global const float* h_vector_1D_values, int DATA_W, int DATA_H, int DATA_D)
{
	int h_vector_element = get_local_id(0);
	int idx;

	float vector_value = 0.0f;
	idx = h_vector_element * DATA_D;

	// Sum over all z positions
	for (int z = 0; z < DATA_D; z++)
	{
		vector_value += h_vector_1D_values[idx + z];
	}

	h_vector[h_vector_element] = vector_value;
}

__kernel void InterpolateVolumeTriLinear(__global float* Volume, int DATA_W, int DATA_H, int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int idx = Calculate_3D_Index(x,y,z,DATA_W, DATA_H);
	float3 Motion_Vector;
	float xf, yf, zf;

    // (motion_vector.x)   (p0)   (p3  p4  p5)   (x)
	// (motion_vector.y) = (p1) + (p6  p7  p8) * (y)
 	// (motion_vector.z)   (p2)   (p9 p10 p11)   (z)

	// Change to coordinate system with origo in (sx - 1)/2 (sy - 1)/2 (sz - 1)/2
	xf = (float)x - ((float)DATA_W - 1.0f) * 0.5f;
	yf = (float)y - ((float)DATA_H - 1.0f) * 0.5f;
	zf = (float)z - ((float)DATA_D - 1.0f) * 0.5f;

	Motion_Vector.x = x + c_Parameter_Vector[0] + c_Parameter_Vector[3] * xf + c_Parameter_Vector[4]   * yf + c_Parameter_Vector[5]  * zf + 0.5f;
	Motion_Vector.y = y + c_Parameter_Vector[1] + c_Parameter_Vector[6] * xf + c_Parameter_Vector[7]   * yf + c_Parameter_Vector[8]  * zf + 0.5f;
	Motion_Vector.z = z + c_Parameter_Vector[2] + c_Parameter_Vector[9] * xf + c_Parameter_Vector[10]  * yf + c_Parameter_Vector[11] * zf + 0.5f;

	//Volume[idx] = tex3D(tex_Modified_Volume, Motion_Vector.x, Motion_Vector.y, Motion_Vector.z);
}

// Statistical functions
__kernel void CalculateBetaValuesGLM(__global float* Beta_Volumes, __global const float* Volumes, __global const float* Mask, __constant float* *c_xtxxt_GLM, __constant struct DATA_PARAMETERS *DATA)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,c,DATA_W,DATA_H)] == 0.0f )
	{
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}
	
		return;
	}

	int t = 0;
	float beta[20];
	
	// Reset all beta values
	for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{
		beta[r] = 0.0f;
	}

	// Calculate betahat, i.e. multiply (x^T x)^(-1) x^T with Y
	// Loop over volumes
	for (v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		float temp = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		// Loop over regressors
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			beta[r] += temp * c_xtxxt_GLM[NUMBER_OF_VOLUMES * r + v];
		}
	}

	// Save beta values
	for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{
		Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)] = beta[r];
	}
}


__kernel void CalculateStatisticalMapsGLM(__global float* Statistical_Maps, __global float* Beta_Contrasts, __global float* Residual_Volumes, __global float* Residual_Variances, __global const float* Volumes, __global const float* Beta_Volumes, __global const float* Mask, __constant float *c_X_GLM, __constant float* c_Contrast_Vectors, __constant float* ctxtxc, __constant struct DATA_PARAMETERS *DATA)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,c,DATA_W,DATA_H)] == 0.0f )
	{
		Residual_Variances[Calculate3DIndex(x,y,z,DATA_W,DATA_H,DATA_D)] = 0.0f;

		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = 0.0f;
			Beta_Contrasts[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}
		
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}
	
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			Residual_Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}

		return;
	}

	int t = 0;
	float eps, meaneps, vareps;

	// Calculate the mean of the error eps
	meaneps = 0.0f;
	for (v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{ 
			eps -= Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)] * c_X_GLM[NUMBER_OF_VALUES * r + v];
		}
		meaneps += eps;
		Residual_Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = eps;
	}
	meaneps /= (float)DATA_T;

	// Now calculate the variance of eps
	vareps = 0.0f;
	for (v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			eps -= Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)] * c_X_GLM[NUMBER_OF_VOLUMES * r + v];
		}
		vareps += (eps - meaneps) * (eps - meaneps);
	}
	vareps /= ((float)DATA_T - (float)NUMBER_OF_REGRESSORS);
	Residual_Variances[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = vareps;

	// Loop over contrasts and calculate t-values
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		float contrast_value = 0.0f;
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			contrast_value += c_Contrast_Vectors[NUMBER_OF_REGRESSORS * c + r] * Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)];
		}	
		Beta_Contrasts[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_values;
		Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_value * rsqrtf(vareps * ctxtxc[c]);
	}
}

// Functions for permutation test

__kernel void CalculateStatisticalMapsGLMPermutation(__global float* Statistical_Maps, __global const float* Volumes, __global const float* Beta_Volumes, __global const float* Mask, __constant float *c_X_GLM, __constant float* c_Contrast_Vectors, __constant float* ctxtxc, __constant struct DATA_PARAMETERS *DATA)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;
	
	if ( Mask[Calculate3DIndex(x,y,z,c,DATA_W,DATA_H)] == 0.0f )
	{
		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}
		return;
	}

	int t = 0;
	float eps, meaneps, vareps;

	// Calculate the mean of the error eps
	meaneps = 0.0f;
	// Loop over volumes
	for (v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		// Loop over regressors
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			eps -= Beta_Volumes[Calculate4DVolumes(x,y,z,r,DATA_W,DATA_H,DATA_D)] * c_X_GLM[NUMBER_OF_VALUES * r + v];
		}
		meaneps += eps;
	}
	meaneps /= (float)DATA_T;

	// Now calculate the variance of eps
	vareps = 0.0f;
	// Loop over volumes
	for (v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		// Loop over regressors
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			eps -= Beta_Volumes[Calculate4DVolumes(x,y,z,r,DATA_W,DATA_H,DATA_D)] * c_X_GLM[NUMBER_OF_VOLUMES * r + v];
		}
		vareps += (eps - meaneps) * (eps - meaneps);
	}
	vareps /= ((float)DATA_T - (float)NUMBER_OF_REGRESSORS);
	
	// Loop over contrasts and calculate t-values
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		float contrast_value = 0.0f;
		// Loop over regressors
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			contrast_value += c_Contrast_Vectors[NUMBER_OF_REGRESSORS * c + r] * Beta_Volumes[Calculate4DVolumes(x,y,z,r,DATA_W,DATA_H,DATA_D)];
		}	
		Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_value * rsqrtf(vareps * ctxtxc[c]);
	}
}


__kernel void GeneratePermutedfMRIVolumesAR4(__global float* Permuted_fMRI_Volumes, __global const float4* Alpha_Volumes, __global const float* Whitened_fMRI_Volumes, __global const float* Mask, __constant unsignet int *c_Permutation_Vector, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Mask[Calculate_3D_Index(x, y, z, DATA_W, DATA_H)] == 1.0f )
    {
        int t = 0;
		float old_value_1, old_value_2, old_value_3, old_value_4, old_value_5;
		float4 alphas = Alpha_Volumes[Calculate_3D_Index(x, y, z, DATA_W, DATA_H)];
        
        old_value1 = Whitened_fMRI_Volumes[Calculate_4D_Index(x, y, z, c_Permutation_Vector[0], DATA_W, DATA_H, DATA_D)];
		old_value2 = alphas.x * old_value1  + Whitened_fMRI_Volumes[Calculate_4D_Index(x, y, z, c_Permutation_Vector[1], DATA_W, DATA_H, DATA_D)];
		old_value3 = alphas.x * old_value2  + alphas.y * old_value1 + Whitened_fMRI_Volumes[Calculate_4D_Index(x, y, z, c_Permutation_Vector[2], DATA_W, DATA_H, DATA_D)];
		old_value4 = alphas.x * old_value3  + alphas.y * old_value2 + alphas.z * old_value1 + Whitened_fMRI_Volumes[Calculate_4D_Index(x, y, z, c_Permutation_Vector[3], DATA_W, DATA_H, DATA_D)];

        Permuted_fMRI_volumes[Calculate_4D_Index(x, y, z, 0, DATA_W, DATA_H, DATA_D)] =  old_value1;
        Permuted_fMRI_volumes[Calculate_4D_Index(x, y, z, 1, DATA_W, DATA_H, DATA_D)] =  old_value2;
        Permuted_fMRI_volumes[Calculate_4D_Index(x, y, z, 2, DATA_W, DATA_H, DATA_D)] =  old_value3;
        Permuted_fMRI_volumes[Calculate_4D_Index(x, y, z, 3, DATA_W, DATA_H, DATA_D)] =  old_value4;

        // Read the data in a permuted order and apply an inverse whitening transform
        for (t = 4; t < DATA_T; t++)
        {
            // Calculate the unwhitened, permuted, timeseries
            old_value5 = alphas.x * old_value4 + alphas.y * old_value3 + alphas.z * old_value2 + alphas.w * old_value1 + whitened_fMRI_volumes[Calculate_4D_Index(x, y, z, c_Permutation_Vector[t], DATA_W, DATA_H, DATA_D)];
			
            Permuted_fMRI_volumes[Calculate_4D_Index(x, y, z, t, DATA_W, DATA_H, DATA_D)] = old_value5;

            // Save old values
			old_value_1 = old_value_2;
            old_value_2 = old_value_3;
            old_value_3 = old_value_4;
            old_value_4 = old_value_5;
        }
    }
}

__kernel void ApplyWhiteningAR4(__global float* Whitened_fMRI_Volumes, __global const float* Detrended_fMRI_Volumes, __global const float4 *Alpha_Volumes, __global const float* Mask, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Mask[Calculate_3D_Index(x, y, z, DATA_W, DATA_H)] == 1.0f )
    {
        int t = 0;
		float old_value_1, old_value_2, old_value_3, old_value_4, old_value_5;
        float4 alphas = Alpha_Volumes[Calculate_3D_Index(x, y, z, DATA_W, DATA_H)];
        
        // Calculate the whitened timeseries

        old_value_1 = Detrended_fMRI_Volumes[Calculate_4D_Index(x, y, z, 0, DATA_W, DATA_H, DATA_D)];
        Whitened_fMRI_Volumes[Calculate_4D_Index(x, y, z, 0, DATA_W, DATA_H, DATA_D)] = old_value_1;
        old_value_2 = Detrended_fMRI_Volumes[Calculate_4D_Index(x, y, z, 1, DATA_W, DATA_H, DATA_D)];
        Whitened_fMRI_Volumes[Calculate_4D_Index(x, y, z, 1, DATA_W, DATA_H, DATA_D)] = old_value_2  - alphas.x * old_value_0;
        old_value_3 = Detrended_fMRI_Volumes[Calculate_4D_Index(x, y, z, 2, DATA_W, DATA_H, DATA_D)];
        Whitened_fMRI_Volumes[Calculate_4D_Index(x, y, z, 2, DATA_W, DATA_H, DATA_D)] = old_value_3 - alphas.x * old_value_2 - alphas.y * old_value_1;
        old_value_4 = Detrended_fMRI_Volumes[Calculate_4D_Index(x, y, z, 3, DATA_W, DATA_H, DATA_D)];
        Whitened_fMRI_Volumes[Calculate_4D_Index(x, y, z, 3, DATA_W, DATA_H, DATA_D)] = old_value_4 - alphas.x * old_value_3 - alphas.y * old_value_2 - alphas.z * old_value_1;

        for (t = 4; t < DATA_T; t++)
        {
            old_value_5 = Detrended_fMRI_Volumes[Calculate_4D_Index(x, y, z, t, DATA_W, DATA_H, DATA_D)];

            Whitened_fMRI_Volumes[Calculate_4D_Index(x, y, z, t, DATA_W, DATA_H, DATA_D)] = old_value_5 - alphas.x * old_value_4 - alphas.y * old_value_3 - alphas.z * old_value_2 - alphas.w * old_value_1;

			// Save old values
            old_value_1 = old_value_2;
            old_value_2 = old_value_3;
            old_value_3 = old_value_4;
            old_value_4 = old_value_5;
        }
    }
}

float Determinant_(float Cxx[4][4])
{
    return Cxx[0][3] * Cxx[1][2] * Cxx[2][1] * Cxx[3][0] - Cxx[0][2] * Cxx[1][3] * Cxx[2][1] * Cxx[3][0] - Cxx[0][3] * Cxx[1][1] * Cxx[2][2] * Cxx[3][0]
         + Cxx[0][1] * Cxx[1][3] * Cxx[2][2] * Cxx[3][0] + Cxx[0][2] * Cxx[1][1] * Cxx[2][3] * Cxx[3][0] - Cxx[0][1] * Cxx[1][2] * Cxx[2][3] * Cxx[3][0]
         - Cxx[0][3] * Cxx[1][2] * Cxx[2][0] * Cxx[3][1] + Cxx[0][2] * Cxx[1][3] * Cxx[2][0] * Cxx[3][1] + Cxx[0][3] * Cxx[1][0] * Cxx[2][2] * Cxx[3][1]
         - Cxx[0][0] * Cxx[1][3] * Cxx[2][2] * Cxx[3][1] - Cxx[0][2] * Cxx[1][0] * Cxx[2][3] * Cxx[3][1] + Cxx[0][0] * Cxx[1][2] * Cxx[2][3] * Cxx[3][1]
         + Cxx[0][3] * Cxx[1][1] * Cxx[2][0] * Cxx[3][2] - Cxx[0][1] * Cxx[1][3] * Cxx[2][0] * Cxx[3][2] - Cxx[0][3] * Cxx[1][0] * Cxx[2][1] * Cxx[3][2]
         + Cxx[0][0] * Cxx[1][3] * Cxx[2][1] * Cxx[3][2] + Cxx[0][1] * Cxx[1][0] * Cxx[2][3] * Cxx[3][2] - Cxx[0][0] * Cxx[1][1] * Cxx[2][3] * Cxx[3][2]
         - Cxx[0][2] * Cxx[1][1] * Cxx[2][0] * Cxx[3][3] + Cxx[0][1] * Cxx[1][2] * Cxx[2][0] * Cxx[3][3] + Cxx[0][2] * Cxx[1][0] * Cxx[2][1] * Cxx[3][3]
		 - Cxx[0][0] * Cxx[1][2] * Cxx[2][1] * Cxx[3][3] - Cxx[0][1] * Cxx[1][0] * Cxx[2][2] * Cxx[3][3] + Cxx[0][0] * Cxx[1][1] * Cxx[2][2] * Cxx[3][3];
}

void Invert_4x4(float Cxx[4][4], float inv_Cxx[4][4])
{
	float determinant = Determinant_(Cxx) + 0.001f;

	inv_Cxx[0][0] = Cxx[1][2]*Cxx[2][3]*Cxx[3][1] - Cxx[1][3]*Cxx[2][2]*Cxx[3][1] + Cxx[1][3]*Cxx[2][1]*Cxx[3][2] - Cxx[1][1]*Cxx[2][3]*Cxx[3][2] - Cxx[1][2]*Cxx[2][1]*Cxx[3][3] + Cxx[1][1]*Cxx[2][2]*Cxx[3][3];
	inv_Cxx[0][1] = Cxx[0][3]*Cxx[2][2]*Cxx[3][1] - Cxx[0][2]*Cxx[2][3]*Cxx[3][1] - Cxx[0][3]*Cxx[2][1]*Cxx[3][2] + Cxx[0][1]*Cxx[2][3]*Cxx[3][2] + Cxx[0][2]*Cxx[2][1]*Cxx[3][3] - Cxx[0][1]*Cxx[2][2]*Cxx[3][3];
	inv_Cxx[0][2] = Cxx[0][2]*Cxx[1][3]*Cxx[3][1] - Cxx[0][3]*Cxx[1][2]*Cxx[3][1] + Cxx[0][3]*Cxx[1][1]*Cxx[3][2] - Cxx[0][1]*Cxx[1][3]*Cxx[3][2] - Cxx[0][2]*Cxx[1][1]*Cxx[3][3] + Cxx[0][1]*Cxx[1][2]*Cxx[3][3];
	inv_Cxx[0][3] = Cxx[0][3]*Cxx[1][2]*Cxx[2][1] - Cxx[0][2]*Cxx[1][3]*Cxx[2][1] - Cxx[0][3]*Cxx[1][1]*Cxx[2][2] + Cxx[0][1]*Cxx[1][3]*Cxx[2][2] + Cxx[0][2]*Cxx[1][1]*Cxx[2][3] - Cxx[0][1]*Cxx[1][2]*Cxx[2][3];
	inv_Cxx[1][0] = Cxx[1][3]*Cxx[2][2]*Cxx[3][0] - Cxx[1][2]*Cxx[2][3]*Cxx[3][0] - Cxx[1][3]*Cxx[2][0]*Cxx[3][2] + Cxx[1][0]*Cxx[2][3]*Cxx[3][2] + Cxx[1][2]*Cxx[2][0]*Cxx[3][3] - Cxx[1][0]*Cxx[2][2]*Cxx[3][3];
	inv_Cxx[1][1] = Cxx[0][2]*Cxx[2][3]*Cxx[3][0] - Cxx[0][3]*Cxx[2][2]*Cxx[3][0] + Cxx[0][3]*Cxx[2][0]*Cxx[3][2] - Cxx[0][0]*Cxx[2][3]*Cxx[3][2] - Cxx[0][2]*Cxx[2][0]*Cxx[3][3] + Cxx[0][0]*Cxx[2][2]*Cxx[3][3];
	inv_Cxx[1][2] = Cxx[0][3]*Cxx[1][2]*Cxx[3][0] - Cxx[0][2]*Cxx[1][3]*Cxx[3][0] - Cxx[0][3]*Cxx[1][0]*Cxx[3][2] + Cxx[0][0]*Cxx[1][3]*Cxx[3][2] + Cxx[0][2]*Cxx[1][0]*Cxx[3][3] - Cxx[0][0]*Cxx[1][2]*Cxx[3][3];
	inv_Cxx[1][3] = Cxx[0][2]*Cxx[1][3]*Cxx[2][0] - Cxx[0][3]*Cxx[1][2]*Cxx[2][0] + Cxx[0][3]*Cxx[1][0]*Cxx[2][2] - Cxx[0][0]*Cxx[1][3]*Cxx[2][2] - Cxx[0][2]*Cxx[1][0]*Cxx[2][3] + Cxx[0][0]*Cxx[1][2]*Cxx[2][3];
	inv_Cxx[2][0] = Cxx[1][1]*Cxx[2][3]*Cxx[3][0] - Cxx[1][3]*Cxx[2][1]*Cxx[3][0] + Cxx[1][3]*Cxx[2][0]*Cxx[3][1] - Cxx[1][0]*Cxx[2][3]*Cxx[3][1] - Cxx[1][1]*Cxx[2][0]*Cxx[3][3] + Cxx[1][0]*Cxx[2][1]*Cxx[3][3];
	inv_Cxx[2][1] = Cxx[0][3]*Cxx[2][1]*Cxx[3][0] - Cxx[0][1]*Cxx[2][3]*Cxx[3][0] - Cxx[0][3]*Cxx[2][0]*Cxx[3][1] + Cxx[0][0]*Cxx[2][3]*Cxx[3][1] + Cxx[0][1]*Cxx[2][0]*Cxx[3][3] - Cxx[0][0]*Cxx[2][1]*Cxx[3][3];
	inv_Cxx[2][2] = Cxx[0][1]*Cxx[1][3]*Cxx[3][0] - Cxx[0][3]*Cxx[1][1]*Cxx[3][0] + Cxx[0][3]*Cxx[1][0]*Cxx[3][1] - Cxx[0][0]*Cxx[1][3]*Cxx[3][1] - Cxx[0][1]*Cxx[1][0]*Cxx[3][3] + Cxx[0][0]*Cxx[1][1]*Cxx[3][3];
	inv_Cxx[2][3] = Cxx[0][3]*Cxx[1][1]*Cxx[2][0] - Cxx[0][1]*Cxx[1][3]*Cxx[2][0] - Cxx[0][3]*Cxx[1][0]*Cxx[2][1] + Cxx[0][0]*Cxx[1][3]*Cxx[2][1] + Cxx[0][1]*Cxx[1][0]*Cxx[2][3] - Cxx[0][0]*Cxx[1][1]*Cxx[2][3];
	inv_Cxx[3][0] = Cxx[1][2]*Cxx[2][1]*Cxx[3][0] - Cxx[1][1]*Cxx[2][2]*Cxx[3][0] - Cxx[1][2]*Cxx[2][0]*Cxx[3][1] + Cxx[1][0]*Cxx[2][2]*Cxx[3][1] + Cxx[1][1]*Cxx[2][0]*Cxx[3][2] - Cxx[1][0]*Cxx[2][1]*Cxx[3][2];
	inv_Cxx[3][1] = Cxx[0][1]*Cxx[2][2]*Cxx[3][0] - Cxx[0][2]*Cxx[2][1]*Cxx[3][0] + Cxx[0][2]*Cxx[2][0]*Cxx[3][1] - Cxx[0][0]*Cxx[2][2]*Cxx[3][1] - Cxx[0][1]*Cxx[2][0]*Cxx[3][2] + Cxx[0][0]*Cxx[2][1]*Cxx[3][2];
	inv_Cxx[3][2] = Cxx[0][2]*Cxx[1][1]*Cxx[3][0] - Cxx[0][1]*Cxx[1][2]*Cxx[3][0] - Cxx[0][2]*Cxx[1][0]*Cxx[3][1] + Cxx[0][0]*Cxx[1][2]*Cxx[3][1] + Cxx[0][1]*Cxx[1][0]*Cxx[3][2] - Cxx[0][0]*Cxx[1][1]*Cxx[3][2];
	inv_Cxx[3][3] = Cxx[0][1]*Cxx[1][2]*Cxx[2][0] - Cxx[0][2]*Cxx[1][1]*Cxx[2][0] + Cxx[0][2]*Cxx[1][0]*Cxx[2][1] - Cxx[0][0]*Cxx[1][2]*Cxx[2][1] - Cxx[0][1]*Cxx[1][0]*Cxx[2][2] + Cxx[0][0]*Cxx[1][1]*Cxx[2][2];

	inv_Cxx[0][0] /= determinant;
	inv_Cxx[0][1] /= determinant;
	inv_Cxx[0][2] /= determinant;
	inv_Cxx[0][3] /= determinant;
	inv_Cxx[1][0] /= determinant;
	inv_Cxx[1][1] /= determinant;
	inv_Cxx[1][2] /= determinant;
	inv_Cxx[1][3] /= determinant;
	inv_Cxx[2][0] /= determinant;
	inv_Cxx[2][1] /= determinant;
	inv_Cxx[2][2] /= determinant;
	inv_Cxx[2][3] /= determinant;
	inv_Cxx[3][0] /= determinant;
	inv_Cxx[3][1] /= determinant;
	inv_Cxx[3][2] /= determinant;
	inv_Cxx[3][3] /= determinant;

}


__kernel void EstimateAR4BrainVoxels(__global float4 *alpha_volumes, __global const float* detrended_fMRI_volumes, __global const float* Brain_Voxels, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Mask[Calculate_3D_Index(x, y, z, DATA_W, DATA_H)] == 1.0f )
    {
        int t = 0;
		float old_value_1, old_value_2, old_value_3, old_value_4, old_value_5;
		float c0 = 0.0f;
        float c1 = 0.0f;
        float c2 = 0.0f;
        float c3 = 0.0f;
        float c4 = 0.0f;

        old_value_1 = detrendedfMRIVolumes[Calculate_4D_Index(x, y, z, 0, DATA_W, DATA_H, DATA_D)];
        c0 += old_value_1 * old_value_1;
        old_value_2 = detrended_fMRI_volumes[Calculate_4D_Index(x, y, z, 1, DATA_W, DATA_H, DATA_D)];
        c0 += old_value_2 * old_value_2;
        c1 += old_value_2 * old_value_1;
        old_value_3 = detrended_fMRI_volumes[Calculate_4D_Index(x, y, z, 2, DATA_W, DATA_H, DATA_D)];
        c0 += old_value_3 * old_value_3;
        c1 += old_value_3 * old_value_2;
        c2 += old_value_3 * old_value_1;
        old_value_4 = detrended_fMRI_volumes[Calculate_4D_Index(x, y, z, 3, DATA_W, DATA_H, DATA_D)];
        c0 += old_value_4 * old_value_4;
        c1 += old_value_4 * old_value_3;
        c2 += old_value_4 * old_value_2;
        c3 += old_value_4 * old_value_1;

        // Estimate c0, c1, c2, c3, c4
        for (t = 4; t < DATA_T; t++)
        {
            // Read data into shared memory
            old_value_5 = detrended_fMRI_volumes[Calculate_4D_Index(x, y, z, t, DATA_W, DATA_H, DATA_D)];
        
            // Sum and multiply the values in shared memory
            c0 += old_value_5 * old_value_5;
            c1 += old_value_5 * old_value_4;
            c2 += old_value_5 * old_value_3;
            c3 += old_value_5 * old_value_2;
            c4 += old_value_5 * old_value_1;

			// Save old values
            old_value_1 = old_value_2;
            old_value_2 = old_value_3;
            old_value_3 = old_value_4;
            old_value_4 = old_value_5;
        }

        c0 /= ((float)(DATA_T) - 1.0f);
        c1 /= ((float)(DATA_T) - 2.0f);
        c2 /= ((float)(DATA_T) - 3.0f);
        c3 /= ((float)(DATA_T) - 4.0f);
        c4 /= ((float)(DATA_T) - 5.0f);

        // Calculate alphas
        float4 r, alphas;

        if (c0 != 0.0f)
        {
            r.x = c1/c0;
            r.y = c2/c0;
            r.z = c3/c0;
            r.w = c4/c0;

            float matrix[4][4];
            matrix[0][0] = 1.0f;
            matrix[1][0] = r.x + 0.001f;
            matrix[2][0] = r.y + 0.001f;
            matrix[3][0] = r.z + 0.001f;

            matrix[0][1] = r.x + 0.001f;
            matrix[1][1] = 1.0f;
            matrix[2][1] = r.x + 0.001f;
            matrix[3][1] = r.y + 0.001f;

            matrix[0][2] = r.y + 0.001f;
            matrix[1][2] = r.x + 0.001f;
            matrix[2][2] = 1.0f;
            matrix[3][2] = r.x + 0.001f;

            matrix[0][3] = r.z + 0.001f;
            matrix[1][3] = r.y + 0.001f;
            matrix[2][3] = r.x + 0.001f;
            matrix[3][3] = 1.0f;

            float inv_matrix[4][4];

            Invert_4x4(matrix, inv_matrix);

            alphas.x = inv_matrix[0][0] * r.x + inv_matrix[0][1] * r.y + inv_matrix[0][2] * r.z + inv_matrix[0][3] * r.w;
            alphas.y = inv_matrix[1][0] * r.x + inv_matrix[1][1] * r.y + inv_matrix[1][2] * r.z + inv_matrix[1][3] * r.w;
            alphas.z = inv_matrix[2][0] * r.x + inv_matrix[2][1] * r.y + inv_matrix[2][2] * r.z + inv_matrix[2][3] * r.w;
            alphas.w = inv_matrix[3][0] * r.x + inv_matrix[3][1] * r.y + inv_matrix[3][2] * r.z + inv_matrix[3][3] * r.w;

            Alpha_Volumes[Calculate_3D_Index(x, y, z, DATA_W, DATA_H)] = alphas;
        }
        else
        {
			alphas.x = 0.0f;
			alphas.y = 0.0f;
			alphas.z = 0.0f;
			alphas.w = 0.0f;
            Alpha_Volumes[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] = alphas;
        }
    }
    else
    {
		alphas.x = 0.0f;
		alphas.y = 0.0f;
		alphas.z = 0.0f;
		alphas.w = 0.0f;
        Alpha_Volumes[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] = alphas;
    }
}

