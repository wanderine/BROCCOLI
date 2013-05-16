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
int Calculate_3D_Index(int a, int b, int c, int DATA_A, int DATA_B)
{
	return a + b * DATA_A + c * DATA_A * DATA_B;
}

int Calculate_4D_Index(int a, int b, int c, int d, int DATA_A, int DATA_B, int DATA_C)
{
	return a + b * DATA_A + c * DATA_A * DATA_B + d * DATA_A * DATA_B * DATA_C;
}

void Get_Parameter_Indices_Kernel(int& i, int& j, int parameter)
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

__kernel void convolutionRows(__global float *Convolved, __global const float* fMRI_Volumes, __constant float *c_Smoothing_Filter_Y, int t, __constant struct DATA_PARAMETERS* DATA)
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
		l_Volume[tIdx.z][tIdx.y][tIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z,t,DATA->DATA_W, DATA->DATA_H, DATA->DATA->DATA_D)];
	}

	if ( (x < DATA->DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA->DATA_H) && ((z + 2) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y][tIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z + 2,t,DATA->DATA_W, DATA->DATA_H, DATA->DATA->DATA_D)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y][tIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z + 4,t,DATA->DATA_W, DATA->DATA_H, DATA->DATA_D)];
	}

	if ( (x < DATA->DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA->DATA_H) && ((z + 6) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y][tIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z + 6,t,DATA->DATA_W, DATA->DATA_H, DATA->DATA_D)];
	}

	// Second half main data + lower apron

	if ( (x < DATA->DATA_W) && ((y + 4) < DATA->DATA_H) && (z < DATA->DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 8][tIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 4,z,t,DATA->DATA_W, DATA->DATA_H, DATA->DATA_D)];
	}

	if ( (x < DATA->DATA_W) && ((y + 4) < DATA->DATA_H) && ((z + 2) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 4,z + 2,t,DATA->DATA_W, DATA->DATA_H, DATA->DATA_D)];
	}

	if ( (x < DATA->DATA_W) && ((y + 4) < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 4,z + 4,t,DATA->DATA_W, DATA->DATA_H, DATA->DATA_D)];
	}

	if ( (x < DATA->DATA_W) && ((y + 4) < DATA->DATA_H) && ((z + 6) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 4,z + 6,t,DATA->DATA_W, DATA->DATA_H, DATA->DATA_D)];
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

		Convolved[Calculate_3D_Index(x,y,z,DATA->DATA_W, DATA->DATA_H)] = sum;
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

		Convolved[Calculate_3D_Index(x,y,z + 2,DATA->DATA_W, DATA->DATA_H)] = sum;
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

		Convolved[Calculate_3D_Index(x,y,z + 4,DATA->DATA_W, DATA->DATA_H)] = sum;
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

		Convolved[Calculate_3D_Index(x,y,z + 6,DATA->DATA_W, DATA->DATA_H)] = sum;
	}
}

__kernel void convolutionColumns(__global float *Convolved, __global const float* fMRI_Volume, __constant float *c_Smoothing_Filter_X, int t, __constant struct DATA_PARAMETERS *DATA)
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
		l_Volume[tIdx.z][tIdx.y][tIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y,z,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA->DATA_W) && (y < DATA->DATA_H) && ((z + 2) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y][tIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y,z + 2,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA->DATA_W) && (y < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y][tIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y,z + 4,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA->DATA_W) && (y < DATA->DATA_H) && ((z + 6) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y][tIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y,z + 6,DATA->DATA_W, DATA->DATA_H)];
	}



	if ( ((x - 4) >= 0) && ((x - 4) < DATA->DATA_W) && ((y + 8) < DATA->DATA_H) && (z < DATA->DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 8][tIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y + 8,z,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA->DATA_W) && ((y + 8) < DATA->DATA_H) && ((z + 2) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y + 8,z + 2,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA->DATA_W) && ((y + 8) < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y + 8,z + 4,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA->DATA_W) && ((y + 8) < DATA->DATA_H) && ((z + 6) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y + 8,z + 6,DATA->DATA_W, DATA->DATA_H)];
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

			Convolved[Calculate_3D_Index(x,y,z,DATA->DATA_W, DATA->DATA_H)] = sum;
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

			Convolved[Calculate_3D_Index(x,y,z + 2,DATA->DATA_W, DATA->DATA_H)] = sum;
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

			Convolved[Calculate_3D_Index(x,y,z + 4,DATA->DATA_W, DATA->DATA_H)] = sum;
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

			Convolved[Calculate_3D_Index(x,y,z + 6,DATA->DATA_W, DATA->DATA_H)] = sum;
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

			Convolved[Calculate_3D_Index(x,y + 8,z,DATA->DATA_W, DATA->DATA_H)] = sum;
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

			Convolved[Calculate_3D_Index(x,y + 8,z + 2,DATA->DATA_W, DATA->DATA_H)] = sum;
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

			Convolved[Calculate_3D_Index(x,y + 8,z + 4,DATA->DATA_W, DATA->DATA_H)] = sum;
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

			Convolved[Calculate_3D_Index(x,y + 8,z + 6,DATA->DATA_W, DATA->DATA_H)] = sum;
		}
	}
}

__kernel void convolutionRods(__global float *Convolved, __global const float* fMRI_Volume, __constant float *c_Smoothing_Filter_Z, int t, __constant struct DATA_PARAMETERS *DATA)
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
		l_Volume[tIdx.z][tIdx.y][tIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y,z - 4,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( (x < DATA->DATA_W) && ((y + 2) < DATA->DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 2][tIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y + 2,z - 4,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( (x < DATA->DATA_W) && ((y + 4) < DATA->DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 4][tIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y + 4,z - 4,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( (x < DATA->DATA_W) && ((y + 6) < DATA->DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 6][tIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y + 6,z - 4,DATA->DATA_W, DATA->DATA_H)];
	}

	// Second half main data + below apron

	if ( (x < DATA->DATA_W) && (y < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y][tIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y,z + 4,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( (x < DATA->DATA_W) && ((y + 2) < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y + 2][tIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y + 2,z + 4,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( (x < DATA->DATA_W) && ((y + 4) < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y + 4][tIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y + 4,z + 4,DATA->DATA_W, DATA->DATA_H)];
	}

	if ( (x < DATA->DATA_W) && ((y + 6) < DATA->DATA_H) && ((z + 4) < DATA->DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y + 6][tIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y + 6,z + 4,DATA->DATA_W, DATA->DATA_H)];
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

		Filter_Responses[Calculate_4D_Index(x,y,z,t,DATA->DATA_W,DATA->DATA_H,DATA->DATA_D)] = sum;
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

		Convolved[Calculate_4D_Index(x,y + 2,z,t,DATA->DATA_W,DATA->DATA_H,DATA->DATA_D)] = sum;
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

		Convolved[Calculate_4D_Index(x,y + 4,z,t,DATA->DATA_W,DATA->DATA_H,DATA->DATA_D)] = sum;
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

		Convolved[Calculate_4D_Index(x,y + 6,z,t,DATA->DATA_W,DATA->DATA_H,DATA->DATA_D)] = sum;
	}
}

// Non-separable 3D convolution

__kernel void convolutionNonseparable3DComplex(float2 *Filter_Response_1, float2 *Filter_Response_2, float2 *Filter_Response_3, const float* Volume, __constant float2 *c_Quadrature_Filter_1, __constant float2 *c_Quadrature_Filter_2, __constant float2 *c_Quadrature_Filter_3, __constant struct DATA_PARAMETERS *DATA)
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


__kernel void CalculateActivityMapGLM(float* Activity_Map, const float* fMRI_Volumes, const float* Brain_Voxels, float ctxtxc, __constant float *c_X_GLM, __constant float* *c_xtxxt_GLM, __constant float* c_Contrast_Vector, __constant struct DATA_PARAMETERS *DATA)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Brain_Voxels[x + y * DATA_W + z * DATA_W * DATA_H] == 0.0f )
	{
		activity_map[x + y * DATA_W + z * DATA_W * DATA_H] = 0.0f;
		return;
	}

	int t = 0;
	float beta1 = 0.0f;
	float beta2 = 0.0f;
	float eps, meaneps, vareps;

	// Calculate betahat, i.e. multiply (x^T x)^(-1) x^T with Y
	for (t = 0; t < DATA_T; t ++)
	{
		// Sum and multiply the values in shared memory
		float temp = fMRI_Volumes[Calculate_4D_Index(x,y,z,t,DATA_W,DATA_H,DATA_D)];
		beta1 += temp * c_xtxxt_GLM[t];
		beta2 += temp * c_xtxxt_GLM[DATA_T + t];
	}

	// Calculate the mean of the error eps
	meaneps = 0.0f;
	for (t = 0; t < DATA_T; t ++)
	{
		meaneps += fMRI_Volumes[Calculate_4D_Index(x,y,z,t,DATA_W,DATA_H,DATA_D)] - beta1 * c_X_GLM[t] - beta2 * c_X_GLM[DATA_T + t];
	}
	meaneps /= (float)DATA_T;

	// Now calculate the variance of eps
	vareps = 0.0f;
	for (t = 0; t < DATA_T; t ++)
	{
		eps = fMRI_Volumes[Calculate_4D_Index(x,y,z,t,DATA_W,DATA_H,DATA_D)];
		eps -= beta1 * c_X_GLM[t];
		eps -= beta2 * c_X_GLM[DATA_T + t];
		vareps += (eps - meaneps) * (eps - meaneps);
	}

	vareps /= ((float)DATA_T - 3.0f);
	
	Activity_Map[Calculate_3D_Index(x,y,z,DATA_W,DATA_H)] = (c_Contrast_Vector[0] * beta1 + c_Contrast_Vector[1] * beta2)  * rsqrtf(vareps * ctxtxc);
}

__kernel void GeneratePermutedfMRIVolumesAR4(float* Permuted_fMRI_Volumes, const float4* Alpha_Volumes, const float* Whitened_fMRI_Volumes, const float* Brain_Voxels, __constant unsignet int *c_Permutation_Vector, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Brain_Voxels[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
    {
        int t = 0;
		float old_value_1, old_value_2, old_value_3, old_value_4, old_value_5;
		float4 alphas = Alpha_Volumes[x + y * DATA_W + z * DATA_W * DATA_H];
        
        old_value1 = Whitened_fMRI_Volumes[Calculate_4D_Index(x, y, z, c_Permutation_Vector[0], DATA_W, DATA_H, DATA_D)];
		old_value2 = alphas.x * old_value1  + Whitened_fMRI_Volumes[Calculate_4D_Index(x, y, z, c_Permutation_Vector[1], DATA_W, DATA_H, DATA_D)];
		old_value3 = alphas.x * old_value2  + alphas.y * old_value1 + Whitened_fMRI_Volumes[Calculate_4D_Index(x, y, z, c_Permutation_Vector[2], DATA_W, DATA_H, DATA_D)];
		old_value4 = alphas.x * old_value3  + alphas.y * old_value2 + alphas.z * old_value1 + Whitened_fMRI_Volumes[Calculate_4D_Index(x, y, z, c_Permutation_Vector[3], DATA_W, DATA_H, DATA_D)];

        permuted_fMRI_volumes[Calculate_4D_Index(x, y, z, 0, DATA_W, DATA_H, DATA_D)] =  old_value1;
        permuted_fMRI_volumes[Calculate_4D_Index(x, y, z, 1, DATA_W, DATA_H, DATA_D)] =  old_value2;
        permuted_fMRI_volumes[Calculate_4D_Index(x, y, z, 2, DATA_W, DATA_H, DATA_D)] =  old_value3;
        permuted_fMRI_volumes[Calculate_4D_Index(x, y, z, 3, DATA_W, DATA_H, DATA_D)] =  old_value4;

        // Read the data in a permuted order and apply an inverse whitening transform
        for (t = 4; t < DATA_T; t++)
        {
            // Calculate the unwhitened, permuted, timeseries
            old_value5 = alphas.x * old_value4 + alphas.y * old_value3 + alphas.z * old_value2 + alphas.w * old_value1 + whitened_fMRI_volumes[Calculate_4D_Index(x, y, z, c_Permutation_Vector[t], DATA_W, DATA_H, DATA_D)];
			
            permuted_fMRI_volumes[Calculate_4D_Index(x, y, z, t, DATA_W, DATA_H, DATA_D)] = old_value5;

            // Save old values
			old_value_1 = old_value_2;
            old_value_2 = old_value_3;
            old_value_3 = old_value_4;
            old_value_4 = old_value_5;
        }
    }
}

__kernel void ApplyWhiteningAR4(float* Whitened_fMRI_Volumes, const float* Detrended_fMRI_Volumes, const float4 *Alpha_Volumes, const float* Brain_Voxels, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Brain_Voxels[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
    {
        int t = 0;
		float old_value_1, old_value_2, old_value_3, old_value_4, old_value_5;
        float4 alphas = alpha_volumes[x + y * DATA_W + z * DATA_W * DATA_H];
        
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


__kernel void EstimateAR4BrainVoxels(float4 *alpha_volumes, const float* detrended_fMRI_volumes, const float* Brain_Voxels, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Brain_Voxels[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
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
            alphas.y = inv_matrix[1][0] * r.y + inv_matrix[1][1] * r.y + inv_matrix[1][2] * r.z + inv_matrix[1][3] * r.w;
            alphas.z = inv_matrix[2][0] * r.z + inv_matrix[2][1] * r.y + inv_matrix[2][2] * r.z + inv_matrix[2][3] * r.w;
            alphas.w = inv_matrix[3][0] * r.w + inv_matrix[3][1] * r.y + inv_matrix[3][2] * r.z + inv_matrix[3][3] * r.w;

            alpha_volumes[x + y * DATA_W + z * DATA_W * DATA_H] = alphas;
        }
        else
        {
			alphas.x = 0.0f;
			alphas.y = 0.0f;
			alphas.z = 0.0f;
			alphas.w = 0.0f;
            alpha_volumes[x + y * DATA_W + z * DATA_W * DATA_H] = alphas;
        }
    }
    else
    {
		alphas.x = 0.0f;
		alphas.y = 0.0f;
		alphas.z = 0.0f;
		alphas.w = 0.0f;
        alpha_volumes[x + y * DATA_W + z * DATA_W * DATA_H] = alphas;
    }
}