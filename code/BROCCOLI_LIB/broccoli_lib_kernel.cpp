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




inline __device__ int Calculate_3D_Index(int a, int b, int c, int DATA_A, int DATA_B)
{
	return a + b * DATA_A + c * DATA_A * DATA_B;
}

inline __device__ int Calculate_4D_Index(int a, int b, int c, int d, int DATA_A, int DATA_B, int DATA_C)
{
	return a + b * DATA_A + c * DATA_A * DATA_B + d * DATA_A * DATA_B * DATA_C;
}

/*
extern "C" __global__ void convolutionRows(float *Filter_Responses, float* fMRI_Volumes, float* Brain_Voxels, int t, int DATA_W, int DATA_H, int DATA_D, int xBlockDifference, int yBlockDifference, int zBlockDifference)
{
	volatile int x = blockIdx.x * blockDim.x + threadIdx.x;
	volatile int y = blockIdx.y * blockDim.y + threadIdx.y;
	volatile int z = blockIdx.z * blockDim.z * 4 + threadIdx.z;

    if(x >= (DATA_W + xBlockDifference) || y >= (DATA_H + yBlockDifference) || z >= (DATA_D + zBlockDifference))
        return;



	// 8 * 8 * 32 valid filter responses = 2048
	__shared__ float s_Volume[8][16][32];

	// Reset shared memory

	s_Volume[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x] = 0.0f;

	s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x] = 0.0f;

    //if (Brain_Voxels[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)] == 0.0f)
    //	return;

	// Read data into shared memory

	// Upper apron + first half main data

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.z][threadIdx.y][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z,t,DATA_W, DATA_H, DATA_D)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 2) < DATA_D) )
	{
		s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z + 2,t,DATA_W, DATA_H, DATA_D)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 4) < DATA_D) )
	{
		s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z + 4,t,DATA_W, DATA_H, DATA_D)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 6) < DATA_D) )
	{
		s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z + 6,t,DATA_W, DATA_H, DATA_D)];
	}

	// Second half main data + lower apron

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 4,z,t,DATA_W, DATA_H, DATA_D)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 2) < DATA_D) )
	{
		s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 4,z + 2,t,DATA_W, DATA_H, DATA_D)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 4) < DATA_D) )
	{
		s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 4,z + 4,t,DATA_W, DATA_H, DATA_D)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 6) < DATA_D) )
	{
		s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 4,z + 6,t,DATA_W, DATA_H, DATA_D)];
	}


	__syncthreads();

	// Only threads within the volume do the convolution
	if ( (x < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z][threadIdx.y + 0][threadIdx.x] * c_Smoothing_Filter_Y[8];
		sum += s_Volume[threadIdx.z][threadIdx.y + 1][threadIdx.x] * c_Smoothing_Filter_Y[7];
		sum += s_Volume[threadIdx.z][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Filter_Y[6];
		sum += s_Volume[threadIdx.z][threadIdx.y + 3][threadIdx.x] * c_Smoothing_Filter_Y[5];
		sum += s_Volume[threadIdx.z][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Filter_Y[4];
		sum += s_Volume[threadIdx.z][threadIdx.y + 5][threadIdx.x] * c_Smoothing_Filter_Y[3];
		sum += s_Volume[threadIdx.z][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Filter_Y[2];
		sum += s_Volume[threadIdx.z][threadIdx.y + 7][threadIdx.x] * c_Smoothing_Filter_Y[1];
		sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x] * c_Smoothing_Filter_Y[0];

		Filter_Responses[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)] = sum;
	}

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 2) < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 0][threadIdx.x] * c_Smoothing_Filter_Y[8];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x] * c_Smoothing_Filter_Y[7];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Filter_Y[6];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x] * c_Smoothing_Filter_Y[5];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Filter_Y[4];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x] * c_Smoothing_Filter_Y[3];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Filter_Y[2];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x] * c_Smoothing_Filter_Y[1];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x] * c_Smoothing_Filter_Y[0];

		Filter_Responses[Calculate_3D_Index(x,y,z + 2,DATA_W, DATA_H)] = sum;
	}

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 4) < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 0][threadIdx.x] * c_Smoothing_Filter_Y[8];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x] * c_Smoothing_Filter_Y[7];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Filter_Y[6];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x] * c_Smoothing_Filter_Y[5];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Filter_Y[4];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x] * c_Smoothing_Filter_Y[3];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Filter_Y[2];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x] * c_Smoothing_Filter_Y[1];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x] * c_Smoothing_Filter_Y[0];

		Filter_Responses[Calculate_3D_Index(x,y,z + 4,DATA_W, DATA_H)] = sum;
	}

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 6) < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 0][threadIdx.x] * c_Smoothing_Filter_Y[8];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x] * c_Smoothing_Filter_Y[7];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Filter_Y[6];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x] * c_Smoothing_Filter_Y[5];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Filter_Y[4];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x] * c_Smoothing_Filter_Y[3];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Filter_Y[2];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x] * c_Smoothing_Filter_Y[1];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x] * c_Smoothing_Filter_Y[0];

		Filter_Responses[Calculate_3D_Index(x,y,z + 6,DATA_W, DATA_H)] = sum;
	}
}


extern "C" __global__ void convolutionRowsNew(float* __restrict__ Filter_Responses, float* __restrict__ fMRI_Volumes, float* __restrict__ Brain_Voxels, int t, int DATA_W, int DATA_H, int DATA_D, int xBlockDifference, int yBlockDifference, int zBlockDifference)
{
	volatile int x = blockIdx.x * VALID_FILTER_RESPONSES_X_CONVOLUTION_ROWS + threadIdx.x;
	volatile int y = blockIdx.y * VALID_FILTER_RESPONSES_Y_CONVOLUTION_ROWS + threadIdx.y;
	volatile int z = blockIdx.z * blockDim.z + threadIdx.z;

    if(x >= (DATA_W + xBlockDifference) || y >= (DATA_H + yBlockDifference) || z >= (DATA_D + zBlockDifference))
        return;



	__shared__ float s_Volume[96][64];

	// Reset shared memory
	s_Volume[threadIdx.y][threadIdx.x] 				= 0.0f;
	s_Volume[threadIdx.y][threadIdx.x + 32] 		= 0.0f;
	s_Volume[threadIdx.y + 32][threadIdx.x] 		= 0.0f;
	s_Volume[threadIdx.y + 32][threadIdx.x + 32] 	= 0.0f;
	s_Volume[threadIdx.y + 64][threadIdx.x] 		= 0.0f;
	s_Volume[threadIdx.y + 64][threadIdx.x + 32] 	= 0.0f;

    //if (Brain_Voxels[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)] == 0.0f)
    //	return;

	// Read data into shared memory

	// Y
	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.y][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z,t,DATA_W, DATA_H, DATA_D)];
	}

	if ( ((x + 32) < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.y][threadIdx.x + 32] = fMRI_Volumes[Calculate_4D_Index(x + 32,y - 4,z,t,DATA_W, DATA_H, DATA_D)];
	}

	// Y + 32
	if ( (x < DATA_W) && ((y + 28) < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.y + 32][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 28,z,t,DATA_W, DATA_H, DATA_D)];
	}

	if ( ((x + 32) < DATA_W) && ((y + 28) < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.y + 32][threadIdx.x + 32] = fMRI_Volumes[Calculate_4D_Index(x + 32,y + 28,z,t,DATA_W, DATA_H, DATA_D)];
	}

	// Y + 64
	if ( (x < DATA_W) && ((y + 60) < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.y + 64][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 60,z,t,DATA_W, DATA_H, DATA_D)];
	}

	if ( ((x + 32) < DATA_W) && ((y + 60) < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.y + 64][threadIdx.x + 32] = fMRI_Volumes[Calculate_4D_Index(x + 32,y + 60,z,t,DATA_W, DATA_H, DATA_D)];
	}

	__syncthreads();

	// Only threads within the volume do the convolution
	if ( (x < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.y + 0][threadIdx.x] * c_Smoothing_Filter_Y[8];
		sum += s_Volume[threadIdx.y + 1][threadIdx.x] * c_Smoothing_Filter_Y[7];
		sum += s_Volume[threadIdx.y + 2][threadIdx.x] * c_Smoothing_Filter_Y[6];
		sum += s_Volume[threadIdx.y + 3][threadIdx.x] * c_Smoothing_Filter_Y[5];
		sum += s_Volume[threadIdx.y + 4][threadIdx.x] * c_Smoothing_Filter_Y[4];
		sum += s_Volume[threadIdx.y + 5][threadIdx.x] * c_Smoothing_Filter_Y[3];
		sum += s_Volume[threadIdx.y + 6][threadIdx.x] * c_Smoothing_Filter_Y[2];
		sum += s_Volume[threadIdx.y + 7][threadIdx.x] * c_Smoothing_Filter_Y[1];
		sum += s_Volume[threadIdx.y + 8][threadIdx.x] * c_Smoothing_Filter_Y[0];

		Filter_Responses[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)] = sum;
	}

	if ( ((x + 32) < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.y + 0][threadIdx.x + 32] * c_Smoothing_Filter_Y[8];
		sum += s_Volume[threadIdx.y + 1][threadIdx.x + 32] * c_Smoothing_Filter_Y[7];
		sum += s_Volume[threadIdx.y + 2][threadIdx.x + 32] * c_Smoothing_Filter_Y[6];
		sum += s_Volume[threadIdx.y + 3][threadIdx.x + 32] * c_Smoothing_Filter_Y[5];
		sum += s_Volume[threadIdx.y + 4][threadIdx.x + 32] * c_Smoothing_Filter_Y[4];
		sum += s_Volume[threadIdx.y + 5][threadIdx.x + 32] * c_Smoothing_Filter_Y[3];
		sum += s_Volume[threadIdx.y + 6][threadIdx.x + 32] * c_Smoothing_Filter_Y[2];
		sum += s_Volume[threadIdx.y + 7][threadIdx.x + 32] * c_Smoothing_Filter_Y[1];
		sum += s_Volume[threadIdx.y + 8][threadIdx.x + 32] * c_Smoothing_Filter_Y[0];

		Filter_Responses[Calculate_3D_Index(x + 32,y,z,DATA_W, DATA_H)] = sum;
	}

	if ( (x < DATA_W) && ((y + 32) < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.y + 32][threadIdx.x] * c_Smoothing_Filter_Y[8];
		sum += s_Volume[threadIdx.y + 33][threadIdx.x] * c_Smoothing_Filter_Y[7];
		sum += s_Volume[threadIdx.y + 34][threadIdx.x] * c_Smoothing_Filter_Y[6];
		sum += s_Volume[threadIdx.y + 35][threadIdx.x] * c_Smoothing_Filter_Y[5];
		sum += s_Volume[threadIdx.y + 36][threadIdx.x] * c_Smoothing_Filter_Y[4];
		sum += s_Volume[threadIdx.y + 37][threadIdx.x] * c_Smoothing_Filter_Y[3];
		sum += s_Volume[threadIdx.y + 38][threadIdx.x] * c_Smoothing_Filter_Y[2];
		sum += s_Volume[threadIdx.y + 39][threadIdx.x] * c_Smoothing_Filter_Y[1];
		sum += s_Volume[threadIdx.y + 40][threadIdx.x] * c_Smoothing_Filter_Y[0];

		Filter_Responses[Calculate_3D_Index(x,y + 32,z,DATA_W, DATA_H)] = sum;
	}

	if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 32] * c_Smoothing_Filter_Y[8];
		sum += s_Volume[threadIdx.y + 33][threadIdx.x + 32] * c_Smoothing_Filter_Y[7];
		sum += s_Volume[threadIdx.y + 34][threadIdx.x + 32] * c_Smoothing_Filter_Y[6];
		sum += s_Volume[threadIdx.y + 35][threadIdx.x + 32] * c_Smoothing_Filter_Y[5];
		sum += s_Volume[threadIdx.y + 36][threadIdx.x + 32] * c_Smoothing_Filter_Y[4];
		sum += s_Volume[threadIdx.y + 37][threadIdx.x + 32] * c_Smoothing_Filter_Y[3];
		sum += s_Volume[threadIdx.y + 38][threadIdx.x + 32] * c_Smoothing_Filter_Y[2];
		sum += s_Volume[threadIdx.y + 39][threadIdx.x + 32] * c_Smoothing_Filter_Y[1];
		sum += s_Volume[threadIdx.y + 40][threadIdx.x + 32] * c_Smoothing_Filter_Y[0];

		Filter_Responses[Calculate_3D_Index(x + 32,y + 32,z,DATA_W, DATA_H)] = sum;
	}


	if (threadIdx.y < 24)
	{
		if ( (x < DATA_W) && ((y + 64) < DATA_H) && (z < DATA_D) )
		{
			float sum = 0.0f;

			sum += s_Volume[threadIdx.y + 64][threadIdx.x] * c_Smoothing_Filter_Y[8];
			sum += s_Volume[threadIdx.y + 65][threadIdx.x] * c_Smoothing_Filter_Y[7];
			sum += s_Volume[threadIdx.y + 66][threadIdx.x] * c_Smoothing_Filter_Y[6];
			sum += s_Volume[threadIdx.y + 67][threadIdx.x] * c_Smoothing_Filter_Y[5];
			sum += s_Volume[threadIdx.y + 68][threadIdx.x] * c_Smoothing_Filter_Y[4];
			sum += s_Volume[threadIdx.y + 69][threadIdx.x] * c_Smoothing_Filter_Y[3];
			sum += s_Volume[threadIdx.y + 70][threadIdx.x] * c_Smoothing_Filter_Y[2];
			sum += s_Volume[threadIdx.y + 71][threadIdx.x] * c_Smoothing_Filter_Y[1];
			sum += s_Volume[threadIdx.y + 72][threadIdx.x] * c_Smoothing_Filter_Y[0];

			Filter_Responses[Calculate_3D_Index(x,y + 64,z,DATA_W, DATA_H)] = sum;
		}

		if ( ((x + 32) < DATA_W) && ((y + 64) < DATA_H) && (z < DATA_D) )
		{
			float sum = 0.0f;

			sum += s_Volume[threadIdx.y + 64][threadIdx.x + 32] * c_Smoothing_Filter_Y[8];
			sum += s_Volume[threadIdx.y + 65][threadIdx.x + 32] * c_Smoothing_Filter_Y[7];
			sum += s_Volume[threadIdx.y + 66][threadIdx.x + 32] * c_Smoothing_Filter_Y[6];
			sum += s_Volume[threadIdx.y + 67][threadIdx.x + 32] * c_Smoothing_Filter_Y[5];
			sum += s_Volume[threadIdx.y + 68][threadIdx.x + 32] * c_Smoothing_Filter_Y[4];
			sum += s_Volume[threadIdx.y + 69][threadIdx.x + 32] * c_Smoothing_Filter_Y[3];
			sum += s_Volume[threadIdx.y + 70][threadIdx.x + 32] * c_Smoothing_Filter_Y[2];
			sum += s_Volume[threadIdx.y + 71][threadIdx.x + 32] * c_Smoothing_Filter_Y[1];
			sum += s_Volume[threadIdx.y + 72][threadIdx.x + 32] * c_Smoothing_Filter_Y[0];

			Filter_Responses[Calculate_3D_Index(x + 32,y + 64,z,DATA_W, DATA_H)] = sum;
		}
	}
}


extern "C" __global__ void convolutionColumns(float *Filter_Responses, float* fMRI_Volume, float* Brain_Voxels, int t, int DATA_W, int DATA_H, int DATA_D, int xBlockDifference, int yBlockDifference, int zBlockDifference)
{
	volatile int x = blockIdx.x * blockDim.x / 32 * 24 + threadIdx.x;
	volatile int y = blockIdx.y * blockDim.y * 2 + threadIdx.y;
	volatile int z = blockIdx.z * blockDim.z * 4 + threadIdx.z;

    if(x >= (DATA_W + xBlockDifference) || y >= (DATA_H + yBlockDifference) || z >= (DATA_D + zBlockDifference))
        return;



	// 8 * 8 * 32 valid filter responses = 2048
	__shared__ float s_Volume[8][16][32];

		// Reset shared memory

	s_Volume[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x] = 0.0f;

	s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x] = 0.0f;

    //if (Brain_Voxels[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)] == 0.0f)
    // 	return;

	// Read data into shared memory


	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.z][threadIdx.y][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y,z,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && ((z + 2) < DATA_D) )
	{
		s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y,z + 2,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && ((z + 4) < DATA_D) )
	{
		s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y,z + 4,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && ((z + 6) < DATA_D) )
	{
		s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y,z + 6,DATA_W, DATA_H)];
	}



	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 8) < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y + 8,z,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 8) < DATA_H) && ((z + 2) < DATA_D) )
	{
		s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y + 8,z + 2,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 8) < DATA_H) && ((z + 4) < DATA_D) )
	{
		s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y + 8,z + 4,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 8) < DATA_H) && ((z + 6) < DATA_D) )
	{
		s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y + 8,z + 6,DATA_W, DATA_H)];
	}

	__syncthreads();

	// Only threads within the volume do the convolution
	if (threadIdx.x < 24)
	{
		if ( (x < DATA_W) && (y < DATA_H) && (z < DATA_D) )
		{
		    float sum = 0.0f;

			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Responses[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)] = sum;
		}

		if ( (x < DATA_W) && (y < DATA_H) && ((z + 2) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Responses[Calculate_3D_Index(x,y,z + 2,DATA_W, DATA_H)] = sum;
		}

		if ( (x < DATA_W) && (y < DATA_H) && ((z + 4) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Responses[Calculate_3D_Index(x,y,z + 4,DATA_W, DATA_H)] = sum;
		}

		if ( (x < DATA_W) && (y < DATA_H) && ((z + 6) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 8] * c_Smoothing_Filter_X[0];


			Filter_Responses[Calculate_3D_Index(x,y,z + 6,DATA_W, DATA_H)] = sum;
		}

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && (z < DATA_D) )
		{
		    float sum = 0.0f;

			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 8] * c_Smoothing_Filter_X[0];


			Filter_Responses[Calculate_3D_Index(x,y + 8,z,DATA_W, DATA_H)] = sum;
		}

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 2) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Responses[Calculate_3D_Index(x,y + 8,z + 2,DATA_W, DATA_H)] = sum;
		}

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 4) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 8] * c_Smoothing_Filter_X[0];


			Filter_Responses[Calculate_3D_Index(x,y + 8,z + 4,DATA_W, DATA_H)] = sum;
		}

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 6) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 8] * c_Smoothing_Filter_X[0];


			Filter_Responses[Calculate_3D_Index(x,y + 8,z + 6,DATA_W, DATA_H)] = sum;
		}

	}
}



extern "C" __global__ void convolutionColumnsNew(float* __restrict__ Filter_Responses, float* __restrict__ fMRI_Volume, float* __restrict__ Brain_Voxels, int t, int DATA_W, int DATA_H, int DATA_D, int xBlockDifference, int yBlockDifference, int zBlockDifference)
{
	volatile int x = blockIdx.x * VALID_FILTER_RESPONSES_X_CONVOLUTION_COLUMNS + threadIdx.x;
	volatile int y = blockIdx.y * VALID_FILTER_RESPONSES_Y_CONVOLUTION_COLUMNS + threadIdx.y;
	volatile int z = blockIdx.z * blockDim.z + threadIdx.z;

    if(x >= (DATA_W + xBlockDifference) || y >= (DATA_H + yBlockDifference) || z >= (DATA_D + zBlockDifference))
        return;



	__shared__ float s_Volume[64][96];

	// Reset shared memory
	s_Volume[threadIdx.y][threadIdx.x] 				= 0.0f;
	s_Volume[threadIdx.y + 32][threadIdx.x] 		= 0.0f;
	s_Volume[threadIdx.y][threadIdx.x + 32] 		= 0.0f;
	s_Volume[threadIdx.y + 32][threadIdx.x + 32] 	= 0.0f;
	s_Volume[threadIdx.y][threadIdx.x + 64] 		= 0.0f;
	s_Volume[threadIdx.y + 32][threadIdx.x + 64] 	= 0.0f;

    //if (Brain_Voxels[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)] == 0.0f)
    //   	return;

	// Read data into shared memory

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.y][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y,z,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 32) < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.y + 32][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x - 4,y + 32,z,DATA_W, DATA_H)];
	}


	if ( ((x + 28) < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.y][threadIdx.x + 32] = fMRI_Volume[Calculate_3D_Index(x + 28,y,z,DATA_W, DATA_H)];
	}

	if ( ((x + 28) < DATA_W) && ((y + 32) < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.y + 32][threadIdx.x + 32] = fMRI_Volume[Calculate_3D_Index(x + 28,y + 32,z,DATA_W, DATA_H)];
	}


	if ( ((x + 64) < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.y][threadIdx.x + 64] = fMRI_Volume[Calculate_3D_Index(x + 60,y,z,DATA_W, DATA_H)];
	}

	if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.y + 32][threadIdx.x + 64] = fMRI_Volume[Calculate_3D_Index(x + 60,y + 32,z,DATA_W, DATA_H)];
	}

	__syncthreads();

	// Only threads within the volume do the convolution
	if ( (x < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

	    sum += s_Volume[threadIdx.y][threadIdx.x + 0] * c_Smoothing_Filter_X[8];
		sum += s_Volume[threadIdx.y][threadIdx.x + 1] * c_Smoothing_Filter_X[7];
		sum += s_Volume[threadIdx.y][threadIdx.x + 2] * c_Smoothing_Filter_X[6];
		sum += s_Volume[threadIdx.y][threadIdx.x + 3] * c_Smoothing_Filter_X[5];
		sum += s_Volume[threadIdx.y][threadIdx.x + 4] * c_Smoothing_Filter_X[4];
		sum += s_Volume[threadIdx.y][threadIdx.x + 5] * c_Smoothing_Filter_X[3];
		sum += s_Volume[threadIdx.y][threadIdx.x + 6] * c_Smoothing_Filter_X[2];
		sum += s_Volume[threadIdx.y][threadIdx.x + 7] * c_Smoothing_Filter_X[1];
		sum += s_Volume[threadIdx.y][threadIdx.x + 8] * c_Smoothing_Filter_X[0];

		Filter_Responses[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)] = sum;
	}

	if ( (x < DATA_W) && ((y + 32) < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

	    sum += s_Volume[threadIdx.y + 32][threadIdx.x + 0] * c_Smoothing_Filter_X[8];
		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 1] * c_Smoothing_Filter_X[7];
		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 2] * c_Smoothing_Filter_X[6];
		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 3] * c_Smoothing_Filter_X[5];
		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 4] * c_Smoothing_Filter_X[4];
		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 5] * c_Smoothing_Filter_X[3];
		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 6] * c_Smoothing_Filter_X[2];
		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 7] * c_Smoothing_Filter_X[1];
		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 8] * c_Smoothing_Filter_X[0];

		Filter_Responses[Calculate_3D_Index(x,y + 32,z,DATA_W, DATA_H)] = sum;
	}

	if ( ((x + 32) < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

	    sum += s_Volume[threadIdx.y][threadIdx.x + 32] * c_Smoothing_Filter_X[8];
		sum += s_Volume[threadIdx.y][threadIdx.x + 33] * c_Smoothing_Filter_X[7];
		sum += s_Volume[threadIdx.y][threadIdx.x + 34] * c_Smoothing_Filter_X[6];
		sum += s_Volume[threadIdx.y][threadIdx.x + 35] * c_Smoothing_Filter_X[5];
		sum += s_Volume[threadIdx.y][threadIdx.x + 36] * c_Smoothing_Filter_X[4];
		sum += s_Volume[threadIdx.y][threadIdx.x + 37] * c_Smoothing_Filter_X[3];
		sum += s_Volume[threadIdx.y][threadIdx.x + 38] * c_Smoothing_Filter_X[2];
		sum += s_Volume[threadIdx.y][threadIdx.x + 39] * c_Smoothing_Filter_X[1];
		sum += s_Volume[threadIdx.y][threadIdx.x + 40] * c_Smoothing_Filter_X[0];

		Filter_Responses[Calculate_3D_Index(x + 32,y,z,DATA_W, DATA_H)] = sum;
	}

	if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

	    sum += s_Volume[threadIdx.y + 32][threadIdx.x + 32] * c_Smoothing_Filter_X[8];
		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 33] * c_Smoothing_Filter_X[7];
		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 34] * c_Smoothing_Filter_X[6];
		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 35] * c_Smoothing_Filter_X[5];
		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 36] * c_Smoothing_Filter_X[4];
		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 37] * c_Smoothing_Filter_X[3];
		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 38] * c_Smoothing_Filter_X[2];
		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 39] * c_Smoothing_Filter_X[1];
		sum += s_Volume[threadIdx.y + 32][threadIdx.x + 40] * c_Smoothing_Filter_X[0];

		Filter_Responses[Calculate_3D_Index(x + 32,y + 32,z,DATA_W, DATA_H)] = sum;
	}

	if (threadIdx.x < 24)
	{
		if ( ((x + 64) < DATA_W) && (y < DATA_H) && (z < DATA_D) )
		{
		    float sum = 0.0f;

		    sum += s_Volume[threadIdx.y][threadIdx.x + 64] * c_Smoothing_Filter_X[8];
			sum += s_Volume[threadIdx.y][threadIdx.x + 65] * c_Smoothing_Filter_X[7];
			sum += s_Volume[threadIdx.y][threadIdx.x + 66] * c_Smoothing_Filter_X[6];
			sum += s_Volume[threadIdx.y][threadIdx.x + 67] * c_Smoothing_Filter_X[5];
			sum += s_Volume[threadIdx.y][threadIdx.x + 68] * c_Smoothing_Filter_X[4];
			sum += s_Volume[threadIdx.y][threadIdx.x + 69] * c_Smoothing_Filter_X[3];
			sum += s_Volume[threadIdx.y][threadIdx.x + 70] * c_Smoothing_Filter_X[2];
			sum += s_Volume[threadIdx.y][threadIdx.x + 71] * c_Smoothing_Filter_X[1];
			sum += s_Volume[threadIdx.y][threadIdx.x + 72] * c_Smoothing_Filter_X[0];

			Filter_Responses[Calculate_3D_Index(x + 64,y,z,DATA_W, DATA_H)] = sum;
		}

		if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) && (z < DATA_D) )
		{
		    float sum = 0.0f;

		    sum += s_Volume[threadIdx.y + 32][threadIdx.x + 64] * c_Smoothing_Filter_X[8];
			sum += s_Volume[threadIdx.y + 32][threadIdx.x + 65] * c_Smoothing_Filter_X[7];
			sum += s_Volume[threadIdx.y + 32][threadIdx.x + 66] * c_Smoothing_Filter_X[6];
			sum += s_Volume[threadIdx.y + 32][threadIdx.x + 67] * c_Smoothing_Filter_X[5];
			sum += s_Volume[threadIdx.y + 32][threadIdx.x + 68] * c_Smoothing_Filter_X[4];
			sum += s_Volume[threadIdx.y + 32][threadIdx.x + 69] * c_Smoothing_Filter_X[3];
			sum += s_Volume[threadIdx.y + 32][threadIdx.x + 70] * c_Smoothing_Filter_X[2];
			sum += s_Volume[threadIdx.y + 32][threadIdx.x + 71] * c_Smoothing_Filter_X[1];
			sum += s_Volume[threadIdx.y + 32][threadIdx.x + 72] * c_Smoothing_Filter_X[0];

			Filter_Responses[Calculate_3D_Index(x + 64,y + 32,z,DATA_W, DATA_H)] = sum;
		}
	}
}



extern "C" __global__ void convolutionRods(float *Filter_Responses, float* fMRI_Volume, float* Brain_Voxels, int t, int DATA_W, int DATA_H, int DATA_D, int xBlockDifference, int yBlockDifference, int zBlockDifference)
{
	volatile int x = blockIdx.x * blockDim.x + threadIdx.x;
	volatile int y = blockIdx.y * blockDim.y * 4 + threadIdx.y;
	volatile int z = blockIdx.z * blockDim.z + threadIdx.z;

    if(x >= (DATA_W + xBlockDifference) || y >= (DATA_H + yBlockDifference) || z >= (DATA_D + zBlockDifference))
        return;



	// 8 * 8 * 32 valid filter responses = 2048
	__shared__ float s_Volume[16][8][32];

	// Reset shared memory

	s_Volume[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z][threadIdx.y + 2][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z][threadIdx.y + 4][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z][threadIdx.y + 6][threadIdx.x] = 0.0f;

	s_Volume[threadIdx.z + 8][threadIdx.y][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z + 8][threadIdx.y + 2][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z + 8][threadIdx.y + 4][threadIdx.x] = 0.0f;
	s_Volume[threadIdx.z + 8][threadIdx.y + 6][threadIdx.x] = 0.0f;

    //if (Brain_Voxels[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)] == 0.0f)
    //   	return;

	// Read data into shared memory

	// Above apron + first half main data

	if ( (x < DATA_W) && (y < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		s_Volume[threadIdx.z][threadIdx.y][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y,z - 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 2) < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		s_Volume[threadIdx.z][threadIdx.y + 2][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y + 2,z - 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		s_Volume[threadIdx.z][threadIdx.y + 4][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y + 4,z - 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 6) < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		s_Volume[threadIdx.z][threadIdx.y + 6][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y + 6,z - 4,DATA_W, DATA_H)];
	}

	// Second half main data + below apron

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 4) < DATA_D) )
	{
		s_Volume[threadIdx.z + 8][threadIdx.y][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 2) < DATA_H) && ((z + 4) < DATA_D) )
	{
		s_Volume[threadIdx.z + 8][threadIdx.y + 2][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y + 2,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 4) < DATA_D) )
	{
		s_Volume[threadIdx.z + 8][threadIdx.y + 4][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y + 4,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 6) < DATA_H) && ((z + 4) < DATA_D) )
	{
		s_Volume[threadIdx.z + 8][threadIdx.y + 6][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y + 6,z + 4,DATA_W, DATA_H)];
	}

	__syncthreads();

	// Only threads within the volume do the convolution
	if ( (x < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z + 0][threadIdx.y][threadIdx.x] * c_Smoothing_Filter_Z[8];
		sum += s_Volume[threadIdx.z + 1][threadIdx.y][threadIdx.x] * c_Smoothing_Filter_Z[7];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x] * c_Smoothing_Filter_Z[6];
		sum += s_Volume[threadIdx.z + 3][threadIdx.y][threadIdx.x] * c_Smoothing_Filter_Z[5];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x] * c_Smoothing_Filter_Z[4];
		sum += s_Volume[threadIdx.z + 5][threadIdx.y][threadIdx.x] * c_Smoothing_Filter_Z[3];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x] * c_Smoothing_Filter_Z[2];
		sum += s_Volume[threadIdx.z + 7][threadIdx.y][threadIdx.x] * c_Smoothing_Filter_Z[1];
		sum += s_Volume[threadIdx.z + 8][threadIdx.y][threadIdx.x] * c_Smoothing_Filter_Z[0];

		Filter_Responses[Calculate_4D_Index(x,y,z,t,DATA_W,DATA_H,DATA_D)] = sum;
		//Filter_Responses[Calculate_4D_Index(x,y,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 2) < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z + 0][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Filter_Z[8];
		sum += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Filter_Z[7];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Filter_Z[6];
		sum += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Filter_Z[5];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Filter_Z[4];
		sum += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Filter_Z[3];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Filter_Z[2];
		sum += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Filter_Z[1];
		sum += s_Volume[threadIdx.z + 8][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Filter_Z[0];


		Filter_Responses[Calculate_4D_Index(x,y + 2,z,t,DATA_W,DATA_H,DATA_D)] = sum;
		//Filter_Responses[Calculate_4D_Index(x,y + 2,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate_3D_Index(x,y + 2,z,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z + 0][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Filter_Z[8];
		sum += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Filter_Z[7];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Filter_Z[6];
		sum += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Filter_Z[5];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Filter_Z[4];
		sum += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Filter_Z[3];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Filter_Z[2];
		sum += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Filter_Z[1];
		sum += s_Volume[threadIdx.z + 8][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Filter_Z[0];

		Filter_Responses[Calculate_4D_Index(x,y + 4,z,t,DATA_W,DATA_H,DATA_D)] = sum;
		//Filter_Responses[Calculate_4D_Index(x,y + 4,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate_3D_Index(x,y + 4,z,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 6) < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z + 0][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Filter_Z[8];
		sum += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Filter_Z[7];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Filter_Z[6];
		sum += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Filter_Z[5];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Filter_Z[4];
		sum += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Filter_Z[3];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Filter_Z[2];
		sum += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Filter_Z[1];
		sum += s_Volume[threadIdx.z + 8][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Filter_Z[0];

		Filter_Responses[Calculate_4D_Index(x,y + 6,z,t,DATA_W,DATA_H,DATA_D)] = sum;
		//Filter_Responses[Calculate_4D_Index(x,y + 6,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate_3D_Index(x,y + 6,z,DATA_W, DATA_H)];
	}
}


extern "C" __global__ void convolutionRodsNew(float* __restrict__ Filter_Responses, float* __restrict__ fMRI_Volume, float* __restrict__ Brain_Voxels, int t, int DATA_W, int DATA_H, int DATA_D, int xBlockDifference, int yBlockDifference, int zBlockDifference)
{
	volatile int x = blockIdx.x * VALID_FILTER_RESPONSES_X_CONVOLUTION_RODS + threadIdx.x;
	volatile int y = blockIdx.y * blockDim.y + threadIdx.y;
	volatile int z = blockIdx.z * VALID_FILTER_RESPONSES_Z_CONVOLUTION_RODS + threadIdx.z;

    if(x >= (DATA_W + xBlockDifference) || y >= (DATA_H + yBlockDifference) || z >= (DATA_D + zBlockDifference))
        return;



	__shared__ float s_Volume[64][96];

	// Reset shared memory
	s_Volume[threadIdx.z][threadIdx.x] 				= 0.0f;
	s_Volume[threadIdx.z + 32][threadIdx.x] 		= 0.0f;
	s_Volume[threadIdx.z][threadIdx.x + 32] 		= 0.0f;
	s_Volume[threadIdx.z + 32][threadIdx.x + 32] 	= 0.0f;
	s_Volume[threadIdx.z][threadIdx.x + 64] 		= 0.0f;
	s_Volume[threadIdx.z + 32][threadIdx.x + 64] 	= 0.0f;

    //if (Brain_Voxels[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)] == 0.0f)
    //   	return;

	// Read data into shared memory


	if ( (x < DATA_W) && (y < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		s_Volume[threadIdx.z][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y,z - 4,DATA_W, DATA_H)];
	}

	if ( ((x + 32) < DATA_W) && (y < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		s_Volume[threadIdx.z][threadIdx.x + 32] = fMRI_Volume[Calculate_3D_Index(x + 32,y,z - 4,DATA_W, DATA_H)];
	}

	if ( ((x + 64) < DATA_W) && (y < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		s_Volume[threadIdx.z][threadIdx.x + 64] = fMRI_Volume[Calculate_3D_Index(x + 64,y,z - 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 28) < DATA_D) )
	{
		s_Volume[threadIdx.z + 32][threadIdx.x] = fMRI_Volume[Calculate_3D_Index(x,y,z + 28,DATA_W, DATA_H)];
	}

	if ( ((x + 32) < DATA_W) && (y < DATA_H) && ((z + 28) < DATA_D) )
	{
		s_Volume[threadIdx.z + 32][threadIdx.x + 32] = fMRI_Volume[Calculate_3D_Index(x + 32,y,z + 28,DATA_W, DATA_H)];
	}

	if ( ((x + 64) < DATA_W) && (y < DATA_H) && ((z + 28) < DATA_D) )
	{
		s_Volume[threadIdx.z + 32][threadIdx.x + 64] = fMRI_Volume[Calculate_3D_Index(x + 64,y,z + 28,DATA_W, DATA_H)];
	}

	__syncthreads();

	// Only threads within the volume do the convolution
	if ( (x < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z + 0][threadIdx.x] * c_Smoothing_Filter_Z[8];
		sum += s_Volume[threadIdx.z + 1][threadIdx.x] * c_Smoothing_Filter_Z[7];
		sum += s_Volume[threadIdx.z + 2][threadIdx.x] * c_Smoothing_Filter_Z[6];
		sum += s_Volume[threadIdx.z + 3][threadIdx.x] * c_Smoothing_Filter_Z[5];
		sum += s_Volume[threadIdx.z + 4][threadIdx.x] * c_Smoothing_Filter_Z[4];
		sum += s_Volume[threadIdx.z + 5][threadIdx.x] * c_Smoothing_Filter_Z[3];
		sum += s_Volume[threadIdx.z + 6][threadIdx.x] * c_Smoothing_Filter_Z[2];
		sum += s_Volume[threadIdx.z + 7][threadIdx.x] * c_Smoothing_Filter_Z[1];
		sum += s_Volume[threadIdx.z + 8][threadIdx.x] * c_Smoothing_Filter_Z[0];

		Filter_Responses[Calculate_4D_Index(x,y,z,t,DATA_W,DATA_H,DATA_D)] = sum;
	}

	if ( ((x + 32) < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z + 0][threadIdx.x + 32] * c_Smoothing_Filter_Z[8];
		sum += s_Volume[threadIdx.z + 1][threadIdx.x + 32] * c_Smoothing_Filter_Z[7];
		sum += s_Volume[threadIdx.z + 2][threadIdx.x + 32] * c_Smoothing_Filter_Z[6];
		sum += s_Volume[threadIdx.z + 3][threadIdx.x + 32] * c_Smoothing_Filter_Z[5];
		sum += s_Volume[threadIdx.z + 4][threadIdx.x + 32] * c_Smoothing_Filter_Z[4];
		sum += s_Volume[threadIdx.z + 5][threadIdx.x + 32] * c_Smoothing_Filter_Z[3];
		sum += s_Volume[threadIdx.z + 6][threadIdx.x + 32] * c_Smoothing_Filter_Z[2];
		sum += s_Volume[threadIdx.z + 7][threadIdx.x + 32] * c_Smoothing_Filter_Z[1];
		sum += s_Volume[threadIdx.z + 8][threadIdx.x + 32] * c_Smoothing_Filter_Z[0];

		Filter_Responses[Calculate_4D_Index(x + 32,y,z,t,DATA_W,DATA_H,DATA_D)] = sum;
	}

	if ( ((x + 64) < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z + 0][threadIdx.x + 64] * c_Smoothing_Filter_Z[8];
		sum += s_Volume[threadIdx.z + 1][threadIdx.x + 64] * c_Smoothing_Filter_Z[7];
		sum += s_Volume[threadIdx.z + 2][threadIdx.x + 64] * c_Smoothing_Filter_Z[6];
		sum += s_Volume[threadIdx.z + 3][threadIdx.x + 64] * c_Smoothing_Filter_Z[5];
		sum += s_Volume[threadIdx.z + 4][threadIdx.x + 64] * c_Smoothing_Filter_Z[4];
		sum += s_Volume[threadIdx.z + 5][threadIdx.x + 64] * c_Smoothing_Filter_Z[3];
		sum += s_Volume[threadIdx.z + 6][threadIdx.x + 64] * c_Smoothing_Filter_Z[2];
		sum += s_Volume[threadIdx.z + 7][threadIdx.x + 64] * c_Smoothing_Filter_Z[1];
		sum += s_Volume[threadIdx.z + 8][threadIdx.x + 64] * c_Smoothing_Filter_Z[0];

		Filter_Responses[Calculate_4D_Index(x + 64,y,z,t,DATA_W,DATA_H,DATA_D)] = sum;
	}

	if (threadIdx.z < 24)
	{
		if ( (x < DATA_W) && (y < DATA_H) && ((z + 32) < DATA_D) )
		{
			float sum = 0.0f;

			sum += s_Volume[threadIdx.z + 32][threadIdx.x] * c_Smoothing_Filter_Z[8];
			sum += s_Volume[threadIdx.z + 33][threadIdx.x] * c_Smoothing_Filter_Z[7];
			sum += s_Volume[threadIdx.z + 34][threadIdx.x] * c_Smoothing_Filter_Z[6];
			sum += s_Volume[threadIdx.z + 35][threadIdx.x] * c_Smoothing_Filter_Z[5];
			sum += s_Volume[threadIdx.z + 36][threadIdx.x] * c_Smoothing_Filter_Z[4];
			sum += s_Volume[threadIdx.z + 37][threadIdx.x] * c_Smoothing_Filter_Z[3];
			sum += s_Volume[threadIdx.z + 38][threadIdx.x] * c_Smoothing_Filter_Z[2];
			sum += s_Volume[threadIdx.z + 39][threadIdx.x] * c_Smoothing_Filter_Z[1];
			sum += s_Volume[threadIdx.z + 40][threadIdx.x] * c_Smoothing_Filter_Z[0];

			Filter_Responses[Calculate_4D_Index(x,y,z + 32,t,DATA_W,DATA_H,DATA_D)] = sum;
		}

		if ( ((x + 32) < DATA_W) && (y < DATA_H) && ((z + 32) < DATA_D) )
		{
			float sum = 0.0f;

			sum += s_Volume[threadIdx.z + 32][threadIdx.x + 32] * c_Smoothing_Filter_Z[8];
			sum += s_Volume[threadIdx.z + 33][threadIdx.x + 32] * c_Smoothing_Filter_Z[7];
			sum += s_Volume[threadIdx.z + 34][threadIdx.x + 32] * c_Smoothing_Filter_Z[6];
			sum += s_Volume[threadIdx.z + 35][threadIdx.x + 32] * c_Smoothing_Filter_Z[5];
			sum += s_Volume[threadIdx.z + 36][threadIdx.x + 32] * c_Smoothing_Filter_Z[4];
			sum += s_Volume[threadIdx.z + 37][threadIdx.x + 32] * c_Smoothing_Filter_Z[3];
			sum += s_Volume[threadIdx.z + 38][threadIdx.x + 32] * c_Smoothing_Filter_Z[2];
			sum += s_Volume[threadIdx.z + 39][threadIdx.x + 32] * c_Smoothing_Filter_Z[1];
			sum += s_Volume[threadIdx.z + 40][threadIdx.x + 32] * c_Smoothing_Filter_Z[0];

			Filter_Responses[Calculate_4D_Index(x + 32,y,z + 32,t,DATA_W,DATA_H,DATA_D)] = sum;
		}

		if ( ((x + 64) < DATA_W) && (y < DATA_H) && ((z + 32) < DATA_D) )
		{
			float sum = 0.0f;

			sum += s_Volume[threadIdx.z + 32][threadIdx.x + 64] * c_Smoothing_Filter_Z[8];
			sum += s_Volume[threadIdx.z + 33][threadIdx.x + 64] * c_Smoothing_Filter_Z[7];
			sum += s_Volume[threadIdx.z + 34][threadIdx.x + 64] * c_Smoothing_Filter_Z[6];
			sum += s_Volume[threadIdx.z + 35][threadIdx.x + 64] * c_Smoothing_Filter_Z[5];
			sum += s_Volume[threadIdx.z + 36][threadIdx.x + 64] * c_Smoothing_Filter_Z[4];
			sum += s_Volume[threadIdx.z + 37][threadIdx.x + 64] * c_Smoothing_Filter_Z[3];
			sum += s_Volume[threadIdx.z + 38][threadIdx.x + 64] * c_Smoothing_Filter_Z[2];
			sum += s_Volume[threadIdx.z + 39][threadIdx.x + 64] * c_Smoothing_Filter_Z[1];
			sum += s_Volume[threadIdx.z + 40][threadIdx.x + 64] * c_Smoothing_Filter_Z[0];

			Filter_Responses[Calculate_4D_Index(x + 64,y,z + 32,t,DATA_W,DATA_H,DATA_D)] = sum;
		}
	}
}
*/



