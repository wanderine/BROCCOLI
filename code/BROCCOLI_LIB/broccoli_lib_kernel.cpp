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

__global__ void ExtractRealNonPaddedVolumes(float* d_Real_Volumes, Complex* d_Complex_Volumes, int z, int DATA_W, int DATA_H, int DATA_D, int DATA_T, int DATA_T_PADDED, int blocksInY, float invBlocksInY)
{	
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
	unsigned int t = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	unsigned int x = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
	unsigned int y = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || t >= DATA_T)
		return;    
	
	int idx, idx_padded;

	idx = t + x * DATA_T + y * DATA_W * DATA_T + z * DATA_W * DATA_H * DATA_T;
	idx_padded = t + x * DATA_T_PADDED + y * DATA_W * DATA_T_PADDED + z * DATA_W * DATA_H * DATA_T_PADDED;
	d_Real_Volumes[idx] = d_Complex_Volumes[idx_padded].x;
}

__device__ void Get_Parameter_Indices_Kernel(int& i, int& j, int parameter)
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


inline __device__ int Calculate_3D_Index(int a, int b, int c, int DATA_A, int DATA_B)
{
	return a + b * DATA_A + c * DATA_A * DATA_B;
}

inline __device__ int Calculate_4D_Index(int a, int b, int c, int d, int DATA_A, int DATA_B, int DATA_C)
{
	return a + b * DATA_A + c * DATA_A * DATA_B + d * DATA_A * DATA_B * DATA_C;
}

// Slice timing correction

__global__ void SliceTimingCorrection(Complex* Slice_Timing_Corrected_Volumes, Complex* fMRI_Volumes, Complex* Shifters, int z, int DATA_W, int DATA_H, int DATA_D, int DATA_T, int blocksInY, float invBlocksInY, float normalizationConstant)
{   
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
	unsigned int t = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	unsigned int x = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
	unsigned int y = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;	

	if (x >= DATA_W || y >= DATA_H || t >= DATA_T)
		return;    
	
	Complex fMRI_Value, Shifter_Value; int idx;

	idx = t + x * DATA_T + y * DATA_W * DATA_T + z * DATA_W * DATA_H * DATA_T;
	fMRI_Value = fMRI_Volumes[idx];
	Shifter_Value = Shifters[t + z * DATA_T];
	Slice_Timing_Corrected_Volumes[idx].x = (fMRI_Value.x * Shifter_Value.x - fMRI_Value.y * Shifter_Value.y) * normalizationConstant;
	Slice_Timing_Corrected_Volumes[idx].y = (fMRI_Value.y * Shifter_Value.x + fMRI_Value.x * Shifter_Value.y) * normalizationConstant;
}

// Motion compensation

__device__ __constant__ Complex c_Quadrature_Filter_1[7][7][7];
__device__ __constant__ Complex c_Quadrature_Filter_2[7][7][7];
__device__ __constant__ Complex c_Quadrature_Filter_3[7][7][7];

#define NUMBER_OF_PARAMETERS 12

texture<float, 3, cudaReadModeElementType> tex_Modified_Volume;

__global__ void Calculate_Phase_Differences_And_Certainties(float* Phase_Differences, float* Certainties, Complex* q11, Complex* q21, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE, int blocksInY, float invBlocksInY)
{
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - blockIdxz * blocksInY;
	volatile unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int y = blockIdxy * blockDim.y + threadIdx.y;
	volatile unsigned int z = blockIdxz * blockDim.z + threadIdx.z;

	if (((x >= (FILTER_SIZE - 1)/2) && (x < DATA_W - (FILTER_SIZE - 1)/2)) && ((y >= (FILTER_SIZE - 1)/2) && (y < DATA_H - (FILTER_SIZE - 1)/2)) && ((z >= (FILTER_SIZE - 1)/2) && (z < DATA_D - (FILTER_SIZE - 1)/2)))
	{
		int idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

		Complex complex_product; 
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
}

__global__ void Calculate_Phase_Gradients_X(float* Phase_Gradients, Complex* q11, Complex* q21, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE, int blocksInY, float invBlocksInY)
{	
	volatile unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	volatile unsigned int blockIdxy = blockIdx.y - blockIdxz * blocksInY;
	volatile unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int y = blockIdxy * blockDim.y + threadIdx.y;
	volatile unsigned int z = blockIdxz * blockDim.z + threadIdx.z;

	if (((x >= (FILTER_SIZE - 1)/2) && (x < DATA_W - (FILTER_SIZE - 1)/2)) && ((y >= (FILTER_SIZE - 1)/2) && (y < DATA_H - (FILTER_SIZE - 1)/2)) && ((z >= (FILTER_SIZE - 1)/2) && (z < DATA_D - (FILTER_SIZE - 1)/2)))
	{
		Complex total_complex_product; 
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

__global__ void Calculate_Phase_Gradients_Y(float* Phase_Gradients, Complex* q12, Complex* q22, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE, int blocksInY, float invBlocksInY)
{	
	volatile unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	volatile unsigned int blockIdxy = blockIdx.y - blockIdxz * blocksInY;
	volatile unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int y = blockIdxy * blockDim.y + threadIdx.y;
	volatile unsigned int z = blockIdxz * blockDim.z + threadIdx.z;

	if (((x >= (FILTER_SIZE - 1)/2) && (x < DATA_W - (FILTER_SIZE - 1)/2)) && ((y >= (FILTER_SIZE - 1)/2) && (y < DATA_H - (FILTER_SIZE - 1)/2)) && ((z >= (FILTER_SIZE - 1)/2) && (z < DATA_D - (FILTER_SIZE - 1)/2)))
	{
		Complex total_complex_product; 
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

__global__ void Calculate_Phase_Gradients_Z(float* Phase_Gradients, Complex* q13, Complex* q23, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE, int blocksInY, float invBlocksInY)
{	
	volatile unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	volatile unsigned int blockIdxy = blockIdx.y - blockIdxz * blocksInY;
	volatile unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int y = blockIdxy * blockDim.y + threadIdx.y;
	volatile unsigned int z = blockIdxz * blockDim.z + threadIdx.z;

	if (((x >= (FILTER_SIZE - 1)/2) && (x < DATA_W - (FILTER_SIZE - 1)/2)) && ((y >= (FILTER_SIZE - 1)/2) && (y < DATA_H - (FILTER_SIZE - 1)/2)) && ((z >= (FILTER_SIZE - 1)/2) && (z < DATA_D - (FILTER_SIZE - 1)/2)))
	{
		Complex total_complex_product; 
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

__global__ void	Calculate_A_matrix_and_h_vector_3D_values_X(float30* A_matrix_3D_values, float12* h_vector_3D_values, float* Phase_Differences, float* Phase_Gradients, float* Certainties, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE, int blocksInY, float invBlocksInY)
{
	volatile unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	volatile unsigned int blockIdxy = blockIdx.y - blockIdxz * blocksInY;
	volatile unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int y = blockIdxy * blockDim.y + threadIdx.y;
	volatile unsigned int z = blockIdxz * blockDim.z + threadIdx.z;

	int idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

	if (((y >= (FILTER_SIZE - 1)/2) && (y < DATA_H - (FILTER_SIZE - 1)/2)) && ((z >= (FILTER_SIZE - 1)/2) && (z < DATA_D - (FILTER_SIZE - 1)/2)))
	{		
		float xf = (float)x - ((float)DATA_W - 1.0f) * 0.5f;
		float yf = (float)y - ((float)DATA_H - 1.0f) * 0.5f;
		float zf = (float)z - ((float)DATA_D - 1.0f) * 0.5f;

		float phase_difference = Phase_Differences[idx];
		float phase_gradient = Phase_Gradients[idx];
		float certainty = Certainties[idx];
		float c_pg_pg = certainty * phase_gradient * phase_gradient;
		float c_pg_pd = certainty * phase_gradient * phase_difference;
			
		A_matrix_3D_values[idx].a = c_pg_pg;
		A_matrix_3D_values[idx].b = xf * c_pg_pg;
		A_matrix_3D_values[idx].c = yf * c_pg_pg;
		A_matrix_3D_values[idx].d = zf * c_pg_pg;
		A_matrix_3D_values[idx].e = xf * xf * c_pg_pg;
		A_matrix_3D_values[idx].f = xf * yf * c_pg_pg;
		A_matrix_3D_values[idx].g = xf * zf * c_pg_pg;
		A_matrix_3D_values[idx].h = yf * yf * c_pg_pg;
		A_matrix_3D_values[idx].i = yf * zf * c_pg_pg;
		A_matrix_3D_values[idx].j = zf * zf * c_pg_pg;

		h_vector_3D_values[idx].a = c_pg_pd;
		h_vector_3D_values[idx].b = xf * c_pg_pd;
		h_vector_3D_values[idx].c = yf * c_pg_pd;
		h_vector_3D_values[idx].d = zf * c_pg_pd;
	}
	else
	{
		A_matrix_3D_values[idx].a = 0.0f;
		A_matrix_3D_values[idx].b = 0.0f;
		A_matrix_3D_values[idx].c = 0.0f;
		A_matrix_3D_values[idx].d = 0.0f;
		A_matrix_3D_values[idx].e = 0.0f;
		A_matrix_3D_values[idx].f = 0.0f;
		A_matrix_3D_values[idx].g = 0.0f;
		A_matrix_3D_values[idx].h = 0.0f;
		A_matrix_3D_values[idx].i = 0.0f;
		A_matrix_3D_values[idx].j = 0.0f;

		h_vector_3D_values[idx].a = 0.0f;
		h_vector_3D_values[idx].b = 0.0f;
		h_vector_3D_values[idx].c = 0.0f;
		h_vector_3D_values[idx].d = 0.0f;
	}
}

__global__ void	Calculate_A_matrix_and_h_vector_3D_values_Y(float30* A_matrix_3D_values, float12* h_vector_3D_values, float* Phase_Differences, float* Phase_Gradients, float* Certainties, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE, int blocksInY, float invBlocksInY)

{
	volatile unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	volatile unsigned int blockIdxy = blockIdx.y - blockIdxz * blocksInY;
	volatile unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int y = blockIdxy * blockDim.y + threadIdx.y;
	volatile unsigned int z = blockIdxz * blockDim.z + threadIdx.z;

	int idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

	if (((y >= (FILTER_SIZE - 1)/2) && (y < DATA_H - (FILTER_SIZE - 1)/2)) && ((z >= (FILTER_SIZE - 1)/2) && (z < DATA_D - (FILTER_SIZE - 1)/2)))
	{
		float xf = (float)x - ((float)DATA_W - 1.0f) * 0.5f;
		float yf = (float)y - ((float)DATA_H - 1.0f) * 0.5f;
		float zf = (float)z - ((float)DATA_D - 1.0f) * 0.5f;

		float phase_difference = Phase_Differences[idx];
		float phase_gradient = Phase_Gradients[idx];
		float certainty = Certainties[idx];
		float c_pg_pg = certainty * phase_gradient * phase_gradient;
		float c_pg_pd = certainty * phase_gradient * phase_difference;
			
		A_matrix_3D_values[idx].k = c_pg_pg;
		A_matrix_3D_values[idx].l = xf * c_pg_pg;
		A_matrix_3D_values[idx].m = yf * c_pg_pg;
		A_matrix_3D_values[idx].n = zf * c_pg_pg;
		A_matrix_3D_values[idx].o = xf * xf * c_pg_pg;
		A_matrix_3D_values[idx].p = xf * yf * c_pg_pg;
		A_matrix_3D_values[idx].q = xf * zf * c_pg_pg;
		A_matrix_3D_values[idx].r = yf * yf * c_pg_pg;
		A_matrix_3D_values[idx].s = yf * zf * c_pg_pg;
		A_matrix_3D_values[idx].t = zf * zf * c_pg_pg;

		h_vector_3D_values[idx].e = c_pg_pd;
		h_vector_3D_values[idx].f = xf * c_pg_pd;
		h_vector_3D_values[idx].g = yf * c_pg_pd;
		h_vector_3D_values[idx].h = zf * c_pg_pd;
	}
	else
	{
		A_matrix_3D_values[idx].k = 0.0f;
		A_matrix_3D_values[idx].l = 0.0f;
		A_matrix_3D_values[idx].m = 0.0f;
		A_matrix_3D_values[idx].n = 0.0f;
		A_matrix_3D_values[idx].o = 0.0f;
		A_matrix_3D_values[idx].p = 0.0f;
		A_matrix_3D_values[idx].q = 0.0f;
		A_matrix_3D_values[idx].r = 0.0f;
		A_matrix_3D_values[idx].s = 0.0f;
		A_matrix_3D_values[idx].t = 0.0f;

		h_vector_3D_values[idx].e = 0.0f;
		h_vector_3D_values[idx].f = 0.0f;
		h_vector_3D_values[idx].g = 0.0f;
		h_vector_3D_values[idx].h = 0.0f;
	}
}

__global__ void	Calculate_A_matrix_and_h_vector_3D_values_Z(float30* A_matrix_3D_values, float12* h_vector_3D_values, float* Phase_Differences, float* Phase_Gradients, float* Certainties, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE, int blocksInY, float invBlocksInY)

{
	volatile unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	volatile unsigned int blockIdxy = blockIdx.y - blockIdxz * blocksInY;
	volatile unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int y = blockIdxy * blockDim.y + threadIdx.y;
	volatile unsigned int z = blockIdxz * blockDim.z + threadIdx.z;

	int idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

	if (((y >= (FILTER_SIZE - 1)/2) && (y < DATA_H - (FILTER_SIZE - 1)/2)) && ((z >= (FILTER_SIZE - 1)/2) && (z < DATA_D - (FILTER_SIZE - 1)/2)))
	{
		float xf = (float)x - ((float)DATA_W - 1.0f) * 0.5f;
		float yf = (float)y - ((float)DATA_H - 1.0f) * 0.5f;
		float zf = (float)z - ((float)DATA_D - 1.0f) * 0.5f;

		float phase_difference = Phase_Differences[idx];
		float phase_gradient = Phase_Gradients[idx];
		float certainty = Certainties[idx];
		float c_pg_pg = certainty * phase_gradient * phase_gradient;
		float c_pg_pd = certainty * phase_gradient * phase_difference;
			
		A_matrix_3D_values[idx].u = c_pg_pg;
		A_matrix_3D_values[idx].v = xf * c_pg_pg;
		A_matrix_3D_values[idx].w = yf * c_pg_pg;
		A_matrix_3D_values[idx].x = zf * c_pg_pg;
		A_matrix_3D_values[idx].y = xf * xf * c_pg_pg;
		A_matrix_3D_values[idx].z = xf * yf * c_pg_pg;
		A_matrix_3D_values[idx].aa = xf * zf * c_pg_pg;
		A_matrix_3D_values[idx].bb = yf * yf * c_pg_pg;
		A_matrix_3D_values[idx].cc = yf * zf * c_pg_pg;
		A_matrix_3D_values[idx].dd = zf * zf * c_pg_pg;

		h_vector_3D_values[idx].i = c_pg_pd;
		h_vector_3D_values[idx].j = xf * c_pg_pd;
		h_vector_3D_values[idx].k = yf * c_pg_pd;
		h_vector_3D_values[idx].l = zf * c_pg_pd;
	}
	else
	{
		A_matrix_3D_values[idx].u = 0.0f;
		A_matrix_3D_values[idx].v = 0.0f;
		A_matrix_3D_values[idx].w = 0.0f;
		A_matrix_3D_values[idx].x = 0.0f;
		A_matrix_3D_values[idx].y = 0.0f;
		A_matrix_3D_values[idx].z = 0.0f;
		A_matrix_3D_values[idx].aa = 0.0f;
		A_matrix_3D_values[idx].bb = 0.0f;
		A_matrix_3D_values[idx].cc = 0.0f;
		A_matrix_3D_values[idx].dd = 0.0f;

		h_vector_3D_values[idx].i = 0.0f;
		h_vector_3D_values[idx].j = 0.0f;
		h_vector_3D_values[idx].k = 0.0f;
		h_vector_3D_values[idx].l = 0.0f;
	}
}

// dimBlock.x = DATA_H; dimBlock.y = 1; dimBlock.z = 1;
// dimGrid.x = DATA_D; dimGrid.y = 1;

__global__ void	Calculate_A_matrix_and_h_vector_2D_values_X(float* A_matrix_2D_values, float* h_vector_2D_values, float* Phase_Differences, float* Phase_Gradients, float* Certainties, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)

{
	volatile int y = threadIdx.x;
	volatile int z = blockIdx.x;
	
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

__global__ void	Calculate_A_matrix_and_h_vector_2D_values_Y(float* A_matrix_2D_values, float* h_vector_2D_values, float* Phase_Differences, float* Phase_Gradients, float* Certainties, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)

{
	volatile int y = threadIdx.x;
	volatile int z = blockIdx.x;
	
	//volatile int y = blockIdx.x * blockDim.x + threadIdx.x;
	//volatile int z = blockIdx.y * blockDim.y + threadIdx.y;

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

__global__ void	Calculate_A_matrix_and_h_vector_2D_values_Z(float* A_matrix_2D_values, float* h_vector_2D_values, float* Phase_Differences, float* Phase_Gradients, float* Certainties, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)

{
	volatile int y = threadIdx.x;
	volatile int z = blockIdx.x;
	
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


// dimBlock.x = DATA_D; dimBlock.y = 1; dimBlock.z = 1;
// dimGrid.x = NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS; dimGrid.y = 1;        

__global__ void Calculate_A_matrix_1D_values(float* A_matrix_1D_values, float* A_matrix_2D_values, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)
{
	int z = threadIdx.x;
	
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

// dimBlock.x = NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS; dimBlock.y = 1; dimBlock.z = 1;
// dimGrid.x = 1; dimGrid.y = 1; 

__global__ void Calculate_A_matrix(float* A_matrix, float* A_matrix_1D_values, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)
{
	int A_matrix_element = threadIdx.x;
	int idx, i, j;
	
	float matrix_value = 0.0f;
	
	idx = A_matrix_element * DATA_D;

	// Sum over all z positions
	//#pragma unroll 128
	for (int z = (FILTER_SIZE - 1)/2; z < (DATA_D - (FILTER_SIZE - 1)/2); z++)
	{		
		matrix_value += A_matrix_1D_values[idx + z];
	}

	Get_Parameter_Indices_Kernel(i,j,A_matrix_element);
	A_matrix_element = i + j * NUMBER_OF_PARAMETERS;

	A_matrix[A_matrix_element] = matrix_value;
}

__global__ void Reset_A_matrix(float* A_matrix)
{
	int i = threadIdx.x;

	A_matrix[i] = 0.0f;
}

// dimBlock.x = DATA_D; dimBlock.y = 1; dimBlock.z = 1;
// dimGrid.x = NUMBER_OF_PARAMETERS; dimGrid.y = 1; 

__global__ void Calculate_h_vector_1D_values(float* h_vector_1D_values, float* h_vector_2D_values, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)
{
	int z = threadIdx.x;

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


// dimBlock.x = NUMBER_OF_PARAMETERS; dimBlock.y = 1; dimBlock.z = 1;
// dimGrid.x = 1; dimGrid.y = 1; 

__global__ void Calculate_h_vector(float* h_vector, float* h_vector_1D_values, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)
{
	int h_vector_element = threadIdx.x;
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

__global__ void InterpolateVolumeTriLinear(float* Volume, float Parameter_Vector_0, float Parameter_Vector_1, float Parameter_Vector_2, float Parameter_Vector_3, float Parameter_Vector_4, float Parameter_Vector_5,
										  float Parameter_Vector_6, float Parameter_Vector_7, float Parameter_Vector_8, float Parameter_Vector_9, float Parameter_Vector_10, float Parameter_Vector_11, int DATA_W, int DATA_H, int DATA_D, int blocksInY, float invBlocksInY)

										   
{		
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - blockIdxz * blocksInY;
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdxy * blockDim.y + threadIdx.y;
	unsigned int z = blockIdxz * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;    
	
	int idx = Calculate_3D_Index(x,y,z,DATA_W, DATA_H);
	float3 Motion_Vector;
	float xf, yf, zf;

    // (motion_vector.x)   (p0)   (p3  p4  p5)   (x)
	// (motion_vector.y) = (p1) + (p6  p7  p8) * (y)
 	// (motion_vector.z)   (p2)   (p9 p10 p11)   (z)
				
	// Change to coordinate system with origo in (sx - 1)/2 (sy - 1)/2 (sz - 1)/2
	xf = (float)x - ((float)DATA_W - 1.0f)*0.5f;
	yf = (float)y - ((float)DATA_H - 1.0f)*0.5f;
	zf = (float)z - ((float)DATA_D - 1.0f)*0.5f;
				
	Motion_Vector.x = x + Parameter_Vector_0 + Parameter_Vector_3 * xf + Parameter_Vector_4   * yf + Parameter_Vector_5  * zf + 0.5f;
	Motion_Vector.y = y + Parameter_Vector_1 + Parameter_Vector_6 * xf + Parameter_Vector_7   * yf + Parameter_Vector_8  * zf + 0.5f;
	Motion_Vector.z = z + Parameter_Vector_2 + Parameter_Vector_9 * xf + Parameter_Vector_10  * yf + Parameter_Vector_11 * zf + 0.5f;

	Volume[idx] = tex3D(tex_Modified_Volume, Motion_Vector.x, Motion_Vector.y, Motion_Vector.z);
}

__global__ void Convolve_3D_Complex_7x7x7(float* Volume, Complex *Filter_Response_1, Complex *Filter_Response_2, Complex *Filter_Response_3, int DATA_W, int DATA_H, int DATA_D, int blocksInY, float invBlocksInY, int xBlockDifference, int yBlockDifference, int zBlockDifference)
{   
	volatile int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	volatile int blockIdxy = blockIdx.y - blockIdxz * blocksInY;
	volatile int x = blockIdx.x * blockDim.x + threadIdx.x;
	volatile int y = blockIdxy * blockDim.y + threadIdx.y;
	volatile int z = blockIdxz * blockDim.z + threadIdx.z;	

    if (x >= (DATA_W + xBlockDifference) || y >= (DATA_H + yBlockDifference) || z >= (DATA_D + zBlockDifference))
        return;
	
	__shared__ float s_Volume[16][16][16];    // z, y, x

	/* 27 blocks in shared memory, filter response is calculated for block 14, 8 x 8 x 8 threads per block
	
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

	// Read data into shared memory
	
	// Top layer

	// Block 1 (4 x 4 x 4)
	if ((threadIdx.x < 4) && (threadIdx.y < 4) && (threadIdx.z < 4))
	{
		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			s_Volume[threadIdx.z][threadIdx.y][threadIdx.x] = Volume[Calculate_3D_Index(x - 4,y - 4,z - 4,DATA_W, DATA_H)];
		}
		else
		{
			s_Volume[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
		}
	}

	__syncthreads();
	
	// Block 2 (8 x 4 x 4)
	if ((threadIdx.x < 8) && (threadIdx.y < 4) && (threadIdx.z < 4))
	{		
		if ( (x < DATA_W) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 4] = Volume[Calculate_3D_Index(x,y - 4,z - 4,DATA_W, DATA_H)];
		}
		else
		{
			s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 4] = 0.0f;
		}
	}

	__syncthreads();
	
	// Block 3 (4 x 4 x 4)
	if ((threadIdx.x < 4) && (threadIdx.y < 4) && (threadIdx.z < 4))
	{		
		if ( ((x + 8) < DATA_W) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y - 4,z - 4,DATA_W, DATA_H)];
		}
		else
		{
			s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 12] = 0.0f;
		}
	}

	__syncthreads();

	// Block 4 (4 x 8 x 4)
	if ((threadIdx.x < 4) && (threadIdx.y < 8) && (threadIdx.z < 4))
	{		
		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && (y < DATA_H) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			s_Volume[threadIdx.z][threadIdx.y + 4][threadIdx.x] = Volume[Calculate_3D_Index(x - 4,y,z - 4,DATA_W, DATA_H)];
		}
		else
		{
			s_Volume[threadIdx.z][threadIdx.y + 4][threadIdx.x] = 0.0f;
		}
	}

	__syncthreads();

	// Block 5 (8 x 8 x 4)
	if ((threadIdx.x < 8) && (threadIdx.y < 8) && (threadIdx.z < 4))
	{		
		if ( (x < DATA_W) && (y < DATA_H) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			s_Volume[threadIdx.z][threadIdx.y + 4][threadIdx.x + 4] = Volume[Calculate_3D_Index(x,y,z - 4,DATA_W, DATA_H)];
		}
		else
		{
			s_Volume[threadIdx.z][threadIdx.y + 4][threadIdx.x + 4] = 0.0f;
		}
	}

	__syncthreads();

	// Block 6 (4 x 8 x 4)
	if ((threadIdx.x < 4) && (threadIdx.y < 8) && (threadIdx.z < 4))
	{		
		if ( ((x + 8) < DATA_W) && (y < DATA_H) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			s_Volume[threadIdx.z][threadIdx.y + 4][threadIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y,z - 4,DATA_W, DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z][threadIdx.y + 4][threadIdx.x + 12] = 0.0f;
		}
	}

	__syncthreads();

	// Block 7 (4 x 4 x 4)
	if ((threadIdx.x < 4) && (threadIdx.y < 4) && (threadIdx.z < 4))
	{		

		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y + 8) < DATA_H) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			s_Volume[threadIdx.z][threadIdx.y + 12][threadIdx.x] = Volume[Calculate_3D_Index(x - 4,y + 8,z - 4,DATA_W, DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z][threadIdx.y + 12][threadIdx.x] = 0.0f;
		}
	}

	__syncthreads();

	// Block 8 (8 x 4 x 4)
	if ((threadIdx.x < 8) && (threadIdx.y < 4) && (threadIdx.z < 4))
	{		
		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			s_Volume[threadIdx.z][threadIdx.y + 12][threadIdx.x + 4] = Volume[Calculate_3D_Index(x,y + 8,z - 4,DATA_W, DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z][threadIdx.y + 12][threadIdx.x + 4] = 0.0f;
		}
	}

	__syncthreads();

	
	// Block 9 (4 x 4 x 4)
	if ((threadIdx.x < 4) && (threadIdx.y < 4) && (threadIdx.z < 4))
	{		
		if ( ((x + 8) < DATA_W) && ((y + 8) < DATA_H) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
		{
			s_Volume[threadIdx.z][threadIdx.y + 12][threadIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y + 8,z - 4,DATA_W, DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z][threadIdx.y + 12][threadIdx.x + 12] = 0.0f;
		}
	}
	

	// Middle layer

	__syncthreads();

	// Block 10 (4 x 4 x 8)
	if ((threadIdx.x < 4) && (threadIdx.y < 4) && (threadIdx.z < 8))
	{		
		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && (z < DATA_D) )
		{
			s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x] = Volume[Calculate_3D_Index(x - 4,y - 4,z,DATA_W, DATA_H)];
		}
		else
		{
			s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x] = 0.0f;
		}
	}

	__syncthreads();

	// Block 11 (8 x 4 x 8)
	if ((threadIdx.x < 8) && (threadIdx.y < 4) && (threadIdx.z < 8))
	{		

		if ( (x < DATA_W) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && (z < DATA_D) )
		{
			s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 4] = Volume[Calculate_3D_Index(x,y - 4,z,DATA_W, DATA_H)];
		}
		else
		{
			s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 4] = 0.0f;
		}
	}

	__syncthreads();

	
	// Block 12 (4 x 4 x 8)
	if ((threadIdx.x < 4) && (threadIdx.y < 4) && (threadIdx.z < 8))
	{		
		if ( ((x  + 8) < DATA_W) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && (z < DATA_D) )
		{
			s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y - 4,z,DATA_W, DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 12] = 0.0f;
		}


	}
	
	__syncthreads();

	// Block 13 (4 x 8 x 8)
	if ((threadIdx.x < 4) && (threadIdx.y < 8) && (threadIdx.z < 8))
	{		
		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && (y < DATA_H) && (z < DATA_D) )
		{
			s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x] = Volume[Calculate_3D_Index(x - 4,y,z,DATA_W, DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x] = 0.0f;
		}
	
	}

	__syncthreads();

	// Block 14, main data (8 x 8 x 8)	
	if ( (x < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 4] = Volume[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)];			
	}
	else
	{
		s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 4] = 0.0f;			
	}

	__syncthreads();

	// Block 15 (4 x 8 x 8)
	if ((threadIdx.x < 4) && (threadIdx.y < 8) && (threadIdx.z < 8))
	{		
		if ( ((x + 8) < DATA_W) && (y < DATA_H) && (z < DATA_D) )
		{
			s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y,z,DATA_W, DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 12] = 0.0f;
		}		
	}

	__syncthreads();

	// Block 16 (4 x 4 x 8)
	if ((threadIdx.x < 4) && (threadIdx.y < 4) && (threadIdx.z < 8))
	{		
		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y + 8) < DATA_H) && (z < DATA_D) )
		{
			s_Volume[threadIdx.z + 4][threadIdx.y + 12][threadIdx.x] = Volume[Calculate_3D_Index(x - 4,y + 8,z,DATA_W, DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z + 4][threadIdx.y + 12][threadIdx.x] = 0.0f;
		}
	}

	__syncthreads();

	// Block 17 (8 x 4 x 8)
	if ((threadIdx.x < 8) && (threadIdx.y < 4) && (threadIdx.z < 8))
	{		
		if ( (x < DATA_W) && ((y + 8) < DATA_H) && (z < DATA_D) )
		{
			s_Volume[threadIdx.z + 4][threadIdx.y + 12][threadIdx.x + 4] = Volume[Calculate_3D_Index(x,y + 8,z,DATA_W, DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z + 4][threadIdx.y + 12][threadIdx.x + 4] = 0.0f;
		}
	}

	__syncthreads();

	// Block 18 (4 x 4 x 8)
	if ((threadIdx.x < 4) && (threadIdx.y < 4) && (threadIdx.z < 8))
	{		
		if ( ((x + 8) < DATA_W) && ((y + 8) < DATA_H) && (z < DATA_D) )
		{
			s_Volume[threadIdx.z + 4][threadIdx.y + 12][threadIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y + 8,z,DATA_W, DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z + 4][threadIdx.y + 12][threadIdx.x + 12] = 0.0f;
		}
	}


	// Bottom layer

	__syncthreads();

	// Block 19 (4 x 4 x 4)
	if ((threadIdx.x < 4) && (threadIdx.y < 4) && (threadIdx.z < 4))
	{		
		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z + 8) < DATA_D) )
		{
			s_Volume[threadIdx.z + 12][threadIdx.y][threadIdx.x] = Volume[Calculate_3D_Index(x - 4,y - 4,z + 8,DATA_W, DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z + 12][threadIdx.y][threadIdx.x] = 0.0f;
		}
	}

	__syncthreads();

	// Block 20 (8 x 4 x 4)
	if ((threadIdx.x < 8) && (threadIdx.y < 4) && (threadIdx.z < 4))
	{	
		if ( (x < DATA_W) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z + 8) < DATA_D) )
		{
			s_Volume[threadIdx.z + 12][threadIdx.y][threadIdx.x + 4] = Volume[Calculate_3D_Index(x,y - 4,z + 8,DATA_W, DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z + 12][threadIdx.y][threadIdx.x + 4] = 0.0f;
		}
	}

	__syncthreads();

	// Block 21 (4 x 4 x 4)
	if ((threadIdx.x < 4) && (threadIdx.y < 4) && (threadIdx.z < 4))
	{		
		if ( ((x + 8) < DATA_W) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z + 8) < DATA_D) )
		{
			s_Volume[threadIdx.z + 12][threadIdx.y][threadIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y - 4,z + 8,DATA_W, DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z + 12][threadIdx.y][threadIdx.x + 12] = 0.0f;
		}
	}

	__syncthreads();

	// Block 22 (4 x 8 x 4)
	if ((threadIdx.x < 4) && (threadIdx.y < 8) && (threadIdx.z < 4))
	{		
		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && (y < DATA_H) && ((z + 8) < DATA_D) )
		{
			s_Volume[threadIdx.z + 12][threadIdx.y + 4][threadIdx.x] = Volume[Calculate_3D_Index(x - 4,y,z + 8,DATA_W,DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z + 12][threadIdx.y + 4][threadIdx.x] = 0.0f;
		}
	}

	__syncthreads();

	// Block 23 (8 x 8 x 4)
	if ((threadIdx.x < 8) && (threadIdx.y < 8) && (threadIdx.z < 4))
	{		
		if ( (x < DATA_W) && (y < DATA_H) && ((z + 8) < DATA_D) )
		{
			s_Volume[threadIdx.z + 12][threadIdx.y + 4][threadIdx.x + 4] = Volume[Calculate_3D_Index(x,y,z + 8,DATA_W,DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z + 12][threadIdx.y + 4][threadIdx.x + 4] = 0.0f;
		}
	}

	__syncthreads();

	// Block 24 (4 x 8 x 4)
	if ((threadIdx.x < 4) && (threadIdx.y < 8) && (threadIdx.z < 4))
	{		
		if ( ((x + 8) < DATA_W) && (y < DATA_H) && ((z + 8) < DATA_D) )
		{
			s_Volume[threadIdx.z + 12][threadIdx.y + 4][threadIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y,z + 8,DATA_W,DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z + 12][threadIdx.y + 4][threadIdx.x + 12] = 0.0f;
		}
	}

	__syncthreads();

	// Block 25 (4 x 4 x 4)
	if ((threadIdx.x < 4) && (threadIdx.y < 4) && (threadIdx.z < 4))
	{		
		if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y + 8) < DATA_H) && ((z + 8) < DATA_D) )
		{
			s_Volume[threadIdx.z + 12][threadIdx.y + 12][threadIdx.x] = Volume[Calculate_3D_Index(x - 4,y + 8,z + 8,DATA_W,DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z + 12][threadIdx.y + 12][threadIdx.x] = 0.0f;
		}
	}

	__syncthreads();

	// Block 26 (8 x 4 x 4)
	if ((threadIdx.x < 8) && (threadIdx.y < 4) && (threadIdx.z < 4))
	{		
		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 8) < DATA_D) )
		{
			s_Volume[threadIdx.z + 12][threadIdx.y + 12][threadIdx.x + 4] = Volume[Calculate_3D_Index(x,y + 8,z + 8,DATA_W,DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z + 12][threadIdx.y + 12][threadIdx.x + 4] = 0.0f;
		}
	}

	__syncthreads();

	// Block 27 (4 x 4 x 4)
	if ((threadIdx.x < 4) && (threadIdx.y < 4) && (threadIdx.z < 4))
	{		
		if ( ((x + 8) < DATA_W) && ((y + 8) < DATA_H) && ((z + 8) < DATA_D) )
		{
			s_Volume[threadIdx.z + 12][threadIdx.y + 12][threadIdx.x + 12] = Volume[Calculate_3D_Index(x + 8,y + 8,z + 8,DATA_W,DATA_H)];
		}	
		else
		{
			s_Volume[threadIdx.z + 12][threadIdx.y + 12][threadIdx.x + 12] = 0.0f;
		}
	}

	// Calculate filter responses for block 14
	__syncthreads();
		
	// Only threads within the volume do the convolution
	if ( (x < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
			Complex sum1, sum2, sum3;
			sum1.x = 0.0f; sum1.y = 0.0f; sum2.x = 0.0f; sum2.y = 0.0f; sum3.x = 0.0f; sum3.y = 0.0f;

            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_1[6][6][6].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_1[6][6][6].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_2[6][6][6].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_2[6][6][6].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_3[6][6][6].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_3[6][6][6].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_1[5][6][6].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_1[5][6][6].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_2[5][6][6].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_2[5][6][6].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_3[5][6][6].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_3[5][6][6].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_1[4][6][6].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_1[4][6][6].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_2[4][6][6].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_2[4][6][6].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_3[4][6][6].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_3[4][6][6].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_1[3][6][6].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_1[3][6][6].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_2[3][6][6].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_2[3][6][6].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_3[3][6][6].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_3[3][6][6].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_1[2][6][6].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_1[2][6][6].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_2[2][6][6].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_2[2][6][6].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_3[2][6][6].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_3[2][6][6].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_1[1][6][6].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_1[1][6][6].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_2[1][6][6].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_2[1][6][6].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_3[1][6][6].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_3[1][6][6].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_1[0][6][6].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_1[0][6][6].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_2[0][6][6].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_2[0][6][6].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_3[0][6][6].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 1] * c_Quadrature_Filter_3[0][6][6].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_1[6][5][6].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_1[6][5][6].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_2[6][5][6].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_2[6][5][6].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_3[6][5][6].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_3[6][5][6].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_1[5][5][6].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_1[5][5][6].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_2[5][5][6].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_2[5][5][6].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_3[5][5][6].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_3[5][5][6].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_1[4][5][6].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_1[4][5][6].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_2[4][5][6].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_2[4][5][6].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_3[4][5][6].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_3[4][5][6].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_1[3][5][6].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_1[3][5][6].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_2[3][5][6].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_2[3][5][6].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_3[3][5][6].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_3[3][5][6].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_1[2][5][6].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_1[2][5][6].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_2[2][5][6].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_2[2][5][6].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_3[2][5][6].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_3[2][5][6].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_1[1][5][6].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_1[1][5][6].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_2[1][5][6].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_2[1][5][6].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_3[1][5][6].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_3[1][5][6].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_1[0][5][6].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_1[0][5][6].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_2[0][5][6].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_2[0][5][6].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_3[0][5][6].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 1] * c_Quadrature_Filter_3[0][5][6].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_1[6][4][6].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_1[6][4][6].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_2[6][4][6].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_2[6][4][6].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_3[6][4][6].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_3[6][4][6].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_1[5][4][6].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_1[5][4][6].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_2[5][4][6].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_2[5][4][6].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_3[5][4][6].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_3[5][4][6].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_1[4][4][6].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_1[4][4][6].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_2[4][4][6].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_2[4][4][6].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_3[4][4][6].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_3[4][4][6].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_1[3][4][6].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_1[3][4][6].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_2[3][4][6].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_2[3][4][6].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_3[3][4][6].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_3[3][4][6].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_1[2][4][6].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_1[2][4][6].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_2[2][4][6].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_2[2][4][6].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_3[2][4][6].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_3[2][4][6].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_1[1][4][6].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_1[1][4][6].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_2[1][4][6].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_2[1][4][6].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_3[1][4][6].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_3[1][4][6].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_1[0][4][6].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_1[0][4][6].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_2[0][4][6].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_2[0][4][6].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_3[0][4][6].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 1] * c_Quadrature_Filter_3[0][4][6].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_1[6][3][6].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_1[6][3][6].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_2[6][3][6].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_2[6][3][6].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_3[6][3][6].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_3[6][3][6].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_1[5][3][6].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_1[5][3][6].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_2[5][3][6].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_2[5][3][6].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_3[5][3][6].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_3[5][3][6].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_1[4][3][6].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_1[4][3][6].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_2[4][3][6].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_2[4][3][6].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_3[4][3][6].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_3[4][3][6].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_1[3][3][6].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_1[3][3][6].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_2[3][3][6].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_2[3][3][6].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_3[3][3][6].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_3[3][3][6].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_1[2][3][6].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_1[2][3][6].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_2[2][3][6].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_2[2][3][6].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_3[2][3][6].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_3[2][3][6].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_1[1][3][6].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_1[1][3][6].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_2[1][3][6].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_2[1][3][6].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_3[1][3][6].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_3[1][3][6].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_1[0][3][6].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_1[0][3][6].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_2[0][3][6].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_2[0][3][6].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_3[0][3][6].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 1] * c_Quadrature_Filter_3[0][3][6].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_1[6][2][6].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_1[6][2][6].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_2[6][2][6].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_2[6][2][6].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_3[6][2][6].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_3[6][2][6].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_1[5][2][6].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_1[5][2][6].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_2[5][2][6].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_2[5][2][6].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_3[5][2][6].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_3[5][2][6].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_1[4][2][6].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_1[4][2][6].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_2[4][2][6].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_2[4][2][6].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_3[4][2][6].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_3[4][2][6].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_1[3][2][6].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_1[3][2][6].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_2[3][2][6].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_2[3][2][6].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_3[3][2][6].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_3[3][2][6].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_1[2][2][6].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_1[2][2][6].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_2[2][2][6].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_2[2][2][6].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_3[2][2][6].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_3[2][2][6].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_1[1][2][6].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_1[1][2][6].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_2[1][2][6].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_2[1][2][6].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_3[1][2][6].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_3[1][2][6].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_1[0][2][6].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_1[0][2][6].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_2[0][2][6].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_2[0][2][6].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_3[0][2][6].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 1] * c_Quadrature_Filter_3[0][2][6].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_1[6][1][6].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_1[6][1][6].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_2[6][1][6].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_2[6][1][6].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_3[6][1][6].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_3[6][1][6].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_1[5][1][6].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_1[5][1][6].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_2[5][1][6].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_2[5][1][6].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_3[5][1][6].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_3[5][1][6].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_1[4][1][6].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_1[4][1][6].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_2[4][1][6].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_2[4][1][6].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_3[4][1][6].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_3[4][1][6].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_1[3][1][6].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_1[3][1][6].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_2[3][1][6].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_2[3][1][6].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_3[3][1][6].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_3[3][1][6].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_1[2][1][6].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_1[2][1][6].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_2[2][1][6].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_2[2][1][6].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_3[2][1][6].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_3[2][1][6].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_1[1][1][6].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_1[1][1][6].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_2[1][1][6].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_2[1][1][6].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_3[1][1][6].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_3[1][1][6].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_1[0][1][6].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_1[0][1][6].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_2[0][1][6].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_2[0][1][6].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_3[0][1][6].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 1] * c_Quadrature_Filter_3[0][1][6].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_1[6][0][6].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_1[6][0][6].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_2[6][0][6].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_2[6][0][6].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_3[6][0][6].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_3[6][0][6].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_1[5][0][6].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_1[5][0][6].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_2[5][0][6].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_2[5][0][6].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_3[5][0][6].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_3[5][0][6].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_1[4][0][6].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_1[4][0][6].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_2[4][0][6].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_2[4][0][6].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_3[4][0][6].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_3[4][0][6].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_1[3][0][6].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_1[3][0][6].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_2[3][0][6].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_2[3][0][6].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_3[3][0][6].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_3[3][0][6].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_1[2][0][6].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_1[2][0][6].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_2[2][0][6].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_2[2][0][6].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_3[2][0][6].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_3[2][0][6].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_1[1][0][6].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_1[1][0][6].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_2[1][0][6].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_2[1][0][6].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_3[1][0][6].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_3[1][0][6].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_1[0][0][6].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_1[0][0][6].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_2[0][0][6].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_2[0][0][6].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_3[0][0][6].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 1] * c_Quadrature_Filter_3[0][0][6].y;

            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_1[6][6][5].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_1[6][6][5].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_2[6][6][5].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_2[6][6][5].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_3[6][6][5].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_3[6][6][5].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_1[5][6][5].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_1[5][6][5].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_2[5][6][5].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_2[5][6][5].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_3[5][6][5].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_3[5][6][5].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_1[4][6][5].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_1[4][6][5].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_2[4][6][5].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_2[4][6][5].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_3[4][6][5].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_3[4][6][5].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_1[3][6][5].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_1[3][6][5].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_2[3][6][5].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_2[3][6][5].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_3[3][6][5].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_3[3][6][5].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_1[2][6][5].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_1[2][6][5].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_2[2][6][5].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_2[2][6][5].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_3[2][6][5].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_3[2][6][5].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_1[1][6][5].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_1[1][6][5].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_2[1][6][5].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_2[1][6][5].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_3[1][6][5].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_3[1][6][5].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_1[0][6][5].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_1[0][6][5].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_2[0][6][5].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_2[0][6][5].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_3[0][6][5].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 2] * c_Quadrature_Filter_3[0][6][5].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_1[6][5][5].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_1[6][5][5].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_2[6][5][5].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_2[6][5][5].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_3[6][5][5].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_3[6][5][5].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_1[5][5][5].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_1[5][5][5].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_2[5][5][5].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_2[5][5][5].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_3[5][5][5].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_3[5][5][5].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_1[4][5][5].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_1[4][5][5].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_2[4][5][5].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_2[4][5][5].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_3[4][5][5].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_3[4][5][5].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_1[3][5][5].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_1[3][5][5].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_2[3][5][5].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_2[3][5][5].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_3[3][5][5].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_3[3][5][5].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_1[2][5][5].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_1[2][5][5].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_2[2][5][5].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_2[2][5][5].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_3[2][5][5].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_3[2][5][5].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_1[1][5][5].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_1[1][5][5].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_2[1][5][5].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_2[1][5][5].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_3[1][5][5].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_3[1][5][5].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_1[0][5][5].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_1[0][5][5].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_2[0][5][5].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_2[0][5][5].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_3[0][5][5].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 2] * c_Quadrature_Filter_3[0][5][5].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_1[6][4][5].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_1[6][4][5].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_2[6][4][5].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_2[6][4][5].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_3[6][4][5].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_3[6][4][5].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_1[5][4][5].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_1[5][4][5].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_2[5][4][5].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_2[5][4][5].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_3[5][4][5].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_3[5][4][5].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_1[4][4][5].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_1[4][4][5].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_2[4][4][5].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_2[4][4][5].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_3[4][4][5].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_3[4][4][5].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_1[3][4][5].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_1[3][4][5].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_2[3][4][5].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_2[3][4][5].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_3[3][4][5].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_3[3][4][5].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_1[2][4][5].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_1[2][4][5].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_2[2][4][5].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_2[2][4][5].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_3[2][4][5].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_3[2][4][5].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_1[1][4][5].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_1[1][4][5].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_2[1][4][5].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_2[1][4][5].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_3[1][4][5].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_3[1][4][5].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_1[0][4][5].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_1[0][4][5].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_2[0][4][5].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_2[0][4][5].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_3[0][4][5].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 2] * c_Quadrature_Filter_3[0][4][5].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_1[6][3][5].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_1[6][3][5].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_2[6][3][5].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_2[6][3][5].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_3[6][3][5].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_3[6][3][5].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_1[5][3][5].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_1[5][3][5].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_2[5][3][5].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_2[5][3][5].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_3[5][3][5].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_3[5][3][5].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_1[4][3][5].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_1[4][3][5].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_2[4][3][5].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_2[4][3][5].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_3[4][3][5].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_3[4][3][5].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_1[3][3][5].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_1[3][3][5].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_2[3][3][5].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_2[3][3][5].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_3[3][3][5].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_3[3][3][5].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_1[2][3][5].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_1[2][3][5].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_2[2][3][5].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_2[2][3][5].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_3[2][3][5].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_3[2][3][5].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_1[1][3][5].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_1[1][3][5].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_2[1][3][5].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_2[1][3][5].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_3[1][3][5].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_3[1][3][5].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_1[0][3][5].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_1[0][3][5].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_2[0][3][5].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_2[0][3][5].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_3[0][3][5].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 2] * c_Quadrature_Filter_3[0][3][5].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_1[6][2][5].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_1[6][2][5].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_2[6][2][5].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_2[6][2][5].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_3[6][2][5].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_3[6][2][5].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_1[5][2][5].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_1[5][2][5].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_2[5][2][5].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_2[5][2][5].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_3[5][2][5].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_3[5][2][5].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_1[4][2][5].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_1[4][2][5].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_2[4][2][5].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_2[4][2][5].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_3[4][2][5].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_3[4][2][5].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_1[3][2][5].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_1[3][2][5].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_2[3][2][5].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_2[3][2][5].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_3[3][2][5].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_3[3][2][5].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_1[2][2][5].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_1[2][2][5].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_2[2][2][5].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_2[2][2][5].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_3[2][2][5].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_3[2][2][5].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_1[1][2][5].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_1[1][2][5].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_2[1][2][5].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_2[1][2][5].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_3[1][2][5].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_3[1][2][5].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_1[0][2][5].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_1[0][2][5].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_2[0][2][5].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_2[0][2][5].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_3[0][2][5].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 2] * c_Quadrature_Filter_3[0][2][5].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_1[6][1][5].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_1[6][1][5].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_2[6][1][5].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_2[6][1][5].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_3[6][1][5].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_3[6][1][5].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_1[5][1][5].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_1[5][1][5].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_2[5][1][5].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_2[5][1][5].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_3[5][1][5].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_3[5][1][5].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_1[4][1][5].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_1[4][1][5].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_2[4][1][5].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_2[4][1][5].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_3[4][1][5].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_3[4][1][5].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_1[3][1][5].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_1[3][1][5].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_2[3][1][5].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_2[3][1][5].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_3[3][1][5].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_3[3][1][5].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_1[2][1][5].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_1[2][1][5].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_2[2][1][5].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_2[2][1][5].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_3[2][1][5].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_3[2][1][5].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_1[1][1][5].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_1[1][1][5].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_2[1][1][5].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_2[1][1][5].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_3[1][1][5].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_3[1][1][5].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_1[0][1][5].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_1[0][1][5].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_2[0][1][5].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_2[0][1][5].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_3[0][1][5].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 2] * c_Quadrature_Filter_3[0][1][5].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_1[6][0][5].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_1[6][0][5].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_2[6][0][5].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_2[6][0][5].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_3[6][0][5].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_3[6][0][5].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_1[5][0][5].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_1[5][0][5].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_2[5][0][5].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_2[5][0][5].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_3[5][0][5].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_3[5][0][5].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_1[4][0][5].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_1[4][0][5].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_2[4][0][5].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_2[4][0][5].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_3[4][0][5].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_3[4][0][5].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_1[3][0][5].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_1[3][0][5].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_2[3][0][5].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_2[3][0][5].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_3[3][0][5].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_3[3][0][5].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_1[2][0][5].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_1[2][0][5].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_2[2][0][5].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_2[2][0][5].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_3[2][0][5].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_3[2][0][5].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_1[1][0][5].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_1[1][0][5].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_2[1][0][5].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_2[1][0][5].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_3[1][0][5].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_3[1][0][5].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_1[0][0][5].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_1[0][0][5].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_2[0][0][5].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_2[0][0][5].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_3[0][0][5].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 2] * c_Quadrature_Filter_3[0][0][5].y;


            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_1[6][6][4].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_1[6][6][4].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_2[6][6][4].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_2[6][6][4].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_3[6][6][4].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_3[6][6][4].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_1[5][6][4].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_1[5][6][4].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_2[5][6][4].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_2[5][6][4].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_3[5][6][4].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_3[5][6][4].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_1[4][6][4].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_1[4][6][4].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_2[4][6][4].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_2[4][6][4].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_3[4][6][4].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_3[4][6][4].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_1[3][6][4].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_1[3][6][4].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_2[3][6][4].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_2[3][6][4].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_3[3][6][4].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_3[3][6][4].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_1[2][6][4].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_1[2][6][4].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_2[2][6][4].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_2[2][6][4].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_3[2][6][4].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_3[2][6][4].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_1[1][6][4].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_1[1][6][4].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_2[1][6][4].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_2[1][6][4].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_3[1][6][4].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_3[1][6][4].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_1[0][6][4].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_1[0][6][4].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_2[0][6][4].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_2[0][6][4].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_3[0][6][4].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 3] * c_Quadrature_Filter_3[0][6][4].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_1[6][5][4].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_1[6][5][4].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_2[6][5][4].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_2[6][5][4].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_3[6][5][4].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_3[6][5][4].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_1[5][5][4].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_1[5][5][4].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_2[5][5][4].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_2[5][5][4].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_3[5][5][4].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_3[5][5][4].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_1[4][5][4].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_1[4][5][4].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_2[4][5][4].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_2[4][5][4].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_3[4][5][4].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_3[4][5][4].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_1[3][5][4].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_1[3][5][4].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_2[3][5][4].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_2[3][5][4].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_3[3][5][4].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_3[3][5][4].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_1[2][5][4].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_1[2][5][4].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_2[2][5][4].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_2[2][5][4].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_3[2][5][4].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_3[2][5][4].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_1[1][5][4].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_1[1][5][4].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_2[1][5][4].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_2[1][5][4].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_3[1][5][4].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_3[1][5][4].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_1[0][5][4].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_1[0][5][4].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_2[0][5][4].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_2[0][5][4].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_3[0][5][4].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 3] * c_Quadrature_Filter_3[0][5][4].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_1[6][4][4].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_1[6][4][4].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_2[6][4][4].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_2[6][4][4].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_3[6][4][4].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_3[6][4][4].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_1[5][4][4].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_1[5][4][4].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_2[5][4][4].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_2[5][4][4].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_3[5][4][4].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_3[5][4][4].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_1[4][4][4].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_1[4][4][4].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_2[4][4][4].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_2[4][4][4].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_3[4][4][4].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_3[4][4][4].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_1[3][4][4].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_1[3][4][4].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_2[3][4][4].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_2[3][4][4].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_3[3][4][4].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_3[3][4][4].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_1[2][4][4].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_1[2][4][4].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_2[2][4][4].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_2[2][4][4].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_3[2][4][4].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_3[2][4][4].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_1[1][4][4].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_1[1][4][4].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_2[1][4][4].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_2[1][4][4].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_3[1][4][4].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_3[1][4][4].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_1[0][4][4].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_1[0][4][4].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_2[0][4][4].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_2[0][4][4].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_3[0][4][4].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 3] * c_Quadrature_Filter_3[0][4][4].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_1[6][3][4].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_1[6][3][4].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_2[6][3][4].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_2[6][3][4].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_3[6][3][4].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_3[6][3][4].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_1[5][3][4].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_1[5][3][4].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_2[5][3][4].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_2[5][3][4].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_3[5][3][4].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_3[5][3][4].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_1[4][3][4].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_1[4][3][4].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_2[4][3][4].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_2[4][3][4].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_3[4][3][4].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_3[4][3][4].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_1[3][3][4].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_1[3][3][4].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_2[3][3][4].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_2[3][3][4].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_3[3][3][4].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_3[3][3][4].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_1[2][3][4].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_1[2][3][4].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_2[2][3][4].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_2[2][3][4].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_3[2][3][4].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_3[2][3][4].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_1[1][3][4].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_1[1][3][4].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_2[1][3][4].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_2[1][3][4].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_3[1][3][4].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_3[1][3][4].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_1[0][3][4].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_1[0][3][4].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_2[0][3][4].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_2[0][3][4].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_3[0][3][4].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 3] * c_Quadrature_Filter_3[0][3][4].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_1[6][2][4].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_1[6][2][4].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_2[6][2][4].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_2[6][2][4].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_3[6][2][4].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_3[6][2][4].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_1[5][2][4].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_1[5][2][4].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_2[5][2][4].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_2[5][2][4].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_3[5][2][4].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_3[5][2][4].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_1[4][2][4].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_1[4][2][4].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_2[4][2][4].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_2[4][2][4].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_3[4][2][4].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_3[4][2][4].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_1[3][2][4].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_1[3][2][4].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_2[3][2][4].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_2[3][2][4].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_3[3][2][4].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_3[3][2][4].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_1[2][2][4].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_1[2][2][4].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_2[2][2][4].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_2[2][2][4].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_3[2][2][4].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_3[2][2][4].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_1[1][2][4].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_1[1][2][4].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_2[1][2][4].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_2[1][2][4].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_3[1][2][4].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_3[1][2][4].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_1[0][2][4].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_1[0][2][4].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_2[0][2][4].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_2[0][2][4].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_3[0][2][4].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 3] * c_Quadrature_Filter_3[0][2][4].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_1[6][1][4].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_1[6][1][4].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_2[6][1][4].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_2[6][1][4].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_3[6][1][4].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_3[6][1][4].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_1[5][1][4].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_1[5][1][4].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_2[5][1][4].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_2[5][1][4].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_3[5][1][4].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_3[5][1][4].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_1[4][1][4].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_1[4][1][4].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_2[4][1][4].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_2[4][1][4].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_3[4][1][4].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_3[4][1][4].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_1[3][1][4].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_1[3][1][4].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_2[3][1][4].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_2[3][1][4].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_3[3][1][4].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_3[3][1][4].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_1[2][1][4].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_1[2][1][4].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_2[2][1][4].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_2[2][1][4].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_3[2][1][4].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_3[2][1][4].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_1[1][1][4].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_1[1][1][4].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_2[1][1][4].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_2[1][1][4].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_3[1][1][4].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_3[1][1][4].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_1[0][1][4].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_1[0][1][4].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_2[0][1][4].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_2[0][1][4].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_3[0][1][4].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 3] * c_Quadrature_Filter_3[0][1][4].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_1[6][0][4].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_1[6][0][4].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_2[6][0][4].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_2[6][0][4].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_3[6][0][4].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_3[6][0][4].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_1[5][0][4].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_1[5][0][4].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_2[5][0][4].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_2[5][0][4].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_3[5][0][4].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_3[5][0][4].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_1[4][0][4].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_1[4][0][4].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_2[4][0][4].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_2[4][0][4].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_3[4][0][4].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_3[4][0][4].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_1[3][0][4].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_1[3][0][4].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_2[3][0][4].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_2[3][0][4].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_3[3][0][4].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_3[3][0][4].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_1[2][0][4].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_1[2][0][4].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_2[2][0][4].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_2[2][0][4].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_3[2][0][4].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_3[2][0][4].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_1[1][0][4].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_1[1][0][4].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_2[1][0][4].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_2[1][0][4].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_3[1][0][4].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_3[1][0][4].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_1[0][0][4].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_1[0][0][4].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_2[0][0][4].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_2[0][0][4].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_3[0][0][4].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 3] * c_Quadrature_Filter_3[0][0][4].y;

            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_1[6][6][3].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_1[6][6][3].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_2[6][6][3].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_2[6][6][3].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_3[6][6][3].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_3[6][6][3].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_1[5][6][3].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_1[5][6][3].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_2[5][6][3].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_2[5][6][3].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_3[5][6][3].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_3[5][6][3].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_1[4][6][3].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_1[4][6][3].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_2[4][6][3].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_2[4][6][3].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_3[4][6][3].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_3[4][6][3].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_1[3][6][3].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_1[3][6][3].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_2[3][6][3].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_2[3][6][3].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_3[3][6][3].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_3[3][6][3].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_1[2][6][3].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_1[2][6][3].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_2[2][6][3].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_2[2][6][3].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_3[2][6][3].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_3[2][6][3].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_1[1][6][3].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_1[1][6][3].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_2[1][6][3].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_2[1][6][3].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_3[1][6][3].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_3[1][6][3].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_1[0][6][3].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_1[0][6][3].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_2[0][6][3].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_2[0][6][3].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_3[0][6][3].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 4] * c_Quadrature_Filter_3[0][6][3].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_1[6][5][3].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_1[6][5][3].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_2[6][5][3].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_2[6][5][3].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_3[6][5][3].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_3[6][5][3].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_1[5][5][3].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_1[5][5][3].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_2[5][5][3].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_2[5][5][3].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_3[5][5][3].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_3[5][5][3].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_1[4][5][3].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_1[4][5][3].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_2[4][5][3].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_2[4][5][3].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_3[4][5][3].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_3[4][5][3].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_1[3][5][3].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_1[3][5][3].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_2[3][5][3].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_2[3][5][3].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_3[3][5][3].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_3[3][5][3].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_1[2][5][3].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_1[2][5][3].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_2[2][5][3].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_2[2][5][3].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_3[2][5][3].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_3[2][5][3].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_1[1][5][3].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_1[1][5][3].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_2[1][5][3].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_2[1][5][3].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_3[1][5][3].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_3[1][5][3].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_1[0][5][3].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_1[0][5][3].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_2[0][5][3].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_2[0][5][3].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_3[0][5][3].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 4] * c_Quadrature_Filter_3[0][5][3].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_1[6][4][3].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_1[6][4][3].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_2[6][4][3].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_2[6][4][3].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_3[6][4][3].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_3[6][4][3].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_1[5][4][3].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_1[5][4][3].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_2[5][4][3].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_2[5][4][3].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_3[5][4][3].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_3[5][4][3].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_1[4][4][3].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_1[4][4][3].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_2[4][4][3].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_2[4][4][3].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_3[4][4][3].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_3[4][4][3].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_1[3][4][3].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_1[3][4][3].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_2[3][4][3].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_2[3][4][3].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_3[3][4][3].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_3[3][4][3].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_1[2][4][3].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_1[2][4][3].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_2[2][4][3].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_2[2][4][3].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_3[2][4][3].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_3[2][4][3].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_1[1][4][3].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_1[1][4][3].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_2[1][4][3].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_2[1][4][3].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_3[1][4][3].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_3[1][4][3].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_1[0][4][3].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_1[0][4][3].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_2[0][4][3].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_2[0][4][3].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_3[0][4][3].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 4] * c_Quadrature_Filter_3[0][4][3].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_1[6][3][3].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_1[6][3][3].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_2[6][3][3].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_2[6][3][3].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_3[6][3][3].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_3[6][3][3].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_1[5][3][3].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_1[5][3][3].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_2[5][3][3].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_2[5][3][3].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_3[5][3][3].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_3[5][3][3].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_1[4][3][3].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_1[4][3][3].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_2[4][3][3].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_2[4][3][3].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_3[4][3][3].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_3[4][3][3].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_1[3][3][3].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_1[3][3][3].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_2[3][3][3].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_2[3][3][3].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_3[3][3][3].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_3[3][3][3].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_1[2][3][3].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_1[2][3][3].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_2[2][3][3].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_2[2][3][3].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_3[2][3][3].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_3[2][3][3].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_1[1][3][3].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_1[1][3][3].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_2[1][3][3].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_2[1][3][3].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_3[1][3][3].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_3[1][3][3].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_1[0][3][3].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_1[0][3][3].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_2[0][3][3].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_2[0][3][3].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_3[0][3][3].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 4] * c_Quadrature_Filter_3[0][3][3].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_1[6][2][3].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_1[6][2][3].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_2[6][2][3].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_2[6][2][3].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_3[6][2][3].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_3[6][2][3].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_1[5][2][3].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_1[5][2][3].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_2[5][2][3].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_2[5][2][3].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_3[5][2][3].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_3[5][2][3].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_1[4][2][3].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_1[4][2][3].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_2[4][2][3].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_2[4][2][3].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_3[4][2][3].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_3[4][2][3].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_1[3][2][3].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_1[3][2][3].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_2[3][2][3].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_2[3][2][3].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_3[3][2][3].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_3[3][2][3].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_1[2][2][3].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_1[2][2][3].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_2[2][2][3].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_2[2][2][3].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_3[2][2][3].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_3[2][2][3].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_1[1][2][3].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_1[1][2][3].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_2[1][2][3].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_2[1][2][3].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_3[1][2][3].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_3[1][2][3].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_1[0][2][3].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_1[0][2][3].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_2[0][2][3].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_2[0][2][3].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_3[0][2][3].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 4] * c_Quadrature_Filter_3[0][2][3].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_1[6][1][3].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_1[6][1][3].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_2[6][1][3].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_2[6][1][3].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_3[6][1][3].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_3[6][1][3].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_1[5][1][3].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_1[5][1][3].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_2[5][1][3].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_2[5][1][3].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_3[5][1][3].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_3[5][1][3].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_1[4][1][3].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_1[4][1][3].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_2[4][1][3].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_2[4][1][3].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_3[4][1][3].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_3[4][1][3].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_1[3][1][3].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_1[3][1][3].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_2[3][1][3].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_2[3][1][3].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_3[3][1][3].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_3[3][1][3].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_1[2][1][3].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_1[2][1][3].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_2[2][1][3].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_2[2][1][3].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_3[2][1][3].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_3[2][1][3].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_1[1][1][3].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_1[1][1][3].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_2[1][1][3].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_2[1][1][3].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_3[1][1][3].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_3[1][1][3].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_1[0][1][3].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_1[0][1][3].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_2[0][1][3].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_2[0][1][3].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_3[0][1][3].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 4] * c_Quadrature_Filter_3[0][1][3].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_1[6][0][3].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_1[6][0][3].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_2[6][0][3].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_2[6][0][3].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_3[6][0][3].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_3[6][0][3].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_1[5][0][3].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_1[5][0][3].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_2[5][0][3].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_2[5][0][3].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_3[5][0][3].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_3[5][0][3].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_1[4][0][3].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_1[4][0][3].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_2[4][0][3].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_2[4][0][3].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_3[4][0][3].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_3[4][0][3].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_1[3][0][3].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_1[3][0][3].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_2[3][0][3].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_2[3][0][3].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_3[3][0][3].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_3[3][0][3].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_1[2][0][3].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_1[2][0][3].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_2[2][0][3].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_2[2][0][3].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_3[2][0][3].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_3[2][0][3].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_1[1][0][3].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_1[1][0][3].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_2[1][0][3].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_2[1][0][3].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_3[1][0][3].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_3[1][0][3].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_1[0][0][3].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_1[0][0][3].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_2[0][0][3].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_2[0][0][3].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_3[0][0][3].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 4] * c_Quadrature_Filter_3[0][0][3].y;

            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_1[6][6][2].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_1[6][6][2].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_2[6][6][2].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_2[6][6][2].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_3[6][6][2].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_3[6][6][2].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_1[5][6][2].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_1[5][6][2].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_2[5][6][2].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_2[5][6][2].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_3[5][6][2].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_3[5][6][2].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_1[4][6][2].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_1[4][6][2].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_2[4][6][2].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_2[4][6][2].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_3[4][6][2].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_3[4][6][2].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_1[3][6][2].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_1[3][6][2].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_2[3][6][2].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_2[3][6][2].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_3[3][6][2].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_3[3][6][2].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_1[2][6][2].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_1[2][6][2].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_2[2][6][2].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_2[2][6][2].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_3[2][6][2].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_3[2][6][2].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_1[1][6][2].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_1[1][6][2].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_2[1][6][2].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_2[1][6][2].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_3[1][6][2].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_3[1][6][2].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_1[0][6][2].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_1[0][6][2].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_2[0][6][2].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_2[0][6][2].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_3[0][6][2].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 5] * c_Quadrature_Filter_3[0][6][2].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_1[6][5][2].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_1[6][5][2].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_2[6][5][2].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_2[6][5][2].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_3[6][5][2].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_3[6][5][2].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_1[5][5][2].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_1[5][5][2].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_2[5][5][2].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_2[5][5][2].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_3[5][5][2].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_3[5][5][2].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_1[4][5][2].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_1[4][5][2].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_2[4][5][2].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_2[4][5][2].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_3[4][5][2].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_3[4][5][2].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_1[3][5][2].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_1[3][5][2].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_2[3][5][2].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_2[3][5][2].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_3[3][5][2].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_3[3][5][2].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_1[2][5][2].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_1[2][5][2].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_2[2][5][2].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_2[2][5][2].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_3[2][5][2].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_3[2][5][2].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_1[1][5][2].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_1[1][5][2].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_2[1][5][2].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_2[1][5][2].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_3[1][5][2].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_3[1][5][2].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_1[0][5][2].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_1[0][5][2].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_2[0][5][2].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_2[0][5][2].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_3[0][5][2].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 5] * c_Quadrature_Filter_3[0][5][2].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_1[6][4][2].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_1[6][4][2].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_2[6][4][2].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_2[6][4][2].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_3[6][4][2].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_3[6][4][2].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_1[5][4][2].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_1[5][4][2].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_2[5][4][2].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_2[5][4][2].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_3[5][4][2].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_3[5][4][2].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_1[4][4][2].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_1[4][4][2].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_2[4][4][2].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_2[4][4][2].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_3[4][4][2].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_3[4][4][2].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_1[3][4][2].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_1[3][4][2].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_2[3][4][2].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_2[3][4][2].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_3[3][4][2].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_3[3][4][2].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_1[2][4][2].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_1[2][4][2].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_2[2][4][2].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_2[2][4][2].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_3[2][4][2].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_3[2][4][2].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_1[1][4][2].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_1[1][4][2].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_2[1][4][2].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_2[1][4][2].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_3[1][4][2].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_3[1][4][2].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_1[0][4][2].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_1[0][4][2].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_2[0][4][2].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_2[0][4][2].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_3[0][4][2].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 5] * c_Quadrature_Filter_3[0][4][2].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_1[6][3][2].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_1[6][3][2].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_2[6][3][2].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_2[6][3][2].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_3[6][3][2].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_3[6][3][2].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_1[5][3][2].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_1[5][3][2].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_2[5][3][2].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_2[5][3][2].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_3[5][3][2].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_3[5][3][2].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_1[4][3][2].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_1[4][3][2].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_2[4][3][2].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_2[4][3][2].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_3[4][3][2].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_3[4][3][2].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_1[3][3][2].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_1[3][3][2].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_2[3][3][2].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_2[3][3][2].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_3[3][3][2].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_3[3][3][2].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_1[2][3][2].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_1[2][3][2].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_2[2][3][2].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_2[2][3][2].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_3[2][3][2].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_3[2][3][2].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_1[1][3][2].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_1[1][3][2].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_2[1][3][2].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_2[1][3][2].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_3[1][3][2].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_3[1][3][2].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_1[0][3][2].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_1[0][3][2].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_2[0][3][2].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_2[0][3][2].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_3[0][3][2].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 5] * c_Quadrature_Filter_3[0][3][2].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_1[6][2][2].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_1[6][2][2].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_2[6][2][2].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_2[6][2][2].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_3[6][2][2].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_3[6][2][2].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_1[5][2][2].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_1[5][2][2].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_2[5][2][2].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_2[5][2][2].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_3[5][2][2].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_3[5][2][2].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_1[4][2][2].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_1[4][2][2].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_2[4][2][2].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_2[4][2][2].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_3[4][2][2].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_3[4][2][2].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_1[3][2][2].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_1[3][2][2].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_2[3][2][2].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_2[3][2][2].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_3[3][2][2].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_3[3][2][2].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_1[2][2][2].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_1[2][2][2].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_2[2][2][2].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_2[2][2][2].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_3[2][2][2].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_3[2][2][2].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_1[1][2][2].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_1[1][2][2].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_2[1][2][2].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_2[1][2][2].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_3[1][2][2].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_3[1][2][2].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_1[0][2][2].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_1[0][2][2].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_2[0][2][2].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_2[0][2][2].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_3[0][2][2].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 5] * c_Quadrature_Filter_3[0][2][2].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_1[6][1][2].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_1[6][1][2].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_2[6][1][2].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_2[6][1][2].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_3[6][1][2].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_3[6][1][2].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_1[5][1][2].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_1[5][1][2].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_2[5][1][2].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_2[5][1][2].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_3[5][1][2].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_3[5][1][2].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_1[4][1][2].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_1[4][1][2].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_2[4][1][2].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_2[4][1][2].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_3[4][1][2].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_3[4][1][2].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_1[3][1][2].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_1[3][1][2].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_2[3][1][2].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_2[3][1][2].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_3[3][1][2].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_3[3][1][2].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_1[2][1][2].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_1[2][1][2].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_2[2][1][2].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_2[2][1][2].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_3[2][1][2].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_3[2][1][2].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_1[1][1][2].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_1[1][1][2].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_2[1][1][2].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_2[1][1][2].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_3[1][1][2].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_3[1][1][2].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_1[0][1][2].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_1[0][1][2].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_2[0][1][2].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_2[0][1][2].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_3[0][1][2].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 5] * c_Quadrature_Filter_3[0][1][2].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_1[6][0][2].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_1[6][0][2].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_2[6][0][2].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_2[6][0][2].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_3[6][0][2].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_3[6][0][2].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_1[5][0][2].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_1[5][0][2].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_2[5][0][2].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_2[5][0][2].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_3[5][0][2].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_3[5][0][2].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_1[4][0][2].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_1[4][0][2].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_2[4][0][2].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_2[4][0][2].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_3[4][0][2].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_3[4][0][2].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_1[3][0][2].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_1[3][0][2].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_2[3][0][2].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_2[3][0][2].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_3[3][0][2].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_3[3][0][2].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_1[2][0][2].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_1[2][0][2].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_2[2][0][2].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_2[2][0][2].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_3[2][0][2].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_3[2][0][2].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_1[1][0][2].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_1[1][0][2].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_2[1][0][2].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_2[1][0][2].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_3[1][0][2].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_3[1][0][2].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_1[0][0][2].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_1[0][0][2].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_2[0][0][2].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_2[0][0][2].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_3[0][0][2].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 5] * c_Quadrature_Filter_3[0][0][2].y;

            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_1[6][6][1].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_1[6][6][1].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_2[6][6][1].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_2[6][6][1].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_3[6][6][1].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_3[6][6][1].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_1[5][6][1].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_1[5][6][1].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_2[5][6][1].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_2[5][6][1].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_3[5][6][1].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_3[5][6][1].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_1[4][6][1].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_1[4][6][1].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_2[4][6][1].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_2[4][6][1].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_3[4][6][1].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_3[4][6][1].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_1[3][6][1].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_1[3][6][1].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_2[3][6][1].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_2[3][6][1].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_3[3][6][1].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_3[3][6][1].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_1[2][6][1].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_1[2][6][1].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_2[2][6][1].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_2[2][6][1].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_3[2][6][1].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_3[2][6][1].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_1[1][6][1].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_1[1][6][1].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_2[1][6][1].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_2[1][6][1].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_3[1][6][1].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_3[1][6][1].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_1[0][6][1].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_1[0][6][1].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_2[0][6][1].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_2[0][6][1].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_3[0][6][1].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 6] * c_Quadrature_Filter_3[0][6][1].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_1[6][5][1].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_1[6][5][1].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_2[6][5][1].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_2[6][5][1].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_3[6][5][1].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_3[6][5][1].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_1[5][5][1].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_1[5][5][1].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_2[5][5][1].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_2[5][5][1].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_3[5][5][1].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_3[5][5][1].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_1[4][5][1].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_1[4][5][1].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_2[4][5][1].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_2[4][5][1].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_3[4][5][1].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_3[4][5][1].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_1[3][5][1].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_1[3][5][1].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_2[3][5][1].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_2[3][5][1].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_3[3][5][1].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_3[3][5][1].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_1[2][5][1].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_1[2][5][1].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_2[2][5][1].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_2[2][5][1].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_3[2][5][1].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_3[2][5][1].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_1[1][5][1].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_1[1][5][1].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_2[1][5][1].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_2[1][5][1].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_3[1][5][1].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_3[1][5][1].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_1[0][5][1].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_1[0][5][1].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_2[0][5][1].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_2[0][5][1].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_3[0][5][1].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 6] * c_Quadrature_Filter_3[0][5][1].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_1[6][4][1].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_1[6][4][1].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_2[6][4][1].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_2[6][4][1].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_3[6][4][1].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_3[6][4][1].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_1[5][4][1].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_1[5][4][1].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_2[5][4][1].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_2[5][4][1].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_3[5][4][1].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_3[5][4][1].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_1[4][4][1].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_1[4][4][1].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_2[4][4][1].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_2[4][4][1].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_3[4][4][1].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_3[4][4][1].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_1[3][4][1].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_1[3][4][1].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_2[3][4][1].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_2[3][4][1].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_3[3][4][1].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_3[3][4][1].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_1[2][4][1].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_1[2][4][1].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_2[2][4][1].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_2[2][4][1].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_3[2][4][1].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_3[2][4][1].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_1[1][4][1].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_1[1][4][1].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_2[1][4][1].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_2[1][4][1].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_3[1][4][1].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_3[1][4][1].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_1[0][4][1].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_1[0][4][1].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_2[0][4][1].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_2[0][4][1].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_3[0][4][1].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 6] * c_Quadrature_Filter_3[0][4][1].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_1[6][3][1].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_1[6][3][1].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_2[6][3][1].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_2[6][3][1].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_3[6][3][1].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_3[6][3][1].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_1[5][3][1].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_1[5][3][1].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_2[5][3][1].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_2[5][3][1].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_3[5][3][1].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_3[5][3][1].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_1[4][3][1].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_1[4][3][1].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_2[4][3][1].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_2[4][3][1].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_3[4][3][1].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_3[4][3][1].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_1[3][3][1].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_1[3][3][1].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_2[3][3][1].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_2[3][3][1].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_3[3][3][1].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_3[3][3][1].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_1[2][3][1].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_1[2][3][1].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_2[2][3][1].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_2[2][3][1].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_3[2][3][1].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_3[2][3][1].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_1[1][3][1].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_1[1][3][1].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_2[1][3][1].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_2[1][3][1].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_3[1][3][1].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_3[1][3][1].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_1[0][3][1].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_1[0][3][1].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_2[0][3][1].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_2[0][3][1].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_3[0][3][1].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 6] * c_Quadrature_Filter_3[0][3][1].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_1[6][2][1].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_1[6][2][1].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_2[6][2][1].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_2[6][2][1].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_3[6][2][1].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_3[6][2][1].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_1[5][2][1].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_1[5][2][1].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_2[5][2][1].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_2[5][2][1].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_3[5][2][1].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_3[5][2][1].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_1[4][2][1].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_1[4][2][1].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_2[4][2][1].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_2[4][2][1].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_3[4][2][1].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_3[4][2][1].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_1[3][2][1].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_1[3][2][1].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_2[3][2][1].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_2[3][2][1].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_3[3][2][1].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_3[3][2][1].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_1[2][2][1].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_1[2][2][1].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_2[2][2][1].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_2[2][2][1].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_3[2][2][1].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_3[2][2][1].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_1[1][2][1].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_1[1][2][1].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_2[1][2][1].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_2[1][2][1].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_3[1][2][1].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_3[1][2][1].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_1[0][2][1].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_1[0][2][1].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_2[0][2][1].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_2[0][2][1].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_3[0][2][1].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 6] * c_Quadrature_Filter_3[0][2][1].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_1[6][1][1].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_1[6][1][1].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_2[6][1][1].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_2[6][1][1].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_3[6][1][1].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_3[6][1][1].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_1[5][1][1].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_1[5][1][1].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_2[5][1][1].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_2[5][1][1].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_3[5][1][1].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_3[5][1][1].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_1[4][1][1].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_1[4][1][1].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_2[4][1][1].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_2[4][1][1].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_3[4][1][1].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_3[4][1][1].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_1[3][1][1].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_1[3][1][1].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_2[3][1][1].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_2[3][1][1].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_3[3][1][1].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_3[3][1][1].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_1[2][1][1].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_1[2][1][1].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_2[2][1][1].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_2[2][1][1].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_3[2][1][1].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_3[2][1][1].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_1[1][1][1].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_1[1][1][1].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_2[1][1][1].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_2[1][1][1].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_3[1][1][1].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_3[1][1][1].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_1[0][1][1].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_1[0][1][1].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_2[0][1][1].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_2[0][1][1].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_3[0][1][1].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 6] * c_Quadrature_Filter_3[0][1][1].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_1[6][0][1].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_1[6][0][1].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_2[6][0][1].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_2[6][0][1].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_3[6][0][1].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_3[6][0][1].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_1[5][0][1].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_1[5][0][1].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_2[5][0][1].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_2[5][0][1].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_3[5][0][1].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_3[5][0][1].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_1[4][0][1].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_1[4][0][1].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_2[4][0][1].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_2[4][0][1].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_3[4][0][1].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_3[4][0][1].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_1[3][0][1].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_1[3][0][1].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_2[3][0][1].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_2[3][0][1].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_3[3][0][1].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_3[3][0][1].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_1[2][0][1].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_1[2][0][1].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_2[2][0][1].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_2[2][0][1].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_3[2][0][1].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_3[2][0][1].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_1[1][0][1].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_1[1][0][1].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_2[1][0][1].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_2[1][0][1].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_3[1][0][1].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_3[1][0][1].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_1[0][0][1].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_1[0][0][1].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_2[0][0][1].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_2[0][0][1].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_3[0][0][1].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 6] * c_Quadrature_Filter_3[0][0][1].y;

            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_1[6][6][0].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_1[6][6][0].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_2[6][6][0].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_2[6][6][0].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_3[6][6][0].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_3[6][6][0].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_1[5][6][0].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_1[5][6][0].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_2[5][6][0].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_2[5][6][0].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_3[5][6][0].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_3[5][6][0].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_1[4][6][0].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_1[4][6][0].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_2[4][6][0].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_2[4][6][0].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_3[4][6][0].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_3[4][6][0].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_1[3][6][0].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_1[3][6][0].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_2[3][6][0].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_2[3][6][0].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_3[3][6][0].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_3[3][6][0].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_1[2][6][0].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_1[2][6][0].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_2[2][6][0].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_2[2][6][0].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_3[2][6][0].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_3[2][6][0].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_1[1][6][0].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_1[1][6][0].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_2[1][6][0].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_2[1][6][0].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_3[1][6][0].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_3[1][6][0].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_1[0][6][0].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_1[0][6][0].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_2[0][6][0].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_2[0][6][0].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_3[0][6][0].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 1][threadIdx.x + 7] * c_Quadrature_Filter_3[0][6][0].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_1[6][5][0].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_1[6][5][0].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_2[6][5][0].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_2[6][5][0].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_3[6][5][0].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_3[6][5][0].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_1[5][5][0].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_1[5][5][0].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_2[5][5][0].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_2[5][5][0].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_3[5][5][0].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_3[5][5][0].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_1[4][5][0].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_1[4][5][0].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_2[4][5][0].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_2[4][5][0].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_3[4][5][0].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_3[4][5][0].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_1[3][5][0].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_1[3][5][0].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_2[3][5][0].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_2[3][5][0].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_3[3][5][0].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_3[3][5][0].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_1[2][5][0].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_1[2][5][0].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_2[2][5][0].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_2[2][5][0].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_3[2][5][0].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_3[2][5][0].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_1[1][5][0].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_1[1][5][0].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_2[1][5][0].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_2[1][5][0].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_3[1][5][0].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_3[1][5][0].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_1[0][5][0].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_1[0][5][0].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_2[0][5][0].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_2[0][5][0].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_3[0][5][0].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x + 7] * c_Quadrature_Filter_3[0][5][0].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_1[6][4][0].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_1[6][4][0].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_2[6][4][0].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_2[6][4][0].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_3[6][4][0].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_3[6][4][0].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_1[5][4][0].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_1[5][4][0].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_2[5][4][0].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_2[5][4][0].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_3[5][4][0].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_3[5][4][0].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_1[4][4][0].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_1[4][4][0].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_2[4][4][0].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_2[4][4][0].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_3[4][4][0].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_3[4][4][0].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_1[3][4][0].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_1[3][4][0].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_2[3][4][0].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_2[3][4][0].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_3[3][4][0].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_3[3][4][0].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_1[2][4][0].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_1[2][4][0].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_2[2][4][0].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_2[2][4][0].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_3[2][4][0].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_3[2][4][0].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_1[1][4][0].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_1[1][4][0].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_2[1][4][0].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_2[1][4][0].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_3[1][4][0].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_3[1][4][0].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_1[0][4][0].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_1[0][4][0].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_2[0][4][0].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_2[0][4][0].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_3[0][4][0].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 3][threadIdx.x + 7] * c_Quadrature_Filter_3[0][4][0].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_1[6][3][0].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_1[6][3][0].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_2[6][3][0].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_2[6][3][0].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_3[6][3][0].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_3[6][3][0].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_1[5][3][0].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_1[5][3][0].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_2[5][3][0].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_2[5][3][0].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_3[5][3][0].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_3[5][3][0].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_1[4][3][0].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_1[4][3][0].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_2[4][3][0].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_2[4][3][0].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_3[4][3][0].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_3[4][3][0].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_1[3][3][0].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_1[3][3][0].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_2[3][3][0].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_2[3][3][0].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_3[3][3][0].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_3[3][3][0].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_1[2][3][0].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_1[2][3][0].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_2[2][3][0].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_2[2][3][0].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_3[2][3][0].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_3[2][3][0].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_1[1][3][0].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_1[1][3][0].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_2[1][3][0].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_2[1][3][0].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_3[1][3][0].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_3[1][3][0].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_1[0][3][0].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_1[0][3][0].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_2[0][3][0].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_2[0][3][0].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_3[0][3][0].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x + 7] * c_Quadrature_Filter_3[0][3][0].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_1[6][2][0].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_1[6][2][0].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_2[6][2][0].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_2[6][2][0].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_3[6][2][0].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_3[6][2][0].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_1[5][2][0].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_1[5][2][0].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_2[5][2][0].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_2[5][2][0].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_3[5][2][0].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_3[5][2][0].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_1[4][2][0].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_1[4][2][0].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_2[4][2][0].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_2[4][2][0].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_3[4][2][0].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_3[4][2][0].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_1[3][2][0].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_1[3][2][0].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_2[3][2][0].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_2[3][2][0].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_3[3][2][0].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_3[3][2][0].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_1[2][2][0].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_1[2][2][0].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_2[2][2][0].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_2[2][2][0].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_3[2][2][0].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_3[2][2][0].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_1[1][2][0].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_1[1][2][0].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_2[1][2][0].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_2[1][2][0].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_3[1][2][0].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_3[1][2][0].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_1[0][2][0].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_1[0][2][0].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_2[0][2][0].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_2[0][2][0].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_3[0][2][0].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 5][threadIdx.x + 7] * c_Quadrature_Filter_3[0][2][0].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_1[6][1][0].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_1[6][1][0].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_2[6][1][0].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_2[6][1][0].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_3[6][1][0].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_3[6][1][0].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_1[5][1][0].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_1[5][1][0].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_2[5][1][0].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_2[5][1][0].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_3[5][1][0].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_3[5][1][0].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_1[4][1][0].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_1[4][1][0].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_2[4][1][0].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_2[4][1][0].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_3[4][1][0].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_3[4][1][0].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_1[3][1][0].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_1[3][1][0].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_2[3][1][0].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_2[3][1][0].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_3[3][1][0].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_3[3][1][0].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_1[2][1][0].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_1[2][1][0].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_2[2][1][0].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_2[2][1][0].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_3[2][1][0].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_3[2][1][0].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_1[1][1][0].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_1[1][1][0].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_2[1][1][0].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_2[1][1][0].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_3[1][1][0].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_3[1][1][0].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_1[0][1][0].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_1[0][1][0].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_2[0][1][0].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_2[0][1][0].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_3[0][1][0].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x + 7] * c_Quadrature_Filter_3[0][1][0].y;
            sum1.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_1[6][0][0].x;
            sum1.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_1[6][0][0].y;
            sum2.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_2[6][0][0].x;
            sum2.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_2[6][0][0].y;
            sum3.x += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_3[6][0][0].x;
            sum3.y += s_Volume[threadIdx.z + 1][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_3[6][0][0].y;
            sum1.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_1[5][0][0].x;
            sum1.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_1[5][0][0].y;
            sum2.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_2[5][0][0].x;
            sum2.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_2[5][0][0].y;
            sum3.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_3[5][0][0].x;
            sum3.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_3[5][0][0].y;
            sum1.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_1[4][0][0].x;
            sum1.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_1[4][0][0].y;
            sum2.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_2[4][0][0].x;
            sum2.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_2[4][0][0].y;
            sum3.x += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_3[4][0][0].x;
            sum3.y += s_Volume[threadIdx.z + 3][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_3[4][0][0].y;
            sum1.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_1[3][0][0].x;
            sum1.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_1[3][0][0].y;
            sum2.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_2[3][0][0].x;
            sum2.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_2[3][0][0].y;
            sum3.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_3[3][0][0].x;
            sum3.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_3[3][0][0].y;
            sum1.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_1[2][0][0].x;
            sum1.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_1[2][0][0].y;
            sum2.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_2[2][0][0].x;
            sum2.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_2[2][0][0].y;
            sum3.x += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_3[2][0][0].x;
            sum3.y += s_Volume[threadIdx.z + 5][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_3[2][0][0].y;
            sum1.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_1[1][0][0].x;
            sum1.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_1[1][0][0].y;
            sum2.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_2[1][0][0].x;
            sum2.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_2[1][0][0].y;
            sum3.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_3[1][0][0].x;
            sum3.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_3[1][0][0].y;
            sum1.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_1[0][0][0].x;
            sum1.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_1[0][0][0].y;
            sum2.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_2[0][0][0].x;
            sum2.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_2[0][0][0].y;
            sum3.x += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_3[0][0][0].x;
            sum3.y += s_Volume[threadIdx.z + 7][threadIdx.y + 7][threadIdx.x + 7] * c_Quadrature_Filter_3[0][0][0].y;


			Filter_Response_1[Calculate_3D_Index(x,y,z,DATA_W,DATA_H)] = sum1;
			Filter_Response_2[Calculate_3D_Index(x,y,z,DATA_W,DATA_H)] = sum2;
			Filter_Response_3[Calculate_3D_Index(x,y,z,DATA_W,DATA_H)] = sum3;
	}	

}

// Smoothing CCA

__device__ __constant__ float4 c_CCA_2D_Filters[9][9];

__device__ __constant__ float2 c_CCA_3D_Filters[9];

// 2D

__device__ float4 Convolve_9x9(float image[64][64], int y, int x)
{
	float pixel; float4 sums;
	sums.x = 0.0f;
	sums.y = 0.0f;
	sums.z = 0.0f;
	sums.w = 0.0f;

    pixel = image[y - 4][x - 4];
    sums.x += pixel * c_CCA_2D_Filters[8][8].x;
    sums.y += pixel * c_CCA_2D_Filters[8][8].y;
    sums.z += pixel * c_CCA_2D_Filters[8][8].z;
    sums.w += pixel * c_CCA_2D_Filters[8][8].w;
    pixel = image[y - 3][x - 4];
    sums.x += pixel * c_CCA_2D_Filters[7][8].x;
    sums.y += pixel * c_CCA_2D_Filters[7][8].y;
    sums.z += pixel * c_CCA_2D_Filters[7][8].z;
    sums.w += pixel * c_CCA_2D_Filters[7][8].w;
    pixel = image[y - 2][x - 4];
    sums.x += pixel * c_CCA_2D_Filters[6][8].x;
    sums.y += pixel * c_CCA_2D_Filters[6][8].y;
    sums.z += pixel * c_CCA_2D_Filters[6][8].z;
    sums.w += pixel * c_CCA_2D_Filters[6][8].w;
    pixel = image[y - 1][x - 4];
    sums.x += pixel * c_CCA_2D_Filters[5][8].x;
    sums.y += pixel * c_CCA_2D_Filters[5][8].y;
    sums.z += pixel * c_CCA_2D_Filters[5][8].z;
    sums.w += pixel * c_CCA_2D_Filters[5][8].w;
    pixel = image[y + 0][x - 4];
    sums.x += pixel * c_CCA_2D_Filters[4][8].x;
    sums.y += pixel * c_CCA_2D_Filters[4][8].y;
    sums.z += pixel * c_CCA_2D_Filters[4][8].z;
    sums.w += pixel * c_CCA_2D_Filters[4][8].w;
    pixel = image[y + 1][x - 4];
    sums.x += pixel * c_CCA_2D_Filters[3][8].x;
    sums.y += pixel * c_CCA_2D_Filters[3][8].y;
    sums.z += pixel * c_CCA_2D_Filters[3][8].z;
    sums.w += pixel * c_CCA_2D_Filters[3][8].w;
    pixel = image[y + 2][x - 4];
    sums.x += pixel * c_CCA_2D_Filters[2][8].x;
    sums.y += pixel * c_CCA_2D_Filters[2][8].y;
    sums.z += pixel * c_CCA_2D_Filters[2][8].z;
    sums.w += pixel * c_CCA_2D_Filters[2][8].w;
    pixel = image[y + 3][x - 4];
    sums.x += pixel * c_CCA_2D_Filters[1][8].x;
    sums.y += pixel * c_CCA_2D_Filters[1][8].y;
    sums.z += pixel * c_CCA_2D_Filters[1][8].z;
    sums.w += pixel * c_CCA_2D_Filters[1][8].w;
    pixel = image[y + 4][x - 4];
    sums.x += pixel * c_CCA_2D_Filters[0][8].x;
    sums.y += pixel * c_CCA_2D_Filters[0][8].y;
    sums.z += pixel * c_CCA_2D_Filters[0][8].z;
    sums.w += pixel * c_CCA_2D_Filters[0][8].w;

    pixel = image[y - 4][x - 3];
    sums.x += pixel * c_CCA_2D_Filters[8][7].x;
    sums.y += pixel * c_CCA_2D_Filters[8][7].y;
    sums.z += pixel * c_CCA_2D_Filters[8][7].z;
    sums.w += pixel * c_CCA_2D_Filters[8][7].w;
    pixel = image[y - 3][x - 3];
    sums.x += pixel * c_CCA_2D_Filters[7][7].x;
    sums.y += pixel * c_CCA_2D_Filters[7][7].y;
    sums.z += pixel * c_CCA_2D_Filters[7][7].z;
    sums.w += pixel * c_CCA_2D_Filters[7][7].w;
    pixel = image[y - 2][x - 3];
    sums.x += pixel * c_CCA_2D_Filters[6][7].x;
    sums.y += pixel * c_CCA_2D_Filters[6][7].y;
    sums.z += pixel * c_CCA_2D_Filters[6][7].z;
    sums.w += pixel * c_CCA_2D_Filters[6][7].w;
    pixel = image[y - 1][x - 3];
    sums.x += pixel * c_CCA_2D_Filters[5][7].x;
    sums.y += pixel * c_CCA_2D_Filters[5][7].y;
    sums.z += pixel * c_CCA_2D_Filters[5][7].z;
    sums.w += pixel * c_CCA_2D_Filters[5][7].w;
    pixel = image[y + 0][x - 3];
    sums.x += pixel * c_CCA_2D_Filters[4][7].x;
    sums.y += pixel * c_CCA_2D_Filters[4][7].y;
    sums.z += pixel * c_CCA_2D_Filters[4][7].z;
    sums.w += pixel * c_CCA_2D_Filters[4][7].w;
    pixel = image[y + 1][x - 3];
    sums.x += pixel * c_CCA_2D_Filters[3][7].x;
    sums.y += pixel * c_CCA_2D_Filters[3][7].y;
    sums.z += pixel * c_CCA_2D_Filters[3][7].z;
    sums.w += pixel * c_CCA_2D_Filters[3][7].w;
    pixel = image[y + 2][x - 3];
    sums.x += pixel * c_CCA_2D_Filters[2][7].x;
    sums.y += pixel * c_CCA_2D_Filters[2][7].y;
    sums.z += pixel * c_CCA_2D_Filters[2][7].z;
    sums.w += pixel * c_CCA_2D_Filters[2][7].w;
    pixel = image[y + 3][x - 3];
    sums.x += pixel * c_CCA_2D_Filters[1][7].x;
    sums.y += pixel * c_CCA_2D_Filters[1][7].y;
    sums.z += pixel * c_CCA_2D_Filters[1][7].z;
    sums.w += pixel * c_CCA_2D_Filters[1][7].w;
    pixel = image[y + 4][x - 3];
    sums.x += pixel * c_CCA_2D_Filters[0][7].x;
    sums.y += pixel * c_CCA_2D_Filters[0][7].y;
    sums.z += pixel * c_CCA_2D_Filters[0][7].z;
    sums.w += pixel * c_CCA_2D_Filters[0][7].w;

    pixel = image[y - 4][x - 2];
    sums.x += pixel * c_CCA_2D_Filters[8][6].x;
    sums.y += pixel * c_CCA_2D_Filters[8][6].y;
    sums.z += pixel * c_CCA_2D_Filters[8][6].z;
    sums.w += pixel * c_CCA_2D_Filters[8][6].w;
    pixel = image[y - 3][x - 2];
    sums.x += pixel * c_CCA_2D_Filters[7][6].x;
    sums.y += pixel * c_CCA_2D_Filters[7][6].y;
    sums.z += pixel * c_CCA_2D_Filters[7][6].z;
    sums.w += pixel * c_CCA_2D_Filters[7][6].w;
    pixel = image[y - 2][x - 2];
    sums.x += pixel * c_CCA_2D_Filters[6][6].x;
    sums.y += pixel * c_CCA_2D_Filters[6][6].y;
    sums.z += pixel * c_CCA_2D_Filters[6][6].z;
    sums.w += pixel * c_CCA_2D_Filters[6][6].w;
    pixel = image[y - 1][x - 2];
    sums.x += pixel * c_CCA_2D_Filters[5][6].x;
    sums.y += pixel * c_CCA_2D_Filters[5][6].y;
    sums.z += pixel * c_CCA_2D_Filters[5][6].z;
    sums.w += pixel * c_CCA_2D_Filters[5][6].w;
    pixel = image[y + 0][x - 2];
    sums.x += pixel * c_CCA_2D_Filters[4][6].x;
    sums.y += pixel * c_CCA_2D_Filters[4][6].y;
    sums.z += pixel * c_CCA_2D_Filters[4][6].z;
    sums.w += pixel * c_CCA_2D_Filters[4][6].w;
    pixel = image[y + 1][x - 2];
    sums.x += pixel * c_CCA_2D_Filters[3][6].x;
    sums.y += pixel * c_CCA_2D_Filters[3][6].y;
    sums.z += pixel * c_CCA_2D_Filters[3][6].z;
    sums.w += pixel * c_CCA_2D_Filters[3][6].w;
    pixel = image[y + 2][x - 2];
    sums.x += pixel * c_CCA_2D_Filters[2][6].x;
    sums.y += pixel * c_CCA_2D_Filters[2][6].y;
    sums.z += pixel * c_CCA_2D_Filters[2][6].z;
    sums.w += pixel * c_CCA_2D_Filters[2][6].w;
    pixel = image[y + 3][x - 2];
    sums.x += pixel * c_CCA_2D_Filters[1][6].x;
    sums.y += pixel * c_CCA_2D_Filters[1][6].y;
    sums.z += pixel * c_CCA_2D_Filters[1][6].z;
    sums.w += pixel * c_CCA_2D_Filters[1][6].w;
    pixel = image[y + 4][x - 2];
    sums.x += pixel * c_CCA_2D_Filters[0][6].x;
    sums.y += pixel * c_CCA_2D_Filters[0][6].y;
    sums.z += pixel * c_CCA_2D_Filters[0][6].z;
    sums.w += pixel * c_CCA_2D_Filters[0][6].w;

    pixel = image[y - 4][x - 1];
    sums.x += pixel * c_CCA_2D_Filters[8][5].x;
    sums.y += pixel * c_CCA_2D_Filters[8][5].y;
    sums.z += pixel * c_CCA_2D_Filters[8][5].z;
    sums.w += pixel * c_CCA_2D_Filters[8][5].w;
    pixel = image[y - 3][x - 1];
    sums.x += pixel * c_CCA_2D_Filters[7][5].x;
    sums.y += pixel * c_CCA_2D_Filters[7][5].y;
    sums.z += pixel * c_CCA_2D_Filters[7][5].z;
    sums.w += pixel * c_CCA_2D_Filters[7][5].w;
    pixel = image[y - 2][x - 1];
    sums.x += pixel * c_CCA_2D_Filters[6][5].x;
    sums.y += pixel * c_CCA_2D_Filters[6][5].y;
    sums.z += pixel * c_CCA_2D_Filters[6][5].z;
    sums.w += pixel * c_CCA_2D_Filters[6][5].w;
    pixel = image[y - 1][x - 1];
    sums.x += pixel * c_CCA_2D_Filters[5][5].x;
    sums.y += pixel * c_CCA_2D_Filters[5][5].y;
    sums.z += pixel * c_CCA_2D_Filters[5][5].z;
    sums.w += pixel * c_CCA_2D_Filters[5][5].w;
    pixel = image[y + 0][x - 1];
    sums.x += pixel * c_CCA_2D_Filters[4][5].x;
    sums.y += pixel * c_CCA_2D_Filters[4][5].y;
    sums.z += pixel * c_CCA_2D_Filters[4][5].z;
    sums.w += pixel * c_CCA_2D_Filters[4][5].w;
    pixel = image[y + 1][x - 1];
    sums.x += pixel * c_CCA_2D_Filters[3][5].x;
    sums.y += pixel * c_CCA_2D_Filters[3][5].y;
    sums.z += pixel * c_CCA_2D_Filters[3][5].z;
    sums.w += pixel * c_CCA_2D_Filters[3][5].w;
    pixel = image[y + 2][x - 1];
    sums.x += pixel * c_CCA_2D_Filters[2][5].x;
    sums.y += pixel * c_CCA_2D_Filters[2][5].y;
    sums.z += pixel * c_CCA_2D_Filters[2][5].z;
    sums.w += pixel * c_CCA_2D_Filters[2][5].w;
    pixel = image[y + 3][x - 1];
    sums.x += pixel * c_CCA_2D_Filters[1][5].x;
    sums.y += pixel * c_CCA_2D_Filters[1][5].y;
    sums.z += pixel * c_CCA_2D_Filters[1][5].z;
    sums.w += pixel * c_CCA_2D_Filters[1][5].w;
    pixel = image[y + 4][x - 1];
    sums.x += pixel * c_CCA_2D_Filters[0][5].x;
    sums.y += pixel * c_CCA_2D_Filters[0][5].y;
    sums.z += pixel * c_CCA_2D_Filters[0][5].z;
    sums.w += pixel * c_CCA_2D_Filters[0][5].w;

    pixel = image[y - 4][x + 0];
    sums.x += pixel * c_CCA_2D_Filters[8][4].x;
    sums.y += pixel * c_CCA_2D_Filters[8][4].y;
    sums.z += pixel * c_CCA_2D_Filters[8][4].z;
    sums.w += pixel * c_CCA_2D_Filters[8][4].w;
    pixel = image[y - 3][x + 0];
    sums.x += pixel * c_CCA_2D_Filters[7][4].x;
    sums.y += pixel * c_CCA_2D_Filters[7][4].y;
    sums.z += pixel * c_CCA_2D_Filters[7][4].z;
    sums.w += pixel * c_CCA_2D_Filters[7][4].w;
    pixel = image[y - 2][x + 0];
    sums.x += pixel * c_CCA_2D_Filters[6][4].x;
    sums.y += pixel * c_CCA_2D_Filters[6][4].y;
    sums.z += pixel * c_CCA_2D_Filters[6][4].z;
    sums.w += pixel * c_CCA_2D_Filters[6][4].w;
    pixel = image[y - 1][x + 0];
    sums.x += pixel * c_CCA_2D_Filters[5][4].x;
    sums.y += pixel * c_CCA_2D_Filters[5][4].y;
    sums.z += pixel * c_CCA_2D_Filters[5][4].z;
    sums.w += pixel * c_CCA_2D_Filters[5][4].w;
    pixel = image[y + 0][x + 0];
    sums.x += pixel * c_CCA_2D_Filters[4][4].x;
    sums.y += pixel * c_CCA_2D_Filters[4][4].y;
    sums.z += pixel * c_CCA_2D_Filters[4][4].z;
    sums.w += pixel * c_CCA_2D_Filters[4][4].w;
    pixel = image[y + 1][x + 0];
    sums.x += pixel * c_CCA_2D_Filters[3][4].x;
    sums.y += pixel * c_CCA_2D_Filters[3][4].y;
    sums.z += pixel * c_CCA_2D_Filters[3][4].z;
    sums.w += pixel * c_CCA_2D_Filters[3][4].w;
    pixel = image[y + 2][x + 0];
    sums.x += pixel * c_CCA_2D_Filters[2][4].x;
    sums.y += pixel * c_CCA_2D_Filters[2][4].y;
    sums.z += pixel * c_CCA_2D_Filters[2][4].z;
    sums.w += pixel * c_CCA_2D_Filters[2][4].w;
    pixel = image[y + 3][x + 0];
    sums.x += pixel * c_CCA_2D_Filters[1][4].x;
    sums.y += pixel * c_CCA_2D_Filters[1][4].y;
    sums.z += pixel * c_CCA_2D_Filters[1][4].z;
    sums.w += pixel * c_CCA_2D_Filters[1][4].w;
    pixel = image[y + 4][x + 0];
    sums.x += pixel * c_CCA_2D_Filters[0][4].x;
    sums.y += pixel * c_CCA_2D_Filters[0][4].y;
    sums.z += pixel * c_CCA_2D_Filters[0][4].z;
    sums.w += pixel * c_CCA_2D_Filters[0][4].w;

    pixel = image[y - 4][x + 1];
    sums.x += pixel * c_CCA_2D_Filters[8][3].x;
    sums.y += pixel * c_CCA_2D_Filters[8][3].y;
    sums.z += pixel * c_CCA_2D_Filters[8][3].z;
    sums.w += pixel * c_CCA_2D_Filters[8][3].w;
    pixel = image[y - 3][x + 1];
    sums.x += pixel * c_CCA_2D_Filters[7][3].x;
    sums.y += pixel * c_CCA_2D_Filters[7][3].y;
    sums.z += pixel * c_CCA_2D_Filters[7][3].z;
    sums.w += pixel * c_CCA_2D_Filters[7][3].w;
    pixel = image[y - 2][x + 1];
    sums.x += pixel * c_CCA_2D_Filters[6][3].x;
    sums.y += pixel * c_CCA_2D_Filters[6][3].y;
    sums.z += pixel * c_CCA_2D_Filters[6][3].z;
    sums.w += pixel * c_CCA_2D_Filters[6][3].w;
    pixel = image[y - 1][x + 1];
    sums.x += pixel * c_CCA_2D_Filters[5][3].x;
    sums.y += pixel * c_CCA_2D_Filters[5][3].y;
    sums.z += pixel * c_CCA_2D_Filters[5][3].z;
    sums.w += pixel * c_CCA_2D_Filters[5][3].w;
    pixel = image[y + 0][x + 1];
    sums.x += pixel * c_CCA_2D_Filters[4][3].x;
    sums.y += pixel * c_CCA_2D_Filters[4][3].y;
    sums.z += pixel * c_CCA_2D_Filters[4][3].z;
    sums.w += pixel * c_CCA_2D_Filters[4][3].w;
    pixel = image[y + 1][x + 1];
    sums.x += pixel * c_CCA_2D_Filters[3][3].x;
    sums.y += pixel * c_CCA_2D_Filters[3][3].y;
    sums.z += pixel * c_CCA_2D_Filters[3][3].z;
    sums.w += pixel * c_CCA_2D_Filters[3][3].w;
    pixel = image[y + 2][x + 1];
    sums.x += pixel * c_CCA_2D_Filters[2][3].x;
    sums.y += pixel * c_CCA_2D_Filters[2][3].y;
    sums.z += pixel * c_CCA_2D_Filters[2][3].z;
    sums.w += pixel * c_CCA_2D_Filters[2][3].w;
    pixel = image[y + 3][x + 1];
    sums.x += pixel * c_CCA_2D_Filters[1][3].x;
    sums.y += pixel * c_CCA_2D_Filters[1][3].y;
    sums.z += pixel * c_CCA_2D_Filters[1][3].z;
    sums.w += pixel * c_CCA_2D_Filters[1][3].w;
    pixel = image[y + 4][x + 1];
    sums.x += pixel * c_CCA_2D_Filters[0][3].x;
    sums.y += pixel * c_CCA_2D_Filters[0][3].y;
    sums.z += pixel * c_CCA_2D_Filters[0][3].z;
    sums.w += pixel * c_CCA_2D_Filters[0][3].w;

    pixel = image[y - 4][x + 2];
    sums.x += pixel * c_CCA_2D_Filters[8][2].x;
    sums.y += pixel * c_CCA_2D_Filters[8][2].y;
    sums.z += pixel * c_CCA_2D_Filters[8][2].z;
    sums.w += pixel * c_CCA_2D_Filters[8][2].w;
    pixel = image[y - 3][x + 2];
    sums.x += pixel * c_CCA_2D_Filters[7][2].x;
    sums.y += pixel * c_CCA_2D_Filters[7][2].y;
    sums.z += pixel * c_CCA_2D_Filters[7][2].z;
    sums.w += pixel * c_CCA_2D_Filters[7][2].w;
    pixel = image[y - 2][x + 2];
    sums.x += pixel * c_CCA_2D_Filters[6][2].x;
    sums.y += pixel * c_CCA_2D_Filters[6][2].y;
    sums.z += pixel * c_CCA_2D_Filters[6][2].z;
    sums.w += pixel * c_CCA_2D_Filters[6][2].w;
    pixel = image[y - 1][x + 2];
    sums.x += pixel * c_CCA_2D_Filters[5][2].x;
    sums.y += pixel * c_CCA_2D_Filters[5][2].y;
    sums.z += pixel * c_CCA_2D_Filters[5][2].z;
    sums.w += pixel * c_CCA_2D_Filters[5][2].w;
    pixel = image[y + 0][x + 2];
    sums.x += pixel * c_CCA_2D_Filters[4][2].x;
    sums.y += pixel * c_CCA_2D_Filters[4][2].y;
    sums.z += pixel * c_CCA_2D_Filters[4][2].z;
    sums.w += pixel * c_CCA_2D_Filters[4][2].w;
    pixel = image[y + 1][x + 2];
    sums.x += pixel * c_CCA_2D_Filters[3][2].x;
    sums.y += pixel * c_CCA_2D_Filters[3][2].y;
    sums.z += pixel * c_CCA_2D_Filters[3][2].z;
    sums.w += pixel * c_CCA_2D_Filters[3][2].w;
    pixel = image[y + 2][x + 2];
    sums.x += pixel * c_CCA_2D_Filters[2][2].x;
    sums.y += pixel * c_CCA_2D_Filters[2][2].y;
    sums.z += pixel * c_CCA_2D_Filters[2][2].z;
    sums.w += pixel * c_CCA_2D_Filters[2][2].w;
    pixel = image[y + 3][x + 2];
    sums.x += pixel * c_CCA_2D_Filters[1][2].x;
    sums.y += pixel * c_CCA_2D_Filters[1][2].y;
    sums.z += pixel * c_CCA_2D_Filters[1][2].z;
    sums.w += pixel * c_CCA_2D_Filters[1][2].w;
    pixel = image[y + 4][x + 2];
    sums.x += pixel * c_CCA_2D_Filters[0][2].x;
    sums.y += pixel * c_CCA_2D_Filters[0][2].y;
    sums.z += pixel * c_CCA_2D_Filters[0][2].z;
    sums.w += pixel * c_CCA_2D_Filters[0][2].w;

    pixel = image[y - 4][x + 3];
    sums.x += pixel * c_CCA_2D_Filters[8][1].x;
    sums.y += pixel * c_CCA_2D_Filters[8][1].y;
    sums.z += pixel * c_CCA_2D_Filters[8][1].z;
    sums.w += pixel * c_CCA_2D_Filters[8][1].w;
    pixel = image[y - 3][x + 3];
    sums.x += pixel * c_CCA_2D_Filters[7][1].x;
    sums.y += pixel * c_CCA_2D_Filters[7][1].y;
    sums.z += pixel * c_CCA_2D_Filters[7][1].z;
    sums.w += pixel * c_CCA_2D_Filters[7][1].w;
    pixel = image[y - 2][x + 3];
    sums.x += pixel * c_CCA_2D_Filters[6][1].x;
    sums.y += pixel * c_CCA_2D_Filters[6][1].y;
    sums.z += pixel * c_CCA_2D_Filters[6][1].z;
    sums.w += pixel * c_CCA_2D_Filters[6][1].w;
    pixel = image[y - 1][x + 3];
    sums.x += pixel * c_CCA_2D_Filters[5][1].x;
    sums.y += pixel * c_CCA_2D_Filters[5][1].y;
    sums.z += pixel * c_CCA_2D_Filters[5][1].z;
    sums.w += pixel * c_CCA_2D_Filters[5][1].w;
    pixel = image[y + 0][x + 3];
    sums.x += pixel * c_CCA_2D_Filters[4][1].x;
    sums.y += pixel * c_CCA_2D_Filters[4][1].y;
    sums.z += pixel * c_CCA_2D_Filters[4][1].z;
    sums.w += pixel * c_CCA_2D_Filters[4][1].w;
    pixel = image[y + 1][x + 3];
    sums.x += pixel * c_CCA_2D_Filters[3][1].x;
    sums.y += pixel * c_CCA_2D_Filters[3][1].y;
    sums.z += pixel * c_CCA_2D_Filters[3][1].z;
    sums.w += pixel * c_CCA_2D_Filters[3][1].w;
    pixel = image[y + 2][x + 3];
    sums.x += pixel * c_CCA_2D_Filters[2][1].x;
    sums.y += pixel * c_CCA_2D_Filters[2][1].y;
    sums.z += pixel * c_CCA_2D_Filters[2][1].z;
    sums.w += pixel * c_CCA_2D_Filters[2][1].w;
    pixel = image[y + 3][x + 3];
    sums.x += pixel * c_CCA_2D_Filters[1][1].x;
    sums.y += pixel * c_CCA_2D_Filters[1][1].y;
    sums.z += pixel * c_CCA_2D_Filters[1][1].z;
    sums.w += pixel * c_CCA_2D_Filters[1][1].w;
    pixel = image[y + 4][x + 3];
    sums.x += pixel * c_CCA_2D_Filters[0][1].x;
    sums.y += pixel * c_CCA_2D_Filters[0][1].y;
    sums.z += pixel * c_CCA_2D_Filters[0][1].z;
    sums.w += pixel * c_CCA_2D_Filters[0][1].w;

    pixel = image[y - 4][x + 4];
    sums.x += pixel * c_CCA_2D_Filters[8][0].x;
    sums.y += pixel * c_CCA_2D_Filters[8][0].y;
    sums.z += pixel * c_CCA_2D_Filters[8][0].z;
    sums.w += pixel * c_CCA_2D_Filters[8][0].w;
    pixel = image[y - 3][x + 4];
    sums.x += pixel * c_CCA_2D_Filters[7][0].x;
    sums.y += pixel * c_CCA_2D_Filters[7][0].y;
    sums.z += pixel * c_CCA_2D_Filters[7][0].z;
    sums.w += pixel * c_CCA_2D_Filters[7][0].w;
    pixel = image[y - 2][x + 4];
    sums.x += pixel * c_CCA_2D_Filters[6][0].x;
    sums.y += pixel * c_CCA_2D_Filters[6][0].y;
    sums.z += pixel * c_CCA_2D_Filters[6][0].z;
    sums.w += pixel * c_CCA_2D_Filters[6][0].w;
    pixel = image[y - 1][x + 4];
    sums.x += pixel * c_CCA_2D_Filters[5][0].x;
    sums.y += pixel * c_CCA_2D_Filters[5][0].y;
    sums.z += pixel * c_CCA_2D_Filters[5][0].z;
    sums.w += pixel * c_CCA_2D_Filters[5][0].w;
    pixel = image[y + 0][x + 4];
    sums.x += pixel * c_CCA_2D_Filters[4][0].x;
    sums.y += pixel * c_CCA_2D_Filters[4][0].y;
    sums.z += pixel * c_CCA_2D_Filters[4][0].z;
    sums.w += pixel * c_CCA_2D_Filters[4][0].w;
    pixel = image[y + 1][x + 4];
    sums.x += pixel * c_CCA_2D_Filters[3][0].x;
    sums.y += pixel * c_CCA_2D_Filters[3][0].y;
    sums.z += pixel * c_CCA_2D_Filters[3][0].z;
    sums.w += pixel * c_CCA_2D_Filters[3][0].w;
    pixel = image[y + 2][x + 4];
    sums.x += pixel * c_CCA_2D_Filters[2][0].x;
    sums.y += pixel * c_CCA_2D_Filters[2][0].y;
    sums.z += pixel * c_CCA_2D_Filters[2][0].z;
    sums.w += pixel * c_CCA_2D_Filters[2][0].w;
    pixel = image[y + 3][x + 4];
    sums.x += pixel * c_CCA_2D_Filters[1][0].x;
    sums.y += pixel * c_CCA_2D_Filters[1][0].y;
    sums.z += pixel * c_CCA_2D_Filters[1][0].z;
    sums.w += pixel * c_CCA_2D_Filters[1][0].w;
    pixel = image[y + 4][x + 4];
    sums.x += pixel * c_CCA_2D_Filters[0][0].x;
    sums.y += pixel * c_CCA_2D_Filters[0][0].y;
    sums.z += pixel * c_CCA_2D_Filters[0][0].z;
    sums.w += pixel * c_CCA_2D_Filters[0][0].w;

	return sums;
}

__global__ void Smoothing_CCA_2D(float *Filter_Responses_1, float* Filter_Responses_2, float* Filter_Responses_3, float* Filter_Responses_4, float* fMRI_Volumes, int z, int DATA_W, int DATA_H, int DATA_D, int DATA_T, int blocksInY, float invBlocksInY, int xBlockDifference, int yBlockDifference)
{   
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
	volatile int x = __umul24(blockIdx.x,blockDim.x / 2) * 3 + threadIdx.x;
	volatile int y = __umul24(blockIdxy ,blockDim.y) * 3 + threadIdx.y;
	int t = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;	

    if (x >= (DATA_W + xBlockDifference) || y >= (DATA_H + yBlockDifference) || t >= DATA_T)
        return;
	
	__shared__ float s_Image[64][64];    

	// Blocks
		
	// 1   2  3  4
	// 5   6  7  8
	// 9  10 11 12
	// 13 14 15 16

	s_Image[threadIdx.y][threadIdx.x] = 0.0f;
	s_Image[threadIdx.y + 16][threadIdx.x] = 0.0f;
	s_Image[threadIdx.y + 32][threadIdx.x] = 0.0f;
	s_Image[threadIdx.y + 48][threadIdx.x] = 0.0f;
	s_Image[threadIdx.y][threadIdx.x + 32] = 0.0f;
	s_Image[threadIdx.y + 16][threadIdx.x + 32] = 0.0f;
	s_Image[threadIdx.y + 32][threadIdx.x + 32] = 0.0f;
	s_Image[threadIdx.y + 48][threadIdx.x + 32] = 0.0f;


	// Read data into shared memory

	
	// First row, blocks 1 + 2
	if ( ((x - 8) >= 0) && ((x - 8) < DATA_W) && ((y - 8) >= 0) && ((y - 8) < DATA_H) )
	{
		s_Image[threadIdx.y][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x - 8,y - 8,z,t,DATA_W, DATA_H, DATA_D)];	
	}

	// First row, blocks 3 + 4
	if ( ((x + 24) < DATA_W) && ((y - 8) >= 0) && ((y - 8) < DATA_H) )
	{
		s_Image[threadIdx.y][threadIdx.x + 32] = fMRI_Volumes[Calculate_4D_Index(x + 24,y - 8,z,t,DATA_W, DATA_H, DATA_D)];	
	}

	// Second row, blocks 5 + 6
	if ( ((x - 8) >= 0) && ((x - 8) < DATA_W) && ((y + 8) < DATA_H) )
	{
		s_Image[threadIdx.y + 16][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x - 8,y + 8,z,t,DATA_W, DATA_H, DATA_D)];	
	}

	// Second row, blocks 7 + 8
	if ( ((x + 24) < DATA_W) && ((y + 8) < DATA_H) )
	{
		s_Image[threadIdx.y + 16][threadIdx.x + 32] = fMRI_Volumes[Calculate_4D_Index(x + 24,y + 8,z,t,DATA_W, DATA_H, DATA_D)];	
	}

	// Third row, blocks 9 + 10
	if ( ((x - 8) >= 0) && ((x - 8) < DATA_W) && ((y + 24) < DATA_H) )
	{
		s_Image[threadIdx.y + 32][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x - 8,y + 24,z,t,DATA_W, DATA_H, DATA_D)];	
	}

	// Third row, blocks 11 + 12
	if ( ((x + 24) < DATA_W) && ((y + 24) < DATA_H) )
	{
		s_Image[threadIdx.y + 32][threadIdx.x + 32] = fMRI_Volumes[Calculate_4D_Index(x + 24,y + 24,z,t,DATA_W, DATA_H, DATA_D)];	
	}

	// Fourth row, blocks 13 + 14
	if ( ((x - 8) >= 0) && ((x - 8) < DATA_W) && ((y + 40) < DATA_H) )
	{
		s_Image[threadIdx.y + 48][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x - 8,y + 40,z,t,DATA_W, DATA_H, DATA_D)];	
	}

	// Fourth row, blocks 15 + 16		
	if ( ((x + 24) < DATA_W) && ((y + 40) < DATA_H) )
	{
		s_Image[threadIdx.y + 48][threadIdx.x + 32] = fMRI_Volumes[Calculate_4D_Index(x + 24,y + 40,z,t,DATA_W, DATA_H, DATA_D)];	
	}
	

	__syncthreads();

	// Only threads inside the image do the convolution, calculate filter responses for 48 x 48 pixels

	if ( (x < DATA_W) && (y < DATA_H) )
	{
		float4 filter_responses = Convolve_9x9(s_Image, threadIdx.y + 8, threadIdx.x + 8);
		Filter_Responses_1[Calculate_4D_Index(x,y,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.x;
		Filter_Responses_2[Calculate_4D_Index(x,y,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.y;
		Filter_Responses_3[Calculate_4D_Index(x,y,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.z;
		Filter_Responses_4[Calculate_4D_Index(x,y,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.w;
	}

	if ( (x < DATA_W) && ((y + 16) < DATA_H) )
	{
		float4 filter_responses = Convolve_9x9(s_Image, threadIdx.y + 24, threadIdx.x + 8);
		Filter_Responses_1[Calculate_4D_Index(x,y + 16,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.x;
		Filter_Responses_2[Calculate_4D_Index(x,y + 16,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.y;
		Filter_Responses_3[Calculate_4D_Index(x,y + 16,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.z;
		Filter_Responses_4[Calculate_4D_Index(x,y + 16,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.w;
	}

	if ( (x < DATA_W) && ((y + 32) < DATA_H) )
	{
		float4 filter_responses = Convolve_9x9(s_Image, threadIdx.y + 40, threadIdx.x + 8);
		Filter_Responses_1[Calculate_4D_Index(x,y + 32,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.x;
		Filter_Responses_2[Calculate_4D_Index(x,y + 32,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.y;
		Filter_Responses_3[Calculate_4D_Index(x,y + 32,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.z;
		Filter_Responses_4[Calculate_4D_Index(x,y + 32,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.w;
	}

	if (threadIdx.x < 16)
	{
		if ( ((x + 32) < DATA_W) && (y < DATA_H) )
		{
			float4 filter_responses = Convolve_9x9(s_Image, threadIdx.y + 8, threadIdx.x + 40);
			Filter_Responses_1[Calculate_4D_Index(x + 32,y,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.x;
			Filter_Responses_2[Calculate_4D_Index(x + 32,y,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.y;
			Filter_Responses_3[Calculate_4D_Index(x + 32,y,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.z;
			Filter_Responses_4[Calculate_4D_Index(x + 32,y,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.w;
		}

		if ( ((x + 32) < DATA_W) && ((y + 16) < DATA_H) )
		{
			float4 filter_responses = Convolve_9x9(s_Image, threadIdx.y + 24, threadIdx.x + 40);
			Filter_Responses_1[Calculate_4D_Index(x + 32,y + 16,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.x;
			Filter_Responses_2[Calculate_4D_Index(x + 32,y + 16,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.y;
			Filter_Responses_3[Calculate_4D_Index(x + 32,y + 16,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.z;
			Filter_Responses_4[Calculate_4D_Index(x + 32,y + 16,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.w;
		}

		if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
		{
			float4 filter_responses = Convolve_9x9(s_Image, threadIdx.y + 40, threadIdx.x + 40);
			Filter_Responses_1[Calculate_4D_Index(x + 32,y + 32,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.x;
			Filter_Responses_2[Calculate_4D_Index(x + 32,y + 32,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.y;
			Filter_Responses_3[Calculate_4D_Index(x + 32,y + 32,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.z;
			Filter_Responses_4[Calculate_4D_Index(x + 32,y + 32,z,t,DATA_W, DATA_H, DATA_D)] = filter_responses.w;
		}
	}
}

// 3D

__device__ __constant__ float c_Smoothing_Kernel[9];

__global__ void convolutionRows_CCA(float2 *Filter_Responses, float* fMRI_Volumes, float* Certainty, int t, int DATA_W, int DATA_H, int DATA_D, int blocksInY, float invBlocksInY, int xBlockDifference, int yBlockDifference, int zBlockDifference)
{   
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
	volatile int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	volatile int y = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
	volatile int z = __umul24(blockIdxz ,blockDim.z) * 4 + threadIdx.z;	

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

	// Read data into shared memory

	// Upper apron + first half main data

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && (z < DATA_D) )
	{		
		s_Volume[threadIdx.z][threadIdx.y][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate_3D_Index(x,y - 4, z, DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 2) < DATA_D) )
	{
		s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z + 2,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate_3D_Index(x,y - 4, z + 2, DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 4) < DATA_D) )
	{
		s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z + 4,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate_3D_Index(x,y - 4, z + 4, DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 6) < DATA_D) )
	{
		s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z + 6,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate_3D_Index(x,y - 4, z + 6, DATA_W, DATA_H)];
	}

	// Second half main data + lower apron

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 4,z,t,DATA_W, DATA_H, DATA_D)]  * Certainty[Calculate_3D_Index(x,y + 4, z, DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 2) < DATA_D) )
	{
		s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 4,z + 2,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate_3D_Index(x,y + 4, z + 2, DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 4) < DATA_D) )
	{
		s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 4,z + 4,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate_3D_Index(x,y + 4, z + 4, DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 6) < DATA_D) )
	{
		s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 4,z + 6,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate_3D_Index(x,y + 4, z + 6, DATA_W, DATA_H)];
	}


	__syncthreads(); 

	// Only threads within the volume do the convolution
	if ( (x < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
	    float2 sum; sum.x = 0.0f; sum.y = 0.0f;

		sum.x += s_Volume[threadIdx.z][threadIdx.y + 0][threadIdx.x] * c_CCA_3D_Filters[8].x;
		sum.x += s_Volume[threadIdx.z][threadIdx.y + 1][threadIdx.x] * c_CCA_3D_Filters[7].x;
		sum.x += s_Volume[threadIdx.z][threadIdx.y + 2][threadIdx.x] * c_CCA_3D_Filters[6].x;
		sum.x += s_Volume[threadIdx.z][threadIdx.y + 3][threadIdx.x] * c_CCA_3D_Filters[5].x;
		sum.x += s_Volume[threadIdx.z][threadIdx.y + 4][threadIdx.x] * c_CCA_3D_Filters[4].x;	
		sum.x += s_Volume[threadIdx.z][threadIdx.y + 5][threadIdx.x] * c_CCA_3D_Filters[3].x;
		sum.x += s_Volume[threadIdx.z][threadIdx.y + 6][threadIdx.x] * c_CCA_3D_Filters[2].x;
		sum.x += s_Volume[threadIdx.z][threadIdx.y + 7][threadIdx.x] * c_CCA_3D_Filters[1].x;
		sum.x += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x] * c_CCA_3D_Filters[0].x;

		sum.y += s_Volume[threadIdx.z][threadIdx.y + 0][threadIdx.x] * c_CCA_3D_Filters[8].y;
		sum.y += s_Volume[threadIdx.z][threadIdx.y + 1][threadIdx.x] * c_CCA_3D_Filters[7].y;
		sum.y += s_Volume[threadIdx.z][threadIdx.y + 2][threadIdx.x] * c_CCA_3D_Filters[6].y;
		sum.y += s_Volume[threadIdx.z][threadIdx.y + 3][threadIdx.x] * c_CCA_3D_Filters[5].y;
		sum.y += s_Volume[threadIdx.z][threadIdx.y + 4][threadIdx.x] * c_CCA_3D_Filters[4].y;	
		sum.y += s_Volume[threadIdx.z][threadIdx.y + 5][threadIdx.x] * c_CCA_3D_Filters[3].y;
		sum.y += s_Volume[threadIdx.z][threadIdx.y + 6][threadIdx.x] * c_CCA_3D_Filters[2].y;
		sum.y += s_Volume[threadIdx.z][threadIdx.y + 7][threadIdx.x] * c_CCA_3D_Filters[1].y;
		sum.y += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x] * c_CCA_3D_Filters[0].y;

		Filter_Responses[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)] = sum;	
	}

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 2) < DATA_D) )
	{
	    float2 sum; sum.x = 0.0f; sum.y = 0.0f;

		sum.x += s_Volume[threadIdx.z + 2][threadIdx.y + 0][threadIdx.x] * c_CCA_3D_Filters[8].x;
		sum.x += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x] * c_CCA_3D_Filters[7].x;
		sum.x += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x] * c_CCA_3D_Filters[6].x;
		sum.x += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x] * c_CCA_3D_Filters[5].x;
		sum.x += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x] * c_CCA_3D_Filters[4].x;	
		sum.x += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x] * c_CCA_3D_Filters[3].x;
		sum.x += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x] * c_CCA_3D_Filters[2].x;
		sum.x += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x] * c_CCA_3D_Filters[1].x;
		sum.x += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x] * c_CCA_3D_Filters[0].x;

		sum.y += s_Volume[threadIdx.z + 2][threadIdx.y + 0][threadIdx.x] * c_CCA_3D_Filters[8].y;
		sum.y += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x] * c_CCA_3D_Filters[7].y;
		sum.y += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x] * c_CCA_3D_Filters[6].y;
		sum.y += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x] * c_CCA_3D_Filters[5].y;
		sum.y += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x] * c_CCA_3D_Filters[4].y;	
		sum.y += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x] * c_CCA_3D_Filters[3].y;
		sum.y += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x] * c_CCA_3D_Filters[2].y;
		sum.y += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x] * c_CCA_3D_Filters[1].y;
		sum.y += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x] * c_CCA_3D_Filters[0].y;

		Filter_Responses[Calculate_3D_Index(x,y,z + 2,DATA_W, DATA_H)] = sum;	
	}

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 4) < DATA_D) )
	{
	    float2 sum; sum.x = 0.0f; sum.y = 0.0f;

		sum.x += s_Volume[threadIdx.z + 4][threadIdx.y + 0][threadIdx.x] * c_CCA_3D_Filters[8].x;
		sum.x += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x] * c_CCA_3D_Filters[7].x;
		sum.x += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x] * c_CCA_3D_Filters[6].x;
		sum.x += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x] * c_CCA_3D_Filters[5].x;
		sum.x += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x] * c_CCA_3D_Filters[4].x;	
		sum.x += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x] * c_CCA_3D_Filters[3].x;
		sum.x += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x] * c_CCA_3D_Filters[2].x;
		sum.x += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x] * c_CCA_3D_Filters[1].x;
		sum.x += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x] * c_CCA_3D_Filters[0].x;

		sum.y += s_Volume[threadIdx.z + 4][threadIdx.y + 0][threadIdx.x] * c_CCA_3D_Filters[8].y;
		sum.y += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x] * c_CCA_3D_Filters[7].y;
		sum.y += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x] * c_CCA_3D_Filters[6].y;
		sum.y += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x] * c_CCA_3D_Filters[5].y;
		sum.y += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x] * c_CCA_3D_Filters[4].y;	
		sum.y += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x] * c_CCA_3D_Filters[3].y;
		sum.y += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x] * c_CCA_3D_Filters[2].y;
		sum.y += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x] * c_CCA_3D_Filters[1].y;
		sum.y += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x] * c_CCA_3D_Filters[0].y;

		Filter_Responses[Calculate_3D_Index(x,y,z + 4,DATA_W, DATA_H)] = sum;	
	}

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 6) < DATA_D) )
	{
	    float2 sum; sum.x = 0.0f; sum.y = 0.0f;

		sum.x += s_Volume[threadIdx.z + 6][threadIdx.y + 0][threadIdx.x] * c_CCA_3D_Filters[8].x;
		sum.x += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x] * c_CCA_3D_Filters[7].x;
		sum.x += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x] * c_CCA_3D_Filters[6].x;
		sum.x += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x] * c_CCA_3D_Filters[5].x;
		sum.x += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x] * c_CCA_3D_Filters[4].x;	
		sum.x += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x] * c_CCA_3D_Filters[3].x;
		sum.x += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x] * c_CCA_3D_Filters[2].x;
		sum.x += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x] * c_CCA_3D_Filters[1].x;
		sum.x += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x] * c_CCA_3D_Filters[0].x;

		sum.y += s_Volume[threadIdx.z + 6][threadIdx.y + 0][threadIdx.x] * c_CCA_3D_Filters[8].y;
		sum.y += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x] * c_CCA_3D_Filters[7].y;
		sum.y += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x] * c_CCA_3D_Filters[6].y;
		sum.y += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x] * c_CCA_3D_Filters[5].y;
		sum.y += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x] * c_CCA_3D_Filters[4].y;	
		sum.y += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x] * c_CCA_3D_Filters[3].y;
		sum.y += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x] * c_CCA_3D_Filters[2].y;
		sum.y += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x] * c_CCA_3D_Filters[1].y;
		sum.y += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x] * c_CCA_3D_Filters[0].y;

		Filter_Responses[Calculate_3D_Index(x,y,z + 6,DATA_W, DATA_H)] = sum;	
	}
}



__global__ void MyMemset(float* data, float value, int DATA_W, int DATA_H, int DATA_D, int blocksInY, float invBlocksInY)
{
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
	volatile unsigned int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	volatile unsigned int y = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
	volatile unsigned int z = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;
	
	if ((x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D))
		return;

	int idx = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);

	data[idx] = value;
}

__global__ void ThresholdfMRIData(float* brain_mask, float* fMRI_volumes, float threshold, int DATA_W, int DATA_H, int DATA_D, int blocksInY, float invBlocksInY)
{
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
	volatile unsigned int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	volatile unsigned int y = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
	volatile unsigned int z = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;
	
	if ((x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D))
		return;
	
	int idx3D = Calculate_3D_Index(x, y, z, DATA_W, DATA_H);
	int idx4D = Calculate_4D_Index(x, y, z, 0, DATA_W, DATA_H, DATA_D);
	float value = fMRI_volumes[idx4D];

	if (value >= threshold)
	{
		brain_mask[idx3D] = 1.0f;
	}
	else
	{
		brain_mask[idx3D] = 0.0f;
	}
}

// Smoothing GLM

__global__ void convolutionRows(float *Filter_Responses, float* fMRI_Volumes, float* Certainty, int t, int DATA_W, int DATA_H, int DATA_D, int blocksInY, float invBlocksInY, int xBlockDifference, int yBlockDifference, int zBlockDifference)
{   
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
	volatile int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	volatile int y = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
	volatile int z = __umul24(blockIdxz ,blockDim.z) * 4 + threadIdx.z;	

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

	// Read data into shared memory

	// Upper apron + first half main data

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && (z < DATA_D) )
	{		
		s_Volume[threadIdx.z][threadIdx.y][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate_3D_Index(x,y - 4, z, DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 2) < DATA_D) )
	{
		s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z + 2,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate_3D_Index(x,y - 4, z + 2, DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 4) < DATA_D) )
	{
		s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z + 4,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate_3D_Index(x,y - 4, z + 4, DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 6) < DATA_D) )
	{
		s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y - 4,z + 6,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate_3D_Index(x,y - 4, z + 6, DATA_W, DATA_H)];
	}

	// Second half main data + lower apron

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && (z < DATA_D) )
	{
		s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 4,z,t,DATA_W, DATA_H, DATA_D)]  * Certainty[Calculate_3D_Index(x,y + 4, z, DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 2) < DATA_D) )
	{
		s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 4,z + 2,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate_3D_Index(x,y + 4, z + 2, DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 4) < DATA_D) )
	{
		s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 4,z + 4,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate_3D_Index(x,y + 4, z + 4, DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 6) < DATA_D) )
	{
		s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x,y + 4,z + 6,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate_3D_Index(x,y + 4, z + 6, DATA_W, DATA_H)];
	}


	__syncthreads(); 

	// Only threads within the volume do the convolution
	if ( (x < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z][threadIdx.y + 0][threadIdx.x] * c_Smoothing_Kernel[8];
		sum += s_Volume[threadIdx.z][threadIdx.y + 1][threadIdx.x] * c_Smoothing_Kernel[7];
		sum += s_Volume[threadIdx.z][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Kernel[6];
		sum += s_Volume[threadIdx.z][threadIdx.y + 3][threadIdx.x] * c_Smoothing_Kernel[5];
		sum += s_Volume[threadIdx.z][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Kernel[4];	
		sum += s_Volume[threadIdx.z][threadIdx.y + 5][threadIdx.x] * c_Smoothing_Kernel[3];
		sum += s_Volume[threadIdx.z][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Kernel[2];
		sum += s_Volume[threadIdx.z][threadIdx.y + 7][threadIdx.x] * c_Smoothing_Kernel[1];
		sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x] * c_Smoothing_Kernel[0];

		Filter_Responses[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)] = sum;	
	}

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 2) < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 0][threadIdx.x] * c_Smoothing_Kernel[8];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x] * c_Smoothing_Kernel[7];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Kernel[6];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 3][threadIdx.x] * c_Smoothing_Kernel[5];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Kernel[4];	
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 5][threadIdx.x] * c_Smoothing_Kernel[3];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Kernel[2];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 7][threadIdx.x] * c_Smoothing_Kernel[1];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x] * c_Smoothing_Kernel[0];

		Filter_Responses[Calculate_3D_Index(x,y,z + 2,DATA_W, DATA_H)] = sum;	
	}

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 4) < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 0][threadIdx.x] * c_Smoothing_Kernel[8];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 1][threadIdx.x] * c_Smoothing_Kernel[7];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Kernel[6];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 3][threadIdx.x] * c_Smoothing_Kernel[5];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Kernel[4];	
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 5][threadIdx.x] * c_Smoothing_Kernel[3];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Kernel[2];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 7][threadIdx.x] * c_Smoothing_Kernel[1];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x] * c_Smoothing_Kernel[0];

		Filter_Responses[Calculate_3D_Index(x,y,z + 4,DATA_W, DATA_H)] = sum;	
	}

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 6) < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 0][threadIdx.x] * c_Smoothing_Kernel[8];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 1][threadIdx.x] * c_Smoothing_Kernel[7];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Kernel[6];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 3][threadIdx.x] * c_Smoothing_Kernel[5];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Kernel[4];	
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 5][threadIdx.x] * c_Smoothing_Kernel[3];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Kernel[2];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 7][threadIdx.x] * c_Smoothing_Kernel[1];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x] * c_Smoothing_Kernel[0];

		Filter_Responses[Calculate_3D_Index(x,y,z + 6,DATA_W, DATA_H)] = sum;	
	}
}

__global__ void convolutionColumns(float *Filter_Responses, float* fMRI_Volume, int t, int DATA_W, int DATA_H, int DATA_D, int blocksInY, float invBlocksInY, int xBlockDifference, int yBlockDifference, int zBlockDifference)
{   
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
	volatile int x = __umul24(blockIdx.x,blockDim.x) / 32 * 24 + threadIdx.x;
	volatile int y = __umul24(blockIdxy ,blockDim.y) * 2 + threadIdx.y;
	volatile int z = __umul24(blockIdxz ,blockDim.z) * 4 + threadIdx.z;	

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
	
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 0] * c_Smoothing_Kernel[8];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 1] * c_Smoothing_Kernel[7];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 2] * c_Smoothing_Kernel[6];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 3] * c_Smoothing_Kernel[5];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 4] * c_Smoothing_Kernel[4];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 5] * c_Smoothing_Kernel[3];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 6] * c_Smoothing_Kernel[2];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 7] * c_Smoothing_Kernel[1];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 8] * c_Smoothing_Kernel[0];	

			Filter_Responses[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)] = sum;	
		}

		if ( (x < DATA_W) && (y < DATA_H) && ((z + 2) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 0] * c_Smoothing_Kernel[8];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 1] * c_Smoothing_Kernel[7];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 2] * c_Smoothing_Kernel[6];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 3] * c_Smoothing_Kernel[5];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 4] * c_Smoothing_Kernel[4];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 5] * c_Smoothing_Kernel[3];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 6] * c_Smoothing_Kernel[2];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 7] * c_Smoothing_Kernel[1];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 8] * c_Smoothing_Kernel[0];	
	
			Filter_Responses[Calculate_3D_Index(x,y,z + 2,DATA_W, DATA_H)] = sum;	
		}

		if ( (x < DATA_W) && (y < DATA_H) && ((z + 4) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 0] * c_Smoothing_Kernel[8];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 1] * c_Smoothing_Kernel[7];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 2] * c_Smoothing_Kernel[6];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 3] * c_Smoothing_Kernel[5];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 4] * c_Smoothing_Kernel[4];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 5] * c_Smoothing_Kernel[3];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 6] * c_Smoothing_Kernel[2];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 7] * c_Smoothing_Kernel[1];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 8] * c_Smoothing_Kernel[0];	
	
			Filter_Responses[Calculate_3D_Index(x,y,z + 4,DATA_W, DATA_H)] = sum;	
		}

		if ( (x < DATA_W) && (y < DATA_H) && ((z + 6) < DATA_D) )
		{
		    float sum = 0.0f;
	
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 0] * c_Smoothing_Kernel[8];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 1] * c_Smoothing_Kernel[7];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 2] * c_Smoothing_Kernel[6];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 3] * c_Smoothing_Kernel[5];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 4] * c_Smoothing_Kernel[4];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 5] * c_Smoothing_Kernel[3];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 6] * c_Smoothing_Kernel[2];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 7] * c_Smoothing_Kernel[1];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 8] * c_Smoothing_Kernel[0];	


			Filter_Responses[Calculate_3D_Index(x,y,z + 6,DATA_W, DATA_H)] = sum;	
		}

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && (z < DATA_D) )
		{
		    float sum = 0.0f;

			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 0] * c_Smoothing_Kernel[8];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 1] * c_Smoothing_Kernel[7];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 2] * c_Smoothing_Kernel[6];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 3] * c_Smoothing_Kernel[5];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 4] * c_Smoothing_Kernel[4];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 5] * c_Smoothing_Kernel[3];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 6] * c_Smoothing_Kernel[2];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 7] * c_Smoothing_Kernel[1];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 8] * c_Smoothing_Kernel[0];	


			Filter_Responses[Calculate_3D_Index(x,y + 8,z,DATA_W, DATA_H)] = sum;	
		}

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 2) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 0] * c_Smoothing_Kernel[8];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 1] * c_Smoothing_Kernel[7];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 2] * c_Smoothing_Kernel[6];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 3] * c_Smoothing_Kernel[5];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 4] * c_Smoothing_Kernel[4];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 5] * c_Smoothing_Kernel[3];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 6] * c_Smoothing_Kernel[2];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 7] * c_Smoothing_Kernel[1];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 8] * c_Smoothing_Kernel[0];	
	
			Filter_Responses[Calculate_3D_Index(x,y + 8,z + 2,DATA_W, DATA_H)] = sum;	
		}

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 4) < DATA_D) )
		{
		    float sum = 0.0f;
	
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 0] * c_Smoothing_Kernel[8];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 1] * c_Smoothing_Kernel[7];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 2] * c_Smoothing_Kernel[6];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 3] * c_Smoothing_Kernel[5];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 4] * c_Smoothing_Kernel[4];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 5] * c_Smoothing_Kernel[3];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 6] * c_Smoothing_Kernel[2];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 7] * c_Smoothing_Kernel[1];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 8] * c_Smoothing_Kernel[0];	

	
			Filter_Responses[Calculate_3D_Index(x,y + 8,z + 4,DATA_W, DATA_H)] = sum;	
		}

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 6) < DATA_D) )
		{
		    float sum = 0.0f;
	
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 0] * c_Smoothing_Kernel[8];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 1] * c_Smoothing_Kernel[7];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 2] * c_Smoothing_Kernel[6];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 3] * c_Smoothing_Kernel[5];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 4] * c_Smoothing_Kernel[4];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 5] * c_Smoothing_Kernel[3];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 6] * c_Smoothing_Kernel[2];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 7] * c_Smoothing_Kernel[1];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 8] * c_Smoothing_Kernel[0];	


			Filter_Responses[Calculate_3D_Index(x,y + 8,z + 6,DATA_W, DATA_H)] = sum;	
		}

	}
}

__global__ void convolutionRods(float *Filter_Responses, float* fMRI_Volume, float* Smoothed_Certainty, int t, int DATA_W, int DATA_H, int DATA_D, int blocksInY, float invBlocksInY, int xBlockDifference, int yBlockDifference, int zBlockDifference)
{   
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
	volatile int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	volatile int y = __umul24(blockIdxy ,blockDim.y) * 4 + threadIdx.y;
	volatile int z = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;	

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

		sum += s_Volume[threadIdx.z + 0][threadIdx.y][threadIdx.x] * c_Smoothing_Kernel[8];
		sum += s_Volume[threadIdx.z + 1][threadIdx.y][threadIdx.x] * c_Smoothing_Kernel[7];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x] * c_Smoothing_Kernel[6];
		sum += s_Volume[threadIdx.z + 3][threadIdx.y][threadIdx.x] * c_Smoothing_Kernel[5];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x] * c_Smoothing_Kernel[4];	
		sum += s_Volume[threadIdx.z + 5][threadIdx.y][threadIdx.x] * c_Smoothing_Kernel[3];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x] * c_Smoothing_Kernel[2];
		sum += s_Volume[threadIdx.z + 7][threadIdx.y][threadIdx.x] * c_Smoothing_Kernel[1];
		sum += s_Volume[threadIdx.z + 8][threadIdx.y][threadIdx.x] * c_Smoothing_Kernel[0];

		//Filter_Responses[Calculate_4D_Index(x,y,z,t,DATA_W,DATA_H,DATA_D)] = sum;
		Filter_Responses[Calculate_4D_Index(x,y,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 2) < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z + 0][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Kernel[8];
		sum += s_Volume[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Kernel[7];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Kernel[6];
		sum += s_Volume[threadIdx.z + 3][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Kernel[5];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Kernel[4];	
		sum += s_Volume[threadIdx.z + 5][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Kernel[3];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Kernel[2];
		sum += s_Volume[threadIdx.z + 7][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Kernel[1];
		sum += s_Volume[threadIdx.z + 8][threadIdx.y + 2][threadIdx.x] * c_Smoothing_Kernel[0];


		//Filter_Responses[Calculate_4D_Index(x,y + 2,z,t,DATA_W,DATA_H,DATA_D)] = sum;
		Filter_Responses[Calculate_4D_Index(x,y + 2,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate_3D_Index(x,y + 2,z,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;	

		sum += s_Volume[threadIdx.z + 0][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Kernel[8];
		sum += s_Volume[threadIdx.z + 1][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Kernel[7];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Kernel[6];
		sum += s_Volume[threadIdx.z + 3][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Kernel[5];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Kernel[4];	
		sum += s_Volume[threadIdx.z + 5][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Kernel[3];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Kernel[2];
		sum += s_Volume[threadIdx.z + 7][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Kernel[1];
		sum += s_Volume[threadIdx.z + 8][threadIdx.y + 4][threadIdx.x] * c_Smoothing_Kernel[0];

		//Filter_Responses[Calculate_4D_Index(x,y + 4,z,t,DATA_W,DATA_H,DATA_D)] = sum;
		Filter_Responses[Calculate_4D_Index(x,y + 4,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate_3D_Index(x,y + 4,z,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 6) < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += s_Volume[threadIdx.z + 0][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Kernel[8];
		sum += s_Volume[threadIdx.z + 1][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Kernel[7];
		sum += s_Volume[threadIdx.z + 2][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Kernel[6];
		sum += s_Volume[threadIdx.z + 3][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Kernel[5];
		sum += s_Volume[threadIdx.z + 4][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Kernel[4];	
		sum += s_Volume[threadIdx.z + 5][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Kernel[3];
		sum += s_Volume[threadIdx.z + 6][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Kernel[2];
		sum += s_Volume[threadIdx.z + 7][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Kernel[1];
		sum += s_Volume[threadIdx.z + 8][threadIdx.y + 6][threadIdx.x] * c_Smoothing_Kernel[0];

		//Filter_Responses[Calculate_4D_Index(x,y + 6,z,t,DATA_W,DATA_H,DATA_D)] = sum;	
		Filter_Responses[Calculate_4D_Index(x,y + 6,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate_3D_Index(x,y + 6,z,DATA_W, DATA_H)];
	}
}

__global__ void convolutionColumns2D(float *Filter_Responses, float* fMRI_Volume, float* Smoothed_Certainty, int t, int DATA_W, int DATA_H, int DATA_D, int blocksInY, float invBlocksInY, int xBlockDifference, int yBlockDifference, int zBlockDifference)
{   
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
	volatile int x = __umul24(blockIdx.x,blockDim.x) / 32 * 24 + threadIdx.x;
	volatile int y = __umul24(blockIdxy ,blockDim.y) * 2 + threadIdx.y;
	volatile int z = __umul24(blockIdxz ,blockDim.z) * 4 + threadIdx.z;	

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
		
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 0] * c_Smoothing_Kernel[8];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 1] * c_Smoothing_Kernel[7];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 2] * c_Smoothing_Kernel[6];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 3] * c_Smoothing_Kernel[5];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 4] * c_Smoothing_Kernel[4];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 5] * c_Smoothing_Kernel[3];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 6] * c_Smoothing_Kernel[2];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 7] * c_Smoothing_Kernel[1];
			sum += s_Volume[threadIdx.z][threadIdx.y][threadIdx.x + 8] * c_Smoothing_Kernel[0];	

			//Filter_Responses[Calculate_4D_Index(x,y,z,t,DATA_W, DATA_H,DATA_D)] = sum;
			Filter_Responses[Calculate_4D_Index(x,y,z,t,DATA_W, DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate_3D_Index(x,y,z,DATA_W, DATA_H)];
		}

		if ( (x < DATA_W) && (y < DATA_H) && ((z + 2) < DATA_D) )
		{
		    float sum = 0.0f;
		
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 0] * c_Smoothing_Kernel[8];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 1] * c_Smoothing_Kernel[7];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 2] * c_Smoothing_Kernel[6];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 3] * c_Smoothing_Kernel[5];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 4] * c_Smoothing_Kernel[4];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 5] * c_Smoothing_Kernel[3];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 6] * c_Smoothing_Kernel[2];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 7] * c_Smoothing_Kernel[1];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y][threadIdx.x + 8] * c_Smoothing_Kernel[0];	

			//Filter_Responses[Calculate_4D_Index(x,y,z + 2,t,DATA_W, DATA_H,DATA_D)] = sum;	
			Filter_Responses[Calculate_4D_Index(x,y,z + 2,t,DATA_W, DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate_3D_Index(x,y,z + 2,DATA_W, DATA_H)];
		}

		if ( (x < DATA_W) && (y < DATA_H) && ((z + 4) < DATA_D) )
		{
		    float sum = 0.0f;
		
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 0] * c_Smoothing_Kernel[8];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 1] * c_Smoothing_Kernel[7];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 2] * c_Smoothing_Kernel[6];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 3] * c_Smoothing_Kernel[5];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 4] * c_Smoothing_Kernel[4];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 5] * c_Smoothing_Kernel[3];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 6] * c_Smoothing_Kernel[2];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 7] * c_Smoothing_Kernel[1];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y][threadIdx.x + 8] * c_Smoothing_Kernel[0];	

			//Filter_Responses[Calculate_4D_Index(x,y,z + 4,t,DATA_W, DATA_H,DATA_D)] = sum;
			Filter_Responses[Calculate_4D_Index(x,y,z + 4,t,DATA_W, DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate_3D_Index(x,y,z + 4,DATA_W, DATA_H)];
		}

		if ( (x < DATA_W) && (y < DATA_H) && ((z + 6) < DATA_D) )
		{
		    float sum = 0.0f;
		
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 0] * c_Smoothing_Kernel[8];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 1] * c_Smoothing_Kernel[7];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 2] * c_Smoothing_Kernel[6];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 3] * c_Smoothing_Kernel[5];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 4] * c_Smoothing_Kernel[4];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 5] * c_Smoothing_Kernel[3];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 6] * c_Smoothing_Kernel[2];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 7] * c_Smoothing_Kernel[1];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y][threadIdx.x + 8] * c_Smoothing_Kernel[0];	

			//Filter_Responses[Calculate_4D_Index(x,y,z + 6,t,DATA_W, DATA_H,DATA_D)] = sum;
			Filter_Responses[Calculate_4D_Index(x,y,z + 6,t,DATA_W, DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate_3D_Index(x,y,z + 6,DATA_W, DATA_H)];
		}

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && (z < DATA_D) )
		{
		    float sum = 0.0f;
		
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 0] * c_Smoothing_Kernel[8];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 1] * c_Smoothing_Kernel[7];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 2] * c_Smoothing_Kernel[6];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 3] * c_Smoothing_Kernel[5];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 4] * c_Smoothing_Kernel[4];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 5] * c_Smoothing_Kernel[3];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 6] * c_Smoothing_Kernel[2];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 7] * c_Smoothing_Kernel[1];
			sum += s_Volume[threadIdx.z][threadIdx.y + 8][threadIdx.x + 8] * c_Smoothing_Kernel[0];	

			//Filter_Responses[Calculate_4D_Index(x,y + 8,z,t,DATA_W, DATA_H,DATA_D)] = sum;
			Filter_Responses[Calculate_4D_Index(x,y + 8,z,t,DATA_W, DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate_3D_Index(x,y + 8,z,DATA_W, DATA_H)];
		}

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 2) < DATA_D) )
		{
		    float sum = 0.0f;
		
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 0] * c_Smoothing_Kernel[8];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 1] * c_Smoothing_Kernel[7];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 2] * c_Smoothing_Kernel[6];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 3] * c_Smoothing_Kernel[5];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 4] * c_Smoothing_Kernel[4];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 5] * c_Smoothing_Kernel[3];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 6] * c_Smoothing_Kernel[2];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 7] * c_Smoothing_Kernel[1];
			sum += s_Volume[threadIdx.z + 2][threadIdx.y + 8][threadIdx.x + 8] * c_Smoothing_Kernel[0];	

			//Filter_Responses[Calculate_4D_Index(x,y + 8,z + 2,t,DATA_W, DATA_H,DATA_D)] = sum;
			Filter_Responses[Calculate_4D_Index(x,y + 8,z + 2,t,DATA_W, DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate_3D_Index(x,y + 8,z + 2,DATA_W, DATA_H)];
		}

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 4) < DATA_D) )
		{
		    float sum = 0.0f;
		
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 0] * c_Smoothing_Kernel[8];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 1] * c_Smoothing_Kernel[7];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 2] * c_Smoothing_Kernel[6];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 3] * c_Smoothing_Kernel[5];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 4] * c_Smoothing_Kernel[4];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 5] * c_Smoothing_Kernel[3];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 6] * c_Smoothing_Kernel[2];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 7] * c_Smoothing_Kernel[1];
			sum += s_Volume[threadIdx.z + 4][threadIdx.y + 8][threadIdx.x + 8] * c_Smoothing_Kernel[0];	

			//Filter_Responses[Calculate_4D_Index(x,y + 8,z + 4,t,DATA_W, DATA_H,DATA_D)] = sum;
			Filter_Responses[Calculate_4D_Index(x,y + 8,z + 4,t,DATA_W, DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate_3D_Index(x,y + 8,z + 4,DATA_W, DATA_H)];
		}

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 6) < DATA_D) )
		{
		    float sum = 0.0f;
	
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 0] * c_Smoothing_Kernel[8];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 1] * c_Smoothing_Kernel[7];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 2] * c_Smoothing_Kernel[6];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 3] * c_Smoothing_Kernel[5];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 4] * c_Smoothing_Kernel[4];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 5] * c_Smoothing_Kernel[3];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 6] * c_Smoothing_Kernel[2];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 7] * c_Smoothing_Kernel[1];
			sum += s_Volume[threadIdx.z + 6][threadIdx.y + 8][threadIdx.x + 8] * c_Smoothing_Kernel[0];	

			//Filter_Responses[Calculate_4D_Index(x,y + 8,z + 6,t,DATA_W, DATA_H,DATA_D)] = sum;
			Filter_Responses[Calculate_4D_Index(x,y + 8,z + 6,t,DATA_W, DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate_3D_Index(x,y + 8,z + 6,DATA_W, DATA_H)];
		}
	}
}


// Detrending

__device__ __constant__ float c_X_Detrend[1000];
__device__ __constant__ float c_xtxxt_Detrend[1000];

__global__ void DetrendCubic(float* detrended_fMRI_data, float* fMRI_Volumes, float* Brain_Voxels, int DATA_W, int DATA_H, int DATA_D, int DATA_T, int blocksInY, float invBlocksInY, int timeMultiples, int timeRest)
{
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
	volatile unsigned int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	volatile unsigned int y = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
	volatile unsigned int z = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;
	
	if ((x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D))
		return;

	//if ( Brain_Voxels[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
	//{
		__shared__ float s_Y[8][8][32]; // y,t,x
	
		int t_offset;
		float beta1 = 0.0f;
		float beta2 = 0.0f;
		float beta3 = 0.0f;
		float beta4 = 0.0f;


		// Calculate betahat, i.e. multiply (x^T x)^(-1) x^T with Y
		for (t_offset = 0; t_offset < timeMultiples * 8; t_offset += 8)
		{ 
			// Load the current voxel timeseries into shared memory 
			s_Y[threadIdx.y][0][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 0, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][1][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 1, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][2][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 2, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][3][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 3, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][4][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 4, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][5][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 5, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][6][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 6, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][7][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 7, DATA_W, DATA_H, DATA_D)];
	
			__syncthreads();
		
			// Sum and multiply the values in shared memory
			beta1 += s_Y[threadIdx.y][0][threadIdx.x] * c_xtxxt_Detrend[t_offset + 0];
			beta1 += s_Y[threadIdx.y][1][threadIdx.x] * c_xtxxt_Detrend[t_offset + 1];
			beta1 += s_Y[threadIdx.y][2][threadIdx.x] * c_xtxxt_Detrend[t_offset + 2];
			beta1 += s_Y[threadIdx.y][3][threadIdx.x] * c_xtxxt_Detrend[t_offset + 3];
			beta1 += s_Y[threadIdx.y][4][threadIdx.x] * c_xtxxt_Detrend[t_offset + 4];
			beta1 += s_Y[threadIdx.y][5][threadIdx.x] * c_xtxxt_Detrend[t_offset + 5];
			beta1 += s_Y[threadIdx.y][6][threadIdx.x] * c_xtxxt_Detrend[t_offset + 6];
			beta1 += s_Y[threadIdx.y][7][threadIdx.x] * c_xtxxt_Detrend[t_offset + 7];
	
			beta2 += s_Y[threadIdx.y][0][threadIdx.x] * c_xtxxt_Detrend[DATA_T + t_offset + 0];
			beta2 += s_Y[threadIdx.y][1][threadIdx.x] * c_xtxxt_Detrend[DATA_T + t_offset + 1];
			beta2 += s_Y[threadIdx.y][2][threadIdx.x] * c_xtxxt_Detrend[DATA_T + t_offset + 2];
			beta2 += s_Y[threadIdx.y][3][threadIdx.x] * c_xtxxt_Detrend[DATA_T + t_offset + 3];
			beta2 += s_Y[threadIdx.y][4][threadIdx.x] * c_xtxxt_Detrend[DATA_T + t_offset + 4];
			beta2 += s_Y[threadIdx.y][5][threadIdx.x] * c_xtxxt_Detrend[DATA_T + t_offset + 5];
			beta2 += s_Y[threadIdx.y][6][threadIdx.x] * c_xtxxt_Detrend[DATA_T + t_offset + 6];
			beta2 += s_Y[threadIdx.y][7][threadIdx.x] * c_xtxxt_Detrend[DATA_T + t_offset + 7];

			beta3 += s_Y[threadIdx.y][0][threadIdx.x] * c_xtxxt_Detrend[2 * DATA_T + t_offset + 0];
			beta3 += s_Y[threadIdx.y][1][threadIdx.x] * c_xtxxt_Detrend[2 * DATA_T + t_offset + 1];
			beta3 += s_Y[threadIdx.y][2][threadIdx.x] * c_xtxxt_Detrend[2 * DATA_T + t_offset + 2];
			beta3 += s_Y[threadIdx.y][3][threadIdx.x] * c_xtxxt_Detrend[2 * DATA_T + t_offset + 3];
			beta3 += s_Y[threadIdx.y][4][threadIdx.x] * c_xtxxt_Detrend[2 * DATA_T + t_offset + 4];
			beta3 += s_Y[threadIdx.y][5][threadIdx.x] * c_xtxxt_Detrend[2 * DATA_T + t_offset + 5];
			beta3 += s_Y[threadIdx.y][6][threadIdx.x] * c_xtxxt_Detrend[2 * DATA_T + t_offset + 6];
			beta3 += s_Y[threadIdx.y][7][threadIdx.x] * c_xtxxt_Detrend[2 * DATA_T + t_offset + 7];

			beta4 += s_Y[threadIdx.y][0][threadIdx.x] * c_xtxxt_Detrend[3 * DATA_T + t_offset + 0];
			beta4 += s_Y[threadIdx.y][1][threadIdx.x] * c_xtxxt_Detrend[3 * DATA_T + t_offset + 1];
			beta4 += s_Y[threadIdx.y][2][threadIdx.x] * c_xtxxt_Detrend[3 * DATA_T + t_offset + 2];
			beta4 += s_Y[threadIdx.y][3][threadIdx.x] * c_xtxxt_Detrend[3 * DATA_T + t_offset + 3];
			beta4 += s_Y[threadIdx.y][4][threadIdx.x] * c_xtxxt_Detrend[3 * DATA_T + t_offset + 4];
			beta4 += s_Y[threadIdx.y][5][threadIdx.x] * c_xtxxt_Detrend[3 * DATA_T + t_offset + 5];
			beta4 += s_Y[threadIdx.y][6][threadIdx.x] * c_xtxxt_Detrend[3 * DATA_T + t_offset + 6];
			beta4 += s_Y[threadIdx.y][7][threadIdx.x] * c_xtxxt_Detrend[3 * DATA_T + t_offset + 7];
		}
	
		t_offset = timeMultiples * 8;	

		for (int t = 0; t < timeRest; t++)
		{
			s_Y[threadIdx.y][0][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + t, DATA_W, DATA_H, DATA_D)];
			beta1 += s_Y[threadIdx.y][0][threadIdx.x] * c_xtxxt_Detrend[t_offset + t];
			beta2 += s_Y[threadIdx.y][0][threadIdx.x] * c_xtxxt_Detrend[DATA_T + t_offset + t];
			beta3 += s_Y[threadIdx.y][0][threadIdx.x] * c_xtxxt_Detrend[2 * DATA_T + t_offset + t];
			beta4 += s_Y[threadIdx.y][0][threadIdx.x] * c_xtxxt_Detrend[3 * DATA_T + t_offset + t];
		}
	
		for (t_offset = 0; t_offset < timeMultiples * 8; t_offset += 8)
		{ 
			// Load the current voxel timeseries into shared memory 
			s_Y[threadIdx.y][0][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 0, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][1][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 1, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][2][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 2, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][3][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 3, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][4][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 4, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][5][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 5, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][6][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 6, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][7][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 7, DATA_W, DATA_H, DATA_D)];
	
			__syncthreads();
		
			// Calculate eps for each timesample and store in shared memory (timesample not needed in shared memory any more)	
			s_Y[threadIdx.y][0][threadIdx.x] = s_Y[threadIdx.y][0][threadIdx.x] - beta1 * c_X_Detrend[t_offset + 0] - beta2 * c_X_Detrend[DATA_T + t_offset + 0] - beta3 * c_X_Detrend[2 * DATA_T + t_offset + 0] - beta4 * c_X_Detrend[3 * DATA_T + t_offset + 0];			
			s_Y[threadIdx.y][1][threadIdx.x] = s_Y[threadIdx.y][1][threadIdx.x] - beta1 * c_X_Detrend[t_offset + 1] - beta2 * c_X_Detrend[DATA_T + t_offset + 1] - beta3 * c_X_Detrend[2 * DATA_T + t_offset + 1] - beta4 * c_X_Detrend[3 * DATA_T + t_offset + 1];			
			s_Y[threadIdx.y][2][threadIdx.x] = s_Y[threadIdx.y][2][threadIdx.x] - beta1 * c_X_Detrend[t_offset + 2] - beta2 * c_X_Detrend[DATA_T + t_offset + 2] - beta3 * c_X_Detrend[2 * DATA_T + t_offset + 2] - beta4 * c_X_Detrend[3 * DATA_T + t_offset + 2];			
			s_Y[threadIdx.y][3][threadIdx.x] = s_Y[threadIdx.y][3][threadIdx.x] - beta1 * c_X_Detrend[t_offset + 3] - beta2 * c_X_Detrend[DATA_T + t_offset + 3] - beta3 * c_X_Detrend[2 * DATA_T + t_offset + 3] - beta4 * c_X_Detrend[3 * DATA_T + t_offset + 3];			
			s_Y[threadIdx.y][4][threadIdx.x] = s_Y[threadIdx.y][4][threadIdx.x] - beta1 * c_X_Detrend[t_offset + 4] - beta2 * c_X_Detrend[DATA_T + t_offset + 4] - beta3 * c_X_Detrend[2 * DATA_T + t_offset + 4] - beta4 * c_X_Detrend[3 * DATA_T + t_offset + 4];			
			s_Y[threadIdx.y][5][threadIdx.x] = s_Y[threadIdx.y][5][threadIdx.x] - beta1 * c_X_Detrend[t_offset + 5] - beta2 * c_X_Detrend[DATA_T + t_offset + 5] - beta3 * c_X_Detrend[2 * DATA_T + t_offset + 5] - beta4 * c_X_Detrend[3 * DATA_T + t_offset + 5];			
			s_Y[threadIdx.y][6][threadIdx.x] = s_Y[threadIdx.y][6][threadIdx.x] - beta1 * c_X_Detrend[t_offset + 6] - beta2 * c_X_Detrend[DATA_T + t_offset + 6] - beta3 * c_X_Detrend[2 * DATA_T + t_offset + 6] - beta4 * c_X_Detrend[3 * DATA_T + t_offset + 6];			
			s_Y[threadIdx.y][7][threadIdx.x] = s_Y[threadIdx.y][7][threadIdx.x] - beta1 * c_X_Detrend[t_offset + 7] - beta2 * c_X_Detrend[DATA_T + t_offset + 7] - beta3 * c_X_Detrend[2 * DATA_T + t_offset + 7] - beta4 * c_X_Detrend[3 * DATA_T + t_offset + 7];			


			__syncthreads();

			// Write the eps back global memory
			detrended_fMRI_data[Calculate_4D_Index(x, y, z, t_offset + 0, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][0][threadIdx.x];
			detrended_fMRI_data[Calculate_4D_Index(x, y, z, t_offset + 1, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][1][threadIdx.x];
			detrended_fMRI_data[Calculate_4D_Index(x, y, z, t_offset + 2, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][2][threadIdx.x];
			detrended_fMRI_data[Calculate_4D_Index(x, y, z, t_offset + 3, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][3][threadIdx.x];
			detrended_fMRI_data[Calculate_4D_Index(x, y, z, t_offset + 4, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][4][threadIdx.x];
			detrended_fMRI_data[Calculate_4D_Index(x, y, z, t_offset + 5, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][5][threadIdx.x];
			detrended_fMRI_data[Calculate_4D_Index(x, y, z, t_offset + 6, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][6][threadIdx.x];
			detrended_fMRI_data[Calculate_4D_Index(x, y, z, t_offset + 7, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][7][threadIdx.x];
		}
	
		t_offset = timeMultiples * 8;	
	
		for (int t = 0; t < timeRest; t++)
		{
			s_Y[threadIdx.y][0][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + t, DATA_W, DATA_H, DATA_D)];
			detrended_fMRI_data[Calculate_4D_Index(x, y, z, t_offset + t, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][0][threadIdx.x] - beta1 * c_X_Detrend[t_offset + t] - beta2 * c_X_Detrend[DATA_T + t_offset + t] - beta3 * c_X_Detrend[2 * DATA_T + t_offset + t] - beta4 * c_X_Detrend[3 * DATA_T + t_offset + t];			
		}
	//}
}


// Statistical analysis CCA

__device__ __constant__ float c_Y[1000];
__device__ __constant__ float c_Cyy[4];
__device__ __constant__ float c_sqrt_inv_Cyy[4];

__device__ float min3(float values[4])
{
	if ( (values[0] < values[1]) && (values[0] < values[2]) )
		return values[0];
	else if ( (values[1] < values[0]) && (values[1] < values[2]) )
		return values[1];
	else
		return values[2];
}

__device__ float max3(float values[4])
{
	if ( (values[0] > values[1]) && (values[0] > values[2]) )
		return values[0];
	else if ( (values[1] > values[0]) && (values[1] > values[2]) )
		return values[1];
	else
		return values[2];
}

__device__ float sign(float s)
{
	if (s < 0.0f)
		return -1.0f;
	else
		return 1.0f;
}

// 2D

__device__ float Determinant(float *Cxx)
{
    return Cxx[3] * Cxx[5] * Cxx[5] * Cxx[3] - Cxx[2] * Cxx[6] * Cxx[5] * Cxx[3] - Cxx[3] * Cxx[4] * Cxx[7] * Cxx[3] 
         + Cxx[1] * Cxx[6] * Cxx[7] * Cxx[3] + Cxx[2] * Cxx[4] * Cxx[8] * Cxx[3] - Cxx[1] * Cxx[5] * Cxx[8] * Cxx[3]
         - Cxx[3] * Cxx[5] * Cxx[2] * Cxx[6] + Cxx[2] * Cxx[6] * Cxx[2] * Cxx[6] + Cxx[3] * Cxx[1] * Cxx[7] * Cxx[6]
         - Cxx[0] * Cxx[6] * Cxx[7] * Cxx[6] - Cxx[2] * Cxx[1] * Cxx[8] * Cxx[6] + Cxx[0] * Cxx[5] * Cxx[8] * Cxx[6]
         + Cxx[3] * Cxx[4] * Cxx[2] * Cxx[8] - Cxx[1] * Cxx[6] * Cxx[2] * Cxx[8] - Cxx[3] * Cxx[1] * Cxx[5] * Cxx[8]
         + Cxx[0] * Cxx[6] * Cxx[5] * Cxx[8] + Cxx[1] * Cxx[1] * Cxx[8] * Cxx[8] - Cxx[0] * Cxx[4] * Cxx[8] * Cxx[8]
         - Cxx[2] * Cxx[4] * Cxx[2] * Cxx[9] + Cxx[1] * Cxx[5] * Cxx[2] * Cxx[9] + Cxx[2] * Cxx[1] * Cxx[5] * Cxx[9]
		 - Cxx[0] * Cxx[5] * Cxx[5] * Cxx[9] - Cxx[1] * Cxx[1] * Cxx[7] * Cxx[9] + Cxx[0] * Cxx[4] * Cxx[7] * Cxx[9];
}

__device__ void Invert_Cxx(float *Cxx, float *inv_Cxx)
{
	float determinant = Determinant(Cxx);

	inv_Cxx[0] = Cxx[5]*Cxx[8]*Cxx[6] - Cxx[6]*Cxx[7]*Cxx[6] + Cxx[6]*Cxx[5]*Cxx[8] - Cxx[4]*Cxx[8]*Cxx[8] - Cxx[5]*Cxx[5]*Cxx[9] + Cxx[4]*Cxx[7]*Cxx[9]; 
	inv_Cxx[1] = Cxx[6]*Cxx[7]*Cxx[3] - Cxx[5]*Cxx[8]*Cxx[3] - Cxx[6]*Cxx[2]*Cxx[8] + Cxx[1]*Cxx[8]*Cxx[8] + Cxx[5]*Cxx[2]*Cxx[9] - Cxx[1]*Cxx[7]*Cxx[9];
	inv_Cxx[2] = Cxx[4]*Cxx[8]*Cxx[3] - Cxx[6]*Cxx[5]*Cxx[3] + Cxx[6]*Cxx[2]*Cxx[6] - Cxx[1]*Cxx[8]*Cxx[6] - Cxx[4]*Cxx[2]*Cxx[9] + Cxx[1]*Cxx[5]*Cxx[9];
	inv_Cxx[3] = Cxx[5]*Cxx[5]*Cxx[3] - Cxx[4]*Cxx[7]*Cxx[3] - Cxx[5]*Cxx[2]*Cxx[6] + Cxx[1]*Cxx[7]*Cxx[6] + Cxx[4]*Cxx[2]*Cxx[8] - Cxx[1]*Cxx[5]*Cxx[8];
	inv_Cxx[4] = Cxx[2]*Cxx[8]*Cxx[3] - Cxx[3]*Cxx[7]*Cxx[3] + Cxx[3]*Cxx[2]*Cxx[8] - Cxx[0]*Cxx[8]*Cxx[8] - Cxx[2]*Cxx[2]*Cxx[9] + Cxx[0]*Cxx[7]*Cxx[9];
	inv_Cxx[5] = Cxx[3]*Cxx[5]*Cxx[3] - Cxx[1]*Cxx[8]*Cxx[3] - Cxx[3]*Cxx[2]*Cxx[6] + Cxx[0]*Cxx[8]*Cxx[6] + Cxx[1]*Cxx[2]*Cxx[9] - Cxx[0]*Cxx[5]*Cxx[9];
	inv_Cxx[6] = Cxx[1]*Cxx[7]*Cxx[3] - Cxx[2]*Cxx[5]*Cxx[3] + Cxx[2]*Cxx[2]*Cxx[6] - Cxx[0]*Cxx[7]*Cxx[6] - Cxx[1]*Cxx[2]*Cxx[8] + Cxx[0]*Cxx[5]*Cxx[8];
	inv_Cxx[7] = Cxx[1]*Cxx[6]*Cxx[3] - Cxx[3]*Cxx[4]*Cxx[3] + Cxx[3]*Cxx[1]*Cxx[6] - Cxx[0]*Cxx[6]*Cxx[6] - Cxx[1]*Cxx[1]*Cxx[9] + Cxx[0]*Cxx[4]*Cxx[9];
	inv_Cxx[8] = Cxx[2]*Cxx[4]*Cxx[3] - Cxx[1]*Cxx[5]*Cxx[3] - Cxx[2]*Cxx[1]*Cxx[6] + Cxx[0]*Cxx[5]*Cxx[6] + Cxx[1]*Cxx[1]*Cxx[8] - Cxx[0]*Cxx[4]*Cxx[8];
	inv_Cxx[9] = Cxx[1]*Cxx[5]*Cxx[2] - Cxx[2]*Cxx[4]*Cxx[2] + Cxx[2]*Cxx[1]*Cxx[5] - Cxx[0]*Cxx[5]*Cxx[5] - Cxx[1]*Cxx[1]*Cxx[7] + Cxx[0]*Cxx[4]*Cxx[7];


	inv_Cxx[0] /= determinant;
	inv_Cxx[1] /= determinant;
	inv_Cxx[2] /= determinant;
	inv_Cxx[3] /= determinant;
	inv_Cxx[4] /= determinant;
	inv_Cxx[5] /= determinant;
	inv_Cxx[6] /= determinant;
	inv_Cxx[7] /= determinant;
	inv_Cxx[8] /= determinant;
	inv_Cxx[9] /= determinant;
}

__global__ void CalculateActivityMapCCA2D(float* activity_map, float* filter_responses_1, float* filter_responses_2, float* filter_responses_3, float* filter_responses_4, float* Brain_Voxels, int DATA_W, int DATA_H, int DATA_D, int DATA_T, int blocksInY, float invBlocksInY, int timeMultiples, int timeRest)
{	
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
	volatile unsigned int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	volatile unsigned int y = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
	volatile unsigned int z = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Brain_Voxels[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
	{
		__shared__ float s_X[4][4][4][32]; // filter responses,y,t,x

		float Cxx[10];
		float inv_Cxx[10];
		float Cxy[4][2];
		float matrix_product[3];
		float temporal_weights[2];
		float spatial_weights[4];	

		Cxx[0] = 0.0f; Cxx[1] = 0.0f; Cxx[2] = 0.0f; Cxx[3] = 0.0f; Cxx[4] = 0.0f; Cxx[5] = 0.0f; Cxx[6] = 0.0f; Cxx[7] = 0.0f; Cxx[8] = 0.0f; Cxx[9] = 0.0f; 

		Cxy[0][0] = 0.0f; Cxy[0][1] = 0.0f;		
		Cxy[1][0] = 0.0f; Cxy[1][1] = 0.0f;		
		Cxy[2][0] = 0.0f; Cxy[2][1] = 0.0f;		
		Cxy[3][0] = 0.0f; Cxy[3][1] = 0.0f;		

		int t_offset = 0;
		for (t_offset = 0; t_offset < timeMultiples * 4; t_offset += 4)
		{ 
			// Read the four filter responses into shared memory
			
			s_X[0][threadIdx.y][0][threadIdx.x] = filter_responses_1[Calculate_4D_Index(x,y,z,t_offset + 0, DATA_W, DATA_H, DATA_D)];
			s_X[1][threadIdx.y][0][threadIdx.x] = filter_responses_2[Calculate_4D_Index(x,y,z,t_offset + 0, DATA_W, DATA_H, DATA_D)];
			s_X[2][threadIdx.y][0][threadIdx.x] = filter_responses_3[Calculate_4D_Index(x,y,z,t_offset + 0, DATA_W, DATA_H, DATA_D)];
			s_X[3][threadIdx.y][0][threadIdx.x] = filter_responses_4[Calculate_4D_Index(x,y,z,t_offset + 0, DATA_W, DATA_H, DATA_D)];

			s_X[0][threadIdx.y][1][threadIdx.x] = filter_responses_1[Calculate_4D_Index(x,y,z,t_offset + 1, DATA_W, DATA_H, DATA_D)];
			s_X[1][threadIdx.y][1][threadIdx.x] = filter_responses_2[Calculate_4D_Index(x,y,z,t_offset + 1, DATA_W, DATA_H, DATA_D)];
			s_X[2][threadIdx.y][1][threadIdx.x] = filter_responses_3[Calculate_4D_Index(x,y,z,t_offset + 1, DATA_W, DATA_H, DATA_D)];
			s_X[3][threadIdx.y][1][threadIdx.x] = filter_responses_4[Calculate_4D_Index(x,y,z,t_offset + 1, DATA_W, DATA_H, DATA_D)];

			s_X[0][threadIdx.y][2][threadIdx.x] = filter_responses_1[Calculate_4D_Index(x,y,z,t_offset + 2, DATA_W, DATA_H, DATA_D)];
			s_X[1][threadIdx.y][2][threadIdx.x] = filter_responses_2[Calculate_4D_Index(x,y,z,t_offset + 2, DATA_W, DATA_H, DATA_D)];
			s_X[2][threadIdx.y][2][threadIdx.x] = filter_responses_3[Calculate_4D_Index(x,y,z,t_offset + 2, DATA_W, DATA_H, DATA_D)];
			s_X[3][threadIdx.y][2][threadIdx.x] = filter_responses_4[Calculate_4D_Index(x,y,z,t_offset + 2, DATA_W, DATA_H, DATA_D)];

			s_X[0][threadIdx.y][3][threadIdx.x] = filter_responses_1[Calculate_4D_Index(x,y,z,t_offset + 3, DATA_W, DATA_H, DATA_D)];
			s_X[1][threadIdx.y][3][threadIdx.x] = filter_responses_2[Calculate_4D_Index(x,y,z,t_offset + 3, DATA_W, DATA_H, DATA_D)];
			s_X[2][threadIdx.y][3][threadIdx.x] = filter_responses_3[Calculate_4D_Index(x,y,z,t_offset + 3, DATA_W, DATA_H, DATA_D)];
			s_X[3][threadIdx.y][3][threadIdx.x] = filter_responses_4[Calculate_4D_Index(x,y,z,t_offset + 3, DATA_W, DATA_H, DATA_D)];


			__syncthreads();

			// Calculate Cxx
			Cxx[0] += s_X[0][threadIdx.y][0][threadIdx.x] * s_X[0][threadIdx.y][0][threadIdx.x];						
			Cxx[1] += s_X[1][threadIdx.y][0][threadIdx.x] * s_X[0][threadIdx.y][0][threadIdx.x];						
			Cxx[2] += s_X[2][threadIdx.y][0][threadIdx.x] * s_X[0][threadIdx.y][0][threadIdx.x];						
			Cxx[3] += s_X[3][threadIdx.y][0][threadIdx.x] * s_X[0][threadIdx.y][0][threadIdx.x];						
			Cxx[4] += s_X[1][threadIdx.y][0][threadIdx.x] * s_X[1][threadIdx.y][0][threadIdx.x];						
			Cxx[5] += s_X[2][threadIdx.y][0][threadIdx.x] * s_X[1][threadIdx.y][0][threadIdx.x];						
			Cxx[6] += s_X[3][threadIdx.y][0][threadIdx.x] * s_X[1][threadIdx.y][0][threadIdx.x];						
			Cxx[7] += s_X[2][threadIdx.y][0][threadIdx.x] * s_X[2][threadIdx.y][0][threadIdx.x];						
			Cxx[8] += s_X[3][threadIdx.y][0][threadIdx.x] * s_X[2][threadIdx.y][0][threadIdx.x];						
			Cxx[9] += s_X[3][threadIdx.y][0][threadIdx.x] * s_X[3][threadIdx.y][0][threadIdx.x];		

			Cxx[0] += s_X[0][threadIdx.y][1][threadIdx.x] * s_X[0][threadIdx.y][1][threadIdx.x];						
			Cxx[1] += s_X[1][threadIdx.y][1][threadIdx.x] * s_X[0][threadIdx.y][1][threadIdx.x];						
			Cxx[2] += s_X[2][threadIdx.y][1][threadIdx.x] * s_X[0][threadIdx.y][1][threadIdx.x];						
			Cxx[3] += s_X[3][threadIdx.y][1][threadIdx.x] * s_X[0][threadIdx.y][1][threadIdx.x];						
			Cxx[4] += s_X[1][threadIdx.y][1][threadIdx.x] * s_X[1][threadIdx.y][1][threadIdx.x];						
			Cxx[5] += s_X[2][threadIdx.y][1][threadIdx.x] * s_X[1][threadIdx.y][1][threadIdx.x];						
			Cxx[6] += s_X[3][threadIdx.y][1][threadIdx.x] * s_X[1][threadIdx.y][1][threadIdx.x];						
			Cxx[7] += s_X[2][threadIdx.y][1][threadIdx.x] * s_X[2][threadIdx.y][1][threadIdx.x];						
			Cxx[8] += s_X[3][threadIdx.y][1][threadIdx.x] * s_X[2][threadIdx.y][1][threadIdx.x];						
			Cxx[9] += s_X[3][threadIdx.y][1][threadIdx.x] * s_X[3][threadIdx.y][1][threadIdx.x];		

			Cxx[0] += s_X[0][threadIdx.y][2][threadIdx.x] * s_X[0][threadIdx.y][2][threadIdx.x];						
			Cxx[1] += s_X[1][threadIdx.y][2][threadIdx.x] * s_X[0][threadIdx.y][2][threadIdx.x];						
			Cxx[2] += s_X[2][threadIdx.y][2][threadIdx.x] * s_X[0][threadIdx.y][2][threadIdx.x];						
			Cxx[3] += s_X[3][threadIdx.y][2][threadIdx.x] * s_X[0][threadIdx.y][2][threadIdx.x];						
			Cxx[4] += s_X[1][threadIdx.y][2][threadIdx.x] * s_X[1][threadIdx.y][2][threadIdx.x];						
			Cxx[5] += s_X[2][threadIdx.y][2][threadIdx.x] * s_X[1][threadIdx.y][2][threadIdx.x];						
			Cxx[6] += s_X[3][threadIdx.y][2][threadIdx.x] * s_X[1][threadIdx.y][2][threadIdx.x];						
			Cxx[7] += s_X[2][threadIdx.y][2][threadIdx.x] * s_X[2][threadIdx.y][2][threadIdx.x];						
			Cxx[8] += s_X[3][threadIdx.y][2][threadIdx.x] * s_X[2][threadIdx.y][2][threadIdx.x];						
			Cxx[9] += s_X[3][threadIdx.y][2][threadIdx.x] * s_X[3][threadIdx.y][2][threadIdx.x];		

			Cxx[0] += s_X[0][threadIdx.y][3][threadIdx.x] * s_X[0][threadIdx.y][3][threadIdx.x];						
			Cxx[1] += s_X[1][threadIdx.y][3][threadIdx.x] * s_X[0][threadIdx.y][3][threadIdx.x];						
			Cxx[2] += s_X[2][threadIdx.y][3][threadIdx.x] * s_X[0][threadIdx.y][3][threadIdx.x];						
			Cxx[3] += s_X[3][threadIdx.y][3][threadIdx.x] * s_X[0][threadIdx.y][3][threadIdx.x];						
			Cxx[4] += s_X[1][threadIdx.y][3][threadIdx.x] * s_X[1][threadIdx.y][3][threadIdx.x];						
			Cxx[5] += s_X[2][threadIdx.y][3][threadIdx.x] * s_X[1][threadIdx.y][3][threadIdx.x];						
			Cxx[6] += s_X[3][threadIdx.y][3][threadIdx.x] * s_X[1][threadIdx.y][3][threadIdx.x];						
			Cxx[7] += s_X[2][threadIdx.y][3][threadIdx.x] * s_X[2][threadIdx.y][3][threadIdx.x];						
			Cxx[8] += s_X[3][threadIdx.y][3][threadIdx.x] * s_X[2][threadIdx.y][3][threadIdx.x];						
			Cxx[9] += s_X[3][threadIdx.y][3][threadIdx.x] * s_X[3][threadIdx.y][3][threadIdx.x];		
			
			// Calculate Cxy		
			Cxy[0][0] += s_X[0][threadIdx.y][0][threadIdx.x] * c_Y[t_offset + 0 + 0 * DATA_T];
			Cxy[1][0] += s_X[1][threadIdx.y][0][threadIdx.x] * c_Y[t_offset + 0 + 0 * DATA_T];
			Cxy[2][0] += s_X[2][threadIdx.y][0][threadIdx.x] * c_Y[t_offset + 0 + 0 * DATA_T];
			Cxy[3][0] += s_X[3][threadIdx.y][0][threadIdx.x] * c_Y[t_offset + 0 + 0 * DATA_T];
	
			Cxy[0][0] += s_X[0][threadIdx.y][1][threadIdx.x] * c_Y[t_offset + 1 + 0 * DATA_T];
			Cxy[1][0] += s_X[1][threadIdx.y][1][threadIdx.x] * c_Y[t_offset + 1 + 0 * DATA_T];
			Cxy[2][0] += s_X[2][threadIdx.y][1][threadIdx.x] * c_Y[t_offset + 1 + 0 * DATA_T];
			Cxy[3][0] += s_X[3][threadIdx.y][1][threadIdx.x] * c_Y[t_offset + 1 + 0 * DATA_T];

			Cxy[0][0] += s_X[0][threadIdx.y][2][threadIdx.x] * c_Y[t_offset + 2 + 0 * DATA_T];
			Cxy[1][0] += s_X[1][threadIdx.y][2][threadIdx.x] * c_Y[t_offset + 2 + 0 * DATA_T];
			Cxy[2][0] += s_X[2][threadIdx.y][2][threadIdx.x] * c_Y[t_offset + 2 + 0 * DATA_T];
			Cxy[3][0] += s_X[3][threadIdx.y][2][threadIdx.x] * c_Y[t_offset + 2 + 0 * DATA_T];

			Cxy[0][0] += s_X[0][threadIdx.y][3][threadIdx.x] * c_Y[t_offset + 3 + 0 * DATA_T];
			Cxy[1][0] += s_X[1][threadIdx.y][3][threadIdx.x] * c_Y[t_offset + 3 + 0 * DATA_T];
			Cxy[2][0] += s_X[2][threadIdx.y][3][threadIdx.x] * c_Y[t_offset + 3 + 0 * DATA_T];
			Cxy[3][0] += s_X[3][threadIdx.y][3][threadIdx.x] * c_Y[t_offset + 3 + 0 * DATA_T];

			Cxy[0][1] += s_X[0][threadIdx.y][0][threadIdx.x] * c_Y[t_offset + 0 + 1 * DATA_T];
			Cxy[1][1] += s_X[1][threadIdx.y][0][threadIdx.x] * c_Y[t_offset + 0 + 1 * DATA_T];
			Cxy[2][1] += s_X[2][threadIdx.y][0][threadIdx.x] * c_Y[t_offset + 0 + 1 * DATA_T];
			Cxy[3][1] += s_X[3][threadIdx.y][0][threadIdx.x] * c_Y[t_offset + 0 + 1 * DATA_T];

			Cxy[0][1] += s_X[0][threadIdx.y][1][threadIdx.x] * c_Y[t_offset + 1 + 1 * DATA_T];
			Cxy[1][1] += s_X[1][threadIdx.y][1][threadIdx.x] * c_Y[t_offset + 1 + 1 * DATA_T];
			Cxy[2][1] += s_X[2][threadIdx.y][1][threadIdx.x] * c_Y[t_offset + 1 + 1 * DATA_T];
			Cxy[3][1] += s_X[3][threadIdx.y][1][threadIdx.x] * c_Y[t_offset + 1 + 1 * DATA_T];

			Cxy[0][1] += s_X[0][threadIdx.y][2][threadIdx.x] * c_Y[t_offset + 2 + 1 * DATA_T];
			Cxy[1][1] += s_X[1][threadIdx.y][2][threadIdx.x] * c_Y[t_offset + 2 + 1 * DATA_T];
			Cxy[2][1] += s_X[2][threadIdx.y][2][threadIdx.x] * c_Y[t_offset + 2 + 1 * DATA_T];
			Cxy[3][1] += s_X[3][threadIdx.y][2][threadIdx.x] * c_Y[t_offset + 2 + 1 * DATA_T];

			Cxy[0][1] += s_X[0][threadIdx.y][3][threadIdx.x] * c_Y[t_offset + 3 + 1 * DATA_T];
			Cxy[1][1] += s_X[1][threadIdx.y][3][threadIdx.x] * c_Y[t_offset + 3 + 1 * DATA_T];
			Cxy[2][1] += s_X[2][threadIdx.y][3][threadIdx.x] * c_Y[t_offset + 3 + 1 * DATA_T];
			Cxy[3][1] += s_X[3][threadIdx.y][3][threadIdx.x] * c_Y[t_offset + 3 + 1 * DATA_T];
		}

		t_offset = timeMultiples * 4;

		for (int t = 0; t < timeRest; t++)
		{			
			s_X[0][threadIdx.y][t][threadIdx.x] = filter_responses_1[Calculate_4D_Index(x,y,z,t_offset + t, DATA_W, DATA_H, DATA_D)];
			s_X[1][threadIdx.y][t][threadIdx.x] = filter_responses_2[Calculate_4D_Index(x,y,z,t_offset + t, DATA_W, DATA_H, DATA_D)];
			s_X[2][threadIdx.y][t][threadIdx.x] = filter_responses_3[Calculate_4D_Index(x,y,z,t_offset + t, DATA_W, DATA_H, DATA_D)];
			s_X[3][threadIdx.y][t][threadIdx.x] = filter_responses_4[Calculate_4D_Index(x,y,z,t_offset + t, DATA_W, DATA_H, DATA_D)];
		}

		for (int t = 0; t < timeRest; t++)
		{
			Cxx[0] += s_X[0][threadIdx.y][t][threadIdx.x] * s_X[0][threadIdx.y][t][threadIdx.x];						
			Cxx[1] += s_X[1][threadIdx.y][t][threadIdx.x] * s_X[0][threadIdx.y][t][threadIdx.x];						
			Cxx[2] += s_X[2][threadIdx.y][t][threadIdx.x] * s_X[0][threadIdx.y][t][threadIdx.x];						
			Cxx[3] += s_X[3][threadIdx.y][t][threadIdx.x] * s_X[0][threadIdx.y][t][threadIdx.x];						
			Cxx[4] += s_X[1][threadIdx.y][t][threadIdx.x] * s_X[1][threadIdx.y][t][threadIdx.x];						
			Cxx[5] += s_X[2][threadIdx.y][t][threadIdx.x] * s_X[1][threadIdx.y][t][threadIdx.x];						
			Cxx[6] += s_X[3][threadIdx.y][t][threadIdx.x] * s_X[1][threadIdx.y][t][threadIdx.x];						
			Cxx[7] += s_X[2][threadIdx.y][t][threadIdx.x] * s_X[2][threadIdx.y][t][threadIdx.x];						
			Cxx[8] += s_X[3][threadIdx.y][t][threadIdx.x] * s_X[2][threadIdx.y][t][threadIdx.x];						
			Cxx[9] += s_X[3][threadIdx.y][t][threadIdx.x] * s_X[3][threadIdx.y][t][threadIdx.x];		

			Cxy[0][0] += s_X[0][threadIdx.y][t][threadIdx.x] * c_Y[t_offset + t + 0 * DATA_T];
			Cxy[1][0] += s_X[1][threadIdx.y][t][threadIdx.x] * c_Y[t_offset + t + 0 * DATA_T];
			Cxy[2][0] += s_X[2][threadIdx.y][t][threadIdx.x] * c_Y[t_offset + t + 0 * DATA_T];
			Cxy[3][0] += s_X[3][threadIdx.y][t][threadIdx.x] * c_Y[t_offset + t + 0 * DATA_T];

			Cxy[0][1] += s_X[0][threadIdx.y][t][threadIdx.x] * c_Y[t_offset + t + 1 * DATA_T];
			Cxy[1][1] += s_X[1][threadIdx.y][t][threadIdx.x] * c_Y[t_offset + t + 1 * DATA_T];
			Cxy[2][1] += s_X[2][threadIdx.y][t][threadIdx.x] * c_Y[t_offset + t + 1 * DATA_T];
			Cxy[3][1] += s_X[3][threadIdx.y][t][threadIdx.x] * c_Y[t_offset + t + 1 * DATA_T];
		}

	
		Cxx[0] /= ((float)DATA_T - 1.0f);
		Cxx[1] /= ((float)DATA_T - 1.0f);
		Cxx[2] /= ((float)DATA_T - 1.0f);
		Cxx[3] /= ((float)DATA_T - 1.0f);
		Cxx[4] /= ((float)DATA_T - 1.0f);
		Cxx[5] /= ((float)DATA_T - 1.0f);
		Cxx[6] /= ((float)DATA_T - 1.0f);
		Cxx[7] /= ((float)DATA_T - 1.0f);
		Cxx[8] /= ((float)DATA_T - 1.0f);
		Cxx[9] /= ((float)DATA_T - 1.0f);
	
		Cxy[0][0] /= ((float)DATA_T - 1.0f);
		Cxy[1][0] /= ((float)DATA_T - 1.0f);
		Cxy[2][0] /= ((float)DATA_T - 1.0f);
		Cxy[3][0] /= ((float)DATA_T - 1.0f);
	
		Cxy[0][1] /= ((float)DATA_T - 1.0f);
		Cxy[1][1] /= ((float)DATA_T - 1.0f);
		Cxy[2][1] /= ((float)DATA_T - 1.0f);
		Cxy[3][1] /= ((float)DATA_T - 1.0f);

		// Calculate the inverse of Cxx
		Invert_Cxx(Cxx, inv_Cxx);

		// Calculate the total matrix product, gives a 2 x 2 matrix,  (Cyy)^(-1/2) * Cyx * (Cxx)^(-1) * Cxy * (Cyy)^(-1/2)
		// First step, calculate Cyx * (Cxx)^(-1) * Cxy, three values sufficient since matrix is symmetric
		float alpha       = Cxy[0][0] * (Cxy[0][0] * inv_Cxx[0] + Cxy[1][0] * inv_Cxx[1] + Cxy[2][0] * inv_Cxx[2] + Cxy[3][0] * inv_Cxx[3]) + 
							Cxy[1][0] * (Cxy[0][0] * inv_Cxx[1] + Cxy[1][0] * inv_Cxx[4] + Cxy[2][0] * inv_Cxx[5] + Cxy[3][0] * inv_Cxx[6]) + 	
							Cxy[2][0] * (Cxy[0][0] * inv_Cxx[2] + Cxy[1][0] * inv_Cxx[5] + Cxy[2][0] * inv_Cxx[7] + Cxy[3][0] * inv_Cxx[8]) + 	
							Cxy[3][0] * (Cxy[0][0] * inv_Cxx[3] + Cxy[1][0] * inv_Cxx[6] + Cxy[2][0] * inv_Cxx[8] + Cxy[3][0] * inv_Cxx[9]);

		float beta        = Cxy[0][1] * (Cxy[0][0] * inv_Cxx[0] + Cxy[1][0] * inv_Cxx[1] + Cxy[2][0] * inv_Cxx[2] + Cxy[3][0] * inv_Cxx[3]) + 
						    Cxy[1][1] * (Cxy[0][0] * inv_Cxx[1] + Cxy[1][0] * inv_Cxx[4] + Cxy[2][0] * inv_Cxx[5] + Cxy[3][0] * inv_Cxx[6]) + 	
						    Cxy[2][1] * (Cxy[0][0] * inv_Cxx[2] + Cxy[1][0] * inv_Cxx[5] + Cxy[2][0] * inv_Cxx[7] + Cxy[3][0] * inv_Cxx[8]) + 	
							Cxy[3][1] * (Cxy[0][0] * inv_Cxx[3] + Cxy[1][0] * inv_Cxx[6] + Cxy[2][0] * inv_Cxx[8] + Cxy[3][0] * inv_Cxx[9]);

		float gamma       = Cxy[0][1] * (Cxy[0][1] * inv_Cxx[0] + Cxy[1][1] * inv_Cxx[1] + Cxy[2][1] * inv_Cxx[2] + Cxy[3][1] * inv_Cxx[3]) + 
						    Cxy[1][1] * (Cxy[0][1] * inv_Cxx[1] + Cxy[1][1] * inv_Cxx[4] + Cxy[2][1] * inv_Cxx[5] + Cxy[3][1] * inv_Cxx[6]) + 	
						    Cxy[2][1] * (Cxy[0][1] * inv_Cxx[2] + Cxy[1][1] * inv_Cxx[5] + Cxy[2][1] * inv_Cxx[7] + Cxy[3][1] * inv_Cxx[8]) + 	
						    Cxy[3][1] * (Cxy[0][1] * inv_Cxx[3] + Cxy[1][1] * inv_Cxx[6] + Cxy[2][1] * inv_Cxx[8] + Cxy[3][1] * inv_Cxx[9]);

		// Second step, calculate the whole product, three values sufficient since matrix is symmetric
		matrix_product[0] = c_sqrt_inv_Cyy[0] * (alpha * c_sqrt_inv_Cyy[0] + beta * c_sqrt_inv_Cyy[1]) + c_sqrt_inv_Cyy[1] * (beta * c_sqrt_inv_Cyy[0] + gamma * c_sqrt_inv_Cyy[1]);
		matrix_product[1] = c_sqrt_inv_Cyy[0] * (alpha * c_sqrt_inv_Cyy[2] + beta * c_sqrt_inv_Cyy[3]) + c_sqrt_inv_Cyy[2] * (beta * c_sqrt_inv_Cyy[2] + gamma * c_sqrt_inv_Cyy[3]);
		matrix_product[2] = c_sqrt_inv_Cyy[2] * (alpha * c_sqrt_inv_Cyy[2] + beta * c_sqrt_inv_Cyy[3]) + c_sqrt_inv_Cyy[3] * (beta * c_sqrt_inv_Cyy[2] + gamma * c_sqrt_inv_Cyy[3]);

		// Calculate the eigen values of the total matrix 
		// lambda 1 = (t1 + t3)/2 + sqrt(t2 * t2 + 1/4 * t1 * t1 + 1/4 * t3 * t3 - t1 * t3/2);
		// lambda 2 = (t1 + t3)/2 - sqrt(t2 * t2 + 1/4 * t1 * t1 + 1/4 * t3 * t3 - t1 * t3/2);	
		
		float lambda_1 = (matrix_product[0] + matrix_product[2]) * 0.5f + sqrtf(matrix_product[1] * matrix_product[1] + 0.25f * matrix_product[0] * matrix_product[0] + 0.25f * matrix_product[2] * matrix_product[2] - matrix_product[0] * matrix_product[2] * 0.5f);
		float lambda_2 = (matrix_product[0] + matrix_product[2]) * 0.5f - sqrtf(matrix_product[1] * matrix_product[1] + 0.25f * matrix_product[0] * matrix_product[0] + 0.25f * matrix_product[2] * matrix_product[2] - matrix_product[0] * matrix_product[2] * 0.5f);

		// Calculate the eigenvector corresponding to the largest eigenvalue = the temporal weight vector wy

		// ee1_1 = t1 - lambda_2;
		// ee1_2 = t2;
		// ee1_3 = t3 - lambda_2;

		// norm = sqrt(ee1_1.^2 + 2*ee1_2.^2 + ee1_3.^2);

		// e11 = sqrt(ee1_1);
		// e12 = sqrt(ee1_3);

		//temporal_weights[0] = lambda_1 - matrix_product[2];
		//temporal_weights[1] = matrix_product[1];

		// Eigenvector to biggest eigenvalue, a
		matrix_product[0] = lambda_1 - matrix_product[2];
		matrix_product[1] = matrix_product[1];

		// Make a change of base, wy = (Cyy)^(-1/2) * a
		temporal_weights[0] = c_sqrt_inv_Cyy[0] * matrix_product[0] + c_sqrt_inv_Cyy[1] * matrix_product[1];
		temporal_weights[1] = c_sqrt_inv_Cyy[2] * matrix_product[0] + c_sqrt_inv_Cyy[3] * matrix_product[1];

		// Normalize weight vector to unit length
		float norm = sqrt(temporal_weights[0] * temporal_weights[0] + temporal_weights[1] * temporal_weights[1]);
		temporal_weights[0] /= norm;
		temporal_weights[1] /= norm;

		// Now get the filter weight from the temporal weight vector by wx = (Cxx)^(-1) * Cxy * wy
		spatial_weights[0] = inv_Cxx[0] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) + 
                             inv_Cxx[1] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1]) + 
                             inv_Cxx[2] * (Cxy[2][0] * temporal_weights[0] + Cxy[2][1] * temporal_weights[1]) + 
                             inv_Cxx[3] * (Cxy[3][0] * temporal_weights[0] + Cxy[3][1] * temporal_weights[1]);

		spatial_weights[1] = inv_Cxx[1] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) +
                             inv_Cxx[4] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1]) + 
                             inv_Cxx[5] * (Cxy[2][0] * temporal_weights[0] + Cxy[2][1] * temporal_weights[1]) +
                             inv_Cxx[6] * (Cxy[3][0] * temporal_weights[0] + Cxy[3][1] * temporal_weights[1]);
 
		spatial_weights[2] = inv_Cxx[2] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) + 
                             inv_Cxx[5] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1]) +
                             inv_Cxx[7] * (Cxy[2][0] * temporal_weights[0] + Cxy[2][1] * temporal_weights[1]) +  
                             inv_Cxx[8] * (Cxy[3][0] * temporal_weights[0] + Cxy[3][1] * temporal_weights[1]);

		spatial_weights[3] = inv_Cxx[3] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) + 
                             inv_Cxx[6] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1]) + 
                             inv_Cxx[8] * (Cxy[2][0] * temporal_weights[0] + Cxy[2][1] * temporal_weights[1]) +  
                             inv_Cxx[9] * (Cxy[3][0] * temporal_weights[0] + Cxy[3][1] * temporal_weights[1]);
		        
		// Normalize weight vector to have unit length
		norm = sqrtf( spatial_weights[0] * spatial_weights[0] + spatial_weights[1] * spatial_weights[1] + spatial_weights[2] * spatial_weights[2] + spatial_weights[3] * spatial_weights[3] );
		spatial_weights[0] /= norm;
		spatial_weights[1] /= norm;
		spatial_weights[2] /= norm;
		spatial_weights[3] /= norm;

		int flip = 0;

		// Adjust the filter weights
		if (min3(spatial_weights) < 0.0f)
        {
			// Count the number of negative and positive filter coefficients
			int negative, positive;	negative = 0; positive = 0;

			#pragma unroll 9
			for (int i = 0; i < 9; i++)
			{	
				#pragma unroll 9
				for (int j = 0; j < 9; j++)
				{					
					float total_coefficient = spatial_weights[0] * c_CCA_2D_Filters[i][j].x + spatial_weights[1] * c_CCA_2D_Filters[i][j].y + spatial_weights[2] * c_CCA_2D_Filters[i][j].z;
				
					if (total_coefficient < 0.0f)
					{
						negative++;
					}	
					else
					{
						positive++;
					}
				}
			}
            
	        if (negative == 81)
			{
				// Flip sign of weights
				spatial_weights[0] = -spatial_weights[0];
				spatial_weights[1] = -spatial_weights[1];
				spatial_weights[2] = -spatial_weights[2];
				// Flip sign of temporal weights also, to keep the sign of the correlation
				temporal_weights[0] = -temporal_weights[0];
				temporal_weights[1] = -temporal_weights[1];
			}
	        else if (negative > positive)
			{
				// Check original sign of correlation
				float s1 = spatial_weights[0] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) +  
                           spatial_weights[1] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1]) + 
                           spatial_weights[2] * (Cxy[2][0] * temporal_weights[0] + Cxy[2][1] * temporal_weights[1]) + 
                           spatial_weights[3] * (Cxy[3][0] * temporal_weights[0] + Cxy[3][1] * temporal_weights[1]);

				if (s1 < 0.0f)
				{
					s1 = -1.0f;
				}
				else
				{
					s1 = 1.0f;
				}

				// Flip sign of weights
				spatial_weights[0] = -spatial_weights[0];
				spatial_weights[1] = -spatial_weights[1];
				spatial_weights[2] = -spatial_weights[2];
				
				// Add most negative weight to the anisotropic weights
				float min_weight = abs(min3(spatial_weights));
				spatial_weights[0] += min_weight;
				spatial_weights[1] += min_weight;
				spatial_weights[2] += min_weight;

				// Check new sign of correlation
				float s2 = spatial_weights[0] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) +  
                           spatial_weights[1] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1]) + 
                           spatial_weights[2] * (Cxy[2][0] * temporal_weights[0] + Cxy[2][1] * temporal_weights[1]) + 
                           spatial_weights[3] * (Cxy[3][0] * temporal_weights[0] + Cxy[3][1] * temporal_weights[1]);

				if (s2 < 0.0f)
				{
					s2 = -1.0f;
				}
				else
				{
					s2 = 1.0f;
				}

				if (s1 != s2)
				{
					// Flip sign of temporal weights, to keep the sign of the correlation
					temporal_weights[0] = -temporal_weights[0];
					temporal_weights[1] = -temporal_weights[1];
					flip = 1;
				}

			}
	        else
			{
				// Check original sign of correlation
				float s1 = spatial_weights[0] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) +  
                           spatial_weights[1] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1]) + 
                           spatial_weights[2] * (Cxy[2][0] * temporal_weights[0] + Cxy[2][1] * temporal_weights[1]) + 
                           spatial_weights[3] * (Cxy[3][0] * temporal_weights[0] + Cxy[3][1] * temporal_weights[1]);

				if (s1 < 0.0f)
				{
					s1 = -1.0f;
				}
				else
				{
					s1 = 1.0f;
				}

				// Add most negative weight to the anisotropic weights
				float min_weight = abs(min3(spatial_weights));
				spatial_weights[0] += min_weight;
				spatial_weights[1] += min_weight;
				spatial_weights[2] += min_weight;

				// Check new sign of correlation
				float s2 = spatial_weights[0] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) +  
                           spatial_weights[1] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1]) + 
                           spatial_weights[2] * (Cxy[2][0] * temporal_weights[0] + Cxy[2][1] * temporal_weights[1]) + 
                           spatial_weights[3] * (Cxy[3][0] * temporal_weights[0] + Cxy[3][1] * temporal_weights[1]);

				if (s2 < 0.0f)
				{
					s2 = -1.0f;
				}
				else
				{
					s2 = 1.0f;
				}

				if (s1 != s2)
				{
					// Flip sign of temporal weights, to keep the sign of the correlation
					temporal_weights[0] = -temporal_weights[0];
					temporal_weights[1] = -temporal_weights[1];
					flip = 1;
				}
			}
        }

		// Make sure center pixel has highest weight
		spatial_weights[3] = 1.2f * max3(spatial_weights);
		// Normalize weight vector to have unit length
	    norm = sqrtf( spatial_weights[0] * spatial_weights[0] + spatial_weights[1] * spatial_weights[1] + spatial_weights[2] * spatial_weights[2] + spatial_weights[3] * spatial_weights[3] );
		spatial_weights[0] /= norm;
		spatial_weights[1] /= norm;
		spatial_weights[2] /= norm;
		spatial_weights[3] /= norm;  
		

		// Calculate the canonical correlation with the adjusted filter weights, rho = wx' * Cxy * wy / sqrt( wx' * Cxx * wx * wy' * Cyy * wy )
		float correlation = ( spatial_weights[0] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) +
                                                               spatial_weights[1] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1]) + 
                                                               spatial_weights[2] * (Cxy[2][0] * temporal_weights[0] + Cxy[2][1] * temporal_weights[1]) + 
                                                               spatial_weights[3] * (Cxy[3][0] * temporal_weights[0] + Cxy[3][1] * temporal_weights[1]) ) 
                                 
                                                                      / 
	  	
      sqrtf( ( spatial_weights[0] * (Cxx[0] * spatial_weights[0] + Cxx[1] * spatial_weights[1] + Cxx[2] * spatial_weights[2] + Cxx[3] * spatial_weights[3]) +
               spatial_weights[1] * (Cxx[1] * spatial_weights[0] + Cxx[4] * spatial_weights[1] + Cxx[5] * spatial_weights[2] + Cxx[6] * spatial_weights[3]) +
               spatial_weights[2] * (Cxx[2] * spatial_weights[0] + Cxx[5] * spatial_weights[1] + Cxx[7] * spatial_weights[2] + Cxx[8] * spatial_weights[3]) + 
               spatial_weights[3] * (Cxx[3] * spatial_weights[0] + Cxx[6] * spatial_weights[1] + Cxx[8] * spatial_weights[2] + Cxx[9] * spatial_weights[3]) )

																	 * 

			(temporal_weights[0] * (c_Cyy[0] * temporal_weights[0] + c_Cyy[1] * temporal_weights[1]) + temporal_weights[1] * (c_Cyy[2] * temporal_weights[0] + c_Cyy[3] * temporal_weights[1])) );	

		if (flip == 0)
		{
			activity_map[x + y * DATA_W + z * DATA_W * DATA_H] = correlation;
		}
		else
		{
			activity_map[x + y * DATA_W + z * DATA_W * DATA_H] = -correlation;
		}
	}
	else
	{
		activity_map[x + y * DATA_W + z * DATA_W * DATA_H] = 0.0f;
	}
}


// 3D

__device__ float Determinant_2x2(float *Cxx)
{
    return Cxx[0] * Cxx[3] - Cxx[1] * Cxx[2];
}

__device__ void Invert_Cxx_2x2(float *Cxx, float *inv_Cxx)
{
	float determinant = Determinant_2x2(Cxx);

	inv_Cxx[0] = Cxx[3];
	inv_Cxx[1] = -Cxx[1];
	inv_Cxx[2] = -Cxx[2];
	inv_Cxx[3] = Cxx[0];

	inv_Cxx[0] /= determinant;
	inv_Cxx[1] /= determinant;
	inv_Cxx[2] /= determinant;
	inv_Cxx[3] /= determinant;
}

__global__ void CalculateActivityMapCCA3D(float* activity_map, float* filter_responses_1, float* filter_responses_2, float* Brain_Voxels, int DATA_W, int DATA_H, int DATA_D, int DATA_T, int blocksInY, float invBlocksInY, int timeMultiples, int timeRest)
{	
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
	volatile unsigned int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	volatile unsigned int y = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
	volatile unsigned int z = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Brain_Voxels[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
	{
		__shared__ float s_X[2][4][4][32]; // filter responses,y,t,x

		float Cxx[4];
		float inv_Cxx[4];
		float Cxy[2][2];
		float matrix_product[3];
		float temporal_weights[2];
		float spatial_weights[2];	

		Cxx[0] = 0.0f; Cxx[1] = 0.0f; Cxx[2] = 0.0f; Cxx[3] = 0.0f; 

		Cxy[0][0] = 0.0f; Cxy[0][1] = 0.0f;		
		Cxy[1][0] = 0.0f; Cxy[1][1] = 0.0f;		

		int t_offset = 0;
		for (t_offset = 0; t_offset < timeMultiples * 4; t_offset += 4)
		{ 
			// Read the two filter responses into shared memory
			
			s_X[0][threadIdx.y][0][threadIdx.x] = filter_responses_1[Calculate_4D_Index(x,y,z,t_offset + 0, DATA_W, DATA_H, DATA_D)];
			s_X[1][threadIdx.y][0][threadIdx.x] = filter_responses_2[Calculate_4D_Index(x,y,z,t_offset + 0, DATA_W, DATA_H, DATA_D)];

			s_X[0][threadIdx.y][1][threadIdx.x] = filter_responses_1[Calculate_4D_Index(x,y,z,t_offset + 1, DATA_W, DATA_H, DATA_D)];
			s_X[1][threadIdx.y][1][threadIdx.x] = filter_responses_2[Calculate_4D_Index(x,y,z,t_offset + 1, DATA_W, DATA_H, DATA_D)];

			s_X[0][threadIdx.y][2][threadIdx.x] = filter_responses_1[Calculate_4D_Index(x,y,z,t_offset + 2, DATA_W, DATA_H, DATA_D)];
			s_X[1][threadIdx.y][2][threadIdx.x] = filter_responses_2[Calculate_4D_Index(x,y,z,t_offset + 2, DATA_W, DATA_H, DATA_D)];

			s_X[0][threadIdx.y][3][threadIdx.x] = filter_responses_1[Calculate_4D_Index(x,y,z,t_offset + 3, DATA_W, DATA_H, DATA_D)];
			s_X[1][threadIdx.y][3][threadIdx.x] = filter_responses_2[Calculate_4D_Index(x,y,z,t_offset + 3, DATA_W, DATA_H, DATA_D)];

			__syncthreads();

			// Calculate Cxx
			Cxx[0] += s_X[0][threadIdx.y][0][threadIdx.x] * s_X[0][threadIdx.y][0][threadIdx.x];						
			Cxx[1] += s_X[1][threadIdx.y][0][threadIdx.x] * s_X[0][threadIdx.y][0][threadIdx.x];						
			Cxx[2] += s_X[0][threadIdx.y][0][threadIdx.x] * s_X[1][threadIdx.y][0][threadIdx.x];						
			Cxx[3] += s_X[1][threadIdx.y][0][threadIdx.x] * s_X[1][threadIdx.y][0][threadIdx.x];						

			Cxx[0] += s_X[0][threadIdx.y][1][threadIdx.x] * s_X[0][threadIdx.y][1][threadIdx.x];						
			Cxx[1] += s_X[1][threadIdx.y][1][threadIdx.x] * s_X[0][threadIdx.y][1][threadIdx.x];						
			Cxx[2] += s_X[0][threadIdx.y][1][threadIdx.x] * s_X[1][threadIdx.y][1][threadIdx.x];						
			Cxx[3] += s_X[1][threadIdx.y][1][threadIdx.x] * s_X[1][threadIdx.y][1][threadIdx.x];						

			Cxx[0] += s_X[0][threadIdx.y][2][threadIdx.x] * s_X[0][threadIdx.y][2][threadIdx.x];						
			Cxx[1] += s_X[1][threadIdx.y][2][threadIdx.x] * s_X[0][threadIdx.y][2][threadIdx.x];						
			Cxx[2] += s_X[0][threadIdx.y][2][threadIdx.x] * s_X[1][threadIdx.y][2][threadIdx.x];						
			Cxx[3] += s_X[1][threadIdx.y][2][threadIdx.x] * s_X[1][threadIdx.y][2][threadIdx.x];						

			Cxx[0] += s_X[0][threadIdx.y][3][threadIdx.x] * s_X[0][threadIdx.y][3][threadIdx.x];						
			Cxx[1] += s_X[1][threadIdx.y][3][threadIdx.x] * s_X[0][threadIdx.y][3][threadIdx.x];						
			Cxx[2] += s_X[0][threadIdx.y][3][threadIdx.x] * s_X[1][threadIdx.y][3][threadIdx.x];						
			Cxx[3] += s_X[1][threadIdx.y][3][threadIdx.x] * s_X[1][threadIdx.y][3][threadIdx.x];						

			
			// Calculate Cxy		
			Cxy[0][0] += s_X[0][threadIdx.y][0][threadIdx.x] * c_Y[t_offset + 0 + 0 * DATA_T];
			Cxy[1][0] += s_X[1][threadIdx.y][0][threadIdx.x] * c_Y[t_offset + 0 + 0 * DATA_T];
	
			Cxy[0][0] += s_X[0][threadIdx.y][1][threadIdx.x] * c_Y[t_offset + 1 + 0 * DATA_T];
			Cxy[1][0] += s_X[1][threadIdx.y][1][threadIdx.x] * c_Y[t_offset + 1 + 0 * DATA_T];

			Cxy[0][0] += s_X[0][threadIdx.y][2][threadIdx.x] * c_Y[t_offset + 2 + 0 * DATA_T];
			Cxy[1][0] += s_X[1][threadIdx.y][2][threadIdx.x] * c_Y[t_offset + 2 + 0 * DATA_T];

			Cxy[0][0] += s_X[0][threadIdx.y][3][threadIdx.x] * c_Y[t_offset + 3 + 0 * DATA_T];
			Cxy[1][0] += s_X[1][threadIdx.y][3][threadIdx.x] * c_Y[t_offset + 3 + 0 * DATA_T];



			Cxy[0][1] += s_X[0][threadIdx.y][0][threadIdx.x] * c_Y[t_offset + 0 + 1 * DATA_T];
			Cxy[1][1] += s_X[1][threadIdx.y][0][threadIdx.x] * c_Y[t_offset + 0 + 1 * DATA_T];

			Cxy[0][1] += s_X[0][threadIdx.y][1][threadIdx.x] * c_Y[t_offset + 1 + 1 * DATA_T];
			Cxy[1][1] += s_X[1][threadIdx.y][1][threadIdx.x] * c_Y[t_offset + 1 + 1 * DATA_T];

			Cxy[0][1] += s_X[0][threadIdx.y][2][threadIdx.x] * c_Y[t_offset + 2 + 1 * DATA_T];
			Cxy[1][1] += s_X[1][threadIdx.y][2][threadIdx.x] * c_Y[t_offset + 2 + 1 * DATA_T];

			Cxy[0][1] += s_X[0][threadIdx.y][3][threadIdx.x] * c_Y[t_offset + 3 + 1 * DATA_T];
			Cxy[1][1] += s_X[1][threadIdx.y][3][threadIdx.x] * c_Y[t_offset + 3 + 1 * DATA_T];
		}

		t_offset = timeMultiples * 4;

		for (int t = 0; t < timeRest; t++)
		{			
			s_X[0][threadIdx.y][t][threadIdx.x] = filter_responses_1[Calculate_4D_Index(x,y,z,t_offset + t, DATA_W, DATA_H, DATA_D)];
			s_X[1][threadIdx.y][t][threadIdx.x] = filter_responses_2[Calculate_4D_Index(x,y,z,t_offset + t, DATA_W, DATA_H, DATA_D)];
		}

		for (int t = 0; t < timeRest; t++)
		{
			Cxx[0] += s_X[0][threadIdx.y][t][threadIdx.x] * s_X[0][threadIdx.y][t][threadIdx.x];						
			Cxx[1] += s_X[1][threadIdx.y][t][threadIdx.x] * s_X[0][threadIdx.y][t][threadIdx.x];						
			Cxx[2] += s_X[0][threadIdx.y][t][threadIdx.x] * s_X[1][threadIdx.y][t][threadIdx.x];						
			Cxx[3] += s_X[1][threadIdx.y][t][threadIdx.x] * s_X[1][threadIdx.y][t][threadIdx.x];						

			Cxy[0][0] += s_X[0][threadIdx.y][t][threadIdx.x] * c_Y[t_offset + t + 0 * DATA_T];
			Cxy[1][0] += s_X[1][threadIdx.y][t][threadIdx.x] * c_Y[t_offset + t + 0 * DATA_T];

			Cxy[0][1] += s_X[0][threadIdx.y][t][threadIdx.x] * c_Y[t_offset + t + 1 * DATA_T];
			Cxy[1][1] += s_X[1][threadIdx.y][t][threadIdx.x] * c_Y[t_offset + t + 1 * DATA_T];
		}

	
		Cxx[0] /= ((float)DATA_T - 1);
		Cxx[1] /= ((float)DATA_T - 1);
		Cxx[2] /= ((float)DATA_T - 1);
		Cxx[3] /= ((float)DATA_T - 1);
	
		Cxy[0][0] /= ((float)DATA_T - 1);
		Cxy[1][0] /= ((float)DATA_T - 1);
	
		Cxy[0][1] /= ((float)DATA_T - 1);
		Cxy[1][1] /= ((float)DATA_T - 1);

		// Calculate the inverse of Cxx
		Invert_Cxx_2x2(Cxx, inv_Cxx);

		/*
		float alpha       = Cxy[0][0] * (Cxy[0][0] * inv_Cxx[0] + Cxy[1][0] * inv_Cxx[1] + Cxy[2][0] * inv_Cxx[2] + Cxy[3][0] * inv_Cxx[3]) + 
							Cxy[1][0] * (Cxy[0][0] * inv_Cxx[1] + Cxy[1][0] * inv_Cxx[4] + Cxy[2][0] * inv_Cxx[5] + Cxy[3][0] * inv_Cxx[6]) + 	
							Cxy[2][0] * (Cxy[0][0] * inv_Cxx[2] + Cxy[1][0] * inv_Cxx[5] + Cxy[2][0] * inv_Cxx[7] + Cxy[3][0] * inv_Cxx[8]) + 	
							Cxy[3][0] * (Cxy[0][0] * inv_Cxx[3] + Cxy[1][0] * inv_Cxx[6] + Cxy[2][0] * inv_Cxx[8] + Cxy[3][0] * inv_Cxx[9]);

		float beta        = Cxy[0][1] * (Cxy[0][0] * inv_Cxx[0] + Cxy[1][0] * inv_Cxx[1] + Cxy[2][0] * inv_Cxx[2] + Cxy[3][0] * inv_Cxx[3]) + 
						    Cxy[1][1] * (Cxy[0][0] * inv_Cxx[1] + Cxy[1][0] * inv_Cxx[4] + Cxy[2][0] * inv_Cxx[5] + Cxy[3][0] * inv_Cxx[6]) + 	
						    Cxy[2][1] * (Cxy[0][0] * inv_Cxx[2] + Cxy[1][0] * inv_Cxx[5] + Cxy[2][0] * inv_Cxx[7] + Cxy[3][0] * inv_Cxx[8]) + 	
							Cxy[3][1] * (Cxy[0][0] * inv_Cxx[3] + Cxy[1][0] * inv_Cxx[6] + Cxy[2][0] * inv_Cxx[8] + Cxy[3][0] * inv_Cxx[9]);

		float gamma       = Cxy[0][1] * (Cxy[0][1] * inv_Cxx[0] + Cxy[1][1] * inv_Cxx[1] + Cxy[2][1] * inv_Cxx[2] + Cxy[3][1] * inv_Cxx[3]) + 
						    Cxy[1][1] * (Cxy[0][1] * inv_Cxx[1] + Cxy[1][1] * inv_Cxx[4] + Cxy[2][1] * inv_Cxx[5] + Cxy[3][1] * inv_Cxx[6]) + 	
						    Cxy[2][1] * (Cxy[0][1] * inv_Cxx[2] + Cxy[1][1] * inv_Cxx[5] + Cxy[2][1] * inv_Cxx[7] + Cxy[3][1] * inv_Cxx[8]) + 	
						    Cxy[3][1] * (Cxy[0][1] * inv_Cxx[3] + Cxy[1][1] * inv_Cxx[6] + Cxy[2][1] * inv_Cxx[8] + Cxy[3][1] * inv_Cxx[9]);
		*/
	

		// Calculate the total matrix product, gives a 2 x 2 matrix,  (Cyy)^(-1/2) * Cyx * (Cxx)^(-1) * Cxy * (Cyy)^(-1/2)
		// First step, calculate Cyx * (Cxx)^(-1) * Cxy, three values sufficient since matrix is symmetric
		float alpha       = Cxy[0][0] * (Cxy[0][0] * inv_Cxx[0] + Cxy[1][0] * inv_Cxx[2]) + 
							Cxy[1][0] * (Cxy[0][0] * inv_Cxx[1] + Cxy[1][0] * inv_Cxx[3]);
							
		float beta        = Cxy[0][1] * (Cxy[0][0] * inv_Cxx[0] + Cxy[1][0] * inv_Cxx[2]) + 
							Cxy[1][1] * (Cxy[0][0] * inv_Cxx[1] + Cxy[1][0] * inv_Cxx[3]); 	

		float gamma       = Cxy[0][1] * (Cxy[0][1] * inv_Cxx[0] + Cxy[1][1] * inv_Cxx[2]) + 
							Cxy[1][1] * (Cxy[0][1] * inv_Cxx[1] + Cxy[1][1] * inv_Cxx[3]); 	
						    
		// Second step, calculate the whole product, three values sufficient since matrix is symmetric
		matrix_product[0] = c_sqrt_inv_Cyy[0] * (alpha * c_sqrt_inv_Cyy[0] + beta * c_sqrt_inv_Cyy[1]) + c_sqrt_inv_Cyy[1] * (beta * c_sqrt_inv_Cyy[0] + gamma * c_sqrt_inv_Cyy[1]);
		matrix_product[1] = c_sqrt_inv_Cyy[0] * (alpha * c_sqrt_inv_Cyy[2] + beta * c_sqrt_inv_Cyy[3]) + c_sqrt_inv_Cyy[2] * (beta * c_sqrt_inv_Cyy[2] + gamma * c_sqrt_inv_Cyy[3]);
		matrix_product[2] = c_sqrt_inv_Cyy[2] * (alpha * c_sqrt_inv_Cyy[2] + beta * c_sqrt_inv_Cyy[3]) + c_sqrt_inv_Cyy[3] * (beta * c_sqrt_inv_Cyy[2] + gamma * c_sqrt_inv_Cyy[3]);

		// Calculate the eigen values of the total matrix 
		// lambda 1 = (t1 + t3)/2 + sqrt(t2 * t2 + 1/4 * t1 * t1 + 1/4 * t3 * t3 - t1 * t3/2);
		// lambda 2 = (t1 + t3)/2 - sqrt(t2 * t2 + 1/4 * t1 * t1 + 1/4 * t3 * t3 - t1 * t3/2);	
		
		float lambda_1 = (matrix_product[0] + matrix_product[2]) * 0.5f + sqrtf(matrix_product[1] * matrix_product[1] + 0.25f * matrix_product[0] * matrix_product[0] + 0.25f * matrix_product[2] * matrix_product[2] - matrix_product[0] * matrix_product[2] * 0.5f);
		float lambda_2 = (matrix_product[0] + matrix_product[2]) * 0.5f - sqrtf(matrix_product[1] * matrix_product[1] + 0.25f * matrix_product[0] * matrix_product[0] + 0.25f * matrix_product[2] * matrix_product[2] - matrix_product[0] * matrix_product[2] * 0.5f);

		// Calculate the eigenvector corresponding to the largest eigenvalue = the temporal weight vector wy

		// ee1_1 = t1 - lambda_2;
		// ee1_2 = t2;
		// ee1_3 = t3 - lambda_2;

		// norm = sqrt(ee1_1.^2 + 2*ee1_2.^2 + ee1_3.^2);

		// e11 = sqrt(ee1_1);
		// e12 = sqrt(ee1_3);

		//temporal_weights[0] = lambda_1 - matrix_product[2];
		//temporal_weights[1] = matrix_product[1];

		// Eigenvector to biggest eigenvalue, a
		matrix_product[0] = lambda_1 - matrix_product[2];
		matrix_product[1] = matrix_product[1];

		// Make a change of base, wy = (Cyy)^(-1/2) * a
		temporal_weights[0] = c_sqrt_inv_Cyy[0] * matrix_product[0] + c_sqrt_inv_Cyy[1] * matrix_product[1];
		temporal_weights[1] = c_sqrt_inv_Cyy[2] * matrix_product[0] + c_sqrt_inv_Cyy[3] * matrix_product[1];

		// Normalize weight vector to unit length
		float norm = sqrt(temporal_weights[0] * temporal_weights[0] + temporal_weights[1] * temporal_weights[1]);
		temporal_weights[0] /= norm;
		temporal_weights[1] /= norm;

		/*
		spatial_weights[0] = inv_Cxx[0] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) + 
                             inv_Cxx[1] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1]) + 
                             inv_Cxx[2] * (Cxy[2][0] * temporal_weights[0] + Cxy[2][1] * temporal_weights[1]) + 
                             inv_Cxx[3] * (Cxy[3][0] * temporal_weights[0] + Cxy[3][1] * temporal_weights[1]);

		spatial_weights[1] = inv_Cxx[1] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) +
                             inv_Cxx[4] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1]) + 
                             inv_Cxx[5] * (Cxy[2][0] * temporal_weights[0] + Cxy[2][1] * temporal_weights[1]) +
                             inv_Cxx[6] * (Cxy[3][0] * temporal_weights[0] + Cxy[3][1] * temporal_weights[1]);
 
		spatial_weights[2] = inv_Cxx[2] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) + 
                             inv_Cxx[5] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1]) +
                             inv_Cxx[7] * (Cxy[2][0] * temporal_weights[0] + Cxy[2][1] * temporal_weights[1]) +  
                             inv_Cxx[8] * (Cxy[3][0] * temporal_weights[0] + Cxy[3][1] * temporal_weights[1]);

		spatial_weights[3] = inv_Cxx[3] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) + 
                             inv_Cxx[6] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1]) + 
                             inv_Cxx[8] * (Cxy[2][0] * temporal_weights[0] + Cxy[2][1] * temporal_weights[1]) +  
                             inv_Cxx[9] * (Cxy[3][0] * temporal_weights[0] + Cxy[3][1] * temporal_weights[1]);
		*/


		// Now get the filter weight from the temporal weight vector by wx = (Cxx)^(-1) * Cxy * wy
		spatial_weights[0] = inv_Cxx[0] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) + 
                             inv_Cxx[1] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1]); 

		spatial_weights[1] = inv_Cxx[2] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) +
                             inv_Cxx[3] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1]); 
 		        
		// Normalize weight vector to have unit length
		norm = sqrtf( spatial_weights[0] * spatial_weights[0] + spatial_weights[1] * spatial_weights[1]);
		spatial_weights[0] /= norm;
		spatial_weights[1] /= norm;

		// Check original sign of correlation
		float s1 = sign(spatial_weights[0] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) +  spatial_weights[1] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1])); 

		float limit1 = -0.2f;
		float limit2 = 0.6f;

		// Adjust the filter weights            
        if (spatial_weights[0] < limit1)
		{
			spatial_weights[0] = -spatial_weights[0];
			
			if (abs(spatial_weights[1]) > limit2 * spatial_weights[0])
			{
				spatial_weights[1] = limit2 * spatial_weights[0] * sign(spatial_weights[1]);
			}						
		}
		else if (abs(spatial_weights[1]) > limit2 * spatial_weights[0])
		{
			spatial_weights[1] = limit2 * spatial_weights[0] * sign(spatial_weights[1]);
		}
		
		float total = spatial_weights[0] * c_CCA_3D_Filters[4].x + spatial_weights[1] * c_CCA_3D_Filters[4].y;
		
		if (total < 0.0f)
		{
			spatial_weights[0] = -spatial_weights[0];
			spatial_weights[1] = -spatial_weights[1];
		}

		// Normalize vector
		norm = sqrt(spatial_weights[0] * spatial_weights[0] + spatial_weights[1] * spatial_weights[1]);
		spatial_weights[0] /= norm;
		spatial_weights[1] /= norm;

		// Check new sign of correlation
		float s2 = sign(spatial_weights[0] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) + spatial_weights[1] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1])); 

		/*
		if (s1 != s2)
		{
			// Flip sign of temporal weights, to keep the sign of the correlation
			temporal_weights[0] = -temporal_weights[0];
			temporal_weights[1] = -temporal_weights[1];
		}
		*/


		// Calculate the canonical correlation with the adjusted filter weights, rho = wx' * Cxy * wy / sqrt( wx' * Cxx * wx * wy' * Cyy * wy )
		activity_map[x + y * DATA_W + z * DATA_W * DATA_H] = ( spatial_weights[0] * (Cxy[0][0] * temporal_weights[0] + Cxy[0][1] * temporal_weights[1]) +
                                                               spatial_weights[1] * (Cxy[1][0] * temporal_weights[0] + Cxy[1][1] * temporal_weights[1]) ) 
                                 
                                                                      / 
	  	
      sqrtf( ( spatial_weights[0] * (Cxx[0] * spatial_weights[0] + Cxx[1] * spatial_weights[1]) + 
			   spatial_weights[1] * (Cxx[1] * spatial_weights[0] + Cxx[3] * spatial_weights[1]) )

																	 * 

			(temporal_weights[0] * (c_Cyy[0] * temporal_weights[0] + c_Cyy[1] * temporal_weights[1]) + temporal_weights[1] * (c_Cyy[2] * temporal_weights[0] + c_Cyy[3] * temporal_weights[1])) );	
	}
	else
	{
		activity_map[x + y * DATA_W + z * DATA_W * DATA_H] = 0.0f;
	}
}







// Statistical analysis GLM

__device__ __constant__ float c_X_GLM[1000];
__device__ __constant__ float c_xtxxt_GLM[1000];

__global__ void CalculateActivityMapGLM(float* activity_map, float* fMRI_Volumes, float* Brain_Voxels, float ctxtxc, int DATA_W, int DATA_H, int DATA_D, int DATA_T, int blocksInY, float invBlocksInY, int timeMultiples, int timeRest)
{
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
	unsigned int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	unsigned int y = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
	unsigned int z = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Brain_Voxels[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
	{
		__shared__ float s_Y[8][8][32]; // y,t,x
		int t_offset = 0;
		float beta1 = 0.0f;
		float beta2 = 0.0f;
		float eps, meaneps, vareps;

		// Calculate betahat, i.e. multiply (x^T x)^(-1) x^T with Y
		for (t_offset = 0; t_offset < timeMultiples * 8; t_offset += 8)
		{ 
			// Read data into shared memory
			s_Y[threadIdx.y][0][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 0, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][1][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 1, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][2][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 2, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][3][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 3, DATA_W, DATA_H, DATA_D)];

			s_Y[threadIdx.y][4][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 4, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][5][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 5, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][6][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 6, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][7][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 7, DATA_W, DATA_H, DATA_D)];

			__syncthreads();
	
			// Sum and multiply the values in shared memory
			beta1 += s_Y[threadIdx.y][0][threadIdx.x] * c_xtxxt_GLM[t_offset + 0];
			beta1 += s_Y[threadIdx.y][1][threadIdx.x] * c_xtxxt_GLM[t_offset + 1];
			beta1 += s_Y[threadIdx.y][2][threadIdx.x] * c_xtxxt_GLM[t_offset + 2];
			beta1 += s_Y[threadIdx.y][3][threadIdx.x] * c_xtxxt_GLM[t_offset + 3];
			beta1 += s_Y[threadIdx.y][4][threadIdx.x] * c_xtxxt_GLM[t_offset + 4];
			beta1 += s_Y[threadIdx.y][5][threadIdx.x] * c_xtxxt_GLM[t_offset + 5];
			beta1 += s_Y[threadIdx.y][6][threadIdx.x] * c_xtxxt_GLM[t_offset + 6];
			beta1 += s_Y[threadIdx.y][7][threadIdx.x] * c_xtxxt_GLM[t_offset + 7];

			beta2 += s_Y[threadIdx.y][0][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + 0];
			beta2 += s_Y[threadIdx.y][1][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + 1];
			beta2 += s_Y[threadIdx.y][2][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + 2];
			beta2 += s_Y[threadIdx.y][3][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + 3];
			beta2 += s_Y[threadIdx.y][4][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + 4];
			beta2 += s_Y[threadIdx.y][5][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + 5];
			beta2 += s_Y[threadIdx.y][6][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + 6];
			beta2 += s_Y[threadIdx.y][7][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + 7];
		}

		t_offset = timeMultiples * 8;	

		for (int t = 0; t < timeRest; t++)
		{
			s_Y[threadIdx.y][0][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + t, DATA_W, DATA_H, DATA_D)];
			beta1 += s_Y[threadIdx.y][0][threadIdx.x] * c_xtxxt_GLM[t_offset + t];
			beta2 += s_Y[threadIdx.y][0][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + t];
		}

		// Calculate the mean of the error eps
		meaneps = 0.0f;

		for (t_offset = 0; t_offset < timeMultiples * 8; t_offset += 8)
		{ 
			// Read data into shared memory again
			s_Y[threadIdx.y][0][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 0, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][1][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 1, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][2][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 2, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][3][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 3, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][4][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 4, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][5][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 5, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][6][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 6, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][7][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 7, DATA_W, DATA_H, DATA_D)];

			__syncthreads();
	
			meaneps += s_Y[threadIdx.y][0][threadIdx.x] - beta1 * c_X_GLM[t_offset + 0] - beta2 * c_X_GLM[DATA_T + t_offset + 0];			
			meaneps += s_Y[threadIdx.y][1][threadIdx.x] - beta1 * c_X_GLM[t_offset + 1] - beta2 * c_X_GLM[DATA_T + t_offset + 1];			
			meaneps += s_Y[threadIdx.y][2][threadIdx.x] - beta1 * c_X_GLM[t_offset + 2] - beta2 * c_X_GLM[DATA_T + t_offset + 2];			
			meaneps += s_Y[threadIdx.y][3][threadIdx.x] - beta1 * c_X_GLM[t_offset + 3] - beta2 * c_X_GLM[DATA_T + t_offset + 3];			
			meaneps += s_Y[threadIdx.y][4][threadIdx.x] - beta1 * c_X_GLM[t_offset + 4] - beta2 * c_X_GLM[DATA_T + t_offset + 4];			
			meaneps += s_Y[threadIdx.y][5][threadIdx.x] - beta1 * c_X_GLM[t_offset + 5] - beta2 * c_X_GLM[DATA_T + t_offset + 5];			
			meaneps += s_Y[threadIdx.y][6][threadIdx.x] - beta1 * c_X_GLM[t_offset + 6] - beta2 * c_X_GLM[DATA_T + t_offset + 6];			
			meaneps += s_Y[threadIdx.y][7][threadIdx.x] - beta1 * c_X_GLM[t_offset + 7] - beta2 * c_X_GLM[DATA_T + t_offset + 7];			
		}

		t_offset = timeMultiples * 8;	

		for (int t = 0; t < timeRest; t++)
		{
			s_Y[threadIdx.y][0][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + t, DATA_W, DATA_H, DATA_D)];
			meaneps += s_Y[threadIdx.y][0][threadIdx.x] - beta1 * c_X_GLM[t_offset + t] - beta2 * c_X_GLM[DATA_T + t_offset + t];			
		}

		meaneps /= (float)DATA_T;

		vareps = 0.0f;
		t_offset = 0;

		// Now calculate the variance of eps
		for (t_offset = 0; t_offset < timeMultiples * 8; t_offset += 8)
		{ 
			// Read data into shared memory
			s_Y[threadIdx.y][0][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 0, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][1][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 1, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][2][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 2, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][3][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 3, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][4][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 4, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][5][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 5, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][6][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 6, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][7][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + 7, DATA_W, DATA_H, DATA_D)];

			__syncthreads();
	
			eps = s_Y[threadIdx.y][0][threadIdx.x] - beta1 * c_X_GLM[t_offset + 0] - beta2 * c_X_GLM[DATA_T + t_offset + 0];			
			vareps += (eps - meaneps) * (eps - meaneps);

			eps = s_Y[threadIdx.y][1][threadIdx.x] - beta1 * c_X_GLM[t_offset + 1] - beta2 * c_X_GLM[DATA_T + t_offset + 1];					
			vareps += (eps - meaneps) * (eps - meaneps);

			eps = s_Y[threadIdx.y][2][threadIdx.x] - beta1 * c_X_GLM[t_offset + 2] - beta2 * c_X_GLM[DATA_T + t_offset + 2];					
			vareps += (eps - meaneps) * (eps - meaneps);

			eps = s_Y[threadIdx.y][3][threadIdx.x] - beta1 * c_X_GLM[t_offset + 3] - beta2 * c_X_GLM[DATA_T + t_offset + 3];					
			vareps += (eps - meaneps) * (eps - meaneps);

			eps = s_Y[threadIdx.y][4][threadIdx.x] - beta1 * c_X_GLM[t_offset + 4] - beta2 * c_X_GLM[DATA_T + t_offset + 4];					
			vareps += (eps - meaneps) * (eps - meaneps);

			eps = s_Y[threadIdx.y][5][threadIdx.x] - beta1 * c_X_GLM[t_offset + 5] - beta2 * c_X_GLM[DATA_T + t_offset + 5];					
			vareps += (eps - meaneps) * (eps - meaneps);

			eps = s_Y[threadIdx.y][6][threadIdx.x] - beta1 * c_X_GLM[t_offset + 6] - beta2 * c_X_GLM[DATA_T + t_offset + 6];					
			vareps += (eps - meaneps) * (eps - meaneps);

			eps = s_Y[threadIdx.y][7][threadIdx.x] - beta1 * c_X_GLM[t_offset + 7] - beta2 * c_X_GLM[DATA_T + t_offset + 7];					
			vareps += (eps - meaneps) * (eps - meaneps);
		}

		t_offset = timeMultiples * 8;	

		for (int t = 0; t < timeRest; t++)
		{
			s_Y[threadIdx.y][0][threadIdx.x] = fMRI_Volumes[Calculate_4D_Index(x, y, z, t_offset + t, DATA_W, DATA_H, DATA_D)];
			eps = s_Y[threadIdx.y][0][threadIdx.x] - beta1 * c_X_GLM[t_offset + t] - beta2 * c_X_GLM[DATA_T + t_offset + t];			
			vareps += (eps - meaneps) * (eps - meaneps);
		}

		vareps /= ((float)DATA_T - 3.0f);

		//activity_map[x + y * DATA_W + z * DATA_W * DATA_H] = (beta1 - beta2) / (sqrt(vareps * ctxtxc));		
		activity_map[x + y * DATA_W + z * DATA_W * DATA_H] = beta1 / (sqrtf(vareps * ctxtxc));				
	}
	else
	{
		activity_map[x + y * DATA_W + z * DATA_W * DATA_H] = 0.0f;
	}
}









__device__ __constant__ unsigned short int c_Permutation_Vector[1000];

__global__ void Regress_BOLD(float* regressed_fMRI_volumes, float* fMRI_volumes, float* Brain_Voxels, int DATA_W, int DATA_H, int DATA_D, int DATA_T, int blocksInY, float invBlocksInY, int timeMultiples, int timeRest)
{
	unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
	unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
	volatile unsigned int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	volatile unsigned int y = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
	volatile unsigned int z = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;
	
	if ((x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D))
		return;

	//if ( Brain_Voxels[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
	//{
		__shared__ float s_Y[8][8][32]; // y,t,x
	
		int t_offset;
		float beta1 = 0.0f;
		float beta2 = 0.0f;

		// Calculate betahat, i.e. multiply (x^T x)^(-1) x^T with Y
		for (t_offset = 0; t_offset < timeMultiples * 8; t_offset += 8)
		{ 
			// Load the current voxel timeseries into shared memory 
			s_Y[threadIdx.y][0][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 0, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][1][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 1, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][2][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 2, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][3][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 3, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][4][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 4, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][5][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 5, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][6][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 6, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][7][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 7, DATA_W, DATA_H, DATA_D)];
	
			__syncthreads();
		
			// Sum and multiply the values in shared memory
			beta1 += s_Y[threadIdx.y][0][threadIdx.x] * c_xtxxt_GLM[t_offset + 0];
			beta1 += s_Y[threadIdx.y][1][threadIdx.x] * c_xtxxt_GLM[t_offset + 1];
			beta1 += s_Y[threadIdx.y][2][threadIdx.x] * c_xtxxt_GLM[t_offset + 2];
			beta1 += s_Y[threadIdx.y][3][threadIdx.x] * c_xtxxt_GLM[t_offset + 3];
			beta1 += s_Y[threadIdx.y][4][threadIdx.x] * c_xtxxt_GLM[t_offset + 4];
			beta1 += s_Y[threadIdx.y][5][threadIdx.x] * c_xtxxt_GLM[t_offset + 5];
			beta1 += s_Y[threadIdx.y][6][threadIdx.x] * c_xtxxt_GLM[t_offset + 6];
			beta1 += s_Y[threadIdx.y][7][threadIdx.x] * c_xtxxt_GLM[t_offset + 7];
	
			beta2 += s_Y[threadIdx.y][0][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + 0];
			beta2 += s_Y[threadIdx.y][1][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + 1];
			beta2 += s_Y[threadIdx.y][2][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + 2];
			beta2 += s_Y[threadIdx.y][3][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + 3];
			beta2 += s_Y[threadIdx.y][4][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + 4];
			beta2 += s_Y[threadIdx.y][5][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + 5];
			beta2 += s_Y[threadIdx.y][6][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + 6];
			beta2 += s_Y[threadIdx.y][7][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + 7];
		}
	
		t_offset = timeMultiples * 8;	

		for (int t = 0; t < timeRest; t++)
		{
			s_Y[threadIdx.y][0][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + t, DATA_W, DATA_H, DATA_D)];
			beta1 += s_Y[threadIdx.y][0][threadIdx.x] * c_xtxxt_GLM[t_offset + t];
			beta2 += s_Y[threadIdx.y][0][threadIdx.x] * c_xtxxt_GLM[DATA_T + t_offset + t];
		}
	
		for (t_offset = 0; t_offset < timeMultiples * 8; t_offset += 8)
		{ 
			// Load the current voxel timeseries into shared memory 
			s_Y[threadIdx.y][0][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 0, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][1][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 1, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][2][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 2, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][3][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 3, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][4][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 4, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][5][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 5, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][6][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 6, DATA_W, DATA_H, DATA_D)];
			s_Y[threadIdx.y][7][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 7, DATA_W, DATA_H, DATA_D)];
	
			__syncthreads();
		
			// Calculate eps for each timesample and store in shared memory (timesample not needed in shared memory any more)	
			s_Y[threadIdx.y][0][threadIdx.x] = s_Y[threadIdx.y][0][threadIdx.x] - beta1 * c_X_GLM[t_offset + 0] - beta2 * c_X_GLM[DATA_T + t_offset + 0];			
			s_Y[threadIdx.y][1][threadIdx.x] = s_Y[threadIdx.y][1][threadIdx.x] - beta1 * c_X_GLM[t_offset + 1] - beta2 * c_X_GLM[DATA_T + t_offset + 1];			
			s_Y[threadIdx.y][2][threadIdx.x] = s_Y[threadIdx.y][2][threadIdx.x] - beta1 * c_X_GLM[t_offset + 2] - beta2 * c_X_GLM[DATA_T + t_offset + 2];			
			s_Y[threadIdx.y][3][threadIdx.x] = s_Y[threadIdx.y][3][threadIdx.x] - beta1 * c_X_GLM[t_offset + 3] - beta2 * c_X_GLM[DATA_T + t_offset + 3];			
			s_Y[threadIdx.y][4][threadIdx.x] = s_Y[threadIdx.y][4][threadIdx.x] - beta1 * c_X_GLM[t_offset + 4] - beta2 * c_X_GLM[DATA_T + t_offset + 4];			
			s_Y[threadIdx.y][5][threadIdx.x] = s_Y[threadIdx.y][5][threadIdx.x] - beta1 * c_X_GLM[t_offset + 5] - beta2 * c_X_GLM[DATA_T + t_offset + 5];			
			s_Y[threadIdx.y][6][threadIdx.x] = s_Y[threadIdx.y][6][threadIdx.x] - beta1 * c_X_GLM[t_offset + 6] - beta2 * c_X_GLM[DATA_T + t_offset + 6];			
			s_Y[threadIdx.y][7][threadIdx.x] = s_Y[threadIdx.y][7][threadIdx.x] - beta1 * c_X_GLM[t_offset + 7] - beta2 * c_X_GLM[DATA_T + t_offset + 7];		

			__syncthreads();

			// Write the eps back global memory
			regressed_fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 0, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][0][threadIdx.x];
			regressed_fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 1, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][1][threadIdx.x];
			regressed_fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 2, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][2][threadIdx.x];
			regressed_fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 3, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][3][threadIdx.x];
			regressed_fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 4, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][4][threadIdx.x];
			regressed_fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 5, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][5][threadIdx.x];
			regressed_fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 6, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][6][threadIdx.x];
			regressed_fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + 7, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][7][threadIdx.x];
		}
	
		t_offset = timeMultiples * 8;	
	
		for (int t = 0; t < timeRest; t++)
		{
			s_Y[threadIdx.y][0][threadIdx.x] = fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + t, DATA_W, DATA_H, DATA_D)];
			regressed_fMRI_volumes[Calculate_4D_Index(x, y, z, t_offset + t, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][0][threadIdx.x] - beta1 * c_X_GLM[t_offset + t] - beta2 * c_X_GLM[DATA_T + t_offset + t];			
		}
	//}
}

__device__ float Determinant_(float Cxx[4][4])
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

__device__ void Invert_4x4(float Cxx[4][4], float inv_Cxx[4][4])
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

__global__ void EstimateAR4BrainVoxels(float* alphas1, float* alphas2, float* alphas3, float* alphas4, float* detrended_fMRI_volumes, float* Brain_Voxels, int DATA_W, int DATA_H, int DATA_D, int DATA_T, int blocksInY, float invBlocksInY)
{
    unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
    unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
    unsigned int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    unsigned int y = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
    unsigned int z = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Brain_Voxels[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
    {
        __shared__ float s_Y[8][8][32]; // y,t,x
        int t = 0;
        float c0 = 0.0f;
        float c1 = 0.0f;
        float c2 = 0.0f;
        float c3 = 0.0f;
        float c4 = 0.0f;

        s_Y[threadIdx.y][0][threadIdx.x] = detrended_fMRI_volumes[Calculate_4D_Index(x, y, z, 0, DATA_W, DATA_H, DATA_D)];
        __syncthreads();
        c0 += s_Y[threadIdx.y][0][threadIdx.x] * s_Y[threadIdx.y][0][threadIdx.x];
        s_Y[threadIdx.y][1][threadIdx.x] = detrended_fMRI_volumes[Calculate_4D_Index(x, y, z, 1, DATA_W, DATA_H, DATA_D)];
        __syncthreads();
        c0 += s_Y[threadIdx.y][1][threadIdx.x] * s_Y[threadIdx.y][1][threadIdx.x];
        c1 += s_Y[threadIdx.y][1][threadIdx.x] * s_Y[threadIdx.y][0][threadIdx.x];
        s_Y[threadIdx.y][2][threadIdx.x] = detrended_fMRI_volumes[Calculate_4D_Index(x, y, z, 2, DATA_W, DATA_H, DATA_D)];
        __syncthreads();
        c0 += s_Y[threadIdx.y][2][threadIdx.x] * s_Y[threadIdx.y][2][threadIdx.x];
        c1 += s_Y[threadIdx.y][2][threadIdx.x] * s_Y[threadIdx.y][1][threadIdx.x];
        c2 += s_Y[threadIdx.y][2][threadIdx.x] * s_Y[threadIdx.y][0][threadIdx.x];
        s_Y[threadIdx.y][3][threadIdx.x] = detrended_fMRI_volumes[Calculate_4D_Index(x, y, z, 3, DATA_W, DATA_H, DATA_D)];
        __syncthreads();
        c0 += s_Y[threadIdx.y][3][threadIdx.x] * s_Y[threadIdx.y][3][threadIdx.x];
        c1 += s_Y[threadIdx.y][3][threadIdx.x] * s_Y[threadIdx.y][2][threadIdx.x];
        c2 += s_Y[threadIdx.y][3][threadIdx.x] * s_Y[threadIdx.y][1][threadIdx.x];
        c3 += s_Y[threadIdx.y][3][threadIdx.x] * s_Y[threadIdx.y][0][threadIdx.x];


        // Estimate c0, c1, c2, c3, c4
        for (t = 4; t < DATA_T; t++)
        {
            // Read data into shared memory
            s_Y[threadIdx.y][4][threadIdx.x] = detrended_fMRI_volumes[Calculate_4D_Index(x, y, z, t, DATA_W, DATA_H, DATA_D)];

            __syncthreads();
   
            // Sum and multiply the values in shared memory
            c0 += s_Y[threadIdx.y][4][threadIdx.x] * s_Y[threadIdx.y][4][threadIdx.x];   
            c1 += s_Y[threadIdx.y][4][threadIdx.x] * s_Y[threadIdx.y][3][threadIdx.x];
            c2 += s_Y[threadIdx.y][4][threadIdx.x] * s_Y[threadIdx.y][2][threadIdx.x];
            c3 += s_Y[threadIdx.y][4][threadIdx.x] * s_Y[threadIdx.y][1][threadIdx.x];
            c4 += s_Y[threadIdx.y][4][threadIdx.x] * s_Y[threadIdx.y][0][threadIdx.x];

            s_Y[threadIdx.y][0][threadIdx.x] = s_Y[threadIdx.y][1][threadIdx.x];
            s_Y[threadIdx.y][1][threadIdx.x] = s_Y[threadIdx.y][2][threadIdx.x];
            s_Y[threadIdx.y][2][threadIdx.x] = s_Y[threadIdx.y][3][threadIdx.x];
            s_Y[threadIdx.y][3][threadIdx.x] = s_Y[threadIdx.y][4][threadIdx.x];
        }



        c0 /= ((float)(DATA_T) - 1.0f);
        c1 /= ((float)(DATA_T) - 2.0f);
        c2 /= ((float)(DATA_T) - 3.0f);
        c3 /= ((float)(DATA_T) - 4.0f);
        c4 /= ((float)(DATA_T) - 5.0f);

        // Calculate alphas
        float r1, r2, r3, r4, alpha1, alpha2, alpha3, alpha4;
   
        if (c0 != 0.0f)
        {
            r1 = c1/c0;
            r2 = c2/c0;
            r3 = c3/c0;
            r4 = c4/c0;
           
            float matrix[4][4];
            matrix[0][0] = 1.0f;
            matrix[1][0] = r1 + 0.001f;
            matrix[2][0] = r2 + 0.001f;
            matrix[3][0] = r3 + 0.001f;

            matrix[0][1] = r1 + 0.001f;
            matrix[1][1] = 1.0f;
            matrix[2][1] = r1 + 0.001f;
            matrix[3][1] = r2 + 0.001f;

            matrix[0][2] = r2 + 0.001f;
            matrix[1][2] = r1 + 0.001f;
            matrix[2][2] = 1.0f;
            matrix[3][2] = r1 + 0.001f;

            matrix[0][3] = r3 + 0.001f;
            matrix[1][3] = r2 + 0.001f;
            matrix[2][3] = r1 + 0.001f;
            matrix[3][3] = 1.0f;

            float inv_matrix[4][4];
           
            Invert_4x4(matrix, inv_matrix);

            alpha1 = inv_matrix[0][0] * r1 + inv_matrix[0][1] * r2 + inv_matrix[0][2] * r3 + inv_matrix[0][3] * r4;
            alpha2 = inv_matrix[1][0] * r1 + inv_matrix[1][1] * r2 + inv_matrix[1][2] * r3 + inv_matrix[1][3] * r4;
            alpha3 = inv_matrix[2][0] * r1 + inv_matrix[2][1] * r2 + inv_matrix[2][2] * r3 + inv_matrix[2][3] * r4;
            alpha4 = inv_matrix[3][0] * r1 + inv_matrix[3][1] * r2 + inv_matrix[3][2] * r3 + inv_matrix[3][3] * r4;

            alphas1[x + y * DATA_W + z * DATA_W * DATA_H] = alpha1;
            alphas2[x + y * DATA_W + z * DATA_W * DATA_H] = alpha2;       
            alphas3[x + y * DATA_W + z * DATA_W * DATA_H] = alpha3;       
            alphas4[x + y * DATA_W + z * DATA_W * DATA_H] = alpha4;                   
        }
        else
        {
            alphas1[x + y * DATA_W + z * DATA_W * DATA_H] = 0.0f;
            alphas2[x + y * DATA_W + z * DATA_W * DATA_H] = 0.0f;       
            alphas3[x + y * DATA_W + z * DATA_W * DATA_H] = 0.0f;       
            alphas4[x + y * DATA_W + z * DATA_W * DATA_H] = 0.0f;       
        }
    }   
    else
    {
        alphas1[x + y * DATA_W + z * DATA_W * DATA_H] = 0.0f;
        alphas2[x + y * DATA_W + z * DATA_W * DATA_H] = 0.0f;       
        alphas3[x + y * DATA_W + z * DATA_W * DATA_H] = 0.0f;       
        alphas4[x + y * DATA_W + z * DATA_W * DATA_H] = 0.0f;       
    }
}

__global__ void ApplyWhiteningAR4(float* whitened_fMRI_volumes, float* detrended_fMRI_volumes, float* alphas1, float* alphas2, float* alphas3, float* alphas4, float* Brain_Voxels, int DATA_W, int DATA_H, int DATA_D, int DATA_T, int blocksInY, float invBlocksInY)
{
    unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
    unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
    unsigned int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    unsigned int y = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
    unsigned int z = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Brain_Voxels[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
    {       
        __shared__ float s_Y[8][8][32]; // y,t,x
        int t = 0;

        float alpha1 = alphas1[x + y * DATA_W + z * DATA_W * DATA_H];
        float alpha2 = alphas2[x + y * DATA_W + z * DATA_W * DATA_H];
        float alpha3 = alphas3[x + y * DATA_W + z * DATA_W * DATA_H];
        float alpha4 = alphas4[x + y * DATA_W + z * DATA_W * DATA_H];

        // Calculate the whitened timeseries

        s_Y[threadIdx.y][0][threadIdx.x] = detrended_fMRI_volumes[Calculate_4D_Index(x, y, z, 0, DATA_W, DATA_H, DATA_D)];
        __syncthreads();
        whitened_fMRI_volumes[Calculate_4D_Index(x, y, z, 0, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][0][threadIdx.x];
        s_Y[threadIdx.y][1][threadIdx.x] = detrended_fMRI_volumes[Calculate_4D_Index(x, y, z, 1, DATA_W, DATA_H, DATA_D)];
        __syncthreads();
        whitened_fMRI_volumes[Calculate_4D_Index(x, y, z, 1, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][1][threadIdx.x]  - alpha1 * s_Y[threadIdx.y][0][threadIdx.x];
        s_Y[threadIdx.y][2][threadIdx.x] = detrended_fMRI_volumes[Calculate_4D_Index(x, y, z, 2, DATA_W, DATA_H, DATA_D)];
        __syncthreads();
        whitened_fMRI_volumes[Calculate_4D_Index(x, y, z, 2, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][2][threadIdx.x] - alpha1 * s_Y[threadIdx.y][1][threadIdx.x] - alpha2 * s_Y[threadIdx.y][0][threadIdx.x];
        s_Y[threadIdx.y][3][threadIdx.x] = detrended_fMRI_volumes[Calculate_4D_Index(x, y, z, 3, DATA_W, DATA_H, DATA_D)];
        __syncthreads();
        whitened_fMRI_volumes[Calculate_4D_Index(x, y, z, 3, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][3][threadIdx.x] - alpha1 * s_Y[threadIdx.y][2][threadIdx.x] - alpha2 * s_Y[threadIdx.y][1][threadIdx.x] - alpha3 * s_Y[threadIdx.y][0][threadIdx.x];

        for (t = 4; t < DATA_T; t++)
        {
            // Read data into shared memory
            s_Y[threadIdx.y][4][threadIdx.x] = detrended_fMRI_volumes[Calculate_4D_Index(x, y, z, t, DATA_W, DATA_H, DATA_D)];
                   
           __syncthreads();

            whitened_fMRI_volumes[Calculate_4D_Index(x, y, z, t, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][4][threadIdx.x] - alpha1 * s_Y[threadIdx.y][3][threadIdx.x] - alpha2 * s_Y[threadIdx.y][2][threadIdx.x] - alpha3 * s_Y[threadIdx.y][1][threadIdx.x] - alpha4 * s_Y[threadIdx.y][0][threadIdx.x];

            s_Y[threadIdx.y][0][threadIdx.x] = s_Y[threadIdx.y][1][threadIdx.x];
            s_Y[threadIdx.y][1][threadIdx.x] = s_Y[threadIdx.y][2][threadIdx.x];           
            s_Y[threadIdx.y][2][threadIdx.x] = s_Y[threadIdx.y][3][threadIdx.x];           
            s_Y[threadIdx.y][3][threadIdx.x] = s_Y[threadIdx.y][4][threadIdx.x];           
        }
    }
}

__global__ void ResetVolumes(float* reset_volumes, float* Brain_Voxels, int DATA_W, int DATA_H, int DATA_D, int DATA_T, int blocksInY, float invBlocksInY)
{
    unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
    unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
    unsigned int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    unsigned int y = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
    unsigned int z = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;


    for (int t = 0; t < DATA_T; t++)
    {
        reset_volumes[Calculate_4D_Index(x, y, z, t, DATA_W, DATA_H, DATA_D)] = 0.0f;
    }

}

__global__ void GeneratePermutedfMRIVolumesAR4(float* permuted_fMRI_volumes, float* alphas1, float* alphas2, float* alphas3, float* alphas4, float* whitened_fMRI_volumes, float* Brain_Voxels, int DATA_W, int DATA_H, int DATA_D, int DATA_T, int blocksInY, float invBlocksInY)
{
    unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
    unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
    unsigned int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    unsigned int y = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
    unsigned int z = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Brain_Voxels[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
    {
        __shared__ float s_Y[8][8][32]; // y,t,x
        int t = 0;
        float alpha1 = alphas1[x + y * DATA_W + z * DATA_W * DATA_H];       
        float alpha2 = alphas2[x + y * DATA_W + z * DATA_W * DATA_H];       
        float alpha3 = alphas3[x + y * DATA_W + z * DATA_W * DATA_H];       
        float alpha4 = alphas4[x + y * DATA_W + z * DATA_W * DATA_H];       

        s_Y[threadIdx.y][0][threadIdx.x] = whitened_fMRI_volumes[Calculate_4D_Index(x, y, z, c_Permutation_Vector[0], DATA_W, DATA_H, DATA_D)];
        __syncthreads();
        s_Y[threadIdx.y][1][threadIdx.x] = alpha1 * s_Y[threadIdx.y][0][threadIdx.x]  + whitened_fMRI_volumes[Calculate_4D_Index(x, y, z, c_Permutation_Vector[1], DATA_W, DATA_H, DATA_D)];
        __syncthreads();
        s_Y[threadIdx.y][2][threadIdx.x] = alpha1 * s_Y[threadIdx.y][1][threadIdx.x]  + alpha2 * s_Y[threadIdx.y][0][threadIdx.x] + whitened_fMRI_volumes[Calculate_4D_Index(x, y, z, c_Permutation_Vector[2], DATA_W, DATA_H, DATA_D)];
        __syncthreads();
        s_Y[threadIdx.y][3][threadIdx.x] = alpha1 * s_Y[threadIdx.y][2][threadIdx.x]  + alpha2 * s_Y[threadIdx.y][1][threadIdx.x] + alpha3 * s_Y[threadIdx.y][0][threadIdx.x] + whitened_fMRI_volumes[Calculate_4D_Index(x, y, z, c_Permutation_Vector[3], DATA_W, DATA_H, DATA_D)];


        permuted_fMRI_volumes[Calculate_4D_Index(x, y, z, 0, DATA_W, DATA_H, DATA_D)] =  s_Y[threadIdx.y][0][threadIdx.x];
        permuted_fMRI_volumes[Calculate_4D_Index(x, y, z, 1, DATA_W, DATA_H, DATA_D)] =  s_Y[threadIdx.y][1][threadIdx.x];
        permuted_fMRI_volumes[Calculate_4D_Index(x, y, z, 2, DATA_W, DATA_H, DATA_D)] =  s_Y[threadIdx.y][2][threadIdx.x];
        permuted_fMRI_volumes[Calculate_4D_Index(x, y, z, 3, DATA_W, DATA_H, DATA_D)] =  s_Y[threadIdx.y][3][threadIdx.x];

        // Read the data in a permuted order and apply an inverse whitening transform
        for (t = 4; t < DATA_T; t++)
        {
            // Calculate the unwhitened, permuted, timeseries           
            s_Y[threadIdx.y][4][threadIdx.x] = alpha1 * s_Y[threadIdx.y][3][threadIdx.x] + alpha2 * s_Y[threadIdx.y][2][threadIdx.x] + alpha3 * s_Y[threadIdx.y][1][threadIdx.x] + alpha4 * s_Y[threadIdx.y][0][threadIdx.x] + whitened_fMRI_volumes[Calculate_4D_Index(x, y, z, c_Permutation_Vector[t], DATA_W, DATA_H, DATA_D)];   

            __syncthreads();

            permuted_fMRI_volumes[Calculate_4D_Index(x, y, z, t, DATA_W, DATA_H, DATA_D)] = s_Y[threadIdx.y][4][threadIdx.x];

            // Save old values
            s_Y[threadIdx.y][0][threadIdx.x] = s_Y[threadIdx.y][1][threadIdx.x];
            s_Y[threadIdx.y][1][threadIdx.x] = s_Y[threadIdx.y][2][threadIdx.x];
            s_Y[threadIdx.y][2][threadIdx.x] = s_Y[threadIdx.y][3][threadIdx.x];
            s_Y[threadIdx.y][3][threadIdx.x] = s_Y[threadIdx.y][4][threadIdx.x];
        }
    }
}

__global__ void FindSliceMax(float* maximum_slice_values, float* activity_volume, int DATA_W, int DATA_H, int DATA_D, int blocksInY, float invBlocksInY)
{
    unsigned int blockIdxz = __float2uint_rd(blockIdx.y * invBlocksInY);
    unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
    unsigned int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    unsigned int y = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
    unsigned int z = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;    

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
    	return;

    __shared__ float s_a[64][64]; // y,x (32 threads in x, 16 in y)

    // Read data into shared memory (one slice)
    s_a[threadIdx.y][threadIdx.x] = activity_volume[Calculate_3D_Index(x, y, z, DATA_W, DATA_H)];		
    s_a[threadIdx.y + 16][threadIdx.x] = activity_volume[Calculate_3D_Index(x, y + 16, z, DATA_W, DATA_H)];
    s_a[threadIdx.y + 32][threadIdx.x] = activity_volume[Calculate_3D_Index(x, y + 32, z, DATA_W, DATA_H)];
    s_a[threadIdx.y + 48][threadIdx.x] = activity_volume[Calculate_3D_Index(x, y + 48, z, DATA_W, DATA_H)];
    s_a[threadIdx.y][threadIdx.x + 32] = activity_volume[Calculate_3D_Index(x + 32, y, z, DATA_W, DATA_H)];
    s_a[threadIdx.y + 16][threadIdx.x + 32] = activity_volume[Calculate_3D_Index(x + 32, y + 16, z, DATA_W, DATA_H)];
    s_a[threadIdx.y + 32][threadIdx.x + 32] = activity_volume[Calculate_3D_Index(x + 32, y + 32, z, DATA_W, DATA_H)];
    s_a[threadIdx.y + 48][threadIdx.x + 32] = activity_volume[Calculate_3D_Index(x + 32, y + 48, z, DATA_W, DATA_H)];

    __syncthreads();
    
    // First reduction, to 2048 values
    s_a[threadIdx.y][threadIdx.x] = max(s_a[threadIdx.y][threadIdx.x], s_a[threadIdx.y][threadIdx.x + 32]);
    s_a[threadIdx.y + 16][threadIdx.x] = max(s_a[threadIdx.y + 16][threadIdx.x], s_a[threadIdx.y + 16][threadIdx.x + 32]);
    s_a[threadIdx.y + 32][threadIdx.x] = max(s_a[threadIdx.y + 32][threadIdx.x], s_a[threadIdx.y + 32][threadIdx.x + 32]);
    s_a[threadIdx.y + 48][threadIdx.x] = max(s_a[threadIdx.y + 48][threadIdx.x], s_a[threadIdx.y + 48][threadIdx.x + 32]);

    __syncthreads();

    // Second reduction, to 1024 values
    s_a[threadIdx.y][threadIdx.x] = max(s_a[threadIdx.y][threadIdx.x], s_a[threadIdx.y + 16][threadIdx.x]);
    s_a[threadIdx.y + 16][threadIdx.x] = max(s_a[threadIdx.y + 32][threadIdx.x], s_a[threadIdx.y + 48][threadIdx.x]);

    __syncthreads();

    // Third reduction, to 512 values
    s_a[threadIdx.y][threadIdx.x] = max(s_a[threadIdx.y][threadIdx.x], s_a[threadIdx.y + 16][threadIdx.x]);

    __syncthreads();

    // Fourth reduction, to 256 values
    if (threadIdx.y < 8)
    {
		s_a[threadIdx.y][threadIdx.x] = max(s_a[threadIdx.y][threadIdx.x], s_a[threadIdx.y + 8][threadIdx.x]);
    }

    __syncthreads();

    // Fifth reduction, to 128 values
    if (threadIdx.y < 4)
    {
		s_a[threadIdx.y][threadIdx.x] = max(s_a[threadIdx.y][threadIdx.x], s_a[threadIdx.y + 4][threadIdx.x]);
    }

    __syncthreads();

    // Sixth reduction, to 64 values
    if (threadIdx.y < 2)
    {
		s_a[threadIdx.y][threadIdx.x] = max(s_a[threadIdx.y][threadIdx.x], s_a[threadIdx.y + 2][threadIdx.x]);
    }

    __syncthreads();

    // Seventh reduction, to 32 values
    if (threadIdx.y == 0)
    {
		s_a[0][threadIdx.x] = max(s_a[0][threadIdx.x], s_a[1][threadIdx.x]);
    }

    __syncthreads();

    // Eigth reduction, to 16 values
    if ((threadIdx.y == 0) && (threadIdx.x < 16))
    {
		s_a[0][threadIdx.x] = max(s_a[0][threadIdx.x], s_a[0][threadIdx.x + 16]);
    }

    __syncthreads();

    // Ninth reduction, to 8 values
    if ((threadIdx.y == 0) && (threadIdx.x < 8))
    {
		s_a[0][threadIdx.x] = max(s_a[0][threadIdx.x], s_a[0][threadIdx.x + 8]);
    }

    __syncthreads();

    // Tenth reduction, to 4 values
    if ((threadIdx.y == 0) && (threadIdx.x < 4))
    {
	    s_a[0][threadIdx.x] = max(s_a[0][threadIdx.x], s_a[0][threadIdx.x + 4]);
    }

    __syncthreads();

    // Eleventh reduction, to 2 values
    if ((threadIdx.y == 0) && (threadIdx.x < 2))
    {
	    s_a[0][threadIdx.x] = max(s_a[0][threadIdx.x], s_a[0][threadIdx.x + 2]);
    }

    __syncthreads();

    // Final reduction
    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {
	    maximum_slice_values[z] = max(s_a[0][0], s_a[0][1]);
    }
}







