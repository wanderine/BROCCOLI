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
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
    DEALINGS IN THE SOFTWARE.
*/

//#include "broccoli_lib.h"
//#include <opencl.h>


// Help functions
int Calculate2DIndex(int x, int y, int DATA_W)
{
	return x + y * DATA_W;
}

int Calculate3DIndex(int x, int y, int z, int DATA_W, int DATA_H)
{
	return x + y * DATA_W + z * DATA_W * DATA_H;
}

int Calculate4DIndex(int x, int y, int z, int t, int DATA_W, int DATA_H, int DATA_D)
{
	return x + y * DATA_W + z * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D;
}

void GetParameterIndices(int* i, int* j, int parameter)
{
	switch(parameter)
	{
		case 0:
			*i = 0; *j = 0;
			break;

		case 1:
			*i = 3; *j = 0;
			break;

		case 2:
			*i = 4; *j = 0;
			break;

		case 3:
			*i = 5; *j = 0;
			break;

		case 4:
			*i = 3; *j = 3;
			break;

		case 5:
			*i = 4; *j = 3;
			break;

		case 6:
			*i = 5; *j = 3;
			break;

		case 7:
			*i = 4; *j = 4;
			break;

		case 8:
			*i = 5; *j = 4;
			break;

		case 9:
			*i = 5; *j = 5;
			break;

		case 10:
			*i = 1; *j = 1;
			break;

		case 11:
			*i = 6; *j = 1;
			break;

		case 12:
			*i = 7; *j = 1;
			break;

		case 13:
			*i = 8; *j = 1;
			break;

		case 14:
			*i = 6; *j = 6;
			break;

		case 15:
			*i = 7; *j = 6;
			break;

		case 16:
			*i = 8; *j = 6;
			break;

		case 17:
			*i = 7; *j = 7;
			break;

		case 18:
			*i = 8; *j = 7;
			break;

		case 19:
			*i = 8; *j = 8;
			break;

		case 20:
			*i = 2; *j = 2;
			break;

		case 21:
			*i = 9; *j = 2;
			break;

		case 22:
			*i = 10; *j = 2;
			break;

		case 23:
			*i = 11; *j = 2;
			break;

		case 24:
			*i = 9; *j = 9;
			break;

		case 25:
			*i = 10; *j = 9;
			break;

		case 26:
			*i = 11; *j = 9;
			break;

		case 27:
			*i = 10; *j = 10;
			break;

		case 28:
			*i = 11; *j = 10;
			break;

		case 29:
			*i = 11; *j = 11;
			break;

		default:
			*i = 0; *j = 0;
			break;
	}
}



// Convolution functions

// Separable 3D convolution

__kernel void SeparableConvolutionRows(__global float *Filter_Response, __global const float* Volume, __constant float *c_Smoothing_Filter_Y, __private int t, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int DATA_T)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_local_size(2) * get_group_id(2) * 4 + get_local_id(2);
	//int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	//volatile int x = blockIdx.x * blockDim.x + tIdx.x;
	//volatile int y = blockIdx.y * blockDim.y + tIdx.y;
	//volatile int z = blockIdx.z * blockDim.z * 4 + tIdx.z;

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

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && (z < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y][tIdx.x] = Volume[Calculate4DIndex(x,y - 4,z,t,DATA_W, DATA_H, DATA_D)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 2) < DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y][tIdx.x] = Volume[Calculate4DIndex(x,y - 4,z + 2,t,DATA_W, DATA_H, DATA_D)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y][tIdx.x] = Volume[Calculate4DIndex(x,y - 4,z + 4,t,DATA_W, DATA_H, DATA_D)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 6) < DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y][tIdx.x] = Volume[Calculate4DIndex(x,y - 4,z + 6,t,DATA_W, DATA_H, DATA_D)];
	}

	// Second half main data + lower apron

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && (z < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 8][tIdx.x] = Volume[Calculate4DIndex(x,y + 4,z,t,DATA_W, DATA_H, DATA_D)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 2) < DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x] = Volume[Calculate4DIndex(x,y + 4,z + 2,t,DATA_W, DATA_H, DATA_D)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x] = Volume[Calculate4DIndex(x,y + 4,z + 4,t,DATA_W, DATA_H, DATA_D)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 6) < DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x] = Volume[Calculate4DIndex(x,y + 4,z + 6,t,DATA_W, DATA_H, DATA_D)];
	}

	// Make sure all threads have written to local memory
	barrier(CLK_LOCAL_MEM_FENCE);
		
	// Only threads within the volume do the convolution
	if ( (x < DATA_W) && (y < DATA_H) && (z < DATA_D) )
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
		
		Filter_Response[Calculate3DIndex(x,y,z,DATA_W, DATA_H)] = sum;
		//Filter_Response[Calculate4DIndex(x,y,z,t,DATA_W, DATA_H,DATA_D)] = sum;		
	}

	
	if ( (x < DATA_W) && (y < DATA_H) && ((z + 2) < DATA_D) )
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

		Filter_Response[Calculate3DIndex(x,y,z + 2,DATA_W, DATA_H)] = sum;
		//Filter_Response[Calculate4DIndex(x,y,z + 2,t,DATA_W, DATA_H,DATA_D)] = sum;
	}

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 4) < DATA_D) )
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

		Filter_Response[Calculate3DIndex(x,y,z + 4,DATA_W, DATA_H)] = sum;
		//Filter_Response[Calculate4DIndex(x,y,z + 4,t,DATA_W, DATA_H,DATA_D)] = sum;
	}

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 6) < DATA_D) )
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

		Filter_Response[Calculate3DIndex(x,y,z + 6,DATA_W, DATA_H)] = sum;		
		//Filter_Response[Calculate4DIndex(x,y,z + 6,t,DATA_W, DATA_H,DATA_D)] = sum;
	}
	
}

__kernel void SeparableConvolutionColumns(__global float *Filter_Response, __global float* Volume, __constant float *c_Smoothing_Filter_X, __private int t, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int DATA_T)
{
	int x = get_local_size(0) * get_group_id(0) / 32 * 24 + get_local_id(0);;
	int y = get_local_size(1) * get_group_id(1) * 2 + get_local_id(1);
	int z = get_local_size(2) * get_group_id(2) * 4 + get_local_id(2);  

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	//volatile int x = blockIdx.x * blockDim.x / 32 * 24 + tIdx.x;
	//volatile int y = blockIdx.y * blockDim.y * 2 + tIdx.y;
	//volatile int z = blockIdx.z * blockDim.z * 4 + tIdx.z;

	// 8 * 16 * 24 valid filter responses = 3072
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

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x - 4,y,z,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && ((z + 2) < DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x - 4,y,z + 2,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x - 4,y,z + 4,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && ((z + 6) < DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x - 4,y,z + 6,DATA_W, DATA_H)];
	}



	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 8) < DATA_H) && (z < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 8][tIdx.x] = Volume[Calculate3DIndex(x - 4,y + 8,z,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 8) < DATA_H) && ((z + 2) < DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x] = Volume[Calculate3DIndex(x - 4,y + 8,z + 2,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 8) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x] = Volume[Calculate3DIndex(x - 4,y + 8,z + 4,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 8) < DATA_H) && ((z + 6) < DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x] = Volume[Calculate3DIndex(x - 4,y + 8,z + 6,DATA_W, DATA_H)];
	}

	// Make sure all threads have written to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Only threads within the volume do the convolution
	if (tIdx.x < 24)
	{
		if ( (x < DATA_W) && (y < DATA_H) && (z < DATA_D) )
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

			Filter_Response[Calculate3DIndex(x,y,z,DATA_W, DATA_H)] = sum;
		}

		if ( (x < DATA_W) && (y < DATA_H) && ((z + 2) < DATA_D) )
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

			Filter_Response[Calculate3DIndex(x,y,z + 2,DATA_W, DATA_H)] = sum;
		}

		if ( (x < DATA_W) && (y < DATA_H) && ((z + 4) < DATA_D) )
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

			Filter_Response[Calculate3DIndex(x,y,z + 4,DATA_W, DATA_H)] = sum;
		}

		if ( (x < DATA_W) && (y < DATA_H) && ((z + 6) < DATA_D) )
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

			Filter_Response[Calculate3DIndex(x,y,z + 6,DATA_W, DATA_H)] = sum;
		}

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && (z < DATA_D) )
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

			Filter_Response[Calculate3DIndex(x,y + 8,z,DATA_W, DATA_H)] = sum;
		}

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 2) < DATA_D) )
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

			Filter_Response[Calculate3DIndex(x,y + 8,z + 2,DATA_W, DATA_H)] = sum;
		}

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 4) < DATA_D) )
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

			Filter_Response[Calculate3DIndex(x,y + 8,z + 4,DATA_W, DATA_H)] = sum;
		}

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 6) < DATA_D) )
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

			Filter_Response[Calculate3DIndex(x,y + 8,z + 6,DATA_W, DATA_H)] = sum;
		}
	}
}

__kernel void SeparableConvolutionRods(__global float *Filter_Response, __global float* Volume, __constant float *c_Smoothing_Filter_Z, __private int t, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int DATA_T)
{
	int x = get_global_id(0);
	int y = get_local_size(1) * get_group_id(1) * 4 + get_local_id(1); 
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	//volatile int x = blockIdx.x * blockDim.x + tIdx.x;
	//volatile int y = blockIdx.y * blockDim.y * 4 + tIdx.y;
	//volatile int z = blockIdx.z * blockDim.z + tIdx.z;

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

	if ( (x < DATA_W) && (y < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x,y,z - 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 2) < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 2][tIdx.x] = Volume[Calculate3DIndex(x,y + 2,z - 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 4][tIdx.x] = Volume[Calculate3DIndex(x,y + 4,z - 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 6) < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 6][tIdx.x] = Volume[Calculate3DIndex(x,y + 6,z - 4,DATA_W, DATA_H)];
	}

	// Second half main data + below apron

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x,y,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 2) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y + 2][tIdx.x] = Volume[Calculate3DIndex(x,y + 2,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y + 4][tIdx.x] = Volume[Calculate3DIndex(x,y + 4,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 6) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y + 6][tIdx.x] = Volume[Calculate3DIndex(x,y + 6,z + 4,DATA_W, DATA_H)];
	}

	// Make sure all threads have written to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Only threads within the volume do the convolution
	if ( (x < DATA_W) && (y < DATA_H) && (z < DATA_D) )
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

		Filter_Response[Calculate4DIndex(x,y,z,t,DATA_W,DATA_H,DATA_D)] = sum;
	}

	if ( (x < DATA_W) && ((y + 2) < DATA_H) && (z < DATA_D) )
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

		Filter_Response[Calculate4DIndex(x,y + 2,z,t,DATA_W,DATA_H,DATA_D)] = sum;
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && (z < DATA_D) )
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

		Filter_Response[Calculate4DIndex(x,y + 4,z,t,DATA_W,DATA_H,DATA_D)] = sum;
	}

	if ( (x < DATA_W) && ((y + 6) < DATA_H) && (z < DATA_D) )
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

		Filter_Response[Calculate4DIndex(x,y + 6,z,t,DATA_W,DATA_H,DATA_D)] = sum;
	}
}

#define HALO 3

#define VALID_FILTER_RESPONSES_X_CONVOLUTION_2D 90
#define VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D 58

// Non-separable 3D convolution

typedef struct tag_float6 {float a; float b; float c; float d; float e; float f;} float6;

float6 Conv_2D_Unrolled_7x7(__local float image[64][96], int y, int x, __constant float Filter_1_Real[7][7], __constant float Filter_1_Imag[7][7], __constant float Filter_2_Real[7][7], __constant float Filter_2_Imag[7][7], __constant float Filter_3_Real[7][7], __constant float Filter_3_Imag[7][7])
{
	float pixel;
	float6 sum;
	sum.a = 0.0f;
	sum.b = 0.0f;
	sum.c = 0.0f;
	sum.d = 0.0f;
	sum.e = 0.0f;
	sum.f = 0.0f;
	
    pixel = image[y - 3][x - 3]; 
    sum.a += pixel * Filter_1_Real[6][6];
	sum.b += pixel * Filter_1_Imag[6][6];
	sum.c += pixel * Filter_2_Real[6][6];
	sum.d += pixel * Filter_2_Imag[6][6];
	sum.e += pixel * Filter_3_Real[6][6];
	sum.f += pixel * Filter_3_Imag[6][6];
    pixel = image[y - 2][x - 3]; 
    sum.a += pixel * Filter_1_Real[5][6];
	sum.b += pixel * Filter_1_Imag[5][6];
	sum.c += pixel * Filter_2_Real[5][6];
	sum.d += pixel * Filter_2_Imag[5][6];
	sum.e += pixel * Filter_3_Real[5][6];
	sum.f += pixel * Filter_3_Imag[5][6];
	pixel = image[y - 1][x - 3]; 
    sum.a += pixel * Filter_1_Real[4][6];
	sum.b += pixel * Filter_1_Imag[4][6];
	sum.c += pixel * Filter_2_Real[4][6];
	sum.d += pixel * Filter_2_Imag[4][6];
	sum.e += pixel * Filter_3_Real[4][6];
	sum.f += pixel * Filter_3_Imag[4][6];
	pixel = image[y + 0][x - 3]; 
    sum.a += pixel * Filter_1_Real[3][6];
	sum.b += pixel * Filter_1_Imag[3][6];
	sum.c += pixel * Filter_2_Real[3][6];
	sum.d += pixel * Filter_2_Imag[3][6];
	sum.e += pixel * Filter_3_Real[3][6];
	sum.f += pixel * Filter_3_Imag[3][6];
    pixel = image[y + 1][x - 3]; 
    sum.a += pixel * Filter_1_Real[2][6];
	sum.b += pixel * Filter_1_Imag[2][6];
	sum.c += pixel * Filter_2_Real[2][6];
	sum.d += pixel * Filter_2_Imag[2][6];
	sum.e += pixel * Filter_3_Real[2][6];
	sum.f += pixel * Filter_3_Imag[2][6];
	pixel = image[y + 2][x - 3]; 
    sum.a += pixel * Filter_1_Real[1][6];
	sum.b += pixel * Filter_1_Imag[1][6];
	sum.c += pixel * Filter_2_Real[1][6];
	sum.d += pixel * Filter_2_Imag[1][6];
	sum.e += pixel * Filter_3_Real[1][6];
	sum.f += pixel * Filter_3_Imag[1][6];
	pixel = image[y + 3][x - 3]; 
    sum.a += pixel * Filter_1_Real[0][6];
	sum.b += pixel * Filter_1_Imag[0][6];
	sum.c += pixel * Filter_2_Real[0][6];
	sum.d += pixel * Filter_2_Imag[0][6];
	sum.e += pixel * Filter_3_Real[0][6];
	sum.f += pixel * Filter_3_Imag[0][6];

    pixel = image[y - 3][x - 2]; 
    sum.a += pixel * Filter_1_Real[6][5];
	sum.b += pixel * Filter_1_Imag[6][5];
	sum.c += pixel * Filter_2_Real[6][5];
	sum.d += pixel * Filter_2_Imag[6][5];
	sum.e += pixel * Filter_3_Real[6][5];
	sum.f += pixel * Filter_3_Imag[6][5];
    pixel = image[y - 2][x - 2]; 
    sum.a += pixel * Filter_1_Real[5][5];
	sum.b += pixel * Filter_1_Imag[5][5];
	sum.c += pixel * Filter_2_Real[5][5];
	sum.d += pixel * Filter_2_Imag[5][5];
	sum.e += pixel * Filter_3_Real[5][5];
	sum.f += pixel * Filter_3_Imag[5][5];
    pixel = image[y - 1][x - 2]; 
    sum.a += pixel * Filter_1_Real[4][5];
	sum.b += pixel * Filter_1_Imag[4][5];
	sum.c += pixel * Filter_2_Real[4][5];
	sum.d += pixel * Filter_2_Imag[4][5];
	sum.e += pixel * Filter_3_Real[4][5];
	sum.f += pixel * Filter_3_Imag[4][5];
    pixel = image[y + 0][x - 2]; 
    sum.a += pixel * Filter_1_Real[3][5];
	sum.b += pixel * Filter_1_Imag[3][5];
	sum.c += pixel * Filter_2_Real[3][5];
	sum.d += pixel * Filter_2_Imag[3][5];
	sum.e += pixel * Filter_3_Real[3][5];
	sum.f += pixel * Filter_3_Imag[3][5];
    pixel = image[y + 1][x - 2]; 
    sum.a += pixel * Filter_1_Real[2][5];
	sum.b += pixel * Filter_1_Imag[2][5];
	sum.c += pixel * Filter_2_Real[2][5];
	sum.d += pixel * Filter_2_Imag[2][5];
	sum.e += pixel * Filter_3_Real[2][5];
	sum.f += pixel * Filter_3_Imag[2][5];
    pixel = image[y + 2][x - 2]; 
    sum.a += pixel * Filter_1_Real[1][5];
	sum.b += pixel * Filter_1_Imag[1][5];
	sum.c += pixel * Filter_2_Real[1][5];
	sum.d += pixel * Filter_2_Imag[1][5];
	sum.e += pixel * Filter_3_Real[1][5];
	sum.f += pixel * Filter_3_Imag[1][5];
    pixel = image[y + 3][x - 2]; 
    sum.a += pixel * Filter_1_Real[0][5];
	sum.b += pixel * Filter_1_Imag[0][5];
	sum.c += pixel * Filter_2_Real[0][5];
	sum.d += pixel * Filter_2_Imag[0][5];
	sum.e += pixel * Filter_3_Real[0][5];
	sum.f += pixel * Filter_3_Imag[0][5];


    pixel = image[y - 3][x - 1]; 
    sum.a += pixel * Filter_1_Real[6][4];
	sum.b += pixel * Filter_1_Imag[6][4];
	sum.c += pixel * Filter_2_Real[6][4];
	sum.d += pixel * Filter_2_Imag[6][4];
	sum.e += pixel * Filter_3_Real[6][4];
	sum.f += pixel * Filter_3_Imag[6][4];
    pixel = image[y - 2][x - 1]; 
    sum.a += pixel * Filter_1_Real[5][4];
	sum.b += pixel * Filter_1_Imag[5][4];
	sum.c += pixel * Filter_2_Real[5][4];
	sum.d += pixel * Filter_2_Imag[5][4];
	sum.e += pixel * Filter_3_Real[5][4];
	sum.f += pixel * Filter_3_Imag[5][4];
    pixel = image[y - 1][x - 1]; 
    sum.a += pixel * Filter_1_Real[4][4];
	sum.b += pixel * Filter_1_Imag[4][4];
	sum.c += pixel * Filter_2_Real[4][4];
	sum.d += pixel * Filter_2_Imag[4][4];
	sum.e += pixel * Filter_3_Real[4][4];
	sum.f += pixel * Filter_3_Imag[4][4];
    pixel = image[y + 0][x - 1]; 
    sum.a += pixel * Filter_1_Real[3][4];
	sum.b += pixel * Filter_1_Imag[3][4];
	sum.c += pixel * Filter_2_Real[3][4];
	sum.d += pixel * Filter_2_Imag[3][4];
	sum.e += pixel * Filter_3_Real[3][4];
	sum.f += pixel * Filter_3_Imag[3][4];
    pixel = image[y + 1][x - 1]; 
    sum.a += pixel * Filter_1_Real[2][4];
	sum.b += pixel * Filter_1_Imag[2][4];
	sum.c += pixel * Filter_2_Real[2][4];
	sum.d += pixel * Filter_2_Imag[2][4];
	sum.e += pixel * Filter_3_Real[2][4];
	sum.f += pixel * Filter_3_Imag[2][4];
    pixel = image[y + 2][x - 1]; 
    sum.a += pixel * Filter_1_Real[1][4];
	sum.b += pixel * Filter_1_Imag[1][4];
	sum.c += pixel * Filter_2_Real[1][4];
	sum.d += pixel * Filter_2_Imag[1][4];
	sum.e += pixel * Filter_3_Real[1][4];
	sum.f += pixel * Filter_3_Imag[1][4];
    pixel = image[y + 3][x - 1]; 
    sum.a += pixel * Filter_1_Real[0][4];
	sum.b += pixel * Filter_1_Imag[0][4];
	sum.c += pixel * Filter_2_Real[0][4];
	sum.d += pixel * Filter_2_Imag[0][4];
	sum.e += pixel * Filter_3_Real[0][4];
	sum.f += pixel * Filter_3_Imag[0][4];


    pixel = image[y - 3][x + 0]; 
    sum.a += pixel * Filter_1_Real[6][3];
	sum.b += pixel * Filter_1_Imag[6][3];
	sum.c += pixel * Filter_2_Real[6][3];
	sum.d += pixel * Filter_2_Imag[6][3];
	sum.e += pixel * Filter_3_Real[6][3];
	sum.f += pixel * Filter_3_Imag[6][3];
    pixel = image[y - 2][x + 0]; 
    sum.a += pixel * Filter_1_Real[5][3];
	sum.b += pixel * Filter_1_Imag[5][3];
	sum.c += pixel * Filter_2_Real[5][3];
	sum.d += pixel * Filter_2_Imag[5][3];
	sum.e += pixel * Filter_3_Real[5][3];
	sum.f += pixel * Filter_3_Imag[5][3];
    pixel = image[y - 1][x + 0]; 
    sum.a += pixel * Filter_1_Real[4][3];
	sum.b += pixel * Filter_1_Imag[4][3];
	sum.c += pixel * Filter_2_Real[4][3];
	sum.d += pixel * Filter_2_Imag[4][3];
	sum.e += pixel * Filter_3_Real[4][3];
	sum.f += pixel * Filter_3_Imag[4][3];
    pixel = image[y + 0][x + 0]; 
    sum.a += pixel * Filter_1_Real[3][3];
	sum.b += pixel * Filter_1_Imag[3][3];
	sum.c += pixel * Filter_2_Real[3][3];
	sum.d += pixel * Filter_2_Imag[3][3];
	sum.e += pixel * Filter_3_Real[3][3];
	sum.f += pixel * Filter_3_Imag[3][3];
    pixel = image[y + 1][x + 0]; 
    sum.a += pixel * Filter_1_Real[2][3];
	sum.b += pixel * Filter_1_Imag[2][3];
	sum.c += pixel * Filter_2_Real[2][3];
	sum.d += pixel * Filter_2_Imag[2][3];
	sum.e += pixel * Filter_3_Real[2][3];
	sum.f += pixel * Filter_3_Imag[2][3];
    pixel = image[y + 2][x + 0]; 
    sum.a += pixel * Filter_1_Real[1][3];
	sum.b += pixel * Filter_1_Imag[1][3];
	sum.c += pixel * Filter_2_Real[1][3];
	sum.d += pixel * Filter_2_Imag[1][3];
	sum.e += pixel * Filter_3_Real[1][3];
	sum.f += pixel * Filter_3_Imag[1][3];
    pixel = image[y + 3][x + 0]; 
    sum.a += pixel * Filter_1_Real[0][3];
	sum.b += pixel * Filter_1_Imag[0][3];
	sum.c += pixel * Filter_2_Real[0][3];
	sum.d += pixel * Filter_2_Imag[0][3];
	sum.e += pixel * Filter_3_Real[0][3];
	sum.f += pixel * Filter_3_Imag[0][3];

	pixel = image[y - 3][x + 1]; 
    sum.a += pixel * Filter_1_Real[6][2];
	sum.b += pixel * Filter_1_Imag[6][2];
	sum.c += pixel * Filter_2_Real[6][2];
	sum.d += pixel * Filter_2_Imag[6][2];
	sum.e += pixel * Filter_3_Real[6][2];
	sum.f += pixel * Filter_3_Imag[6][2];
    pixel = image[y - 2][x + 1]; 
    sum.a += pixel * Filter_1_Real[5][2];
	sum.b += pixel * Filter_1_Imag[5][2];
	sum.c += pixel * Filter_2_Real[5][2];
	sum.d += pixel * Filter_2_Imag[5][2];
	sum.e += pixel * Filter_3_Real[5][2];
	sum.f += pixel * Filter_3_Imag[5][2];
    pixel = image[y - 1][x + 1]; 
    sum.a += pixel * Filter_1_Real[4][2];
	sum.b += pixel * Filter_1_Imag[4][2];
	sum.c += pixel * Filter_2_Real[4][2];
	sum.d += pixel * Filter_2_Imag[4][2];
	sum.e += pixel * Filter_3_Real[4][2];
	sum.f += pixel * Filter_3_Imag[4][2];
    pixel = image[y + 0][x + 1]; 
    sum.a += pixel * Filter_1_Real[3][2];
	sum.b += pixel * Filter_1_Imag[3][2];
	sum.c += pixel * Filter_2_Real[3][2];
	sum.d += pixel * Filter_2_Imag[3][2];
	sum.e += pixel * Filter_3_Real[3][2];
	sum.f += pixel * Filter_3_Imag[3][2];
    pixel = image[y + 1][x + 1]; 
    sum.a += pixel * Filter_1_Real[2][2];
	sum.b += pixel * Filter_1_Imag[2][2];
	sum.c += pixel * Filter_2_Real[2][2];
	sum.d += pixel * Filter_2_Imag[2][2];
	sum.e += pixel * Filter_3_Real[2][2];
	sum.f += pixel * Filter_3_Imag[2][2];
    pixel = image[y + 2][x + 1]; 
    sum.a += pixel * Filter_1_Real[1][2];
	sum.b += pixel * Filter_1_Imag[1][2];
	sum.c += pixel * Filter_2_Real[1][2];
	sum.d += pixel * Filter_2_Imag[1][2];
	sum.e += pixel * Filter_3_Real[1][2];
	sum.f += pixel * Filter_3_Imag[1][2];
    pixel = image[y + 3][x + 1]; 
    sum.a += pixel * Filter_1_Real[0][2];
	sum.b += pixel * Filter_1_Imag[0][2];
	sum.c += pixel * Filter_2_Real[0][2];
	sum.d += pixel * Filter_2_Imag[0][2];
	sum.e += pixel * Filter_3_Real[0][2];
	sum.f += pixel * Filter_3_Imag[0][2];
 
    pixel = image[y - 3][x + 2]; 
    sum.a += pixel * Filter_1_Real[6][1];
	sum.b += pixel * Filter_1_Imag[6][1];
	sum.c += pixel * Filter_2_Real[6][1];
	sum.d += pixel * Filter_2_Imag[6][1];
	sum.e += pixel * Filter_3_Real[6][1];
	sum.f += pixel * Filter_3_Imag[6][1];
    pixel = image[y - 2][x + 2]; 
    sum.a += pixel * Filter_1_Real[5][1];
	sum.b += pixel * Filter_1_Imag[5][1];
	sum.c += pixel * Filter_2_Real[5][1];
	sum.d += pixel * Filter_2_Imag[5][1];
	sum.e += pixel * Filter_3_Real[5][1];
	sum.f += pixel * Filter_3_Imag[5][1];
    pixel = image[y - 1][x + 2]; 
    sum.a += pixel * Filter_1_Real[4][1];
	sum.b += pixel * Filter_1_Imag[4][1];
	sum.c += pixel * Filter_2_Real[4][1];
	sum.d += pixel * Filter_2_Imag[4][1];
	sum.e += pixel * Filter_3_Real[4][1];
	sum.f += pixel * Filter_3_Imag[4][1];
    pixel = image[y + 0][x + 2]; 
    sum.a += pixel * Filter_1_Real[3][1];
	sum.b += pixel * Filter_1_Imag[3][1];
	sum.c += pixel * Filter_2_Real[3][1];
	sum.d += pixel * Filter_2_Imag[3][1];
	sum.e += pixel * Filter_3_Real[3][1];
	sum.f += pixel * Filter_3_Imag[3][1];
	pixel = image[y + 1][x + 2]; 
    sum.a += pixel * Filter_1_Real[2][1];
	sum.b += pixel * Filter_1_Imag[2][1];
	sum.c += pixel * Filter_2_Real[2][1];
	sum.d += pixel * Filter_2_Imag[2][1];
	sum.e += pixel * Filter_3_Real[2][1];
	sum.f += pixel * Filter_3_Imag[2][1];
    pixel = image[y + 2][x + 2]; 
    sum.a += pixel * Filter_1_Real[1][1];
	sum.b += pixel * Filter_1_Imag[1][1];
	sum.c += pixel * Filter_2_Real[1][1];
	sum.d += pixel * Filter_2_Imag[1][1];
	sum.e += pixel * Filter_3_Real[1][1];
	sum.f += pixel * Filter_3_Imag[1][1];
    pixel = image[y + 3][x + 2]; 
    sum.a += pixel * Filter_1_Real[0][1];
	sum.b += pixel * Filter_1_Imag[0][1];
	sum.c += pixel * Filter_2_Real[0][1];
	sum.d += pixel * Filter_2_Imag[0][1];
	sum.e += pixel * Filter_3_Real[0][1];
	sum.f += pixel * Filter_3_Imag[0][1];

    pixel = image[y - 3][x + 3]; 
    sum.a += pixel * Filter_1_Real[6][0];
	sum.b += pixel * Filter_1_Imag[6][0];
	sum.c += pixel * Filter_2_Real[6][0];
	sum.d += pixel * Filter_2_Imag[6][0];
	sum.e += pixel * Filter_3_Real[6][0];
	sum.f += pixel * Filter_3_Imag[6][0];
    pixel = image[y - 2][x + 3]; 
    sum.a += pixel * Filter_1_Real[5][0];
	sum.b += pixel * Filter_1_Imag[5][0];
	sum.c += pixel * Filter_2_Real[5][0];
	sum.d += pixel * Filter_2_Imag[5][0];
	sum.e += pixel * Filter_3_Real[5][0];
	sum.f += pixel * Filter_3_Imag[5][0];
    pixel = image[y - 1][x + 3]; 
    sum.a += pixel * Filter_1_Real[4][0];
	sum.b += pixel * Filter_1_Imag[4][0];
	sum.c += pixel * Filter_2_Real[4][0];
	sum.d += pixel * Filter_2_Imag[4][0];
	sum.e += pixel * Filter_3_Real[4][0];
	sum.f += pixel * Filter_3_Imag[4][0];
    pixel = image[y + 0][x + 3]; 
    sum.a += pixel * Filter_1_Real[3][0];
	sum.b += pixel * Filter_1_Imag[3][0];
	sum.c += pixel * Filter_2_Real[3][0];
	sum.d += pixel * Filter_2_Imag[3][0];
	sum.e += pixel * Filter_3_Real[3][0];
	sum.f += pixel * Filter_3_Imag[3][0];
    pixel = image[y + 1][x + 3]; 
    sum.a += pixel * Filter_1_Real[2][0];
	sum.b += pixel * Filter_1_Imag[2][0];
	sum.c += pixel * Filter_2_Real[2][0];
	sum.d += pixel * Filter_2_Imag[2][0];
	sum.e += pixel * Filter_3_Real[2][0];
	sum.f += pixel * Filter_3_Imag[2][0];
    pixel = image[y + 2][x + 3]; 
    sum.a += pixel * Filter_1_Real[1][0];
	sum.b += pixel * Filter_1_Imag[1][0];
	sum.c += pixel * Filter_2_Real[1][0];
	sum.d += pixel * Filter_2_Imag[1][0];
	sum.e += pixel * Filter_3_Real[1][0];
	sum.f += pixel * Filter_3_Imag[1][0];
    pixel = image[y + 3][x + 3]; 
    sum.a += pixel * Filter_1_Real[0][0];
	sum.b += pixel * Filter_1_Imag[0][0];
	sum.c += pixel * Filter_2_Real[0][0];
	sum.d += pixel * Filter_2_Imag[0][0];
	sum.e += pixel * Filter_3_Real[0][0];
	sum.f += pixel * Filter_3_Imag[0][0];

	return sum;
}

__kernel void Memset(__global float *Data, __private float value, __private int N)
{
	int i = get_global_id(0);

	if (i >= N)
		return;

	Data[i] = value;
}

#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void MemsetDouble(__global double *Data, __private double value, __private int N)
{
	int i = get_global_id(0);

	if (i >= N)
		return;

	Data[i] = value;
}

__kernel void Nonseparable3DConvolutionComplex(__global float *Filter_Response_1_Real, __global float *Filter_Response_1_Imag, __global float *Filter_Response_2_Real, __global float *Filter_Response_2_Imag, __global float *Filter_Response_3_Real, __global float *Filter_Response_3_Imag, __global const float* Volume, __constant float c_Quadrature_Filter_1_Real[7][7][7], __constant float c_Quadrature_Filter_1_Imag[7][7][7], __constant float c_Quadrature_Filter_2_Real[7][7][7], __constant float c_Quadrature_Filter_2_Imag[7][7][7], __constant float c_Quadrature_Filter_3_Real[7][7][7], __constant float c_Quadrature_Filter_3_Imag[7][7][7], __private int z_offset, __private int DATA_W, __private int DATA_H, __private int DATA_D)
{   
    int x = get_group_id(0) * VALID_FILTER_RESPONSES_X_CONVOLUTION_2D + get_local_id(0);
	int y = get_group_id(1) * VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D + get_local_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};
    
    __local float l_Image[64][96]; // y, x

    // Reset shared memory
    l_Image[tIdx.y][tIdx.x]           = 0.0f;
    l_Image[tIdx.y][tIdx.x + 32]      = 0.0f;
    l_Image[tIdx.y][tIdx.x + 64]      = 0.0f;
    l_Image[tIdx.y + 32][tIdx.x]      = 0.0f;
    l_Image[tIdx.y + 32][tIdx.x + 32] = 0.0f;
    l_Image[tIdx.y + 32][tIdx.x + 64] = 0.0f;

    // Read data into shared memory

    if ( ((z + z_offset) >= 0) && ((z + z_offset) < DATA_D) )
    {
        if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  )   
            l_Image[tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+32-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  )
            l_Image[tIdx.y][tIdx.x + 32] = Volume[Calculate3DIndex(x+32-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+64-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  ) 
            l_Image[tIdx.y][tIdx.x + 64] = Volume[Calculate3DIndex(x+64-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )
            l_Image[tIdx.y + 32][tIdx.x] = Volume[Calculate3DIndex(x-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+32-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )
            l_Image[tIdx.y + 32][tIdx.x + 32] = Volume[Calculate3DIndex(x+32-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+64-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )
            l_Image[tIdx.y + 32][tIdx.x + 64] = Volume[Calculate3DIndex(x+64-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];
    }
	
   	// Make sure all threads have written to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

    // Only threads inside the image do the convolution

    if ( (x < DATA_W) && (y < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7(l_Image,tIdx.y+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1_Real[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += temp.a;
	    Filter_Response_1_Imag[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += temp.b;
	    Filter_Response_2_Real[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += temp.c;
	    Filter_Response_2_Imag[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += temp.d;
	    Filter_Response_3_Real[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += temp.e;
	    Filter_Response_3_Imag[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += temp.f;
    }

    if ( ((x + 32) < DATA_W) && (y < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7(l_Image,tIdx.y+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1_Real[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += temp.a;
		Filter_Response_1_Imag[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += temp.b;
		Filter_Response_2_Real[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += temp.c;
		Filter_Response_2_Imag[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += temp.d;
		Filter_Response_3_Real[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += temp.e;
		Filter_Response_3_Imag[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += temp.f;
    }

    if (tIdx.x < (32 - HALO*2))
    {
        if ( ((x + 64) < DATA_W) && (y < DATA_H) )
	    {
		    float6 temp = Conv_2D_Unrolled_7x7(l_Image,tIdx.y+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
            Filter_Response_1_Real[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += temp.a;
		    Filter_Response_1_Imag[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += temp.b;
		    Filter_Response_2_Real[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += temp.c;
		    Filter_Response_2_Imag[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += temp.d;
		    Filter_Response_3_Real[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += temp.e;
		    Filter_Response_3_Imag[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += temp.f;
	    }
    }

    if (tIdx.y < (32 - HALO*2))
    {
        if ( (x < DATA_W) && ((y + 32) < DATA_H) )
	    {
 		    float6 temp = Conv_2D_Unrolled_7x7(l_Image,tIdx.y+32+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
            Filter_Response_1_Real[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += temp.a;
		    Filter_Response_1_Imag[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += temp.b;
		    Filter_Response_2_Real[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += temp.c;
		    Filter_Response_2_Imag[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += temp.d;
		    Filter_Response_3_Real[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += temp.e;
		    Filter_Response_3_Imag[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += temp.f;
	    }
    }

    if (tIdx.y < (32 - HALO*2))
    {
        if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
	    {
		    float6 temp = Conv_2D_Unrolled_7x7(l_Image,tIdx.y+32+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
            Filter_Response_1_Real[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += temp.a;
		    Filter_Response_1_Imag[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += temp.b;
		    Filter_Response_2_Real[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += temp.c;
		    Filter_Response_2_Imag[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += temp.d;
		    Filter_Response_3_Real[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += temp.e;
		    Filter_Response_3_Imag[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += temp.f;
	    }
     } 

    if ( (tIdx.x < (32 - HALO*2)) && (tIdx.y < (32 - HALO*2)) )
    {
        if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
	    {
		    float6 temp = Conv_2D_Unrolled_7x7(l_Image,tIdx.y+32+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
            Filter_Response_1_Real[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += temp.a;
		    Filter_Response_1_Imag[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += temp.b;
		    Filter_Response_2_Real[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += temp.c;
		    Filter_Response_2_Imag[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += temp.d;
		    Filter_Response_3_Real[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += temp.e;
		    Filter_Response_3_Imag[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += temp.f;
	    }
     }
}

	

// Functions for motion correction


__kernel void CalculatePhaseDifferencesAndCertainties(__global float* Phase_Differences, __global float* Certainties, __global const float* q11_Real, __global const float* q11_Imag, __global const float* q21_Real, __global const float* q21_Imag, __private int DATA_W, __private int DATA_H, __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	
	if ( (x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D))
		return;

	int idx = Calculate3DIndex(x, y, z, DATA_W, DATA_H);

	float complex_product_real, complex_product_imag;
	float a, b, c, d, phase_difference;

	// q1 = a + i * b
	// q2 = c + i * d
	a = q11_Real[idx];
	b = q11_Imag[idx];
	c = q21_Real[idx];
	d = q21_Imag[idx];

	// phase difference = arg (q1 * (complex conjugate of q2))
	complex_product_real = a * c + b * d;
	complex_product_imag = b * c - a * d;

	phase_difference = atan2(complex_product_imag, complex_product_real);

	complex_product_real = a * c - b * d;
  	complex_product_imag = b * c + a * d;

	c = cos( phase_difference * 0.5f );
	Phase_Differences[idx] = phase_difference;
	Certainties[idx] = sqrt(complex_product_real * complex_product_real + complex_product_imag * complex_product_imag) * c * c;
}


__kernel void CalculatePhaseGradientsX(__global float* Phase_Gradients, __global const float* q11_Real, __global const float* q11_Imag, __global const float* q21_Real, __global const float* q21_Imag, __private int DATA_W, __private int DATA_H, __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	
	if ( (x >= DATA_W) || ((x + 1) >= DATA_W) || ((x - 1) < 0) || (y >= DATA_H) || (z >= DATA_D))
		return;

	float total_complex_product_real, total_complex_product_imag;
	float a, b, c, d;
	int idx_minus_1, idx_plus_1, idx;

	idx = Calculate3DIndex(x, y, z, DATA_W, DATA_H);

	// X
	idx_minus_1 = Calculate3DIndex(x - 1, y, z, DATA_W, DATA_H);
	idx_plus_1 = Calculate3DIndex(x + 1, y, z, DATA_W, DATA_H);

	total_complex_product_real = 0.0f;
	total_complex_product_imag = 0.0f;

	a = q11_Real[idx_plus_1];
	b = q11_Imag[idx_plus_1];
	c = q11_Real[idx];
	d = q11_Imag[idx];

	total_complex_product_real += a * c + b * d;
	total_complex_product_imag += b * c - a * d;

	a = c;
	b = d;
	c = q11_Real[idx_minus_1];
	d = q11_Imag[idx_minus_1];

	total_complex_product_real += a * c + b * d;
	total_complex_product_imag += b * c - a * d;

	a = q21_Real[idx_plus_1];
	b = q21_Imag[idx_plus_1];
	c = q21_Real[idx];
	d = q21_Imag[idx];

	total_complex_product_real += a * c + b * d;
	total_complex_product_imag += b * c - a * d;

	a = c;
	b = d;
	c = q21_Real[idx_minus_1];
	d = q21_Imag[idx_minus_1];

	total_complex_product_real += a * c + b * d;
	total_complex_product_imag += b * c - a * d;

	Phase_Gradients[idx] = atan2(total_complex_product_imag, total_complex_product_real);
}


__kernel void CalculatePhaseGradientsY(__global float* Phase_Gradients, __global const float* q12_Real, __global const float* q12_Imag, __global const float* q22_Real, __global const float* q22_Imag, __private int DATA_W, __private int DATA_H, __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	
	if ( (x >= DATA_W) || (y >= DATA_H) || ((y + 1) >= DATA_H) || ((y - 1) < 0) || (z >= DATA_D))	
		return;

	float total_complex_product_real, total_complex_product_imag;
	float a, b, c, d;
	int idx_minus_1, idx_plus_1, idx;

	idx = Calculate3DIndex(x, y, z, DATA_W, DATA_H);

	// Y

	idx_plus_1 =  Calculate3DIndex(x, y + 1, z, DATA_W, DATA_H);
	idx_minus_1 =  Calculate3DIndex(x, y - 1, z, DATA_W, DATA_H);

	total_complex_product_real = 0.0f;
	total_complex_product_imag = 0.0f;

	a = q12_Real[idx_plus_1];
	b = q12_Imag[idx_plus_1];
	c = q12_Real[idx];
	d = q12_Imag[idx];

	total_complex_product_real += a * c + b * d;
	total_complex_product_imag += b * c - a * d;

	a = c;
	b = d;
	c = q12_Real[idx_minus_1];
	d = q12_Imag[idx_minus_1];

	total_complex_product_real += a * c + b * d;
	total_complex_product_imag += b * c - a * d;

	a = q22_Real[idx_plus_1];
	b = q22_Imag[idx_plus_1];
	c = q22_Real[idx];
	d = q22_Imag[idx];
	
	total_complex_product_real += a * c + b * d;
	total_complex_product_imag += b * c - a * d;

	a = c;
	b = d;
	c = q22_Real[idx_minus_1];
	d = q22_Imag[idx_minus_1];

	total_complex_product_real += a * c + b * d;
	total_complex_product_imag += b * c - a * d;

	Phase_Gradients[idx] = atan2(total_complex_product_imag, total_complex_product_real);
}

__kernel void CalculatePhaseGradientsZ(__global float* Phase_Gradients, __global const float* q13_Real, __global const float* q13_Imag, __global const float* q23_Real, __global const float* q23_Imag, __private int DATA_W, __private int DATA_H, __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	
	if ( (x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D) || ((z + 1) >= DATA_D) || ((z - 1) < 0) )	
		return;

	float total_complex_product_real, total_complex_product_imag;
	float a, b, c, d;
	int idx_minus_1, idx_plus_1, idx;

	idx = Calculate3DIndex(x, y, z, DATA_W, DATA_H);

	// Z

	idx_plus_1 = Calculate3DIndex(x, y, z + 1, DATA_W, DATA_H);
	idx_minus_1 = Calculate3DIndex(x, y, z - 1, DATA_W, DATA_H);

	total_complex_product_real = 0.0f;
	total_complex_product_imag = 0.0f;

	a = q13_Real[idx_plus_1];
	b = q13_Imag[idx_plus_1];
	c = q13_Real[idx];
	d = q13_Imag[idx];

	total_complex_product_real += a * c + b * d;
	total_complex_product_imag += b * c - a * d;

	a = c;
	b = d;
	c = q13_Real[idx_minus_1];
	d = q13_Imag[idx_minus_1];

	total_complex_product_real += a * c + b * d;
	total_complex_product_imag += b * c - a * d;

	a = q23_Real[idx_plus_1];
	b = q23_Imag[idx_plus_1];
	c = q23_Real[idx];
	d = q23_Imag[idx];

	total_complex_product_real += a * c + b * d;
	total_complex_product_imag += b * c - a * d;

	a = c;
	b = d;
	c = q23_Real[idx_minus_1];
	d = q23_Imag[idx_minus_1];

	total_complex_product_real += a * c + b * d;
	total_complex_product_imag += b * c - a * d;

	Phase_Gradients[idx] = atan2(total_complex_product_imag, total_complex_product_real);
}



// dimBlock.x = DATA_H; dimBlock.y = 1; dimBlock.z = 1;
// dimGrid.x = DATA_D; dimGrid.y = 1;

__kernel void CalculateAMatrixAndHVector2DValuesX(__global float* A_matrix_2D_values, __global float* h_vector_2D_values, __global const float* Phase_Differences, __global const float* Phase_Gradients, __global const float* Phase_Certainties, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int FILTER_SIZE)
{
	int y = get_local_id(0);
	int z = get_group_id(1); 
				
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
			int idx = Calculate3DIndex(x, y, z, DATA_W, DATA_H);

			float phase_difference = Phase_Differences[idx];
			float phase_gradient = Phase_Gradients[idx];
			float phase_certainty = Phase_Certainties[idx];
			float c_pg_pg = phase_certainty * phase_gradient * phase_gradient;
			float c_pg_pd = phase_certainty * phase_gradient * phase_difference;

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

__kernel void CalculateAMatrixAndHVector2DValuesXDouble(__global double* A_matrix_2D_values, __global double* h_vector_2D_values, __global const float* Phase_Differences, __global const float* Phase_Gradients, __global const float* Phase_Certainties, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int FILTER_SIZE)
{
	int y = get_local_id(0);
	int z = get_group_id(1); 
				
	if (((y >= (FILTER_SIZE - 1)/2) && (y < DATA_H - (FILTER_SIZE - 1)/2)) && ((z >= (FILTER_SIZE - 1)/2) && (z < DATA_D - (FILTER_SIZE - 1)/2)))
	{
		double yf, zf;
		int matrix_element_idx, vector_element_idx;
		double A_matrix_2D_value[10], h_vector_2D_value[4];

    	yf = (double)y - ((double)DATA_H - 1.0) * 0.5;
		zf = (double)z - ((double)DATA_D - 1.0) * 0.5;

		// X

		A_matrix_2D_value[0] = 0.0;
		A_matrix_2D_value[1] = 0.0;
		A_matrix_2D_value[2] = 0.0;
		A_matrix_2D_value[3] = 0.0;
		A_matrix_2D_value[4] = 0.0;
		A_matrix_2D_value[5] = 0.0;
		A_matrix_2D_value[6] = 0.0;
		A_matrix_2D_value[7] = 0.0;
		A_matrix_2D_value[8] = 0.0;
		A_matrix_2D_value[9] = 0.0;

		h_vector_2D_value[0] = 0.0;
		h_vector_2D_value[1] = 0.0;
		h_vector_2D_value[2] = 0.0;
		h_vector_2D_value[3] = 0.0;

		for (int x = (FILTER_SIZE - 1)/2; x < (DATA_W - (FILTER_SIZE - 1)/2); x++)
		{
			double xf = (double)x - ((double)DATA_W - 1.0) * 0.5;
			int idx = Calculate3DIndex(x, y, z, DATA_W, DATA_H);

			double phase_difference = (double)Phase_Differences[idx];
			double phase_gradient = (double)Phase_Gradients[idx];
			double phase_certainty = (double)Phase_Certainties[idx];
			double c_pg_pg = phase_certainty * phase_gradient * phase_gradient;
			double c_pg_pd = phase_certainty * phase_gradient * phase_difference;

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


__kernel void CalculateAMatrixAndHVector2DValuesY(__global float* A_matrix_2D_values, __global float* h_vector_2D_values, __global const float* Phase_Differences, __global const float* Phase_Gradients, __global const float* Phase_Certainties, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int FILTER_SIZE)
{
	int y = get_local_id(0);
	int z = get_group_id(1);

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
			int idx = Calculate3DIndex(x, y, z, DATA_W, DATA_H);

			float phase_difference = Phase_Differences[idx];
			float phase_gradient = Phase_Gradients[idx];
			float phase_certainty = Phase_Certainties[idx];
			float c_pg_pg = phase_certainty * phase_gradient * phase_gradient;
			float c_pg_pd = phase_certainty * phase_gradient * phase_difference;

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

__kernel void CalculateAMatrixAndHVector2DValuesYDouble(__global double* A_matrix_2D_values, __global double* h_vector_2D_values, __global const float* Phase_Differences, __global const float* Phase_Gradients, __global const float* Phase_Certainties, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int FILTER_SIZE)
{
	int y = get_local_id(0);
	int z = get_group_id(1);

	if (((y >= (FILTER_SIZE - 1)/2) && (y < DATA_H - (FILTER_SIZE - 1)/2)) && ((z >= (FILTER_SIZE - 1)/2) && (z < DATA_D - (FILTER_SIZE - 1)/2)))
	{
		double yf, zf;
		int matrix_element_idx, vector_element_idx;
		double A_matrix_2D_value[10], h_vector_2D_value[4];

    	yf = (double)y - ((double)DATA_H - 1.0) * 0.5;
		zf = (double)z - ((double)DATA_D - 1.0) * 0.5;

		// Y

		A_matrix_2D_value[0] = 0.0;
		A_matrix_2D_value[1] = 0.0;
		A_matrix_2D_value[2] = 0.0;
		A_matrix_2D_value[3] = 0.0;
		A_matrix_2D_value[4] = 0.0;
		A_matrix_2D_value[5] = 0.0;
		A_matrix_2D_value[6] = 0.0;
		A_matrix_2D_value[7] = 0.0;
		A_matrix_2D_value[8] = 0.0;
		A_matrix_2D_value[9] = 0.0;

		h_vector_2D_value[0] = 0.0;
		h_vector_2D_value[1] = 0.0;
		h_vector_2D_value[2] = 0.0;
		h_vector_2D_value[3] = 0.0;

		for (int x = (FILTER_SIZE - 1)/2; x < (DATA_W - (FILTER_SIZE - 1)/2); x++)
		{
			double xf = (double)x - ((double)DATA_W - 1.0) * 0.5;
			int idx = Calculate3DIndex(x, y, z, DATA_W, DATA_H);

			double phase_difference = (double)Phase_Differences[idx];
			double phase_gradient = (double)Phase_Gradients[idx];
			double phase_certainty = (double)Phase_Certainties[idx];
			double c_pg_pg = phase_certainty * phase_gradient * phase_gradient;
			double c_pg_pd = phase_certainty * phase_gradient * phase_difference;

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



__kernel void CalculateAMatrixAndHVector2DValuesZ(__global float* A_matrix_2D_values, __global float* h_vector_2D_values, __global const float* Phase_Differences, __global const float* Phase_Gradients, __global const float* Phase_Certainties, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)
{
	int y = get_local_id(0);
	int z = get_group_id(1);

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
			int idx = Calculate3DIndex(x, y, z, DATA_W, DATA_H);

			float phase_difference = Phase_Differences[idx];
			float phase_gradient = Phase_Gradients[idx];
			float phase_certainty = Phase_Certainties[idx];
			float c_pg_pg = phase_certainty * phase_gradient * phase_gradient;
			float c_pg_pd = phase_certainty * phase_gradient * phase_difference;

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


__kernel void CalculateAMatrixAndHVector2DValuesZDouble(__global double* A_matrix_2D_values, __global double* h_vector_2D_values, __global const float* Phase_Differences, __global const float* Phase_Gradients, __global const float* Phase_Certainties, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)
{
	int y = get_local_id(0);
	int z = get_group_id(1);

	if (((y >= (FILTER_SIZE - 1)/2) && (y < DATA_H - (FILTER_SIZE - 1)/2)) && ((z >= (FILTER_SIZE - 1)/2) && (z < DATA_D - (FILTER_SIZE - 1)/2)))
	{
	    double yf, zf;
		int matrix_element_idx, vector_element_idx;
		double A_matrix_2D_value[10], h_vector_2D_value[4];

    	yf = (double)y - ((double)DATA_H - 1.0) * 0.5;
		zf = (double)z - ((double)DATA_D - 1.0) * 0.5;

		// Z

		A_matrix_2D_value[0] = 0.0;
		A_matrix_2D_value[1] = 0.0;
		A_matrix_2D_value[2] = 0.0;
		A_matrix_2D_value[3] = 0.0;
		A_matrix_2D_value[4] = 0.0;
		A_matrix_2D_value[5] = 0.0;
		A_matrix_2D_value[6] = 0.0;
		A_matrix_2D_value[7] = 0.0;
		A_matrix_2D_value[8] = 0.0;
		A_matrix_2D_value[9] = 0.0;

		h_vector_2D_value[0] = 0.0;
		h_vector_2D_value[1] = 0.0;
		h_vector_2D_value[2] = 0.0;
		h_vector_2D_value[3] = 0.0;

		for (int x = (FILTER_SIZE - 1)/2; x < (DATA_W - (FILTER_SIZE - 1)/2); x++)
		{
			double xf = (double)x - ((double)DATA_W - 1.0) * 0.5;
			int idx = Calculate3DIndex(x, y, z, DATA_W, DATA_H);

			double phase_difference = (double)Phase_Differences[idx];
			double phase_gradient = (double)Phase_Gradients[idx];
			double phase_certainty = (double)Phase_Certainties[idx];
			double c_pg_pg = phase_certainty * phase_gradient * phase_gradient;
			double c_pg_pd = phase_certainty * phase_gradient * phase_difference;

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

__kernel void CalculateAMatrix1DValues(__global float* A_matrix_1D_values, __global const float* A_matrix_2D_values, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int FILTER_SIZE)
{
	int z = get_local_id(0);
	int A_matrix_element   = get_group_id(1); // blockIdx.x; // 144 element (12 x 12 matrix) (30 that are != 0)

	if (z >= (FILTER_SIZE - 1)/2 && z < (DATA_D - (FILTER_SIZE - 1)/2))
	{		
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

__kernel void CalculateAMatrix1DValuesDouble(__global double* A_matrix_1D_values, __global const double* A_matrix_2D_values, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int FILTER_SIZE)
{
	int z = get_local_id(0);
	int A_matrix_element   = get_group_id(1); // blockIdx.x; // 144 element (12 x 12 matrix) (30 that are != 0)

	if (z >= (FILTER_SIZE - 1)/2 && z < (DATA_D - (FILTER_SIZE - 1)/2))
	{		
		int matrix_element_idx = z + A_matrix_element * DATA_D;
		int idx;
		double matrix_1D_value = 0.0;

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

__kernel void CalculateAMatrix(__global float* A_matrix, __global const float* A_matrix_1D_values, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int FILTER_SIZE)
{
	int A_matrix_element = get_local_id(0);
	int idx, i, j;

	float matrix_value = 0.0f;

	idx = A_matrix_element * DATA_D;

	// Sum over all z positions	
	for (int z = (FILTER_SIZE - 1)/2; z < (DATA_D - (FILTER_SIZE - 1)/2); z++)
	{
		matrix_value += A_matrix_1D_values[idx + z];
	}

	GetParameterIndices(&i,&j,A_matrix_element);
	A_matrix_element = i + j * 12; //NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS;

	A_matrix[A_matrix_element] = matrix_value;
}

__kernel void CalculateAMatrixDouble(__global double* A_matrix, __global const double* A_matrix_1D_values, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int FILTER_SIZE)
{
	int A_matrix_element = get_local_id(0);
	int idx, i, j;

	double matrix_value = 0.0;

	idx = A_matrix_element * DATA_D;

	// Sum over all z positions	
	for (int z = (FILTER_SIZE - 1)/2; z < (DATA_D - (FILTER_SIZE - 1)/2); z++)
	{
		matrix_value += A_matrix_1D_values[idx + z];
	}

	GetParameterIndices(&i,&j,A_matrix_element);
	A_matrix_element = i + j * 12; //NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS;

	A_matrix[A_matrix_element] = matrix_value;
}

// dimBlock.x = DATA_D; dimBlock.y = 1; dimBlock.z = 1;
// dimGrid.x = NUMBER_OF_PARAMETERS; dimGrid.y = 1;

__kernel void CalculateHVector1DValues(__global float* h_vector_1D_values, __global const float* h_vector_2D_values, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int FILTER_SIZE)
{
	int z = get_local_id(0);
	int h_vector_element   = get_global_id(1); //blockIdx.x; // 12 parameters

	if (z >= (FILTER_SIZE - 1)/2 && z < (DATA_D - (FILTER_SIZE - 1)/2))
	{		
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

__kernel void CalculateHVector1DValuesDouble(__global double* h_vector_1D_values, __global const double* h_vector_2D_values, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int FILTER_SIZE)
{
	int z = get_local_id(0);
	int h_vector_element   = get_global_id(1); //blockIdx.x; // 12 parameters

	if (z >= (FILTER_SIZE - 1)/2 && z < (DATA_D - (FILTER_SIZE - 1)/2))
	{		
		int vector_element_idx = z + h_vector_element * DATA_D;
		int idx;

		double vector_1D_value = 0.0;

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

__kernel void CalculateHVector(__global float* h_vector, __global const float* h_vector_1D_values, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int FILTER_SIZE)
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

__kernel void CalculateHVectorDouble(__global double* h_vector, __global const double* h_vector_1D_values, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int FILTER_SIZE)
{
	int h_vector_element = get_local_id(0);
	int idx;

	double vector_value = 0.0;
	idx = h_vector_element * DATA_D;

	// Sum over all z positions
	for (int z = (FILTER_SIZE - 1)/2; z < (DATA_D - (FILTER_SIZE - 1)/2); z++)
	{
		vector_value += h_vector_1D_values[idx + z];
	}

	h_vector[h_vector_element] = vector_value;
}

__constant sampler_t volume_sampler_nearest = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
	

__kernel void InterpolateVolumeNearest(__global float* Volume, read_only image3d_t Original_Volume, __constant float* c_Parameter_Vector, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int VOLUME)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int idx = Calculate4DIndex(x,y,z,VOLUME,DATA_W,DATA_H,DATA_D);
	float4 Motion_Vector;
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
	Motion_Vector.w = 0.0f;

	float4 Interpolated_Value = read_imagef(Original_Volume, volume_sampler_nearest, Motion_Vector);
	Volume[idx] = Interpolated_Value.x;
}

__constant sampler_t volume_sampler_linear = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
	

__kernel void InterpolateVolumeLinear(__global float* Volume, read_only image3d_t Original_Volume, __constant float* c_Parameter_Vector, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int VOLUME)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int idx = Calculate4DIndex(x,y,z,VOLUME,DATA_W,DATA_H,DATA_D);
	float4 Motion_Vector;
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
	Motion_Vector.w = 0.0f;

	float4 Interpolated_Value = read_imagef(Original_Volume, volume_sampler_linear, Motion_Vector);
	Volume[idx] = Interpolated_Value.x;
}




__kernel void RescaleVolumeLinear(__global float* Volume, read_only image3d_t Original_Volume, __private float VOXEL_DIFFERENCE_X, __private float VOXEL_DIFFERENCE_Y, __private float VOXEL_DIFFERENCE_Z, __private int DATA_W, __private int DATA_H, __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int idx = Calculate3DIndex(x,y,z,DATA_W, DATA_H);
	float4 Motion_Vector;
	
	Motion_Vector.x = x * VOXEL_DIFFERENCE_X + 0.5f;
	Motion_Vector.y = y * VOXEL_DIFFERENCE_Y + 0.5f;
	Motion_Vector.z = z * VOXEL_DIFFERENCE_Z + 0.5f;
	Motion_Vector.w = 0.0f;

	float4 Interpolated_Value = read_imagef(Original_Volume, volume_sampler_linear, Motion_Vector);
	Volume[idx] = Interpolated_Value.x;
}

__kernel void CalculateMagnitudes(__global float* Magnitudes, __global const float* Real, __global const float* Imag, __private int DATA_W, __private int DATA_H, __private int DATA_D)
{
	int x = get_global_id(0);	
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (y >= DATA_H || z >= DATA_D)
		return;

	float r = Real[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];
	float i = Imag[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];
	Magnitudes[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = sqrt(r * r + i * i);
}

__kernel void CalculateColumnSums(__global float* Sums, __global const float* Volume, __private int DATA_W, __private int DATA_H, __private int DATA_D)
{
	int y = get_global_id(0);	
	int z = get_global_id(1);

	if (y >= DATA_H || z >= DATA_D)
		return;

	float sum = 0.0f;
	for (int x = 0; x < DATA_W; x++)
	{
		sum += Volume[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];
	}

	Sums[Calculate2DIndex(y,z,DATA_H)] = sum;
}

__kernel void CalculateRowSums(__global float* Sums, __global const float* Image, __private int DATA_H, __private int DATA_D)
{
	int z = get_global_id(0);

	if (z >= DATA_D)
		return;

	float sum = 0.0f;
	for (int y = 0; y < DATA_H; y++)
	{
		sum += Image[Calculate2DIndex(y,z,DATA_H)];
	}

	Sums[z] = sum;
}

__kernel void CopyT1VolumeToMNI(__global float* MNI_T1_Volume,__global float* Interpolated_T1_Volume, __private int MNI_DATA_W, __private int MNI_DATA_H, __private int MNI_DATA_D, __private int T1_DATA_W_INTERPOLATED, __private int T1_DATA_H_INTERPOLATED, __private int T1_DATA_D_INTERPOLATED, __private int x_diff, __private int y_diff, __private int z_diff, __private int MM_T1_Z_CUT, __private float MNI_VOXEL_SIZE_Z)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int MNI_T1_idx, Interpolated_T1_idx;
	int x_MNI, x_Interpolated;
	int y_MNI, y_Interpolated;
	int z_MNI, z_Interpolated;
	
	// Interpolated T1 volume larger than MNI volume
	// Remove half of columns in each direction
	if (x_diff > 0)
	{
		x_MNI = x;
		x_Interpolated = x + (int)round((float)x_diff/2.0);
	}
	// Interpolated T1 volume smaller than MNI volume
	// Put interpolated T1 volume in the middle of the MNI volume
	else
	{
		x_MNI = x + (int)round((float)abs(x_diff)/2.0);
		x_Interpolated = x;
	}
	// Interpolated T1 volume larger than MNI volume
	// Remove half of rows in each direction
	if (y_diff > 0)
	{
		y_MNI = y;
		y_Interpolated = y + (int)round((float)y_diff/2.0);
	}
	// Interpolated T1 volume smaller than MNI volume
	// Put interpolated T1 volume in the middle of the MNI volume
	else
	{
		y_MNI = y + (int)round((float)abs(y_diff)/2.0);
		y_Interpolated = y;
	}
	// Interpolated T1 volume larger than MNI volume
	// Remove bottom slices
	if (z_diff > 0)
	{
		z_MNI = z;
		z_Interpolated = z + z_diff + (int)round((float)MM_T1_Z_CUT/MNI_VOXEL_SIZE_Z);
	}
	// Interpolated T1 volume smaller than MNI volume
	// Put interpolated T1 volume in the middle of the MNI volume
	else
	{
		z_MNI = z + (int)round((float)abs(z_diff)/2.0);
		z_Interpolated = z + (int)round((float)MM_T1_Z_CUT/MNI_VOXEL_SIZE_Z);
	}

	// Make sure we are not reading or writing outside any volume
	if ( (x_Interpolated >= T1_DATA_W_INTERPOLATED) || (y_Interpolated >= T1_DATA_H_INTERPOLATED) || (z_Interpolated >= T1_DATA_D_INTERPOLATED) || (x_MNI >= MNI_DATA_W) || (y_MNI >= MNI_DATA_H) || (z_MNI >= MNI_DATA_D) )
	{
		return;
	}
	else if ( (x_Interpolated < 0) || (y_Interpolated < 0) || (z_Interpolated < 0) || (x_MNI < 0) || (y_MNI < 0) || (z_MNI < 0) )
	{
		return;
	}
	else
	{
		MNI_T1_idx = Calculate3DIndex(x_MNI,y_MNI,z_MNI,MNI_DATA_W,MNI_DATA_H);
		Interpolated_T1_idx = Calculate3DIndex(x_Interpolated,y_Interpolated,z_Interpolated,T1_DATA_W_INTERPOLATED,T1_DATA_H_INTERPOLATED);
		MNI_T1_Volume[MNI_T1_idx] = Interpolated_T1_Volume[Interpolated_T1_idx];
	}			
}


__kernel void CopyEPIVolumeToT1(__global float* T1_EPI_Volume,__global float* Interpolated_EPI_Volume, __private int T1_DATA_W, __private int T1_DATA_H, __private int T1_DATA_D, __private int EPI_DATA_W_INTERPOLATED, __private int EPI_DATA_H_INTERPOLATED, __private int EPI_DATA_D_INTERPOLATED, __private int x_diff, __private int y_diff, __private int z_diff, __private int MM_EPI_Z_CUT, __private float T1_VOXEL_SIZE_Z)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int T1_EPI_idx, Interpolated_EPI_idx;
	int x_T1, x_Interpolated;
	int y_T1, y_Interpolated;
	int z_T1, z_Interpolated;

	// Interpolated EPI volume larger than T1 volume
	// Remove half of the columns in each direction
	if (x_diff > 0)
	{
		x_T1 = x;
		x_Interpolated = x + (int)round((float)x_diff/2.0);
	}
	// Interpolated EPI volume smaller than T1 volume
	// Put interpolated EPI volume in the middle of the T1 volume
	else
	{
		x_T1 = x + (int)round((float)abs(x_diff)/2.0);
		x_Interpolated = x;
	}
	// Interpolated EPI volume larger than T1 volume
	// Remove half of the rows in each direction
	if (y_diff > 0)
	{
		y_T1 = y;
		y_Interpolated = y + (int)round((float)y_diff/2.0);
	}
	// Interpolated EPI volume smaller than T1 volume
	// Put interpolated EPI volume in the middle of the T1 volume
	else
	{
		y_T1 = y + (int)round((float)abs(y_diff)/2.0);
		y_Interpolated = y;
	}
	// Interpolated EPI volume larger than T1 volume
	// Remove half the slices in each direction
	if (z_diff > 0)
	{
		z_T1 = z;
		z_Interpolated = z + (int)round((float)z_diff/2.0) + (int)round((float)MM_EPI_Z_CUT/T1_VOXEL_SIZE_Z);
	}
	// Interpolated EPI volume smaller than T1 volume
	// Put interpolated EPI volume in the middle of the T1 volume
	else
	{
		z_T1 = z + (int)round((float)abs(z_diff)/2.0);
		z_Interpolated = z + (int)round((float)MM_EPI_Z_CUT/T1_VOXEL_SIZE_Z);
	}

	// Make sure we are not reading outside any volume
	if ( (x_Interpolated >= EPI_DATA_W_INTERPOLATED) || (y_Interpolated >= EPI_DATA_H_INTERPOLATED) || (z_Interpolated >= EPI_DATA_D_INTERPOLATED) || (x_T1 >= T1_DATA_W) || (y_T1 >= T1_DATA_H) || (z_T1 >= T1_DATA_D) )
	{
		return;
	}
	else if ( (x_Interpolated < 0) || (y_Interpolated < 0) || (z_Interpolated < 0) || (x_T1 < 0) || (y_T1 < 0) || (z_T1 < 0) )
	{
		return;
	}
	else
	{
		T1_EPI_idx = Calculate3DIndex(x_T1,y_T1,z_T1,T1_DATA_W,T1_DATA_H);
		Interpolated_EPI_idx = Calculate3DIndex(x_Interpolated,y_Interpolated,z_Interpolated,EPI_DATA_W_INTERPOLATED,EPI_DATA_H_INTERPOLATED);
		T1_EPI_Volume[T1_EPI_idx] = Interpolated_EPI_Volume[Interpolated_EPI_idx];
	}			
}


__kernel void CopyVolumeToNew(__global float* New_Volume,__global float* Interpolated_Volume, __private int NEW_DATA_W, __private int NEW_DATA_H, __private int NEW_DATA_D, __private int DATA_W_INTERPOLATED, __private int DATA_H_INTERPOLATED, __private int DATA_D_INTERPOLATED, __private int x_diff, __private int y_diff, __private int z_diff, __private int MM_Z_CUT, __private float NEW_VOXEL_SIZE_Z, __private int VOLUME)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int NEW_idx, Interpolated_idx;
	int x_NEW, x_Interpolated;
	int y_NEW, y_Interpolated;
	int z_NEW, z_Interpolated;
	
	// Interpolated volume larger than new volume
	// Remove half of the columns in each direction
	if (x_diff > 0)
	{
		x_NEW = x;
		x_Interpolated = x + (int)round((float)x_diff/2.0);
	}
	// Interpolated volume smaller than new volume
	// Put interpolated volume in the middle of the new volume
	else
	{
		x_NEW = x + (int)round((float)abs(x_diff)/2.0);
		x_Interpolated = x;
	}
	// Interpolated EPI volume larger than T1 volume
	// Remove half of the rows in each direction
	if (y_diff > 0)
	{
		y_NEW = y;
		y_Interpolated = y + (int)round((float)y_diff/2.0);
	}
	// Interpolated EPI volume smaller than T1 volume
	// Put interpolated EPI volume in the middle of the T1 volume
	else
	{
		y_NEW = y + (int)round((float)abs(y_diff)/2.0);
		y_Interpolated = y;
	}
	// Interpolated EPI volume larger than T1 volume
	// Remove half the slices in each direction
	if (z_diff > 0)
	{
		z_NEW = z;
		z_Interpolated = z + (int)round((float)z_diff/2.0) + (int)round((float)MM_Z_CUT/NEW_VOXEL_SIZE_Z);
	}
	// Interpolated EPI volume smaller than T1 volume
	// Put interpolated EPI volume in the middle of the T1 volume
	else
	{
		z_NEW = z + (int)round((float)abs(z_diff)/2.0);
		z_Interpolated = z + (int)round((float)MM_Z_CUT/NEW_VOXEL_SIZE_Z);
	}

	// Make sure we are not reading outside any volume
	if ( (x_Interpolated >= DATA_W_INTERPOLATED) || (y_Interpolated >= DATA_H_INTERPOLATED) || (z_Interpolated >= DATA_D_INTERPOLATED) || (x_NEW >= NEW_DATA_W) || (y_NEW >= NEW_DATA_H) || (z_NEW >= NEW_DATA_D) )
	{
		return;
	}
	else if ( (x_Interpolated < 0) || (y_Interpolated < 0) || (z_Interpolated < 0) || (x_NEW < 0) || (y_NEW < 0) || (z_NEW < 0) )
	{
		return;
	}
	else
	{
		NEW_idx = Calculate4DIndex(x_NEW,y_NEW,z_NEW,VOLUME,NEW_DATA_W,NEW_DATA_H,NEW_DATA_D);
		Interpolated_idx = Calculate3DIndex(x_Interpolated,y_Interpolated,z_Interpolated,DATA_W_INTERPOLATED,DATA_H_INTERPOLATED);
		New_Volume[NEW_idx] = Interpolated_Volume[Interpolated_idx];
	}			
}


__kernel void MultiplyVolumes(__global float* Result, __global const float* Volume1, __global const float* Volume2, __private int DATA_W, __private int DATA_H, __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int idx = Calculate3DIndex(x,y,z,DATA_W,DATA_H);

	Result[idx] = Volume1[idx] * Volume2[idx];
}

__kernel void MultiplyVolumesOverwrite(__global float* Volumes1, __global const float* Volume2, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int VOLUME)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int idx3D = Calculate3DIndex(x,y,z,DATA_W,DATA_H);
	int idx4D = Calculate4DIndex(x,y,z,VOLUME,DATA_W,DATA_H,DATA_D);

	Volumes1[idx4D] = Volumes1[idx4D] * Volume2[idx3D];
}

// Statistical functions

__kernel void CalculateBetaValuesGLM(__global float* Beta_Volumes, __global const float* Volumes, __global const float* Mask, __constant float* c_xtxxt_GLM, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int NUMBER_OF_VOLUMES, __private int NUMBER_OF_REGRESSORS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] == 0.0f )
	{
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}
		return;
	}

	int t = 0;
	float beta[20];
	//__local float beta[16][32][16]; // y, x, regressors, For a maximum of 16 regressors per thread (32 KB)
	
	// Reset all beta values
	for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{
		beta[r] = 0.0f;
		//beta[tIdx.y][tIdx.x][r] = 0.0f;
	}

	// Calculate betahat, i.e. multiply (x^T x)^(-1) x^T with Y
	// Loop over volumes
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		//if (c_Censor[v] == 0.0f)
		//{
			float temp = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
			// Loop over regressors
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				beta[r] += temp * c_xtxxt_GLM[NUMBER_OF_VOLUMES * r + v];
				//beta[tIdx.y][tIdx.x][r] += temp * c_xtxxt_GLM[NUMBER_OF_VOLUMES * r + v];
			}
		//}
	}

	// Save beta values
	for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{
		Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)] = beta[r];
		//Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)] = beta[tIdx.y][tIdx.x][r];
	}
}

__kernel void CalculateStatisticalMapsGLM(__global float* Statistical_Maps, __global float* Beta_Contrasts, __global float* Residuals, __global float* Residual_Variances, __global const float* Volumes, __global float* Beta_Volumes, __global const float* Mask, __constant float *c_X_GLM, __constant float* c_Contrasts, __constant float* c_ctxtxc_GLM, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int NUMBER_OF_VOLUMES, __private int NUMBER_OF_REGRESSORS, __private int NUMBER_OF_CONTRASTS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] == 0.0f )
	{
		Residual_Variances[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;

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
			Residuals[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}

		return;
	}

	int t = 0;
	float eps, meaneps, vareps;
	float beta[20];
	//__local float beta[16][32][16]; // y, x, regressors, For a maximum of 16 regressors per thread (32 KB)

	// Load beta values into shared memory
    for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{ 
		beta[r] = Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)];
		//beta[tIdx.y][tIdx.x][r] = Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)];
	}

	// Calculate the mean of the error eps
	meaneps = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		//if (c_Censor[v] == 0.0f)
		//{
			eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{ 
				eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
				//eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[tIdx.y][tIdx.x][r];
			}
			meaneps += eps;
			Residuals[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = eps;
		//}
	}
	meaneps /= (float)NUMBER_OF_VOLUMES;

	// Now calculate the variance of eps
	vareps = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		//if (c_Censor[v] == 0.0f)
		//{
			eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
				//eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[tIdx.y][tIdx.x][r];
			}
			vareps += (eps - meaneps) * (eps - meaneps);
		//}
	}
	//vareps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_REGRESSORS); // correct for number of censor points?
	vareps /= (float)(NUMBER_OF_VOLUMES-1); // correct for number of censor points?
	Residual_Variances[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = vareps;
	
	// Loop over contrasts and calculate t-values
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		float contrast_value = 0.0f;
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + r] * beta[r];
			//contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + r] * beta[tIdx.y][tIdx.x][r];
		}	
		Beta_Contrasts[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_value;
		Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_value * rsqrt(vareps * c_ctxtxc_GLM[c]);		
	}
}

__kernel void RemoveLinearFit(__global float* Residual_Volumes, __global const float* Volumes, __global const float* Beta_Volumes, __global const float* Mask, __constant float *c_X_Detrend, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int NUMBER_OF_VOLUMES, __private int NUMBER_OF_REGRESSORS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] == 0.0f )
	{
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			Residual_Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}

		return;
	}

	int t = 0;
	float eps, meaneps, vareps;

	// Calculate the residual
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{ 
			eps -= Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)] * c_X_Detrend[NUMBER_OF_VOLUMES * r + v];
		}
		Residual_Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = eps;
	}
}


// Functions for permutation test

__kernel void CalculateStatisticalMapsGLMPermutation(__global float* Statistical_Maps, __global const float* Volumes, __global const float* Beta_Volumes, __global const float* Mask, __constant float *c_X_GLM, __constant float* c_Contrast_Vectors, __constant float* ctxtxc, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int NUMBER_OF_VOLUMES, __private int NUMBER_OF_REGRESSORS, __private int NUMBER_OF_CONTRASTS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;
	
	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] == 0.0f )
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
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		// Loop over regressors
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			eps -= Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)] * c_X_GLM[NUMBER_OF_VOLUMES * r + v];
		}
		meaneps += eps;
	}
	meaneps /= (float)NUMBER_OF_VOLUMES;

	// Now calculate the variance of eps
	vareps = 0.0f;
	// Loop over volumes
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		// Loop over regressors
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			eps -= Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)] * c_X_GLM[NUMBER_OF_VOLUMES * r + v];
		}
		vareps += (eps - meaneps) * (eps - meaneps);
	}
	vareps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_REGRESSORS);
	
	// Loop over contrasts and calculate t-values
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		float contrast_value = 0.0f;
		// Loop over regressors
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			contrast_value += c_Contrast_Vectors[NUMBER_OF_REGRESSORS * c + r] * Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)];
		}	
		Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_value * rsqrt(vareps * ctxtxc[c]);
	}
}


__kernel void GeneratePermutedfMRIVolumesAR4(__global float* Permuted_fMRI_Volumes, __global const float* Whitened_fMRI_Volumes, __global const float* AR1_Estimates, __global const float* AR2_Estimates, __global const float* AR3_Estimates, __global const float* AR4_Estimates, __global const float* Mask, __constant unsigned int *c_Permutation_Vector, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int DATA_T)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if ( x >= DATA_W || y >= DATA_H || z >= DATA_D )
        return;

    if ( Mask[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] == 0.0f )
		return;

    int t = 0;
	float old_value_1, old_value_2, old_value_3, old_value_4, old_value_5;
	float4 alphas;
	alphas.x = AR1_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)];
    alphas.y = AR2_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)];
    alphas.z = AR3_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)];
    alphas.w = AR4_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)];

    old_value_1 = Whitened_fMRI_Volumes[Calculate4DIndex(x, y, z, c_Permutation_Vector[0], DATA_W, DATA_H, DATA_D)];
	old_value_2 = alphas.x * old_value_1  + Whitened_fMRI_Volumes[Calculate4DIndex(x, y, z, c_Permutation_Vector[1], DATA_W, DATA_H, DATA_D)];
	old_value_3 = alphas.x * old_value_2  + alphas.y * old_value_1 + Whitened_fMRI_Volumes[Calculate4DIndex(x, y, z, c_Permutation_Vector[2], DATA_W, DATA_H, DATA_D)];
	old_value_4 = alphas.x * old_value_3  + alphas.y * old_value_2 + alphas.z * old_value_1 + Whitened_fMRI_Volumes[Calculate4DIndex(x, y, z, c_Permutation_Vector[3], DATA_W, DATA_H, DATA_D)];

    Permuted_fMRI_Volumes[Calculate4DIndex(x, y, z, 0, DATA_W, DATA_H, DATA_D)] =  old_value_1;
    Permuted_fMRI_Volumes[Calculate4DIndex(x, y, z, 1, DATA_W, DATA_H, DATA_D)] =  old_value_2;
    Permuted_fMRI_Volumes[Calculate4DIndex(x, y, z, 2, DATA_W, DATA_H, DATA_D)] =  old_value_3;
    Permuted_fMRI_Volumes[Calculate4DIndex(x, y, z, 3, DATA_W, DATA_H, DATA_D)] =  old_value_4;

    // Read the data in a permuted order and apply an inverse whitening transform
    for (t = 4; t < DATA_T; t++)
    {
        // Calculate the unwhitened, permuted, timeseries
        old_value_5 = alphas.x * old_value_1 + alphas.y * old_value_2 + alphas.z * old_value_3 + alphas.w * old_value_4 + Whitened_fMRI_Volumes[Calculate4DIndex(x, y, z, c_Permutation_Vector[t], DATA_W, DATA_H, DATA_D)];
			
        Permuted_fMRI_Volumes[Calculate4DIndex(x, y, z, t, DATA_W, DATA_H, DATA_D)] = old_value_5;

        // Save old values
		old_value_1 = old_value_2;
        old_value_2 = old_value_3;
        old_value_3 = old_value_4;
        old_value_4 = old_value_5;
    }
}

__kernel void ApplyWhiteningAR4(__global float* Whitened_fMRI_Volumes, __global const float* fMRI_Volumes, __global const float* AR1_Estimates, __global const float* AR2_Estimates, __global const float* AR3_Estimates, __global const float* AR4_Estimates, __global const float* Mask, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int DATA_T)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if ( x >= DATA_W || y >= DATA_H || z >= DATA_D )
        return;

    if ( Mask[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] == 0.0f )
		return;

    int t = 0;
	float old_value_1, old_value_2, old_value_3, old_value_4, old_value_5;
    float4 alphas;
	alphas.x = AR1_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)];
    alphas.y = AR2_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)];
    alphas.z = AR3_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)];
    alphas.w = AR4_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)];

    // Calculate the whitened timeseries

    //old_value_1 = fMRI_Volumes[Calculate4DIndex(x, y, z, 0, DATA_W, DATA_H, DATA_D)];
	old_value_1 = 0.0f;
    Whitened_fMRI_Volumes[Calculate4DIndex(x, y, z, 0, DATA_W, DATA_H, DATA_D)] = old_value_1;
    old_value_2 = fMRI_Volumes[Calculate4DIndex(x, y, z, 1, DATA_W, DATA_H, DATA_D)];
    Whitened_fMRI_Volumes[Calculate4DIndex(x, y, z, 1, DATA_W, DATA_H, DATA_D)] = old_value_2  - alphas.x * old_value_1;
    old_value_3 = fMRI_Volumes[Calculate4DIndex(x, y, z, 2, DATA_W, DATA_H, DATA_D)];
    Whitened_fMRI_Volumes[Calculate4DIndex(x, y, z, 2, DATA_W, DATA_H, DATA_D)] = old_value_3 - alphas.x * old_value_2 - alphas.y * old_value_1;
    old_value_4 = fMRI_Volumes[Calculate4DIndex(x, y, z, 3, DATA_W, DATA_H, DATA_D)];
    Whitened_fMRI_Volumes[Calculate4DIndex(x, y, z, 3, DATA_W, DATA_H, DATA_D)] = old_value_4 - alphas.x * old_value_3 - alphas.y * old_value_2 - alphas.z * old_value_1;

    for (t = 4; t < DATA_T; t++)
    {
        old_value_5 = fMRI_Volumes[Calculate4DIndex(x, y, z, t, DATA_W, DATA_H, DATA_D)];

        Whitened_fMRI_Volumes[Calculate4DIndex(x, y, z, t, DATA_W, DATA_H, DATA_D)] = old_value_5 - alphas.x * old_value_4 - alphas.y * old_value_3 - alphas.z * old_value_2 - alphas.w * old_value_1;

		// Save old values
        old_value_1 = old_value_2;
        old_value_2 = old_value_3;
        old_value_3 = old_value_4;
        old_value_4 = old_value_5;
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


	
__kernel void EstimateAR4Models(__global float* AR1_Estimates, __global float* AR2_Estimates, __global float* AR3_Estimates, __global float* AR4_Estimates, __global const float* fMRI_Volumes, __global const float* Mask, __private int DATA_W, __private int DATA_H, __private int DATA_D, __private int DATA_T)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Mask[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] == 0.0f )
	{
        AR1_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] = 0.0f;
		AR2_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] = 0.0f;
		AR3_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] = 0.0f;
		AR4_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] = 0.0f;

		return;
	}

    int t = 0;
	float old_value_1, old_value_2, old_value_3, old_value_4, old_value_5;
	float c0 = 0.0f;
    float c1 = 0.0f;
    float c2 = 0.0f;
    float c3 = 0.0f;
    float c4 = 0.0f;

    old_value_1 = fMRI_Volumes[Calculate4DIndex(x, y, z, 0, DATA_W, DATA_H, DATA_D)];
    c0 += old_value_1 * old_value_1;
    old_value_2 = fMRI_Volumes[Calculate4DIndex(x, y, z, 1, DATA_W, DATA_H, DATA_D)];
    c0 += old_value_2 * old_value_2;
    c1 += old_value_2 * old_value_1;
    old_value_3 = fMRI_Volumes[Calculate4DIndex(x, y, z, 2, DATA_W, DATA_H, DATA_D)];
    c0 += old_value_3 * old_value_3;
    c1 += old_value_3 * old_value_2;
    c2 += old_value_3 * old_value_1;
    old_value_4 = fMRI_Volumes[Calculate4DIndex(x, y, z, 3, DATA_W, DATA_H, DATA_D)];
    c0 += old_value_4 * old_value_4;
    c1 += old_value_4 * old_value_3;
    c2 += old_value_4 * old_value_2;
    c3 += old_value_4 * old_value_1;

    // Estimate c0, c1, c2, c3, c4
    for (t = 4; t < DATA_T; t++)
    {
        // Read data into shared memory
        old_value_5 = fMRI_Volumes[Calculate4DIndex(x, y, z, t, DATA_W, DATA_H, DATA_D)];
        
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

        AR1_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] = alphas.x;
		AR2_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] = alphas.y;
		AR3_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] = alphas.z;
		AR4_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] = alphas.w;
    }
    else
    {
		AR1_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] = 0.0f;
        AR2_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] = 0.0f;
		AR3_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] = 0.0f;
		AR4_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] = 0.0f;
    }
}

