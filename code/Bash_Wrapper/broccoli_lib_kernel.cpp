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

int Calculate5DIndex(int x, int y, int z, int t, int v, int DATA_W, int DATA_H, int DATA_D, int VALUES)
{
	return x + y * DATA_W + z * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D + v * DATA_W * DATA_H * DATA_D * VALUES;
}

// For parametric image registration
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

#define VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_ROWS 32
#define VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_ROWS 8
#define VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_ROWS 8



__kernel void SeparableConvolutionRows(__global float *Filter_Response,
	                                   __global const float* Volume, 
									   __global const float* Certainty, 
									   __constant float *c_Smoothing_Filter_Y, 
									   __private int t, 
									   __private int DATA_W, 
									   __private int DATA_H, 
									   __private int DATA_D, 
									   __private int DATA_T)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_group_id(2) * VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_ROWS + get_local_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

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
		l_Volume[tIdx.z][tIdx.y][tIdx.x] = Volume[Calculate4DIndex(x,y - 4,z,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y - 4,z,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 2) < DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y][tIdx.x] = Volume[Calculate4DIndex(x,y - 4,z + 2,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y - 4,z + 2,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y][tIdx.x] = Volume[Calculate4DIndex(x,y - 4,z + 4,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y - 4,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 6) < DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y][tIdx.x] = Volume[Calculate4DIndex(x,y - 4,z + 6,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y - 4,z + 6,DATA_W, DATA_H)];
	}

	// Second half main data + lower apron

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && (z < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 8][tIdx.x] = Volume[Calculate4DIndex(x,y + 4,z,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y + 4,z,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 2) < DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x] = Volume[Calculate4DIndex(x,y + 4,z + 2,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y + 4,z + 2,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x] = Volume[Calculate4DIndex(x,y + 4,z + 4,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y + 4,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 6) < DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x] = Volume[Calculate4DIndex(x,y + 4,z + 6,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y + 4,z + 6,DATA_W, DATA_H)];
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

__kernel void SeparableConvolutionRowsAMD(__global float *Filter_Response, 
	                                      __global const float* Volume, 
										  __global const float* Certainty, 
										  __constant float *c_Smoothing_Filter_Y, 
										  __private int t, 
										  __private int DATA_W, 
										  __private int DATA_H, 
										  __private int DATA_D, 
										  __private int DATA_T)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_group_id(2) * VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_ROWS + get_local_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	// 8 * 8 * 32 valid filter responses = 2048
	
	
	__local float l_Volume[8][16][32];

	// Reset local memory

	l_Volume[tIdx.z][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 1][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 2][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 3][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 4][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 5][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 6][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 7][tIdx.y][tIdx.x] = 0.0f;

	l_Volume[tIdx.z][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 1][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 3][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 5][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 7][tIdx.y + 8][tIdx.x] = 0.0f;

	// Read data into shared memory

	// Upper apron + first half main data

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && (z < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y][tIdx.x] = Volume[Calculate4DIndex(x,y - 4,z,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y - 4,z,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 1) < DATA_D) )
	{
		l_Volume[tIdx.z + 1][tIdx.y][tIdx.x] = Volume[Calculate4DIndex(x,y - 4,z + 1,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y - 4,z + 1,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 2) < DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y][tIdx.x] = Volume[Calculate4DIndex(x,y - 4,z + 2,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y - 4,z + 2,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 3) < DATA_D) )
	{
		l_Volume[tIdx.z + 3][tIdx.y][tIdx.x] = Volume[Calculate4DIndex(x,y - 4,z + 3,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y - 4,z + 3,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y][tIdx.x] = Volume[Calculate4DIndex(x,y - 4,z + 4,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y - 4,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 5) < DATA_D) )
	{
		l_Volume[tIdx.z + 5][tIdx.y][tIdx.x] = Volume[Calculate4DIndex(x,y - 4,z + 5,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y - 4,z + 5,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 6) < DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y][tIdx.x] = Volume[Calculate4DIndex(x,y - 4,z + 6,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y - 4,z + 6,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y - 4) >= 0) && ((y - 4) < DATA_H) && ((z + 7) < DATA_D) )
	{
		l_Volume[tIdx.z + 7][tIdx.y][tIdx.x] = Volume[Calculate4DIndex(x,y - 4,z + 7,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y - 4,z + 7,DATA_W, DATA_H)];
	}

	// Second half main data + lower apron

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && (z < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 8][tIdx.x] = Volume[Calculate4DIndex(x,y + 4,z,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y + 4,z,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 1) < DATA_D) )
	{
		l_Volume[tIdx.z + 1][tIdx.y + 8][tIdx.x] = Volume[Calculate4DIndex(x,y + 4,z + 1,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y + 4,z + 1,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 2) < DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x] = Volume[Calculate4DIndex(x,y + 4,z + 2,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y + 4,z + 2,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 3) < DATA_D) )
	{
		l_Volume[tIdx.z + 3][tIdx.y + 8][tIdx.x] = Volume[Calculate4DIndex(x,y + 4,z + 3,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y + 4,z + 3,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x] = Volume[Calculate4DIndex(x,y + 4,z + 4,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y + 4,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 5) < DATA_D) )
	{
		l_Volume[tIdx.z + 5][tIdx.y + 8][tIdx.x] = Volume[Calculate4DIndex(x,y + 4,z + 5,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y + 4,z + 5,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 6) < DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x] = Volume[Calculate4DIndex(x,y + 4,z + 6,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y + 4,z + 6,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 7) < DATA_D) )
	{
		l_Volume[tIdx.z + 7][tIdx.y + 8][tIdx.x] = Volume[Calculate4DIndex(x,y + 4,z + 7,t,DATA_W, DATA_H, DATA_D)] * Certainty[Calculate3DIndex(x,y + 4,z + 7,DATA_W, DATA_H)];
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

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 1) < DATA_D) )
	{
	    float sum = 0.0f;

		sum += l_Volume[tIdx.z + 1][tIdx.y + 0][tIdx.x] * c_Smoothing_Filter_Y[8];
		sum += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Y[7];
		sum += l_Volume[tIdx.z + 1][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Y[6];
		sum += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Y[5];
		sum += l_Volume[tIdx.z + 1][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Y[4];
		sum += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Y[3];
		sum += l_Volume[tIdx.z + 1][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Y[2];
		sum += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Y[1];
		sum += l_Volume[tIdx.z + 1][tIdx.y + 8][tIdx.x] * c_Smoothing_Filter_Y[0];

		Filter_Response[Calculate3DIndex(x,y,z + 1,DATA_W, DATA_H)] = sum;
		//Filter_Response[Calculate4DIndex(x,y,z + 1,t,DATA_W, DATA_H,DATA_D)] = sum;
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

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 3) < DATA_D) )
	{
	    float sum = 0.0f;

		sum += l_Volume[tIdx.z + 3][tIdx.y + 0][tIdx.x] * c_Smoothing_Filter_Y[8];
		sum += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Y[7];
		sum += l_Volume[tIdx.z + 3][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Y[6];
		sum += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Y[5];
		sum += l_Volume[tIdx.z + 3][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Y[4];
		sum += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Y[3];
		sum += l_Volume[tIdx.z + 3][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Y[2];
		sum += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Y[1];
		sum += l_Volume[tIdx.z + 3][tIdx.y + 8][tIdx.x] * c_Smoothing_Filter_Y[0];

		Filter_Response[Calculate3DIndex(x,y,z + 3,DATA_W, DATA_H)] = sum;
		//Filter_Response[Calculate4DIndex(x,y,z + 3,t,DATA_W, DATA_H,DATA_D)] = sum;
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

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 5) < DATA_D) )
	{
	    float sum = 0.0f;

		sum += l_Volume[tIdx.z + 5][tIdx.y + 0][tIdx.x] * c_Smoothing_Filter_Y[8];
		sum += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Y[7];
		sum += l_Volume[tIdx.z + 5][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Y[6];
		sum += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Y[5];
		sum += l_Volume[tIdx.z + 5][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Y[4];
		sum += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Y[3];
		sum += l_Volume[tIdx.z + 5][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Y[2];
		sum += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Y[1];
		sum += l_Volume[tIdx.z + 5][tIdx.y + 8][tIdx.x] * c_Smoothing_Filter_Y[0];

		Filter_Response[Calculate3DIndex(x,y,z + 5,DATA_W, DATA_H)] = sum;
		//Filter_Response[Calculate4DIndex(x,y,z + 5,t,DATA_W, DATA_H,DATA_D)] = sum;
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

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 7) < DATA_D) )
	{
	    float sum = 0.0f;

		sum += l_Volume[tIdx.z + 7][tIdx.y + 0][tIdx.x] * c_Smoothing_Filter_Y[8];
		sum += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Y[7];
		sum += l_Volume[tIdx.z + 7][tIdx.y + 2][tIdx.x] * c_Smoothing_Filter_Y[6];
		sum += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Y[5];
		sum += l_Volume[tIdx.z + 7][tIdx.y + 4][tIdx.x] * c_Smoothing_Filter_Y[4];
		sum += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Y[3];
		sum += l_Volume[tIdx.z + 7][tIdx.y + 6][tIdx.x] * c_Smoothing_Filter_Y[2];
		sum += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Y[1];
		sum += l_Volume[tIdx.z + 7][tIdx.y + 8][tIdx.x] * c_Smoothing_Filter_Y[0];

		Filter_Response[Calculate3DIndex(x,y,z + 7,DATA_W, DATA_H)] = sum;
		//Filter_Response[Calculate4DIndex(x,y,z + 7,t,DATA_W, DATA_H,DATA_D)] = sum;
	}
	
}

#define VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_COLUMNS 24
#define VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_COLUMNS 16
#define VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_COLUMNS 8


__kernel void SeparableConvolutionColumns(__global float *Filter_Response, 
	                                      __global float* Volume, 
										  __constant float *c_Smoothing_Filter_X, 
										  __private int t, 
										  __private int DATA_W, 
										  __private int DATA_H, 
										  __private int DATA_D, 
										  __private int DATA_T)
{
	//int x = get_local_size(0) * get_group_id(0) / 32 * 24 + get_local_id(0);;
	//int y = get_local_size(1) * get_group_id(1) * 2 + get_local_id(1);
	//int z = get_local_size(2) * get_group_id(2) * 4 + get_local_id(2);  

	int x = get_group_id(0) * VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_COLUMNS + get_local_id(0);
	int y = get_group_id(1) * VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_COLUMNS + get_local_id(1);
	int z = get_group_id(2) * VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_COLUMNS + get_local_id(2);	

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};
	
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

__kernel void SeparableConvolutionColumnsAMD(__global float *Filter_Response, 
	                                         __global float* Volume, 
											 __constant float *c_Smoothing_Filter_X, 
											 __private int t, 
											 __private int DATA_W, 
											 __private int DATA_H, 
											 __private int DATA_D, 
											 __private int DATA_T)
{
	//int x = get_local_size(0) * get_group_id(0) / 32 * 24 + get_local_id(0);;
	//int y = get_local_size(1) * get_group_id(1) * 2 + get_local_id(1);
	//int z = get_local_size(2) * get_group_id(2) * 4 + get_local_id(2);  

	int x = get_group_id(0) * VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_COLUMNS + get_local_id(0);
	int y = get_group_id(1) * VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_COLUMNS + get_local_id(1);
	int z = get_group_id(2) * VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_COLUMNS + get_local_id(2);	

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};
	
	// 8 * 16 * 24 valid filter responses = 3072
	__local float l_Volume[8][16][32];

	// Reset shared memory
	l_Volume[tIdx.z][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 1][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 2][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 3][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 4][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 5][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 6][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 7][tIdx.y][tIdx.x] = 0.0f;

	l_Volume[tIdx.z][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 1][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 3][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 5][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 7][tIdx.y + 8][tIdx.x] = 0.0f;

	// Read data into shared memory

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && (z < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x - 4,y,z,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && ((z + 1) < DATA_D) )
	{
		l_Volume[tIdx.z + 1][tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x - 4,y,z + 1,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && ((z + 2) < DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x - 4,y,z + 2,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && ((z + 3) < DATA_D) )
	{
		l_Volume[tIdx.z + 3][tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x - 4,y,z + 3,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x - 4,y,z + 4,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && ((z + 5) < DATA_D) )
	{
		l_Volume[tIdx.z + 5][tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x - 4,y,z + 5,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && ((z + 6) < DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x - 4,y,z + 6,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && (y < DATA_H) && ((z + 7) < DATA_D) )
	{
		l_Volume[tIdx.z + 7][tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x - 4,y,z + 7,DATA_W, DATA_H)];
	}


	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 8) < DATA_H) && (z < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 8][tIdx.x] = Volume[Calculate3DIndex(x - 4,y + 8,z,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 8) < DATA_H) && ((z + 1) < DATA_D) )
	{
		l_Volume[tIdx.z + 1][tIdx.y + 8][tIdx.x] = Volume[Calculate3DIndex(x - 4,y + 8,z + 1,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 8) < DATA_H) && ((z + 2) < DATA_D) )
	{
		l_Volume[tIdx.z + 2][tIdx.y + 8][tIdx.x] = Volume[Calculate3DIndex(x - 4,y + 8,z + 2,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 8) < DATA_H) && ((z + 3) < DATA_D) )
	{
		l_Volume[tIdx.z + 3][tIdx.y + 8][tIdx.x] = Volume[Calculate3DIndex(x - 4,y + 8,z + 3,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 8) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 4][tIdx.y + 8][tIdx.x] = Volume[Calculate3DIndex(x - 4,y + 8,z + 4,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 8) < DATA_H) && ((z + 5) < DATA_D) )
	{
		l_Volume[tIdx.z + 5][tIdx.y + 8][tIdx.x] = Volume[Calculate3DIndex(x - 4,y + 8,z + 5,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 8) < DATA_H) && ((z + 6) < DATA_D) )
	{
		l_Volume[tIdx.z + 6][tIdx.y + 8][tIdx.x] = Volume[Calculate3DIndex(x - 4,y + 8,z + 6,DATA_W, DATA_H)];
	}

	if ( ((x - 4) >= 0) && ((x - 4) < DATA_W) && ((y + 8) < DATA_H) && ((z + 7) < DATA_D) )
	{
		l_Volume[tIdx.z + 7][tIdx.y + 8][tIdx.x] = Volume[Calculate3DIndex(x - 4,y + 8,z + 7,DATA_W, DATA_H)];
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

		if ( (x < DATA_W) && (y < DATA_H) && ((z + 1) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += l_Volume[tIdx.z + 1][tIdx.y][tIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += l_Volume[tIdx.z + 1][tIdx.y][tIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += l_Volume[tIdx.z + 1][tIdx.y][tIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += l_Volume[tIdx.z + 1][tIdx.y][tIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += l_Volume[tIdx.z + 1][tIdx.y][tIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += l_Volume[tIdx.z + 1][tIdx.y][tIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += l_Volume[tIdx.z + 1][tIdx.y][tIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += l_Volume[tIdx.z + 1][tIdx.y][tIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += l_Volume[tIdx.z + 1][tIdx.y][tIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Response[Calculate3DIndex(x,y,z + 1,DATA_W, DATA_H)] = sum;
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

		if ( (x < DATA_W) && (y < DATA_H) && ((z + 3) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += l_Volume[tIdx.z + 3][tIdx.y][tIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += l_Volume[tIdx.z + 3][tIdx.y][tIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += l_Volume[tIdx.z + 3][tIdx.y][tIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += l_Volume[tIdx.z + 3][tIdx.y][tIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += l_Volume[tIdx.z + 3][tIdx.y][tIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += l_Volume[tIdx.z + 3][tIdx.y][tIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += l_Volume[tIdx.z + 3][tIdx.y][tIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += l_Volume[tIdx.z + 3][tIdx.y][tIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += l_Volume[tIdx.z + 3][tIdx.y][tIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Response[Calculate3DIndex(x,y,z + 3,DATA_W, DATA_H)] = sum;
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

		if ( (x < DATA_W) && (y < DATA_H) && ((z + 5) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += l_Volume[tIdx.z + 5][tIdx.y][tIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += l_Volume[tIdx.z + 5][tIdx.y][tIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += l_Volume[tIdx.z + 5][tIdx.y][tIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += l_Volume[tIdx.z + 5][tIdx.y][tIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += l_Volume[tIdx.z + 5][tIdx.y][tIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += l_Volume[tIdx.z + 5][tIdx.y][tIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += l_Volume[tIdx.z + 5][tIdx.y][tIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += l_Volume[tIdx.z + 5][tIdx.y][tIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += l_Volume[tIdx.z + 5][tIdx.y][tIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Response[Calculate3DIndex(x,y,z + 5,DATA_W, DATA_H)] = sum;
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

		if ( (x < DATA_W) && (y < DATA_H) && ((z + 7) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += l_Volume[tIdx.z + 7][tIdx.y][tIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += l_Volume[tIdx.z + 7][tIdx.y][tIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += l_Volume[tIdx.z + 7][tIdx.y][tIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += l_Volume[tIdx.z + 7][tIdx.y][tIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += l_Volume[tIdx.z + 7][tIdx.y][tIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += l_Volume[tIdx.z + 7][tIdx.y][tIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += l_Volume[tIdx.z + 7][tIdx.y][tIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += l_Volume[tIdx.z + 7][tIdx.y][tIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += l_Volume[tIdx.z + 7][tIdx.y][tIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Response[Calculate3DIndex(x,y,z + 7,DATA_W, DATA_H)] = sum;
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

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 1) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += l_Volume[tIdx.z + 1][tIdx.y + 8][tIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += l_Volume[tIdx.z + 1][tIdx.y + 8][tIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += l_Volume[tIdx.z + 1][tIdx.y + 8][tIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += l_Volume[tIdx.z + 1][tIdx.y + 8][tIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += l_Volume[tIdx.z + 1][tIdx.y + 8][tIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += l_Volume[tIdx.z + 1][tIdx.y + 8][tIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += l_Volume[tIdx.z + 1][tIdx.y + 8][tIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += l_Volume[tIdx.z + 1][tIdx.y + 8][tIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += l_Volume[tIdx.z + 1][tIdx.y + 8][tIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Response[Calculate3DIndex(x,y + 8,z + 1,DATA_W, DATA_H)] = sum;
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

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 3) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += l_Volume[tIdx.z + 3][tIdx.y + 8][tIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += l_Volume[tIdx.z + 3][tIdx.y + 8][tIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += l_Volume[tIdx.z + 3][tIdx.y + 8][tIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += l_Volume[tIdx.z + 3][tIdx.y + 8][tIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += l_Volume[tIdx.z + 3][tIdx.y + 8][tIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += l_Volume[tIdx.z + 3][tIdx.y + 8][tIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += l_Volume[tIdx.z + 3][tIdx.y + 8][tIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += l_Volume[tIdx.z + 3][tIdx.y + 8][tIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += l_Volume[tIdx.z + 3][tIdx.y + 8][tIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Response[Calculate3DIndex(x,y + 8,z + 3,DATA_W, DATA_H)] = sum;
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

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 5) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += l_Volume[tIdx.z + 5][tIdx.y + 8][tIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += l_Volume[tIdx.z + 5][tIdx.y + 8][tIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += l_Volume[tIdx.z + 5][tIdx.y + 8][tIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += l_Volume[tIdx.z + 5][tIdx.y + 8][tIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += l_Volume[tIdx.z + 5][tIdx.y + 8][tIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += l_Volume[tIdx.z + 5][tIdx.y + 8][tIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += l_Volume[tIdx.z + 5][tIdx.y + 8][tIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += l_Volume[tIdx.z + 5][tIdx.y + 8][tIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += l_Volume[tIdx.z + 5][tIdx.y + 8][tIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Response[Calculate3DIndex(x,y + 8,z + 5,DATA_W, DATA_H)] = sum;
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

		if ( (x < DATA_W) && ((y + 8) < DATA_H) && ((z + 7) < DATA_D) )
		{
		    float sum = 0.0f;

			sum += l_Volume[tIdx.z + 7][tIdx.y + 8][tIdx.x + 0] * c_Smoothing_Filter_X[8];
			sum += l_Volume[tIdx.z + 7][tIdx.y + 8][tIdx.x + 1] * c_Smoothing_Filter_X[7];
			sum += l_Volume[tIdx.z + 7][tIdx.y + 8][tIdx.x + 2] * c_Smoothing_Filter_X[6];
			sum += l_Volume[tIdx.z + 7][tIdx.y + 8][tIdx.x + 3] * c_Smoothing_Filter_X[5];
			sum += l_Volume[tIdx.z + 7][tIdx.y + 8][tIdx.x + 4] * c_Smoothing_Filter_X[4];
			sum += l_Volume[tIdx.z + 7][tIdx.y + 8][tIdx.x + 5] * c_Smoothing_Filter_X[3];
			sum += l_Volume[tIdx.z + 7][tIdx.y + 8][tIdx.x + 6] * c_Smoothing_Filter_X[2];
			sum += l_Volume[tIdx.z + 7][tIdx.y + 8][tIdx.x + 7] * c_Smoothing_Filter_X[1];
			sum += l_Volume[tIdx.z + 7][tIdx.y + 8][tIdx.x + 8] * c_Smoothing_Filter_X[0];

			Filter_Response[Calculate3DIndex(x,y + 8,z + 7,DATA_W, DATA_H)] = sum;
		}
	}
}

#define VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_RODS 32
#define VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_RODS 8
#define VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_RODS 8


__kernel void SeparableConvolutionRods(__global float *Filter_Response, 
	                                   __global float* Volume, 
									   __global const float* Smoothed_Certainty, 
									   __constant float *c_Smoothing_Filter_Z, 
									   __private int t, 
									   __private int DATA_W, 
									   __private int DATA_H, 
									   __private int DATA_D, 
									   __private int DATA_T)
{
	//int x = get_global_id(0);
	//int y = get_local_size(1) * get_group_id(1) * 4 + get_local_id(1); 
	//int z = get_global_id(2);

	int x = get_global_id(0);
	int y = get_group_id(1) * VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_RODS + get_local_id(1); 
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	
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

		Filter_Response[Calculate4DIndex(x,y,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];
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

		Filter_Response[Calculate4DIndex(x,y + 2,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate3DIndex(x,y + 2,z,DATA_W,DATA_H)];
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

		Filter_Response[Calculate4DIndex(x,y + 4,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate3DIndex(x,y + 4,z,DATA_W,DATA_H)];
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

		Filter_Response[Calculate4DIndex(x,y + 6,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate3DIndex(x,y + 6,z,DATA_W,DATA_H)];
	}
}


__kernel void SeparableConvolutionRodsAMD(__global float *Filter_Response,
	                                      __global float* Volume, 
										  __global const float* Smoothed_Certainty, 
										  __constant float *c_Smoothing_Filter_Z, 
										  __private int t, 
										  __private int DATA_W, 
										  __private int DATA_H, 
										  __private int DATA_D, 
										  __private int DATA_T)
{
	int x = get_global_id(0);
	int y = get_group_id(1) * VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_RODS + get_local_id(1); 
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	
	// 8 * 8 * 32 valid filter responses = 2048
	__local float l_Volume[16][8][32];

	// Reset shared memory
	l_Volume[tIdx.z][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z][tIdx.y + 1][tIdx.x] = 0.0f;
	l_Volume[tIdx.z][tIdx.y + 2][tIdx.x] = 0.0f;
	l_Volume[tIdx.z][tIdx.y + 3][tIdx.x] = 0.0f;
	l_Volume[tIdx.z][tIdx.y + 4][tIdx.x] = 0.0f;
	l_Volume[tIdx.z][tIdx.y + 5][tIdx.x] = 0.0f;
	l_Volume[tIdx.z][tIdx.y + 6][tIdx.x] = 0.0f;
	l_Volume[tIdx.z][tIdx.y + 7][tIdx.x] = 0.0f;

	l_Volume[tIdx.z + 8][tIdx.y][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 8][tIdx.y + 1][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 8][tIdx.y + 2][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 8][tIdx.y + 3][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 8][tIdx.y + 4][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 8][tIdx.y + 5][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 8][tIdx.y + 6][tIdx.x] = 0.0f;
	l_Volume[tIdx.z + 8][tIdx.y + 7][tIdx.x] = 0.0f;
    
	// Read data into shared memory

	// Above apron + first half main data

	if ( (x < DATA_W) && (y < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x,y,z - 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 1) < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 1][tIdx.x] = Volume[Calculate3DIndex(x,y + 1,z - 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 2) < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 2][tIdx.x] = Volume[Calculate3DIndex(x,y + 2,z - 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 3) < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 3][tIdx.x] = Volume[Calculate3DIndex(x,y + 3,z - 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 4][tIdx.x] = Volume[Calculate3DIndex(x,y + 4,z - 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 5) < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 5][tIdx.x] = Volume[Calculate3DIndex(x,y + 5,z - 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 6) < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 6][tIdx.x] = Volume[Calculate3DIndex(x,y + 6,z - 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 7) < DATA_H) && ((z - 4) >= 0) && ((z - 4) < DATA_D) )
	{
		l_Volume[tIdx.z][tIdx.y + 7][tIdx.x] = Volume[Calculate3DIndex(x,y + 7,z - 4,DATA_W, DATA_H)];
	}

	// Second half main data + below apron

	if ( (x < DATA_W) && (y < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x,y,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 1) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y + 1][tIdx.x] = Volume[Calculate3DIndex(x,y + 1,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 2) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y + 2][tIdx.x] = Volume[Calculate3DIndex(x,y + 2,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 3) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y + 3][tIdx.x] = Volume[Calculate3DIndex(x,y + 3,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 4) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y + 4][tIdx.x] = Volume[Calculate3DIndex(x,y + 4,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 5) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y + 5][tIdx.x] = Volume[Calculate3DIndex(x,y + 5,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 6) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y + 6][tIdx.x] = Volume[Calculate3DIndex(x,y + 6,z + 4,DATA_W, DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 7) < DATA_H) && ((z + 4) < DATA_D) )
	{
		l_Volume[tIdx.z + 8][tIdx.y + 7][tIdx.x] = Volume[Calculate3DIndex(x,y + 7,z + 4,DATA_W, DATA_H)];
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

		Filter_Response[Calculate4DIndex(x,y,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 1) < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += l_Volume[tIdx.z + 0][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Z[8];
		sum += l_Volume[tIdx.z + 1][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Z[7];
		sum += l_Volume[tIdx.z + 2][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Z[6];
		sum += l_Volume[tIdx.z + 3][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Z[5];
		sum += l_Volume[tIdx.z + 4][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Z[4];
		sum += l_Volume[tIdx.z + 5][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Z[3];
		sum += l_Volume[tIdx.z + 6][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Z[2];
		sum += l_Volume[tIdx.z + 7][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Z[1];
		sum += l_Volume[tIdx.z + 8][tIdx.y + 1][tIdx.x] * c_Smoothing_Filter_Z[0];

		Filter_Response[Calculate4DIndex(x,y + 1,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate3DIndex(x,y + 1,z,DATA_W,DATA_H)];
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

		Filter_Response[Calculate4DIndex(x,y + 2,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate3DIndex(x,y + 2,z,DATA_W,DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 3) < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += l_Volume[tIdx.z + 0][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Z[8];
		sum += l_Volume[tIdx.z + 1][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Z[7];
		sum += l_Volume[tIdx.z + 2][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Z[6];
		sum += l_Volume[tIdx.z + 3][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Z[5];
		sum += l_Volume[tIdx.z + 4][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Z[4];
		sum += l_Volume[tIdx.z + 5][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Z[3];
		sum += l_Volume[tIdx.z + 6][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Z[2];
		sum += l_Volume[tIdx.z + 7][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Z[1];
		sum += l_Volume[tIdx.z + 8][tIdx.y + 3][tIdx.x] * c_Smoothing_Filter_Z[0];

		Filter_Response[Calculate4DIndex(x,y + 3,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate3DIndex(x,y + 3,z,DATA_W,DATA_H)];
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

		Filter_Response[Calculate4DIndex(x,y + 4,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate3DIndex(x,y + 4,z,DATA_W,DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 5) < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += l_Volume[tIdx.z + 0][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Z[8];
		sum += l_Volume[tIdx.z + 1][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Z[7];
		sum += l_Volume[tIdx.z + 2][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Z[6];
		sum += l_Volume[tIdx.z + 3][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Z[5];
		sum += l_Volume[tIdx.z + 4][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Z[4];
		sum += l_Volume[tIdx.z + 5][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Z[3];
		sum += l_Volume[tIdx.z + 6][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Z[2];
		sum += l_Volume[tIdx.z + 7][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Z[1];
		sum += l_Volume[tIdx.z + 8][tIdx.y + 5][tIdx.x] * c_Smoothing_Filter_Z[0];

		Filter_Response[Calculate4DIndex(x,y + 5,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate3DIndex(x,y + 5,z,DATA_W,DATA_H)];
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

		Filter_Response[Calculate4DIndex(x,y + 6,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate3DIndex(x,y + 6,z,DATA_W,DATA_H)];
	}

	if ( (x < DATA_W) && ((y + 7) < DATA_H) && (z < DATA_D) )
	{
	    float sum = 0.0f;

		sum += l_Volume[tIdx.z + 0][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Z[8];
		sum += l_Volume[tIdx.z + 1][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Z[7];
		sum += l_Volume[tIdx.z + 2][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Z[6];
		sum += l_Volume[tIdx.z + 3][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Z[5];
		sum += l_Volume[tIdx.z + 4][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Z[4];
		sum += l_Volume[tIdx.z + 5][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Z[3];
		sum += l_Volume[tIdx.z + 6][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Z[2];
		sum += l_Volume[tIdx.z + 7][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Z[1];
		sum += l_Volume[tIdx.z + 8][tIdx.y + 7][tIdx.x] * c_Smoothing_Filter_Z[0];

		Filter_Response[Calculate4DIndex(x,y + 7,z,t,DATA_W,DATA_H,DATA_D)] = sum / Smoothed_Certainty[Calculate3DIndex(x,y + 7,z,DATA_W,DATA_H)];
	}
}

#define HALO 3

#define VALID_FILTER_RESPONSES_X_CONVOLUTION_2D 90
#define VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D 58

#define VALID_FILTER_RESPONSES_X_CONVOLUTION_2D_AMD 122
#define VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D_AMD 58

// Non-separable 3D convolution

typedef struct tag_float6 {float a; float b; float c; float d; float e; float f;} float6;

// Non-separable 2D convolution for three complex valued 7 x 7 filters, unrolled for performance
float6 Conv_2D_Unrolled_7x7_ThreeFilters(__local float image[64][96],
	                                     int y, 
										 int x, 
										 __constant float2* Filter_1, 
										 __constant float2* Filter_2, 
										 __constant float2* Filter_3)
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
    sum.a += pixel * Filter_1[6*7 + 6].x;
	sum.b += pixel * Filter_1[6*7 + 6].y;
	sum.c += pixel * Filter_2[6*7 + 6].x;
	sum.d += pixel * Filter_2[6*7 + 6].y;
	sum.e += pixel * Filter_3[6*7 + 6].x;
	sum.f += pixel * Filter_3[6*7 + 6].y;
    pixel = image[y - 2][x - 3]; 
    sum.a += pixel * Filter_1[5*7 + 6].x;
	sum.b += pixel * Filter_1[5*7 + 6].y;
	sum.c += pixel * Filter_2[5*7 + 6].x;
	sum.d += pixel * Filter_2[5*7 + 6].y;
	sum.e += pixel * Filter_3[5*7 + 6].x;
	sum.f += pixel * Filter_3[5*7 + 6].y;
	pixel = image[y - 1][x - 3]; 
    sum.a += pixel * Filter_1[4*7 + 6].x;
	sum.b += pixel * Filter_1[4*7 + 6].y;
	sum.c += pixel * Filter_2[4*7 + 6].x;
	sum.d += pixel * Filter_2[4*7 + 6].y;
	sum.e += pixel * Filter_3[4*7 + 6].x;
	sum.f += pixel * Filter_3[4*7 + 6].y;
	pixel = image[y + 0][x - 3]; 
    sum.a += pixel * Filter_1[3*7 + 6].x;
	sum.b += pixel * Filter_1[3*7 + 6].y;
	sum.c += pixel * Filter_2[3*7 + 6].x;
	sum.d += pixel * Filter_2[3*7 + 6].y;
	sum.e += pixel * Filter_3[3*7 + 6].x;
	sum.f += pixel * Filter_3[3*7 + 6].y;
    pixel = image[y + 1][x - 3]; 
    sum.a += pixel * Filter_1[2*7 + 6].x;
	sum.b += pixel * Filter_1[2*7 + 6].y;
	sum.c += pixel * Filter_2[2*7 + 6].x;
	sum.d += pixel * Filter_2[2*7 + 6].y;
	sum.e += pixel * Filter_3[2*7 + 6].x;
	sum.f += pixel * Filter_3[2*7 + 6].y;
	pixel = image[y + 2][x - 3]; 
    sum.a += pixel * Filter_1[1*7 + 6].x;
	sum.b += pixel * Filter_1[1*7 + 6].y;
	sum.c += pixel * Filter_2[1*7 + 6].x;
	sum.d += pixel * Filter_2[1*7 + 6].y;
	sum.e += pixel * Filter_3[1*7 + 6].x;
	sum.f += pixel * Filter_3[1*7 + 6].y;
	pixel = image[y + 3][x - 3]; 
    sum.a += pixel * Filter_1[6].x;
	sum.b += pixel * Filter_1[6].y;
	sum.c += pixel * Filter_2[6].x;
	sum.d += pixel * Filter_2[6].y;
	sum.e += pixel * Filter_3[6].x;
	sum.f += pixel * Filter_3[6].y;

    pixel = image[y - 3][x - 2]; 
    sum.a += pixel * Filter_1[6*7 + 5].x;
	sum.b += pixel * Filter_1[6*7 + 5].y;
	sum.c += pixel * Filter_2[6*7 + 5].x;
	sum.d += pixel * Filter_2[6*7 + 5].y;
	sum.e += pixel * Filter_3[6*7 + 5].x;
	sum.f += pixel * Filter_3[6*7 + 5].y;
    pixel = image[y - 2][x - 2]; 
    sum.a += pixel * Filter_1[5*7 + 5].x;
	sum.b += pixel * Filter_1[5*7 + 5].y;
	sum.c += pixel * Filter_2[5*7 + 5].x;
	sum.d += pixel * Filter_2[5*7 + 5].y;
	sum.e += pixel * Filter_3[5*7 + 5].x;
	sum.f += pixel * Filter_3[5*7 + 5].y;
    pixel = image[y - 1][x - 2]; 
    sum.a += pixel * Filter_1[4*7 + 5].x;
	sum.b += pixel * Filter_1[4*7 + 5].y;
	sum.c += pixel * Filter_2[4*7 + 5].x;
	sum.d += pixel * Filter_2[4*7 + 5].y;
	sum.e += pixel * Filter_3[4*7 + 5].x;
	sum.f += pixel * Filter_3[4*7 + 5].y;
    pixel = image[y + 0][x - 2]; 
    sum.a += pixel * Filter_1[3*7 + 5].x;
	sum.b += pixel * Filter_1[3*7 + 5].y;
	sum.c += pixel * Filter_2[3*7 + 5].x;
	sum.d += pixel * Filter_2[3*7 + 5].y;
	sum.e += pixel * Filter_3[3*7 + 5].x;
	sum.f += pixel * Filter_3[3*7 + 5].y;
    pixel = image[y + 1][x - 2]; 
    sum.a += pixel * Filter_1[2*7 + 5].x;
	sum.b += pixel * Filter_1[2*7 + 5].y;
	sum.c += pixel * Filter_2[2*7 + 5].x;
	sum.d += pixel * Filter_2[2*7 + 5].y;
	sum.e += pixel * Filter_3[2*7 + 5].x;
	sum.f += pixel * Filter_3[2*7 + 5].y;
    pixel = image[y + 2][x - 2]; 
    sum.a += pixel * Filter_1[1*7 + 5].x;
	sum.b += pixel * Filter_1[1*7 + 5].y;
	sum.c += pixel * Filter_2[1*7 + 5].x;
	sum.d += pixel * Filter_2[1*7 + 5].y;
	sum.e += pixel * Filter_3[1*7 + 5].x;
	sum.f += pixel * Filter_3[1*7 + 5].y;
    pixel = image[y + 3][x - 2]; 
    sum.a += pixel * Filter_1[5].x;
	sum.b += pixel * Filter_1[5].y;
	sum.c += pixel * Filter_2[5].x;
	sum.d += pixel * Filter_2[5].y;
	sum.e += pixel * Filter_3[5].x;
	sum.f += pixel * Filter_3[5].y;


    pixel = image[y - 3][x - 1]; 
    sum.a += pixel * Filter_1[6*7 + 4].x;
	sum.b += pixel * Filter_1[6*7 + 4].y;
	sum.c += pixel * Filter_2[6*7 + 4].x;
	sum.d += pixel * Filter_2[6*7 + 4].y;
	sum.e += pixel * Filter_3[6*7 + 4].x;
	sum.f += pixel * Filter_3[6*7 + 4].y;
    pixel = image[y - 2][x - 1]; 
    sum.a += pixel * Filter_1[5*7 + 4].x;
	sum.b += pixel * Filter_1[5*7 + 4].y;
	sum.c += pixel * Filter_2[5*7 + 4].x;
	sum.d += pixel * Filter_2[5*7 + 4].y;
	sum.e += pixel * Filter_3[5*7 + 4].x;
	sum.f += pixel * Filter_3[5*7 + 4].y;
    pixel = image[y - 1][x - 1]; 
    sum.a += pixel * Filter_1[4*7 + 4].x;
	sum.b += pixel * Filter_1[4*7 + 4].y;
	sum.c += pixel * Filter_2[4*7 + 4].x;
	sum.d += pixel * Filter_2[4*7 + 4].y;
	sum.e += pixel * Filter_3[4*7 + 4].x;
	sum.f += pixel * Filter_3[4*7 + 4].y;
    pixel = image[y + 0][x - 1]; 
    sum.a += pixel * Filter_1[3*7 + 4].x;
	sum.b += pixel * Filter_1[3*7 + 4].y;
	sum.c += pixel * Filter_2[3*7 + 4].x;
	sum.d += pixel * Filter_2[3*7 + 4].y;
	sum.e += pixel * Filter_3[3*7 + 4].x;
	sum.f += pixel * Filter_3[3*7 + 4].y;
    pixel = image[y + 1][x - 1]; 
    sum.a += pixel * Filter_1[2*7 + 4].x;
	sum.b += pixel * Filter_1[2*7 + 4].y;
	sum.c += pixel * Filter_2[2*7 + 4].x;
	sum.d += pixel * Filter_2[2*7 + 4].y;
	sum.e += pixel * Filter_3[2*7 + 4].x;
	sum.f += pixel * Filter_3[2*7 + 4].y;
    pixel = image[y + 2][x - 1]; 
    sum.a += pixel * Filter_1[1*7 + 4].x;
	sum.b += pixel * Filter_1[1*7 + 4].y;
	sum.c += pixel * Filter_2[1*7 + 4].x;
	sum.d += pixel * Filter_2[1*7 + 4].y;
	sum.e += pixel * Filter_3[1*7 + 4].x;
	sum.f += pixel * Filter_3[1*7 + 4].y;
    pixel = image[y + 3][x - 1]; 
    sum.a += pixel * Filter_1[4].x;
	sum.b += pixel * Filter_1[4].y;
	sum.c += pixel * Filter_2[4].x;
	sum.d += pixel * Filter_2[4].y;
	sum.e += pixel * Filter_3[4].x;
	sum.f += pixel * Filter_3[4].y;


    pixel = image[y - 3][x + 0]; 
    sum.a += pixel * Filter_1[6*7 + 3].x;
	sum.b += pixel * Filter_1[6*7 + 3].y;
	sum.c += pixel * Filter_2[6*7 + 3].x;
	sum.d += pixel * Filter_2[6*7 + 3].y;
	sum.e += pixel * Filter_3[6*7 + 3].x;
	sum.f += pixel * Filter_3[6*7 + 3].y;
    pixel = image[y - 2][x + 0]; 
    sum.a += pixel * Filter_1[5*7 + 3].x;
	sum.b += pixel * Filter_1[5*7 + 3].y;
	sum.c += pixel * Filter_2[5*7 + 3].x;
	sum.d += pixel * Filter_2[5*7 + 3].y;
	sum.e += pixel * Filter_3[5*7 + 3].x;
	sum.f += pixel * Filter_3[5*7 + 3].y;
    pixel = image[y - 1][x + 0]; 
    sum.a += pixel * Filter_1[4*7 + 3].x;
	sum.b += pixel * Filter_1[4*7 + 3].y;
	sum.c += pixel * Filter_2[4*7 + 3].x;
	sum.d += pixel * Filter_2[4*7 + 3].y;
	sum.e += pixel * Filter_3[4*7 + 3].x;
	sum.f += pixel * Filter_3[4*7 + 3].y;
    pixel = image[y + 0][x + 0]; 
    sum.a += pixel * Filter_1[3*7 + 3].x;
	sum.b += pixel * Filter_1[3*7 + 3].y;
	sum.c += pixel * Filter_2[3*7 + 3].x;
	sum.d += pixel * Filter_2[3*7 + 3].y;
	sum.e += pixel * Filter_3[3*7 + 3].x;
	sum.f += pixel * Filter_3[3*7 + 3].y;
    pixel = image[y + 1][x + 0]; 
    sum.a += pixel * Filter_1[2*7 + 3].x;
	sum.b += pixel * Filter_1[2*7 + 3].y;
	sum.c += pixel * Filter_2[2*7 + 3].x;
	sum.d += pixel * Filter_2[2*7 + 3].y;
	sum.e += pixel * Filter_3[2*7 + 3].x;
	sum.f += pixel * Filter_3[2*7 + 3].y;
    pixel = image[y + 2][x + 0]; 
    sum.a += pixel * Filter_1[1*7 + 3].x;
	sum.b += pixel * Filter_1[1*7 + 3].y;
	sum.c += pixel * Filter_2[1*7 + 3].x;
	sum.d += pixel * Filter_2[1*7 + 3].y;
	sum.e += pixel * Filter_3[1*7 + 3].x;
	sum.f += pixel * Filter_3[1*7 + 3].y;
    pixel = image[y + 3][x + 0]; 
    sum.a += pixel * Filter_1[3].x;
	sum.b += pixel * Filter_1[3].y;
	sum.c += pixel * Filter_2[3].x;
	sum.d += pixel * Filter_2[3].y;
	sum.e += pixel * Filter_3[3].x;
	sum.f += pixel * Filter_3[3].y;

	pixel = image[y - 3][x + 1]; 
    sum.a += pixel * Filter_1[6*7 + 2].x;
	sum.b += pixel * Filter_1[6*7 + 2].y;
	sum.c += pixel * Filter_2[6*7 + 2].x;
	sum.d += pixel * Filter_2[6*7 + 2].y;
	sum.e += pixel * Filter_3[6*7 + 2].x;
	sum.f += pixel * Filter_3[6*7 + 2].y;
    pixel = image[y - 2][x + 1]; 
    sum.a += pixel * Filter_1[5*7 + 2].x;
	sum.b += pixel * Filter_1[5*7 + 2].y;
	sum.c += pixel * Filter_2[5*7 + 2].x;
	sum.d += pixel * Filter_2[5*7 + 2].y;
	sum.e += pixel * Filter_3[5*7 + 2].x;
	sum.f += pixel * Filter_3[5*7 + 2].y;
    pixel = image[y - 1][x + 1]; 
    sum.a += pixel * Filter_1[4*7 + 2].x;
	sum.b += pixel * Filter_1[4*7 + 2].y;
	sum.c += pixel * Filter_2[4*7 + 2].x;
	sum.d += pixel * Filter_2[4*7 + 2].y;
	sum.e += pixel * Filter_3[4*7 + 2].x;
	sum.f += pixel * Filter_3[4*7 + 2].y;
    pixel = image[y + 0][x + 1]; 
    sum.a += pixel * Filter_1[3*7 + 2].x;
	sum.b += pixel * Filter_1[3*7 + 2].y;
	sum.c += pixel * Filter_2[3*7 + 2].x;
	sum.d += pixel * Filter_2[3*7 + 2].y;
	sum.e += pixel * Filter_3[3*7 + 2].x;
	sum.f += pixel * Filter_3[3*7 + 2].y;
    pixel = image[y + 1][x + 1]; 
    sum.a += pixel * Filter_1[2*7 + 2].x;
	sum.b += pixel * Filter_1[2*7 + 2].y;
	sum.c += pixel * Filter_2[2*7 + 2].x;
	sum.d += pixel * Filter_2[2*7 + 2].y;
	sum.e += pixel * Filter_3[2*7 + 2].x;
	sum.f += pixel * Filter_3[2*7 + 2].y;
    pixel = image[y + 2][x + 1]; 
    sum.a += pixel * Filter_1[1*7 + 2].x;
	sum.b += pixel * Filter_1[1*7 + 2].y;
	sum.c += pixel * Filter_2[1*7 + 2].x;
	sum.d += pixel * Filter_2[1*7 + 2].y;
	sum.e += pixel * Filter_3[1*7 + 2].x;
	sum.f += pixel * Filter_3[1*7 + 2].y;
    pixel = image[y + 3][x + 1]; 
    sum.a += pixel * Filter_1[2].x;
	sum.b += pixel * Filter_1[2].y;
	sum.c += pixel * Filter_2[2].x;
	sum.d += pixel * Filter_2[2].y;
	sum.e += pixel * Filter_3[2].x;
	sum.f += pixel * Filter_3[2].y;
 
    pixel = image[y - 3][x + 2]; 
    sum.a += pixel * Filter_1[6*7 + 1].x;
	sum.b += pixel * Filter_1[6*7 + 1].y;
	sum.c += pixel * Filter_2[6*7 + 1].x;
	sum.d += pixel * Filter_2[6*7 + 1].y;
	sum.e += pixel * Filter_3[6*7 + 1].x;
	sum.f += pixel * Filter_3[6*7 + 1].y;
    pixel = image[y - 2][x + 2]; 
    sum.a += pixel * Filter_1[5*7 + 1].x;
	sum.b += pixel * Filter_1[5*7 + 1].y;
	sum.c += pixel * Filter_2[5*7 + 1].x;
	sum.d += pixel * Filter_2[5*7 + 1].y;
	sum.e += pixel * Filter_3[5*7 + 1].x;
	sum.f += pixel * Filter_3[5*7 + 1].y;
    pixel = image[y - 1][x + 2]; 
    sum.a += pixel * Filter_1[4*7 + 1].x;
	sum.b += pixel * Filter_1[4*7 + 1].y;
	sum.c += pixel * Filter_2[4*7 + 1].x;
	sum.d += pixel * Filter_2[4*7 + 1].y;
	sum.e += pixel * Filter_3[4*7 + 1].x;
	sum.f += pixel * Filter_3[4*7 + 1].y;
    pixel = image[y + 0][x + 2]; 
    sum.a += pixel * Filter_1[3*7 + 1].x;
	sum.b += pixel * Filter_1[3*7 + 1].y;
	sum.c += pixel * Filter_2[3*7 + 1].x;
	sum.d += pixel * Filter_2[3*7 + 1].y;
	sum.e += pixel * Filter_3[3*7 + 1].x;
	sum.f += pixel * Filter_3[3*7 + 1].y;
	pixel = image[y + 1][x + 2]; 
    sum.a += pixel * Filter_1[2*7 + 1].x;
	sum.b += pixel * Filter_1[2*7 + 1].y;
	sum.c += pixel * Filter_2[2*7 + 1].x;
	sum.d += pixel * Filter_2[2*7 + 1].y;
	sum.e += pixel * Filter_3[2*7 + 1].x;
	sum.f += pixel * Filter_3[2*7 + 1].y;
    pixel = image[y + 2][x + 2]; 
    sum.a += pixel * Filter_1[1*7 + 1].x;
	sum.b += pixel * Filter_1[1*7 + 1].y;
	sum.c += pixel * Filter_2[1*7 + 1].x;
	sum.d += pixel * Filter_2[1*7 + 1].y;
	sum.e += pixel * Filter_3[1*7 + 1].x;
	sum.f += pixel * Filter_3[1*7 + 1].y;
    pixel = image[y + 3][x + 2]; 
    sum.a += pixel * Filter_1[1].x;
	sum.b += pixel * Filter_1[1].y;
	sum.c += pixel * Filter_2[1].x;
	sum.d += pixel * Filter_2[1].y;
	sum.e += pixel * Filter_3[1].x;
	sum.f += pixel * Filter_3[1].y;

    pixel = image[y - 3][x + 3]; 
    sum.a += pixel * Filter_1[6*7].x;
	sum.b += pixel * Filter_1[6*7].y;
	sum.c += pixel * Filter_2[6*7].x;
	sum.d += pixel * Filter_2[6*7].y;
	sum.e += pixel * Filter_3[6*7].x;
	sum.f += pixel * Filter_3[6*7].y;
    pixel = image[y - 2][x + 3]; 
    sum.a += pixel * Filter_1[5*7].x;
	sum.b += pixel * Filter_1[5*7].y;
	sum.c += pixel * Filter_2[5*7].x;
	sum.d += pixel * Filter_2[5*7].y;
	sum.e += pixel * Filter_3[5*7].x;
	sum.f += pixel * Filter_3[5*7].y;
    pixel = image[y - 1][x + 3]; 
    sum.a += pixel * Filter_1[4*7].x;
	sum.b += pixel * Filter_1[4*7].y;
	sum.c += pixel * Filter_2[4*7].x;
	sum.d += pixel * Filter_2[4*7].y;
	sum.e += pixel * Filter_3[4*7].x;
	sum.f += pixel * Filter_3[4*7].y;
    pixel = image[y + 0][x + 3]; 
    sum.a += pixel * Filter_1[3*7].x;
	sum.b += pixel * Filter_1[3*7].y;
	sum.c += pixel * Filter_2[3*7].x;
	sum.d += pixel * Filter_2[3*7].y;
	sum.e += pixel * Filter_3[3*7].x;
	sum.f += pixel * Filter_3[3*7].y;
    pixel = image[y + 1][x + 3]; 
    sum.a += pixel * Filter_1[2*7].x;
	sum.b += pixel * Filter_1[2*7].y;
	sum.c += pixel * Filter_2[2*7].x;
	sum.d += pixel * Filter_2[2*7].y;
	sum.e += pixel * Filter_3[2*7].x;
	sum.f += pixel * Filter_3[2*7].y;
    pixel = image[y + 2][x + 3]; 
    sum.a += pixel * Filter_1[1*7].x;
	sum.b += pixel * Filter_1[1*7].y;
	sum.c += pixel * Filter_2[1*7].x;
	sum.d += pixel * Filter_2[1*7].y;
	sum.e += pixel * Filter_3[1*7].x;
	sum.f += pixel * Filter_3[1*7].y;
    pixel = image[y + 3][x + 3]; 
    sum.a += pixel * Filter_1[0].x;
	sum.b += pixel * Filter_1[0].y;
	sum.c += pixel * Filter_2[0].x;
	sum.d += pixel * Filter_2[0].y;
	sum.e += pixel * Filter_3[0].x;
	sum.f += pixel * Filter_3[0].y;

	return sum;
}

float6 Conv_2D_Unrolled_7x7_ThreeFilters_(__local float image[64][96], 
	                                      int y, 
										  int x, 
										  __constant float* Filter_1_Real, 
										  __constant float* Filter_1_Imag, 
										  __constant float* Filter_2_Real, 
										  __constant float* Filter_2_Imag, 
										  __constant float* Filter_3_Real, 
										  __constant float* Filter_3_Imag)
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
    sum.a += pixel * Filter_1_Real[6*7 + 6];
	sum.b += pixel * Filter_1_Imag[6*7 + 6];
	sum.c += pixel * Filter_2_Real[6*7 + 6];
	sum.d += pixel * Filter_2_Imag[6*7 + 6];
	sum.e += pixel * Filter_3_Real[6*7 + 6];
	sum.f += pixel * Filter_3_Imag[6*7 + 6];
    pixel = image[y - 2][x - 3]; 
    sum.a += pixel * Filter_1_Real[5*7 + 6];
	sum.b += pixel * Filter_1_Imag[5*7 + 6];
	sum.c += pixel * Filter_2_Real[5*7 + 6];
	sum.d += pixel * Filter_2_Imag[5*7 + 6];
	sum.e += pixel * Filter_3_Real[5*7 + 6];
	sum.f += pixel * Filter_3_Imag[5*7 + 6];
	pixel = image[y - 1][x - 3]; 
    sum.a += pixel * Filter_1_Real[4*7 + 6];
	sum.b += pixel * Filter_1_Imag[4*7 + 6];
	sum.c += pixel * Filter_2_Real[4*7 + 6];
	sum.d += pixel * Filter_2_Imag[4*7 + 6];
	sum.e += pixel * Filter_3_Real[4*7 + 6];
	sum.f += pixel * Filter_3_Imag[4*7 + 6];
	pixel = image[y + 0][x - 3]; 
    sum.a += pixel * Filter_1_Real[3*7 + 6];
	sum.b += pixel * Filter_1_Imag[3*7 + 6];
	sum.c += pixel * Filter_2_Real[3*7 + 6];
	sum.d += pixel * Filter_2_Imag[3*7 + 6];
	sum.e += pixel * Filter_3_Real[3*7 + 6];
	sum.f += pixel * Filter_3_Imag[3*7 + 6];
    pixel = image[y + 1][x - 3]; 
    sum.a += pixel * Filter_1_Real[2*7 + 6];
	sum.b += pixel * Filter_1_Imag[2*7 + 6];
	sum.c += pixel * Filter_2_Real[2*7 + 6];
	sum.d += pixel * Filter_2_Imag[2*7 + 6];
	sum.e += pixel * Filter_3_Real[2*7 + 6];
	sum.f += pixel * Filter_3_Imag[2*7 + 6];
	pixel = image[y + 2][x - 3]; 
    sum.a += pixel * Filter_1_Real[1*7 + 6];
	sum.b += pixel * Filter_1_Imag[1*7 + 6];
	sum.c += pixel * Filter_2_Real[1*7 + 6];
	sum.d += pixel * Filter_2_Imag[1*7 + 6];
	sum.e += pixel * Filter_3_Real[1*7 + 6];
	sum.f += pixel * Filter_3_Imag[1*7 + 6];
	pixel = image[y + 3][x - 3]; 
    sum.a += pixel * Filter_1_Real[6];
	sum.b += pixel * Filter_1_Imag[6];
	sum.c += pixel * Filter_2_Real[6];
	sum.d += pixel * Filter_2_Imag[6];
	sum.e += pixel * Filter_3_Real[6];
	sum.f += pixel * Filter_3_Imag[6];

    pixel = image[y - 3][x - 2]; 
    sum.a += pixel * Filter_1_Real[6*7 + 5];
	sum.b += pixel * Filter_1_Imag[6*7 + 5];
	sum.c += pixel * Filter_2_Real[6*7 + 5];
	sum.d += pixel * Filter_2_Imag[6*7 + 5];
	sum.e += pixel * Filter_3_Real[6*7 + 5];
	sum.f += pixel * Filter_3_Imag[6*7 + 5];
    pixel = image[y - 2][x - 2]; 
    sum.a += pixel * Filter_1_Real[5*7 + 5];
	sum.b += pixel * Filter_1_Imag[5*7 + 5];
	sum.c += pixel * Filter_2_Real[5*7 + 5];
	sum.d += pixel * Filter_2_Imag[5*7 + 5];
	sum.e += pixel * Filter_3_Real[5*7 + 5];
	sum.f += pixel * Filter_3_Imag[5*7 + 5];
    pixel = image[y - 1][x - 2]; 
    sum.a += pixel * Filter_1_Real[4*7 + 5];
	sum.b += pixel * Filter_1_Imag[4*7 + 5];
	sum.c += pixel * Filter_2_Real[4*7 + 5];
	sum.d += pixel * Filter_2_Imag[4*7 + 5];
	sum.e += pixel * Filter_3_Real[4*7 + 5];
	sum.f += pixel * Filter_3_Imag[4*7 + 5];
    pixel = image[y + 0][x - 2]; 
    sum.a += pixel * Filter_1_Real[3*7 + 5];
	sum.b += pixel * Filter_1_Imag[3*7 + 5];
	sum.c += pixel * Filter_2_Real[3*7 + 5];
	sum.d += pixel * Filter_2_Imag[3*7 + 5];
	sum.e += pixel * Filter_3_Real[3*7 + 5];
	sum.f += pixel * Filter_3_Imag[3*7 + 5];
    pixel = image[y + 1][x - 2]; 
    sum.a += pixel * Filter_1_Real[2*7 + 5];
	sum.b += pixel * Filter_1_Imag[2*7 + 5];
	sum.c += pixel * Filter_2_Real[2*7 + 5];
	sum.d += pixel * Filter_2_Imag[2*7 + 5];
	sum.e += pixel * Filter_3_Real[2*7 + 5];
	sum.f += pixel * Filter_3_Imag[2*7 + 5];
    pixel = image[y + 2][x - 2]; 
    sum.a += pixel * Filter_1_Real[1*7 + 5];
	sum.b += pixel * Filter_1_Imag[1*7 + 5];
	sum.c += pixel * Filter_2_Real[1*7 + 5];
	sum.d += pixel * Filter_2_Imag[1*7 + 5];
	sum.e += pixel * Filter_3_Real[1*7 + 5];
	sum.f += pixel * Filter_3_Imag[1*7 + 5];
    pixel = image[y + 3][x - 2]; 
    sum.a += pixel * Filter_1_Real[5];
	sum.b += pixel * Filter_1_Imag[5];
	sum.c += pixel * Filter_2_Real[5];
	sum.d += pixel * Filter_2_Imag[5];
	sum.e += pixel * Filter_3_Real[5];
	sum.f += pixel * Filter_3_Imag[5];


    pixel = image[y - 3][x - 1]; 
    sum.a += pixel * Filter_1_Real[6*7 + 4];
	sum.b += pixel * Filter_1_Imag[6*7 + 4];
	sum.c += pixel * Filter_2_Real[6*7 + 4];
	sum.d += pixel * Filter_2_Imag[6*7 + 4];
	sum.e += pixel * Filter_3_Real[6*7 + 4];
	sum.f += pixel * Filter_3_Imag[6*7 + 4];
    pixel = image[y - 2][x - 1]; 
    sum.a += pixel * Filter_1_Real[5*7 + 4];
	sum.b += pixel * Filter_1_Imag[5*7 + 4];
	sum.c += pixel * Filter_2_Real[5*7 + 4];
	sum.d += pixel * Filter_2_Imag[5*7 + 4];
	sum.e += pixel * Filter_3_Real[5*7 + 4];
	sum.f += pixel * Filter_3_Imag[5*7 + 4];
    pixel = image[y - 1][x - 1]; 
    sum.a += pixel * Filter_1_Real[4*7 + 4];
	sum.b += pixel * Filter_1_Imag[4*7 + 4];
	sum.c += pixel * Filter_2_Real[4*7 + 4];
	sum.d += pixel * Filter_2_Imag[4*7 + 4];
	sum.e += pixel * Filter_3_Real[4*7 + 4];
	sum.f += pixel * Filter_3_Imag[4*7 + 4];
    pixel = image[y + 0][x - 1]; 
    sum.a += pixel * Filter_1_Real[3*7 + 4];
	sum.b += pixel * Filter_1_Imag[3*7 + 4];
	sum.c += pixel * Filter_2_Real[3*7 + 4];
	sum.d += pixel * Filter_2_Imag[3*7 + 4];
	sum.e += pixel * Filter_3_Real[3*7 + 4];
	sum.f += pixel * Filter_3_Imag[3*7 + 4];
    pixel = image[y + 1][x - 1]; 
    sum.a += pixel * Filter_1_Real[2*7 + 4];
	sum.b += pixel * Filter_1_Imag[2*7 + 4];
	sum.c += pixel * Filter_2_Real[2*7 + 4];
	sum.d += pixel * Filter_2_Imag[2*7 + 4];
	sum.e += pixel * Filter_3_Real[2*7 + 4];
	sum.f += pixel * Filter_3_Imag[2*7 + 4];
    pixel = image[y + 2][x - 1]; 
    sum.a += pixel * Filter_1_Real[1*7 + 4];
	sum.b += pixel * Filter_1_Imag[1*7 + 4];
	sum.c += pixel * Filter_2_Real[1*7 + 4];
	sum.d += pixel * Filter_2_Imag[1*7 + 4];
	sum.e += pixel * Filter_3_Real[1*7 + 4];
	sum.f += pixel * Filter_3_Imag[1*7 + 4];
    pixel = image[y + 3][x - 1]; 
    sum.a += pixel * Filter_1_Real[4];
	sum.b += pixel * Filter_1_Imag[4];
	sum.c += pixel * Filter_2_Real[4];
	sum.d += pixel * Filter_2_Imag[4];
	sum.e += pixel * Filter_3_Real[4];
	sum.f += pixel * Filter_3_Imag[4];


    pixel = image[y - 3][x + 0]; 
    sum.a += pixel * Filter_1_Real[6*7 + 3];
	sum.b += pixel * Filter_1_Imag[6*7 + 3];
	sum.c += pixel * Filter_2_Real[6*7 + 3];
	sum.d += pixel * Filter_2_Imag[6*7 + 3];
	sum.e += pixel * Filter_3_Real[6*7 + 3];
	sum.f += pixel * Filter_3_Imag[6*7 + 3];
    pixel = image[y - 2][x + 0]; 
    sum.a += pixel * Filter_1_Real[5*7 + 3];
	sum.b += pixel * Filter_1_Imag[5*7 + 3];
	sum.c += pixel * Filter_2_Real[5*7 + 3];
	sum.d += pixel * Filter_2_Imag[5*7 + 3];
	sum.e += pixel * Filter_3_Real[5*7 + 3];
	sum.f += pixel * Filter_3_Imag[5*7 + 3];
    pixel = image[y - 1][x + 0]; 
    sum.a += pixel * Filter_1_Real[4*7 + 3];
	sum.b += pixel * Filter_1_Imag[4*7 + 3];
	sum.c += pixel * Filter_2_Real[4*7 + 3];
	sum.d += pixel * Filter_2_Imag[4*7 + 3];
	sum.e += pixel * Filter_3_Real[4*7 + 3];
	sum.f += pixel * Filter_3_Imag[4*7 + 3];
    pixel = image[y + 0][x + 0]; 
    sum.a += pixel * Filter_1_Real[3*7 + 3];
	sum.b += pixel * Filter_1_Imag[3*7 + 3];
	sum.c += pixel * Filter_2_Real[3*7 + 3];
	sum.d += pixel * Filter_2_Imag[3*7 + 3];
	sum.e += pixel * Filter_3_Real[3*7 + 3];
	sum.f += pixel * Filter_3_Imag[3*7 + 3];
    pixel = image[y + 1][x + 0]; 
    sum.a += pixel * Filter_1_Real[2*7 + 3];
	sum.b += pixel * Filter_1_Imag[2*7 + 3];
	sum.c += pixel * Filter_2_Real[2*7 + 3];
	sum.d += pixel * Filter_2_Imag[2*7 + 3];
	sum.e += pixel * Filter_3_Real[2*7 + 3];
	sum.f += pixel * Filter_3_Imag[2*7 + 3];
    pixel = image[y + 2][x + 0]; 
    sum.a += pixel * Filter_1_Real[1*7 + 3];
	sum.b += pixel * Filter_1_Imag[1*7 + 3];
	sum.c += pixel * Filter_2_Real[1*7 + 3];
	sum.d += pixel * Filter_2_Imag[1*7 + 3];
	sum.e += pixel * Filter_3_Real[1*7 + 3];
	sum.f += pixel * Filter_3_Imag[1*7 + 3];
    pixel = image[y + 3][x + 0]; 
    sum.a += pixel * Filter_1_Real[3];
	sum.b += pixel * Filter_1_Imag[3];
	sum.c += pixel * Filter_2_Real[3];
	sum.d += pixel * Filter_2_Imag[3];
	sum.e += pixel * Filter_3_Real[3];
	sum.f += pixel * Filter_3_Imag[3];

	pixel = image[y - 3][x + 1]; 
    sum.a += pixel * Filter_1_Real[6*7 + 2];
	sum.b += pixel * Filter_1_Imag[6*7 + 2];
	sum.c += pixel * Filter_2_Real[6*7 + 2];
	sum.d += pixel * Filter_2_Imag[6*7 + 2];
	sum.e += pixel * Filter_3_Real[6*7 + 2];
	sum.f += pixel * Filter_3_Imag[6*7 + 2];
    pixel = image[y - 2][x + 1]; 
    sum.a += pixel * Filter_1_Real[5*7 + 2];
	sum.b += pixel * Filter_1_Imag[5*7 + 2];
	sum.c += pixel * Filter_2_Real[5*7 + 2];
	sum.d += pixel * Filter_2_Imag[5*7 + 2];
	sum.e += pixel * Filter_3_Real[5*7 + 2];
	sum.f += pixel * Filter_3_Imag[5*7 + 2];
    pixel = image[y - 1][x + 1]; 
    sum.a += pixel * Filter_1_Real[4*7 + 2];
	sum.b += pixel * Filter_1_Imag[4*7 + 2];
	sum.c += pixel * Filter_2_Real[4*7 + 2];
	sum.d += pixel * Filter_2_Imag[4*7 + 2];
	sum.e += pixel * Filter_3_Real[4*7 + 2];
	sum.f += pixel * Filter_3_Imag[4*7 + 2];
    pixel = image[y + 0][x + 1]; 
    sum.a += pixel * Filter_1_Real[3*7 + 2];
	sum.b += pixel * Filter_1_Imag[3*7 + 2];
	sum.c += pixel * Filter_2_Real[3*7 + 2];
	sum.d += pixel * Filter_2_Imag[3*7 + 2];
	sum.e += pixel * Filter_3_Real[3*7 + 2];
	sum.f += pixel * Filter_3_Imag[3*7 + 2];
    pixel = image[y + 1][x + 1]; 
    sum.a += pixel * Filter_1_Real[2*7 + 2];
	sum.b += pixel * Filter_1_Imag[2*7 + 2];
	sum.c += pixel * Filter_2_Real[2*7 + 2];
	sum.d += pixel * Filter_2_Imag[2*7 + 2];
	sum.e += pixel * Filter_3_Real[2*7 + 2];
	sum.f += pixel * Filter_3_Imag[2*7 + 2];
    pixel = image[y + 2][x + 1]; 
    sum.a += pixel * Filter_1_Real[1*7 + 2];
	sum.b += pixel * Filter_1_Imag[1*7 + 2];
	sum.c += pixel * Filter_2_Real[1*7 + 2];
	sum.d += pixel * Filter_2_Imag[1*7 + 2];
	sum.e += pixel * Filter_3_Real[1*7 + 2];
	sum.f += pixel * Filter_3_Imag[1*7 + 2];
    pixel = image[y + 3][x + 1]; 
    sum.a += pixel * Filter_1_Real[2];
	sum.b += pixel * Filter_1_Imag[2];
	sum.c += pixel * Filter_2_Real[2];
	sum.d += pixel * Filter_2_Imag[2];
	sum.e += pixel * Filter_3_Real[2];
	sum.f += pixel * Filter_3_Imag[2];
 
    pixel = image[y - 3][x + 2]; 
    sum.a += pixel * Filter_1_Real[6*7 + 1];
	sum.b += pixel * Filter_1_Imag[6*7 + 1];
	sum.c += pixel * Filter_2_Real[6*7 + 1];
	sum.d += pixel * Filter_2_Imag[6*7 + 1];
	sum.e += pixel * Filter_3_Real[6*7 + 1];
	sum.f += pixel * Filter_3_Imag[6*7 + 1];
    pixel = image[y - 2][x + 2]; 
    sum.a += pixel * Filter_1_Real[5*7 + 1];
	sum.b += pixel * Filter_1_Imag[5*7 + 1];
	sum.c += pixel * Filter_2_Real[5*7 + 1];
	sum.d += pixel * Filter_2_Imag[5*7 + 1];
	sum.e += pixel * Filter_3_Real[5*7 + 1];
	sum.f += pixel * Filter_3_Imag[5*7 + 1];
    pixel = image[y - 1][x + 2]; 
    sum.a += pixel * Filter_1_Real[4*7 + 1];
	sum.b += pixel * Filter_1_Imag[4*7 + 1];
	sum.c += pixel * Filter_2_Real[4*7 + 1];
	sum.d += pixel * Filter_2_Imag[4*7 + 1];
	sum.e += pixel * Filter_3_Real[4*7 + 1];
	sum.f += pixel * Filter_3_Imag[4*7 + 1];
    pixel = image[y + 0][x + 2]; 
    sum.a += pixel * Filter_1_Real[3*7 + 1];
	sum.b += pixel * Filter_1_Imag[3*7 + 1];
	sum.c += pixel * Filter_2_Real[3*7 + 1];
	sum.d += pixel * Filter_2_Imag[3*7 + 1];
	sum.e += pixel * Filter_3_Real[3*7 + 1];
	sum.f += pixel * Filter_3_Imag[3*7 + 1];
	pixel = image[y + 1][x + 2]; 
    sum.a += pixel * Filter_1_Real[2*7 + 1];
	sum.b += pixel * Filter_1_Imag[2*7 + 1];
	sum.c += pixel * Filter_2_Real[2*7 + 1];
	sum.d += pixel * Filter_2_Imag[2*7 + 1];
	sum.e += pixel * Filter_3_Real[2*7 + 1];
	sum.f += pixel * Filter_3_Imag[2*7 + 1];
    pixel = image[y + 2][x + 2]; 
    sum.a += pixel * Filter_1_Real[1*7 + 1];
	sum.b += pixel * Filter_1_Imag[1*7 + 1];
	sum.c += pixel * Filter_2_Real[1*7 + 1];
	sum.d += pixel * Filter_2_Imag[1*7 + 1];
	sum.e += pixel * Filter_3_Real[1*7 + 1];
	sum.f += pixel * Filter_3_Imag[1*7 + 1];
    pixel = image[y + 3][x + 2]; 
    sum.a += pixel * Filter_1_Real[1];
	sum.b += pixel * Filter_1_Imag[1];
	sum.c += pixel * Filter_2_Real[1];
	sum.d += pixel * Filter_2_Imag[1];
	sum.e += pixel * Filter_3_Real[1];
	sum.f += pixel * Filter_3_Imag[1];

    pixel = image[y - 3][x + 3]; 
    sum.a += pixel * Filter_1_Real[6*7];
	sum.b += pixel * Filter_1_Imag[6*7];
	sum.c += pixel * Filter_2_Real[6*7];
	sum.d += pixel * Filter_2_Imag[6*7];
	sum.e += pixel * Filter_3_Real[6*7];
	sum.f += pixel * Filter_3_Imag[6*7];
    pixel = image[y - 2][x + 3]; 
    sum.a += pixel * Filter_1_Real[5*7];
	sum.b += pixel * Filter_1_Imag[5*7];
	sum.c += pixel * Filter_2_Real[5*7];
	sum.d += pixel * Filter_2_Imag[5*7];
	sum.e += pixel * Filter_3_Real[5*7];
	sum.f += pixel * Filter_3_Imag[5*7];
    pixel = image[y - 1][x + 3]; 
    sum.a += pixel * Filter_1_Real[4*7];
	sum.b += pixel * Filter_1_Imag[4*7];
	sum.c += pixel * Filter_2_Real[4*7];
	sum.d += pixel * Filter_2_Imag[4*7];
	sum.e += pixel * Filter_3_Real[4*7];
	sum.f += pixel * Filter_3_Imag[4*7];
    pixel = image[y + 0][x + 3]; 
    sum.a += pixel * Filter_1_Real[3*7];
	sum.b += pixel * Filter_1_Imag[3*7];
	sum.c += pixel * Filter_2_Real[3*7];
	sum.d += pixel * Filter_2_Imag[3*7];
	sum.e += pixel * Filter_3_Real[3*7];
	sum.f += pixel * Filter_3_Imag[3*7];
    pixel = image[y + 1][x + 3]; 
    sum.a += pixel * Filter_1_Real[2*7];
	sum.b += pixel * Filter_1_Imag[2*7];
	sum.c += pixel * Filter_2_Real[2*7];
	sum.d += pixel * Filter_2_Imag[2*7];
	sum.e += pixel * Filter_3_Real[2*7];
	sum.f += pixel * Filter_3_Imag[2*7];
    pixel = image[y + 2][x + 3]; 
    sum.a += pixel * Filter_1_Real[1*7];
	sum.b += pixel * Filter_1_Imag[1*7];
	sum.c += pixel * Filter_2_Real[1*7];
	sum.d += pixel * Filter_2_Imag[1*7];
	sum.e += pixel * Filter_3_Real[1*7];
	sum.f += pixel * Filter_3_Imag[1*7];
    pixel = image[y + 3][x + 3]; 
    sum.a += pixel * Filter_1_Real[0];
	sum.b += pixel * Filter_1_Imag[0];
	sum.c += pixel * Filter_2_Real[0];
	sum.d += pixel * Filter_2_Imag[0];
	sum.e += pixel * Filter_3_Real[0];
	sum.f += pixel * Filter_3_Imag[0];

	return sum;
}



float6 Conv_2D_Unrolled_7x7_ThreeFilters_AMD(__local float image[64][128],
	                                         int y,
	                                         int x,
	                                         __constant float2* Filter_1,
	                                         __constant float2* Filter_2,
	                                         __constant float2* Filter_3)
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
    sum.a += pixel * Filter_1[6*7 + 6].x;
	sum.b += pixel * Filter_1[6*7 + 6].y;
	sum.c += pixel * Filter_2[6*7 + 6].x;
	sum.d += pixel * Filter_2[6*7 + 6].y;
	sum.e += pixel * Filter_3[6*7 + 6].x;
	sum.f += pixel * Filter_3[6*7 + 6].y;
    pixel = image[y - 2][x - 3]; 
    sum.a += pixel * Filter_1[5*7 + 6].x;
	sum.b += pixel * Filter_1[5*7 + 6].y;
	sum.c += pixel * Filter_2[5*7 + 6].x;
	sum.d += pixel * Filter_2[5*7 + 6].y;
	sum.e += pixel * Filter_3[5*7 + 6].x;
	sum.f += pixel * Filter_3[5*7 + 6].y;
	pixel = image[y - 1][x - 3]; 
    sum.a += pixel * Filter_1[4*7 + 6].x;
	sum.b += pixel * Filter_1[4*7 + 6].y;
	sum.c += pixel * Filter_2[4*7 + 6].x;
	sum.d += pixel * Filter_2[4*7 + 6].y;
	sum.e += pixel * Filter_3[4*7 + 6].x;
	sum.f += pixel * Filter_3[4*7 + 6].y;
	pixel = image[y + 0][x - 3]; 
    sum.a += pixel * Filter_1[3*7 + 6].x;
	sum.b += pixel * Filter_1[3*7 + 6].y;
	sum.c += pixel * Filter_2[3*7 + 6].x;
	sum.d += pixel * Filter_2[3*7 + 6].y;
	sum.e += pixel * Filter_3[3*7 + 6].x;
	sum.f += pixel * Filter_3[3*7 + 6].y;
    pixel = image[y + 1][x - 3]; 
    sum.a += pixel * Filter_1[2*7 + 6].x;
	sum.b += pixel * Filter_1[2*7 + 6].y;
	sum.c += pixel * Filter_2[2*7 + 6].x;
	sum.d += pixel * Filter_2[2*7 + 6].y;
	sum.e += pixel * Filter_3[2*7 + 6].x;
	sum.f += pixel * Filter_3[2*7 + 6].y;
	pixel = image[y + 2][x - 3]; 
    sum.a += pixel * Filter_1[1*7 + 6].x;
	sum.b += pixel * Filter_1[1*7 + 6].y;
	sum.c += pixel * Filter_2[1*7 + 6].x;
	sum.d += pixel * Filter_2[1*7 + 6].y;
	sum.e += pixel * Filter_3[1*7 + 6].x;
	sum.f += pixel * Filter_3[1*7 + 6].y;
	pixel = image[y + 3][x - 3]; 
    sum.a += pixel * Filter_1[6].x;
	sum.b += pixel * Filter_1[6].y;
	sum.c += pixel * Filter_2[6].x;
	sum.d += pixel * Filter_2[6].y;
	sum.e += pixel * Filter_3[6].x;
	sum.f += pixel * Filter_3[6].y;

    pixel = image[y - 3][x - 2]; 
    sum.a += pixel * Filter_1[6*7 + 5].x;
	sum.b += pixel * Filter_1[6*7 + 5].y;
	sum.c += pixel * Filter_2[6*7 + 5].x;
	sum.d += pixel * Filter_2[6*7 + 5].y;
	sum.e += pixel * Filter_3[6*7 + 5].x;
	sum.f += pixel * Filter_3[6*7 + 5].y;
    pixel = image[y - 2][x - 2]; 
    sum.a += pixel * Filter_1[5*7 + 5].x;
	sum.b += pixel * Filter_1[5*7 + 5].y;
	sum.c += pixel * Filter_2[5*7 + 5].x;
	sum.d += pixel * Filter_2[5*7 + 5].y;
	sum.e += pixel * Filter_3[5*7 + 5].x;
	sum.f += pixel * Filter_3[5*7 + 5].y;
    pixel = image[y - 1][x - 2]; 
    sum.a += pixel * Filter_1[4*7 + 5].x;
	sum.b += pixel * Filter_1[4*7 + 5].y;
	sum.c += pixel * Filter_2[4*7 + 5].x;
	sum.d += pixel * Filter_2[4*7 + 5].y;
	sum.e += pixel * Filter_3[4*7 + 5].x;
	sum.f += pixel * Filter_3[4*7 + 5].y;
    pixel = image[y + 0][x - 2]; 
    sum.a += pixel * Filter_1[3*7 + 5].x;
	sum.b += pixel * Filter_1[3*7 + 5].y;
	sum.c += pixel * Filter_2[3*7 + 5].x;
	sum.d += pixel * Filter_2[3*7 + 5].y;
	sum.e += pixel * Filter_3[3*7 + 5].x;
	sum.f += pixel * Filter_3[3*7 + 5].y;
    pixel = image[y + 1][x - 2]; 
    sum.a += pixel * Filter_1[2*7 + 5].x;
	sum.b += pixel * Filter_1[2*7 + 5].y;
	sum.c += pixel * Filter_2[2*7 + 5].x;
	sum.d += pixel * Filter_2[2*7 + 5].y;
	sum.e += pixel * Filter_3[2*7 + 5].x;
	sum.f += pixel * Filter_3[2*7 + 5].y;
    pixel = image[y + 2][x - 2]; 
    sum.a += pixel * Filter_1[1*7 + 5].x;
	sum.b += pixel * Filter_1[1*7 + 5].y;
	sum.c += pixel * Filter_2[1*7 + 5].x;
	sum.d += pixel * Filter_2[1*7 + 5].y;
	sum.e += pixel * Filter_3[1*7 + 5].x;
	sum.f += pixel * Filter_3[1*7 + 5].y;
    pixel = image[y + 3][x - 2]; 
    sum.a += pixel * Filter_1[5].x;
	sum.b += pixel * Filter_1[5].y;
	sum.c += pixel * Filter_2[5].x;
	sum.d += pixel * Filter_2[5].y;
	sum.e += pixel * Filter_3[5].x;
	sum.f += pixel * Filter_3[5].y;


    pixel = image[y - 3][x - 1]; 
    sum.a += pixel * Filter_1[6*7 + 4].x;
	sum.b += pixel * Filter_1[6*7 + 4].y;
	sum.c += pixel * Filter_2[6*7 + 4].x;
	sum.d += pixel * Filter_2[6*7 + 4].y;
	sum.e += pixel * Filter_3[6*7 + 4].x;
	sum.f += pixel * Filter_3[6*7 + 4].y;
    pixel = image[y - 2][x - 1]; 
    sum.a += pixel * Filter_1[5*7 + 4].x;
	sum.b += pixel * Filter_1[5*7 + 4].y;
	sum.c += pixel * Filter_2[5*7 + 4].x;
	sum.d += pixel * Filter_2[5*7 + 4].y;
	sum.e += pixel * Filter_3[5*7 + 4].x;
	sum.f += pixel * Filter_3[5*7 + 4].y;
    pixel = image[y - 1][x - 1]; 
    sum.a += pixel * Filter_1[4*7 + 4].x;
	sum.b += pixel * Filter_1[4*7 + 4].y;
	sum.c += pixel * Filter_2[4*7 + 4].x;
	sum.d += pixel * Filter_2[4*7 + 4].y;
	sum.e += pixel * Filter_3[4*7 + 4].x;
	sum.f += pixel * Filter_3[4*7 + 4].y;
    pixel = image[y + 0][x - 1]; 
    sum.a += pixel * Filter_1[3*7 + 4].x;
	sum.b += pixel * Filter_1[3*7 + 4].y;
	sum.c += pixel * Filter_2[3*7 + 4].x;
	sum.d += pixel * Filter_2[3*7 + 4].y;
	sum.e += pixel * Filter_3[3*7 + 4].x;
	sum.f += pixel * Filter_3[3*7 + 4].y;
    pixel = image[y + 1][x - 1]; 
    sum.a += pixel * Filter_1[2*7 + 4].x;
	sum.b += pixel * Filter_1[2*7 + 4].y;
	sum.c += pixel * Filter_2[2*7 + 4].x;
	sum.d += pixel * Filter_2[2*7 + 4].y;
	sum.e += pixel * Filter_3[2*7 + 4].x;
	sum.f += pixel * Filter_3[2*7 + 4].y;
    pixel = image[y + 2][x - 1]; 
    sum.a += pixel * Filter_1[1*7 + 4].x;
	sum.b += pixel * Filter_1[1*7 + 4].y;
	sum.c += pixel * Filter_2[1*7 + 4].x;
	sum.d += pixel * Filter_2[1*7 + 4].y;
	sum.e += pixel * Filter_3[1*7 + 4].x;
	sum.f += pixel * Filter_3[1*7 + 4].y;
    pixel = image[y + 3][x - 1]; 
    sum.a += pixel * Filter_1[4].x;
	sum.b += pixel * Filter_1[4].y;
	sum.c += pixel * Filter_2[4].x;
	sum.d += pixel * Filter_2[4].y;
	sum.e += pixel * Filter_3[4].x;
	sum.f += pixel * Filter_3[4].y;


    pixel = image[y - 3][x + 0]; 
    sum.a += pixel * Filter_1[6*7 + 3].x;
	sum.b += pixel * Filter_1[6*7 + 3].y;
	sum.c += pixel * Filter_2[6*7 + 3].x;
	sum.d += pixel * Filter_2[6*7 + 3].y;
	sum.e += pixel * Filter_3[6*7 + 3].x;
	sum.f += pixel * Filter_3[6*7 + 3].y;
    pixel = image[y - 2][x + 0]; 
    sum.a += pixel * Filter_1[5*7 + 3].x;
	sum.b += pixel * Filter_1[5*7 + 3].y;
	sum.c += pixel * Filter_2[5*7 + 3].x;
	sum.d += pixel * Filter_2[5*7 + 3].y;
	sum.e += pixel * Filter_3[5*7 + 3].x;
	sum.f += pixel * Filter_3[5*7 + 3].y;
    pixel = image[y - 1][x + 0]; 
    sum.a += pixel * Filter_1[4*7 + 3].x;
	sum.b += pixel * Filter_1[4*7 + 3].y;
	sum.c += pixel * Filter_2[4*7 + 3].x;
	sum.d += pixel * Filter_2[4*7 + 3].y;
	sum.e += pixel * Filter_3[4*7 + 3].x;
	sum.f += pixel * Filter_3[4*7 + 3].y;
    pixel = image[y + 0][x + 0]; 
    sum.a += pixel * Filter_1[3*7 + 3].x;
	sum.b += pixel * Filter_1[3*7 + 3].y;
	sum.c += pixel * Filter_2[3*7 + 3].x;
	sum.d += pixel * Filter_2[3*7 + 3].y;
	sum.e += pixel * Filter_3[3*7 + 3].x;
	sum.f += pixel * Filter_3[3*7 + 3].y;
    pixel = image[y + 1][x + 0]; 
    sum.a += pixel * Filter_1[2*7 + 3].x;
	sum.b += pixel * Filter_1[2*7 + 3].y;
	sum.c += pixel * Filter_2[2*7 + 3].x;
	sum.d += pixel * Filter_2[2*7 + 3].y;
	sum.e += pixel * Filter_3[2*7 + 3].x;
	sum.f += pixel * Filter_3[2*7 + 3].y;
    pixel = image[y + 2][x + 0]; 
    sum.a += pixel * Filter_1[1*7 + 3].x;
	sum.b += pixel * Filter_1[1*7 + 3].y;
	sum.c += pixel * Filter_2[1*7 + 3].x;
	sum.d += pixel * Filter_2[1*7 + 3].y;
	sum.e += pixel * Filter_3[1*7 + 3].x;
	sum.f += pixel * Filter_3[1*7 + 3].y;
    pixel = image[y + 3][x + 0]; 
    sum.a += pixel * Filter_1[3].x;
	sum.b += pixel * Filter_1[3].y;
	sum.c += pixel * Filter_2[3].x;
	sum.d += pixel * Filter_2[3].y;
	sum.e += pixel * Filter_3[3].x;
	sum.f += pixel * Filter_3[3].y;

	pixel = image[y - 3][x + 1]; 
    sum.a += pixel * Filter_1[6*7 + 2].x;
	sum.b += pixel * Filter_1[6*7 + 2].y;
	sum.c += pixel * Filter_2[6*7 + 2].x;
	sum.d += pixel * Filter_2[6*7 + 2].y;
	sum.e += pixel * Filter_3[6*7 + 2].x;
	sum.f += pixel * Filter_3[6*7 + 2].y;
    pixel = image[y - 2][x + 1]; 
    sum.a += pixel * Filter_1[5*7 + 2].x;
	sum.b += pixel * Filter_1[5*7 + 2].y;
	sum.c += pixel * Filter_2[5*7 + 2].x;
	sum.d += pixel * Filter_2[5*7 + 2].y;
	sum.e += pixel * Filter_3[5*7 + 2].x;
	sum.f += pixel * Filter_3[5*7 + 2].y;
    pixel = image[y - 1][x + 1]; 
    sum.a += pixel * Filter_1[4*7 + 2].x;
	sum.b += pixel * Filter_1[4*7 + 2].y;
	sum.c += pixel * Filter_2[4*7 + 2].x;
	sum.d += pixel * Filter_2[4*7 + 2].y;
	sum.e += pixel * Filter_3[4*7 + 2].x;
	sum.f += pixel * Filter_3[4*7 + 2].y;
    pixel = image[y + 0][x + 1]; 
    sum.a += pixel * Filter_1[3*7 + 2].x;
	sum.b += pixel * Filter_1[3*7 + 2].y;
	sum.c += pixel * Filter_2[3*7 + 2].x;
	sum.d += pixel * Filter_2[3*7 + 2].y;
	sum.e += pixel * Filter_3[3*7 + 2].x;
	sum.f += pixel * Filter_3[3*7 + 2].y;
    pixel = image[y + 1][x + 1]; 
    sum.a += pixel * Filter_1[2*7 + 2].x;
	sum.b += pixel * Filter_1[2*7 + 2].y;
	sum.c += pixel * Filter_2[2*7 + 2].x;
	sum.d += pixel * Filter_2[2*7 + 2].y;
	sum.e += pixel * Filter_3[2*7 + 2].x;
	sum.f += pixel * Filter_3[2*7 + 2].y;
    pixel = image[y + 2][x + 1]; 
    sum.a += pixel * Filter_1[1*7 + 2].x;
	sum.b += pixel * Filter_1[1*7 + 2].y;
	sum.c += pixel * Filter_2[1*7 + 2].x;
	sum.d += pixel * Filter_2[1*7 + 2].y;
	sum.e += pixel * Filter_3[1*7 + 2].x;
	sum.f += pixel * Filter_3[1*7 + 2].y;
    pixel = image[y + 3][x + 1]; 
    sum.a += pixel * Filter_1[2].x;
	sum.b += pixel * Filter_1[2].y;
	sum.c += pixel * Filter_2[2].x;
	sum.d += pixel * Filter_2[2].y;
	sum.e += pixel * Filter_3[2].x;
	sum.f += pixel * Filter_3[2].y;
 
    pixel = image[y - 3][x + 2]; 
    sum.a += pixel * Filter_1[6*7 + 1].x;
	sum.b += pixel * Filter_1[6*7 + 1].y;
	sum.c += pixel * Filter_2[6*7 + 1].x;
	sum.d += pixel * Filter_2[6*7 + 1].y;
	sum.e += pixel * Filter_3[6*7 + 1].x;
	sum.f += pixel * Filter_3[6*7 + 1].y;
    pixel = image[y - 2][x + 2]; 
    sum.a += pixel * Filter_1[5*7 + 1].x;
	sum.b += pixel * Filter_1[5*7 + 1].y;
	sum.c += pixel * Filter_2[5*7 + 1].x;
	sum.d += pixel * Filter_2[5*7 + 1].y;
	sum.e += pixel * Filter_3[5*7 + 1].x;
	sum.f += pixel * Filter_3[5*7 + 1].y;
    pixel = image[y - 1][x + 2]; 
    sum.a += pixel * Filter_1[4*7 + 1].x;
	sum.b += pixel * Filter_1[4*7 + 1].y;
	sum.c += pixel * Filter_2[4*7 + 1].x;
	sum.d += pixel * Filter_2[4*7 + 1].y;
	sum.e += pixel * Filter_3[4*7 + 1].x;
	sum.f += pixel * Filter_3[4*7 + 1].y;
    pixel = image[y + 0][x + 2]; 
    sum.a += pixel * Filter_1[3*7 + 1].x;
	sum.b += pixel * Filter_1[3*7 + 1].y;
	sum.c += pixel * Filter_2[3*7 + 1].x;
	sum.d += pixel * Filter_2[3*7 + 1].y;
	sum.e += pixel * Filter_3[3*7 + 1].x;
	sum.f += pixel * Filter_3[3*7 + 1].y;
	pixel = image[y + 1][x + 2]; 
    sum.a += pixel * Filter_1[2*7 + 1].x;
	sum.b += pixel * Filter_1[2*7 + 1].y;
	sum.c += pixel * Filter_2[2*7 + 1].x;
	sum.d += pixel * Filter_2[2*7 + 1].y;
	sum.e += pixel * Filter_3[2*7 + 1].x;
	sum.f += pixel * Filter_3[2*7 + 1].y;
    pixel = image[y + 2][x + 2]; 
    sum.a += pixel * Filter_1[1*7 + 1].x;
	sum.b += pixel * Filter_1[1*7 + 1].y;
	sum.c += pixel * Filter_2[1*7 + 1].x;
	sum.d += pixel * Filter_2[1*7 + 1].y;
	sum.e += pixel * Filter_3[1*7 + 1].x;
	sum.f += pixel * Filter_3[1*7 + 1].y;
    pixel = image[y + 3][x + 2]; 
    sum.a += pixel * Filter_1[1].x;
	sum.b += pixel * Filter_1[1].y;
	sum.c += pixel * Filter_2[1].x;
	sum.d += pixel * Filter_2[1].y;
	sum.e += pixel * Filter_3[1].x;
	sum.f += pixel * Filter_3[1].y;

    pixel = image[y - 3][x + 3]; 
    sum.a += pixel * Filter_1[6*7].x;
	sum.b += pixel * Filter_1[6*7].y;
	sum.c += pixel * Filter_2[6*7].x;
	sum.d += pixel * Filter_2[6*7].y;
	sum.e += pixel * Filter_3[6*7].x;
	sum.f += pixel * Filter_3[6*7].y;
    pixel = image[y - 2][x + 3]; 
    sum.a += pixel * Filter_1[5*7].x;
	sum.b += pixel * Filter_1[5*7].y;
	sum.c += pixel * Filter_2[5*7].x;
	sum.d += pixel * Filter_2[5*7].y;
	sum.e += pixel * Filter_3[5*7].x;
	sum.f += pixel * Filter_3[5*7].y;
    pixel = image[y - 1][x + 3]; 
    sum.a += pixel * Filter_1[4*7].x;
	sum.b += pixel * Filter_1[4*7].y;
	sum.c += pixel * Filter_2[4*7].x;
	sum.d += pixel * Filter_2[4*7].y;
	sum.e += pixel * Filter_3[4*7].x;
	sum.f += pixel * Filter_3[4*7].y;
    pixel = image[y + 0][x + 3]; 
    sum.a += pixel * Filter_1[3*7].x;
	sum.b += pixel * Filter_1[3*7].y;
	sum.c += pixel * Filter_2[3*7].x;
	sum.d += pixel * Filter_2[3*7].y;
	sum.e += pixel * Filter_3[3*7].x;
	sum.f += pixel * Filter_3[3*7].y;
    pixel = image[y + 1][x + 3]; 
    sum.a += pixel * Filter_1[2*7].x;
	sum.b += pixel * Filter_1[2*7].y;
	sum.c += pixel * Filter_2[2*7].x;
	sum.d += pixel * Filter_2[2*7].y;
	sum.e += pixel * Filter_3[2*7].x;
	sum.f += pixel * Filter_3[2*7].y;
    pixel = image[y + 2][x + 3]; 
    sum.a += pixel * Filter_1[1*7].x;
	sum.b += pixel * Filter_1[1*7].y;
	sum.c += pixel * Filter_2[1*7].x;
	sum.d += pixel * Filter_2[1*7].y;
	sum.e += pixel * Filter_3[1*7].x;
	sum.f += pixel * Filter_3[1*7].y;
    pixel = image[y + 3][x + 3]; 
    sum.a += pixel * Filter_1[0].x;
	sum.b += pixel * Filter_1[0].y;
	sum.c += pixel * Filter_2[0].x;
	sum.d += pixel * Filter_2[0].y;
	sum.e += pixel * Filter_3[0].x;
	sum.f += pixel * Filter_3[0].y;

	return sum;
}

float6 Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(__local float image[64][128], 
	                                      	  int y,
	                                      	  int x,
	                                      	  __constant float* Filter_1_Real,
	                                      	  __constant float* Filter_1_Imag,
	                                      	  __constant float* Filter_2_Real,
	                                      	  __constant float* Filter_2_Imag,
	                                      	  __constant float* Filter_3_Real,
	                                      	  __constant float* Filter_3_Imag)
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
    sum.a += pixel * Filter_1_Real[6*7 + 6];
	sum.b += pixel * Filter_1_Imag[6*7 + 6];
	sum.c += pixel * Filter_2_Real[6*7 + 6];
	sum.d += pixel * Filter_2_Imag[6*7 + 6];
	sum.e += pixel * Filter_3_Real[6*7 + 6];
	sum.f += pixel * Filter_3_Imag[6*7 + 6];
    pixel = image[y - 2][x - 3]; 
    sum.a += pixel * Filter_1_Real[5*7 + 6];
	sum.b += pixel * Filter_1_Imag[5*7 + 6];
	sum.c += pixel * Filter_2_Real[5*7 + 6];
	sum.d += pixel * Filter_2_Imag[5*7 + 6];
	sum.e += pixel * Filter_3_Real[5*7 + 6];
	sum.f += pixel * Filter_3_Imag[5*7 + 6];
	pixel = image[y - 1][x - 3]; 
    sum.a += pixel * Filter_1_Real[4*7 + 6];
	sum.b += pixel * Filter_1_Imag[4*7 + 6];
	sum.c += pixel * Filter_2_Real[4*7 + 6];
	sum.d += pixel * Filter_2_Imag[4*7 + 6];
	sum.e += pixel * Filter_3_Real[4*7 + 6];
	sum.f += pixel * Filter_3_Imag[4*7 + 6];
	pixel = image[y + 0][x - 3]; 
    sum.a += pixel * Filter_1_Real[3*7 + 6];
	sum.b += pixel * Filter_1_Imag[3*7 + 6];
	sum.c += pixel * Filter_2_Real[3*7 + 6];
	sum.d += pixel * Filter_2_Imag[3*7 + 6];
	sum.e += pixel * Filter_3_Real[3*7 + 6];
	sum.f += pixel * Filter_3_Imag[3*7 + 6];
    pixel = image[y + 1][x - 3]; 
    sum.a += pixel * Filter_1_Real[2*7 + 6];
	sum.b += pixel * Filter_1_Imag[2*7 + 6];
	sum.c += pixel * Filter_2_Real[2*7 + 6];
	sum.d += pixel * Filter_2_Imag[2*7 + 6];
	sum.e += pixel * Filter_3_Real[2*7 + 6];
	sum.f += pixel * Filter_3_Imag[2*7 + 6];
	pixel = image[y + 2][x - 3]; 
    sum.a += pixel * Filter_1_Real[1*7 + 6];
	sum.b += pixel * Filter_1_Imag[1*7 + 6];
	sum.c += pixel * Filter_2_Real[1*7 + 6];
	sum.d += pixel * Filter_2_Imag[1*7 + 6];
	sum.e += pixel * Filter_3_Real[1*7 + 6];
	sum.f += pixel * Filter_3_Imag[1*7 + 6];
	pixel = image[y + 3][x - 3]; 
    sum.a += pixel * Filter_1_Real[6];
	sum.b += pixel * Filter_1_Imag[6];
	sum.c += pixel * Filter_2_Real[6];
	sum.d += pixel * Filter_2_Imag[6];
	sum.e += pixel * Filter_3_Real[6];
	sum.f += pixel * Filter_3_Imag[6];

    pixel = image[y - 3][x - 2]; 
    sum.a += pixel * Filter_1_Real[6*7 + 5];
	sum.b += pixel * Filter_1_Imag[6*7 + 5];
	sum.c += pixel * Filter_2_Real[6*7 + 5];
	sum.d += pixel * Filter_2_Imag[6*7 + 5];
	sum.e += pixel * Filter_3_Real[6*7 + 5];
	sum.f += pixel * Filter_3_Imag[6*7 + 5];
    pixel = image[y - 2][x - 2]; 
    sum.a += pixel * Filter_1_Real[5*7 + 5];
	sum.b += pixel * Filter_1_Imag[5*7 + 5];
	sum.c += pixel * Filter_2_Real[5*7 + 5];
	sum.d += pixel * Filter_2_Imag[5*7 + 5];
	sum.e += pixel * Filter_3_Real[5*7 + 5];
	sum.f += pixel * Filter_3_Imag[5*7 + 5];
    pixel = image[y - 1][x - 2]; 
    sum.a += pixel * Filter_1_Real[4*7 + 5];
	sum.b += pixel * Filter_1_Imag[4*7 + 5];
	sum.c += pixel * Filter_2_Real[4*7 + 5];
	sum.d += pixel * Filter_2_Imag[4*7 + 5];
	sum.e += pixel * Filter_3_Real[4*7 + 5];
	sum.f += pixel * Filter_3_Imag[4*7 + 5];
    pixel = image[y + 0][x - 2]; 
    sum.a += pixel * Filter_1_Real[3*7 + 5];
	sum.b += pixel * Filter_1_Imag[3*7 + 5];
	sum.c += pixel * Filter_2_Real[3*7 + 5];
	sum.d += pixel * Filter_2_Imag[3*7 + 5];
	sum.e += pixel * Filter_3_Real[3*7 + 5];
	sum.f += pixel * Filter_3_Imag[3*7 + 5];
    pixel = image[y + 1][x - 2]; 
    sum.a += pixel * Filter_1_Real[2*7 + 5];
	sum.b += pixel * Filter_1_Imag[2*7 + 5];
	sum.c += pixel * Filter_2_Real[2*7 + 5];
	sum.d += pixel * Filter_2_Imag[2*7 + 5];
	sum.e += pixel * Filter_3_Real[2*7 + 5];
	sum.f += pixel * Filter_3_Imag[2*7 + 5];
    pixel = image[y + 2][x - 2]; 
    sum.a += pixel * Filter_1_Real[1*7 + 5];
	sum.b += pixel * Filter_1_Imag[1*7 + 5];
	sum.c += pixel * Filter_2_Real[1*7 + 5];
	sum.d += pixel * Filter_2_Imag[1*7 + 5];
	sum.e += pixel * Filter_3_Real[1*7 + 5];
	sum.f += pixel * Filter_3_Imag[1*7 + 5];
    pixel = image[y + 3][x - 2]; 
    sum.a += pixel * Filter_1_Real[5];
	sum.b += pixel * Filter_1_Imag[5];
	sum.c += pixel * Filter_2_Real[5];
	sum.d += pixel * Filter_2_Imag[5];
	sum.e += pixel * Filter_3_Real[5];
	sum.f += pixel * Filter_3_Imag[5];


    pixel = image[y - 3][x - 1]; 
    sum.a += pixel * Filter_1_Real[6*7 + 4];
	sum.b += pixel * Filter_1_Imag[6*7 + 4];
	sum.c += pixel * Filter_2_Real[6*7 + 4];
	sum.d += pixel * Filter_2_Imag[6*7 + 4];
	sum.e += pixel * Filter_3_Real[6*7 + 4];
	sum.f += pixel * Filter_3_Imag[6*7 + 4];
    pixel = image[y - 2][x - 1]; 
    sum.a += pixel * Filter_1_Real[5*7 + 4];
	sum.b += pixel * Filter_1_Imag[5*7 + 4];
	sum.c += pixel * Filter_2_Real[5*7 + 4];
	sum.d += pixel * Filter_2_Imag[5*7 + 4];
	sum.e += pixel * Filter_3_Real[5*7 + 4];
	sum.f += pixel * Filter_3_Imag[5*7 + 4];
    pixel = image[y - 1][x - 1]; 
    sum.a += pixel * Filter_1_Real[4*7 + 4];
	sum.b += pixel * Filter_1_Imag[4*7 + 4];
	sum.c += pixel * Filter_2_Real[4*7 + 4];
	sum.d += pixel * Filter_2_Imag[4*7 + 4];
	sum.e += pixel * Filter_3_Real[4*7 + 4];
	sum.f += pixel * Filter_3_Imag[4*7 + 4];
    pixel = image[y + 0][x - 1]; 
    sum.a += pixel * Filter_1_Real[3*7 + 4];
	sum.b += pixel * Filter_1_Imag[3*7 + 4];
	sum.c += pixel * Filter_2_Real[3*7 + 4];
	sum.d += pixel * Filter_2_Imag[3*7 + 4];
	sum.e += pixel * Filter_3_Real[3*7 + 4];
	sum.f += pixel * Filter_3_Imag[3*7 + 4];
    pixel = image[y + 1][x - 1]; 
    sum.a += pixel * Filter_1_Real[2*7 + 4];
	sum.b += pixel * Filter_1_Imag[2*7 + 4];
	sum.c += pixel * Filter_2_Real[2*7 + 4];
	sum.d += pixel * Filter_2_Imag[2*7 + 4];
	sum.e += pixel * Filter_3_Real[2*7 + 4];
	sum.f += pixel * Filter_3_Imag[2*7 + 4];
    pixel = image[y + 2][x - 1]; 
    sum.a += pixel * Filter_1_Real[1*7 + 4];
	sum.b += pixel * Filter_1_Imag[1*7 + 4];
	sum.c += pixel * Filter_2_Real[1*7 + 4];
	sum.d += pixel * Filter_2_Imag[1*7 + 4];
	sum.e += pixel * Filter_3_Real[1*7 + 4];
	sum.f += pixel * Filter_3_Imag[1*7 + 4];
    pixel = image[y + 3][x - 1]; 
    sum.a += pixel * Filter_1_Real[4];
	sum.b += pixel * Filter_1_Imag[4];
	sum.c += pixel * Filter_2_Real[4];
	sum.d += pixel * Filter_2_Imag[4];
	sum.e += pixel * Filter_3_Real[4];
	sum.f += pixel * Filter_3_Imag[4];


    pixel = image[y - 3][x + 0]; 
    sum.a += pixel * Filter_1_Real[6*7 + 3];
	sum.b += pixel * Filter_1_Imag[6*7 + 3];
	sum.c += pixel * Filter_2_Real[6*7 + 3];
	sum.d += pixel * Filter_2_Imag[6*7 + 3];
	sum.e += pixel * Filter_3_Real[6*7 + 3];
	sum.f += pixel * Filter_3_Imag[6*7 + 3];
    pixel = image[y - 2][x + 0]; 
    sum.a += pixel * Filter_1_Real[5*7 + 3];
	sum.b += pixel * Filter_1_Imag[5*7 + 3];
	sum.c += pixel * Filter_2_Real[5*7 + 3];
	sum.d += pixel * Filter_2_Imag[5*7 + 3];
	sum.e += pixel * Filter_3_Real[5*7 + 3];
	sum.f += pixel * Filter_3_Imag[5*7 + 3];
    pixel = image[y - 1][x + 0]; 
    sum.a += pixel * Filter_1_Real[4*7 + 3];
	sum.b += pixel * Filter_1_Imag[4*7 + 3];
	sum.c += pixel * Filter_2_Real[4*7 + 3];
	sum.d += pixel * Filter_2_Imag[4*7 + 3];
	sum.e += pixel * Filter_3_Real[4*7 + 3];
	sum.f += pixel * Filter_3_Imag[4*7 + 3];
    pixel = image[y + 0][x + 0]; 
    sum.a += pixel * Filter_1_Real[3*7 + 3];
	sum.b += pixel * Filter_1_Imag[3*7 + 3];
	sum.c += pixel * Filter_2_Real[3*7 + 3];
	sum.d += pixel * Filter_2_Imag[3*7 + 3];
	sum.e += pixel * Filter_3_Real[3*7 + 3];
	sum.f += pixel * Filter_3_Imag[3*7 + 3];
    pixel = image[y + 1][x + 0]; 
    sum.a += pixel * Filter_1_Real[2*7 + 3];
	sum.b += pixel * Filter_1_Imag[2*7 + 3];
	sum.c += pixel * Filter_2_Real[2*7 + 3];
	sum.d += pixel * Filter_2_Imag[2*7 + 3];
	sum.e += pixel * Filter_3_Real[2*7 + 3];
	sum.f += pixel * Filter_3_Imag[2*7 + 3];
    pixel = image[y + 2][x + 0]; 
    sum.a += pixel * Filter_1_Real[1*7 + 3];
	sum.b += pixel * Filter_1_Imag[1*7 + 3];
	sum.c += pixel * Filter_2_Real[1*7 + 3];
	sum.d += pixel * Filter_2_Imag[1*7 + 3];
	sum.e += pixel * Filter_3_Real[1*7 + 3];
	sum.f += pixel * Filter_3_Imag[1*7 + 3];
    pixel = image[y + 3][x + 0]; 
    sum.a += pixel * Filter_1_Real[3];
	sum.b += pixel * Filter_1_Imag[3];
	sum.c += pixel * Filter_2_Real[3];
	sum.d += pixel * Filter_2_Imag[3];
	sum.e += pixel * Filter_3_Real[3];
	sum.f += pixel * Filter_3_Imag[3];

	pixel = image[y - 3][x + 1]; 
    sum.a += pixel * Filter_1_Real[6*7 + 2];
	sum.b += pixel * Filter_1_Imag[6*7 + 2];
	sum.c += pixel * Filter_2_Real[6*7 + 2];
	sum.d += pixel * Filter_2_Imag[6*7 + 2];
	sum.e += pixel * Filter_3_Real[6*7 + 2];
	sum.f += pixel * Filter_3_Imag[6*7 + 2];
    pixel = image[y - 2][x + 1]; 
    sum.a += pixel * Filter_1_Real[5*7 + 2];
	sum.b += pixel * Filter_1_Imag[5*7 + 2];
	sum.c += pixel * Filter_2_Real[5*7 + 2];
	sum.d += pixel * Filter_2_Imag[5*7 + 2];
	sum.e += pixel * Filter_3_Real[5*7 + 2];
	sum.f += pixel * Filter_3_Imag[5*7 + 2];
    pixel = image[y - 1][x + 1]; 
    sum.a += pixel * Filter_1_Real[4*7 + 2];
	sum.b += pixel * Filter_1_Imag[4*7 + 2];
	sum.c += pixel * Filter_2_Real[4*7 + 2];
	sum.d += pixel * Filter_2_Imag[4*7 + 2];
	sum.e += pixel * Filter_3_Real[4*7 + 2];
	sum.f += pixel * Filter_3_Imag[4*7 + 2];
    pixel = image[y + 0][x + 1]; 
    sum.a += pixel * Filter_1_Real[3*7 + 2];
	sum.b += pixel * Filter_1_Imag[3*7 + 2];
	sum.c += pixel * Filter_2_Real[3*7 + 2];
	sum.d += pixel * Filter_2_Imag[3*7 + 2];
	sum.e += pixel * Filter_3_Real[3*7 + 2];
	sum.f += pixel * Filter_3_Imag[3*7 + 2];
    pixel = image[y + 1][x + 1]; 
    sum.a += pixel * Filter_1_Real[2*7 + 2];
	sum.b += pixel * Filter_1_Imag[2*7 + 2];
	sum.c += pixel * Filter_2_Real[2*7 + 2];
	sum.d += pixel * Filter_2_Imag[2*7 + 2];
	sum.e += pixel * Filter_3_Real[2*7 + 2];
	sum.f += pixel * Filter_3_Imag[2*7 + 2];
    pixel = image[y + 2][x + 1]; 
    sum.a += pixel * Filter_1_Real[1*7 + 2];
	sum.b += pixel * Filter_1_Imag[1*7 + 2];
	sum.c += pixel * Filter_2_Real[1*7 + 2];
	sum.d += pixel * Filter_2_Imag[1*7 + 2];
	sum.e += pixel * Filter_3_Real[1*7 + 2];
	sum.f += pixel * Filter_3_Imag[1*7 + 2];
    pixel = image[y + 3][x + 1]; 
    sum.a += pixel * Filter_1_Real[2];
	sum.b += pixel * Filter_1_Imag[2];
	sum.c += pixel * Filter_2_Real[2];
	sum.d += pixel * Filter_2_Imag[2];
	sum.e += pixel * Filter_3_Real[2];
	sum.f += pixel * Filter_3_Imag[2];
 
    pixel = image[y - 3][x + 2]; 
    sum.a += pixel * Filter_1_Real[6*7 + 1];
	sum.b += pixel * Filter_1_Imag[6*7 + 1];
	sum.c += pixel * Filter_2_Real[6*7 + 1];
	sum.d += pixel * Filter_2_Imag[6*7 + 1];
	sum.e += pixel * Filter_3_Real[6*7 + 1];
	sum.f += pixel * Filter_3_Imag[6*7 + 1];
    pixel = image[y - 2][x + 2]; 
    sum.a += pixel * Filter_1_Real[5*7 + 1];
	sum.b += pixel * Filter_1_Imag[5*7 + 1];
	sum.c += pixel * Filter_2_Real[5*7 + 1];
	sum.d += pixel * Filter_2_Imag[5*7 + 1];
	sum.e += pixel * Filter_3_Real[5*7 + 1];
	sum.f += pixel * Filter_3_Imag[5*7 + 1];
    pixel = image[y - 1][x + 2]; 
    sum.a += pixel * Filter_1_Real[4*7 + 1];
	sum.b += pixel * Filter_1_Imag[4*7 + 1];
	sum.c += pixel * Filter_2_Real[4*7 + 1];
	sum.d += pixel * Filter_2_Imag[4*7 + 1];
	sum.e += pixel * Filter_3_Real[4*7 + 1];
	sum.f += pixel * Filter_3_Imag[4*7 + 1];
    pixel = image[y + 0][x + 2]; 
    sum.a += pixel * Filter_1_Real[3*7 + 1];
	sum.b += pixel * Filter_1_Imag[3*7 + 1];
	sum.c += pixel * Filter_2_Real[3*7 + 1];
	sum.d += pixel * Filter_2_Imag[3*7 + 1];
	sum.e += pixel * Filter_3_Real[3*7 + 1];
	sum.f += pixel * Filter_3_Imag[3*7 + 1];
	pixel = image[y + 1][x + 2]; 
    sum.a += pixel * Filter_1_Real[2*7 + 1];
	sum.b += pixel * Filter_1_Imag[2*7 + 1];
	sum.c += pixel * Filter_2_Real[2*7 + 1];
	sum.d += pixel * Filter_2_Imag[2*7 + 1];
	sum.e += pixel * Filter_3_Real[2*7 + 1];
	sum.f += pixel * Filter_3_Imag[2*7 + 1];
    pixel = image[y + 2][x + 2]; 
    sum.a += pixel * Filter_1_Real[1*7 + 1];
	sum.b += pixel * Filter_1_Imag[1*7 + 1];
	sum.c += pixel * Filter_2_Real[1*7 + 1];
	sum.d += pixel * Filter_2_Imag[1*7 + 1];
	sum.e += pixel * Filter_3_Real[1*7 + 1];
	sum.f += pixel * Filter_3_Imag[1*7 + 1];
    pixel = image[y + 3][x + 2]; 
    sum.a += pixel * Filter_1_Real[1];
	sum.b += pixel * Filter_1_Imag[1];
	sum.c += pixel * Filter_2_Real[1];
	sum.d += pixel * Filter_2_Imag[1];
	sum.e += pixel * Filter_3_Real[1];
	sum.f += pixel * Filter_3_Imag[1];

    pixel = image[y - 3][x + 3]; 
    sum.a += pixel * Filter_1_Real[6*7];
	sum.b += pixel * Filter_1_Imag[6*7];
	sum.c += pixel * Filter_2_Real[6*7];
	sum.d += pixel * Filter_2_Imag[6*7];
	sum.e += pixel * Filter_3_Real[6*7];
	sum.f += pixel * Filter_3_Imag[6*7];
    pixel = image[y - 2][x + 3]; 
    sum.a += pixel * Filter_1_Real[5*7];
	sum.b += pixel * Filter_1_Imag[5*7];
	sum.c += pixel * Filter_2_Real[5*7];
	sum.d += pixel * Filter_2_Imag[5*7];
	sum.e += pixel * Filter_3_Real[5*7];
	sum.f += pixel * Filter_3_Imag[5*7];
    pixel = image[y - 1][x + 3]; 
    sum.a += pixel * Filter_1_Real[4*7];
	sum.b += pixel * Filter_1_Imag[4*7];
	sum.c += pixel * Filter_2_Real[4*7];
	sum.d += pixel * Filter_2_Imag[4*7];
	sum.e += pixel * Filter_3_Real[4*7];
	sum.f += pixel * Filter_3_Imag[4*7];
    pixel = image[y + 0][x + 3]; 
    sum.a += pixel * Filter_1_Real[3*7];
	sum.b += pixel * Filter_1_Imag[3*7];
	sum.c += pixel * Filter_2_Real[3*7];
	sum.d += pixel * Filter_2_Imag[3*7];
	sum.e += pixel * Filter_3_Real[3*7];
	sum.f += pixel * Filter_3_Imag[3*7];
    pixel = image[y + 1][x + 3]; 
    sum.a += pixel * Filter_1_Real[2*7];
	sum.b += pixel * Filter_1_Imag[2*7];
	sum.c += pixel * Filter_2_Real[2*7];
	sum.d += pixel * Filter_2_Imag[2*7];
	sum.e += pixel * Filter_3_Real[2*7];
	sum.f += pixel * Filter_3_Imag[2*7];
    pixel = image[y + 2][x + 3]; 
    sum.a += pixel * Filter_1_Real[1*7];
	sum.b += pixel * Filter_1_Imag[1*7];
	sum.c += pixel * Filter_2_Real[1*7];
	sum.d += pixel * Filter_2_Imag[1*7];
	sum.e += pixel * Filter_3_Real[1*7];
	sum.f += pixel * Filter_3_Imag[1*7];
    pixel = image[y + 3][x + 3]; 
    sum.a += pixel * Filter_1_Real[0];
	sum.b += pixel * Filter_1_Imag[0];
	sum.c += pixel * Filter_2_Real[0];
	sum.d += pixel * Filter_2_Imag[0];
	sum.e += pixel * Filter_3_Real[0];
	sum.f += pixel * Filter_3_Imag[0];

	return sum;
}

__kernel void MemsetInt(__global int *Data,
	                    __private int value,
					    __private int N)
{
	int i = get_global_id(0);

	if (i >= N)
		return;

	Data[i] = value;
}


__kernel void Memset(__global float *Data, 
	                 __private float value, 
					 __private int N)
{
	int i = get_global_id(0);

	if (i >= N)
		return;

	Data[i] = value;
}

__kernel void MemsetFloat2(__global float2 *Data, 
	                       __private float value, 
						   __private int N)
{
	int i = get_global_id(0);

	if (i >= N)
		return;

	float2 values;
	values.x = value;
	values.y = value;

	Data[i] = values;
}




__kernel void Nonseparable3DConvolutionComplexThreeQuadratureFilters(/*
																	 __global float* Filter_Response_1_Real,
	                                                                 __global float* Filter_Response_1_Imag,
																	 __global float* Filter_Response_2_Real,
																	 __global float* Filter_Response_2_Imag,
																	 __global float* Filter_Response_3_Real,
																	 __global float* Filter_Response_3_Imag, 
																	 */

																	 __global float2* Filter_Response_1,
	                                                                 __global float2* Filter_Response_2,
																	 __global float2* Filter_Response_3,
																	 
																	 __global const float* Volume, 
																	 
																	 __constant float* c_Quadrature_Filter_1_Real, 
																	 __constant float* c_Quadrature_Filter_1_Imag, 
																	 __constant float* c_Quadrature_Filter_2_Real, 
																	 __constant float* c_Quadrature_Filter_2_Imag, 
																	 __constant float* c_Quadrature_Filter_3_Real, 
																	 __constant float* c_Quadrature_Filter_3_Imag, 

																	 __private int z_offset, 
																	 __private int DATA_W, 
																	 __private int DATA_H, 
																	 __private int DATA_D)
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
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_(l_Image,tIdx.y+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += (float2)(temp.a,temp.b);
	    Filter_Response_2[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += (float2)(temp.c,temp.d);
	    Filter_Response_3[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += (float2)(temp.e,temp.f);	    
    }

    if ( ((x + 32) < DATA_W) && (y < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_(l_Image,tIdx.y+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += (float2)(temp.a,temp.b);
	    Filter_Response_2[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += (float2)(temp.c,temp.d);
	    Filter_Response_3[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += (float2)(temp.e,temp.f);

    }

    if (tIdx.x < (32 - HALO*2))
    {
        if ( ((x + 64) < DATA_W) && (y < DATA_H) )
	    {
		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_(l_Image,tIdx.y+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += (float2)(temp.a,temp.b);
			Filter_Response_2[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += (float2)(temp.c,temp.d);
			Filter_Response_3[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += (float2)(temp.e,temp.f);
	    }
    }

    if (tIdx.y < (32 - HALO*2))
    {
        if ( (x < DATA_W) && ((y + 32) < DATA_H) )
	    {
 		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_(l_Image,tIdx.y+32+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a,temp.b);
			Filter_Response_2[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c,temp.d);
			Filter_Response_3[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e,temp.f);
	    }
    }

    if (tIdx.y < (32 - HALO*2))
    {
        if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
	    {
		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_(l_Image,tIdx.y+32+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a,temp.b);
			Filter_Response_2[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c,temp.d);
			Filter_Response_3[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e,temp.f);
	    }
     } 

    if ( (tIdx.x < (32 - HALO*2)) && (tIdx.y < (32 - HALO*2)) )
    {
        if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
	    {
		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_(l_Image,tIdx.y+32+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a,temp.b);
			Filter_Response_2[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c,temp.d);
			Filter_Response_3[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e,temp.f);
	    }
     }
}




__kernel void Nonseparable3DConvolutionComplexThreeQuadratureFiltersAMD(__global float2 *Filter_Response_1,
	                                                                    __global float2 *Filter_Response_2, 
																		__global float2 *Filter_Response_3, 																		
																		__global const float* Volume, 
																		__constant float* c_Quadrature_Filter_1_Real, 
																		__constant float* c_Quadrature_Filter_1_Imag, 
																		__constant float* c_Quadrature_Filter_2_Real, 
																		__constant float* c_Quadrature_Filter_2_Imag, 
																		__constant float* c_Quadrature_Filter_3_Real, 
																		__constant float* c_Quadrature_Filter_3_Imag, 
																		__private int z_offset, 
																		__private int DATA_W, 
																		__private int DATA_H, 
																		__private int DATA_D)
{   
    int x = get_group_id(0) * VALID_FILTER_RESPONSES_X_CONVOLUTION_2D_AMD + get_local_id(0);
	int y = get_group_id(1) * VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D_AMD + get_local_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};
    
	
    __local float l_Image[64][128]; // y, x

    // Reset shared memory
    l_Image[tIdx.y][tIdx.x]           = 0.0f;
    l_Image[tIdx.y][tIdx.x + 16]      = 0.0f;
    l_Image[tIdx.y][tIdx.x + 32]      = 0.0f;
	l_Image[tIdx.y][tIdx.x + 48]      = 0.0f;
	l_Image[tIdx.y][tIdx.x + 64]      = 0.0f;
    l_Image[tIdx.y][tIdx.x + 80]      = 0.0f;
	l_Image[tIdx.y][tIdx.x + 96]      = 0.0f;
	l_Image[tIdx.y][tIdx.x + 112]     = 0.0f;

	l_Image[tIdx.y + 16][tIdx.x]           = 0.0f;
    l_Image[tIdx.y + 16][tIdx.x + 16]      = 0.0f;
    l_Image[tIdx.y + 16][tIdx.x + 32]      = 0.0f;
	l_Image[tIdx.y + 16][tIdx.x + 48]      = 0.0f;
	l_Image[tIdx.y + 16][tIdx.x + 64]      = 0.0f;
    l_Image[tIdx.y + 16][tIdx.x + 80]      = 0.0f;
	l_Image[tIdx.y + 16][tIdx.x + 96]      = 0.0f;
	l_Image[tIdx.y + 16][tIdx.x + 112]     = 0.0f;

	l_Image[tIdx.y + 32][tIdx.x]           = 0.0f;
    l_Image[tIdx.y + 32][tIdx.x + 16]      = 0.0f;
    l_Image[tIdx.y + 32][tIdx.x + 32]      = 0.0f;
	l_Image[tIdx.y + 32][tIdx.x + 48]      = 0.0f;
	l_Image[tIdx.y + 32][tIdx.x + 64]      = 0.0f;
    l_Image[tIdx.y + 32][tIdx.x + 80]      = 0.0f;
	l_Image[tIdx.y + 32][tIdx.x + 96]      = 0.0f;
	l_Image[tIdx.y + 32][tIdx.x + 112]     = 0.0f;

	l_Image[tIdx.y + 48][tIdx.x]           = 0.0f;
    l_Image[tIdx.y + 48][tIdx.x + 16]      = 0.0f;
    l_Image[tIdx.y + 48][tIdx.x + 32]      = 0.0f;
	l_Image[tIdx.y + 48][tIdx.x + 48]      = 0.0f;
	l_Image[tIdx.y + 48][tIdx.x + 64]      = 0.0f;
    l_Image[tIdx.y + 48][tIdx.x + 80]      = 0.0f;
	l_Image[tIdx.y + 48][tIdx.x + 96]      = 0.0f;
	l_Image[tIdx.y + 48][tIdx.x + 112]     = 0.0f;

    // Read data into shared memory

    if ( ((z + z_offset) >= 0) && ((z + z_offset) < DATA_D) )
    {
        if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  )   
            l_Image[tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+16-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  )
            l_Image[tIdx.y][tIdx.x + 16] = Volume[Calculate3DIndex(x+16-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+32-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  ) 
            l_Image[tIdx.y][tIdx.x + 32] = Volume[Calculate3DIndex(x+32-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

		if ( ((x+48-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  )   
            l_Image[tIdx.y][tIdx.x + 48] = Volume[Calculate3DIndex(x+48-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+64-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  )
            l_Image[tIdx.y][tIdx.x + 64] = Volume[Calculate3DIndex(x+64-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+80-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  ) 
            l_Image[tIdx.y][tIdx.x + 80] = Volume[Calculate3DIndex(x+80-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

		if ( ((x+96-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  )
            l_Image[tIdx.y][tIdx.x + 96] = Volume[Calculate3DIndex(x+96-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+112-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  ) 
            l_Image[tIdx.y][tIdx.x + 112] = Volume[Calculate3DIndex(x+112-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];


		if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+16-HALO) < DATA_H)  )   
            l_Image[tIdx.y + 16][tIdx.x] = Volume[Calculate3DIndex(x-HALO,y+16-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+16-HALO) < DATA_W) && ((y+16-HALO) < DATA_H)  )
            l_Image[tIdx.y + 16][tIdx.x + 16] = Volume[Calculate3DIndex(x+16-HALO,y+16-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+32-HALO) < DATA_W) && ((y+16-HALO) < DATA_H)  ) 
            l_Image[tIdx.y + 16][tIdx.x + 32] = Volume[Calculate3DIndex(x+32-HALO,y+16-HALO,z+z_offset,DATA_W,DATA_H)];

		if ( ((x+48-HALO) < DATA_W) && ((y+16-HALO) < DATA_H)  )   
            l_Image[tIdx.y + 16][tIdx.x + 48] = Volume[Calculate3DIndex(x+48-HALO,y+16-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+64-HALO) < DATA_W) && ((y+16-HALO) < DATA_H)  )
            l_Image[tIdx.y + 16][tIdx.x + 64] = Volume[Calculate3DIndex(x+64-HALO,y+16-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+80-HALO) < DATA_W) && ((y+16-HALO) < DATA_H)  ) 
            l_Image[tIdx.y + 16][tIdx.x + 80] = Volume[Calculate3DIndex(x+80-HALO,y+16-HALO,z+z_offset,DATA_W,DATA_H)];

		if ( ((x+96-HALO) < DATA_W) && ((y+16-HALO) < DATA_H)  )
            l_Image[tIdx.y + 16][tIdx.x + 96] = Volume[Calculate3DIndex(x+96-HALO,y+16-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+112-HALO) < DATA_W) && ((y+16-HALO) < DATA_H)  ) 
            l_Image[tIdx.y + 16][tIdx.x + 112] = Volume[Calculate3DIndex(x+112-HALO,y+16-HALO,z+z_offset,DATA_W,DATA_H)];


		if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )   
            l_Image[tIdx.y + 32][tIdx.x] = Volume[Calculate3DIndex(x-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+16-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )
            l_Image[tIdx.y + 32][tIdx.x + 16] = Volume[Calculate3DIndex(x+16-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+32-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  ) 
            l_Image[tIdx.y + 32][tIdx.x + 32] = Volume[Calculate3DIndex(x+32-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];

		if ( ((x+48-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )   
            l_Image[tIdx.y + 32][tIdx.x + 48] = Volume[Calculate3DIndex(x+48-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+64-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )
            l_Image[tIdx.y + 32][tIdx.x + 64] = Volume[Calculate3DIndex(x+64-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+80-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  ) 
            l_Image[tIdx.y + 32][tIdx.x + 80] = Volume[Calculate3DIndex(x+80-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];

		if ( ((x+96-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )
            l_Image[tIdx.y + 32][tIdx.x + 96] = Volume[Calculate3DIndex(x+96-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+112-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  ) 
            l_Image[tIdx.y + 32][tIdx.x + 112] = Volume[Calculate3DIndex(x+112-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];


		if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+48-HALO) < DATA_H)  )   
            l_Image[tIdx.y + 48][tIdx.x] = Volume[Calculate3DIndex(x-HALO,y+48-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+16-HALO) < DATA_W) && ((y+48-HALO) < DATA_H)  )
            l_Image[tIdx.y + 48][tIdx.x + 16] = Volume[Calculate3DIndex(x+16-HALO,y+48-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+32-HALO) < DATA_W) && ((y+48-HALO) < DATA_H)  ) 
            l_Image[tIdx.y + 48][tIdx.x + 32] = Volume[Calculate3DIndex(x+32-HALO,y+48-HALO,z+z_offset,DATA_W,DATA_H)];

		if ( ((x+48-HALO) < DATA_W) && ((y+48-HALO) < DATA_H)  )   
            l_Image[tIdx.y + 48][tIdx.x + 48] = Volume[Calculate3DIndex(x+48-HALO,y+48-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+64-HALO) < DATA_W) && ((y+48-HALO) < DATA_H)  )
            l_Image[tIdx.y + 48][tIdx.x + 64] = Volume[Calculate3DIndex(x+64-HALO,y+48-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+80-HALO) < DATA_W) && ((y+48-HALO) < DATA_H)  ) 
            l_Image[tIdx.y + 48][tIdx.x + 80] = Volume[Calculate3DIndex(x+80-HALO,y+48-HALO,z+z_offset,DATA_W,DATA_H)];

		if ( ((x+96-HALO) < DATA_W) && ((y+48-HALO) < DATA_H)  )
            l_Image[tIdx.y + 48][tIdx.x + 96] = Volume[Calculate3DIndex(x+96-HALO,y+48-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+112-HALO) < DATA_W) && ((y+48-HALO) < DATA_H)  ) 
            l_Image[tIdx.y + 48][tIdx.x + 112] = Volume[Calculate3DIndex(x+112-HALO,y+48-HALO,z+z_offset,DATA_W,DATA_H)];

    }
	
   	// Make sure all threads have written to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

    // Only threads inside the image do the convolution

    if ( (x < DATA_W) && (y < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
	    Filter_Response_2[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
	    Filter_Response_3[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);			    
    }

	if ( ((x + 16) < DATA_W) && (y < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+HALO,tIdx.x+16+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+16,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
	    Filter_Response_2[Calculate3DIndex(x+16,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
	    Filter_Response_3[Calculate3DIndex(x+16,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				    
    }

    if ( ((x + 32) < DATA_W) && (y < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

	if ( ((x + 48) < DATA_W) && (y < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+HALO,tIdx.x+48+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+48,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+48,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+48,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				
    }

	if ( ((x + 64) < DATA_W) && (y < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

	if ( ((x + 80) < DATA_W) && (y < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+HALO,tIdx.x+80+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+80,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+80,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+80,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				
    }

	if ( ((x + 96) < DATA_W) && (y < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+HALO,tIdx.x+96+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+96,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+96,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+96,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

    if (tIdx.x < (16 - HALO*2))
    {
        if ( ((x + 112) < DATA_W) && (y < DATA_H) )
	    {
		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+HALO,tIdx.x+112+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
            Filter_Response_1[Calculate3DIndex(x+112,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		    Filter_Response_2[Calculate3DIndex(x+112,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		    Filter_Response_3[Calculate3DIndex(x+112,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				    
	    }
    }

	if ( (x < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+16+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
	    Filter_Response_2[Calculate3DIndex(x,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
	    Filter_Response_3[Calculate3DIndex(x,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);			    
    }

	if ( ((x + 16) < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+16+HALO,tIdx.x+16+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+16,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
	    Filter_Response_2[Calculate3DIndex(x+16,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
	    Filter_Response_3[Calculate3DIndex(x+16,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				    
    }

    if ( ((x + 32) < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+16+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+32,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+32,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+32,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				
    }

	if ( ((x + 48) < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+16+HALO,tIdx.x+48+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+48,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+48,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+48,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				
    }

	if ( ((x + 64) < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+16+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+64,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+64,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+64,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

	if ( ((x + 80) < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+16+HALO,tIdx.x+80+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+80,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+80,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+80,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

	if ( ((x + 96) < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+16+HALO,tIdx.x+96+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+96,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+96,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+96,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

    if (tIdx.x < (16 - HALO*2))
    {
        if ( ((x + 112) < DATA_W) && ((y + 16) < DATA_H) )
	    {
		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+16+HALO,tIdx.x+112+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
            Filter_Response_1[Calculate3DIndex(x+112,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		    Filter_Response_2[Calculate3DIndex(x+112,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		    Filter_Response_3[Calculate3DIndex(x+112,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				    
	    }
    }


	if ( (x < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+32+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
	    Filter_Response_2[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
	    Filter_Response_3[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);			    
    }

	if ( ((x + 16) < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+32+HALO,tIdx.x+16+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+16,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
	    Filter_Response_2[Calculate3DIndex(x+16,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
	    Filter_Response_3[Calculate3DIndex(x+16,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);			    
    }

    if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+32+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

	if ( ((x + 48) < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+32+HALO,tIdx.x+48+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+48,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+48,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+48,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				
    }

	if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+32+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				
    }

	if ( ((x + 80) < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+32+HALO,tIdx.x+80+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+80,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+80,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+80,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

	if ( ((x + 96) < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+32+HALO,tIdx.x+96+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+96,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+96,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+96,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

    if (tIdx.x < (16 - HALO*2))
    {
        if ( ((x + 112) < DATA_W) && ((y + 32) < DATA_H) )
	    {
		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+32+HALO,tIdx.x+112+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
            Filter_Response_1[Calculate3DIndex(x+112,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		    Filter_Response_2[Calculate3DIndex(x+112,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		    Filter_Response_3[Calculate3DIndex(x+112,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				    
	    }
    }

	if (tIdx.y < (16 - HALO*2))
    {	
		if ( (x < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+48+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
		}

		if ( ((x + 16) < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+48+HALO,tIdx.x+16+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+16,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+16,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+16,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);						
		}

		if ( ((x + 32) < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+48+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+32,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+32,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+32,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);						
		}

		if ( ((x + 48) < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+48+HALO,tIdx.x+48+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
	        Filter_Response_1[Calculate3DIndex(x+48,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+48,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+48,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);						
		}

		if ( ((x + 64) < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+48+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+64,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+64,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+64,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);						
		}

		if ( ((x + 80) < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+48+HALO,tIdx.x+80+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+80,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+80,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+80,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
		}

		if ( ((x + 96) < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+48+HALO,tIdx.x+96+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+96,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+96,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+96,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
		}

		if (tIdx.x < (16 - HALO*2))
		{
	        if ( ((x + 112) < DATA_W) && ((y + 48) < DATA_H) )
			{
			    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_AMD_(l_Image,tIdx.y+48+HALO,tIdx.x+112+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
				Filter_Response_1[Calculate3DIndex(x+112,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
				Filter_Response_2[Calculate3DIndex(x+112,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
				Filter_Response_3[Calculate3DIndex(x+112,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);			
			}
		}
	}	
}





__kernel void CalculatePhaseDifferencesAndCertainties(__global float* Phase_Differences, 
	                                                  __global float* Certainties, 
													  __global const float2* q11, 
													  __global const float2* q21, 
													  __private int DATA_W, 
													  __private int DATA_H, 
													  __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	
	if ( (x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D))
		return;

	int idx = Calculate3DIndex(x, y, z, DATA_W, DATA_H);

	float complex_product_real, complex_product_imag;
	float2 a, c;
	float phase_difference;

	// q1 = a + i * b
	// q2 = c + i * d
	a = q11[idx];
	c = q21[idx];

	// phase difference = arg (q1 * (complex conjugate of q2))	
	complex_product_real = a.x * c.x + a.y * c.y;
	complex_product_imag = a.y * c.x - a.x * c.y;

	phase_difference = atan2(complex_product_imag, complex_product_real);

	complex_product_real = a.x * c.x - a.y * c.y;
  	complex_product_imag = a.y * c.x + a.x * c.y;

	c.x = cos( phase_difference * 0.5f );
	Phase_Differences[idx] = phase_difference;
	Certainties[idx] = sqrt(complex_product_real * complex_product_real + complex_product_imag * complex_product_imag) * c.x * c.x;
}

__kernel void CalculatePhaseGradientsX(__global float* Phase_Gradients, 
	                                   __global const float2* q11, 
									   __global const float2* q21, 
									   __private int DATA_W, 
									   __private int DATA_H, 
									   __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	
	if ( (x >= DATA_W) || ((x + 1) >= DATA_W) || ((x - 1) < 0) || (y >= DATA_H) || (z >= DATA_D))
		return;

	float total_complex_product_real, total_complex_product_imag;
	float2 a, c;
	int idx_minus_1, idx_plus_1, idx;

	idx = Calculate3DIndex(x, y, z, DATA_W, DATA_H);

	// X
	idx_minus_1 = Calculate3DIndex(x - 1, y, z, DATA_W, DATA_H);
	idx_plus_1 = Calculate3DIndex(x + 1, y, z, DATA_W, DATA_H);

	total_complex_product_real = 0.0f;
	total_complex_product_imag = 0.0f;

	a = q11[idx_plus_1];
	c = q11[idx];

	total_complex_product_real += a.x * c.x + a.y * c.y;
	total_complex_product_imag += a.y * c.x - a.x * c.y;

	a.x = c.x;
	a.y = c.y;
	c = q11[idx_minus_1];
	
	total_complex_product_real += a.x * c.x + a.y * c.y;
	total_complex_product_imag += a.y * c.x - a.x * c.y;

	a = q21[idx_plus_1];
	c = q21[idx];

	total_complex_product_real += a.x * c.x + a.y * c.y;
	total_complex_product_imag += a.y * c.x - a.x * c.y;

	a.x = c.x;
	a.y = c.y;
	c = q21[idx_minus_1];
	
	total_complex_product_real += a.x * c.x + a.y * c.y;
	total_complex_product_imag += a.y * c.x - a.x * c.y;

	Phase_Gradients[idx] = atan2(total_complex_product_imag, total_complex_product_real);
}


__kernel void CalculatePhaseGradientsY(__global float* Phase_Gradients, 
	                                   __global const float2* q12, 
									   __global const float2* q22, 
									   __private int DATA_W, 
									   __private int DATA_H, 
									   __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	
	if ( (x >= DATA_W) || (y >= DATA_H) || ((y + 1) >= DATA_H) || ((y - 1) < 0) || (z >= DATA_D))	
		return;

	float total_complex_product_real, total_complex_product_imag;
	float2 a, c;
	int idx_minus_1, idx_plus_1, idx;

	idx = Calculate3DIndex(x, y, z, DATA_W, DATA_H);

	// Y

	idx_plus_1 =  Calculate3DIndex(x, y + 1, z, DATA_W, DATA_H);
	idx_minus_1 =  Calculate3DIndex(x, y - 1, z, DATA_W, DATA_H);

	total_complex_product_real = 0.0f;
	total_complex_product_imag = 0.0f;

	a = q12[idx_plus_1];
	c = q12[idx];
	
	total_complex_product_real += a.x * c.x + a.y * c.y;
	total_complex_product_imag += a.y * c.x - a.x * c.y;

	a.x = c.x;
	a.y = c.y;
	c = q12[idx_minus_1];
	
	total_complex_product_real += a.x * c.x + a.y * c.y;
	total_complex_product_imag += a.y * c.x - a.x * c.y;

	a = q22[idx_plus_1];
	c = q22[idx];
	
	total_complex_product_real += a.x * c.x + a.y * c.y;
	total_complex_product_imag += a.y * c.x - a.x * c.y;

	a.x = c.x;
	a.y = c.y;
	c = q22[idx_minus_1];
	
	total_complex_product_real += a.x * c.x + a.y * c.y;
	total_complex_product_imag += a.y * c.x - a.x * c.y;

	Phase_Gradients[idx] = atan2(total_complex_product_imag, total_complex_product_real);
}

__kernel void CalculatePhaseGradientsZ(__global float* Phase_Gradients,
	                                   __global const float2* q13, 
									   __global const float2* q23, 
									   __private int DATA_W, 
									   __private int DATA_H, 
									   __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	
	if ( (x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D) || ((z + 1) >= DATA_D) || ((z - 1) < 0) )	
		return;

	float total_complex_product_real, total_complex_product_imag;
	float2 a, c;
	int idx_minus_1, idx_plus_1, idx;

	idx = Calculate3DIndex(x, y, z, DATA_W, DATA_H);

	// Z

	idx_plus_1 = Calculate3DIndex(x, y, z + 1, DATA_W, DATA_H);
	idx_minus_1 = Calculate3DIndex(x, y, z - 1, DATA_W, DATA_H);

	total_complex_product_real = 0.0f;
	total_complex_product_imag = 0.0f;

	a = q13[idx_plus_1];
	c = q13[idx];
	
	total_complex_product_real += a.x * c.x + a.y * c.y;
	total_complex_product_imag += a.y * c.x - a.x * c.y;

	a.x = c.x;
	a.y = c.y;
	c = q13[idx_minus_1];

	total_complex_product_real += a.x * c.x + a.y * c.y;
	total_complex_product_imag += a.y * c.x - a.x * c.y;

	a = q23[idx_plus_1];
	c = q23[idx];

	total_complex_product_real += a.x * c.x + a.y * c.y;
	total_complex_product_imag += a.y * c.x - a.x * c.y;

	a.x = c.x;
	a.y = c.y;
	c = q23[idx_minus_1];

	total_complex_product_real += a.x * c.x + a.y * c.y;
	total_complex_product_imag += a.y * c.x - a.x * c.y;

	Phase_Gradients[idx] = atan2(total_complex_product_imag, total_complex_product_real);
}




__kernel void CalculateAMatrixAndHVector2DValuesX(__global float* A_matrix_2D_values, 
	                                              __global float* h_vector_2D_values, 
												  __global const float* Phase_Differences, 
												  __global const float* Phase_Gradients, 
												  __global const float* Phase_Certainties, 
												  __private int DATA_W, 
												  __private int DATA_H, 
												  __private int DATA_D, 
												  __private int FILTER_SIZE)
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



__kernel void CalculateAMatrixAndHVector2DValuesY(__global float* A_matrix_2D_values, 
	                                              __global float* h_vector_2D_values, 
												  __global const float* Phase_Differences, 
												  __global const float* Phase_Gradients, 
												  __global const float* Phase_Certainties, 
												  __private int DATA_W, 
												  __private int DATA_H, 
												  __private int DATA_D, 
												  __private int FILTER_SIZE)
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



__kernel void CalculateAMatrixAndHVector2DValuesZ(__global float* A_matrix_2D_values, 
	                                              __global float* h_vector_2D_values, 
												  __global const float* Phase_Differences, 
												  __global const float* Phase_Gradients, 
												  __global const float* Phase_Certainties, 
												  __private int DATA_W, 
												  __private int DATA_H, 
												  __private int DATA_D, 
												  __private int FILTER_SIZE)
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




__kernel void CalculateAMatrix1DValues(__global float* A_matrix_1D_values, 
	                                   __global const float* A_matrix_2D_values, 
									   __private int DATA_W, 
									   __private int DATA_H, 
									   __private int DATA_D, 
									   __private int FILTER_SIZE)
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


__kernel void CalculateAMatrix(__global float* A_matrix, 
	                           __global const float* A_matrix_1D_values, 
							   __private int DATA_W, 
							   __private int DATA_H, 
							   __private int DATA_D, 
							   __private int FILTER_SIZE)
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


__kernel void CalculateHVector1DValues(__global float* h_vector_1D_values, 
	                                   __global const float* h_vector_2D_values, 
									   __private int DATA_W, 
									   __private int DATA_H, 
									   __private int DATA_D, 
									   __private int FILTER_SIZE)
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



__kernel void CalculateHVector(__global float* h_vector, 
	                           __global const float* h_vector_1D_values, 
							   __private int DATA_W, 
							   __private int DATA_H, 
							   __private int DATA_D, 
							   __private int FILTER_SIZE)
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





__kernel void CalculateTensorComponents(__global float* t11,
										__global float* t12,
										__global float* t13,
										__global float* t22,
										__global float* t23,
										__global float* t33,
										__global const float2* q1,
										__global const float2* q2,
										__private float m11,
										__private float m12,
										__private float m13,
										__private float m22,
										__private float m23,
										__private float m33,
										__private int DATA_W,
										__private int DATA_H,
									    __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if ((x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D) )
		return;

	int idx = x + y * DATA_W + z * DATA_W * DATA_H;

	float2 q1_ = q1[idx];
	float2 q2_ = q2[idx];

	// Estimate structure tensor for the deformed volume
	float magnitude = sqrt(q2_.x * q2_.x + q2_.y * q2_.y);

	t11[idx] += magnitude * m11;
	t12[idx] += magnitude * m12;
	t13[idx] += magnitude * m13;
	t22[idx] += magnitude * m22;
	t23[idx] += magnitude * m23;
	t33[idx] += magnitude * m33;
}

__kernel void CalculateTensorNorms(__global float* Tensor_Norm, 
	                               __global const float* t11, 
								   __global const float* t12, 
								   __global const float* t13, 
								   __global const float* t22, 
								   __global const float* t23, 
								   __global const float* t33, 
								   __private int DATA_W, 
								   __private int DATA_H, 
								   __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if ((x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D) )
		return;

	int idx = x + y * DATA_W + z * DATA_W * DATA_H;
	
	float t11_ = t11[idx];
	float t12_ = t12[idx];
	float t13_ = t13[idx];
	float t22_ = t22[idx];
	float t23_ = t23[idx];
	float t33_ = t33[idx];	

	Tensor_Norm[idx] = sqrt(t11_*t11_ + 2.0f*t12_*t12_ + 2.0f*t13_*t13_ + t22_*t22_ + 2.0f*t23_*t23_ + t33_*t33_);	
}




__kernel void CalculateAMatricesAndHVectors(__global float* a11,
	                                        __global float* a12,
											__global float* a13,
											__global float* a22,
											__global float* a23,
											__global float* a33,
											__global float* h1,
											__global float* h2,
											__global float* h3,
											__global const float2* q1,
											__global const float2* q2,
											__global const float* t11,
											__global const float* t12,
											__global const float* t13,
											__global const float* t22,
											__global const float* t23,
											__global const float *t33,
											__constant float* c_Filter_Directions_X,
											__constant float* c_Filter_Directions_Y,
											__constant float* c_Filter_Directions_Z,
											__private int DATA_W,
											__private int DATA_H,
											__private int DATA_D,
											__private int FILTER)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if ((x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D) )
		return;

	int idx = x + y * DATA_W + z * DATA_W * DATA_H;

	float2 q1_ = q1[idx];
	float2 q2_ = q2[idx];

	// q1 * conj(q2)
	float qqReal = q1_.x * q2_.x + q1_.y * q2_.y;
	float qqImag = -q1_.x * q2_.y + q1_.y * q2_.x;
	float phase_difference = atan2(qqImag,qqReal);
	float Aqq = sqrt(qqReal * qqReal + qqImag * qqImag);
	float certainty = sqrt(Aqq) * cos(phase_difference/2.0f) * cos(phase_difference/2.0f);

	float tt11, tt12, tt13, tt22, tt23, tt33;

	tt11 = t11[idx] * t11[idx] + t12[idx] * t12[idx] + t13[idx] * t13[idx];
    tt12 = t11[idx] * t12[idx] + t12[idx] * t22[idx] + t13[idx] * t23[idx];
    tt13 = t11[idx] * t13[idx] + t12[idx] * t23[idx] + t13[idx] * t33[idx];
    tt22 = t12[idx] * t12[idx] + t22[idx] * t22[idx] + t23[idx] * t23[idx];
    tt23 = t12[idx] * t13[idx] + t22[idx] * t23[idx] + t23[idx] * t33[idx];
    tt33 = t13[idx] * t13[idx] + t23[idx] * t23[idx] + t33[idx] * t33[idx];

	a11[idx] += certainty * tt11;
	a12[idx] += certainty * tt12;
	a13[idx] += certainty * tt13;
	a22[idx] += certainty * tt22;
	a23[idx] += certainty * tt23;
	a33[idx] += certainty * tt33;

	h1[idx] += certainty * phase_difference * (c_Filter_Directions_X[FILTER] * tt11 + c_Filter_Directions_Y[FILTER] * tt12 + c_Filter_Directions_Z[FILTER] * tt13);
	h2[idx] += certainty * phase_difference * (c_Filter_Directions_X[FILTER] * tt12 + c_Filter_Directions_Y[FILTER] * tt22 + c_Filter_Directions_Z[FILTER] * tt23);
	h3[idx] += certainty * phase_difference * (c_Filter_Directions_X[FILTER] * tt13 + c_Filter_Directions_Y[FILTER] * tt23 + c_Filter_Directions_Z[FILTER] * tt33);
}




__kernel void CalculateDisplacementUpdate(__global float* DisplacementX,
	                                      __global float* DisplacementY,
	                                      __global float* DisplacementZ,
										  __global const float* a11,
										  __global const float* a12,
										  __global const float* a13,
										  __global const float* a22,
										  __global const float* a23,
										  __global const float* a33,
										  __global const float* h1,
										  __global const float* h2,
										  __global const float* h3,
										  __private int DATA_W,
										  __private int DATA_H,
										  __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if ((x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D) )
		return;

	int idx = x + y * DATA_W + z * DATA_W * DATA_H;
	
	float a11Temp = a11[idx];
	float a12Temp = a12[idx];
	float a13Temp = a13[idx];
	float a22Temp = a22[idx];
	float a23Temp = a23[idx];
	float a33Temp = a33[idx];
	float h1Temp = h1[idx];
	float h2Temp = h2[idx];
	float h3Temp = h3[idx];

	float norm = 1.0f / (a11Temp * a22Temp * a33Temp - a11Temp * a23Temp * a23Temp - a12Temp * a12Temp * a33Temp + a12Temp * a23Temp * a13Temp + a13Temp * a12Temp * a23Temp - a13Temp * a22Temp * a13Temp + 1E-16f);
		
	DisplacementX[idx] = norm * ((h3Temp * (a12Temp * a23Temp - a13Temp * a22Temp)) - (h2Temp * (a12Temp * a33Temp - a13Temp * a23Temp)) + (h1Temp * (a22Temp * a33Temp - a23Temp * a23Temp)));
	DisplacementY[idx] = norm * ((h2Temp * (a11Temp * a33Temp - a13Temp * a13Temp)) - (h3Temp * (a11Temp * a23Temp - a13Temp * a12Temp)) - (h1Temp * (a12Temp * a33Temp - a23Temp * a13Temp)));
	DisplacementZ[idx] = norm * ((h3Temp * (a11Temp * a22Temp - a12Temp * a12Temp)) - (h2Temp * (a11Temp * a23Temp - a12Temp * a13Temp)) + (h1Temp * (a12Temp * a23Temp - a22Temp * a13Temp)));
}



__constant sampler_t volume_sampler_nearest = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	

__kernel void InterpolateVolumeNearestParametric(__global float* Volume, 
	                                             read_only image3d_t Original_Volume, 
												 __constant float* c_Parameter_Vector, 
												 __private int DATA_W, 
												 __private int DATA_H, 
												 __private int DATA_D, 
												 __private int VOLUME)
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

__kernel void InterpolateVolumeNearestNonParametric(__global float* Volume, 
	                                                read_only image3d_t Original_Volume, 
													__global const float* d_Displacement_Field_X, 
													__global const float* d_Displacement_Field_Y, 
													__global const float* d_Displacement_Field_Z, 
													__private int DATA_W, 
													__private int DATA_H, 
													__private int DATA_D, 
													__private int VOLUME)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int idx = Calculate4DIndex(x,y,z,VOLUME,DATA_W,DATA_H,DATA_D);
	float4 Motion_Vector;
	
	Motion_Vector.x = (float)x + d_Displacement_Field_X[idx] + 0.5f;
	Motion_Vector.y = (float)y + d_Displacement_Field_Y[idx] + 0.5f;
	Motion_Vector.z = (float)z + d_Displacement_Field_Z[idx] + 0.5f;
	Motion_Vector.w = 0.0f;

	float4 Interpolated_Value = read_imagef(Original_Volume, volume_sampler_nearest, Motion_Vector);
	Volume[idx] = Interpolated_Value.x;
}

__constant sampler_t volume_sampler_linear = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
	

__kernel void InterpolateVolumeLinearParametric(__global float* Volume, 
	                                            read_only image3d_t Original_Volume, 
												__constant float* c_Parameter_Vector,
												__private int DATA_W,
												__private int DATA_H,
												__private int DATA_D,
												__private int VOLUME)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if ((x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D))
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

float  myabs(float value)
{
	if (value < 0.0f)
		return -value;
	else
		return value;
}

__kernel void InterpolateVolumeLinearNonParametric(__global float* Volume, 
	                                               read_only image3d_t Original_Volume, 
												   __global const float* d_Displacement_Field_X, 
												   __global const float* d_Displacement_Field_Y, 
												   __global const float* d_Displacement_Field_Z, 
												   __private int DATA_W, 
												   __private int DATA_H, 
												   __private int DATA_D, 
												   __private int VOLUME)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if ((x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D))
		return;

	int idx4D = Calculate4DIndex(x,y,z,VOLUME,DATA_W,DATA_H,DATA_D);
	int idx3D = Calculate3DIndex(x,y,z,DATA_W,DATA_H);

	float4 Motion_Vector;
	
	Motion_Vector.x = (float)x + d_Displacement_Field_X[idx3D] + 0.5f;
	Motion_Vector.y = (float)y + d_Displacement_Field_Y[idx3D] + 0.5f;
	Motion_Vector.z = (float)z + d_Displacement_Field_Z[idx3D] + 0.5f;
	Motion_Vector.w = 0.0f;

	float4 Interpolated_Value = read_imagef(Original_Volume, volume_sampler_linear, Motion_Vector);
	Volume[idx4D] = Interpolated_Value.x;
}

__kernel void AddParametricAndNonParametricDisplacement(__global float* d_Displacement_Field_X,
		   	   	   	   	   	   	   	   	   	   	   	    __global float* d_Displacement_Field_Y,
		   	   	   	   	   	   	   	   	   	   	   	    __global float* d_Displacement_Field_Z,
	                                            	 	__constant float* c_Parameter_Vector,
	                                            	 	__private int DATA_W,
	                                            	 	__private int DATA_H,
	                                            	 	__private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if ((x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D))
		return;

	int idx = Calculate3DIndex(x,y,z,DATA_W,DATA_H);
	float4 Motion_Vector;
	float xf, yf, zf;

    // (motion_vector.x)   (p0)   (p3  p4  p5)   (x)
	// (motion_vector.y) = (p1) + (p6  p7  p8) * (y)
 	// (motion_vector.z)   (p2)   (p9 p10 p11)   (z)

	// Change to coordinate system with origo in (sx - 1)/2 (sy - 1)/2 (sz - 1)/2
	xf = (float)x - ((float)DATA_W - 1.0f) * 0.5f;
	yf = (float)y - ((float)DATA_H - 1.0f) * 0.5f;
	zf = (float)z - ((float)DATA_D - 1.0f) * 0.5f;

	Motion_Vector.x = c_Parameter_Vector[0] + c_Parameter_Vector[3] * xf + c_Parameter_Vector[4]   * yf + c_Parameter_Vector[5]  * zf;
	Motion_Vector.y = c_Parameter_Vector[1] + c_Parameter_Vector[6] * xf + c_Parameter_Vector[7]   * yf + c_Parameter_Vector[8]  * zf;
	Motion_Vector.z = c_Parameter_Vector[2] + c_Parameter_Vector[9] * xf + c_Parameter_Vector[10]  * yf + c_Parameter_Vector[11] * zf;
	Motion_Vector.w = 0.0f;

	d_Displacement_Field_X[idx] += Motion_Vector.x;
	d_Displacement_Field_Y[idx] += Motion_Vector.y;
	d_Displacement_Field_Z[idx] += Motion_Vector.z;
}

float bspline(float t)
{
	t = fabs(t);
	const float a = 2.0f - t;

	if (t < 1.0f) return 2.0f/3.0f - 0.5f*t*t*a;
	else if (t < 2.0f) return a*a*a / 6.0f;
	else return 0.0f;
}



__kernel void InterpolateVolumeCubicParametric(__global float* Volume, 
	                                           read_only image3d_t Original_Volume, 
											   __constant float* c_Parameter_Vector, 
											   __private int DATA_W, 
											   __private int DATA_H, 
											   __private int DATA_D, 
											   __private int VOLUME)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int idx = Calculate4DIndex(x,y,z,VOLUME,DATA_W,DATA_H,DATA_D);
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
	

	const float3 coord_grid = Motion_Vector - 0.5f;
	float3 index = floor(coord_grid);
	const float3 fraction = coord_grid - index;
	index = index + 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

	float result;
	
	for (float z=-1.0f; z < 2.5f; z += 1.0f)  //range [-1, 2]
	{
		float bsplineZ = bspline(z-fraction.z);
		float w = index.z + z;
		for (float y=-1.0f; y < 2.5f; y += 1.0f)
		{
			float bsplineYZ = bspline(y-fraction.y) * bsplineZ;
			float v = index.y + y;
			for (float x=-1.0f; x < 2.5f; x += 1.0f)
			{
				float bsplineXYZ = bspline(x-fraction.x) * bsplineYZ;
				float u = index.x + x;
				float4 vector;
				vector.x = u;
				vector.y = v;
				vector.z = w;
				vector.w = 0.0f;
				float4 temp = read_imagef(Original_Volume, volume_sampler_linear, vector);
				result += temp.x * bsplineXYZ;
			}
		}
	}
	
	Volume[idx] = result;
}


__kernel void InterpolateVolumeCubicNonParametric(__global float* Volume, 
	                                              read_only image3d_t Original_Volume, 
												  __global const float* d_Displacement_Field, 
												  __private int DATA_W, 
												  __private int DATA_H, 
												  __private int DATA_D, 
												  __private int VOLUME)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int idx = Calculate4DIndex(x,y,z,VOLUME,DATA_W,DATA_H,DATA_D);
	float3 Motion_Vector;



	const float3 coord_grid = Motion_Vector - 0.5f;
	float3 index = floor(coord_grid);
	const float3 fraction = coord_grid - index;
	index = index + 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

	float result;
	
	for (float z=-1.0f; z < 2.5f; z += 1.0f)  //range [-1, 2]
	{
		float bsplineZ = bspline(z-fraction.z);
		float w = index.z + z;
		for (float y=-1.0f; y < 2.5f; y += 1.0f)
		{
			float bsplineYZ = bspline(y-fraction.y) * bsplineZ;
			float v = index.y + y;
			for (float x=-1.0f; x < 2.5f; x += 1.0f)
			{
				float bsplineXYZ = bspline(x-fraction.x) * bsplineYZ;
				float u = index.x + x;
				float4 vector;
				vector.x = u;
				vector.y = v;
				vector.z = w;
				vector.w = 0.0f;
				float4 temp = read_imagef(Original_Volume, volume_sampler_linear, vector);
				result += temp.x * bsplineXYZ;
			}
		}
	}
	
	Volume[idx] = result;
}

__kernel void RescaleVolumeLinear(__global float* Volume,
	                              read_only image3d_t Original_Volume,
								  __private float VOXEL_DIFFERENCE_X,
								  __private float VOXEL_DIFFERENCE_Y,
								  __private float VOXEL_DIFFERENCE_Z,
								  __private int DATA_W,
								  __private int DATA_H,
								  __private int DATA_D)
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

__kernel void RescaleVolumeCubic(__global float* Volume, 
	                             read_only image3d_t Original_Volume, 
								 __private float VOXEL_DIFFERENCE_X, 
								 __private float VOXEL_DIFFERENCE_Y, 
								 __private float VOXEL_DIFFERENCE_Z, 
								 __private int DATA_W, 
								 __private int DATA_H, 
								 __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int idx = Calculate3DIndex(x,y,z,DATA_W, DATA_H);
	float3 Motion_Vector;
	
	Motion_Vector.x = x * VOXEL_DIFFERENCE_X + 0.5f;
	Motion_Vector.y = y * VOXEL_DIFFERENCE_Y + 0.5f;
	Motion_Vector.z = z * VOXEL_DIFFERENCE_Z + 0.5f;
	
	const float3 coord_grid = Motion_Vector - 0.5f;
	float3 index = floor(coord_grid);
	const float3 fraction = coord_grid - index;
	index = index + 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

	float result;

	for (float z=-1.0f; z < 2.5f; z += 1.0f)  //range [-1, 2]
	{
		float bsplineZ = bspline(z-fraction.z);
		float w = index.z + z;
		for (float y=-1.0f; y < 2.5f; y += 1.0f)
		{
			float bsplineYZ = bspline(y-fraction.y) * bsplineZ;
			float v = index.y + y;
			for (float x=-1.0f; x < 2.5f; x += 1.0f)
			{
				float bsplineXYZ = bspline(x-fraction.x) * bsplineYZ;
				float u = index.x + x;
				float4 vector;
				vector.x = u;
				vector.y = v;
				vector.z = w;
				vector.w = 0.0f;
				float4 temp = read_imagef(Original_Volume, volume_sampler_linear, vector);
				result += bsplineXYZ * temp.x;
			}
		}
	}
	
	Volume[idx] = result;
}



__kernel void CalculateMagnitudes(__global float* Magnitudes,
	                              __global const float2* Complex,
								  __private int DATA_W, 
								  __private int DATA_H, 
								  __private int DATA_D)
{
	int x = get_global_id(0);	
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (y >= DATA_H || z >= DATA_D)
		return;

	float r = Complex[Calculate3DIndex(x,y,z,DATA_W,DATA_H)].x;
	float i = Complex[Calculate3DIndex(x,y,z,DATA_W,DATA_H)].y;
	Magnitudes[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = sqrt(r * r + i * i);
}

__kernel void CalculateColumnSums(__global float* Sums, 
	                              __global const float* Volume, 
								  __private int DATA_W, 
								  __private int DATA_H, 
								  __private int DATA_D)
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

__kernel void CalculateRowSums(__global float* Sums, 
	                           __global const float* Image, 
							   __private int DATA_H, 
							   __private int DATA_D)
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

float mymax(float a, float b)
{
	if (a > b)
		return a;
	else
		return b;
}

__kernel void CalculateColumnMaxs(__global float* Maxs, 
	                              __global const float* Volume, 
								  __private int DATA_W, 
								  __private int DATA_H, 
								  __private int DATA_D)
{
	int y = get_global_id(0);	
	int z = get_global_id(1);

	if (y >= DATA_H || z >= DATA_D)
		return;

	float max = -10000.0f;
	for (int x = 0; x < DATA_W; x++)
	{
		max = mymax(max, Volume[Calculate3DIndex(x,y,z,DATA_W,DATA_H)]);
	}

	Maxs[Calculate2DIndex(y,z,DATA_H)] = max;
}

__kernel void CalculateRowMaxs(__global float* Maxs, 
	                           __global const float* Image, 
							   __private int DATA_H, 
							   __private int DATA_D)
{
	int z = get_global_id(0);

	if (z >= DATA_D)
		return;

	float max = -10000.0f;
	for (int y = 0; y < DATA_H; y++)
	{
		max = mymax(max, Image[Calculate2DIndex(y,z,DATA_H)]);
	}

	Maxs[z] = max;
}

// Ugly quick solution, since atomic max does not work for floats
__kernel void CalculateMaxAtomic(volatile __global int* max_value,
	                             __global const float* Volume,
								 __global const float* Mask,
								 __private int DATA_W,
								 __private int DATA_H,
								 __private int DATA_D)
{
	int x = get_global_id(0);	
	int y = get_global_id(1);
	int z = get_global_id(2);

	if ( x >= DATA_W || y >= DATA_H || z >= DATA_D )
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
		return;

	int value = (int)(Volume[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] * 10000.0f);
	atomic_max(max_value, value);
}

__kernel void CopyT1VolumeToMNI(__global float* MNI_T1_Volume,
		                        __global float* Interpolated_T1_Volume,
		                        __private int MNI_DATA_W,
		                        __private int MNI_DATA_H,
		                        __private int MNI_DATA_D,
		                        __private int T1_DATA_W_INTERPOLATED,
		                        __private int T1_DATA_H_INTERPOLATED,
		                        __private int T1_DATA_D_INTERPOLATED,
		                        __private int x_diff,
		                        __private int y_diff,
		                        __private int z_diff,
		                        __private int MM_T1_Z_CUT,
		                        __private float MNI_VOXEL_SIZE_Z)
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


__kernel void CopyEPIVolumeToT1(__global float* T1_EPI_Volume,
		                        __global float* Interpolated_EPI_Volume,
		                        __private int T1_DATA_W,
		                        __private int T1_DATA_H,
		                        __private int T1_DATA_D,
		                        __private int EPI_DATA_W_INTERPOLATED,
		                        __private int EPI_DATA_H_INTERPOLATED,
		                        __private int EPI_DATA_D_INTERPOLATED,
		                        __private int x_diff,
		                        __private int y_diff,
		                        __private int z_diff,
		                        __private int MM_EPI_Z_CUT,
		                        __private float T1_VOXEL_SIZE_Z)
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


__kernel void CopyVolumeToNew(__global float* New_Volume,
		                      __global float* Interpolated_Volume,
		                      __private int NEW_DATA_W,
		                      __private int NEW_DATA_H,
		                      __private int NEW_DATA_D,
		                      __private int DATA_W_INTERPOLATED,
		                      __private int DATA_H_INTERPOLATED,
		                      __private int DATA_D_INTERPOLATED,
		                      __private int x_diff,
		                      __private int y_diff,
		                      __private int z_diff,
		                      __private int MM_Z_CUT,
		                      __private float NEW_VOXEL_SIZE_Z,
		                      __private int VOLUME)
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



float InterpolateCubic(float p0, float p1, float p2, float p3, float delta)
{
   float a0,a1,a2,a3,delta2;

   delta2 = delta * delta;
   a0 = p3 - p2 - p0 + p1;
   a1 = p0 - p1 - a0;
   a2 = p2 - p0;
   a3 = p1;

   return(a0 * delta * delta2 + a1 * delta2 + a2 * delta + a3);
}

__kernel void SliceTimingCorrection(__global float* Corrected_Volumes, 
                                    __global const float* Volumes, 									 
									__constant float* c_Slice_Differences, 									 
									__private int DATA_W, 
									__private int DATA_H, 
									__private int DATA_D, 
									__private int DATA_T)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	float delta = c_Slice_Differences[z];
	float t0, t1, t2, t3;

	// Forward interpolation
	if (delta > 0.0f)
	{
		t0 = Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
		t1 = t0;
		t2 = Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
		t3 = Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];

		// Loop over timepoints
		for (int t = 0; t < DATA_T - 3; t++)
		{
			// Cubic interpolation in time
			Corrected_Volumes[Calculate4DIndex(x,y,z,t,DATA_W,DATA_H,DATA_D)] = InterpolateCubic(t0,t1,t2,t3,delta); 
		
			// Shift old values backwards
			t0 = t1;
			t1 = t2;
			t2 = t3;

			// Read one new value
			t3 = Volumes[Calculate4DIndex(x,y,z,t+3,DATA_W,DATA_H,DATA_D)];
		}

		int t = DATA_T - 3;	
		Corrected_Volumes[Calculate4DIndex(x,y,z,t,DATA_W,DATA_H,DATA_D)] = InterpolateCubic(t0,t1,t2,t3,delta); 
	
		t = DATA_T - 2;
		t0 = t1;
		t1 = t2;
		t2 = t3;	
		Corrected_Volumes[Calculate4DIndex(x,y,z,t,DATA_W,DATA_H,DATA_D)] = InterpolateCubic(t0,t1,t2,t3,delta); 

		t = DATA_T - 1;
		t0 = t1;
		t1 = t2;
		Corrected_Volumes[Calculate4DIndex(x,y,z,t,DATA_W,DATA_H,DATA_D)] = InterpolateCubic(t0,t1,t2,t3,delta); 
	}
	// Backward interpolation
	else
	{
		delta = 1.0f - (-delta);

		t0 = Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
		t1 = t0;
		t2 = t0;
		t3 = Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];

		// Loop over timepoints
		for (int t = 0; t < DATA_T - 2; t++)
		{
			// Cubic interpolation in time
			Corrected_Volumes[Calculate4DIndex(x,y,z,t,DATA_W,DATA_H,DATA_D)] = InterpolateCubic(t0,t1,t2,t3,delta); 
		
			// Shift old values backwards
			t0 = t1;
			t1 = t2;
			t2 = t3;

			// Read one new value
			t3 = Volumes[Calculate4DIndex(x,y,z,t+2,DATA_W,DATA_H,DATA_D)];
		}

		int t = DATA_T - 2;	
		Corrected_Volumes[Calculate4DIndex(x,y,z,t,DATA_W,DATA_H,DATA_D)] = InterpolateCubic(t0,t1,t2,t3,delta); 
	
		t = DATA_T - 1;
		t0 = t1;
		t1 = t2;
		t2 = t3;	
		Corrected_Volumes[Calculate4DIndex(x,y,z,t,DATA_W,DATA_H,DATA_D)] = InterpolateCubic(t0,t1,t2,t3,delta); 
	}
}

// Statistical functions

// General function for calculating beta weights, all voxels use the same design matrix, not optimized for speed

__kernel void CalculateBetaWeightsGLM(__global float* Beta_Volumes, 
                                      __global const float* Volumes, 
									  __global const float* Mask, 
									  __constant float* c_xtxxt_GLM, 
									  __constant float* c_Censored_Timepoints,
									  __private int DATA_W, 
									  __private int DATA_H, 
									  __private int DATA_D, 
									  __private int NUMBER_OF_VOLUMES, 
									  __private int NUMBER_OF_REGRESSORS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
	{
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}
		return;
	}

	int t = 0;
	float beta[25];
	
	// Reset all beta values
	for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{
		beta[r] = 0.0f;
	}

	// Calculate betahat, i.e. multiply (x^T x)^(-1) x^T with Y
	// Loop over volumes
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		float temp = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] * c_Censored_Timepoints[v];

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

// Special function for calculating beta weights, all voxels use different design matrices (needed for Cochrane-Orcutt procedure)

__kernel void CalculateBetaWeightsGLMFirstLevel(__global float* Beta_Volumes, 
												__global const float* Volumes, 
												__global const float* Mask, 
												__global const float* d_xtxxt_GLM, 
												__global const float* d_Voxel_Numbers, 
												__constant float* c_Censored_Timepoints,
												__private int DATA_W, 
												__private int DATA_H, 
												__private int DATA_D, 
												__private int NUMBER_OF_VOLUMES, 
												__private int NUMBER_OF_REGRESSORS,
												__private int NUMBER_OF_INVALID_TIMEPOINTS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
	{
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}
		return;
	}

	int t = 0;
	float beta[25];
	
	// Reset all beta values
	for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{
		beta[r] = 0.0f;
	}

	// Get the specific voxel number for this brain voxel
	int voxel_number = (int)d_Voxel_Numbers[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];

	// Calculate betahat, i.e. multiply (x^T x)^(-1) x^T with Y
	// Loop over volumes
	for (int v = NUMBER_OF_INVALID_TIMEPOINTS; v < NUMBER_OF_VOLUMES; v++)
	{
		float temp = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];

		// Loop over regressors
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			beta[r] += temp * d_xtxxt_GLM[voxel_number * NUMBER_OF_VOLUMES * NUMBER_OF_REGRESSORS + NUMBER_OF_VOLUMES * r + v];
		}
	}

	// Save beta values
	for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{
		Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)] = beta[r];
	}
}

__kernel void CalculateGLMResiduals(__global float* Residuals,
		                            __global const float* Volumes,
		                            __global const float* Beta_Volumes,
		                            __global const float* Mask,
		                            __constant float *c_X_GLM,
		                            __private int DATA_W,
		                            __private int DATA_H,
		                            __private int DATA_D,
		                            __private int NUMBER_OF_VOLUMES,
		                            __private int NUMBER_OF_REGRESSORS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
	{
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			Residuals[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}

		return;
	}

	int t = 0;
	float eps, meaneps, vareps;
	float beta[25];

	// Load beta values into registers
    for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{
		beta[r] = Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)];
	}

	// Calculate the residual
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
		}

		Residuals[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = eps;
	}
}

__kernel void CalculateStatisticalMapsGLMTTestFirstLevel(__global float* Statistical_Maps,
		                                       	   	   	 __global float* Residuals,
		                                       	   	   	 __global float* Residual_Variances,
		                                       	   	   	 __global const float* Volumes,
		                                       	   	   	 __global const float* Beta_Volumes,
		                                       	   	   	 __global const float* Mask,
		                                       	   	   	 __global const float* d_X_GLM,
		                                       	   	   	 __global const float* d_GLM_Scalars,
		                                       	   	   	 __global const float* d_Voxel_Numbers,
		                                       	   	   	 __constant float* c_Contrasts,
		                                       	   	   	 __constant float* c_Censored_Timepoints,
		                                       	   	   	 __private int DATA_W,
		                                       	   	   	 __private int DATA_H,
		                                       	   	   	 __private int DATA_D,
		                                       	   	   	 __private int NUMBER_OF_VOLUMES,
		                                       	   	   	 __private int NUMBER_OF_REGRESSORS,
		                                       	   	   	 __private int NUMBER_OF_CONTRASTS,
		                                       	   	   	 __private int NUMBER_OF_CENSORED_TIMEPOINTS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
	{
		Residual_Variances[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;

		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}

		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			Residuals[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}

		return;
	}

	int t = 0;
	float eps, meaneps, vareps;
	float beta[25];

	// Load beta values into registers
    for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{
		beta[r] = Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)];
	}

    // Get the specific voxel number for this brain voxel
    int voxel_number = (int)d_Voxel_Numbers[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];

	// Calculate the mean of the error eps, using voxel-specific design models
	meaneps = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			eps -= d_X_GLM[voxel_number * NUMBER_OF_VOLUMES * NUMBER_OF_REGRESSORS + NUMBER_OF_VOLUMES * r + v] * beta[r];
			//eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
		}
		eps *= c_Censored_Timepoints[v];
		meaneps += eps;
		Residuals[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = eps;
	}
	//meaneps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_CENSORED_TIMEPOINTS);
	meaneps /= ((float)NUMBER_OF_VOLUMES);


	// Now calculate the variance of eps, using voxel-specific design models
	vareps = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			eps -= d_X_GLM[voxel_number * NUMBER_OF_VOLUMES * NUMBER_OF_REGRESSORS + NUMBER_OF_VOLUMES * r + v] * beta[r];
			//eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
		}
		vareps += (eps - meaneps) * (eps - meaneps) * c_Censored_Timepoints[v];
		//vareps += (eps - meaneps) * (eps - meaneps);
	}
	//vareps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_REGRESSORS - (float)NUMBER_OF_CENSORED_TIMEPOINTS - 1.0f);
	//vareps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_CENSORED_TIMEPOINTS - 1.0f);
	//vareps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_REGRESSORS - 1.0f);
	vareps /= ((float)NUMBER_OF_VOLUMES - 1.0f);
	Residual_Variances[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = vareps;

	// Loop over contrasts and calculate t-values, using a voxel-specific GLM scalar
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		float contrast_value = 0.0f;
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + r] * beta[r];
		}
		Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_value * rsqrt(vareps * d_GLM_Scalars[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)]);
	}
}

__kernel void CalculateStatisticalMapsGLMFTestFirstLevel(__global float* Statistical_Maps,
		                                       	   	   	 __global float* Residuals,
		                                       	   	   	 __global float* Residual_Variances,
		                                       	   	   	 __global const float* Volumes,
		                                       	   	   	 __global const float* Beta_Volumes,
		                                       	   	   	 __global const float* Mask,
		                                       	   	   	 __global const float* d_X_GLM,
		                                       	   	   	 __global const float* d_GLM_Scalars,
		                                       	   	   	 __global const float* d_Voxel_Numbers,
		                                       	   	   	 __constant float* c_Contrasts,
		                                       	   	   	 __constant float* c_Censored_Timepoints,
		                                       	   	   	 __private int DATA_W,
		                                       	   	   	 __private int DATA_H,
		                                       	   	   	 __private int DATA_D,
		                                       	   	   	 __private int NUMBER_OF_VOLUMES,
		                                       	   	   	 __private int NUMBER_OF_REGRESSORS,
		                                       	   	   	 __private int NUMBER_OF_CONTRASTS,
		                                       	   	   	 __private int NUMBER_OF_CENSORED_TIMEPOINTS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
	{
		Residual_Variances[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;

		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}

		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			Residuals[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}

		return;
	}

	int t = 0;
	float eps, meaneps, vareps;
	float beta[25];

	// Load beta values into registers
    for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{
		beta[r] = Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)];
	}

    // Get the specific voxel number for this brain voxel
    int voxel_number = (int)d_Voxel_Numbers[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];

	// Calculate the mean of the error eps, using voxel-specific design models
	meaneps = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			eps -= d_X_GLM[voxel_number * NUMBER_OF_VOLUMES * NUMBER_OF_REGRESSORS + NUMBER_OF_VOLUMES * r + v] * beta[r];
			//eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
		}
		eps *= c_Censored_Timepoints[v];
		meaneps += eps;
		Residuals[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = eps;
	}
	//meaneps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_CENSORED_TIMEPOINTS);
	meaneps /= ((float)NUMBER_OF_VOLUMES);


	// Now calculate the variance of eps, using voxel-specific design models
	vareps = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			eps -= d_X_GLM[voxel_number * NUMBER_OF_VOLUMES * NUMBER_OF_REGRESSORS + NUMBER_OF_VOLUMES * r + v] * beta[r];
			//eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
		}
		vareps += (eps - meaneps) * (eps - meaneps) * c_Censored_Timepoints[v];
	}
	//vareps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_REGRESSORS - (float)NUMBER_OF_CENSORED_TIMEPOINTS - 1.0f);
	//vareps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_CENSORED_TIMEPOINTS - 1.0f);
	//vareps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_REGRESSORS - 1.0f);
	vareps /= ((float)NUMBER_OF_VOLUMES - 1.0f);
	Residual_Variances[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = vareps;

	// Calculate matrix vector product C*beta (minus u)
	float cbeta[25];
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		cbeta[c] = 0.0f;
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + r] * beta[r];
		}
	}

	// Calculate total vector matrix vector product (C*beta)^T ( 1/vareps * (C^T (X^T X)^(-1) C^T)^(-1) ) (C*beta)

	// Calculate right hand side, temp = ( 1/vareps * (C^T (X^T X)^(-1) C^T)^(-1) ) (C*beta)
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		beta[c] = 0.0f;
		for (int cc = 0; cc < NUMBER_OF_CONTRASTS; cc++)
		{
			//beta[c] += 1.0f/vareps * c_ctxtxc_GLM[cc + c * NUMBER_OF_CONTRASTS] * cbeta[cc];
			beta[c] += 1.0f/vareps * d_GLM_Scalars[Calculate4DIndex(x,y,z,cc + c * NUMBER_OF_CONTRASTS,DATA_W,DATA_H,DATA_D)] * cbeta[cc];
		}
	}

	// Finally calculate (C*beta)^T * temp
	float scalar = 0.0f;
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		scalar += cbeta[c] * beta[c];
	}

	// Save F-value
	Statistical_Maps[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = scalar/(float)NUMBER_OF_CONTRASTS;
}

// Unoptimized kernel for calculating t-values, not a problem for regular first and second level analysis

__kernel void CalculateStatisticalMapsGLMTTest(__global float* Statistical_Maps,
		                                       __global float* Residuals,
		                                       __global float* Residual_Variances,
		                                       __global const float* Volumes,
		                                       __global const float* Beta_Volumes,
		                                       __global const float* Mask,
		                                       __constant float *c_X_GLM,
		                                       __constant float* c_Contrasts,
		                                       __constant float* c_ctxtxc_GLM,
											   __constant float* c_Censored_Timepoints,
		                                       __private int DATA_W,
		                                       __private int DATA_H,
		                                       __private int DATA_D,
		                                       __private int NUMBER_OF_VOLUMES,
		                                       __private int NUMBER_OF_REGRESSORS,
		                                       __private int NUMBER_OF_CONTRASTS,
											   __private int NUMBER_OF_CENSORED_TIMEPOINTS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
	{
		Residual_Variances[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;

		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}
	
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			Residuals[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}

		return;
	}

	int t = 0;
	float eps, meaneps, vareps;
	float beta[25];

	// Load beta values into registers
    for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{ 
		beta[r] = Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)];
	}

	// Calculate the mean of the error eps
	meaneps = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{ 
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
		}
		//eps *= c_Censored_Timepoints[v];
		meaneps += eps;
		Residuals[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = eps;
	}
	//meaneps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_CENSORED_TIMEPOINTS);
	meaneps /= ((float)NUMBER_OF_VOLUMES);


	// Now calculate the variance of eps
	vareps = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
		}
		//vareps += (eps - meaneps) * (eps - meaneps) * c_Censored_Timepoints[v];
		vareps += (eps - meaneps) * (eps - meaneps);
	}
	//vareps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_REGRESSORS - (float)NUMBER_OF_CENSORED_TIMEPOINTS - 1.0f);
	//vareps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_CENSORED_TIMEPOINTS - 1.0f);
	//vareps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_REGRESSORS - 1.0f);
	vareps /= ((float)NUMBER_OF_VOLUMES - 1.0f);
	Residual_Variances[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = vareps;
	
	// Loop over contrasts and calculate t-values
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		float contrast_value = 0.0f;
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + r] * beta[r];
		}
		Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_value * rsqrt(vareps * c_ctxtxc_GLM[c]);
	}
}

// Unoptimized kernel for calculating F-values, not a problem for regular first and second level analysis

__kernel void CalculateStatisticalMapsGLMFTest(__global float* Statistical_Maps,
		                                       __global float* Residuals,
		                                       __global float* Residual_Variances,
		                                       __global const float* Volumes,
		                                       __global const float* Beta_Volumes,
		                                       __global const float* Mask,
		                                       __constant float* c_X_GLM,
		                                       __constant float* c_Contrasts,
		                                       __constant float* c_ctxtxc_GLM,
		                                       __constant float* c_Censored_Timepoints,
		                                       __private int DATA_W,
		                                       __private int DATA_H,
		                                       __private int DATA_D,
		                                       __private int NUMBER_OF_VOLUMES,
		                                       __private int NUMBER_OF_REGRESSORS,
		                                       __private int NUMBER_OF_CONTRASTS,
		                                       __private int NUMBER_OF_CENSORED_TIMEPOINTS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
	{
		Residual_Variances[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;

		Statistical_Maps[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
		
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			Residuals[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}

		return;
	}

	int t = 0;
	float eps, meaneps, vareps;
	float beta[25];

	// Load beta values into registers
    for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{
		beta[r] = Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)];
	}

	// Calculate the mean of the error eps
	meaneps = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
		}
		meaneps += eps;
		Residuals[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = eps;
	}
	meaneps /= (float)NUMBER_OF_VOLUMES;

	// Now calculate the variance of eps
	vareps = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
		}
		vareps += (eps - meaneps) * (eps - meaneps);
	}
	vareps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_REGRESSORS - 1.0f); 
	Residual_Variances[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = vareps;

	//-------------------------

	// Calculate matrix vector product C*beta (minus u)
	float cbeta[25];
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		cbeta[c] = 0.0f;
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + r] * beta[r];
		}
	}

	// Calculate total vector matrix vector product (C*beta)^T ( 1/vareps * (C^T (X^T X)^(-1) C^T)^(-1) ) (C*beta)

	// Calculate right hand side, temp = ( 1/vareps * (C^T (X^T X)^(-1) C^T)^(-1) ) (C*beta)
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		beta[c] = 0.0f;
		for (int cc = 0; cc < NUMBER_OF_CONTRASTS; cc++)
		{
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[cc + c * NUMBER_OF_CONTRASTS] * cbeta[cc];
		}
	}

	// Finally calculate (C*beta)^T * temp
	float scalar = 0.0f;
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		scalar += cbeta[c] * beta[c];
	}

	// Save F-value
	Statistical_Maps[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = scalar/(float)NUMBER_OF_CONTRASTS;
}
	
#pragma OPENCL EXTENSION cl_khr_fp64: enable

// Generate random uniform number by modulo operation


double unirand(__private int* seed)
{
	double const a = 16807.0; //ie 7**5
	double const m = 2147483647.0; //ie 2**31-1
	double const reciprocal_m = 1.0/m;
	double temp = (*seed) * a;
	*seed = (int)(temp - m * floor(temp * reciprocal_m));

	return ((double)(*seed) * reciprocal_m);
}

#define pi 3.141592653589793

// Generate random normal number by Box-Muller transform
double normalrand(__private int* seed)
{
	double u = unirand(seed);
	double v = unirand(seed);

	return sqrt(-2.0*log(u))*cos(2.0*pi*v);
}

// Generate inverse Gamma number
double gamrnd(float a, float b, __private int* seed)
{
	double x = 0.0;
	for (int i = 0; i < 2*(int)round(a); i++)
	{
		double rand_value = normalrand(seed);
		x += rand_value * rand_value;
	}

	return 2.0 * b / x;
}


/*
float unirand(__private int* seed)
{
	float const a = 16807.0f; //ie 7**5
	float const m = 2147483647.0f; //ie 2**31-1
	float const reciprocal_m = 1.0f/m;
	float temp = (*seed) * a;
	*seed = (int)(temp - m * floor(temp * reciprocal_m));

	return ((float)(*seed) * reciprocal_m);
}

#define pi 3.141592653589793

// Generate random normal number by Box-Muller transform
float normalrand(__private int* seed)
{
	float u = unirand(seed);
	float v = unirand(seed);

	return sqrt(-2.0f*log(u))*cos(2.0f*pi*v);
}

// Generate inverse Gamma number
float gamrnd(float a, float b, __private int* seed)
{
	float x = 0.0f;
	for (int i = 0; i < 2*(int)round(a); i++)
	{
		float rand_value = normalrand(seed);
		x += rand_value * rand_value;
	}

	return 2.0f * b / x;
}
*/

// Cholesky factorization, not optimized
int Cholesky(float* cholA, float factor, __constant float* A, int N)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{   
			cholA[i + j*N] = 0.0f;

			if (i == j)
			{
				float value = 0.0f;
				for (int k = 0; k <= (j-1); k++)
				{
					value += cholA[k + j*N] * cholA[k + j*N];
				}
				cholA[j + j*N] = sqrt(factor*A[j + j*N] - value);
			}
			else if (i > j)
			{
				float value = 0.0f;
				for (int k = 0; k <= (j-1); k++)
				{
					value += cholA[k + i*N] * cholA[k + j*N]; 
				}
				cholA[j + i*N] = 1/cholA[j + j*N] * (factor*A[j + i*N] - value);
			}
		}
	}

	return 0;
}

int MultivariateRandomOld(float* random, float* mu, __constant float* Cov, float Sigma, int N, __private int* seed)
{
	float randvalues[2];
	float cholCov[4];
			
	switch(N)
	{
		case 2:
			
			randvalues[0] = normalrand(seed);
			randvalues[1] = normalrand(seed);
	
			Cholesky(cholCov, Sigma, Cov, N);

			random[0] = mu[0] + cholCov[0 + 0 * N] * randvalues[0] + cholCov[1 + 0 * N] * randvalues[1];
			random[1] = mu[1] + cholCov[0 + 1 * N] * randvalues[0] + cholCov[1 + 1 * N] * randvalues[1];
		
			break;

		case 3:

		
			break;

		case 4:

		
			break;

		default:
			1;
			break;
	}

	return 0;
}

int Cholesky1(float* cholA, float factor, float A)
{
	*cholA = sqrt(factor * A);

	return 0;
}

int Cholesky2(float cholA[2][2], float factor, float A[2][2])
{
	// i = 0, j = 0
	cholA[0][0] = sqrt(factor * A[0][0]);

	// i = 1, j = 0
	cholA[1][0] = 1.0f / cholA[0][0] * (factor * A[1][0]);

	// i = 0, j = 1
	cholA[0][1] = 0.0f;

	// i = 1, j = 1
	cholA[1][1] = sqrt(factor * A[1][1] - cholA[0][1] * cholA[0][1] - cholA[1][0] * cholA[1][0] );

	return 0;
}



int MultivariateRandom1(float* random, float mu, __private float Cov, float Sigma, __private int* seed)
{
	float randvalues;
	float cholCov;
			
	randvalues = normalrand(seed);		
	Cholesky1(&cholCov, Sigma, Cov);
	random[0] = mu + cholCov * randvalues;

	return 0;
}

int MultivariateRandom2(float* random, float* mu, __private float Cov[2][2], float Sigma, __private int* seed)
{
	float randvalues[2];
	float cholCov[2][2];
			
	randvalues[0] = normalrand(seed);
	randvalues[1] = normalrand(seed);
	
	Cholesky2(cholCov, Sigma, Cov);

	random[0] = mu[0] + cholCov[0][0] * randvalues[0] + cholCov[0][1] * randvalues[1];
	random[1] = mu[1] + cholCov[0][1] * randvalues[0] + cholCov[1][1] * randvalues[1];
	
	return 0;
}

int CalculateBetaWeightsBayesian(__private float* beta,
								 __private float value,
								 __constant float* c_X_GLM,
								 int v,
								 int NUMBER_OF_VOLUMES,
								 int NUMBER_OF_REGRESSORS)
{
	switch(NUMBER_OF_REGRESSORS)
	{
		case 1:

			beta[0] += value * c_X_GLM[NUMBER_OF_VOLUMES * 0 + v];

			break;

		case 2:

			beta[0] += value * c_X_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_X_GLM[NUMBER_OF_VOLUMES * 1 + v];

			break;

		case 3:

			beta[0] += value * c_X_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_X_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_X_GLM[NUMBER_OF_VOLUMES * 2 + v];

			break;

		case 4:

			beta[0] += value * c_X_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_X_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_X_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_X_GLM[NUMBER_OF_VOLUMES * 3 + v];

			break;

		default:
			1;
			break;
	}

	return 0;
}



float Determinant_4x4(float Cxx[4][4])
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
	float determinant = Determinant_4x4(Cxx) + 0.001f;

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

float Determinant_2x2(float Cxx[2][2])
{
    return Cxx[0][0] * Cxx[1][1] - Cxx[0][1] * Cxx[1][0];
}

void Invert_2x2(float Cxx[2][2], float inv_Cxx[2][2])
{
	float determinant = Determinant_2x2(Cxx) + 0.001f;

	inv_Cxx[0][0] = Cxx[1][1] / determinant;
	inv_Cxx[0][1] = -Cxx[0][1] / determinant;
	inv_Cxx[1][0] = -Cxx[1][0] / determinant;
	inv_Cxx[1][1] = Cxx[0][0] / determinant;
}

// Generates a posterior probability map (PPM) using Gibbs sampling

__kernel void CalculateStatisticalMapsGLMBayesian(__global float* Statistical_Maps,
		                                          __global const float* Volumes,
		                                          __global const float* Mask,
		                                          __global const int* Seeds,
		                                          __constant float* c_X_GLM,
		                                          __constant float* c_InvOmega0,
											      __constant float* c_S00,
											      __constant float* c_S01,
											      __constant float* c_S11,
		                                          __private int DATA_W,
		                                          __private int DATA_H,
		                                          __private int DATA_D,
		                                          __private int NUMBER_OF_VOLUMES,
		                                          __private int NUMBER_OF_REGRESSORS,
											      __private int NUMBER_OF_ITERATIONS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int seed = Calculate3DIndex(x,y,z,DATA_W,DATA_H) * 1000;


	Statistical_Maps[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)] = (float)unirand(&seed);
	Statistical_Maps[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)] = (float)normalrand(&seed);

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
	{
		Statistical_Maps[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)] = 0.0f;

		/*
		Statistical_Maps[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)] = 0.0f;
		*/

		return;
	}

	// Get seed from host
	//int seed = Seeds[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];

	// Prior options
	float iota = 1.0f;                 // Decay factor for lag length in prior for rho.
	float r = 0.5f;                    // Prior mean on rho1
	float c = 0.3f;                    // Prior standard deviation on first lag.
	float a0 = 0.01f;                  // First parameter in IG prior for sigma^2
	float b0 = 0.01f;                  // Second parameter in IG prior for sigma^2

	float InvA0 = c * c;

	// Algorithmic options
	float prcBurnin = 10.0f;             // Percentage of nIter used for burnin. Note: effective number of iter is nIter.

	float beta[2];
	float betaT[2];

	int nBurnin = (int)round((float)NUMBER_OF_ITERATIONS*(prcBurnin/100.0f));
	int probability = 0;
	
	float m00[2];
	float m01[2];
	float m10[2];
	float m11[2];

	m00[0] = 0.0f;
	m00[1] = 0.0f;

	m01[0] = 0.0f;
	m01[1] = 0.0f;

	m10[0] = 0.0f;
	m10[1] = 0.0f;

	m11[0] = 0.0f;
	m11[1] = 0.0f;

	float g00 = 0.0f;
	float g01 = 0.0f;
	float g11 = 0.0f;

	float old_value = Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];

	m00[0] += c_X_GLM[NUMBER_OF_VOLUMES * 0 + 0] * old_value;
	m00[1] += c_X_GLM[NUMBER_OF_VOLUMES * 1 + 0] * old_value;

	g00 += old_value * old_value;

	for (int v = 1; v < NUMBER_OF_VOLUMES; v++)
	{
		float value = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];

		m00[0] += c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * value;
		m00[1] += c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * value;

		m01[0] += c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * old_value;
		m01[1] += c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * old_value;

		m10[0] += c_X_GLM[NUMBER_OF_VOLUMES * 0 + (v - 1)] * value;
		m10[1] += c_X_GLM[NUMBER_OF_VOLUMES * 1 + (v - 1)] * value;

		m11[0] += c_X_GLM[NUMBER_OF_VOLUMES * 0 + (v - 1)] * old_value;
		m11[1] += c_X_GLM[NUMBER_OF_VOLUMES * 1 + (v - 1)] * old_value;

		g00 += value * value;
		g01 += value * old_value;
		g11 += old_value * old_value;

		old_value = value;
	}
	
	float InvOmegaT[2][2];
	float OmegaT[2][2];
	float Xtildesquared[2][2];
	float XtildeYtilde[2];
	float Ytildesquared;

	Xtildesquared[0][0] = c_S00[0 + 0*2];
	Xtildesquared[0][1] = c_S00[0 + 1*2];
	Xtildesquared[1][0] = c_S00[1 + 0*2];
	Xtildesquared[1][1] = c_S00[1 + 1*2];
		
	XtildeYtilde[0] = m00[0];
	XtildeYtilde[1] = m00[1];

	Ytildesquared = g00;

	float sigma2;
	float rho, rhoT, rhoProp, bT;

	rho = 0.0f;

	// Loop over iterations
	for (int i = 0; i < (nBurnin + NUMBER_OF_ITERATIONS); i++)
	//for (int i = 0; i < 100; i++)
	{
		InvOmegaT[0][0] = c_InvOmega0[0 + 0 * NUMBER_OF_REGRESSORS] + Xtildesquared[0][0];
		InvOmegaT[0][1] = c_InvOmega0[0 + 1 * NUMBER_OF_REGRESSORS] + Xtildesquared[0][1];
		InvOmegaT[1][0] = c_InvOmega0[1 + 0 * NUMBER_OF_REGRESSORS] + Xtildesquared[1][0];
		InvOmegaT[1][1] = c_InvOmega0[1 + 1 * NUMBER_OF_REGRESSORS] + Xtildesquared[1][1];
		Invert_2x2(InvOmegaT, OmegaT);

		betaT[0] = OmegaT[0][0] * XtildeYtilde[0] + OmegaT[0][1] * XtildeYtilde[1];
		betaT[1] = OmegaT[1][0] * XtildeYtilde[0] + OmegaT[1][1] * XtildeYtilde[1];

		float aT = a0 + (float)NUMBER_OF_VOLUMES/2.0f;
		float temp[2];
		temp[0] = InvOmegaT[0][0] * betaT[0] + InvOmegaT[0][1] * betaT[1];
		temp[1] = InvOmegaT[1][0] * betaT[0] + InvOmegaT[1][1] * betaT[1];
		bT = b0 + 0.5f * (Ytildesquared - betaT[0] * temp[0] - betaT[1] * temp[1]);

		// Block 1 - Step 1a. Update sigma2
		sigma2 = gamrnd(aT,bT,&seed);
		
		// Block 1 - Step 1b. Update beta | sigma2
		MultivariateRandom2(beta,betaT,OmegaT,sigma2,&seed);
		
		if (i > nBurnin)
		{
			if (beta[0] > 0.0f)
			{
				probability++;
			}
		}  

		// Block 2, update rho
		float zsquared = 0.0f;
		float zu = 0.0f;
		float old_eps = 0.0f;

		// Calculate residuals
		for (int v = 1; v < NUMBER_OF_VOLUMES; v++)
		{
			float eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];

			zsquared += eps * eps;
			zu += eps * old_eps;

			old_eps = eps;
		}

		// Generate rho
		float InvAT = InvA0 + zsquared / sigma2;
		float AT = 1.0f / InvAT;
		rhoT = AT * zu / sigma2;
		MultivariateRandom1(&rhoProp,rhoT,AT,sigma2,&seed);

		if (myabs(rhoProp) < 1.0f)
		{
			rho = rhoProp;
		}

		// Prewhitening of regressors and data
		Xtildesquared[0][0] = c_S00[0 + 0*2] - 2.0f * rho * c_S01[0 + 0*2] + rho * rho * c_S11[0 + 0*2];
		Xtildesquared[0][1] = c_S00[0 + 1*2] - 2.0f * rho * c_S01[0 + 1*2] + rho * rho * c_S11[0 + 1*2];
		Xtildesquared[1][0] = c_S00[1 + 0*2] - 2.0f * rho * c_S01[1 + 0*2] + rho * rho * c_S11[1 + 0*2];
		Xtildesquared[1][1] = c_S00[1 + 1*2] - 2.0f * rho * c_S01[1 + 1*2] + rho * rho * c_S11[1 + 1*2];
		
		XtildeYtilde[0] = m00[0] - rho * (m01[0] + m10[0]) + rho * rho * m11[0];
		XtildeYtilde[1] = m00[1] - rho * (m01[1] + m10[1]) + rho * rho * m11[1];

		Ytildesquared = g00 - 2.0f * rho * g01 + rho * rho * g11;
	}
	
	Statistical_Maps[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)] = (float)probability/(float)NUMBER_OF_ITERATIONS;
	Statistical_Maps[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)] = rhoT;

	/*
	float Sigma = 1.0f;
	float Cov[2][2];
	float cholCov[2][2];

	Cov[0][0] = 1.0f;
	Cov[0][1] = 2.0f	;
	Cov[1][0] = 2.0f;
	Cov[1][1] = 15.0f;

	Cholesky2(cholCov, Sigma, Cov);

	Statistical_Maps[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)] = cholCov[0][0];
	Statistical_Maps[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)] = cholCov[0][1];
	Statistical_Maps[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)] = cholCov[1][0];
	Statistical_Maps[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)] = cholCov[1][1];
	*/

	/*
	Statistical_Maps[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)] = beta[0];
	Statistical_Maps[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)] = beta[1];
	Statistical_Maps[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)] = betaT[0];
	Statistical_Maps[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)] = betaT[1];
	
	Statistical_Maps[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)] = rhoT;
	Statistical_Maps[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)] = rho;
	Statistical_Maps[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)] = bT;
	Statistical_Maps[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)] = InvOmegaT[0][1];
	Statistical_Maps[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)] = InvOmegaT[1][0];
	Statistical_Maps[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)] = InvOmegaT[1][1];
	Statistical_Maps[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)] = OmegaT[0][0];
	Statistical_Maps[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)] = OmegaT[0][1];
	Statistical_Maps[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)] = OmegaT[1][0];
	Statistical_Maps[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)] = (float)probability/(float)NUMBER_OF_ITERATIONS;
	*/
}


// Functions for permutation test

int LoadBetaWeights(__private float* beta, 
	                __global const float* Beta_Volumes, 
					int x, 
					int y, 
					int z, 
					int DATA_W, 
					int DATA_H, 
					int DATA_D, 
					int NUMBER_OF_REGRESSORS)
{
	switch(NUMBER_OF_REGRESSORS)
	{
		case 1:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			
			break;

		case 2:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];

			break;

		case 3:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];

			break;

		case 4:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];

			break;

		case 5:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];

			break;

		case 6:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];

			break;

		case 7:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];			

			break;

		case 8:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];

			break;

		case 9:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];

			break;

		case 10:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
			beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];

			break;

		case 11:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
			beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
			beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];

			break;

		case 12:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
			beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
			beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
			beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];

			break;

		case 13:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
			beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
			beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
			beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
			beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
			
			break;

		case 14:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
			beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
			beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
			beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
			beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
			beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
			
			break;

		case 15:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
			beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
			beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
			beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
			beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
			beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
			beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
			
			break;

		case 16:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
			beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
			beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
			beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
			beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
			beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
			beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
			beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
			
			break;

		case 17:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
			beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
			beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
			beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
			beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
			beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
			beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
			beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
			beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
			
			break;

		case 18:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
			beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
			beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
			beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
			beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
			beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
			beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
			beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
			beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
			beta[17] = Beta_Volumes[Calculate4DIndex(x,y,z,17,DATA_W,DATA_H,DATA_D)];
			
			break;

		case 19:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
			beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
			beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
			beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
			beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
			beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
			beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
			beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
			beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
			beta[17] = Beta_Volumes[Calculate4DIndex(x,y,z,17,DATA_W,DATA_H,DATA_D)];
			beta[18] = Beta_Volumes[Calculate4DIndex(x,y,z,18,DATA_W,DATA_H,DATA_D)];			

			break;

		case 20:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
			beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
			beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
			beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
			beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
			beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
			beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
			beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
			beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
			beta[17] = Beta_Volumes[Calculate4DIndex(x,y,z,17,DATA_W,DATA_H,DATA_D)];
			beta[18] = Beta_Volumes[Calculate4DIndex(x,y,z,18,DATA_W,DATA_H,DATA_D)];
			beta[19] = Beta_Volumes[Calculate4DIndex(x,y,z,19,DATA_W,DATA_H,DATA_D)];
			
			break;

		case 21:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
			beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
			beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
			beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
			beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
			beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
			beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
			beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
			beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
			beta[17] = Beta_Volumes[Calculate4DIndex(x,y,z,17,DATA_W,DATA_H,DATA_D)];
			beta[18] = Beta_Volumes[Calculate4DIndex(x,y,z,18,DATA_W,DATA_H,DATA_D)];
			beta[19] = Beta_Volumes[Calculate4DIndex(x,y,z,19,DATA_W,DATA_H,DATA_D)];
			beta[20] = Beta_Volumes[Calculate4DIndex(x,y,z,20,DATA_W,DATA_H,DATA_D)];
			
			break;

		case 22:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
			beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
			beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
			beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
			beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
			beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
			beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
			beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
			beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
			beta[17] = Beta_Volumes[Calculate4DIndex(x,y,z,17,DATA_W,DATA_H,DATA_D)];
			beta[18] = Beta_Volumes[Calculate4DIndex(x,y,z,18,DATA_W,DATA_H,DATA_D)];
			beta[19] = Beta_Volumes[Calculate4DIndex(x,y,z,19,DATA_W,DATA_H,DATA_D)];
			beta[20] = Beta_Volumes[Calculate4DIndex(x,y,z,20,DATA_W,DATA_H,DATA_D)];
			beta[21] = Beta_Volumes[Calculate4DIndex(x,y,z,21,DATA_W,DATA_H,DATA_D)];
			
			break;

		case 23:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
			beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
			beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
			beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
			beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
			beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
			beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
			beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
			beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
			beta[17] = Beta_Volumes[Calculate4DIndex(x,y,z,17,DATA_W,DATA_H,DATA_D)];
			beta[18] = Beta_Volumes[Calculate4DIndex(x,y,z,18,DATA_W,DATA_H,DATA_D)];
			beta[19] = Beta_Volumes[Calculate4DIndex(x,y,z,19,DATA_W,DATA_H,DATA_D)];
			beta[20] = Beta_Volumes[Calculate4DIndex(x,y,z,20,DATA_W,DATA_H,DATA_D)];
			beta[21] = Beta_Volumes[Calculate4DIndex(x,y,z,21,DATA_W,DATA_H,DATA_D)];
			beta[22] = Beta_Volumes[Calculate4DIndex(x,y,z,22,DATA_W,DATA_H,DATA_D)];

			break;

		case 24:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
			beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
			beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
			beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
			beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
			beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
			beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
			beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
			beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
			beta[17] = Beta_Volumes[Calculate4DIndex(x,y,z,17,DATA_W,DATA_H,DATA_D)];
			beta[18] = Beta_Volumes[Calculate4DIndex(x,y,z,18,DATA_W,DATA_H,DATA_D)];
			beta[19] = Beta_Volumes[Calculate4DIndex(x,y,z,19,DATA_W,DATA_H,DATA_D)];
			beta[20] = Beta_Volumes[Calculate4DIndex(x,y,z,20,DATA_W,DATA_H,DATA_D)];
			beta[21] = Beta_Volumes[Calculate4DIndex(x,y,z,21,DATA_W,DATA_H,DATA_D)];
			beta[22] = Beta_Volumes[Calculate4DIndex(x,y,z,22,DATA_W,DATA_H,DATA_D)];
			beta[23] = Beta_Volumes[Calculate4DIndex(x,y,z,23,DATA_W,DATA_H,DATA_D)];
			
			break;

		case 25:

			beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
			beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
			beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
			beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
			beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
			beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
			beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
			beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
			beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
			beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
			beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
			beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
			beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
			beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
			beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
			beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
			beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
			beta[17] = Beta_Volumes[Calculate4DIndex(x,y,z,17,DATA_W,DATA_H,DATA_D)];
			beta[18] = Beta_Volumes[Calculate4DIndex(x,y,z,18,DATA_W,DATA_H,DATA_D)];
			beta[19] = Beta_Volumes[Calculate4DIndex(x,y,z,19,DATA_W,DATA_H,DATA_D)];
			beta[20] = Beta_Volumes[Calculate4DIndex(x,y,z,20,DATA_W,DATA_H,DATA_D)];
			beta[21] = Beta_Volumes[Calculate4DIndex(x,y,z,21,DATA_W,DATA_H,DATA_D)];
			beta[22] = Beta_Volumes[Calculate4DIndex(x,y,z,22,DATA_W,DATA_H,DATA_D)];
			beta[23] = Beta_Volumes[Calculate4DIndex(x,y,z,23,DATA_W,DATA_H,DATA_D)];
			beta[24] = Beta_Volumes[Calculate4DIndex(x,y,z,24,DATA_W,DATA_H,DATA_D)];
			
			break;


		default:
			1;
			break;
	}

	return 0;
}

int CalculateBetaWeightsFirstLevel(__private float* beta,
		                 	       __private float value,
		                 	 	   __constant float* c_xtxxt_GLM,
		                 	 	   int v,
		                 	 	   int NUMBER_OF_VOLUMES,
		                 	 	   int NUMBER_OF_REGRESSORS)
{
	switch(NUMBER_OF_REGRESSORS)
	{
		case 1:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];

			break;

		case 2:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];

			break;

		case 3:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];

			break;

		case 4:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];

			break;

		case 5:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];

			break;

		case 6:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];

			break;

		case 7:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];

			break;

		case 8:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];

			break;

		case 9:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];

			break;

		case 10:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];

			break;

		case 11:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];

			break;

		case 12:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];

			break;

		case 13:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
			
			break;

		case 14:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
			
			break;

		case 15:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
			
			break;

		case 16:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
			
			break;

		case 17:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
			
			break;

		case 18:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
			beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + v];
			
			break;

		case 19:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
			beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + v];
			beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + v];

			break;

		case 20:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
			beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + v];
			beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + v];
			beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + v];
			
			break;

		case 21:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
			beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + v];
			beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + v];
			beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + v];
			beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + v];
			
			break;

		case 22:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
			beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + v];
			beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + v];
			beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + v];
			beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + v];
			beta[21] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 21 + v];

			break;

		case 23:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
			beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + v];
			beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + v];
			beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + v];
			beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + v];
			beta[21] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 21 + v];
			beta[22] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 22 + v];

			break;

		case 24:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
			beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + v];
			beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + v];
			beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + v];
			beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + v];
			beta[21] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 21 + v];
			beta[22] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 22 + v];
			beta[23] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 23 + v];
			
			break;

		case 25:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
			beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + v];
			beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + v];
			beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + v];
			beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + v];
			beta[21] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 21 + v];
			beta[22] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 22 + v];
			beta[23] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 23 + v];
			beta[24] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 24 + v];
			
			break;


		default:
			1;
			break;
	}

	return 0;
}


int CalculateBetaWeightsSecondLevel(__private float* beta,
		                 	 	    __private float value,
		                 	 	    __constant float* c_xtxxt_GLM,
		                 	 	    int v,
		                 	 	    __constant unsigned short int* c_Permutation_Vector,
		                 	 	    int NUMBER_OF_VOLUMES,
		                 	 	    int NUMBER_OF_REGRESSORS)
{
	switch(NUMBER_OF_REGRESSORS)
	{
		case 1:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];

			break;

		case 2:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];

			break;

		case 3:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];

			break;

		case 4:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];

			break;

		case 5:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];

			break;

		case 6:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];

			break;

		case 7:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];

			break;

		case 8:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];

			break;

		case 9:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];

			break;

		case 10:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];

			break;

		case 11:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];

			break;

		case 12:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];

			break;

		case 13:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];

			break;

		case 14:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];

			break;

		case 15:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];

			break;

		case 16:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];

			break;

		case 17:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];

			break;

		case 18:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
			beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]];

			break;

		case 19:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
			beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]];
			beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]];

			break;

		case 20:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
			beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]];
			beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]];
			beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]];

			break;

		case 21:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
			beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]];
			beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]];
			beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]];
			beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]];

			break;

		case 22:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
			beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]];
			beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]];
			beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]];
			beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]];
			beta[21] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 21 + c_Permutation_Vector[v]];

			break;

		case 23:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
			beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]];
			beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]];
			beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]];
			beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]];
			beta[21] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 21 + c_Permutation_Vector[v]];
			beta[22] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 22 + c_Permutation_Vector[v]];

			break;

		case 24:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
			beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]];
			beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]];
			beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]];
			beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]];
			beta[21] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 21 + c_Permutation_Vector[v]];
			beta[22] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 22 + c_Permutation_Vector[v]];
			beta[23] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 23 + c_Permutation_Vector[v]];

			break;

		case 25:

			beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
			beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
			beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
			beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
			beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
			beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
			beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
			beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
			beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
			beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
			beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
			beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
			beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
			beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
			beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
			beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
			beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
			beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]];
			beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]];
			beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]];
			beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]];
			beta[21] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 21 + c_Permutation_Vector[v]];
			beta[22] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 22 + c_Permutation_Vector[v]];
			beta[23] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 23 + c_Permutation_Vector[v]];
			beta[24] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 24 + c_Permutation_Vector[v]];

			break;


		default:
			1;
			break;
	}

	return 0;
}





// For first level, volumes already permuted
float CalculateEpsFirstLevel(__private float eps,
							 __private float* beta,
							 __constant float* c_X_GLM,
							 int v,		           
							 int NUMBER_OF_VOLUMES,
							 int NUMBER_OF_REGRESSORS)
{
	switch(NUMBER_OF_REGRESSORS)
	{
		case 1:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];

			break;

		case 2:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];

			break;

		case 3:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];

			break;

		case 4:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];

			break;

		case 5:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];

			break;

		case 6:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];

			break;

		case 7:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];

			break;

		case 8:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];

			break;

		case 9:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];

			break;

		case 10:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];			

			break;

		case 11:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
			
			break;

		case 12:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
			
			break;

		case 13:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
			
			break;

		case 14:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
			
			break;

		case 15:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
			
			break;

		case 16:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
			
			break;

		case 17:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
			
			break;

		case 18:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + v] * beta[17];
			
			break;

		case 19:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + v] * beta[17];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + v] * beta[18];
			
			break;

		case 20:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + v] * beta[17];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + v] * beta[18];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + v] * beta[19];
			
			break;

		case 21:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + v] * beta[17];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + v] * beta[18];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + v] * beta[19];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + v] * beta[20];
			
			break;

		case 22:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + v] * beta[17];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + v] * beta[18];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + v] * beta[19];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + v] * beta[20];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 21 + v] * beta[21];
			
			break;

		case 23:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + v] * beta[17];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + v] * beta[18];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + v] * beta[19];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + v] * beta[20];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 21 + v] * beta[21];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 22 + v] * beta[22];
			
			break;

		case 24:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + v] * beta[17];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + v] * beta[18];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + v] * beta[19];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + v] * beta[20];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 21 + v] * beta[21];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 22 + v] * beta[22];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 23 + v] * beta[23];
			
			break;

		case 25:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + v] * beta[17];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + v] * beta[18];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + v] * beta[19];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + v] * beta[20];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 21 + v] * beta[21];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 22 + v] * beta[22];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 23 + v] * beta[23];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 24 + v] * beta[24];

			break;

		default:
			1;
			break;
	}

	return eps;
}


// For second level, permutation of rows in design matrix (as in FSL)
float CalculateEpsSecondLevel(__private float eps,
							  __private float* beta,
							  __constant float* c_X_GLM,
							  int v,
							  __constant unsigned short int* c_Permutation_Vector,
							  int NUMBER_OF_VOLUMES,
							  int NUMBER_OF_REGRESSORS)
{
	switch(NUMBER_OF_REGRESSORS)
	{
		case 1:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];

			break;

		case 2:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];

			break;

		case 3:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];

			break;

		case 4:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];

			break;

		case 5:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];

			break;

		case 6:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];

			break;

		case 7:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];

			break;

		case 8:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];

			break;

		case 9:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];

			break;

		case 10:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];

			break;

		case 11:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
			
			break;

		case 12:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
			
			break;

		case 13:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
			
			break;

		case 14:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
			
			break;

		case 15:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
			
			break;

		case 16:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
			
			break;

		case 17:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
			
			break;

		case 18:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]] * beta[17];
			
			break;

		case 19:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]] * beta[17];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]] * beta[18];
			
			break;

		case 20:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]] * beta[17];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]] * beta[18];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]] * beta[19];
			
			break;

		case 21:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]] * beta[17];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]] * beta[18];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]] * beta[19];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]] * beta[20];
			
			break;

		case 22:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]] * beta[17];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]] * beta[18];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]] * beta[19];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]] * beta[20];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 21 + c_Permutation_Vector[v]] * beta[21];
			
			break;

		case 23:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]] * beta[17];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]] * beta[18];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]] * beta[19];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]] * beta[20];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 21 + c_Permutation_Vector[v]] * beta[21];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 22 + c_Permutation_Vector[v]] * beta[22];
			
			break;

		case 24:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]] * beta[17];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]] * beta[18];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]] * beta[19];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]] * beta[20];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 21 + c_Permutation_Vector[v]] * beta[21];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 22 + c_Permutation_Vector[v]] * beta[22];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 23 + c_Permutation_Vector[v]] * beta[23];
			
			break;

		case 25:

			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]] * beta[17];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]] * beta[18];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]] * beta[19];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]] * beta[20];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 21 + c_Permutation_Vector[v]] * beta[21];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 22 + c_Permutation_Vector[v]] * beta[22];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 23 + c_Permutation_Vector[v]] * beta[23];
			eps -= c_X_GLM[NUMBER_OF_VOLUMES * 24 + c_Permutation_Vector[v]] * beta[24];

			break;

		default:
			1;
			break;
	}

	return eps;
}



float CalculateContrastValue(__private float* beta, __constant float* c_Contrasts, int c, int NUMBER_OF_REGRESSORS)
{
	float contrast_value = 0.0f;

	switch(NUMBER_OF_REGRESSORS)
	{
		case 1:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];

			break;

		case 2:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];

			break;

		case 3:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];

			break;

		case 4:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];

			break;

		case 5:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];

			break;

		case 6:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];

			break;

		case 7:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];

			break;

		case 8:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];

			break;

		case 9:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];

			break;

		case 10:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];

			break;

		case 11:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			
			break;

		case 12:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			
			break;

		case 13:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			
			break;

		case 14:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			
			break;

		case 15:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			
			break;

		case 16:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			
			break;

		case 17:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			
			break;

		case 18:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
			
			break;

		case 19:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
			
			break;

		case 20:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
			
			break;

		case 21:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];

			break;

		case 22:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 21] * beta[21];
			
			break;

		case 23:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 21] * beta[21];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 22] * beta[22];
			
			break;

		case 24:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 21] * beta[21];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 22] * beta[22];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 23] * beta[23];
			
			break;

		case 25:

			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 21] * beta[21];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 22] * beta[22];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 23] * beta[23];
			contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 24] * beta[24];

			break;


		default:
			1;
			break;
	}

	return contrast_value;
}



int CalculateCBeta(__private float* cbeta, __private float* beta, __constant float* c_Contrasts, int c, int NUMBER_OF_REGRESSORS)	
{
	cbeta[c] = 0.0f;

	switch(NUMBER_OF_REGRESSORS)
	{
		case 1:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];

			break;

		case 2:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			
			break;

		case 3:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			
			break;

		case 4:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			
			break;

		case 5:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			
			break;

		case 6:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			
			break;

		case 7:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			
			break;

		case 8:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			
			break;

		case 9:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			
			break;

		case 10:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			
			break;

		case 11:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			
			break;

		case 12:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			
			break;

		case 13:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			
			break;

		case 14:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			
			break;

		case 15:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			
			break;

		case 16:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			
			break;

		case 17:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			
			break;

		case 18:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
			
			break;

		case 19:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
			
			break;

		case 20:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
			
			break;

		case 21:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
			
			break;

		case 22:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 21] * beta[21];
			
			break;

		case 23:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 21] * beta[21];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 22] * beta[22];
			
			break;

		case 24:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 21] * beta[21];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 22] * beta[22];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 23] * beta[23];
			
			break;

		case 25:

			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 21] * beta[21];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 22] * beta[22];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 23] * beta[23];
			cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 24] * beta[24];			
			
			break;

		default:
			1;
			break;
	}

	return 0;
}

int CalculateCBetas(__private float* cbeta, __private float* beta, __constant float* c_Contrasts, int NUMBER_OF_REGRESSORS, int NUMBER_OF_CONTRASTS)
{
	switch(NUMBER_OF_CONTRASTS)
	{
		case 1:

			CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);

			break;

		case 2:

			CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);

			break;

		case 3:

			CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 2, NUMBER_OF_REGRESSORS);
			
			break;

		case 4:

			CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 2, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 3, NUMBER_OF_REGRESSORS);
			
			break;

		case 5:

			CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 2, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 3, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 4, NUMBER_OF_REGRESSORS);

			break;

		case 6:

			CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 2, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 3, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 4, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 5, NUMBER_OF_REGRESSORS);

			break;

		case 7:

			CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 2, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 3, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 4, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 5, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 6, NUMBER_OF_REGRESSORS);

			break;

		case 8:
			
			CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 2, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 3, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 4, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 5, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 6, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 7, NUMBER_OF_REGRESSORS);

			break;

		case 9:

			CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 2, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 3, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 4, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 5, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 6, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 7, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 8, NUMBER_OF_REGRESSORS);

			break;

		case 10:

			CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 2, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 3, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 4, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 5, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 6, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 7, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 8, NUMBER_OF_REGRESSORS);
			CalculateCBeta(cbeta, beta, c_Contrasts, 9, NUMBER_OF_REGRESSORS);
			
			break;		

		default:
			1;
			break;
	}	

	return 0;
}

int CalculateCTXTXCCBeta(__private float* beta, float vareps, __constant float* c_ctxtxc_GLM, __private float* cbeta, int c,  int NUMBER_OF_CONTRASTS)
{
	beta[c] = 0.0f;

	switch(NUMBER_OF_CONTRASTS)
	{
		case 1:

			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
			
			break;

		case 2:

			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
			
			break;

		case 3:

			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[2 + c * NUMBER_OF_CONTRASTS] * cbeta[2];
			
			break;

		case 4:

			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[2 + c * NUMBER_OF_CONTRASTS] * cbeta[2];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[3 + c * NUMBER_OF_CONTRASTS] * cbeta[3];
			
			break;

		case 5:

			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[2 + c * NUMBER_OF_CONTRASTS] * cbeta[2];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[3 + c * NUMBER_OF_CONTRASTS] * cbeta[3];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[4 + c * NUMBER_OF_CONTRASTS] * cbeta[4];
			
			break;

		case 6:

			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[2 + c * NUMBER_OF_CONTRASTS] * cbeta[2];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[3 + c * NUMBER_OF_CONTRASTS] * cbeta[3];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[4 + c * NUMBER_OF_CONTRASTS] * cbeta[4];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[5 + c * NUMBER_OF_CONTRASTS] * cbeta[5];
			
			break;

		case 7:

			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[2 + c * NUMBER_OF_CONTRASTS] * cbeta[2];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[3 + c * NUMBER_OF_CONTRASTS] * cbeta[3];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[4 + c * NUMBER_OF_CONTRASTS] * cbeta[4];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[5 + c * NUMBER_OF_CONTRASTS] * cbeta[5];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[6 + c * NUMBER_OF_CONTRASTS] * cbeta[6];
			
			break;

		case 8:

			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[2 + c * NUMBER_OF_CONTRASTS] * cbeta[2];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[3 + c * NUMBER_OF_CONTRASTS] * cbeta[3];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[4 + c * NUMBER_OF_CONTRASTS] * cbeta[4];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[5 + c * NUMBER_OF_CONTRASTS] * cbeta[5];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[6 + c * NUMBER_OF_CONTRASTS] * cbeta[6];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[7 + c * NUMBER_OF_CONTRASTS] * cbeta[7];
			
			break;

		case 9:

			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[2 + c * NUMBER_OF_CONTRASTS] * cbeta[2];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[3 + c * NUMBER_OF_CONTRASTS] * cbeta[3];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[4 + c * NUMBER_OF_CONTRASTS] * cbeta[4];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[5 + c * NUMBER_OF_CONTRASTS] * cbeta[5];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[6 + c * NUMBER_OF_CONTRASTS] * cbeta[6];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[7 + c * NUMBER_OF_CONTRASTS] * cbeta[7];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[8 + c * NUMBER_OF_CONTRASTS] * cbeta[8];
			
			break;

		case 10:

			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[2 + c * NUMBER_OF_CONTRASTS] * cbeta[2];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[3 + c * NUMBER_OF_CONTRASTS] * cbeta[3];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[4 + c * NUMBER_OF_CONTRASTS] * cbeta[4];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[5 + c * NUMBER_OF_CONTRASTS] * cbeta[5];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[6 + c * NUMBER_OF_CONTRASTS] * cbeta[6];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[7 + c * NUMBER_OF_CONTRASTS] * cbeta[7];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[8 + c * NUMBER_OF_CONTRASTS] * cbeta[8];
			beta[c] += 1.0f/vareps * c_ctxtxc_GLM[9 + c * NUMBER_OF_CONTRASTS] * cbeta[9];

			break;		

		default:
			1;
			break;
	}	

	return 0;	
}			


int CalculateCTXTXCCBetas(__private float* beta, float vareps, __constant float* c_ctxtxc_GLM, __private float* cbeta, int NUMBER_OF_CONTRASTS)
{
	switch(NUMBER_OF_CONTRASTS)
	{
		case 1:

			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
			
			break;

		case 2:
			
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
						
			break;

		case 3:

			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 2, NUMBER_OF_CONTRASTS);

			break;

		case 4:

			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 2, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 3, NUMBER_OF_CONTRASTS);
			
			break;

		case 5:

			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 2, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 3, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 4, NUMBER_OF_CONTRASTS);

			break;

		case 6:

			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 2, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 3, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 4, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 5, NUMBER_OF_CONTRASTS);
			
			break;

		case 7:

			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 2, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 3, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 4, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 5, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 6, NUMBER_OF_CONTRASTS);
			
			break;

		case 8:

			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 2, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 3, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 4, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 5, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 6, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 7, NUMBER_OF_CONTRASTS);

			break;

		case 9:

			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 2, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 3, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 4, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 5, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 6, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 7, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 8, NUMBER_OF_CONTRASTS);

			break;

		case 10:

			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 2, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 3, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 4, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 5, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 6, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 7, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 8, NUMBER_OF_CONTRASTS);
			CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 9, NUMBER_OF_CONTRASTS);
			
			break;		

		default:
			1;
			break;
	}	

	return 0;	
}			


float CalculateFTestScalar(__private float* cbeta, __private float* beta, int NUMBER_OF_CONTRASTS)
{
	float scalar = 0.0f;

	switch(NUMBER_OF_CONTRASTS)
	{
		case 1:

			scalar += cbeta[0] * beta[0];
			
			break;

		case 2:

			scalar += cbeta[0] * beta[0];
			scalar += cbeta[1] * beta[1];
			
			break;

		case 3:

			scalar += cbeta[0] * beta[0];
			scalar += cbeta[1] * beta[1];
			scalar += cbeta[2] * beta[2];
			
			break;

		case 4:

			scalar += cbeta[0] * beta[0];
			scalar += cbeta[1] * beta[1];
			scalar += cbeta[2] * beta[2];
			scalar += cbeta[3] * beta[3];
			
			break;

		case 5:

			scalar += cbeta[0] * beta[0];
			scalar += cbeta[1] * beta[1];
			scalar += cbeta[2] * beta[2];
			scalar += cbeta[3] * beta[3];
			scalar += cbeta[4] * beta[4];
			
			break;

		case 6:

			scalar += cbeta[0] * beta[0];
			scalar += cbeta[1] * beta[1];
			scalar += cbeta[2] * beta[2];
			scalar += cbeta[3] * beta[3];
			scalar += cbeta[4] * beta[4];
			scalar += cbeta[5] * beta[5];
			
			break;

		case 7:

			scalar += cbeta[0] * beta[0];
			scalar += cbeta[1] * beta[1];
			scalar += cbeta[2] * beta[2];
			scalar += cbeta[3] * beta[3];
			scalar += cbeta[4] * beta[4];
			scalar += cbeta[5] * beta[5];
			scalar += cbeta[6] * beta[6];
			scalar += cbeta[7] * beta[7];
			
			break;

		case 8:

			scalar += cbeta[0] * beta[0];
			scalar += cbeta[1] * beta[1];
			scalar += cbeta[2] * beta[2];
			scalar += cbeta[3] * beta[3];
			scalar += cbeta[4] * beta[4];
			scalar += cbeta[5] * beta[5];
			scalar += cbeta[6] * beta[6];
			scalar += cbeta[7] * beta[7];
			
			break;

		case 9:

			scalar += cbeta[0] * beta[0];
			scalar += cbeta[1] * beta[1];
			scalar += cbeta[2] * beta[2];
			scalar += cbeta[3] * beta[3];
			scalar += cbeta[4] * beta[4];
			scalar += cbeta[5] * beta[5];
			scalar += cbeta[6] * beta[6];
			scalar += cbeta[7] * beta[7];
			scalar += cbeta[8] * beta[8];
			
			break;

		case 10:

			scalar += cbeta[0] * beta[0];
			scalar += cbeta[1] * beta[1];
			scalar += cbeta[2] * beta[2];
			scalar += cbeta[3] * beta[3];
			scalar += cbeta[4] * beta[4];
			scalar += cbeta[5] * beta[5];
			scalar += cbeta[6] * beta[6];
			scalar += cbeta[7] * beta[7];
			scalar += cbeta[8] * beta[8];
			scalar += cbeta[9] * beta[9];
			
			break;		

		default:
			1;
			break;
	}	

	return scalar;
}

	



__kernel void CalculateStatisticalMapsGLMTTestFirstLevelPermutation(__global float* Statistical_Maps,
																	__global const float* Volumes,
																	__global const float* Mask,
																	__constant float* c_X_GLM,
																	__constant float* c_xtxxt_GLM,
																	__constant float* c_Contrasts,	
																	__constant float* c_ctxtxc_GLM,
																	__private int DATA_W,
																	__private int DATA_H,
																	__private int DATA_D,
																	__private int NUMBER_OF_VOLUMES,
																	__private int NUMBER_OF_REGRESSORS,
																	__private int NUMBER_OF_CONTRASTS)
{	
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
		return;

	int t = 0;
	float eps, meaneps, vareps;
	float beta[25];

	// Reset beta weights
	beta[0] = 0.0f;
	beta[1] = 0.0f;
	beta[2] = 0.0f;
	beta[3] = 0.0f;
	beta[4] = 0.0f;
	beta[5] = 0.0f;
	beta[6] = 0.0f;
	beta[7] = 0.0f;
	beta[8] = 0.0f;
	beta[9] = 0.0f;
	beta[10] = 0.0f;
	beta[11] = 0.0f;
	beta[12] = 0.0f;
	beta[13] = 0.0f;
	beta[14] = 0.0f;
	beta[15] = 0.0f;
	beta[16] = 0.0f;
	beta[17] = 0.0f;
	beta[18] = 0.0f;
	beta[19] = 0.0f;
	beta[20] = 0.0f;
	beta[21] = 0.0f;
	beta[22] = 0.0f;
	beta[23] = 0.0f;
	beta[24] = 0.0f;

	// Calculate betahat, i.e. multiply (x^T x)^(-1) x^T with Y
	// Loop over volumes
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		float value = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];

		// Loop over regressors using unrolled code for performance
		CalculateBetaWeightsFirstLevel(beta, value, c_xtxxt_GLM, v, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);
	}

	// Calculate the mean and variance of the error eps
	meaneps = 0.0f;
	vareps = 0.0f;
	float n = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		eps = CalculateEpsFirstLevel(eps, beta, c_X_GLM, v, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);
		
		n += 1.0f;
		float delta = eps - meaneps;
		meaneps += delta/n;
		vareps += delta * (eps - meaneps);
	}
	vareps = vareps / (n - 1.0f);

	// Loop over contrasts and calculate t-values
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		float contrast_value = 0.0f;
		contrast_value = CalculateContrastValue(beta, c_Contrasts, c, NUMBER_OF_REGRESSORS);
		Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_value * rsqrt(vareps * c_ctxtxc_GLM[c]);
	}
}

__kernel void CalculateStatisticalMapsGLMFTestFirstLevelPermutation(__global float* Statistical_Maps,
					 		                                        __global const float* Volumes,
																	__global const float* Mask,
																	__constant float* c_X_GLM,
																	__constant float* c_xtxxt_GLM,
																	__constant float* c_Contrasts,
																	__constant float* c_ctxtxc_GLM,
																	__private int DATA_W,
																	__private int DATA_H,
																	__private int DATA_D,
																	__private int NUMBER_OF_VOLUMES,
																	__private int NUMBER_OF_REGRESSORS,
																	__private int NUMBER_OF_CONTRASTS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
		return;

	int t = 0;
	float eps, meaneps, vareps;
	float beta[25];

	// Reset beta weights
	beta[0] = 0.0f;
	beta[1] = 0.0f;
	beta[2] = 0.0f;
	beta[3] = 0.0f;
	beta[4] = 0.0f;
	beta[5] = 0.0f;
	beta[6] = 0.0f;
	beta[7] = 0.0f;
	beta[8] = 0.0f;
	beta[9] = 0.0f;
	beta[10] = 0.0f;
	beta[11] = 0.0f;
	beta[12] = 0.0f;
	beta[13] = 0.0f;
	beta[14] = 0.0f;
	beta[15] = 0.0f;
	beta[16] = 0.0f;
	beta[17] = 0.0f;
	beta[18] = 0.0f;
	beta[19] = 0.0f;
	beta[20] = 0.0f;
	beta[21] = 0.0f;
	beta[22] = 0.0f;
	beta[23] = 0.0f;
	beta[24] = 0.0f;

	// Calculate betahat, i.e. multiply (x^T x)^(-1) x^T with Y
	// Loop over volumes
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		float value = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];

		// Loop over regressors using unrolled code for performance
		CalculateBetaWeightsFirstLevel(beta, value, c_xtxxt_GLM, v, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);
	}

	// Calculate the mean and variance of the error eps
	meaneps = 0.0f;
	vareps = 0.0f;
	float n = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		eps = CalculateEpsFirstLevel(eps, beta, c_X_GLM, v, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);
		
		n += 1.0f;
		float delta = eps - meaneps;
		meaneps += delta/n;
		vareps += delta * (eps - meaneps);
	}
	vareps = vareps / (n - 1.0f);

	//-------------------------

	// Calculate matrix vector product C*beta (minus u)
	float cbeta[10];
	CalculateCBetas(cbeta, beta, c_Contrasts, NUMBER_OF_REGRESSORS, NUMBER_OF_CONTRASTS);		

	// Calculate total vector matrix vector product (C*beta)^T ( 1/vareps * (C^T (X^T X)^(-1) C^T)^(-1) ) (C*beta)

	// Calculate right hand side, temp = ( 1/vareps * (C^T (X^T X)^(-1) C^T)^(-1) ) (C*beta)	
	CalculateCTXTXCCBetas(beta, vareps, c_ctxtxc_GLM, cbeta, NUMBER_OF_CONTRASTS);

	// Finally calculate (C*beta)^T * temp
	float scalar = CalculateFTestScalar(cbeta,beta,NUMBER_OF_CONTRASTS);

	// Save F-value
	Statistical_Maps[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = scalar/(float)NUMBER_OF_CONTRASTS;
}

// Optimized kernel for calculating t-test values for permutations, second level

__kernel void CalculateStatisticalMapsGLMTTestSecondLevelPermutation(__global float* Statistical_Maps,
		                                       	   	   				 __global const float* Volumes,
		                                       	   	   				 __global const float* Mask,
		                                       	   	   				 __constant float* c_X_GLM,
		                                       	   	   				 __constant float* c_xtxxt_GLM,
		                                       	   	   				 __constant float* c_Contrasts,
		                                       	   	   				 __constant float* c_ctxtxc_GLM,
		                                       	   	   				 __constant unsigned short int* c_Permutation_Vector,
		                                       	   	   				 __private int DATA_W,
		                                       	   	   				 __private int DATA_H,
		                                       	   	   				 __private int DATA_D,
		                                       	   	   				 __private int NUMBER_OF_VOLUMES,
		                                       	   	   				 __private int NUMBER_OF_REGRESSORS,
		                                       	   	   				 __private int NUMBER_OF_CONTRASTS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
		return;

	int t = 0;
	float eps, meaneps, vareps;
	float beta[25];

	// Reset beta weights
	beta[0] = 0.0f;
	beta[1] = 0.0f;
	beta[2] = 0.0f;
	beta[3] = 0.0f;
	beta[4] = 0.0f;
	beta[5] = 0.0f;
	beta[6] = 0.0f;
	beta[7] = 0.0f;
	beta[8] = 0.0f;
	beta[9] = 0.0f;
	beta[10] = 0.0f;
	beta[11] = 0.0f;
	beta[12] = 0.0f;
	beta[13] = 0.0f;
	beta[14] = 0.0f;
	beta[15] = 0.0f;
	beta[16] = 0.0f;
	beta[17] = 0.0f;
	beta[18] = 0.0f;
	beta[19] = 0.0f;
	beta[20] = 0.0f;
	beta[21] = 0.0f;
	beta[22] = 0.0f;
	beta[23] = 0.0f;
	beta[24] = 0.0f;


	// Calculate betahat, i.e. multiply (x^T x)^(-1) x^T with Y
	// Loop over volumes
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		float value = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];

		// Loop over regressors using unrolled code for performance
		CalculateBetaWeightsSecondLevel(beta, value, c_xtxxt_GLM, v, c_Permutation_Vector, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);
	}

	/*
	// Calculate the mean and variance of the error eps
	meaneps = 0.0f;
	vareps = 0.0f;
	float n = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];

		// Loop over regressors using unrolled code for performance
		eps = CalculateEpsSecondLevel(eps, beta, c_X_GLM, v, c_Permutation_Vector, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);

		n += 1.0f;
		float delta = eps - meaneps;
		meaneps += delta/n;
		vareps += delta * (eps - meaneps);
	}
	vareps = vareps / (n - 1.0f);
	*/

	vareps = 0.0f;
	float n = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];

		// Loop over regressors using unrolled code for performance
		eps = CalculateEpsSecondLevel(eps, beta, c_X_GLM, v, c_Permutation_Vector, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);

		vareps += eps * eps;
	}
	vareps = vareps / ((float)NUMBER_OF_VOLUMES - 1.0f);


	// Loop over contrasts and calculate t-values

	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		float contrast_value = 0.0f;
		contrast_value = CalculateContrastValue(beta, c_Contrasts, c, NUMBER_OF_REGRESSORS);
		Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_value * rsqrt(vareps * c_ctxtxc_GLM[c]);
	}
}


// Optimized kernel for calculating F-test values for permutations, second level

__kernel void CalculateStatisticalMapsGLMFTestSecondLevelPermutation(__global float* Statistical_Maps,
		                                       	   	   	             __global const float* Volumes,
		                                       	   	   	             __global const float* Mask,
		                                       	   	   				 __constant float* c_X_GLM,
																	 __constant float* c_xtxxt_GLM,
		                                       	   	   				 __constant float* c_Contrasts,
		                                       	   	   				 __constant float* c_ctxtxc_GLM,
																	 __constant unsigned short int* c_Permutation_Vector,
		                                       	   	   				 __private int DATA_W,
		                                       	   	   				 __private int DATA_H,
		                                       	   	   				 __private int DATA_D,
		                                       	   	   				 __private int NUMBER_OF_VOLUMES,
		                                       	   	   				 __private int NUMBER_OF_REGRESSORS,
		                                       	   	   				 __private int NUMBER_OF_CONTRASTS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
		return;

	int t = 0;
	float eps, meaneps, vareps;
	float beta[25];

	beta[0] = 0.0f;
	beta[1] = 0.0f;
	beta[2] = 0.0f;
	beta[3] = 0.0f;
	beta[4] = 0.0f;
	beta[5] = 0.0f;
	beta[6] = 0.0f;
	beta[7] = 0.0f;
	beta[8] = 0.0f;
	beta[9] = 0.0f;
	beta[10] = 0.0f;
	beta[11] = 0.0f;
	beta[12] = 0.0f;
	beta[13] = 0.0f;
	beta[14] = 0.0f;
	beta[15] = 0.0f;
	beta[16] = 0.0f;
	beta[17] = 0.0f;
	beta[18] = 0.0f;
	beta[19] = 0.0f;
	beta[20] = 0.0f;
	beta[21] = 0.0f;
	beta[22] = 0.0f;
	beta[23] = 0.0f;
	beta[24] = 0.0f;


	// Calculate betahat, i.e. multiply (x^T x)^(-1) x^T with Y
	// Loop over volumes
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		float value = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];

		// Loop over regressors using unrolled code for performance
		CalculateBetaWeightsSecondLevel(beta, value, c_xtxxt_GLM, v, c_Permutation_Vector, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);
	}

	// Calculate the mean and variance of the error eps
	meaneps = 0.0f;
	vareps = 0.0f;
	float n = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];

		// Loop over regressors using unrolled code for performance
		eps = CalculateEpsSecondLevel(eps, beta, c_X_GLM, v, c_Permutation_Vector, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);

		n += 1.0f;
		float delta = eps - meaneps;
		meaneps += delta/n;
		vareps += delta * (eps - meaneps);
	}
	vareps = vareps / (n - 1.0f);

	//-------------------------

	// Calculate matrix vector product C*beta (minus u)
	float cbeta[10];
	CalculateCBetas(cbeta, beta, c_Contrasts, NUMBER_OF_REGRESSORS, NUMBER_OF_CONTRASTS);		

	// Calculate total vector matrix vector product (C*beta)^T ( 1/vareps * (C^T (X^T X)^(-1) C^T)^(-1) ) (C*beta)

	// Calculate right hand side, temp = ( 1/vareps * (C^T (X^T X)^(-1) C^T)^(-1) ) (C*beta)
	CalculateCTXTXCCBetas(beta, vareps, c_ctxtxc_GLM, cbeta, NUMBER_OF_CONTRASTS);

	// Finally calculate (C*beta)^T * temp
	float scalar = CalculateFTestScalar(cbeta,beta,NUMBER_OF_CONTRASTS);

	// Save F-value
	Statistical_Maps[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = scalar/(float)NUMBER_OF_CONTRASTS;
}




// Estimates voxel specific AR(4) models
__kernel void EstimateAR4Models(__global float* AR1_Estimates, 
                                __global float* AR2_Estimates, 
								__global float* AR3_Estimates, 
								__global float* AR4_Estimates, 
								__global const float* fMRI_Volumes, 
								__global const float* Mask, 
								__private int DATA_W, 
								__private int DATA_H, 
								__private int DATA_D, 
								__private int DATA_T,
								__private int INVALID_TIMEPOINTS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Mask[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] != 1.0f )
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

    old_value_1 = fMRI_Volumes[Calculate4DIndex(x, y, z, 0 + INVALID_TIMEPOINTS, DATA_W, DATA_H, DATA_D)];
    c0 += old_value_1 * old_value_1;
    old_value_2 = fMRI_Volumes[Calculate4DIndex(x, y, z, 1 + INVALID_TIMEPOINTS, DATA_W, DATA_H, DATA_D)];
    c0 += old_value_2 * old_value_2;
    c1 += old_value_2 * old_value_1;
    old_value_3 = fMRI_Volumes[Calculate4DIndex(x, y, z, 2 + INVALID_TIMEPOINTS, DATA_W, DATA_H, DATA_D)];
    c0 += old_value_3 * old_value_3;
    c1 += old_value_3 * old_value_2;
    c2 += old_value_3 * old_value_1;
    old_value_4 = fMRI_Volumes[Calculate4DIndex(x, y, z, 3 + INVALID_TIMEPOINTS, DATA_W, DATA_H, DATA_D)];
    c0 += old_value_4 * old_value_4;
    c1 += old_value_4 * old_value_3;
    c2 += old_value_4 * old_value_2;
    c3 += old_value_4 * old_value_1;

    // Estimate c0, c1, c2, c3, c4
    for (t = 4 + INVALID_TIMEPOINTS; t < DATA_T; t++)
    {
        // Read data into register
        old_value_5 = fMRI_Volumes[Calculate4DIndex(x, y, z, t, DATA_W, DATA_H, DATA_D)];
        
        // Sum and multiply the values in fast registers
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

    c0 /= ((float)DATA_T - 1.0f - (float)INVALID_TIMEPOINTS);
    c1 /= ((float)DATA_T - 2.0f - (float)INVALID_TIMEPOINTS);
    c2 /= ((float)DATA_T - 3.0f - (float)INVALID_TIMEPOINTS);
    c3 /= ((float)DATA_T - 4.0f - (float)INVALID_TIMEPOINTS);
    c4 /= ((float)DATA_T - 5.0f - (float)INVALID_TIMEPOINTS);

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


__kernel void ApplyWhiteningAR4(__global float* Whitened_fMRI_Volumes, 
                                __global float* fMRI_Volumes, 
								__global const float* AR1_Estimates, 
								__global const float* AR2_Estimates, 
								__global const float* AR3_Estimates, 
								__global const float* AR4_Estimates, 
								__global const float* Mask, 
								__private int DATA_W, 
								__private int DATA_H, 
								__private int DATA_D, 
								__private int DATA_T)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if ( x >= DATA_W || y >= DATA_H || z >= DATA_D )
        return;

    if ( Mask[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] != 1.0f )
		return;

    int t = 0;
	float old_value_1, old_value_2, old_value_3, old_value_4, old_value_5;
    float4 alphas;
	alphas.x = AR1_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)];
    alphas.y = AR2_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)];
    alphas.z = AR3_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)];
    alphas.w = AR4_Estimates[Calculate3DIndex(x, y, z, DATA_W, DATA_H)];

    // Calculate the whitened timeseries

    old_value_1 = fMRI_Volumes[Calculate4DIndex(x, y, z, 0, DATA_W, DATA_H, DATA_D)];	
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

__kernel void GeneratePermutedVolumesFirstLevel(__global float* Permuted_fMRI_Volumes, 
                                                __global const float* Whitened_fMRI_Volumes, 
												__global const float* AR1_Estimates, 
												__global const float* AR2_Estimates, 
												__global const float* AR3_Estimates, 
												__global const float* AR4_Estimates, 
												__global const float* Mask, 
												__constant unsigned short int *c_Permutation_Vector, 
												__private int DATA_W, 
												__private int DATA_H, 
												__private int DATA_D, 
												__private int DATA_T)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if ( x >= DATA_W || y >= DATA_H || z >= DATA_D )
        return;

    if ( Mask[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] != 1.0f )
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
        old_value_5 = alphas.x * old_value_4 + alphas.y * old_value_3 + alphas.z * old_value_2 + alphas.w * old_value_1 + Whitened_fMRI_Volumes[Calculate4DIndex(x, y, z, c_Permutation_Vector[t], DATA_W, DATA_H, DATA_D)];
			
        Permuted_fMRI_Volumes[Calculate4DIndex(x, y, z, t, DATA_W, DATA_H, DATA_D)] = old_value_5;

        // Save old values
		old_value_1 = old_value_2;
        old_value_2 = old_value_3;
        old_value_3 = old_value_4;
        old_value_4 = old_value_5;
    }
}



__kernel void SetStartClusterIndicesKernel(__global int* Cluster_Indices,
										   __global const float* Data,
										   __global const float* Mask,
										   __private float threshold,
										   __private int DATA_W,
										   __private int DATA_H,
										   __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	// Threshold data
	if ( (Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] == 1.0f) && (Data[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] > threshold) )
	{
		// Set an unique index
		Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = Calculate3DIndex(x,y,z,DATA_W,DATA_H);
	}
	else
	{
		// Make sure that all other voxels have a higher start index
		Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = DATA_W * DATA_H * DATA_D * 3;
	}
}






__kernel void ClusterizeScan(__global int* Cluster_Indices,
						  	  volatile __global float* Updated,
						  	  __global const float* Data,
						  	  __global const float* Mask,
						  	  __private float threshold,
						  	  __private int DATA_W,
						  	  __private int DATA_H,
						  	  __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
		return;

	// Threshold data
	if ( Data[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] > threshold )
	{
		int label1, label2, temp;

		label2 = DATA_W * DATA_H * DATA_D * 3;

		// Original index
		label1 = Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];

		// z - 1
		temp = Cluster_Indices[Calculate3DIndex(x-1,y,z-1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x-1,y-1,z-1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x-1,y+1,z-1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x,y,z-1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x,y-1,z-1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x,y+1,z-1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x+1,y,z-1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x+1,y-1,z-1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x+1,y+1,z-1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		// z

		temp = Cluster_Indices[Calculate3DIndex(x-1,y,z,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x-1,y-1,z,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x-1,y+1,z,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x,y-1,z,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x,y+1,z,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x+1,y,z,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x+1,y-1,z,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x+1,y+1,z,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		// z + 1

		temp = Cluster_Indices[Calculate3DIndex(x-1,y,z+1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x-1,y-1,z+1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x-1,y+1,z+1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x,y,z+1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x,y-1,z+1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x,y+1,z+1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x+1,y,z+1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x+1,y-1,z+1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		temp = Cluster_Indices[Calculate3DIndex(x+1,y+1,z+1,DATA_W,DATA_H)];
		if (temp < label2)
		{
			label2 = temp;
		}

		if (label2 < label1)
		{
			Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = label2;
			float one = 1.0f;
			atomic_xchg(Updated,one);
		}

	}
}



__kernel void ClusterizeRelabel(__global int* Cluster_Indices,
						  	  	__global const float* Data,
						  	  	__global const float* Mask,
						  	  	__private float threshold,
						  	  	__private int DATA_W,
						  	  	__private int DATA_H,
						  	  	__private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	// Threshold data
	if ( (Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] == 1.0f) && (Data[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] > threshold) )
	{
		// Relabel voxels
		int label = Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];
		int next = Cluster_Indices[label];
		while (next != label)
		{
			label = next;
			next = Cluster_Indices[label];
		}
		Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = label;
	}
}

__kernel void CalculateClusterSizes(__global int* Cluster_Indices,
						  	  	    volatile __global int* Cluster_Sizes,
						  	  	    __global const float* Data,
						  	  	    __global const float* Mask,
						  	  	    __private float threshold,
						  	  	    __private int DATA_W,
						  	  	    __private int DATA_H,
						  	  	    __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
		return;

	// Threshold data
	if ( Data[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] > threshold )
	{
		// Increment counter for the current cluster index
		atomic_inc(&Cluster_Sizes[Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)]]);
	}
}

__kernel void CalculateLargestCluster(__global int* Cluster_Sizes,
								      volatile global int* Largest_Cluster,
   						  	  	      __private int DATA_W,
									  __private int DATA_H,
									  __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int cluster_size = Cluster_Sizes[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];

	// Most cluster size counters are zero, so avoid running atomic max for those
	if (cluster_size == 0)
		return;

	atomic_max(Largest_Cluster,cluster_size);
}


__kernel void ThresholdVolume(__global float* Thresholded_Volume, 
	                          __global const float* Volume, 
							  __private float threshold, 
							  __private int DATA_W, 
							  __private int DATA_H, 
							  __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

	if ( Volume[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] > threshold )
	{
		Thresholded_Volume[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 1.0f;
	}
	else
	{
		Thresholded_Volume[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.001f;
	}
}

// Removes a linear fit estimated with CalculateGLMBetaWeights
__kernel void RemoveLinearFit(__global float* Residual_Volumes, 
                              __global const float* Volumes, 
							  __global const float* Beta_Volumes, 
							  __global const float* Mask, 
							  __constant float *c_X_Detrend, 
							  __private int DATA_W, 
							  __private int DATA_H, 
							  __private int DATA_D, 
							  __private int NUMBER_OF_VOLUMES, 
							  __private int NUMBER_OF_REGRESSORS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
	{
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			Residual_Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}

		return;
	}
	
	float eps;
	float beta[10];

	// Load beta values into regressors
    for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{ 
		beta[r] = Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)];
	}

	// Calculate the residual
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{ 			
			eps -= beta[r] * c_X_Detrend[NUMBER_OF_VOLUMES * r + v];
		}
		Residual_Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = eps;
	}
}

__kernel void RemoveMean(__global float* Volumes, 
					     __private int DATA_W, 
						 __private int DATA_H, 
						 __private int DATA_D, 
						 __private int NUMBER_OF_VOLUMES)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;


	float mean = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		mean += Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
	}
	mean /= (float)NUMBER_OF_VOLUMES;
	
	// Calculate the residual
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] -= mean;
	}

}





__kernel void AddVolume(__global float* Volume, 
	                    __private float value, 
						__private int DATA_W, 
						__private int DATA_H, 
						__private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int idx = Calculate3DIndex(x,y,z,DATA_W,DATA_H);

	Volume[idx] += value;
}


__kernel void AddVolumes(__global float* Result, 
	                     __global const float* Volume1, 
						 __global const float* Volume2, 
						 __private int DATA_W, 
						 __private int DATA_H, 
						 __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int idx = Calculate3DIndex(x,y,z,DATA_W,DATA_H);

	Result[idx] = Volume1[idx] + Volume2[idx];
}

__kernel void AddVolumesOverwrite(__global float* Volume1, 
	                              __global const float* Volume2, 
								  __private int DATA_W, 
								  __private int DATA_H, 
								  __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int idx = Calculate3DIndex(x,y,z,DATA_W,DATA_H);

	Volume1[idx] = Volume1[idx] + Volume2[idx];
}

__kernel void MultiplyVolume(__global float* Volume, 
	                         __private float factor, 
							 __private int DATA_W, 
							 __private int DATA_H, 
							 __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if ((x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D))
		return;

	int idx = Calculate3DIndex(x,y,z,DATA_W,DATA_H);

	Volume[idx] = Volume[idx] * factor;
}

__kernel void MultiplyVolumes(__global float* Result, 
	                          __global const float* Volume1, 
							  __global const float* Volume2, 
							  __private int DATA_W, 
							  __private int DATA_H, 
							  __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int idx = Calculate3DIndex(x,y,z,DATA_W,DATA_H);

	Result[idx] = Volume1[idx] * Volume2[idx];
}

__kernel void MultiplyVolumesOverwrite(__global float* Volume1, 
	                                   __global const float* Volume2, 
									   __private int DATA_W, 
									   __private int DATA_H, 
									   __private int DATA_D, 
									   __private int VOLUME)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int idx3D = Calculate3DIndex(x,y,z,DATA_W,DATA_H);
	int idx4D = Calculate4DIndex(x,y,z,VOLUME,DATA_W,DATA_H,DATA_D);

	Volume1[idx4D] = Volume1[idx4D] * Volume2[idx3D];
}


