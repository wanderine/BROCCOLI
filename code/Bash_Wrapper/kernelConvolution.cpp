/*
    BROCCOLI: Software for Fast fMRI Analysis on Many-Core CPUs and GPUs
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



// Convolution functions

// Separable 3D convolution

#define VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_ROWS 32
#define VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_ROWS 8
#define VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_ROWS 8



__kernel void SeparableConvolutionRows_16KB_512threads(__global float *Filter_Response,
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

__kernel void SeparableConvolutionRows_16KB_256threads(__global float *Filter_Response, 
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


__kernel void SeparableConvolutionColumns_16KB_512threads(__global float *Filter_Response, 
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

__kernel void SeparableConvolutionColumns_16KB_256threads(__global float *Filter_Response, 
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


__kernel void SeparableConvolutionRods_16KB_512threads(__global float *Filter_Response, 
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


__kernel void SeparableConvolutionRods_16KB_256threads(__global float *Filter_Response,
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

#define VALID_FILTER_RESPONSES_X_CONVOLUTION_2D_24KB 90
#define VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D_24KB 58

#define VALID_FILTER_RESPONSES_X_CONVOLUTION_2D_32KB 122
#define VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D_32KB 58

// Non-separable 3D convolution

// float66 since Apple seems to have predefined float6
typedef struct tag_float66 {float a; float b; float c; float d; float e; float f;} float66;

// Non-separable 2D convolution for three complex valued 7 x 7 filters, unrolled for performance

// Shared memory is 64 * 96 pixels
float66 Conv_2D_Unrolled_7x7_ThreeFilters_24KB(__local float* image,
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
	float66 sum;
	sum.a = 0.0f;
	sum.b = 0.0f;
	sum.c = 0.0f;
	sum.d = 0.0f;
	sum.e = 0.0f;
	sum.f = 0.0f;
	
    pixel = image[(y - 3)*96 + x - 3];
    sum.a += pixel * Filter_1_Real[6*7 + 6];
	sum.b += pixel * Filter_1_Imag[6*7 + 6];
	sum.c += pixel * Filter_2_Real[6*7 + 6];
	sum.d += pixel * Filter_2_Imag[6*7 + 6];
	sum.e += pixel * Filter_3_Real[6*7 + 6];
	sum.f += pixel * Filter_3_Imag[6*7 + 6];
    pixel = image[(y - 2)*96 + x - 3];
    sum.a += pixel * Filter_1_Real[5*7 + 6];
	sum.b += pixel * Filter_1_Imag[5*7 + 6];
	sum.c += pixel * Filter_2_Real[5*7 + 6];
	sum.d += pixel * Filter_2_Imag[5*7 + 6];
	sum.e += pixel * Filter_3_Real[5*7 + 6];
	sum.f += pixel * Filter_3_Imag[5*7 + 6];
	pixel = image[(y - 1)*96 + x - 3];
    sum.a += pixel * Filter_1_Real[4*7 + 6];
	sum.b += pixel * Filter_1_Imag[4*7 + 6];
	sum.c += pixel * Filter_2_Real[4*7 + 6];
	sum.d += pixel * Filter_2_Imag[4*7 + 6];
	sum.e += pixel * Filter_3_Real[4*7 + 6];
	sum.f += pixel * Filter_3_Imag[4*7 + 6];
	pixel = image[y*96 + x - 3];
    sum.a += pixel * Filter_1_Real[3*7 + 6];
	sum.b += pixel * Filter_1_Imag[3*7 + 6];
	sum.c += pixel * Filter_2_Real[3*7 + 6];
	sum.d += pixel * Filter_2_Imag[3*7 + 6];
	sum.e += pixel * Filter_3_Real[3*7 + 6];
	sum.f += pixel * Filter_3_Imag[3*7 + 6];
    pixel = image[(y + 1)*96 + x - 3];
    sum.a += pixel * Filter_1_Real[2*7 + 6];
	sum.b += pixel * Filter_1_Imag[2*7 + 6];
	sum.c += pixel * Filter_2_Real[2*7 + 6];
	sum.d += pixel * Filter_2_Imag[2*7 + 6];
	sum.e += pixel * Filter_3_Real[2*7 + 6];
	sum.f += pixel * Filter_3_Imag[2*7 + 6];
	pixel = image[(y + 2)*96 + x - 3];
    sum.a += pixel * Filter_1_Real[1*7 + 6];
	sum.b += pixel * Filter_1_Imag[1*7 + 6];
	sum.c += pixel * Filter_2_Real[1*7 + 6];
	sum.d += pixel * Filter_2_Imag[1*7 + 6];
	sum.e += pixel * Filter_3_Real[1*7 + 6];
	sum.f += pixel * Filter_3_Imag[1*7 + 6];
	pixel = image[(y + 3)*96 + x - 3];
    sum.a += pixel * Filter_1_Real[6];
	sum.b += pixel * Filter_1_Imag[6];
	sum.c += pixel * Filter_2_Real[6];
	sum.d += pixel * Filter_2_Imag[6];
	sum.e += pixel * Filter_3_Real[6];
	sum.f += pixel * Filter_3_Imag[6];

    pixel = image[(y - 3)*96 + x - 2];
    sum.a += pixel * Filter_1_Real[6*7 + 5];
	sum.b += pixel * Filter_1_Imag[6*7 + 5];
	sum.c += pixel * Filter_2_Real[6*7 + 5];
	sum.d += pixel * Filter_2_Imag[6*7 + 5];
	sum.e += pixel * Filter_3_Real[6*7 + 5];
	sum.f += pixel * Filter_3_Imag[6*7 + 5];
    pixel = image[(y - 2)*96 + x - 2];
    sum.a += pixel * Filter_1_Real[5*7 + 5];
	sum.b += pixel * Filter_1_Imag[5*7 + 5];
	sum.c += pixel * Filter_2_Real[5*7 + 5];
	sum.d += pixel * Filter_2_Imag[5*7 + 5];
	sum.e += pixel * Filter_3_Real[5*7 + 5];
	sum.f += pixel * Filter_3_Imag[5*7 + 5];
	pixel = image[(y - 1)*96 + x - 2];
    sum.a += pixel * Filter_1_Real[4*7 + 5];
	sum.b += pixel * Filter_1_Imag[4*7 + 5];
	sum.c += pixel * Filter_2_Real[4*7 + 5];
	sum.d += pixel * Filter_2_Imag[4*7 + 5];
	sum.e += pixel * Filter_3_Real[4*7 + 5];
	sum.f += pixel * Filter_3_Imag[4*7 + 5];
	pixel = image[y*96 + x - 2];
    sum.a += pixel * Filter_1_Real[3*7 + 5];
	sum.b += pixel * Filter_1_Imag[3*7 + 5];
	sum.c += pixel * Filter_2_Real[3*7 + 5];
	sum.d += pixel * Filter_2_Imag[3*7 + 5];
	sum.e += pixel * Filter_3_Real[3*7 + 5];
	sum.f += pixel * Filter_3_Imag[3*7 + 5];
    pixel = image[(y + 1)*96 + x - 2];
    sum.a += pixel * Filter_1_Real[2*7 + 5];
	sum.b += pixel * Filter_1_Imag[2*7 + 5];
	sum.c += pixel * Filter_2_Real[2*7 + 5];
	sum.d += pixel * Filter_2_Imag[2*7 + 5];
	sum.e += pixel * Filter_3_Real[2*7 + 5];
	sum.f += pixel * Filter_3_Imag[2*7 + 5];
	pixel = image[(y + 2)*96 + x - 2];
    sum.a += pixel * Filter_1_Real[1*7 + 5];
	sum.b += pixel * Filter_1_Imag[1*7 + 5];
	sum.c += pixel * Filter_2_Real[1*7 + 5];
	sum.d += pixel * Filter_2_Imag[1*7 + 5];
	sum.e += pixel * Filter_3_Real[1*7 + 5];
	sum.f += pixel * Filter_3_Imag[1*7 + 5];
	pixel = image[(y + 3)*96 + x - 2];
    sum.a += pixel * Filter_1_Real[5];
	sum.b += pixel * Filter_1_Imag[5];
	sum.c += pixel * Filter_2_Real[5];
	sum.d += pixel * Filter_2_Imag[5];
	sum.e += pixel * Filter_3_Real[5];
	sum.f += pixel * Filter_3_Imag[5];

    pixel = image[(y - 3)*96 + x - 1];
    sum.a += pixel * Filter_1_Real[6*7 + 4];
	sum.b += pixel * Filter_1_Imag[6*7 + 4];
	sum.c += pixel * Filter_2_Real[6*7 + 4];
	sum.d += pixel * Filter_2_Imag[6*7 + 4];
	sum.e += pixel * Filter_3_Real[6*7 + 4];
	sum.f += pixel * Filter_3_Imag[6*7 + 4];
    pixel = image[(y - 2)*96 + x - 1];
    sum.a += pixel * Filter_1_Real[5*7 + 4];
	sum.b += pixel * Filter_1_Imag[5*7 + 4];
	sum.c += pixel * Filter_2_Real[5*7 + 4];
	sum.d += pixel * Filter_2_Imag[5*7 + 4];
	sum.e += pixel * Filter_3_Real[5*7 + 4];
	sum.f += pixel * Filter_3_Imag[5*7 + 4];
	pixel = image[(y - 1)*96 + x - 1];
    sum.a += pixel * Filter_1_Real[4*7 + 4];
	sum.b += pixel * Filter_1_Imag[4*7 + 4];
	sum.c += pixel * Filter_2_Real[4*7 + 4];
	sum.d += pixel * Filter_2_Imag[4*7 + 4];
	sum.e += pixel * Filter_3_Real[4*7 + 4];
	sum.f += pixel * Filter_3_Imag[4*7 + 4];
	pixel = image[y*96 + x - 1];
    sum.a += pixel * Filter_1_Real[3*7 + 4];
	sum.b += pixel * Filter_1_Imag[3*7 + 4];
	sum.c += pixel * Filter_2_Real[3*7 + 4];
	sum.d += pixel * Filter_2_Imag[3*7 + 4];
	sum.e += pixel * Filter_3_Real[3*7 + 4];
	sum.f += pixel * Filter_3_Imag[3*7 + 4];
    pixel = image[(y + 1)*96 + x - 1];
    sum.a += pixel * Filter_1_Real[2*7 + 4];
	sum.b += pixel * Filter_1_Imag[2*7 + 4];
	sum.c += pixel * Filter_2_Real[2*7 + 4];
	sum.d += pixel * Filter_2_Imag[2*7 + 4];
	sum.e += pixel * Filter_3_Real[2*7 + 4];
	sum.f += pixel * Filter_3_Imag[2*7 + 4];
	pixel = image[(y + 2)*96 + x - 1];
    sum.a += pixel * Filter_1_Real[1*7 + 4];
	sum.b += pixel * Filter_1_Imag[1*7 + 4];
	sum.c += pixel * Filter_2_Real[1*7 + 4];
	sum.d += pixel * Filter_2_Imag[1*7 + 4];
	sum.e += pixel * Filter_3_Real[1*7 + 4];
	sum.f += pixel * Filter_3_Imag[1*7 + 4];
	pixel = image[(y + 3)*96 + x - 1];
    sum.a += pixel * Filter_1_Real[4];
	sum.b += pixel * Filter_1_Imag[4];
	sum.c += pixel * Filter_2_Real[4];
	sum.d += pixel * Filter_2_Imag[4];
	sum.e += pixel * Filter_3_Real[4];
	sum.f += pixel * Filter_3_Imag[4];

    pixel = image[(y - 3)*96 + x];
    sum.a += pixel * Filter_1_Real[6*7 + 3];
	sum.b += pixel * Filter_1_Imag[6*7 + 3];
	sum.c += pixel * Filter_2_Real[6*7 + 3];
	sum.d += pixel * Filter_2_Imag[6*7 + 3];
	sum.e += pixel * Filter_3_Real[6*7 + 3];
	sum.f += pixel * Filter_3_Imag[6*7 + 3];
    pixel = image[(y - 2)*96 + x];
    sum.a += pixel * Filter_1_Real[5*7 + 3];
	sum.b += pixel * Filter_1_Imag[5*7 + 3];
	sum.c += pixel * Filter_2_Real[5*7 + 3];
	sum.d += pixel * Filter_2_Imag[5*7 + 3];
	sum.e += pixel * Filter_3_Real[5*7 + 3];
	sum.f += pixel * Filter_3_Imag[5*7 + 3];
	pixel = image[(y - 1)*96 + x];
    sum.a += pixel * Filter_1_Real[4*7 + 3];
	sum.b += pixel * Filter_1_Imag[4*7 + 3];
	sum.c += pixel * Filter_2_Real[4*7 + 3];
	sum.d += pixel * Filter_2_Imag[4*7 + 3];
	sum.e += pixel * Filter_3_Real[4*7 + 3];
	sum.f += pixel * Filter_3_Imag[4*7 + 3];
	pixel = image[y*96 + x];
    sum.a += pixel * Filter_1_Real[3*7 + 3];
	sum.b += pixel * Filter_1_Imag[3*7 + 3];
	sum.c += pixel * Filter_2_Real[3*7 + 3];
	sum.d += pixel * Filter_2_Imag[3*7 + 3];
	sum.e += pixel * Filter_3_Real[3*7 + 3];
	sum.f += pixel * Filter_3_Imag[3*7 + 3];
    pixel = image[(y + 1)*96 + x];
    sum.a += pixel * Filter_1_Real[2*7 + 3];
	sum.b += pixel * Filter_1_Imag[2*7 + 3];
	sum.c += pixel * Filter_2_Real[2*7 + 3];
	sum.d += pixel * Filter_2_Imag[2*7 + 3];
	sum.e += pixel * Filter_3_Real[2*7 + 3];
	sum.f += pixel * Filter_3_Imag[2*7 + 3];
	pixel = image[(y + 2)*96 + x];
    sum.a += pixel * Filter_1_Real[1*7 + 3];
	sum.b += pixel * Filter_1_Imag[1*7 + 3];
	sum.c += pixel * Filter_2_Real[1*7 + 3];
	sum.d += pixel * Filter_2_Imag[1*7 + 3];
	sum.e += pixel * Filter_3_Real[1*7 + 3];
	sum.f += pixel * Filter_3_Imag[1*7 + 3];
	pixel = image[(y + 3)*96 + x];
    sum.a += pixel * Filter_1_Real[3];
	sum.b += pixel * Filter_1_Imag[3];
	sum.c += pixel * Filter_2_Real[3];
	sum.d += pixel * Filter_2_Imag[3];
	sum.e += pixel * Filter_3_Real[3];
	sum.f += pixel * Filter_3_Imag[3];

    pixel = image[(y - 3)*96 + x + 1];
    sum.a += pixel * Filter_1_Real[6*7 + 2];
	sum.b += pixel * Filter_1_Imag[6*7 + 2];
	sum.c += pixel * Filter_2_Real[6*7 + 2];
	sum.d += pixel * Filter_2_Imag[6*7 + 2];
	sum.e += pixel * Filter_3_Real[6*7 + 2];
	sum.f += pixel * Filter_3_Imag[6*7 + 2];
    pixel = image[(y - 2)*96 + x + 1];
    sum.a += pixel * Filter_1_Real[5*7 + 2];
	sum.b += pixel * Filter_1_Imag[5*7 + 2];
	sum.c += pixel * Filter_2_Real[5*7 + 2];
	sum.d += pixel * Filter_2_Imag[5*7 + 2];
	sum.e += pixel * Filter_3_Real[5*7 + 2];
	sum.f += pixel * Filter_3_Imag[5*7 + 2];
	pixel = image[(y - 1)*96 + x + 1];
    sum.a += pixel * Filter_1_Real[4*7 + 2];
	sum.b += pixel * Filter_1_Imag[4*7 + 2];
	sum.c += pixel * Filter_2_Real[4*7 + 2];
	sum.d += pixel * Filter_2_Imag[4*7 + 2];
	sum.e += pixel * Filter_3_Real[4*7 + 2];
	sum.f += pixel * Filter_3_Imag[4*7 + 2];
	pixel = image[y*96 + x + 1];
    sum.a += pixel * Filter_1_Real[3*7 + 2];
	sum.b += pixel * Filter_1_Imag[3*7 + 2];
	sum.c += pixel * Filter_2_Real[3*7 + 2];
	sum.d += pixel * Filter_2_Imag[3*7 + 2];
	sum.e += pixel * Filter_3_Real[3*7 + 2];
	sum.f += pixel * Filter_3_Imag[3*7 + 2];
    pixel = image[(y + 1)*96 + x + 1];
    sum.a += pixel * Filter_1_Real[2*7 + 2];
	sum.b += pixel * Filter_1_Imag[2*7 + 2];
	sum.c += pixel * Filter_2_Real[2*7 + 2];
	sum.d += pixel * Filter_2_Imag[2*7 + 2];
	sum.e += pixel * Filter_3_Real[2*7 + 2];
	sum.f += pixel * Filter_3_Imag[2*7 + 2];
	pixel = image[(y + 2)*96 + x + 1];
    sum.a += pixel * Filter_1_Real[1*7 + 2];
	sum.b += pixel * Filter_1_Imag[1*7 + 2];
	sum.c += pixel * Filter_2_Real[1*7 + 2];
	sum.d += pixel * Filter_2_Imag[1*7 + 2];
	sum.e += pixel * Filter_3_Real[1*7 + 2];
	sum.f += pixel * Filter_3_Imag[1*7 + 2];
	pixel = image[(y + 3)*96 + x + 1];
    sum.a += pixel * Filter_1_Real[2];
	sum.b += pixel * Filter_1_Imag[2];
	sum.c += pixel * Filter_2_Real[2];
	sum.d += pixel * Filter_2_Imag[2];
	sum.e += pixel * Filter_3_Real[2];
	sum.f += pixel * Filter_3_Imag[2];

    pixel = image[(y - 3)*96 + x + 2];
    sum.a += pixel * Filter_1_Real[6*7 + 1];
	sum.b += pixel * Filter_1_Imag[6*7 + 1];
	sum.c += pixel * Filter_2_Real[6*7 + 1];
	sum.d += pixel * Filter_2_Imag[6*7 + 1];
	sum.e += pixel * Filter_3_Real[6*7 + 1];
	sum.f += pixel * Filter_3_Imag[6*7 + 1];
    pixel = image[(y - 2)*96 + x + 2];
    sum.a += pixel * Filter_1_Real[5*7 + 1];
	sum.b += pixel * Filter_1_Imag[5*7 + 1];
	sum.c += pixel * Filter_2_Real[5*7 + 1];
	sum.d += pixel * Filter_2_Imag[5*7 + 1];
	sum.e += pixel * Filter_3_Real[5*7 + 1];
	sum.f += pixel * Filter_3_Imag[5*7 + 1];
	pixel = image[(y - 1)*96 + x + 2];
    sum.a += pixel * Filter_1_Real[4*7 + 1];
	sum.b += pixel * Filter_1_Imag[4*7 + 1];
	sum.c += pixel * Filter_2_Real[4*7 + 1];
	sum.d += pixel * Filter_2_Imag[4*7 + 1];
	sum.e += pixel * Filter_3_Real[4*7 + 1];
	sum.f += pixel * Filter_3_Imag[4*7 + 1];
	pixel = image[y*96 + x + 2];
    sum.a += pixel * Filter_1_Real[3*7 + 1];
	sum.b += pixel * Filter_1_Imag[3*7 + 1];
	sum.c += pixel * Filter_2_Real[3*7 + 1];
	sum.d += pixel * Filter_2_Imag[3*7 + 1];
	sum.e += pixel * Filter_3_Real[3*7 + 1];
	sum.f += pixel * Filter_3_Imag[3*7 + 1];
    pixel = image[(y + 1)*96 + x + 2];
    sum.a += pixel * Filter_1_Real[2*7 + 1];
	sum.b += pixel * Filter_1_Imag[2*7 + 1];
	sum.c += pixel * Filter_2_Real[2*7 + 1];
	sum.d += pixel * Filter_2_Imag[2*7 + 1];
	sum.e += pixel * Filter_3_Real[2*7 + 1];
	sum.f += pixel * Filter_3_Imag[2*7 + 1];
	pixel = image[(y + 2)*96 + x + 2];
    sum.a += pixel * Filter_1_Real[1*7 + 1];
	sum.b += pixel * Filter_1_Imag[1*7 + 1];
	sum.c += pixel * Filter_2_Real[1*7 + 1];
	sum.d += pixel * Filter_2_Imag[1*7 + 1];
	sum.e += pixel * Filter_3_Real[1*7 + 1];
	sum.f += pixel * Filter_3_Imag[1*7 + 1];
	pixel = image[(y + 3)*96 + x + 2];
    sum.a += pixel * Filter_1_Real[1];
	sum.b += pixel * Filter_1_Imag[1];
	sum.c += pixel * Filter_2_Real[1];
	sum.d += pixel * Filter_2_Imag[1];
	sum.e += pixel * Filter_3_Real[1];
	sum.f += pixel * Filter_3_Imag[1];

    pixel = image[(y - 3)*96 + x + 3];
    sum.a += pixel * Filter_1_Real[6*7];
	sum.b += pixel * Filter_1_Imag[6*7];
	sum.c += pixel * Filter_2_Real[6*7];
	sum.d += pixel * Filter_2_Imag[6*7];
	sum.e += pixel * Filter_3_Real[6*7];
	sum.f += pixel * Filter_3_Imag[6*7];
    pixel = image[(y - 2)*96 + x + 3];
    sum.a += pixel * Filter_1_Real[5*7];
	sum.b += pixel * Filter_1_Imag[5*7];
	sum.c += pixel * Filter_2_Real[5*7];
	sum.d += pixel * Filter_2_Imag[5*7];
	sum.e += pixel * Filter_3_Real[5*7];
	sum.f += pixel * Filter_3_Imag[5*7];
	pixel = image[(y - 1)*96 + x + 3];
    sum.a += pixel * Filter_1_Real[4*7];
	sum.b += pixel * Filter_1_Imag[4*7];
	sum.c += pixel * Filter_2_Real[4*7];
	sum.d += pixel * Filter_2_Imag[4*7];
	sum.e += pixel * Filter_3_Real[4*7];
	sum.f += pixel * Filter_3_Imag[4*7];
	pixel = image[y*96 + x + 3];
    sum.a += pixel * Filter_1_Real[3*7];
	sum.b += pixel * Filter_1_Imag[3*7];
	sum.c += pixel * Filter_2_Real[3*7];
	sum.d += pixel * Filter_2_Imag[3*7];
	sum.e += pixel * Filter_3_Real[3*7];
	sum.f += pixel * Filter_3_Imag[3*7];
    pixel = image[(y + 1)*96 + x + 3];
    sum.a += pixel * Filter_1_Real[2*7];
	sum.b += pixel * Filter_1_Imag[2*7];
	sum.c += pixel * Filter_2_Real[2*7];
	sum.d += pixel * Filter_2_Imag[2*7];
	sum.e += pixel * Filter_3_Real[2*7];
	sum.f += pixel * Filter_3_Imag[2*7];
	pixel = image[(y + 2)*96 + x + 3];
    sum.a += pixel * Filter_1_Real[1*7];
	sum.b += pixel * Filter_1_Imag[1*7];
	sum.c += pixel * Filter_2_Real[1*7];
	sum.d += pixel * Filter_2_Imag[1*7];
	sum.e += pixel * Filter_3_Real[1*7];
	sum.f += pixel * Filter_3_Imag[1*7];
	pixel = image[(y + 3)*96 + x + 3];
    sum.a += pixel * Filter_1_Real[0];
	sum.b += pixel * Filter_1_Imag[0];
	sum.c += pixel * Filter_2_Real[0];
	sum.d += pixel * Filter_2_Imag[0];
	sum.e += pixel * Filter_3_Real[0];
	sum.f += pixel * Filter_3_Imag[0];

	return sum;
}



// Shared memory is 64 * 128 pixels
float66 Conv_2D_Unrolled_7x7_ThreeFilters_32KB(__local float* image,
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
	float66 sum;
	sum.a = 0.0f;
	sum.b = 0.0f;
	sum.c = 0.0f;
	sum.d = 0.0f;
	sum.e = 0.0f;
	sum.f = 0.0f;
	
    pixel = image[(y - 3)*128 + x - 3];
    sum.a += pixel * Filter_1_Real[6*7 + 6];
	sum.b += pixel * Filter_1_Imag[6*7 + 6];
	sum.c += pixel * Filter_2_Real[6*7 + 6];
	sum.d += pixel * Filter_2_Imag[6*7 + 6];
	sum.e += pixel * Filter_3_Real[6*7 + 6];
	sum.f += pixel * Filter_3_Imag[6*7 + 6];
    pixel = image[(y - 2)*128 + x - 3];
    sum.a += pixel * Filter_1_Real[5*7 + 6];
	sum.b += pixel * Filter_1_Imag[5*7 + 6];
	sum.c += pixel * Filter_2_Real[5*7 + 6];
	sum.d += pixel * Filter_2_Imag[5*7 + 6];
	sum.e += pixel * Filter_3_Real[5*7 + 6];
	sum.f += pixel * Filter_3_Imag[5*7 + 6];
	pixel = image[(y - 1)*128 + x - 3];
    sum.a += pixel * Filter_1_Real[4*7 + 6];
	sum.b += pixel * Filter_1_Imag[4*7 + 6];
	sum.c += pixel * Filter_2_Real[4*7 + 6];
	sum.d += pixel * Filter_2_Imag[4*7 + 6];
	sum.e += pixel * Filter_3_Real[4*7 + 6];
	sum.f += pixel * Filter_3_Imag[4*7 + 6];
	pixel = image[y*128 + x - 3];
    sum.a += pixel * Filter_1_Real[3*7 + 6];
	sum.b += pixel * Filter_1_Imag[3*7 + 6];
	sum.c += pixel * Filter_2_Real[3*7 + 6];
	sum.d += pixel * Filter_2_Imag[3*7 + 6];
	sum.e += pixel * Filter_3_Real[3*7 + 6];
	sum.f += pixel * Filter_3_Imag[3*7 + 6];
    pixel = image[(y + 1)*128 + x - 3];
    sum.a += pixel * Filter_1_Real[2*7 + 6];
	sum.b += pixel * Filter_1_Imag[2*7 + 6];
	sum.c += pixel * Filter_2_Real[2*7 + 6];
	sum.d += pixel * Filter_2_Imag[2*7 + 6];
	sum.e += pixel * Filter_3_Real[2*7 + 6];
	sum.f += pixel * Filter_3_Imag[2*7 + 6];
	pixel = image[(y + 2)*128 + x - 3];
    sum.a += pixel * Filter_1_Real[1*7 + 6];
	sum.b += pixel * Filter_1_Imag[1*7 + 6];
	sum.c += pixel * Filter_2_Real[1*7 + 6];
	sum.d += pixel * Filter_2_Imag[1*7 + 6];
	sum.e += pixel * Filter_3_Real[1*7 + 6];
	sum.f += pixel * Filter_3_Imag[1*7 + 6];
	pixel = image[(y + 3)*128 + x - 3];
    sum.a += pixel * Filter_1_Real[6];
	sum.b += pixel * Filter_1_Imag[6];
	sum.c += pixel * Filter_2_Real[6];
	sum.d += pixel * Filter_2_Imag[6];
	sum.e += pixel * Filter_3_Real[6];
	sum.f += pixel * Filter_3_Imag[6];

    pixel = image[(y - 3)*128 + x - 2];
    sum.a += pixel * Filter_1_Real[6*7 + 5];
	sum.b += pixel * Filter_1_Imag[6*7 + 5];
	sum.c += pixel * Filter_2_Real[6*7 + 5];
	sum.d += pixel * Filter_2_Imag[6*7 + 5];
	sum.e += pixel * Filter_3_Real[6*7 + 5];
	sum.f += pixel * Filter_3_Imag[6*7 + 5];
    pixel = image[(y - 2)*128 + x - 2];
    sum.a += pixel * Filter_1_Real[5*7 + 5];
	sum.b += pixel * Filter_1_Imag[5*7 + 5];
	sum.c += pixel * Filter_2_Real[5*7 + 5];
	sum.d += pixel * Filter_2_Imag[5*7 + 5];
	sum.e += pixel * Filter_3_Real[5*7 + 5];
	sum.f += pixel * Filter_3_Imag[5*7 + 5];
	pixel = image[(y - 1)*128 + x - 2];
    sum.a += pixel * Filter_1_Real[4*7 + 5];
	sum.b += pixel * Filter_1_Imag[4*7 + 5];
	sum.c += pixel * Filter_2_Real[4*7 + 5];
	sum.d += pixel * Filter_2_Imag[4*7 + 5];
	sum.e += pixel * Filter_3_Real[4*7 + 5];
	sum.f += pixel * Filter_3_Imag[4*7 + 5];
	pixel = image[y*128 + x - 2];
    sum.a += pixel * Filter_1_Real[3*7 + 5];
	sum.b += pixel * Filter_1_Imag[3*7 + 5];
	sum.c += pixel * Filter_2_Real[3*7 + 5];
	sum.d += pixel * Filter_2_Imag[3*7 + 5];
	sum.e += pixel * Filter_3_Real[3*7 + 5];
	sum.f += pixel * Filter_3_Imag[3*7 + 5];
    pixel = image[(y + 1)*128 + x - 2];
    sum.a += pixel * Filter_1_Real[2*7 + 5];
	sum.b += pixel * Filter_1_Imag[2*7 + 5];
	sum.c += pixel * Filter_2_Real[2*7 + 5];
	sum.d += pixel * Filter_2_Imag[2*7 + 5];
	sum.e += pixel * Filter_3_Real[2*7 + 5];
	sum.f += pixel * Filter_3_Imag[2*7 + 5];
	pixel = image[(y + 2)*128 + x - 2];
    sum.a += pixel * Filter_1_Real[1*7 + 5];
	sum.b += pixel * Filter_1_Imag[1*7 + 5];
	sum.c += pixel * Filter_2_Real[1*7 + 5];
	sum.d += pixel * Filter_2_Imag[1*7 + 5];
	sum.e += pixel * Filter_3_Real[1*7 + 5];
	sum.f += pixel * Filter_3_Imag[1*7 + 5];
	pixel = image[(y + 3)*128 + x - 2];
    sum.a += pixel * Filter_1_Real[5];
	sum.b += pixel * Filter_1_Imag[5];
	sum.c += pixel * Filter_2_Real[5];
	sum.d += pixel * Filter_2_Imag[5];
	sum.e += pixel * Filter_3_Real[5];
	sum.f += pixel * Filter_3_Imag[5];

    pixel = image[(y - 3)*128 + x - 1];
    sum.a += pixel * Filter_1_Real[6*7 + 4];
	sum.b += pixel * Filter_1_Imag[6*7 + 4];
	sum.c += pixel * Filter_2_Real[6*7 + 4];
	sum.d += pixel * Filter_2_Imag[6*7 + 4];
	sum.e += pixel * Filter_3_Real[6*7 + 4];
	sum.f += pixel * Filter_3_Imag[6*7 + 4];
    pixel = image[(y - 2)*128 + x - 1];
    sum.a += pixel * Filter_1_Real[5*7 + 4];
	sum.b += pixel * Filter_1_Imag[5*7 + 4];
	sum.c += pixel * Filter_2_Real[5*7 + 4];
	sum.d += pixel * Filter_2_Imag[5*7 + 4];
	sum.e += pixel * Filter_3_Real[5*7 + 4];
	sum.f += pixel * Filter_3_Imag[5*7 + 4];
	pixel = image[(y - 1)*128 + x - 1];
    sum.a += pixel * Filter_1_Real[4*7 + 4];
	sum.b += pixel * Filter_1_Imag[4*7 + 4];
	sum.c += pixel * Filter_2_Real[4*7 + 4];
	sum.d += pixel * Filter_2_Imag[4*7 + 4];
	sum.e += pixel * Filter_3_Real[4*7 + 4];
	sum.f += pixel * Filter_3_Imag[4*7 + 4];
	pixel = image[y*128 + x - 1];
    sum.a += pixel * Filter_1_Real[3*7 + 4];
	sum.b += pixel * Filter_1_Imag[3*7 + 4];
	sum.c += pixel * Filter_2_Real[3*7 + 4];
	sum.d += pixel * Filter_2_Imag[3*7 + 4];
	sum.e += pixel * Filter_3_Real[3*7 + 4];
	sum.f += pixel * Filter_3_Imag[3*7 + 4];
    pixel = image[(y + 1)*128 + x - 1];
    sum.a += pixel * Filter_1_Real[2*7 + 4];
	sum.b += pixel * Filter_1_Imag[2*7 + 4];
	sum.c += pixel * Filter_2_Real[2*7 + 4];
	sum.d += pixel * Filter_2_Imag[2*7 + 4];
	sum.e += pixel * Filter_3_Real[2*7 + 4];
	sum.f += pixel * Filter_3_Imag[2*7 + 4];
	pixel = image[(y + 2)*128 + x - 1];
    sum.a += pixel * Filter_1_Real[1*7 + 4];
	sum.b += pixel * Filter_1_Imag[1*7 + 4];
	sum.c += pixel * Filter_2_Real[1*7 + 4];
	sum.d += pixel * Filter_2_Imag[1*7 + 4];
	sum.e += pixel * Filter_3_Real[1*7 + 4];
	sum.f += pixel * Filter_3_Imag[1*7 + 4];
	pixel = image[(y + 3)*128 + x - 1];
    sum.a += pixel * Filter_1_Real[4];
	sum.b += pixel * Filter_1_Imag[4];
	sum.c += pixel * Filter_2_Real[4];
	sum.d += pixel * Filter_2_Imag[4];
	sum.e += pixel * Filter_3_Real[4];
	sum.f += pixel * Filter_3_Imag[4];

    pixel = image[(y - 3)*128 + x];
    sum.a += pixel * Filter_1_Real[6*7 + 3];
	sum.b += pixel * Filter_1_Imag[6*7 + 3];
	sum.c += pixel * Filter_2_Real[6*7 + 3];
	sum.d += pixel * Filter_2_Imag[6*7 + 3];
	sum.e += pixel * Filter_3_Real[6*7 + 3];
	sum.f += pixel * Filter_3_Imag[6*7 + 3];
    pixel = image[(y - 2)*128 + x];
    sum.a += pixel * Filter_1_Real[5*7 + 3];
	sum.b += pixel * Filter_1_Imag[5*7 + 3];
	sum.c += pixel * Filter_2_Real[5*7 + 3];
	sum.d += pixel * Filter_2_Imag[5*7 + 3];
	sum.e += pixel * Filter_3_Real[5*7 + 3];
	sum.f += pixel * Filter_3_Imag[5*7 + 3];
	pixel = image[(y - 1)*128 + x];
    sum.a += pixel * Filter_1_Real[4*7 + 3];
	sum.b += pixel * Filter_1_Imag[4*7 + 3];
	sum.c += pixel * Filter_2_Real[4*7 + 3];
	sum.d += pixel * Filter_2_Imag[4*7 + 3];
	sum.e += pixel * Filter_3_Real[4*7 + 3];
	sum.f += pixel * Filter_3_Imag[4*7 + 3];
	pixel = image[y*128 + x];
    sum.a += pixel * Filter_1_Real[3*7 + 3];
	sum.b += pixel * Filter_1_Imag[3*7 + 3];
	sum.c += pixel * Filter_2_Real[3*7 + 3];
	sum.d += pixel * Filter_2_Imag[3*7 + 3];
	sum.e += pixel * Filter_3_Real[3*7 + 3];
	sum.f += pixel * Filter_3_Imag[3*7 + 3];
    pixel = image[(y + 1)*128 + x];
    sum.a += pixel * Filter_1_Real[2*7 + 3];
	sum.b += pixel * Filter_1_Imag[2*7 + 3];
	sum.c += pixel * Filter_2_Real[2*7 + 3];
	sum.d += pixel * Filter_2_Imag[2*7 + 3];
	sum.e += pixel * Filter_3_Real[2*7 + 3];
	sum.f += pixel * Filter_3_Imag[2*7 + 3];
	pixel = image[(y + 2)*128 + x];
    sum.a += pixel * Filter_1_Real[1*7 + 3];
	sum.b += pixel * Filter_1_Imag[1*7 + 3];
	sum.c += pixel * Filter_2_Real[1*7 + 3];
	sum.d += pixel * Filter_2_Imag[1*7 + 3];
	sum.e += pixel * Filter_3_Real[1*7 + 3];
	sum.f += pixel * Filter_3_Imag[1*7 + 3];
	pixel = image[(y + 3)*128 + x];
    sum.a += pixel * Filter_1_Real[3];
	sum.b += pixel * Filter_1_Imag[3];
	sum.c += pixel * Filter_2_Real[3];
	sum.d += pixel * Filter_2_Imag[3];
	sum.e += pixel * Filter_3_Real[3];
	sum.f += pixel * Filter_3_Imag[3];

    pixel = image[(y - 3)*128 + x + 1];
    sum.a += pixel * Filter_1_Real[6*7 + 2];
	sum.b += pixel * Filter_1_Imag[6*7 + 2];
	sum.c += pixel * Filter_2_Real[6*7 + 2];
	sum.d += pixel * Filter_2_Imag[6*7 + 2];
	sum.e += pixel * Filter_3_Real[6*7 + 2];
	sum.f += pixel * Filter_3_Imag[6*7 + 2];
    pixel = image[(y - 2)*128 + x + 1];
    sum.a += pixel * Filter_1_Real[5*7 + 2];
	sum.b += pixel * Filter_1_Imag[5*7 + 2];
	sum.c += pixel * Filter_2_Real[5*7 + 2];
	sum.d += pixel * Filter_2_Imag[5*7 + 2];
	sum.e += pixel * Filter_3_Real[5*7 + 2];
	sum.f += pixel * Filter_3_Imag[5*7 + 2];
	pixel = image[(y - 1)*128 + x + 1];
    sum.a += pixel * Filter_1_Real[4*7 + 2];
	sum.b += pixel * Filter_1_Imag[4*7 + 2];
	sum.c += pixel * Filter_2_Real[4*7 + 2];
	sum.d += pixel * Filter_2_Imag[4*7 + 2];
	sum.e += pixel * Filter_3_Real[4*7 + 2];
	sum.f += pixel * Filter_3_Imag[4*7 + 2];
	pixel = image[y*128 + x + 1];
    sum.a += pixel * Filter_1_Real[3*7 + 2];
	sum.b += pixel * Filter_1_Imag[3*7 + 2];
	sum.c += pixel * Filter_2_Real[3*7 + 2];
	sum.d += pixel * Filter_2_Imag[3*7 + 2];
	sum.e += pixel * Filter_3_Real[3*7 + 2];
	sum.f += pixel * Filter_3_Imag[3*7 + 2];
    pixel = image[(y + 1)*128 + x + 1];
    sum.a += pixel * Filter_1_Real[2*7 + 2];
	sum.b += pixel * Filter_1_Imag[2*7 + 2];
	sum.c += pixel * Filter_2_Real[2*7 + 2];
	sum.d += pixel * Filter_2_Imag[2*7 + 2];
	sum.e += pixel * Filter_3_Real[2*7 + 2];
	sum.f += pixel * Filter_3_Imag[2*7 + 2];
	pixel = image[(y + 2)*128 + x + 1];
    sum.a += pixel * Filter_1_Real[1*7 + 2];
	sum.b += pixel * Filter_1_Imag[1*7 + 2];
	sum.c += pixel * Filter_2_Real[1*7 + 2];
	sum.d += pixel * Filter_2_Imag[1*7 + 2];
	sum.e += pixel * Filter_3_Real[1*7 + 2];
	sum.f += pixel * Filter_3_Imag[1*7 + 2];
	pixel = image[(y + 3)*128 + x + 1];
    sum.a += pixel * Filter_1_Real[2];
	sum.b += pixel * Filter_1_Imag[2];
	sum.c += pixel * Filter_2_Real[2];
	sum.d += pixel * Filter_2_Imag[2];
	sum.e += pixel * Filter_3_Real[2];
	sum.f += pixel * Filter_3_Imag[2];

    pixel = image[(y - 3)*128 + x + 2];
    sum.a += pixel * Filter_1_Real[6*7 + 1];
	sum.b += pixel * Filter_1_Imag[6*7 + 1];
	sum.c += pixel * Filter_2_Real[6*7 + 1];
	sum.d += pixel * Filter_2_Imag[6*7 + 1];
	sum.e += pixel * Filter_3_Real[6*7 + 1];
	sum.f += pixel * Filter_3_Imag[6*7 + 1];
    pixel = image[(y - 2)*128 + x + 2];
    sum.a += pixel * Filter_1_Real[5*7 + 1];
	sum.b += pixel * Filter_1_Imag[5*7 + 1];
	sum.c += pixel * Filter_2_Real[5*7 + 1];
	sum.d += pixel * Filter_2_Imag[5*7 + 1];
	sum.e += pixel * Filter_3_Real[5*7 + 1];
	sum.f += pixel * Filter_3_Imag[5*7 + 1];
	pixel = image[(y - 1)*128 + x + 2];
    sum.a += pixel * Filter_1_Real[4*7 + 1];
	sum.b += pixel * Filter_1_Imag[4*7 + 1];
	sum.c += pixel * Filter_2_Real[4*7 + 1];
	sum.d += pixel * Filter_2_Imag[4*7 + 1];
	sum.e += pixel * Filter_3_Real[4*7 + 1];
	sum.f += pixel * Filter_3_Imag[4*7 + 1];
	pixel = image[y*128 + x + 2];
    sum.a += pixel * Filter_1_Real[3*7 + 1];
	sum.b += pixel * Filter_1_Imag[3*7 + 1];
	sum.c += pixel * Filter_2_Real[3*7 + 1];
	sum.d += pixel * Filter_2_Imag[3*7 + 1];
	sum.e += pixel * Filter_3_Real[3*7 + 1];
	sum.f += pixel * Filter_3_Imag[3*7 + 1];
    pixel = image[(y + 1)*128 + x + 2];
    sum.a += pixel * Filter_1_Real[2*7 + 1];
	sum.b += pixel * Filter_1_Imag[2*7 + 1];
	sum.c += pixel * Filter_2_Real[2*7 + 1];
	sum.d += pixel * Filter_2_Imag[2*7 + 1];
	sum.e += pixel * Filter_3_Real[2*7 + 1];
	sum.f += pixel * Filter_3_Imag[2*7 + 1];
	pixel = image[(y + 2)*128 + x + 2];
    sum.a += pixel * Filter_1_Real[1*7 + 1];
	sum.b += pixel * Filter_1_Imag[1*7 + 1];
	sum.c += pixel * Filter_2_Real[1*7 + 1];
	sum.d += pixel * Filter_2_Imag[1*7 + 1];
	sum.e += pixel * Filter_3_Real[1*7 + 1];
	sum.f += pixel * Filter_3_Imag[1*7 + 1];
	pixel = image[(y + 3)*128 + x + 2];
    sum.a += pixel * Filter_1_Real[1];
	sum.b += pixel * Filter_1_Imag[1];
	sum.c += pixel * Filter_2_Real[1];
	sum.d += pixel * Filter_2_Imag[1];
	sum.e += pixel * Filter_3_Real[1];
	sum.f += pixel * Filter_3_Imag[1];

    pixel = image[(y - 3)*128 + x + 3];
    sum.a += pixel * Filter_1_Real[6*7];
	sum.b += pixel * Filter_1_Imag[6*7];
	sum.c += pixel * Filter_2_Real[6*7];
	sum.d += pixel * Filter_2_Imag[6*7];
	sum.e += pixel * Filter_3_Real[6*7];
	sum.f += pixel * Filter_3_Imag[6*7];
    pixel = image[(y - 2)*128 + x + 3];
    sum.a += pixel * Filter_1_Real[5*7];
	sum.b += pixel * Filter_1_Imag[5*7];
	sum.c += pixel * Filter_2_Real[5*7];
	sum.d += pixel * Filter_2_Imag[5*7];
	sum.e += pixel * Filter_3_Real[5*7];
	sum.f += pixel * Filter_3_Imag[5*7];
	pixel = image[(y - 1)*128 + x + 3];
    sum.a += pixel * Filter_1_Real[4*7];
	sum.b += pixel * Filter_1_Imag[4*7];
	sum.c += pixel * Filter_2_Real[4*7];
	sum.d += pixel * Filter_2_Imag[4*7];
	sum.e += pixel * Filter_3_Real[4*7];
	sum.f += pixel * Filter_3_Imag[4*7];
	pixel = image[y*128 + x + 3];
    sum.a += pixel * Filter_1_Real[3*7];
	sum.b += pixel * Filter_1_Imag[3*7];
	sum.c += pixel * Filter_2_Real[3*7];
	sum.d += pixel * Filter_2_Imag[3*7];
	sum.e += pixel * Filter_3_Real[3*7];
	sum.f += pixel * Filter_3_Imag[3*7];
    pixel = image[(y + 1)*128 + x + 3];
    sum.a += pixel * Filter_1_Real[2*7];
	sum.b += pixel * Filter_1_Imag[2*7];
	sum.c += pixel * Filter_2_Real[2*7];
	sum.d += pixel * Filter_2_Imag[2*7];
	sum.e += pixel * Filter_3_Real[2*7];
	sum.f += pixel * Filter_3_Imag[2*7];
	pixel = image[(y + 2)*128 + x + 3];
    sum.a += pixel * Filter_1_Real[1*7];
	sum.b += pixel * Filter_1_Imag[1*7];
	sum.c += pixel * Filter_2_Real[1*7];
	sum.d += pixel * Filter_2_Imag[1*7];
	sum.e += pixel * Filter_3_Real[1*7];
	sum.f += pixel * Filter_3_Imag[1*7];
	pixel = image[(y + 3)*128 + x + 3];
    sum.a += pixel * Filter_1_Real[0];
	sum.b += pixel * Filter_1_Imag[0];
	sum.c += pixel * Filter_2_Real[0];
	sum.d += pixel * Filter_2_Imag[0];
	sum.e += pixel * Filter_3_Real[0];
	sum.f += pixel * Filter_3_Imag[0];

	return sum;
}

// Shared memory is 128 * 128 pixels
float66 Conv_2D_Unrolled_7x7_ThreeFilters_64KB(__local float* image,
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
	float66 sum;
	sum.a = 0.0f;
	sum.b = 0.0f;
	sum.c = 0.0f;
	sum.d = 0.0f;
	sum.e = 0.0f;
	sum.f = 0.0f;
	
    pixel = image[(y - 3)*128 + x - 3];
    sum.a += pixel * Filter_1_Real[6*7 + 6];
	sum.b += pixel * Filter_1_Imag[6*7 + 6];
	sum.c += pixel * Filter_2_Real[6*7 + 6];
	sum.d += pixel * Filter_2_Imag[6*7 + 6];
	sum.e += pixel * Filter_3_Real[6*7 + 6];
	sum.f += pixel * Filter_3_Imag[6*7 + 6];
    pixel = image[(y - 2)*128 + x - 3];
    sum.a += pixel * Filter_1_Real[5*7 + 6];
	sum.b += pixel * Filter_1_Imag[5*7 + 6];
	sum.c += pixel * Filter_2_Real[5*7 + 6];
	sum.d += pixel * Filter_2_Imag[5*7 + 6];
	sum.e += pixel * Filter_3_Real[5*7 + 6];
	sum.f += pixel * Filter_3_Imag[5*7 + 6];
	pixel = image[(y - 1)*128 + x - 3];
    sum.a += pixel * Filter_1_Real[4*7 + 6];
	sum.b += pixel * Filter_1_Imag[4*7 + 6];
	sum.c += pixel * Filter_2_Real[4*7 + 6];
	sum.d += pixel * Filter_2_Imag[4*7 + 6];
	sum.e += pixel * Filter_3_Real[4*7 + 6];
	sum.f += pixel * Filter_3_Imag[4*7 + 6];
	pixel = image[y*128 + x - 3];
    sum.a += pixel * Filter_1_Real[3*7 + 6];
	sum.b += pixel * Filter_1_Imag[3*7 + 6];
	sum.c += pixel * Filter_2_Real[3*7 + 6];
	sum.d += pixel * Filter_2_Imag[3*7 + 6];
	sum.e += pixel * Filter_3_Real[3*7 + 6];
	sum.f += pixel * Filter_3_Imag[3*7 + 6];
    pixel = image[(y + 1)*128 + x - 3];
    sum.a += pixel * Filter_1_Real[2*7 + 6];
	sum.b += pixel * Filter_1_Imag[2*7 + 6];
	sum.c += pixel * Filter_2_Real[2*7 + 6];
	sum.d += pixel * Filter_2_Imag[2*7 + 6];
	sum.e += pixel * Filter_3_Real[2*7 + 6];
	sum.f += pixel * Filter_3_Imag[2*7 + 6];
	pixel = image[(y + 2)*128 + x - 3];
    sum.a += pixel * Filter_1_Real[1*7 + 6];
	sum.b += pixel * Filter_1_Imag[1*7 + 6];
	sum.c += pixel * Filter_2_Real[1*7 + 6];
	sum.d += pixel * Filter_2_Imag[1*7 + 6];
	sum.e += pixel * Filter_3_Real[1*7 + 6];
	sum.f += pixel * Filter_3_Imag[1*7 + 6];
	pixel = image[(y + 3)*128 + x - 3];
    sum.a += pixel * Filter_1_Real[6];
	sum.b += pixel * Filter_1_Imag[6];
	sum.c += pixel * Filter_2_Real[6];
	sum.d += pixel * Filter_2_Imag[6];
	sum.e += pixel * Filter_3_Real[6];
	sum.f += pixel * Filter_3_Imag[6];

    pixel = image[(y - 3)*128 + x - 2];
    sum.a += pixel * Filter_1_Real[6*7 + 5];
	sum.b += pixel * Filter_1_Imag[6*7 + 5];
	sum.c += pixel * Filter_2_Real[6*7 + 5];
	sum.d += pixel * Filter_2_Imag[6*7 + 5];
	sum.e += pixel * Filter_3_Real[6*7 + 5];
	sum.f += pixel * Filter_3_Imag[6*7 + 5];
    pixel = image[(y - 2)*128 + x - 2];
    sum.a += pixel * Filter_1_Real[5*7 + 5];
	sum.b += pixel * Filter_1_Imag[5*7 + 5];
	sum.c += pixel * Filter_2_Real[5*7 + 5];
	sum.d += pixel * Filter_2_Imag[5*7 + 5];
	sum.e += pixel * Filter_3_Real[5*7 + 5];
	sum.f += pixel * Filter_3_Imag[5*7 + 5];
	pixel = image[(y - 1)*128 + x - 2];
    sum.a += pixel * Filter_1_Real[4*7 + 5];
	sum.b += pixel * Filter_1_Imag[4*7 + 5];
	sum.c += pixel * Filter_2_Real[4*7 + 5];
	sum.d += pixel * Filter_2_Imag[4*7 + 5];
	sum.e += pixel * Filter_3_Real[4*7 + 5];
	sum.f += pixel * Filter_3_Imag[4*7 + 5];
	pixel = image[y*128 + x - 2];
    sum.a += pixel * Filter_1_Real[3*7 + 5];
	sum.b += pixel * Filter_1_Imag[3*7 + 5];
	sum.c += pixel * Filter_2_Real[3*7 + 5];
	sum.d += pixel * Filter_2_Imag[3*7 + 5];
	sum.e += pixel * Filter_3_Real[3*7 + 5];
	sum.f += pixel * Filter_3_Imag[3*7 + 5];
    pixel = image[(y + 1)*128 + x - 2];
    sum.a += pixel * Filter_1_Real[2*7 + 5];
	sum.b += pixel * Filter_1_Imag[2*7 + 5];
	sum.c += pixel * Filter_2_Real[2*7 + 5];
	sum.d += pixel * Filter_2_Imag[2*7 + 5];
	sum.e += pixel * Filter_3_Real[2*7 + 5];
	sum.f += pixel * Filter_3_Imag[2*7 + 5];
	pixel = image[(y + 2)*128 + x - 2];
    sum.a += pixel * Filter_1_Real[1*7 + 5];
	sum.b += pixel * Filter_1_Imag[1*7 + 5];
	sum.c += pixel * Filter_2_Real[1*7 + 5];
	sum.d += pixel * Filter_2_Imag[1*7 + 5];
	sum.e += pixel * Filter_3_Real[1*7 + 5];
	sum.f += pixel * Filter_3_Imag[1*7 + 5];
	pixel = image[(y + 3)*128 + x - 2];
    sum.a += pixel * Filter_1_Real[5];
	sum.b += pixel * Filter_1_Imag[5];
	sum.c += pixel * Filter_2_Real[5];
	sum.d += pixel * Filter_2_Imag[5];
	sum.e += pixel * Filter_3_Real[5];
	sum.f += pixel * Filter_3_Imag[5];

    pixel = image[(y - 3)*128 + x - 1];
    sum.a += pixel * Filter_1_Real[6*7 + 4];
	sum.b += pixel * Filter_1_Imag[6*7 + 4];
	sum.c += pixel * Filter_2_Real[6*7 + 4];
	sum.d += pixel * Filter_2_Imag[6*7 + 4];
	sum.e += pixel * Filter_3_Real[6*7 + 4];
	sum.f += pixel * Filter_3_Imag[6*7 + 4];
    pixel = image[(y - 2)*128 + x - 1];
    sum.a += pixel * Filter_1_Real[5*7 + 4];
	sum.b += pixel * Filter_1_Imag[5*7 + 4];
	sum.c += pixel * Filter_2_Real[5*7 + 4];
	sum.d += pixel * Filter_2_Imag[5*7 + 4];
	sum.e += pixel * Filter_3_Real[5*7 + 4];
	sum.f += pixel * Filter_3_Imag[5*7 + 4];
	pixel = image[(y - 1)*128 + x - 1];
    sum.a += pixel * Filter_1_Real[4*7 + 4];
	sum.b += pixel * Filter_1_Imag[4*7 + 4];
	sum.c += pixel * Filter_2_Real[4*7 + 4];
	sum.d += pixel * Filter_2_Imag[4*7 + 4];
	sum.e += pixel * Filter_3_Real[4*7 + 4];
	sum.f += pixel * Filter_3_Imag[4*7 + 4];
	pixel = image[y*128 + x - 1];
    sum.a += pixel * Filter_1_Real[3*7 + 4];
	sum.b += pixel * Filter_1_Imag[3*7 + 4];
	sum.c += pixel * Filter_2_Real[3*7 + 4];
	sum.d += pixel * Filter_2_Imag[3*7 + 4];
	sum.e += pixel * Filter_3_Real[3*7 + 4];
	sum.f += pixel * Filter_3_Imag[3*7 + 4];
    pixel = image[(y + 1)*128 + x - 1];
    sum.a += pixel * Filter_1_Real[2*7 + 4];
	sum.b += pixel * Filter_1_Imag[2*7 + 4];
	sum.c += pixel * Filter_2_Real[2*7 + 4];
	sum.d += pixel * Filter_2_Imag[2*7 + 4];
	sum.e += pixel * Filter_3_Real[2*7 + 4];
	sum.f += pixel * Filter_3_Imag[2*7 + 4];
	pixel = image[(y + 2)*128 + x - 1];
    sum.a += pixel * Filter_1_Real[1*7 + 4];
	sum.b += pixel * Filter_1_Imag[1*7 + 4];
	sum.c += pixel * Filter_2_Real[1*7 + 4];
	sum.d += pixel * Filter_2_Imag[1*7 + 4];
	sum.e += pixel * Filter_3_Real[1*7 + 4];
	sum.f += pixel * Filter_3_Imag[1*7 + 4];
	pixel = image[(y + 3)*128 + x - 1];
    sum.a += pixel * Filter_1_Real[4];
	sum.b += pixel * Filter_1_Imag[4];
	sum.c += pixel * Filter_2_Real[4];
	sum.d += pixel * Filter_2_Imag[4];
	sum.e += pixel * Filter_3_Real[4];
	sum.f += pixel * Filter_3_Imag[4];

    pixel = image[(y - 3)*128 + x];
    sum.a += pixel * Filter_1_Real[6*7 + 3];
	sum.b += pixel * Filter_1_Imag[6*7 + 3];
	sum.c += pixel * Filter_2_Real[6*7 + 3];
	sum.d += pixel * Filter_2_Imag[6*7 + 3];
	sum.e += pixel * Filter_3_Real[6*7 + 3];
	sum.f += pixel * Filter_3_Imag[6*7 + 3];
    pixel = image[(y - 2)*128 + x];
    sum.a += pixel * Filter_1_Real[5*7 + 3];
	sum.b += pixel * Filter_1_Imag[5*7 + 3];
	sum.c += pixel * Filter_2_Real[5*7 + 3];
	sum.d += pixel * Filter_2_Imag[5*7 + 3];
	sum.e += pixel * Filter_3_Real[5*7 + 3];
	sum.f += pixel * Filter_3_Imag[5*7 + 3];
	pixel = image[(y - 1)*128 + x];
    sum.a += pixel * Filter_1_Real[4*7 + 3];
	sum.b += pixel * Filter_1_Imag[4*7 + 3];
	sum.c += pixel * Filter_2_Real[4*7 + 3];
	sum.d += pixel * Filter_2_Imag[4*7 + 3];
	sum.e += pixel * Filter_3_Real[4*7 + 3];
	sum.f += pixel * Filter_3_Imag[4*7 + 3];
	pixel = image[y*128 + x];
    sum.a += pixel * Filter_1_Real[3*7 + 3];
	sum.b += pixel * Filter_1_Imag[3*7 + 3];
	sum.c += pixel * Filter_2_Real[3*7 + 3];
	sum.d += pixel * Filter_2_Imag[3*7 + 3];
	sum.e += pixel * Filter_3_Real[3*7 + 3];
	sum.f += pixel * Filter_3_Imag[3*7 + 3];
    pixel = image[(y + 1)*128 + x];
    sum.a += pixel * Filter_1_Real[2*7 + 3];
	sum.b += pixel * Filter_1_Imag[2*7 + 3];
	sum.c += pixel * Filter_2_Real[2*7 + 3];
	sum.d += pixel * Filter_2_Imag[2*7 + 3];
	sum.e += pixel * Filter_3_Real[2*7 + 3];
	sum.f += pixel * Filter_3_Imag[2*7 + 3];
	pixel = image[(y + 2)*128 + x];
    sum.a += pixel * Filter_1_Real[1*7 + 3];
	sum.b += pixel * Filter_1_Imag[1*7 + 3];
	sum.c += pixel * Filter_2_Real[1*7 + 3];
	sum.d += pixel * Filter_2_Imag[1*7 + 3];
	sum.e += pixel * Filter_3_Real[1*7 + 3];
	sum.f += pixel * Filter_3_Imag[1*7 + 3];
	pixel = image[(y + 3)*128 + x];
    sum.a += pixel * Filter_1_Real[3];
	sum.b += pixel * Filter_1_Imag[3];
	sum.c += pixel * Filter_2_Real[3];
	sum.d += pixel * Filter_2_Imag[3];
	sum.e += pixel * Filter_3_Real[3];
	sum.f += pixel * Filter_3_Imag[3];

    pixel = image[(y - 3)*128 + x + 1];
    sum.a += pixel * Filter_1_Real[6*7 + 2];
	sum.b += pixel * Filter_1_Imag[6*7 + 2];
	sum.c += pixel * Filter_2_Real[6*7 + 2];
	sum.d += pixel * Filter_2_Imag[6*7 + 2];
	sum.e += pixel * Filter_3_Real[6*7 + 2];
	sum.f += pixel * Filter_3_Imag[6*7 + 2];
    pixel = image[(y - 2)*128 + x + 1];
    sum.a += pixel * Filter_1_Real[5*7 + 2];
	sum.b += pixel * Filter_1_Imag[5*7 + 2];
	sum.c += pixel * Filter_2_Real[5*7 + 2];
	sum.d += pixel * Filter_2_Imag[5*7 + 2];
	sum.e += pixel * Filter_3_Real[5*7 + 2];
	sum.f += pixel * Filter_3_Imag[5*7 + 2];
	pixel = image[(y - 1)*128 + x + 1];
    sum.a += pixel * Filter_1_Real[4*7 + 2];
	sum.b += pixel * Filter_1_Imag[4*7 + 2];
	sum.c += pixel * Filter_2_Real[4*7 + 2];
	sum.d += pixel * Filter_2_Imag[4*7 + 2];
	sum.e += pixel * Filter_3_Real[4*7 + 2];
	sum.f += pixel * Filter_3_Imag[4*7 + 2];
	pixel = image[y*128 + x + 1];
    sum.a += pixel * Filter_1_Real[3*7 + 2];
	sum.b += pixel * Filter_1_Imag[3*7 + 2];
	sum.c += pixel * Filter_2_Real[3*7 + 2];
	sum.d += pixel * Filter_2_Imag[3*7 + 2];
	sum.e += pixel * Filter_3_Real[3*7 + 2];
	sum.f += pixel * Filter_3_Imag[3*7 + 2];
    pixel = image[(y + 1)*128 + x + 1];
    sum.a += pixel * Filter_1_Real[2*7 + 2];
	sum.b += pixel * Filter_1_Imag[2*7 + 2];
	sum.c += pixel * Filter_2_Real[2*7 + 2];
	sum.d += pixel * Filter_2_Imag[2*7 + 2];
	sum.e += pixel * Filter_3_Real[2*7 + 2];
	sum.f += pixel * Filter_3_Imag[2*7 + 2];
	pixel = image[(y + 2)*128 + x + 1];
    sum.a += pixel * Filter_1_Real[1*7 + 2];
	sum.b += pixel * Filter_1_Imag[1*7 + 2];
	sum.c += pixel * Filter_2_Real[1*7 + 2];
	sum.d += pixel * Filter_2_Imag[1*7 + 2];
	sum.e += pixel * Filter_3_Real[1*7 + 2];
	sum.f += pixel * Filter_3_Imag[1*7 + 2];
	pixel = image[(y + 3)*128 + x + 1];
    sum.a += pixel * Filter_1_Real[2];
	sum.b += pixel * Filter_1_Imag[2];
	sum.c += pixel * Filter_2_Real[2];
	sum.d += pixel * Filter_2_Imag[2];
	sum.e += pixel * Filter_3_Real[2];
	sum.f += pixel * Filter_3_Imag[2];

    pixel = image[(y - 3)*128 + x + 2];
    sum.a += pixel * Filter_1_Real[6*7 + 1];
	sum.b += pixel * Filter_1_Imag[6*7 + 1];
	sum.c += pixel * Filter_2_Real[6*7 + 1];
	sum.d += pixel * Filter_2_Imag[6*7 + 1];
	sum.e += pixel * Filter_3_Real[6*7 + 1];
	sum.f += pixel * Filter_3_Imag[6*7 + 1];
    pixel = image[(y - 2)*128 + x + 2];
    sum.a += pixel * Filter_1_Real[5*7 + 1];
	sum.b += pixel * Filter_1_Imag[5*7 + 1];
	sum.c += pixel * Filter_2_Real[5*7 + 1];
	sum.d += pixel * Filter_2_Imag[5*7 + 1];
	sum.e += pixel * Filter_3_Real[5*7 + 1];
	sum.f += pixel * Filter_3_Imag[5*7 + 1];
	pixel = image[(y - 1)*128 + x + 2];
    sum.a += pixel * Filter_1_Real[4*7 + 1];
	sum.b += pixel * Filter_1_Imag[4*7 + 1];
	sum.c += pixel * Filter_2_Real[4*7 + 1];
	sum.d += pixel * Filter_2_Imag[4*7 + 1];
	sum.e += pixel * Filter_3_Real[4*7 + 1];
	sum.f += pixel * Filter_3_Imag[4*7 + 1];
	pixel = image[y*128 + x + 2];
    sum.a += pixel * Filter_1_Real[3*7 + 1];
	sum.b += pixel * Filter_1_Imag[3*7 + 1];
	sum.c += pixel * Filter_2_Real[3*7 + 1];
	sum.d += pixel * Filter_2_Imag[3*7 + 1];
	sum.e += pixel * Filter_3_Real[3*7 + 1];
	sum.f += pixel * Filter_3_Imag[3*7 + 1];
    pixel = image[(y + 1)*128 + x + 2];
    sum.a += pixel * Filter_1_Real[2*7 + 1];
	sum.b += pixel * Filter_1_Imag[2*7 + 1];
	sum.c += pixel * Filter_2_Real[2*7 + 1];
	sum.d += pixel * Filter_2_Imag[2*7 + 1];
	sum.e += pixel * Filter_3_Real[2*7 + 1];
	sum.f += pixel * Filter_3_Imag[2*7 + 1];
	pixel = image[(y + 2)*128 + x + 2];
    sum.a += pixel * Filter_1_Real[1*7 + 1];
	sum.b += pixel * Filter_1_Imag[1*7 + 1];
	sum.c += pixel * Filter_2_Real[1*7 + 1];
	sum.d += pixel * Filter_2_Imag[1*7 + 1];
	sum.e += pixel * Filter_3_Real[1*7 + 1];
	sum.f += pixel * Filter_3_Imag[1*7 + 1];
	pixel = image[(y + 3)*128 + x + 2];
    sum.a += pixel * Filter_1_Real[1];
	sum.b += pixel * Filter_1_Imag[1];
	sum.c += pixel * Filter_2_Real[1];
	sum.d += pixel * Filter_2_Imag[1];
	sum.e += pixel * Filter_3_Real[1];
	sum.f += pixel * Filter_3_Imag[1];

    pixel = image[(y - 3)*128 + x + 3];
    sum.a += pixel * Filter_1_Real[6*7];
	sum.b += pixel * Filter_1_Imag[6*7];
	sum.c += pixel * Filter_2_Real[6*7];
	sum.d += pixel * Filter_2_Imag[6*7];
	sum.e += pixel * Filter_3_Real[6*7];
	sum.f += pixel * Filter_3_Imag[6*7];
    pixel = image[(y - 2)*128 + x + 3];
    sum.a += pixel * Filter_1_Real[5*7];
	sum.b += pixel * Filter_1_Imag[5*7];
	sum.c += pixel * Filter_2_Real[5*7];
	sum.d += pixel * Filter_2_Imag[5*7];
	sum.e += pixel * Filter_3_Real[5*7];
	sum.f += pixel * Filter_3_Imag[5*7];
	pixel = image[(y - 1)*128 + x + 3];
    sum.a += pixel * Filter_1_Real[4*7];
	sum.b += pixel * Filter_1_Imag[4*7];
	sum.c += pixel * Filter_2_Real[4*7];
	sum.d += pixel * Filter_2_Imag[4*7];
	sum.e += pixel * Filter_3_Real[4*7];
	sum.f += pixel * Filter_3_Imag[4*7];
	pixel = image[y*128 + x + 3];
    sum.a += pixel * Filter_1_Real[3*7];
	sum.b += pixel * Filter_1_Imag[3*7];
	sum.c += pixel * Filter_2_Real[3*7];
	sum.d += pixel * Filter_2_Imag[3*7];
	sum.e += pixel * Filter_3_Real[3*7];
	sum.f += pixel * Filter_3_Imag[3*7];
    pixel = image[(y + 1)*128 + x + 3];
    sum.a += pixel * Filter_1_Real[2*7];
	sum.b += pixel * Filter_1_Imag[2*7];
	sum.c += pixel * Filter_2_Real[2*7];
	sum.d += pixel * Filter_2_Imag[2*7];
	sum.e += pixel * Filter_3_Real[2*7];
	sum.f += pixel * Filter_3_Imag[2*7];
	pixel = image[(y + 2)*128 + x + 3];
    sum.a += pixel * Filter_1_Real[1*7];
	sum.b += pixel * Filter_1_Imag[1*7];
	sum.c += pixel * Filter_2_Real[1*7];
	sum.d += pixel * Filter_2_Imag[1*7];
	sum.e += pixel * Filter_3_Real[1*7];
	sum.f += pixel * Filter_3_Imag[1*7];
	pixel = image[(y + 3)*128 + x + 3];
    sum.a += pixel * Filter_1_Real[0];
	sum.b += pixel * Filter_1_Imag[0];
	sum.c += pixel * Filter_2_Real[0];
	sum.d += pixel * Filter_2_Imag[0];
	sum.e += pixel * Filter_3_Real[0];
	sum.f += pixel * Filter_3_Imag[0];

	return sum;
}


__kernel void Nonseparable3DConvolutionComplexThreeQuadratureFilters_24KB_1024threads(
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
    int x = get_group_id(0) * VALID_FILTER_RESPONSES_X_CONVOLUTION_2D_24KB + get_local_id(0);
	int y = get_group_id(1) * VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D_24KB + get_local_id(1);
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
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_24KB((__local float*)l_Image,tIdx.y+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += (float2)(temp.a,temp.b);
	    Filter_Response_2[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += (float2)(temp.c,temp.d);
	    Filter_Response_3[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += (float2)(temp.e,temp.f);	    
    }

    if ( ((x + 32) < DATA_W) && (y < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_24KB((__local float*)l_Image,tIdx.y+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += (float2)(temp.a,temp.b);
	    Filter_Response_2[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += (float2)(temp.c,temp.d);
	    Filter_Response_3[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += (float2)(temp.e,temp.f);

    }

    if (tIdx.x < (32 - HALO*2))
    {
        if ( ((x + 64) < DATA_W) && (y < DATA_H) )
	    {
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_24KB((__local float*)l_Image,tIdx.y+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += (float2)(temp.a,temp.b);
			Filter_Response_2[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += (float2)(temp.c,temp.d);
			Filter_Response_3[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += (float2)(temp.e,temp.f);
	    }
    }

    if (tIdx.y < (32 - HALO*2))
    {
        if ( (x < DATA_W) && ((y + 32) < DATA_H) )
	    {
 		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_24KB((__local float*)l_Image,tIdx.y+32+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a,temp.b);
			Filter_Response_2[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c,temp.d);
			Filter_Response_3[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e,temp.f);
	    }
    }

    if (tIdx.y < (32 - HALO*2))
    {
        if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
	    {
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_24KB((__local float*)l_Image,tIdx.y+32+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a,temp.b);
			Filter_Response_2[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c,temp.d);
			Filter_Response_3[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e,temp.f);
	    }
     } 

    if ( (tIdx.x < (32 - HALO*2)) && (tIdx.y < (32 - HALO*2)) )
    {
        if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
	    {
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_24KB((__local float*)l_Image,tIdx.y+32+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a,temp.b);
			Filter_Response_2[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c,temp.d);
			Filter_Response_3[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e,temp.f);
	    }
     }
}



__kernel void Nonseparable3DConvolutionComplexThreeQuadratureFilters_32KB_512threads(__global float2 *Filter_Response_1,
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
    int x = get_group_id(0) * VALID_FILTER_RESPONSES_X_CONVOLUTION_2D_32KB + get_local_id(0);
	int y = get_group_id(1) * VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D_32KB + get_local_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};
    
	
    __local float l_Image[64][128]; // y, x

    // Reset shared memory
    l_Image[tIdx.y][tIdx.x]           = 0.0f;
    l_Image[tIdx.y][tIdx.x + 32]      = 0.0f;
	l_Image[tIdx.y][tIdx.x + 64]      = 0.0f;
	l_Image[tIdx.y][tIdx.x + 96]      = 0.0f;

	l_Image[tIdx.y + 16][tIdx.x]           = 0.0f;
    l_Image[tIdx.y + 16][tIdx.x + 32]      = 0.0f;
	l_Image[tIdx.y + 16][tIdx.x + 64]      = 0.0f;
	l_Image[tIdx.y + 16][tIdx.x + 96]      = 0.0f;

	l_Image[tIdx.y + 32][tIdx.x]           = 0.0f;
    l_Image[tIdx.y + 32][tIdx.x + 32]      = 0.0f;
	l_Image[tIdx.y + 32][tIdx.x + 64]      = 0.0f;
	l_Image[tIdx.y + 32][tIdx.x + 96]      = 0.0f;

	l_Image[tIdx.y + 48][tIdx.x]           = 0.0f;
    l_Image[tIdx.y + 48][tIdx.x + 32]      = 0.0f;
	l_Image[tIdx.y + 48][tIdx.x + 64]      = 0.0f;
	l_Image[tIdx.y + 48][tIdx.x + 96]      = 0.0f;

    // Read data into shared memory

    if ( ((z + z_offset) >= 0) && ((z + z_offset) < DATA_D) )
    {
        if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  )   
            l_Image[tIdx.y][tIdx.x] = Volume[Calculate3DIndex(x-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+32-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  ) 
            l_Image[tIdx.y][tIdx.x + 32] = Volume[Calculate3DIndex(x+32-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+64-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  )
            l_Image[tIdx.y][tIdx.x + 64] = Volume[Calculate3DIndex(x+64-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

		if ( ((x+96-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  )
            l_Image[tIdx.y][tIdx.x + 96] = Volume[Calculate3DIndex(x+96-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

		if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+16-HALO) < DATA_H)  )   
            l_Image[tIdx.y + 16][tIdx.x] = Volume[Calculate3DIndex(x-HALO,y+16-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+32-HALO) < DATA_W) && ((y+16-HALO) < DATA_H)  ) 
            l_Image[tIdx.y + 16][tIdx.x + 32] = Volume[Calculate3DIndex(x+32-HALO,y+16-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+64-HALO) < DATA_W) && ((y+16-HALO) < DATA_H)  )
            l_Image[tIdx.y + 16][tIdx.x + 64] = Volume[Calculate3DIndex(x+64-HALO,y+16-HALO,z+z_offset,DATA_W,DATA_H)];

		if ( ((x+96-HALO) < DATA_W) && ((y+16-HALO) < DATA_H)  )
            l_Image[tIdx.y + 16][tIdx.x + 96] = Volume[Calculate3DIndex(x+96-HALO,y+16-HALO,z+z_offset,DATA_W,DATA_H)];


		if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )   
            l_Image[tIdx.y + 32][tIdx.x] = Volume[Calculate3DIndex(x-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+32-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  ) 
            l_Image[tIdx.y + 32][tIdx.x + 32] = Volume[Calculate3DIndex(x+32-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+64-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )
            l_Image[tIdx.y + 32][tIdx.x + 64] = Volume[Calculate3DIndex(x+64-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];

		if ( ((x+96-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )
            l_Image[tIdx.y + 32][tIdx.x + 96] = Volume[Calculate3DIndex(x+96-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];


		if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+48-HALO) < DATA_H)  )   
            l_Image[tIdx.y + 48][tIdx.x] = Volume[Calculate3DIndex(x-HALO,y+48-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+32-HALO) < DATA_W) && ((y+48-HALO) < DATA_H)  ) 
            l_Image[tIdx.y + 48][tIdx.x + 32] = Volume[Calculate3DIndex(x+32-HALO,y+48-HALO,z+z_offset,DATA_W,DATA_H)];

        if ( ((x+64-HALO) < DATA_W) && ((y+48-HALO) < DATA_H)  )
            l_Image[tIdx.y + 48][tIdx.x + 64] = Volume[Calculate3DIndex(x+64-HALO,y+48-HALO,z+z_offset,DATA_W,DATA_H)];

		if ( ((x+96-HALO) < DATA_W) && ((y+48-HALO) < DATA_H)  )
            l_Image[tIdx.y + 48][tIdx.x + 96] = Volume[Calculate3DIndex(x+96-HALO,y+48-HALO,z+z_offset,DATA_W,DATA_H)];

    }
	
   	// Make sure all threads have written to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

    // Only threads inside the image do the convolution

    if ( (x < DATA_W) && (y < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
	    Filter_Response_2[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
	    Filter_Response_3[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);			    
    }

    if ( ((x + 32) < DATA_W) && (y < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

	if ( ((x + 64) < DATA_W) && (y < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

    if (tIdx.x < (32 - HALO*2))
    {
		if ( ((x + 96) < DATA_W) && (y < DATA_H) )
	    {
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+HALO,tIdx.x+96+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
	        Filter_Response_1[Calculate3DIndex(x+96,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+96,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+96,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
	    }
	}

	if ( (x < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+16+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
	    Filter_Response_2[Calculate3DIndex(x,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
	    Filter_Response_3[Calculate3DIndex(x,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);			    
    }

    if ( ((x + 32) < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+16+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+32,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+32,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+32,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				
    }

	if ( ((x + 64) < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+16+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+64,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+64,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+64,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

	if (tIdx.x < (32 - HALO*2))
    {
		if ( ((x + 96) < DATA_W) && ((y + 16) < DATA_H) )
    	{
	    	float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+16+HALO,tIdx.x+96+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
	        Filter_Response_1[Calculate3DIndex(x+96,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+96,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+96,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
	    }
	}

	if ( (x < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+32+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
	    Filter_Response_2[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
	    Filter_Response_3[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);			    
    }

    if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+32+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

	if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+32+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				
    }

    if (tIdx.x < (32 - HALO*2))
    {
		if ( ((x + 96) < DATA_W) && ((y + 32) < DATA_H) )
	    {
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+32+HALO,tIdx.x+96+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
	        Filter_Response_1[Calculate3DIndex(x+96,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+96,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+96,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
	    }
	}
    
	if (tIdx.y < (16 - HALO*2))
    {	
		if ( (x < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+48+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
		}

		if ( ((x + 32) < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+48+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+32,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+32,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+32,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);						
		}

		if ( ((x + 64) < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+48+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+64,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+64,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+64,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);						
		}

		if (tIdx.x < (32 - HALO*2))
		{
			if ( ((x + 96) < DATA_W) && ((y + 48) < DATA_H) )
			{
			    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+48+HALO,tIdx.x+96+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
				Filter_Response_1[Calculate3DIndex(x+96,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
				Filter_Response_2[Calculate3DIndex(x+96,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
				Filter_Response_3[Calculate3DIndex(x+96,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
			}
		}
	}	
}


__kernel void Nonseparable3DConvolutionComplexThreeQuadratureFilters_32KB_256threads(__global float2 *Filter_Response_1,
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
    int x = get_group_id(0) * VALID_FILTER_RESPONSES_X_CONVOLUTION_2D_32KB + get_local_id(0);
	int y = get_group_id(1) * VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D_32KB + get_local_id(1);
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
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
	    Filter_Response_2[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
	    Filter_Response_3[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);			    
    }

	if ( ((x + 16) < DATA_W) && (y < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+HALO,tIdx.x+16+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+16,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
	    Filter_Response_2[Calculate3DIndex(x+16,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
	    Filter_Response_3[Calculate3DIndex(x+16,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				    
    }

    if ( ((x + 32) < DATA_W) && (y < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+32,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

	if ( ((x + 48) < DATA_W) && (y < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+HALO,tIdx.x+48+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+48,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+48,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+48,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				
    }

	if ( ((x + 64) < DATA_W) && (y < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+64,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

	if ( ((x + 80) < DATA_W) && (y < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+HALO,tIdx.x+80+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+80,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+80,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+80,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				
    }

	if ( ((x + 96) < DATA_W) && (y < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+HALO,tIdx.x+96+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+96,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+96,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+96,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

    if (tIdx.x < (16 - HALO*2))
    {
        if ( ((x + 112) < DATA_W) && (y < DATA_H) )
	    {
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+HALO,tIdx.x+112+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
            Filter_Response_1[Calculate3DIndex(x+112,y,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		    Filter_Response_2[Calculate3DIndex(x+112,y,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		    Filter_Response_3[Calculate3DIndex(x+112,y,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				    
	    }
    }

	if ( (x < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+16+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
	    Filter_Response_2[Calculate3DIndex(x,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
	    Filter_Response_3[Calculate3DIndex(x,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);			    
    }

	if ( ((x + 16) < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+16+HALO,tIdx.x+16+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+16,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
	    Filter_Response_2[Calculate3DIndex(x+16,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
	    Filter_Response_3[Calculate3DIndex(x+16,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				    
    }

    if ( ((x + 32) < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+16+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+32,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+32,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+32,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				
    }

	if ( ((x + 48) < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+16+HALO,tIdx.x+48+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+48,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+48,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+48,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				
    }

	if ( ((x + 64) < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+16+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+64,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+64,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+64,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

	if ( ((x + 80) < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+16+HALO,tIdx.x+80+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+80,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+80,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+80,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

	if ( ((x + 96) < DATA_W) && ((y + 16) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+16+HALO,tIdx.x+96+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+96,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+96,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+96,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

    if (tIdx.x < (16 - HALO*2))
    {
        if ( ((x + 112) < DATA_W) && ((y + 16) < DATA_H) )
	    {
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+16+HALO,tIdx.x+112+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
            Filter_Response_1[Calculate3DIndex(x+112,y+16,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		    Filter_Response_2[Calculate3DIndex(x+112,y+16,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		    Filter_Response_3[Calculate3DIndex(x+112,y+16,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				    
	    }
    }


	if ( (x < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+32+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
	    Filter_Response_2[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
	    Filter_Response_3[Calculate3DIndex(x,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);			    
    }

	if ( ((x + 16) < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+32+HALO,tIdx.x+16+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+16,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
	    Filter_Response_2[Calculate3DIndex(x+16,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
	    Filter_Response_3[Calculate3DIndex(x+16,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);			    
    }

    if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+32+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+32,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

	if ( ((x + 48) < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+32+HALO,tIdx.x+48+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+48,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+48,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+48,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				
    }

	if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+32+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				
    }

	if ( ((x + 80) < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+32+HALO,tIdx.x+80+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+80,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+80,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+80,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

	if ( ((x + 96) < DATA_W) && ((y + 32) < DATA_H) )
    {
	    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+32+HALO,tIdx.x+96+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
        Filter_Response_1[Calculate3DIndex(x+96,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		Filter_Response_2[Calculate3DIndex(x+96,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		Filter_Response_3[Calculate3DIndex(x+96,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
    }

    if (tIdx.x < (16 - HALO*2))
    {
        if ( ((x + 112) < DATA_W) && ((y + 32) < DATA_H) )
	    {
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+32+HALO,tIdx.x+112+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
            Filter_Response_1[Calculate3DIndex(x+112,y+32,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
		    Filter_Response_2[Calculate3DIndex(x+112,y+32,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
		    Filter_Response_3[Calculate3DIndex(x+112,y+32,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);				    
	    }
    }

	if (tIdx.y < (16 - HALO*2))
    {	
		if ( (x < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+48+HALO,tIdx.x+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
		}

		if ( ((x + 16) < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+48+HALO,tIdx.x+16+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+16,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+16,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+16,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);						
		}

		if ( ((x + 32) < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+48+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+32,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+32,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+32,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);						
		}

		if ( ((x + 48) < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+48+HALO,tIdx.x+48+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
	        Filter_Response_1[Calculate3DIndex(x+48,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+48,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+48,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);						
		}

		if ( ((x + 64) < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+48+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+64,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+64,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+64,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);						
		}

		if ( ((x + 80) < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+48+HALO,tIdx.x+80+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+80,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+80,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+80,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
		}

		if ( ((x + 96) < DATA_W) && ((y + 48) < DATA_H) )
		{
		    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+48+HALO,tIdx.x+96+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
			Filter_Response_1[Calculate3DIndex(x+96,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
			Filter_Response_2[Calculate3DIndex(x+96,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
			Filter_Response_3[Calculate3DIndex(x+96,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);					
		}

		if (tIdx.x < (16 - HALO*2))
		{
	        if ( ((x + 112) < DATA_W) && ((y + 48) < DATA_H) )
			{
			    float66 temp = Conv_2D_Unrolled_7x7_ThreeFilters_32KB((__local float*)l_Image,tIdx.y+48+HALO,tIdx.x+112+HALO,c_Quadrature_Filter_1_Real,c_Quadrature_Filter_1_Imag,c_Quadrature_Filter_2_Real,c_Quadrature_Filter_2_Imag,c_Quadrature_Filter_3_Real,c_Quadrature_Filter_3_Imag);
				Filter_Response_1[Calculate3DIndex(x+112,y+48,z,DATA_W,DATA_H)] += (float2)(temp.a, temp.b);
				Filter_Response_2[Calculate3DIndex(x+112,y+48,z,DATA_W,DATA_H)] += (float2)(temp.c, temp.d);
				Filter_Response_3[Calculate3DIndex(x+112,y+48,z,DATA_W,DATA_H)] += (float2)(temp.e, temp.f);			
			}
		}
	}	
}




