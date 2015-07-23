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


__kernel void IdentityMatrix(__global float* Matrix, 
  			     			 __private int N)
{
	int x = get_global_id(0);	
	int y = get_global_id(1);

	if (x >= N || y >= N)
		return;

	if (x == y)
	{
		Matrix[Calculate2DIndex(x,y,N)] = 1.0f;
	}
	else
	{
		Matrix[Calculate2DIndex(x,y,N)] = 0.0f;
	}
}

__kernel void GetSubMatrix(__global float* Small_Matrix, 
                           __global const float* Matrix, 
  			     		   __private int rows)
{
	int x = get_global_id(0);	
	int y = get_global_id(1);

	if (x >= rows || y >= rows)
		return;

	Small_Matrix[x] = Matrix[x];
}


__kernel void PermuteMatrix(__global float* Small_Matrix, 
                           __global const float* Matrix, 
  			     		   __private int rows)
{
	int x = get_global_id(0);	
	int y = get_global_id(1);

	if (x >= rows || y >= rows)
		return;

	Small_Matrix[x] = Matrix[x];

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

__kernel void SubtractVolumes(__global float* Result, 
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

	Result[idx] = Volume1[idx] - Volume2[idx];
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


__kernel void SubtractVolumesOverwrite(__global float* Volume1, 
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

	Volume1[idx] = Volume1[idx] - Volume2[idx];
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
									__private float delta, 									 
									__private int DATA_W, 
									__private int DATA_H, 
									__private int DATA_D, 
									__private int DATA_T)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H)
		return;

	float t0, t1, t2, t3;

	// Forward interpolation
	if (delta > 0.0f)
	{
		t0 = Volumes[Calculate3DIndex(x,y,0,DATA_W,DATA_H)];
		t1 = t0;
		t2 = Volumes[Calculate3DIndex(x,y,1,DATA_W,DATA_H)];
		t3 = Volumes[Calculate3DIndex(x,y,2,DATA_W,DATA_H)];

		// Loop over timepoints
		for (int t = 0; t < DATA_T - 3; t++)
		{
			// Cubic interpolation in time
			Corrected_Volumes[Calculate3DIndex(x,y,t,DATA_W,DATA_H)] = InterpolateCubic(t0,t1,t2,t3,delta); 
		
			// Shift old values backwards
			t0 = t1;
			t1 = t2;
			t2 = t3;

			// Read one new value
			t3 = Volumes[Calculate3DIndex(x,y,t+3,DATA_W,DATA_H)];
		}

		int t = DATA_T - 3;	
		Corrected_Volumes[Calculate3DIndex(x,y,t,DATA_W,DATA_H)] = InterpolateCubic(t0,t1,t2,t3,delta); 
	
		t = DATA_T - 2;
		t0 = t1;
		t1 = t2;
		t2 = t3;	
		Corrected_Volumes[Calculate3DIndex(x,y,t,DATA_W,DATA_H)] = InterpolateCubic(t0,t1,t2,t3,delta); 

		t = DATA_T - 1;
		t0 = t1;
		t1 = t2;
		Corrected_Volumes[Calculate3DIndex(x,y,t,DATA_W,DATA_H)] = InterpolateCubic(t0,t1,t2,t3,delta); 
	}
	// Backward interpolation
	else
	{
		delta = 1.0f - (-delta);

		t0 = Volumes[Calculate3DIndex(x,y,0,DATA_W,DATA_H)];
		t1 = t0;
		t2 = t0;
		t3 = Volumes[Calculate3DIndex(x,y,1,DATA_W,DATA_H)];

		// Loop over timepoints
		for (int t = 0; t < DATA_T - 2; t++)
		{
			// Cubic interpolation in time
			Corrected_Volumes[Calculate3DIndex(x,y,t,DATA_W,DATA_H)] = InterpolateCubic(t0,t1,t2,t3,delta); 
		
			// Shift old values backwards
			t0 = t1;
			t1 = t2;
			t2 = t3;

			// Read one new value
			t3 = Volumes[Calculate3DIndex(x,y,t+2,DATA_W,DATA_H)];
		}

		int t = DATA_T - 2;	
		Corrected_Volumes[Calculate3DIndex(x,y,t,DATA_W,DATA_H)] = InterpolateCubic(t0,t1,t2,t3,delta); 
	
		t = DATA_T - 1;
		t0 = t1;
		t1 = t2;
		t2 = t3;	
		Corrected_Volumes[Calculate3DIndex(x,y,t,DATA_W,DATA_H)] = InterpolateCubic(t0,t1,t2,t3,delta); 
	}
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
