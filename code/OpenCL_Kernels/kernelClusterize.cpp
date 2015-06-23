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

__kernel void SetStartClusterIndicesKernel(__global unsigned int* Cluster_Indices,
										   __global const float* Data,
										   __global const float* Mask,
										   __private float threshold,
 									       __private int contrast,
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
	if ( (Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] == 1.0f) && (Data[Calculate4DIndex(x,y,z,contrast,DATA_W,DATA_H,DATA_D)] > threshold) )
	{
		// Set an unique index
		Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = (unsigned int)Calculate3DIndex(x,y,z,DATA_W,DATA_H);
	}
	else
	{
		// Make sure that all other voxels have a higher start index
		Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = (unsigned int)(DATA_W * DATA_H * DATA_D * 3);
	}
}




bool IsInsideVolume(int x, int y, int z, int DATA_W, int DATA_H, int DATA_D)
{
	if (z < 0)
		return false;
	else if (z >= DATA_D)
		return false;
	else if (y < 0)
		return false;
	else if (y >= DATA_H)
		return false;
	else if (x < 0)
		return false;
	else if (x >= DATA_W)
		return false;
	else
		return true;
}


__kernel void ClusterizeScan(__global unsigned int* Cluster_Indices,
						  	  volatile __global float* Updated,
						  	  __global const float* Data,
						  	  __global const float* Mask,
						  	  __private float threshold,
 							  __private int contrast,
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
	if ( Data[Calculate4DIndex(x,y,z,contrast,DATA_W,DATA_H,DATA_D)] > threshold )
	{
		unsigned int label1, label2, temp;

		label2 = DATA_W * DATA_H * DATA_D * 3;

		// Original index
		label1 = Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];

		// z - 1
		if ( IsInsideVolume(x-1,y,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x-1,y-1,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y-1,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x-1,y+1,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y+1,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x,y,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x,y,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x,y-1,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x,y-1,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x,y+1,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x,y+1,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y-1,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y-1,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y+1,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y+1,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		// z

		if ( IsInsideVolume(x-1,y,z,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y,z,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x-1,y-1,z,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y-1,z,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x-1,y+1,z,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y+1,z,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x,y-1,z,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x,y-1,z,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x,y+1,z,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x,y+1,z,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y,z,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y,z,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y-1,z,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y-1,z,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y+1,z,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y+1,z,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		// z + 1

		if ( IsInsideVolume(x-1,y,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x-1,y-1,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y-1,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x-1,y+1,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y+1,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x,y,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x,y,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x,y-1,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x,y-1,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x,y+1,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x,y+1,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y-1,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y-1,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y+1,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y+1,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if (label2 < label1)
		{
			Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = label2;
			float one = 1.0f;
			atomic_xchg(Updated,one);
		}
	}
}


__kernel void ClusterizeRelabel(__global unsigned int* Cluster_Indices,
						  	  	__global const float* Data,
						  	  	__global const float* Mask,
						  	  	__private float threshold,
								__private int contrast,
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
	if ( (Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] == 1.0f) && (Data[Calculate4DIndex(x,y,z,contrast,DATA_W,DATA_H,DATA_D)] > threshold) )
	{
		// Relabel voxels
		unsigned int label = Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];
		unsigned int next = Cluster_Indices[label];
		while (next != label)
		{
			label = next;
			next = Cluster_Indices[label];
		}
		Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = label;
	}
}

__kernel void CalculateClusterSizes(__global unsigned int* Cluster_Indices,
						  	  	    volatile __global unsigned int* Cluster_Sizes,
						  	  	    __global const float* Data,
						  	  	    __global const float* Mask,
						  	  	    __private float threshold,	
									__private int contrast,
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
	if ( Data[Calculate4DIndex(x,y,z,contrast,DATA_W,DATA_H,DATA_D)] > threshold )
	{
		unsigned int one = 1;
		// Increment counter for the current cluster index
		atomic_add(&Cluster_Sizes[Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)]],one);		
	}
}

__kernel void CalculateClusterMasses(__global unsigned int* Cluster_Indices,
						  	  	     volatile __global unsigned int* Cluster_Masses,
						  	  	     __global const float* Data,
						  	  	     __global const float* Mask,
						  	  	     __private float threshold,
									 __private int contrast,
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
	if ( Data[Calculate4DIndex(x,y,z,contrast,DATA_W,DATA_H,DATA_D)] > threshold )
	{
		// Increment mass for the current cluster index, done in an ugly way since atomic floats are not supported
		atomic_add( &Cluster_Masses[Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)]], (unsigned int)(Data[Calculate4DIndex(x,y,z,contrast,DATA_W,DATA_H,DATA_D)] * 10000.0f) );
	}
}

__kernel void CalculateLargestCluster(__global unsigned int* Cluster_Sizes,
								      volatile global unsigned int* Largest_Cluster,
   						  	  	      __private int DATA_W,
									  __private int DATA_H,
									  __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	unsigned int cluster_size = Cluster_Sizes[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];

	// Most cluster size counters are zero, so avoid running atomic max for those
	if (cluster_size == 0)
		return;

	atomic_max(Largest_Cluster,cluster_size);
}

__kernel void CalculateTFCEValues(__global float* TFCE_Values,
								  __global const float* Mask,
	  	    					  __private float threshold,
								  __global const unsigned int* Cluster_Indices,
							      __global const unsigned int* Cluster_Sizes,
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

	// Check if the current voxel belongs to a cluster
	if ( Cluster_Indices[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] < (DATA_W * DATA_H * DATA_D) )
	{
		// Get extent of cluster for current voxel
		float clusterSize = (float)Cluster_Sizes[Cluster_Indices[Calculate3DIndex(x, y, z, DATA_W, DATA_H)]];
		float value = sqrt(clusterSize) * threshold * threshold;

		TFCE_Values[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += value;
	}
}




__kernel void CalculatePermutationPValuesVoxelLevelInference(__global float* P_Values,
							   	   	   	   	   	  	  	  	 __global const float* Test_Values,
							   	   	   	   	   	  	  	  	 __global const float* Mask,
							   	   	   	   	   	  	  	  	 __constant float* c_Max_Values,
							   	   	   	   	   	  	  	  	 __private int contrast,
							   	   	   	   	   	  	  	  	 __private int DATA_W,
							   	   	   	   	   	  	  	  	 __private int DATA_H,
							   	   	   	   	   	  	  	  	 __private int DATA_D,
							   	   	   	   	   	  	  	  	 __private int NUMBER_OF_PERMUTATIONS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Mask[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] == 1.0f )
	{
    	float Test_Value = Test_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)];

    	float sum = 0.0f;
    	for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
    	{
    		//sum += Test_Value > c_Max_Values[p];
    		if (Test_Value > c_Max_Values[p])
    		{
    			sum += 1.0f;
    		}
    	}
    	P_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)] = sum / (float)NUMBER_OF_PERMUTATIONS;
	}
    else
    {
    	P_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)] = 0.0f;
    }
}

__kernel void CalculatePermutationPValuesClusterLevelInference(__global float* P_Values,
															   __global const float* Test_Values,
															   __global const unsigned int* Cluster_Indices,
															   __global const unsigned int* Cluster_Sizes,
							   	   	   	   	   	  	  	  	   __global const float* Mask,
							   	   	   	   	   	  	  	  	   __constant float* c_Max_Values,
							   	   	   	   	   	  	  	  	   __private float threshold,
							   	   	   	   	   	  	  	  	   __private int contrast,
							   	   	   	   	   	  	  	  	   __private int DATA_W,
							   	   	   	   	   	  	  	  	   __private int DATA_H,
							   	   	   	   	   	  	  	  	   __private int DATA_D,
							   	   	   	   	   	  	  	  	   __private int NUMBER_OF_PERMUTATIONS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Mask[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] == 1.0f )
	{
    	// Check if the current voxel belongs to a cluster
    	if ( Test_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)] > threshold )
    	{
    		// Get cluster extent or cluster mass of current cluster
    		float Test_Value = (float)Cluster_Sizes[Cluster_Indices[Calculate3DIndex(x, y, z, DATA_W, DATA_H)]];

    		float sum = 0.0f;
    		for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
    		{
    			if (Test_Value > c_Max_Values[p])
    			{
    				sum += 1.0f;
    			}
    		}
    		P_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)] = sum / (float)NUMBER_OF_PERMUTATIONS;
    	}
    	// Voxel is not part of a cluster, so p-value should be 0
    	else
    	{
    		P_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)] = 0.0f;
    	}
	}
    else
    {
    	P_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)] = 0.0f;
    }
}

