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

__kernel void CalculateBetaWeightsGLMSlice(__global float* Beta_Volumes, 
                                      	   __global const float* Volumes, 
									       __global const float* Mask, 
									       __global const float* c_xtxxt_GLM, 
									       __constant float* c_Censored_Timepoints,
									       __private int DATA_W, 
									       __private int DATA_H, 
									       __private int DATA_D, 
									       __private int NUMBER_OF_VOLUMES, 
									       __private int NUMBER_OF_REGRESSORS,
									       __private int slice)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int NUMBER_OF_REGRESSORS_PER_CHUNK = 25;
	int REGRESSOR_GROUPS = (int)ceil((float)NUMBER_OF_REGRESSORS / (float)NUMBER_OF_REGRESSORS_PER_CHUNK);
	int NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK = 0;

	// First deal with voxels outside the mask
	if ( Mask[Calculate3DIndex(x,y,slice,DATA_W,DATA_H)] != 1.0f )
	{
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			Beta_Volumes[Calculate4DIndex(x,y,slice,r,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}
		return;
	}

	// Loop over chunks of 25 regressors at a time, since it is not possible to use for example 400 registers per thread
	for (int regressor_group = 0; regressor_group < REGRESSOR_GROUPS; regressor_group++)
	{
		// Check how many regressors that are left
		if ( (NUMBER_OF_REGRESSORS - regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK) >= 25 )
		{
			NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK = 25;
		}	
		else
		{
			NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK = NUMBER_OF_REGRESSORS - regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK;
		}

		int t = 0;
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
			float temp = Volumes[Calculate3DIndex(x,y,v,DATA_W,DATA_H)] * c_Censored_Timepoints[v];

			// Loop over regressors
			for (int r = 0; r < NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK; r++)
			{
				beta[r] += temp * c_xtxxt_GLM[NUMBER_OF_VOLUMES * (r + regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK) + v];
			}
		}

		// Save beta values for current chunk
		for (int r = 0; r < NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK; r++)
		{
			Beta_Volumes[Calculate4DIndex(x,y,slice,r + regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK,DATA_W,DATA_H,DATA_D)] = beta[r];
		}
	}
}



__kernel void CalculateBetaWeightsAndContrastsGLM(__global float* Beta_Volumes, 
                                      	   		  __global float* Contrast_Volumes, 
                                      	   		  __global const float* Volumes, 
									       		  __global const float* Mask, 
									       		  __global const float* c_xtxxt_GLM, 
												  __global const float* c_Contrasts,
									       		  __constant float* c_Censored_Timepoints,
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

	int NUMBER_OF_REGRESSORS_PER_CHUNK = 25;
	int REGRESSOR_GROUPS = (int)ceil((float)NUMBER_OF_REGRESSORS / (float)NUMBER_OF_REGRESSORS_PER_CHUNK);
	int NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK = 0;

	// First deal with voxels outside the mask
	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
	{
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}
		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			Contrast_Volumes[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}
		return;
	}

	float beta[25];

	// Loop over chunks of 25 regressors at a time, since it is not possible to use for example 400 registers per thread
	for (int regressor_group = 0; regressor_group < REGRESSOR_GROUPS; regressor_group++)
	{
		// Check how many regressors that are left
		if ( (NUMBER_OF_REGRESSORS - regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK) >= 25 )
		{
			NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK = 25;
		}	
		else
		{
			NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK = NUMBER_OF_REGRESSORS - regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK;
		}

		int t = 0;		
	
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
			float temp = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] * c_Censored_Timepoints[v];

			// Loop over regressors
			for (int r = 0; r < NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK; r++)
			{
				beta[r] += temp * c_xtxxt_GLM[NUMBER_OF_VOLUMES * (r + regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK) + v];
			}
		}

		// Save beta values for current chunk
		for (int r = 0; r < NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK; r++)
		{
			Beta_Volumes[Calculate4DIndex(x,y,z,r + regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK,DATA_W,DATA_H,DATA_D)] = beta[r];
		}
	}

	if (NUMBER_OF_REGRESSORS <= 25)
	{
		// Loop over contrasts and calculate t-values, using a voxel-specific GLM scalar
		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			float contrast_value = 0.0f;
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + r] * beta[r];
			}
			Contrast_Volumes[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_value;
		}
	}
	else
	{
		// Loop over contrasts and calculate t-values, using a voxel-specific GLM scalar
		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			float contrast_value = 0.0f;
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + r] * Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)];
			}
			Contrast_Volumes[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_value;
		}
	}
}



__kernel void CalculateBetaWeightsAndContrastsGLMSlice(__global float* Beta_Volumes, 
                                      	   			   __global float* Contrast_Volumes, 
                                      	   			   __global const float* Volumes, 
									       			   __global const float* Mask, 
									       			   __global const float* c_xtxxt_GLM, 
													   __global const float* c_Contrasts,
									       			   __constant float* c_Censored_Timepoints,
									       			   __private int DATA_W, 
									       			   __private int DATA_H, 
									       			   __private int DATA_D, 
									       			   __private int NUMBER_OF_VOLUMES, 
									      			   __private int NUMBER_OF_REGRESSORS,
									      			   __private int NUMBER_OF_CONTRASTS,
									       			   __private int slice)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int NUMBER_OF_REGRESSORS_PER_CHUNK = 25;
	int REGRESSOR_GROUPS = (int)ceil((float)NUMBER_OF_REGRESSORS / (float)NUMBER_OF_REGRESSORS_PER_CHUNK);
	int NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK = 0;

	// First deal with voxels outside the mask
	if ( Mask[Calculate3DIndex(x,y,slice,DATA_W,DATA_H)] != 1.0f )
	{
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			Beta_Volumes[Calculate4DIndex(x,y,slice,r,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}
		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			Contrast_Volumes[Calculate4DIndex(x,y,slice,c,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}
		return;
	}

	float beta[25];

	// Loop over chunks of 25 regressors at a time, since it is not possible to use for example 400 registers per thread
	for (int regressor_group = 0; regressor_group < REGRESSOR_GROUPS; regressor_group++)
	{
		// Check how many regressors that are left
		if ( (NUMBER_OF_REGRESSORS - regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK) >= 25 )
		{
			NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK = 25;
		}	
		else
		{
			NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK = NUMBER_OF_REGRESSORS - regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK;
		}

		int t = 0;		
	
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
			float temp = Volumes[Calculate3DIndex(x,y,v,DATA_W,DATA_H)] * c_Censored_Timepoints[v];

			// Loop over regressors
			for (int r = 0; r < NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK; r++)
			{
				beta[r] += temp * c_xtxxt_GLM[NUMBER_OF_VOLUMES * (r + regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK) + v];
			}
		}

		// Save beta values for current chunk
		for (int r = 0; r < NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK; r++)
		{
			Beta_Volumes[Calculate4DIndex(x,y,slice,r + regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK,DATA_W,DATA_H,DATA_D)] = beta[r];
		}
	}

	if (NUMBER_OF_REGRESSORS <= 25)
	{
		// Loop over contrasts and calculate t-values, using a voxel-specific GLM scalar
		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			float contrast_value = 0.0f;
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + r] * beta[r];
			}
			Contrast_Volumes[Calculate4DIndex(x,y,slice,c,DATA_W,DATA_H,DATA_D)] = contrast_value;
		}
	}
	else
	{
		// Loop over contrasts and calculate t-values, using a voxel-specific GLM scalar
		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			float contrast_value = 0.0f;
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + r] * Beta_Volumes[Calculate4DIndex(x,y,slice,r,DATA_W,DATA_H,DATA_D)];
			}
			Contrast_Volumes[Calculate4DIndex(x,y,slice,c,DATA_W,DATA_H,DATA_D)] = contrast_value;
		}
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

	int NUMBER_OF_REGRESSORS_PER_CHUNK = 25;
	int REGRESSOR_GROUPS = (int)ceil((float)NUMBER_OF_REGRESSORS / (float)NUMBER_OF_REGRESSORS_PER_CHUNK);
	int NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK = 0;

	// First deal with voxels outside the mask
	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
	{
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}
		return;
	}

	// Loop over chunks of 25 regressors at a time, since it is not possible to use for example 400 registers per thread
	for (int regressor_group = 0; regressor_group < REGRESSOR_GROUPS; regressor_group++)
	{
		// Check how many regressors that are left
		if ( (NUMBER_OF_REGRESSORS - regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK) >= 25 )
		{
			NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK = 25;
		}	
		else
		{
			NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK = NUMBER_OF_REGRESSORS - regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK;
		}

		int t = 0;
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

		// Get the specific voxel number for this brain voxel
		int voxel_number = (int)d_Voxel_Numbers[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];
		
		// Calculate betahat, i.e. multiply (x^T x)^(-1) x^T with Y
		// Loop over volumes
		for (int v = NUMBER_OF_INVALID_TIMEPOINTS; v < NUMBER_OF_VOLUMES; v++)
		{
			float temp = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];

			// Loop over regressors
			for (int r = 0; r < NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK; r++)
			{
				beta[r] += temp * d_xtxxt_GLM[voxel_number * NUMBER_OF_VOLUMES * NUMBER_OF_REGRESSORS + NUMBER_OF_VOLUMES * (r + regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK) + v];
			}
		}

		// Save beta values for the current chunk of regressors
		for (int r = 0; r < NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK; r++)
		{
			Beta_Volumes[Calculate4DIndex(x,y,z,r + regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK,DATA_W,DATA_H,DATA_D)] = beta[r];
		}
	}
}

__kernel void CalculateBetaWeightsGLMFirstLevelSlice(__global float* Beta_Volumes, 
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
												     __private int NUMBER_OF_INVALID_TIMEPOINTS,
												     __private int slice)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int NUMBER_OF_REGRESSORS_PER_CHUNK = 25;
	int REGRESSOR_GROUPS = (int)ceil((float)NUMBER_OF_REGRESSORS / (float)NUMBER_OF_REGRESSORS_PER_CHUNK);
	int NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK = 0;

	// First deal with voxels outside the mask
	if ( Mask[Calculate3DIndex(x,y,slice,DATA_W,DATA_H)] != 1.0f )
	{
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			Beta_Volumes[Calculate4DIndex(x,y,slice,r,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}
		return;
	}

	// Loop over chunks of 25 regressors at a time, since it is not possible to use for example 400 registers per thread
	for (int regressor_group = 0; regressor_group < REGRESSOR_GROUPS; regressor_group++)
	{
		// Check how many regressors that are left
		if ( (NUMBER_OF_REGRESSORS - regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK) >= 25 )
		{
			NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK = 25;
		}	
		else
		{
			NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK = NUMBER_OF_REGRESSORS - regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK;
		}

		int t = 0;
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

		// Get the specific voxel number for this brain voxel
		//int voxel_number = (int)d_Voxel_Numbers[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];
		int voxel_number = (int)d_Voxel_Numbers[Calculate2DIndex(x,y,DATA_W)];

		// Calculate betahat, i.e. multiply (x^T x)^(-1) x^T with Y
		// Loop over volumes
		for (int v = NUMBER_OF_INVALID_TIMEPOINTS; v < NUMBER_OF_VOLUMES; v++)
		{
			//float temp = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
			float temp = Volumes[Calculate3DIndex(x,y,v,DATA_W,DATA_H)];

			// Loop over regressors
			for (int r = 0; r < NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK; r++)
			{
				beta[r] += temp * d_xtxxt_GLM[voxel_number * NUMBER_OF_VOLUMES * NUMBER_OF_REGRESSORS + NUMBER_OF_VOLUMES * (r + regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK) + v];
			}
		}

		// Save beta values for the current chunk of regressors
		for (int r = 0; r < NUMBER_OF_REGRESSORS_IN_CURRENT_CHUNK; r++)
		{
			Beta_Volumes[Calculate4DIndex(x,y,slice,r + regressor_group * NUMBER_OF_REGRESSORS_PER_CHUNK,DATA_W,DATA_H,DATA_D)] = beta[r];
		}
	}
}




__kernel void CalculateGLMResiduals(__global float* Residuals,
		                            __global const float* Volumes,
		                            __global const float* Beta_Volumes,
		                            __global const float* Mask,
		                            __global const float *c_X_GLM,
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

	// Special case for low number of regressors, store beta scores in registers for faster performance
	if (NUMBER_OF_REGRESSORS <= 25)
	{
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
	// General case for large number of regressors (slower)
	else
	{
		// Calculate the residual
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)];
			}

			Residuals[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = eps;			
		}
	}
}


__kernel void CalculateGLMResidualsSlice(__global float* Residuals,
		                                 __global const float* Volumes,
		                                 __global const float* Beta_Volumes,
		                                 __global const float* Mask,
		                                 __global const float *c_X_GLM,
		                                 __private int DATA_W,
		                                 __private int DATA_H,
		                                 __private int DATA_D,
		                                 __private int NUMBER_OF_VOLUMES,
		                                 __private int NUMBER_OF_REGRESSORS,
                                         __private int slice)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,slice,DATA_W,DATA_H)] != 1.0f )
	{
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			Residuals[Calculate3DIndex(x,y,v,DATA_W,DATA_H)] = 0.0f;
		}

		return;
	}

	int t = 0;
	float eps, meaneps, vareps;

	// Special case for low number of regressors, store beta scores in registers for faster performance
	if (NUMBER_OF_REGRESSORS <= 25)
	{
		float beta[25];

		// Load beta values into registers
	    for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			beta[r] = Beta_Volumes[Calculate4DIndex(x,y,slice,r,DATA_W,DATA_H,DATA_D)];
		}

		// Calculate the residual
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			eps = Volumes[Calculate3DIndex(x,y,v,DATA_W,DATA_H)];
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
			}

			Residuals[Calculate3DIndex(x,y,v,DATA_W,DATA_H)] = eps;
		}
	}
	// General case for large number of regressors (slower)
	else
	{
		// Calculate the residual
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			eps = Volumes[Calculate3DIndex(x,y,v,DATA_W,DATA_H)];
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * Beta_Volumes[Calculate4DIndex(x,y,slice,r,DATA_W,DATA_H,DATA_D)];
			}

			Residuals[Calculate3DIndex(x,y,v,DATA_W,DATA_H)] = eps;
		}
	}
}



__kernel void CalculateStatisticalMapsGLMTTestFirstLevel(__global float* Statistical_Maps,
														 __global float* Contrast_Volumes,
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

	// First deal with voxels outside the mask
	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
	{
		Residual_Variances[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;

		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			Contrast_Volumes[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = 0.0f;
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

    // Get the specific voxel number for this brain voxel
    int voxel_number = (int)d_Voxel_Numbers[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];
	
	// Special case for a low number of regressors
	if (NUMBER_OF_REGRESSORS <= 25)
	{
		float beta[25];
		
		// Load beta values into registers
	    for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			beta[r] = Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)];
		}

		// Calculate the mean of the error eps, using voxel-specific design models
		meaneps = 0.0f;
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];

			// Calculate eps
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				eps -= d_X_GLM[voxel_number * NUMBER_OF_VOLUMES * NUMBER_OF_REGRESSORS + NUMBER_OF_VOLUMES * r + v] * beta[r];
			}
			eps *= c_Censored_Timepoints[v];
			meaneps += eps;
	
			Residuals[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = eps;		
		}
		meaneps /= ((float)NUMBER_OF_VOLUMES);

		// Now calculate the variance of eps, using voxel-specific design models
		vareps = 0.0f;
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];

			// Calculate eps
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				eps -= d_X_GLM[voxel_number * NUMBER_OF_VOLUMES * NUMBER_OF_REGRESSORS + NUMBER_OF_VOLUMES * r + v] * beta[r];
			}
			vareps += (eps - meaneps) * (eps - meaneps) * c_Censored_Timepoints[v];
		}
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
			Contrast_Volumes[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_value;
			Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_value * rsqrt(vareps * d_GLM_Scalars[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)]);
		}
	}
	// General case for large number of regressors, slower
	else
	{
		// Calculate the mean of the error eps, using voxel-specific design models
		meaneps = 0.0f;
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];

			// Calculate eps
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				eps -= d_X_GLM[voxel_number * NUMBER_OF_VOLUMES * NUMBER_OF_REGRESSORS + NUMBER_OF_VOLUMES * r + v] * Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)];
			}
			eps *= c_Censored_Timepoints[v];
			meaneps += eps;
	
			Residuals[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = eps;		
		}
		meaneps /= ((float)NUMBER_OF_VOLUMES);

		// Now calculate the variance of eps, using voxel-specific design models
		vareps = 0.0f;
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];

			// Calculate eps
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				eps -= d_X_GLM[voxel_number * NUMBER_OF_VOLUMES * NUMBER_OF_REGRESSORS + NUMBER_OF_VOLUMES * r + v] * Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)];
			}
			vareps += (eps - meaneps) * (eps - meaneps) * c_Censored_Timepoints[v];
		}
		vareps /= ((float)NUMBER_OF_VOLUMES - 1.0f);
		Residual_Variances[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = vareps;

		// Loop over contrasts and calculate t-values, using a voxel-specific GLM scalar
		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			float contrast_value = 0.0f;
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + r] * Beta_Volumes[Calculate4DIndex(x,y,z,r,DATA_W,DATA_H,DATA_D)];
			}
			Contrast_Volumes[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_value;
			Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_value * rsqrt(vareps * d_GLM_Scalars[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)]);
		}
	}
}


__kernel void CalculateStatisticalMapsGLMTTestFirstLevelSlice(__global float* Statistical_Maps,
														      __global float* Contrast_Volumes,
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
		                                       	   	   	      __private int NUMBER_OF_CENSORED_TIMEPOINTS,
                                                              __private int slice)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	// First deal with voxels outside the mask
	if ( Mask[Calculate3DIndex(x,y,slice,DATA_W,DATA_H)] != 1.0f )
	{
		Residual_Variances[Calculate3DIndex(x,y,slice,DATA_W,DATA_H)] = 0.0f;

		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			Contrast_Volumes[Calculate4DIndex(x,y,slice,c,DATA_W,DATA_H,DATA_D)] = 0.0f;
			Statistical_Maps[Calculate4DIndex(x,y,slice,c,DATA_W,DATA_H,DATA_D)] = 0.0f;
		}

		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			//Residuals[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = 0.0f;
			Residuals[Calculate3DIndex(x,y,v,DATA_W,DATA_H)] = 0.0f;
		}

		return;
	}

	int t = 0;
	float eps, meaneps, vareps;

    // Get the specific voxel number for this brain voxel
    //int voxel_number = (int)d_Voxel_Numbers[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];
	int voxel_number = (int)d_Voxel_Numbers[Calculate2DIndex(x,y,DATA_W)];

	// Special case for a low number of regressors
	if (NUMBER_OF_REGRESSORS <= 25)
	{
		float beta[25];
		
		// Load beta values into registers
	    for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			beta[r] = Beta_Volumes[Calculate4DIndex(x,y,slice,r,DATA_W,DATA_H,DATA_D)];
		}

		// Calculate the mean of the error eps, using voxel-specific design models
		meaneps = 0.0f;
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			eps = Volumes[Calculate3DIndex(x,y,v,DATA_W,DATA_H)];

			// Calculate eps
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				eps -= d_X_GLM[voxel_number * NUMBER_OF_VOLUMES * NUMBER_OF_REGRESSORS + NUMBER_OF_VOLUMES * r + v] * beta[r];
				//eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
			}
			eps *= c_Censored_Timepoints[v];
			meaneps += eps;
	
			Residuals[Calculate3DIndex(x,y,v,DATA_W,DATA_H)] = eps;		
		}
		meaneps /= ((float)NUMBER_OF_VOLUMES);

		// Now calculate the variance of eps, using voxel-specific design models
		vareps = 0.0f;
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			eps = Volumes[Calculate3DIndex(x,y,v,DATA_W,DATA_H)];

			// Calculate eps
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				eps -= d_X_GLM[voxel_number * NUMBER_OF_VOLUMES * NUMBER_OF_REGRESSORS + NUMBER_OF_VOLUMES * r + v] * beta[r];
				//eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
			}
			vareps += (eps - meaneps) * (eps - meaneps) * c_Censored_Timepoints[v];
		}
		vareps /= ((float)NUMBER_OF_VOLUMES - 1.0f);
		Residual_Variances[Calculate3DIndex(x,y,slice,DATA_W,DATA_H)] = vareps;

		// Loop over contrasts and calculate t-values, using a voxel-specific GLM scalar
		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			float contrast_value = 0.0f;
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + r] * beta[r];
			}
			Contrast_Volumes[Calculate4DIndex(x,y,slice,c,DATA_W,DATA_H,DATA_D)] = contrast_value;
			Statistical_Maps[Calculate4DIndex(x,y,slice,c,DATA_W,DATA_H,DATA_D)] = contrast_value * rsqrt(vareps * d_GLM_Scalars[Calculate3DIndex(x,y,c,DATA_W,DATA_H)]);
		}
	}
	// General case for large number of regressors, slower
	else
	{
		// Calculate the mean of the error eps, using voxel-specific design models
		meaneps = 0.0f;
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			eps = Volumes[Calculate3DIndex(x,y,v,DATA_W,DATA_H)];

			// Calculate eps
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				eps -= d_X_GLM[voxel_number * NUMBER_OF_VOLUMES * NUMBER_OF_REGRESSORS + NUMBER_OF_VOLUMES * r + v] * Beta_Volumes[Calculate4DIndex(x,y,slice,r,DATA_W,DATA_H,DATA_D)];
				//eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
			}
			eps *= c_Censored_Timepoints[v];
			meaneps += eps;
	
			Residuals[Calculate3DIndex(x,y,v,DATA_W,DATA_H)] = eps;		
		}
		meaneps /= ((float)NUMBER_OF_VOLUMES);

		// Now calculate the variance of eps, using voxel-specific design models
		vareps = 0.0f;
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			eps = Volumes[Calculate3DIndex(x,y,v,DATA_W,DATA_H)];

			// Calculate eps
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				eps -= d_X_GLM[voxel_number * NUMBER_OF_VOLUMES * NUMBER_OF_REGRESSORS + NUMBER_OF_VOLUMES * r + v] * Beta_Volumes[Calculate4DIndex(x,y,slice,r,DATA_W,DATA_H,DATA_D)];
				//eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
			}
			vareps += (eps - meaneps) * (eps - meaneps) * c_Censored_Timepoints[v];
		}
		vareps /= ((float)NUMBER_OF_VOLUMES - 1.0f);
		Residual_Variances[Calculate3DIndex(x,y,slice,DATA_W,DATA_H)] = vareps;

		// Loop over contrasts and calculate t-values, using a voxel-specific GLM scalar
		for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			float contrast_value = 0.0f;
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + r] * Beta_Volumes[Calculate4DIndex(x,y,slice,r,DATA_W,DATA_H,DATA_D)];
			}
			Contrast_Volumes[Calculate4DIndex(x,y,slice,c,DATA_W,DATA_H,DATA_D)] = contrast_value;
			Statistical_Maps[Calculate4DIndex(x,y,slice,c,DATA_W,DATA_H,DATA_D)] = contrast_value * rsqrt(vareps * d_GLM_Scalars[Calculate3DIndex(x,y,c,DATA_W,DATA_H)]);
		}
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

	// First deal with voxels outside the mask
	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
	{
		Residual_Variances[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;

		Statistical_Maps[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)] = 0.0f;

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



__kernel void CalculateStatisticalMapsGLMFTestFirstLevelSlice(__global float* Statistical_Maps,
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
		                                       	   	   	      __private int NUMBER_OF_CENSORED_TIMEPOINTS,
 	 	 												      __private int slice)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	// First deal with voxels outside the mask
	if ( Mask[Calculate3DIndex(x,y,slice,DATA_W,DATA_H)] != 1.0f )
	{
		Residual_Variances[Calculate3DIndex(x,y,slice,DATA_W,DATA_H)] = 0.0f;

		Statistical_Maps[Calculate4DIndex(x,y,slice,0,DATA_W,DATA_H,DATA_D)] = 0.0f;

		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			Residuals[Calculate3DIndex(x,y,v,DATA_W,DATA_H)] = 0.0f;
		}

		return;
	}

	int t = 0;
	float eps, meaneps, vareps;
	float beta[25];

	// Load beta values into registers
    for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{
		beta[r] = Beta_Volumes[Calculate4DIndex(x,y,slice,r,DATA_W,DATA_H,DATA_D)];
	}

    // Get the specific voxel number for this brain voxel
    int voxel_number = (int)d_Voxel_Numbers[Calculate2DIndex(x,y,DATA_W)];

	// Calculate the mean of the error eps, using voxel-specific design models
	meaneps = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate3DIndex(x,y,v,DATA_W,DATA_H)];
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			eps -= d_X_GLM[voxel_number * NUMBER_OF_VOLUMES * NUMBER_OF_REGRESSORS + NUMBER_OF_VOLUMES * r + v] * beta[r];
			//eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
		}
		eps *= c_Censored_Timepoints[v];
		meaneps += eps;
		Residuals[Calculate3DIndex(x,y,v,DATA_W,DATA_H)] = eps;
	}
	//meaneps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_CENSORED_TIMEPOINTS);
	meaneps /= ((float)NUMBER_OF_VOLUMES);

	// Now calculate the variance of eps, using voxel-specific design models
	vareps = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate3DIndex(x,y,v,DATA_W,DATA_H)];
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
	Residual_Variances[Calculate3DIndex(x,y,slice,DATA_W,DATA_H)] = vareps;

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
			beta[c] += 1.0f/vareps * d_GLM_Scalars[Calculate3DIndex(x,y,cc + c * NUMBER_OF_CONTRASTS,DATA_W,DATA_H)] * cbeta[cc];
		}
	}

	// Finally calculate (C*beta)^T * temp
	float scalar = 0.0f;
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		scalar += cbeta[c] * beta[c];
	}

	// Save F-value
	Statistical_Maps[Calculate3DIndex(x,y,slice,DATA_W,DATA_H)] = scalar/(float)NUMBER_OF_CONTRASTS;
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
		//vareps += (eps - meaneps) * (eps - meaneps);
		vareps += eps*eps;
	}
	//vareps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_REGRESSORS - (float)NUMBER_OF_CENSORED_TIMEPOINTS - 1.0f);
	//vareps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_CENSORED_TIMEPOINTS - 1.0f);
	//vareps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_REGRESSORS - 1.0f);
	vareps /= ((float)NUMBER_OF_VOLUMES - NUMBER_OF_REGRESSORS);
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
	




// Optimized kernel for calculating t-test values for permutations, second level




__kernel void TransformData(__global float* Transformed_Volumes,
							__global float* Volumes,
		                    __global const float* Mask,
   	   	   				    __constant float* c_X,
		                    __private int DATA_W,
		                    __private int DATA_H,
		                    __private int DATA_D,
		                    __private int NUMBER_OF_VOLUMES)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
		return;

	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		Transformed_Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = 0.0f;

		for (int vv = 0; vv < NUMBER_OF_VOLUMES; vv++)
		{
			Transformed_Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] += c_X[vv + v * NUMBER_OF_VOLUMES] * Volumes[Calculate4DIndex(x,y,z,vv,DATA_W,DATA_H,DATA_D)];
		}
	}

	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)] = Transformed_Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
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
	float beta[100];

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


// Removes a linear fit estimated with CalculateGLMBetaWeights, for one slice
__kernel void RemoveLinearFitSlice(__global float* Residual_Volumes, 
		                           __global const float* Volumes, 
								   __global const float* Beta_Volumes, 
							  	   __global const float* Mask, 
							  	   __constant float *c_X_Detrend, 
							  	   __private int DATA_W, 
							  	   __private int DATA_H, 
							  	   __private int DATA_D, 
							  	   __private int NUMBER_OF_VOLUMES, 
							  	   __private int NUMBER_OF_REGRESSORS,
							  	   __private int slice)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,slice,DATA_W,DATA_H)] != 1.0f )
	{
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			Residual_Volumes[Calculate3DIndex(x,y,v,DATA_W,DATA_H)] = 0.0f;
		}

		return;
	}
	
	float eps;
	float beta[100];

	// Load beta values into regressors
    for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{ 
		beta[r] = Beta_Volumes[Calculate4DIndex(x,y,slice,r,DATA_W,DATA_H,DATA_D)];
	}

	// Calculate the residual
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		eps = Volumes[Calculate3DIndex(x,y,v,DATA_W,DATA_H)];
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{ 			
			eps -= beta[r] * c_X_Detrend[NUMBER_OF_VOLUMES * r + v];
		}
		Residual_Volumes[Calculate3DIndex(x,y,v,DATA_W,DATA_H)] = eps;
	}
}




