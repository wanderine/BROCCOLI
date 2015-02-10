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
								__private int INVALID_TIMEPOINTS,
                                __private int slice)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Mask[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)] != 1.0f )
	{
        AR1_Estimates[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)] = 0.0f;
		AR2_Estimates[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)] = 0.0f;
		AR3_Estimates[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)] = 0.0f;
		AR4_Estimates[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)] = 0.0f;

		return;
	}

    int t = 0;
	float old_value_1, old_value_2, old_value_3, old_value_4, old_value_5;
	float c0 = 0.0f;
    float c1 = 0.0f;
    float c2 = 0.0f;
    float c3 = 0.0f;
    float c4 = 0.0f;

    //old_value_1 = fMRI_Volumes[Calculate4DIndex(x, y, z, 0 + INVALID_TIMEPOINTS, DATA_W, DATA_H, DATA_D)];
	old_value_1 = fMRI_Volumes[Calculate3DIndex(x, y, 0 + INVALID_TIMEPOINTS, DATA_W, DATA_H)];
    c0 += old_value_1 * old_value_1;
    //old_value_2 = fMRI_Volumes[Calculate4DIndex(x, y, z, 1 + INVALID_TIMEPOINTS, DATA_W, DATA_H, DATA_D)];
	old_value_2 = fMRI_Volumes[Calculate3DIndex(x, y, 1 + INVALID_TIMEPOINTS, DATA_W, DATA_H)];
    c0 += old_value_2 * old_value_2;
    c1 += old_value_2 * old_value_1;
    //old_value_3 = fMRI_Volumes[Calculate4DIndex(x, y, z, 2 + INVALID_TIMEPOINTS, DATA_W, DATA_H, DATA_D)];
	old_value_3 = fMRI_Volumes[Calculate3DIndex(x, y, 2 + INVALID_TIMEPOINTS, DATA_W, DATA_H)];
    c0 += old_value_3 * old_value_3;
    c1 += old_value_3 * old_value_2;
    c2 += old_value_3 * old_value_1;
    //old_value_4 = fMRI_Volumes[Calculate4DIndex(x, y, z, 3 + INVALID_TIMEPOINTS, DATA_W, DATA_H, DATA_D)];
	old_value_4 = fMRI_Volumes[Calculate3DIndex(x, y, 3 + INVALID_TIMEPOINTS, DATA_W, DATA_H)];
    c0 += old_value_4 * old_value_4;
    c1 += old_value_4 * old_value_3;
    c2 += old_value_4 * old_value_2;
    c3 += old_value_4 * old_value_1;

    // Estimate c0, c1, c2, c3, c4
    for (t = 4 + INVALID_TIMEPOINTS; t < DATA_T; t++)
    {
        // Read data into register
        //old_value_5 = fMRI_Volumes[Calculate4DIndex(x, y, z, t, DATA_W, DATA_H, DATA_D)];
		old_value_5 = fMRI_Volumes[Calculate3DIndex(x, y, t, DATA_W, DATA_H)];
        
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

        AR1_Estimates[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)] = alphas.x;
		AR2_Estimates[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)] = alphas.y;
		AR3_Estimates[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)] = alphas.z;
		AR4_Estimates[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)] = alphas.w;
    }
    else
    {
		AR1_Estimates[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)] = 0.0f;
        AR2_Estimates[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)] = 0.0f;
		AR3_Estimates[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)] = 0.0f;
		AR4_Estimates[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)] = 0.0f;
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
								__private int DATA_T,
                                __private int slice)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if ( x >= DATA_W || y >= DATA_H || z >= DATA_D )
        return;

    if ( Mask[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)] != 1.0f )
		return;

    int t = 0;
	float old_value_1, old_value_2, old_value_3, old_value_4, old_value_5;
    float4 alphas;
	alphas.x = AR1_Estimates[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)];
    alphas.y = AR2_Estimates[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)];
    alphas.z = AR3_Estimates[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)];
    alphas.w = AR4_Estimates[Calculate3DIndex(x, y, slice, DATA_W, DATA_H)];

    // Calculate the whitened timeseries

	/*
    old_value_1 = fMRI_Volumes[Calculate4DIndex(x, y, z, 0, DATA_W, DATA_H, DATA_D)];	
    Whitened_fMRI_Volumes[Calculate4DIndex(x, y, z, 0, DATA_W, DATA_H, DATA_D)] = old_value_1;
    old_value_2 = fMRI_Volumes[Calculate4DIndex(x, y, z, 1, DATA_W, DATA_H, DATA_D)];
    Whitened_fMRI_Volumes[Calculate4DIndex(x, y, z, 1, DATA_W, DATA_H, DATA_D)] = old_value_2  - alphas.x * old_value_1;
    old_value_3 = fMRI_Volumes[Calculate4DIndex(x, y, z, 2, DATA_W, DATA_H, DATA_D)];
    Whitened_fMRI_Volumes[Calculate4DIndex(x, y, z, 2, DATA_W, DATA_H, DATA_D)] = old_value_3 - alphas.x * old_value_2 - alphas.y * old_value_1;
    old_value_4 = fMRI_Volumes[Calculate4DIndex(x, y, z, 3, DATA_W, DATA_H, DATA_D)];
    Whitened_fMRI_Volumes[Calculate4DIndex(x, y, z, 3, DATA_W, DATA_H, DATA_D)] = old_value_4 - alphas.x * old_value_3 - alphas.y * old_value_2 - alphas.z * old_value_1;
	*/

    old_value_1 = fMRI_Volumes[Calculate3DIndex(x, y,  0, DATA_W, DATA_H)];	
    Whitened_fMRI_Volumes[Calculate3DIndex(x, y, 0, DATA_W, DATA_H)] = old_value_1;
    old_value_2 = fMRI_Volumes[Calculate3DIndex(x, y, 1, DATA_W, DATA_H)];
    Whitened_fMRI_Volumes[Calculate3DIndex(x, y, 1, DATA_W, DATA_H)] = old_value_2  - alphas.x * old_value_1;
    old_value_3 = fMRI_Volumes[Calculate3DIndex(x, y, 2, DATA_W, DATA_H)];
    Whitened_fMRI_Volumes[Calculate3DIndex(x, y, 2, DATA_W, DATA_H)] = old_value_3 - alphas.x * old_value_2 - alphas.y * old_value_1;
    old_value_4 = fMRI_Volumes[Calculate3DIndex(x, y, 3, DATA_W, DATA_H)];
    Whitened_fMRI_Volumes[Calculate3DIndex(x, y, 3, DATA_W, DATA_H)] = old_value_4 - alphas.x * old_value_3 - alphas.y * old_value_2 - alphas.z * old_value_1;

    for (t = 4; t < DATA_T; t++)
    {
        //old_value_5 = fMRI_Volumes[Calculate4DIndex(x, y, z, t, DATA_W, DATA_H, DATA_D)];
        //Whitened_fMRI_Volumes[Calculate4DIndex(x, y, z, t, DATA_W, DATA_H, DATA_D)] = old_value_5 - alphas.x * old_value_4 - alphas.y * old_value_3 - alphas.z * old_value_2 - alphas.w * old_value_1;

        old_value_5 = fMRI_Volumes[Calculate3DIndex(x, y, t, DATA_W, DATA_H)];
        Whitened_fMRI_Volumes[Calculate3DIndex(x, y, t, DATA_W, DATA_H)] = old_value_5 - alphas.x * old_value_4 - alphas.y * old_value_3 - alphas.z * old_value_2 - alphas.w * old_value_1;

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



