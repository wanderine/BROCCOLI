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

// For Linear image registration
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
	

__kernel void InterpolateVolumeNearestLinear(__global float* Volume,
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

__kernel void InterpolateVolumeNearestNonLinear(__global float* Volume,
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
	

__kernel void InterpolateVolumeLinearLinear(__global float* Volume,
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

__kernel void InterpolateVolumeLinearNonLinear(__global float* Volume,
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

__kernel void AddLinearAndNonLinearDisplacement(__global float* d_Displacement_Field_X,
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



__kernel void InterpolateVolumeCubicLinear(__global float* Volume,
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


__kernel void InterpolateVolumeCubicNonLinear(__global float* Volume,
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

__kernel void RescaleVolumeNearest(__global float* Volume,
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

	float4 Interpolated_Value = read_imagef(Original_Volume, volume_sampler_nearest, Motion_Vector);
	Volume[idx] = Interpolated_Value.x;
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



