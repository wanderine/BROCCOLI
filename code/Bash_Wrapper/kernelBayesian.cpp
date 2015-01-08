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


#pragma OPENCL EXTENSION cl_khr_fp64: enable

// Generate random uniform number by modulo operation


double unirand(__private int* seed)
{
	double const a = 16807.0; //ie 7**5
	double const m = 2147483647.0; //ie 2**31-1
	double const reciprocal_m = 1.0/m;
	double temp = (*seed) * a;
	//*seed = (int)(temp - m * floor(temp * reciprocal_m));

	return ((double)(*seed) * reciprocal_m);
}

#define pi 3.141592653589793

// Generate random normal number by Box-Muller transform
double normalrand(__private int* seed)
{
	double u = unirand(seed);
	double v = unirand(seed);

	//return sqrt(-2.0*log(u))*cos(2.0*pi*v);
	return 1.0;
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
												  __global float* Beta_Volumes,
												  __global float* AR_Estimates,
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
/*

	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
	{
		Statistical_Maps[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Statistical_Maps[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)] = 0.0f;

		Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)] = 0.0f;
		Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)] = 0.0f;

		AR_Estimates[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;

		return;
	}

	// Get seed from host
	int seed = Seeds[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];

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

	int probability1 = 0;
	int probability2 = 0;
	int probability3 = 0;
	int probability4 = 0;
	int probability5 = 0;
	int probability6 = 0;
	
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
				probability1++;
			}

			if (beta[1] > 0.0f)
			{
				probability2++;
			}

			if (beta[0] < 0.0f)
			{
				probability3++;
			}

			if (beta[1] < 0.0f)
			{
				probability4++;
			}

			if ((beta[0] - beta[1]) > 0.0f)
			{
				probability5++;
			}

			if ((beta[1] - beta[0]) > 0.0f)
			{
				probability6++;
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
	
	Statistical_Maps[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)] = (float)probability1/(float)NUMBER_OF_ITERATIONS;
	Statistical_Maps[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)] = (float)probability2/(float)NUMBER_OF_ITERATIONS;
	Statistical_Maps[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)] = (float)probability3/(float)NUMBER_OF_ITERATIONS;
	Statistical_Maps[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)] = (float)probability4/(float)NUMBER_OF_ITERATIONS;
	Statistical_Maps[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)] = (float)probability5/(float)NUMBER_OF_ITERATIONS;
	Statistical_Maps[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)] = (float)probability6/(float)NUMBER_OF_ITERATIONS;

	Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)] = beta[0];
	Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)] = beta[1];

	AR_Estimates[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = rhoT;
*/
}


