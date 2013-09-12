
void BROCCOLI_LIB::InvertAffineRegistrationParameters(float* h_Inverse_Parameters, float* h_Parameters)
{
	// Put parameters in 4 x 4 affine transformation matrix

	// (p3 p4  p5  tx)
	// (p6 p7  p8  ty)
	// (p9 p10 p11 tz)
	// (0  0   0   1 )

	double Affine_Matrix[16], Inverse_Affine_Matrix[16];

	// Add ones in the diagonal, to get a transformation matrix
	// First row
	Affine_Matrix[0] = (double)(h_Parameters[3] + 1.0f);
	Affine_Matrix[1] = (double)h_Parameters[4];
	Affine_Matrix[2] = (double)h_Parameters[5];
	Affine_Matrix[3] = (double)h_Parameters[0];

	// Second row
	Affine_Matrix[4] = (double)h_Parameters[6];
	Affine_Matrix[5] = (double)(h_Parameters[7] + 1.0f);
	Affine_Matrix[6] = (double)h_Parameters[8];
	Affine_Matrix[7] = (double)h_Parameters[1];

	// Third row
	Affine_Matrix[8]  = (double)h_Parameters[9];
	Affine_Matrix[9]  = (double)h_Parameters[10];
	Affine_Matrix[10] = (double)(h_Parameters[11] + 1.0f);
	Affine_Matrix[11] = (double)h_Parameters[2];

	// Fourth row
	Affine_Matrix[12] = 0.0;
	Affine_Matrix[13] = 0.0;
	Affine_Matrix[14] = 0.0;
	Affine_Matrix[15] = 1.0;

	// Invert the affine transformation matrix, th get the inverse parameters
	InvertMatrixDouble(Inverse_Affine_Matrix, Affine_Matrix, 4);

	// Subtract ones in the diagonal
	// First row
	h_Inverse_Parameters[0] = (float)Inverse_Affine_Matrix[3];
	h_Inverse_Parameters[1] = (float)Inverse_Affine_Matrix[7];
	h_Inverse_Parameters[2] = (float)Inverse_Affine_Matrix[11];

	// Second row
	h_Inverse_Parameters[3] = (float)(Inverse_Affine_Matrix[0] - 1.0);
	h_Inverse_Parameters[4] = (float)Inverse_Affine_Matrix[1];
	h_Inverse_Parameters[5] = (float)Inverse_Affine_Matrix[2];

	// Third row
	h_Inverse_Parameters[6] = (float)Inverse_Affine_Matrix[4];
	h_Inverse_Parameters[7] = (float)(Inverse_Affine_Matrix[5] - 1.0);
	h_Inverse_Parameters[8] = (float)Inverse_Affine_Matrix[6];

	// Fourth row
	h_Inverse_Parameters[9] = (float)Inverse_Affine_Matrix[8];
	h_Inverse_Parameters[10] = (float)Inverse_Affine_Matrix[9];
	h_Inverse_Parameters[11] = (float)(Inverse_Affine_Matrix[10] - 1.0);
}

void BROCCOLI_LIB::AddAffineRegistrationParameters(float* h_Old_Parameters, float* h_New_Parameters)
{
	// Put parameters in 4 x 4 affine transformation matrix

	// (p3 p4  p5  tx)
	// (p6 p7  p8  ty)
	// (p9 p10 p11 tz)
	// (0  0   0   1 )

	double Old_Affine_Matrix[16], New_Affine_Matrix[16], Temp_Affine_Matrix[16];
	
	// Add ones in the diagonal, to get a transformation matrix
	// First row
	Old_Affine_Matrix[0] = (double)(h_Old_Parameters[3] + 1.0f);
	Old_Affine_Matrix[1] = (double)h_Old_Parameters[4];
	Old_Affine_Matrix[2] = (double)h_Old_Parameters[5];
	Old_Affine_Matrix[3] = (double)h_Old_Parameters[0];

	// Second row
	Old_Affine_Matrix[4] = (double)h_Old_Parameters[6];
	Old_Affine_Matrix[5] = (double)(h_Old_Parameters[7] + 1.0f);
	Old_Affine_Matrix[6] = (double)h_Old_Parameters[8];
	Old_Affine_Matrix[7] = (double)h_Old_Parameters[1];

	// Third row
	Old_Affine_Matrix[8]  = (double)h_Old_Parameters[9];
	Old_Affine_Matrix[9]  = (double)h_Old_Parameters[10];
	Old_Affine_Matrix[10] = (double)(h_Old_Parameters[11] + 1.0f);
	Old_Affine_Matrix[11] = (double)h_Old_Parameters[2];

	// Fourth row
	Old_Affine_Matrix[12] = 0.0;
	Old_Affine_Matrix[13] = 0.0;
	Old_Affine_Matrix[14] = 0.0;
	Old_Affine_Matrix[15] = 1.0;

	// Add ones in the diagonal, to get a transformation matrix
	// First row
	New_Affine_Matrix[0] = (double)(h_New_Parameters[3] + 1.0f);
	New_Affine_Matrix[1] = (double)h_New_Parameters[4];
	New_Affine_Matrix[2] = (double)h_New_Parameters[5];
	New_Affine_Matrix[3] = (double)h_New_Parameters[0];

	// Second row
	New_Affine_Matrix[4] = (double)h_New_Parameters[6];
	New_Affine_Matrix[5] = (double)(h_New_Parameters[7] + 1.0f);
	New_Affine_Matrix[6] = (double)h_New_Parameters[8];
	New_Affine_Matrix[7] = (double)h_New_Parameters[1];

	// Third row
	New_Affine_Matrix[8]  = (double)h_New_Parameters[9];
	New_Affine_Matrix[9]  = (double)h_New_Parameters[10];
	New_Affine_Matrix[10] = (double)(h_New_Parameters[11] + 1.0f);
	New_Affine_Matrix[11] = (double)h_New_Parameters[2];

	// Fourth row
	New_Affine_Matrix[12] = 0.0;
	New_Affine_Matrix[13] = 0.0;
	New_Affine_Matrix[14] = 0.0;
	New_Affine_Matrix[15] = 1.0;

	//
	MatMul4x4(Temp_Affine_Matrix, New_Affine_Matrix, Old_Affine_Matrix);

	// Subtract ones in the diagonal
	// First row
	h_Old_Parameters[0] = (float)Temp_Affine_Matrix[3];
	h_Old_Parameters[1] = (float)Temp_Affine_Matrix[7];
	h_Old_Parameters[2] = (float)Temp_Affine_Matrix[11];

	// Second row
	h_Old_Parameters[3] = (float)(Temp_Affine_Matrix[0] - 1.0);
	h_Old_Parameters[4] = (float)Temp_Affine_Matrix[1];
	h_Old_Parameters[5] = (float)Temp_Affine_Matrix[2];

	// Third row
	h_Old_Parameters[6] = (float)Temp_Affine_Matrix[4];
	h_Old_Parameters[7] = (float)(Temp_Affine_Matrix[5] - 1.0);
	h_Old_Parameters[8] = (float)Temp_Affine_Matrix[6];

	// Fourth row
	h_Old_Parameters[9] = (float)Temp_Affine_Matrix[8];
	h_Old_Parameters[10] = (float)Temp_Affine_Matrix[9];
	h_Old_Parameters[11] = (float)(Temp_Affine_Matrix[10] - 1.0);
}


void BROCCOLI_LIB::AddAffineRegistrationParametersNextScale(float* h_Old_Parameters, float* h_New_Parameters)
{
	// Put parameters in 4 x 4 affine transformation matrix
	
	// (p3 p4  p5  tx)
	// (p6 p7  p8  ty)
	// (p9 p10 p11 tz)
	// (0  0   0   1 )


	double Old_Affine_Matrix[16], New_Affine_Matrix[16], Temp_Affine_Matrix[16];
	
	// Add ones in the diagonal, to get a transformation matrix
	// First row
	Old_Affine_Matrix[0] = (double)(h_Old_Parameters[3] + 1.0f);
	Old_Affine_Matrix[1] = (double)h_Old_Parameters[4];
	Old_Affine_Matrix[2] = (double)h_Old_Parameters[5];
	Old_Affine_Matrix[3] = (double)(h_Old_Parameters[0] * 2.0f);

	// Second row
	Old_Affine_Matrix[4] = (double)h_Old_Parameters[6];
	Old_Affine_Matrix[5] = (double)(h_Old_Parameters[7] + 1.0f);
	Old_Affine_Matrix[6] = (double)h_Old_Parameters[8];
	Old_Affine_Matrix[7] = (double)(h_Old_Parameters[1] * 2.0f);

	// Third row
	Old_Affine_Matrix[8]  = (double)h_Old_Parameters[9];
	Old_Affine_Matrix[9]  = (double)h_Old_Parameters[10];
	Old_Affine_Matrix[10] = (double)(h_Old_Parameters[11] + 1.0f);
	Old_Affine_Matrix[11] = (double)(h_Old_Parameters[2] * 2.0f);

	// Fourth row
	Old_Affine_Matrix[12] = 0.0;
	Old_Affine_Matrix[13] = 0.0;
	Old_Affine_Matrix[14] = 0.0;
	Old_Affine_Matrix[15] = 1.0;

	// Add ones in the diagonal, to get a transformation matrix
	// First row
	New_Affine_Matrix[0] = (double)(h_New_Parameters[3] + 1.0f);
	New_Affine_Matrix[1] = (double)h_New_Parameters[4];
	New_Affine_Matrix[2] = (double)h_New_Parameters[5];
	New_Affine_Matrix[3] = (double)(h_New_Parameters[0] * 2.0f);

	// Second row
	New_Affine_Matrix[4] = (double)h_New_Parameters[6];
	New_Affine_Matrix[5] = (double)(h_New_Parameters[7] + 1.0f);
	New_Affine_Matrix[6] = (double)h_New_Parameters[8];
	New_Affine_Matrix[7] = (double)(h_New_Parameters[1] * 2.0f);

	// Third row
	New_Affine_Matrix[8]  = (double)h_New_Parameters[9];
	New_Affine_Matrix[9]  = (double)h_New_Parameters[10];
	New_Affine_Matrix[10] = (double)(h_New_Parameters[11] + 1.0f);
	New_Affine_Matrix[11] = (double)(h_New_Parameters[2] * 2.0f);

	// Fourth row
	New_Affine_Matrix[12] = 0.0;
	New_Affine_Matrix[13] = 0.0;
	New_Affine_Matrix[14] = 0.0;
	New_Affine_Matrix[15] = 1.0;

	//
	MatMul4x4(Temp_Affine_Matrix, New_Affine_Matrix, Old_Affine_Matrix);

	// Subtract ones in the diagonal
	// First row
	h_Old_Parameters[0] = (float)Temp_Affine_Matrix[3];
	h_Old_Parameters[1] = (float)Temp_Affine_Matrix[7];
	h_Old_Parameters[2] = (float)Temp_Affine_Matrix[11];

	// Second row
	h_Old_Parameters[3] = (float)(Temp_Affine_Matrix[0] - 1.0);
	h_Old_Parameters[4] = (float)Temp_Affine_Matrix[1];
	h_Old_Parameters[5] = (float)Temp_Affine_Matrix[2];

	// Third row
	h_Old_Parameters[6] = (float)Temp_Affine_Matrix[4];
	h_Old_Parameters[7] = (float)(Temp_Affine_Matrix[5] - 1.0);
	h_Old_Parameters[8] = (float)Temp_Affine_Matrix[6];

	// Fourth row
	h_Old_Parameters[9] = (float)Temp_Affine_Matrix[8];
	h_Old_Parameters[10] = (float)Temp_Affine_Matrix[9];
	h_Old_Parameters[11] = (float)(Temp_Affine_Matrix[10] - 1.0);
}


void BROCCOLI_LIB::AddAffineRegistrationParameters(float* h_Resulting_Parameters, float* h_Old_Parameters, float* h_New_Parameters)
{
	// Put parameters in 4 x 4 affine transformation matrix

	// (p3 p4  p5  tx)
	// (p6 p7  p8  ty)
	// (p9 p10 p11 tz)
	// (0  0   0   1 )


	double Old_Affine_Matrix[16], New_Affine_Matrix[16], Temp_Affine_Matrix[16];
	
	// Add ones in the diagonal, to get a transformation matrix
	// First row
	Old_Affine_Matrix[0] = (double)(h_Old_Parameters[3] + 1.0f);
	Old_Affine_Matrix[1] = (double)h_Old_Parameters[4];
	Old_Affine_Matrix[2] = (double)h_Old_Parameters[5];
	Old_Affine_Matrix[3] = (double)h_Old_Parameters[0];

	// Second row
	Old_Affine_Matrix[4] = (double)h_Old_Parameters[6];
	Old_Affine_Matrix[5] = (double)(h_Old_Parameters[7] + 1.0f);
	Old_Affine_Matrix[6] = (double)h_Old_Parameters[8];
	Old_Affine_Matrix[7] = (double)h_Old_Parameters[1];

	// Third row
	Old_Affine_Matrix[8]  = (double)h_Old_Parameters[9];
	Old_Affine_Matrix[9]  = (double)h_Old_Parameters[10];
	Old_Affine_Matrix[10] = (double)(h_Old_Parameters[11] + 1.0f);
	Old_Affine_Matrix[11] = (double)h_Old_Parameters[2];

	// Fourth row
	Old_Affine_Matrix[12] = 0.0;
	Old_Affine_Matrix[13] = 0.0;
	Old_Affine_Matrix[14] = 0.0;
	Old_Affine_Matrix[15] = 1.0;

	// Add ones in the diagonal, to get a transformation matrix
	// First row
	New_Affine_Matrix[0] = (double)(h_New_Parameters[3] + 1.0f);
	New_Affine_Matrix[1] = (double)h_New_Parameters[4];
	New_Affine_Matrix[2] = (double)h_New_Parameters[5];
	New_Affine_Matrix[3] = (double)h_New_Parameters[0];

	// Second row
	New_Affine_Matrix[4] = (double)h_New_Parameters[6];
	New_Affine_Matrix[5] = (double)(h_New_Parameters[7] + 1.0f);
	New_Affine_Matrix[6] = (double)h_New_Parameters[8];
	New_Affine_Matrix[7] = (double)h_New_Parameters[1];

	// Third row
	New_Affine_Matrix[8]  = (double)h_New_Parameters[9];
	New_Affine_Matrix[9]  = (double)h_New_Parameters[10];
	New_Affine_Matrix[10] = (double)(h_New_Parameters[11] + 1.0f);
	New_Affine_Matrix[11] = (double)h_New_Parameters[2];

	// Fourth row
	New_Affine_Matrix[12] = 0.0;
	New_Affine_Matrix[13] = 0.0;
	New_Affine_Matrix[14] = 0.0;
	New_Affine_Matrix[15] = 1.0;

	//
	MatMul4x4(Temp_Affine_Matrix, New_Affine_Matrix, Old_Affine_Matrix);

	// Subtract ones in the diagonal
	// First row
	h_Resulting_Parameters[0] = (float)Temp_Affine_Matrix[3];
	h_Resulting_Parameters[1] = (float)Temp_Affine_Matrix[7];
	h_Resulting_Parameters[2] = (float)Temp_Affine_Matrix[11];

	// Second row
	h_Resulting_Parameters[3] = (float)(Temp_Affine_Matrix[0] - 1.0);
	h_Resulting_Parameters[4] = (float)Temp_Affine_Matrix[1];
	h_Resulting_Parameters[5] = (float)Temp_Affine_Matrix[2];

	// Third row
	h_Resulting_Parameters[6] = (float)Temp_Affine_Matrix[4];
	h_Resulting_Parameters[7] = (float)(Temp_Affine_Matrix[5] - 1.0);
	h_Resulting_Parameters[8] = (float)Temp_Affine_Matrix[6];

	// Fourth row
	h_Resulting_Parameters[9] = (float)Temp_Affine_Matrix[8];
	h_Resulting_Parameters[10] = (float)Temp_Affine_Matrix[9];
	h_Resulting_Parameters[11] = (float)(Temp_Affine_Matrix[10] - 1.0);
}


void BROCCOLI_LIB::InvertMatrix(float* inverse_matrix, float* matrix, int N)
{
    int i = 0;
    int j = 0;
    int k = 0;

	int NUMBER_OF_ROWS = N;
    int NUMBER_OF_COLUMNS = N;
    int n = N;
	int m = N;

    float* LU = (float*)malloc(sizeof(float) * N * N);

    /* Copy A to LU matrix */
    for(i = 0; i < NUMBER_OF_ROWS * NUMBER_OF_COLUMNS; i++)
    {
        LU[i] = matrix[i];
    }

    /* Perform LU decomposition */
    float* piv = (float*)malloc(sizeof(float) * N);
    for (i = 0; i < m; i++)
    {
        piv[i] = (float)i;
    }
    float pivsign = 1.0f;
    /* Main loop */
    for (k = 0; k < n; k++)
    {
        /* Find pivot */
        int p = k;
        for (i = k+1; i < m; i++)
        {
            if (abs(LU[i + k * NUMBER_OF_ROWS]) > abs(LU[p + k * NUMBER_OF_ROWS]))
            {
                p = i;
            }
        }
        /* Exchange if necessary */
        if (p != k)
        {
            for (j = 0; j < n; j++)
            {
                float t = LU[p + j*NUMBER_OF_ROWS]; LU[p + j*NUMBER_OF_ROWS] = LU[k + j*NUMBER_OF_ROWS]; LU[k + j*NUMBER_OF_ROWS] = t;
            }
            int t = (int)piv[p]; piv[p] = piv[k]; piv[k] = (float)t;
            pivsign = -pivsign;
        }
        /* Compute multipliers and eliminate k-th column */
        if (LU[k + k*NUMBER_OF_ROWS] != 0.0f)
        {
            for (i = k+1; i < m; i++)
            {
                LU[i + k*NUMBER_OF_ROWS] /= LU[k + k*NUMBER_OF_ROWS];
                for (j = k+1; j < n; j++)
                {
                    LU[i + j*NUMBER_OF_ROWS] -= LU[i + k*NUMBER_OF_ROWS]*LU[k + j*NUMBER_OF_ROWS];
                }
            }
        }
    }

    /* "Solve" equation system AX = B with B = identity matrix
     to get matrix inverse */

    /* Make an identity matrix of the right size */
    float* B = (float*)malloc(sizeof(float) * N * N);
    float* X = (float*)malloc(sizeof(float) * N * N);

    for (i = 0; i < NUMBER_OF_ROWS; i++)
    {
        for (j = 0; j < NUMBER_OF_COLUMNS; j++)
        {
            if (i == j)
            {
                B[i + j * NUMBER_OF_ROWS] = 1;
            }
            else
            {
                B[i + j * NUMBER_OF_ROWS] = 0;
            }
        }
    }

    /* Pivot the identity matrix */
    for (i = 0; i < NUMBER_OF_ROWS; i++)
    {
        int current_row = (int)piv[i];

        for (j = 0; j < NUMBER_OF_COLUMNS; j++)
        {
            X[i + j * NUMBER_OF_ROWS] = B[current_row + j * NUMBER_OF_ROWS];
        }
    }

    /* Solve L*Y = B(piv,:) */
    for (k = 0; k < n; k++)
    {
        for (i = k+1; i < n; i++)
        {
            for (j = 0; j < NUMBER_OF_COLUMNS; j++)
            {
                X[i + j*NUMBER_OF_ROWS] -= X[k + j*NUMBER_OF_ROWS]*LU[i + k*NUMBER_OF_ROWS];
            }
        }
    }
    /* Solve U*X = Y */
    for (k = n-1; k >= 0; k--)
    {
        for (j = 0; j < NUMBER_OF_COLUMNS; j++)
        {
            X[k + j*NUMBER_OF_ROWS] /= LU[k + k*NUMBER_OF_ROWS];
        }
        for (i = 0; i < k; i++)
        {
            for (j = 0; j < NUMBER_OF_COLUMNS; j++)
            {
                X[i + j*NUMBER_OF_ROWS] -= X[k + j*NUMBER_OF_ROWS]*LU[i + k*NUMBER_OF_ROWS];
            }
        }
    }

    for (i = 0; i < NUMBER_OF_ROWS; i++)
    {
        for (j = 0; j < NUMBER_OF_COLUMNS; j++)
        {
            inverse_matrix[i + j * NUMBER_OF_ROWS] = X[i + j * NUMBER_OF_ROWS];
        }
    }

	free(LU);
	free(piv);
	free(B);
	free(X);
}

void BROCCOLI_LIB::InvertMatrixDouble(double* inverse_matrix, double* matrix, int N)
{
    int i = 0;
    int j = 0;
    int k = 0;

	int NUMBER_OF_ROWS = N;
    int NUMBER_OF_COLUMNS = N;
    int n = N;
	int m = N;

    double* LU = (double*)malloc(sizeof(double) * N * N);

    /* Copy A to LU matrix */
    for(i = 0; i < NUMBER_OF_ROWS * NUMBER_OF_COLUMNS; i++)
    {
        LU[i] = matrix[i];
    }

    /* Perform LU decomposition */
    double* piv = (double*)malloc(sizeof(double) * N);
    for (i = 0; i < m; i++)
    {
        piv[i] = (double)i;
    }
    double pivsign = 1.0;
    /* Main loop */
    for (k = 0; k < n; k++)
    {
        /* Find pivot */
        int p = k;
        for (i = k+1; i < m; i++)
        {
            if (abs(LU[i + k * NUMBER_OF_ROWS]) > abs(LU[p + k * NUMBER_OF_ROWS]))
            {
                p = i;
            }
        }
        /* Exchange if necessary */
        if (p != k)
        {
            for (j = 0; j < n; j++)
            {
                double t = LU[p + j*NUMBER_OF_ROWS]; LU[p + j*NUMBER_OF_ROWS] = LU[k + j*NUMBER_OF_ROWS]; LU[k + j*NUMBER_OF_ROWS] = t;
            }
            int t = (int)piv[p]; piv[p] = piv[k]; piv[k] = (double)t;
            pivsign = -pivsign;
        }
        /* Compute multipliers and eliminate k-th column */
        if (LU[k + k*NUMBER_OF_ROWS] != 0.0)
        {
            for (i = k+1; i < m; i++)
            {
                LU[i + k*NUMBER_OF_ROWS] /= LU[k + k*NUMBER_OF_ROWS];
                for (j = k+1; j < n; j++)
                {
                    LU[i + j*NUMBER_OF_ROWS] -= LU[i + k*NUMBER_OF_ROWS]*LU[k + j*NUMBER_OF_ROWS];
                }
            }
        }
    }

    /* "Solve" equation system AX = B with B = identity matrix
     to get matrix inverse */

    /* Make an identity matrix of the right size */
    double* B = (double*)malloc(sizeof(double) * N * N);
    double* X = (double*)malloc(sizeof(double) * N * N);

    for (i = 0; i < NUMBER_OF_ROWS; i++)
    {
        for (j = 0; j < NUMBER_OF_COLUMNS; j++)
        {
            if (i == j)
            {
                B[i + j * NUMBER_OF_ROWS] = 1;
            }
            else
            {
                B[i + j * NUMBER_OF_ROWS] = 0;
            }
        }
    }

    /* Pivot the identity matrix */
    for (i = 0; i < NUMBER_OF_ROWS; i++)
    {
        int current_row = (int)piv[i];

        for (j = 0; j < NUMBER_OF_COLUMNS; j++)
        {
            X[i + j * NUMBER_OF_ROWS] = B[current_row + j * NUMBER_OF_ROWS];
        }
    }

    /* Solve L*Y = B(piv,:) */
    for (k = 0; k < n; k++)
    {
        for (i = k+1; i < n; i++)
        {
            for (j = 0; j < NUMBER_OF_COLUMNS; j++)
            {
                X[i + j*NUMBER_OF_ROWS] -= X[k + j*NUMBER_OF_ROWS]*LU[i + k*NUMBER_OF_ROWS];
            }
        }
    }
    /* Solve U*X = Y */
    for (k = n-1; k >= 0; k--)
    {
        for (j = 0; j < NUMBER_OF_COLUMNS; j++)
        {
            X[k + j*NUMBER_OF_ROWS] /= LU[k + k*NUMBER_OF_ROWS];
        }
        for (i = 0; i < k; i++)
        {
            for (j = 0; j < NUMBER_OF_COLUMNS; j++)
            {
                X[i + j*NUMBER_OF_ROWS] -= X[k + j*NUMBER_OF_ROWS]*LU[i + k*NUMBER_OF_ROWS];
            }
        }
    }

    for (i = 0; i < NUMBER_OF_ROWS; i++)
    {
        for (j = 0; j < NUMBER_OF_COLUMNS; j++)
        {
            inverse_matrix[i + j * NUMBER_OF_ROWS] = X[i + j * NUMBER_OF_ROWS];
        }
    }

	free(LU);
	free(piv);
	free(B);
	free(X);
}





/*
__kernel void CalculateAMatricesAndHVectors(__global float* a11, 
	                                        __global float* a12, 
											__global float* a13, 
											__global float* a22, 
											__global float* a23, 
											__global float* a33, 
											__global float* h1, 
											__global float* h2, 
											__global float* h3, 
											__global const float* Phase_Differences, 
											__global const float* Certainties, 
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
											__private int DATA_D) 
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	
	if ((x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D) )
		return;

	int idx = x + y * DATA_W + z * DATA_W * DATA_H;
	int offset = DATA_W * DATA_H * DATA_D;

	float c, pd;
	float a11Temp = 0.0f;
	float a12Temp = 0.0f;
	float a13Temp = 0.0f;
	float a22Temp = 0.0f;
	float a23Temp = 0.0f;
	float a33Temp = 0.0f;
	float h1Temp = 0.0f;
	float h2Temp = 0.0f;
	float h3Temp = 0.0f;
	float tt11, tt12, tt13, tt22, tt23, tt33;
		
	tt11 = t11[idx] * t11[idx] + t12[idx] * t12[idx] + t13[idx] * t13[idx];
    tt12 = t11[idx] * t12[idx] + t12[idx] * t22[idx] + t13[idx] * t23[idx];
    tt13 = t11[idx] * t13[idx] + t12[idx] * t23[idx] + t13[idx] * t33[idx];
    tt22 = t12[idx] * t12[idx] + t22[idx] * t22[idx] + t23[idx] * t23[idx];
    tt23 = t12[idx] * t13[idx] + t22[idx] * t23[idx] + t23[idx] * t33[idx];
    tt33 = t13[idx] * t13[idx] + t23[idx] * t23[idx] + t33[idx] * t33[idx];
        
	// First quadrature filter
	c = Certainties[idx + 0*offset];
	a11Temp += c * tt11;
	a12Temp += c * tt12;
	a13Temp += c * tt13;
	a22Temp += c * tt22;
	a23Temp += c * tt23;
	a33Temp += c * tt33;
			
	pd = Phase_Differences[idx + 0*offset];
	h1Temp += c * pd * (c_Filter_Directions_X[0] * tt11 + c_Filter_Directions_Y[0] * tt12 + c_Filter_Directions_Z[0] * tt13);
	h2Temp += c * pd * (c_Filter_Directions_X[0] * tt12 + c_Filter_Directions_Y[0] * tt22 + c_Filter_Directions_Z[0] * tt23);
	h3Temp += c * pd * (c_Filter_Directions_X[0] * tt13 + c_Filter_Directions_Y[0] * tt23 + c_Filter_Directions_Z[0] * tt33);
		
	// Second quadrature filter
	c = Certainties[idx + 1*offset];
	a11Temp += c * tt11;
	a12Temp += c * tt12;
	a13Temp += c * tt13;
	a22Temp += c * tt22;
	a23Temp += c * tt23;
	a33Temp += c * tt33;
			
	pd = Phase_Differences[idx + 1*offset];
	h1Temp += c * pd * (c_Filter_Directions_X[1] * tt11 + c_Filter_Directions_Y[1] * tt12 + c_Filter_Directions_Z[1] * tt13);
	h2Temp += c * pd * (c_Filter_Directions_X[1] * tt12 + c_Filter_Directions_Y[1] * tt22 + c_Filter_Directions_Z[1] * tt23);
	h3Temp += c * pd * (c_Filter_Directions_X[1] * tt13 + c_Filter_Directions_Y[1] * tt23 + c_Filter_Directions_Z[1] * tt33);
	
	// Third quadrature filter
	c = Certainties[idx + 2*offset];
	a11Temp += c * tt11;
	a12Temp += c * tt12;
	a13Temp += c * tt13;
	a22Temp += c * tt22;
	a23Temp += c * tt23;
	a33Temp += c * tt33;
			
	pd = Phase_Differences[idx + 2*offset];
	h1Temp += c * pd * (c_Filter_Directions_X[2] * tt11 + c_Filter_Directions_Y[2] * tt12 + c_Filter_Directions_Z[2] * tt13);
	h2Temp += c * pd * (c_Filter_Directions_X[2] * tt12 + c_Filter_Directions_Y[2] * tt22 + c_Filter_Directions_Z[2] * tt23);
	h3Temp += c * pd * (c_Filter_Directions_X[2] * tt13 + c_Filter_Directions_Y[2] * tt23 + c_Filter_Directions_Z[2] * tt33);
	
	// Fourth quadrature filter
	c = Certainties[idx + 3*offset];
	a11Temp += c * tt11;
	a12Temp += c * tt12;
	a13Temp += c * tt13;
	a22Temp += c * tt22;
	a23Temp += c * tt23;
	a33Temp += c * tt33;
			
	pd = Phase_Differences[idx + 3*offset];
	h1Temp += c * pd * (c_Filter_Directions_X[3] * tt11 + c_Filter_Directions_Y[3] * tt12 + c_Filter_Directions_Z[3] * tt13);
	h2Temp += c * pd * (c_Filter_Directions_X[3] * tt12 + c_Filter_Directions_Y[3] * tt22 + c_Filter_Directions_Z[3] * tt23);
	h3Temp += c * pd * (c_Filter_Directions_X[3] * tt13 + c_Filter_Directions_Y[3] * tt23 + c_Filter_Directions_Z[3] * tt33);
	
	// Fifth quadrature filter
	c = Certainties[idx + 4*offset];
	a11Temp += c * tt11;
	a12Temp += c * tt12;
	a13Temp += c * tt13;
	a22Temp += c * tt22;
	a23Temp += c * tt23;
	a33Temp += c * tt33;
			
	pd = Phase_Differences[idx + 4*offset];
	h1Temp += c * pd * (c_Filter_Directions_X[4] * tt11 + c_Filter_Directions_Y[4] * tt12 + c_Filter_Directions_Z[4] * tt13);
	h2Temp += c * pd * (c_Filter_Directions_X[4] * tt12 + c_Filter_Directions_Y[4] * tt22 + c_Filter_Directions_Z[4] * tt23);
	h3Temp += c * pd * (c_Filter_Directions_X[4] * tt13 + c_Filter_Directions_Y[4] * tt23 + c_Filter_Directions_Z[4] * tt33);
	
	// Sixth quadrature filter
	c = Certainties[idx + 5*offset];
	a11Temp += c * tt11;
	a12Temp += c * tt12;
	a13Temp += c * tt13;
	a22Temp += c * tt22;
	a23Temp += c * tt23;
	a33Temp += c * tt33;
			
	pd = Phase_Differences[idx + 5*offset];
	h1Temp += c * pd * (c_Filter_Directions_X[5] * tt11 + c_Filter_Directions_Y[5] * tt12 + c_Filter_Directions_Z[5] * tt13);
	h2Temp += c * pd * (c_Filter_Directions_X[5] * tt12 + c_Filter_Directions_Y[5] * tt22 + c_Filter_Directions_Z[5] * tt23);
	h3Temp += c * pd * (c_Filter_Directions_X[5] * tt13 + c_Filter_Directions_Y[5] * tt23 + c_Filter_Directions_Z[5] * tt33);
	
	a11[idx] = a11Temp;
	a12[idx] = a12Temp;
	a13[idx] = a13Temp;
	a22[idx] = a22Temp;
	a23[idx] = a23Temp;
	a33[idx] = a33Temp;
	h1[idx] = h1Temp;
	h2[idx] = h2Temp;
	h3[idx] = h3Temp;	
}
*/



// Estimate Dk Ck and T
/*
__kernel void CalculatePhaseDifferencesCertaintiesAndTensorComponents(__global float* Phase_Differences, 
	                                                                  __global float* Certainties, 
																	  __global float* t11, 
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
																	  __private int DATA_D,
																	  __private int filter) 
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if ((x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D) )
		return;

	int idx = x + y * DATA_W + z * DATA_W * DATA_H;

	int offset = filter * DATA_W * DATA_H * DATA_D;

	float2 q1_ = q1[idx];
	float2 q2_ = q2[idx];

	// q1 * conj(q2)
	float qqReal = q1_.x * q2_.x + q1_.y * q2_.y;
	float qqImag = -q1_.x * q2_.y + q1_.y * q2_.x;
	float phaseDifference = atan2(qqImag,qqReal);
	Phase_Differences[idx + offset] = phaseDifference;
	float Aqq = sqrt(qqReal * qqReal + qqImag * qqImag);
	Certainties[idx + offset] = sqrt(Aqq) * cos(phaseDifference/2.0f) * cos(phaseDifference/2.0f);
		
	// Estimate structure tensor for the deformed volume
	float magnitude = sqrt(q2_.x * q2_.x + q2_.y * q2_.y);

	t11[idx] += magnitude * m11;
	t12[idx] += magnitude * m12;
	t13[idx] += magnitude * m13;
	t22[idx] += magnitude * m22;
	t23[idx] += magnitude * m23;
	t33[idx] += magnitude * m33;	
}
*/



/*
__kernel void Nonseparable3DConvolutionComplexThreeQuadratureFilters(__global float2 *Filter_Response_1,
	                                                                 __global float2 *Filter_Response_2,
																	 __global float2 *Filter_Response_3,
																	 __global const float* Volume, 
																	 __constant float2* c_Quadrature_Filter_1, 
																	 __constant float2* c_Quadrature_Filter_2, 
																	 __constant float2* c_Quadrature_Filter_3, 
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
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_(l_Image,tIdx.y+HALO,tIdx.x+HALO,c_Quadrature_Filter_1,c_Quadrature_Filter_2,c_Quadrature_Filter_3);
        Filter_Response_1_Real[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += temp.a;
	    Filter_Response_1_Imag[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += temp.b;
	    Filter_Response_2_Real[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += temp.c;
	    Filter_Response_2_Imag[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += temp.d;
	    Filter_Response_3_Real[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += temp.e;
	    Filter_Response_3_Imag[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += temp.f;
    }

    if ( ((x + 32) < DATA_W) && (y < DATA_H) )
    {
	    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_(l_Image,tIdx.y+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1,c_Quadrature_Filter_2,c_Quadrature_Filter_3);
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
		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_(l_Image,tIdx.y+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1,c_Quadrature_Filter_2,c_Quadrature_Filter_3);
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
 		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_(l_Image,tIdx.y+32+HALO,tIdx.x+HALO,c_Quadrature_Filter_1,c_Quadrature_Filter_2,c_Quadrature_Filter_3);
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
		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_(l_Image,tIdx.y+32+HALO,tIdx.x+32+HALO,c_Quadrature_Filter_1,c_Quadrature_Filter_2,c_Quadrature_Filter_3);
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
		    float6 temp = Conv_2D_Unrolled_7x7_ThreeFilters_(l_Image,tIdx.y+32+HALO,tIdx.x+64+HALO,c_Quadrature_Filter_1,c_Quadrature_Filter_2,c_Quadrature_Filter_3);
            Filter_Response_1_Real[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += temp.a;
		    Filter_Response_1_Imag[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += temp.b;
		    Filter_Response_2_Real[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += temp.c;
		    Filter_Response_2_Imag[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += temp.d;
		    Filter_Response_3_Real[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += temp.e;
		    Filter_Response_3_Imag[Calculate3DIndex(x+64,y+32,z,DATA_W,DATA_H)] += temp.f;
	    }
     }
}
*/



/*
		// Calculate phase differences, certainties and tensor components, first quadrature filter
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 8, sizeof(cl_mem), &d_q11);
  	    clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 9, sizeof(cl_mem), &d_q21);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 10, sizeof(float), &M11_1);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 11, sizeof(float), &M12_1);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 12, sizeof(float), &M13_1);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 13, sizeof(float), &M22_1);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 14, sizeof(float), &M23_1);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 15, sizeof(float), &M33_1);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 19, sizeof(int), &zero);
		runKernelErrorCalculatePhaseDifferencesCertaintiesAndTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		// Calculate phase differences, certainties and tensor components, second quadrature filter
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 8, sizeof(cl_mem), &d_q12);
  	    clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 9, sizeof(cl_mem), &d_q22);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 10, sizeof(float), &M11_2);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 11, sizeof(float), &M12_2);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 12, sizeof(float), &M13_2);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 13, sizeof(float), &M22_2);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 14, sizeof(float), &M23_2);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 15, sizeof(float), &M33_2);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 19, sizeof(int), &one);
		runKernelErrorCalculatePhaseDifferencesCertaintiesAndTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		// Calculate phase differences, certainties and tensor components, third quadrature filter
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 8, sizeof(cl_mem), &d_q13);
  	    clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 9, sizeof(cl_mem), &d_q23);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 10, sizeof(float), &M11_3);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 11, sizeof(float), &M12_3);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 12, sizeof(float), &M13_3);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 13, sizeof(float), &M22_3);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 14, sizeof(float), &M23_3);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 15, sizeof(float), &M33_3);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 19, sizeof(int), &two);
		runKernelErrorCalculatePhaseDifferencesCertaintiesAndTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		// Calculate phase differences, certainties and tensor components, fourth quadrature filter
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 8, sizeof(cl_mem), &d_q14);
  	    clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 9, sizeof(cl_mem), &d_q24);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 10, sizeof(float), &M11_4);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 11, sizeof(float), &M12_4);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 12, sizeof(float), &M13_4);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 13, sizeof(float), &M22_4);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 14, sizeof(float), &M23_4);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 15, sizeof(float), &M33_4);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 19, sizeof(int), &three);
		runKernelErrorCalculatePhaseDifferencesCertaintiesAndTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		// Calculate phase differences, certainties and tensor components, fifth quadrature filter
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 8, sizeof(cl_mem), &d_q15);
  	    clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 9, sizeof(cl_mem), &d_q25);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 10, sizeof(float), &M11_5);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 11, sizeof(float), &M12_5);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 12, sizeof(float), &M13_5);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 13, sizeof(float), &M22_5);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 14, sizeof(float), &M23_5);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 15, sizeof(float), &M33_5);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 19, sizeof(int), &four);
		runKernelErrorCalculatePhaseDifferencesCertaintiesAndTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		// Calculate phase differences, certainties and tensor components, sixth quadrature filter
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 8, sizeof(cl_mem), &d_q16);
  	    clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 9, sizeof(cl_mem), &d_q26);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 10, sizeof(float), &M11_6);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 11, sizeof(float), &M12_6);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 12, sizeof(float), &M13_6);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 13, sizeof(float), &M22_6);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 14, sizeof(float), &M23_6);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 15, sizeof(float), &M33_6);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 19, sizeof(int), &five);
		runKernelErrorCalculatePhaseDifferencesCertaintiesAndTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);
		*/



/*
clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 0, sizeof(cl_mem), &d_Temp_Displacement_Field_X);
clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 1, sizeof(cl_mem), &d_Temp_Displacement_Field_Y);
clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 2, sizeof(cl_mem), &d_Temp_Displacement_Field_Z);
clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 3, sizeof(cl_mem), &d_Update_Certainty);
clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 4, sizeof(cl_mem), &d_a11);
clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 5, sizeof(cl_mem), &d_a12);
clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 6, sizeof(cl_mem), &d_a13);
clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 7, sizeof(cl_mem), &d_a22);
clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 8, sizeof(cl_mem), &d_a23);
clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 9, sizeof(cl_mem), &d_a33);
clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 10, sizeof(cl_mem), &d_h1);
clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 11, sizeof(cl_mem), &d_h2);
clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 12, sizeof(cl_mem), &d_h3);
clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 13, sizeof(int), &DATA_W);
clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 14, sizeof(int), &DATA_H);
clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 15, sizeof(int), &DATA_D);
*/



/*
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 0, sizeof(cl_mem), &d_a11);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 1, sizeof(cl_mem), &d_a12);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 2, sizeof(cl_mem), &d_a13);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 3, sizeof(cl_mem), &d_a22);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 4, sizeof(cl_mem), &d_a23);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 5, sizeof(cl_mem), &d_a33);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 6, sizeof(cl_mem), &d_h1);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 7, sizeof(cl_mem), &d_h2);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 8, sizeof(cl_mem), &d_h3);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 9, sizeof(cl_mem), &d_Phase_Differences);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 10, sizeof(cl_mem), &d_Phase_Certainties);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 11, sizeof(cl_mem), &d_t11);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 12, sizeof(cl_mem), &d_t12);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 13, sizeof(cl_mem), &d_t13);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 14, sizeof(cl_mem), &d_t22);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 15, sizeof(cl_mem), &d_t23);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 16, sizeof(cl_mem), &d_t33);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 17, sizeof(cl_mem), &c_Filter_Directions_X);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 18, sizeof(cl_mem), &c_Filter_Directions_Y);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 19, sizeof(cl_mem), &c_Filter_Directions_Z);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 20, sizeof(int), &DATA_W);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 21, sizeof(int), &DATA_H);
clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 22, sizeof(int), &DATA_D);
*/




/*
clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 0, sizeof(cl_mem), &d_Phase_Differences);
clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 1, sizeof(cl_mem), &d_Phase_Certainties);
clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 2, sizeof(cl_mem), &d_t11);
clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 3, sizeof(cl_mem), &d_t12);
clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 4, sizeof(cl_mem), &d_t13);
clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 5, sizeof(cl_mem), &d_t22);
clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 6, sizeof(cl_mem), &d_t23);
clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 7, sizeof(cl_mem), &d_t33);
clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 16, sizeof(int), &DATA_W);
clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 17, sizeof(int), &DATA_H);
clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 18, sizeof(int), &DATA_D);
*/


// Free all the allocated memory

	for (int i = 0; i < NUMBER_OF_HOST_POINTERS; i++)
	{
		void* pointer = host_pointers[i];
		if (pointer != NULL)
		{
			free(pointer);
		}
	}

	for (int i = 0; i < NUMBER_OF_HOST_POINTERS; i++)
	{
		void* pointer = host_pointers_static[i];
		if (pointer != NULL)
		{
			free(pointer);
		}
	}

	for (int i = 0; i < NUMBER_OF_HOST_POINTERS; i++)
	{
		void* pointer = host_pointers_permutation[i];
		if (pointer != NULL)
		{
			free(pointer);
		}
	}

	for (int i = 0; i < NUMBER_OF_DEVICE_POINTERS; i++)
	{
		float* pointer = device_pointers[i];
		if (pointer != NULL)
		{
			//clReleaseMemObject();
		}
	}

	for (int i = 0; i < NUMBER_OF_DEVICE_POINTERS; i++)
	{
		float* pointer = device_pointers_permutation[i];
		if (pointer != NULL)
		{
			//clReleaseMemObject();
		}
	}


	void BROCCOLI_LIB::ResetAllPointers()
	{
		for (int i = 0; i < NUMBER_OF_HOST_POINTERS; i++)
		{
			host_pointers[i] = NULL;
		}

		for (int i = 0; i < NUMBER_OF_HOST_POINTERS; i++)
		{
			host_pointers_static[i] = NULL;
		}

		for (int i = 0; i < NUMBER_OF_HOST_POINTERS; i++)
		{
			host_pointers_permutation[i] = NULL;
		}

		for (int i = 0; i < NUMBER_OF_DEVICE_POINTERS; i++)
		{
			device_pointers[i] = NULL;
		}

		for (int i = 0; i < NUMBER_OF_DEVICE_POINTERS; i++)
		{
			device_pointers_permutation[i] = NULL;
		}
	}



	void BROCCOLI_LIB::AllocateMemoryForFilters()
	{
		/*
		h_Quadrature_Filter_1_Real = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL);
		h_Quadrature_Filter_1_Imag = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL);
		h_Quadrature_Filter_2_Real = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL);
		h_Quadrature_Filter_2_Imag = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL);
		h_Quadrature_Filter_3_Real = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL);
		h_Quadrature_Filter_3_Imag = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL);
		//h_Quadrature_Filter_1 = (Complex*)malloc(DATA_SIZE_QUADRATURE_FILTER_COMPLEX);
		//h_Quadrature_Filter_2 = (Complex*)malloc(DATA_SIZE_QUADRATURE_FILTER_COMPLEX);
		//h_Quadrature_Filter_3 = (Complex*)malloc(DATA_SIZE_QUADRATURE_FILTER_COMPLEX);

		//h_GLM_Filter = (float*)malloc(DATA_SIZE_SMOOTHING_FILTER_GLM);

		host_pointers_static[QF1R]   = (void*)h_Quadrature_Filter_1_Real;
		host_pointers_static[QF1I]   = (void*)h_Quadrature_Filter_1_Imag;
		host_pointers_static[QF2R]   = (void*)h_Quadrature_Filter_2_Real;
		host_pointers_static[QF2I]   = (void*)h_Quadrature_Filter_2_Imag;
		host_pointers_static[QF3R]   = (void*)h_Quadrature_Filter_3_Real;
		host_pointers_static[QF3I]   = (void*)h_Quadrature_Filter_3_Imag;
		//host_pointers_static[QF1]    = (void*)h_Quadrature_Filter_1;
		//host_pointers_static[QF2]    = (void*)h_Quadrature_Filter_2;
		//host_pointers_static[QF3]	 = (void*)h_Quadrature_Filter_3;
		*/
	}
