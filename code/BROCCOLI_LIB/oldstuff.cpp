
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



	

// Read functions, public


/*
void BROCCOLI_LIB::ReadfMRIDataNIFTI()
{
	nifti_data = new nifti_image;
	// Read nifti data
	nifti_data = nifti_image_read(filename_fMRI_data_nifti.c_str(), 1);

	if (nifti_data->datatype == DT_SIGNED_SHORT)
	{
		DATA_TYPE = INT16;
	}
	else if (nifti_data->datatype == DT_SIGNED_INT)
	{
		DATA_TYPE = INT32;
	}
	else if (nifti_data->datatype == DT_FLOAT)
	{
		DATA_TYPE = FLOAT;
	}
	else if (nifti_data->datatype == DT_DOUBLE)
	{
		DATA_TYPE = DOUBLE;
	}
	else if (nifti_data->datatype == DT_UNSIGNED_CHAR)
	{
		DATA_TYPE = UINT8;
	}

	// Get number of data points in each direction
	DATA_W = nifti_data->nx;
	DATA_H = nifti_data->ny;
	DATA_D = nifti_data->nz;
	DATA_T = nifti_data->nt;

	FMRI_VOXEL_SIZE_X = nifti_data->dx;
	FMRI_VOXEL_SIZE_Y = nifti_data->dy;
	FMRI_VOXEL_SIZE_Z = nifti_data->dz;
	TR = nifti_data->dt;

	SetupParametersReadData();


	// Get data from nifti image
	if (DATA_TYPE == FLOAT)
	{
		float* data = (float*)nifti_data->data;
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = data[i];
		}
	}
	else if (DATA_TYPE == INT32)
	{
		int* data = (int*)nifti_data->data;
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)data[i];
		}
	}
	else if (DATA_TYPE == INT16)
	{
		short int* data = (short int*)nifti_data->data;
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)data[i];
		}
	}
	else if (DATA_TYPE == DOUBLE)
	{
		double* data = (double*)nifti_data->data;
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)data[i];
		}
	}
	else if (DATA_TYPE == UINT8)
	{
		unsigned char* data = (unsigned char*)nifti_data->data;
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)data[i];
		}
	}

	// Scale if necessary
	if (nifti_data->scl_slope != 0.0f)
	{
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = h_fMRI_Volumes[i] * nifti_data->scl_slope + nifti_data->scl_inter;
		}
	}

	delete nifti_data;

	// Copy fMRI volumes to global memory, as floats
	//cudaMemcpy(d_fMRI_Volumes, h_fMRI_Volumes, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyHostToDevice);

	for (int i = 0; i < DATA_T; i++)
	{
		plot_values_x[i] = (double)i * (double)TR;
	}

	SegmentBrainData();

	SetupStatisticalAnalysisBasisFunctions();
	SetupDetrendingBasisFunctions();

	CalculateSlicesfMRIData();
}
*/

/*

void BROCCOLI_LIB::ReadNIFTIHeader()
{
	// Read nifti header only
	nifti_data = nifti_image_read(filename_fMRI_data_nifti.c_str(), 0);

	// Get dimensions
	DATA_W = nifti_data->nx;
	DATA_H = nifti_data->ny;
	DATA_D = nifti_data->nz;
	DATA_T = nifti_data->nt;

	FMRI_VOXEL_SIZE_X = nifti_data->dx;
	FMRI_VOXEL_SIZE_Y = nifti_data->dy;
	FMRI_VOXEL_SIZE_Z = nifti_data->dz;
	TR = nifti_data->dt;
}

*/

// Read functions, private

void BROCCOLI_LIB::ReadRealDataInt32(int* data, std::string filename, int N)
{
	std::fstream file(filename.c_str(), std::ios::in | std::ios::binary);
	int current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			file.read( (char*) &current_value, sizeof(current_value) );
			data[i] = current_value;
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}

void BROCCOLI_LIB::ReadRealDataInt16(short int* data, std::string filename, int N)
{
	std::fstream file(filename.c_str(), std::ios::in | std::ios::binary);
	short int current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			file.read( (char*) &current_value, sizeof(current_value) );
			data[i] = current_value;
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}

void BROCCOLI_LIB::ReadRealDataUint32(unsigned int* data, std::string filename, int N)
{
	std::fstream file(filename.c_str(), std::ios::in | std::ios::binary);
	unsigned int current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			file.read( (char*) &current_value, sizeof(current_value) );
			data[i] = current_value;
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}

void BROCCOLI_LIB::ReadRealDataUint16(unsigned short int* data, std::string filename, int N)
{
	std::fstream file(filename.c_str(), std::ios::in | std::ios::binary);
	unsigned short int current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			file.read( (char*) &current_value, sizeof(current_value) );
			data[i] = current_value;
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}

void BROCCOLI_LIB::ReadRealDataFloat(float* data, std::string filename, int N)
{
	std::fstream file(filename.c_str(), std::ios::in | std::ios::binary);
	float current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			file.read( (char*) &current_value, sizeof(current_value) );
			data[i] = current_value;
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}

void BROCCOLI_LIB::ReadRealDataDouble(double* data, std::string filename, int N)
{
	std::fstream file(filename.c_str(), std::ios::in | std::ios::binary);
	double current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			file.read( (char*) &current_value, sizeof(current_value) );
			data[i] = current_value;
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}


/*
void BROCCOLI_LIB::ReadComplexData(Complex* data, std::string real_filename, std::string imag_filename, int N)
{
	std::fstream real_file(real_filename, std::ios::in | std::ios::binary);
	std::fstream imag_file(imag_filename, std::ios::in | std::ios::binary);
	float current_value;

	if (real_file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			real_file.read( (char*) &current_value, sizeof(current_value) );
			data[i] = current_value;
		}
	}
	else
	{
		std::cout << "Could not find file " << real_filename << std::endl;
	}

	if (imag_file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			imag_file.read( (char*) &current_value, sizeof(current_value) );
			data[i] = current_value;
		}
	}
	else
	{
		std::cout << "Could not find file " << imag_filename << std::endl;
	}

	real_file.close();
	imag_file.close();
}
*/

void BROCCOLI_LIB::ReadImageRegistrationFilters()
{
	// Read the quadrature filters from file
	//ReadComplexData(h_Quadrature_Filter_1, filename_real_quadrature_filter_1, filename_imag_quadrature_filter_1, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE);
	//ReadComplexData(h_Quadrature_Filter_2, filename_real_quadrature_filter_2, filename_imag_quadrature_filter_2, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE);
	//ReadComplexData(h_Quadrature_Filter_3, filename_real_quadrature_filter_3, filename_imag_quadrature_filter_3, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE);
}

/*
void BROCCOLI_LIB::SetupParametersReadData()
{
	// Reset all pointers
	
	for (int i = 0; i < NUMBER_OF_HOST_POINTERS; i++)
	{
		void* pointer = host_pointers[i];
		if (pointer != NULL)
		{
			free(pointer);
			host_pointers[i] = NULL;
		}
	}

	for (int i = 0; i < NUMBER_OF_DEVICE_POINTERS; i++)
	{
		float* pointer = device_pointers[i];
		if (pointer != NULL)
		{
			//cudaFree(pointer);
			device_pointers[i] = NULL;
		}
	}

	MOTION_CORRECTED = false;

	X_SLICE_LOCATION_fMRI_DATA = EPI_DATA_W / 2;
	Y_SLICE_LOCATION_fMRI_DATA = EPI_DATA_H / 2;
	Z_SLICE_LOCATION_fMRI_DATA = EPI_DATA_D / 2;
	TIMEPOINT_fMRI_DATA = 0;

	h_fMRI_Volumes = (float*)malloc(DATA_SIZE_FMRI_VOLUMES);
	h_Motion_Corrected_fMRI_Volumes = (float*)malloc(DATA_SIZE_FMRI_VOLUMES);
	h_Smoothed_fMRI_Volumes = (float*)malloc(DATA_SIZE_FMRI_VOLUMES);
	h_Detrended_fMRI_Volumes = (float*)malloc(DATA_SIZE_FMRI_VOLUMES);

	h_X_Detrend = (float*)malloc(DATA_SIZE_DETRENDING);
	h_xtxxt_Detrend = (float*)malloc(DATA_SIZE_DETRENDING);

	h_X_GLM = (float*)malloc(DATA_SIZE_TEMPORAL_BASIS_FUNCTIONS);
	h_xtxxt_GLM = (float*)malloc(DATA_SIZE_TEMPORAL_BASIS_FUNCTIONS);
	h_Contrasts = (float*)malloc(sizeof(float) * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS * NUMBER_OF_CONTRASTS);
	
	//h_Activity_Volume = (float*)malloc(DATA_SIZE_fMRI_VOLUME);
	
	host_pointers[fMRI_VOLUMES] = (void*)h_fMRI_Volumes;
	host_pointers[MOTION_CORRECTED_VOLUMES] = (void*)h_Motion_Corrected_fMRI_Volumes;
	host_pointers[SMOOTHED1] = (void*)h_Smoothed_fMRI_Volumes;
	host_pointers[DETRENDED1] = (void*)h_Detrended_fMRI_Volumes;
	host_pointers[XDETREND1] = (void*)h_X_Detrend;
	host_pointers[XDETREND2] = (void*)h_xtxxt_Detrend;
	host_pointers[XGLM1] = (void*)h_X_GLM;
	host_pointers[XGLM2] = (void*)h_xtxxt_GLM;
	host_pointers[CONTRAST_VECTOR] = (void*)h_Contrasts;
	//host_pointers[ACTIVITY_VOLUME] = (void*)h_Activity_Volume;

	x_slice_fMRI_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_H * DATA_D);
	y_slice_fMRI_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_W * DATA_D);
	z_slice_fMRI_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_W * DATA_H);

	x_slice_preprocessed_fMRI_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_H * DATA_D);
	y_slice_preprocessed_fMRI_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_W * DATA_D);
	z_slice_preprocessed_fMRI_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_W * DATA_H);

	x_slice_activity_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_H * DATA_D);
	y_slice_activity_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_W * DATA_D);
	z_slice_activity_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_W * DATA_H);

	motion_parameters_x = (double*)malloc(sizeof(double) * DATA_T);
	motion_parameters_y = (double*)malloc(sizeof(double) * DATA_T);
	motion_parameters_z = (double*)malloc(sizeof(double) * DATA_T);
	plot_values_x = (double*)malloc(sizeof(double) * DATA_T);
	motion_corrected_curve = (double*)malloc(sizeof(double) * DATA_T);
	smoothed_curve = (double*)malloc(sizeof(double) * DATA_T);
	detrended_curve = (double*)malloc(sizeof(double) * DATA_T);

	host_pointers[X_SLICE_fMRI] = (void*)x_slice_fMRI_data;
	host_pointers[Y_SLICE_fMRI] = (void*)y_slice_fMRI_data;
	host_pointers[Z_SLICE_fMRI] = (void*)z_slice_fMRI_data;
	host_pointers[X_SLICE_PREPROCESSED_fMRI] = (void*)x_slice_preprocessed_fMRI_data;
	host_pointers[Y_SLICE_PREPROCESSED_fMRI] = (void*)y_slice_preprocessed_fMRI_data;
	host_pointers[Z_SLICE_PREPROCESSED_fMRI] = (void*)z_slice_preprocessed_fMRI_data;
	host_pointers[X_SLICE_ACTIVITY] = (void*)x_slice_activity_data;
	host_pointers[Y_SLICE_ACTIVITY] = (void*)y_slice_activity_data;
	host_pointers[Z_SLICE_ACTIVITY] = (void*)z_slice_activity_data;
	host_pointers[MOTION_PARAMETERS_X] = (void*)motion_parameters_x;
	host_pointers[MOTION_PARAMETERS_Y] = (void*)motion_parameters_y;
	host_pointers[MOTION_PARAMETERS_Z] = (void*)motion_parameters_z;
	host_pointers[PLOT_VALUES_X] = (void*)plot_values_x;
	host_pointers[MOTION_CORRECTED_CURVE] = (void*)motion_corrected_curve;
	host_pointers[SMOOTHED_CURVE] = (void*)smoothed_curve;
	host_pointers[DETRENDED_CURVE] = (void*)detrended_curve;


	//device_pointers[fMRI_VOLUMES] = d_fMRI_Volumes;
	//device_pointers[BRAIN_VOXELS] = d_Brain_Voxels;
	//device_pointers[SMOOTHED_CERTAINTY] = d_Smoothed_Certainty;
	//device_pointers[ACTIVITY_VOLUME] = d_Activity_Volume;
}
*/

// Write functions, public

void BROCCOLI_LIB::WritefMRIDataNIFTI()
{


}

// Write functions, private

void BROCCOLI_LIB::WriteRealDataUint16(unsigned short int* data, std::string filename, int N)
{
	std::fstream file(filename.c_str(), std::ios::out | std::ios::binary);
	unsigned short int current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			current_value = data[i];
			file.write( (const char*) &current_value, sizeof(current_value) );
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}

void BROCCOLI_LIB::WriteRealDataFloat(float* data, std::string filename, int N)
{
	std::fstream file(filename.c_str(), std::ios::out | std::ios::binary);
	float current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			current_value = data[i];
			file.write( (const char*) &current_value, sizeof(current_value) );
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}

void BROCCOLI_LIB::WriteRealDataDouble(double* data, std::string filename, int N)
{
	std::fstream file(filename.c_str(), std::ios::out | std::ios::binary);
	double current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			current_value = data[i];
			file.write( (const char*) &current_value, sizeof(current_value) );
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}

/*
void BROCCOLI_LIB::WriteComplexData(Complex* data, std::string real_filename, std::string imag_filename, int N)
{
	std::fstream real_file(real_filename, std::ios::out | std::ios::binary);
	std::fstream imag_file(imag_filename, std::ios::out | std::ios::binary);

	float current_value;

	if (real_file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			current_value = data[i];
			real_file.write( (const char*) &current_value, sizeof(current_value) );
		}
	}
	else
	{
		std::cout << "Could not find file " << real_filename << std::endl;
	}
	real_file.close();

	if (imag_file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			current_value = data[i];
			imag_file.write( (const char*) &current_value, sizeof(current_value) );
		}
	}
	else
	{
		std::cout << "Could not find file " << imag_filename << std::endl;
	}
	imag_file.close();
}
*/




// Help functions

/*
void BROCCOLI_LIB::CalculateSlicesActivityData()
{
	//float max = CalculateMax(h_Activity_Volume, DATA_W * DATA_H * DATA_D);
	//float min = CalculateMin(h_Activity_Volume, DATA_W * DATA_H * DATA_D);
	float adder = -min;
	float multiplier = 255.0f / (max + adder);

	for (int x = 0; x < DATA_W; x++)
	{
		for (int y = 0; y < DATA_H; y++)
		{
			if (THRESHOLD_ACTIVITY_MAP)
			{
				if (h_Activity_Volume[x + y * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H] >= ACTIVITY_THRESHOLD)
				{
					z_slice_activity_data[x + y * DATA_W] = (unsigned char)((h_Activity_Volume[x + y * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H] + adder) * multiplier);
				}
				else
				{
					z_slice_activity_data[x + y * DATA_W] = 0;
				}
			}
			else
			{
				z_slice_activity_data[x + y * DATA_W] = (unsigned char)((h_Activity_Volume[x + y * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H] + adder) * multiplier);
			}
		}
	}

	for (int x = 0; x < DATA_W; x++)
	{
		for (int z = 0; z < DATA_D; z++)
		{
			int inv_z = DATA_D - 1 - z;
			if (THRESHOLD_ACTIVITY_MAP)
			{
				if (h_Activity_Volume[x + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + z * DATA_W * DATA_H] >= ACTIVITY_THRESHOLD)
				{
					y_slice_activity_data[x + inv_z * DATA_W] = (unsigned char)((h_Activity_Volume[x + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + z * DATA_W * DATA_H] + adder) * multiplier);
				}
				else
				{
					y_slice_activity_data[x + inv_z * DATA_W] = 0;
				}
			}
			else
			{
				y_slice_activity_data[x + inv_z * DATA_W] = (unsigned char)((h_Activity_Volume[x + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + z * DATA_W * DATA_H] + adder) * multiplier);
			}
		}
	}

	for (int y = 0; y < DATA_H; y++)
	{
		for (int z = 0; z < DATA_D; z++)
		{
			int inv_z = DATA_D - 1 - z;
			if (THRESHOLD_ACTIVITY_MAP)
			{
				if (h_Activity_Volume[X_SLICE_LOCATION_fMRI_DATA + y * DATA_W + z * DATA_W * DATA_H] >= ACTIVITY_THRESHOLD)
				{
					x_slice_activity_data[y + inv_z * DATA_H] = (unsigned char)((h_Activity_Volume[X_SLICE_LOCATION_fMRI_DATA + y * DATA_W + z * DATA_W * DATA_H] + adder) * multiplier);
				}
				else
				{
					x_slice_activity_data[y + inv_z * DATA_H] = 0;
				}
			}
			else
			{
				x_slice_activity_data[y + inv_z * DATA_H] = (unsigned char)((h_Activity_Volume[X_SLICE_LOCATION_fMRI_DATA + y * DATA_W + z * DATA_W * DATA_H] + adder) * multiplier);
			}
		}
	}
}
*/

void BROCCOLI_LIB::CalculateSlicesfMRIData()
{
	float max = CalculateMax(h_fMRI_Volumes, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T);
	float min = CalculateMin(h_fMRI_Volumes, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T);
	float adder = -min;
	float multiplier = 255.0f / (max + adder);

	for (int x = 0; x < EPI_DATA_W; x++)
	{
		for (int y = 0; y < EPI_DATA_H; y++)
		{
			z_slice_fMRI_data[x + y * EPI_DATA_W] = (unsigned char)((h_fMRI_Volumes[x + y * EPI_DATA_W + Z_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W * EPI_DATA_H + TIMEPOINT_fMRI_DATA * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] + adder) * multiplier);
		}
	}

	for (int x = 0; x < EPI_DATA_W; x++)
	{
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			int inv_z = EPI_DATA_D - 1 - z;
			y_slice_fMRI_data[x + inv_z * EPI_DATA_W] = (unsigned char)((h_fMRI_Volumes[x + Y_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + TIMEPOINT_fMRI_DATA * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] + adder) * multiplier);
		}
	}

	for (int y = 0; y < EPI_DATA_H; y++)
	{
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			int inv_z = EPI_DATA_D - 1 - z;
			x_slice_fMRI_data[y + inv_z * EPI_DATA_H] = (unsigned char)((h_fMRI_Volumes[X_SLICE_LOCATION_fMRI_DATA + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + TIMEPOINT_fMRI_DATA * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] + adder) * multiplier);
		}
	}
}

void BROCCOLI_LIB::CalculateSlicesPreprocessedfMRIData()
{
	float* pointer = NULL;

	/*
	if (PREPROCESSED == MOTION_CORRECTION)
	{
		pointer = h_Motion_Corrected_fMRI_Volumes;
	}
	else if (PREPROCESSED == SMOOTHING)
	{
		pointer = h_Smoothed_fMRI_Volumes;
	}
	else if (PREPROCESSED == DETRENDING)
	{
		pointer = h_Detrended_fMRI_Volumes;
	}
	*/

	float max = CalculateMax(pointer, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T);
	float min = CalculateMin(pointer, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T);
	float adder = -min;
	float multiplier = 255.0f / (max + adder);


	for (int x = 0; x < EPI_DATA_W; x++)
	{
		for (int y = 0; y < EPI_DATA_H; y++)
		{
			z_slice_preprocessed_fMRI_data[x + y * EPI_DATA_W] = (unsigned char)((pointer[x + y * EPI_DATA_W + Z_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W * EPI_DATA_H + TIMEPOINT_fMRI_DATA * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] + adder) * multiplier);
		}
	}

	for (int x = 0; x < EPI_DATA_W; x++)
	{
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			int inv_z = EPI_DATA_D - 1 - z;
			y_slice_preprocessed_fMRI_data[x + inv_z * EPI_DATA_W] = (unsigned char)((pointer[x + Y_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + TIMEPOINT_fMRI_DATA * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] + adder) * multiplier);
		}
	}

	for (int y = 0; y < EPI_DATA_H; y++)
	{
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			int inv_z = EPI_DATA_D - 1 - z;
			x_slice_preprocessed_fMRI_data[y + inv_z * EPI_DATA_H] = (unsigned char)((pointer[X_SLICE_LOCATION_fMRI_DATA + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + TIMEPOINT_fMRI_DATA * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] + adder) * multiplier);
		}
	}
}

/*
void BROCCOLI_LIB::Convert4FloatToFloat4(float4* floats, float* float_1, float* float_2, float* float_3, float* float_4, int N)
{
	for (int i = 0; i < N; i++)
	{
		floats[i] = float_1[i];
		floats[i] = float_2[i];
		floats[i].z = float_3[i];
		floats[i].w = float_4[i];
	}
}
*/

/*
void BROCCOLI_LIB::Convert2FloatToFloat2(float2* floats, float* float_1, float* float_2, int N)
{
	for (int i = 0; i < N; i++)
	{
		floats[i] = float_1[i];
		floats[i] = float_2[i];
	}
}
*/

/*
void BROCCOLI_LIB::ConvertRealToComplex(Complex* complex_data, float* real_data, int N)
{
	for (int i = 0; i < N; i++)
	{
		complex_data[i] = real_data[i];
		complex_data[i] = 0.0f;
	}
}
*/

/*
void BROCCOLI_LIB::ExtractRealData(float* real_data, Complex* complex_data, int N)
{
	for (int i = 0; i < N; i++)
	{
		real_data[i] = complex_data[i];
	}
}
*/







void BROCCOLI_LIB::CalculateClusterSizes(int* Cluster_Sizes, int* Cluster_Indices, int NUMBER_OF_CLUSTERS, float* Mask, int DATA_W, int DATA_H, int DATA_D)
{
	for (int c = 0; c < NUMBER_OF_CLUSTERS; c++)
	{
		Cluster_Sizes[c] = 0;
	}

	// Loop over clusters
	for (int z = 0; z < DATA_D; z++)
	{
		for (int y = 0; y < DATA_H; y++)
		{
			for (int x = 0; x < DATA_W; x++)
			{
				if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] == 1.0f )
				{
					int cluster_index = Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];
					if ( cluster_index != 0 )
					{
						if (cluster_index < NUMBER_OF_CLUSTERS)
						{
							// Increment number of voxels for current cluster
							Cluster_Sizes[cluster_index]++;
						}
					}
				}
			}
		}
	}
}




// Forward relabelling of cluster indices
int ForwardScan(int* Cluster_Indices, float* Thresholded, int DATA_W, int DATA_H, int DATA_D)
{
	int changed = 0;

	// Loop through voxels
	for (int z = 0; z < DATA_D; z++)
	{
		for (int y = 0; y < DATA_H; y++)
		{
			for (int x = 0; x < DATA_W; x++)
			{
				// Only look at voxels that survived the threshold
				if ( Thresholded[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
				{
					// Find the local maximal cluster index
					int cluster_index = 0;
					for (int zz = -1; zz < 2; zz++)
					{
						for (int yy = -1; yy < 2; yy++)
						{
							for (int xx = -1; xx < 2; xx++)
							{
								// Do not include center (current) voxel
								int sum = abs(xx) + abs(yy) + abs(zz);
								if (sum != 0)
								{
									// Do not read outside volume
									if ( ((x + xx) >= 0) && ((y + yy) >= 0) && ((z + zz) >= 0) && ((x + xx) < DATA_W) && ((y + yy) < DATA_H) && ((z + zz) < DATA_D) )
									{
										// Only consider voxels that survived threshold (to avoid get a 0 cluster index)
										if ( Thresholded[xx + x + (yy + y) * DATA_W + (zz + z) * DATA_W * DATA_H] == 1.0f )
										{
											cluster_index = mymax(Cluster_Indices[xx + x + (yy + y) * DATA_W + (zz + z) * DATA_W * DATA_H],cluster_index);
										}
									}
								}
							}
						}
					}

					// Check if the local maxima exists
					if (cluster_index != 0)
					{
						// Check if local maxima is different compared to current index, then increase number of changed labels
						if (Cluster_Indices[x + y * DATA_W + z * DATA_W * DATA_H] != cluster_index)
						{
							changed++;
							Cluster_Indices[x + y * DATA_W + z * DATA_W * DATA_H] = cluster_index;
						}
					}
				}
			}
		}
	}
	return changed;
}

// Backward relabelling of cluster indices
int BackwardScan(int* Cluster_Indices, float* Thresholded, int DATA_W, int DATA_H, int DATA_D)
{
	int changed = 0;

	// Loop through voxels
	for (int z = DATA_D-1; z >= 0; z--)
	{
		for (int y = DATA_H-1; y >= 0; y--)
		{
			for (int x = DATA_W-1; x >= 0; x--)
			{
				// Only look at voxels that survived the threshold
				if ( Thresholded[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
				{
					// Find the local maximal cluster index
					int cluster_index = 0;
					for (int zz = -1; zz < 2; zz++)
					{
						for (int yy = -1; yy < 2; yy++)
						{
							for (int xx = -1; xx < 2; xx++)
							{
								// Do not include center (current) voxel
								int sum = abs(xx) + abs(yy) + abs(zz);
								if (sum != 0)
								{
									// Do not read outside volume
									if ( ((x + xx) >= 0) && ((y + yy) >= 0) && ((z + zz) >= 0) && ((x + xx) < DATA_W) && ((y + yy) < DATA_H) && ((z + zz) < DATA_D) )
									{
										// Only consider voxels that survived threshold (to avoid get a 0 cluster index)
										if ( Thresholded[xx + x + (yy + y) * DATA_W + (zz + z) * DATA_W * DATA_H] == 1.0f )
										{
											cluster_index = mymax(Cluster_Indices[xx + x + (yy + y) * DATA_W + (zz + z) * DATA_W * DATA_H],cluster_index);
										}
									}
								}
							}
						}
					}

					// Check if the local maxima exists
					if (cluster_index != 0)
					{
						// Check if local maxima is different compared to current index, then increase number of changed labels
						if (Cluster_Indices[x + y * DATA_W + z * DATA_W * DATA_H] != cluster_index)
						{
							changed++;
							Cluster_Indices[x + y * DATA_W + z * DATA_W * DATA_H] = cluster_index;
						}
					}
				}
			}
		}
	}
	return changed;
}





void BROCCOLI_LIB::ClusterizeOld(int* Cluster_Indices, int& NUMBER_OF_CLUSTERS, float* Data, float Threshold, float* Mask, int DATA_W, int DATA_H, int DATA_D)
{
	float* Thresholded = (float*)malloc(DATA_W * DATA_H * DATA_W * sizeof(float));

	int current_cluster = 0;

	// Set all indices to 0
	for (int z = 0; z < DATA_D; z++)
	{
		for (int y = 0; y < DATA_H; y++)
		{
			for (int x = 0; x < DATA_W; x++)
			{
				Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0;

				if ( Data[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] > Threshold )
				{
					Thresholded[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 1.0f;
				}
			}
		}
	}

	// Make an initial labelling
	for (int z = 0; z < DATA_D; z++)
	{
		for (int y = 0; y < DATA_H; y++)
		{
			for (int x = 0; x < DATA_W; x++)
			{
				if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] == 1.0f )
				{
					if ( Thresholded[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] == 1.0f )
					{
						// Check if neighbour already has a cluster index
						int cluster_index = 0;

						for (int zz = -1; zz < 2; zz++)
						{
							for (int yy = -1; yy < 2; yy++)
							{
								for (int xx = -1; xx < 2; xx++)
								{
									if ( ((x + xx) >= 0) && ((y + yy) >= 0) && ((z + zz) >= 0) && ((x + xx) < DATA_W) && ((y + yy) < DATA_H) && ((z + zz) < DATA_D) )
									{
										cluster_index = mymax(Cluster_Indices[Calculate3DIndex(x+xx,y+yy,z+zz,DATA_W,DATA_H)],cluster_index);
									}
								}
							}
						}

						// Use existing cluster index
						if (cluster_index != 0)
						{
							Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = cluster_index;
						}
						// Use new cluster index
						else
						{
							current_cluster++;
							Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = current_cluster;
						}

					}
				}
			}
		}
	}

	//NUMBER_OF_CLUSTERS = (int)current_cluster; // Note that some clusters have 0 voxels after relabelling

	// Perform backward and forward relabellings of cluster indices until no changes are made
	int changed_labels = 1;
	while (changed_labels > 0)
	{
		changed_labels = BackwardScan(Cluster_Indices, Thresholded, DATA_W, DATA_H, DATA_D);
		changed_labels += ForwardScan(Cluster_Indices, Thresholded, DATA_W, DATA_H, DATA_D);
	}

	//NUMBER_OF_CLUSTERS = (int)CalculateMax(Cluster_Indices, DATA_W * DATA_H * DATA_D);
	NUMBER_OF_CLUSTERS = current_cluster;

	free(Thresholded);
}



//h_Registration_Parameters_Align_Two_Volumes_Several_Scales = h_Registration_Parameters_Align_Two_Volumes_Several_Scales*2 + h_Registration_Parameters_Temp*2




   		//clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps, 0, NULL, NULL);
   		//ClusterizeOld(h_Cluster_Indices, NUMBER_OF_CLUSTERS, h_Statistical_Maps, 5.0f, h_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
   		//h_Cluster_Sizes = (int*)malloc(NUMBER_OF_CLUSTERS * sizeof(int));
   		//CalculateClusterSizes(h_Cluster_Sizes, h_Cluster_Indices, NUMBER_OF_CLUSTERS, h_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
   		//h_Permutation_Distribution[p] = CalculateMax(h_Cluster_Sizes, NUMBER_OF_CLUSTERS);
   		//free(h_Cluster_Sizes);



#define QF1R 0
#define QF1I 1
#define QF2R 2
#define QF2I 3
#define QF3R 4
#define QF3I 5
#define QF1  6
#define QF2  7
#define QF3  8
#define CCA3D1 14
#define CCA3D2 15
#define CCA3D 16
#define fMRI_VOLUMES 17
#define XDETREND1 18
#define XDETREND2 19
#define CXX 20
#define SQRTINVCXX 21
#define XGLM1 22
#define XGLM2 23
#define CONTRAST_VECTOR 24
#define ACTIVITY_VOLUME 25
#define BRAIN_VOXELS 26
#define MOTION_CORRECTED_VOLUMES 27
#define REGISTRATION_PARAMETERS 28
#define SMOOTHED1 29
#define SMOOTHED2 30
#define SMOOTHED3 31
#define SMOOTHED4 32
#define DETRENDED1 33
#define DETRENDED2 34
#define DETRENDED3 35
#define DETRENDED4 36
#define X_SLICE_fMRI 37
#define Y_SLICE_fMRI 38
#define Z_SLICE_fMRI 39
#define X_SLICE_PREPROCESSED_fMRI 40
#define Y_SLICE_PREPROCESSED_fMRI 41
#define Z_SLICE_PREPROCESSED_fMRI 42
#define X_SLICE_ACTIVITY 43
#define Y_SLICE_ACTIVITY 44
#define Z_SLICE_ACTIVITY 45
#define MOTION_PARAMETERS_X 46
#define MOTION_PARAMETERS_Y 47
#define MOTION_PARAMETERS_Z 48
#define PLOT_VALUES_X 49
#define MOTION_CORRECTED_CURVE 50
#define SMOOTHED_CURVE 51
#define DETRENDED_CURVE 52	
#define ALPHAS1 53
#define ALPHAS2 54
#define ALPHAS3 55
#define ALPHAS4 56
#define SMOOTHED_ALPHAS1 57
#define SMOOTHED_ALPHAS2 58
#define SMOOTHED_ALPHAS3 59
#define SMOOTHED_ALPHAS4 60
#define SMOOTHED_CERTAINTY 61
#define BOLD_REGRESSED_VOLUMES 62
#define WHITENED_VOLUMES 63
#define PERMUTED_VOLUMES 64
#define MAXIMUM_TEST_VALUES 65
#define PERMUTATION_MATRIX 66

#define NUMBER_OF_HOST_POINTERS 100
#define NUMBER_OF_DEVICE_POINTERS 100



		unsigned char* GetZSlicefMRIData();
		unsigned char* GetYSlicefMRIData();
		unsigned char* GetXSlicefMRIData();
		unsigned char* GetZSlicePreprocessedfMRIData();
		unsigned char* GetYSlicePreprocessedfMRIData();
		unsigned char* GetXSlicePreprocessedfMRIData();
		unsigned char* GetZSliceActivityData();
		unsigned char* GetYSliceActivityData();
		unsigned char* GetXSliceActivityData();



				double* GetMotionParametersX();
		double* GetMotionParametersY();
		double* GetMotionParametersZ();
		double* GetPlotValuesX();

		double* GetMotionCorrectedCurve();
		double* GetSmoothedCurve();
		double* GetDetrendedCurve();



				//void ConvertRealToComplex(Complex* complex_data, float* real_data, int N);
		//void ExtractRealData(float* real_data, Complex* complex_data, int N);
		//void Convert4FloatToFloat4(float4* floats, float* float_1, float* float_2, float* float_3, float* float_4, int N);
		//void Convert2FloatToFloat2(float2* floats, float* float_1, float* float_2, int N);




		//------------------------------------------------
		// Read functions
		//------------------------------------------------
		void ReadRealDataInt32(int* data, std::string filename, int N);
		void ReadRealDataInt16(short int* data, std::string filename, int N);
		void ReadRealDataUint32(unsigned int* data, std::string filename, int N);
		void ReadRealDataUint16(unsigned short int* data, std::string filename, int N);
		void ReadRealDataFloat(float* data, std::string filename, int N);
		void ReadRealDataDouble(double* data, std::string filename, int N);
		//void ReadComplexData(Complex* data, std::string real_filename, std::string imag_filename, int N);
		void ReadImageRegistrationFilters();
		void SetupParametersReadData();




		//------------------------------------------------
		// Write functions
		//------------------------------------------------
		void WriteRealDataUint16(unsigned short int* data, std::string filename, int N);
		void WriteRealDataFloat(float* data, std::string filename, int N);
		void WriteRealDataDouble(double* data, std::string filename, int N);
		//void WriteComplexData(Complex* data, std::string real_filename, std::string imag_filename, int N);



		unsigned char* x_slice_fMRI_data;
		unsigned char* y_slice_fMRI_data;
		unsigned char* z_slice_fMRI_data;
		unsigned char* x_slice_preprocessed_fMRI_data;
		unsigned char* y_slice_preprocessed_fMRI_data;
		unsigned char* z_slice_preprocessed_fMRI_data;
		unsigned char* x_slice_activity_data;
		unsigned char* y_slice_activity_data;
		unsigned char* z_slice_activity_data;

		double* plot_values_x;




				double* smoothed_curve;


		double* detrended_curve;

		int X_SLICE_LOCATION_fMRI_DATA, Y_SLICE_LOCATION_fMRI_DATA, Z_SLICE_LOCATION_fMRI_DATA, TIMEPOINT_fMRI_DATA;


void BROCCOLI_LIB::SetDataType(int type)
{
	DATA_TYPE = type;
}

void BROCCOLI_LIB::SetFileType(int type)
{
	FILE_TYPE = type;
}

void BROCCOLI_LIB::SetfMRIDataSliceLocationX(int location)
{
	X_SLICE_LOCATION_fMRI_DATA = location;
}
			
void BROCCOLI_LIB::SetfMRIDataSliceLocationY(int location)
{
	Y_SLICE_LOCATION_fMRI_DATA = location;
}
		
void BROCCOLI_LIB::SetfMRIDataSliceLocationZ(int location)
{
	Z_SLICE_LOCATION_fMRI_DATA = location;
}

void BROCCOLI_LIB::SetfMRIDataSliceTimepoint(int timepoint)
{
	TIMEPOINT_fMRI_DATA = timepoint;
}


void BROCCOLI_LIB::SetfMRIDataFilename(std::string filename)
{
	filename_fMRI_data_nifti = filename;
}


int BROCCOLI_LIB::GetfMRIDataSliceLocationX()
{
	return X_SLICE_LOCATION_fMRI_DATA;
}
			
int BROCCOLI_LIB::GetfMRIDataSliceLocationY()
{
	return Y_SLICE_LOCATION_fMRI_DATA;
}
		
int BROCCOLI_LIB::GetfMRIDataSliceLocationZ()
{
	return Z_SLICE_LOCATION_fMRI_DATA;
}



// Returns a z slice of the original fMRI data, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetZSlicefMRIData()
{
	return z_slice_fMRI_data;
}

// Returns a y slice of the original fMRI data, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetYSlicefMRIData()
{
	return y_slice_fMRI_data;
}

// Returns a x slice of the original fMRI data, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetXSlicefMRIData()
{
	return x_slice_fMRI_data;
}

// Returns a z slice of the preprocessed fMRI data, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetZSlicePreprocessedfMRIData()
{
	return z_slice_preprocessed_fMRI_data;
}

// Returns a y slice of the preprocessed fMRI data, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetYSlicePreprocessedfMRIData()
{
	return y_slice_preprocessed_fMRI_data;
}

// Returns a x slice of the preprocessed fMRI data, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetXSlicePreprocessedfMRIData()
{
	return x_slice_preprocessed_fMRI_data;
}

// Returns a z slice of the activity map, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetZSliceActivityData()
{
	return z_slice_activity_data;
}

// Returns a y slice of the activity map, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetYSliceActivityData()
{
	return y_slice_activity_data;
}

// Returns a x slice of the activity map, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetXSliceActivityData()
{
	return x_slice_activity_data;
}

// Returns estimated motion parameters in the x direction to be viewed in the GUI
double* BROCCOLI_LIB::GetMotionParametersX()
{
	return motion_parameters_x;
}

// Returns estimated motion parameters in the y direction to be viewed in the GUI
double* BROCCOLI_LIB::GetMotionParametersY()
{
	return motion_parameters_y;
}

// Returns estimated motion parameters in the z direction to be viewed in the GUI
double* BROCCOLI_LIB::GetMotionParametersZ()
{
	return motion_parameters_z;
}

double* BROCCOLI_LIB::GetPlotValuesX()
{
	return plot_values_x;
}

// Returns the timeseries of the motion corrected data for the current mouse location in the GUI
double* BROCCOLI_LIB::GetMotionCorrectedCurve()
{
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		motion_corrected_curve[t] = (double)h_Motion_Corrected_fMRI_Volumes[X_SLICE_LOCATION_fMRI_DATA + Y_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W + Z_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
	}

	return motion_corrected_curve;
}

// Returns the timeseries of the smoothed data for the current mouse location in the GUI
double* BROCCOLI_LIB::GetSmoothedCurve()
{
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		smoothed_curve[t] = (double)h_Smoothed_fMRI_Volumes[X_SLICE_LOCATION_fMRI_DATA + Y_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W + Z_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
	}

	return smoothed_curve;
}

// Returns the timeseries of the detrended data for the current mouse location in the GUI
double* BROCCOLI_LIB::GetDetrendedCurve()
{
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		detrended_curve[t] = (double)h_Detrended_fMRI_Volumes[X_SLICE_LOCATION_fMRI_DATA + Y_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W + Z_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
	}

	return detrended_curve;
}

// Returns the filename of the current fMRI dataset
std::string BROCCOLI_LIB::GetfMRIDataFilename()
{
	return filename_fMRI_data_nifti;
}





filename_real_quadrature_filter_1 = "filters\\quadrature_filter_1_real.raw";
	filename_real_quadrature_filter_2 = "filters\\quadrature_filter_2_real.raw";
	filename_real_quadrature_filter_3 = "filters\\quadrature_filter_3_real.raw";
	filename_imag_quadrature_filter_1 = "filters\\quadrature_filter_1_imag.raw";
	filename_imag_quadrature_filter_2 = "filters\\quadrature_filter_2_imag.raw";
	filename_imag_quadrature_filter_3 = "filters\\quadrature_filter_3_imag.raw";

	filename_GLM_filter = "filters\\GLM_smoothing_filter";

	filename_fMRI_data_raw = "fMRI_data.raw";
	filename_slice_timing_corrected_fMRI_volumes_raw = "output\\slice_timing_corrected_fMRI_volumes.raw";
	filename_registration_parameters_raw = "output\\registration_parameters.raw";
	filename_motion_corrected_fMRI_volumes_raw = "output\\motion_compensated_fMRI_volumes.raw";
	filename_smoothed_fMRI_volumes_raw = "output\\smoothed_fMRI_volumes_1.raw";
	filename_detrended_fMRI_volumes_raw = "output\\detrended_fMRI_volumes_1.raw";
	filename_activity_volume_raw = "output\\activity_volume.raw";
	
	filename_fMRI_data_nifti = "fMRI_data.nii";
	filename_slice_timing_corrected_fMRI_volumes_nifti = "output\\slice_timing_corrected_fMRI_volumes.nii";
	filename_registration_parameters_nifti = "output\\registration_parameters.nii";
	filename_motion_corrected_fMRI_volumes_nifti = "output\\motion_compensated_fMRI_volumes.nii";
	filename_smoothed_fMRI_volumes_nifti = "output\\smoothed_fMRI_volumes_1.nii";
	filename_detrended_fMRI_volumes_nifti = "output\\detrended_fMRI_volumes_1.nii";
	filename_activity_volume_nifti = "output\\activity_volume.nii";
	


			//--------------------------------------------------
		// Filenames
		//--------------------------------------------------

		std::string		filename_real_quadrature_filter_1;
		std::string		filename_real_quadrature_filter_2;
		std::string		filename_real_quadrature_filter_3;
		std::string		filename_imag_quadrature_filter_1;
		std::string		filename_imag_quadrature_filter_2;
		std::string		filename_imag_quadrature_filter_3;
		std::string		filename_GLM_filter;

		std::string		filename_fMRI_data_raw;
		std::string		filename_slice_timing_corrected_fMRI_volumes_raw;
		std::string		filename_registration_parameters_raw;
		std::string		filename_motion_corrected_fMRI_volumes_raw;
		std::string		filename_smoothed_fMRI_volumes_raw;
		std::string		filename_detrended_fMRI_volumes_raw;
		std::string		filename_activity_volume_raw;

		std::string		filename_fMRI_data_nifti;
		std::string		filename_slice_timing_corrected_fMRI_volumes_nifti;
		std::string		filename_registration_parameters_nifti;
		std::string		filename_motion_corrected_fMRI_volumes_nifti;
		std::string		filename_smoothed_fMRI_volumes_nifti;
		std::string		filename_detrended_fMRI_volumes_nifti;
		std::string		filename_activity_volume_nifti;




		/*
__kernel void CalculateStatisticalMapsGLMTTestPermutation(__global float* Statistical_Maps,
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
	{
		//for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		//{
		//	Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = 0.0f;
		//}

		return;
	}

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
		//float value = Volumes[Calculate4DIndex(x,y,z,c_Permutation_Vector[v],DATA_W,DATA_H,DATA_D)];
		float value = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		// Loop over regressors
		//for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		//{
		//	beta[r] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * r + c_Permutation_Vector[v]];
		//}
		CalculateBetaWeights(beta, value, c_xtxxt_GLM, v, c_Permutation_Vector, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);
	}

	// Calculate the mean of the error eps
	meaneps = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		//eps = Volumes[Calculate4DIndex(x,y,z,c_Permutation_Vector[v],DATA_W,DATA_H,DATA_D)];
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		//for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		//{
		//	eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + c_Permutation_Vector[v]] * beta[r];
		//}
		eps = CalculateEps(eps, beta, c_X_GLM, v, c_Permutation_Vector, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);
		//meaneps += eps;
		meaneps += eps / ((float)NUMBER_OF_VOLUMES);
	}
	//meaneps /= ((float)NUMBER_OF_VOLUMES);

	// Now calculate the variance of eps
	vareps = 0.0f;
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		//eps = Volumes[Calculate4DIndex(x,y,z,c_Permutation_Vector[v],DATA_W,DATA_H,DATA_D)];
		eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
		//for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		//{
		//	eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + c_Permutation_Vector[v]] * beta[r];
		//}
		eps = CalculateEps(eps, beta, c_X_GLM, v, c_Permutation_Vector, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);
		//vareps += (eps - meaneps) * (eps - meaneps);
		vareps += (eps - meaneps) * (eps - meaneps) / ((float)NUMBER_OF_VOLUMES - 1.0f);
	}
	//vareps /= ((float)NUMBER_OF_VOLUMES - (float)NUMBER_OF_REGRESSORS - 1.0f);
	//vareps /= ((float)NUMBER_OF_VOLUMES - 1.0f);

	// Loop over contrasts and calculate t-values
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		float contrast_value = 0.0f;
		//for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		//{
			//contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + r] * beta[r];
		//}
		contrast_value = CalculateContrastValue(beta, c_Contrasts, c, NUMBER_OF_REGRESSORS);
		Statistical_Maps[Calculate4DIndex(x,y,z,c,DATA_W,DATA_H,DATA_D)] = contrast_value * rsqrt(vareps * c_ctxtxc_GLM[c]);
	}
}
*/




/*
		cluster_index *= Cluster_Indices[Calculate3DIndex(x-1,y,z-1,DATA_W,DATA_H)];
		cluster_index *= Cluster_Indices[Calculate3DIndex(x,y-1,z-1,DATA_W,DATA_H)];
		cluster_index *= Cluster_Indices[Calculate3DIndex(x,y,z-1,DATA_W,DATA_H)];
		cluster_index *= Cluster_Indices[Calculate3DIndex(x+1,y,z-1,DATA_W,DATA_H)];
		cluster_index *= Cluster_Indices[Calculate3DIndex(x,y+1,z-1,DATA_W,DATA_H)];
		
		cluster_index *= Cluster_Indices[Calculate3DIndex(x-1,y-1,z,DATA_W,DATA_H)];
		cluster_index *= Cluster_Indices[Calculate3DIndex(x-1,y,z,DATA_W,DATA_H)];
		cluster_index *= Cluster_Indices[Calculate3DIndex(x,y-1,z,DATA_W,DATA_H)];
		cluster_index *= Cluster_Indices[Calculate3DIndex(x+1,y,z,DATA_W,DATA_H)];
		cluster_index *= Cluster_Indices[Calculate3DIndex(x,y+1,z,DATA_W,DATA_H)];
		cluster_index *= Cluster_Indices[Calculate3DIndex(x+1,y+1,z,DATA_W,DATA_H)];
		cluster_index *= Cluster_Indices[Calculate3DIndex(x-1,y+1,z,DATA_W,DATA_H)];
		cluster_index *= Cluster_Indices[Calculate3DIndex(x+1,y-1,z,DATA_W,DATA_H)];

		cluster_index *= Cluster_Indices[Calculate3DIndex(x-1,y,z+1,DATA_W,DATA_H)];
		cluster_index *= Cluster_Indices[Calculate3DIndex(x,y-1,z+1,DATA_W,DATA_H)];
		cluster_index *= Cluster_Indices[Calculate3DIndex(x,y,z+1,DATA_W,DATA_H)];
		cluster_index *= Cluster_Indices[Calculate3DIndex(x+1,y,z+1,DATA_W,DATA_H)];
		cluster_index *= Cluster_Indices[Calculate3DIndex(x,y+1,z+1,DATA_W,DATA_H)];
		*/





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



		//clSetKernelArg(RemoveMeanKernel, 0, sizeof(cl_mem), &d_Volumes);
	//clSetKernelArg(RemoveMeanKernel, 1, sizeof(int), &EPI_DATA_W);
	//clSetKernelArg(RemoveMeanKernel, 2, sizeof(int), &EPI_DATA_H);
	//clSetKernelArg(RemoveMeanKernel, 3, sizeof(int), &EPI_DATA_D);
	//clSetKernelArg(RemoveMeanKernel, 4, sizeof(int), &EPI_DATA_T);
	//clEnqueueNDRangeKernel(commandQueue, RemoveMeanKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);




	/*
void BROCCOLI_LIB::CalculateStatisticalMapsGLMTTestFirstLevel(cl_mem d_Volumes)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, AR_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	PerformSmoothing(d_Smoothed_EPI_Mask, d_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

	// All timepoints are valid the first run
	NUMBER_OF_INVALID_TIMEPOINTS = 0;
	c_Censored_Timepoints = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(float), NULL, NULL);
	SetMemory(c_Censored_Timepoints, 1.0f, EPI_DATA_T);

	// Reset all AR parameters
	SetMemory(d_AR1_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_AR2_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_AR3_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_AR4_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);

	clEnqueueCopyBuffer(commandQueue, d_Volumes, d_Whitened_fMRI_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), 0, NULL, NULL);

	// Cochrane-Orcutt procedure
	for (int it = 0; it < 1; it++)
	{
		// Calculate beta values, using whitened data and the whitened voxel-specific model
		clSetKernelArg(CalculateBetaWeightsGLMKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateBetaWeightsGLMKernel, 1, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(CalculateBetaWeightsGLMKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateBetaWeightsGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_GLM);
		clSetKernelArg(CalculateBetaWeightsGLMKernel, 4, sizeof(cl_mem), &c_Censored_Timepoints);
		clSetKernelArg(CalculateBetaWeightsGLMKernel, 5, sizeof(int), &EPI_DATA_W);
		clSetKernelArg(CalculateBetaWeightsGLMKernel, 6, sizeof(int), &EPI_DATA_H);
		clSetKernelArg(CalculateBetaWeightsGLMKernel, 7, sizeof(int), &EPI_DATA_D);
		clSetKernelArg(CalculateBetaWeightsGLMKernel, 8, sizeof(int), &EPI_DATA_T);
		clSetKernelArg(CalculateBetaWeightsGLMKernel, 9, sizeof(int), &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		runKernelErrorCalculateBetaWeightsGLM = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
		clFinish(commandQueue);

		// Calculate t-values and residuals, using original data and the original model
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 1, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 2, sizeof(cl_mem), &d_Residual_Variances);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 3, sizeof(cl_mem), &d_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 4, sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 5, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 6, sizeof(cl_mem), &c_X_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 7, sizeof(cl_mem), &c_Contrasts);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 8, sizeof(cl_mem), &c_ctxtxc_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 9, sizeof(cl_mem), &c_Censored_Timepoints);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 10, sizeof(int),   &EPI_DATA_W);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 11, sizeof(int),   &EPI_DATA_H);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 12, sizeof(int),   &EPI_DATA_D);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 13, sizeof(int),   &EPI_DATA_T);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 14, sizeof(int),   &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 15, sizeof(int),   &NUMBER_OF_CONTRASTS);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 16, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorCalculateStatisticalMapsGLMTTest = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMTTestKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
		clFinish(commandQueue);

		// Estimate auto correlation from residuals
		clSetKernelArg(EstimateAR4ModelsKernel, 0, sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 1, sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 2, sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 3, sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 4, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(EstimateAR4ModelsKernel, 5, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(EstimateAR4ModelsKernel, 6, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(EstimateAR4ModelsKernel, 7, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(EstimateAR4ModelsKernel, 8, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(EstimateAR4ModelsKernel, 9, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(EstimateAR4ModelsKernel, 10, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorEstimateAR4Models = clEnqueueNDRangeKernel(commandQueue, EstimateAR4ModelsKernel, 3, NULL, globalWorkSizeEstimateAR4Models, localWorkSizeEstimateAR4Models, 0, NULL, NULL);

		// Smooth auto correlation estimates
		PerformSmoothingNormalized(d_AR1_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR2_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR3_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR4_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

		// Apply whitening to data
		clSetKernelArg(ApplyWhiteningAR4Kernel, 0, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 1, sizeof(cl_mem), &d_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 2, sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 3, sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 4, sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 5, sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 6, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 7, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 8, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 9, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 10, sizeof(int),   &EPI_DATA_T);
		runKernelErrorApplyWhiteningAR4 = clEnqueueNDRangeKernel(commandQueue, ApplyWhiteningAR4Kernel, 3, NULL, globalWorkSizeApplyWhiteningAR4, localWorkSizeApplyWhiteningAR4, 0, NULL, NULL);

		// First four timepoints are now invalid
		SetMemory(c_Censored_Timepoints, 0.0f, 4);
		NUMBER_OF_INVALID_TIMEPOINTS = 4;
	}

	clReleaseMemObject(c_Censored_Timepoints);
}
*/

