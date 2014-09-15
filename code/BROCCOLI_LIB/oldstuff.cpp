
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



	/*
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_1_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_1_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_1_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_1_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_2_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_2_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_2_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_2_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_3_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_3_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_3_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_3_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	*/

	/*
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_1_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_1_Parametric_Registration_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_1_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_1_Parametric_Registration_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_2_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_2_Parametric_Registration_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_2_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_2_Parametric_Registration_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_3_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_3_Parametric_Registration_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_3_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_3_Parametric_Registration_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	*/



	//clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	//clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	//convolution_time += time_end - time_start;


	int MEAN_REGRESSOR = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1);

	// Demean regressors
	for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
	{
		if (r != MEAN_REGRESSOR)
		{
			Eigen::VectorXd regressor = X.block(NUMBER_OF_INVALID_TIMEPOINTS,r,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS,1);
			DemeanRegressor(regressor,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS);
			X.block(NUMBER_OF_INVALID_TIMEPOINTS,r,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS,1) = regressor;
		}
	}




	/*
	__kernel void CalculateStatisticalMapsGLMBayesianOld(__global float* Statistical_Maps,
			                                       __global const float* Volumes,
			                                       __global const float* Mask,
			                                       __constant float* c_X_GLM,
			                                       __constant float* c_Contrasts,
			                                       __constant float* c_OmegaT,
												   __constant float* c_InvOmegaT,
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

		if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
		{
			Statistical_Maps[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;

			return;
		}

		int seed = Calculate3DIndex(x,y,z,DATA_W,DATA_H) * 1000;

		// Prior options
		float beta0 = 0.0f;                // Prior mean of beta. p-vector. Scalar value here will be replicated in the vector
		float tau = 100.0f;                // Prior covariance matrix of beta is tau^2*(X'X)^-1
		float iota = 1.0f;                 // Decay factor for lag length in prior for rho.
		float r = 0.5f;                    // Prior mean on rho1
		float c = 0.3f;                    // Prior standard deviation on first lag.
		float a0 = 0.01f;                  // First parameter in IG prior for sigma^2
		float b0 = 0.01f;                  // Second parameter in IG prior for sigma^2

		// Algorithmic options
		float prcBurnin = 10.0f;             // Percentage of nIter used for burnin. Note: effective number of iter is nIter.

		float beta[2];
		float betaT[2];

		betaT[0] = 0.0f;
		betaT[1] = 0.0f;

		beta[0] = 0.0f;
		beta[1] = 0.0f;


		float ysquared = 0.0f;
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			float value = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
			//CalculateBetaWeightsBayesian(beta, value, c_X_GLM, v, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);

			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				betaT[r] += value * c_X_GLM[NUMBER_OF_VOLUMES * r + v];
			}

			ysquared += value*value;
		}

		beta[0] = c_OmegaT[0 + 0 * NUMBER_OF_REGRESSORS] * betaT[0] + c_OmegaT[0 + 1 * NUMBER_OF_REGRESSORS] * betaT[1];
		beta[1] = c_OmegaT[1 + 0 * NUMBER_OF_REGRESSORS] * betaT[0] + c_OmegaT[1 + 1 * NUMBER_OF_REGRESSORS] * betaT[1];

		betaT[0] = beta[0];
		betaT[1] = beta[1];

		float aT = a0 + (float)NUMBER_OF_VOLUMES/2.0f;

		float betaomega[2];
		betaomega[0] = c_InvOmegaT[0 + 0 * NUMBER_OF_REGRESSORS] * betaT[0] + c_InvOmegaT[1 + 0 * NUMBER_OF_REGRESSORS] * betaT[1];
		betaomega[1] = c_InvOmegaT[0 + 1 * NUMBER_OF_REGRESSORS] * betaT[0] + c_InvOmegaT[1 + 1 * NUMBER_OF_REGRESSORS] * betaT[1];
		float scalar = betaomega[0] * betaT[0] + betaomega[1] * betaT[1];

		float bT = b0 + 0.5f*(ysquared - scalar);

		int nBurnin = (int)round((float)NUMBER_OF_ITERATIONS*(prcBurnin/100.0f));
		int probability = 0;

		float sigma2;


		// Loop over iterations
		for (int i = 0; i < (nBurnin + NUMBER_OF_ITERATIONS); i++)
		//for (int i = 0; i < 100; i++)
		{
			// Block 1 - Step 1a. Update sigma2
			sigma2 = gamrnd(aT,bT,&seed);

			// Block 1 - Step 1b. Update beta | sigma2
			MultivariateRandom(beta,betaT,c_OmegaT,sigma2,NUMBER_OF_REGRESSORS,&seed);

			if (i > nBurnin)
			{
				//float contrast_value = 0.0f;
				//for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
				//{
				//	contrast_value += c_Contrasts[r] * beta[r];
				//}

				if (beta[0] > 0.0f)
				{
					probability++;
				}
			}
		}


		//float chol[4];
		//Cholesky(chol, 1.0f, c_OmegaT, NUMBER_OF_REGRESSORS);


		Statistical_Maps[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = (float)probability/(float)NUMBER_OF_ITERATIONS;

		//Statistical_Maps[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = (float)probability/100.0f;


		//aT = 39.51;
		//bT = 1216.9f;
		//sigma2 = gamrnd(aT,bT,&seed);
		//MultivariateRandom(beta,betaT,c_OmegaT,sigma2,NUMBER_OF_REGRESSORS,&seed);


		//sigma2 = (float)unirand(&seed);
		//Statistical_Maps[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = sigma2;
		//Statistical_Maps[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)] = bT;
		//Statistical_Maps[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)] = beta[1];
		//Statistical_Maps[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)] = chol[2];
		//Statistical_Maps[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)] = chol[3];

	}
	*/


	/*
	int MultivariateRandom(float* random, float* mu, __private float* Cov, float Sigma, int N, __private int* seed)
	{
		float randvalues[2];
		float cholCov[4];

		switch(N)
		{
			case 1:

				randvalues[0] = normalrand(seed);

				Cholesky1(cholCov, Sigma, Cov);

				random[0] = mu[0] + cholCov[0 + 0 * N] * randvalues[0];

				break;


			case 2:

				randvalues[0] = normalrand(seed);
				randvalues[1] = normalrand(seed);

				Cholesky2(cholCov, Sigma, Cov);

				random[0] = mu[0] + cholCov[0 + 0 * N] * randvalues[0] + cholCov[1 + 0 * N] * randvalues[1];
				random[1] = mu[1] + cholCov[0 + 1 * N] * randvalues[0] + cholCov[1 + 1 * N] * randvalues[1];

				break;


			default:
				1;
				break;
		}

		return 0;
	}
	*/




	// Calculate betahat, i.e. multiply (x^T x)^(-1) x^T with Y
		// Loop over volumes
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			float temp = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
	        //float temp = Volumes[Calculate4DIndex(x,y,z,c_Permutation_Vector[v],DATA_W,DATA_H,DATA_D)];
			// Loop over regressors
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				beta[r] += temp * c_xtxxt_GLM[NUMBER_OF_VOLUMES * r + c_Permutation_Vector[v]];
	            //beta[r] += temp * c_xtxxt_GLM[NUMBER_OF_VOLUMES * r + v];
			}
		}


	    // Calculate sigma squared
		vareps = 0.0f;
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
	        //eps = Volumes[Calculate4DIndex(x,y,z,c_Permutation_Vector[v],DATA_W,DATA_H,DATA_D)];
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + c_Permutation_Vector[v]] * beta[r];
	            //eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
			}
			vareps += eps*eps;
		}
		vareps /= ((float)NUMBER_OF_VOLUMES - 1.0f);


	    /*
	    // Calculate the mean of the error eps
		meaneps = 0.0f;
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
	        //eps = Volumes[Calculate4DIndex(x,y,z,c_Permutation_Vector[v],DATA_W,DATA_H,DATA_D)];
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + c_Permutation_Vector[v]] * beta[r];
	            //eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
			}
			meaneps += eps;
		}
		meaneps /= ((float)NUMBER_OF_VOLUMES);

		// Now calculate the variance of eps
		vareps = 0.0f;
		for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
		{
			eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
	        //eps = Volumes[Calculate4DIndex(x,y,z,c_Permutation_Vector[v],DATA_W,DATA_H,DATA_D)];
			for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
			{
				eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + c_Permutation_Vector[v]] * beta[r];
	            //eps -= c_X_GLM[NUMBER_OF_VOLUMES * r + v] * beta[r];
			}
			vareps += (eps - meaneps) * (eps - meaneps);
		}
		vareps /= ((float)NUMBER_OF_VOLUMES - 1.0f);

	     */
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



/*
 * Old version of OpenCLInitiate, hard to read and lacks some error checking
 *

void BROCCOLI_LIB::OpenCLInitiate(cl_uint OPENCL_PLATFORM, cl_uint OPENCL_DEVICE)
{
	char* value;
	size_t valueSize;
	cl_device_id *clDevices;

  	// Get number of platforms
	cl_uint platformIdCount = 0;
	getPlatformIDsError = clGetPlatformIDs (0, NULL, &platformIdCount);

	if (getPlatformIDsError == SUCCESS)
	{
		// Get platform IDs
		std::vector<cl_platform_id> platformIds(platformIdCount);
		getPlatformIDsError = clGetPlatformIDs(platformIdCount, platformIds.data(), NULL);              

		if (getPlatformIDsError == SUCCESS)
		{	
			// Check if the requested platform exists
			if ((OPENCL_PLATFORM >= 0) &&  (OPENCL_PLATFORM < platformIdCount))
			{
				// Create context
				const cl_context_properties contextProperties [] =
				{
					CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[OPENCL_PLATFORM]), 0, 0
				};

				// Get number of devices for selected platform
				cl_uint deviceIdCount = 0;
				getDeviceIDsError = clGetDeviceIDs(platformIds[OPENCL_PLATFORM], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceIdCount);
	
				if (getDeviceIDsError == SUCCESS)
				{
					// Get device IDs for selected platform
					std::vector<cl_device_id> deviceIds(deviceIdCount);
					getDeviceIDsError = clGetDeviceIDs(platformIds[OPENCL_PLATFORM], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), NULL);

					// Check if the requested device exists
					if ((OPENCL_DEVICE >= 0) &&  (OPENCL_DEVICE < deviceIdCount))
					{
						if (getDeviceIDsError == SUCCESS)
						{
							// Create context for selected device
							//context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), NULL, NULL, &createContextError);
							context = clCreateContext(contextProperties, 1, &deviceIds[OPENCL_DEVICE], NULL, NULL, &createContextError);

							if (createContextError == SUCCESS)
							{
								// Get size of context info
								getContextInfoError = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &valueSize);

								if (getContextInfoError == SUCCESS)
								{
									// Get context info
									clDevices = (cl_device_id *) malloc(valueSize);
									getContextInfoError = clGetContextInfo(context, CL_CONTEXT_DEVICES, valueSize, clDevices, NULL);

									if (getContextInfoError == SUCCESS)
									{
										// Get size of name of current platform
										clGetPlatformInfo(platformIds[OPENCL_PLATFORM], CL_PLATFORM_NAME, 0, NULL, &valueSize);
										value = (char*) malloc(valueSize);
										// Get name of current platform
										clGetPlatformInfo(platformIds[OPENCL_PLATFORM], CL_PLATFORM_NAME, valueSize, value, NULL);
										std::string vendor_string(value);
										free(value);

										// Figure out the vendor
										size_t npos = vendor_string.find("NVIDIA");
										size_t ipos = vendor_string.find("Intel");
										size_t apos = vendor_string.find("AMD");

										binaryFilename = "broccoli_lib_kernel_unknown";
										if (npos != std::string::npos)
										{
											VENDOR = NVIDIA;
											binaryFilename = "broccoli_lib_kernel_Nvidia";
										}
										else if (ipos != std::string::npos)
										{
											VENDOR = INTEL;
											binaryFilename = "broccoli_lib_kernel_Intel";
										}
										else if (apos != std::string::npos)
										{
											VENDOR = AMD;
											binaryFilename = "broccoli_lib_kernel_AMD";
										}
										else if (WRAPPER == BASH)
										{
											printf("\nUnsupported OpenCL vendor!\n\n");
										}

										// Create a command queue for the selected device
										commandQueue = clCreateCommandQueue(context, deviceIds[OPENCL_DEVICE], CL_QUEUE_PROFILING_ENABLE, &createCommandQueueError);

										if (createCommandQueueError == SUCCESS)
										{
											// Support for running functions from any folder
											//std::string kernelFileName = Getexepath();
											//kernelFileName.erase(kernelFileName.end()-16, kernelFileName.end());
											//kernelFileName.append(binaryFilename);

											std::string kernelFileName;
											kernelFileName.append(binaryFilename);

											// First try to compile from binary file for the selected device
											//createProgramError = CreateProgramFromBinary(program, context, deviceIds[OPENCL_DEVICE], kernelFileName);
											createProgramError = CreateProgramFromBinary(program, context, deviceIds[OPENCL_DEVICE], binaryFilename);
											//buildProgramError = clBuildProgram(program, deviceIdCount, deviceIds.data(), NULL, NULL, NULL);

											if (VENDOR == NVIDIA)
											{
												buildProgramError = clBuildProgram(program, 1, &deviceIds[OPENCL_DEVICE], "-cl-nv-verbose", NULL, NULL);
											}
											else
											{
												//buildProgramError = clBuildProgram(program, 1, &deviceIds[OPENCL_DEVICE], "-cl-opt-disable", NULL, NULL);
												buildProgramError = clBuildProgram(program, 1, &deviceIds[OPENCL_DEVICE], NULL, NULL, NULL);
											}

											// Otherwise compile from source code
											if (buildProgramError != SUCCESS)
											{
												// Read the kernel code from file
												//std::string kernelFileName = Getexepath();
												//kernelFileName.erase(kernelFileName.end()-16, kernelFileName.end());

												//std::string kernelFileName;
												//kernelFileName.append("broccoli_lib_kernel.cpp");
												//std::fstream kernelFile(kernelFileName.c_str(),std::ios::in);
												std::fstream kernelFile("broccoli_lib_kernel.cpp",std::ios::in);
												std::ostringstream oss;
												oss << kernelFile.rdbuf();
												std::string src = oss.str();
												const char *srcstr = src.c_str();

												// Create program and build the code for the selected device
												program = clCreateProgramWithSource(context, 1, (const char**)&srcstr , NULL, &createProgramError);
												//buildProgramError = clBuildProgram(program, deviceIdCount, deviceIds.data(), NULL, NULL, NULL);

												if (VENDOR == NVIDIA)
												{
													buildProgramError = clBuildProgram(program, 1, &deviceIds[OPENCL_DEVICE], "-cl-nv-verbose", NULL, NULL);
												}
												else
												{
													buildProgramError = clBuildProgram(program, 1, &deviceIds[OPENCL_DEVICE], NULL, NULL, NULL);
												}

												// If successful build, save to binary file
												if (buildProgramError == SUCCESS)
												{
													SaveProgramBinary(program,deviceIds[OPENCL_DEVICE],binaryFilename);
												}
											}

											// Always get build info

											// Get size of build info
											valueSize = 0;
											getProgramBuildInfoError = clGetProgramBuildInfo(program, deviceIds[OPENCL_DEVICE], CL_PROGRAM_BUILD_LOG, 0, NULL, &valueSize);

											// Get build info
											if (getProgramBuildInfoError == SUCCESS)
											{
												value = (char*)malloc(valueSize);
												getProgramBuildInfoError = clGetProgramBuildInfo(program, deviceIds[OPENCL_DEVICE], CL_PROGRAM_BUILD_LOG, valueSize, value, NULL);

												if (getProgramBuildInfoError == SUCCESS)
												{
													build_info.append(value);
												}
												else if (WRAPPER == BASH)
												{
													printf("\nUnable to get OpenCL build info! \n\n");
												}
												free(value);
											}
											else if (WRAPPER == BASH)
											{
												printf("\nUnable to get size of OpenCL build info!\n\n");
											}

											if (buildProgramError == SUCCESS)
											{
												// Create kernels

												// Convolution kernels
												if ( (VENDOR == NVIDIA) || (VENDOR == INTEL))
												{
													NonseparableConvolution3DComplexThreeFiltersKernel = clCreateKernel(program,"Nonseparable3DConvolutionComplexThreeQuadratureFilters",&createKernelErrorNonseparableConvolution3DComplexThreeFilters);
													SeparableConvolutionRowsKernel = clCreateKernel(program,"SeparableConvolutionRows",&createKernelErrorSeparableConvolutionRows);
													SeparableConvolutionColumnsKernel = clCreateKernel(program,"SeparableConvolutionColumns",&createKernelErrorSeparableConvolutionColumns);
													SeparableConvolutionRodsKernel = clCreateKernel(program,"SeparableConvolutionRods",&createKernelErrorSeparableConvolutionRods);
												}
												else if (VENDOR == AMD)
												{
													NonseparableConvolution3DComplexThreeFiltersKernel = clCreateKernel(program,"Nonseparable3DConvolutionComplexThreeQuadratureFiltersAMD",&createKernelErrorNonseparableConvolution3DComplexThreeFilters);
													SeparableConvolutionRowsKernel = clCreateKernel(program,"SeparableConvolutionRowsAMD",&createKernelErrorSeparableConvolutionRows);
													SeparableConvolutionColumnsKernel = clCreateKernel(program,"SeparableConvolutionColumnsAMD",&createKernelErrorSeparableConvolutionColumns);
													SeparableConvolutionRodsKernel = clCreateKernel(program,"SeparableConvolutionRodsAMD",&createKernelErrorSeparableConvolutionRods);
												}

												OpenCLKernels[0] = NonseparableConvolution3DComplexThreeFiltersKernel;
												OpenCLKernels[1] = SeparableConvolutionRowsKernel;
												OpenCLKernels[2] = SeparableConvolutionColumnsKernel;
												OpenCLKernels[3] = SeparableConvolutionRodsKernel;

												SliceTimingCorrectionKernel = clCreateKernel(program,"SliceTimingCorrection",&createKernelErrorSliceTimingCorrection);

												OpenCLKernels[4] = SliceTimingCorrectionKernel;

												// Kernels for Linear registration
												CalculatePhaseDifferencesAndCertaintiesKernel = clCreateKernel(program,"CalculatePhaseDifferencesAndCertainties",&createKernelErrorCalculatePhaseDifferencesAndCertainties);
												CalculatePhaseGradientsXKernel = clCreateKernel(program,"CalculatePhaseGradientsX",&createKernelErrorCalculatePhaseGradientsX);
												CalculatePhaseGradientsYKernel = clCreateKernel(program,"CalculatePhaseGradientsY",&createKernelErrorCalculatePhaseGradientsY);
												CalculatePhaseGradientsZKernel = clCreateKernel(program,"CalculatePhaseGradientsZ",&createKernelErrorCalculatePhaseGradientsZ);
												CalculateAMatrixAndHVector2DValuesXKernel = clCreateKernel(program,"CalculateAMatrixAndHVector2DValuesX",&createKernelErrorCalculateAMatrixAndHVector2DValuesX);
												CalculateAMatrixAndHVector2DValuesYKernel = clCreateKernel(program,"CalculateAMatrixAndHVector2DValuesY",&createKernelErrorCalculateAMatrixAndHVector2DValuesY);
												CalculateAMatrixAndHVector2DValuesZKernel = clCreateKernel(program,"CalculateAMatrixAndHVector2DValuesZ",&createKernelErrorCalculateAMatrixAndHVector2DValuesZ);
												CalculateAMatrix1DValuesKernel = clCreateKernel(program,"CalculateAMatrix1DValues",&createKernelErrorCalculateAMatrix1DValues);
												CalculateHVector1DValuesKernel = clCreateKernel(program,"CalculateHVector1DValues",&createKernelErrorCalculateHVector1DValues);
												CalculateAMatrixKernel = clCreateKernel(program,"CalculateAMatrix",&createKernelErrorCalculateAMatrix);
												CalculateHVectorKernel = clCreateKernel(program,"CalculateHVector",&createKernelErrorCalculateHVector);

												OpenCLKernels[5] = CalculatePhaseDifferencesAndCertaintiesKernel;
												OpenCLKernels[6] = CalculatePhaseGradientsXKernel;
												OpenCLKernels[7] = CalculatePhaseGradientsYKernel;
												OpenCLKernels[8] = CalculatePhaseGradientsZKernel;
												OpenCLKernels[9] = CalculateAMatrixAndHVector2DValuesXKernel;
												OpenCLKernels[10] = CalculateAMatrixAndHVector2DValuesYKernel;
												OpenCLKernels[11] = CalculateAMatrixAndHVector2DValuesZKernel;
												OpenCLKernels[12] = CalculateAMatrix1DValuesKernel;
												OpenCLKernels[13] = CalculateHVector1DValuesKernel;
												OpenCLKernels[14] = CalculateAMatrixKernel;
												OpenCLKernels[15] = CalculateHVectorKernel;

												// Kernels for non-Linear registration
												CalculateTensorComponentsKernel = clCreateKernel(program, "CalculateTensorComponents", &createKernelErrorCalculateTensorComponents);
												CalculateTensorNormsKernel = clCreateKernel(program, "CalculateTensorNorms", &createKernelErrorCalculateTensorNorms);
												CalculateAMatricesAndHVectorsKernel = clCreateKernel(program, "CalculateAMatricesAndHVectors", &createKernelErrorCalculateAMatricesAndHVectors);
												CalculateDisplacementUpdateKernel = clCreateKernel(program, "CalculateDisplacementUpdate", &createKernelErrorCalculateDisplacementUpdate);
												AddLinearAndNonLinearDisplacementKernel = clCreateKernel(program, "AddLinearAndNonLinearDisplacement", &createKernelErrorAddLinearAndNonLinearDisplacement);

												OpenCLKernels[16] = CalculateTensorComponentsKernel;
												OpenCLKernels[17] = CalculateTensorNormsKernel;
												OpenCLKernels[18] = CalculateAMatricesAndHVectorsKernel;
												OpenCLKernels[19] = CalculateDisplacementUpdateKernel;
												OpenCLKernels[20] = AddLinearAndNonLinearDisplacementKernel;

												CalculateMagnitudesKernel = clCreateKernel(program,"CalculateMagnitudes",&createKernelErrorCalculateMagnitudes);
												CalculateColumnSumsKernel = clCreateKernel(program,"CalculateColumnSums",&createKernelErrorCalculateColumnSums);
												CalculateRowSumsKernel = clCreateKernel(program,"CalculateRowSums",&createKernelErrorCalculateRowSums);
												CalculateColumnMaxsKernel = clCreateKernel(program,"CalculateColumnMaxs",&createKernelErrorCalculateColumnMaxs);
												CalculateRowMaxsKernel = clCreateKernel(program,"CalculateRowMaxs",&createKernelErrorCalculateRowMaxs);
												CalculateMaxAtomicKernel = clCreateKernel(program,"CalculateMaxAtomic",&createKernelErrorCalculateMaxAtomic);
												ThresholdVolumeKernel = clCreateKernel(program,"ThresholdVolume",&createKernelErrorThresholdVolume);

												OpenCLKernels[21] = CalculateMagnitudesKernel;
												OpenCLKernels[22] = CalculateColumnSumsKernel;
												OpenCLKernels[23] = CalculateRowSumsKernel;
												OpenCLKernels[24] = CalculateColumnMaxsKernel;
												OpenCLKernels[25] = CalculateRowMaxsKernel;
												OpenCLKernels[26] = CalculateMaxAtomicKernel;
												OpenCLKernels[27] = ThresholdVolumeKernel;

												// Interpolation kernels
												InterpolateVolumeNearestLinearKernel = clCreateKernel(program,"InterpolateVolumeNearestLinear",&createKernelErrorInterpolateVolumeNearestLinear);
												InterpolateVolumeLinearLinearKernel = clCreateKernel(program,"InterpolateVolumeLinearLinear",&createKernelErrorInterpolateVolumeLinearLinear);
												InterpolateVolumeCubicLinearKernel = clCreateKernel(program,"InterpolateVolumeCubicLinear",&createKernelErrorInterpolateVolumeCubicLinear);
												InterpolateVolumeNearestNonLinearKernel = clCreateKernel(program,"InterpolateVolumeNearestNonLinear",&createKernelErrorInterpolateVolumeNearestNonLinear);
												InterpolateVolumeLinearNonLinearKernel = clCreateKernel(program,"InterpolateVolumeLinearNonLinear",&createKernelErrorInterpolateVolumeLinearNonLinear);
												InterpolateVolumeCubicNonLinearKernel = clCreateKernel(program,"InterpolateVolumeCubicNonLinear",&createKernelErrorInterpolateVolumeCubicNonLinear);

												OpenCLKernels[28] = InterpolateVolumeNearestLinearKernel;
												OpenCLKernels[29] = InterpolateVolumeLinearLinearKernel;
												OpenCLKernels[30] = InterpolateVolumeCubicLinearKernel;
												OpenCLKernels[31] = InterpolateVolumeNearestNonLinearKernel;
												OpenCLKernels[32] = InterpolateVolumeLinearNonLinearKernel;
												OpenCLKernels[33] = InterpolateVolumeCubicNonLinearKernel;

												RescaleVolumeLinearKernel = clCreateKernel(program,"RescaleVolumeLinear",&createKernelErrorRescaleVolumeLinear);
												RescaleVolumeCubicKernel = clCreateKernel(program,"RescaleVolumeCubic",&createKernelErrorRescaleVolumeCubic);
												RescaleVolumeNearestKernel = clCreateKernel(program,"RescaleVolumeNearest",&createKernelErrorRescaleVolumeNearest);

												OpenCLKernels[34] = RescaleVolumeLinearKernel;
												OpenCLKernels[35] = RescaleVolumeCubicKernel;
												OpenCLKernels[36] = RescaleVolumeNearestKernel;

												CopyT1VolumeToMNIKernel = clCreateKernel(program,"CopyT1VolumeToMNI",&createKernelErrorCopyT1VolumeToMNI);
												CopyEPIVolumeToT1Kernel = clCreateKernel(program,"CopyEPIVolumeToT1",&createKernelErrorCopyEPIVolumeToT1);
												CopyVolumeToNewKernel = clCreateKernel(program,"CopyVolumeToNew",&createKernelErrorCopyVolumeToNew);

												OpenCLKernels[37] = CopyT1VolumeToMNIKernel;
												OpenCLKernels[38] = CopyEPIVolumeToT1Kernel;
												OpenCLKernels[39] = CopyVolumeToNewKernel;

												// Help kernels
												MemsetKernel = clCreateKernel(program,"Memset",&createKernelErrorMemset);
												MemsetIntKernel = clCreateKernel(program,"MemsetInt",&createKernelErrorMemsetInt);
												MemsetFloat2Kernel = clCreateKernel(program,"MemsetFloat2",&createKernelErrorMemsetFloat2);
												MultiplyVolumeKernel = clCreateKernel(program,"MultiplyVolume",&createKernelErrorMultiplyVolume);
												MultiplyVolumesKernel = clCreateKernel(program,"MultiplyVolumes",&createKernelErrorMultiplyVolumes);
												MultiplyVolumesOverwriteKernel = clCreateKernel(program,"MultiplyVolumesOverwrite",&createKernelErrorMultiplyVolumesOverwrite);
												AddVolumeKernel = clCreateKernel(program,"AddVolume",&createKernelErrorAddVolume);
												AddVolumesKernel = clCreateKernel(program,"AddVolumes",&createKernelErrorAddVolumes);
												AddVolumesOverwriteKernel = clCreateKernel(program,"AddVolumesOverwrite",&createKernelErrorAddVolumesOverwrite);
												RemoveMeanKernel = clCreateKernel(program,"RemoveMean",&createKernelErrorRemoveMean);
												SetStartClusterIndicesKernel = clCreateKernel(program,"SetStartClusterIndicesKernel",&createKernelErrorSetStartClusterIndices);
												ClusterizeScanKernel = clCreateKernel(program,"ClusterizeScan",&createKernelErrorClusterizeScan);
												ClusterizeRelabelKernel = clCreateKernel(program,"ClusterizeRelabel",&createKernelErrorClusterizeRelabel);
												CalculateClusterSizesKernel = clCreateKernel(program,"CalculateClusterSizes",&createKernelErrorCalculateClusterSizes);
												CalculateLargestClusterKernel = clCreateKernel(program,"CalculateLargestCluster",&createKernelErrorCalculateLargestCluster);


												OpenCLKernels[40] = MemsetKernel;
												OpenCLKernels[41] = MemsetIntKernel;
												OpenCLKernels[42] = MemsetFloat2Kernel;
												OpenCLKernels[43] = MultiplyVolumeKernel;
												OpenCLKernels[44] = MultiplyVolumesKernel;
												OpenCLKernels[45] = MultiplyVolumesOverwriteKernel;
												OpenCLKernels[46] = AddVolumeKernel;
												OpenCLKernels[47] = AddVolumesKernel;
												OpenCLKernels[48] = AddVolumesOverwriteKernel;
												OpenCLKernels[49] = RemoveMeanKernel;
												OpenCLKernels[50] = SetStartClusterIndicesKernel;
												OpenCLKernels[51] = ClusterizeScanKernel;
												OpenCLKernels[52] = ClusterizeRelabelKernel;
												OpenCLKernels[53] = CalculateClusterSizesKernel;
												OpenCLKernels[54] = CalculateLargestClusterKernel;

												// Statistical kernels
												CalculateBetaWeightsGLMKernel = clCreateKernel(program,"CalculateBetaWeightsGLM",&createKernelErrorCalculateBetaWeightsGLM);
												CalculateBetaWeightsGLMFirstLevelKernel = clCreateKernel(program,"CalculateBetaWeightsGLMFirstLevel",&createKernelErrorCalculateBetaWeightsGLMFirstLevel);
												CalculateGLMResidualsKernel = clCreateKernel(program,"CalculateGLMResiduals",&createKernelErrorCalculateGLMResiduals);
												CalculateStatisticalMapsGLMTTestFirstLevelKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMTTestFirstLevel",&createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel);
												CalculateStatisticalMapsGLMFTestFirstLevelKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMFTestFirstLevel",&createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevel);
												CalculateStatisticalMapsGLMTTestKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMTTest",&createKernelErrorCalculateStatisticalMapsGLMTTest);
												CalculateStatisticalMapsGLMFTestKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMFTest",&createKernelErrorCalculateStatisticalMapsGLMFTest);
												CalculateStatisticalMapsGLMBayesianKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMBayesian",&createKernelErrorCalculateStatisticalMapsGLMBayesian);
												CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMTTestFirstLevelPermutation",&createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelPermutation);
												CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMFTestFirstLevelPermutation",&createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelPermutation);
												CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMTTestSecondLevelPermutation",&createKernelErrorCalculateStatisticalMapsGLMTTestSecondLevelPermutation);
												CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMFTestSecondLevelPermutation",&createKernelErrorCalculateStatisticalMapsGLMFTestSecondLevelPermutation);
												EstimateAR4ModelsKernel = clCreateKernel(program,"EstimateAR4Models",&createKernelErrorEstimateAR4Models);
												ApplyWhiteningAR4Kernel = clCreateKernel(program,"ApplyWhiteningAR4",&createKernelErrorApplyWhiteningAR4);
												GeneratePermutedVolumesFirstLevelKernel = clCreateKernel(program,"GeneratePermutedVolumesFirstLevel",&createKernelErrorGeneratePermutedVolumesFirstLevel);
												RemoveLinearFitKernel = clCreateKernel(program,"RemoveLinearFit",&createKernelErrorRemoveLinearFit);

												OpenCLKernels[55] = CalculateBetaWeightsGLMKernel;
												OpenCLKernels[56] = CalculateBetaWeightsGLMFirstLevelKernel;
												OpenCLKernels[57] = CalculateGLMResidualsKernel;
												OpenCLKernels[58] = CalculateStatisticalMapsGLMTTestFirstLevelKernel;
												OpenCLKernels[59] = CalculateStatisticalMapsGLMFTestFirstLevelKernel;
												OpenCLKernels[60] = CalculateStatisticalMapsGLMTTestKernel;
												OpenCLKernels[61] = CalculateStatisticalMapsGLMFTestKernel;
												OpenCLKernels[62] = CalculateStatisticalMapsGLMBayesianKernel;
												OpenCLKernels[63] = CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel;
												OpenCLKernels[64] = CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel;
												OpenCLKernels[65] = CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel;
												OpenCLKernels[66] = CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel;
												OpenCLKernels[67] = EstimateAR4ModelsKernel;
												OpenCLKernels[68] = ApplyWhiteningAR4Kernel;
												OpenCLKernels[69] = GeneratePermutedVolumesFirstLevelKernel;
												OpenCLKernels[70] = RemoveLinearFitKernel;

												OPENCL_INITIATED = 1;
											}
											else if (WRAPPER == BASH)
											{
												printf("\nUnable to build OpenCL program. Aborting! \n\n");
											}
										}
										else if (WRAPPER == BASH)
										{
											printf("\nUnable to create an OpenCL command queue. Aborting! \n\n");
										}
									}
									else if (WRAPPER == BASH)
									{
										printf("\nUnable to get OpenCL context info. Aborting! \n\n");
									}
									free(clDevices);
								}
								else if (WRAPPER == BASH)
								{
									printf("\nUnable to get size of OpenCL context info. Aborting! \n\n");
								}
							}
							else if (WRAPPER == BASH)
							{
								printf("\nUnable to create an OpenCL context. Aborting! \n\n");
							}
						}
						else if (WRAPPER == BASH)
						{
							printf("\nUnable to get OpenCL device id's for the specified platform. Aborting! \n\n");
						}
					}
					else if (WRAPPER == BASH)
					{
						printf("\nYou tried to use the invalid OpenCL device %i, valid devices for the selected platform are 0 <= device < %i. Aborting! \n\n",OPENCL_DEVICE,deviceIdCount);
					}
				}
				else if (WRAPPER == BASH)
				{
					printf("\nUnable to get number of OpenCL devices for the specified platform. Aborting! \n\n");
				}
			}
			else if (WRAPPER == BASH)
			{
				printf("\nYou tried to use the invalid OpenCL platform %i, valid platforms are 0 <= platform < %i. Aborting! \n\n",OPENCL_PLATFORM,platformIdCount);
			}
		}
		else if (WRAPPER == BASH)
		{
			printf("\nUnable to get OpenCL platform id's. Aborting! \n\n");
		}
	}
	else if (WRAPPER == BASH)
	{
		printf("\nUnable to get number of OpenCL platforms. Aborting! \n\n");
	}
}
*/




void BROCCOLI_LIB::PerformRegistrationEPIT1Wrapper()
{
	// Reset total registration parameters
	for (int p = 0; p < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; p++)
	{
		h_Registration_Parameters_EPI_T1_Affine[p] = 0.0f;
	}

	// Allocate memory for EPI volume, T1 volume and EPI volume of T1 size
	d_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	d_T1_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to EPI volume and T1 volume
	clEnqueueWriteBuffer(commandQueue, d_EPI_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);

	// Make a segmentation of the EPI volume first
	SegmentEPIData(d_EPI_Volume);

	// Interpolate EPI volume to T1 resolution and make sure it has the same size
	ChangeEPIVolumeResolutionAndSize(d_T1_EPI_Volume, d_EPI_Volume, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, INTERPOLATION_MODE);

	// Copy the interpolated EPI T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_Interpolated_EPI_Volume, 0, NULL, NULL);

	// Calculate tensor magnitudes
	cl_mem d_T1_EPI_Tensor_Magnitude = clCreateBuffer(context, CL_MEM_READ_WRITE, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_T1_Tensor_Magnitude = clCreateBuffer(context, CL_MEM_READ_WRITE, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);

	CalculateTensorMagnitude(d_T1_EPI_Tensor_Magnitude, d_T1_EPI_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D);
	CalculateTensorMagnitude(d_T1_Tensor_Magnitude, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D);

	// Do the registration between EPI and skullstripped T1 with several scales, first translation
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Translation, h_Rotations, d_T1_EPI_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, TRANSLATION, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Tensor_Magnitude, h_Registration_Parameters_EPI_T1_Translation, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Translation);

	// Do the registration between EPI and skullstripped T1 with several scales, first translation
	//AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, COARSEST_SCALE_EPI_T1/2, NUMBER_OF_ITERATIONS_FOR_Linear_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);
	//TransformVolumesLinear(d_T1_EPI_Tensor_Magnitude, h_Registration_Parameters_EPI_T1_Rigid, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	//AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Rigid);


	// Rigid with tensor magnitudes
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Tensor_Magnitude, d_T1_Tensor_Magnitude, T1_DATA_W, T1_DATA_H, T1_DATA_D, 2, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Rigid, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Rigid);


	// Affine with tensor magnitudes
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Tensor_Magnitude, d_T1_Tensor_Magnitude, T1_DATA_W, T1_DATA_H, T1_DATA_D, 2, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Rigid, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Rigid);


	// Copy the aligned EPI volume to host
	clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_Aligned_EPI_Volume_T1, 0, NULL, NULL);

	// Get translations
	h_Registration_Parameters_EPI_T1_Out[0] = h_Registration_Parameters_EPI_T1_Affine[0];
	h_Registration_Parameters_EPI_T1_Out[1] = h_Registration_Parameters_EPI_T1_Affine[1];
	h_Registration_Parameters_EPI_T1_Out[2] = h_Registration_Parameters_EPI_T1_Affine[2];

	// Get rotations
	h_Registration_Parameters_EPI_T1_Out[3] = h_Rotations[0];
	h_Registration_Parameters_EPI_T1_Out[4] = h_Rotations[1];
	h_Registration_Parameters_EPI_T1_Out[5] = h_Rotations[2];

	// Cleanup
	clReleaseMemObject(d_EPI_Volume);
	clReleaseMemObject(d_T1_Volume);
	clReleaseMemObject(d_T1_EPI_Volume);

	clReleaseMemObject(d_T1_EPI_Tensor_Magnitude);
	clReleaseMemObject(d_T1_Tensor_Magnitude);
}




void BROCCOLI_LIB::PerformRegistrationT1MNIWrapper()
{
	// Allocate memory for T1 volume, MNI volume and T1 volume of MNI size
	d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to T1 volume and MNI volume
	clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Volume , 0, NULL, NULL);

	// Interpolate T1 volume to MNI resolution and make sure it has the same size
	ChangeT1VolumeResolutionAndSize(d_MNI_T1_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, INTERPOLATION_MODE, NOT_SKULL_STRIPPED);

	clReleaseMemObject(d_T1_Volume);

	// Copy the MNI T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Interpolated_T1_Volume, 0, NULL, NULL);

	// Do the registration between T1 and MNI with several scales (with skull)
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_T1_MNI_Out, h_Rotations, d_MNI_T1_Volume, d_MNI_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, NO_OVERWRITE, INTERPOLATION_MODE);

	clReleaseMemObject(d_MNI_Volume);

	// Copy the aligned T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_Linear, 0, NULL, NULL);

	// Allocate memory for MNI brain mask
	d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy MNI brain mask to device
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

	// Calculate inverse transform between T1 and MNI
	InvertAffineRegistrationParameters(h_Inverse_Registration_Parameters, h_Registration_Parameters_T1_MNI_Out);

	// Now apply the inverse transformation between MNI and T1, to transform MNI brain mask to original T1 space
	TransformVolumesLinear(d_MNI_Brain_Mask, h_Inverse_Registration_Parameters, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, NEAREST);

	// Create skullstripped volume, by multiplying original T1 volume with transformed MNI brain mask
	MultiplyVolumes(d_MNI_T1_Volume, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);


	d_MNI_Brain_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Volume , 0, NULL, NULL);


	// Now align skullstripped volume with MNI brain
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_T1_MNI_Out, h_Rotations, d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, NO_OVERWRITE, INTERPOLATION_MODE);

	// Copy MNI brain mask to device again
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

	// Calculate inverse transform between T1 and MNI (to get better skullstrip)
	InvertAffineRegistrationParameters(h_Inverse_Registration_Parameters, h_Registration_Parameters_T1_MNI_Out);

	// Apply inverse transform
	TransformVolumesLinear(d_MNI_Brain_Mask, h_Inverse_Registration_Parameters, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, NEAREST);

	// Multiply inverse transformed mask with original volume (to get better skullstrip)
	MultiplyVolumes(d_MNI_T1_Volume, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	// Copy the skullstripped T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Skullstripped_T1_Volume, 0, NULL, NULL);

	// Apply forward transform to skullstripped volume
	TransformVolumesLinear(d_MNI_T1_Volume, h_Registration_Parameters_T1_MNI_Out, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_Linear, 0, NULL, NULL);

	// Perform non-Linear registration between tramsformed skullstripped volume and MNI brain volume
	AlignTwoVolumesNonLinearSeveralScales(d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION, DO_OVERWRITE, INTERPOLATION_MODE, DISCARD_DISPLACEMENT_FIELD);

	// Copy the aligned T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_NonLinear, 0, NULL, NULL);

	// Cleanup

	clReleaseMemObject(d_MNI_Brain_Volume);
	clReleaseMemObject(d_MNI_T1_Volume);
	clReleaseMemObject(d_MNI_Brain_Mask);
}




void BROCCOLI_LIB::PerformRegistrationT1MNINoSkullstripWrapper()
{
	// Allocate memory for T1 volume, MNI volume and T1 volume of MNI size
	d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Brain_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to T1 volume and MNI volume
    clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);
    clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Volume , 0, NULL, NULL);

	// Interpolate T1 volume to MNI resolution and make sure it has the same size
    ChangeT1VolumeResolutionAndSize(d_MNI_T1_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, INTERPOLATION_MODE, SKULL_STRIPPED);

	clReleaseMemObject(d_T1_Volume);

	// Copy the MNI T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Interpolated_T1_Volume, 0, NULL, NULL);

	// Do Linear registration between T1 and MNI with several scales (without skull)
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_T1_MNI_Out, h_Rotations, d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);

	// Copy the aligned T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_Linear, 0, NULL, NULL);


	// Perform non-Linear registration between tramsformed skullstripped volume and MNI brain volume
	//if (NUMBER_OF_ITERATIONS_FOR_NONLinear_IMAGE_REGISTRATION > 0)
	//{
		AlignTwoVolumesNonLinearSeveralScales(d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION, DO_OVERWRITE, INTERPOLATION_MODE, DISCARD_DISPLACEMENT_FIELD);
	//}

	// Copy the aligned T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_NonLinear, 0, NULL, NULL);

	// Cleanup
	clReleaseMemObject(d_MNI_Brain_Volume);
	clReleaseMemObject(d_MNI_T1_Volume);
}





// Old version
/*
void BROCCOLI_LIB::PerformFirstLevelAnalysisBayesianWrapper()
{
	//------------------------

	// Allocate memory on device
	d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Brain_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Skullstripped_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

	PerformRegistrationT1MNINoSkullstrip();
	//PerformRegistrationT1MNI();

	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_NonLinear, 0, NULL, NULL);

	h_Registration_Parameters_T1_MNI_Out[0] = h_Registration_Parameters_T1_MNI[0];
	h_Registration_Parameters_T1_MNI_Out[1] = h_Registration_Parameters_T1_MNI[1];
	h_Registration_Parameters_T1_MNI_Out[2] = h_Registration_Parameters_T1_MNI[2];
	h_Registration_Parameters_T1_MNI_Out[3] = h_Registration_Parameters_T1_MNI[3];
	h_Registration_Parameters_T1_MNI_Out[4] = h_Registration_Parameters_T1_MNI[4];
	h_Registration_Parameters_T1_MNI_Out[5] = h_Registration_Parameters_T1_MNI[5];
	h_Registration_Parameters_T1_MNI_Out[6] = h_Registration_Parameters_T1_MNI[6];
	h_Registration_Parameters_T1_MNI_Out[7] = h_Registration_Parameters_T1_MNI[7];
	h_Registration_Parameters_T1_MNI_Out[8] = h_Registration_Parameters_T1_MNI[8];
	h_Registration_Parameters_T1_MNI_Out[9] = h_Registration_Parameters_T1_MNI[9];
	h_Registration_Parameters_T1_MNI_Out[10] = h_Registration_Parameters_T1_MNI[10];
	h_Registration_Parameters_T1_MNI_Out[11] = h_Registration_Parameters_T1_MNI[11];

	// Cleanup
	clReleaseMemObject(d_T1_Volume);
	clReleaseMemObject(d_MNI_Volume);
	clReleaseMemObject(d_MNI_Brain_Volume);
	//clReleaseMemObject(d_MNI_Brain_Mask);



	//------------------------

	// Allocate memory on device
	d_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_T1_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_EPI_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

	PerformRegistrationEPIT1();

	TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_T1_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	TransformVolumesNonLinear(d_T1_EPI_Volume, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_EPI_Volume_MNI, 0, NULL, NULL);

	h_Registration_Parameters_EPI_T1_Out[0] = h_Registration_Parameters_EPI_T1[0];
	h_Registration_Parameters_EPI_T1_Out[1] = h_Registration_Parameters_EPI_T1[1];
	h_Registration_Parameters_EPI_T1_Out[2] = h_Registration_Parameters_EPI_T1[2];
	h_Registration_Parameters_EPI_T1_Out[3] = h_Registration_Parameters_EPI_T1[3];
	h_Registration_Parameters_EPI_T1_Out[4] = h_Registration_Parameters_EPI_T1[4];
	h_Registration_Parameters_EPI_T1_Out[5] = h_Registration_Parameters_EPI_T1[5];

	clReleaseMemObject(d_MNI_T1_Volume);
	clReleaseMemObject(d_Skullstripped_T1_Volume);
	clReleaseMemObject(d_EPI_Volume);
	clReleaseMemObject(d_T1_EPI_Volume);


	//------------------------


	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Motion_Corrected_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

	PerformMotionCorrection();

	for (int t = 0; t < EPI_DATA_T; t++)
	{
		for (int p = 0; p < 6; p++)
		{
			h_Motion_Parameters_Out[t + p * EPI_DATA_T] = h_Motion_Parameters[t + p * EPI_DATA_T];
		}
	}

	clEnqueueReadBuffer(commandQueue, d_Motion_Corrected_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Motion_Corrected_fMRI_Volumes, 0, NULL, NULL);



	//------------------------

	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Smoothed_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	SegmentEPIData();

	//-------------------------------

	d_Smoothed_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, EPI_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	//PerformSmoothing(d_Smoothed_fMRI_Volumes, d_Motion_Corrected_fMRI_Volumes, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	PerformSmoothing(d_Smoothed_EPI_Mask, d_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
	PerformSmoothingNormalized(d_Smoothed_fMRI_Volumes, d_Motion_Corrected_fMRI_Volumes, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	clEnqueueReadBuffer(commandQueue, d_Smoothed_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Smoothed_fMRI_Volumes, 0, NULL, NULL);


	//-------------------------------

	//NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS*2 + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS;
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + NUMBER_OF_CONFOUND_REGRESSORS*REGRESS_CONFOUNDS;

	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Residuals = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device

	//clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM_In, 0, NULL, NULL);
	//clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM_In, 0, NULL, NULL);
	//clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts_In, 0, NULL, NULL);
	//clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In, 0, NULL, NULL);

	d_AR1_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR2_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR3_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR4_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Whitened_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	//SetMemory(d_EPI_Mask, 1.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);


	h_X_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
	h_xtxxt_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
	h_Contrasts = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float));
	h_ctxtxc_GLM = (float*)malloc(NUMBER_OF_CONTRASTS * sizeof(float));

	h_X_GLM_With_Temporal_Derivatives = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * 2 * EPI_DATA_T * sizeof(float));
	h_X_GLM_Convolved = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * (USE_TEMPORAL_DERIVATIVES+1) * EPI_DATA_T * sizeof(float));


	SetupTTestFirstLevel(EPI_DATA_T);

	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM , 0, NULL, NULL);

	for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
	{
		for (int t = 0; t < EPI_DATA_T; t++)
		{
			h_X_GLM_Out[t + r * EPI_DATA_T] = h_X_GLM[t + r * EPI_DATA_T];
			h_xtxxt_GLM_Out[t + r * EPI_DATA_T] = h_xtxxt_GLM[t + r * EPI_DATA_T];
		}
	}



	CalculateStatisticalMapsGLMBayesianFirstLevel(d_Smoothed_fMRI_Volumes);

	free(h_X_GLM);
	free(h_xtxxt_GLM);
	free(h_Contrasts);
	free(h_ctxtxc_GLM);
	free(h_X_GLM_With_Temporal_Derivatives);
	free(h_X_GLM_Convolved);

	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);

	clReleaseMemObject(d_AR1_Estimates);
	clReleaseMemObject(d_AR2_Estimates);
	clReleaseMemObject(d_AR3_Estimates);
	clReleaseMemObject(d_AR4_Estimates);
	clReleaseMemObject(d_Whitened_fMRI_Volumes);

	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_MNI,h_Registration_Parameters_T1_MNI,h_Registration_Parameters_EPI_T1_Affine);

	h_Registration_Parameters_EPI_MNI_Out[0] = h_Registration_Parameters_EPI_MNI[0];
	h_Registration_Parameters_EPI_MNI_Out[1] = h_Registration_Parameters_EPI_MNI[1];
	h_Registration_Parameters_EPI_MNI_Out[2] = h_Registration_Parameters_EPI_MNI[2];
	h_Registration_Parameters_EPI_MNI_Out[3] = h_Registration_Parameters_EPI_MNI[3];
	h_Registration_Parameters_EPI_MNI_Out[4] = h_Registration_Parameters_EPI_MNI[4];
	h_Registration_Parameters_EPI_MNI_Out[5] = h_Registration_Parameters_EPI_MNI[5];
	h_Registration_Parameters_EPI_MNI_Out[6] = h_Registration_Parameters_EPI_MNI[6];
	h_Registration_Parameters_EPI_MNI_Out[7] = h_Registration_Parameters_EPI_MNI[7];
	h_Registration_Parameters_EPI_MNI_Out[8] = h_Registration_Parameters_EPI_MNI[8];
	h_Registration_Parameters_EPI_MNI_Out[9] = h_Registration_Parameters_EPI_MNI[9];
	h_Registration_Parameters_EPI_MNI_Out[10] = h_Registration_Parameters_EPI_MNI[10];
	h_Registration_Parameters_EPI_MNI_Out[11] = h_Registration_Parameters_EPI_MNI[11];

	// Transform results to MNI space and copy to host
	if (BETA_SPACE == MNI)
	{
		// Allocate memory on device
		d_Beta_Volumes_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, &createBufferErrorBetaVolumesMNI);
		d_Statistical_Maps_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, &createBufferErrorStatisticalMapsMNI);
		d_Residual_Variances_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, &createBufferErrorResidualVariancesMNI);

		clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

		TransformFirstLevelResultsToMNI();

		clEnqueueReadBuffer(commandQueue, d_Beta_Volumes_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_MNI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Statistical_Maps_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Residual_Variances_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);

		clReleaseMemObject(d_Beta_Volumes_MNI);
		clReleaseMemObject(d_Statistical_Maps_MNI);
		clReleaseMemObject(d_Residual_Variances_MNI);
	}
	// Copy data to host
	else if (BETA_SPACE == EPI)
	{
		clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_EPI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_EPI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);

		clEnqueueReadBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask, 0, NULL, NULL);

		//Clusterize(h_Cluster_Indices, MAX_CLUSTER_SIZE, MAX_CLUSTER_MASS, NUMBER_OF_CLUSTERS, h_Statistical_Maps, 2.0f, h_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, CALCULATE_VOXEL_LABELS, CALCULATE_CLUSTER_MASS);
	}

	clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Residuals, 0, NULL, NULL);

	clReleaseMemObject(d_MNI_Brain_Mask);

	clReleaseMemObject(d_Total_Displacement_Field_X);
	clReleaseMemObject(d_Total_Displacement_Field_Y);
	clReleaseMemObject(d_Total_Displacement_Field_Z);


	//free(h_Motion_Parameters);
	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Motion_Corrected_fMRI_Volumes);
	clReleaseMemObject(d_Smoothed_fMRI_Volumes);

	clReleaseMemObject(d_EPI_Mask);
	clReleaseMemObject(d_Smoothed_EPI_Mask);

	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);

	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);

}
*/

// Transforms results from EPI space to MNI space
// old version
/*
void BROCCOLI_LIB::TransformFirstLevelResultsToMNI()
{
	ChangeVolumesResolutionAndSize(d_Beta_Volumes_MNI, d_Beta_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_TOTAL_GLM_REGRESSORS, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	TransformVolumesLinear(d_Beta_Volumes_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_TOTAL_GLM_REGRESSORS, INTERPOLATION_MODE);
	TransformVolumesNonLinear(d_Beta_Volumes_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_TOTAL_GLM_REGRESSORS, INTERPOLATION_MODE);

	ChangeVolumesResolutionAndSize(d_Contrast_Volumes_MNI, d_Contrast_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	TransformVolumesLinear(d_Contrast_Volumes_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);
	TransformVolumesNonLinear(d_Contrast_Volumes_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);

	ChangeVolumesResolutionAndSize(d_Statistical_Maps_MNI, d_Statistical_Maps, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	TransformVolumesLinear(d_Statistical_Maps_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);
	TransformVolumesNonLinear(d_Statistical_Maps_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);

	ChangeVolumesResolutionAndSize(d_Residual_Variances_MNI, d_Residual_Variances, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	TransformVolumesLinear(d_Residual_Variances_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	TransformVolumesNonLinear(d_Residual_Variances_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

	MultiplyVolumes(d_Beta_Volumes_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_GLM_REGRESSORS);
	MultiplyVolumes(d_Contrast_Volumes_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS);
	MultiplyVolumes(d_Statistical_Maps_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS);
	MultiplyVolumes(d_Residual_Variances_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1);

	if (WRITE_AR_ESTIMATES_MNI)
	{
		ChangeVolumesResolutionAndSize(d_AR1_Estimates_MNI, d_AR1_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
		TransformVolumesLinear(d_AR1_Estimates_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesNonLinear(d_AR1_Estimates_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

		ChangeVolumesResolutionAndSize(d_AR2_Estimates_MNI, d_AR2_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
		TransformVolumesLinear(d_AR2_Estimates_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesNonLinear(d_AR2_Estimates_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

		ChangeVolumesResolutionAndSize(d_AR3_Estimates_MNI, d_AR3_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
		TransformVolumesLinear(d_AR3_Estimates_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesNonLinear(d_AR3_Estimates_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

		ChangeVolumesResolutionAndSize(d_AR4_Estimates_MNI, d_AR4_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
		TransformVolumesLinear(d_AR4_Estimates_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesNonLinear(d_AR4_Estimates_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	}
}
*/




// Transforms results from EPI space to T1 space
// old version, using a lot of memory
/*
void BROCCOLI_LIB::TransformFirstLevelResultsToT1()
{
	ChangeVolumesResolutionAndSize(d_Beta_Volumes_T1, d_Beta_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_TOTAL_GLM_REGRESSORS, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);
	TransformVolumesLinear(d_Beta_Volumes_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, NUMBER_OF_TOTAL_GLM_REGRESSORS, INTERPOLATION_MODE);

	ChangeVolumesResolutionAndSize(d_Contrast_Volumes_T1, d_Contrast_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);
	TransformVolumesLinear(d_Contrast_Volumes_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);

	ChangeVolumesResolutionAndSize(d_Statistical_Maps_T1, d_Statistical_Maps, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);
	TransformVolumesLinear(d_Statistical_Maps_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);

	//ChangeVolumesResolutionAndSize(d_Residual_Variances_T1, d_Residual_Variances, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	//TransformVolumesLinear(d_Residual_Variances_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);

	if (WRITE_AR_ESTIMATES_T1)
	{
		ChangeVolumesResolutionAndSize(d_AR1_Estimates_T1, d_AR1_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);
		TransformVolumesLinear(d_AR1_Estimates_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);

		ChangeVolumesResolutionAndSize(d_AR2_Estimates_T1, d_AR2_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);
		TransformVolumesLinear(d_AR2_Estimates_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);

		ChangeVolumesResolutionAndSize(d_AR3_Estimates_T1, d_AR3_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);
		TransformVolumesLinear(d_AR3_Estimates_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);

		ChangeVolumesResolutionAndSize(d_AR4_Estimates_T1, d_AR4_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);
		TransformVolumesLinear(d_AR4_Estimates_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	}
}
*/



/* Old version  using a lot of memory
void BROCCOLI_LIB::PerformMotionCorrectionWrapper()
{
	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Motion_Corrected_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Copy volumes to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

	clFinish(commandQueue);

	// Setup all parameters and allocate memory on device
	AlignTwoVolumesLinearSetup(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Set the first volume as the reference volume
	clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Reference_Volume, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

	// Copy the first volume to the corrected volumes
	clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Motion_Corrected_fMRI_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

	// Translations
	h_Motion_Parameters_Out[0 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters_Out[1 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters_Out[2 * EPI_DATA_T] = 0.0f;

	// Rotations
	h_Motion_Parameters_Out[3 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters_Out[4 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters_Out[5 * EPI_DATA_T] = 0.0f;

	// Run the registration for each volume
	for (int t = 1; t < EPI_DATA_T; t++)
	{
		// Set a new volume to be aligned
		clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Aligned_Volume, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

		// Also copy the same volume to an image to interpolate from
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {EPI_DATA_W, EPI_DATA_H, EPI_DATA_D};
		clEnqueueCopyBufferToImage(commandQueue, d_fMRI_Volumes, d_Original_Volume, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), origin, region, 0, NULL, NULL);

		// Do rigid registration with only one scale
		AlignTwoVolumesLinear(h_Registration_Parameters_Motion_Correction, h_Rotations, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION, RIGID, INTERPOLATION_MODE);

		// Copy the corrected volume to the corrected volumes
		clEnqueueCopyBuffer(commandQueue, d_Aligned_Volume, d_Motion_Corrected_fMRI_Volumes, 0, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

		// Write the total parameter vector to host

		// Translations
		h_Motion_Parameters_Out[t + 0 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[0] * EPI_VOXEL_SIZE_X;
		h_Motion_Parameters_Out[t + 1 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[1] * EPI_VOXEL_SIZE_Y;
		h_Motion_Parameters_Out[t + 2 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[2] * EPI_VOXEL_SIZE_Z;

		// Rotations
		h_Motion_Parameters_Out[t + 3 * EPI_DATA_T] = h_Rotations[0];
		h_Motion_Parameters_Out[t + 4 * EPI_DATA_T] = h_Rotations[1];
		h_Motion_Parameters_Out[t + 5 * EPI_DATA_T] = h_Rotations[2];
	}

	// Copy all corrected volumes to host
	clEnqueueReadBuffer(commandQueue, d_Motion_Corrected_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Motion_Corrected_fMRI_Volumes, 0, NULL, NULL);

	// Cleanup allocated memory
	AlignTwoVolumesLinearCleanup();

	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Motion_Corrected_fMRI_Volumes);
}
*/





void BROCCOLI_LIB::PerformMotionCorrectionWrapperSeveralScales()
{
	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Motion_Corrected_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Copy volumes to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);
	clFinish(commandQueue);

	cl_mem d_Current_fMRI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);
	cl_mem d_Current_Reference_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);

	// Set the first volume as the reference volume
	clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Current_Reference_Volume, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

	// Copy the first volume to the corrected volumes
	clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Motion_Corrected_fMRI_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

	// Translations
	h_Motion_Parameters_Out[0 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters_Out[1 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters_Out[2 * EPI_DATA_T] = 0.0f;

	// Rotations
	h_Motion_Parameters_Out[3 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters_Out[4 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters_Out[5 * EPI_DATA_T] = 0.0f;

	// Run the registration for each volume
	for (int t = 1; t < EPI_DATA_T; t++)
	{
		// Set a new volume to be aligned
		clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Current_fMRI_Volume, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

		// Align to reference volume using 2 scales
		AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_Motion_Correction, h_Rotations, d_Current_fMRI_Volume, d_Current_Reference_Volume, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);

		// Copy the corrected volume to the corrected volumes
		clEnqueueCopyBuffer(commandQueue, d_Current_fMRI_Volume, d_Motion_Corrected_fMRI_Volumes, 0, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

		// Write the total parameter vector to host

		// Translations
		h_Motion_Parameters_Out[t + 0 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[0]; // * EPI_VOXEL_SIZE_X;
		h_Motion_Parameters_Out[t + 1 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[1]; // * EPI_VOXEL_SIZE_Y;
		h_Motion_Parameters_Out[t + 2 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[2]; // * EPI_VOXEL_SIZE_Z;

		// Rotations
		h_Motion_Parameters_Out[t + 3 * EPI_DATA_T] = h_Rotations[0];
		h_Motion_Parameters_Out[t + 4 * EPI_DATA_T] = h_Rotations[1];
		h_Motion_Parameters_Out[t + 5 * EPI_DATA_T] = h_Rotations[2];
	}

	// Copy all corrected volumes to host
	clEnqueueReadBuffer(commandQueue, d_Motion_Corrected_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Motion_Corrected_fMRI_Volumes, 0, NULL, NULL);

	// Cleanup allocated memory
	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Motion_Corrected_fMRI_Volumes);
	clReleaseMemObject(d_Current_fMRI_Volume);
	clReleaseMemObject(d_Current_Reference_Volume);
}






// Parallel version of clustering, using texture memory
void BROCCOLI_LIB::ClusterizeOpenCLWrapper()
{
	/*
	cl_mem d_Mask = clCreateBuffer(context, CL_MEM_READ_ONLY, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Data = clCreateBuffer(context, CL_MEM_READ_ONLY, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Cluster_Indices = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_First_Level_Results, 0, NULL, NULL);


	SetGlobalAndLocalWorkSizesClusterize(MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	// Create a 3D image (texture) for fast access of neighbouring indices
	cl_image_format format;
	format.image_channel_data_type = CL_SIGNED_INT32;
	format.image_channel_order = CL_INTENSITY;
	cl_mem d_Cluster_Indices_Texture = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 0, 0, NULL, NULL);

	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {MNI_DATA_W, MNI_DATA_H, MNI_DATA_D};

	cl_mem d_Updated = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
	cl_mem d_Current_Cluster = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

	SetMemoryInt(d_Current_Cluster, 0, 1);

	for (int i = 0; i < 100; i++)
	{
	clSetKernelArg(SetStartClusterIndicesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(SetStartClusterIndicesKernel, 1, sizeof(cl_mem), &d_Data);
	clSetKernelArg(SetStartClusterIndicesKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(SetStartClusterIndicesKernel, 3, sizeof(cl_mem), &d_Current_Cluster);
	clSetKernelArg(SetStartClusterIndicesKernel, 4, sizeof(float), &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(SetStartClusterIndicesKernel, 5, sizeof(int), &MNI_DATA_W);
	clSetKernelArg(SetStartClusterIndicesKernel, 6, sizeof(int), &MNI_DATA_H);
	clSetKernelArg(SetStartClusterIndicesKernel, 7, sizeof(int), &MNI_DATA_D);
	runKernelErrorClusterizeScan = clEnqueueNDRangeKernel(commandQueue, SetStartClusterIndicesKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
	clFinish(commandQueue);

	clSetKernelArg(ClusterizeKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeKernel, 1, sizeof(cl_mem), &d_Cluster_Indices_Texture);
	clSetKernelArg(ClusterizeKernel, 2, sizeof(cl_mem), &d_Updated);
	clSetKernelArg(ClusterizeKernel, 3, sizeof(cl_mem), &d_Data);
	clSetKernelArg(ClusterizeKernel, 4, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(ClusterizeKernel, 5, sizeof(float), &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(ClusterizeKernel, 6, sizeof(int), &MNI_DATA_W);
	clSetKernelArg(ClusterizeKernel, 7, sizeof(int), &MNI_DATA_H);
	clSetKernelArg(ClusterizeKernel, 8, sizeof(int), &MNI_DATA_D);


	float UPDATED = 1.0f;
	while (UPDATED == 1.0f)
	{
		// Copy the current cluster indices to a texture, for fast spatial access
		clEnqueueCopyBufferToImage(commandQueue, d_Cluster_Indices, d_Cluster_Indices_Texture, 0, origin, region, 0, NULL, NULL);
		// Set updated to 0
		SetMemory(d_Updated, 0.0f, 1);
		// Run the clustering
		runKernelErrorClusterize = clEnqueueNDRangeKernel(commandQueue, ClusterizeKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);
		// Copy update parameter to host
		clEnqueueReadBuffer(commandQueue, d_Updated, CL_TRUE, 0, sizeof(float), &UPDATED, 0, NULL, NULL);
	}
	}

	clEnqueueReadBuffer(commandQueue, d_Cluster_Indices, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), h_Cluster_Indices, 0, NULL, NULL);

	clReleaseMemObject(d_Cluster_Indices_Texture);
	clReleaseMemObject(d_Updated);
	clReleaseMemObject(d_Current_Cluster);

	clReleaseMemObject(d_Mask);
	clReleaseMemObject(d_Data);
	clReleaseMemObject(d_Cluster_Indices);
	*/
}


void BROCCOLI_LIB::ClusterizeOpenCLWrapper2()
{
	cl_mem d_Mask = clCreateBuffer(context, CL_MEM_READ_ONLY, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Data = clCreateBuffer(context, CL_MEM_READ_ONLY, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Cluster_Indices = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), NULL, NULL);
	d_Cluster_Sizes = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), NULL, NULL);
	d_Largest_Cluster = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_First_Level_Results, 0, NULL, NULL);

	SetGlobalAndLocalWorkSizesClusterize(MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	d_Updated = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);

	clSetKernelArg(ClusterizeScanKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeScanKernel, 1, sizeof(cl_mem), &d_Updated);
	clSetKernelArg(ClusterizeScanKernel, 2, sizeof(cl_mem), &d_Data);
	clSetKernelArg(ClusterizeScanKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(ClusterizeScanKernel, 4, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(ClusterizeScanKernel, 5, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(ClusterizeScanKernel, 6, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(ClusterizeScanKernel, 7, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(ClusterizeRelabelKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeRelabelKernel, 1, sizeof(cl_mem), &d_Data);
	clSetKernelArg(ClusterizeRelabelKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(ClusterizeRelabelKernel, 3, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(ClusterizeRelabelKernel, 4, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(ClusterizeRelabelKernel, 5, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(ClusterizeRelabelKernel, 6, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(CalculateClusterSizesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(CalculateClusterSizesKernel, 1, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateClusterSizesKernel, 2, sizeof(cl_mem), &d_Data);
	clSetKernelArg(CalculateClusterSizesKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateClusterSizesKernel, 4, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(CalculateClusterSizesKernel, 5, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateClusterSizesKernel, 6, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateClusterSizesKernel, 7, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(CalculateLargestClusterKernel, 0, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateLargestClusterKernel, 1, sizeof(cl_mem), &d_Largest_Cluster);
	clSetKernelArg(CalculateLargestClusterKernel, 2, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateLargestClusterKernel, 3, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateLargestClusterKernel, 4, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(SetStartClusterIndicesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(SetStartClusterIndicesKernel, 1, sizeof(cl_mem), &d_Data);
	clSetKernelArg(SetStartClusterIndicesKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(SetStartClusterIndicesKernel, 3, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(SetStartClusterIndicesKernel, 4, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(SetStartClusterIndicesKernel, 5, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(SetStartClusterIndicesKernel, 6, sizeof(int),    &MNI_DATA_D);

	for (int i = 0; i < 1000; i++)
	{



	runKernelErrorClusterizeScan = clEnqueueNDRangeKernel(commandQueue, SetStartClusterIndicesKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
	clFinish(commandQueue);



	float UPDATED = 1.0f;
	while (UPDATED == 1.0f)
	{
		// Set updated to 0
		SetMemory(d_Updated, 0.0f, 1);
		// Run the clustering
		runKernelErrorClusterizeScan = clEnqueueNDRangeKernel(commandQueue, ClusterizeScanKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);
		runKernelErrorClusterizeRelabel = clEnqueueNDRangeKernel(commandQueue, ClusterizeRelabelKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);

		// Copy update parameter to host
		clEnqueueReadBuffer(commandQueue, d_Updated, CL_TRUE, 0, sizeof(float), &UPDATED, 0, NULL, NULL);
	}

	SetMemoryInt(d_Largest_Cluster, -100, 1);
	SetMemoryInt(d_Cluster_Sizes, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D);

	runKernelErrorCalculateClusterSizes = clEnqueueNDRangeKernel(commandQueue, CalculateClusterSizesKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
	clFinish(commandQueue);
	runKernelErrorCalculateLargestCluster = clEnqueueNDRangeKernel(commandQueue, CalculateLargestClusterKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
	clFinish(commandQueue);
	// Copy largest cluster to host
	clEnqueueReadBuffer(commandQueue, d_Largest_Cluster, CL_TRUE, 0, sizeof(int), h_Largest_Cluster, 0, NULL, NULL);

	}

	clEnqueueReadBuffer(commandQueue, d_Cluster_Indices, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), h_Cluster_Indices, 0, NULL, NULL);

	clEnqueueReadBuffer(commandQueue, d_Cluster_Sizes, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), h_Cluster_Indices, 0, NULL, NULL);

	clReleaseMemObject(d_Updated);

	clReleaseMemObject(d_Mask);
	clReleaseMemObject(d_Data);
	clReleaseMemObject(d_Cluster_Indices);

	clReleaseMemObject(d_Cluster_Sizes);
	clReleaseMemObject(d_Largest_Cluster);
}

void BROCCOLI_LIB::ClusterizeOpenCLWrapper3()
{
	cl_mem d_Data = clCreateBuffer(context, CL_MEM_READ_ONLY, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	float* h_Data = (float*)malloc(MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float));

	for (int i = 0; i < 1000; i++)
	{
		clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Data, 0, NULL, NULL);
		clFinish(commandQueue);
		//Clusterize(h_Cluster_Indices, MAX_CLUSTER_SIZE, MAX_CLUSTER_MASS, NUMBER_OF_CLUSTERS, h_First_Level_Results, CLUSTER_DEFINING_THRESHOLD, h_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, CALCULATE_VOXEL_LABELS, DONT_CALCULATE_CLUSTER_MASS);
		//Clusterize(h_Cluster_Indices, MAX_CLUSTER_SIZE, MAX_CLUSTER_MASS, NUMBER_OF_CLUSTERS, h_First_Level_Results, CLUSTER_DEFINING_THRESHOLD, h_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, DONT_CALCULATE_VOXEL_LABELS, DONT_CALCULATE_CLUSTER_MASS);
	}

	clReleaseMemObject(d_Data);
	free(h_Data);
}



