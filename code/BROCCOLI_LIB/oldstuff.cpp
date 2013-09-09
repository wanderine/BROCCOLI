
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



