#include "mex.h"

void Get_Parameter_Indices(int* i, int* j, int parameter)
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
			*i = 1; *j = 1;
			break;

		case 5:
			*i = 6; *j = 1;
			break;

		case 6:
			*i = 7; *j = 1;
			break;

		case 7:
			*i = 8; *j = 1;
			break;

		case 8:
			*i = 2; *j = 2;
			break;

		case 9:
			*i = 9; *j = 2;
			break;

		case 10:
			*i = 10; *j = 2;
			break;

		case 11:
			*i = 11; *j = 2;
			break;

		case 12:
			*i = 3; *j = 3;
			break;

		case 13:
			*i = 4; *j = 3;
			break;

		case 14:
			*i = 5; *j = 3;
			break;

		case 15:
			*i = 4; *j = 4;
			break;

		case 16:
			*i = 5; *j = 4;
			break;

		case 17:
			*i = 5; *j = 5;
			break;

		case 18:
			*i = 6; *j = 6;
			break;

		case 19:
			*i = 7; *j = 6;
			break;

		case 20:
			*i = 8; *j = 6;
			break;

		case 21:
			*i = 7; *j = 7;
			break;

		case 22:
			*i = 8; *j = 7;
			break;

		case 23:
			*i = 8; *j = 8;
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



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Check the number of input and output arguments. */
    if(nrhs<9)
	mexErrMsgTxt("Too few input arguments.");
    if(nrhs>9)
	mexErrMsgTxt("Too many input arguments.");
    if(nlhs<2)
	mexErrMsgTxt("Too few output arguments.");
    if(nlhs>2)
	mexErrMsgTxt("Too many output arguments.");
       
    double *certainties_1 = (double*)mxGetPr(prhs[0]);
    double *certainties_2 = (double*)mxGetPr(prhs[1]);
    double *certainties_3 = (double*)mxGetPr(prhs[2]);
    
    double *phase_differences_1 = (double*)mxGetPr(prhs[3]);
    double *phase_differences_2 = (double*)mxGetPr(prhs[4]);
    double *phase_differences_3 = (double*)mxGetPr(prhs[5]);
    
    double *phase_gradients_1 = (double*)mxGetPr(prhs[6]);
    double *phase_gradients_2 = (double*)mxGetPr(prhs[7]);
    double *phase_gradients_3 = (double*)mxGetPr(prhs[8]);
    
    
    int NUMBER_OF_DIMENSIONS = mxGetNumberOfDimensions(prhs[0]);
	const int *ARRAY_DIMENSIONS = mxGetDimensions(prhs[0]);
	const int DATA_H = ARRAY_DIMENSIONS[0];
	const int DATA_W = ARRAY_DIMENSIONS[1];
	const int DATA_D = ARRAY_DIMENSIONS[2];
	                
    /*printf("Volume is size %i x %i x %i \n",DATA_W, DATA_H, DATA_D);*/
    
    int NUMBER_OF_VOXELS = DATA_W * DATA_H * DATA_D;
    int NUMBER_OF_PARAMETERS = 12;
    
    /* Output arguments */
    double *A, *h;
    
    /* Create the A matrix */
    NUMBER_OF_DIMENSIONS = 2;
    int A_DIMENSIONS[2];    
    A_DIMENSIONS[0] = 12;
    A_DIMENSIONS[1] = 12;
    plhs[0] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS, A_DIMENSIONS, mxDOUBLE_CLASS, mxREAL);
    A = (double*)mxGetPr(plhs[0]);
    
    /* Create the h vector */
    NUMBER_OF_DIMENSIONS = 2;
    int H_DIMENSIONS[2];
    H_DIMENSIONS[0] = 12;
    H_DIMENSIONS[1] = 1;
    plhs[1] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS, H_DIMENSIONS, mxDOUBLE_CLASS, mxREAL);
    h = (double*)mxGetPr(plhs[1]);
        
    double certainty_1, certainty_2, certainty_3;
    double phase_gradient_1, phase_gradient_2, phase_gradient_3;
    double phase_difference_1, phase_difference_2, phase_difference_3;
    
    int voxel,i,j,parameter,x,y,z;
    
    
    /* Reset A matrix */
    for (i = 0; i < NUMBER_OF_PARAMETERS; ++i)
    {
        for (j = 0; j < NUMBER_OF_PARAMETERS; ++j)
        {
            A[i*NUMBER_OF_PARAMETERS + j] = 0;
        }
    }
    
    /* Reset h vector */
    for (i = 0; i < NUMBER_OF_PARAMETERS; ++i)
    {
        h[i] = 0;
    }
    
    double A_matrix_values[30];
    double xf, yf, zf;
        
    for (i = 0; i < 30; i++)
    { 
        A_matrix_values[i] = 0;
    }
    
    /*printf("before loops \n");
    fflush(stdout);*/
    
    /* Calculate the A matrix */
    for (x = 0; x < DATA_W; x++)
    {
        for (y = 0; y < DATA_H; y++)
        {
            for (z = 0; z < DATA_D; z++)
            {
                
                /*printf("x %d y %d z %d \n",x,y,z);
                fflush(stdout);*/
                
                
                voxel = y + x * DATA_H + z * DATA_W * DATA_H;
                
                xf = (double)x - ((double)DATA_W - 1.0f)/2.0f;
                yf = (double)y - ((double)DATA_H - 1.0f)/2.0f;
                zf = (double)z - ((double)DATA_D - 1.0f)/2.0f;
        
                phase_gradient_1 = phase_gradients_1[voxel]; /* 1 2 0 */
                phase_gradient_2 = phase_gradients_2[voxel];
                phase_gradient_3 = phase_gradients_3[voxel];
                
                phase_difference_1 = phase_differences_1[voxel];
                phase_difference_2 = phase_differences_2[voxel];
                phase_difference_3 = phase_differences_3[voxel];
                
                certainty_1 = certainties_1[voxel];
                certainty_2 = certainties_2[voxel];
                certainty_3 = certainties_3[voxel];
                
                A_matrix_values[0] += certainty_1 * phase_gradient_1 * phase_gradient_1;
                A_matrix_values[1] += xf * certainty_1 * phase_gradient_1 * phase_gradient_1;
                A_matrix_values[2] += yf * certainty_1 * phase_gradient_1 * phase_gradient_1;
                A_matrix_values[3] += zf * certainty_1 * phase_gradient_1 * phase_gradient_1;
                A_matrix_values[4] += certainty_2 * phase_gradient_2 * phase_gradient_2;
                A_matrix_values[5] += xf * certainty_2 * phase_gradient_2 * phase_gradient_2;
                A_matrix_values[6] += yf * certainty_2 * phase_gradient_2 * phase_gradient_2;
                A_matrix_values[7] += zf * certainty_2 * phase_gradient_2 * phase_gradient_2;
                A_matrix_values[8] += certainty_3 * phase_gradient_3 * phase_gradient_3;
                A_matrix_values[9] += xf * certainty_3 * phase_gradient_3 * phase_gradient_3;
                A_matrix_values[10] += yf * certainty_3 * phase_gradient_3 * phase_gradient_3;
                A_matrix_values[11] += zf * certainty_3 * phase_gradient_3 * phase_gradient_3;
                A_matrix_values[12] += xf * xf * certainty_1 * phase_gradient_1 * phase_gradient_1;
                A_matrix_values[13] += xf * yf * certainty_1 * phase_gradient_1 * phase_gradient_1;
                A_matrix_values[14] += xf * zf * certainty_1 * phase_gradient_1 * phase_gradient_1;
                A_matrix_values[15] += yf * yf * certainty_1 * phase_gradient_1 * phase_gradient_1;
                A_matrix_values[16] += yf * zf * certainty_1 * phase_gradient_1 * phase_gradient_1;
                A_matrix_values[17] += zf * zf * certainty_1 * phase_gradient_1 * phase_gradient_1;
                A_matrix_values[18] += xf * xf * certainty_2 * phase_gradient_2 * phase_gradient_2;
                A_matrix_values[19] += xf * yf * certainty_2 * phase_gradient_2 * phase_gradient_2;
                A_matrix_values[20] += xf * zf * certainty_2 * phase_gradient_2 * phase_gradient_2;
                A_matrix_values[21] += yf * yf * certainty_2 * phase_gradient_2 * phase_gradient_2;
                A_matrix_values[22] += yf * zf * certainty_2 * phase_gradient_2 * phase_gradient_2;
                A_matrix_values[23] += zf * zf * certainty_2 * phase_gradient_2 * phase_gradient_2;
                A_matrix_values[24] += xf * xf * certainty_3 * phase_gradient_3 * phase_gradient_3;
                A_matrix_values[25] += xf * yf * certainty_3 * phase_gradient_3 * phase_gradient_3;
                A_matrix_values[26] += xf * zf * certainty_3 * phase_gradient_3 * phase_gradient_3;
                A_matrix_values[27] += yf * yf * certainty_3 * phase_gradient_3 * phase_gradient_3;
                A_matrix_values[28] += yf * zf * certainty_3 * phase_gradient_3 * phase_gradient_3;
                A_matrix_values[29] += zf * zf * certainty_3 * phase_gradient_3 * phase_gradient_3;
                
                
                h[0] += certainty_1 * phase_gradient_1 * phase_difference_1;
                h[1] += certainty_2 * phase_gradient_2 * phase_difference_2;
                h[2] += certainty_3 * phase_gradient_3 * phase_difference_3;
                h[3] += xf * certainty_1 * phase_gradient_1 * phase_difference_1;
                h[4] += yf * certainty_1 * phase_gradient_1 * phase_difference_1;
                h[5] += zf * certainty_1 * phase_gradient_1 * phase_difference_1;
                h[6] += xf * certainty_2 * phase_gradient_2 * phase_difference_2;
                h[7] += yf * certainty_2 * phase_gradient_2 * phase_difference_2;
                h[8] += zf * certainty_2 * phase_gradient_2 * phase_difference_2;
                h[9] += xf * certainty_3 * phase_gradient_3 * phase_difference_3;
                h[10] += yf * certainty_3 * phase_gradient_3 * phase_difference_3;
                h[11] += zf * certainty_3 * phase_gradient_3 * phase_difference_3;
            }
        }
    }
    
    /*printf("after loops \n");*/
    
    for (parameter = 0; parameter < 30; parameter++)
    {
        Get_Parameter_Indices(&i,&j,parameter);
        A[i + j * 12] = A_matrix_values[parameter];
    }
    
    /* Mirror the matrix values */
	for (j = 0; j < NUMBER_OF_PARAMETERS; j++)
	{
		for (i = 0; i < NUMBER_OF_PARAMETERS; i++)
		{
			A[j + i*NUMBER_OF_PARAMETERS] = A[i + j*NUMBER_OF_PARAMETERS];
		}
	}
}
