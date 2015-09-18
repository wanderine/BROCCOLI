/*
 * BROCCOLI: Software for Fast fMRI Analysis on Many-Core CPUs and GPUs
 * Copyright (C) <2013>  Anders Eklund, andek034@gmail.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <Dense>

#define ADD_FILENAME true
#define DONT_ADD_FILENAME true

#define CHECK_EXISTING_FILE true
#define DONT_CHECK_EXISTING_FILE false

int main(int argc, char **argv)
{
    /* Input arguments */
    FILE *fp = NULL; 
    
    // No inputs, so print help text
    if (argc == 1)
    {   
		printf("Combines two affine transformation matrices \n\n");     
        printf("Usage:\n\n");
        printf("CombineAffineTransformations matrix1.txt matrix2.txt\n\n");
        printf("\n\n");
        
        return EXIT_SUCCESS;
    }
    // Try to open files
    else if (argc > 1)
    {        
        fp = fopen(argv[1],"r");
        if (fp == NULL)
        {
            printf("Could not open first matrix file %s !\n",argv[1]);
            return EXIT_FAILURE;
        }
        fclose(fp);     
        
        fp = fopen(argv[2],"r");
        if (fp == NULL)
        {
            printf("Could not open second matrix file %s !\n",argv[2]);
            return EXIT_FAILURE;
        }
        fclose(fp);   
    }
    
	float h_Affine_Parameters1[16];
	float h_Affine_Parameters2[16]; 
   	float h_Total_Parameters[16]; 

	std::vector<int> indices;
	indices.push_back(3);
	indices.push_back(4);
	indices.push_back(5);
	indices.push_back(0);
	indices.push_back(6);
	indices.push_back(7);
	indices.push_back(8);
	indices.push_back(1);
	indices.push_back(9);
	indices.push_back(10);
	indices.push_back(11);
	indices.push_back(2);
	indices.push_back(12);
	indices.push_back(13);
	indices.push_back(14);
	indices.push_back(15);

	std::ifstream matrix1, matrix2;

	// Read transformation matrix 1 from file

	matrix1.open(argv[1]);

    if (!matrix1.good())
    {
        matrix1.close();
        printf("Unable to open first affine matrix file %s. Aborting! \n",argv[1]);
        return EXIT_FAILURE;
    }

	for (size_t p = 0; p < 16; p++)
	{
		if (! (matrix1 >> h_Affine_Parameters1[indices[p]]) )
		{
			matrix1.close();
	        printf("Could not read all values of the first affine matrix file %s, aborting! Stopped reading at value %zu. \n",argv[1],p);      
	        return EXIT_FAILURE;
		}
	}
	matrix1.close();


	h_Affine_Parameters1[3] -= 1.0f;
	h_Affine_Parameters1[7] -= 1.0f;
	h_Affine_Parameters1[11] -= 1.0f;


	// Read transformation matrix 2 from file

	matrix2.open(argv[2]);

    if (!matrix2.good())
    {
        matrix2.close();
        printf("Unable to open first affine matrix file %s. Aborting! \n",argv[2]);
        return EXIT_FAILURE;
    }

	for (size_t p = 0; p < 16; p++)
	{
		if (! (matrix2 >> h_Affine_Parameters2[indices[p]]) )
		{
			matrix2.close();
	        printf("Could not read all values of the second affine matrix file %s, aborting! Stopped reading at value %zu. \n",argv[2],p);      
	        return EXIT_FAILURE;
		}
	}
	matrix2.close();

	h_Affine_Parameters2[3] -= 1.0f;
	h_Affine_Parameters2[7] -= 1.0f;
	h_Affine_Parameters2[11] -= 1.0f;


            
    //------------------------
    
	// Put parameters in 4 x 4 affine transformation matrix

	// (p3 p4  p5  tx)
	// (p6 p7  p8  ty)
	// (p9 p10 p11 tz)
	// (0  0   0   1 )

	Eigen::MatrixXd Affine_Matrix1(4,4), Affine_Matrix2(4,4), Total_Affine_Matrix(4,4);

	// Put values into an Eigen matrix, and convert to double
	// Add ones in the diagonal, to get a transformation matrix

	// First row
	Affine_Matrix1(0,0) = (double)(h_Affine_Parameters1[3] + 1.0f);
	Affine_Matrix1(0,1) = (double)h_Affine_Parameters1[4];
	Affine_Matrix1(0,2) = (double)h_Affine_Parameters1[5];
	Affine_Matrix1(0,3) = (double)h_Affine_Parameters1[0];

	// Second row
	Affine_Matrix1(1,0) = (double)h_Affine_Parameters1[6];
	Affine_Matrix1(1,1) = (double)(h_Affine_Parameters1[7] + 1.0f);
	Affine_Matrix1(1,2) = (double)h_Affine_Parameters1[8];
	Affine_Matrix1(1,3) = (double)h_Affine_Parameters1[1];

	// Third row
	Affine_Matrix1(2,0)  = (double)h_Affine_Parameters1[9];
	Affine_Matrix1(2,1)  = (double)h_Affine_Parameters1[10];
	Affine_Matrix1(2,2) = (double)(h_Affine_Parameters1[11] + 1.0f);
	Affine_Matrix1(2,3) = (double)h_Affine_Parameters1[2];

	// Fourth row
	Affine_Matrix1(3,0) = 0.0;
	Affine_Matrix1(3,1) = 0.0;
	Affine_Matrix1(3,2) = 0.0;
	Affine_Matrix1(3,3) = 1.0;

	// Put values into an Eigen matrix, and convert to double
	// Add ones in the diagonal, to get a transformation matrix
	// First row
	Affine_Matrix2(0,0) = (double)(h_Affine_Parameters2[3] + 1.0f);
	Affine_Matrix2(0,1) = (double)h_Affine_Parameters2[4];
	Affine_Matrix2(0,2) = (double)h_Affine_Parameters2[5];
	Affine_Matrix2(0,3) = (double)h_Affine_Parameters2[0];

	// Second row
	Affine_Matrix2(1,0) = (double)h_Affine_Parameters2[6];
	Affine_Matrix2(1,1) = (double)(h_Affine_Parameters2[7] + 1.0f);
	Affine_Matrix2(1,2) = (double)h_Affine_Parameters2[8];
	Affine_Matrix2(1,3) = (double)h_Affine_Parameters2[1];

	// Third row
	Affine_Matrix2(2,0)  = (double)h_Affine_Parameters2[9];
	Affine_Matrix2(2,1)  = (double)h_Affine_Parameters2[10];
	Affine_Matrix2(2,2) = (double)(h_Affine_Parameters2[11] + 1.0f);
	Affine_Matrix2(2,3) = (double)h_Affine_Parameters2[2];

	// Fourth row
	Affine_Matrix2(3,0) = 0.0;
	Affine_Matrix2(3,1) = 0.0;
	Affine_Matrix2(3,2) = 0.0;
	Affine_Matrix2(3,3) = 1.0;

	// Multiply the two matrices
	Total_Affine_Matrix = Affine_Matrix1 * Affine_Matrix2;

	// Put values back into array
	// Subtract ones in the diagonal

	// Translation parameters
	h_Total_Parameters[0] = (float)Total_Affine_Matrix(0,3);
	h_Total_Parameters[1] = (float)Total_Affine_Matrix(1,3);
	h_Total_Parameters[2] = (float)Total_Affine_Matrix(2,3);

	// First row
	h_Total_Parameters[3] = (float)(Total_Affine_Matrix(0,0) - 1.0);
	h_Total_Parameters[4] = (float)Total_Affine_Matrix(0,1);
	h_Total_Parameters[5] = (float)Total_Affine_Matrix(0,2);

	// Second row
	h_Total_Parameters[6] = (float)Total_Affine_Matrix(1,0);
	h_Total_Parameters[7] = (float)(Total_Affine_Matrix(1,1) - 1.0);
	h_Total_Parameters[8] = (float)Total_Affine_Matrix(1,2);

	// Third row
	h_Total_Parameters[9] = (float)Total_Affine_Matrix(2,0);
	h_Total_Parameters[10] = (float)Total_Affine_Matrix(2,1);
	h_Total_Parameters[11] = (float)(Total_Affine_Matrix(2,2) - 1.0);

	//------------
    
	std::ofstream totalMatrix;
    totalMatrix.open("total_matrix.txt");      
    if ( totalMatrix.good() )
    {
        totalMatrix.precision(6);

        totalMatrix << h_Total_Parameters[3]+1.0f << std::setw(2) << " " << h_Total_Parameters[4] << std::setw(2) << " " << h_Total_Parameters[5] << std::setw(2) << " " << h_Total_Parameters[0] << std::endl;
        totalMatrix << h_Total_Parameters[6] << std::setw(2) << " " << h_Total_Parameters[7] + 1.0f << std::setw(2) << " " << h_Total_Parameters[8] << std::setw(2) << " " << h_Total_Parameters[1] << std::endl;
	    totalMatrix << h_Total_Parameters[9] << std::setw(2) << " " << h_Total_Parameters[10] << std::setw(2) << " " << h_Total_Parameters[11] + 1.0f << std::setw(2) << " " << h_Total_Parameters[2] << std::endl;
	    totalMatrix << 0.0f << std::setw(2) << " " << 0.0f << std::setw(2) << " " << 0.0f << std::setw(2) << " " << 1.0f << std::endl;

        totalMatrix.close();
    }
    else
    {
        printf("Could not open total matrix for writing!\n");
    }
    
	return EXIT_SUCCESS;
}



