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

#include "broccoli_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include "nifti1_io.h"
#include <iostream>
#include <fstream>
#include <iomanip>

#include <limits.h>
#include <unistd.h>

#include "HelpFunctions.cpp"

#define ADD_FILENAME true
#define DONT_ADD_FILENAME true

#define CHECK_EXISTING_FILE true
#define DONT_CHECK_EXISTING_FILE false



int main(int argc, char ** argv)
{
    //-----------------------
    // Input pointers
    
    float           *h_Volumes = NULL;
    float           *h_Mask = NULL;

	//--------------

    void*           allMemoryPointers[500];
	for (int i = 0; i < 500; i++)
	{
		allMemoryPointers[i] = NULL;
	}
    
	nifti_image*	allNiftiImages[500];
	for (int i = 0; i < 500; i++)
	{
		allNiftiImages[i] = NULL;
	}

    int             numberOfMemoryPointers = 0;
	int				numberOfNiftiImages = 0;

	size_t			allocatedHostMemory = 0;

	//--------------
  
    // Default parameters
    bool            PRINT = true;
	bool			VERBOS = false;
    
    size_t          DATA_W, DATA_H, DATA_D, DATA_T;
    float           VOXEL_SIZE_X, VOXEL_SIZE_Y, VOXEL_SIZE_Z;

	bool			CHANGE_OUTPUT_FILENAME = false;

    //-----------------------
    // Output parameters
    
    const char      *outputFilename;
       
    //---------------------
    
    /* Input arguments */
    FILE *fp = NULL; 
    
    // No inputs, so print help text
    if (argc == 1)
    {        
        printf("Usage:\n\n");
        printf("ExtractTimeseries input.nii mask.nii [options]\n\n");
        printf("Options:\n\n");
        printf(" -output      Set filename of text file  \n");
        printf("\n\n");
        
        return EXIT_SUCCESS;
    }
    // Try to open files
    else if (argc > 1)
    {        
        fp = fopen(argv[1],"r");
        if (fp == NULL)
        {            
            printf("Could not open file %s !\n",argv[1]);
            return EXIT_FAILURE;
        }
        fclose(fp);        

        fp = fopen(argv[2],"r");
        if (fp == NULL)
        {            
            printf("Could not open file %s !\n",argv[2]);
            return EXIT_FAILURE;
        }
        fclose(fp);        

    }
    
    // Loop over additional inputs
    int i = 3;
    while (i < argc)
    {
        char *input = argv[i];
        char *p;
        if (strcmp(input,"-output") == 0)
        {
			CHANGE_OUTPUT_FILENAME = true;

			if ( (i+1) >= argc  )
			{
			    printf("Unable to read name after -output !\n");
                return EXIT_FAILURE;
			}

            outputFilename = argv[i+1];
            i += 2;
        }
        else
        {
            printf("Unrecognized option! %s \n",argv[i]);
            return EXIT_FAILURE;
        }                
    }
    
    double startTime = GetWallTime();

	// ---------------------
    // Read data
	// ---------------------
    nifti_image *inputData = nifti_image_read(argv[1],1);
    
    if (inputData == NULL)
    {
        printf("Could not open nifti file!\n");
        return EXIT_FAILURE;
    }
    allNiftiImages[numberOfNiftiImages] = inputData;
	numberOfNiftiImages++;


    nifti_image *inputMask = nifti_image_read(argv[2],1);
    
    if (inputMask == NULL)
    {
        printf("Could not open nifti file!\n");
        return EXIT_FAILURE;
    }
    allNiftiImages[numberOfNiftiImages] = inputMask;
	numberOfNiftiImages++;

	double endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to read the nifti file\n",(float)(endTime - startTime));
	}

    // Get data dimensions
    DATA_W = inputData->nx;
    DATA_H = inputData->ny;
    DATA_D = inputData->nz;
    DATA_T = inputData->nt;
    	
    // Calculate size, in bytes
    size_t DATA_SIZE = DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float);
    size_t VOLUME_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);
    
    // Print some info
    if (PRINT)
    {
        printf("Authored by K.A. Eklund \n");
        printf("Data size: %zu x %zu x %zu x %zu \n",  DATA_W, DATA_H, DATA_D, DATA_T);
    } 
   
    // ------------------------------------------------
    
    // Allocate memory on the host
    
	startTime = GetWallTime();

	AllocateMemory(h_Volumes, DATA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "INPUT_DATA");
	AllocateMemory(h_Mask, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "MASK");

	endTime = GetWallTime();
    
	if (VERBOS)
 	{
		printf("It took %f seconds to allocate memory\n",(float)(endTime - startTime));
	}

	startTime = GetWallTime();

    // Convert data to floats
    if ( inputData->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D * DATA_T; i++)
        {
            h_Volumes[i] = (float)p[i];
        }
    }
    else if ( inputData->datatype == DT_UINT8 )
    {
        unsigned char *p = (unsigned char*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D * DATA_T; i++)
        {
            h_Volumes[i] = (float)p[i];
        }
    }
    else if ( inputData->datatype == DT_UINT16 )
    {
        unsigned short int *p = (unsigned short int*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D * DATA_T; i++)
        {
            h_Volumes[i] = (float)p[i];
        }
    }
	else if ( inputData->datatype == DT_FLOAT )
    {
        float *p = (float*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D * DATA_T; i++)
        {
            h_Volumes[i] = p[i];
        }
    }
    else
    {
        printf("Unknown data type in input data, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }




    // Convert data to floats
    if ( inputMask->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputMask->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        {
            h_Mask[i] = (float)p[i];
        }
    }
    else if ( inputMask->datatype == DT_UINT8 )
    {
        unsigned char *p = (unsigned char*)inputMask->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        {
            h_Mask[i] = (float)p[i];
        }
    }
    else if ( inputMask->datatype == DT_UINT16 )
    {
        unsigned short int *p = (unsigned short int*)inputMask->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        {
            h_Mask[i] = (float)p[i];
        }
    }
	else if ( inputMask->datatype == DT_FLOAT )
    {
        float *p = (float*)inputMask->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        {
            h_Mask[i] = p[i];
        }
    }
    else
    {
        printf("Unknown data type in input data, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }


	endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to convert data to floats\n",(float)(endTime - startTime));
	}

    //------------------------

	float* h_Timeseries = (float*)malloc(DATA_T * sizeof(float));

	for (int t = 0; t < DATA_T; t++)
	{
		h_Timeseries[t] = 0.0f;
	}

	// Calculate average time series in mask
	float NUMBER_OF_VOXELS = 0.0f;
	for (int x = 0; x < DATA_W; x++)
	{
		for (int y = 0; y < DATA_H; y++)
		{
			for (int z = 0; z < DATA_D; z++)
			{
				if ( h_Mask[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
				{
					NUMBER_OF_VOXELS += 1.0f;
					for (int t = 0; t < DATA_T; t++)
					{
						h_Timeseries[t] += h_Volumes[x + y * DATA_W + z * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D];
					}
				}
			}
		}
	}

	printf("There are %i voxels in the mask\n",(int)NUMBER_OF_VOXELS);

	for (int t = 0; t < DATA_T; t++)
	{
		h_Timeseries[t] /= NUMBER_OF_VOXELS;
	}

    //------------------------
         
    // Write results to text file            
    // Print motion parameters to file
    std::ofstream timeseries;
  
    // Add the provided filename extension to the original filename, before the dot

	const char* extension = "_timeseries.1D";
	char* filenameWithExtension;

	CreateFilename(filenameWithExtension, inputData, extension, CHANGE_OUTPUT_FILENAME, outputFilename);

    timeseries.open(filenameWithExtension);      

    if ( timeseries.good() )
    {
        timeseries.precision(6);
        for (size_t t = 0; t < DATA_T; t++)
        {
            timeseries << h_Timeseries[t] << std::setw(2) << std::endl;
        }
        timeseries.close();
    }
    else
    {
        printf("Could not open %s for writing!\n",filenameWithExtension);
    }
	free(filenameWithExtension);

	//---------------
    
    // Free all memory
    FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);            
    FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
    
	free(h_Timeseries);

    return EXIT_SUCCESS;
}


