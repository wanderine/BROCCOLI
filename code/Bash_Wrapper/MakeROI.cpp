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
    
    float           *h_Volume = NULL;

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
    const char*     FILENAME_EXTENSION = "_roi";
    bool            PRINT = true;
	bool			VERBOS = false;
    
    size_t          DATA_W, DATA_H, DATA_D, DATA_T;
    float           VOXEL_SIZE_X, VOXEL_SIZE_Y, VOXEL_SIZE_Z;

	bool			CHANGE_OUTPUT_FILENAME = false;

	// Settings
	float			RADIUS = 5.0f;	
	float			RADIUSV = 5.0f;	

	float			XCOORDINATE = 0.0f;
	float			YCOORDINATE = 0.0f;
	float			ZCOORDINATE = 0.0f;

	float			XCOORDINATEV = 0.0f;
	float			YCOORDINATEV = 0.0f;
	float			ZCOORDINATEV = 0.0f;
	
	bool			MMRADIUS = false;
	bool			VOXELSRADIUS = false;

	bool			MMCOORDINATE = false;
	bool			VOXELSCOORDINATE = false;

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
        printf("MakeROI input.nii [options]\n\n");
        printf("Options:\n\n");
        printf(" -coordinate      Center of ROI (x,y,z), in mm  \n");
        printf(" -coordinatev     Center of ROI (x,y,z), in voxels  \n");
        printf(" -radius          Radius of ROI, in millimeters \n");
        printf(" -radiusv         Radius of ROI, in voxels \n");
		printf(" -output      	  Set filename of nifti file  \n");
        printf("\n\n");
        
        return EXIT_SUCCESS;
    }
    // Try to open file
    else if (argc > 1)
    {        
        fp = fopen(argv[1],"r");
        if (fp == NULL)
        {            
            printf("Could not open file %s !\n",argv[1]);
            return EXIT_FAILURE;
        }
        fclose(fp);        
    }
    
    // Loop over additional inputs
    int i = 2;
    while (i < argc)
    {
        char *input = argv[i];
        char *p;
        if (strcmp(input,"-coordinate") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read first value after -coordinate !\n");
                return EXIT_FAILURE;
			}

			if ( (i+2) >= argc  )
			{
			    printf("Unable to read second value after -coordinate !\n");
                return EXIT_FAILURE;
			}

			if ( (i+3) >= argc  )
			{
			    printf("Unable to read third value after -coordinate !\n");
                return EXIT_FAILURE;
			}
            
			MMCOORDINATE = true;
            XCOORDINATE = (float)strtod(argv[i+1], &p);
			YCOORDINATE = (float)strtod(argv[i+2], &p);
			ZCOORDINATE = (float)strtod(argv[i+3], &p);
            
            i += 4;
        }        
        else if (strcmp(input,"-coordinatev") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read first value after -coordinatev !\n");
                return EXIT_FAILURE;
			}

			if ( (i+2) >= argc  )
			{
			    printf("Unable to read second value after -coordinatev !\n");
                return EXIT_FAILURE;
			}

			if ( (i+3) >= argc  )
			{
			    printf("Unable to read third value after -coordinatev !\n");
                return EXIT_FAILURE;
			}
            
			VOXELSCOORDINATE = true;
            XCOORDINATEV = (float)strtod(argv[i+1], &p);
			YCOORDINATEV = (float)strtod(argv[i+2], &p);
			ZCOORDINATEV = (float)strtod(argv[i+3], &p);
            
            i += 4;
        }        
        else if (strcmp(input,"-radius") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -radius !\n");
                return EXIT_FAILURE;
			}
            
            RADIUS = (float)strtod(argv[i+1], &p);
			MMRADIUS = true;
            
			if (!isspace(*p) && *p != 0)
		    {
		        printf("Radius must be a float! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
  			else if ( RADIUS <= 0.0f )
            {
                printf("Radius must be > 0.0 !\n");
                return EXIT_FAILURE;
            }
            i += 2;
        } 
        else if (strcmp(input,"-radiusv") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -radiusv !\n");
                return EXIT_FAILURE;
			}
            
            RADIUSV = (float)strtod(argv[i+1], &p);
			VOXELSRADIUS = true;
            
  			if ( RADIUSV <= 0.0f )
            {
                printf("Radius must be > 0 !\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }        
        else if (strcmp(input,"-output") == 0)
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

    if (!MMCOORDINATE && !VOXELSCOORDINATE)
    {
        printf("Have to define center in mm or in voxels!\n");
        return EXIT_FAILURE;
    }
    if (MMCOORDINATE && VOXELSCOORDINATE)
    {
        printf("Cannot define center in both mm and in voxels!\n");
        return EXIT_FAILURE;
    }


    if (!MMRADIUS && !VOXELSRADIUS)
    {
        printf("Have to define radius in mm or in voxels!\n");
        return EXIT_FAILURE;
    }
    if (MMRADIUS && VOXELSRADIUS)
    {
        printf("Cannot define radius in both mm and in voxels!\n");
        return EXIT_FAILURE;
    }

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

	XCOORDINATE = (float)DATA_W - XCOORDINATE - 1.0f;
	YCOORDINATE = (float)DATA_H - YCOORDINATE - 1.0f;

	if (VOXELSCOORDINATE)
	{
		//XCOORDINATEV = (float)DATA_W - XCOORDINATE - 1.0f;
		//YCOORDINATEV = (float)DATA_H - YCOORDINATE - 1.0f;
		//ZCOORDINATE -= 1.0f;
	}

    // Get voxel sizes
    VOXEL_SIZE_X = inputData->dx;
    VOXEL_SIZE_Y = inputData->dy;
    VOXEL_SIZE_Z = inputData->dz;
    	
    // Calculate size, in bytes
    size_t DATA_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);
    size_t VOLUME_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);
    
    // Print some info
    if (PRINT)
    {
        printf("Authored by K.A. Eklund \n");
        printf("Data size: %zu x %zu x %zu \n",  DATA_W, DATA_H, DATA_D);
        printf("Voxel size: %f x %f x %f mm \n", VOXEL_SIZE_X, VOXEL_SIZE_Y, VOXEL_SIZE_Z);   
    } 
   
    // ------------------------------------------------
    
    // Allocate memory on the host
    
	startTime = GetWallTime();

	AllocateMemory(h_Volume, DATA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "INPUT_DATA");

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
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        {
            h_Volume[i] = (float)p[i];
        }
    }
    else if ( inputData->datatype == DT_UINT8 )
    {
        unsigned char *p = (unsigned char*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        {
            h_Volume[i] = (float)p[i];
        }
    }
    else if ( inputData->datatype == DT_UINT16 )
    {
        unsigned short int *p = (unsigned short int*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        {
            h_Volume[i] = (float)p[i];
        }
    }
	// Correct data type, just copy the pointer
	else if ( inputData->datatype == DT_FLOAT )
    {
        float *p = (float*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        {
            h_Volume[i] = p[i];
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

	if (MMRADIUS)
	{
		for (int x = 0; x < DATA_W; x++)
		{
			for (int y = 0; y < DATA_H; y++)
			{
				for (int z = 0; z < DATA_D; z++)
				{
					float distance = sqrt( ((float) x - XCOORDINATE)*((float) x - XCOORDINATE)*VOXEL_SIZE_X*VOXEL_SIZE_X + ((float) y - YCOORDINATE)*((float) y - YCOORDINATE)*VOXEL_SIZE_Y*VOXEL_SIZE_Y + ((float) z - ZCOORDINATE)*((float) z - ZCOORDINATE)*VOXEL_SIZE_Z*VOXEL_SIZE_Z   );
					if ( distance <= RADIUS )
					{
						h_Volume[x + y * DATA_W + z * DATA_W * DATA_H] = 1.0f;
					}
					else
					{
						h_Volume[x + y * DATA_W + z * DATA_W * DATA_H] = 0.0f;
					}
				}
			}
		}
	}
	else if (VOXELSRADIUS)
	{
		for (int x = 0; x < DATA_W; x++)
		{
			for (int y = 0; y < DATA_H; y++)
			{
				for (int z = 0; z < DATA_D; z++)
				{
					float distance = sqrt( ((float) x - XCOORDINATE)*((float) x - XCOORDINATE) + ((float) y - YCOORDINATE)*((float) y - YCOORDINATE) + ((float) z - ZCOORDINATE)*((float) z - ZCOORDINATE) );
					if ( distance <= RADIUSV )
					{
						h_Volume[x + y * DATA_W + z * DATA_W * DATA_H] = 1.0f;
					}
					else
					{
						h_Volume[x + y * DATA_W + z * DATA_W * DATA_H] = 0.0f;
					}
				}
			}
		}
	}



    //------------------------
         
    // Write results to file            

    nifti_image *outputNifti = nifti_copy_nim_info(inputData);
    outputNifti->nt = 1;	
    outputNifti->dim[0] = 3;
    outputNifti->dim[4] = 1;
    outputNifti->nvox = DATA_W * DATA_H * DATA_D;
    allNiftiImages[numberOfNiftiImages] = outputNifti;
	numberOfNiftiImages++;

    startTime = GetWallTime();

	if (!CHANGE_OUTPUT_FILENAME)
	{
	    WriteNifti(outputNifti,h_Volume,FILENAME_EXTENSION,ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}
	else
	{
		nifti_set_filenames(outputNifti, outputFilename, 0, 1);
		WriteNifti(outputNifti,h_Volume,"",DONT_ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}

	endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to write the nifti file\n",(float)(endTime - startTime));
	}
    
    // Free all memory
    FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);            
    FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
    
    return EXIT_SUCCESS;
}


