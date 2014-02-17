/*
 * BROCCOLI: An open source multi-platform software for parallel analysis of fMRI data on many core CPUs and GPUS
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
#include "opencl.h"

#define ADD_FILENAME true
#define DONT_ADD_FILENAME true

#define CHECK_EXISTING_FILE true
#define DONT_CHECK_EXISTING_FILE false

void FreeAllMemory(void **pointers, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (pointers[i] != NULL)
        {
			printf("Freeing pointer %i which is %i \n",i,pointers[i]);
            free(pointers[i]);
			printf("Freed pointer %i\n",i);
        }
    }
}


bool AllocateMemory(float *& pointer, int size, void** pointers, int& N)
{
    pointer = (float*)malloc(size);
    if (pointer != NULL)
    {
        pointers[N] = (void*)pointer;
        N++;
        return true;
    }
    else
    {
        printf("Could not allocate host memory! \n");        
        return false;
    }
}
    

bool WriteNifti(nifti_image* inputNifti, float* data, const char* filename, bool addFilename, bool checkFilename)
{       
    char* filenameWithExtension;
    
    // Add the provided filename to the original filename, before the dot
    if (addFilename)
    {
        // Find the dot in the original filename
        const char* p = inputNifti->fname;
        int dotPosition = 0;
        while ( (p != NULL) && ((*p) != '.') )
        {
            p++;
            dotPosition++;
        }
    
        // Allocate temporary array
        filenameWithExtension = (char*)malloc(strlen(inputNifti->fname) + strlen(filename) + 1);
        if (filenameWithExtension == NULL)
        {
            printf("Could not allocate temporary host memory! \n");      
            return false;
        }
    
        // Copy filename to the dot
        strncpy(filenameWithExtension,inputNifti->fname,dotPosition);
        filenameWithExtension[dotPosition] = '\0';
        // Add the extension
        strcat(filenameWithExtension,filename);
        // Add the rest of the original filename
        strcat(filenameWithExtension,inputNifti->fname+dotPosition);    
    }
        
    // Create new nifti image
    nifti_image *outputNifti = new nifti_image;
    // Copy information from input data
    outputNifti = nifti_copy_nim_info(inputNifti);    
    // Set data pointer 
    outputNifti->data = (void*)data;        
    // Set data type to float
    outputNifti->datatype = DT_FLOAT;
    outputNifti->nbyper = 4;    
    
    // Change filename and write
    bool written = false;
    if (addFilename)
    {
        if ( nifti_set_filenames(outputNifti, filenameWithExtension, checkFilename, 1) == 0)
        {
            nifti_image_write(outputNifti);
            written = true;
        }
    }
    else if (!addFilename)
    {
        if ( nifti_set_filenames(outputNifti, filename, checkFilename, 1) == 0)
        {
            nifti_image_write(outputNifti);
            written = true;
        }                
    }    
    
    outputNifti->data = NULL;
    nifti_image_free(outputNifti);
    if (addFilename)
    {
        free(filenameWithExtension);
    } 
        
    if (written)
    {      
        return true;
    }
    else
    {
        return false;
    }                        
}

int main(int argc, char **argv)
{
    //-----------------------
    // Input pointers
    
    float           *h_Input_Volume;
    float           *h_Displacement_Field_X, *h_Displacement_Field_Y, *h_Displacement_Field_Z;
            
    //-----------------------
    // Output pointers        
        
    float           *h_Interpolated_Volume;

    void            *allMemoryPointers[500];
    int             numberOfMemoryPointers = 0;
    
    // Default parameters
        
    int             OPENCL_PLATFORM = 0;
    int             OPENCL_DEVICE = 0;
    bool            DEBUG = false;
    bool            PRINT = true;
	bool			CHANGE_OUTPUT_NAME = false;    

	const char*		outputFilename;

    // Size parameters
    int             INPUT_DATA_H, INPUT_DATA_W, INPUT_DATA_D;
    int             REFERENCE_DATA_H, REFERENCE_DATA_W, REFERENCE_DATA_D;
           
	float			INPUT_VOXEL_SIZE_X, INPUT_VOXEL_SIZE_Y, INPUT_VOXEL_SIZE_Z;
	float			REFERENCE_VOXEL_SIZE_X, REFERENCE_VOXEL_SIZE_Y, REFERENCE_VOXEL_SIZE_Z;

    //---------------------    
    
    /* Input arguments */
    FILE *fp = NULL; 
    
    // No inputs, so print help text
    if (argc == 1)
    {   
		printf("Transforms a volume using provided displacement fields, which have to be of the same size as the reference volume. The input volume is automagically resized and rescaled to match the input volume. \n\n");     
        printf("Usage:\n\n");
        printf("TransformVolume volume_to_transform.nii reference_volume.nii displacement_field_x.nii displacement_field_y.nii displacement_field_z.nii  [options]\n\n");
        printf("Options:\n\n");
        printf(" -platform                  The OpenCL platform to use (default 0) \n");
        printf(" -device                    The OpenCL device to use for the specificed platform (default 0) \n");
		printf(" -output                    Set output filename (default volume_to_transform_warped.nii) \n");
        printf(" -quiet                     Don't print anything to the terminal (default false) \n");
        printf("\n\n");
        
        return 1;
    }
    else if (argc < 6)
    {
        printf("Need one volume to warp, one reference volume and three displacement field volumes!\n\n");
		return -1;
    }
    // Try to open files
    else if (argc > 1)
    {        
        fp = fopen(argv[1],"r");
        if (fp == NULL)
        {
            printf("Could not open file %s !\n",argv[1]);
            return -1;
        }
        fclose(fp);     
        
        fp = fopen(argv[2],"r");
        if (fp == NULL)
        {
            printf("Could not open file %s !\n",argv[2]);
            return -1;
        }
        fclose(fp);   

        fp = fopen(argv[3],"r");
        if (fp == NULL)
        {
            printf("Could not open file %s !\n",argv[3]);
            return -1;
        }
        fclose(fp);   

        fp = fopen(argv[4],"r");
        if (fp == NULL)
        {
            printf("Could not open file %s !\n",argv[4]);
            return -1;
        }
        fclose(fp);   

        fp = fopen(argv[5],"r");
        if (fp == NULL)
        {
            printf("Could not open file %s !\n",argv[5]);
            return -1;
        }
        fclose(fp);            
    }
    
    // Loop over additional inputs
    int i = 6;
    while (i < argc)
    {
        char *input = argv[i];
        char *p;
        if (strcmp(input,"-platform") == 0)
        {
            OPENCL_PLATFORM = (int)strtol(argv[i+1], &p, 10);
            if (OPENCL_PLATFORM < 0)
            {
                printf("OpenCL platform must be >= 0!\n");
                return -1;
            }
            i += 2;
        }
        else if (strcmp(input,"-device") == 0)
        {
            OPENCL_DEVICE = (int)strtol(argv[i+1], &p, 10);
            if (OPENCL_DEVICE < 0)
            {
                printf("OpenCL device must be >= 0!\n");
                return -1;
            }
            i += 2;
        }      
        else if (strcmp(input,"-quiet") == 0)
        {
            PRINT = false;
            i += 1;
        }
        else if (strcmp(input,"-output") == 0)
        {
			CHANGE_OUTPUT_NAME = true;
            outputFilename = argv[i+1];
            i += 2;
        }
        else
        {
            printf("Unrecognized option! %s \n",argv[i]);
            return -1;
        }                
    }
    
    
    // Read data
    nifti_image *inputVolume = nifti_image_read(argv[1],1);    
    if (inputVolume == NULL)
    {
        printf("Could not open volume to transform!\n");
        return -1;
    }

    nifti_image *referenceVolume = nifti_image_read(argv[2],1);    
    if (referenceVolume == NULL)
    {
        printf("Could not open reference volume!\n");
        return -1;
    }
    
    nifti_image *inputDisplacementX = nifti_image_read(argv[3],1);   
    if (inputDisplacementX == NULL)
    {
        printf("Could not open displacement X volume!\n");
        return -1;
    }

    nifti_image *inputDisplacementY = nifti_image_read(argv[4],1);   
    if (inputDisplacementY == NULL)
    {
        printf("Could not open displacement Y volume!\n");
        return -1;
    }

    nifti_image *inputDisplacementZ = nifti_image_read(argv[5],1);   
    if (inputDisplacementZ == NULL)
    {
        printf("Could not open displacement Z volume!\n");
        return -1;
    }
    
    // Get data dimensions from input data
    INPUT_DATA_W = inputVolume->nx;
    INPUT_DATA_H = inputVolume->ny;
	INPUT_DATA_D = inputVolume->nz;    

	INPUT_VOXEL_SIZE_X = inputVolume->dx;
	INPUT_VOXEL_SIZE_Y = inputVolume->dy;
	INPUT_VOXEL_SIZE_Z = inputVolume->dz;

    REFERENCE_DATA_W = referenceVolume->nx;
    REFERENCE_DATA_H = referenceVolume->ny;
	REFERENCE_DATA_D = referenceVolume->nz;    
    
	REFERENCE_VOXEL_SIZE_X = referenceVolume->dx;
	REFERENCE_VOXEL_SIZE_Y = referenceVolume->dy;
	REFERENCE_VOXEL_SIZE_Z = referenceVolume->dz;

    // Check if the displacement volumes have the same size
	int DISPLACEMENT_DATA_W, DISPLACEMENT_DATA_H, DISPLACEMENT_DATA_D;

	DISPLACEMENT_DATA_W = inputDisplacementX->nx;
	DISPLACEMENT_DATA_H = inputDisplacementX->ny;
	DISPLACEMENT_DATA_D = inputDisplacementX->nz;
	
	if ( (DISPLACEMENT_DATA_W != REFERENCE_DATA_W) || (DISPLACEMENT_DATA_H != REFERENCE_DATA_H) || (DISPLACEMENT_DATA_D != REFERENCE_DATA_D) )
	{
        printf("Dimensions of displacement field x does not match the reference volume!\n");
        return -1;
	}

	DISPLACEMENT_DATA_W = inputDisplacementY->nx;
	DISPLACEMENT_DATA_H = inputDisplacementY->ny;
	DISPLACEMENT_DATA_D = inputDisplacementY->nz;

	if ( (DISPLACEMENT_DATA_W != REFERENCE_DATA_W) || (DISPLACEMENT_DATA_H != REFERENCE_DATA_H) || (DISPLACEMENT_DATA_D != REFERENCE_DATA_D) )
	{
        printf("Dimensions of displacement field y does not match the reference volume!\n");
        return -1;
	}

	DISPLACEMENT_DATA_W = inputDisplacementZ->nx;
	DISPLACEMENT_DATA_H = inputDisplacementZ->ny;
	DISPLACEMENT_DATA_D = inputDisplacementZ->nz;

	if ( (DISPLACEMENT_DATA_W != REFERENCE_DATA_W) || (DISPLACEMENT_DATA_H != REFERENCE_DATA_H) || (DISPLACEMENT_DATA_D != REFERENCE_DATA_D) )
	{
        printf("Dimensions of displacement field z does not match the reference volume!\n");
        return -1;
	}

    // Calculate size, in bytes
    
    int INPUT_VOLUME_SIZE = INPUT_DATA_W * INPUT_DATA_H * INPUT_DATA_D * sizeof(float);
    int REFERENCE_VOLUME_SIZE = REFERENCE_DATA_W * REFERENCE_DATA_H * REFERENCE_DATA_D * sizeof(float);

    // Print some info
    if (PRINT)
    {
        printf("Authored by K.A. Eklund \n");
        printf("Input volume size: %i x %i x %i \n",  INPUT_DATA_W, INPUT_DATA_H, INPUT_DATA_D);
        printf("Input volume voxel size: %f x %f x %f \n",  INPUT_VOXEL_SIZE_X, INPUT_VOXEL_SIZE_Y, INPUT_VOXEL_SIZE_Z);
        printf("Reference volume size: %i x %i x %i \n",  REFERENCE_DATA_W, REFERENCE_DATA_H, REFERENCE_DATA_D);
        printf("Reference volume voxel size: %f x %f x %f \n",  REFERENCE_VOXEL_SIZE_X, REFERENCE_VOXEL_SIZE_Y, REFERENCE_VOXEL_SIZE_Z);
    }

	if (PRINT)
	{
		printf("Input volume size in bytes: %i \n",  INPUT_VOLUME_SIZE);
	}
            
    // ------------------------------------------------
    
    // Allocate memory on the host        
    
    if (!AllocateMemory(h_Input_Volume, INPUT_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers))
    {
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputVolume);
        nifti_image_free(referenceVolume);
        nifti_image_free(inputDisplacementX);
        nifti_image_free(inputDisplacementY);
        nifti_image_free(inputDisplacementZ);
        return -1;
    }
    
    if (!AllocateMemory(h_Interpolated_Volume, REFERENCE_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers))
    {
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputVolume);
        nifti_image_free(referenceVolume);
        nifti_image_free(inputDisplacementX);
        nifti_image_free(inputDisplacementY);
        nifti_image_free(inputDisplacementZ);
        return -1;
    }    
    
    
    if (!AllocateMemory(h_Displacement_Field_X, REFERENCE_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers))
    {
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputVolume);
        nifti_image_free(referenceVolume);
        nifti_image_free(inputDisplacementX);
        nifti_image_free(inputDisplacementY);
        nifti_image_free(inputDisplacementZ);
        return -1;
    }
    if (!AllocateMemory(h_Displacement_Field_Y, REFERENCE_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers))
    {
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputVolume);
        nifti_image_free(referenceVolume);
        nifti_image_free(inputDisplacementX);
        nifti_image_free(inputDisplacementY);
        nifti_image_free(inputDisplacementZ);
        return -1;
    }
    if (!AllocateMemory(h_Displacement_Field_Z, REFERENCE_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers))
    {
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputVolume);
        nifti_image_free(referenceVolume);
        nifti_image_free(inputDisplacementX);
        nifti_image_free(inputDisplacementY);
        nifti_image_free(inputDisplacementZ);
        return -1;
    }   
        
    // Convert data to floats
    if ( inputVolume->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputVolume->data;
    
        for (int i = 0; i < INPUT_DATA_W * INPUT_DATA_H * INPUT_DATA_D; i++)
        {
            h_Input_Volume[i] = (float)p[i];
        }
    }
    else if ( inputVolume->datatype == DT_FLOAT )
    {
        float *p = (float*)inputVolume->data;
    
        for (int i = 0; i < INPUT_DATA_W * INPUT_DATA_H * INPUT_DATA_D; i++)
        {
            h_Input_Volume[i] = p[i];
        }
    }
    else
    {
        printf("Unknown data type in volume to transform, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputVolume);
        nifti_image_free(referenceVolume);
        nifti_image_free(inputDisplacementX);
        nifti_image_free(inputDisplacementY);
        nifti_image_free(inputDisplacementZ);
        return -1;
    }
    
    if ( inputDisplacementX->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputDisplacementX->data;
    
        for (int i = 0; i < REFERENCE_DATA_W * REFERENCE_DATA_H * REFERENCE_DATA_D; i++)
        {
            h_Displacement_Field_X[i] = (float)p[i];
        }
    }
    else if ( inputDisplacementX->datatype == DT_FLOAT )
    {
        float *p = (float*)inputDisplacementX->data;
    
        for (int i = 0; i < REFERENCE_DATA_W * REFERENCE_DATA_H * REFERENCE_DATA_D; i++)
        {
            h_Displacement_Field_X[i] = p[i];
        }
    }
    else
    {
        printf("Unknown data type in displacement x volume, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputVolume);
        nifti_image_free(referenceVolume);
        nifti_image_free(inputDisplacementX);
        nifti_image_free(inputDisplacementY);
        nifti_image_free(inputDisplacementZ);
        return -1;
    }

    if ( inputDisplacementY->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputDisplacementY->data;
    
        for (int i = 0; i < REFERENCE_DATA_W * REFERENCE_DATA_H * REFERENCE_DATA_D; i++)
        {
	        h_Displacement_Field_Y[i] = (float)p[i];
        }
    }
    else if ( inputDisplacementY->datatype == DT_FLOAT )
    {
        float *p = (float*)inputDisplacementY->data;
    
        for (int i = 0; i < REFERENCE_DATA_W * REFERENCE_DATA_H * REFERENCE_DATA_D; i++)
        {
            h_Displacement_Field_Y[i] = p[i];			
        }
    }
    else
    {
        printf("Unknown data type in displacement y volume, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputVolume);
        nifti_image_free(referenceVolume);
        nifti_image_free(inputDisplacementX);
        nifti_image_free(inputDisplacementY);
        nifti_image_free(inputDisplacementZ);
        return -1;
    }

    if ( inputDisplacementZ->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputDisplacementZ->data;
    
        for (int i = 0; i < REFERENCE_DATA_W * REFERENCE_DATA_H * REFERENCE_DATA_D; i++)
        {
            h_Displacement_Field_Z[i] = (float)p[i];
        }
    }
    else if ( inputDisplacementZ->datatype == DT_FLOAT )
    {
        float *p = (float*)inputDisplacementZ->data;
    
        for (int i = 0; i < REFERENCE_DATA_W * REFERENCE_DATA_H * REFERENCE_DATA_D; i++)
        {
            h_Displacement_Field_Z[i] = p[i];
        }
    }
    else
    {
        printf("Unknown data type in displacement z volume, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputVolume);
        nifti_image_free(referenceVolume);
        nifti_image_free(inputDisplacementX);
        nifti_image_free(inputDisplacementY);
        nifti_image_free(inputDisplacementZ);
        return -1;
    }
            
    //------------------------
    
    BROCCOLI_LIB BROCCOLI(OPENCL_PLATFORM,OPENCL_DEVICE);
    
    // Something went wrong...
    if (BROCCOLI.GetOpenCLInitiated() == 0)
    {              
        printf("Get platform IDs error is %s \n",BROCCOLI.GetOpenCLErrorMessage(BROCCOLI.GetOpenCLPlatformIDsError()));
        printf("Get device IDs error is %s \n",BROCCOLI.GetOpenCLErrorMessage(BROCCOLI.GetOpenCLDeviceIDsError()));
        printf("Create context error is %s \n",BROCCOLI.GetOpenCLErrorMessage(BROCCOLI.GetOpenCLCreateContextError()));
        printf("Get create context info error is %s \n",BROCCOLI.GetOpenCLErrorMessage(BROCCOLI.GetOpenCLContextInfoError()));
        printf("Create command queue error is %s \n",BROCCOLI.GetOpenCLErrorMessage(BROCCOLI.GetOpenCLCreateCommandQueueError()));
        printf("Create program error is %s \n",BROCCOLI.GetOpenCLErrorMessage(BROCCOLI.GetOpenCLCreateProgramError()));
        printf("Build program error is %s \n",BROCCOLI.GetOpenCLErrorMessage(BROCCOLI.GetOpenCLBuildProgramError()));
        printf("Get program build info error is %s \n",BROCCOLI.GetOpenCLErrorMessage(BROCCOLI.GetOpenCLProgramBuildInfoError()));
    
        // Print create kernel errors
        int* createKernelErrors = BROCCOLI.GetOpenCLCreateKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createKernelErrors[i] != 0)
            {
                printf("Create kernel error %i is %s \n",i,BROCCOLI.GetOpenCLErrorMessage(createKernelErrors[i]));
            }
        }                
        
        // Print build info to file    
        fp = fopen("buildinfo.txt","w");
        if (fp == NULL)
        {     
            printf("Could not open buildinfo.txt! \n");
        }
        if (BROCCOLI.GetOpenCLBuildInfoChar() != NULL)
        {
            int error = fputs(BROCCOLI.GetOpenCLBuildInfoChar(),fp);
            if (error == EOF)
            {
                printf("Could not write to buildinfo.txt! \n");
            }
        }
        fclose(fp);
        
        printf("OpenCL initialization failed, aborting! \nSee buildinfo.txt for output of OpenCL compilation!\n");      
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputVolume);
        nifti_image_free(referenceVolume);
        nifti_image_free(inputDisplacementX);
        nifti_image_free(inputDisplacementY);
        nifti_image_free(inputDisplacementZ);
        return -1;
    }
    // Initialization OK
    else if (BROCCOLI.GetOpenCLInitiated() == 1)
    {
        // Set all necessary pointers and values
        BROCCOLI.SetInputT1Volume(h_Input_Volume);        
        BROCCOLI.SetT1Width(INPUT_DATA_W);
        BROCCOLI.SetT1Height(INPUT_DATA_H);
        BROCCOLI.SetT1Depth(INPUT_DATA_D);  
		BROCCOLI.SetT1VoxelSizeX(INPUT_VOXEL_SIZE_X);
		BROCCOLI.SetT1VoxelSizeY(INPUT_VOXEL_SIZE_Y);
		BROCCOLI.SetT1VoxelSizeZ(INPUT_VOXEL_SIZE_Z);
             
        BROCCOLI.SetMNIWidth(REFERENCE_DATA_W);
        BROCCOLI.SetMNIHeight(REFERENCE_DATA_H);
        BROCCOLI.SetMNIDepth(REFERENCE_DATA_D);
		BROCCOLI.SetMNIVoxelSizeX(REFERENCE_VOXEL_SIZE_X);
		BROCCOLI.SetMNIVoxelSizeY(REFERENCE_VOXEL_SIZE_Y);
		BROCCOLI.SetMNIVoxelSizeZ(REFERENCE_VOXEL_SIZE_Z);

		BROCCOLI.SetMMT1ZCUT(0);  

        BROCCOLI.SetInterpolationMode(LINEAR);  
		BROCCOLI.SetOutputDisplacementField(h_Displacement_Field_X,h_Displacement_Field_Y,h_Displacement_Field_Z);      
        BROCCOLI.SetOutputInterpolatedT1Volume(h_Interpolated_Volume);
                
        if (DEBUG)
        {
            BROCCOLI.SetDebug(true);            
        }
        
		for (int i = 0; i < numberOfMemoryPointers; i++)
		{
			printf("Pointer i is %i \n",allMemoryPointers[i]);
		}
                    
        // Run the actual transformation
        BROCCOLI.TransformVolumesNonParametricWrapper();

		for (int i = 0; i < numberOfMemoryPointers; i++)
		{
			printf("Pointer i is %i \n",allMemoryPointers[i]);
		}
     
        // Print create buffer errors
        int* createBufferErrors = BROCCOLI.GetOpenCLCreateBufferErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createBufferErrors[i] != 0)
            {
                printf("Create buffer error %i is %s \n",i,BROCCOLI.GetOpenCLErrorMessage(createBufferErrors[i]));
            }
        }
        
        // Print run kernel errors
        int* runKernelErrors = BROCCOLI.GetOpenCLRunKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (runKernelErrors[i] != 0)
            {
                printf("Run kernel error %i is %s \n",i,BROCCOLI.GetOpenCLErrorMessage(runKernelErrors[i]));
            }
        } 
    }        

    // Create new nifti image
    nifti_image *outputNifti = new nifti_image;
    
    // Copy information from input data
	outputNifti = nifti_copy_nim_info(referenceVolume);    
	

	// Change filename
	if (!CHANGE_OUTPUT_NAME)
	{
    	nifti_set_filenames(outputNifti, inputVolume->fname, 0, 1);

	    // Write transformed data to file
	    WriteNifti(outputNifti,h_Interpolated_Volume,"_warped",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);

		//nifti_set_filenames(outputNifti, "warped.nii", 0, 1);        
		//WriteNifti(outputNifti,h_Interpolated_Volume,"",DONT_ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}
	else 
	{
    	//nifti_set_filenames(outputNifti, outputFilename, 0, 1);
    	// Write transformed data to file
	    //WriteNifti(outputNifti,h_Interpolated_Volume,"",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);    
	}
                 
            
    // Free all memory
	FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);	   
	nifti_image_free(inputVolume);
    nifti_image_free(referenceVolume);
    nifti_image_free(inputDisplacementX);
    nifti_image_free(inputDisplacementY);
    nifti_image_free(inputDisplacementZ);
    nifti_image_free(outputNifti);
	

    return 1;
}



