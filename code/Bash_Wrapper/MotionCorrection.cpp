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
    
    float           *h_fMRI_Volumes = NULL;
	float			*h_Reference_Volume = NULL;
    float           *h_Quadrature_Filter_1_Real = NULL;
    float           *h_Quadrature_Filter_2_Real = NULL;
    float           *h_Quadrature_Filter_3_Real = NULL;
    float           *h_Quadrature_Filter_1_Imag = NULL;
    float           *h_Quadrature_Filter_2_Imag = NULL;
    float           *h_Quadrature_Filter_3_Imag = NULL;
    
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
    int             MOTION_CORRECTION_FILTER_SIZE = 7; 
    int             NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION = 5;
    int             OPENCL_PLATFORM = 0;
    int             OPENCL_DEVICE = 0;
    int             NUMBER_OF_MOTION_CORRECTION_PARAMETERS = 6;    
    bool            DEBUG = false;
    const char*     FILENAME_EXTENSION = "_mc";
    bool            PRINT = true;
	bool			VERBOS = false;
	bool			CHANGE_OUTPUT_FILENAME = false;
	bool			CHANGE_REFERENCE_VOLUME = false;
	const char*		referenceVolumeFilename;
    
    size_t          DATA_W, DATA_H, DATA_D, DATA_T;
    float           EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z;
    
    //-----------------------
    // Output parameters
    
    const char      *outputFilename;
    
    float           *h_Quadrature_Filter_Response_1_Real = NULL;
    float           *h_Quadrature_Filter_Response_1_Imag = NULL;
    float           *h_Quadrature_Filter_Response_2_Real = NULL;
    float           *h_Quadrature_Filter_Response_2_Imag = NULL;
    float           *h_Quadrature_Filter_Response_3_Real = NULL;
    float           *h_Quadrature_Filter_Response_3_Imag = NULL;  
    float           *h_Phase_Differences = NULL;
    float           *h_Phase_Certainties = NULL;
    float           *h_Phase_Gradients = NULL;
    float           *h_Motion_Corrected_fMRI_Volumes = NULL;
    float           *h_Motion_Parameters = NULL;
    
    //---------------------
        
    /* Input arguments */
    FILE *fp = NULL; 
    
    // No inputs, so print help text
    if (argc == 1)
    {        
        printf("Usage:\n\n");
        printf("MotionCorrection input.nii [options]\n\n");
        printf("Options:\n\n");
        printf(" -platform           The OpenCL platform to use (default 0) \n");
        printf(" -device             The OpenCL device to use for the specificed platform (default 0) \n");
        printf(" -referencevolume    Give a reference volume to align all other volumes to (default false) \n");        
        printf(" -iterations         Number of iterations for the motion correction algorithm (default 5) \n");        
        printf(" -output             Set output filename (default input_mc.nii) \n");
        printf(" -quiet              Don't print anything to the terminal (default false) \n");
        printf(" -verbose            Print extra stuff (default false) \n");
        printf(" -debug              Get additional debug information (default false) \n");
        printf("\n\n");
        
        return EXIT_SUCCESS;
    }
    // Try to open file
    else if (argc > 1)
    {        
		// Check that file extension is .nii or .nii.gz
		std::string extension;
		bool extensionOK;
		CheckFileExtension(argv[1],extensionOK,extension);
		if (!extensionOK)
		{
            printf("File extension is not .nii or .nii.gz, %s is not allowed!\n",extension.c_str());
            return EXIT_FAILURE;
		}

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
        if (strcmp(input,"-platform") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -platform !\n");
                return EXIT_FAILURE;
			}

            OPENCL_PLATFORM = (int)strtol(argv[i+1], &p, 10);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("OpenCL platform must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
            else if (OPENCL_PLATFORM < 0)
            {
                printf("OpenCL platform must be >= 0!\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        else if (strcmp(input,"-device") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -device !\n");
                return EXIT_FAILURE;
			}

            OPENCL_DEVICE = (int)strtol(argv[i+1], &p, 10);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("OpenCL device must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
            else if (OPENCL_DEVICE < 0)
            {
                printf("OpenCL device must be >= 0!\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        else if (strcmp(input,"-referencevolume") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read filename after -referencevolume !\n");
                return EXIT_FAILURE;
			}

			referenceVolumeFilename = argv[i+1];
            CHANGE_REFERENCE_VOLUME = true;
            i += 2;
        }
        else if (strcmp(input,"-iterations") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -iterations !\n");
                return EXIT_FAILURE;
			}

            NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION = (int)strtol(argv[i+1], &p, 10);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("Number of iterations must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
            else if (NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION <= 0)
            {
                printf("Number of iterations must be a positive number!\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        else if (strcmp(input,"-debug") == 0)
        {
            DEBUG = true;
            i += 1;
        }
        else if (strcmp(input,"-quiet") == 0)
        {
            PRINT = false;
            i += 1;
        }
        else if (strcmp(input,"-verbose") == 0)
        {
            VERBOS = true;
            i += 1;
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
    
	// Check if BROCCOLI_DIR variable is set
	if (getenv("BROCCOLI_DIR") == NULL)
	{
        printf("The environment variable BROCCOLI_DIR is not set!\n");
        return EXIT_FAILURE;
	}

    double startTime = GetWallTime();

    // Read data
    nifti_image *inputData = nifti_image_read(argv[1],1);
    
    if (inputData == NULL)
    {
        printf("Could not open fMRI data nifti file!\n");
        return EXIT_FAILURE;
    }
    allNiftiImages[numberOfNiftiImages] = inputData;
	numberOfNiftiImages++;

    nifti_image *referenceVolume;
	if (CHANGE_REFERENCE_VOLUME)
	{
		referenceVolume = nifti_image_read(referenceVolumeFilename,1);
		if (referenceVolume == NULL)
		{
	        printf("Could not open reference volume nifti file!\n");
	        return EXIT_FAILURE;
		}
	    allNiftiImages[numberOfNiftiImages] = referenceVolume;
		numberOfNiftiImages++;
	}

	double endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to read the nifti file\n",(float)(endTime - startTime));
	}

    // Get data dimensions from input data
    DATA_W = inputData->nx;
    DATA_H = inputData->ny;
    DATA_D = inputData->nz;
    DATA_T = inputData->nt;
    
	// Check if reference volume has same size
	if (CHANGE_REFERENCE_VOLUME)
	{
    	size_t TEMP_DATA_W = referenceVolume->nx;
   		size_t TEMP_DATA_H = referenceVolume->ny;
    	size_t TEMP_DATA_D = referenceVolume->nz;

		if ( (TEMP_DATA_W != DATA_W) || (TEMP_DATA_H != DATA_H) || (TEMP_DATA_D != DATA_D) )
		{
        	printf("fMRI data has a size of %zu x %zu x %zu, while the reference volume has a size of %zu x %zu x %zu , aborting!\n",DATA_W,DATA_H,DATA_D,TEMP_DATA_W,TEMP_DATA_H,TEMP_DATA_D);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
    	    return EXIT_FAILURE;
		}
	}

    // Get voxel sizes from input data
    EPI_VOXEL_SIZE_X = inputData->dx;
    EPI_VOXEL_SIZE_Y = inputData->dy;
    EPI_VOXEL_SIZE_Z = inputData->dz;

	// Check if reference volume has same voxel size
	if (CHANGE_REFERENCE_VOLUME)
	{
    	float TEMP_VOXEL_SIZE_X = referenceVolume->dx;
   		float TEMP_VOXEL_SIZE_Y = referenceVolume->dy;
    	float TEMP_VOXEL_SIZE_Z = referenceVolume->dz;

		if ( (TEMP_VOXEL_SIZE_X != EPI_VOXEL_SIZE_X) || (TEMP_VOXEL_SIZE_Y != EPI_VOXEL_SIZE_Y) || (TEMP_VOXEL_SIZE_Z != EPI_VOXEL_SIZE_Z) )
		{
        	printf("fMRI data has a  voxel size of %f x %f x %f, while the reference volume has a voxel size of %f x %f x %f, aborting!\n",EPI_VOXEL_SIZE_X,EPI_VOXEL_SIZE_Y,EPI_VOXEL_SIZE_Z,TEMP_VOXEL_SIZE_X,TEMP_VOXEL_SIZE_Y,TEMP_VOXEL_SIZE_Z);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
    	    return EXIT_FAILURE;
		}
	}
                               
    // Calculate size, in bytes
    size_t DATA_SIZE = DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float);
    size_t MOTION_PARAMETERS_SIZE = NUMBER_OF_MOTION_CORRECTION_PARAMETERS * DATA_T * sizeof(float);
    size_t FILTER_SIZE = MOTION_CORRECTION_FILTER_SIZE * MOTION_CORRECTION_FILTER_SIZE * MOTION_CORRECTION_FILTER_SIZE * sizeof(float);
    size_t VOLUME_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);
    
    // Print some info
    if (PRINT)
    {
        printf("Authored by K.A. Eklund \n");
        printf("Data size: %zu x %zu x %zu x %zu \n",  DATA_W, DATA_H, DATA_D, DATA_T);
        printf("Voxel size: %f x %f x %f mm \n", EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);    
        printf("Number of iterations for motion correction: %i \n",  NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION);
    } 
    
    // ------------------------------------------------
    
    // Allocate memory on the host
    
	startTime = GetWallTime();

	// If the data is in float format, we can just copy the pointer
	if ( inputData->datatype != DT_FLOAT )
	{
		AllocateMemory(h_fMRI_Volumes, DATA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "INPUT_DATA");
	}
	else
	{
		allocatedHostMemory += DATA_SIZE;
	}

	if (CHANGE_REFERENCE_VOLUME)
	{
		AllocateMemory(h_Reference_Volume, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "REFERENCE_VOLUME");
	}

	AllocateMemory(h_Quadrature_Filter_1_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_1_REAL");    
  	AllocateMemory(h_Quadrature_Filter_1_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_1_IMAG");    
	AllocateMemory(h_Quadrature_Filter_2_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_2_REAL");    
  	AllocateMemory(h_Quadrature_Filter_2_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_2_IMAG");    
	AllocateMemory(h_Quadrature_Filter_3_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_3_REAL");    
  	AllocateMemory(h_Quadrature_Filter_3_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_3_IMAG");    
	AllocateMemory(h_Motion_Parameters, MOTION_PARAMETERS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "MOTION_PARAMETERS");       
    
    if (DEBUG)
    {    
		AllocateMemory(h_Quadrature_Filter_Response_1_Real, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_RESPONSE_1_REAL");
		AllocateMemory(h_Quadrature_Filter_Response_1_Imag, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_RESPONSE_1_IMAG");        
		AllocateMemory(h_Quadrature_Filter_Response_2_Real, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_RESPONSE_2_REAL");
		AllocateMemory(h_Quadrature_Filter_Response_2_Imag, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_RESPONSE_2_IMAG");        
		AllocateMemory(h_Quadrature_Filter_Response_3_Real, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_RESPONSE_3_REAL");
		AllocateMemory(h_Quadrature_Filter_Response_3_Imag, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_RESPONSE_3_IMAG");        
		AllocateMemory(h_Phase_Differences, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "PHASE_DIFFERENCES");        
		AllocateMemory(h_Phase_Certainties, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "PHASE_CERTAINTIES");        
		AllocateMemory(h_Phase_Gradients, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "PHASE_GRADIENTS");        
    }
    
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
            h_fMRI_Volumes[i] = (float)p[i];
        }
    }
    else if ( inputData->datatype == DT_UINT8 )
    {
        unsigned char *p = (unsigned char*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D * DATA_T; i++)
        {
            h_fMRI_Volumes[i] = (float)p[i];
        }
    }
    else if ( inputData->datatype == DT_UINT16 )
    {
        unsigned short int *p = (unsigned short int*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D * DATA_T; i++)
        {
            h_fMRI_Volumes[i] = (float)p[i];
        }
    }
	// Correct data type, just copy the pointer
	else if ( inputData->datatype == DT_FLOAT )
    {
		h_fMRI_Volumes = (float*)inputData->data;

		// Save the pointer in the pointer list
		allMemoryPointers[numberOfMemoryPointers] = (void*)h_fMRI_Volumes;
        numberOfMemoryPointers++;

        //float *p = (float*)inputData->data;
    
        //for (size_t i = 0; i < DATA_W * DATA_H * DATA_D * DATA_T; i++)
        //{
        //    h_fMRI_Volumes[i] = p[i];
        //}
    }
    else
    {
        printf("Unknown data type in fMRI data, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
    
	// Free input fMRI data, it has been converted to floats
	if ( inputData->datatype != DT_FLOAT )
	{		
		free(inputData->data);
		inputData->data = NULL;
	}
	// Pointer has been copied to h_fMRI_Volumes and pointer list, so set the input data pointer to NULL
	else
	{		
		inputData->data = NULL;
	}

	if (CHANGE_REFERENCE_VOLUME)
	{
    	// Convert data to floats
	    if ( referenceVolume->datatype == DT_SIGNED_SHORT )
	    {
	        short int *p = (short int*)referenceVolume->data;
	    
	        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
	        {
	            h_Reference_Volume[i] = (float)p[i];
	        }
	    }
	    else if ( referenceVolume->datatype == DT_UINT8 )
	    {
	        unsigned char *p = (unsigned char*)referenceVolume->data;
	    
	        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
	        {
	            h_Reference_Volume[i] = (float)p[i];
    	    }
    	}
    	else if ( referenceVolume->datatype == DT_UINT16 )
    	{
    	    unsigned short int *p = (unsigned short int*)referenceVolume->data;
    	
    	    for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
    	    {
    	        h_Reference_Volume[i] = (float)p[i];
    	    }
    	}
		else if ( referenceVolume->datatype == DT_FLOAT )
    	{	
			float *p = (float*)referenceVolume->data;
    
        	for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        	{
        	    h_Reference_Volume[i] = p[i];
        	}
	    }
    	else
    	{
    	    printf("Unknown data type in reference volume, aborting!\n");
    	    FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
    	    FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
    	    return EXIT_FAILURE;
    	}
	}
	
	endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to convert data to floats\n",(float)(endTime - startTime));
	}

	startTime = GetWallTime();

    // Read quadrature filters, three real valued and three imaginary valued

	std::string filter1RealLinearPathAndName;
	std::string filter1ImagLinearPathAndName;
	std::string filter2RealLinearPathAndName;
	std::string filter2ImagLinearPathAndName;
	std::string filter3RealLinearPathAndName;
	std::string filter3ImagLinearPathAndName;

	filter1RealLinearPathAndName.append(getenv("BROCCOLI_DIR"));
	filter1ImagLinearPathAndName.append(getenv("BROCCOLI_DIR"));
	filter2RealLinearPathAndName.append(getenv("BROCCOLI_DIR"));
	filter2ImagLinearPathAndName.append(getenv("BROCCOLI_DIR"));
	filter3RealLinearPathAndName.append(getenv("BROCCOLI_DIR"));
	filter3ImagLinearPathAndName.append(getenv("BROCCOLI_DIR"));

	filter1RealLinearPathAndName.append("filters/filter1_real_linear_registration.bin");
	filter1ImagLinearPathAndName.append("filters/filter1_imag_linear_registration.bin");
	filter2RealLinearPathAndName.append("filters/filter2_real_linear_registration.bin");
	filter2ImagLinearPathAndName.append("filters/filter2_imag_linear_registration.bin");
	filter3RealLinearPathAndName.append("filters/filter3_real_linear_registration.bin");
	filter3ImagLinearPathAndName.append("filters/filter3_imag_linear_registration.bin");

	ReadBinaryFile(h_Quadrature_Filter_1_Real,MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,filter1RealLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_1_Imag,MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,filter1ImagLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_2_Real,MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,filter2RealLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_2_Imag,MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,filter2ImagLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_3_Real,MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,filter3RealLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_3_Imag,MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,filter3ImagLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages);     
    
	endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to read all binary files\n",(float)(endTime - startTime));
	}    

    //------------------------
    
	startTime = GetWallTime();

	// Initialize BROCCOLI
    BROCCOLI_LIB BROCCOLI(OPENCL_PLATFORM,OPENCL_DEVICE,2,VERBOS); // 2 = Bash wrapper

	endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to initiate BROCCOLI\n",(float)(endTime - startTime));
	}

    // Print build info to file (always)
	std::vector<std::string> buildInfo = BROCCOLI.GetOpenCLBuildInfo();
	std::vector<std::string> kernelFileNames = BROCCOLI.GetKernelFileNames();

	std::string buildInfoPath;
	buildInfoPath.append(getenv("BROCCOLI_DIR"));
	buildInfoPath.append("compiled/Kernels/");

	for (int k = 0; k < BROCCOLI.GetNumberOfKernelFiles(); k++)
	{
		std::string temp = buildInfoPath;
		temp.append("buildInfo_");
		temp.append(BROCCOLI.GetOpenCLPlatformName());
		temp.append("_");	
		temp.append(BROCCOLI.GetOpenCLDeviceName());
		temp.append("_");	
		std::string name = kernelFileNames[k];
		// Remove "kernel" and ".cpp" from kernel filename
		name = name.substr(0,name.size()-4);
		name = name.substr(6,name.size());
		temp.append(name);
		temp.append(".txt");
		fp = fopen(temp.c_str(),"w");
		if (fp == NULL)
		{     
		    printf("Could not open %s for writing ! \n",temp.c_str());
		}
		else
		{	
			if (buildInfo[k].c_str() != NULL)
			{
			    int error = fputs(buildInfo[k].c_str(),fp);
			    if (error == EOF)
			    {
			        printf("Could not write to %s ! \n",temp.c_str());
			    }
			}
			fclose(fp);
		}
	}
    
    // Something went wrong...
    if (!BROCCOLI.GetOpenCLInitiated())
    {              
        printf("Initialization error is \"%s\" \n",BROCCOLI.GetOpenCLInitializationError().c_str());
		printf("OpenCL error is \"%s\" \n",BROCCOLI.GetOpenCLError());

        // Print create kernel errors
        int* createKernelErrors = BROCCOLI.GetOpenCLCreateKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createKernelErrors[i] != 0)
            {
                printf("Create kernel error for kernel '%s' is '%s' \n",BROCCOLI.GetOpenCLKernelName(i),BROCCOLI.GetOpenCLErrorMessage(createKernelErrors[i]));
            }
        }                        
                
        printf("OpenCL initialization failed, aborting! \nSee buildInfo* for output of OpenCL compilation!\n");      
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
    // Initialization OK
    else
    {
        // Set all necessary pointers and values
        BROCCOLI.SetInputfMRIVolumes(h_fMRI_Volumes);
        
		BROCCOLI.SetAllocatedHostMemory(allocatedHostMemory);

        BROCCOLI.SetEPIWidth(DATA_W);
        BROCCOLI.SetEPIHeight(DATA_H);
        BROCCOLI.SetEPIDepth(DATA_D);
        BROCCOLI.SetEPITimepoints(DATA_T);   
        
        BROCCOLI.SetEPIVoxelSizeX(EPI_VOXEL_SIZE_X);
        BROCCOLI.SetEPIVoxelSizeY(EPI_VOXEL_SIZE_Y);
        BROCCOLI.SetEPIVoxelSizeZ(EPI_VOXEL_SIZE_Z);        
        
		BROCCOLI.SetChangeMotionCorrectionReferenceVolume(CHANGE_REFERENCE_VOLUME);
		BROCCOLI.SetMotionCorrectionReferenceVolume(h_Reference_Volume);

        BROCCOLI.SetImageRegistrationFilterSize(MOTION_CORRECTION_FILTER_SIZE);
        BROCCOLI.SetLinearImageRegistrationFilters(h_Quadrature_Filter_1_Real, h_Quadrature_Filter_1_Imag, h_Quadrature_Filter_2_Real, h_Quadrature_Filter_2_Imag, h_Quadrature_Filter_3_Real, h_Quadrature_Filter_3_Imag);
        BROCCOLI.SetNumberOfIterationsForMotionCorrection(NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION);
        
        BROCCOLI.SetOutputMotionParameters(h_Motion_Parameters);
      
        if (DEBUG)
        {
			BROCCOLI.SetDebug(true);
            //BROCCOLI.SetOutputQuadratureFilterResponses(h_Quadrature_Filter_Response_1_Real, h_Quadrature_Filter_Response_1_Imag, h_Quadrature_Filter_Response_2_Real, h_Quadrature_Filter_Response_2_Imag, h_Quadrature_Filter_Response_3_Real, h_Quadrature_Filter_Response_3_Imag);
            BROCCOLI.SetOutputPhaseDifferences(h_Phase_Differences);
            BROCCOLI.SetOutputPhaseCertainties(h_Phase_Certainties);
            BROCCOLI.SetOutputPhaseGradients(h_Phase_Gradients);
        }
             
        // Run the actual motion correction
		startTime = GetWallTime();        
		BROCCOLI.PerformMotionCorrectionWrapper();        
		endTime = GetWallTime();

		if (VERBOS)
	 	{
			printf("\nIt took %f seconds to run the motion correction\n",(float)(endTime - startTime));
		}    

        // Print create buffer errors
        int* createBufferErrors = BROCCOLI.GetOpenCLCreateBufferErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createBufferErrors[i] != 0)
            {
                printf("Create buffer error for kernel '%s' is '%s' \n",BROCCOLI.GetOpenCLKernelName(i),BROCCOLI.GetOpenCLErrorMessage(createBufferErrors[i]));
            }
        }
        
        // Print create kernel errors
        int* createKernelErrors = BROCCOLI.GetOpenCLCreateKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createKernelErrors[i] != 0)
            {
                printf("Create kernel error for kernel '%s' is '%s' \n",BROCCOLI.GetOpenCLKernelName(i),BROCCOLI.GetOpenCLErrorMessage(createKernelErrors[i]));
            }
        } 

        // Print run kernel errors
        int* runKernelErrors = BROCCOLI.GetOpenCLRunKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (runKernelErrors[i] != 0)
            {
                printf("Run kernel error for kernel '%s' is '%s' \n",BROCCOLI.GetOpenCLKernelName(i),BROCCOLI.GetOpenCLErrorMessage(runKernelErrors[i]));
            }
        } 
    }
    
        
    // Find max displacement
    float maxDisplacement = 0.0f;
    int maxVolume = 0;
    for (size_t t = 1; t < DATA_T; t++)
    {        
        float displacement = fabs(h_Motion_Parameters[t + 0*DATA_T]) + fabs(h_Motion_Parameters[t + 1*DATA_T]) + fabs(h_Motion_Parameters[t + 2*DATA_T]) + fabs(h_Motion_Parameters[t + 3*DATA_T]) + fabs(h_Motion_Parameters[t + 4*DATA_T]) + fabs(h_Motion_Parameters[t + 5*DATA_T]);
        
        if (displacement > maxDisplacement)
        {
            maxDisplacement = displacement;
            maxVolume = t;
        }
    }

    if (PRINT)
    {
        printf("Max displacement = %f (mm) at volume %i \n",maxDisplacement,maxVolume);
    }
            
    // Print motion parameters to file
    std::ofstream motion;
  
    // Add the provided filename extension to the original filename, before the dot

	const char* extension = "_motionparameters.1D";
	char* filenameWithExtension;

	CreateFilename(filenameWithExtension, inputData, extension, CHANGE_OUTPUT_FILENAME, outputFilename);

    motion.open(filenameWithExtension);      

    if ( motion.good() )
    {
        //motion.setf(ios::scientific);
        motion.precision(6);
        for (size_t t = 0; t < DATA_T; t++)
        {
            //printf("X translation for timepoint %i is %f\n",t+1,h_Motion_Parameters[t + DATA_T]);
            //motion << h_Motion_Parameters[t + 0*DATA_T] << std::setw(2) << " " << h_Motion_Parameters[t + 1*DATA_T] << std::setw(2) << " " << h_Motion_Parameters[t + 2*DATA_T] << std::setw(2) << " " << h_Motion_Parameters[t + 3*DATA_T] << std::setw(2) << " " << h_Motion_Parameters[t + 4*DATA_T] << std::setw(2) << " " << h_Motion_Parameters[t + 5*DATA_T] << std::endl;
            motion << h_Motion_Parameters[t + 4*DATA_T] << std::setw(2) << " " << -h_Motion_Parameters[t + 3*DATA_T] << std::setw(2) << " " << h_Motion_Parameters[t + 5*DATA_T] << std::setw(2) << " " << -h_Motion_Parameters[t + 2*DATA_T] << std::setw(2) << " " << -h_Motion_Parameters[t + 0*DATA_T] << std::setw(2) << " " << -h_Motion_Parameters[t + 1*DATA_T] << std::endl;
        }
        motion.close();
    }
    else
    {
        printf("Could not open %s for writing!\n",filenameWithExtension);
    }
	free(filenameWithExtension);
        
    // Write motion corrected data to file            
    startTime = GetWallTime();
    
	if (!CHANGE_OUTPUT_FILENAME)
	{
	    WriteNifti(inputData,h_fMRI_Volumes,FILENAME_EXTENSION,ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}
	else
	{
		nifti_set_filenames(inputData, outputFilename, 0, 1);
		WriteNifti(inputData,h_fMRI_Volumes,"",DONT_ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}

    if (DEBUG)
    {
        WriteNifti(inputData,h_Phase_Differences,"_phase_differences",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(inputData,h_Phase_Gradients,"_phase_gradients",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(inputData,h_Phase_Certainties,"_phase_certainties",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(inputData,h_Quadrature_Filter_Response_1_Real,"_quadrature_filter_responses_1_real",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(inputData,h_Quadrature_Filter_Response_1_Imag,"_quadrature_filter_responses_1_imag",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(inputData,h_Quadrature_Filter_Response_2_Real,"_quadrature_filter_responses_2_real",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(inputData,h_Quadrature_Filter_Response_2_Imag,"_quadrature_filter_responses_2_imag",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(inputData,h_Quadrature_Filter_Response_3_Real,"_quadrature_filter_responses_3_real",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(inputData,h_Quadrature_Filter_Response_3_Imag,"_quadrature_filter_responses_3_imag",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);        
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

/*
fid = fopen('filter1_imag_Linear_registration.bin','w')
a = f1_Linear_registration;
a = imag(a);
a(:,:,1) = a(:,:,1)';
a(:,:,2) = a(:,:,2)';
a(:,:,3) = a(:,:,3)';
a(:,:,4) = a(:,:,4)';
a(:,:,5) = a(:,:,5)';
a(:,:,6) = a(:,:,6)';
a(:,:,7) = a(:,:,7)';        
count = fwrite(fid,single(a(:)),'float')
fclose(fid)
*/

