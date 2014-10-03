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

#include <limits.h>
#include <unistd.h>

#include <time.h>
#include <sys/time.h>

#define ADD_FILENAME true
#define DONT_ADD_FILENAME true

#define CHECK_EXISTING_FILE true
#define DONT_CHECK_EXISTING_FILE false

std::string Getexepath()
{
  char result[ PATH_MAX ];
  ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );
  return std::string( result, (count > 0) ? count : 0 );
}

void FreeAllMemory(void **pointers, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (pointers[i] != NULL)
        {
            free(pointers[i]);
        }
    }
}

void FreeAllNiftiImages(nifti_image **niftiImages, int N)
{
    for (int i = 0; i < N; i++)
    {
		if (niftiImages[i] != NULL)
		{
			nifti_image_free(niftiImages[i]);
		}
    }
}

void ReadBinaryFile(float* pointer, int size, const char* filename, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages)
{
	if (pointer == NULL)
    {
        printf("The provided pointer for file %s is NULL, aborting! \n",filename);
        FreeAllMemory(pointers,Npointers);
		FreeAllNiftiImages(niftiImages,Nimages);
        exit(EXIT_FAILURE);
	}	

	FILE *fp = NULL; 
	fp = fopen(filename,"rb");

    if (fp != NULL)
    {
        fread(pointer,sizeof(float),size,fp);
        fclose(fp);
    }
    else
    {
        printf("Could not open %s , aborting! \n",filename);
        FreeAllMemory(pointers,Npointers);
		FreeAllNiftiImages(niftiImages,Nimages);
        exit(EXIT_FAILURE);
    }
}

void AllocateMemory(float *& pointer, int size, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages, const char* variable)
{
    pointer = (float*)malloc(size);
    if (pointer != NULL)
    {
        pointers[Npointers] = (void*)pointer;
        Npointers++;
    }
    else
    {
        printf("Could not allocate host memory for variable %s ! \n",variable);        
		FreeAllMemory(pointers, Npointers);
		FreeAllNiftiImages(niftiImages, Nimages);
		exit(EXIT_FAILURE);        
    }
}


bool WriteNifti(nifti_image* inputNifti, float* data, const char* filename, bool addFilename, bool checkFilename)
{       
	if (data == NULL)
    {
        printf("The provided data pointer for file %s is NULL, aborting writing nifti file! \n",filename);
		return false;
	}	
	if (inputNifti == NULL)
    {
        printf("The provided nifti pointer for file %s is NULL, aborting writing nifti file! \n",filename);
		return false;
	}	


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
        
    // Copy information from input data
    nifti_image *outputNifti = nifti_copy_nim_info(inputNifti);    
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

double GetWallTime()
{
    struct timeval time;
    if (gettimeofday(&time,NULL))
    {
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


int main(int argc, char ** argv)
{
    //-----------------------
    // Input pointers
    
    float           *h_fMRI_Volumes = NULL;
    float           *h_Quadrature_Filter_1_Real = NULL;
    float           *h_Quadrature_Filter_2_Real = NULL;
    float           *h_Quadrature_Filter_3_Real = NULL;
    float           *h_Quadrature_Filter_1_Imag = NULL;
    float           *h_Quadrature_Filter_2_Imag = NULL;
    float           *h_Quadrature_Filter_3_Imag = NULL;
    
    void*			allMemoryPointers[500];
    int             numberOfMemoryPointers = 0;

	nifti_image*	allNiftiImages[500];
    int             numberOfNiftiImages = 0;

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
    
    int             DATA_W, DATA_H, DATA_D, DATA_T;
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
        printf(" -platform   The OpenCL platform to use (default 0) \n");
        printf(" -device     The OpenCL device to use for the specificed platform (default 0) \n");
        printf(" -iterations Number of iterations for the motion correction algorithm (default 5) \n");        
        printf(" -output     Set output filename (default input_mc.nii) \n");
        printf(" -quiet      Don't print anything to the terminal (default false) \n");
        printf(" -verbose    Print extra stuff (default false) \n");
        printf(" -debug      Get additional debug information (default false) \n");
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

    // Read data
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

    // Get data dimensions from input data
    DATA_W = inputData->nx;
    DATA_H = inputData->ny;
    DATA_D = inputData->nz;
    DATA_T = inputData->nt;
    
    // Get voxel sizes from input data
    EPI_VOXEL_SIZE_X = inputData->dx;
    EPI_VOXEL_SIZE_Y = inputData->dy;
    EPI_VOXEL_SIZE_Z = inputData->dz;
                               
    // Calculate size, in bytes
    int DATA_SIZE = DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float);
    int MOTION_PARAMETERS_SIZE = NUMBER_OF_MOTION_CORRECTION_PARAMETERS * DATA_T * sizeof(float);
    int FILTER_SIZE = MOTION_CORRECTION_FILTER_SIZE * MOTION_CORRECTION_FILTER_SIZE * MOTION_CORRECTION_FILTER_SIZE * sizeof(float);
    int VOLUME_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);
    
    // Print some info
    if (PRINT)
    {
        printf("Authored by K.A. Eklund \n");
        printf("Data size: %i x %i x %i x %i \n",  DATA_W, DATA_H, DATA_D, DATA_T);
        printf("Voxel size: %f x %f x %f mm \n", EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);    
        printf("Number of iterations for motion correction: %i \n",  NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION);
    } 
    
    // ------------------------------------------------
    
    // Allocate memory on the host
    
	startTime = GetWallTime();

	AllocateMemory(h_fMRI_Volumes, DATA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "INPUT_DATA");
	AllocateMemory(h_Motion_Corrected_fMRI_Volumes, DATA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "MOTION_CORRECTED_DATA");
	AllocateMemory(h_Quadrature_Filter_1_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_1_REAL");    
  	AllocateMemory(h_Quadrature_Filter_1_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_1_IMAG");    
	AllocateMemory(h_Quadrature_Filter_2_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_2_REAL");    
  	AllocateMemory(h_Quadrature_Filter_2_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_2_IMAG");    
	AllocateMemory(h_Quadrature_Filter_3_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_3_REAL");    
  	AllocateMemory(h_Quadrature_Filter_3_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_3_IMAG");    
	AllocateMemory(h_Motion_Parameters, MOTION_PARAMETERS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "MOTION_PARAMETERS");       
    
    if (DEBUG)
    {    
		AllocateMemory(h_Quadrature_Filter_Response_1_Real, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_RESPONSE_1_REAL");
		AllocateMemory(h_Quadrature_Filter_Response_1_Imag, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_RESPONSE_1_IMAG");        
		AllocateMemory(h_Quadrature_Filter_Response_2_Real, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_RESPONSE_2_REAL");
		AllocateMemory(h_Quadrature_Filter_Response_2_Imag, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_RESPONSE_2_IMAG");        
		AllocateMemory(h_Quadrature_Filter_Response_3_Real, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_RESPONSE_3_REAL");
		AllocateMemory(h_Quadrature_Filter_Response_3_Imag, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_RESPONSE_3_IMAG");        
		AllocateMemory(h_Phase_Differences, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "PHASE_DIFFERENCES");        
		AllocateMemory(h_Phase_Certainties, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "PHASE_CERTAINTIES");        
		AllocateMemory(h_Phase_Gradients, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "PHASE_GRADIENTS");        
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
    
        for (int i = 0; i < DATA_W * DATA_H * DATA_D * DATA_T; i++)
        {
            h_fMRI_Volumes[i] = (float)p[i];
        }
    }
	else if ( inputData->datatype == DT_FLOAT )
    {
        float *p = (float*)inputData->data;
    
        for (int i = 0; i < DATA_W * DATA_H * DATA_D * DATA_T; i++)
        {
            h_fMRI_Volumes[i] = p[i];
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

	startTime = GetWallTime();

    // Read quadrature filters, three real valued and three imaginary valued

	/*
	std::string path = Getexepath();
	path.erase(path.end()-16, path.end()); // 16 is the number of characters in 'MotionCorrection'
	std::string filter1RealName = path;
	std::string filter1ImagName = path;
	std::string filter2RealName = path;
	std::string filter2ImagName = path;
	std::string filter3RealName = path;
	std::string filter3ImagName = path;
	*/

	std::string filter1RealName;
	std::string filter1ImagName;
	std::string filter2RealName;
	std::string filter2ImagName;
	std::string filter3RealName;
	std::string filter3ImagName;

	filter1RealName.append("filter1_real_linear_registration.bin");
	filter1ImagName.append("filter1_imag_linear_registration.bin");
	filter2RealName.append("filter2_real_linear_registration.bin");
	filter2ImagName.append("filter2_imag_linear_registration.bin");
	filter3RealName.append("filter3_real_linear_registration.bin");
	filter3ImagName.append("filter3_imag_linear_registration.bin");

	ReadBinaryFile(h_Quadrature_Filter_1_Real,MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,filter1RealName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_1_Imag,MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,filter1ImagName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_2_Real,MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,filter2RealName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_2_Imag,MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,filter2ImagName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_3_Real,MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,filter3RealName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_3_Imag,MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,filter3ImagName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages);     
    
	endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to read all binary files\n",(float)(endTime - startTime));
	}    

    //------------------------
    
	startTime = GetWallTime();

	// Initialize BROCCOLI
    BROCCOLI_LIB BROCCOLI(OPENCL_PLATFORM,OPENCL_DEVICE,2); // 2 = Bash wrapper

	endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to initiate BROCCOLI\n",(float)(endTime - startTime));
	}
    
    // Something went wrong...
    if (!BROCCOLI.GetOpenCLInitiated())
    {              
        printf("Initialization error is \"%s\" \n",BROCCOLI.GetOpenCLInitializationError());
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
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
    // Initialization OK
    else
    {
        // Set all necessary pointers and values
        BROCCOLI.SetInputfMRIVolumes(h_fMRI_Volumes);
        
        BROCCOLI.SetEPIWidth(DATA_W);
        BROCCOLI.SetEPIHeight(DATA_H);
        BROCCOLI.SetEPIDepth(DATA_D);
        BROCCOLI.SetEPITimepoints(DATA_T);   
        
        BROCCOLI.SetEPIVoxelSizeX(EPI_VOXEL_SIZE_X);
        BROCCOLI.SetEPIVoxelSizeY(EPI_VOXEL_SIZE_Y);
        BROCCOLI.SetEPIVoxelSizeZ(EPI_VOXEL_SIZE_Z);        
        
        BROCCOLI.SetImageRegistrationFilterSize(MOTION_CORRECTION_FILTER_SIZE);
        BROCCOLI.SetLinearImageRegistrationFilters(h_Quadrature_Filter_1_Real, h_Quadrature_Filter_1_Imag, h_Quadrature_Filter_2_Real, h_Quadrature_Filter_2_Imag, h_Quadrature_Filter_3_Real, h_Quadrature_Filter_3_Imag);
        BROCCOLI.SetNumberOfIterationsForMotionCorrection(NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION);
        
        BROCCOLI.SetOutputMotionCorrectedfMRIVolumes(h_Motion_Corrected_fMRI_Volumes);
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
    for (int t = 1; t < DATA_T; t++)
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
    motion.open("motion.1D");      
    if ( motion.good() )
    {
        //motion.setf(ios::scientific);
        motion.precision(6);
        for (int t = 0; t < DATA_T; t++)
        {
            //printf("X translation for timepoint %i is %f\n",t+1,h_Motion_Parameters[t + DATA_T]);
            //motion << h_Motion_Parameters[t + 0*DATA_T] << std::setw(2) << " " << h_Motion_Parameters[t + 1*DATA_T] << std::setw(2) << " " << h_Motion_Parameters[t + 2*DATA_T] << std::setw(2) << " " << h_Motion_Parameters[t + 3*DATA_T] << std::setw(2) << " " << h_Motion_Parameters[t + 4*DATA_T] << std::setw(2) << " " << h_Motion_Parameters[t + 5*DATA_T] << std::endl;
            motion << h_Motion_Parameters[t + 4*DATA_T] << std::setw(2) << " " << -h_Motion_Parameters[t + 3*DATA_T] << std::setw(2) << " " << h_Motion_Parameters[t + 5*DATA_T] << std::setw(2) << " " << -h_Motion_Parameters[t + 2*DATA_T] << std::setw(2) << " " << -h_Motion_Parameters[t + 0*DATA_T] << std::setw(2) << " " << -h_Motion_Parameters[t + 1*DATA_T] << std::endl;
        }
        motion.close();
    }
    else
    {
        printf("Could not open motion.1D for writing!\n");
    }
        
    // Write motion corrected data to file            
    startTime = GetWallTime();

    WriteNifti(inputData,h_Motion_Corrected_fMRI_Volumes,FILENAME_EXTENSION,ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
    
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

