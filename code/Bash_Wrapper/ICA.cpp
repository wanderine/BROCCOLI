/*
    BROCCOLI: Software for Fast fMRI Analysis on Many-Core CPUs and GPUs
    
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
	float			*h_EPI_Mask = NULL;

    float           *h_Quadrature_Filter_1_Real = NULL;
    float           *h_Quadrature_Filter_2_Real = NULL;
    float           *h_Quadrature_Filter_3_Real = NULL;
    float           *h_Quadrature_Filter_1_Imag = NULL;
    float           *h_Quadrature_Filter_2_Imag = NULL;
    float           *h_Quadrature_Filter_3_Imag = NULL;

    size_t          DATA_W, DATA_H, DATA_D, DATA_T;
    float           EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z;

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
    int             OPENCL_PLATFORM = 0;
    int             OPENCL_DEVICE = 0;
    bool            DEBUG = false;
    const char*     FILENAME_EXTENSION = "_ica";
    bool            PRINT = true;
	bool			VERBOS = false;
    
	bool			CHANGE_OUTPUT_FILENAME = false;

	// Settings
	bool			AUTO_MASK = true;
	bool			MASK = false;
	const char*		MASK_NAME;
	bool			Z_SCORE = false;
	bool			CPU = false;
	bool			DOUBLEPRECISION = false;
	
	size_t			NUMBER_OF_ICA_COMPONENTS = 55;

	double			PROPORTION_OF_VARIANCE_TO_SAVE_BEFORE_ICA = 80.0;

    //-----------------------
    // Output parameters
    
	float			*h_Motion_Parameters;
    const char      *outputFilename;
       
    //---------------------
    
    /* Input arguments */
    FILE *fp = NULL; 
    
    // No inputs, so print help text
    if (argc == 1)
    {        
        printf("Usage:\n\n");
        printf("ICA input.nii [options]\n\n");
        printf("Options:\n\n");
        printf(" -platform           The OpenCL platform to use (default 0) \n");
        printf(" -device             The OpenCL device to use for the specificed platform (default 0) \n");
        printf(" -var                Proportion of variance to save before ICA (default 80 %%) \n");
		printf(" -mask               Provide a spatial mask (default false) \n");
		printf(" -zscore             Z-score each time series before ICA (default false) \n");
		printf(" -cpu	             Use the CPU only (default false) \n");
		printf(" -double             Use double precision (default false) \n");
        printf(" -output             Set output filename (default input_ica.nii) \n");
        printf(" -quiet              Don't print anything to the terminal (default false) \n");
        printf(" -verbose            Print extra stuff (default false) \n");
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
        else if (strcmp(input,"-var") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -var !\n");
                return EXIT_FAILURE;
			}
            
            PROPORTION_OF_VARIANCE_TO_SAVE_BEFORE_ICA = (float)strtod(argv[i+1], &p);
            
			if (!isspace(*p) && *p != 0)
		    {
		        printf("Variance proportion must be a float! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
  			if ( PROPORTION_OF_VARIANCE_TO_SAVE_BEFORE_ICA <= 0.0f )
            {
                printf("Variance proportion must be > 0.0 !\n");
                return EXIT_FAILURE;
            }
  			else if ( PROPORTION_OF_VARIANCE_TO_SAVE_BEFORE_ICA >= 100.0f )
            {
                printf("Variance proportion must be < 100.0 !\n");
                return EXIT_FAILURE;
            }

            i += 2;
        }        
        else if (strcmp(input,"-mask") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read name after -mask !\n");
                return EXIT_FAILURE;
			}
            
			AUTO_MASK = false;
			MASK = true;
            MASK_NAME = argv[i+1];
            i += 2;
        }
        else if (strcmp(input,"-zscore") == 0)
        {
            Z_SCORE = true;
            i += 1;
        }
        else if (strcmp(input,"-cpu") == 0)
        {
            CPU = true;
            i += 1;
        }
        else if (strcmp(input,"-double") == 0)
        {
            DOUBLEPRECISION = true;
            i += 1;
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

	// -----------------------    
    // Read mask
	// -----------------------

    nifti_image *inputMask;
    if (MASK)
    {
        inputMask = nifti_image_read(MASK_NAME,1);
        if (inputMask == NULL)
        {
            printf("Could not open mask volume!\n");
            FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
            return EXIT_FAILURE;
        }
        allNiftiImages[numberOfNiftiImages] = inputMask;
        numberOfNiftiImages++;
    }
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

	// Check if mask volume has the same dimensions as the data
	if (MASK)
	{
		size_t TEMP_DATA_W = inputMask->nx;
		size_t TEMP_DATA_H = inputMask->ny;
		size_t TEMP_DATA_D = inputMask->nz;

		if ( (TEMP_DATA_W != DATA_W) || (TEMP_DATA_H != DATA_H) || (TEMP_DATA_D != DATA_D) )
		{
			printf("Input data has the dimensions %zu x %zu x %zu, while the mask volume has the dimensions %zu x %zu x %zu. Aborting! \n",DATA_W,DATA_H,DATA_D,TEMP_DATA_W,TEMP_DATA_H,TEMP_DATA_D);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}
	}

	// Get voxel sizes
	EPI_VOXEL_SIZE_X = inputData->dx;
	EPI_VOXEL_SIZE_Y = inputData->dy;
	EPI_VOXEL_SIZE_Z = inputData->dz;

	// Check if data has more than one time point
	if (DATA_T <= 1)
	{
		printf("Input data has only one volume, cannot do ICA!\n");
	}
   	
    // Calculate size, in bytes
    size_t DATA_SIZE = DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float);
    size_t VOLUME_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);

    // Print some info
    if (PRINT)
    {
        printf("Authored by K.A. Eklund \n");
        printf("Data size: %zu x %zu x %zu x %zu \n",  DATA_W, DATA_H, DATA_D, DATA_T);
        printf("Voxel size: %f x %f x %f mm \n", EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);   
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
	AllocateMemory(h_EPI_Mask, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "EPI_MASK");

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
        printf("Unknown data type in input data, aborting!\n");
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
    

	// Mask is provided by user
	if (MASK)
	{
	    if ( inputMask->datatype == DT_SIGNED_SHORT )
	    {
	        short int *p = (short int*)inputMask->data;
    
	        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
	        {
	            h_EPI_Mask[i] = (float)p[i];
	        }
	    }
	    else if ( inputMask->datatype == DT_UINT16 )
	    {
	        unsigned short int *p = (unsigned short int*)inputMask->data;
    
	        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        	{
	            h_EPI_Mask[i] = (float)p[i];
	        }
	    }
	    else if ( inputMask->datatype == DT_FLOAT )
	    {
	        float *p = (float*)inputMask->data;
    
	        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        	{
	            h_EPI_Mask[i] = p[i];
	        }
	    }
	    else if ( inputMask->datatype == DT_UINT8 )
	    {
    	    unsigned char *p = (unsigned char*)inputMask->data;
    
	        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
	        {
	            h_EPI_Mask[i] = (float)p[i];
	        }
	    }
	    else
	    {
	        printf("Unknown data type in mask volume, aborting!\n");
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
        BROCCOLI.SetEPIWidth(DATA_W);
        BROCCOLI.SetEPIHeight(DATA_H);
        BROCCOLI.SetEPIDepth(DATA_D);
        BROCCOLI.SetEPITimepoints(DATA_T);  

		BROCCOLI.SetAutoMask(AUTO_MASK);
		BROCCOLI.SetZScore(Z_SCORE);

		BROCCOLI.SetOutputEPIMask(h_EPI_Mask);
		
		BROCCOLI.SetAllocatedHostMemory(allocatedHostMemory);
          
		BROCCOLI.SetVarianceToSaveBeforeICA(PROPORTION_OF_VARIANCE_TO_SAVE_BEFORE_ICA);                  
		BROCCOLI.SetNumberOfICAComponents(NUMBER_OF_ICA_COMPONENTS);
   
        // Run the actual ICA
		startTime = GetWallTime();   
		if (DOUBLEPRECISION)
		{
			if (CPU)
			{     
				BROCCOLI.PerformICADoubleCPUWrapper();
			}
			else
			{
				BROCCOLI.PerformICADoubleWrapper();
			}
		}
		else
		{
			if (CPU)
			{     
				BROCCOLI.PerformICACPUWrapper();
			}
			else
			{
				BROCCOLI.PerformICAWrapper();
			}
		}
		endTime = GetWallTime();

		if (VERBOS)
	 	{
			printf("\nIt took %f seconds to run the ICA\n",(float)(endTime - startTime));
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
        
    // Write results to file           
    startTime = GetWallTime();

	nifti_image *outputData = nifti_copy_nim_info(inputData);
	outputData->nt = BROCCOLI.GetNumberOfICAComponents();
	outputData->dim[4] = BROCCOLI.GetNumberOfICAComponents();
	outputData->nvox = DATA_W * DATA_H * DATA_D * BROCCOLI.GetNumberOfICAComponents();
	nifti_free_extensions(outputData);
	allNiftiImages[numberOfNiftiImages] = outputData;
	numberOfNiftiImages++;

	if (!CHANGE_OUTPUT_FILENAME)
	{
	    WriteNifti(outputData,h_fMRI_Volumes,FILENAME_EXTENSION,ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}
	else
	{
		nifti_set_filenames(outputData, outputFilename, 0, 1);
		WriteNifti(outputData,h_fMRI_Volumes,"",DONT_ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
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


