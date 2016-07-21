/*
 * BROCCOLI: Software for fast fMRI analysis on many core CPUs and GPUS
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
#include <sstream>
#include <iomanip>
#include <math.h>

#include "HelpFunctions.cpp"

#define ADD_FILENAME true
#define DONT_ADD_FILENAME true

#define CHECK_EXISTING_FILE true
#define DONT_CHECK_EXISTING_FILE false


double factorial(unsigned long int n)
{
    // Stirling approximation
    return (n == 1 || n == 0) ? 1.0 : round( sqrt(2.0*3.14*(double)n) * pow( ((double)n / 2.7183), double(n) ) );
}

int main(int argc, char **argv)
{
    //-----------------------
    // Input
    
    float           *h_Data, *h_Mask;

    unsigned short int        **h_Permutation_Matrices, *h_Permutation_Matrix;
	float			*h_Sign_Matrix;
    
    float           *h_Correct_Classes, *h_d;
                  
    //-----------------------
    // Output
    
    int             *h_Cluster_Indices, *h_Cluster_Indices_Out;
    float           *h_Permutation_Distribution;
    float           *h_Classifier_Weights, *h_Classifier_Performance, *h_P_Values;

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
    bool            PRINT = true;
	bool			VERBOS = false;
   	bool			CHANGE_OUTPUT_NAME = false;    
                   
    float           CLUSTER_DEFINING_THRESHOLD = 2.5f;
	size_t			NUMBER_OF_PERMUTATIONS = 5000;
	float			SIGNIFICANCE_LEVEL = 0.05f;
	int				INFERENCE_MODE = 1;
	bool			MASK = false;
	const char*		MASK_NAME;
	const char*		CLASS_FILE;
	const char* 	PERMUTATION_INPUT_FILE;
	const char* 	PERMUTATION_VALUES_FILE;
	const char* 	PERMUTATION_VECTORS_FILE;

	bool FOUND_CLASSES = false;
	bool USE_PERMUTATION_FILE = false;
	bool WRITE_PERMUTATION_VALUES = false;
	bool WRITE_PERMUTATION_VECTORS = false;
	bool DO_ALL_PERMUTATIONS = false;
	int	 NUMBER_OF_STATISTICAL_MAPS = 1;

	const char*		outputFilename;

    // Size parameters
    size_t  DATA_W, DATA_H, DATA_D, NUMBER_OF_VOLUMES;
    
    //---------------------    
    
    /* Input arguments */
    FILE *fp = NULL; 
    
    // No inputs, so print help text
    if (argc == 1)
    {   
		printf("\nThe function runs the searchlight algorithm in parallel for all voxels (in mask).\n\n");
        printf("General usage:\n\n");
        printf("Searchlight volumes.nii -classes classes.txt [options]\n\n");
        printf("Options:\n\n");
        printf(" -platform                  The OpenCL platform to use (default 0) \n");
        printf(" -device                    The OpenCL device to use for the specificed platform (default 0) \n");
        printf(" -classes                   Classes for training and testing of the classifier \n");
        printf(" -mask                      A mask that defines which voxels to analyze (default none) \n");
        //printf(" -radius                  Radius of search light (default 1 = 7 voxels) \n");
        //printf(" -classifier              Classifier to use, 0 = neural network, 1 = SVM (default 1) \n");
        //printf(" -inferencemode             Inference mode to use, 0 = voxel, 1 = cluster extent, 2 = cluster mass, 3 = TFCE (default 1) \n");
        //printf(" -cdt                       Cluster defining threshold for cluster inference (default 2.5) \n");
        //printf(" -significance              The significance level to calculate the threshold for (default 0.05) \n");
		//printf(" -output                    Set output filename (default volumes_perm_tvalues.nii and volumes_perm_pvalues.nii) \n");
		//printf(" -writepermutationvalues    Write all the permutation values to a text file \n");
		//printf(" -writepermutations         Write all the random permutations (or sign flips) to a text file \n");
		//printf(" -permutationfile           Use a specific permutation file or sign flipping file (e.g. from FSL) \n");
        printf(" -quiet                     Don't print anything to the terminal (default false) \n");
        printf(" -verbose                   Print extra stuff (default false) \n");
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
        else if (strcmp(input,"-classes") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -classes !\n");
                return EXIT_FAILURE;
			}

            CLASS_FILE = argv[i+1];
			FOUND_CLASSES = true;
            i += 2;
        }
        else if (strcmp(input,"-permutations") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -permutations !\n");
                return EXIT_FAILURE;
			}

            NUMBER_OF_PERMUTATIONS = (int)strtol(argv[i+1], &p, 10);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("Number of permutations must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
            else if (NUMBER_OF_PERMUTATIONS <= 0)
            {
                printf("Number of permutations must be > 0!\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        else if (strcmp(input,"-inferencemode") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -inferencemode !\n");
                return EXIT_FAILURE;
			}

            INFERENCE_MODE = (int)strtol(argv[i+1], &p, 10);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("Inference mode must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
            else if ( (INFERENCE_MODE != 0) && (INFERENCE_MODE != 1) && (INFERENCE_MODE != 2) && (INFERENCE_MODE != 3) )
            {
                printf("Inference mode must be 0, 1, 2 or 3 !\n");
                return EXIT_FAILURE;
            }
            i += 2;

			if (INFERENCE_MODE == 3)
			{
				printf("TFCE is currently turned off!\n");
    	        return EXIT_FAILURE;
			}
        }
        else if (strcmp(input,"-cdt") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -cdt !\n");
                return EXIT_FAILURE;
			}

            CLUSTER_DEFINING_THRESHOLD = (float)strtod(argv[i+1], &p);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("Cluster defining threshold must be a float! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
            i += 2;
        }
        else if (strcmp(input,"-significance") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -significance !\n");
                return EXIT_FAILURE;
			}

            SIGNIFICANCE_LEVEL = (float)strtod(argv[i+1], &p);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("Significance level must be a float! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
			if ( (SIGNIFICANCE_LEVEL <= 0.0f) || (SIGNIFICANCE_LEVEL >= 1.0f) )
		    {
				float zero = 0.0f;
				float one = 1.0f;
		        printf("Significance level must be between %f and %f ! You provided %f \n",zero,one,SIGNIFICANCE_LEVEL);
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

			MASK = true;
            MASK_NAME = argv[i+1];
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

			CHANGE_OUTPUT_NAME = true;
            outputFilename = argv[i+1];
            i += 2;
        }
        else if (strcmp(input,"-writepermutationvalues") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read name after -writepermutationvalues !\n");
                return EXIT_FAILURE;
			}

			WRITE_PERMUTATION_VALUES = true;
            PERMUTATION_VALUES_FILE = argv[i+1];
            i += 2;
        }
        else if (strcmp(input,"-writepermutations") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read name after -writepermutations !\n");
                return EXIT_FAILURE;
			}

			WRITE_PERMUTATION_VECTORS = true;
            PERMUTATION_VECTORS_FILE = argv[i+1];
            i += 2;
        }
        else if (strcmp(input,"-permutationfile") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read name after -permutationfile !\n");
                return EXIT_FAILURE;
			}

			USE_PERMUTATION_FILE = true;
            PERMUTATION_INPUT_FILE = argv[i+1];
            i += 2;
        }
        else
        {
            printf("Unrecognized option! %s \n",argv[i]);
            return EXIT_FAILURE;
        }                
    }

	if (!FOUND_CLASSES)
	{
    	printf("No class file detected, aborting! \n");
        return EXIT_FAILURE;
	}

	// Check if BROCCOLI_DIR variable is set
	if (getenv("BROCCOLI_DIR") == NULL)
	{
        printf("The environment variable BROCCOLI_DIR is not set!\n");
        return EXIT_FAILURE;
	}

	//-------------------------------

	double startTime = GetWallTime();
    
    // Read data

    nifti_image *inputData = nifti_image_read(argv[1],1);
    
    if (inputData == NULL)
    {
        printf("Could not open volumes!\n");
        return EXIT_FAILURE;
    }
	allNiftiImages[numberOfNiftiImages] = inputData;
	numberOfNiftiImages++;
    
	nifti_image *inputMask;
	if (MASK)
	{
	    inputMask = nifti_image_read(MASK_NAME,1);
    
	    if (inputMask == NULL)
	    {
        	printf("Could not open mask volume!\n");
	        return EXIT_FAILURE;
	    }
		allNiftiImages[numberOfNiftiImages] = inputMask;
		numberOfNiftiImages++;
	}
	else
	{
       	printf("\nWarning: No mask being used, doing analysis for all voxels.\n\n");
	}
    	
	double endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to read the nifti file(s)\n",(float)(endTime - startTime));
	}

    // Get data dimensions from input data
   	DATA_W = inputData->nx;
    DATA_H = inputData->ny;
    DATA_D = inputData->nz;    
    NUMBER_OF_VOLUMES = inputData->nt;

	// Check if there is more than one volume
	if (NUMBER_OF_VOLUMES <= 1)
	{
		printf("Input data is a single volume! \n");
		FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
		return EXIT_FAILURE;	
	}
    	
	// Check if mask volume has the same dimensions as the data
	if (MASK)
	{
		size_t TEMP_DATA_W = inputMask->nx;
		size_t TEMP_DATA_H = inputMask->ny;
		size_t TEMP_DATA_D = inputMask->nz;

		if ( (TEMP_DATA_W != DATA_W) || (TEMP_DATA_H != DATA_H) || (TEMP_DATA_D != DATA_D) )
		{
			printf("Input data has the dimensions %zu x %zu %zu, while the mask volume has the dimensions %zu x %zu x %zu. Aborting! \n",DATA_W,DATA_H,DATA_D,TEMP_DATA_W,TEMP_DATA_H,TEMP_DATA_D);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}
	}
    
    // ------------------------------------------------
    
	// Read number of subjects from class file

	std::ifstream design;
    design.open(CLASS_FILE);

    if (!design.good())
    {
        design.close();
        printf("Unable to open class file %s. Aborting! \n",CLASS_FILE);
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }

    // Get number of volumes
    std::string tempString;
    int tempNumber;
    design >> tempString; // NumVolumes as string
    std::string NV("NumVolumes");
    if (tempString.compare(NV) != 0)
    {
        design.close();
        printf("First element of the class file should be the string 'NumVolumes', but it is %s. Aborting! \n",tempString.c_str());
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
    design >> tempNumber;

    if (tempNumber <= 0)
    {
        design.close();
        printf("Number of volumes must be > 0 ! You provided %i in the class file. Aborting! \n",tempNumber);
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
	
    // Check for consistency
    if ( tempNumber != NUMBER_OF_VOLUMES )
    {
        design.close();
        printf("Input data contains %zu volumes, while the class file says %i volumes. Aborting! \n",NUMBER_OF_VOLUMES,tempNumber);
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
		
	
    // ------------------------------------------------

    // Calculate size, in bytes 
    size_t DATA_SIZE = DATA_W * DATA_H * DATA_D * NUMBER_OF_VOLUMES * sizeof(float);
    size_t VOLUME_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);
    size_t CLASS_SIZE = NUMBER_OF_VOLUMES * sizeof(float);
                        
    // ------------------------------------------------

    // Allocate memory on the host

	startTime = GetWallTime();
    
	AllocateMemory(h_Data, DATA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "INPUT_DATA");
	AllocateMemory(h_Mask, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "MASK");
    AllocateMemory(h_Classifier_Performance, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "CLASSIFIER_PERFORMANCE");
	AllocateMemory(h_Correct_Classes, CLASS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "CLASSES");
    AllocateMemory(h_d, CLASS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "D");
                        
	//AllocateMemory(h_P_Values, STATISTICAL_MAPS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "PERMUTATION_PVALUES");

	//h_Permutation_Distributions = (float**)malloc(NUMBER_OF_CONTRASTS * sizeof(float*));
	//h_Permutation_Matrices = (unsigned short int**)malloc(NUMBER_OF_CONTRASTS * sizeof(unsigned short int*));

	endTime = GetWallTime();
    
	if (VERBOS)
 	{
		printf("It took %f seconds to allocate memory\n",(float)(endTime - startTime));
	}

	startTime = GetWallTime();

    // ------------------------------------------------
	// Read classes from file
    // ------------------------------------------------

    // Read design matrix from file, should check for errors
    float tempFloat;
    for (size_t v = 0; v < NUMBER_OF_VOLUMES; v++)
    {
        if (! (design >> h_Correct_Classes[v]) )
        {
            design.close();
            printf("Could not read all values of the class file %s, aborting! Please check if the number of volumes are correct. \n",CLASS_FILE);
            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
            FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
            return EXIT_FAILURE;
        }
    }
    design.close();

	int uncensoredVolumes = 0;

    for (size_t v = 0; v < NUMBER_OF_VOLUMES; v++)
    {
        if (h_Correct_Classes[v] == 0.0f)
        {
            h_d[v] = 1.0f;
        }
        else
        {
            h_d[v] = -1.0f;
        }

		if (h_Correct_Classes[v] != 9999.0f)
		{
			uncensoredVolumes++;
		}
    }

	
	
    NUMBER_OF_STATISTICAL_MAPS = 1;
	

    // ------------------------------------------------

    // ------------------------------------------------

	// Read data

    // Convert data to floats
    if ( inputData->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D * NUMBER_OF_VOLUMES; i++)
        {
            h_Data[i] = (float)p[i];
        }
    }
	else if ( inputData->datatype == DT_UINT8 )
    {
        unsigned char *p = (unsigned char*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D * NUMBER_OF_VOLUMES; i++)
        {
            h_Data[i] = (float)p[i];
        }
    }
    else if ( inputData->datatype == DT_UINT16 )
    {
        unsigned short int *p = (unsigned short int*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D * NUMBER_OF_VOLUMES; i++)
        {
            h_Data[i] = (float)p[i];
        }
    }
    else if ( inputData->datatype == DT_FLOAT )
    {
        float *p = (float*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D * NUMBER_OF_VOLUMES; i++)
        {
            h_Data[i] = p[i];
        }
    }
    else
    {
        printf("Unknown data type in input data, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
		FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
    
	int maskVoxels = 0;

	// Mask is provided by user
	if (MASK)
	{
	    if ( inputMask->datatype == DT_SIGNED_SHORT )
	    {
	        short int *p = (short int*)inputMask->data;
    
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
	    else if ( inputMask->datatype == DT_UINT8 )
	    {
    	    unsigned char *p = (unsigned char*)inputMask->data;
    
	        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
	        {
	            h_Mask[i] = (float)p[i];
	        }
	    }
	    else
	    {
	        printf("Unknown data type in mask volume, aborting!\n");
	        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
	        return EXIT_FAILURE;
	    }

        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        {
            if (h_Mask[i] == 1.0f)
			{
				maskVoxels++;
			}
        }

	}
	// Mask is NOT provided by user, set all mask voxels to 1
	else
	{
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        {
            h_Mask[i] = 1.0f;
        }
	}

    // Print some info
    if (PRINT)
    {
        printf("Authored by K.A. Eklund \n");
        printf("Data size: %zu x %zu x %zu x %zu \n",  DATA_W, DATA_H, DATA_D, NUMBER_OF_VOLUMES);
	    printf("Uncensored volumes: %i\n",uncensoredVolumes);
		if (MASK)
		{
			printf("Voxels in mask: %i\n",maskVoxels);
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
        BROCCOLI.SetInputFirstLevelResults(h_Data);
        BROCCOLI.SetInputMNIBrainMask(h_Mask);        
        BROCCOLI.SetMNIWidth(DATA_W);
        BROCCOLI.SetMNIHeight(DATA_H);
        BROCCOLI.SetMNIDepth(DATA_D);                
        BROCCOLI.SetNumberOfSubjects(NUMBER_OF_VOLUMES);
        
		BROCCOLI.SetAllocatedHostMemory(allocatedHostMemory);

        BROCCOLI.SetInferenceMode(INFERENCE_MODE);        
        BROCCOLI.SetClusterDefiningThreshold(CLUSTER_DEFINING_THRESHOLD);
        BROCCOLI.SetSignificanceLevel(SIGNIFICANCE_LEVEL);		
        
        //BROCCOLI.SetNumberOfPermutations(NUMBER_OF_PERMUTATIONS);
        //BROCCOLI.SetNumberOfGroupPermutations(NUMBER_OF_PERMUTATIONS_PER_CONTRAST);
        BROCCOLI.SetCorrectClasses(h_Correct_Classes, h_d);
        
        BROCCOLI.SetOutputStatisticalMapsMNI(h_Classifier_Performance);
        //BROCCOLI.SetOutputPermutationDistributions(h_Permutation_Distributions);
        //BROCCOLI.SetOutputPValuesMNI(h_P_Values);

		//BROCCOLI.SetPermutationFileUsage(USE_PERMUTATION_FILE);
		BROCCOLI.SetPrint(PRINT);

        // Run the permutation test

		startTime = GetWallTime();
        BROCCOLI.PerformSearchlightWrapper();
		endTime = GetWallTime();

		if (VERBOS)
	 	{
			printf("\nIt took %f seconds to run the searchlight\n",(float)(endTime - startTime));
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
       




    // Create new nifti image
	nifti_image *outputNifti = nifti_copy_nim_info(inputData);      
	nifti_free_extensions(outputNifti);

    // Change number of output volumes
    outputNifti->nt = 1;
    outputNifti->dim[4] = 1;
    outputNifti->nvox = DATA_W * DATA_H * DATA_D;
	
                        
	allNiftiImages[numberOfNiftiImages] = outputNifti;
	numberOfNiftiImages++;    

	if (!CHANGE_OUTPUT_NAME)
	{
    	nifti_set_filenames(outputNifti, inputData->fname, 0, 1);    
	}
	else
	{
		nifti_set_filenames(outputNifti, outputFilename, 0, 1);    
	}

    startTime = GetWallTime(); 
        
    WriteNifti(outputNifti,h_Classifier_Performance,"_classifier_performance",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);

	endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to write the nifti file(s)\n",(float)(endTime - startTime));
	}

    // Free all memory
    FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
    FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);

    return EXIT_SUCCESS;
}



