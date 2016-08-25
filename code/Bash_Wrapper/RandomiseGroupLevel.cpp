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
    
    float           *h_First_Level_Results, *h_Mask; 

    unsigned short int        **h_Permutation_Matrices, *h_Permutation_Matrix;
	float			*h_Sign_Matrix;
    
    float           *h_X_GLM, *h_xtxxt_GLM, *h_Contrasts, *h_ctxtxc_GLM;  
                  
    //-----------------------
    // Output
    
    int             *h_Cluster_Indices, *h_Cluster_Indices_Out;
    float           **h_Permutation_Distributions, *h_Permutation_Distribution;
    float           *h_Beta_Volumes, *h_Residuals, *h_Residual_Variances, *h_Statistical_Maps, *h_P_Values;        
    float           *h_Permuted_First_Level_Results;

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
                   
    size_t          NUMBER_OF_GLM_REGRESSORS = 1;
	size_t			NUMBER_OF_CONTRASTS = 1; 
    float           CLUSTER_DEFINING_THRESHOLD = 2.5f;
	size_t			NUMBER_OF_PERMUTATIONS = 5000;
	size_t			NUMBER_OF_PERMUTATIONS_PER_CONTRAST[1000];
	float			SIGNIFICANCE_LEVEL = 0.05f;
	int				STATISTICAL_TEST = 0;
	int				INFERENCE_MODE = 1;
	bool			MASK = false;
	const char*		MASK_NAME;
	const char*		DESIGN_FILE;        
	const char*		CONTRASTS_FILE;
	const char* 	PERMUTATION_INPUT_FILE;
	const char* 	PERMUTATION_VALUES_FILE;
	const char* 	PERMUTATION_VECTORS_FILE;

	bool FOUND_DESIGN = false;
	bool FOUND_CONTRASTS = false;
	bool ANALYZE_GROUP_MEAN = false;
	bool ANALYZE_TTEST = false;
	bool ANALYZE_FTEST = false;
	bool CORRELATION_DESIGN[1000];
	bool TWOSAMPLE_DESIGN[1000];
	bool MEAN_DESIGN[1000];
	int  GROUP_DESIGNS[1000];
	bool USE_PERMUTATION_FILE = false;
	bool WRITE_PERMUTATION_VALUES = false;
	bool WRITE_PERMUTATION_VECTORS = false;
	bool DO_ALL_PERMUTATIONS = false;
	int	 NUMBER_OF_STATISTICAL_MAPS = 1;

	for (int i = 0; i < 1000; i++)
	{
		TWOSAMPLE_DESIGN[i] = false;
		CORRELATION_DESIGN[i] = false;
		MEAN_DESIGN[i] = false;
		GROUP_DESIGNS[i] = 100;
		NUMBER_OF_PERMUTATIONS_PER_CONTRAST[i] = 5000;
	}

	const char*		outputFilename;

    // Size parameters
    size_t  DATA_W, DATA_H, DATA_D, NUMBER_OF_SUBJECTS;
	int NUMBER_OF_SUBJECTS_IN_GROUP1[1000], NUMBER_OF_SUBJECTS_IN_GROUP2[1000];
        
    //---------------------    
    
    /* Input arguments */
    FILE *fp = NULL; 
    
    // No inputs, so print help text
    if (argc == 1)
    {   
		printf("\nThe function performs permutation testing for group analyses.\n\n");     
        printf("General usage:\n\n");
        printf("RandomiseGroupLevel volumes.nii -design design.mat -contrasts design.con [options]\n\n");
        printf("Testing a group mean:\n\n");
        printf("RandomiseGroupLevel volumes.nii -groupmean [options]\n\n");
        printf("Options:\n\n");
        printf(" -platform                  The OpenCL platform to use (default 0) \n");
        printf(" -device                    The OpenCL device to use for the specificed platform (default 0) \n");
        printf(" -design                    The design matrix to apply in each permutation \n");
        printf(" -contrasts                 The contrast vector(s) to apply to the estimated beta values \n");
	    printf(" -groupmean                 Test for group mean, using sign flipping (design and contrast not needed) \n");
        printf(" -mask                      A mask that defines which voxels to permute (default none) \n");
        printf(" -permutations              Number of permutations to use (default 5,000) \n");
        printf(" -teststatistics            Test statistics to use, 0 = GLM t-test, 1 = GLM F-test  (default 0) \n");
        printf(" -inferencemode             Inference mode to use, 0 = voxel, 1 = cluster extent, 2 = cluster mass, 3 = TFCE (default 1) \n");
        printf(" -cdt                       Cluster defining threshold for cluster inference (default 2.5) \n");
        printf(" -significance              The significance level to calculate the threshold for (default 0.05) \n");		
		printf(" -output                    Set output filename (default volumes_perm_tvalues.nii and volumes_perm_pvalues.nii) \n");
		printf(" -writepermutationvalues    Write all the permutation values to a text file \n");
		printf(" -writepermutations         Write all the random permutations (or sign flips) to a text file \n");
		printf(" -permutationfile           Use a specific permutation file or sign flipping file (e.g. from FSL) \n");
        printf(" -quiet                     Don't print anything to the terminal (default false) \n");
        printf(" -verbose                   Print extra stuff (default false) \n");
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
        else if (strcmp(input,"-design") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -design !\n");
                return EXIT_FAILURE;
			}

            DESIGN_FILE = argv[i+1];
			FOUND_DESIGN = true;
            i += 2;
        }
        else if (strcmp(input,"-contrasts") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -contrasts !\n");
                return EXIT_FAILURE;
			}

            CONTRASTS_FILE = argv[i+1];
			FOUND_CONTRASTS = true;
            i += 2;
        }
        else if (strcmp(input,"-groupmean") == 0)
        {
			ANALYZE_GROUP_MEAN = true;
			FOUND_DESIGN = true;
			FOUND_CONTRASTS = true;
            i += 1;
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
        else if (strcmp(input,"-teststatistics") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -teststatistics !\n");
                return EXIT_FAILURE;
			}

            STATISTICAL_TEST = (int)strtol(argv[i+1], &p, 10);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("Test statistics must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
            if ((STATISTICAL_TEST != 0) && (STATISTICAL_TEST != 1))
            {
                printf("Test statistics must be 0 or 1!\n");
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

	if (!FOUND_DESIGN)
	{
    	printf("No design file detected, aborting! \n");
        return EXIT_FAILURE;
	}

	if (!FOUND_CONTRASTS)
	{
    	printf("No contrasts file detected, aborting! \n");
        return EXIT_FAILURE;
	}

	if (ANALYZE_GROUP_MEAN && (STATISTICAL_TEST == 1))
	{
    	printf("Cannot use F-test for group mean, aborting! \n");
        return EXIT_FAILURE;
	}

	// Check if BROCCOLI_DIR variable is set
	if (getenv("BROCCOLI_DIR") == NULL)
	{
        printf("The environment variable BROCCOLI_DIR is not set!\n");
        return EXIT_FAILURE;
	}

	if (STATISTICAL_TEST == 0)
	{
		ANALYZE_TTEST = true;
	}
	else if (STATISTICAL_TEST == 1)
	{
		ANALYZE_FTEST = true;
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
		// Check that file extension is .nii or .nii.gz
		std::string extension;
		bool extensionOK;
		CheckFileExtension(MASK_NAME,extensionOK,extension);
		if (!extensionOK)
		{
            printf("File extension is not .nii or .nii.gz, %s is not allowed!\n",extension.c_str());
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
            return EXIT_FAILURE;
		}

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
	else
	{
       	printf("\nWarning: No mask being used, doing permutations for all voxels.\n\n");
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
    NUMBER_OF_SUBJECTS = inputData->nt;    

	// Check if there is more than one volume
	if (NUMBER_OF_SUBJECTS <= 1)
	{
		printf("Input data is a single volume, nothing to permute! \n");
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
    
	// Read number of regressors and number of subjects from design matrix file

	std::ifstream design;
	std::ifstream contrasts;

	if (!ANALYZE_GROUP_MEAN)
	{
	    design.open(DESIGN_FILE); 

		if (!design.good())
		{
			design.close();
			printf("Unable to open design file %s. Aborting! \n",DESIGN_FILE);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}

		// Get number of regressors
		std::string tempString;
		int tempNumber;
		design >> tempString; // NumRegressors as string
		std::string NR("NumRegressors");
		if (tempString.compare(NR) != 0)
		{
			design.close();
			printf("First element of the design file should be the string 'NumRegressors', but it is %s. Aborting! \n",tempString.c_str());
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}
		design >> NUMBER_OF_GLM_REGRESSORS;
	
		if (NUMBER_OF_GLM_REGRESSORS <= 0)
		{
			design.close();
			printf("Number of regressors must be > 0 ! You provided %zu regressors in the design file. Aborting! \n",NUMBER_OF_GLM_REGRESSORS);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}
		else if (NUMBER_OF_GLM_REGRESSORS > 25)
		{
			design.close();
			printf("Number of regressors must be <= 25 ! You provided %zu regressors in the design file. Aborting! \n",NUMBER_OF_GLM_REGRESSORS);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}
		else if (NUMBER_OF_GLM_REGRESSORS > NUMBER_OF_SUBJECTS)
		{
			design.close();
			printf("Number of regressors must be smaller or equal to the number of subjects! You provided %zu regressors in the design file and there are %zu subjects. Aborting! \n",NUMBER_OF_GLM_REGRESSORS,NUMBER_OF_SUBJECTS);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}

		// Get number of subjects
		design >> tempString; // NumSubjects as string
		std::string NS("NumSubjects");
		if (tempString.compare(NS) != 0)
		{
			design.close();
			printf("Third element of the design file should be the string 'NumSubjects', but it is %s. Aborting! \n",tempString.c_str());
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}
		design >> tempNumber;

		if (tempNumber <= 0)
		{
			design.close();
			printf("Number of subjects must be > 0 ! You provided %i in the design file. Aborting! \n",tempNumber);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}
	
		// Check for consistency
		if ( tempNumber != NUMBER_OF_SUBJECTS )
		{
			design.close();
			printf("Input data contains %zu volumes, while the design file says %i subjects. Aborting! \n",NUMBER_OF_SUBJECTS,tempNumber);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}
		
	    // ------------------------------------------------
  
		// Read number of regressors and number of contrasts from contrasts file

	    contrasts.open(CONTRASTS_FILE); 

		if (!contrasts.good())
		{
			contrasts.close();
			printf("Unable to open contrasts file %s. Aborting! \n",CONTRASTS_FILE);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}

		// Get number of regressors and number of subjects
		contrasts >> tempString; // NumRegressors as string	
		if (tempString.compare(NR) != 0)
		{
			contrasts.close();
			printf("First element of the contrasts file should be the string 'NumRegressors', but it is %s. Aborting! \n",tempString.c_str());
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}
		contrasts >> tempNumber;
	
		// Check for consistency
		if ( tempNumber != NUMBER_OF_GLM_REGRESSORS )
		{
			contrasts.close();
			printf("Design file says that number of regressors is %zu, while contrast file says there are %i regressors. Aborting! \n",NUMBER_OF_GLM_REGRESSORS,tempNumber);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}

		contrasts >> tempString; // NumContrasts as string
		std::string NC("NumContrasts");
		if (tempString.compare(NC) != 0)
		{
			contrasts.close();
			printf("Third element of the contrasts file should be the string 'NumContrasts', but it is %s. Aborting! \n",tempString.c_str());
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}
		contrasts >> NUMBER_OF_CONTRASTS;
	
		if (NUMBER_OF_CONTRASTS <= 0)
		{
			contrasts.close();
			printf("Number of contrasts must be > 0 ! You provided %zu in the contrasts file. Aborting! \n",NUMBER_OF_CONTRASTS);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}
		if ((STATISTICAL_TEST == 1) && (NUMBER_OF_CONTRASTS > 10))
		{
			contrasts.close();
			printf("Number of contrasts must currently be <= 10 for F-test! You provided %zu contrasts in the contrasts file. Aborting! \n",NUMBER_OF_CONTRASTS);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}
	}
		
    // ------------------------------------------------

    // Print some info
    if (PRINT)
    {
        printf("Authored by K.A. Eklund \n");
        printf("Data size: %zu x %zu x %zu x %zu \n",  DATA_W, DATA_H, DATA_D, NUMBER_OF_SUBJECTS);
        printf("Number of permutations: %zu \n",  NUMBER_OF_PERMUTATIONS);
        printf("Number of regressors: %zu \n",  NUMBER_OF_GLM_REGRESSORS);
        printf("Number of contrasts: %zu \n",  NUMBER_OF_CONTRASTS);
		if (ANALYZE_TTEST)
		{
			printf("Performing %zu t-tests \n",  NUMBER_OF_CONTRASTS);
		}
		else if (ANALYZE_FTEST)
		{
			printf("Performing one F-test \n");
		}
    }
	if (VERBOS)
	{
		printf("Using a cluster defining threshold of %f \n",CLUSTER_DEFINING_THRESHOLD);
	}
	
    // ------------------------------------------------

    // Calculate size, in bytes 
    size_t DATA_SIZE = DATA_W * DATA_H * DATA_D * NUMBER_OF_SUBJECTS * sizeof(float);
    size_t VOLUME_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);
  	size_t GLM_SIZE = NUMBER_OF_SUBJECTS * NUMBER_OF_GLM_REGRESSORS * sizeof(float);
    size_t CONTRAST_SIZE = NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float);

    size_t STATISTICAL_MAPS_SIZE, CONTRAST_SCALAR_SIZE;
	if (ANALYZE_TTEST)
	{
		CONTRAST_SCALAR_SIZE = NUMBER_OF_CONTRASTS * sizeof(float);
		STATISTICAL_MAPS_SIZE = DATA_W * DATA_H * DATA_D * NUMBER_OF_CONTRASTS * sizeof(float);
	}
	else if (ANALYZE_FTEST)
	{
		CONTRAST_SCALAR_SIZE = NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float);
		STATISTICAL_MAPS_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);
	}
            
    // ------------------------------------------------

    // Allocate memory on the host

	startTime = GetWallTime();
    
	AllocateMemory(h_First_Level_Results, DATA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "INPUT_DATA");
	AllocateMemory(h_Mask, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "MASK");
	AllocateMemory(h_X_GLM, GLM_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "DESIGN_MATRIX");
	AllocateMemory(h_xtxxt_GLM, GLM_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "DESIGN_MATRIX_PSEUDO_INVERSE");
	AllocateMemory(h_Contrasts, CONTRAST_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "CONTRASTS");
	AllocateMemory(h_ctxtxc_GLM, CONTRAST_SCALAR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "CONTRAST_SCALARS");
	AllocateMemory(h_Statistical_Maps, STATISTICAL_MAPS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "STATISTICAL_MAPS");             
	AllocateMemory(h_P_Values, STATISTICAL_MAPS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "PERMUTATION_PVALUES");             

	h_Permutation_Distributions = (float**)malloc(NUMBER_OF_CONTRASTS * sizeof(float*));
	h_Permutation_Matrices = (unsigned short int**)malloc(NUMBER_OF_CONTRASTS * sizeof(unsigned short int*));

	endTime = GetWallTime();
    
	if (VERBOS)
 	{
		printf("It took %f seconds to allocate memory\n",(float)(endTime - startTime));
	}

	startTime = GetWallTime();

    // ------------------------------------------------
	// Read design matrix from file
    // ------------------------------------------------

	Eigen::MatrixXd X(NUMBER_OF_SUBJECTS,NUMBER_OF_GLM_REGRESSORS);
	
	if (!ANALYZE_GROUP_MEAN)
    {
		// Read design matrix from file, should check for errors
		float tempFloat;	
		for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
		{
			for (size_t r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				if (! (design >> h_X_GLM[NUMBER_OF_SUBJECTS * r + s]) )
				{
					design.close();
			        printf("Could not read all values of the design file %s, aborting! Please check if the number of regressors and subjects are correct. \n",DESIGN_FILE);      
			        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
			        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			        return EXIT_FAILURE;
				}
			}
		}	
		design.close();

		// Put design matrix into Eigen object 
		for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
		{
			for (size_t r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				X(s,r) = (double)h_X_GLM[s + r * NUMBER_OF_SUBJECTS];
			}
		}
	}
	else if (ANALYZE_GROUP_MEAN)
	{
		// Create regressor with all ones
		for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
		{
			h_X_GLM[s] = 1.0f;
			X(s) = 1.0;
		}
	}

    // ------------------------------------------------
	// Read contrasts from file
    // ------------------------------------------------

	if (!ANALYZE_GROUP_MEAN)
	{
		// Read contrasts from file, should check for errors
		for (size_t c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			for (size_t r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				if (! (contrasts >> h_Contrasts[r + c * NUMBER_OF_GLM_REGRESSORS]) )
				{
					contrasts.close();
			        printf("Could not read all values of the contrasts file %s, aborting! Please check if the number of regressors and contrasts are correct. \n",CONTRASTS_FILE);      
			        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
			        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			        return EXIT_FAILURE;			
				}
			}
		}
		contrasts.close();
	}
	else if (ANALYZE_GROUP_MEAN)
	{
		h_Contrasts[0] = 1.0f;
	}

    // ------------------------------------------------
	// Check if design matrix is two sample test or correlation, for each contrast
    // ------------------------------------------------

	Eigen::MatrixXd Contrasts(NUMBER_OF_CONTRASTS,NUMBER_OF_GLM_REGRESSORS);

	// Put contrast vectors into eigen object
	for (size_t c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		for (size_t r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
		{
			Contrasts(c,r) = h_Contrasts[r + c * NUMBER_OF_GLM_REGRESSORS];		
		}
	}

	if (ANALYZE_FTEST)
	{
		// Check if contrast matrix has full rank
		Eigen::FullPivLU<Eigen::MatrixXd> luA(Contrasts);
		int rank = luA.rank();
		if (rank < NUMBER_OF_CONTRASTS)
		{
	        printf("Contrast matrix does not have full rank, at least one contrast can be written as a linear combination of other contrasts, not OK for F-test, aborting!\n");      
	        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
	        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
	        return EXIT_FAILURE;	
		}
	}

	if (ANALYZE_TTEST)
	{    
		// Calculate current contrast
		for (size_t c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			Eigen::MatrixXd currentContrast = Contrasts.row(c);
			Eigen::MatrixXd currentVector = X * currentContrast.transpose();

			// Check how many unique values there are
			std::vector<float> allUniqueValues;        

	        for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
	        {
				float value = currentVector(s);
				bool unique = true;
				// Check if value is in list of unique values
				for (size_t i = 0; i < allUniqueValues.size(); i++)
				{
					if (value == allUniqueValues[i])
					{
						unique = false;
						break;
					}
				}

				if (unique)
				{
					allUniqueValues.push_back(value);
				}
	 		}
    
			if ( allUniqueValues.size() == 2 )
			{
				TWOSAMPLE_DESIGN[c] = true;
				GROUP_DESIGNS[c] = 0;
			}
			else if ( allUniqueValues.size() > 2 )
			{
				CORRELATION_DESIGN[c] = true;
				GROUP_DESIGNS[c] = 1;
			}
			else if ( allUniqueValues.size() == 1 )
			{
				MEAN_DESIGN[c] = true;
				GROUP_DESIGNS[c] = 2;
			}

			if (TWOSAMPLE_DESIGN[c])
			{
				NUMBER_OF_SUBJECTS_IN_GROUP1[c] = 0;
				for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
				{
					if (currentVector(s) == allUniqueValues[0])
					{
						NUMBER_OF_SUBJECTS_IN_GROUP1[c]++;
					}
				}
				NUMBER_OF_SUBJECTS_IN_GROUP2[c] = NUMBER_OF_SUBJECTS - NUMBER_OF_SUBJECTS_IN_GROUP1[c];
		        printf("Two sample t-test design detected for t-contrast %zu, %i subjects in group 1 and %i subjects in group 2\n",c+1,NUMBER_OF_SUBJECTS_IN_GROUP1[c],NUMBER_OF_SUBJECTS_IN_GROUP2[c]);
			}
			else if (CORRELATION_DESIGN[c])
			{
			   printf("Correlation design detected for t-contrast %zu\n",c+1);
			}	
			else if (MEAN_DESIGN[c])
			{
			   printf("Mean design detected for t-contrast %zu\n",c+1);
			}	

		}
	}
	else if (ANALYZE_FTEST)
	{
		// Calculate design matrix * contrast
		Eigen::MatrixXd temp = X * Contrasts.transpose();
		
		// Check how many unique values there are
		std::vector<float> allUniqueValues;        

        for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
        {
	        for (size_t c = 0; c < NUMBER_OF_CONTRASTS; c++)
    	    {
				float value = temp(s,c);
				bool unique = true;
				// Check if value is in list of unique values
				for (size_t i = 0; i < allUniqueValues.size(); i++)
				{
					if (value == allUniqueValues[i])
					{
						unique = false;
						break;
					}
				}

				if (unique)
				{
					allUniqueValues.push_back(value);
				}
	 		}
		}
    
		if ( allUniqueValues.size() == 2 )
		{
			TWOSAMPLE_DESIGN[0] = true;
			GROUP_DESIGNS[0] = 0;
		}
		else if ( allUniqueValues.size() > 2 )
		{
			CORRELATION_DESIGN[0] = true;
			GROUP_DESIGNS[0] = 1;
		}
		else if ( allUniqueValues.size() == 1 )
		{
			MEAN_DESIGN[0] = true;
			GROUP_DESIGNS[0] = 2;
		}
		

		if (TWOSAMPLE_DESIGN[0])
		{
			NUMBER_OF_SUBJECTS_IN_GROUP1[0] = 0;
			for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
			{
				if (temp(s,0) == allUniqueValues[0])
				{
					NUMBER_OF_SUBJECTS_IN_GROUP1[0]++;
				}
			}
			NUMBER_OF_SUBJECTS_IN_GROUP2[0] = NUMBER_OF_SUBJECTS - NUMBER_OF_SUBJECTS_IN_GROUP1[0];
	        printf("\nTwo sample t-test design detected for F-test, %i subjects in group 1 and %i subjects in group 2\n\n",NUMBER_OF_SUBJECTS_IN_GROUP1[0],NUMBER_OF_SUBJECTS_IN_GROUP2[0]);
		}
		else if (CORRELATION_DESIGN[0])
		{
		   printf("\nCorrelation design detected for F-test\n\n");
		}
	}


	// Check if requested number of permutations is larger than number of possible sign flips
	if (ANALYZE_GROUP_MEAN)
	{
		NUMBER_OF_PERMUTATIONS_PER_CONTRAST[0] = NUMBER_OF_PERMUTATIONS;

		// Calculate maximum number of sign flips
		double MAX_SIGN_FLIPS = pow(2.0, (double)NUMBER_OF_SUBJECTS);
		if ((double)NUMBER_OF_PERMUTATIONS > MAX_SIGN_FLIPS)
		{
			printf("Warning: Number of possible sign flips for group mean is %g, but %g permutations were requested. Lowering number of permutations to number of possible sign flips. \n",MAX_SIGN_FLIPS,(double)NUMBER_OF_PERMUTATIONS);
			NUMBER_OF_PERMUTATIONS_PER_CONTRAST[0] = (int)MAX_SIGN_FLIPS;
			DO_ALL_PERMUTATIONS = true;
		}
		else if ((double)NUMBER_OF_PERMUTATIONS == MAX_SIGN_FLIPS)
		{
			DO_ALL_PERMUTATIONS = true;
			printf("Max number of sign flips is %g \n",MAX_SIGN_FLIPS);
		}
		else
		{
			printf("Max number of sign flips is %g \n",MAX_SIGN_FLIPS);
		}
	}

	if (ANALYZE_GROUP_MEAN)
	{
		NUMBER_OF_STATISTICAL_MAPS = 1;
	}
	if (ANALYZE_TTEST)
	{
		NUMBER_OF_STATISTICAL_MAPS = NUMBER_OF_CONTRASTS;
	}
	else if (ANALYZE_FTEST)
	{
		NUMBER_OF_STATISTICAL_MAPS = 1;
	}

	if (!ANALYZE_GROUP_MEAN)
	{
		for (size_t c = 0; c < NUMBER_OF_STATISTICAL_MAPS; c++)
		{
			NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c] = NUMBER_OF_PERMUTATIONS;

			if (TWOSAMPLE_DESIGN[c])
			{
		       	double MAX_PERMS = round(exp(lgamma(NUMBER_OF_SUBJECTS+1)-lgamma(NUMBER_OF_SUBJECTS-NUMBER_OF_SUBJECTS_IN_GROUP2[c]+1)-lgamma(NUMBER_OF_SUBJECTS_IN_GROUP2[c]+1)));
				if ((double)NUMBER_OF_PERMUTATIONS > MAX_PERMS)
				{
					printf("Warning: Number of possible permutations for your design is %g for contrast %zu, but %g permutations were requested. Lowering number of permutations to number of possible permutations. \n",MAX_PERMS,c+1,(double)NUMBER_OF_PERMUTATIONS);
					NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c] = (int)MAX_PERMS; 
					DO_ALL_PERMUTATIONS = true;
				}
				else if ((double)NUMBER_OF_PERMUTATIONS == MAX_PERMS)
				{
					DO_ALL_PERMUTATIONS = true;
					printf("Max number of permutations for contrast %zu is %g \n",c+1,MAX_PERMS);
				}
				else
				{
					printf("Max number of permutations for contrast %zu is %g \n",c+1,MAX_PERMS);
				}
			}
			else if (CORRELATION_DESIGN[c])
			{
				double MAX_PERMS = factorial(NUMBER_OF_SUBJECTS);
				if ((double)NUMBER_OF_PERMUTATIONS > MAX_PERMS)
				{
					printf("Warning: Number of possible permutations for your design is %g for contrast %zu, but %g permutations were requested. Lowering number of permutations to number of possible permutations. \n",MAX_PERMS,c+1,(double)NUMBER_OF_PERMUTATIONS);
					NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c] = (int)MAX_PERMS; 
					DO_ALL_PERMUTATIONS = true;
				}
				else if ((double)NUMBER_OF_PERMUTATIONS == MAX_PERMS)
				{
					DO_ALL_PERMUTATIONS = true;
					printf("Max number of permutations for contrast %zu is %g \n",c+1,MAX_PERMS);
				}
				else
				{
					printf("Max number of permutations for contrast %zu is %g \n",c+1,MAX_PERMS);
				}
			}
			else if (MEAN_DESIGN[c])
			{
				printf("Warning: Contrast %zu leads to a simple mean value, only doing 1 permutation!\n",c+1);
				NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c] = 1; 
			}
		}
	}


	// Calculate pseudo inverse
	Eigen::MatrixXd xtx(NUMBER_OF_GLM_REGRESSORS,NUMBER_OF_GLM_REGRESSORS);
	xtx = X.transpose() * X;
	Eigen::MatrixXd inv_xtx = xtx.inverse();
	Eigen::MatrixXd xtxxt = inv_xtx * X.transpose();
	
	// Put pseudo inverse into regular array
	for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
	{
		for (size_t r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
		{
			h_xtxxt_GLM[s + r * NUMBER_OF_SUBJECTS] = (float)xtxxt(r,s);
		}
	}

	// Print design matrix
	if (VERBOS)
	{
		for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
		{
			printf("Design matrix is ");
			for (size_t r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				printf(" %f ",h_X_GLM[s + r * NUMBER_OF_SUBJECTS]);
			}
			printf("\n");
		}
	}

	// Calculate contrast scalars
	if (ANALYZE_TTEST)
	{
		for (size_t c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			// Put contrast vector into eigen object
			Eigen::VectorXd Contrast(NUMBER_OF_GLM_REGRESSORS);
			for (size_t r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				Contrast(r) = h_Contrasts[r + c * NUMBER_OF_GLM_REGRESSORS];		
			}
	
			Eigen::VectorXd scalar = Contrast.transpose() * inv_xtx * Contrast;
			h_ctxtxc_GLM[c] = scalar(0);
		}
	}
	else if (ANALYZE_FTEST)
	{
		Eigen::MatrixXd temp = Contrasts * inv_xtx * Contrasts.transpose();
		Eigen::MatrixXd ctxtxc = temp.inverse();

		for (size_t c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			for (size_t cc = 0; cc < NUMBER_OF_CONTRASTS; cc++)
			{
				h_ctxtxc_GLM[c + cc  * NUMBER_OF_CONTRASTS] = ctxtxc(c,cc);
			}
		}
	}

	if (!ANALYZE_GROUP_MEAN)
	{
		// Print contrasts
		if (VERBOS)
		{
			for (size_t c = 0; c < NUMBER_OF_CONTRASTS; c++)
			{
				printf("Contrast is ");
				for (size_t r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
				{
					printf(" %f ",h_Contrasts[r + c * NUMBER_OF_GLM_REGRESSORS]);
				}
				printf("\n");
			}
		}
	}

    // ------------------------------------------------
		
	size_t SIGN_MATRIX_SIZE = NUMBER_OF_PERMUTATIONS_PER_CONTRAST[0] * NUMBER_OF_SUBJECTS * sizeof(float);

	AllocateMemory(h_Sign_Matrix, SIGN_MATRIX_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "SIGN_MATRIX");

	for (size_t c = 0; c < NUMBER_OF_STATISTICAL_MAPS; c++)
	{ 
	    size_t NULL_DISTRIBUTION_SIZE = NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c] * sizeof(float);
		size_t PERMUTATION_MATRIX_SIZE = NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c] * NUMBER_OF_SUBJECTS * sizeof(unsigned short int);

		AllocateMemoryInt(h_Permutation_Matrix, PERMUTATION_MATRIX_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages,allocatedHostMemory, "PERMUTATION_MATRIX");
		AllocateMemory(h_Permutation_Distribution, NULL_DISTRIBUTION_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "PERMUTATION_DISTRIBUTION");             
		h_Permutation_Matrices[c] = h_Permutation_Matrix;
		h_Permutation_Distributions[c] = h_Permutation_Distribution;
	}

    // ------------------------------------------------

	// Read permutation file
	if (USE_PERMUTATION_FILE && (!ANALYZE_GROUP_MEAN))
	{
		h_Permutation_Matrix = h_Permutation_Matrices[0];

		std::ifstream permutations;
    	permutations.open(PERMUTATION_INPUT_FILE); 

		// Should check if number of permutations and number of subjects is OK

		if (permutations.good())
		{
			for (size_t p = 0; p < NUMBER_OF_PERMUTATIONS_PER_CONTRAST[0]; p++)
			{
				for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
				{
					float temp;
					if (permutations >> temp)
					{
						h_Permutation_Matrix[s + p * NUMBER_OF_SUBJECTS] = (unsigned short int)temp;
						h_Permutation_Matrix[s + p * NUMBER_OF_SUBJECTS] -= 1;				
					}
					else
					{
						permutations.close();
				        printf("Could not read all values of the permutation file %s, aborting! \n",PERMUTATION_INPUT_FILE);      
				        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
				        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
				        return EXIT_FAILURE;
					}
				}			
			}
			permutations.close();
		}
		else	
		{
			permutations.close();
	        printf("Could not open permutation file %s, aborting! \n",PERMUTATION_INPUT_FILE);      
	        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
	        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
	        return EXIT_FAILURE;
		}	
	}
	// Read sign flipping file
	else if (USE_PERMUTATION_FILE && ANALYZE_GROUP_MEAN)
	{
		std::ifstream permutations;
    	permutations.open(PERMUTATION_INPUT_FILE); 

		// Should check if number of permutations and number of subjects is OK

		if (permutations.good())
		{
			for (size_t p = 0; p < NUMBER_OF_PERMUTATIONS_PER_CONTRAST[0]; p++)
			{
				for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
				{
					float temp;
					permutations >> temp;
					h_Sign_Matrix[s + p * NUMBER_OF_SUBJECTS] = temp;
				}			
			}
			permutations.close();
		}
		else	
		{
			permutations.close();
	        printf("Could not open sign file %s, aborting! \n",PERMUTATION_INPUT_FILE);      
	        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
	        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
	        return EXIT_FAILURE;
		}	
	}

    // ------------------------------------------------

	// Read data

    // Convert data to floats
    if ( inputData->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D * NUMBER_OF_SUBJECTS; i++)
        {
            h_First_Level_Results[i] = (float)p[i];
        }
    }
	else if ( inputData->datatype == DT_UINT8 )
    {
        unsigned char *p = (unsigned char*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D * NUMBER_OF_SUBJECTS; i++)
        {
            h_First_Level_Results[i] = (float)p[i];
        }
    }
    else if ( inputData->datatype == DT_UINT16 )
    {
        unsigned short int *p = (unsigned short int*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D * NUMBER_OF_SUBJECTS; i++)
        {
            h_First_Level_Results[i] = (float)p[i];
        }
    }
    else if ( inputData->datatype == DT_FLOAT )
    {
        float *p = (float*)inputData->data;
    
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D * NUMBER_OF_SUBJECTS; i++)
        {
            h_First_Level_Results[i] = p[i];
        }
    }
    else
    {
        printf("Unknown data type in input data, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
		FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
    
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
	}
	// Mask is NOT provided by user, set all mask voxels to 1
	else
	{
        for (size_t i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        {
            h_Mask[i] = 1.0f;
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
        BROCCOLI.SetInputFirstLevelResults(h_First_Level_Results);        
        BROCCOLI.SetInputMNIBrainMask(h_Mask);        
        BROCCOLI.SetMNIWidth(DATA_W);
        BROCCOLI.SetMNIHeight(DATA_H);
        BROCCOLI.SetMNIDepth(DATA_D);                

		BROCCOLI.SetAllocatedHostMemory(allocatedHostMemory);

        BROCCOLI.SetInferenceMode(INFERENCE_MODE);        
        BROCCOLI.SetClusterDefiningThreshold(CLUSTER_DEFINING_THRESHOLD);
        BROCCOLI.SetSignificanceLevel(SIGNIFICANCE_LEVEL);		
        BROCCOLI.SetNumberOfSubjects(NUMBER_OF_SUBJECTS);
        BROCCOLI.SetNumberOfSubjectsGroup1(NUMBER_OF_SUBJECTS_IN_GROUP1);
        BROCCOLI.SetNumberOfSubjectsGroup2(NUMBER_OF_SUBJECTS_IN_GROUP2);
        BROCCOLI.SetNumberOfPermutations(NUMBER_OF_PERMUTATIONS);
        BROCCOLI.SetNumberOfGroupPermutations(NUMBER_OF_PERMUTATIONS_PER_CONTRAST);
        BROCCOLI.SetNumberOfGLMRegressors(NUMBER_OF_GLM_REGRESSORS);
        BROCCOLI.SetNumberOfContrasts(NUMBER_OF_CONTRASTS);    
        BROCCOLI.SetDesignMatrix(h_X_GLM, h_xtxxt_GLM);
        BROCCOLI.SetContrasts(h_Contrasts);
        BROCCOLI.SetGLMScalars(h_ctxtxc_GLM);

        //BROCCOLI.SetOutputBetaVolumes(h_Beta_Volumes);        
        //BROCCOLI.SetOutputResiduals(h_Residuals);        
        //BROCCOLI.SetOutputResidualVariances(h_Residual_Variances);        
        BROCCOLI.SetOutputStatisticalMapsMNI(h_Statistical_Maps);        
        //BROCCOLI.SetOutputClusterIndices(h_Cluster_Indices);
        BROCCOLI.SetOutputPermutationDistributions(h_Permutation_Distributions);
        //BROCCOLI.SetOutputPermutedFirstLevelResults(h_Permuted_First_Level_Results);       
        BROCCOLI.SetOutputPValuesMNI(h_P_Values);        

		BROCCOLI.SetDoAllPermutations(DO_ALL_PERMUTATIONS);

		BROCCOLI.SetPermutationFileUsage(USE_PERMUTATION_FILE);
		BROCCOLI.SetPrint(PRINT);

		BROCCOLI.SetGroupDesigns(GROUP_DESIGNS);

        // Run the permutation test

		startTime = GetWallTime();
		if (ANALYZE_GROUP_MEAN)
		{
		    BROCCOLI.SetSignMatrix(h_Sign_Matrix);          
	        BROCCOLI.SetStatisticalTest(2); // Group mean
			BROCCOLI.PerformMeanSecondLevelPermutationWrapper();                            
		}
		else if (ANALYZE_TTEST)
		{
	        BROCCOLI.SetPermutationMatrices(h_Permutation_Matrices);        
	        BROCCOLI.SetStatisticalTest(0); // t-test
	        BROCCOLI.PerformGLMTTestSecondLevelPermutationWrapper();                            
		}
		else if (ANALYZE_FTEST)
		{
	        BROCCOLI.SetPermutationMatrices(h_Permutation_Matrices);        
	        BROCCOLI.SetStatisticalTest(1); // F-test
	        BROCCOLI.PerformGLMFTestSecondLevelPermutationWrapper();                            
		}

		endTime = GetWallTime();

		if (VERBOS)
	 	{
			printf("\nIt took %f seconds to run the permutation test\n",(float)(endTime - startTime));
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
       
	// Print the permutation values to a text file
	if (WRITE_PERMUTATION_VALUES)
	{
		for (size_t c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			h_Permutation_Distribution = h_Permutation_Distributions[c];

			std::ofstream permutationValues;
			std::string permValues(PERMUTATION_VALUES_FILE);
			char tmp[1000];
			sprintf(tmp, "%zu", c+1);
			permValues.insert(permValues.find("."),std::string(tmp));

		    permutationValues.open(permValues.c_str());      

		    if ( permutationValues.good() )
		    {
    		    for (size_t p = 0; p < NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c]; p++)
		        {
    	        	permutationValues << std::setprecision(6) << std::fixed << (double)h_Permutation_Distribution[p] << " " << std::endl;
				}
			    permutationValues.close();
    	    } 	
		    else
		    {
				permutationValues.close();
		        printf("Could not open %s for writing permutation values!\n",permValues.c_str());
		    }
		}
	}

	// Print the permutation vectors or sign flips to a text file
	if (WRITE_PERMUTATION_VECTORS)
	{
		std::ofstream permutationVectors;
	    permutationVectors.open(PERMUTATION_VECTORS_FILE);      

		h_Permutation_Matrix = h_Permutation_Matrices[0];

	    if ( permutationVectors.good() )
	    {
			if (ANALYZE_GROUP_MEAN)
			{
	    	    for (size_t p = 0; p < NUMBER_OF_PERMUTATIONS_PER_CONTRAST[0]; p++)
		        {
		    	    for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
			        {
		            	permutationVectors << std::setprecision(6) << std::fixed << (double)h_Sign_Matrix[s + p * NUMBER_OF_SUBJECTS] << " ";
					}
					permutationVectors << std::endl;
				}
			}
			else
			{
	    	    for (size_t p = 0; p < NUMBER_OF_PERMUTATIONS_PER_CONTRAST[0]; p++)
		        {
		    	    for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
			        {
		            	permutationVectors << std::setprecision(6) << std::fixed << (double)(h_Permutation_Matrix[s + p * NUMBER_OF_SUBJECTS] + 1) << " ";
					}
					permutationVectors << std::endl;
				}
			}
		    permutationVectors.close();
        } 	
	    else
	    {
			permutationVectors.close();
	        printf("Could not open %s for writing permutation vectors!\n",PERMUTATION_VECTORS_FILE);
	    }
	}

    // Create new nifti image
	nifti_image *outputNifti = nifti_copy_nim_info(inputData);      
	nifti_free_extensions(outputNifti);

	if (ANALYZE_TTEST)
	{
		// Change number of output volumes
		outputNifti->nt = NUMBER_OF_CONTRASTS;
		outputNifti->dim[4] = NUMBER_OF_CONTRASTS;
		outputNifti->nvox = DATA_W * DATA_H * DATA_D * NUMBER_OF_CONTRASTS;
	}
	else if (ANALYZE_FTEST)
	{
		// Change number of output volumes
		outputNifti->nt = 1;
		outputNifti->dim[4] = 1;
		outputNifti->nvox = DATA_W * DATA_H * DATA_D;
	}
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
        
	if (!ANALYZE_FTEST)
	{
	    WriteNifti(outputNifti,h_Statistical_Maps,"_perm_tvalues",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}
	else
	{
	    WriteNifti(outputNifti,h_Statistical_Maps,"_perm_fvalues",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}
    WriteNifti(outputNifti,h_P_Values,"_perm_pvalues",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);

	endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to write the nifti file(s)\n",(float)(endTime - startTime));
	}

    // Free all memory
    FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
    FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);

	free(h_Permutation_Distributions);
	free(h_Permutation_Matrices);
        
    return EXIT_SUCCESS;
}



