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
#include <time.h>
#include <sys/time.h>

#define ADD_FILENAME true
#define DONT_ADD_FILENAME true

//#define HAVE_ZLIB 1

#define CHECK_EXISTING_FILE true
#define DONT_CHECK_EXISTING_FILE false

void ConvertFloat2ToFloats(float* Real, float* Imag, cl_float2* Complex, int DATA_W, int DATA_H, int DATA_D)
{
    for (int z = 0; z < DATA_D; z++)
    {
        for (int y = 0; y < DATA_H; y++)
        {   
           for (int x = 0; x < DATA_W; x++)
           {    
               Real[x + y * DATA_W + z * DATA_W * DATA_H] = Complex[x + y * DATA_W + z * DATA_W * DATA_H].x;
               Imag[x + y * DATA_W + z * DATA_W * DATA_H] = Complex[x + y * DATA_W + z * DATA_W * DATA_H].y;
           }
        }
    }
    
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
    
void AllocateMemoryFloat2(cl_float2 *& pointer, int size, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages, const char* variable)
{
    pointer = (cl_float2*)malloc(size);
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


int main(int argc, char **argv)
{
    //-----------------------
    // Input pointers
    
    float           *h_fMRI_Volumes, *h_T1_Volume, *h_Interpolated_T1_Volume, *h_Aligned_T1_Volume_Linear, *h_Aligned_T1_Volume_NonLinear, *h_MNI_Volume, *h_MNI_Brain_Volume, *h_MNI_Brain_Mask, *h_Aligned_EPI_Volume_T1, *h_Aligned_EPI_Volume_MNI; 
  
    float           *h_Quadrature_Filter_1_Linear_Registration_Real, *h_Quadrature_Filter_2_Linear_Registration_Real, *h_Quadrature_Filter_3_Linear_Registration_Real, *h_Quadrature_Filter_1_Linear_Registration_Imag, *h_Quadrature_Filter_2_Linear_Registration_Imag, *h_Quadrature_Filter_3_Linear_Registration_Imag;
    float           *h_Quadrature_Filter_1_NonLinear_Registration_Real, *h_Quadrature_Filter_2_NonLinear_Registration_Real, *h_Quadrature_Filter_3_NonLinear_Registration_Real, *h_Quadrature_Filter_1_NonLinear_Registration_Imag, *h_Quadrature_Filter_2_NonLinear_Registration_Imag, *h_Quadrature_Filter_3_NonLinear_Registration_Imag;
    float           *h_Quadrature_Filter_4_NonLinear_Registration_Real, *h_Quadrature_Filter_5_NonLinear_Registration_Real, *h_Quadrature_Filter_6_NonLinear_Registration_Real, *h_Quadrature_Filter_4_NonLinear_Registration_Imag, *h_Quadrature_Filter_5_NonLinear_Registration_Imag, *h_Quadrature_Filter_6_NonLinear_Registration_Imag;
  
    int             IMAGE_REGISTRATION_FILTER_SIZE = 7;
    int 			COARSEST_SCALE_T1_MNI = 4;
	int				COARSEST_SCALE_EPI_T1 = 4;
	int				MM_T1_Z_CUT = 0;
	int				MM_EPI_Z_CUT = 0;
    int             NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_RIGID = 6;
    int             NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_AFFINE = 12;
    
    float           h_T1_MNI_Registration_Parameters[NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_AFFINE];
    float           h_EPI_T1_Registration_Parameters[NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_RIGID];
    float           h_EPI_MNI_Registration_Parameters[NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_AFFINE];
    float           *h_Motion_Parameters;
    
    float           *h_Projection_Tensor_1, *h_Projection_Tensor_2, *h_Projection_Tensor_3, *h_Projection_Tensor_4, *h_Projection_Tensor_5, *h_Projection_Tensor_6;    
    
    float           *h_Filter_Directions_X, *h_Filter_Directions_Y, *h_Filter_Directions_Z;
    
    float           *h_Slice_Timing_Corrected_fMRI_Volumes;
    float           *h_Motion_Corrected_fMRI_Volumes;
    float           *h_Smoothed_fMRI_Volumes;    
    
    float           *h_X_GLM, *h_xtxxt_GLM, *h_X_GLM_Confounds, *h_Contrasts, *h_ctxtxc_GLM, *h_Highres_Regressor;
    
	float			*h_Beta_Volumes_MNI, *h_Statistical_Maps_MNI, *h_P_Values_MNI;
	float			*h_Beta_Volumes_EPI, *h_Statistical_Maps_EPI, *h_P_Values_EPI;
    float           *h_AR1_Estimates_MNI, *h_AR2_Estimates_MNI, *h_AR3_Estimates_MNI, *h_AR4_Estimates_MNI;
    float           *h_AR1_Estimates_EPI, *h_AR2_Estimates_EPI, *h_AR3_Estimates_EPI, *h_AR4_Estimates_EPI;
        
    int             EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T;
    int             T1_DATA_H, T1_DATA_W, T1_DATA_D;
    int             MNI_DATA_W, MNI_DATA_H, MNI_DATA_D;
                
    float           EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, TR;
    float           T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z;
    float           MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z;
    
    int             NUMBER_OF_GLM_REGRESSORS, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_CONFOUND_REGRESSORS, NUMBER_OF_CONTRASTS, BETA_SPACE;
    
    int             NUMBER_OF_DETRENDING_REGRESSORS = 4;
    int             NUMBER_OF_MOTION_REGRESSORS = 6;	

	int				NUMBER_OF_EVENTS;
	int				HIGHRES_FACTOR = 100;
    
    //-----------------------
    // Output pointers
    
    int             *h_Cluster_Indices_Out, *h_Cluster_Indices;
    float           *h_Beta_Volumes, *h_Residuals, *h_Residual_Variances, *h_Statistical_Maps;    
    float           *h_Design_Matrix, *h_Design_Matrix2;
    float           *h_Whitened_Models;
    float           *h_EPI_Mask;
    
    //----------
    
    void*           allMemoryPointers[500];
    int             numberOfMemoryPointers = 0;
    
	nifti_image*	allNiftiImages[500];
	int				numberOfNiftiImages = 0;
    
    //---------------------
    // Settings
    
    int             OPENCL_PLATFORM = 0;
    int             OPENCL_DEVICE = 0;
    
    int             NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION = 10;
    int             NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION = 10;
    int             COARSEST_SCALE = 4;
    float           TSIGMA = 5.0f;
    float           ESIGMA = 5.0f;
    float           DSIGMA = 5.0f;
    
    int             NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION = 5;
    int             REGRESS_MOTION = 0;
	int				REGRESS_CONFOUNDS = 0;
    float           EPI_SMOOTHING_AMOUNT = 6.0f;
    float           AR_SMOOTHING_AMOUNT = 6.0f;
    
    int             USE_TEMPORAL_DERIVATIVES = 0;
    bool            PERMUTE = false;
    int				NUMBER_OF_PERMUTATIONS = 10000;
    int				INFERENCE_MODE = 1;
    float           CLUSTER_DEFINING_THRESHOLD = 2.5f;
    bool            BAYESIAN = false;
    int             NUMBER_OF_ITERATIONS_FOR_MCMC = 1000;
	bool			MASK = false;
	const char*		MASK_NAME;
    float			SIGNIFICANCE_LEVEL = 0.05f;
	int				TEST_STATISTICS = 0;
    
    bool            WRITE_INTERPOLATED_T1 = false;
    bool            WRITE_ALIGNED_T1_LINEAR = false;
    bool            WRITE_ALIGNED_T1_NONLINEAR = false;
    bool            WRITE_ALIGNED_EPI_T1 = false;
    bool            WRITE_ALIGNED_EPI_MNI = false;
    bool            WRITE_SLICETIMING_CORRECTED = false;
    bool            WRITE_MOTION_CORRECTED = false;
    bool            WRITE_SMOOTHED = false;
    bool            WRITE_ACTIVITY_EPI = false;
    bool            WRITE_RESIDUALS = false;
    bool            WRITE_RESIDUALS_MNI = false;
    bool            WRITE_DESIGNMATRIX = false;
    bool            WRITE_AR_ESTIMATES_EPI = false;
    bool            WRITE_AR_ESTIMATES_MNI = false;
    
    bool            PRINT = true;
    bool            VERBOS = false;
    bool            DEBUG = false;
    
    //---------------------
    
   
    /* Input arguments */
    FILE *fp = NULL;
    
    // No inputs, so print help text
    if (argc == 1)
    {   
		printf("\nThe function performs first level analysis of one fMRI dataset. The processing includes registration between T1 and MNI, registration between fMRI and T1, slice timing correction, motion correction, smoothing and statistical analysis. \n\n");     
        printf("Usage:\n\n");
        printf("FirstLevelAnalysis fMRI_data.nii T1_volume.nii MNI_volume.nii regressors.txt contrasts.txt  [options]\n\n");
        
        printf("OpenCL options:\n\n");
        printf(" -platform                  The OpenCL platform to use (default 0) \n");
        printf(" -device                    The OpenCL device to use for the specificed platform (default 0) \n\n");
        
        printf("Registration options:\n\n");
        printf(" -iterationslinear          Number of iterations for the linear registration (default 10) \n");        
        printf(" -iterationsnonlinear       Number of iterations for the non-linear registration (default 10), 0 means that no non-linear registration is performed \n");        
        printf(" -lowestscale               The lowest scale for the linear and non-linear registration, should be 1, 2, 4 or 8 (default 4), x means downsampling a factor x in each dimension  \n");        
        printf(" -tsigma                    Amount of Gaussian smoothing applied to the estimated tensor components, defined as sigma of the Gaussian kernel (default 5.0)  \n");        
        printf(" -esigma                    Amount of Gaussian smoothing applied to the equation systems (one in each voxel), defined as sigma of the Gaussian kernel (default 5.0)  \n");        
        printf(" -dsigma                    Amount of Gaussian smoothing applied to the displacement fields (x,y,z), defined as sigma of the Gaussian kernel (default 5.0)  \n");        
        printf(" -zcut                      Number of mm to cut from the bottom of the T1 volume, can be negative, useful if the head in the volume is placed very high or low (default 0) \n\n");
        
        printf("Preprocessing options:\n\n");
        printf(" -iterationsmc              Number of iterations for motion correction (default 5) \n");
        printf(" -regressmotion             Include motion parameters in design matrix (default no) \n");
        printf(" -smoothing                 Amount of smoothing to apply to the fMRI data (default 6.0 mm) \n\n");
        
        printf("Statistical options:\n\n");
        printf(" -temporalderivatives       Use temporal derivatives for the activity regressors (default no) \n");
        printf(" -permute                   Apply a permutation test to get p-values (default no) \n");
        printf(" -permutations              Number of permutations to use (default 10,000) \n");
        printf(" -inferencemode             Inference mode to use for permutation test, 0 = voxel, 1 = cluster extent, 2 = cluster mass (default 1) \n");
        printf(" -cdt                       Cluster defining threshold for cluster inference (default 2.5) \n");
        printf(" -bayesian                  Do Bayesian analysis using MCMC, currently only supports 2 regressors (default no) \n");
        printf(" -iterationsmcmc            Number of iterations for MCMC chains (default 1,000) \n");
        printf(" -mask                      Apply a mask to the statistical maps after the statistical analysis, in MNI space (default none) \n\n");


        printf("Misc options:\n\n");
        printf(" -savet1interpolated        Save T1 volume after resampling to MNI voxel size (default no) \n");
        printf(" -savet1alignedlinear       Save T1 volume linearly aligned to the MNI volume (default no) \n");
        printf(" -savet1alignednonlinear    Save T1 volume non-linearly aligned to the MNI volume (default no) \n");
        printf(" -saveepialignedt1          Save EPI volume aligned to the T1 volume (default no) \n");
        printf(" -saveepialignedmni         Save EPI volume aligned to the MNI volume (default no) \n");
        printf(" -saveslicetimingcorrected  Save slice timing corrected fMRI volumes  (default no) \n");
        printf(" -savemotioncorrected       Save motion corrected fMRI volumes (default no) \n");
        printf(" -savesmoothed              Save smoothed fMRI volumes (default no) \n");
        printf(" -saveactivityepi           Save activity maps in EPI space (in addition to MNI space, default no) \n");
        printf(" -saveresiduals             Save residuals after GLM analysis (default no) \n");
        printf(" -saveresidualsmni          Save residuals after GLM analysis, in MNI space (default no) \n");
        printf(" -savedesignmatrix          Save the total design matrix used (default no) \n");
        printf(" -savearparameters          Save the estimated AR coefficients (default no) \n");
        printf(" -savearparametersmni       Save the estimated AR coefficients, in MNI space (default no) \n");
        printf(" -saveall                   Save everything (default no) \n");
        printf(" -quiet                     Don't print anything to the terminal (default false) \n");
        printf(" -verbose                   Print extra stuff (default false) \n");
        printf(" -debug                     Get additional debug information saved as nifti files (default no). Warning: This will use a lot of extra memory! \n");
        printf("\n\n");
        
        return EXIT_SUCCESS;
    }
    else if (argc < 6)
    {
        printf("\nNeed fMRI data, T1 volume, MNI volume, regressors and contrasts!\n\n");
        printf("Usage:\n\n");
        printf("FirstLevelAnalysis fMRI_data.nii T1_volume.nii MNI_volume.nii regressors.txt contrasts.txt  [options]\n\n");
		return EXIT_FAILURE;
    }
    // Try to open all files
    else if (argc > 1)
    {
        for (int i = 1; i <= 5; i++)
        {
            fp = fopen(argv[i],"r");
            if (fp == NULL)
            {
                printf("Could not open file %s !\n",argv[i]);
                return EXIT_FAILURE;
            }
            fclose(fp);
        }
    }
    
    // Loop over additional inputs
    int i = 6;
    while (i < argc)
    {
        char *input = argv[i];
        char *p;
        
        // OpenCL options
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
        
        // Registration options
        else if (strcmp(input,"-iterationslinear") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -iterationslinear !\n");
                return EXIT_FAILURE;
			}

            NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION = (int)strtol(argv[i+1], &p, 10);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("Number of linear iterations must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
            else if (NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION <= 0)
            {
                printf("Number of linear iterations must be a positive number!\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        else if (strcmp(input,"-iterationsnonlinear") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -iterationsnonlinear !\n");
                return EXIT_FAILURE;
			}

            NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION = (int)strtol(argv[i+1], &p, 10);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("Number of non-linear iterations must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
            else if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION < 0)
            {
                printf("Number of non-linear iterations must be >= 0 !\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        else if (strcmp(input,"-lowestscale") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -lowestscale !\n");
                return EXIT_FAILURE;
			}

            COARSEST_SCALE = (int)strtol(argv[i+1], &p, 10);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("Lowest scale must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
  			else if ( (COARSEST_SCALE != 1) && (COARSEST_SCALE != 2) && (COARSEST_SCALE != 4) && (COARSEST_SCALE != 8) )
            {
                printf("Lowest scale must be 1, 2, 4 or 8!\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        else if (strcmp(input,"-tsigma") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -tsigma !\n");
                return EXIT_FAILURE;
			}

            TSIGMA = strtod(argv[i+1], &p);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("tsigma must be a float! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
  			else if ( TSIGMA < 0.0f )
            {
                printf("tsigma must be >= 0.0 !\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        else if (strcmp(input,"-esigma") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -esigma !\n");
                return EXIT_FAILURE;
			}

            ESIGMA = (float)strtod(argv[i+1], &p);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("esigma must be a float! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
  			else if ( ESIGMA < 0.0f )
            {
                printf("esigma must be >= 0.0 !\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        else if (strcmp(input,"-dsigma") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -dsigma !\n");
                return EXIT_FAILURE;
			}

            DSIGMA = (float)strtod(argv[i+1], &p);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("dsigma must be a float! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
  			else if ( DSIGMA < 0.0f )
            {
                printf("dsigma must be >= 0.0 !\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        else if (strcmp(input,"-zcut") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -zcut !\n");
                return EXIT_FAILURE;
			}

            MM_T1_Z_CUT = (int)strtol(argv[i+1], &p, 10);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("zcut must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }

            i += 2;
        }
        
        // Preprocessing options
        else if (strcmp(input,"-iterationsmc") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -iterationsmc !\n");
                return EXIT_FAILURE;
			}
            
            NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION = (int)strtol(argv[i+1], &p, 10);
            
			if (!isspace(*p) && *p != 0)
		    {
		        printf("Number of iterations for motion correction must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
            else if (NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION < 0)
            {
                printf("Number of iterations for motion correction must be >= 0 !\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        else if (strcmp(input,"-regressmotion") == 0)
        {
            REGRESS_MOTION = 1;
            i += 1;
        }
        else if (strcmp(input,"-smoothing") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -smoothing !\n");
                return EXIT_FAILURE;
			}
            
            EPI_SMOOTHING_AMOUNT = (float)strtod(argv[i+1], &p);
            
			if (!isspace(*p) && *p != 0)
		    {
		        printf("Smoothing must be a float! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
  			else if ( EPI_SMOOTHING_AMOUNT < 0.0f )
            {
                printf("Smoothing must be >= 0.0 !\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        
        // Statistical options
        else if (strcmp(input,"-temporalderivatives") == 0)
        {
            USE_TEMPORAL_DERIVATIVES = 1;
            i += 1;
        }
        else if (strcmp(input,"-permute") == 0)
        {
            PERMUTE = true;
            i += 1;
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
            else if ( (INFERENCE_MODE != 0) && (INFERENCE_MODE != 1) && (INFERENCE_MODE != 2) )
            {
                printf("Inference mode must be 0, 1 or 2 !\n");
                return EXIT_FAILURE;
            }
            i += 2;
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
        else if (strcmp(input,"-bayesian") == 0)
        {
            BAYESIAN = true;
            i += 1;
        }
        else if (strcmp(input,"-iterationsmcmc") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -iterationsmcmc !\n");
                return EXIT_FAILURE;
			}
            
            NUMBER_OF_ITERATIONS_FOR_MCMC = (int)strtol(argv[i+1], &p, 10);
            
			if (!isspace(*p) && *p != 0)
		    {
		        printf("Number of iterations for MCMC must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
            else if (NUMBER_OF_ITERATIONS_FOR_MCMC <= 0)
            {
                printf("Number of iterations for MCMC must be > 0 !\n");
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
        
        // Misc options
        else if (strcmp(input,"-savet1interpolated") == 0)
        {
            WRITE_INTERPOLATED_T1 = true;
            i += 1;
        }
        else if (strcmp(input,"-savet1alignedlinear") == 0)
        {
            WRITE_ALIGNED_T1_LINEAR = true;
            i += 1;
        }
        else if (strcmp(input,"-savet1alignednonlinear") == 0)
        {
            WRITE_ALIGNED_T1_NONLINEAR = true;
            i += 1;
        }
        else if (strcmp(input,"-saveepialignedt1") == 0)
        {
            WRITE_ALIGNED_EPI_T1 = true;
            i += 1;
        }
        else if (strcmp(input,"-saveepialignedmni") == 0)
        {
            WRITE_ALIGNED_EPI_MNI = true;
            i += 1;
        }
        else if (strcmp(input,"-saveslicetimingcorrected") == 0)
        {
            WRITE_SLICETIMING_CORRECTED = true;
            i += 1;
        }
        else if (strcmp(input,"-savemotioncorrected") == 0)
        {
            WRITE_MOTION_CORRECTED = true;
            i += 1;
        }
        else if (strcmp(input,"-savesmoothed") == 0)
        {
            WRITE_SMOOTHED = true;
            i += 1;
        }
        else if (strcmp(input,"-saveactivityepi") == 0)
        {
            WRITE_ACTIVITY_EPI = true;
            i += 1;
        }
        else if (strcmp(input,"-saveresiduals") == 0)
        {
            WRITE_RESIDUALS = true;
            i += 1;
        }
        else if (strcmp(input,"-saveresidualsmni") == 0)
        {
            WRITE_RESIDUALS_MNI = true;
            i += 1;
        }
        else if (strcmp(input,"-savedesignmatrix") == 0)
        {
            WRITE_DESIGNMATRIX = true;
            i += 1;
        }
        else if (strcmp(input,"-savearparameters") == 0)
        {
            WRITE_AR_ESTIMATES_EPI = true;
            i += 1;
        }
        else if (strcmp(input,"-savearparametersmni") == 0)
        {
            WRITE_AR_ESTIMATES_MNI = true;
            i += 1;
        }
        else if (strcmp(input,"-saveall") == 0)
        {
            WRITE_INTERPOLATED_T1 = true;
            WRITE_ALIGNED_T1_LINEAR = true;
            WRITE_ALIGNED_T1_NONLINEAR = true;
            WRITE_ALIGNED_EPI_T1 = true;
            WRITE_ALIGNED_EPI_MNI = true;
            WRITE_SLICETIMING_CORRECTED = true;
            WRITE_MOTION_CORRECTED = true;
            WRITE_SMOOTHED = true;
            WRITE_ACTIVITY_EPI = true;
            WRITE_RESIDUALS = true;
            WRITE_RESIDUALS_MNI = true;
            WRITE_AR_ESTIMATES_EPI = true;
            WRITE_AR_ESTIMATES_MNI = true;
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
        else if (strcmp(input,"-debug") == 0)
        {
            DEBUG = true;
            i += 1;
        }
        else
        {
            printf("Unrecognized option! %s \n",argv[i]);
            return EXIT_FAILURE;
        }                
    }
    
    if (BAYESIAN && PERMUTE)
    {
        printf("Cannot do both Bayesian and non-parametric fMRI analysis, pick one!\n");
        return EXIT_FAILURE;
    }

    //------------------------------------------

    // Read number of regressors from design matrix file
    
	std::ifstream design;
    design.open(argv[4]);
    
    if (!design.good())
    {
        design.close();
        printf("Unable to open design file %s. Aborting! \n",argv[4]);
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
        printf("Number of regressors must be > 0 ! You provided %i regressors in the design file. Aborting! \n",NUMBER_OF_GLM_REGRESSORS);
        return EXIT_FAILURE;
    }
    else if (NUMBER_OF_GLM_REGRESSORS > 25)
    {
        design.close();
        printf("Number of regressors must be <= 25 ! You provided %i regressors in the design file. Aborting! \n",NUMBER_OF_GLM_REGRESSORS);
        return EXIT_FAILURE;
    }
    design.close();
    
    // Read contrasts
    
    std::ifstream contrasts;
    contrasts.open(argv[5]);
    
    if (!contrasts.good())
    {
        contrasts.close();
        printf("Unable to open contrasts file %s. Aborting! \n",argv[5]);
        return EXIT_FAILURE;
    }
    
    contrasts >> tempString; // NumRegressors as string
    if (tempString.compare(NR) != 0)
    {
        contrasts.close();
        printf("First element of the contrasts file should be the string 'NumRegressors', but it is %s. Aborting! \n",tempString.c_str());
        return EXIT_FAILURE;
    }
    contrasts >> tempNumber;
    
    // Check for consistency
    if ( tempNumber != NUMBER_OF_GLM_REGRESSORS )
    {
        contrasts.close();
        printf("Design file says that number of regressors is %i, while contrast file says there are %i regressors. Aborting! \n",NUMBER_OF_GLM_REGRESSORS,tempNumber);
        return EXIT_FAILURE;
    }
    
    contrasts >> tempString; // NumContrasts as string
    std::string NC("NumContrasts");
    if (tempString.compare(NC) != 0)
    {
        contrasts.close();
        printf("Third element of the contrasts file should be the string 'NumContrasts', but it is %s. Aborting! \n",tempString.c_str());
        return EXIT_FAILURE;
    }
    contrasts >> NUMBER_OF_CONTRASTS;
	
    if (NUMBER_OF_CONTRASTS <= 0)
    {
        contrasts.close();
        printf("Number of contrasts must be > 0 ! You provided %i in the contrasts file. Aborting! \n",NUMBER_OF_CONTRASTS);
        return EXIT_FAILURE;
    }
    contrasts.close();
    
    
	//------------------------------------------

	double startTime = GetWallTime();
    
    // Read data

	nifti_image *inputfMRI = nifti_image_read(argv[1],1);
    
    if (inputfMRI == NULL)
    {
        printf("Could not open fMRI data!\n");
        return EXIT_FAILURE;
    }
	allNiftiImages[numberOfNiftiImages] = inputfMRI;
	numberOfNiftiImages++;

	nifti_image *inputT1 = nifti_image_read(argv[2],1);
    
    if (inputT1 == NULL)
    {
        printf("Could not open T1 volume!\n");
		FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
	allNiftiImages[numberOfNiftiImages] = inputT1;
	numberOfNiftiImages++;

	nifti_image *inputMNI = nifti_image_read(argv[3],1);
    
    if (inputMNI == NULL)
    {
        printf("Could not open MNI volume!\n");
		FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
	allNiftiImages[numberOfNiftiImages] = inputMNI;
	numberOfNiftiImages++;
    
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
		printf("It took %f seconds to read the nifti file(s)\n",(float)(endTime - startTime));
	}

    // Get data dimensions from input data
    EPI_DATA_W = inputfMRI->nx;
    EPI_DATA_H = inputfMRI->ny;
    EPI_DATA_D = inputfMRI->nz;    
    EPI_DATA_T = inputfMRI->nt;    

    T1_DATA_W = inputT1->nx;
    T1_DATA_H = inputT1->ny;
    T1_DATA_D = inputT1->nz;    
    
    MNI_DATA_W = inputMNI->nx;
    MNI_DATA_H = inputMNI->ny;
    MNI_DATA_D = inputMNI->nz;    
    
    // Get voxel sizes from input data
    EPI_VOXEL_SIZE_X = inputfMRI->dx;
    EPI_VOXEL_SIZE_Y = inputfMRI->dy;
    EPI_VOXEL_SIZE_Z = inputfMRI->dz;
    TR = inputfMRI->dt;

    T1_VOXEL_SIZE_X = inputT1->dx;
    T1_VOXEL_SIZE_Y = inputT1->dy;
    T1_VOXEL_SIZE_Z = inputT1->dz;
                             
    MNI_VOXEL_SIZE_X = inputMNI->dx;
    MNI_VOXEL_SIZE_Y = inputMNI->dy;
    MNI_VOXEL_SIZE_Z = inputMNI->dz;

    // Calculate sizes, in bytes
    
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS * (USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS * REGRESS_MOTION; //NUMBER_OF_CONFOUND_REGRESSORS*REGRESS_CONFOUNDS;
    
    if (NUMBER_OF_TOTAL_GLM_REGRESSORS > 25)
    {
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        printf("Number of total regressors must be <= 25 ! You provided %i regressors in the design file, with regressors for motion and detrending, this comes to a total of %i regressors. Aborting! \n",NUMBER_OF_GLM_REGRESSORS,NUMBER_OF_TOTAL_GLM_REGRESSORS);
        return EXIT_FAILURE;
    }

    int EPI_DATA_SIZE = EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);
    int T1_VOLUME_SIZE = T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float);
    int MNI_VOLUME_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float);
    
    int EPI_VOLUME_SIZE = EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);
    int EPI_VOLUME_SIZE_INT = EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int);
    
    int FILTER_SIZE = IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float);
    
    int MOTION_PARAMETERS_SIZE = NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_RIGID * EPI_DATA_T * sizeof(float);
    
    int GLM_SIZE = EPI_DATA_T * NUMBER_OF_GLM_REGRESSORS * sizeof(float);
    int CONTRAST_SIZE = NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float);
    int CONTRAST_SCALAR_SIZE = NUMBER_OF_CONTRASTS * sizeof(float);
    int DESIGN_MATRIX_SIZE = NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float);
	int HIGHRES_REGRESSOR_SIZE = EPI_DATA_T * HIGHRES_FACTOR * sizeof(float);    

    int CONFOUNDS_SIZE = NUMBER_OF_CONFOUND_REGRESSORS * EPI_DATA_T * sizeof(float);
    
    int PROJECTION_TENSOR_SIZE = NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION * sizeof(float);
    int FILTER_DIRECTIONS_SIZE = NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION * sizeof(float);
    
    int BETA_DATA_SIZE_MNI = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float);
    int STATISTICAL_MAPS_DATA_SIZE_MNI = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float);
    int RESIDUALS_DATA_SIZE_MNI = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * EPI_DATA_T * sizeof(float);
 
    int BETA_DATA_SIZE_EPI = EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float);
    int STATISTICAL_MAPS_DATA_SIZE_EPI = EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float);
    int RESIDUALS_DATA_SIZE_EPI = EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);
    
    // Print some info
    if (PRINT)
    {
        printf("Authored by K.A. Eklund \n");
	    printf("fMRI data size : %i x %i x %i x %i \n", EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
		printf("fMRI voxel size : %f x %f x %f mm \n", EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
		printf("fMRI TR  : %f s \n", TR);		
    	printf("T1 data size : %i x %i x %i \n", T1_DATA_W, T1_DATA_H, T1_DATA_D);
		printf("T1 voxel size : %f x %f x %f mm \n", T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z);
	    printf("MNI data size : %i x %i x %i \n", MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
		printf("MNI voxel size : %f x %f x %f mm \n", MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z);
	    printf("Number of GLM regressors : %i \n",  NUMBER_OF_GLM_REGRESSORS);
		if (REGRESS_CONFOUNDS)
		{
	    	printf("Number of confound regressors : %i \n",  NUMBER_OF_CONFOUND_REGRESSORS);
		}
	    printf("Number of total GLM regressors : %i \n",  NUMBER_OF_TOTAL_GLM_REGRESSORS);
	    printf("Number of contrasts : %i \n",  NUMBER_OF_CONTRASTS);
    } 
    
    // ------------------------------------------------
    
    // Allocate memory on the host

	startTime = GetWallTime();

	AllocateMemory(h_fMRI_Volumes, EPI_DATA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "fMRI_VOLUMES");
	AllocateMemory(h_T1_Volume, T1_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "T1_VOLUME");
	AllocateMemory(h_MNI_Volume, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "MNI_VOLUME");
	AllocateMemory(h_MNI_Brain_Volume, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "MNI_BRAIN_VOLUME");

    //h_MNI_Brain_Mask                    = (float *)mxMalloc(MNI_DATA_SIZE);

	//AllocateMemory(h_EPI_Mask, EPI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "EPI_MASK");
    
    //h_Cluster_Indices                   = (int *)mxMalloc(EPI_VOLUME_SIZE_INT);
   
    if (WRITE_INTERPOLATED_T1)
    {
        AllocateMemory(h_Interpolated_T1_Volume, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "T1_INTERPOLATED");
    }
    if (WRITE_ALIGNED_T1_LINEAR)
    {
        AllocateMemory(h_Aligned_T1_Volume_Linear, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "ALIGNED_T1_LINEAR");
    }
    if (WRITE_ALIGNED_T1_NONLINEAR)
    {
        AllocateMemory(h_Aligned_T1_Volume_NonLinear, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "ALIGNED_T1_NONLINEAR");
    }
	if (WRITE_ALIGNED_EPI_T1)
	{
        AllocateMemory(h_Aligned_EPI_Volume_T1, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "ALIGNED_EPI_T1");
	}
	if (WRITE_ALIGNED_EPI_MNI)
	{
        AllocateMemory(h_Aligned_EPI_Volume_MNI, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "ALIGNED_EPI_MNI");
	}

   
    //h_Aligned_EPI_Volume                                = (float *)mxMalloc(MNI_DATA_SIZE);

	AllocateMemory(h_Quadrature_Filter_1_Linear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_1_LINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_1_Linear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_1_LINEAR_REGISTRATION_IMAG");    
	AllocateMemory(h_Quadrature_Filter_2_Linear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_2_LINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_2_Linear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_2_LINEAR_REGISTRATION_IMAG");    
	AllocateMemory(h_Quadrature_Filter_3_Linear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_3_LINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_3_Linear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_3_LINEAR_REGISTRATION_IMAG");    
    
	AllocateMemory(h_Quadrature_Filter_1_NonLinear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_1_NONLINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_1_NonLinear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_1_NONLINEAR_REGISTRATION_IMAG");    
	AllocateMemory(h_Quadrature_Filter_2_NonLinear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_2_NONLINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_2_NonLinear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_2_NONLINEAR_REGISTRATION_IMAG");    
	AllocateMemory(h_Quadrature_Filter_3_NonLinear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_3_NONLINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_3_NonLinear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_3_NONLINEAR_REGISTRATION_IMAG");    
	AllocateMemory(h_Quadrature_Filter_4_NonLinear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_4_NONLINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_4_NonLinear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_4_NONLINEAR_REGISTRATION_IMAG");    
	AllocateMemory(h_Quadrature_Filter_5_NonLinear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_5_NONLINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_5_NonLinear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_5_NONLINEAR_REGISTRATION_IMAG");    
	AllocateMemory(h_Quadrature_Filter_6_NonLinear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_6_NONLINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_6_NonLinear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "QUADRATURE_FILTER_6_NONLINEAR_REGISTRATION_IMAG");    

    AllocateMemory(h_Projection_Tensor_1, PROJECTION_TENSOR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "PROJECTION_TENSOR_1");    
    AllocateMemory(h_Projection_Tensor_2, PROJECTION_TENSOR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "PROJECTION_TENSOR_2");    
    AllocateMemory(h_Projection_Tensor_3, PROJECTION_TENSOR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "PROJECTION_TENSOR_3");    
    AllocateMemory(h_Projection_Tensor_4, PROJECTION_TENSOR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "PROJECTION_TENSOR_4");    
    AllocateMemory(h_Projection_Tensor_5, PROJECTION_TENSOR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "PROJECTION_TENSOR_5");    
    AllocateMemory(h_Projection_Tensor_6, PROJECTION_TENSOR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "PROJECTION_TENSOR_6");    

    AllocateMemory(h_Filter_Directions_X, FILTER_DIRECTIONS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "FILTER_DIRECTIONS_X");
    AllocateMemory(h_Filter_Directions_Y, FILTER_DIRECTIONS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "FILTER_DIRECTIONS_Y");        
    AllocateMemory(h_Filter_Directions_Z, FILTER_DIRECTIONS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "FILTER_DIRECTIONS_Z");                
   
	AllocateMemory(h_Motion_Parameters, MOTION_PARAMETERS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "MOTION_PARAMETERS");       

	if (WRITE_SLICETIMING_CORRECTED)
	{
		AllocateMemory(h_Slice_Timing_Corrected_fMRI_Volumes, EPI_DATA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "SLICETIMINGCORRECTED_fMRI_VOLUMES");
	}
	if (WRITE_MOTION_CORRECTED)
	{
		AllocateMemory(h_Motion_Corrected_fMRI_Volumes, EPI_DATA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "MOTIONCORRECTED_fMRI_VOLUMES");
	}
	if (WRITE_SMOOTHED)
	{
		AllocateMemory(h_Smoothed_fMRI_Volumes, EPI_DATA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "SMOOTHED_fMRI_VOLUMES");
	}

    AllocateMemory(h_Beta_Volumes_MNI, BETA_DATA_SIZE_MNI, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "BETA_VOLUMES_MNI");
	AllocateMemory(h_Statistical_Maps_MNI, STATISTICAL_MAPS_DATA_SIZE_MNI, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "STATISTICALMAPS_MNI");
    
    if (WRITE_ACTIVITY_EPI)
    {
        AllocateMemory(h_Beta_Volumes_EPI, BETA_DATA_SIZE_EPI, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "BETA_VOLUMES_EPI");
        AllocateMemory(h_Statistical_Maps_EPI, STATISTICAL_MAPS_DATA_SIZE_EPI, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "STATISTICALMAPS_EPI");

		if (PERMUTE)
		{
			AllocateMemory(h_P_Values_EPI, STATISTICAL_MAPS_DATA_SIZE_EPI, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "PVALUES_EPI");
		}
    }        

	if (PERMUTE)
	{
		AllocateMemory(h_P_Values_MNI, STATISTICAL_MAPS_DATA_SIZE_MNI, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "PVALUES_MNI");
	}
    
    AllocateMemory(h_X_GLM, GLM_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "DESIGN_MATRIX");
    AllocateMemory(h_Highres_Regressor, HIGHRES_REGRESSOR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "HIGHRES_REGRESSOR");
    AllocateMemory(h_xtxxt_GLM, GLM_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "DESIGN_MATRIX_INVERSE");
    AllocateMemory(h_Contrasts, CONTRAST_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "CONTRASTS");
    AllocateMemory(h_ctxtxc_GLM, CONTRAST_SCALAR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "CONTRAST_SCALARS");
    AllocateMemory(h_Design_Matrix, DESIGN_MATRIX_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "TOTAL_DESIGN_MATRIX");
    AllocateMemory(h_Design_Matrix2, DESIGN_MATRIX_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "TOTAL_DESIGN_MATRIX2");

    AllocateMemory(h_AR1_Estimates_EPI, EPI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "AR1_ESTIMATES");
    AllocateMemory(h_AR2_Estimates_EPI, EPI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "AR2_ESTIMATES");
    AllocateMemory(h_AR3_Estimates_EPI, EPI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "AR3_ESTIMATES");
    AllocateMemory(h_AR4_Estimates_EPI, EPI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "AR4_ESTIMATES");

    if (WRITE_AR_ESTIMATES_MNI)
    {
        AllocateMemory(h_AR1_Estimates_MNI, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "AR1_ESTIMATES_MNI");
        AllocateMemory(h_AR2_Estimates_MNI, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "AR2_ESTIMATES_MNI");
        AllocateMemory(h_AR3_Estimates_MNI, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "AR3_ESTIMATES_MNI");
        AllocateMemory(h_AR4_Estimates_MNI, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "AR4_ESTIMATES_MNI");
    }

    
	endTime = GetWallTime();
    
	if (VERBOS)
 	{
		printf("It took %f seconds to allocate memory\n",(float)(endTime - startTime));
	}
    
    // ------------------------------------------------
	// Read events for each regressor
    
	startTime = GetWallTime();

    // Each line of the design file is a filename
    
    // Open design file again
    design.open(argv[4]);
    // Read first two values again
    design >> tempString; // NumRegressors as string
    design >> NUMBER_OF_GLM_REGRESSORS;
    
    // Loop over the number of regressors provided in the design file
    for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
    {
		// Reset highres regressor
	    for (int t = 0; t < EPI_DATA_T * HIGHRES_FACTOR; t++)
    	{
			h_Highres_Regressor[t] = 0.0f;
		}

        // Each regressor is a filename, so try to open the file
        std::ifstream regressor;
        std::string filename;
        design >> filename;
        regressor.open(filename.c_str());
        if (!regressor.good())
        {
            regressor.close();
            printf("Unable to open the regressor file %s . Aborting! \n",filename.c_str());
            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
            FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
            return EXIT_FAILURE;
        }
        
        // Read number of events for current regressor
        regressor >> tempString; // NumEvents as string
        std::string NE("NumEvents");
        if (tempString.compare(NE) != 0)
        {
            contrasts.close();
            printf("First element of each regressor file should be the string 'NumEvents', but it is %s for the regressor file %s. Aborting! \n",tempString.c_str(),filename.c_str());
            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
            FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
            return EXIT_FAILURE;
        }
        regressor >> NUMBER_OF_EVENTS;

		if (DEBUG)
		{
			printf("Number of events for regressor %i is %i \n",r,NUMBER_OF_EVENTS);
		}
        
        // Loop over events
        for (int e = 0; e < NUMBER_OF_EVENTS; e++)
        {
            float onset;
            float duration;
            float value;
            
            // Read onset, duration and value for current event
            regressor >> onset;
            regressor >> duration;
            regressor >> value;
        
			if (DEBUG)
			{
				printf("Event %i: Onset is %f, duration is %f and value is %f \n",e,onset,duration,value);
			}
            
            int start = (int)round(onset * (float)HIGHRES_FACTOR / TR);
            int activityLength = (int)round(duration * (float)HIGHRES_FACTOR / TR);
            
			if (DEBUG)
			{
				printf("Event %i: Start is %i, activity length is %i \n",e,start,activityLength);
			}

            // Put values into highres GLM
            for (int i = 0; i < activityLength; i++)
            {
                if ((start + i) < (EPI_DATA_T * HIGHRES_FACTOR) )
                {
                    h_Highres_Regressor[start + i] = value;
                }
                else
                {
                    design.close();
                    printf("The activity start or duration for event %i in regressor file %s is longer than the fMRI data, aborting! \n",e,filename.c_str());
                    FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
                    FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
                    return EXIT_FAILURE;
                }
            }            
        }
        
        // Downsample highres GLM and put values into regular GLM
        // Should do some lowpass filtering first...
        // Loop over TRs
        for (int t = 0; t < EPI_DATA_T; t++)
        {
            h_X_GLM[t + r * EPI_DATA_T] = h_Highres_Regressor[t*HIGHRES_FACTOR];
        }
        
    }
    design.close();
    
    // Open contrast file again
    contrasts.open(argv[5]);

    // Read first two values again
	contrasts >> tempString; // NumRegressors as string
    contrasts >> tempNumber;
    contrasts >> tempString; // NumContrasts as string
    contrasts >> tempNumber;
   
	if (VERBOS)
 	{
		printf("It took %f seconds to read regressors and contrasts\n",(float)(endTime - startTime));
	}
    
	// Write original design matrix to file
	if (DEBUG)
	{
		std::ofstream designmatrix;
	    designmatrix.open("original_designmatrix.txt");  

	    if ( designmatrix.good() )
	    {
    	    for (int t = 0; t < EPI_DATA_T; t++)
	        {
	    	    for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
		        {
            		designmatrix << std::setprecision(6) << std::fixed << (double)h_X_GLM[t + r * EPI_DATA_T] << "  ";
				}
				designmatrix << std::endl;
			}
		    designmatrix.close();
        } 	
	    else
	    {
			designmatrix.close();
	        printf("Could not open the file for writing the design matrix!\n");
	    }
	}

    // ------------------------------------------------
	// Read data
    
	startTime = GetWallTime();
    
	// Convert fMRI data to floats
    if ( inputfMRI->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputfMRI->data;
    
        for (int i = 0; i < EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T; i++)
        {
            h_fMRI_Volumes[i] = (float)p[i];
        }
    }
    else if ( inputfMRI->datatype == DT_FLOAT )
    {
        float *p = (float*)inputfMRI->data;
    
        for (int i = 0; i < EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T; i++)
        {
            h_fMRI_Volumes[i] = p[i];
        }
    }
    else
    {
        printf("Unknown data type in fMRI data, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
		FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }

	// Convert T1 volume to floats
    if ( inputT1->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputT1->data;
    
        for (int i = 0; i < T1_DATA_W * T1_DATA_H * T1_DATA_D; i++)
        {
            h_T1_Volume[i] = (float)p[i];
        }
    }
    else if ( inputT1->datatype == DT_FLOAT )
    {
        float *p = (float*)inputT1->data;
    
        for (int i = 0; i < T1_DATA_W * T1_DATA_H * T1_DATA_D; i++)
        {
            h_T1_Volume[i] = p[i];
        }
    }
    else
    {
        printf("Unknown data type in T1 volume, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
		FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }

	// Convert MNI volume to floats
    if ( inputMNI->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputMNI->data;
    
        for (int i = 0; i < MNI_DATA_W * MNI_DATA_H * MNI_DATA_D; i++)
        {
            h_MNI_Brain_Volume[i] = (float)p[i];
        }
    }
    else if ( inputMNI->datatype == DT_FLOAT )
    {
        float *p = (float*)inputMNI->data;
    
        for (int i = 0; i < MNI_DATA_W * MNI_DATA_H * MNI_DATA_D; i++)
        {
            h_MNI_Brain_Volume[i] = p[i];
        }
    }
    else
    {
        printf("Unknown data type in MNI volume, aborting!\n");
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

	// Read quadrature filters for Linear registration, three real valued and three imaginary valued
	ReadBinaryFile(h_Quadrature_Filter_1_Linear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter1_real_linear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_1_Linear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter1_imag_linear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_2_Linear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter2_real_linear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_2_Linear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter2_imag_linear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_3_Linear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter3_real_linear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_3_Linear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter3_imag_linear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 

        // Read quadrature filters for nonLinear registration, six real valued and six imaginary valued
	ReadBinaryFile(h_Quadrature_Filter_1_NonLinear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter1_real_nonlinear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_1_NonLinear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter1_imag_nonlinear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_2_NonLinear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter2_real_nonlinear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_2_NonLinear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter2_imag_nonlinear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_3_NonLinear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter3_real_nonlinear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_3_NonLinear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter3_imag_nonlinear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_4_NonLinear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter4_real_nonlinear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_4_NonLinear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter4_imag_nonlinear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_5_NonLinear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter5_real_nonlinear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_5_NonLinear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter5_imag_nonlinear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_6_NonLinear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter6_real_nonlinear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_6_NonLinear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,"filter6_imag_nonlinear_registration.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 

    // Read projection tensors   
    ReadBinaryFile(h_Projection_Tensor_1,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,"projection_tensor1.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
    ReadBinaryFile(h_Projection_Tensor_2,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,"projection_tensor2.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
    ReadBinaryFile(h_Projection_Tensor_3,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,"projection_tensor3.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
    ReadBinaryFile(h_Projection_Tensor_4,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,"projection_tensor4.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
    ReadBinaryFile(h_Projection_Tensor_5,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,"projection_tensor5.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
    ReadBinaryFile(h_Projection_Tensor_6,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,"projection_tensor6.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
        
    // Read filter directions
    ReadBinaryFile(h_Filter_Directions_X,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,"filter_directions_x.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages);
    ReadBinaryFile(h_Filter_Directions_Y,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,"filter_directions_y.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
    ReadBinaryFile(h_Filter_Directions_Z,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,"filter_directions_z.bin",allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages);  

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
    if ( !BROCCOLI.GetOpenCLInitiated() )
    {  
        printf("Initialization error is \"%s\" \n",BROCCOLI.GetOpenCLInitializationError());
		printf("OpenCL error is \"%s\" \n",BROCCOLI.GetOpenCLError());
		
        // Print create kernel errors
        int* createKernelErrors = BROCCOLI.GetOpenCLCreateKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createKernelErrors[i] != 0)
            {
                printf("Create kernel error %i is %s \n",i,BROCCOLI.GetOpenCLErrorMessage(createKernelErrors[i]));
            }
        } 
       
       	// Print build info to file (always)
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
	// Initialization went OK
    else
    {
        BROCCOLI.SetEPIWidth(EPI_DATA_W);
        BROCCOLI.SetEPIHeight(EPI_DATA_H);
        BROCCOLI.SetEPIDepth(EPI_DATA_D);
        BROCCOLI.SetEPITimepoints(EPI_DATA_T);     
        BROCCOLI.SetT1Width(T1_DATA_W);
        BROCCOLI.SetT1Height(T1_DATA_H);
        BROCCOLI.SetT1Depth(T1_DATA_D);
        BROCCOLI.SetMNIWidth(MNI_DATA_W);
        BROCCOLI.SetMNIHeight(MNI_DATA_H);
        BROCCOLI.SetMNIDepth(MNI_DATA_D);
        
        BROCCOLI.SetInputfMRIVolumes(h_fMRI_Volumes);
        BROCCOLI.SetInputT1Volume(h_T1_Volume);
        //BROCCOLI.SetInputMNIVolume(h_MNI_Volume);
        BROCCOLI.SetInputMNIBrainVolume(h_MNI_Brain_Volume);
		//BROCCOLI.SetInputMNIBrainMask(h_MNI_Brain_Mask);
        
        BROCCOLI.SetEPIVoxelSizeX(EPI_VOXEL_SIZE_X);
        BROCCOLI.SetEPIVoxelSizeY(EPI_VOXEL_SIZE_Y);
        BROCCOLI.SetEPIVoxelSizeZ(EPI_VOXEL_SIZE_Z);       
        BROCCOLI.SetT1VoxelSizeX(T1_VOXEL_SIZE_X);
        BROCCOLI.SetT1VoxelSizeY(T1_VOXEL_SIZE_Y);
        BROCCOLI.SetT1VoxelSizeZ(T1_VOXEL_SIZE_Z);   
        BROCCOLI.SetMNIVoxelSizeX(MNI_VOXEL_SIZE_X);
        BROCCOLI.SetMNIVoxelSizeY(MNI_VOXEL_SIZE_Y);
        BROCCOLI.SetMNIVoxelSizeZ(MNI_VOXEL_SIZE_Z); 
        BROCCOLI.SetInterpolationMode(LINEAR);
    
        BROCCOLI.SetNumberOfIterationsForLinearImageRegistration(NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION);
        BROCCOLI.SetNumberOfIterationsForNonLinearImageRegistration(NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION);
        BROCCOLI.SetImageRegistrationFilterSize(IMAGE_REGISTRATION_FILTER_SIZE);    
        BROCCOLI.SetLinearImageRegistrationFilters(h_Quadrature_Filter_1_Linear_Registration_Real, h_Quadrature_Filter_1_Linear_Registration_Imag, h_Quadrature_Filter_2_Linear_Registration_Real, h_Quadrature_Filter_2_Linear_Registration_Imag, h_Quadrature_Filter_3_Linear_Registration_Real, h_Quadrature_Filter_3_Linear_Registration_Imag);
        BROCCOLI.SetNonLinearImageRegistrationFilters(h_Quadrature_Filter_1_NonLinear_Registration_Real, h_Quadrature_Filter_1_NonLinear_Registration_Imag, h_Quadrature_Filter_2_NonLinear_Registration_Real, h_Quadrature_Filter_2_NonLinear_Registration_Imag, h_Quadrature_Filter_3_NonLinear_Registration_Real, h_Quadrature_Filter_3_NonLinear_Registration_Imag, h_Quadrature_Filter_4_NonLinear_Registration_Real, h_Quadrature_Filter_4_NonLinear_Registration_Imag, h_Quadrature_Filter_5_NonLinear_Registration_Real, h_Quadrature_Filter_5_NonLinear_Registration_Imag, h_Quadrature_Filter_6_NonLinear_Registration_Real, h_Quadrature_Filter_6_NonLinear_Registration_Imag);    
        BROCCOLI.SetProjectionTensorMatrixFirstFilter(h_Projection_Tensor_1[0],h_Projection_Tensor_1[1],h_Projection_Tensor_1[2],h_Projection_Tensor_1[3],h_Projection_Tensor_1[4],h_Projection_Tensor_1[5]);
        BROCCOLI.SetProjectionTensorMatrixSecondFilter(h_Projection_Tensor_2[0],h_Projection_Tensor_2[1],h_Projection_Tensor_2[2],h_Projection_Tensor_2[3],h_Projection_Tensor_2[4],h_Projection_Tensor_2[5]);
        BROCCOLI.SetProjectionTensorMatrixThirdFilter(h_Projection_Tensor_3[0],h_Projection_Tensor_3[1],h_Projection_Tensor_3[2],h_Projection_Tensor_3[3],h_Projection_Tensor_3[4],h_Projection_Tensor_3[5]);
        BROCCOLI.SetProjectionTensorMatrixFourthFilter(h_Projection_Tensor_4[0],h_Projection_Tensor_4[1],h_Projection_Tensor_4[2],h_Projection_Tensor_4[3],h_Projection_Tensor_4[4],h_Projection_Tensor_4[5]);
        BROCCOLI.SetProjectionTensorMatrixFifthFilter(h_Projection_Tensor_5[0],h_Projection_Tensor_5[1],h_Projection_Tensor_5[2],h_Projection_Tensor_5[3],h_Projection_Tensor_5[4],h_Projection_Tensor_5[5]);
        BROCCOLI.SetProjectionTensorMatrixSixthFilter(h_Projection_Tensor_6[0],h_Projection_Tensor_6[1],h_Projection_Tensor_6[2],h_Projection_Tensor_6[3],h_Projection_Tensor_6[4],h_Projection_Tensor_6[5]);
        BROCCOLI.SetFilterDirections(h_Filter_Directions_X, h_Filter_Directions_Y, h_Filter_Directions_Z);
    
        BROCCOLI.SetNumberOfIterationsForMotionCorrection(NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION);    
        BROCCOLI.SetCoarsestScaleT1MNI(COARSEST_SCALE_T1_MNI);
        BROCCOLI.SetCoarsestScaleEPIT1(COARSEST_SCALE_EPI_T1);
        BROCCOLI.SetMMT1ZCUT(MM_T1_Z_CUT);   
        BROCCOLI.SetMMEPIZCUT(MM_EPI_Z_CUT);   
        BROCCOLI.SetOutputT1MNIRegistrationParameters(h_T1_MNI_Registration_Parameters);
        BROCCOLI.SetOutputEPIT1RegistrationParameters(h_EPI_T1_Registration_Parameters);
        BROCCOLI.SetOutputEPIMNIRegistrationParameters(h_EPI_MNI_Registration_Parameters);
        BROCCOLI.SetOutputMotionParameters(h_Motion_Parameters);
        BROCCOLI.SetEPISmoothingAmount(EPI_SMOOTHING_AMOUNT);
        BROCCOLI.SetARSmoothingAmount(AR_SMOOTHING_AMOUNT);

		BROCCOLI.SetSaveInterpolatedT1(WRITE_INTERPOLATED_T1);
		BROCCOLI.SetSaveAlignedT1Linear(WRITE_ALIGNED_T1_LINEAR);
		BROCCOLI.SetSaveAlignedT1NonLinear(WRITE_ALIGNED_T1_NONLINEAR);	
		BROCCOLI.SetSaveAlignedEPIT1(WRITE_ALIGNED_EPI_T1);	
		BROCCOLI.SetSaveAlignedEPIMNI(WRITE_ALIGNED_EPI_MNI);	
		BROCCOLI.SetSaveSliceTimingCorrected(WRITE_SLICETIMING_CORRECTED);
		BROCCOLI.SetSaveMotionCorrected(WRITE_MOTION_CORRECTED);
		BROCCOLI.SetSaveSmoothed(WRITE_SMOOTHED);				
		BROCCOLI.SetSaveActivityEPI(WRITE_ACTIVITY_EPI);
		BROCCOLI.SetSaveDesignMatrix(WRITE_DESIGNMATRIX);

		if (WRITE_SLICETIMING_CORRECTED)
		{
	        BROCCOLI.SetOutputSliceTimingCorrectedfMRIVolumes(h_Slice_Timing_Corrected_fMRI_Volumes);
		}
		if (WRITE_MOTION_CORRECTED)
		{
	        BROCCOLI.SetOutputMotionCorrectedfMRIVolumes(h_Motion_Corrected_fMRI_Volumes);
		}
		if (WRITE_SMOOTHED)
		{
	        BROCCOLI.SetOutputSmoothedfMRIVolumes(h_Smoothed_fMRI_Volumes);
		}
    
        BROCCOLI.SetTemporalDerivatives(USE_TEMPORAL_DERIVATIVES);
        BROCCOLI.SetRegressMotion(REGRESS_MOTION);
        BROCCOLI.SetRegressConfounds(REGRESS_CONFOUNDS);
        BROCCOLI.SetBetaSpace(BETA_SPACE);
    
        if (REGRESS_CONFOUNDS == 1)
        {
            BROCCOLI.SetNumberOfConfoundRegressors(NUMBER_OF_CONFOUND_REGRESSORS);
            BROCCOLI.SetConfoundRegressors(h_X_GLM_Confounds);
        }
    
        BROCCOLI.SetNumberOfGLMRegressors(NUMBER_OF_GLM_REGRESSORS);
        BROCCOLI.SetNumberOfContrasts(NUMBER_OF_CONTRASTS);    
        BROCCOLI.SetDesignMatrix(h_X_GLM, h_xtxxt_GLM);
        BROCCOLI.SetContrasts(h_Contrasts);
        BROCCOLI.SetGLMScalars(h_ctxtxc_GLM);
    
        BROCCOLI.SetOutputInterpolatedT1Volume(h_Interpolated_T1_Volume);
        BROCCOLI.SetOutputAlignedT1VolumeLinear(h_Aligned_T1_Volume_Linear);
        BROCCOLI.SetOutputAlignedT1VolumeNonLinear(h_Aligned_T1_Volume_NonLinear);
        BROCCOLI.SetOutputAlignedEPIVolumeT1(h_Aligned_EPI_Volume_T1);
        BROCCOLI.SetOutputAlignedEPIVolumeMNI(h_Aligned_EPI_Volume_MNI);
        BROCCOLI.SetOutputSliceTimingCorrectedfMRIVolumes(h_Slice_Timing_Corrected_fMRI_Volumes);
        BROCCOLI.SetOutputMotionCorrectedfMRIVolumes(h_Motion_Corrected_fMRI_Volumes);
        BROCCOLI.SetOutputSmoothedfMRIVolumes(h_Smoothed_fMRI_Volumes);

        BROCCOLI.SetOutputBetaVolumesEPI(h_Beta_Volumes_EPI);
        BROCCOLI.SetOutputStatisticalMapsEPI(h_Statistical_Maps_EPI);

        BROCCOLI.SetOutputBetaVolumesMNI(h_Beta_Volumes_MNI);
        BROCCOLI.SetOutputStatisticalMapsMNI(h_Statistical_Maps_MNI);

        BROCCOLI.SetOutputResiduals(h_Residuals);
        BROCCOLI.SetOutputResidualVariances(h_Residual_Variances);
        BROCCOLI.SetOutputAREstimates(h_AR1_Estimates_EPI, h_AR2_Estimates_EPI, h_AR3_Estimates_EPI, h_AR4_Estimates_EPI);
        BROCCOLI.SetOutputWhitenedModels(h_Whitened_Models);
		    
        BROCCOLI.SetOutputDesignMatrix(h_Design_Matrix, h_Design_Matrix2);

        BROCCOLI.SetOutputClusterIndices(h_Cluster_Indices);
        BROCCOLI.SetOutputEPIMask(h_EPI_Mask);

		startTime = GetWallTime();
		if (!BAYESIAN)
		{
        	BROCCOLI.PerformFirstLevelAnalysisWrapper();
		}
		else
		{
			//BROCCOLI.PerformFirstLevelAnalysisBayesianWrapper();	        
		}
		endTime = GetWallTime();

		if (VERBOS)
	 	{
			printf("\nIt took %f seconds to run the first level analysis\n",(float)(endTime - startTime));
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
    
    startTime = GetWallTime();

	// Write total design matrix to file
	if (WRITE_DESIGNMATRIX)
	{
		std::ofstream designmatrix;
	    designmatrix.open("total_designmatrix.txt");  

	    if ( designmatrix.good() )
	    {
    	    for (int t = 0; t < EPI_DATA_T; t++)
	        {
	    	    for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
		        {
            		designmatrix << std::setprecision(6) << std::fixed << (double)h_Design_Matrix[t + r * EPI_DATA_T] << "  ";
				}
				designmatrix << std::endl;
			}
		    designmatrix.close();
        } 	
	    else
	    {
			designmatrix.close();
	        printf("Could not open the file for writing the design matrix!\n");
	    }
	}

    //----------------------------
    // Write aligned data
    //----------------------------
    
    // Create new nifti image
    nifti_image *outputNiftiT1 = nifti_copy_nim_info(inputMNI);
    nifti_set_filenames(outputNiftiT1, inputT1->fname, 0, 1);
    allNiftiImages[numberOfNiftiImages] = outputNiftiT1;
	numberOfNiftiImages++;
    
    if (WRITE_INTERPOLATED_T1)
	{
    	WriteNifti(outputNiftiT1,h_Interpolated_T1_Volume,"_interpolated",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}
    if (WRITE_ALIGNED_T1_LINEAR)
	{
    	WriteNifti(outputNiftiT1,h_Aligned_T1_Volume_Linear,"_aligned_linear",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}
    if (WRITE_ALIGNED_T1_NONLINEAR)
	{
    	WriteNifti(outputNiftiT1,h_Aligned_T1_Volume_NonLinear,"_aligned_nonlinear",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}

    // Create new nifti image
    nifti_image *outputNiftiEPI = nifti_copy_nim_info(inputMNI);
    nifti_set_filenames(outputNiftiEPI, inputfMRI->fname, 0, 1);
    allNiftiImages[numberOfNiftiImages] = outputNiftiEPI;
	numberOfNiftiImages++;

	if (WRITE_ALIGNED_EPI_T1)
	{
    	WriteNifti(outputNiftiEPI,h_Aligned_EPI_Volume_T1,"_aligned_t1",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}    
	if (WRITE_ALIGNED_EPI_MNI)
	{
    	WriteNifti(outputNiftiEPI,h_Aligned_EPI_Volume_MNI,"_aligned_mni",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}    

    //----------------------------
    // Write preprocessed data
    //----------------------------
    
    // Create new nifti image
    nifti_image *outputNiftifMRI = nifti_copy_nim_info(inputfMRI);
    allNiftiImages[numberOfNiftiImages] = outputNiftifMRI;
	numberOfNiftiImages++;
    
    if (WRITE_SLICETIMING_CORRECTED)
	{
    	WriteNifti(outputNiftifMRI,h_Slice_Timing_Corrected_fMRI_Volumes,"_slice_timing_corrected",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}
    if (WRITE_MOTION_CORRECTED)
	{
    	WriteNifti(outputNiftifMRI,h_Motion_Corrected_fMRI_Volumes,"_motion_corrected",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}
    if (WRITE_SMOOTHED)
	{
    	WriteNifti(outputNiftifMRI,h_Smoothed_fMRI_Volumes,"_smoothed",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}
    
    //------------------------------------------
    // Write statistical results, MNI space
    //------------------------------------------
    
    std::string beta = "_beta";
    std::string tscores = "_tscores";
    std::string pvalues = "_pvalues";
    std::string PPM = "_PPM";
    std::string mni = "_mni";
    std::string epi = "_epi";
    
    // Create new nifti image
    nifti_image *outputNiftiStatisticsMNI = nifti_copy_nim_info(inputMNI);
    nifti_set_filenames(outputNiftiStatisticsMNI, inputfMRI->fname, 0, 1);
    allNiftiImages[numberOfNiftiImages] = outputNiftiStatisticsMNI;
	numberOfNiftiImages++;
    
    // Write each beta weight as a separate file
    if (!BAYESIAN)
    {
        for (int i = 0; i < NUMBER_OF_TOTAL_GLM_REGRESSORS; i++)
        {
            std::string temp = beta;
            std::stringstream ss;
            ss << "_regressor";
            ss << i + 1;
            temp.append(ss.str());
            temp.append(mni);
            WriteNifti(outputNiftiStatisticsMNI,&h_Beta_Volumes_MNI[i * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D],temp.c_str(),ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        }
        // Write each t-map as a separate file
        for (int i = 0; i < NUMBER_OF_CONTRASTS; i++)
        {
            std::string temp = tscores;
            std::stringstream ss;
            ss << "_contrast";
            ss << i + 1;
            temp.append(ss.str());
            temp.append(mni);
            WriteNifti(outputNiftiStatisticsMNI,&h_Statistical_Maps_MNI[i * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D],temp.c_str(),ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        }
        if (PERMUTE)
        {
            // Write each p-map as a separate file
            for (int i = 0; i < NUMBER_OF_CONTRASTS; i++)
            {
                std::string temp = pvalues;
		        std::stringstream ss;
                ss << "_contrast";
                ss << i + 1;
                temp.append(ss.str());
                temp.append(mni);
                WriteNifti(outputNiftiStatisticsMNI,&h_P_Values_MNI[i * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D],temp.c_str(),ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
            }
        }
    }
    else if (BAYESIAN)
    {
        // Write each PPM as a separate file
        for (int i = 0; i < NUMBER_OF_CONTRASTS; i++)
        {
            std::string temp = PPM;
            std::stringstream ss;
            ss << "_contrast";
            ss << i + 1;
            temp.append(ss.str());
            temp.append(mni);
            WriteNifti(outputNiftiStatisticsMNI,&h_Statistical_Maps_MNI[i * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D],temp.c_str(),ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        }
    }

    if (WRITE_AR_ESTIMATES_MNI)
    {
        WriteNifti(outputNiftiStatisticsMNI,h_AR1_Estimates_MNI,"_ar1_estimates_mni",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(outputNiftiStatisticsMNI,h_AR2_Estimates_MNI,"_ar2_estimates_mni",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(outputNiftiStatisticsMNI,h_AR3_Estimates_MNI,"_ar3_estimates_mni",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(outputNiftiStatisticsMNI,h_AR4_Estimates_MNI,"_ar4_estimates_mni",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
    }

    //------------------------------------------
    // Write statistical results, EPI space
    //------------------------------------------
    
    // Create new nifti image
    nifti_image *outputNiftiStatisticsEPI = nifti_copy_nim_info(inputfMRI);
    outputNiftiStatisticsEPI->nt = 1;
    outputNiftiStatisticsEPI->dim[4] = 1;
    outputNiftiStatisticsEPI->nvox = EPI_DATA_W * EPI_DATA_H * EPI_DATA_D;
    allNiftiImages[numberOfNiftiImages] = outputNiftiStatisticsEPI;
    numberOfNiftiImages++;
    
    if (WRITE_ACTIVITY_EPI)
    {
        if (!BAYESIAN)
        {
            // Write each beta weight as a separate file
            for (int i = 0; i < NUMBER_OF_TOTAL_GLM_REGRESSORS; i++)
            {
                std::string temp = beta;
                std::stringstream ss;
                ss << "_regressor";
				ss << i + 1;
                temp.append(ss.str());
                temp.append(epi);
                WriteNifti(outputNiftiStatisticsEPI,&h_Beta_Volumes_EPI[i * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D],temp.c_str(),ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
            }
            // Write each t-map as a separate file
            for (int i = 0; i < NUMBER_OF_CONTRASTS; i++)
            {
                std::string temp = tscores;
                std::stringstream ss;
                ss << "_contrast";
                ss << i + 1;
                temp.append(ss.str());
                temp.append(epi);
                WriteNifti(outputNiftiStatisticsEPI,&h_Statistical_Maps_EPI[i * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D],temp.c_str(),ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
            }
            if (PERMUTE)
            {
                // Write each p-map as a separate file
                for (int i = 0; i < NUMBER_OF_CONTRASTS; i++)
                {
                    std::string temp = pvalues;
                    std::stringstream ss;
	                ss << "_contrast";
                    ss << i + 1;
                    temp.append(ss.str());
                    temp.append(epi);
                    WriteNifti(outputNiftiStatisticsEPI,&h_P_Values_EPI[i * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D],temp.c_str(),ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
                }
            }
        }
        else if (BAYESIAN)
        {
            // Write each PPM as a separate file
            for (int i = 0; i < NUMBER_OF_CONTRASTS; i++)
            {
                std::string temp = PPM;
                std::stringstream ss;
                ss << "_contrast";
                ss << i + 1;
                temp.append(ss.str());
                temp.append(epi);
                WriteNifti(outputNiftiStatisticsEPI,&h_Statistical_Maps_EPI[i * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D],temp.c_str(),ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
            }
        }
    }
    
    if (WRITE_AR_ESTIMATES_EPI)
    {
        WriteNifti(outputNiftiStatisticsEPI,h_AR1_Estimates_EPI,"_ar1_estimates",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(outputNiftiStatisticsEPI,h_AR2_Estimates_EPI,"_ar2_estimates",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(outputNiftiStatisticsEPI,h_AR3_Estimates_EPI,"_ar3_estimates",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(outputNiftiStatisticsEPI,h_AR4_Estimates_EPI,"_ar4_estimates",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
    }    
    
    endTime = GetWallTime();
    
	if (VERBOS)
 	{
		printf("It took %f seconds to write the nifti files\n",(float)(endTime - startTime));
	}
    
    
    // Free all memory
    FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
    FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
    
    return EXIT_SUCCESS;
}



