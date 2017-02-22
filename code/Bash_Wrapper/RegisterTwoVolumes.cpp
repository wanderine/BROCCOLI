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

#include "HelpFunctions.cpp"

#define ADD_FILENAME true
#define DONT_ADD_FILENAME true

#define CHECK_EXISTING_FILE true
#define DONT_CHECK_EXISTING_FILE false

int main(int argc, char **argv)
{
    //-----------------------
    // Input pointers
    
    float           *h_T1_Volume, *h_MNI_Volume, *h_MNI_Brain_Volume, *h_MNI_Brain_Mask;
    
    float           *h_Quadrature_Filter_1_Linear_Registration_Real, *h_Quadrature_Filter_2_Linear_Registration_Real, *h_Quadrature_Filter_3_Linear_Registration_Real, *h_Quadrature_Filter_1_Linear_Registration_Imag, *h_Quadrature_Filter_2_Linear_Registration_Imag, *h_Quadrature_Filter_3_Linear_Registration_Imag;
    float           *h_Quadrature_Filter_1_NonLinear_Registration_Real, *h_Quadrature_Filter_2_NonLinear_Registration_Real, *h_Quadrature_Filter_3_NonLinear_Registration_Real, *h_Quadrature_Filter_1_NonLinear_Registration_Imag, *h_Quadrature_Filter_2_NonLinear_Registration_Imag, *h_Quadrature_Filter_3_NonLinear_Registration_Imag;
    float           *h_Quadrature_Filter_4_NonLinear_Registration_Real, *h_Quadrature_Filter_5_NonLinear_Registration_Real, *h_Quadrature_Filter_6_NonLinear_Registration_Real, *h_Quadrature_Filter_4_NonLinear_Registration_Imag, *h_Quadrature_Filter_5_NonLinear_Registration_Imag, *h_Quadrature_Filter_6_NonLinear_Registration_Imag;
        
    float           *h_Projection_Tensor_1, *h_Projection_Tensor_2, *h_Projection_Tensor_3, *h_Projection_Tensor_4, *h_Projection_Tensor_5, *h_Projection_Tensor_6;
        
    float           *h_Filter_Directions_X, *h_Filter_Directions_Y, *h_Filter_Directions_Z;
        
    //-----------------------
    // Output pointers        
    
    float           *h_t11, *h_t12, *h_t13, *h_t22, *h_t23, *h_t33;
    
    float           *h_Displacement_Field_X, *h_Displacement_Field_Y, *h_Displacement_Field_Z;
    
    cl_float2       *h_Quadrature_Filter_Response_1, *h_Quadrature_Filter_Response_2, *h_Quadrature_Filter_Response_3, *h_Quadrature_Filter_Response_4, *h_Quadrature_Filter_Response_5, *h_Quadrature_Filter_Response_6;
    
    float           *h_Quadrature_Filter_Response_1_Real, *h_Quadrature_Filter_Response_1_Imag;
    float           *h_Quadrature_Filter_Response_2_Real, *h_Quadrature_Filter_Response_2_Imag;
    float           *h_Quadrature_Filter_Response_3_Real, *h_Quadrature_Filter_Response_3_Imag;  
    float           *h_Quadrature_Filter_Response_4_Real, *h_Quadrature_Filter_Response_4_Imag;
    float           *h_Quadrature_Filter_Response_5_Real, *h_Quadrature_Filter_Response_5_Imag;
    float           *h_Quadrature_Filter_Response_6_Real, *h_Quadrature_Filter_Response_6_Imag;      
    float           *h_Phase_Differences, *h_Phase_Certainties, *h_Phase_Gradients;
    float           *h_Aligned_T1_Volume, *h_Aligned_T1_Volume_NonLinear, *h_Interpolated_T1_Volume, *h_Registration_Parameters;
    float           *h_Downsampled_Volume;
    float           *h_Skullstripped_T1_Volume;
    float           *h_Slice_Sums, *h_Top_Slice;
    float           *h_A_Matrix, *h_h_Vector;

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
        
    int             IMAGE_REGISTRATION_FILTER_SIZE = 7;
    int             NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS = 12;
    int             NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION = 10;
    int             NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION = 10;
    int             COARSEST_SCALE = 4;
    int             MM_T1_Z_CUT = 0;
    int             OPENCL_PLATFORM = 0;
    int             OPENCL_DEVICE = 0;
    bool            DEBUG = false;
    const char*     FILENAME_EXTENSION = "_MNI";
    bool            PRINT = true;
	bool			VERBOS = false;
    bool            WRITE_TRANSFORMATION_MATRIX = false;
    bool            WRITE_DISPLACEMENT_FIELD = false;
	bool			WRITE_INTERPOLATED = false;
   	bool			CHANGE_OUTPUT_FILENAME = false;    
	float			SIGMA = 5.0f;
	bool			MASK = false;
	bool			MASK_ORIGINAL = false;
	const char* 	MASK_NAME;
	bool			PRECENTER_REGISTRATION = false;

	const char*		outputFilename;

    // Size parameters
    size_t          T1_DATA_H, T1_DATA_W, T1_DATA_D, T1_DATA_T;
    size_t          MNI_DATA_W, MNI_DATA_H, MNI_DATA_D;
    
    float           T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z;
    float           MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z;    
        
    //---------------------    
    
    /* Input arguments */
    FILE *fp = NULL; 
    
    // No inputs, so print help text
    if (argc == 1)
    {   
		printf("\nThe function aligns two volumes using linear as well as non-linear registration. The function automagically resizes and rescales the input volume to match the reference volume.\n\n");     
        printf("Usage:\n\n");
        printf("RegisterTwoVolumes input_volume.nii reference_volume.nii [options]\n\n");
        printf("Options:\n\n");
        printf(" -platform                  The OpenCL platform to use (default 0) \n");
        printf(" -device                    The OpenCL device to use for the specificed platform (default 0) \n");
        printf(" -iterationslinear          Number of iterations for the linear registration (default 10), 0 means that no linear registration is performed \n");        
        printf(" -iterationsnonlinear       Number of iterations for the non-linear registration (default 10), 0 means that no non-linear registration is performed \n");        

        printf(" -sigma                     Amount of Gaussian smoothing applied for regularization of the displacement field, defined as sigma of the Gaussian kernel (default 5.0)  \n");        
        printf(" -zcut                      Number of mm to cut from the bottom of the input volume, can be negative, useful if the head in the volume is placed very high or low (default 0) \n");        
        printf(" -precenter                 Center the input volume before the registration starts (default off) \n");        
        printf(" -mask                      Mask to apply after linear and non-linear registration, to for example do a skullstrip (default none) \n");        
        printf(" -maskoriginal              Mask to apply after linear registration, to for example do a skullstrip. Returns the volume skullstripped and unregistered (but interpolated to the reference volume size) (default none) \n");        

		printf(" -savematrix                Saves the affine transformation matrix to file (default false) \n");        
		printf(" -savefield                 Saves the displacement field to file (default false) \n");        
		printf(" -saveinterpolated          Saves the input volume rescaled and resized to the size and resolution of the reference volume, before alignment (default false) \n");        
		printf(" -output                    Set output filename (default input_volume_aligned_linear.nii and input_volume_aligned_nonlinear.nii) \n");
        printf(" -quiet                     Don't print anything to the terminal (default false) \n");
        printf(" -verbose                   Print extra stuff (default false) \n");
        printf(" -debug                     Get additional debug information saved as nifti files (default no). Warning: This will use a lot of extra memory! \n");
        printf("\n\n");
        
        return EXIT_SUCCESS;
    }
    else if (argc < 3)
    {
        printf("\nNeed two input volumes!\n\n");
        printf("Usage:\n\n");
        printf("RegisterTwoVolumes input_volume.nii reference_volume.nii [options]\n\n");
		return EXIT_FAILURE;
    }
    // Try to open files
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


		// Check that file extension is .nii or .nii.gz
		CheckFileExtension(argv[2],extensionOK,extension);
		if (!extensionOK)
		{
            printf("File extension is not .nii or .nii.gz, %s is not allowed!\n",extension.c_str());
            return EXIT_FAILURE;
		}
        
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
            else if (NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION < 0)
            {
                printf("Number of linear iterations must be >= 0!\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        else if (strcmp(input,"-iterationsnonlinear") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -iterationslinear !\n");
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
		/*
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
		*/
        else if (strcmp(input,"-sigma") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -sigma !\n");
                return EXIT_FAILURE;
			}

            SIGMA = (float)strtod(argv[i+1], &p);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("sigma must be a float! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
  			else if ( SIGMA < 0.0f )
            {
                printf("sigma must be >= 0.0 !\n");
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
		else if (strcmp(input,"-precenter") == 0)
        {
			PRECENTER_REGISTRATION = true;
            i += 1;
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
		else if (strcmp(input,"-maskoriginal") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read name after -maskoriginal !\n");
                return EXIT_FAILURE;
			}

			MASK_ORIGINAL = true;
            MASK_NAME = argv[i+1];
            i += 2;
        }

        else if (strcmp(input,"-savematrix") == 0)
        {
            WRITE_TRANSFORMATION_MATRIX = true;
            i += 1;
        }
        else if (strcmp(input,"-savefield") == 0)
        {
            WRITE_DISPLACEMENT_FIELD = true;
            i += 1;
        }
        else if (strcmp(input,"-saveinterpolated") == 0)
        {
            WRITE_INTERPOLATED = true;
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

	if ((NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION == 0) && (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION == 0))
	{
        printf("Number of iterations is 0 both for linear and non-linear registration!\n");
        return EXIT_FAILURE;
	}

	// Check if BROCCOLI_DIR variable is set
	if (getenv("BROCCOLI_DIR") == NULL)
	{
        printf("The environment variable BROCCOLI_DIR is not set!\n");
        return EXIT_FAILURE;
	}
    	
	double startTime = GetWallTime(); 

    // Read first volume (to transform)
	// -----------------------------------

    nifti_image *inputT1 = nifti_image_read(argv[1],1);
    
    if (inputT1 == NULL)
    {
        printf("Could not open first volume!\n");
        return EXIT_FAILURE;
    }
	allNiftiImages[numberOfNiftiImages] = inputT1;
	numberOfNiftiImages++;
    
	// Get data dimensions from input data
    T1_DATA_W = inputT1->nx;
    T1_DATA_H = inputT1->ny;
    T1_DATA_D = inputT1->nz;    
    T1_DATA_T = inputT1->nt;    

	if (T1_DATA_T > 1)
	{
		if (MASK)
		{
			printf("The mask option is currently only valid for a single volume\n");
			MASK = false;
		}
		else if (MASK_ORIGINAL)
		{
			printf("The mask original option is currently only valid for a single volume\n");
			MASK_ORIGINAL = false;
		}
		else if (WRITE_TRANSFORMATION_MATRIX)
		{
			printf("The save matrix option is currently only valid for a single volume\n");
			WRITE_TRANSFORMATION_MATRIX = false;
		}
		else if (WRITE_DISPLACEMENT_FIELD)
		{
			printf("The save displacement field option is currently only valid for a single volume\n");
			WRITE_DISPLACEMENT_FIELD = false;
		}
		else if (WRITE_INTERPOLATED)
		{
			printf("The save interpolated option is currently only valid for a single volume\n");
			WRITE_INTERPOLATED = false;
		}
	}


	// -----------------------------------
	// Read second volume (reference)
	// -----------------------------------

    nifti_image *inputMNI = nifti_image_read(argv[2],1);
    
    if (inputMNI == NULL)
    {
        printf("Could not open second volume!\n");
		FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
	allNiftiImages[numberOfNiftiImages] = inputMNI;
	numberOfNiftiImages++;

	nifti_image *inputMask;
	if (MASK || MASK_ORIGINAL)
	{
		// Check that file extension is .nii or .nii.gz
		std::string extension;
		bool extensionOK;
		CheckFileExtension(MASK_NAME,extensionOK,extension);
		if (!extensionOK)
		{
            printf("File extension is not .nii or .nii.gz, %s is not allowed!\n",extension.c_str());
            return EXIT_FAILURE;
		}

	    inputMask = nifti_image_read(MASK_NAME,1);
    
	    if (inputMask == NULL)
	    {
        	printf("Could not open mask volume!\n");
	        return EXIT_FAILURE;
	    }
		allNiftiImages[numberOfNiftiImages] = inputMask;
		numberOfNiftiImages++;
	}
    	
	// -----------------------------------

	double endTime = GetWallTime();

	if (VERBOS)
 	{
		printf("It took %f seconds to read the two nifti files\n",(float)(endTime - startTime));
	}
        
    MNI_DATA_W = inputMNI->nx;
    MNI_DATA_H = inputMNI->ny;
    MNI_DATA_D = inputMNI->nz;    
    
    // Get voxel sizes from input data
    T1_VOXEL_SIZE_X = inputT1->dx;
    T1_VOXEL_SIZE_Y = inputT1->dy;
    T1_VOXEL_SIZE_Z = inputT1->dz;
                             
    MNI_VOXEL_SIZE_X = inputMNI->dx;
    MNI_VOXEL_SIZE_Y = inputMNI->dy;
    MNI_VOXEL_SIZE_Z = inputMNI->dz;

	// Check if sizes match, for non-linear registration only
	if (NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION == 0)
	{
		if ( (T1_DATA_W != MNI_DATA_W) || (T1_DATA_H != MNI_DATA_H) || (T1_DATA_D != MNI_DATA_D) )
		{
			printf("Reference volume has the dimensions %zu x %zu x %zu, while the input volume has the dimensions %zu x %zu x %zu. Not OK when only doing non-linear registration, aborting! \n",MNI_DATA_W,MNI_DATA_H,MNI_DATA_D,T1_DATA_W,T1_DATA_H,T1_DATA_D);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}
	}
    
	// The filter size is 7, so select a lowest scale that gives at least 10 valid samples (3 data points are lost on each side in each dimension, i.e. 6 total)
	if ( (MNI_DATA_W/16 >= 16) && (MNI_DATA_H/16 >= 16) && (MNI_DATA_D/16 >= 16) )
	{
		COARSEST_SCALE = 16;
	}
	else if ( (MNI_DATA_W/8 >= 16) && (MNI_DATA_H/8 >= 16) && (MNI_DATA_D/8 >= 16) )
	{
		COARSEST_SCALE = 8;
	}
	else if ( (MNI_DATA_W/4 >= 16) && (MNI_DATA_H/4 >= 16) && (MNI_DATA_D/4 >= 16) )
	{
		COARSEST_SCALE = 4;
	}
	else if ( (MNI_DATA_W/2 >= 16) && (MNI_DATA_H/2 >= 16) && (MNI_DATA_D/2 >= 16) )
	{
		COARSEST_SCALE = 2;
	}
	else
	{
		COARSEST_SCALE = 1;
	}
    
	// Check  if mask has same dimensions as reference volume
	if (MASK || MASK_ORIGINAL)
	{
		size_t TEMP_DATA_W = inputMask->nx;
		size_t TEMP_DATA_H = inputMask->ny;
		size_t TEMP_DATA_D = inputMask->nz;

		if ( (TEMP_DATA_W != MNI_DATA_W) || (TEMP_DATA_H != MNI_DATA_H) || (TEMP_DATA_D != MNI_DATA_D) )
		{
			printf("Reference volume has the dimensions %zu x %zu x %zu, while the mask volume has the dimensions %zu x %zu x %zu. Aborting! \n",MNI_DATA_W,MNI_DATA_H,MNI_DATA_D,TEMP_DATA_W,TEMP_DATA_H,TEMP_DATA_D);
			FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
			return EXIT_FAILURE;
		}
	}

    // Calculate sizes, in bytes
    
    size_t DOWNSAMPLED_DATA_W = (int)round((float)MNI_DATA_W/(float)COARSEST_SCALE);
	size_t DOWNSAMPLED_DATA_H = (int)round((float)MNI_DATA_H/(float)COARSEST_SCALE);
	size_t DOWNSAMPLED_DATA_D = (int)round((float)MNI_DATA_D/(float)COARSEST_SCALE);
                
   	size_t T1_DATA_W_INTERPOLATED = (int)round((float)T1_DATA_W * T1_VOXEL_SIZE_X / MNI_VOXEL_SIZE_X);
	size_t T1_DATA_H_INTERPOLATED = (int)round((float)T1_DATA_H * T1_VOXEL_SIZE_Y / MNI_VOXEL_SIZE_Y);
	size_t T1_DATA_D_INTERPOLATED = (int)round((float)T1_DATA_D * T1_VOXEL_SIZE_Z / MNI_VOXEL_SIZE_Z);

    size_t IMAGE_REGISTRATION_PARAMETERS_SIZE = NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS * sizeof(float);
    size_t FILTER_SIZE = IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float);
    size_t FILTER_SIZE2 = IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(cl_float2);
    size_t T1_VOLUME_SIZE = T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float);
    size_t T1_VOLUMES_SIZE = T1_DATA_W * T1_DATA_H * T1_DATA_D * T1_DATA_T * sizeof(float);
    size_t MNI_VOLUME_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float);
    size_t MNI_VOLUMES_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * T1_DATA_T * sizeof(float);
    size_t MNI2_VOLUME_SIZE = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(cl_float2);
    size_t INTERPOLATED_T1_VOLUME_SIZE = T1_DATA_W_INTERPOLATED * T1_DATA_H_INTERPOLATED * T1_DATA_D_INTERPOLATED * sizeof(float);
    size_t DOWNSAMPLED_VOLUME_SIZE = DOWNSAMPLED_DATA_W * DOWNSAMPLED_DATA_H * DOWNSAMPLED_DATA_D * sizeof(float);
    size_t PROJECTION_TENSOR_SIZE = NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION * sizeof(float);
    size_t FILTER_DIRECTIONS_SIZE = NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION * sizeof(float);
    
    // Print some info
    if (PRINT)
    {
        printf("Authored by K.A. Eklund \n");
		if (T1_DATA_T == 1)
		{
	        printf("Volume 1 size: %zu x %zu x %zu \n",  T1_DATA_W, T1_DATA_H, T1_DATA_D);
		}
		else
		{
	        printf("Volumes 1 size: %zu x %zu x %zu x %zu \n",  T1_DATA_W, T1_DATA_H, T1_DATA_D, T1_DATA_T);
		}
        printf("Volume 1 voxel size: %f x %f x %f mm \n", T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z);    
        printf("Volume 2 size: %zu x %zu x %zu \n",  MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
        printf("Volume 2 voxel size: %f x %f x %f mm \n", MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z);    
    }
   	if (VERBOS)
 	{
		printf("Selected lowest scale %i for the registration \n",COARSEST_SCALE);
	}
        
    // ------------------------------------------------
    
    // Allocate memory on the host

	startTime = GetWallTime();
    
	AllocateMemory(h_T1_Volume, T1_VOLUMES_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "INPUT_VOLUME");
	AllocateMemory(h_Interpolated_T1_Volume, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "INTERPOLATED_INPUT_VOLUME");
	AllocateMemory(h_MNI_Volume, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "REFERENCE_VOLUME");
	AllocateMemory(h_Aligned_T1_Volume, MNI_VOLUMES_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "LINEARLY_ALIGNED_INPUT_VOLUME");    
   	AllocateMemory(h_Aligned_T1_Volume_NonLinear, MNI_VOLUMES_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "NONLINEARLY_ALIGNED_INPUT_VOLUME");    
   	AllocateMemory(h_Registration_Parameters, IMAGE_REGISTRATION_PARAMETERS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "REGISTRATION_PARAMETERS");    
        
	if (MASK || MASK_ORIGINAL)
	{
		AllocateMemory(h_MNI_Brain_Mask, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "MNI_BRAIN_MASK");    
		AllocateMemory(h_Skullstripped_T1_Volume, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "SKULLSTRIPPED_VOLUME");    
	}

	AllocateMemory(h_Quadrature_Filter_1_Linear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_1_LINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_1_Linear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_1_LINEAR_REGISTRATION_IMAG");    
	AllocateMemory(h_Quadrature_Filter_2_Linear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_2_LINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_2_Linear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_2_LINEAR_REGISTRATION_IMAG");    
	AllocateMemory(h_Quadrature_Filter_3_Linear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_3_LINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_3_Linear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_3_LINEAR_REGISTRATION_IMAG");    
    
	AllocateMemory(h_Quadrature_Filter_1_NonLinear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_1_NONLINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_1_NonLinear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_1_NONLINEAR_REGISTRATION_IMAG");    
	AllocateMemory(h_Quadrature_Filter_2_NonLinear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_2_NONLINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_2_NonLinear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_2_NONLINEAR_REGISTRATION_IMAG");    
	AllocateMemory(h_Quadrature_Filter_3_NonLinear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_3_NONLINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_3_NonLinear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_3_NONLINEAR_REGISTRATION_IMAG");    
	AllocateMemory(h_Quadrature_Filter_4_NonLinear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_4_NONLINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_4_NonLinear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_4_NONLINEAR_REGISTRATION_IMAG");    
	AllocateMemory(h_Quadrature_Filter_5_NonLinear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_5_NONLINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_5_NonLinear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_5_NONLINEAR_REGISTRATION_IMAG");    
	AllocateMemory(h_Quadrature_Filter_6_NonLinear_Registration_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_6_NONLINEAR_REGISTRATION_REAL");    
	AllocateMemory(h_Quadrature_Filter_6_NonLinear_Registration_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_6_NONLINEAR_REGISTRATION_IMAG");    

    AllocateMemory(h_Projection_Tensor_1, PROJECTION_TENSOR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "PROJECTION_TENSOR_1");    
    AllocateMemory(h_Projection_Tensor_2, PROJECTION_TENSOR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "PROJECTION_TENSOR_2");    
    AllocateMemory(h_Projection_Tensor_3, PROJECTION_TENSOR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "PROJECTION_TENSOR_3");    
    AllocateMemory(h_Projection_Tensor_4, PROJECTION_TENSOR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "PROJECTION_TENSOR_4");    
    AllocateMemory(h_Projection_Tensor_5, PROJECTION_TENSOR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "PROJECTION_TENSOR_5");    
    AllocateMemory(h_Projection_Tensor_6, PROJECTION_TENSOR_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "PROJECTION_TENSOR_6");    

    AllocateMemory(h_Filter_Directions_X, FILTER_DIRECTIONS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "FILTER_DIRECTIONS_X");
    AllocateMemory(h_Filter_Directions_Y, FILTER_DIRECTIONS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "FILTER_DIRECTIONS_Y");        
    AllocateMemory(h_Filter_Directions_Z, FILTER_DIRECTIONS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "FILTER_DIRECTIONS_Z");                
      
    if (WRITE_DISPLACEMENT_FIELD)
    {
	    AllocateMemory(h_Displacement_Field_X, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "DISPLACEMENT_FIELD_X");
		AllocateMemory(h_Displacement_Field_Y, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "DISPLACEMENT_FIELD_Y");        
		AllocateMemory(h_Displacement_Field_Z, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "DISPLACEMENT_FIELD_Z");                
    }
    
    if (DEBUG)
    {                    
		AllocateMemory(h_Quadrature_Filter_Response_1_Real, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_RESPONSE_1_REAL");
		AllocateMemory(h_Quadrature_Filter_Response_1_Imag, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_RESPONSE_1_IMAG");                
		AllocateMemory(h_Quadrature_Filter_Response_2_Real, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_RESPONSE_2_REAL");
		AllocateMemory(h_Quadrature_Filter_Response_2_Imag, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_RESPONSE_2_IMAG");                
		AllocateMemory(h_Quadrature_Filter_Response_3_Real, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_RESPONSE_3_REAL");
		AllocateMemory(h_Quadrature_Filter_Response_3_Imag, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_RESPONSE_3_IMAG");                

		AllocateMemoryFloat2(h_Quadrature_Filter_Response_1, MNI2_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_RESPONSE_1");
		AllocateMemoryFloat2(h_Quadrature_Filter_Response_2, MNI2_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_RESPONSE_2");
		AllocateMemoryFloat2(h_Quadrature_Filter_Response_3, MNI2_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "QUADRATURE_FILTER_RESPONSE_3");

		AllocateMemory(h_Phase_Differences, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "PHASE_DIFFERENCES");                
		AllocateMemory(h_Phase_Certainties, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "PHASE_CERTAINTIES");                
		AllocateMemory(h_Phase_Gradients, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "PHASE_GRADIENTS");                

		AllocateMemory(h_t11, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "TENSOR_11");
		AllocateMemory(h_t12, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "TENSOR_12");                
		AllocateMemory(h_t13, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "TENSOR_13");                
		AllocateMemory(h_t22, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "TENSOR_22");                
		AllocateMemory(h_t23, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "TENSOR_23");                
		AllocateMemory(h_t33, MNI_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "TENSOR_33");                                
    }

	endTime = GetWallTime();
    
	if (VERBOS)
 	{
		printf("It took %f seconds to allocate memory\n",(float)(endTime - startTime));
	}

	startTime = GetWallTime();

    // Convert data to floats
    if ( inputT1->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputT1->data;
    
        for (size_t i = 0; i < T1_DATA_W * T1_DATA_H * T1_DATA_D * T1_DATA_T; i++)
        {
            h_T1_Volume[i] = (float)p[i];
        }
    }
    else if ( inputT1->datatype == DT_UINT8 )
    {
        unsigned char *p = (unsigned char*)inputT1->data;
    
        for (size_t i = 0; i < T1_DATA_W * T1_DATA_H * T1_DATA_D * T1_DATA_T; i++)
        {
            h_T1_Volume[i] = (float)p[i];
        }
    }
    else if ( inputT1->datatype == DT_FLOAT )
    {
        float *p = (float*)inputT1->data;
    
        for (size_t i = 0; i < T1_DATA_W * T1_DATA_H * T1_DATA_D * T1_DATA_T; i++)
        {
            h_T1_Volume[i] = p[i];
        }
    }
    else
    {
        printf("Unknown data type in input volume, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
		FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
    
    if ( inputMNI->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputMNI->data;
    
        for (size_t i = 0; i < MNI_DATA_W * MNI_DATA_H * MNI_DATA_D; i++)
        {
            h_MNI_Volume[i] = (float)p[i];
        }
    }
    else if ( inputMNI->datatype == DT_UINT8 )
    {
        unsigned char *p = (unsigned char*)inputMNI->data;
    
        for (size_t i = 0; i < MNI_DATA_W * MNI_DATA_H * MNI_DATA_D; i++)
        {
            h_MNI_Volume[i] = (float)p[i];
        }
    }
    else if ( inputMNI->datatype == DT_FLOAT )
    {
        float *p = (float*)inputMNI->data;
    
        for (size_t i = 0; i < MNI_DATA_W * MNI_DATA_H * MNI_DATA_D; i++)
        {
            h_MNI_Volume[i] = p[i];
        }
    }
    else
    {
        printf("Unknown data type %i in reference volume, aborting!\n",inputMNI->datatype);
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
		FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
    
	if (MASK || MASK_ORIGINAL)
	{
	    if ( inputMask->datatype == DT_SIGNED_SHORT )
	    {
	        short int *p = (short int*)inputMask->data;
    
	        for (size_t i = 0; i < MNI_DATA_W * MNI_DATA_H * MNI_DATA_D; i++)
	        {
	            h_MNI_Brain_Mask[i] = (float)p[i];
	        }
	    }
	    else if ( inputMask->datatype == DT_FLOAT )
	    {
	        float *p = (float*)inputMask->data;
    
	        for (size_t i = 0; i < MNI_DATA_W * MNI_DATA_H * MNI_DATA_D; i++)
        	{
	            h_MNI_Brain_Mask[i] = p[i];
	        }
	    }
	    else if ( inputMask->datatype == DT_UINT8 )
	    {
    	    unsigned char *p = (unsigned char*)inputMask->data;
    
	        for (size_t i = 0; i < MNI_DATA_W * MNI_DATA_H * MNI_DATA_D; i++)
	        {
	            h_MNI_Brain_Mask[i] = (float)p[i];
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

	startTime = GetWallTime();

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
    
    // Read quadrature filters for linear registration, three real valued and three imaginary valued
	ReadBinaryFile(h_Quadrature_Filter_1_Linear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter1RealLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_1_Linear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter1ImagLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_2_Linear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter2RealLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_2_Linear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter2ImagLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_3_Linear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter3RealLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_3_Linear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter3ImagLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 

	std::string filter1RealNonLinearPathAndName;
	std::string filter1ImagNonLinearPathAndName;
	std::string filter2RealNonLinearPathAndName;
	std::string filter2ImagNonLinearPathAndName;
	std::string filter3RealNonLinearPathAndName;
	std::string filter3ImagNonLinearPathAndName;
	std::string filter4RealNonLinearPathAndName;
	std::string filter4ImagNonLinearPathAndName;
	std::string filter5RealNonLinearPathAndName;
	std::string filter5ImagNonLinearPathAndName;
	std::string filter6RealNonLinearPathAndName;
	std::string filter6ImagNonLinearPathAndName;

	filter1RealNonLinearPathAndName.append(getenv("BROCCOLI_DIR"));
	filter1ImagNonLinearPathAndName.append(getenv("BROCCOLI_DIR"));
	filter2RealNonLinearPathAndName.append(getenv("BROCCOLI_DIR"));
	filter2ImagNonLinearPathAndName.append(getenv("BROCCOLI_DIR"));
	filter3RealNonLinearPathAndName.append(getenv("BROCCOLI_DIR"));
	filter3ImagNonLinearPathAndName.append(getenv("BROCCOLI_DIR"));
	filter4RealNonLinearPathAndName.append(getenv("BROCCOLI_DIR"));
	filter4ImagNonLinearPathAndName.append(getenv("BROCCOLI_DIR"));
	filter5RealNonLinearPathAndName.append(getenv("BROCCOLI_DIR"));
	filter5ImagNonLinearPathAndName.append(getenv("BROCCOLI_DIR"));
	filter6RealNonLinearPathAndName.append(getenv("BROCCOLI_DIR"));
	filter6ImagNonLinearPathAndName.append(getenv("BROCCOLI_DIR"));

	filter1RealNonLinearPathAndName.append("filters/filter1_real_nonlinear_registration.bin");
	filter1ImagNonLinearPathAndName.append("filters/filter1_imag_nonlinear_registration.bin");
	filter2RealNonLinearPathAndName.append("filters/filter2_real_nonlinear_registration.bin");
	filter2ImagNonLinearPathAndName.append("filters/filter2_imag_nonlinear_registration.bin");
	filter3RealNonLinearPathAndName.append("filters/filter3_real_nonlinear_registration.bin");
	filter3ImagNonLinearPathAndName.append("filters/filter3_imag_nonlinear_registration.bin");
	filter4RealNonLinearPathAndName.append("filters/filter4_real_nonlinear_registration.bin");
	filter4ImagNonLinearPathAndName.append("filters/filter4_imag_nonlinear_registration.bin");
	filter5RealNonLinearPathAndName.append("filters/filter5_real_nonlinear_registration.bin");
	filter5ImagNonLinearPathAndName.append("filters/filter5_imag_nonlinear_registration.bin");
	filter6RealNonLinearPathAndName.append("filters/filter6_real_nonlinear_registration.bin");
	filter6ImagNonLinearPathAndName.append("filters/filter6_imag_nonlinear_registration.bin");

	// Read quadrature filters for nonLinear registration, six real valued and six imaginary valued
	ReadBinaryFile(h_Quadrature_Filter_1_NonLinear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter1RealNonLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_1_NonLinear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter1ImagNonLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_2_NonLinear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter2RealNonLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_2_NonLinear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter2ImagNonLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_3_NonLinear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter3RealNonLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_3_NonLinear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter3ImagNonLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_4_NonLinear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter4RealNonLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_4_NonLinear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter4ImagNonLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_5_NonLinear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter5RealNonLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_5_NonLinear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter5ImagNonLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_6_NonLinear_Registration_Real,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter6RealNonLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
	ReadBinaryFile(h_Quadrature_Filter_6_NonLinear_Registration_Imag,IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE*IMAGE_REGISTRATION_FILTER_SIZE,filter6ImagNonLinearPathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 

	std::string projectionTensor1PathAndName;
	std::string projectionTensor2PathAndName;
	std::string projectionTensor3PathAndName;
	std::string projectionTensor4PathAndName;
	std::string projectionTensor5PathAndName;
	std::string projectionTensor6PathAndName;

	projectionTensor1PathAndName.append(getenv("BROCCOLI_DIR"));
	projectionTensor2PathAndName.append(getenv("BROCCOLI_DIR"));
	projectionTensor3PathAndName.append(getenv("BROCCOLI_DIR"));
	projectionTensor4PathAndName.append(getenv("BROCCOLI_DIR"));
	projectionTensor5PathAndName.append(getenv("BROCCOLI_DIR"));
	projectionTensor6PathAndName.append(getenv("BROCCOLI_DIR"));

	projectionTensor1PathAndName.append("filters/projection_tensor1.bin");
	projectionTensor2PathAndName.append("filters/projection_tensor2.bin");
	projectionTensor3PathAndName.append("filters/projection_tensor3.bin");
	projectionTensor4PathAndName.append("filters/projection_tensor4.bin");
	projectionTensor5PathAndName.append("filters/projection_tensor5.bin");
	projectionTensor6PathAndName.append("filters/projection_tensor6.bin");

    // Read projection tensors   
    ReadBinaryFile(h_Projection_Tensor_1,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,projectionTensor1PathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
    ReadBinaryFile(h_Projection_Tensor_2,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,projectionTensor2PathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
    ReadBinaryFile(h_Projection_Tensor_3,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,projectionTensor3PathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
    ReadBinaryFile(h_Projection_Tensor_4,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,projectionTensor4PathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
    ReadBinaryFile(h_Projection_Tensor_5,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,projectionTensor5PathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
    ReadBinaryFile(h_Projection_Tensor_6,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,projectionTensor6PathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
        
	std::string filterDirections1PathAndName;
	std::string filterDirections2PathAndName;
	std::string filterDirections3PathAndName;

	filterDirections1PathAndName.append(getenv("BROCCOLI_DIR"));
	filterDirections2PathAndName.append(getenv("BROCCOLI_DIR"));
	filterDirections3PathAndName.append(getenv("BROCCOLI_DIR"));

	filterDirections1PathAndName.append("filters/filter_directions_x.bin");
	filterDirections2PathAndName.append("filters/filter_directions_y.bin");
	filterDirections3PathAndName.append("filters/filter_directions_z.bin");

    // Read filter directions
    ReadBinaryFile(h_Filter_Directions_X,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,filterDirections1PathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages);
    ReadBinaryFile(h_Filter_Directions_Y,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,filterDirections2PathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages); 
    ReadBinaryFile(h_Filter_Directions_Z,NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION,filterDirections3PathAndName.c_str(),allMemoryPointers,numberOfMemoryPointers,allNiftiImages,numberOfNiftiImages);  

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
    if ( !BROCCOLI.GetOpenCLInitiated() )
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
        BROCCOLI.SetInputT1Volume(h_T1_Volume);
        BROCCOLI.SetInputMNIBrainVolume(h_MNI_Volume);
        BROCCOLI.SetInputMNIBrainMask(h_MNI_Brain_Mask);

		BROCCOLI.SetAllocatedHostMemory(allocatedHostMemory);
        
        BROCCOLI.SetT1Width(T1_DATA_W);
        BROCCOLI.SetT1Height(T1_DATA_H);
        BROCCOLI.SetT1Depth(T1_DATA_D);
        BROCCOLI.SetT1Timepoints(T1_DATA_T);
        BROCCOLI.SetT1VoxelSizeX(T1_VOXEL_SIZE_X);
        BROCCOLI.SetT1VoxelSizeY(T1_VOXEL_SIZE_Y);
        BROCCOLI.SetT1VoxelSizeZ(T1_VOXEL_SIZE_Z);   
        
        BROCCOLI.SetMNIWidth(MNI_DATA_W);
        BROCCOLI.SetMNIHeight(MNI_DATA_H);
        BROCCOLI.SetMNIDepth(MNI_DATA_D);    
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
        BROCCOLI.SetCoarsestScaleT1MNI(COARSEST_SCALE);
        BROCCOLI.SetMMT1ZCUT(MM_T1_Z_CUT);   

		BROCCOLI.SetTsigma(SIGMA);
		BROCCOLI.SetEsigma(SIGMA);
		BROCCOLI.SetDsigma(SIGMA);
        
        BROCCOLI.SetOutputInterpolatedT1Volume(h_Interpolated_T1_Volume);
        BROCCOLI.SetOutputAlignedT1VolumeLinear(h_Aligned_T1_Volume);
        BROCCOLI.SetOutputAlignedT1VolumeNonLinear(h_Aligned_T1_Volume_NonLinear);                
		BROCCOLI.SetOutputSkullstrippedT1Volume(h_Skullstripped_T1_Volume);
        BROCCOLI.SetOutputT1MNIRegistrationParameters(h_Registration_Parameters);
        
		BROCCOLI.SetPrecenterRegistration(PRECENTER_REGISTRATION);

		BROCCOLI.SetDoSkullstrip(MASK);
		BROCCOLI.SetDoSkullstripOriginal(MASK_ORIGINAL);

		BROCCOLI.SetSaveDisplacementField(WRITE_DISPLACEMENT_FIELD);
		BROCCOLI.SetSaveInterpolatedT1(WRITE_INTERPOLATED);
		BROCCOLI.SetSaveAlignedT1MNILinear(true);
		BROCCOLI.SetSaveAlignedT1MNINonLinear(true);		

        BROCCOLI.SetOutputDisplacementField(h_Displacement_Field_X,h_Displacement_Field_Y,h_Displacement_Field_Z);

        if (DEBUG)
        {
            BROCCOLI.SetDebug(true);
            
            //BROCCOLI.SetOutputSkullstrippedT1Volume(h_Skullstripped_T1_Volume);
            
            //BROCCOLI.SetOutputDownsampledVolume(h_Downsampled_Volume);
        
            BROCCOLI.SetOutputQuadratureFilterResponses(h_Quadrature_Filter_Response_1, h_Quadrature_Filter_Response_2, h_Quadrature_Filter_Response_3);
            //BROCCOLI.SetOutputQuadratureFilterResponses(h_Quadrature_Filter_Response_1_Real, h_Quadrature_Filter_Response_1_Imag, h_Quadrature_Filter_Response_2_Real, h_Quadrature_Filter_Response_2_Imag, h_Quadrature_Filter_Response_3_Real, h_Quadrature_Filter_Response_3_Imag);
            //BROCCOLI.SetOutputQuadratureFilterResponses(h_Quadrature_Filter_Response_1_Real, h_Quadrature_Filter_Response_1_Imag, h_Quadrature_Filter_Response_2_Real, h_Quadrature_Filter_Response_2_Imag, h_Quadrature_Filter_Response_3_Real, h_Quadrature_Filter_Response_3_Imag);
            //BROCCOLI.SetOutputQuadratureFilterResponses(h_Quadrature_Filter_Response_1, h_Quadrature_Filter_Response_2, h_Quadrature_Filter_Response_3, h_Quadrature_Filter_Response_4, h_Quadrature_Filter_Response_5, h_Quadrature_Filter_Response_6);
            //BROCCOLI.SetOutputTensorComponents(h_t11, h_t12, h_t13, h_t22, h_t23, h_t33);            
            BROCCOLI.SetOutputPhaseDifferences(h_Phase_Differences);
            BROCCOLI.SetOutputPhaseCertainties(h_Phase_Certainties);
            BROCCOLI.SetOutputPhaseGradients(h_Phase_Gradients);
            //BROCCOLI.SetOutputSliceSums(h_Slice_Sums);
            //BROCCOLI.SetOutputTopSlice(h_Top_Slice);
            //BROCCOLI.SetOutputAMatrix(h_A_Matrix);
            //BROCCOLI.SetOutputHVector(h_h_Vector);
        }
                            
        // Run the actual registration

		startTime = GetWallTime();
        BROCCOLI.PerformRegistrationTwoVolumesWrapper();     
		endTime = GetWallTime();

		if (VERBOS)
	 	{
			printf("\nIt took %f seconds to run the registration\n",(float)(endTime - startTime));
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
           

	if (NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION > 0 && PRINT && (T1_DATA_T == 1))
	{
		printf("\nAffine registration parameters\n");
		printf(" %f %f %f %f\n", h_Registration_Parameters[3]+1.0f,h_Registration_Parameters[4],h_Registration_Parameters[5],h_Registration_Parameters[0]);
		printf(" %f %f %f %f\n", h_Registration_Parameters[6],h_Registration_Parameters[7]+1.0f,h_Registration_Parameters[8],h_Registration_Parameters[1]);
		printf(" %f %f %f %f\n", h_Registration_Parameters[9],h_Registration_Parameters[10],h_Registration_Parameters[11]+1.0f,h_Registration_Parameters[2]);
		printf(" %f %f %f %f\n\n", 0.0f,0.0f,0.0f,1.0f);
	}


    // Print registration parameters to file
	if (WRITE_TRANSFORMATION_MATRIX && (NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION > 0))
	{
		// Add the provided filename extension to the original filename, before the dot

		const char* extension = "_affinematrix.txt";
		char* filenameWithExtension;

		CreateFilename(filenameWithExtension, inputT1, extension, CHANGE_OUTPUT_FILENAME, outputFilename);

	    std::ofstream matrix;
	    matrix.open(filenameWithExtension);      
	    if ( matrix.good() )
	    {
	        matrix.precision(6);

	        matrix << h_Registration_Parameters[3]+1.0f << std::setw(2) << " " << h_Registration_Parameters[4] << std::setw(2) << " " << h_Registration_Parameters[5] << std::setw(2) << " " << h_Registration_Parameters[0] << std::endl;
	        matrix << h_Registration_Parameters[6] << std::setw(2) << " " << h_Registration_Parameters[7] + 1.0f << std::setw(2) << " " << h_Registration_Parameters[8] << std::setw(2) << " " << h_Registration_Parameters[1] << std::endl;
	        matrix << h_Registration_Parameters[9] << std::setw(2) << " " << h_Registration_Parameters[10] << std::setw(2) << " " << h_Registration_Parameters[11] + 1.0f << std::setw(2) << " " << h_Registration_Parameters[2] << std::endl;
	        matrix << 0.0f << std::setw(2) << " " << 0.0f << std::setw(2) << " " << 0.0f << std::setw(2) << " " << 1.0f << std::endl;

	        matrix.close();
	    }
	    else
	    {
	        printf("Could not open %s for writing!\n",filenameWithExtension);
	    }
		free(filenameWithExtension);
	}

    // Create new nifti image
	nifti_image *outputNifti = nifti_copy_nim_info(inputMNI);      
	allNiftiImages[numberOfNiftiImages] = outputNifti;
	numberOfNiftiImages++;    

	if (T1_DATA_T > 1)
	{
		outputNifti->nt = T1_DATA_T;
		outputNifti->ndim = 4;
		outputNifti->dim[0] = 4;
	    outputNifti->dim[4] = T1_DATA_T;
	    outputNifti->nvox = MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * T1_DATA_T;
	}

    // Copy information from input data    	
	if (!CHANGE_OUTPUT_FILENAME)
	{
    	nifti_set_filenames(outputNifti, inputT1->fname, 0, 1);    
	}
	else
	{
		nifti_set_filenames(outputNifti, outputFilename, 0, 1);    
	}

    startTime = GetWallTime();

    // Write aligned data to file, as the size of the MNI volume  
	if (WRITE_INTERPOLATED)
	{          
    	WriteNifti(outputNifti,h_Interpolated_T1_Volume,"_interpolated",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}

	if (NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION > 0)
	{
	    WriteNifti(outputNifti,h_Aligned_T1_Volume,"_aligned_linear",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}
	
	if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
	{
   		WriteNifti(outputNifti,h_Aligned_T1_Volume_NonLinear,"_aligned_nonlinear",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}

	if (MASK || MASK_ORIGINAL)
	{
   		WriteNifti(outputNifti,h_Skullstripped_T1_Volume,"_masked",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}
              	
	if (WRITE_DISPLACEMENT_FIELD && (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0))
    {
        WriteNifti(outputNifti,h_Displacement_Field_X,"_displacement_x",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(outputNifti,h_Displacement_Field_Y,"_displacement_y",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(outputNifti,h_Displacement_Field_Z,"_displacement_z",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
    }
    
    if (DEBUG)
    {
        WriteNifti(outputNifti,h_Phase_Gradients,"_phase_gradients",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(outputNifti,h_Phase_Certainties,"_phase_certainties",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(outputNifti,h_Phase_Differences,"_phase_differences",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        
        ConvertFloat2ToFloats(h_Quadrature_Filter_Response_1_Real, h_Quadrature_Filter_Response_1_Imag, h_Quadrature_Filter_Response_1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
        ConvertFloat2ToFloats(h_Quadrature_Filter_Response_2_Real, h_Quadrature_Filter_Response_2_Imag, h_Quadrature_Filter_Response_2, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
        ConvertFloat2ToFloats(h_Quadrature_Filter_Response_3_Real, h_Quadrature_Filter_Response_3_Imag, h_Quadrature_Filter_Response_3, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
        
        WriteNifti(outputNifti,h_Quadrature_Filter_Response_1_Real,"_quadrature_filter_response_1_real",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(outputNifti,h_Quadrature_Filter_Response_1_Imag,"_quadrature_filter_response_1_imag",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(outputNifti,h_Quadrature_Filter_Response_2_Real,"_quadrature_filter_response_2_real",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(outputNifti,h_Quadrature_Filter_Response_2_Imag,"_quadrature_filter_response_2_imag",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(outputNifti,h_Quadrature_Filter_Response_3_Real,"_quadrature_filter_response_3_real",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
        WriteNifti(outputNifti,h_Quadrature_Filter_Response_3_Imag,"_quadrature_filter_response_3_imag",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);        
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

/*
fid = fopen('filter6_real_nonLinear_registration.bin','w')
a = f6_nonLinear_registration;
a = real(a);
a(:,:,1) = a(:,:,1)';
a(:,:,2) = a(:,:,2)';
a(:,:,3) = a(:,:,3)';
a(:,:,4) = a(:,:,4)';
a(:,:,5) = a(:,:,5)';
a(:,:,6) = a(:,:,6)';
a(:,:,7) = a(:,:,7)';        
count = fwrite(fid,single(a(:)),'float')
fclose(fid)
 
 fid = fopen('projection_tensor6.bin','w')
 count = fwrite(fid,single(m6),'float')
 fclose(fid)
 
 
*/

