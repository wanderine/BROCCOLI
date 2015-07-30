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

#include "HelpFunctions.cpp"

#define ADD_FILENAME true
#define DONT_ADD_FILENAME true

#define CHECK_EXISTING_FILE true
#define DONT_CHECK_EXISTING_FILE false

int main(int argc, char **argv)
{
    //-----------------------
    // Input pointers
    
    float           *h_Input_Volume;
    float           *h_Displacement_Field_X, *h_Displacement_Field_Y, *h_Displacement_Field_Z;            

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

    //-----------------------
    // Output pointers        
        
    float           *h_Interpolated_Volume;
    
    // Default parameters
        
    int             OPENCL_PLATFORM = 0;
    int             OPENCL_DEVICE = 0;
	int				INTERPOLATION_MODE = 1;
    bool            DEBUG = false;
    bool            PRINT = true;
	bool			CHANGE_OUTPUT_FILENAME = false;    
	int 			MM_T1_Z_CUT = 0;

	const char*		outputFilename;

	bool			VERBOS = false;

    // Size parameters
    size_t          INPUT_DATA_H, INPUT_DATA_W, INPUT_DATA_D;
    size_t          REFERENCE_DATA_H, REFERENCE_DATA_W, REFERENCE_DATA_D;
           
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
		printf(" -interpolation             The interpolation to use, 0 = nearest, 1 = trilinear (default 1) \n");
		printf(" -zcut                      Number of mm to cut from the bottom of the input volume, can be negative (default 0). Should be the same as for the call to RegisterTwoVolumes\n"); 
		printf(" -output                    Set output filename (default volume_to_transform_warped.nii) \n");
        printf(" -quiet                     Don't print anything to the terminal (default false) \n");
        printf("\n\n");
        
        return EXIT_SUCCESS;
    }
    else if (argc < 6)
    {
        printf("Need one volume to warp, one reference volume and three displacement field volumes!\n\n");
		return EXIT_FAILURE;
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

        fp = fopen(argv[3],"r");
        if (fp == NULL)
        {
            printf("Could not open file %s !\n",argv[3]);
            return EXIT_FAILURE;
        }
        fclose(fp);   

        fp = fopen(argv[4],"r");
        if (fp == NULL)
        {
            printf("Could not open file %s !\n",argv[4]);
            return EXIT_FAILURE;
        }
        fclose(fp);   

        fp = fopen(argv[5],"r");
        if (fp == NULL)
        {
            printf("Could not open file %s !\n",argv[5]);
            return EXIT_FAILURE;
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
		else if (strcmp(input,"-interpolation") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -interpolation !\n");
                return EXIT_FAILURE;
			}

            INTERPOLATION_MODE = (int)strtol(argv[i+1], &p, 10);

			if (!isspace(*p) && *p != 0)
		    {
		        printf("Interpolation mode must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
			if ( (INTERPOLATION_MODE != 0) && (INTERPOLATION_MODE != 1) )
            {
			    printf("Interpolation mode has to be 0 or 1!\n");
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
        else if (strcmp(input,"-quiet") == 0)
        {
            PRINT = false;
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

    // Read data
    nifti_image *inputVolume = nifti_image_read(argv[1],1);    
    if (inputVolume == NULL)
    {
        printf("Could not open volume to transform!\n");
        return EXIT_FAILURE;
    }
	allNiftiImages[numberOfNiftiImages] = inputVolume;
	numberOfNiftiImages++;

    nifti_image *referenceVolume = nifti_image_read(argv[2],1);    
    if (referenceVolume == NULL)
    {
        printf("Could not open reference volume!\n");
        return EXIT_FAILURE;
    }
   	allNiftiImages[numberOfNiftiImages] = referenceVolume;
	numberOfNiftiImages++;

    nifti_image *inputDisplacementX = nifti_image_read(argv[3],1);   
    if (inputDisplacementX == NULL)
    {
        printf("Could not open displacement X volume!\n");
        return EXIT_FAILURE;
    }
	allNiftiImages[numberOfNiftiImages] = inputDisplacementX;
	numberOfNiftiImages++;

    nifti_image *inputDisplacementY = nifti_image_read(argv[4],1);   
    if (inputDisplacementY == NULL)
    {
        printf("Could not open displacement Y volume!\n");
        return EXIT_FAILURE;
    }
	allNiftiImages[numberOfNiftiImages] = inputDisplacementY;
	numberOfNiftiImages++;

    nifti_image *inputDisplacementZ = nifti_image_read(argv[5],1);   
    if (inputDisplacementZ == NULL)
    {
        printf("Could not open displacement Z volume!\n");
        return EXIT_FAILURE;
    }
	allNiftiImages[numberOfNiftiImages] = inputDisplacementZ;
	numberOfNiftiImages++;
    
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
        printf("Dimensions of displacement field X does not match the reference volume!\n");
        return -1;
	}

	DISPLACEMENT_DATA_W = inputDisplacementY->nx;
	DISPLACEMENT_DATA_H = inputDisplacementY->ny;
	DISPLACEMENT_DATA_D = inputDisplacementY->nz;

	if ( (DISPLACEMENT_DATA_W != REFERENCE_DATA_W) || (DISPLACEMENT_DATA_H != REFERENCE_DATA_H) || (DISPLACEMENT_DATA_D != REFERENCE_DATA_D) )
	{
        printf("Dimensions of displacement field Y does not match the reference volume!\n");
        return -1;
	}

	DISPLACEMENT_DATA_W = inputDisplacementZ->nx;
	DISPLACEMENT_DATA_H = inputDisplacementZ->ny;
	DISPLACEMENT_DATA_D = inputDisplacementZ->nz;

	if ( (DISPLACEMENT_DATA_W != REFERENCE_DATA_W) || (DISPLACEMENT_DATA_H != REFERENCE_DATA_H) || (DISPLACEMENT_DATA_D != REFERENCE_DATA_D) )
	{
        printf("Dimensions of displacement field Z does not match the reference volume!\n");
        return -1;
	}

    // Calculate size, in bytes
    
    int INPUT_VOLUME_SIZE = INPUT_DATA_W * INPUT_DATA_H * INPUT_DATA_D * sizeof(float);
    int REFERENCE_VOLUME_SIZE = REFERENCE_DATA_W * REFERENCE_DATA_H * REFERENCE_DATA_D * sizeof(float);

    // Print some info
    if (PRINT)
    {
        printf("Authored by K.A. Eklund \n");
        printf("Input volume size: %zu x %zu x %zu \n",  INPUT_DATA_W, INPUT_DATA_H, INPUT_DATA_D);
        printf("Input volume voxel size: %f x %f x %f \n",  INPUT_VOXEL_SIZE_X, INPUT_VOXEL_SIZE_Y, INPUT_VOXEL_SIZE_Z);
        printf("Reference volume size: %zu x %zu x %zu \n",  REFERENCE_DATA_W, REFERENCE_DATA_H, REFERENCE_DATA_D);
        printf("Reference volume voxel size: %f x %f x %f \n",  REFERENCE_VOXEL_SIZE_X, REFERENCE_VOXEL_SIZE_Y, REFERENCE_VOXEL_SIZE_Z);
    }
           
    // ------------------------------------------------
    
    // Allocate memory on the host        

	AllocateMemory(h_Input_Volume, INPUT_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "INPUT_VOLUME");
	AllocateMemory(h_Interpolated_Volume, REFERENCE_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "INTERPOLATED_VOLUME");
	AllocateMemory(h_Displacement_Field_X, REFERENCE_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "DISPLACEMENT_FIELD_X");
	AllocateMemory(h_Displacement_Field_Y, REFERENCE_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "DISPLACEMENT_FIELD_Y");
	AllocateMemory(h_Displacement_Field_Z, REFERENCE_VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "DISPLACEMENT_FIELD_Z");
			           
    // Convert data to floats
    if ( inputVolume->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputVolume->data;
    
        for (size_t i = 0; i < INPUT_DATA_W * INPUT_DATA_H * INPUT_DATA_D; i++)
        {
            h_Input_Volume[i] = (float)p[i];
        }
    }
    else if ( inputVolume->datatype == DT_FLOAT )
    {
        float *p = (float*)inputVolume->data;
    
        for (size_t i = 0; i < INPUT_DATA_W * INPUT_DATA_H * INPUT_DATA_D; i++)
        {
            h_Input_Volume[i] = p[i];
        }
    }
    else if ( inputVolume->datatype == DT_UINT8 )
    {
        unsigned char *p = (unsigned char*)inputVolume->data;
    
        for (size_t i = 0; i < INPUT_DATA_W * INPUT_DATA_H * INPUT_DATA_D; i++)
        {
            h_Input_Volume[i] = (float)p[i];
        }
    }
    else
    {
        printf("Unknown data type in volume to transform, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
    
    if ( inputDisplacementX->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputDisplacementX->data;
    
        for (size_t i = 0; i < REFERENCE_DATA_W * REFERENCE_DATA_H * REFERENCE_DATA_D; i++)
        {
            h_Displacement_Field_X[i] = (float)p[i];
        }
    }
    else if ( inputDisplacementX->datatype == DT_FLOAT )
    {
        float *p = (float*)inputDisplacementX->data;
    
        for (size_t i = 0; i < REFERENCE_DATA_W * REFERENCE_DATA_H * REFERENCE_DATA_D; i++)
        {
            h_Displacement_Field_X[i] = p[i];
        }
    }
    else
    {
        printf("Unknown data type in displacement x volume, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }

    if ( inputDisplacementY->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputDisplacementY->data;
    
        for (size_t i = 0; i < REFERENCE_DATA_W * REFERENCE_DATA_H * REFERENCE_DATA_D; i++)
        {
	        h_Displacement_Field_Y[i] = (float)p[i];
        }
    }
    else if ( inputDisplacementY->datatype == DT_FLOAT )
    {
        float *p = (float*)inputDisplacementY->data;
    
        for (size_t i = 0; i < REFERENCE_DATA_W * REFERENCE_DATA_H * REFERENCE_DATA_D; i++)
        {
            h_Displacement_Field_Y[i] = p[i];			
        }
    }
    else
    {
        printf("Unknown data type in displacement y volume, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }

    if ( inputDisplacementZ->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputDisplacementZ->data;
    
        for (size_t i = 0; i < REFERENCE_DATA_W * REFERENCE_DATA_H * REFERENCE_DATA_D; i++)
        {
            h_Displacement_Field_Z[i] = (float)p[i];
        }
    }
    else if ( inputDisplacementZ->datatype == DT_FLOAT )
    {
        float *p = (float*)inputDisplacementZ->data;
    
        for (size_t i = 0; i < REFERENCE_DATA_W * REFERENCE_DATA_H * REFERENCE_DATA_D; i++)
        {
            h_Displacement_Field_Z[i] = p[i];
        }
    }
    else
    {
        printf("Unknown data type in displacement z volume, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
            
    //------------------------
    
    BROCCOLI_LIB BROCCOLI(OPENCL_PLATFORM,OPENCL_DEVICE,2,VERBOS); // 2 = Bash wrapper    

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
        BROCCOLI.SetInputT1Volume(h_Input_Volume);        
        BROCCOLI.SetT1Width(INPUT_DATA_W);
        BROCCOLI.SetT1Height(INPUT_DATA_H);
        BROCCOLI.SetT1Depth(INPUT_DATA_D);  
		BROCCOLI.SetT1VoxelSizeX(INPUT_VOXEL_SIZE_X);
		BROCCOLI.SetT1VoxelSizeY(INPUT_VOXEL_SIZE_Y);
		BROCCOLI.SetT1VoxelSizeZ(INPUT_VOXEL_SIZE_Z);
     
		BROCCOLI.SetAllocatedHostMemory(allocatedHostMemory);
        
        BROCCOLI.SetMNIWidth(REFERENCE_DATA_W);
        BROCCOLI.SetMNIHeight(REFERENCE_DATA_H);
        BROCCOLI.SetMNIDepth(REFERENCE_DATA_D);
		BROCCOLI.SetMNIVoxelSizeX(REFERENCE_VOXEL_SIZE_X);
		BROCCOLI.SetMNIVoxelSizeY(REFERENCE_VOXEL_SIZE_Y);
		BROCCOLI.SetMNIVoxelSizeZ(REFERENCE_VOXEL_SIZE_Z);
		
		BROCCOLI.SetMMT1ZCUT(MM_T1_Z_CUT); 
        BROCCOLI.SetInterpolationMode(INTERPOLATION_MODE);  

		BROCCOLI.SetOutputDisplacementField(h_Displacement_Field_X,h_Displacement_Field_Y,h_Displacement_Field_Z);      
        BROCCOLI.SetOutputInterpolatedT1Volume(h_Interpolated_Volume);
                
        if (DEBUG)
        {
            BROCCOLI.SetDebug(true);            
        }
                          
        // Run the actual transformation
        BROCCOLI.TransformVolumesNonLinearWrapper();
		
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
   
    // Copy information from input data
	nifti_image* outputNifti = nifti_copy_nim_info(referenceVolume);    	
	allNiftiImages[numberOfNiftiImages] = outputNifti;
	numberOfNiftiImages++;    

	// Change filename and write transformed data to file
	if (!CHANGE_OUTPUT_FILENAME)
	{
    	nifti_set_filenames(outputNifti, inputVolume->fname, 0, 1);
	    WriteNifti(outputNifti,h_Interpolated_Volume,"_warped",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
	}
	else 
	{
    	nifti_set_filenames(outputNifti, outputFilename, 0, 1);    	
	    WriteNifti(outputNifti,h_Interpolated_Volume,"",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);    
	}
                             
    // Free all memory
	FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);	   
	FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
    
	return EXIT_SUCCESS;
}



