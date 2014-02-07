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
            free(pointers[i]);
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
    
    void            *allMemoryPointers[500];
    int             numberOfMemoryPointers = 0;
    
    // Default parameters
    int             MOTION_CORRECTION_FILTER_SIZE = 7; 
    int             NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION = 5;
    int             OPENCL_PLATFORM = 2;
    int             OPENCL_DEVICE = 0;
    int             NUMBER_OF_MOTION_CORRECTION_PARAMETERS = 6;    
    bool            DEBUG = false;
    //bool            DEBUG = true;
    const char*     FILENAME_EXTENSION = "_mc";
    bool            PRINT = true;
    
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
        printf("-platform   The OpenCL platform to use (default 0) \n");
        printf("-device     The OpenCL device to use for the specificed platform (default 0) \n");
        printf("-iterations Number of iterations for the motion correction algorithm (default 5) \n");        
        printf("-output     Set output filename (default input_mc.nii) \n");
        printf("-quiet      Don't print anything to the terminal (default false) \n");
        printf("-debug      Get additional debug information (default no) \n");
        printf("\n\n");
        
        return 1;
    }
    // Try to open file
    else if (argc > 1)
    {        
        fp = fopen(argv[1],"r");
        if (fp == NULL)
        {            
            printf("Could not open file %s !\n",argv[1]);
            return -1;
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
        else if (strcmp(input,"-iterations") == 0)
        {
            NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION = (int)strtol(argv[i+1], &p, 10);
            if (NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION <= 0)
            {
                printf("Number of iterations must be a positive number!\n");
                return -1;
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
        else if (strcmp(input,"-output") == 0)
        {
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
    nifti_image *inputData = nifti_image_read(argv[1],1);
    
    if (inputData == NULL)
    {
        printf("Could not open nifti file!\n");
        return -1;
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
        printf("OpenCL Platform: %i OpenCL Device %i\n", OPENCL_PLATFORM, OPENCL_DEVICE);
        printf("Data size: %i x %i x %i x %i \n",  DATA_W, DATA_H, DATA_D, DATA_T);
        printf("Voxel size: %f x %f x %f mm \n", EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);    
        printf("Number of iterations for motion correction: %i \n",  NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION);
    } 
    
    // ------------------------------------------------
    
    // Allocate memory on the host
    
    if (!AllocateMemory(h_fMRI_Volumes, DATA_SIZE, allMemoryPointers, numberOfMemoryPointers))
    {
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputData);
        return -1;
    }
    
    if (!AllocateMemory(h_Motion_Corrected_fMRI_Volumes, DATA_SIZE, allMemoryPointers, numberOfMemoryPointers))
    {
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputData);
        return -1;
    }        
    
    if (!AllocateMemory(h_Quadrature_Filter_1_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers))
    {
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputData);
        return -1;
    }
    
    if (!AllocateMemory(h_Quadrature_Filter_1_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers))
    {
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputData);
        return -1;
    }
    
    if (!AllocateMemory(h_Quadrature_Filter_2_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers))
    {
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputData);
        return -1;
    }
    
    if (!AllocateMemory(h_Quadrature_Filter_2_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers))
    {
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputData);
        return -1;
    }
    
    if (!AllocateMemory(h_Quadrature_Filter_3_Real, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers))
    {
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputData);
        return -1;
    }
    
    if (!AllocateMemory(h_Quadrature_Filter_3_Imag, FILTER_SIZE, allMemoryPointers, numberOfMemoryPointers))
    {
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputData);
        return -1;
    }
    
    if (!AllocateMemory(h_Motion_Parameters, MOTION_PARAMETERS_SIZE, allMemoryPointers, numberOfMemoryPointers))
    {
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputData);
        return -1;
    }
    
    
    if (DEBUG)
    {    
        if (!AllocateMemory(h_Quadrature_Filter_Response_1_Real, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers))
        {
            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
            nifti_image_free(inputData);
            return -1;
        }
        if (!AllocateMemory(h_Quadrature_Filter_Response_1_Imag, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers))
        {
            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
            nifti_image_free(inputData);
            return -1;
        }
        if (!AllocateMemory(h_Quadrature_Filter_Response_2_Real, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers))
        {
            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
            nifti_image_free(inputData);
            return -1;
        }
        if (!AllocateMemory(h_Quadrature_Filter_Response_2_Imag, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers))
        {
            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
            nifti_image_free(inputData);
            return -1;
        }
        if (!AllocateMemory(h_Quadrature_Filter_Response_3_Real, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers))
        {
            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
            nifti_image_free(inputData);
            return -1;
        }
        if (!AllocateMemory(h_Quadrature_Filter_Response_3_Imag, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers))
        {
            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
            nifti_image_free(inputData);
            return -1;
        }
        if (!AllocateMemory(h_Phase_Differences, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers))
        {
            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
            nifti_image_free(inputData);
            return -1;
        }
        if (!AllocateMemory(h_Phase_Certainties, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers))
        {
            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
            nifti_image_free(inputData);
            return -1;
        }
        if (!AllocateMemory(h_Phase_Gradients, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers))
        {
            FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
            nifti_image_free(inputData);
            return -1;
        }               
    }
    
    // Convert data to floats
    if ( inputData->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputData->data;
    
        for (int i = 0; i < DATA_W * DATA_H * DATA_D * DATA_T; i++)
        {
            h_fMRI_Volumes[i] = (float)p[i];
        }
    }
    else
    {
        printf("Unknown data type in input data, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputData);
        return -1;
    }
    
    // Read quadrature filters, three real valued and three imaginary valued
    fp = fopen("filter1_real_parametric_registration.bin","rb");
    if (fp != NULL)
    {
        fread(h_Quadrature_Filter_1_Real,sizeof(float),MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,fp);
        fclose(fp);
    }
    else
    {
        printf("Could not open filter1_real_parametric_registration.bin \n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputData);
        return -1;
    }
    
    fp = fopen("filter1_imag_parametric_registration.bin","rb");
    if (fp != NULL)
    {
        fread(h_Quadrature_Filter_1_Imag,sizeof(float),MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,fp);        
        fclose(fp);
    }
    else
    {
        printf("Could not open filter1_imag_parametric_registration.bin \n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputData);
        return -1;
    }
    
    fp = fopen("filter2_real_parametric_registration.bin","rb");
    if (fp != NULL)
    {
        fread(h_Quadrature_Filter_2_Real,sizeof(float),MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,fp);
        fclose(fp);
    }
    else
    {
        printf("Could not open filter2_real_parametric_registration.bin \n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputData);
        return -1;
    }
    
    fp = fopen("filter2_imag_parametric_registration.bin","rb");
    if (fp != NULL)
    {
        fread(h_Quadrature_Filter_2_Imag,sizeof(float),MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,fp);
        fclose(fp);
    }
    else
    {
        printf("Could not open filter2_imag_parametric_registration.bin \n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputData);
        return -1;
    }
    
    fp = fopen("filter3_real_parametric_registration.bin","rb");
    if (fp != NULL)
    {
        fread(h_Quadrature_Filter_3_Real,sizeof(float),MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,fp);
        fclose(fp);
    }
    else
    {
        printf("Could not open filter3_real_parametric_registration.bin \n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputData);
        return -1;
    }
    
    fp = fopen("filter3_imag_parametric_registration.bin","rb");
    if (fp != NULL)
    {
        fread(h_Quadrature_Filter_3_Imag,sizeof(float),MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE*MOTION_CORRECTION_FILTER_SIZE,fp);
        fclose(fp);
    }
    else
    {
        printf("Could not open filter3_imag_parametric_registration.bin \n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        nifti_image_free(inputData);
        return -1;
    }
    
    //------------------------
    
    BROCCOLI_LIB BROCCOLI(OPENCL_PLATFORM,OPENCL_DEVICE);
    
    // Something went wrong...
    if (BROCCOLI.GetOpenCLInitiated() == 0)
    {              
        printf("Get platform IDs error is %d \n",BROCCOLI.GetOpenCLPlatformIDsError());
        printf("Get device IDs error is %d \n",BROCCOLI.GetOpenCLDeviceIDsError());
        printf("Create context error is %d \n",BROCCOLI.GetOpenCLCreateContextError());
        printf("Get create context info error is %d \n",BROCCOLI.GetOpenCLContextInfoError());
        printf("Create command queue error is %d \n",BROCCOLI.GetOpenCLCreateCommandQueueError());
        printf("Create program error is %d \n",BROCCOLI.GetOpenCLCreateProgramError());
        printf("Build program error is %d \n",BROCCOLI.GetOpenCLBuildProgramError());
        printf("Get program build info error is %d \n",BROCCOLI.GetOpenCLProgramBuildInfoError());
    
        // Print create kernel errors
        int* createKernelErrors = BROCCOLI.GetOpenCLCreateKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createKernelErrors[i] != 0)
            {
                printf("Create kernel error %i is %d \n",i,createKernelErrors[i]);
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
        nifti_image_free(inputData);
        return -1;
    }
    // Initialization OK
    else if (BROCCOLI.GetOpenCLInitiated() == 1)
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
        BROCCOLI.SetParametricImageRegistrationFilters(h_Quadrature_Filter_1_Real, h_Quadrature_Filter_1_Imag, h_Quadrature_Filter_2_Real, h_Quadrature_Filter_2_Imag, h_Quadrature_Filter_3_Real, h_Quadrature_Filter_3_Imag);
        BROCCOLI.SetNumberOfIterationsForMotionCorrection(NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION);
        
        BROCCOLI.SetOutputMotionCorrectedfMRIVolumes(h_Motion_Corrected_fMRI_Volumes);
        BROCCOLI.SetOutputMotionParameters(h_Motion_Parameters);
      
        if (DEBUG)
        {
            //BROCCOLI.SetOutputQuadratureFilterResponses(h_Quadrature_Filter_Response_1_Real, h_Quadrature_Filter_Response_1_Imag, h_Quadrature_Filter_Response_2_Real, h_Quadrature_Filter_Response_2_Imag, h_Quadrature_Filter_Response_3_Real, h_Quadrature_Filter_Response_3_Imag);
            BROCCOLI.SetOutputPhaseDifferences(h_Phase_Differences);
            BROCCOLI.SetOutputPhaseCertainties(h_Phase_Certainties);
            BROCCOLI.SetOutputPhaseGradients(h_Phase_Gradients);
        }
             
        // Run the actual motion correction
        BROCCOLI.PerformMotionCorrectionWrapper();        
    
        // Print create buffer errors
        int* createBufferErrors = BROCCOLI.GetOpenCLCreateBufferErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createBufferErrors[i] != 0)
            {
                printf("Create buffer error %i is %d \n",i,createBufferErrors[i]);
            }
        }
        
        // Print run kernel errors
        int* runKernelErrors = BROCCOLI.GetOpenCLRunKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (runKernelErrors[i] != 0)
            {
                printf("Run kernel error %i is %d \n",i,runKernelErrors[i]);
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
    
    // Free all memory
    FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);            
    nifti_image_free(inputData);    
    
    return 1;
}

/*
fid = fopen('filter1_imag_parametric_registration.bin','w')
a = f1_parametric_registration;
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

