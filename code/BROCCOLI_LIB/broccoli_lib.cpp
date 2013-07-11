/*
    BROCCOLI: An Open Source Multi-Platform Software for Parallel Analysis of fMRI Data on Many-Core CPUs and GPUs
    Copyright (C) <2013>  Anders Eklund, andek034@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/



#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
//#include <ifstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>

#include <opencl.h>

//#include <shrUtils.h>
//#include <shrQATest.h>
#include "broccoli_lib.h"

#include "nifti1.h"
#include "nifti1_io.h"

#include <cstdlib>




// public


// Constructor
BROCCOLI_LIB::BROCCOLI_LIB()
{
	OpenCLInitiate();
	SetStartValues();
	ResetAllPointers();
	//AllocateMemory();
	//ReadImageRegistrationFilters();
	//ReadSmoothingFilters();	
	
}

// Destructor
BROCCOLI_LIB::~BROCCOLI_LIB()
{
	// Free all the allocated memory
	
	for (int i = 0; i < NUMBER_OF_HOST_POINTERS; i++)
	{
		void* pointer = host_pointers[i];
		if (pointer != NULL)
		{
			free(pointer);
		}
	}

	for (int i = 0; i < NUMBER_OF_HOST_POINTERS; i++)
	{
		void* pointer = host_pointers_static[i];
		if (pointer != NULL)
		{
			free(pointer);
		}
	}

	for (int i = 0; i < NUMBER_OF_HOST_POINTERS; i++)
	{
		void* pointer = host_pointers_permutation[i];
		if (pointer != NULL)
		{
			free(pointer);
		}
	}

	for (int i = 0; i < NUMBER_OF_DEVICE_POINTERS; i++)
	{
		float* pointer = device_pointers[i];
		if (pointer != NULL)
		{
			//clReleaseMemObject();
		}
	}
	
	for (int i = 0; i < NUMBER_OF_DEVICE_POINTERS; i++)
	{
		float* pointer = device_pointers_permutation[i];
		if (pointer != NULL)
		{
			//clReleaseMemObject();
		}
	}

	OpenCLCleanup();
}

void BROCCOLI_LIB::SetStartValues()
{
	FILE_TYPE = RAW;
	DATA_TYPE = FLOAT;

	DATA_W = 64;
	DATA_H = 64;
	DATA_D = 22;
	DATA_T = 79;

	FMRI_VOXEL_SIZE_X = 3.75f;
	FMRI_VOXEL_SIZE_Y = 3.75f;
	FMRI_VOXEL_SIZE_Z = 3.75f;
	TR = 2.0f;
	
	NUMBER_OF_PERMUTATIONS = 1000;
	significance_threshold = 0.05f;

	filename_real_quadrature_filter_1 = "filters\\quadrature_filter_1_real.raw";
	filename_real_quadrature_filter_2 = "filters\\quadrature_filter_2_real.raw";
	filename_real_quadrature_filter_3 = "filters\\quadrature_filter_3_real.raw";
	filename_imag_quadrature_filter_1 = "filters\\quadrature_filter_1_imag.raw";
	filename_imag_quadrature_filter_2 = "filters\\quadrature_filter_2_imag.raw";
	filename_imag_quadrature_filter_3 = "filters\\quadrature_filter_3_imag.raw";

	filename_GLM_filter = "filters\\GLM_smoothing_filter";

	filename_fMRI_data_raw = "fMRI_data.raw";
	filename_slice_timing_corrected_fMRI_volumes_raw = "output\\slice_timing_corrected_fMRI_volumes.raw";
	filename_registration_parameters_raw = "output\\registration_parameters.raw";
	filename_motion_corrected_fMRI_volumes_raw = "output\\motion_compensated_fMRI_volumes.raw";
	filename_smoothed_fMRI_volumes_raw = "output\\smoothed_fMRI_volumes_1.raw";
	filename_detrended_fMRI_volumes_raw = "output\\detrended_fMRI_volumes_1.raw";
	filename_activity_volume_raw = "output\\activity_volume.raw";
	
	filename_fMRI_data_nifti = "fMRI_data.nii";
	filename_slice_timing_corrected_fMRI_volumes_nifti = "output\\slice_timing_corrected_fMRI_volumes.nii";
	filename_registration_parameters_nifti = "output\\registration_parameters.nii";
	filename_motion_corrected_fMRI_volumes_nifti = "output\\motion_compensated_fMRI_volumes.nii";
	filename_smoothed_fMRI_volumes_nifti = "output\\smoothed_fMRI_volumes_1.nii";
	filename_detrended_fMRI_volumes_nifti = "output\\detrended_fMRI_volumes_1.nii";
	filename_activity_volume_nifti = "output\\activity_volume.nii";
	
	
	THRESHOLD_ACTIVITY_MAP = false;
	ACTIVITY_THRESHOLD = 0.05f;

	MOTION_CORRECTED = false;
	IMAGE_REGISTRATION_FILTER_SIZE = 7;
	NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION = 3;
	NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS = 30;
	
	SMOOTHING_AMOUNT_MM = 8;
	SMOOTHING_FILTER_SIZE = 9;
	
	NUMBER_OF_DETRENDING_BASIS_FUNCTIONS = 4;

	SEGMENTATION_THRESHOLD = 600.0f;
	NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS = 2;
	NUMBER_OF_PERIODS = 4;
	PERIOD_TIME = 20;

	PRINT = VERBOSE;
	WRITE_DATA = NO;

	int DATA_SIZE_QUADRATURE_FILTER_REAL = sizeof(float) * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE;
	//int DATA_SIZE_QUADRATURE_FILTER_COMPLEX = sizeof(Complex) * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE;

	int DATA_SIZE_SMOOTHING_FILTER_GLM = sizeof(float) * SMOOTHING_FILTER_SIZE;
}

void BROCCOLI_LIB::ResetAllPointers()
{
	for (int i = 0; i < NUMBER_OF_HOST_POINTERS; i++)
	{	
		host_pointers[i] = NULL;
	}

	for (int i = 0; i < NUMBER_OF_HOST_POINTERS; i++)
	{	
		host_pointers_static[i] = NULL;
	}

	for (int i = 0; i < NUMBER_OF_HOST_POINTERS; i++)
	{	
		host_pointers_permutation[i] = NULL;
	}

	for (int i = 0; i < NUMBER_OF_DEVICE_POINTERS; i++)
	{
		device_pointers[i] = NULL;
	}

	for (int i = 0; i < NUMBER_OF_DEVICE_POINTERS; i++)
	{
		device_pointers_permutation[i] = NULL;
	}
}

void BROCCOLI_LIB::AllocateMemoryForFilters()
{
	h_Quadrature_Filter_1_Real = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL);
	h_Quadrature_Filter_1_Imag = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL);
	h_Quadrature_Filter_2_Real = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL);
	h_Quadrature_Filter_2_Imag = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL);
	h_Quadrature_Filter_3_Real = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL);
	h_Quadrature_Filter_3_Imag = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL); 		
	//h_Quadrature_Filter_1 = (Complex*)malloc(DATA_SIZE_QUADRATURE_FILTER_COMPLEX);
	//h_Quadrature_Filter_2 = (Complex*)malloc(DATA_SIZE_QUADRATURE_FILTER_COMPLEX);
	//h_Quadrature_Filter_3 = (Complex*)malloc(DATA_SIZE_QUADRATURE_FILTER_COMPLEX);

	//h_GLM_Filter = (float*)malloc(DATA_SIZE_SMOOTHING_FILTER_GLM);

	host_pointers_static[QF1R]   = (void*)h_Quadrature_Filter_1_Real;
	host_pointers_static[QF1I]   = (void*)h_Quadrature_Filter_1_Imag;
	host_pointers_static[QF2R]   = (void*)h_Quadrature_Filter_2_Real;
	host_pointers_static[QF2I]   = (void*)h_Quadrature_Filter_2_Imag;
	host_pointers_static[QF3R]   = (void*)h_Quadrature_Filter_3_Real;
	host_pointers_static[QF3I]   = (void*)h_Quadrature_Filter_3_Imag;
	//host_pointers_static[QF1]    = (void*)h_Quadrature_Filter_1;
	//host_pointers_static[QF2]    = (void*)h_Quadrature_Filter_2;
	//host_pointers_static[QF3]	 = (void*)h_Quadrature_Filter_3;
}	


// Add compilation from binary
// Add to select CPU or GPU

void BROCCOLI_LIB::OpenCLInitiate()
{
	std::string temp_string; std::ostringstream temp_stream;
	char* value;
	size_t valueSize, valueSizes[3];
	cl_uint maxComputeUnits, clockFrequency;
	cl_ulong memorySize;
	cl_int err;
	
	// Get platforms
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, NULL, &platformIdCount);
	std::vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs (platformIdCount, platformIds.data (), NULL);

	// Get devices
	cl_uint deviceIdCount = 0;
    clGetDeviceIDs (platformIds[0], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceIdCount);
	std::vector<cl_device_id> deviceIds (deviceIdCount);
	clGetDeviceIDs (platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), NULL);

	// Get information for for each device and save as a long string
    for (uint j = 0; j < deviceIdCount; j++) 
    {
        // Get vendor name
        clGetDeviceInfo(deviceIds[j], CL_DEVICE_VENDOR, 0, NULL, &valueSize);
        value = (char*) malloc(valueSize);
        clGetDeviceInfo(deviceIds[j], CL_DEVICE_VENDOR, valueSize, value, NULL);            
        device_info.append("Vendor name: ");
        device_info.append(value);
        device_info.append("\n");
        free(value);	
        	
        // Get device name
        clGetDeviceInfo(deviceIds[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
        value = (char*) malloc(valueSize);
        clGetDeviceInfo(deviceIds[j], CL_DEVICE_NAME, valueSize, value, NULL);            
        device_info.append("Device name: ");
        device_info.append(value);
        device_info.append("\n");
        free(value);

        // Get hardware device version
        clGetDeviceInfo(deviceIds[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
        value = (char*) malloc(valueSize);
        clGetDeviceInfo(deviceIds[j], CL_DEVICE_VERSION, valueSize, value, NULL);
        device_info.append("Hardware version: ");
        device_info.append(value);
        device_info.append("\n");
        free(value);

        // Get software driver version
        clGetDeviceInfo(deviceIds[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
        value = (char*) malloc(valueSize);
        clGetDeviceInfo(deviceIds[j], CL_DRIVER_VERSION, valueSize, value, NULL);
        device_info.append("Software version: ");
        device_info.append(value);
        device_info.append("\n");
        free(value);

        // Get C version supported by compiler for device
        clGetDeviceInfo(deviceIds[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
        value = (char*) malloc(valueSize);
        clGetDeviceInfo(deviceIds[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
        device_info.append("OpenCL C version: ");
        device_info.append(value);
        device_info.append("\n");
        free(value);
            
        // Get global memory size
        clGetDeviceInfo(deviceIds[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memorySize), &memorySize, NULL);            
        device_info.append("Global memory size in MB: ");
        temp_stream.str("");
	    temp_stream.clear();
        temp_stream << memorySize/ (1024*1024);            
        device_info.append(temp_stream.str());
        device_info.append("\n");
            
        // Get local (shared) memory size
        clGetDeviceInfo(deviceIds[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(memorySize), &memorySize, NULL);            
        device_info.append("Local memory size in KB: ");
        temp_stream.str("");
	    temp_stream.clear();
        temp_stream << memorySize/1024;            
        device_info.append(temp_stream.str());
        device_info.append("\n");
            
        // Get constant memory size
        clGetDeviceInfo(deviceIds[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(memorySize), &memorySize, NULL);            
        device_info.append("Constant memory size in KB: ");
        temp_stream.str("");
	    temp_stream.clear();
        temp_stream << memorySize/1024;            
        device_info.append(temp_stream.str());
        device_info.append("\n");                       
            
	    // Get parallel compute units
        clGetDeviceInfo(deviceIds[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);            
        device_info.append("Parallel compute units: ");
        temp_stream.str("");
	    temp_stream.clear();
        temp_stream << maxComputeUnits;            
        device_info.append(temp_stream.str());
        device_info.append("\n");
            
        // Get clock frequency
        clGetDeviceInfo(deviceIds[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockFrequency), &clockFrequency, NULL);            
        device_info.append("Clock frequency in MHz: ");
        temp_stream.str("");
	    temp_stream.clear();
        temp_stream << clockFrequency;            
        device_info.append(temp_stream.str());
        device_info.append("\n");                                  

		// Get maximum number of threads per block
        clGetDeviceInfo(deviceIds[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(valueSize), &valueSize, NULL);            
        device_info.append("Max number of threads per block: ");
        temp_stream.str("");
	    temp_stream.clear();
        temp_stream << valueSize;            
        device_info.append(temp_stream.str());
        device_info.append("\n");                                  
        
		// Get maximum block dimensions
        clGetDeviceInfo(deviceIds[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(valueSizes), valueSizes, NULL);            
        device_info.append("Max number of threads in each dimension: ");
        temp_stream.str("");
	    temp_stream.clear();
        temp_stream << valueSizes[0];
		temp_stream << " ";
		temp_stream << valueSizes[1];
		temp_stream << " ";
		temp_stream << valueSizes[2];
		device_info.append(temp_stream.str());
        device_info.append("\n");                                  
        				
		device_info.append("\n");		
    }
                

	// Create context
	const cl_context_properties contextProperties [] =
	{
	    CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[0]), 0, 0
	};

	context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), NULL, NULL, &err);	
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &valueSize);
	cl_device_id *clDevices = (cl_device_id *) malloc(valueSize);
	errNum |= clGetContextInfo(context, CL_CONTEXT_DEVICES, valueSize, clDevices, NULL);

	// Create a command queue
	commandQueue = clCreateCommandQueue(context, deviceIds[0], 0, &err);

	// Read the kernel code from file
	std::fstream kernelFile("broccoli_lib_kernel.cl",std::ios::in);
	std::ostringstream oss;
	oss << kernelFile.rdbuf();
	std::string src = oss.str();
	const char *srcstr = src.c_str();

	// Create a program and build the code
	program = clCreateProgramWithSource(context, 1, (const char**)&srcstr , NULL, &err);
	clBuildProgram(program, deviceIdCount, deviceIds.data(), NULL, NULL, NULL);

	// Get build info        
    clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &valueSize);        
    value = (char*)malloc(valueSize);
    clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, valueSize, value, NULL);
	build_info.append(value);
	free(value);

	// Create kernels
	SeparableConvolutionRowsKernel = clCreateKernel(program,"SeparableConvolutionRows",&kernel_error);	
	SeparableConvolutionColumnsKernel = clCreateKernel(program,"SeparableConvolutionColumns",&kernel_error);
	SeparableConvolutionRodsKernel = clCreateKernel(program,"SeparableConvolutionRods",&kernel_error);	
	//NonseparableConvolution3DComplexKernel = clCreateKernel(program,"convolutionNonSeparable3DComplex",&err);
	
	AddKernel = clCreateKernel(program,"Add",&err);
	
	// Kernels for statistical analysis
	//CalculateStatisticalMapsGLMKernel = clCreateKernel(program,"CalculateActivityMapGLM",&err);
	
	//clFinish(commandQueue);
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizes()
{


	// Number of threads per block
	
	localWorkSizeSeparableConvolutionRows[0] = 32;
	localWorkSizeSeparableConvolutionRows[1] = 8;
	localWorkSizeSeparableConvolutionRows[2] = 2;

	localWorkSizeSeparableConvolutionColumns[0] = 32;
	localWorkSizeSeparableConvolutionColumns[1] = 8;
	localWorkSizeSeparableConvolutionColumns[2] = 2;

	localWorkSizeSeparableConvolutionRods[0] = 32;
	localWorkSizeSeparableConvolutionRods[1] = 2;
	localWorkSizeSeparableConvolutionRods[2] = 8;

	localWorkSizeCalculateBetaValuesGLM[0] = 32;
	localWorkSizeCalculateBetaValuesGLM[1] = 8;
	localWorkSizeCalculateBetaValuesGLM[2] = 1;

	localWorkSizeCalculateStatisticalMapsGLM[0] = 32;
	localWorkSizeCalculateStatisticalMapsGLM[1] = 8;
	localWorkSizeCalculateStatisticalMapsGLM[2] = 1;
	
	// Calculate how many blocks are required
	// ConvolutionRows yields 32 * 8 * 8 valid filter responses per block (x,y,z)
	xBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_ROWS);
	yBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_ROWS);
	zBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_RODS);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	threadsX = xBlocks * localWorkSizeSeparableConvolutionRows[0];
	threadsY = yBlocks * localWorkSizeSeparableConvolutionRows[1];
	threadsZ = zBlocks * localWorkSizeSeparableConvolutionRows[2];
	
	globalWorkSizeSeparableConvolutionRows[0] = threadsX;
	globalWorkSizeSeparableConvolutionRows[1] = threadsY;
	globalWorkSizeSeparableConvolutionRows[2] = threadsZ;

    // Calculate how many blocks are required
	// ConvolutionColumns yields 24 * 16 * 8 valid filter responses per block (x,y,z)
	xBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_COLUMNS);
	yBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_COLUMNS);
	zBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_COLUMNS);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	threadsX = xBlocks * localWorkSizeSeparableConvolutionColumns[0];
	threadsY = yBlocks * localWorkSizeSeparableConvolutionColumns[1];
	threadsZ = zBlocks * localWorkSizeSeparableConvolutionColumns[2];

	globalWorkSizeSeparableConvolutionColumns[0] = threadsX;
	globalWorkSizeSeparableConvolutionColumns[1] = threadsY;
	globalWorkSizeSeparableConvolutionColumns[2] = threadsX;

	// Calculate how many blocks are required
	// ConvolutionRods yields 32 * 8 * 8 valid filter responses per block (x,y,z)
	xBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_RODS);
	yBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_RODS);
	zBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_RODS);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	threadsX = xBlocks * localWorkSizeSeparableConvolutionRods[0];
	threadsY = yBlocks * localWorkSizeSeparableConvolutionRods[1];
	threadsZ = zBlocks * localWorkSizeSeparableConvolutionRods[2];

	globalWorkSizeSeparableConvolutionRods[0] = threadsX;
	globalWorkSizeSeparableConvolutionRods[1] = threadsY;
	globalWorkSizeSeparableConvolutionRods[2] = threadsZ;
}

void BROCCOLI_LIB::OpenCLCleanup()
{
    clReleaseKernel(SeparableConvolutionRowsKernel);
    clReleaseProgram(program);    
    clReleaseCommandQueue(commandQueue);
    int err = clReleaseContext(context);
}







// Set functions for GUI / Wrappers

void BROCCOLI_LIB::SetInputData(float* data)
{
	h_fMRI_Volumes = data;
}

void BROCCOLI_LIB::SetOutputData(float* data)
{
	h_Result = data;
}

void BROCCOLI_LIB::SetSmoothingFilters(float* Smoothing_Filter_X, float* Smoothing_Filter_Y, float* Smoothing_Filter_Z)
{
	h_Smoothing_Filter_X = Smoothing_Filter_X;
	h_Smoothing_Filter_Y = Smoothing_Filter_Y;
	h_Smoothing_Filter_Z = Smoothing_Filter_Z;
}


void BROCCOLI_LIB::SetDataType(int type)
{
	DATA_TYPE = type;
}

void BROCCOLI_LIB::SetFileType(int type)
{
	FILE_TYPE = type;
}

void BROCCOLI_LIB::SetNumberOfIterationsForMotionCorrection(int N)
{
	NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION = N;
}

void BROCCOLI_LIB::SetfMRIDataSliceLocationX(int location)
{
	X_SLICE_LOCATION_fMRI_DATA = location;
}
			
void BROCCOLI_LIB::SetfMRIDataSliceLocationY(int location)
{
	Y_SLICE_LOCATION_fMRI_DATA = location;
}
		
void BROCCOLI_LIB::SetfMRIDataSliceLocationZ(int location)
{
	Z_SLICE_LOCATION_fMRI_DATA = location;
}

void BROCCOLI_LIB::SetfMRIDataSliceTimepoint(int timepoint)
{
	TIMEPOINT_fMRI_DATA = timepoint;
}

void BROCCOLI_LIB::SetActivityThreshold(float threshold)
{
	ACTIVITY_THRESHOLD = threshold;
}

void BROCCOLI_LIB::SetThresholdStatus(bool status)
{
	THRESHOLD_ACTIVITY_MAP = status;
}

void BROCCOLI_LIB::SetSmoothingAmount(int amount)
{
	SMOOTHING_AMOUNT_MM = amount;
	ReadSmoothingFilters();
}

void BROCCOLI_LIB::SetNumberOfBasisFunctionsDetrending(int N)
{
	NUMBER_OF_DETRENDING_BASIS_FUNCTIONS = N;
}

void BROCCOLI_LIB::SetfMRIDataFilename(std::string filename)
{
	filename_fMRI_data_nifti = filename;
}

void BROCCOLI_LIB::SetAnalysisMethod(int method)
{
	ANALYSIS_METHOD = method;
	ReadSmoothingFilters();
}

void BROCCOLI_LIB::SetWriteStatus(bool status)
{
	WRITE_DATA = status;
}

void BROCCOLI_LIB::SetShowPreprocessedType(int value)
{
	PREPROCESSED = value;
}

void BROCCOLI_LIB::SetWidth(int w)
{
	DATA_W = w;
}
			
void BROCCOLI_LIB::SetHeight(int h)
{
	DATA_H = h;
}

void BROCCOLI_LIB::SetDepth(int d)
{
	DATA_D = d;
}

void BROCCOLI_LIB::SetTimepoints(int t)
{
	DATA_T = t;
}

void BROCCOLI_LIB::SetfMRIVoxelSizeX(float value)
{
	FMRI_VOXEL_SIZE_X = value;
}

void BROCCOLI_LIB::SetfMRIVoxelSizeY(float value)
{
	FMRI_VOXEL_SIZE_Y = value;
}

void BROCCOLI_LIB::SetfMRIVoxelSizeZ(float value)
{
	FMRI_VOXEL_SIZE_Z = value;
}

void BROCCOLI_LIB::SetTR(float value)
{
	TR = value;
}

void BROCCOLI_LIB::SetSignificanceThreshold(float value)
{
	significance_threshold = value;
}

void BROCCOLI_LIB::SetNumberOfPermutations(int value)
{
	NUMBER_OF_PERMUTATIONS = value;
}









// Get functions for GUI / Wrappers

const char* BROCCOLI_LIB::GetDeviceInfoChar()
{
	return device_info.c_str();
}

const char* BROCCOLI_LIB::GetBuildInfoChar()
{
	return build_info.c_str();
}

std::string BROCCOLI_LIB::GetDeviceInfoString()
{
	return device_info;
}

std::string BROCCOLI_LIB::GetBuildInfoString()
{
	return build_info;
}



int BROCCOLI_LIB::GetfMRIDataSliceLocationX()
{
	return X_SLICE_LOCATION_fMRI_DATA;
}
			
int BROCCOLI_LIB::GetfMRIDataSliceLocationY()
{
	return Y_SLICE_LOCATION_fMRI_DATA;
}
		
int BROCCOLI_LIB::GetfMRIDataSliceLocationZ()
{
	return Z_SLICE_LOCATION_fMRI_DATA;
}

// Returns the processing time for slice timing correction			
double BROCCOLI_LIB::GetProcessingTimeSliceTimingCorrection()
{
	return processing_times[SLICE_TIMING_CORRECTION];
}

// Returns the processing time for motion correction			
double BROCCOLI_LIB::GetProcessingTimeMotionCorrection()
{
	return processing_times[MOTION_CORRECTION];
}

// Returns the processing time for smoothing	
double BROCCOLI_LIB::GetProcessingTimeSmoothing()
{
	return processing_times[SMOOTHING];
}

// Returns the processing time for detrending
double BROCCOLI_LIB::GetProcessingTimeDetrending()
{
	return processing_times[DETRENDING];
}

// Returns the processing time for the statistical analysis
double BROCCOLI_LIB::GetProcessingTimeStatisticalAnalysis()
{
	return processing_times[STATISTICAL_ANALYSIS];
}

// Returns the processing time for the permutation test
double BROCCOLI_LIB::GetProcessingTimePermutationTest()
{
	return processing_times[PERMUTATION_TEST];
}

// Returns the processing time for copying of data
double BROCCOLI_LIB::GetProcessingTimeCopy()
{
	return processing_times[COPY];
}

// Returns the processing time for convolution in the motion correction	step
double BROCCOLI_LIB::GetProcessingTimeConvolution()
{
	return processing_times[CONVOLVE];
}

// Returns the processing time for calculation of phase differences in the motion correction step
double BROCCOLI_LIB::GetProcessingTimePhaseDifferences()
{
	return processing_times[PHASEDC];
}

// Returns the processing time for calculation of phase gradients in the motion correction step
double BROCCOLI_LIB::GetProcessingTimePhaseGradients()
{
	return processing_times[PHASEG];
}

// Returns the processing time for calculation of A-matrix and h-vector in the motion correction step
double BROCCOLI_LIB::GetProcessingTimeAH()
{
	return processing_times[AH2D];
}

// Returns the processing time for solving the equation system in the motion correction step
double BROCCOLI_LIB::GetProcessingTimeEquationSystem()
{
	return processing_times[EQSYSTEM];
}

// Returns the processing time for the interpolation step in the motion correction step
double BROCCOLI_LIB::GetProcessingTimeInterpolation()
{
	return processing_times[INTERPOLATION];
}

// Returns the width dimension (x) of the current fMRI dataset
int BROCCOLI_LIB::GetWidth()
{
	return DATA_W;
}

// Returns the height dimension (y) of the current fMRI dataset
int BROCCOLI_LIB::GetHeight()
{
	return DATA_H;
}

// Returns the depth dimension (z) of the current fMRI dataset
int BROCCOLI_LIB::GetDepth()
{
	return DATA_D;
}

// Returns the number of timepoints of the current fMRI dataset
int BROCCOLI_LIB::GetTimepoints()
{
	return DATA_T;
}

// Returns the voxel size (in mm) in the x direction
float BROCCOLI_LIB::GetfMRIVoxelSizeX()
{
	return FMRI_VOXEL_SIZE_X;
}

// Returns the voxel size (in mm) in the y direction
float BROCCOLI_LIB::GetfMRIVoxelSizeY()
{
	return FMRI_VOXEL_SIZE_Y;
}

// Returns the voxel size (in mm) in the z direction
float BROCCOLI_LIB::GetfMRIVoxelSizeZ()
{
	return FMRI_VOXEL_SIZE_Z;
}

// Returns the repetition time of the current fMRI dataset
float BROCCOLI_LIB::GetTR()
{
	return TR;
}

// Returns a z slice of the original fMRI data, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetZSlicefMRIData()
{
	return z_slice_fMRI_data;
}

// Returns a y slice of the original fMRI data, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetYSlicefMRIData()
{
	return y_slice_fMRI_data;
}

// Returns a x slice of the original fMRI data, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetXSlicefMRIData()
{
	return x_slice_fMRI_data;
}

// Returns a z slice of the preprocessed fMRI data, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetZSlicePreprocessedfMRIData()
{
	return z_slice_preprocessed_fMRI_data;
}

// Returns a y slice of the preprocessed fMRI data, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetYSlicePreprocessedfMRIData()
{
	return y_slice_preprocessed_fMRI_data;
}

// Returns a x slice of the preprocessed fMRI data, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetXSlicePreprocessedfMRIData()
{
	return x_slice_preprocessed_fMRI_data;
}

// Returns a z slice of the activity map, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetZSliceActivityData()
{
	return z_slice_activity_data;
}

// Returns a y slice of the activity map, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetYSliceActivityData()
{
	return y_slice_activity_data;
}

// Returns a x slice of the activity map, to be viewed in the GUI
unsigned char* BROCCOLI_LIB::GetXSliceActivityData()
{
	return x_slice_activity_data;
}

// Returns estimated motion parameters in the x direction to be viewed in the GUI
double* BROCCOLI_LIB::GetMotionParametersX()
{
	return motion_parameters_x;
}

// Returns estimated motion parameters in the y direction to be viewed in the GUI
double* BROCCOLI_LIB::GetMotionParametersY()
{
	return motion_parameters_y;
}

// Returns estimated motion parameters in the z direction to be viewed in the GUI
double* BROCCOLI_LIB::GetMotionParametersZ()
{
	return motion_parameters_z;
}

double* BROCCOLI_LIB::GetPlotValuesX()
{
	return plot_values_x;
}

// Returns the timeseries of the motion corrected data for the current mouse location in the GUI
double* BROCCOLI_LIB::GetMotionCorrectedCurve()
{
	for (int t = 0; t < DATA_T; t++)
	{
		motion_corrected_curve[t] = (double)h_Motion_Corrected_fMRI_Volumes[X_SLICE_LOCATION_fMRI_DATA + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D];
	}

	return motion_corrected_curve;
}

// Returns the timeseries of the smoothed data for the current mouse location in the GUI
double* BROCCOLI_LIB::GetSmoothedCurve()
{
	for (int t = 0; t < DATA_T; t++)
	{
		smoothed_curve[t] = (double)h_Smoothed_fMRI_Volumes[X_SLICE_LOCATION_fMRI_DATA + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D];
	}

	return smoothed_curve;
}

// Returns the timeseries of the detrended data for the current mouse location in the GUI
double* BROCCOLI_LIB::GetDetrendedCurve()
{
	for (int t = 0; t < DATA_T; t++)
	{
		detrended_curve[t] = (double)h_Detrended_fMRI_Volumes[X_SLICE_LOCATION_fMRI_DATA + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D];
	}

	return detrended_curve;
}

// Returns the filename of the current fMRI dataset
std::string BROCCOLI_LIB::GetfMRIDataFilename()
{
	return filename_fMRI_data_nifti;
}

// Returns the significance threshold calculated with a permutation test
float BROCCOLI_LIB::GetPermutationThreshold()
{
	return permutation_test_threshold;
}

// Returns the number of voxels that pass the significance threshold
int BROCCOLI_LIB::GetNumberOfSignificantlyActiveVoxels()
{
	return NUMBER_OF_SIGNIFICANTLY_ACTIVE_VOXELS;
}

// Returns the number of clusters that pass the significance threshold
int BROCCOLI_LIB::GetNumberOfSignificantlyActiveClusters()
{
	return NUMBER_OF_SIGNIFICANTLY_ACTIVE_CLUSTERS;
}

// Returns a string containing info about the device(s) used for computations
std::string BROCCOLI_LIB::PrintDeviceInfo()
{
	std::string s;
	return s;
}













// Processing



// Preprocessing

/*
void BROCCOLI_LIB::AlignTwoVolumesSeveralScales(float *h_Registration_Parameters, cl_mem d_Reference_Volume, cl_mem d_Aligned_Volume, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_SCALES)
{
	// Loop registration over scales
	for (int s = NUMBER_OF_SCALES-1; s >= 0; s--)
	{
		CURRENT_DATA_W = DATA_W/(2*s);
		CURRENT_DATA_H = DATA_H/(2*s);
		CURRENT_DATA_D = DATA_D/(2*s);

		ChangeVolumeResolutionAndSize(d_Aligned_Volume, d_Interpolated_Aligned_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, VOXEL_SIZE_X, VOXEL_SIZE_Y, VOXEL_SIZE_Z, CURRENT_VOXEL_SIZE_X, CURRENT_VOXEL_SIZE_Y, CURRENT_VOXEL_SIZE_Z);       
		ChangeVolumeResolutionAndSize(d_Reference_Volume, d_Interpolated_Reference_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, VOXEL_SIZE_X, VOXEL_SIZE_Y, VOXEL_SIZE_Z, CURRENT_VOXEL_SIZE_X, CURRENT_VOXEL_SIZE_Y, CURRENT_VOXEL_SIZE_Z);       

		// Setup all parameters and allocate memory on host
		AlignTwoVolumesSetup(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);

		// Set the  volume as the reference volume
		clEnqueueCopyBuffer(commandQueue, d_T1_Volume, d_Reference_Volume, 0, 0, DATA_SIZE_T1_VOLUME, 0, NULL, NULL);

		// Set the interpolated volume as the volume to be aligned
		clEnqueueCopyBuffer(commandQueue, d_Interpolated_fMRI_Volume, d_Aligned_Volume, 0, 0, DATA_SIZE_T1_VOLUME, 0, NULL, NULL);

		// Set the interpolated fMRI volume as the original volume to interpolate from
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {T1_DATA_W, T1_DATA_H, T1_DATA_D};
		clEnqueueCopyBufferToImage(commandQueue, d_Interpolated_fMRI_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);

		AlignTwoVolumes(h_Registration_Parameters);

		AlignTwoVolumesCleanup();

		// Multiply the transformations by a factor 2 for the next scale

	}
}
*/


// This function is the foundation for all the image registration functions
void BROCCOLI_LIB::AlignTwoVolumes(float *h_Registration_Parameters)
{
	// Calculate the filter responses for the reference volume (only needed once)
	clEnqueueNDRangeKernel(commandQueue, NonseparableConvolution3DComplexKernel, 3, NULL, globalWorkSizeNonseparableConvolution3DComplex, localWorkSizeNonseparableConvolution3DComplex, 0, NULL, NULL);

	// Set kernel arguments for following convolutions
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 0, sizeof(cl_mem), &d_q21);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 1, sizeof(cl_mem), &d_q22);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 2, sizeof(cl_mem), &d_q23);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 3, sizeof(cl_mem), &d_Aligned_Volume);

	// Reset the parameter vector
	for (int p = 0; p < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; p++)
	{
		h_Registration_Parameters[p] = 0;
	}

	// Run the registration algorithm for a number of iterations
	for (int it = 0; it < NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION; it++)
	{
		// Apply convolution with 3 quadrature filters
		clEnqueueNDRangeKernel(commandQueue, NonseparableConvolution3DComplexKernel, 3, NULL, globalWorkSizeNonseparableConvolution3DComplex, localWorkSizeNonseparableConvolution3DComplex, 0, NULL, NULL);
		clFinish(commandQueue);

		// Calculate phase differences, certainties and phase gradients in the X direction
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 2, sizeof(cl_mem), &d_q11);
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 3, sizeof(cl_mem), &d_q21);
		clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesAndCertaintiesKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);
		clFinish(commandQueue);
			
		clEnqueueNDRangeKernel(commandQueue, CalculatePhaseGradientsXKernel, 3, NULL, globalWorkSizeCalculatePhaseGradients, localWorkSizeCalculatePhaseGradients, 0, NULL, NULL);
		clFinish(commandQueue);

		// Calculate values for the A-matrix and h-vector in the X direction
		clEnqueueNDRangeKernel(commandQueue, CalculateAMatrixAndHVector2DValuesXKernel, 1, NULL, globalWorkSizeCalculateAMatrixAndHVector2DValuesX, localWorkSizeCalculateAMatrixAndHVector2DValuesX, 0, NULL, NULL);
		clFinish(commandQueue);

		// Calculate phase differences, certainties and phase gradients in the Y direction
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 2, sizeof(cl_mem), &d_q12);
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 3, sizeof(cl_mem), &d_q22);
		clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesAndCertaintiesKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);
		clFinish(commandQueue);
			
		clEnqueueNDRangeKernel(commandQueue, CalculatePhaseGradientsYKernel, 3, NULL, globalWorkSizeCalculatePhaseGradients, localWorkSizeCalculatePhaseGradients, 0, NULL, NULL);
		clFinish(commandQueue);

		// Calculate values for the A-matrix and h-vector in the Y direction
		clEnqueueNDRangeKernel(commandQueue, CalculateAMatrixAndHVector2DValuesYKernel, 1, NULL, globalWorkSizeCalculateAMatrixAndHVector2DValuesY, localWorkSizeCalculateAMatrixAndHVector2DValuesY, 0, NULL, NULL);
		clFinish(commandQueue);

		// Calculate phase differences, certainties and phase gradients in the Z direction
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 2, sizeof(cl_mem), &d_q13);
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 3, sizeof(cl_mem), &d_q23);			
		clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesAndCertaintiesKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);
		clFinish(commandQueue);
			
		clEnqueueNDRangeKernel(commandQueue, CalculatePhaseGradientsZKernel, 3, NULL, globalWorkSizeCalculatePhaseGradients, localWorkSizeCalculatePhaseGradients, 0, NULL, NULL);
		clFinish(commandQueue);
			
		// Calculate values for the A-matrix and h-vector in the Z direction
		clEnqueueNDRangeKernel(commandQueue, CalculateAMatrixAndHVector2DValuesZKernel, 1, NULL, globalWorkSizeCalculateAMatrixAndHVector2DValuesZ, localWorkSizeCalculateAMatrixAndHVector2DValuesZ, 0, NULL, NULL);
		clFinish(commandQueue);
			
   		// Setup final equation system

		// Sum in one direction to get 1D values
		clEnqueueNDRangeKernel(commandQueue, CalculateAMatrix1DValuesKernel, 1, NULL, globalWorkSizeCalculateAMatrix1DValues, localWorkSizeCalculateAMatrix1DValues, 0, NULL, NULL);
		clFinish(commandQueue);
			
		clEnqueueNDRangeKernel(commandQueue, CalculateHVector1DValuesKernel, 1, NULL, globalWorkSizeCalculateHVector1DValues, localWorkSizeCalculateHVector1DValues, 0, NULL, NULL);
		clFinish(commandQueue);
			
		clEnqueueNDRangeKernel(commandQueue, ResetAMatrixKernel, 1, NULL, globalWorkSizeResetAMatrix, localWorkSizeResetAMatrix, 0, NULL, NULL);
		clFinish(commandQueue);
			
		// Calculate final A-matrix
		clEnqueueNDRangeKernel(commandQueue, CalculateAMatrixKernel, 1, NULL, globalWorkSizeCalculateAMatrix, localWorkSizeCalculateAMatrix, 0, NULL, NULL);
		clFinish(commandQueue);

		// Calculate final h-vector
		clEnqueueNDRangeKernel(commandQueue, CalculateHVectorKernel, 1, NULL, globalWorkSizeCalculateHVector, localWorkSizeCalculateHVector, 0, NULL, NULL);
		clFinish(commandQueue);

		// Copy A-matrix and h-vector from device to host
		clEnqueueReadBuffer(commandQueue, d_A_Matrix, CL_TRUE, 0, sizeof(float) * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS, h_A_Matrix, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_h_Vector, CL_TRUE, 0, sizeof(float) * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS, h_h_Vector, 0, NULL, NULL);

		// Mirror the matrix values
		for (int j = 0; j < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; j++)
		{
			for (int i = 0; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; i++)
			{
				h_A_Matrix[j + i*NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS] = h_A_Matrix[i + j*NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS];
			}
		}

		// Solve the equation system A * p = h to obtain the parameter vector
		SolveEquationSystem(h_A_Matrix, h_Inverse_A_Matrix, h_h_Vector, h_Parameter_Vector, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS);

		// Update the total parameter vector
		h_Registration_Parameters[0]  += h_Parameter_Vector[0];
		h_Registration_Parameters[1]  += h_Parameter_Vector[1];
		h_Registration_Parameters[2]  += h_Parameter_Vector[2];
		h_Registration_Parameters[3]  += h_Parameter_Vector[3];
		h_Registration_Parameters[4]  += h_Parameter_Vector[4];
		h_Registration_Parameters[5]  += h_Parameter_Vector[5];
		h_Registration_Parameters[6]  += h_Parameter_Vector[6];
		h_Registration_Parameters[7]  += h_Parameter_Vector[7];
		h_Registration_Parameters[8]  += h_Parameter_Vector[8];
		h_Registration_Parameters[9]  += h_Parameter_Vector[9];
		h_Registration_Parameters[10] += h_Parameter_Vector[10];
		h_Registration_Parameters[11] += h_Parameter_Vector[11];

		// Copy parameter vector to constant memory
		clEnqueueWriteBuffer(commandQueue, c_Parameter_Vector, CL_TRUE, 0, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), h_Registration_Parameters, 0, NULL, NULL);

		// Interpolate to get the new volume
		clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeTrilinearKernel, 3, NULL, globalWorkSizeInterpolateVolumeTrilinear, localWorkSizeInterpolateVolumeTrilinear, 0, NULL, NULL);
		clFinish(commandQueue);
	}
}

// This function is used by all registration functions, to cleanup
void BROCCOLI_LIB::AlignTwoVolumesCleanup()
{
	// Free all the allocated memory on the device

	clReleaseMemObject(d_Reference_Volume);
	clReleaseMemObject(d_Aligned_Volume);
	clReleaseMemObject(d_Original_Volume);

	clReleaseMemObject(d_q11);
	clReleaseMemObject(d_q12);
	clReleaseMemObject(d_q13);
	clReleaseMemObject(d_q21);
	clReleaseMemObject(d_q22);
	clReleaseMemObject(d_q23);

	clReleaseMemObject(d_Phase_Differences);
	clReleaseMemObject(d_Phase_Gradients);
	clReleaseMemObject(d_Phase_Certainties);

	clReleaseMemObject(d_A_Matrix);
	clReleaseMemObject(d_h_Vector);

	clReleaseMemObject(d_A_Matrix_2D_Values);
	clReleaseMemObject(d_A_Matrix_1D_Values);

	clReleaseMemObject(d_h_Vector_2D_Values);
	clReleaseMemObject(d_h_Vector_1D_Values);

	clReleaseMemObject(c_Quadrature_Filter_1);
	clReleaseMemObject(c_Quadrature_Filter_2);
	clReleaseMemObject(c_Quadrature_Filter_3);
	clReleaseMemObject(c_Parameter_Vector);

	// Free all host allocated memory

	free(h_A_Matrix);
	free(h_Inverse_A_Matrix);
	free(h_h_Vector);
}

// This function is used by all registration functions, to setup necessary parameters
void BROCCOLI_LIB::AlignTwoVolumesSetup(int DATA_W, int DATA_H, int DATA_D)
{
	DATA_SIZE_VOLUME = sizeof(float) * DATA_W * DATA_H * DATA_D;
	//DATA_SIZE_COMPLEX_VOLUME = sizeof(float2) * DATA_W * DATA_H * DATA_D;

	// Allocate memory on the host
	h_A_Matrix = (float *)malloc(sizeof(float) * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS);
	h_Inverse_A_Matrix = (float *)malloc(sizeof(float) * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS);
	h_h_Vector = (float *)malloc(sizeof(float) * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS);

	// Create a 3D image (texture) for fast interpolation
	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_INTENSITY;
	d_Original_Volume = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, DATA_W, DATA_H, DATA_D, 0, 0, NULL, NULL);
	
	// Allocate memory on the device
	d_Aligned_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_SIZE_VOLUME, NULL, NULL);
	d_Reference_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_SIZE_VOLUME, NULL, NULL);

	d_q11 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_SIZE_COMPLEX_VOLUME, NULL, NULL);
	d_q12 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_SIZE_COMPLEX_VOLUME, NULL, NULL);
	d_q13 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_SIZE_COMPLEX_VOLUME, NULL, NULL);
	d_q21 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_SIZE_COMPLEX_VOLUME, NULL, NULL);
	d_q22 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_SIZE_COMPLEX_VOLUME, NULL, NULL);
	d_q23 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_SIZE_COMPLEX_VOLUME, NULL, NULL);

	d_Phase_Differences = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_SIZE_VOLUME, NULL, NULL);
	d_Phase_Gradients = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_SIZE_VOLUME, NULL, NULL);
	d_Phase_Certainties = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_SIZE_VOLUME, NULL, NULL);

	d_A_Matrix = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS, NULL, NULL);
	d_h_Vector = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS, NULL, NULL);

	d_A_Matrix_2D_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_H * DATA_D * NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS, NULL, NULL);
	d_A_Matrix_1D_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_D * NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS, NULL, NULL);
	
	d_h_Vector_2D_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_H * DATA_D * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS, NULL, NULL);
	d_h_Vector_1D_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_D * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS, NULL, NULL);

	// Allocate constant memory
	//c_Quadrature_Filter_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float2) * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE, NULL, NULL);
	//c_Quadrature_Filter_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float2) * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE, NULL, NULL);
	//c_Quadrature_Filter_3 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float2) * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE, NULL, NULL);
	c_Parameter_Vector = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS, NULL, NULL);

	// Set all kernel arguments

	clSetKernelArg(NonseparableConvolution3DComplexKernel, 0, sizeof(cl_mem), &d_q11);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 1, sizeof(cl_mem), &d_q12);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 2, sizeof(cl_mem), &d_q13);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 3, sizeof(cl_mem), &d_Reference_Volume);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 4, sizeof(cl_mem), &c_Quadrature_Filter_1);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 5, sizeof(cl_mem), &c_Quadrature_Filter_2);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 6, sizeof(cl_mem), &c_Quadrature_Filter_3);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 7, sizeof(int), &DATA_W);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 8, sizeof(int), &DATA_H);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 9, sizeof(int), &DATA_D);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 10, sizeof(int), &xBlockDifference);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 11, sizeof(int), &yBlockDifference);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 12, sizeof(int), &zBlockDifference);

	clSetKernelArg(CalculatePhaseGradientsXKernel, 0, sizeof(cl_mem), &d_Phase_Gradients);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 1, sizeof(cl_mem), &d_q11);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 2, sizeof(cl_mem), &d_q21);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 5, sizeof(int), &DATA_D);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 6, sizeof(int), &IMAGE_REGISTRATION_FILTER_SIZE);

	clSetKernelArg(CalculatePhaseGradientsYKernel, 0, sizeof(cl_mem), &d_Phase_Gradients);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 1, sizeof(cl_mem), &d_q12);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 2, sizeof(cl_mem), &d_q22);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 5, sizeof(int), &DATA_D);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 6, sizeof(int), &IMAGE_REGISTRATION_FILTER_SIZE);

	clSetKernelArg(CalculatePhaseGradientsZKernel, 0, sizeof(cl_mem), &d_Phase_Gradients);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 1, sizeof(cl_mem), &d_q13);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 2, sizeof(cl_mem), &d_q23);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 5, sizeof(int), &DATA_D);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 6, sizeof(int), &IMAGE_REGISTRATION_FILTER_SIZE);

	clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 0, sizeof(cl_mem), &d_Phase_Differences);
	clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 1, sizeof(cl_mem), &d_Phase_Certainties);
	clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 4, sizeof(int), &DATA_W);
	clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 5, sizeof(int), &DATA_H);
	clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 6, sizeof(int), &DATA_D);
	clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 7, sizeof(int), &IMAGE_REGISTRATION_FILTER_SIZE);

	clSetKernelArg(CalculateAMatrixAndHVector2DValuesXKernel, 0, sizeof(cl_mem), &d_A_Matrix_2D_Values);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesXKernel, 1, sizeof(cl_mem), &d_h_Vector_2D_Values);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesXKernel, 2, sizeof(cl_mem), &d_Phase_Differences);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesXKernel, 3, sizeof(cl_mem), &d_Phase_Gradients);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesXKernel, 4, sizeof(cl_mem), &d_Phase_Certainties);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesXKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesXKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesXKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesXKernel, 8, sizeof(int), &IMAGE_REGISTRATION_FILTER_SIZE);

	clSetKernelArg(CalculateAMatrixAndHVector2DValuesYKernel, 0, sizeof(cl_mem), &d_A_Matrix_2D_Values);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesYKernel, 1, sizeof(cl_mem), &d_h_Vector_2D_Values);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesYKernel, 2, sizeof(cl_mem), &d_Phase_Differences);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesYKernel, 3, sizeof(cl_mem), &d_Phase_Gradients);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesYKernel, 4, sizeof(cl_mem), &d_Phase_Certainties);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesYKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesYKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesYKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesYKernel, 8, sizeof(int), &IMAGE_REGISTRATION_FILTER_SIZE);

	clSetKernelArg(CalculateAMatrixAndHVector2DValuesZKernel, 0, sizeof(cl_mem), &d_A_Matrix_2D_Values);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesZKernel, 1, sizeof(cl_mem), &d_h_Vector_2D_Values);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesZKernel, 2, sizeof(cl_mem), &d_Phase_Differences);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesZKernel, 3, sizeof(cl_mem), &d_Phase_Gradients);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesZKernel, 4, sizeof(cl_mem), &d_Phase_Certainties);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesZKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesZKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesZKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(CalculateAMatrixAndHVector2DValuesZKernel, 8, sizeof(int), &IMAGE_REGISTRATION_FILTER_SIZE);

	clSetKernelArg(CalculateAMatrix1DValuesKernel, 0, sizeof(cl_mem), &d_A_Matrix_1D_Values);
	clSetKernelArg(CalculateAMatrix1DValuesKernel, 1, sizeof(cl_mem), &d_A_Matrix_2D_Values);
	clSetKernelArg(CalculateAMatrix1DValuesKernel, 2, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateAMatrix1DValuesKernel, 3, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateAMatrix1DValuesKernel, 4, sizeof(int), &DATA_D);
	clSetKernelArg(CalculateAMatrix1DValuesKernel, 5, sizeof(int), &IMAGE_REGISTRATION_FILTER_SIZE);

	clSetKernelArg(CalculateHVector1DValuesKernel, 0, sizeof(cl_mem), &d_h_Vector_1D_Values);
	clSetKernelArg(CalculateHVector1DValuesKernel, 1, sizeof(cl_mem), &d_h_Vector_2D_Values);
	clSetKernelArg(CalculateHVector1DValuesKernel, 2, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateHVector1DValuesKernel, 3, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateHVector1DValuesKernel, 4, sizeof(int), &DATA_D);
	clSetKernelArg(CalculateHVector1DValuesKernel, 5, sizeof(int), &IMAGE_REGISTRATION_FILTER_SIZE);

	clSetKernelArg(ResetAMatrixKernel, 0, sizeof(cl_mem), &d_A_Matrix);
	
	clSetKernelArg(CalculateAMatrixKernel, 0, sizeof(cl_mem), &d_A_Matrix);
	clSetKernelArg(CalculateAMatrixKernel, 1, sizeof(cl_mem), &d_A_Matrix_1D_Values);
	clSetKernelArg(CalculateAMatrixKernel, 2, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateAMatrixKernel, 3, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateAMatrixKernel, 4, sizeof(int), &DATA_D);
	clSetKernelArg(CalculateAMatrixKernel, 5, sizeof(int), &IMAGE_REGISTRATION_FILTER_SIZE);
		
	clSetKernelArg(CalculateHVectorKernel, 0, sizeof(cl_mem), &d_h_Vector);
	clSetKernelArg(CalculateHVectorKernel, 1, sizeof(cl_mem), &d_h_Vector_1D_Values);
	clSetKernelArg(CalculateHVectorKernel, 2, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateHVectorKernel, 3, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateHVectorKernel, 4, sizeof(int), &DATA_D);
	clSetKernelArg(CalculateHVectorKernel, 5, sizeof(int), &IMAGE_REGISTRATION_FILTER_SIZE);

	clSetKernelArg(InterpolateVolumeTrilinearKernel, 0, sizeof(cl_mem), &d_Aligned_Volume);
	clSetKernelArg(InterpolateVolumeTrilinearKernel, 1, sizeof(cl_mem), &d_Original_Volume);
	clSetKernelArg(InterpolateVolumeTrilinearKernel, 2, sizeof(cl_mem), &c_Parameter_Vector);
	clSetKernelArg(InterpolateVolumeTrilinearKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(InterpolateVolumeTrilinearKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(InterpolateVolumeTrilinearKernel, 5, sizeof(int), &DATA_D);
}


void BROCCOLI_LIB::ChangeVolumeResolutionAndSize(cl_mem d_Original_Volume, cl_mem d_Interpolated_Volume, int ORIGINAL_DATA_W, int ORIGINAL_DATA_H, int ORIGINAL_DATA_D, int INTERPOLATED__DATA_W, int INTERPOLATED__DATA_H, int INTERPOLATED__DATA_D, float ORIGINAL_VOXEL_SIZE_X, float ORIGINAL_VOXEL_SIZE_Y, float ORIGINAL_VOXEL_SIZE_Z, float INTERPOLATED_VOXEL_SIZE_X, float INTERPOLATED_VOXEL_SIZE_Y, float INTERPOLATED_VOXEL_SIZE_Z)
{

}


// Performs registration between one low resolution fMRI volume and a high resolution T1 volume
void BROCCOLI_LIB::PerformRegistrationEPIT1(int t)
{
	// Interpolate FMRI volume to T1 resolution (use volume at timepoint t)
	ChangeVolumeResolutionAndSize(d_fMRI_Volumes, d_Interpolated_fMRI_Volume, FMRI_DATA_W, FMRI_DATA_H, FMRI_DATA_D, T1_DATA_W, T1_DATA_H, T1_DATA_D, FMRI_VOXEL_SIZE_X, FMRI_VOXEL_SIZE_Y, FMRI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z);       

		
	// Do the registration with several scales
	//AlignTwoVolumesSeveralScales(h_Registration_Parameters_EPI_T1,T1_DATA_W, T1_DATA_H, T1_DATA_D, 1);

	// Copy the aligned volume to host
	clEnqueueReadBuffer(commandQueue, d_Aligned_Volume, CL_TRUE, 0, DATA_SIZE_T1_VOLUME, h_Aligned_fMRI_Volume, 0, NULL, NULL);

	// Cleanup allocated memory
	AlignTwoVolumesCleanup();	
}

// Performs registration between one high resolution T1 volume and a high resolution MNI volume (brain template)
void BROCCOLI_LIB::PerformRegistrationT1MNI()
{
	// Interpolate T1 volume to MNI resolution
	ChangeVolumeResolutionAndSize(d_T1_Volume, d_Interpolated_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z);       

	// Setup all parameters and allocate memory on host
	AlignTwoVolumesSetup(MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	// Set the MNI volume as the reference volume
	clEnqueueCopyBuffer(commandQueue, d_MNI_Volume, d_Reference_Volume, 0, 0, DATA_SIZE_MNI_VOLUME, 0, NULL, NULL);

	// Set the T1 volume as the volume to be aligned
	clEnqueueCopyBuffer(commandQueue, d_Interpolated_T1_Volume, d_Aligned_Volume, 0, 0, DATA_SIZE_MNI_VOLUME, 0, NULL, NULL);

	// Set the T1 volume as the original volume to interpolate from
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {MNI_DATA_W, MNI_DATA_H, MNI_DATA_D};
	clEnqueueCopyBufferToImage(commandQueue, d_Interpolated_T1_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);
		
	// Do the registration with several scales
	//AlignTwoVolumesSeveralScales(h_Registration_Parameters_T1_MNI,1);

	// Copy the aligned volume to host
	clEnqueueReadBuffer(commandQueue, d_Aligned_Volume, CL_TRUE, 0, DATA_SIZE_MNI_VOLUME, h_Aligned_T1_Volume, 0, NULL, NULL);

	// Cleanup allocated memory
	AlignTwoVolumesCleanup();	
}

// Performs registration between one low resolution fMRI volume and a high resolution MNI volume
//void BROCCOLI_LIB::PerformRegistrationEPIMNI()
//{
//}


// Performs slice timing correction of an fMRI dataset
void BROCCOLI_LIB::PerformSliceTimingCorrection()
{
	
}



// Performs motion correction of an fMRI dataset
void BROCCOLI_LIB::PerformMotionCorrection()
{
	// Setup all parameters and allocate memory on host
	AlignTwoVolumesSetup(FMRI_DATA_W, FMRI_DATA_H, FMRI_DATA_D);

	// Set the first volume as the reference volume
	clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Reference_Volume, 0, 0, DATA_SIZE_FMRI_VOLUME, 0, NULL, NULL);

	// Run the registration for each volume
	for (int t = 0; t < FMRI_DATA_T; t++)
	{
		// Set a new volume to be aligned
		clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Aligned_Volume, t * FMRI_DATA_W * FMRI_DATA_H * FMRI_DATA_D, 0, DATA_SIZE_FMRI_VOLUME, 0, NULL, NULL);

		// Also copy the same volume to an image to interpolate from
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {FMRI_DATA_W, FMRI_DATA_H, FMRI_DATA_D};
		clEnqueueCopyBufferToImage(commandQueue, d_fMRI_Volumes, d_Original_Volume, t * FMRI_DATA_W * FMRI_DATA_H * FMRI_DATA_D, origin, region, 0, NULL, NULL);
		
		// Do the registration with only one scale
		AlignTwoVolumes(h_Registration_Parameters);

		// Copy the corrected volume to the corrected volumes
		clEnqueueCopyBuffer(commandQueue, d_Aligned_Volume, d_Motion_Corrected_fMRI_Volumes, 0, t * FMRI_DATA_W * FMRI_DATA_H * FMRI_DATA_D, DATA_SIZE_FMRI_VOLUME, 0, NULL, NULL);
	
		// Write the total parameter vector to host
		for (int i = 0; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; i++)
		{
			h_Motion_Parameters[t + i * DATA_T] = h_Registration_Parameters[i];
		}
	}
	
	// Copy all corrected volumes to host
	clEnqueueReadBuffer(commandQueue, d_Motion_Corrected_fMRI_Volumes, CL_TRUE, 0, DATA_SIZE_FMRI_VOLUMES, h_Motion_Corrected_fMRI_Volumes, 0, NULL, NULL);

	// Cleanup allocated memory
	AlignTwoVolumesCleanup();
}

void BROCCOLI_LIB::AddVolumes()
{
	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_ONLY, DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float), NULL, NULL);
	d_Motion_Corrected_fMRI_Volumes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float), NULL, NULL);

	// Copy volumes to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

	int N = DATA_W * DATA_H * DATA_D * DATA_T;
	clSetKernelArg(AddKernel, 0, sizeof(cl_mem), &d_Motion_Corrected_fMRI_Volumes);
	clSetKernelArg(AddKernel, 1, sizeof(cl_mem), &d_fMRI_Volumes);	
	clSetKernelArg(AddKernel, 2, sizeof(int), &N);


	size_t globalWorkSize[1] = {DATA_W * DATA_H * DATA_D * DATA_T}; // Total number of threads
	size_t localWorkSize[1] = {512}; // Number of threads per block

	// Launch kernel
	clEnqueueNDRangeKernel(commandQueue, AddKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

	// Copy result back to host
	clEnqueueReadBuffer(commandQueue, d_Motion_Corrected_fMRI_Volumes, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float), h_Result, 0, NULL, NULL);

	// Free memory
	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Motion_Corrected_fMRI_Volumes);
}

int BROCCOLI_LIB::GetOpenCLError()
{
	return error;
}

int BROCCOLI_LIB::GetOpenCLKernelError()
{
	return kernel_error;
}


//void BROCCOLI_LIB::PerformSmoothingTest(cl_mem d_Smoothed_Volumes, cl_mem d_Volumes, int NUMBER_OF_VOLUMES, cl_mem c_Smoothing_Filter_X, cl_mem c_Smoothing_Filter_Y, cl_mem c_Smoothing_Filter_Z)
void BROCCOLI_LIB::PerformSmoothingTest()
{
	// Allocate memory for smoothing filters
	c_Smoothing_Filter_X = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Y = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Z = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);

	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_ONLY, DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float), NULL, NULL);
	d_Smoothed_fMRI_Volumes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float), NULL, NULL);

	// Copy volumes to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

	// Allocate temporary memory
	cl_mem d_Convolved_Rows = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Convolved_Columns = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);

	// Copy smoothing filters to constant memory
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_X, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_X , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Y, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Y , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Z, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Z , 0, NULL, NULL);

	// Set arguments for the kernels
	clSetKernelArg(SeparableConvolutionRowsKernel, 0, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionRowsKernel, 1, sizeof(cl_mem), &d_fMRI_Volumes);
	clSetKernelArg(SeparableConvolutionRowsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_Y);
	clSetKernelArg(SeparableConvolutionRowsKernel, 4, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionRowsKernel, 5, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionRowsKernel, 6, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionRowsKernel, 7, sizeof(int), &DATA_T);

	clSetKernelArg(SeparableConvolutionColumnsKernel, 0, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 1, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_X);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 4, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 5, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 6, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 7, sizeof(int), &DATA_T);

	clSetKernelArg(SeparableConvolutionRodsKernel, 0, sizeof(cl_mem), &d_Smoothed_fMRI_Volumes);
	clSetKernelArg(SeparableConvolutionRodsKernel, 1, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionRodsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_Z);
    clSetKernelArg(SeparableConvolutionRodsKernel, 4, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionRodsKernel, 5, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionRodsKernel, 6, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionRodsKernel, 7, sizeof(int), &DATA_T);

	// Loop over volumes
	for (int v = 0; v < DATA_T; v++)
	{		
		clSetKernelArg(SeparableConvolutionRowsKernel, 3, sizeof(int), &v);
		error = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRowsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRows, localWorkSizeSeparableConvolutionRows, 0, NULL, NULL);
		clFinish(commandQueue);
		
		clSetKernelArg(SeparableConvolutionColumnsKernel, 3, sizeof(int), &v);
		error = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionColumnsKernel, 3, NULL, globalWorkSizeSeparableConvolutionColumns, localWorkSizeSeparableConvolutionColumns, 0, NULL, NULL);
		clFinish(commandQueue);
		
		clSetKernelArg(SeparableConvolutionRodsKernel, 3, sizeof(int), &v);
		error = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRodsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRods, localWorkSizeSeparableConvolutionRods, 0, NULL, NULL);
		clFinish(commandQueue);	
	}

	
	// Copy result back to host
	clEnqueueReadBuffer(commandQueue, d_Smoothed_fMRI_Volumes, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float), h_Result, 0, NULL, NULL);
	clFinish(commandQueue);

	// Free memory
	clReleaseMemObject(d_Convolved_Rows);
	clReleaseMemObject(d_Convolved_Columns);

	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Smoothed_fMRI_Volumes);

	clReleaseMemObject(c_Smoothing_Filter_X);
	clReleaseMemObject(c_Smoothing_Filter_Y);
	clReleaseMemObject(c_Smoothing_Filter_Z);
}

// Performs smoothing of a number of volumes
void BROCCOLI_LIB::PerformSmoothing(cl_mem d_Smoothed_Volumes, cl_mem d_Volumes, int NUMBER_OF_VOLUMES, cl_mem c_Smoothing_Filter_X, cl_mem c_Smoothing_Filter_Y, cl_mem c_Smoothing_Filter_Z)
{
	// Allocate temporary memory
	cl_mem d_Convolved_Rows = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Convolved_Columns = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);

	// Set arguments for the kernels
	clSetKernelArg(SeparableConvolutionRowsKernel, 0, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionRowsKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(SeparableConvolutionRowsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_Y);
	clSetKernelArg(SeparableConvolutionRowsKernel, 4, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionRowsKernel, 5, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionRowsKernel, 6, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionRowsKernel, 7, sizeof(int), &DATA_T);

	clSetKernelArg(SeparableConvolutionColumnsKernel, 0, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 1, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_X);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 4, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 5, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 6, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 7, sizeof(int), &DATA_T);

	clSetKernelArg(SeparableConvolutionRodsKernel, 0, sizeof(cl_mem), &d_Smoothed_Volumes);
	clSetKernelArg(SeparableConvolutionRodsKernel, 1, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionRodsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_Z);
	clSetKernelArg(SeparableConvolutionRodsKernel, 4, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionRodsKernel, 5, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionRodsKernel, 6, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionRodsKernel, 7, sizeof(int), &DATA_T);


	// Loop over volumes
	for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
	{
		clSetKernelArg(SeparableConvolutionRowsKernel, 3, sizeof(int), &v);
		clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRowsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRows, localWorkSizeSeparableConvolutionRows, 0, NULL, NULL);
		clFinish(commandQueue);

		clSetKernelArg(SeparableConvolutionColumnsKernel, 3, sizeof(int), &v);
		clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionColumnsKernel, 3, NULL, globalWorkSizeSeparableConvolutionColumns, localWorkSizeSeparableConvolutionColumns, 0, NULL, NULL);
		clFinish(commandQueue);
	
		clSetKernelArg(SeparableConvolutionRodsKernel, 3, sizeof(int), &v);
		clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRodsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRods, localWorkSizeSeparableConvolutionRods, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	// Free temporary memory
	clReleaseMemObject(d_Convolved_Rows);
	clReleaseMemObject(d_Convolved_Columns);
}

// Performs detrending of an fMRI dataset
void BROCCOLI_LIB::PerformDetrending()
{
	// First estimate beta weights
	clSetKernelArg(CalculateBetaValuesGLMKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_Detrend);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 4, sizeof(cl_mem), &c_Censor);
	clEnqueueNDRangeKernel(commandQueue, CalculateBetaValuesGLMKernel, 3, NULL, globalWorkSizeCalculateBetaValuesGLM, localWorkSizeCalculateBetaValuesGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Then remove linear fit
	clSetKernelArg(RemoveLinearFitKernel, 0, sizeof(cl_mem), &d_Detrended_fMRI_Volumes);
	clSetKernelArg(RemoveLinearFitKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(RemoveLinearFitKernel, 2, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(RemoveLinearFitKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(RemoveLinearFitKernel, 4, sizeof(cl_mem), &c_X_Detrend);
	clEnqueueNDRangeKernel(commandQueue, RemoveLinearFitKernel, 3, NULL, globalWorkSizeRemoveLinearFit, localWorkSizeRemoveLinearFit, 0, NULL, NULL);
	clFinish(commandQueue);
}

// Processing

// Runs all the preprocessing steps and the statistical analysis for one subject
void BROCCOLI_LIB::PerformPreprocessingAndCalculateStatisticalMaps()
{
	//PerformRegistrationEPIT1();
	//PerformRegistrationT1MNI();

    //PerformSliceTimingCorrection();
	PerformMotionCorrection();
	//PerformSmoothing();	
	PerformDetrending();
	//CalculateStatisticalMapsFirstLevel();

	//CalculateSlicesPreprocessedfMRIData();
}

// Calculates a statistical map for first level analysis
void BROCCOLI_LIB::CalculateStatisticalMapsGLMFirstLevel()
{
	// Calculate beta values
	clSetKernelArg(CalculateBetaValuesGLMKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_GLM);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 4, sizeof(cl_mem), &c_Censor);
	clEnqueueNDRangeKernel(commandQueue, CalculateBetaValuesGLMKernel, 3, NULL, globalWorkSizeCalculateBetaValuesGLM, localWorkSizeCalculateBetaValuesGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Calculate t-values and residuals
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 1, sizeof(cl_mem), &d_Beta_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 2, sizeof(cl_mem), &d_Residual_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 3, sizeof(cl_mem), &d_Residual_Variances);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 4, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 5, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 6, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 7, sizeof(cl_mem), &c_X_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 8, sizeof(cl_mem), &c_Contrast_Vectors);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 9, sizeof(float), &ctxtxc);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 10, sizeof(cl_mem), &c_Censor);
	clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Estimate auto correlation from residuals

	// Remove auto correlation from regressors and data


}

// Calculates a statistical map for second level analysis
void BROCCOLI_LIB::CalculateStatisticalMapsGLMSecondLevel()
{
	// Calculate beta values
	clSetKernelArg(CalculateBetaValuesGLMKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_GLM);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 4, sizeof(cl_mem), &c_Censor);
	clEnqueueNDRangeKernel(commandQueue, CalculateBetaValuesGLMKernel, 3, NULL, globalWorkSizeCalculateBetaValuesGLM, localWorkSizeCalculateBetaValuesGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Calculate t-values and residuals
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 1, sizeof(cl_mem), &d_Beta_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 2, sizeof(cl_mem), &d_Residual_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 3, sizeof(cl_mem), &d_Residual_Variances);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 4, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 5, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 6, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 7, sizeof(cl_mem), &c_X_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 8, sizeof(cl_mem), &c_Contrast_Vectors);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 9, sizeof(cl_mem), &ctxtxc);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 10, sizeof(cl_mem), &c_Censor);
	clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);
}


// Calculates a significance threshold for a single subject
void BROCCOLI_LIB::CalculatePermutationTestThresholdSingleSubject()
{
	SetupParametersPermutationSingleSubject();
	GeneratePermutationMatrixSingleSubject();

	// Make the timeseries white prior to the random permutations (if single subject)
	WhitenfMRIVolumes();
	CreateBOLDRegressedVolumes();
	//PerformWhiteningPriorPermutation();
	
    // Loop over all the permutations, save the maximum test value from each permutation
    for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
    {
        // Copy a new permutation vector to constant memory
   
        GeneratePermutedfMRIVolumes();
        PerformSmoothingPermutation();
        PerformDetrendingPermutation();
        //PerformWhiteningPermutation();
        CalculateActivityMapPermutation();
		h_Maximum_Test_Values[p] = FindMaxTestvaluePermutation();  
    }

	
    // Sort the maximum test values
	
	// Find the threshold for the significance level
	
	NUMBER_OF_SIGNIFICANTLY_ACTIVE_VOXELS = 0;
	for (int i = 0; i < DATA_W * DATA_H * DATA_D; i++)
	{
		//if (h_Activity_Volume[i] >= permutation_test_threshold)
		//{
		//	NUMBER_OF_SIGNIFICANTLY_ACTIVE_VOXELS++;
		//}
	}
}

// Calculates a significance threshold for a group of subjects
void BROCCOLI_LIB::CalculatePermutationTestThresholdMultiSubject()
{
	SetupParametersPermutationMultiSubject();
	GeneratePermutationMatrixMultiSubject();

    // Loop over all the permutations, save the maximum test value from each permutation
    for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
    {
         // Copy a new permutation vector to constant memory
   
        GeneratePermutedfMRIVolumes();
        CalculateActivityMapPermutation();
		h_Maximum_Test_Values[p] = FindMaxTestvaluePermutation();  
    }

	
    // Sort the maximum test values
	
	// Find the threshold for the significance level
	
	NUMBER_OF_SIGNIFICANTLY_ACTIVE_VOXELS = 0;
	for (int i = 0; i < DATA_W * DATA_H * DATA_D; i++)
	{
		//if (h_Activity_Volume[i] >= permutation_test_threshold)
		//{
		//	NUMBER_OF_SIGNIFICANTLY_ACTIVE_VOXELS++;
		//}
	}
}





// Functions for permutations, single subject

void BROCCOLI_LIB::SetupParametersPermutationSingleSubject()
{	

}

// Generates a permutation matrix for a single subject
void BROCCOLI_LIB::GeneratePermutationMatrixSingleSubject()
{
    for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
    {
		// Generate numbers from 0 to number of timepoints
        for (int i = 0; i < DATA_T; i++)
        {			
            h_Permutation_Matrix[i + p * DATA_T] = (unsigned short int)i;
        }

		// Generate random number and switch position of two existing numbers
        for (int i = 0; i < DATA_T; i++)
        {			
            int j = rand() % (DATA_T - i) + i;
            unsigned short int temp = h_Permutation_Matrix[j + p * DATA_T];
            h_Permutation_Matrix[j + p * DATA_T] = h_Permutation_Matrix[i + p * DATA_T];
            h_Permutation_Matrix[i + p * DATA_T] = temp;
        }
    }
}

void BROCCOLI_LIB::PerformDetrendingPriorPermutation()
{	
	
}

void BROCCOLI_LIB::CreateBOLDRegressedVolumes()
{	
	
}


void BROCCOLI_LIB::WhitenfMRIVolumes()
{
	clSetKernelArg(EstimateAR4ModelsKernel, 0, sizeof(cl_mem), &d_AR1_Estimates);
	clSetKernelArg(EstimateAR4ModelsKernel, 1, sizeof(cl_mem), &d_AR2_Estimates);
	clSetKernelArg(EstimateAR4ModelsKernel, 2, sizeof(cl_mem), &d_AR3_Estimates);
	clSetKernelArg(EstimateAR4ModelsKernel, 3, sizeof(cl_mem), &d_AR4_Estimates);
	clSetKernelArg(EstimateAR4ModelsKernel, 4, sizeof(cl_mem), &d_Detrended_fMRI_Volumes);
	clSetKernelArg(EstimateAR4ModelsKernel, 5, sizeof(cl_mem), &d_Mask);
	clEnqueueNDRangeKernel(commandQueue, EstimateAR4ModelsKernel, 3, NULL, globalWorkSizeEstimateAR4Models, localWorkSizeEstimateAR4Models, 0, NULL, NULL);
	clFinish(commandQueue);

	PerformSmoothing(d_Smoothed_AR1_Estimates, d_AR1_Estimates, 1, c_AR_Smoothing_Filter_X, c_AR_Smoothing_Filter_Y, c_AR_Smoothing_Filter_Z);
	PerformSmoothing(d_Smoothed_AR2_Estimates, d_AR2_Estimates, 1, c_AR_Smoothing_Filter_X, c_AR_Smoothing_Filter_Y, c_AR_Smoothing_Filter_Z);
	PerformSmoothing(d_Smoothed_AR3_Estimates, d_AR3_Estimates, 1, c_AR_Smoothing_Filter_X, c_AR_Smoothing_Filter_Y, c_AR_Smoothing_Filter_Z);
	PerformSmoothing(d_Smoothed_AR4_Estimates, d_AR4_Estimates, 1, c_AR_Smoothing_Filter_X, c_AR_Smoothing_Filter_Y, c_AR_Smoothing_Filter_Z);
	
	clSetKernelArg(ApplyWhiteningAR4Kernel, 0, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
	clSetKernelArg(ApplyWhiteningAR4Kernel, 1, sizeof(cl_mem), &d_Detrended_fMRI_Volumes);
	clSetKernelArg(ApplyWhiteningAR4Kernel, 3, sizeof(cl_mem), &d_Smoothed_AR1_Estimates);
	clSetKernelArg(ApplyWhiteningAR4Kernel, 4, sizeof(cl_mem), &d_Smoothed_AR2_Estimates);
	clSetKernelArg(ApplyWhiteningAR4Kernel, 5, sizeof(cl_mem), &d_Smoothed_AR3_Estimates);
	clSetKernelArg(ApplyWhiteningAR4Kernel, 6, sizeof(cl_mem), &d_Smoothed_AR4_Estimates);
	clSetKernelArg(ApplyWhiteningAR4Kernel, 7, sizeof(cl_mem), &d_Mask);
	clEnqueueNDRangeKernel(commandQueue, ApplyWhiteningAR4Kernel, 3, NULL, globalWorkSizeApplyWhiteningAR4, localWorkSizeApplyWhiteningAR4, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::GeneratePermutedfMRIVolumes()
{
	clSetKernelArg(GeneratePermutedfMRIVolumesAR4Kernel, 0, sizeof(cl_mem), &d_Permuted_fMRI_Volumes);
	clSetKernelArg(GeneratePermutedfMRIVolumesAR4Kernel, 1, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
	clSetKernelArg(GeneratePermutedfMRIVolumesAR4Kernel, 2, sizeof(cl_mem), &d_Smoothed_AR1_Estimates);
	clSetKernelArg(GeneratePermutedfMRIVolumesAR4Kernel, 3, sizeof(cl_mem), &d_Smoothed_AR2_Estimates);
	clSetKernelArg(GeneratePermutedfMRIVolumesAR4Kernel, 4, sizeof(cl_mem), &d_Smoothed_AR3_Estimates);
	clSetKernelArg(GeneratePermutedfMRIVolumesAR4Kernel, 5, sizeof(cl_mem), &d_Smoothed_AR4_Estimates);
	clSetKernelArg(GeneratePermutedfMRIVolumesAR4Kernel, 6, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(GeneratePermutedfMRIVolumesAR4Kernel, 7, sizeof(cl_mem), &c_Permutation_Vector);
	clEnqueueNDRangeKernel(commandQueue, GeneratePermutedfMRIVolumesAR4Kernel, 3, NULL, globalWorkSizeGeneratePermutedfMRIVolumesAR4, localWorkSizeGeneratePermutedfMRIVolumesAR4, 0, NULL, NULL);
	clFinish(commandQueue);
}


void BROCCOLI_LIB::PerformSmoothingPermutation()
{

}

void BROCCOLI_LIB::PerformDetrendingPermutation()
{
	
}

void BROCCOLI_LIB::CalculateActivityMapPermutation()
{
	
}

float BROCCOLI_LIB::FindMaxTestvaluePermutation()
{
	//cudaMemcpy(h_Activity_Volume, d_Activity_Volume, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyDeviceToHost);
	//thrust::host_vector<float> h_vec(h_Activity_Volume, &h_Activity_Volume[DATA_W * DATA_H * DATA_D]); 
	
	//thrust::device_vector<float> d_vec = h_vec;
	//thrust::device_vector<float> d_vec(d_Activity_Volume, &d_Activity_Volume[DATA_W * DATA_H * DATA_D]);

    //return thrust::reduce(d_vec.begin(), d_vec.end(), -1000.0f, thrust::maximum<float>());
	return 1.0f;
}

// Functions for permutations, multi subject

void BROCCOLI_LIB::SetupParametersPermutationMultiSubject()
{	

}

// Generates a permutation matrix for several subjects
void BROCCOLI_LIB::GeneratePermutationMatrixMultiSubject()
{
    for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
    {
		// Generate numbers from 0 to number of subjects
        for (int i = 0; i < NUMBER_OF_SUBJECTS; i++)
        {			
            h_Permutation_Matrix[i + p * NUMBER_OF_SUBJECTS] = (unsigned short int)i;
        }

		// Generate random number and switch position of existing numbers
        for (int i = 0; i < NUMBER_OF_SUBJECTS; i++)
        {			
            int j = rand() % (NUMBER_OF_SUBJECTS - i) + i;

			// Check if random permutation is valid?!

            unsigned short int t = h_Permutation_Matrix[j + p * DATA_T];
            h_Permutation_Matrix[j + p * DATA_T] = h_Permutation_Matrix[i + p * DATA_T];
            h_Permutation_Matrix[i + p * DATA_T] = t;
        }
    }
}






// Read functions, public

void BROCCOLI_LIB::ReadfMRIDataRAW()
{
	SetupParametersReadData();
	
	// Read fMRI volumes from file
	if (DATA_TYPE == FLOAT)
	{
		ReadRealDataFloat(h_fMRI_Volumes, filename_fMRI_data_raw, DATA_W * DATA_H * DATA_D * DATA_T);
	}
	else if (DATA_TYPE == INT32)
	{
		int* h_Temp_Volumes = (int*)malloc(sizeof(int) * DATA_W * DATA_H * DATA_D * DATA_T);
		ReadRealDataInt32(h_Temp_Volumes, filename_fMRI_data_raw, DATA_W * DATA_H * DATA_D * DATA_T);
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)h_Temp_Volumes[i];
		}
		free(h_Temp_Volumes);
	}
	else if (DATA_TYPE == INT16)
	{
		short int* h_Temp_Volumes = (short int*)malloc(sizeof(short int) * DATA_W * DATA_H * DATA_D * DATA_T);
		ReadRealDataInt16(h_Temp_Volumes, filename_fMRI_data_raw, DATA_W * DATA_H * DATA_D * DATA_T);
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)h_Temp_Volumes[i];
		}
		free(h_Temp_Volumes);
	}
	else if (DATA_TYPE == UINT32)
	{
		unsigned int* h_Temp_Volumes = (unsigned int*)malloc(sizeof(unsigned int) * DATA_W * DATA_H * DATA_D * DATA_T);
		ReadRealDataUint32(h_Temp_Volumes, filename_fMRI_data_raw, DATA_W * DATA_H * DATA_D * DATA_T);
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)h_Temp_Volumes[i];
		}
		free(h_Temp_Volumes);
	}
	else if (DATA_TYPE == UINT16)
	{
		unsigned short int* h_Temp_Volumes = (unsigned short int*)malloc(sizeof(unsigned short int) * DATA_W * DATA_H * DATA_D * DATA_T);
		ReadRealDataUint16(h_Temp_Volumes, filename_fMRI_data_raw, DATA_W * DATA_H * DATA_D * DATA_T);
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)h_Temp_Volumes[i];
		}
		free(h_Temp_Volumes);
	}
	else if (DATA_TYPE == DOUBLE)
	{
		double* h_Temp_Volumes = (double*)malloc(sizeof(double) * DATA_W * DATA_H * DATA_D * DATA_T);
		ReadRealDataDouble(h_Temp_Volumes, filename_fMRI_data_raw, DATA_W * DATA_H * DATA_D * DATA_T);
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)h_Temp_Volumes[i];
		}
		free(h_Temp_Volumes);
	}

	// Copy fMRI volumes to global memory, as floats
	//cudaMemcpy(d_fMRI_Volumes, h_fMRI_Volumes, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyHostToDevice);

	for (int i = 0; i < DATA_T; i++)
	{
		plot_values_x[i] = (double)i * (double)TR;
	}


	SegmentBrainData();

	SetupStatisticalAnalysisBasisFunctions();
	SetupDetrendingBasisFunctions();

	CalculateSlicesfMRIData();
}

/*
void BROCCOLI_LIB::ReadfMRIDataNIFTI()
{
	nifti_data = new nifti_image;
	// Read nifti data
	nifti_data = nifti_image_read(filename_fMRI_data_nifti.c_str(), 1);

	if (nifti_data->datatype == DT_SIGNED_SHORT)
	{
		DATA_TYPE = INT16;
	}
	else if (nifti_data->datatype == DT_SIGNED_INT)
	{
		DATA_TYPE = INT32;
	}
	else if (nifti_data->datatype == DT_FLOAT)
	{
		DATA_TYPE = FLOAT;
	}
	else if (nifti_data->datatype == DT_DOUBLE)
	{
		DATA_TYPE = DOUBLE;
	}
	else if (nifti_data->datatype == DT_UNSIGNED_CHAR)
	{
		DATA_TYPE = UINT8;
	}

	// Get number of data points in each direction
	DATA_W = nifti_data->nx;
	DATA_H = nifti_data->ny;
	DATA_D = nifti_data->nz;
	DATA_T = nifti_data->nt;

	FMRI_VOXEL_SIZE_X = nifti_data->dx;
	FMRI_VOXEL_SIZE_Y = nifti_data->dy;
	FMRI_VOXEL_SIZE_Z = nifti_data->dz;
	TR = nifti_data->dt;

	SetupParametersReadData();


	// Get data from nifti image
	if (DATA_TYPE == FLOAT)
	{
		float* data = (float*)nifti_data->data;
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = data[i];
		}
	}
	else if (DATA_TYPE == INT32)
	{
		int* data = (int*)nifti_data->data;
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)data[i];
		}
	}
	else if (DATA_TYPE == INT16)
	{
		short int* data = (short int*)nifti_data->data;
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)data[i];
		}
	}
	else if (DATA_TYPE == DOUBLE)
	{
		double* data = (double*)nifti_data->data;
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)data[i];
		}
	}
	else if (DATA_TYPE == UINT8)
	{
		unsigned char* data = (unsigned char*)nifti_data->data;
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)data[i];
		}
	}

	// Scale if necessary
	if (nifti_data->scl_slope != 0.0f)
	{
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = h_fMRI_Volumes[i] * nifti_data->scl_slope + nifti_data->scl_inter;
		}
	}

	delete nifti_data;

	// Copy fMRI volumes to global memory, as floats
	//cudaMemcpy(d_fMRI_Volumes, h_fMRI_Volumes, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyHostToDevice);

	for (int i = 0; i < DATA_T; i++)
	{
		plot_values_x[i] = (double)i * (double)TR;
	}

	SegmentBrainData();

	SetupStatisticalAnalysisBasisFunctions();
	SetupDetrendingBasisFunctions();

	CalculateSlicesfMRIData();
}
*/

/*

void BROCCOLI_LIB::ReadNIFTIHeader()
{
	// Read nifti header only
	nifti_data = nifti_image_read(filename_fMRI_data_nifti.c_str(), 0);

	// Get dimensions
	DATA_W = nifti_data->nx;
	DATA_H = nifti_data->ny;
	DATA_D = nifti_data->nz;
	DATA_T = nifti_data->nt;

	FMRI_VOXEL_SIZE_X = nifti_data->dx;
	FMRI_VOXEL_SIZE_Y = nifti_data->dy;
	FMRI_VOXEL_SIZE_Z = nifti_data->dz;
	TR = nifti_data->dt;
}

*/

// Read functions, private

void BROCCOLI_LIB::ReadRealDataInt32(int* data, std::string filename, int N)
{
	std::fstream file(filename, std::ios::in | std::ios::binary);
	int current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			file.read( (char*) &current_value, sizeof(current_value) );
			data[i] = current_value;
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}

void BROCCOLI_LIB::ReadRealDataInt16(short int* data, std::string filename, int N)
{
	std::fstream file(filename, std::ios::in | std::ios::binary);
	short int current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			file.read( (char*) &current_value, sizeof(current_value) );
			data[i] = current_value;
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}

void BROCCOLI_LIB::ReadRealDataUint32(unsigned int* data, std::string filename, int N)
{
	std::fstream file(filename, std::ios::in | std::ios::binary);
	unsigned int current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			file.read( (char*) &current_value, sizeof(current_value) );
			data[i] = current_value;
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}

void BROCCOLI_LIB::ReadRealDataUint16(unsigned short int* data, std::string filename, int N)
{
	std::fstream file(filename, std::ios::in | std::ios::binary);
	unsigned short int current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			file.read( (char*) &current_value, sizeof(current_value) );
			data[i] = current_value;
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}

void BROCCOLI_LIB::ReadRealDataFloat(float* data, std::string filename, int N)
{
	std::fstream file(filename, std::ios::in | std::ios::binary);
	float current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			file.read( (char*) &current_value, sizeof(current_value) );
			data[i] = current_value;
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}

void BROCCOLI_LIB::ReadRealDataDouble(double* data, std::string filename, int N)
{
	std::fstream file(filename, std::ios::in | std::ios::binary);
	double current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			file.read( (char*) &current_value, sizeof(current_value) );
			data[i] = current_value;
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}


/*
void BROCCOLI_LIB::ReadComplexData(Complex* data, std::string real_filename, std::string imag_filename, int N)
{
	std::fstream real_file(real_filename, std::ios::in | std::ios::binary);
	std::fstream imag_file(imag_filename, std::ios::in | std::ios::binary);
	float current_value;

	if (real_file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			real_file.read( (char*) &current_value, sizeof(current_value) );
			data[i].x = current_value;
		}
	}
	else
	{
		std::cout << "Could not find file " << real_filename << std::endl;
	}

	if (imag_file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			imag_file.read( (char*) &current_value, sizeof(current_value) );
			data[i].y = current_value;
		}
	}
	else
	{
		std::cout << "Could not find file " << imag_filename << std::endl;
	}

	real_file.close();
	imag_file.close();
}
*/

void BROCCOLI_LIB::ReadImageRegistrationFilters()
{
	// Read the quadrature filters from file
	//ReadComplexData(h_Quadrature_Filter_1, filename_real_quadrature_filter_1, filename_imag_quadrature_filter_1, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE);
	//ReadComplexData(h_Quadrature_Filter_2, filename_real_quadrature_filter_2, filename_imag_quadrature_filter_2, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE);
	//ReadComplexData(h_Quadrature_Filter_3, filename_real_quadrature_filter_3, filename_imag_quadrature_filter_3, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE);
}

void BROCCOLI_LIB::ReadSmoothingFilters()
{
	// Read smoothing filters from file
	std::string mm_string;
	std::stringstream out;
	out << SMOOTHING_AMOUNT_MM;
	mm_string = out.str();

	std::string filename_GLM = filename_GLM_filter + mm_string + "mm.raw";
	//ReadRealDataFloat(h_GLM_Filter, filename_GLM, SMOOTHING_FILTER_SIZE);
	
	//std::string filename_CCA_3D_1 = filename_CCA_3D_filter_1 + mm_string + "mm.raw";
	//std::string filename_CCA_3D_2 = filename_CCA_3D_filter_2 + mm_string + "mm.raw";
	//ReadRealDataFloat(h_CCA_3D_Filter_1, filename_CCA_3D_1, SMOOTHING_FILTER_SIZE);
	//ReadRealDataFloat(h_CCA_3D_Filter_2, filename_CCA_3D_2, SMOOTHING_FILTER_SIZE);
	//Convert2FloatToFloat2(h_CCA_3D_Filters, h_CCA_3D_Filter_1, h_CCA_3D_Filter_2, SMOOTHING_FILTER_SIZE);
}

void BROCCOLI_LIB::SetupParametersReadData()
{
	// Reset all pointers
	
	for (int i = 0; i < NUMBER_OF_HOST_POINTERS; i++)
	{
		void* pointer = host_pointers[i];
		if (pointer != NULL)
		{
			free(pointer);
			host_pointers[i] = NULL;
		}
	}

	for (int i = 0; i < NUMBER_OF_DEVICE_POINTERS; i++)
	{
		float* pointer = device_pointers[i];
		if (pointer != NULL)
		{
			//cudaFree(pointer);
			device_pointers[i] = NULL;
		}
	}

	MOTION_CORRECTED = false;

	X_SLICE_LOCATION_fMRI_DATA = DATA_W / 2;
	Y_SLICE_LOCATION_fMRI_DATA = DATA_H / 2;
	Z_SLICE_LOCATION_fMRI_DATA = DATA_D / 2;
	TIMEPOINT_fMRI_DATA = 0;

	DATA_SIZE_FMRI_VOLUME = sizeof(float) * DATA_W * DATA_H * DATA_D;
	DATA_SIZE_FMRI_VOLUMES = sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T;
	
	int DATA_SIZE_DETRENDING = sizeof(float) * DATA_T * NUMBER_OF_DETRENDING_BASIS_FUNCTIONS;
	
	int DATA_SIZE_TEMPORAL_BASIS_FUNCTIONS = sizeof(float) * DATA_T * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS;
	int DATA_SIZE_COVARIANCE_MATRIX = sizeof(float) * 4;

	h_fMRI_Volumes = (float*)malloc(DATA_SIZE_FMRI_VOLUMES);
	h_Motion_Corrected_fMRI_Volumes = (float*)malloc(DATA_SIZE_FMRI_VOLUMES);
	h_Smoothed_fMRI_Volumes = (float*)malloc(DATA_SIZE_FMRI_VOLUMES);
	h_Detrended_fMRI_Volumes = (float*)malloc(DATA_SIZE_FMRI_VOLUMES);
	
	h_X_Detrend = (float*)malloc(DATA_SIZE_DETRENDING);
	h_xtxxt_Detrend = (float*)malloc(DATA_SIZE_DETRENDING);	

	h_X_GLM = (float*)malloc(DATA_SIZE_TEMPORAL_BASIS_FUNCTIONS);
	h_xtxxt_GLM = (float*)malloc(DATA_SIZE_TEMPORAL_BASIS_FUNCTIONS);
	h_Contrast_Vectors = (float*)malloc(sizeof(float) * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS * NUMBER_OF_CONTRASTS);
	
	//h_Activity_Volume = (float*)malloc(DATA_SIZE_fMRI_VOLUME);
	
	host_pointers[fMRI_VOLUMES] = (void*)h_fMRI_Volumes;
	host_pointers[MOTION_CORRECTED_VOLUMES] = (void*)h_Motion_Corrected_fMRI_Volumes;
	host_pointers[SMOOTHED1] = (void*)h_Smoothed_fMRI_Volumes;
	host_pointers[DETRENDED1] = (void*)h_Detrended_fMRI_Volumes;
	host_pointers[XDETREND1] = (void*)h_X_Detrend;
	host_pointers[XDETREND2] = (void*)h_xtxxt_Detrend;
	host_pointers[XGLM1] = (void*)h_X_GLM;
	host_pointers[XGLM2] = (void*)h_xtxxt_GLM;
	host_pointers[CONTRAST_VECTOR] = (void*)h_Contrast_Vectors;
	//host_pointers[ACTIVITY_VOLUME] = (void*)h_Activity_Volume;

	x_slice_fMRI_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_H * DATA_D);
	y_slice_fMRI_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_W * DATA_D);
	z_slice_fMRI_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_W * DATA_H);

	x_slice_preprocessed_fMRI_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_H * DATA_D);
	y_slice_preprocessed_fMRI_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_W * DATA_D);
	z_slice_preprocessed_fMRI_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_W * DATA_H);

	x_slice_activity_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_H * DATA_D);
	y_slice_activity_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_W * DATA_D);
	z_slice_activity_data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_W * DATA_H);

	motion_parameters_x = (double*)malloc(sizeof(double) * DATA_T);
	motion_parameters_y = (double*)malloc(sizeof(double) * DATA_T);
	motion_parameters_z = (double*)malloc(sizeof(double) * DATA_T);
	plot_values_x = (double*)malloc(sizeof(double) * DATA_T);
	motion_corrected_curve = (double*)malloc(sizeof(double) * DATA_T);
	smoothed_curve = (double*)malloc(sizeof(double) * DATA_T);
	detrended_curve = (double*)malloc(sizeof(double) * DATA_T);

	host_pointers[X_SLICE_fMRI] = (void*)x_slice_fMRI_data;
	host_pointers[Y_SLICE_fMRI] = (void*)y_slice_fMRI_data;
	host_pointers[Z_SLICE_fMRI] = (void*)z_slice_fMRI_data;
	host_pointers[X_SLICE_PREPROCESSED_fMRI] = (void*)x_slice_preprocessed_fMRI_data;
	host_pointers[Y_SLICE_PREPROCESSED_fMRI] = (void*)y_slice_preprocessed_fMRI_data;
	host_pointers[Z_SLICE_PREPROCESSED_fMRI] = (void*)z_slice_preprocessed_fMRI_data;
	host_pointers[X_SLICE_ACTIVITY] = (void*)x_slice_activity_data;
	host_pointers[Y_SLICE_ACTIVITY] = (void*)y_slice_activity_data;
	host_pointers[Z_SLICE_ACTIVITY] = (void*)z_slice_activity_data;
	host_pointers[MOTION_PARAMETERS_X] = (void*)motion_parameters_x;
	host_pointers[MOTION_PARAMETERS_Y] = (void*)motion_parameters_y;
	host_pointers[MOTION_PARAMETERS_Z] = (void*)motion_parameters_z;
	host_pointers[PLOT_VALUES_X] = (void*)plot_values_x;
	host_pointers[MOTION_CORRECTED_CURVE] = (void*)motion_corrected_curve;
	host_pointers[SMOOTHED_CURVE] = (void*)smoothed_curve;
	host_pointers[DETRENDED_CURVE] = (void*)detrended_curve;

	
	//device_pointers[fMRI_VOLUMES] = d_fMRI_Volumes;
	//device_pointers[BRAIN_VOXELS] = d_Brain_Voxels;
	//device_pointers[SMOOTHED_CERTAINTY] = d_Smoothed_Certainty;
	//device_pointers[ACTIVITY_VOLUME] = d_Activity_Volume;
}


// Write functions, public

void BROCCOLI_LIB::WritefMRIDataNIFTI()
{


}

// Write functions, private

void BROCCOLI_LIB::WriteRealDataUint16(unsigned short int* data, std::string filename, int N)
{
	std::fstream file(filename, std::ios::out | std::ios::binary);
	unsigned short int current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			current_value = data[i];
			file.write( (const char*) &current_value, sizeof(current_value) );
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}

void BROCCOLI_LIB::WriteRealDataFloat(float* data, std::string filename, int N)
{
	std::fstream file(filename, std::ios::out | std::ios::binary);
	float current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			current_value = data[i];
			file.write( (const char*) &current_value, sizeof(current_value) );
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}

void BROCCOLI_LIB::WriteRealDataDouble(double* data, std::string filename, int N)
{
	std::fstream file(filename, std::ios::out | std::ios::binary);
	double current_value;

	if (file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			current_value = data[i];
			file.write( (const char*) &current_value, sizeof(current_value) );
		}
	}
	else
	{
		std::cout << "Could not find file " << filename << std::endl;
	}

	file.close();
}

/*
void BROCCOLI_LIB::WriteComplexData(Complex* data, std::string real_filename, std::string imag_filename, int N)
{
	std::fstream real_file(real_filename, std::ios::out | std::ios::binary);
	std::fstream imag_file(imag_filename, std::ios::out | std::ios::binary);

	float current_value;

	if (real_file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			current_value = data[i].x;
			real_file.write( (const char*) &current_value, sizeof(current_value) );
		}
	}
	else
	{
		std::cout << "Could not find file " << real_filename << std::endl;
	}
	real_file.close();

	if (imag_file.good())
	{
		for (int i = 0; i < N ; i++)
		{
			current_value = data[i].y;
			imag_file.write( (const char*) &current_value, sizeof(current_value) );
		}
	}
	else
	{
		std::cout << "Could not find file " << imag_filename << std::endl;
	}
	imag_file.close();
}
*/




// Help functions

/*
void BROCCOLI_LIB::CalculateSlicesActivityData()
{
	//float max = CalculateMax(h_Activity_Volume, DATA_W * DATA_H * DATA_D);
	//float min = CalculateMin(h_Activity_Volume, DATA_W * DATA_H * DATA_D);
	float adder = -min;
	float multiplier = 255.0f / (max + adder);

	for (int x = 0; x < DATA_W; x++)
	{
		for (int y = 0; y < DATA_H; y++)
		{
			if (THRESHOLD_ACTIVITY_MAP)
			{
				if (h_Activity_Volume[x + y * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H] >= ACTIVITY_THRESHOLD)
				{
					z_slice_activity_data[x + y * DATA_W] = (unsigned char)((h_Activity_Volume[x + y * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H] + adder) * multiplier);
				}
				else
				{
					z_slice_activity_data[x + y * DATA_W] = 0;
				}
			}
			else
			{
				z_slice_activity_data[x + y * DATA_W] = (unsigned char)((h_Activity_Volume[x + y * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H] + adder) * multiplier);
			}
		}
	}

	for (int x = 0; x < DATA_W; x++)
	{
		for (int z = 0; z < DATA_D; z++)
		{
			int inv_z = DATA_D - 1 - z;
			if (THRESHOLD_ACTIVITY_MAP)
			{
				if (h_Activity_Volume[x + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + z * DATA_W * DATA_H] >= ACTIVITY_THRESHOLD)
				{
					y_slice_activity_data[x + inv_z * DATA_W] = (unsigned char)((h_Activity_Volume[x + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + z * DATA_W * DATA_H] + adder) * multiplier);
				}
				else
				{
					y_slice_activity_data[x + inv_z * DATA_W] = 0;
				}
			}
			else
			{
				y_slice_activity_data[x + inv_z * DATA_W] = (unsigned char)((h_Activity_Volume[x + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + z * DATA_W * DATA_H] + adder) * multiplier);
			}
		}
	}

	for (int y = 0; y < DATA_H; y++)
	{
		for (int z = 0; z < DATA_D; z++)
		{
			int inv_z = DATA_D - 1 - z;
			if (THRESHOLD_ACTIVITY_MAP)
			{
				if (h_Activity_Volume[X_SLICE_LOCATION_fMRI_DATA + y * DATA_W + z * DATA_W * DATA_H] >= ACTIVITY_THRESHOLD)
				{
					x_slice_activity_data[y + inv_z * DATA_H] = (unsigned char)((h_Activity_Volume[X_SLICE_LOCATION_fMRI_DATA + y * DATA_W + z * DATA_W * DATA_H] + adder) * multiplier);
				}
				else
				{
					x_slice_activity_data[y + inv_z * DATA_H] = 0;
				}
			}
			else
			{
				x_slice_activity_data[y + inv_z * DATA_H] = (unsigned char)((h_Activity_Volume[X_SLICE_LOCATION_fMRI_DATA + y * DATA_W + z * DATA_W * DATA_H] + adder) * multiplier);
			}
		}
	}
}
*/

void BROCCOLI_LIB::CalculateSlicesfMRIData()
{
	float max = CalculateMax(h_fMRI_Volumes, DATA_W * DATA_H * DATA_D * DATA_T);
	float min = CalculateMin(h_fMRI_Volumes, DATA_W * DATA_H * DATA_D * DATA_T);
	float adder = -min;
	float multiplier = 255.0f / (max + adder);

	for (int x = 0; x < DATA_W; x++)
	{
		for (int y = 0; y < DATA_H; y++)
		{
			z_slice_fMRI_data[x + y * DATA_W] = (unsigned char)((h_fMRI_Volumes[x + y * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H + TIMEPOINT_fMRI_DATA * DATA_W * DATA_H * DATA_D] + adder) * multiplier);
		}
	}

	for (int x = 0; x < DATA_W; x++)
	{
		for (int z = 0; z < DATA_D; z++)
		{
			int inv_z = DATA_D - 1 - z;
			y_slice_fMRI_data[x + inv_z * DATA_W] = (unsigned char)((h_fMRI_Volumes[x + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + z * DATA_W * DATA_H + TIMEPOINT_fMRI_DATA * DATA_W * DATA_H * DATA_D] + adder) * multiplier);
		}
	}

	for (int y = 0; y < DATA_H; y++)
	{
		for (int z = 0; z < DATA_D; z++)
		{
			int inv_z = DATA_D - 1 - z;
			x_slice_fMRI_data[y + inv_z * DATA_H] = (unsigned char)((h_fMRI_Volumes[X_SLICE_LOCATION_fMRI_DATA + y * DATA_W + z * DATA_W * DATA_H + TIMEPOINT_fMRI_DATA * DATA_W * DATA_H * DATA_D] + adder) * multiplier);
		}
	}
}

void BROCCOLI_LIB::CalculateSlicesPreprocessedfMRIData()
{
	float* pointer = NULL;

	if (PREPROCESSED == MOTION_CORRECTION)
	{
		pointer = h_Motion_Corrected_fMRI_Volumes;
	}
	else if (PREPROCESSED == SMOOTHING)
	{
		pointer = h_Smoothed_fMRI_Volumes;
	}
	else if (PREPROCESSED == DETRENDING)
	{
		pointer = h_Detrended_fMRI_Volumes;
	}

	float max = CalculateMax(pointer, DATA_W * DATA_H * DATA_D * DATA_T);
	float min = CalculateMin(pointer, DATA_W * DATA_H * DATA_D * DATA_T);
	float adder = -min;
	float multiplier = 255.0f / (max + adder);


	for (int x = 0; x < DATA_W; x++)
	{
		for (int y = 0; y < DATA_H; y++)
		{
			z_slice_preprocessed_fMRI_data[x + y * DATA_W] = (unsigned char)((pointer[x + y * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H + TIMEPOINT_fMRI_DATA * DATA_W * DATA_H * DATA_D] + adder) * multiplier);
		}
	}

	for (int x = 0; x < DATA_W; x++)
	{
		for (int z = 0; z < DATA_D; z++)
		{
			int inv_z = DATA_D - 1 - z;
			y_slice_preprocessed_fMRI_data[x + inv_z * DATA_W] = (unsigned char)((pointer[x + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + z * DATA_W * DATA_H + TIMEPOINT_fMRI_DATA * DATA_W * DATA_H * DATA_D] + adder) * multiplier);
		}
	}

	for (int y = 0; y < DATA_H; y++)
	{
		for (int z = 0; z < DATA_D; z++)
		{
			int inv_z = DATA_D - 1 - z;
			x_slice_preprocessed_fMRI_data[y + inv_z * DATA_H] = (unsigned char)((pointer[X_SLICE_LOCATION_fMRI_DATA + y * DATA_W + z * DATA_W * DATA_H + TIMEPOINT_fMRI_DATA * DATA_W * DATA_H * DATA_D] + adder) * multiplier);
		}
	}
}

/*
void BROCCOLI_LIB::Convert4FloatToFloat4(float4* floats, float* float_1, float* float_2, float* float_3, float* float_4, int N)
{
	for (int i = 0; i < N; i++)
	{
		floats[i].x = float_1[i];
		floats[i].y = float_2[i];
		floats[i].z = float_3[i];
		floats[i].w = float_4[i];
	}
}
*/

/*
void BROCCOLI_LIB::Convert2FloatToFloat2(float2* floats, float* float_1, float* float_2, int N)
{
	for (int i = 0; i < N; i++)
	{
		floats[i].x = float_1[i];
		floats[i].y = float_2[i];
	}
}
*/

/*
void BROCCOLI_LIB::ConvertRealToComplex(Complex* complex_data, float* real_data, int N)
{
	for (int i = 0; i < N; i++)
	{
		complex_data[i].x = real_data[i];
		complex_data[i].y = 0.0f;
	}
}
*/

/*
void BROCCOLI_LIB::ExtractRealData(float* real_data, Complex* complex_data, int N)
{
	for (int i = 0; i < N; i++)
	{
		real_data[i] = complex_data[i].x;
	}
}
*/


void BROCCOLI_LIB::Invert_Matrix(float* inverse_matrix, float* matrix, int N)
{      
    int i = 0;
    int j = 0;
    int k = 0;
    
	int NUMBER_OF_ROWS = N;
    int NUMBER_OF_COLUMNS = N;
    int n = N;
	int m = N;

    float* LU = (float*)malloc(sizeof(float) * N * N);

    /* Copy A to LU matrix */
    for(i = 0; i < NUMBER_OF_ROWS * NUMBER_OF_COLUMNS; i++)
    {
        LU[i] = matrix[i];
    }
    
    /* Perform LU decomposition */
    float* piv = (float*)malloc(sizeof(float) * N);
    for (i = 0; i < m; i++) 
    {
        piv[i] = i;
    }
    float pivsign = 1;
    /* Main loop */
    for (k = 0; k < n; k++) 
    {
        /* Find pivot */
        int p = k;
        for (i = k+1; i < m; i++) 
        {
            if (abs(LU[i + k * NUMBER_OF_ROWS]) > abs(LU[p + k * NUMBER_OF_ROWS])) 
            {
                p = i;
            }
        }
        /* Exchange if necessary */
        if (p != k) 
        {
            for (j = 0; j < n; j++) 
            {
                float t = LU[p + j*NUMBER_OF_ROWS]; LU[p + j*NUMBER_OF_ROWS] = LU[k + j*NUMBER_OF_ROWS]; LU[k + j*NUMBER_OF_ROWS] = t;
            }
            int t = piv[p]; piv[p] = piv[k]; piv[k] = t;
            pivsign = -pivsign;
        }
        /* Compute multipliers and eliminate k-th column */
        if (LU[k + k*NUMBER_OF_ROWS] != 0.0) 
        {
            for (i = k+1; i < m; i++) 
            {
                LU[i + k*NUMBER_OF_ROWS] /= LU[k + k*NUMBER_OF_ROWS];
                for (j = k+1; j < n; j++) 
                {
                    LU[i + j*NUMBER_OF_ROWS] -= LU[i + k*NUMBER_OF_ROWS]*LU[k + j*NUMBER_OF_ROWS];
                }
            }
        }
    }
    
    /* "Solve" equation system AX = B with B = identity matrix
     to get matrix inverse */
    
    /* Make an identity matrix of the right size */
    float* B = (float*)malloc(sizeof(float) * N * N);
    float* X = (float*)malloc(sizeof(float) * N * N);
    
    for (i = 0; i < NUMBER_OF_ROWS; i++)
    {
        for (j = 0; j < NUMBER_OF_COLUMNS; j++)
        {
            if (i == j)
            {
                B[i + j * NUMBER_OF_ROWS] = 1;
            }
            else
            {
                B[i + j * NUMBER_OF_ROWS] = 0;
            }           
        }
    }
    
    /* Pivot the identity matrix */
    for (i = 0; i < NUMBER_OF_ROWS; i++)
    {
        int current_row = piv[i];
        
        for (j = 0; j < NUMBER_OF_COLUMNS; j++)
        {
            X[i + j * NUMBER_OF_ROWS] = B[current_row + j * NUMBER_OF_ROWS];
        }
    }
    
    /* Solve L*Y = B(piv,:) */
    for (k = 0; k < n; k++) 
    {
        for (i = k+1; i < n; i++) 
        {
            for (j = 0; j < NUMBER_OF_COLUMNS; j++) 
            {
                X[i + j*NUMBER_OF_ROWS] -= X[k + j*NUMBER_OF_ROWS]*LU[i + k*NUMBER_OF_ROWS];
            }
        }
    }
    /* Solve U*X = Y */
    for (k = n-1; k >= 0; k--) 
    {
        for (j = 0; j < NUMBER_OF_COLUMNS; j++) 
        {
            X[k + j*NUMBER_OF_ROWS] /= LU[k + k*NUMBER_OF_ROWS];
        }
        for (i = 0; i < k; i++) 
        {
            for (j = 0; j < NUMBER_OF_COLUMNS; j++) 
            {
                X[i + j*NUMBER_OF_ROWS] -= X[k + j*NUMBER_OF_ROWS]*LU[i + k*NUMBER_OF_ROWS];
            }
        }
    }
    
    for (i = 0; i < NUMBER_OF_ROWS; i++)
    {
        for (j = 0; j < NUMBER_OF_COLUMNS; j++)
        {
            inverse_matrix[i + j * NUMBER_OF_ROWS] = X[i + j * NUMBER_OF_ROWS];
        }
    }

	free(LU);
	free(piv);
	free(B);
	free(X);
}

void BROCCOLI_LIB::Calculate_Square_Root_of_Matrix(float* sqrt_matrix, float* matrix, int N)
{
	float* tempinv = (float*)malloc(sizeof(float) * N * N); 

	for (int i = 0; i < N * N; i++)
	{
		sqrt_matrix[i] = 0.0f;
	}

	for (int i = 0; i < N; i++)
	{
		sqrt_matrix[i + i * N] = 1.0f;
	}

	for (int iteration = 0; iteration < 15; iteration++)
	{
		Invert_Matrix(tempinv, sqrt_matrix, N);
		sqrt_matrix[0] = 0.5f * sqrt_matrix[0] + 0.5f * (matrix[0] * tempinv[0] + matrix[1] * tempinv[2]);
		sqrt_matrix[1] = 0.5f * sqrt_matrix[1] + 0.5f * (matrix[0] * tempinv[1] + matrix[1] * tempinv[3]);
		sqrt_matrix[2] = 0.5f * sqrt_matrix[2] + 0.5f * (matrix[2] * tempinv[0] + matrix[3] * tempinv[2]);
		sqrt_matrix[3] = 0.5f * sqrt_matrix[3] + 0.5f * (matrix[2] * tempinv[1] + matrix[3] * tempinv[3]);
	}

	free(tempinv);
}

void BROCCOLI_LIB::SolveEquationSystem(float* h_A_matrix, float* h_inverse_A_matrix, float* h_h_vector, float* h_Parameter_Vector, int N)
{
	Invert_Matrix(h_inverse_A_matrix, h_A_matrix, N);

    for (int row = 0; row < N; row++)
	{
		h_Parameter_Vector[row] = 0;
		
		for (int i = 0; i < N; i++)
		{
			h_Parameter_Vector[row] += h_inverse_A_matrix[i + row*N]*h_h_vector[i];
		}	
	}
}

void BROCCOLI_LIB::SetupDetrendingBasisFunctions()
{
	/*
	Matlab equivalent

	X_Detrend = zeros(st,4);
	X_Detrend(:,1) = ones(st,1);
	X_Detrend(:,2) = -(st-1)/2:(st-1)/2;	
	X_Detrend(:,3) = X_Detrend(:,2).^2;
	X_Detrend(:,4) = X_Detrend(:,2).^3;

	X_Detrend(:,1) = X_Detrend(:,1) / norm(X_Detrend(:,1));
	X_Detrend(:,2) = X_Detrend(:,2) / norm(X_Detrend(:,2));
	X_Detrend(:,3) = X_Detrend(:,3) / norm(X_Detrend(:,3));
	X_Detrend(:,4) = X_Detrend(:,4) / norm(X_Detrend(:,4));

	xtxxt_Detrend = inv(X_Detrend'*X_Detrend)*X_Detrend';
	*/

	// 1 and X
	float offset = -((float)DATA_T - 1.0f)/2.0f;
	for (int t = 0; t < DATA_T; t++)
	{
		h_X_Detrend[t + 0 * DATA_T] = 1.0f;
		h_X_Detrend[t + 1 * DATA_T] = offset + (float)t;
	}

	// X^2 and X^3
	for (int t = 0; t < DATA_T; t++)
	{
		h_X_Detrend[t + 2 * DATA_T] = h_X_Detrend[t + 1 * DATA_T] * h_X_Detrend[t + 1 * DATA_T];
		h_X_Detrend[t + 3 * DATA_T] = h_X_Detrend[t + 1 * DATA_T] * h_X_Detrend[t + 1 * DATA_T] * h_X_Detrend[t + 1 * DATA_T];
	}

	// Normalize

	// 1
	float norm = 0.0f;
	for (int t = 0; t < DATA_T; t++)
	{
		norm += h_X_Detrend[t + 0 * DATA_T] * h_X_Detrend[t + 0 * DATA_T];
	}
	norm = sqrt(norm);
	for (int t = 0; t < DATA_T; t++)
	{
		h_X_Detrend[t + 0 * DATA_T] /= norm;
	}

	// X
	norm = 0.0f;
	for (int t = 0; t < DATA_T; t++)
	{
		norm += h_X_Detrend[t + 1 * DATA_T] * h_X_Detrend[t + 1 * DATA_T];
	}
	norm = sqrt(norm);
	for (int t = 0; t < DATA_T; t++)
	{
		h_X_Detrend[t + 1 * DATA_T] /= norm;
	}

	// X^2
	norm = 0.0f;
	for (int t = 0; t < DATA_T; t++)
	{
		norm += h_X_Detrend[t + 2 * DATA_T] * h_X_Detrend[t + 2 * DATA_T];
	}
	norm = sqrt(norm);
	for (int t = 0; t < DATA_T; t++)
	{
		h_X_Detrend[t + 2 * DATA_T] /= norm;
	}

	// X^3
	norm = 0.0f;
	for (int t = 0; t < DATA_T; t++)
	{
		norm += h_X_Detrend[t + 3 * DATA_T] * h_X_Detrend[t + 3 * DATA_T];
	}
	norm = sqrt(norm);
	for (int t = 0; t < DATA_T; t++)
	{
		h_X_Detrend[t + 3 * DATA_T] /= norm;
	}

	// Calculate X_Detrend'*X_Detrend
	float xtx[16];
	float inv_xtx[16];

	for (int i = 0; i < NUMBER_OF_DETRENDING_BASIS_FUNCTIONS; i++)
	{
		for (int j = 0; j < NUMBER_OF_DETRENDING_BASIS_FUNCTIONS; j++)
		{
			xtx[i + j * NUMBER_OF_DETRENDING_BASIS_FUNCTIONS] = 0.0f;
			for (int t = 0; t < DATA_T; t++)
			{
				xtx[i + j * NUMBER_OF_DETRENDING_BASIS_FUNCTIONS] += h_X_Detrend[t + i * DATA_T] * h_X_Detrend[t + j * DATA_T];
			}
		}
	}

	// Calculate inverse of X_Detrend'*X_Detrend
	Invert_Matrix(inv_xtx, xtx, NUMBER_OF_DETRENDING_BASIS_FUNCTIONS);

	// Calculate inv(X_Detrend'*X_Detrend)*X_Detrend'
	for (int t = 0; t < DATA_T; t++)
	{
		h_xtxxt_Detrend[t + 0 * DATA_T] = inv_xtx[0] * h_X_Detrend[t + 0 * DATA_T] + inv_xtx[1] * h_X_Detrend[t + 1 * DATA_T] + inv_xtx[2] * h_X_Detrend[t + 2 * DATA_T] + inv_xtx[3] * h_X_Detrend[t + 3 * DATA_T];
		h_xtxxt_Detrend[t + 1 * DATA_T] = inv_xtx[4] * h_X_Detrend[t + 0 * DATA_T] + inv_xtx[5] * h_X_Detrend[t + 1 * DATA_T] + inv_xtx[6] * h_X_Detrend[t + 2 * DATA_T] + inv_xtx[7] * h_X_Detrend[t + 3 * DATA_T];
		h_xtxxt_Detrend[t + 2 * DATA_T] = inv_xtx[8] * h_X_Detrend[t + 0 * DATA_T] + inv_xtx[9] * h_X_Detrend[t + 1 * DATA_T] + inv_xtx[10] * h_X_Detrend[t + 2 * DATA_T] + inv_xtx[11] * h_X_Detrend[t + 3 * DATA_T];
		h_xtxxt_Detrend[t + 3 * DATA_T] = inv_xtx[12] * h_X_Detrend[t + 0 * DATA_T] + inv_xtx[13] * h_X_Detrend[t + 1 * DATA_T] + inv_xtx[14] * h_X_Detrend[t + 2 * DATA_T] + inv_xtx[15] * h_X_Detrend[t + 3 * DATA_T];
	}

	//cudaMemcpyToSymbol(c_X_Detrend, h_X_Detrend, sizeof(float) * NUMBER_OF_DETRENDING_BASIS_FUNCTIONS * DATA_T);
	//cudaMemcpyToSymbol(c_xtxxt_Detrend, h_xtxxt_Detrend, sizeof(float) * NUMBER_OF_DETRENDING_BASIS_FUNCTIONS * DATA_T);
}

void BROCCOLI_LIB::SegmentBrainData()
{
	
}

void BROCCOLI_LIB::CreateHRF()
{
	/*
	% p    - parameters of the response function (two gamma functions)
	%
	%							defaults
	%							(seconds)
	%	p(1) - delay of response (relative to onset)	   6
	%	p(2) - delay of undershoot (relative to onset)    16
	%	p(3) - dispersion of response			   1
	%	p(4) - dispersion of undershoot			   1
	%	p(5) - ratio of response to undershoot		   6
	%	p(6) - onset (seconds)				   0
	%	p(7) - length of kernel (seconds)		  32
	*/

	double p[7];
	p[0] = 6.0;
	p[1] = 16.0;
	p[2] = 1.0;
	p[3] = 1.0;
	p[4] = 6.0;
	p[5] = 0.0;
	p[6] = 32.0;
	double fMRI_T = 16.0;
	double dt = ((double)TR)/fMRI_T;

	int length = (int)(p[6]/dt);
	double* highres_hrf = (double*)malloc(sizeof(double) * length);

	for (int i = 0; i < length; i++)
	{
		highres_hrf[i] = (double)i - p[5]/dt;
	}

	for (int i = 0; i < length; i++)
	{
		highres_hrf[i] = Gpdf(highres_hrf[i],p[0]/p[2],dt/p[2]) - 1.0/p[4] * Gpdf(highres_hrf[i],p[1]/p[3],dt/p[3]);
	}




	// Downsample the hrf
	int downsample_factor = 16;
	hrf_length = length/downsample_factor + 1;
	//hrf_length = 17;

	std::cout << "length is " << length << " and hrf length is " << hrf_length << std::endl;

	/*
	for (int i = 0; i < hrf_length; i++)
	{
		std::cout << "Loggamma of " << i << " is " << loggamma(i) << std::endl;
		//std::cout << "Gpdf of " << i << " is " << Gpdf((float)i,p[0]/p[2],dt/p[2]) << std::endl;
	}
	*/
	

	hrf = (float*)malloc(sizeof(float) * hrf_length);

	for (int i = 0; i < hrf_length; i++)
	{
		if ((i * downsample_factor) < length)
		{
			hrf[i] = (float)highres_hrf[i*downsample_factor];
		}
		else
		{
			hrf[i] = 0.0f;
		}
	}

	float sum = 0.0f;
	for (int i = 0; i < hrf_length; i++)
	{
		sum += hrf[i];
	}
	for (int i = 0; i < hrf_length; i++)
	{
		hrf[i] /= sum;
	}

	WriteRealDataDouble(highres_hrf, "highres_hrf.raw", length);
	WriteRealDataFloat(hrf, "hrf.raw", hrf_length);


	free(highres_hrf);

	/*
	for (int i = 0; i < hrf_length; i++)
	{
		std::cout << "hrf coefficient " << i << " is " << hrf[i] << std::endl;
	}
	*/
}

void BROCCOLI_LIB::ConvolveWithHRF(float* temp_GLM)
{
	
}

void BROCCOLI_LIB::SetupStatisticalAnalysisBasisFunctions()
{
	
}

float BROCCOLI_LIB::CalculateMax(float *data, int N)
{
    float max = -1000000.0f;
	for (int i = 0; i < N; i++)
	{
	    if (data[i] > max)
		{
			max = data[i];
		}
	}
	return max;
}

float BROCCOLI_LIB::CalculateMin(float *data, int N)
{
    float min = 1000000.0f;
	for (int i = 0; i < N; i++)
	{
	    if (data[i] < min)
		{
			min = data[i];
		}
	}
	return min;
}

float BROCCOLI_LIB::loggamma(int value)
{
	int product = 1;
	for (int i = 1; i < value; i++)
	{
		product *= i;
	}
	return log((double)product);
}

float BROCCOLI_LIB::Gpdf(double value, double shape, double scale)
{
	//return pow(value, shape - scale) * exp(-value / scale) / (pow(scale,shape) * gamma((int)shape));

	return (exp( (shape - 1.0) * log(value) + shape * log(scale) - scale * value - loggamma(shape) ));
}
