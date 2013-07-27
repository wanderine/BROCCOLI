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
        
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
    DEALINGS IN THE SOFTWARE.
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
#include <math.h>

#include <opencl.h>

//#include <shrUtils.h>
//#include <shrQATest.h>
#include "broccoli_lib.h"

#include "nifti1.h"
#include "nifti1_io.h"

#include <cstdlib>




// public
float round( float d )
{
    return floor( d + 0.5f );
}

// Constructors

BROCCOLI_LIB::BROCCOLI_LIB()
{
	SetStartValues();
	OPENCL_INITIATED = 0;
	ResetAllPointers();
}

BROCCOLI_LIB::BROCCOLI_LIB(cl_uint platform)
{
	OpenCLInitiate(platform);
	OPENCL_INITIATED = 1;
	SetStartValues();
	ResetAllPointers();
	//AllocateMemory();
	//ReadImageRegistrationFilters();
	//ReadSmoothingFilters();	
	//SetGlobalAndLocalWorkSizes();
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

	EPI_DATA_W = 64;
	EPI_DATA_H = 64;
	EPI_DATA_D = 22;
	EPI_DATA_T = 79;

	EPI_VOXEL_SIZE_X = 3.75f;
	EPI_VOXEL_SIZE_Y = 3.75f;
	EPI_VOXEL_SIZE_Z = 3.75f;
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

	error = 0;
	kernel_error = 0;

	NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS = 12;

	convolution_time = 0.0;

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
	/*
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
	*/
}	

void BROCCOLI_LIB::GetOpenCLInfo()
{
	std::string temp_string; std::ostringstream temp_stream;
	char* value;
	size_t valueSize, valueSizes[3];
	cl_uint maxComputeUnits, clockFrequency;
	cl_ulong memorySize;
	
	// Get platforms
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, NULL, &platformIdCount);
	std::vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs (platformIdCount, platformIds.data (), NULL);

	// Loop over platforms
	for (uint i = 0; i < platformIdCount; i++) 
    {
	    // Get platform vendor
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, 0, NULL, &valueSize);
		value = (char*) malloc(valueSize);
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, valueSize, value, NULL);            
		device_info.append("Platform vendor: ");
		device_info.append(value);
		device_info.append("\n");
		free(value);		

		// Get platform name
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, 0, NULL, &valueSize);
		value = (char*) malloc(valueSize);
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, valueSize, value, NULL);            
		device_info.append("Platform name: ");
		device_info.append(value);
		device_info.append("\n");
		free(value);		

		// Get platform extensions
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &valueSize);
		value = (char*) malloc(valueSize);
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_EXTENSIONS, valueSize, value, NULL);            
		device_info.append("Platform extentions: ");
		device_info.append(value);
		device_info.append("\n");
		free(value);		

		// Get platform profile
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_PROFILE, 0, NULL, &valueSize);
		value = (char*) malloc(valueSize);
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_PROFILE, valueSize, value, NULL);            
		device_info.append("Platform profile: ");
		device_info.append(value);
		device_info.append("\n");
		device_info.append("\n");
		free(value);		

		// Get devices for each platform
		cl_uint deviceIdCount = 0;
		clGetDeviceIDs (platformIds[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceIdCount);
		std::vector<cl_device_id> deviceIds (deviceIdCount);
		clGetDeviceIDs (platformIds[i], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), NULL);

		// Get information for for each device and save as a long string
		for (uint j = 0; j < deviceIdCount; j++) 
		{
	        // Get vendor name
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_VENDOR, 0, NULL, &valueSize);
			value = (char*) malloc(valueSize);
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_VENDOR, valueSize, value, NULL);            
			device_info.append("Device vendor: ");
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
	}
}

// Add compilation from binary
// Add to select CPU or GPU

void BROCCOLI_LIB::OpenCLInitiate(cl_uint OPENCL_PLATFORM)
{
	char* value;
	size_t valueSize;

  	// Get platforms
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, NULL, &platformIdCount);
	std::vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs (platformIdCount, platformIds.data(), NULL);              
	
	// Create context
	const cl_context_properties contextProperties [] =
	{
	    CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[OPENCL_PLATFORM]), 0, 0
	};

	// Get devices for current platform
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs (platformIds[OPENCL_PLATFORM], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceIdCount);
	std::vector<cl_device_id> deviceIds (deviceIdCount);
	clGetDeviceIDs (platformIds[OPENCL_PLATFORM], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), NULL);

	context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), NULL, NULL, &error);	
	error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &valueSize);
	cl_device_id *clDevices = (cl_device_id *) malloc(valueSize);
	error = clGetContextInfo(context, CL_CONTEXT_DEVICES, valueSize, clDevices, NULL);

	// Create a command queue
	commandQueue = clCreateCommandQueue(context, deviceIds[0], CL_QUEUE_PROFILING_ENABLE, &error);

	// Read the kernel code from file
	std::fstream kernelFile("broccoli_lib_kernel.cpp",std::ios::in);
	std::ostringstream oss;
	oss << kernelFile.rdbuf();
	std::string src = oss.str();
	const char *srcstr = src.c_str();

	// Create a program and build the code
	program = clCreateProgramWithSource(context, 1, (const char**)&srcstr , NULL, &error);
	clBuildProgram(program, deviceIdCount, deviceIds.data(), NULL, NULL, NULL);

	// Get build info        
    clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &valueSize);        
    value = (char*)malloc(valueSize);
    clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, valueSize, value, NULL);
	build_info.append(value);
	free(value);

	// Create kernels

	SeparableConvolutionRowsKernel = clCreateKernel(program,"SeparableConvolutionRows",&createKernelErrorSeparableConvolutionRows);	
	SeparableConvolutionColumnsKernel = clCreateKernel(program,"SeparableConvolutionColumns",&createKernelErrorSeparableConvolutionColumns);
	SeparableConvolutionRodsKernel = clCreateKernel(program,"SeparableConvolutionRods",&createKernelErrorSeparableConvolutionRods);	
	
	// Kernels for image registration	
	MemsetKernel = clCreateKernel(program,"Memset",&createKernelErrorMemset);
	NonseparableConvolution3DComplexKernel = clCreateKernel(program,"Nonseparable3DConvolutionComplex",&createKernelErrorNonseparableConvolution3DComplex);
	CalculatePhaseDifferencesAndCertaintiesKernel = clCreateKernel(program,"CalculatePhaseDifferencesAndCertainties",&createKernelErrorCalculatePhaseDifferencesAndCertainties);
	CalculatePhaseGradientsXKernel = clCreateKernel(program,"CalculatePhaseGradientsX",&createKernelErrorCalculatePhaseGradientsX);
	CalculatePhaseGradientsYKernel = clCreateKernel(program,"CalculatePhaseGradientsY",&createKernelErrorCalculatePhaseGradientsY);
	CalculatePhaseGradientsZKernel = clCreateKernel(program,"CalculatePhaseGradientsZ",&createKernelErrorCalculatePhaseGradientsZ);
	CalculateAMatrixAndHVector2DValuesXKernel = clCreateKernel(program,"CalculateAMatrixAndHVector2DValuesX",&createKernelErrorCalculateAMatrixAndHVector2DValuesX);
	CalculateAMatrixAndHVector2DValuesYKernel = clCreateKernel(program,"CalculateAMatrixAndHVector2DValuesY",&createKernelErrorCalculateAMatrixAndHVector2DValuesY);
	CalculateAMatrixAndHVector2DValuesZKernel = clCreateKernel(program,"CalculateAMatrixAndHVector2DValuesZ",&createKernelErrorCalculateAMatrixAndHVector2DValuesZ);
	CalculateAMatrix1DValuesKernel = clCreateKernel(program,"CalculateAMatrix1DValues",&createKernelErrorCalculateAMatrix1DValues);
	CalculateHVector1DValuesKernel = clCreateKernel(program,"CalculateHVector1DValues",&createKernelErrorCalculateHVector1DValues);
	CalculateAMatrixKernel = clCreateKernel(program,"CalculateAMatrix",&createKernelErrorCalculateAMatrix);
	CalculateHVectorKernel = clCreateKernel(program,"CalculateHVector",&createKernelErrorCalculateHVector);	
	InterpolateVolumeNearestKernel = clCreateKernel(program,"InterpolateVolumeNearest",&createKernelErrorInterpolateVolumeNearest);       
	InterpolateVolumeLinearKernel = clCreateKernel(program,"InterpolateVolumeLinear",&createKernelErrorInterpolateVolumeLinear);       
	//InterpolateVolumeCubicKernel = clCreateKernel(program,"InterpolateVolumeCubic",&createKernelErrorInterpolateVolumeCubic);       
	RescaleVolumeLinearKernel = clCreateKernel(program,"RescaleVolumeLinear",&createKernelErrorRescaleVolumeLinear);
	//RescaleVolumeCubicKernel = clCreateKernel(program,"RescaleVolumeCubic",&createKernelErrorRescaleVolumeCubic);
	CopyT1VolumeToMNIKernel = clCreateKernel(program,"CopyT1VolumeToMNI",&createKernelErrorCopyT1VolumeToMNI);       
	CopyEPIVolumeToT1Kernel = clCreateKernel(program,"CopyEPIVolumeToT1",&createKernelErrorCopyEPIVolumeToT1);       
	CopyVolumeToNewKernel = clCreateKernel(program,"CopyVolumeToNew",&createKernelErrorCopyVolumeToNew);       
	MultiplyVolumesKernel = clCreateKernel(program,"MultiplyVolumes",&createKernelErrorMultiplyVolumes);       
	MultiplyVolumesOverwriteKernel = clCreateKernel(program,"MultiplyVolumesOverwrite",&createKernelErrorMultiplyVolumesOverwrite);       

	// Kernels for statistical analysis	
	CalculateBetaValuesGLMKernel = clCreateKernel(program,"CalculateBetaValuesGLM",&createKernelErrorCalculateBetaValuesGLM);
	CalculateStatisticalMapsGLMKernel = clCreateKernel(program,"CalculateStatisticalMapsGLM",&createKernelErrorCalculateStatisticalMapsGLM);
	

	//clFinish(commandQueue);
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesSeparableConvolution(int DATA_W, int DATA_H, int DATA_D)
{		
	//----------------------------------
	// Separable convolution rows
	//----------------------------------

	localWorkSizeSeparableConvolutionRows[0] = 32;
	localWorkSizeSeparableConvolutionRows[1] = 8;
	localWorkSizeSeparableConvolutionRows[2] = 2;

	// Calculate how many blocks are required
	// ConvolutionRows yields 32 * 8 * 8 valid filter responses per block (x,y,z)
	xBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_ROWS);
	yBlocks = (size_t)ceil((float)DATA_H / (float)VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_ROWS);
	zBlocks = (size_t)ceil((float)DATA_D / (float)VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_ROWS);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeSeparableConvolutionRows[0] = xBlocks * localWorkSizeSeparableConvolutionRows[0];
	globalWorkSizeSeparableConvolutionRows[1] = yBlocks * localWorkSizeSeparableConvolutionRows[1];
	globalWorkSizeSeparableConvolutionRows[2] = zBlocks * localWorkSizeSeparableConvolutionRows[2];

	//----------------------------------
	// Separable convolution columns
	//----------------------------------

	localWorkSizeSeparableConvolutionColumns[0] = 32;
	localWorkSizeSeparableConvolutionColumns[1] = 8;
	localWorkSizeSeparableConvolutionColumns[2] = 2;

    // Calculate how many blocks are required
	// ConvolutionColumns yields 24 * 16 * 8 valid filter responses per block (x,y,z)
	xBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_COLUMNS);
	yBlocks = (size_t)ceil((float)DATA_H / (float)VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_COLUMNS);
	zBlocks = (size_t)ceil((float)DATA_D / (float)VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_COLUMNS);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeSeparableConvolutionColumns[0] = xBlocks * localWorkSizeSeparableConvolutionColumns[0];
	globalWorkSizeSeparableConvolutionColumns[1] = yBlocks * localWorkSizeSeparableConvolutionColumns[1];
	globalWorkSizeSeparableConvolutionColumns[2] = zBlocks * localWorkSizeSeparableConvolutionColumns[2];

	//----------------------------------
	// Separable convolution rods
	//----------------------------------

	localWorkSizeSeparableConvolutionRods[0] = 32;
	localWorkSizeSeparableConvolutionRods[1] = 2;
	localWorkSizeSeparableConvolutionRods[2] = 8;

	// Calculate how many blocks are required
	// ConvolutionRods yields 32 * 8 * 8 valid filter responses per block (x,y,z)
	xBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_RODS);
	yBlocks = (size_t)ceil((float)DATA_H / (float)VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_RODS);
	zBlocks = (size_t)ceil((float)DATA_D / (float)VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_RODS);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeSeparableConvolutionRods[0] = xBlocks * localWorkSizeSeparableConvolutionRods[0];
	globalWorkSizeSeparableConvolutionRods[1] = yBlocks * localWorkSizeSeparableConvolutionRods[1];
	globalWorkSizeSeparableConvolutionRods[2] = zBlocks * localWorkSizeSeparableConvolutionRods[2];
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesMemset(int N)
{	
	localWorkSizeMemset[0] = 512;
	localWorkSizeMemset[1] = 1;
	localWorkSizeMemset[2] = 1;

	xBlocks = (size_t)ceil((float)(N) / (float)localWorkSizeMemset[0]);
	
	globalWorkSizeMemset[0] = xBlocks * localWorkSizeMemset[0];
	globalWorkSizeMemset[1] = 1;
	globalWorkSizeMemset[2] = 1;
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesNonSeparableConvolution(int DATA_W, int DATA_H, int DATA_D)
{	
	//----------------------------------
	// Non-separable convolution
	//----------------------------------

	localWorkSizeNonseparableConvolution3DComplex[0] = 32;
	localWorkSizeNonseparableConvolution3DComplex[1] = 32;
	localWorkSizeNonseparableConvolution3DComplex[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_X_CONVOLUTION_2D);
	yBlocks = (size_t)ceil((float)DATA_H / (float)VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeNonseparableConvolution3DComplex[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeNonseparableConvolution3DComplex[0] = xBlocks * localWorkSizeNonseparableConvolution3DComplex[0];
	globalWorkSizeNonseparableConvolution3DComplex[1] = yBlocks * localWorkSizeNonseparableConvolution3DComplex[1];
	globalWorkSizeNonseparableConvolution3DComplex[2] = zBlocks * localWorkSizeNonseparableConvolution3DComplex[2];
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesImageRegistration(int DATA_W, int DATA_H, int DATA_D)
{	
	//----------------------------------
	// Phase differences and certainties
	//----------------------------------

	localWorkSizeCalculatePhaseDifferencesAndCertainties[0] = 32;
	localWorkSizeCalculatePhaseDifferencesAndCertainties[1] = 16;
	localWorkSizeCalculatePhaseDifferencesAndCertainties[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculatePhaseDifferencesAndCertainties[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculatePhaseDifferencesAndCertainties[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculatePhaseDifferencesAndCertainties[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculatePhaseDifferencesAndCertainties[0] = xBlocks * localWorkSizeCalculatePhaseDifferencesAndCertainties[0];
	globalWorkSizeCalculatePhaseDifferencesAndCertainties[1] = yBlocks * localWorkSizeCalculatePhaseDifferencesAndCertainties[1];
	globalWorkSizeCalculatePhaseDifferencesAndCertainties[2] = zBlocks * localWorkSizeCalculatePhaseDifferencesAndCertainties[2];

	//----------------------------------
	// Phase gradients
	//----------------------------------

	localWorkSizeCalculatePhaseGradients[0] = 32;
	localWorkSizeCalculatePhaseGradients[1] = 16;
	localWorkSizeCalculatePhaseGradients[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculatePhaseGradients[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculatePhaseGradients[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculatePhaseGradients[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculatePhaseGradients[0] = xBlocks * localWorkSizeCalculatePhaseGradients[0];
	globalWorkSizeCalculatePhaseGradients[1] = yBlocks * localWorkSizeCalculatePhaseGradients[1];
	globalWorkSizeCalculatePhaseGradients[2] = zBlocks * localWorkSizeCalculatePhaseGradients[2];

	//----------------------------------
	// A-matrix and h-vector
	//----------------------------------

	localWorkSizeCalculateAMatrixAndHVector2DValuesX[0] = DATA_H;
	localWorkSizeCalculateAMatrixAndHVector2DValuesX[1] = 1;
	localWorkSizeCalculateAMatrixAndHVector2DValuesX[2] = 1;

	globalWorkSizeCalculateAMatrixAndHVector2DValuesX[0] = DATA_H;
	globalWorkSizeCalculateAMatrixAndHVector2DValuesX[1] = DATA_D;
	globalWorkSizeCalculateAMatrixAndHVector2DValuesX[2] = 1;

	localWorkSizeCalculateAMatrixAndHVector2DValuesY[0] = DATA_H;
	localWorkSizeCalculateAMatrixAndHVector2DValuesY[1] = 1;
	localWorkSizeCalculateAMatrixAndHVector2DValuesY[2] = 1;
	
	globalWorkSizeCalculateAMatrixAndHVector2DValuesY[0] = DATA_H;
	globalWorkSizeCalculateAMatrixAndHVector2DValuesY[1] = DATA_D;
	globalWorkSizeCalculateAMatrixAndHVector2DValuesY[2] = 1;

	localWorkSizeCalculateAMatrixAndHVector2DValuesZ[0] = DATA_H;
	localWorkSizeCalculateAMatrixAndHVector2DValuesZ[1] = 1;
	localWorkSizeCalculateAMatrixAndHVector2DValuesZ[2] = 1;
	
	globalWorkSizeCalculateAMatrixAndHVector2DValuesZ[0] = DATA_H;
	globalWorkSizeCalculateAMatrixAndHVector2DValuesZ[1] = DATA_D;
	globalWorkSizeCalculateAMatrixAndHVector2DValuesZ[2] = 1;

	localWorkSizeCalculateAMatrix1DValues[0] = DATA_D;
	localWorkSizeCalculateAMatrix1DValues[1] = 1;
	localWorkSizeCalculateAMatrix1DValues[2] = 1;
	
	globalWorkSizeCalculateAMatrix1DValues[0] = DATA_D;
	globalWorkSizeCalculateAMatrix1DValues[1] = NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS;
	globalWorkSizeCalculateAMatrix1DValues[2] = 1;

	localWorkSizeCalculateHVector1DValues[0] = DATA_D;
	localWorkSizeCalculateHVector1DValues[1] = 1;
	localWorkSizeCalculateHVector1DValues[2] = 1;

	globalWorkSizeCalculateHVector1DValues[0] = DATA_D;
	globalWorkSizeCalculateHVector1DValues[1] = NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS;
	globalWorkSizeCalculateHVector1DValues[2] = 1;

	localWorkSizeCalculateAMatrix[0] = NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS;
	localWorkSizeCalculateAMatrix[1] = 1;
	localWorkSizeCalculateAMatrix[2] = 1;

	globalWorkSizeCalculateAMatrix[0] = NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS;
	globalWorkSizeCalculateAMatrix[1] = 1;
	globalWorkSizeCalculateAMatrix[2] = 1;

	localWorkSizeCalculateHVector[0] = NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS;
	localWorkSizeCalculateHVector[1] = 1;
	localWorkSizeCalculateHVector[2] = 1;

	globalWorkSizeCalculateHVector[0] = NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS;
	globalWorkSizeCalculateHVector[1] = 1;
	globalWorkSizeCalculateHVector[2] = 1;

	SetGlobalAndLocalWorkSizesInterpolateVolume(DATA_W, DATA_H, DATA_D);
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesRescaleVolume(int DATA_W, int DATA_H, int DATA_D)
{
	localWorkSizeRescaleVolumeLinear[0] = 32;
	localWorkSizeRescaleVolumeLinear[1] = 16;
	localWorkSizeRescaleVolumeLinear[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeRescaleVolumeLinear[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeRescaleVolumeLinear[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeRescaleVolumeLinear[2]);

	globalWorkSizeRescaleVolumeLinear[0] = xBlocks * localWorkSizeRescaleVolumeLinear[0];
	globalWorkSizeRescaleVolumeLinear[1] = yBlocks * localWorkSizeRescaleVolumeLinear[1];
	globalWorkSizeRescaleVolumeLinear[2] = zBlocks * localWorkSizeRescaleVolumeLinear[2];

	localWorkSizeRescaleVolumeCubic[0] = 32;
	localWorkSizeRescaleVolumeCubic[1] = 16;
	localWorkSizeRescaleVolumeCubic[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeRescaleVolumeCubic[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeRescaleVolumeCubic[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeRescaleVolumeCubic[2]);

	globalWorkSizeRescaleVolumeCubic[0] = xBlocks * localWorkSizeRescaleVolumeCubic[0];
	globalWorkSizeRescaleVolumeCubic[1] = yBlocks * localWorkSizeRescaleVolumeCubic[1];
	globalWorkSizeRescaleVolumeCubic[2] = zBlocks * localWorkSizeRescaleVolumeCubic[2];
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesInterpolateVolume(int DATA_W, int DATA_H, int DATA_D)
{
	localWorkSizeInterpolateVolumeNearest[0] = 32;
	localWorkSizeInterpolateVolumeNearest[1] = 16;
	localWorkSizeInterpolateVolumeNearest[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeInterpolateVolumeNearest[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeInterpolateVolumeNearest[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeInterpolateVolumeNearest[2]);

	globalWorkSizeInterpolateVolumeNearest[0] = xBlocks * localWorkSizeInterpolateVolumeNearest[0];
	globalWorkSizeInterpolateVolumeNearest[1] = yBlocks * localWorkSizeInterpolateVolumeNearest[1];
	globalWorkSizeInterpolateVolumeNearest[2] = zBlocks * localWorkSizeInterpolateVolumeNearest[2];

	localWorkSizeInterpolateVolumeLinear[0] = 32;
	localWorkSizeInterpolateVolumeLinear[1] = 16;
	localWorkSizeInterpolateVolumeLinear[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeInterpolateVolumeLinear[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeInterpolateVolumeLinear[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeInterpolateVolumeLinear[2]);

	globalWorkSizeInterpolateVolumeLinear[0] = xBlocks * localWorkSizeInterpolateVolumeLinear[0];
	globalWorkSizeInterpolateVolumeLinear[1] = yBlocks * localWorkSizeInterpolateVolumeLinear[1];
	globalWorkSizeInterpolateVolumeLinear[2] = zBlocks * localWorkSizeInterpolateVolumeLinear[2];

	localWorkSizeInterpolateVolumeCubic[0] = 32;
	localWorkSizeInterpolateVolumeCubic[1] = 16;
	localWorkSizeInterpolateVolumeCubic[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeInterpolateVolumeCubic[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeInterpolateVolumeCubic[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeInterpolateVolumeCubic[2]);

	globalWorkSizeInterpolateVolumeCubic[0] = xBlocks * localWorkSizeInterpolateVolumeCubic[0];
	globalWorkSizeInterpolateVolumeCubic[1] = yBlocks * localWorkSizeInterpolateVolumeCubic[1];
	globalWorkSizeInterpolateVolumeCubic[2] = zBlocks * localWorkSizeInterpolateVolumeCubic[2];
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesCopyVolumeToNew(int DATA_W, int DATA_H, int DATA_D)
{
	localWorkSizeCopyVolumeToNew[0] = 32;
	localWorkSizeCopyVolumeToNew[1] = 16;
	localWorkSizeCopyVolumeToNew[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCopyVolumeToNew[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCopyVolumeToNew[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCopyVolumeToNew[2]);

	globalWorkSizeCopyVolumeToNew[0] = xBlocks * localWorkSizeCopyVolumeToNew[0];
	globalWorkSizeCopyVolumeToNew[1] = yBlocks * localWorkSizeCopyVolumeToNew[1];
	globalWorkSizeCopyVolumeToNew[2] = zBlocks * localWorkSizeCopyVolumeToNew[2];
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesMultiplyVolumes(int DATA_W, int DATA_H, int DATA_D)
{	
	//----------------------------------
	// Statistical calculations
	//----------------------------------

	localWorkSizeMultiplyVolumes[0] = 32;
	localWorkSizeMultiplyVolumes[1] = 16;
	localWorkSizeMultiplyVolumes[2] = 1;
	
	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeMultiplyVolumes[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeMultiplyVolumes[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeMultiplyVolumes[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeMultiplyVolumes[0] = xBlocks * localWorkSizeMultiplyVolumes[0];
	globalWorkSizeMultiplyVolumes[1] = yBlocks * localWorkSizeMultiplyVolumes[1];
	globalWorkSizeMultiplyVolumes[2] = zBlocks * localWorkSizeMultiplyVolumes[2];
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesStatisticalCalculations(int DATA_W, int DATA_H, int DATA_D)
{	
	//----------------------------------
	// Statistical calculations
	//----------------------------------

	localWorkSizeCalculateBetaValuesGLM[0] = 32;
	localWorkSizeCalculateBetaValuesGLM[1] = 16;
	localWorkSizeCalculateBetaValuesGLM[2] = 1;
	
	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculateBetaValuesGLM[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculateBetaValuesGLM[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateBetaValuesGLM[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculateBetaValuesGLM[0] = xBlocks * localWorkSizeCalculateBetaValuesGLM[0];
	globalWorkSizeCalculateBetaValuesGLM[1] = yBlocks * localWorkSizeCalculateBetaValuesGLM[1];
	globalWorkSizeCalculateBetaValuesGLM[2] = zBlocks * localWorkSizeCalculateBetaValuesGLM[2];

	localWorkSizeCalculateStatisticalMapsGLM[0] = 32;
	localWorkSizeCalculateStatisticalMapsGLM[1] = 16;
	localWorkSizeCalculateStatisticalMapsGLM[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculateStatisticalMapsGLM[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculateStatisticalMapsGLM[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateStatisticalMapsGLM[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculateStatisticalMapsGLM[0] = xBlocks * localWorkSizeCalculateStatisticalMapsGLM[0];
	globalWorkSizeCalculateStatisticalMapsGLM[1] = yBlocks * localWorkSizeCalculateStatisticalMapsGLM[1];
	globalWorkSizeCalculateStatisticalMapsGLM[2] = zBlocks * localWorkSizeCalculateStatisticalMapsGLM[2];
}

void BROCCOLI_LIB::OpenCLCleanup()
{
	if (OPENCL_INITIATED == 1)
	{
	    clReleaseKernel(SeparableConvolutionRowsKernel);
		clReleaseKernel(SeparableConvolutionColumnsKernel);
		clReleaseKernel(SeparableConvolutionRodsKernel);
		clReleaseKernel(MemsetKernel);
		clReleaseKernel(NonseparableConvolution3DComplexKernel);
		clReleaseKernel(CalculatePhaseDifferencesAndCertaintiesKernel);
		clReleaseKernel(CalculatePhaseGradientsXKernel);
		clReleaseKernel(CalculatePhaseGradientsYKernel);
		clReleaseKernel(CalculatePhaseGradientsZKernel);
		clReleaseKernel(CalculateAMatrixAndHVector2DValuesXKernel);
		clReleaseKernel(CalculateAMatrixAndHVector2DValuesYKernel);
		clReleaseKernel(CalculateAMatrixAndHVector2DValuesZKernel);
		clReleaseKernel(CalculateAMatrix1DValuesKernel);
		clReleaseKernel(CalculateHVector1DValuesKernel);
		clReleaseKernel(CalculateAMatrixKernel);
		clReleaseKernel(CalculateHVectorKernel);
		clReleaseKernel(InterpolateVolumeNearestKernel);
		clReleaseKernel(InterpolateVolumeLinearKernel);
		//clReleaseKernel(InterpolateVolumeCubicKernel);
		//clReleaseKernel(RescaleVolumeNearestKernel);
		clReleaseKernel(RescaleVolumeLinearKernel);
		//clReleaseKernel(RescaleVolumeCubicKernel);
		clReleaseKernel(CopyT1VolumeToMNIKernel);
		clReleaseKernel(CopyEPIVolumeToT1Kernel);
		clReleaseKernel(CopyVolumeToNewKernel);
		clReleaseKernel(CalculateBetaValuesGLMKernel);
		clReleaseKernel(CalculateStatisticalMapsGLMKernel);
		clReleaseProgram(program);    
		clReleaseCommandQueue(commandQueue);
		clReleaseContext(context);
	}
}







// Set functions for GUI / Wrappers


void BROCCOLI_LIB::SetInputfMRIVolumes(float* data)
{
	h_fMRI_Volumes = data;
}

void BROCCOLI_LIB::SetInputEPIVolume(float* data)
{
	h_EPI_Volume = data;
}

void BROCCOLI_LIB::SetInputT1Volume(float* data)
{
	h_T1_Volume = data;
}


void BROCCOLI_LIB::SetInputMNIVolume(float* data)
{
	h_MNI_Volume = data;
}

void BROCCOLI_LIB::SetInputMNIBrainMask(float* data)
{
	h_MNI_Brain_Mask = data;
}

void BROCCOLI_LIB::SetMask(float* data)
{
	h_Mask = data;
}

void BROCCOLI_LIB::SetNumberOfRegressors(int N)
{
	NUMBER_OF_REGRESSORS = N;
}

void BROCCOLI_LIB::SetNumberOfContrasts(int N)
{
	NUMBER_OF_CONTRASTS = N;
}

void BROCCOLI_LIB::SetDesignMatrix(float* data1, float* data2)
{
	h_X_GLM = data1;
	h_xtxxt_GLM = data2;
}

void BROCCOLI_LIB::SetGLMScalars(float* data)
{
	h_ctxtxc_GLM = data;
}

void BROCCOLI_LIB::SetContrasts(float* data)
{
	h_Contrasts = data;
}

void BROCCOLI_LIB::SetSmoothingFilters(float* Smoothing_Filter_X, float* Smoothing_Filter_Y, float* Smoothing_Filter_Z)
{
	h_Smoothing_Filter_X = Smoothing_Filter_X;
	h_Smoothing_Filter_Y = Smoothing_Filter_Y;
	h_Smoothing_Filter_Z = Smoothing_Filter_Z;
}

void BROCCOLI_LIB::SetImageRegistrationFilterSize(int N) 
{
	IMAGE_REGISTRATION_FILTER_SIZE = N;
}

void BROCCOLI_LIB::SetImageRegistrationFilters(float* qf1r, float* qf1i, float* qf2r, float* qf2i, float* qf3r, float* qf3i)    
{
	h_Quadrature_Filter_1_Real = qf1r;
	h_Quadrature_Filter_1_Imag = qf1i;
	h_Quadrature_Filter_2_Real = qf2r;
	h_Quadrature_Filter_2_Imag = qf2i;
	h_Quadrature_Filter_3_Real = qf3r;
	h_Quadrature_Filter_3_Imag = qf3i;	
}

void BROCCOLI_LIB::SetNumberOfIterationsForImageRegistration(int N)
{
	NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION = N;
}

void BROCCOLI_LIB::SetNumberOfIterationsForMotionCorrection(int N)
{
	NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION = N;
}

void BROCCOLI_LIB::SetCoarsestScaleT1MNI(int N)
{
	COARSEST_SCALE_T1_MNI = N;
}

void BROCCOLI_LIB::SetCoarsestScaleEPIT1(int N)
{
	COARSEST_SCALE_EPI_T1 = N;
}

void BROCCOLI_LIB::SetMMT1ZCUT(int mm)
{
	MM_T1_Z_CUT = mm;
}

void BROCCOLI_LIB::SetMMEPIZCUT(int mm)
{
	MM_EPI_Z_CUT = mm;
}

void BROCCOLI_LIB::SetOutputBetaVolumes(float* data)
{
	h_Beta_Volumes = data;
}

void BROCCOLI_LIB::SetOutputResiduals(float* data)
{
	h_Residuals = data;
}

void BROCCOLI_LIB::SetOutputResidualVariances(float* data)
{
	h_Residual_Variances = data;
}

void BROCCOLI_LIB::SetOutputStatisticalMaps(float* data)
{
	h_Statistical_Maps = data;
}

void BROCCOLI_LIB::SetOutputMotionParameters(float* output)
{
	h_Motion_Parameters_Out = output;
}

void BROCCOLI_LIB::SetOutputT1MNIRegistrationParameters(float* output)
{
	h_Registration_Parameters_T1_MNI_Out = output;
}

void BROCCOLI_LIB::SetOutputEPIT1RegistrationParameters(float* output)
{
	h_Registration_Parameters_EPI_T1_Out = output;
}

void BROCCOLI_LIB::SetOutputQuadratureFilterResponses(float* qfr1r, float* qfr1i, float* qfr2r, float* qfr2i, float* qfr3r, float* qfr3i)
{
	h_Quadrature_Filter_Response_1_Real = qfr1r;
	h_Quadrature_Filter_Response_1_Imag = qfr1i;
	h_Quadrature_Filter_Response_2_Real = qfr2r;
	h_Quadrature_Filter_Response_2_Imag = qfr2i;
	h_Quadrature_Filter_Response_3_Real = qfr3r;
	h_Quadrature_Filter_Response_3_Imag = qfr3i;	
}

void BROCCOLI_LIB::SetOutputPhaseDifferences(float* pd)
{
	h_Phase_Differences = pd;
}

void BROCCOLI_LIB::SetOutputPhaseCertainties(float* pc)
{
	h_Phase_Certainties = pc;
}

void BROCCOLI_LIB::SetOutputPhaseGradients(float* pg)
{
	h_Phase_Gradients = pg;
}

void BROCCOLI_LIB::SetOutputAlignedT1Volume(float* aligned)
{
	h_Aligned_T1_Volume = aligned;
}

void BROCCOLI_LIB::SetOutputAlignedEPIVolume(float* aligned)
{
	h_Aligned_EPI_Volume = aligned;
}


void BROCCOLI_LIB::SetOutputSkullstrippedT1Volume(float* skullstripped)
{
	h_Skullstripped_T1_Volume = skullstripped;
}

void BROCCOLI_LIB::SetOutputInterpolatedT1Volume(float* interpolated)
{
	h_Interpolated_T1_Volume = interpolated;
}

void BROCCOLI_LIB::SetOutputInterpolatedEPIVolume(float* interpolated)
{
	h_Interpolated_EPI_Volume = interpolated;
}


void BROCCOLI_LIB::SetOutputMotionCorrectedfMRIVolumes(float* motion_corrected)
{
	h_Motion_Corrected_fMRI_Volumes = motion_corrected;
}

void BROCCOLI_LIB::SetOutputSmoothedfMRIVolumes(float* smoothed)
{
	h_Smoothed_fMRI_Volumes = smoothed;
}

void BROCCOLI_LIB::SetOutputDownsampledVolume(float* downsampled)
{
	h_Downsampled_Volume = downsampled;
}




void BROCCOLI_LIB::SetDataType(int type)
{
	DATA_TYPE = type;
}

void BROCCOLI_LIB::SetFileType(int type)
{
	FILE_TYPE = type;
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

void BROCCOLI_LIB::SetEPIWidth(int w)
{
	EPI_DATA_W = w;
}
			
void BROCCOLI_LIB::SetEPIHeight(int h)
{
	EPI_DATA_H = h;
}

void BROCCOLI_LIB::SetEPIDepth(int d)
{
	EPI_DATA_D = d;
}

void BROCCOLI_LIB::SetEPITimepoints(int t)
{
	EPI_DATA_T = t;
}

void BROCCOLI_LIB::SetT1Width(int w)
{
	T1_DATA_W = w;
}
			
void BROCCOLI_LIB::SetT1Height(int h)
{
	T1_DATA_H = h;
}

void BROCCOLI_LIB::SetT1Depth(int d)
{
	T1_DATA_D = d;
}

void BROCCOLI_LIB::SetMNIWidth(int w)
{
	MNI_DATA_W = w;
}
			
void BROCCOLI_LIB::SetMNIHeight(int h)
{
	MNI_DATA_H = h;
}

void BROCCOLI_LIB::SetMNIDepth(int d)
{
	MNI_DATA_D = d;
}

void BROCCOLI_LIB::SetEPIVoxelSizeX(float value)
{
	EPI_VOXEL_SIZE_X = value;
}

void BROCCOLI_LIB::SetEPIVoxelSizeY(float value)
{
	EPI_VOXEL_SIZE_Y = value;
}

void BROCCOLI_LIB::SetEPIVoxelSizeZ(float value)
{
	EPI_VOXEL_SIZE_Z = value;
}

void BROCCOLI_LIB::SetEPITR(float value)
{
	TR = value;
}

void BROCCOLI_LIB::SetT1VoxelSizeX(float value)
{
	T1_VOXEL_SIZE_X = value;
}

void BROCCOLI_LIB::SetT1VoxelSizeY(float value)
{
	T1_VOXEL_SIZE_Y = value;
}

void BROCCOLI_LIB::SetT1VoxelSizeZ(float value)
{
	T1_VOXEL_SIZE_Z = value;
}

void BROCCOLI_LIB::SetMNIVoxelSizeX(float value)
{
	MNI_VOXEL_SIZE_X = value;
}

void BROCCOLI_LIB::SetMNIVoxelSizeY(float value)
{
	MNI_VOXEL_SIZE_Y = value;
}

void BROCCOLI_LIB::SetMNIVoxelSizeZ(float value)
{
	MNI_VOXEL_SIZE_Z = value;
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

const char* BROCCOLI_LIB::GetOpenCLDeviceInfoChar()
{
	return device_info.c_str();
}

const char* BROCCOLI_LIB::GetOpenCLBuildInfoChar()
{
	return build_info.c_str();
}

std::string BROCCOLI_LIB::GetOpenCLDeviceInfoString()
{
	return device_info;
}

std::string BROCCOLI_LIB::GetOpenCLBuildInfoString()
{
	return build_info;
}

int BROCCOLI_LIB::GetOpenCLError()
{
	return error;
}

int BROCCOLI_LIB::GetOpenCLCreateKernelError()
{
	//return createKernelErrorNonseparableConvolution3DComplex;
	//return createKernelErrorCalculatePhaseDifferencesAndCertainties;
	//return createKernelErrorCalculatePhaseGradientsX;
	//return createKernelErrorInterpolateVolumeLinear;
	return createKernelErrorRescaleVolumeLinear;
}

int* BROCCOLI_LIB::GetOpenCLCreateBufferErrors()
{
	OpenCLCreateBufferErrors[0] = createBufferErrorAlignedVolume;
	OpenCLCreateBufferErrors[1] = createBufferErrorReferenceVolume;
	OpenCLCreateBufferErrors[2] = createBufferErrorq11Real;
	OpenCLCreateBufferErrors[3] = createBufferErrorq11Imag; 
	OpenCLCreateBufferErrors[4] = createBufferErrorq12Real;
	OpenCLCreateBufferErrors[5] = createBufferErrorq12Imag;
	OpenCLCreateBufferErrors[6] = createBufferErrorq13Real;
	OpenCLCreateBufferErrors[7] = createBufferErrorq13Imag;
	OpenCLCreateBufferErrors[8] = createBufferErrorq21Real;
	OpenCLCreateBufferErrors[9] = createBufferErrorq21Imag;
	OpenCLCreateBufferErrors[10] = createBufferErrorq22Real;
	OpenCLCreateBufferErrors[11] = createBufferErrorq22Imag;
	OpenCLCreateBufferErrors[12] = createBufferErrorq23Real;
	OpenCLCreateBufferErrors[13] = createBufferErrorq23Imag;
	OpenCLCreateBufferErrors[14] = createBufferErrorPhaseDifferences;
	OpenCLCreateBufferErrors[15] = createBufferErrorPhaseCertainties;
	OpenCLCreateBufferErrors[16] = createBufferErrorPhaseGradients;
	OpenCLCreateBufferErrors[17] = createBufferErrorAMatrix;
	OpenCLCreateBufferErrors[18] = createBufferErrorHVector;
	OpenCLCreateBufferErrors[19] = createBufferErrorAMatrix2DValues;
	OpenCLCreateBufferErrors[20] = createBufferErrorAMatrix1DValues;
	OpenCLCreateBufferErrors[21] = createBufferErrorHVector2DValues;
	OpenCLCreateBufferErrors[22] = createBufferErrorHVector1DValues;
	OpenCLCreateBufferErrors[23] = createBufferErrorQuadratureFilter1Real;
	OpenCLCreateBufferErrors[24] = createBufferErrorQuadratureFilter1Imag;
	OpenCLCreateBufferErrors[25] = createBufferErrorQuadratureFilter2Real;
	OpenCLCreateBufferErrors[26] = createBufferErrorQuadratureFilter2Imag;
	OpenCLCreateBufferErrors[27] = createBufferErrorQuadratureFilter3Real;
	OpenCLCreateBufferErrors[28] = createBufferErrorQuadratureFilter3Imag;   
	OpenCLCreateBufferErrors[29] = createBufferErrorRegistrationParameters;
	OpenCLCreateBufferErrors[30] = createBufferErrorBetaVolumesMNI;

    return OpenCLCreateBufferErrors;
}

int* BROCCOLI_LIB::GetOpenCLRunKernelErrors()
{
    OpenCLRunKernelErrors[0] = runKernelErrorNonseparableConvolution3DComplex;
	OpenCLRunKernelErrors[1] = runKernelErrorMemset;
	OpenCLRunKernelErrors[2] = runKernelErrorCalculatePhaseDifferencesAndCertainties;
	OpenCLRunKernelErrors[3] = runKernelErrorCalculatePhaseGradientsX;
	OpenCLRunKernelErrors[4] = runKernelErrorCalculatePhaseGradientsY;
	OpenCLRunKernelErrors[5] = runKernelErrorCalculatePhaseGradientsZ;
	OpenCLRunKernelErrors[6] = runKernelErrorCalculateAMatrixAndHVector2DValuesX;
	OpenCLRunKernelErrors[7] = runKernelErrorCalculateAMatrixAndHVector2DValuesY;
	OpenCLRunKernelErrors[8] = runKernelErrorCalculateAMatrixAndHVector2DValuesZ;
	OpenCLRunKernelErrors[9] = runKernelErrorCalculateAMatrix1DValues;
	OpenCLRunKernelErrors[10] = runKernelErrorCalculateHVector1DValues;
	OpenCLRunKernelErrors[11] = runKernelErrorCalculateAMatrix;
	OpenCLRunKernelErrors[12] = runKernelErrorCalculateHVector;
	OpenCLRunKernelErrors[13] = runKernelErrorInterpolateVolume;

	return OpenCLRunKernelErrors;
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
	//return processing_times[CONVOLVE];
	return convolution_time;
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
int BROCCOLI_LIB::GetEPIWidth()
{
	return EPI_DATA_W;
}

// Returns the height dimension (y) of the current fMRI dataset
int BROCCOLI_LIB::GetEPIHeight()
{
	return EPI_DATA_H;
}

// Returns the depth dimension (z) of the current fMRI dataset
int BROCCOLI_LIB::GetEPIDepth()
{
	return EPI_DATA_D;
}

// Returns the number of timepoints of the current fMRI dataset
int BROCCOLI_LIB::GetEPITimepoints()
{
	return EPI_DATA_T;
}

int BROCCOLI_LIB::GetT1Width()
{
	return T1_DATA_W;
}

int BROCCOLI_LIB::GetT1Height()
{
	return T1_DATA_H;
}

int BROCCOLI_LIB::GetT1Depth()
{
	return T1_DATA_D;
}


// Returns the voxel size (in mm) in the x direction
float BROCCOLI_LIB::GetEPIVoxelSizeX()
{
	return EPI_VOXEL_SIZE_X;
}

// Returns the voxel size (in mm) in the y direction
float BROCCOLI_LIB::GetEPIVoxelSizeY()
{
	return EPI_VOXEL_SIZE_Y;
}

// Returns the voxel size (in mm) in the z direction
float BROCCOLI_LIB::GetEPIVoxelSizeZ()
{
	return EPI_VOXEL_SIZE_Z;
}

// Returns the repetition time of the current fMRI dataset
float BROCCOLI_LIB::GetEPITR()
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
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		motion_corrected_curve[t] = (double)h_Motion_Corrected_fMRI_Volumes[X_SLICE_LOCATION_fMRI_DATA + Y_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W + Z_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
	}

	return motion_corrected_curve;
}

// Returns the timeseries of the smoothed data for the current mouse location in the GUI
double* BROCCOLI_LIB::GetSmoothedCurve()
{
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		smoothed_curve[t] = (double)h_Smoothed_fMRI_Volumes[X_SLICE_LOCATION_fMRI_DATA + Y_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W + Z_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
	}

	return smoothed_curve;
}

// Returns the timeseries of the detrended data for the current mouse location in the GUI
double* BROCCOLI_LIB::GetDetrendedCurve()
{
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		detrended_curve[t] = (double)h_Detrended_fMRI_Volumes[X_SLICE_LOCATION_fMRI_DATA + Y_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W + Z_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
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


// Copy a slice of the quadrature filters to constant memory
void BROCCOLI_LIB::Copy3DFiltersToConstantMemory(int z, int FILTER_SIZE)
{
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_1_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_1_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_1_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_1_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_2_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_2_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_2_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_2_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_3_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_3_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_3_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_3_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
}

void BROCCOLI_LIB::NonseparableConvolution3D(cl_mem d_q1_Real, cl_mem d_q1_Imag, cl_mem d_q2_Real, cl_mem d_q2_Imag, cl_mem d_q3_Real, cl_mem d_q3_Imag, cl_mem d_Volume, int DATA_W, int DATA_H, int DATA_D)
{	
	SetGlobalAndLocalWorkSizesNonSeparableConvolution(DATA_W, DATA_H, DATA_D);

	clSetKernelArg(NonseparableConvolution3DComplexKernel, 0, sizeof(cl_mem), &d_q1_Real);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 1, sizeof(cl_mem), &d_q1_Imag);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 2, sizeof(cl_mem), &d_q2_Real);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 3, sizeof(cl_mem), &d_q2_Imag);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 4, sizeof(cl_mem), &d_q3_Real);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 5, sizeof(cl_mem), &d_q3_Imag);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 6, sizeof(cl_mem), &d_Volume);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 14, sizeof(int), &DATA_W);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 15, sizeof(int), &DATA_H);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 16, sizeof(int), &DATA_D);

	// Reset filter responses
	SetMemory(d_q1_Real, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_q1_Imag, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_q2_Real, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_q2_Imag, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_q3_Real, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_q3_Imag, 0.0f, DATA_W * DATA_H * DATA_D);
	
	// Do 3D convolution by summing 2D convolutions
	int z_offset = -(IMAGE_REGISTRATION_FILTER_SIZE - 1)/2;
	for (int zz = IMAGE_REGISTRATION_FILTER_SIZE -1; zz >= 0; zz--)
	{
		Copy3DFiltersToConstantMemory(zz, IMAGE_REGISTRATION_FILTER_SIZE);

		clSetKernelArg(NonseparableConvolution3DComplexKernel, 13, sizeof(int), &z_offset);
		runKernelErrorNonseparableConvolution3DComplex = clEnqueueNDRangeKernel(commandQueue, NonseparableConvolution3DComplexKernel, 3, NULL, globalWorkSizeNonseparableConvolution3DComplex, localWorkSizeNonseparableConvolution3DComplex, 0, NULL, NULL);
		//error = clEnqueueNDRangeKernel(commandQueue, NonseparableConvolution3DComplexKernel, 3, NULL, globalWorkSizeNonseparableConvolution3DComplex, localWorkSizeNonseparableConvolution3DComplex, 0, NULL, &event);
		clFinish(commandQueue);
		z_offset++;

		//clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		//clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		//convolution_time += time_end - time_start;
	}	
}

void BROCCOLI_LIB::SetMemory(cl_mem memory, float value, int N)
{
	SetGlobalAndLocalWorkSizesMemset(N);
	clSetKernelArg(MemsetKernel, 0, sizeof(cl_mem), &memory);
	clSetKernelArg(MemsetKernel, 1, sizeof(float), &value);
	clSetKernelArg(MemsetKernel, 2, sizeof(int), &N);		
	runKernelErrorMemset = clEnqueueNDRangeKernel(commandQueue, MemsetKernel, 1, NULL, globalWorkSizeMemset, localWorkSizeMemset, 0, NULL, NULL);
	clFinish(commandQueue);
}

// This function is the foundation for all the image registration functions
void BROCCOLI_LIB::AlignTwoVolumes(float *h_Registration_Parameters_Align_Two_Volumes, float* h_Rotations, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_ITERATIONS, int ALIGNMENT_TYPE)
{
	// Calculate the filter responses for the reference volume (only needed once)	
	NonseparableConvolution3D(d_q11_Real, d_q11_Imag, d_q12_Real, d_q12_Imag, d_q13_Real, d_q13_Imag, d_Reference_Volume, DATA_W, DATA_H, DATA_D);

	// Reset the parameter vector
	for (int p = 0; p < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; p++)
	{
		h_Registration_Parameters_Align_Two_Volumes[p] = 0.0f;
		h_Registration_Parameters[p] = 0.0f;
	}
	
	// Run the registration algorithm for a number of iterations
	for (int it = 0; it < NUMBER_OF_ITERATIONS; it++)
	{
		NonseparableConvolution3D(d_q21_Real, d_q21_Imag, d_q22_Real, d_q22_Imag, d_q23_Real, d_q23_Imag, d_Aligned_Volume, DATA_W, DATA_H, DATA_D);
			
		// Calculate phase differences, certainties and phase gradients in the X direction
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 2, sizeof(cl_mem), &d_q11_Real);
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 3, sizeof(cl_mem), &d_q11_Imag);
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 4, sizeof(cl_mem), &d_q21_Real);
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 5, sizeof(cl_mem), &d_q21_Imag);
		runKernelErrorCalculatePhaseDifferencesAndCertainties = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesAndCertaintiesKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);
		clFinish(commandQueue);
	
		
		runKernelErrorCalculatePhaseGradientsX = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseGradientsXKernel, 3, NULL, globalWorkSizeCalculatePhaseGradients, localWorkSizeCalculatePhaseGradients, 0, NULL, NULL);
		clFinish(commandQueue);
		
		
		// Calculate values for the A-matrix and h-vector in the X direction
		runKernelErrorCalculateAMatrixAndHVector2DValuesX = clEnqueueNDRangeKernel(commandQueue, CalculateAMatrixAndHVector2DValuesXKernel, 3, NULL, globalWorkSizeCalculateAMatrixAndHVector2DValuesX, localWorkSizeCalculateAMatrixAndHVector2DValuesX, 0, NULL, NULL);
		clFinish(commandQueue);

		
		// Calculate phase differences, certainties and phase gradients in the Y direction
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 2, sizeof(cl_mem), &d_q12_Real);
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 3, sizeof(cl_mem), &d_q12_Imag);
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 4, sizeof(cl_mem), &d_q22_Real);
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 5, sizeof(cl_mem), &d_q22_Imag);
		runKernelErrorCalculatePhaseDifferencesAndCertainties = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesAndCertaintiesKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);
		clFinish(commandQueue);
		
		
		runKernelErrorCalculatePhaseGradientsY = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseGradientsYKernel, 3, NULL, globalWorkSizeCalculatePhaseGradients, localWorkSizeCalculatePhaseGradients, 0, NULL, NULL);
		clFinish(commandQueue);

		
		// Calculate values for the A-matrix and h-vector in the Y direction
		runKernelErrorCalculateAMatrixAndHVector2DValuesY = clEnqueueNDRangeKernel(commandQueue, CalculateAMatrixAndHVector2DValuesYKernel, 3, NULL, globalWorkSizeCalculateAMatrixAndHVector2DValuesY, localWorkSizeCalculateAMatrixAndHVector2DValuesY, 0, NULL, NULL);
		clFinish(commandQueue);
		
		// Calculate phase differences, certainties and phase gradients in the Z direction
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 2, sizeof(cl_mem), &d_q13_Real);
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 3, sizeof(cl_mem), &d_q13_Imag);
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 4, sizeof(cl_mem), &d_q23_Real);
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 5, sizeof(cl_mem), &d_q23_Imag);		
		runKernelErrorCalculatePhaseDifferencesAndCertainties = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesAndCertaintiesKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);
		clFinish(commandQueue);
		
		
		runKernelErrorCalculatePhaseGradientsZ = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseGradientsZKernel, 3, NULL, globalWorkSizeCalculatePhaseGradients, localWorkSizeCalculatePhaseGradients, 0, NULL, NULL);
		clFinish(commandQueue);
				
		// Calculate values for the A-matrix and h-vector in the Z direction
		runKernelErrorCalculateAMatrixAndHVector2DValuesZ = clEnqueueNDRangeKernel(commandQueue, CalculateAMatrixAndHVector2DValuesZKernel, 3, NULL, globalWorkSizeCalculateAMatrixAndHVector2DValuesZ, localWorkSizeCalculateAMatrixAndHVector2DValuesZ, 0, NULL, NULL);
		clFinish(commandQueue);
			
   		// Setup final equation system

		// Sum in one direction to get 1D values
		runKernelErrorCalculateAMatrix1DValues = clEnqueueNDRangeKernel(commandQueue, CalculateAMatrix1DValuesKernel, 3, NULL, globalWorkSizeCalculateAMatrix1DValues, localWorkSizeCalculateAMatrix1DValues, 0, NULL, NULL);
		clFinish(commandQueue);
			
		runKernelErrorCalculateHVector1DValues = clEnqueueNDRangeKernel(commandQueue, CalculateHVector1DValuesKernel, 3, NULL, globalWorkSizeCalculateHVector1DValues, localWorkSizeCalculateHVector1DValues, 0, NULL, NULL);
		clFinish(commandQueue);
						
		SetMemory(d_A_Matrix,0.0f,NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS);

		// Calculate final A-matrix
		runKernelErrorCalculateAMatrix = clEnqueueNDRangeKernel(commandQueue, CalculateAMatrixKernel, 1, NULL, globalWorkSizeCalculateAMatrix, localWorkSizeCalculateAMatrix, 0, NULL, NULL);
		clFinish(commandQueue);

		// Calculate final h-vector
		runKernelErrorCalculateHVector = clEnqueueNDRangeKernel(commandQueue, CalculateHVectorKernel, 1, NULL, globalWorkSizeCalculateHVector, localWorkSizeCalculateHVector, 0, NULL, NULL);
		clFinish(commandQueue);

		// Copy A-matrix and h-vector from device to host
		clEnqueueReadBuffer(commandQueue, d_A_Matrix, CL_TRUE, 0, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), h_A_Matrix, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_h_Vector, CL_TRUE, 0, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), h_h_Vector, 0, NULL, NULL);

		// Mirror the matrix values
		for (int j = 0; j < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; j++)
		{
			for (int i = 0; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; i++)
			{
				h_A_Matrix[j + i*NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS] = h_A_Matrix[i + j*NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS];
			}
		}

		// Solve the equation system A * p = h to obtain the parameter vector
		SolveEquationSystem(h_A_Matrix, h_Inverse_A_Matrix, h_h_Vector, h_Registration_Parameters, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS);

		// Remove everything but translation
		if (ALIGNMENT_TYPE == TRANSLATION)
		{
			// Increment translation
			h_Registration_Parameters_Align_Two_Volumes[0] += h_Registration_Parameters[0];
			h_Registration_Parameters_Align_Two_Volumes[1] += h_Registration_Parameters[1];
			h_Registration_Parameters_Align_Two_Volumes[2] += h_Registration_Parameters[2];

			// Set transformation matrix to zeros
			for (int i = 3; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; i++)
			{
				h_Registration_Parameters_Align_Two_Volumes[i] = 0.0f;
			}
		}
		// Remove scaling by doing a SVD and forcing all singular values to be 1
		else if (ALIGNMENT_TYPE == RIGID)
		{
			for (int i = 0; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; i++)
			{
				h_Registration_Parameters_Align_Two_Volumes[i] += h_Registration_Parameters[i];
			}		

			RemoveTransformationScaling(h_Registration_Parameters_Align_Two_Volumes);			
		}
		// Keep all parameters
		else if (ALIGNMENT_TYPE == AFFINE)
		{
			for (int i = 0; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; i++)
			{
				h_Registration_Parameters_Align_Two_Volumes[i] += h_Registration_Parameters[i];
			}		
		}

		// Copy parameter vector to constant memory
		clEnqueueWriteBuffer(commandQueue, c_Registration_Parameters, CL_TRUE, 0, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), h_Registration_Parameters_Align_Two_Volumes, 0, NULL, NULL);

		// Interpolate to get the new volume
		runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearKernel, 3, NULL, globalWorkSizeInterpolateVolumeLinear, localWorkSizeInterpolateVolumeLinear, 0, NULL, NULL);
		clFinish(commandQueue);				
	}

	if (ALIGNMENT_TYPE == RIGID)
	{
		CalculateRotationAnglesFromRotationMatrix(h_Rotations, h_Registration_Parameters_Align_Two_Volumes);
	}
}

// Remove scaling from transformation matrix, to get a rotation matrix
void BROCCOLI_LIB::RemoveTransformationScaling(float* h_Registration_Parameters)
{
	double h_Transformation_Parameters_double[9];

	// Make a copy of transformation matrix parameters
	for (int i = 3; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; i++)
	{
		h_Transformation_Parameters_double[i-3] = (double)h_Registration_Parameters[i];
	}		

	// Add one to diagonal
	h_Transformation_Parameters_double[0] += 1.0;
	h_Transformation_Parameters_double[4] += 1.0;
	h_Transformation_Parameters_double[8] += 1.0;

	double U[9], S[3], V[9], Rotation_Matrix[9];

	// Calculate singular value decomposition of transformation matrix
	SVD3x3(U, S, V, h_Transformation_Parameters_double);

	// Ignore singular values (scaling)
	// Rotation matrix = U * V'
	Transpose3x3(V);
	MatMul3x3(Rotation_Matrix, U, V);
	
	// Remove one from diagonal
	Rotation_Matrix[0] -= 1.0;
	Rotation_Matrix[4] -= 1.0;
	Rotation_Matrix[8] -= 1.0;

	for (int i = 0; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS - 3; i++)
	{
		h_Registration_Parameters[i+3] = (float)Rotation_Matrix[i];
	}		
}

// Calculate Euler rotation angles from a rotation matrix
void BROCCOLI_LIB::CalculateRotationAnglesFromRotationMatrix(float* h_Rotations, float* h_Registration_Parameters)
{
	float h_Transformation_Matrix[9];
	float c1, c2, s1;

	// Make a copy of transformation matrix parameters
	for (int i = 3; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; i++)
	{
		h_Transformation_Matrix[i-3] = h_Registration_Parameters[i];
	}		

	// Add ones in the diagonal
	h_Transformation_Matrix[0] += 1.0f;
	h_Transformation_Matrix[4] += 1.0f;
	h_Transformation_Matrix[8] += 1.0f;
	
	// Calculate rotation angles

	// (p0  p1  p2)
	// (p3  p4  p5)
 	// (p6  p7  p8)
	
	/*
	angle1 = atan2(p_matrix(2,3),p_matrix(3,3))*180/pi;
	c2 = sqrt(p_matrix(1,1)^2 + p_matrix(1,2)^2);
	angle2 = atan2(-p_matrix(1,3),c2)*180/pi;
	s1 = sind(angle1);
	c1 = cosd(angle1);
	angle3 = atan2(s1*p_matrix(3,1)-c1*p_matrix(2,1),c1*p_matrix(2,2)-s1*p_matrix(3,2))*180/pi;
	rotations = [angle1, angle2, angle3];
	*/
	
	h_Rotations[0] = -atan2(h_Transformation_Matrix[5], h_Transformation_Matrix[8]) * 180.0f/PI;
	c2 = sqrt(h_Transformation_Matrix[0] * h_Transformation_Matrix[0] + h_Transformation_Matrix[1] * h_Transformation_Matrix[1]);
	h_Rotations[1] = -atan2(-h_Transformation_Matrix[2],c2)*180.0f/PI;
	s1 = sin(h_Rotations[0]*PI/180.0f);
	c1 = cos(h_Rotations[0]*PI/180.0f);
	h_Rotations[2] = -atan2(s1*h_Transformation_Matrix[6] - c1*h_Transformation_Matrix[3],c1*h_Transformation_Matrix[4] - s1*h_Transformation_Matrix[7])*180.0f/PI;
}

void BROCCOLI_LIB::ChangeVolumeSize(cl_mem d_Changed_Volume, cl_mem d_Original_Volume_, int ORIGINAL_DATA_W, int ORIGINAL_DATA_H, int ORIGINAL_DATA_D, int NEW_DATA_W, int NEW_DATA_H, int NEW_DATA_D)
{
	// Create a 3D image (texture) for fast interpolation
	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_INTENSITY;
	cl_mem d_Volume_Texture = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, ORIGINAL_DATA_W, ORIGINAL_DATA_H, ORIGINAL_DATA_D, 0, 0, NULL, NULL);
	
	// Copy the T1 volume to an image to interpolate from
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {ORIGINAL_DATA_W, ORIGINAL_DATA_H, ORIGINAL_DATA_D};
	clEnqueueCopyBufferToImage(commandQueue, d_Original_Volume_, d_Volume_Texture, 0, origin, region, 0, NULL, NULL);
	
	float VOXEL_DIFFERENCE_X = (float)(ORIGINAL_DATA_W-1)/(float)(NEW_DATA_W-1);
	float VOXEL_DIFFERENCE_Y = (float)(ORIGINAL_DATA_H-1)/(float)(NEW_DATA_H-1);
	float VOXEL_DIFFERENCE_Z = (float)(ORIGINAL_DATA_D-1)/(float)(NEW_DATA_D-1);

	SetGlobalAndLocalWorkSizesRescaleVolume(NEW_DATA_W, NEW_DATA_H, NEW_DATA_D);

	clSetKernelArg(RescaleVolumeLinearKernel, 0, sizeof(cl_mem), &d_Changed_Volume);
	clSetKernelArg(RescaleVolumeLinearKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
	clSetKernelArg(RescaleVolumeLinearKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
	clSetKernelArg(RescaleVolumeLinearKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
	clSetKernelArg(RescaleVolumeLinearKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
	clSetKernelArg(RescaleVolumeLinearKernel, 5, sizeof(int), &NEW_DATA_W);
	clSetKernelArg(RescaleVolumeLinearKernel, 6, sizeof(int), &NEW_DATA_H);
	clSetKernelArg(RescaleVolumeLinearKernel, 7, sizeof(int), &NEW_DATA_D);	
	
	error = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeLinearKernel, 3, NULL, globalWorkSizeRescaleVolumeLinear, localWorkSizeRescaleVolumeLinear, 0, NULL, NULL);
	clFinish(commandQueue);	

	clReleaseMemObject(d_Volume_Texture);
}

// Runs registration over several scales, COARSEST_SCALE should be 8, 4, 2 or 1
void BROCCOLI_LIB::AlignTwoVolumesSeveralScales(float *h_Registration_Parameters_Align_Two_Volumes_Several_Scales, float* h_Rotations, cl_mem d_Original_Aligned_Volume, cl_mem d_Original_Reference_Volume, int DATA_W, int DATA_H, int DATA_D, int COARSEST_SCALE_, int NUMBER_OF_ITERATIONS, int ALIGNMENT_TYPE, int OVERWRITE)
{
	for (int i = 0; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; i++)
	{
		h_Registration_Parameters_Align_Two_Volumes_Several_Scales[i] = 0.0f;
		h_Registration_Parameters_Temp[i] = 0.0f;
	}
	h_Rotations[0] = 0.0f;
	h_Rotations[1] = 0.0f;
	h_Rotations[2] = 0.0f;

	// Calculate volume size for coarsest scale
	CURRENT_DATA_W = (int)round((float)DATA_W/(float)COARSEST_SCALE_);
	CURRENT_DATA_H = (int)round((float)DATA_H/(float)COARSEST_SCALE_);
	CURRENT_DATA_D = (int)round((float)DATA_D/(float)COARSEST_SCALE_);

	// Setup all parameters and allocate memory on host
	AlignTwoVolumesSetup(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);
	
	// Change size of original volumes to current scale
	ChangeVolumeSize(d_Aligned_Volume, d_Original_Aligned_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);       
	ChangeVolumeSize(d_Reference_Volume, d_Original_Reference_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);       		
						
	// Copy volume to be aligned to an image (texture)
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D};
	clEnqueueCopyBufferToImage(commandQueue, d_Aligned_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);
	
	// Loop registration over scales	
	for (int current_scale = COARSEST_SCALE_; current_scale >= 1; current_scale = current_scale/2)
	{
		if (current_scale == 1)
		{
			AlignTwoVolumes(h_Registration_Parameters_Temp, h_Rotations_Temp, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, (int)ceil((float)NUMBER_OF_ITERATIONS/10.0f), ALIGNMENT_TYPE);
		}
		else
		{
			AlignTwoVolumes(h_Registration_Parameters_Temp, h_Rotations_Temp, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, NUMBER_OF_ITERATIONS, ALIGNMENT_TYPE);
		}	
		
		if (current_scale != 1)
		{
			h_Rotations[0] += h_Rotations_Temp[0];
			h_Rotations[1] += h_Rotations_Temp[1];
			h_Rotations[2] += h_Rotations_Temp[2];

			// Multiply the transformations by a factor 2 for the next scale and add to previous parameters
			h_Registration_Parameters_Align_Two_Volumes_Several_Scales[0] = 2*h_Registration_Parameters_Align_Two_Volumes_Several_Scales[0] + 2*h_Registration_Parameters_Temp[0]; 
			h_Registration_Parameters_Align_Two_Volumes_Several_Scales[1] = 2*h_Registration_Parameters_Align_Two_Volumes_Several_Scales[1] + 2*h_Registration_Parameters_Temp[1]; 
			h_Registration_Parameters_Align_Two_Volumes_Several_Scales[2] = 2*h_Registration_Parameters_Align_Two_Volumes_Several_Scales[2] + 2*h_Registration_Parameters_Temp[2]; 
			
			// Add transformation parameters for next scale
			for (int i = 3; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; i++)
			{
				h_Registration_Parameters_Align_Two_Volumes_Several_Scales[i] += h_Registration_Parameters_Temp[i]; 
			}
			
			// Clean up before the next scale
			AlignTwoVolumesCleanup();

			// Prepare for the next scale
			CURRENT_DATA_W = (int)round((float)DATA_W/((float)current_scale/2.0f));
			CURRENT_DATA_H = (int)round((float)DATA_H/((float)current_scale/2.0f));
			CURRENT_DATA_D = (int)round((float)DATA_D/((float)current_scale/2.0f));

			// Setup all parameters and allocate memory on host
			AlignTwoVolumesSetup(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);
					
			// Change size of original volumes to current scale
			ChangeVolumeSize(d_Aligned_Volume, d_Original_Aligned_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);       
			ChangeVolumeSize(d_Reference_Volume, d_Original_Reference_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);       
								
			// Copy volume to be aligned to an image (texture)
			size_t origin[3] = {0, 0, 0};
			size_t region[3] = {CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D};
			clEnqueueCopyBufferToImage(commandQueue, d_Aligned_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);

			// Copy incremented parameter vector to constant memory
			clEnqueueWriteBuffer(commandQueue, c_Registration_Parameters, CL_TRUE, 0, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), h_Registration_Parameters_Align_Two_Volumes_Several_Scales, 0, NULL, NULL);

			// Apply transformation to next scale
			error = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearKernel, 3, NULL, globalWorkSizeInterpolateVolumeLinear, localWorkSizeInterpolateVolumeLinear, 0, NULL, NULL);
			clFinish(commandQueue);	

			// Copy transformed volume back to image (texture)
			clEnqueueCopyBufferToImage(commandQueue, d_Aligned_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);
		}
		else // Last scale, nothing more to do
		{	
			// Clean up 
			AlignTwoVolumesCleanup();			

			// Calculate final registration parameters
			for (int i = 0; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; i++)
			{
				h_Registration_Parameters_Align_Two_Volumes_Several_Scales[i] += h_Registration_Parameters_Temp[i]; 
			}
			h_Rotations[0] += h_Rotations_Temp[0];
			h_Rotations[1] += h_Rotations_Temp[1];
			h_Rotations[2] += h_Rotations_Temp[2];

			if (OVERWRITE == DO_OVERWRITE)
			{
				// Transform the original volume once with the final registration parameters, to remove effects of several interpolations
				TransformVolume(d_Original_Aligned_Volume, h_Registration_Parameters_Align_Two_Volumes_Several_Scales, DATA_W, DATA_H, DATA_D, LINEAR);
			}
		}		
	}	
}



// This function is used by all registration functions, to setup necessary parameters
void BROCCOLI_LIB::AlignTwoVolumesSetup(int DATA_W, int DATA_H, int DATA_D)
{	
	// Set global and local work sizes
	SetGlobalAndLocalWorkSizesImageRegistration(DATA_W, DATA_H, DATA_D);

	// Create a 3D image (texture) for fast interpolation
	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_INTENSITY;
	d_Original_Volume = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, DATA_W, DATA_H, DATA_D, 0, 0, NULL, NULL);
	
	// Allocate global memory on the device
	d_Aligned_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorAlignedVolume);
	d_Reference_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorReferenceVolume);

	d_q11_Real = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorq11Real);
	d_q11_Imag = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorq11Imag);
	d_q12_Real = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorq12Real);
	d_q12_Imag = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorq12Imag);
	d_q13_Real = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorq13Real);
	d_q13_Imag = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorq13Imag);
	
	d_q21_Real = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorq21Real);
	d_q21_Imag = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorq21Imag);
	d_q22_Real = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorq22Real);
	d_q22_Imag = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorq22Imag);
	d_q23_Real = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorq23Real);
	d_q23_Imag = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorq23Imag);
	
	d_Phase_Differences = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseDifferences);
	d_Phase_Certainties = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);
	d_Phase_Gradients = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseGradients);
	
	d_A_Matrix = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), NULL, &createBufferErrorAMatrix);
	d_h_Vector = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), NULL, &createBufferErrorHVector);

	d_A_Matrix_2D_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_H * DATA_D * NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS * sizeof(float), NULL, &createBufferErrorAMatrix2DValues);
	d_A_Matrix_1D_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_D * NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS * sizeof(float), NULL, &createBufferErrorAMatrix1DValues);
	
	d_h_Vector_2D_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_H * DATA_D * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), NULL, &createBufferErrorHVector2DValues);
	d_h_Vector_1D_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_D * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), NULL, &createBufferErrorHVector1DValues);

	// Allocate constant memory
	c_Quadrature_Filter_1_Real = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Quadrature_Filter_1_Imag = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Imag);
	c_Quadrature_Filter_2_Real = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter2Real);
	c_Quadrature_Filter_2_Imag = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter2Imag);
	c_Quadrature_Filter_3_Real = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter3Real);
	c_Quadrature_Filter_3_Imag = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter3Imag);	
	c_Registration_Parameters = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), NULL, &createBufferErrorRegistrationParameters);

	// Set all kernel arguments

	clSetKernelArg(NonseparableConvolution3DComplexKernel, 7, sizeof(cl_mem), &c_Quadrature_Filter_1_Real);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 8, sizeof(cl_mem), &c_Quadrature_Filter_1_Imag);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 9, sizeof(cl_mem), &c_Quadrature_Filter_2_Real);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 10, sizeof(cl_mem), &c_Quadrature_Filter_2_Imag);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 11, sizeof(cl_mem), &c_Quadrature_Filter_3_Real);
	clSetKernelArg(NonseparableConvolution3DComplexKernel, 12, sizeof(cl_mem), &c_Quadrature_Filter_3_Imag);	

	clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 0, sizeof(cl_mem), &d_Phase_Differences);
	clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 1, sizeof(cl_mem), &d_Phase_Certainties);
	clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 6, sizeof(int), &DATA_W);
	clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 7, sizeof(int), &DATA_H);
	clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 8, sizeof(int), &DATA_D);
		
	clSetKernelArg(CalculatePhaseGradientsXKernel, 0, sizeof(cl_mem), &d_Phase_Gradients);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 1, sizeof(cl_mem), &d_q11_Real);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 2, sizeof(cl_mem), &d_q11_Imag);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 3, sizeof(cl_mem), &d_q21_Real);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 4, sizeof(cl_mem), &d_q21_Imag);	
	clSetKernelArg(CalculatePhaseGradientsXKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 7, sizeof(int), &DATA_D);
		
	clSetKernelArg(CalculatePhaseGradientsYKernel, 0, sizeof(cl_mem), &d_Phase_Gradients);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 1, sizeof(cl_mem), &d_q12_Real);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 2, sizeof(cl_mem), &d_q12_Imag);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 3, sizeof(cl_mem), &d_q22_Real);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 4, sizeof(cl_mem), &d_q22_Imag);	
	clSetKernelArg(CalculatePhaseGradientsYKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 7, sizeof(int), &DATA_D);
	
	clSetKernelArg(CalculatePhaseGradientsZKernel, 0, sizeof(cl_mem), &d_Phase_Gradients);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 1, sizeof(cl_mem), &d_q13_Real);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 2, sizeof(cl_mem), &d_q13_Imag);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 3, sizeof(cl_mem), &d_q23_Real);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 4, sizeof(cl_mem), &d_q23_Imag);	
	clSetKernelArg(CalculatePhaseGradientsZKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 7, sizeof(int), &DATA_D);
		
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

	int volume = 0;

	clSetKernelArg(InterpolateVolumeNearestKernel, 0, sizeof(cl_mem), &d_Aligned_Volume);
	clSetKernelArg(InterpolateVolumeNearestKernel, 1, sizeof(cl_mem), &d_Original_Volume);
	clSetKernelArg(InterpolateVolumeNearestKernel, 2, sizeof(cl_mem), &c_Registration_Parameters);
	clSetKernelArg(InterpolateVolumeNearestKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(InterpolateVolumeNearestKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(InterpolateVolumeNearestKernel, 5, sizeof(int), &DATA_D);	
	clSetKernelArg(InterpolateVolumeNearestKernel, 6, sizeof(int), &volume);	

	clSetKernelArg(InterpolateVolumeLinearKernel, 0, sizeof(cl_mem), &d_Aligned_Volume);
	clSetKernelArg(InterpolateVolumeLinearKernel, 1, sizeof(cl_mem), &d_Original_Volume);
	clSetKernelArg(InterpolateVolumeLinearKernel, 2, sizeof(cl_mem), &c_Registration_Parameters);
	clSetKernelArg(InterpolateVolumeLinearKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(InterpolateVolumeLinearKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(InterpolateVolumeLinearKernel, 5, sizeof(int), &DATA_D);	
	clSetKernelArg(InterpolateVolumeLinearKernel, 6, sizeof(int), &volume);	

	clSetKernelArg(InterpolateVolumeCubicKernel, 0, sizeof(cl_mem), &d_Aligned_Volume);
	clSetKernelArg(InterpolateVolumeCubicKernel, 1, sizeof(cl_mem), &d_Original_Volume);
	clSetKernelArg(InterpolateVolumeCubicKernel, 2, sizeof(cl_mem), &c_Registration_Parameters);
	clSetKernelArg(InterpolateVolumeCubicKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(InterpolateVolumeCubicKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(InterpolateVolumeCubicKernel, 5, sizeof(int), &DATA_D);	
	clSetKernelArg(InterpolateVolumeCubicKernel, 6, sizeof(int), &volume);	
}

// This function is used by all registration functions, to cleanup
void BROCCOLI_LIB::AlignTwoVolumesCleanup()
{
	// Free all the allocated memory on the device

	clReleaseMemObject(d_Original_Volume);
	clReleaseMemObject(d_Reference_Volume);
	clReleaseMemObject(d_Aligned_Volume);
	
	clReleaseMemObject(d_q11_Real);
	clReleaseMemObject(d_q11_Imag);
	clReleaseMemObject(d_q12_Real);
	clReleaseMemObject(d_q12_Imag);
	clReleaseMemObject(d_q13_Real);
	clReleaseMemObject(d_q13_Imag);

	clReleaseMemObject(d_q21_Real);
	clReleaseMemObject(d_q21_Imag);
	clReleaseMemObject(d_q22_Real);
	clReleaseMemObject(d_q22_Imag);
	clReleaseMemObject(d_q23_Real);
	clReleaseMemObject(d_q23_Imag);

	clReleaseMemObject(d_Phase_Differences);
	clReleaseMemObject(d_Phase_Gradients);
	clReleaseMemObject(d_Phase_Certainties);

	clReleaseMemObject(d_A_Matrix);
	clReleaseMemObject(d_h_Vector);

	clReleaseMemObject(d_A_Matrix_2D_Values);
	clReleaseMemObject(d_A_Matrix_1D_Values);

	clReleaseMemObject(d_h_Vector_2D_Values);
	clReleaseMemObject(d_h_Vector_1D_Values);

	clReleaseMemObject(c_Quadrature_Filter_1_Real);
	clReleaseMemObject(c_Quadrature_Filter_1_Imag);
	clReleaseMemObject(c_Quadrature_Filter_2_Real);
	clReleaseMemObject(c_Quadrature_Filter_2_Imag);
	clReleaseMemObject(c_Quadrature_Filter_3_Real);
	clReleaseMemObject(c_Quadrature_Filter_3_Imag);
	clReleaseMemObject(c_Registration_Parameters);	
}





// Performs registration between one low resolution fMRI volume and a high resolution T1 volume



int mymax(int a, int b)
{
	if (a > b)
		return a;
	else
		return b;
}

void BROCCOLI_LIB::ChangeT1VolumeResolutionAndSizeWrapper()
{
	// Allocate memory for T1 volume
	d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	// Copy data to T1 volume
	clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);

	// Calculate volume size for the same voxel size
	int T1_DATA_W_INTERPOLATED = (int)round((float)T1_DATA_W * T1_VOXEL_SIZE_X / MNI_VOXEL_SIZE_X);
	int T1_DATA_H_INTERPOLATED = (int)round((float)T1_DATA_H * T1_VOXEL_SIZE_Y / MNI_VOXEL_SIZE_Y);
	int T1_DATA_D_INTERPOLATED = (int)round((float)T1_DATA_D * T1_VOXEL_SIZE_Z / MNI_VOXEL_SIZE_Z);
    
	// Allocate memory for interpolated volume
	cl_mem d_Interpolated_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W_INTERPOLATED * T1_DATA_H_INTERPOLATED * T1_DATA_D_INTERPOLATED * sizeof(float), NULL, NULL);
	
	// Create a 3D image (texture) for fast interpolation
	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_INTENSITY;
	cl_mem d_T1_Volume_Texture = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, T1_DATA_W, T1_DATA_H, T1_DATA_D, 0, 0, NULL, NULL);
	
	// Copy the T1 volume to an image to interpolate from
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {T1_DATA_W, T1_DATA_H, T1_DATA_D};
	clEnqueueCopyBufferToImage(commandQueue, d_T1_Volume, d_T1_Volume_Texture, 0, origin, region, 0, NULL, NULL);
	
	float VOXEL_DIFFERENCE_X = (float)(T1_DATA_W-1)/(float)(T1_DATA_W_INTERPOLATED-1);
	float VOXEL_DIFFERENCE_Y = (float)(T1_DATA_H-1)/(float)(T1_DATA_H_INTERPOLATED-1);
	float VOXEL_DIFFERENCE_Z = (float)(T1_DATA_D-1)/(float)(T1_DATA_D_INTERPOLATED-1);

	SetGlobalAndLocalWorkSizesRescaleVolume(T1_DATA_W_INTERPOLATED, T1_DATA_H_INTERPOLATED, T1_DATA_D_INTERPOLATED);

	clSetKernelArg(RescaleVolumeLinearKernel, 0, sizeof(cl_mem), &d_Interpolated_T1_Volume);
	clSetKernelArg(RescaleVolumeLinearKernel, 1, sizeof(cl_mem), &d_T1_Volume_Texture);
	clSetKernelArg(RescaleVolumeLinearKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
	clSetKernelArg(RescaleVolumeLinearKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
	clSetKernelArg(RescaleVolumeLinearKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
	clSetKernelArg(RescaleVolumeLinearKernel, 5, sizeof(int), &T1_DATA_W_INTERPOLATED);
	clSetKernelArg(RescaleVolumeLinearKernel, 6, sizeof(int), &T1_DATA_H_INTERPOLATED);
	clSetKernelArg(RescaleVolumeLinearKernel, 7, sizeof(int), &T1_DATA_D_INTERPOLATED);	
	
	// Interpolate T1 volume to the same voxel size as the MNI volume
	error = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeLinearKernel, 3, NULL, globalWorkSizeRescaleVolumeLinear, localWorkSizeRescaleVolumeLinear, 0, NULL, NULL);
	clFinish(commandQueue);	

	clEnqueueReadBuffer(commandQueue, d_Interpolated_T1_Volume, CL_TRUE, 0, T1_DATA_W_INTERPOLATED * T1_DATA_H_INTERPOLATED * T1_DATA_D_INTERPOLATED * sizeof(float), h_Interpolated_T1_Volume, 0, NULL, NULL);


	// Now make sure that the interpolated T1 volume has the same number of voxels as the MNI volume in each direction
	int x_diff = T1_DATA_W_INTERPOLATED - MNI_DATA_W;
	int y_diff = T1_DATA_H_INTERPOLATED - MNI_DATA_H;
	int z_diff = T1_DATA_D_INTERPOLATED - MNI_DATA_D;

	// Allocate memory for T1 volume of MNI size
	cl_mem d_MNI_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	
	// Set all values to zero
	SetMemory(d_MNI_T1_Volume, 0.0f, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D);

	SetGlobalAndLocalWorkSizesCopyVolumeToNew(mymax(MNI_DATA_W,T1_DATA_W_INTERPOLATED),mymax(MNI_DATA_H,T1_DATA_H_INTERPOLATED),mymax(MNI_DATA_D,T1_DATA_D_INTERPOLATED));

	clSetKernelArg(CopyT1VolumeToMNIKernel, 0, sizeof(cl_mem), &d_MNI_T1_Volume);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 1, sizeof(cl_mem), &d_Interpolated_T1_Volume);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 2, sizeof(int), &MNI_DATA_W);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 3, sizeof(int), &MNI_DATA_H);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 4, sizeof(int), &MNI_DATA_D);	
	clSetKernelArg(CopyT1VolumeToMNIKernel, 5, sizeof(int), &T1_DATA_W_INTERPOLATED);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 6, sizeof(int), &T1_DATA_H_INTERPOLATED);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 7, sizeof(int), &T1_DATA_D_INTERPOLATED);	
	clSetKernelArg(CopyT1VolumeToMNIKernel, 8, sizeof(int), &x_diff);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 9, sizeof(int), &y_diff);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 10, sizeof(int), &z_diff);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 11, sizeof(int), &MM_T1_Z_CUT);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 12, sizeof(float), &MNI_VOXEL_SIZE_Z);
	
	error = clEnqueueNDRangeKernel(commandQueue, CopyT1VolumeToMNIKernel, 3, NULL, globalWorkSizeCopyVolumeToNew, localWorkSizeCopyVolumeToNew, 0, NULL, NULL);
	clFinish(commandQueue);	

	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume, 0, NULL, NULL);
	
	clReleaseMemObject(d_T1_Volume);
	clReleaseMemObject(d_Interpolated_T1_Volume);
	clReleaseMemObject(d_MNI_T1_Volume);
	clReleaseMemObject(d_T1_Volume_Texture);
}

void BROCCOLI_LIB::ChangeT1VolumeResolutionAndSize(cl_mem d_MNI_T1_Volume, cl_mem d_T1_Volume, int T1_DATA_W, int T1_DATA_H, int T1_DATA_D, int MNI_DATA_W, int MNI_DATA_H, int MNI_DATA_D, float T1_VOXEL_SIZE_X, float T1_VOXEL_SIZE_Y, float T1_VOXEL_SIZE_Z, float MNI_VOXEL_SIZE_X, float MNI_VOXEL_SIZE_Y, float MNI_VOXEL_SIZE_Z)
{	
	// Calculate volume size for the same voxel size
	int T1_DATA_W_INTERPOLATED = (int)round((float)T1_DATA_W * T1_VOXEL_SIZE_X / MNI_VOXEL_SIZE_X);
	int T1_DATA_H_INTERPOLATED = (int)round((float)T1_DATA_H * T1_VOXEL_SIZE_Y / MNI_VOXEL_SIZE_Y);
	int T1_DATA_D_INTERPOLATED = (int)round((float)T1_DATA_D * T1_VOXEL_SIZE_Z / MNI_VOXEL_SIZE_Z);
    
	// Allocate memory for interpolated volume
	cl_mem d_Interpolated_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W_INTERPOLATED * T1_DATA_H_INTERPOLATED * T1_DATA_D_INTERPOLATED * sizeof(float), NULL, NULL);
	
	// Create a 3D image (texture) for fast interpolation
	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_INTENSITY;
	cl_mem d_T1_Volume_Texture = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, T1_DATA_W, T1_DATA_H, T1_DATA_D, 0, 0, NULL, NULL);
	
	// Copy the T1 volume to an image to interpolate from
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {T1_DATA_W, T1_DATA_H, T1_DATA_D};
	clEnqueueCopyBufferToImage(commandQueue, d_T1_Volume, d_T1_Volume_Texture, 0, origin, region, 0, NULL, NULL);
	
	float VOXEL_DIFFERENCE_X = (float)(T1_DATA_W-1)/(float)(T1_DATA_W_INTERPOLATED-1);
	float VOXEL_DIFFERENCE_Y = (float)(T1_DATA_H-1)/(float)(T1_DATA_H_INTERPOLATED-1);
	float VOXEL_DIFFERENCE_Z = (float)(T1_DATA_D-1)/(float)(T1_DATA_D_INTERPOLATED-1);

	SetGlobalAndLocalWorkSizesRescaleVolume(T1_DATA_W_INTERPOLATED, T1_DATA_H_INTERPOLATED, T1_DATA_D_INTERPOLATED);

	clSetKernelArg(RescaleVolumeLinearKernel, 0, sizeof(cl_mem), &d_Interpolated_T1_Volume);
	clSetKernelArg(RescaleVolumeLinearKernel, 1, sizeof(cl_mem), &d_T1_Volume_Texture);
	clSetKernelArg(RescaleVolumeLinearKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
	clSetKernelArg(RescaleVolumeLinearKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
	clSetKernelArg(RescaleVolumeLinearKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
	clSetKernelArg(RescaleVolumeLinearKernel, 5, sizeof(int), &T1_DATA_W_INTERPOLATED);
	clSetKernelArg(RescaleVolumeLinearKernel, 6, sizeof(int), &T1_DATA_H_INTERPOLATED);
	clSetKernelArg(RescaleVolumeLinearKernel, 7, sizeof(int), &T1_DATA_D_INTERPOLATED);	
	
	// Interpolate T1 volume to the same voxel size as the MNI volume
	error = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeLinearKernel, 3, NULL, globalWorkSizeRescaleVolumeLinear, localWorkSizeRescaleVolumeLinear, 0, NULL, NULL);
	clFinish(commandQueue);	
	
	// Now make sure that the interpolated T1 volume has the same number of voxels as the MNI volume in each direction
	int x_diff = T1_DATA_W_INTERPOLATED - MNI_DATA_W;
	int y_diff = T1_DATA_H_INTERPOLATED - MNI_DATA_H;
	int z_diff = T1_DATA_D_INTERPOLATED - MNI_DATA_D;
	
	// Set all values to zero
	SetMemory(d_MNI_T1_Volume, 0.0f, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D);

	SetGlobalAndLocalWorkSizesCopyVolumeToNew(mymax(MNI_DATA_W,T1_DATA_W_INTERPOLATED),mymax(MNI_DATA_H,T1_DATA_H_INTERPOLATED),mymax(MNI_DATA_D,T1_DATA_D_INTERPOLATED));

	clSetKernelArg(CopyT1VolumeToMNIKernel, 0, sizeof(cl_mem), &d_MNI_T1_Volume);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 1, sizeof(cl_mem), &d_Interpolated_T1_Volume);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 2, sizeof(int), &MNI_DATA_W);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 3, sizeof(int), &MNI_DATA_H);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 4, sizeof(int), &MNI_DATA_D);	
	clSetKernelArg(CopyT1VolumeToMNIKernel, 5, sizeof(int), &T1_DATA_W_INTERPOLATED);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 6, sizeof(int), &T1_DATA_H_INTERPOLATED);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 7, sizeof(int), &T1_DATA_D_INTERPOLATED);	
	clSetKernelArg(CopyT1VolumeToMNIKernel, 8, sizeof(int), &x_diff);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 9, sizeof(int), &y_diff);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 10, sizeof(int), &z_diff);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 11, sizeof(int), &MM_T1_Z_CUT);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 12, sizeof(float), &MNI_VOXEL_SIZE_Z);
	
	error = clEnqueueNDRangeKernel(commandQueue, CopyT1VolumeToMNIKernel, 3, NULL, globalWorkSizeCopyVolumeToNew, localWorkSizeCopyVolumeToNew, 0, NULL, NULL);
	clFinish(commandQueue);	

	clReleaseMemObject(d_Interpolated_T1_Volume);
	clReleaseMemObject(d_T1_Volume_Texture);
}

void BROCCOLI_LIB::ChangeEPIVolumeResolutionAndSize(cl_mem d_T1_EPI_Volume, cl_mem d_EPI_Volume, int EPI_DATA_W, int EPI_DATA_H, int EPI_DATA_D, int T1_DATA_W, int T1_DATA_H, int T1_DATA_D, float EPI_VOXEL_SIZE_X, float EPI_VOXEL_SIZE_Y, float EPI_VOXEL_SIZE_Z, float T1_VOXEL_SIZE_X, float T1_VOXEL_SIZE_Y, float T1_VOXEL_SIZE_Z)
{	
	// Calculate volume size for the same voxel size
	int EPI_DATA_W_INTERPOLATED = (int)round((float)EPI_DATA_W * EPI_VOXEL_SIZE_X / T1_VOXEL_SIZE_X);
	int EPI_DATA_H_INTERPOLATED = (int)round((float)EPI_DATA_H * EPI_VOXEL_SIZE_Y / T1_VOXEL_SIZE_Y);
	int EPI_DATA_D_INTERPOLATED = (int)round((float)EPI_DATA_D * EPI_VOXEL_SIZE_Z / T1_VOXEL_SIZE_Z);
    
	// Allocate memory for interpolated volume
	cl_mem d_Interpolated_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  EPI_DATA_W_INTERPOLATED * EPI_DATA_H_INTERPOLATED * EPI_DATA_D_INTERPOLATED * sizeof(float), NULL, NULL);
	
	// Create a 3D image (texture) for fast interpolation
	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_INTENSITY;
	cl_mem d_EPI_Volume_Texture = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 0, 0, NULL, NULL);
	
	// Copy the T1 volume to an image to interpolate from
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {EPI_DATA_W, EPI_DATA_H, EPI_DATA_D};
	clEnqueueCopyBufferToImage(commandQueue, d_EPI_Volume, d_EPI_Volume_Texture, 0, origin, region, 0, NULL, NULL);
	
	float VOXEL_DIFFERENCE_X = (float)(EPI_DATA_W-1)/(float)(EPI_DATA_W_INTERPOLATED-1);
	float VOXEL_DIFFERENCE_Y = (float)(EPI_DATA_H-1)/(float)(EPI_DATA_H_INTERPOLATED-1);
	float VOXEL_DIFFERENCE_Z = (float)(EPI_DATA_D-1)/(float)(EPI_DATA_D_INTERPOLATED-1);

	SetGlobalAndLocalWorkSizesRescaleVolume(EPI_DATA_W_INTERPOLATED, EPI_DATA_H_INTERPOLATED, EPI_DATA_D_INTERPOLATED);

	clSetKernelArg(RescaleVolumeLinearKernel, 0, sizeof(cl_mem), &d_Interpolated_EPI_Volume);
	clSetKernelArg(RescaleVolumeLinearKernel, 1, sizeof(cl_mem), &d_EPI_Volume_Texture);
	clSetKernelArg(RescaleVolumeLinearKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
	clSetKernelArg(RescaleVolumeLinearKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
	clSetKernelArg(RescaleVolumeLinearKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
	clSetKernelArg(RescaleVolumeLinearKernel, 5, sizeof(int), &EPI_DATA_W_INTERPOLATED);
	clSetKernelArg(RescaleVolumeLinearKernel, 6, sizeof(int), &EPI_DATA_H_INTERPOLATED);
	clSetKernelArg(RescaleVolumeLinearKernel, 7, sizeof(int), &EPI_DATA_D_INTERPOLATED);	
	
	// Interpolate EPI volume to the same voxel size as the new volume
	error = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeLinearKernel, 3, NULL, globalWorkSizeRescaleVolumeLinear, localWorkSizeRescaleVolumeLinear, 0, NULL, NULL);
	clFinish(commandQueue);	
	
	// Now make sure that the interpolated T1 volume has the same number of voxels as the MNI volume in each direction
	int x_diff = EPI_DATA_W_INTERPOLATED - T1_DATA_W;
	int y_diff = EPI_DATA_H_INTERPOLATED - T1_DATA_H;
	int z_diff = EPI_DATA_D_INTERPOLATED - T1_DATA_D;
	
	// Set all values to zero
	SetMemory(d_T1_EPI_Volume, 0.0f, T1_DATA_W * T1_DATA_H * T1_DATA_D);

	SetGlobalAndLocalWorkSizesCopyVolumeToNew(mymax(T1_DATA_W,EPI_DATA_W_INTERPOLATED),mymax(T1_DATA_H,EPI_DATA_H_INTERPOLATED),mymax(T1_DATA_D,EPI_DATA_D_INTERPOLATED));

	clSetKernelArg(CopyEPIVolumeToT1Kernel, 0, sizeof(cl_mem), &d_T1_EPI_Volume);
	clSetKernelArg(CopyEPIVolumeToT1Kernel, 1, sizeof(cl_mem), &d_Interpolated_EPI_Volume);
	clSetKernelArg(CopyEPIVolumeToT1Kernel, 2, sizeof(int), &T1_DATA_W);
	clSetKernelArg(CopyEPIVolumeToT1Kernel, 3, sizeof(int), &T1_DATA_H);
	clSetKernelArg(CopyEPIVolumeToT1Kernel, 4, sizeof(int), &T1_DATA_D);	
	clSetKernelArg(CopyEPIVolumeToT1Kernel, 5, sizeof(int), &EPI_DATA_W_INTERPOLATED);
	clSetKernelArg(CopyEPIVolumeToT1Kernel, 6, sizeof(int), &EPI_DATA_H_INTERPOLATED);
	clSetKernelArg(CopyEPIVolumeToT1Kernel, 7, sizeof(int), &EPI_DATA_D_INTERPOLATED);	
	clSetKernelArg(CopyEPIVolumeToT1Kernel, 8, sizeof(int), &x_diff);
	clSetKernelArg(CopyEPIVolumeToT1Kernel, 9, sizeof(int), &y_diff);
	clSetKernelArg(CopyEPIVolumeToT1Kernel, 10, sizeof(int), &z_diff);
	clSetKernelArg(CopyEPIVolumeToT1Kernel, 11, sizeof(int), &MM_EPI_Z_CUT);
	clSetKernelArg(CopyEPIVolumeToT1Kernel, 12, sizeof(float), &T1_VOXEL_SIZE_Z);
	
	error = clEnqueueNDRangeKernel(commandQueue, CopyEPIVolumeToT1Kernel, 3, NULL, globalWorkSizeCopyVolumeToNew, localWorkSizeCopyVolumeToNew, 0, NULL, NULL);
	clFinish(commandQueue);	

	clReleaseMemObject(d_Interpolated_EPI_Volume);
	clReleaseMemObject(d_EPI_Volume_Texture);
}

void BROCCOLI_LIB::ChangeVolumesResolutionAndSize(cl_mem d_New_Volumes, cl_mem d_Volumes, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_VOLUMES, int NEW_DATA_W, int NEW_DATA_H, int NEW_DATA_D, float VOXEL_SIZE_X, float VOXEL_SIZE_Y, float VOXEL_SIZE_Z, float NEW_VOXEL_SIZE_X, float NEW_VOXEL_SIZE_Y, float NEW_VOXEL_SIZE_Z, int MM_Z_CUT)
{	
	// Calculate volume size for the same voxel size
	int DATA_W_INTERPOLATED = (int)round((float)DATA_W * VOXEL_SIZE_X / NEW_VOXEL_SIZE_X);
	int DATA_H_INTERPOLATED = (int)round((float)DATA_H * VOXEL_SIZE_Y / NEW_VOXEL_SIZE_Y);
	int DATA_D_INTERPOLATED = (int)round((float)DATA_D * VOXEL_SIZE_Z / NEW_VOXEL_SIZE_Z);
    
	// Allocate memory for interpolated volume
	cl_mem d_Interpolated_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W_INTERPOLATED * DATA_H_INTERPOLATED * DATA_D_INTERPOLATED * sizeof(float), NULL, NULL);
	
	// Create a 3D image (texture) for fast interpolation
	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_INTENSITY;
	cl_mem d_Volume_Texture = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, DATA_W, DATA_H, DATA_D, 0, 0, NULL, NULL);
	
	float VOXEL_DIFFERENCE_X = (float)(DATA_W-1)/(float)(DATA_W_INTERPOLATED-1);
	float VOXEL_DIFFERENCE_Y = (float)(DATA_H-1)/(float)(DATA_H_INTERPOLATED-1);
	float VOXEL_DIFFERENCE_Z = (float)(DATA_D-1)/(float)(DATA_D_INTERPOLATED-1);

	SetGlobalAndLocalWorkSizesRescaleVolume(DATA_W_INTERPOLATED, DATA_H_INTERPOLATED, DATA_D_INTERPOLATED);

	SetGlobalAndLocalWorkSizesCopyVolumeToNew(mymax(NEW_DATA_W,DATA_W_INTERPOLATED),mymax(NEW_DATA_H,DATA_H_INTERPOLATED),mymax(NEW_DATA_D,DATA_D_INTERPOLATED));
		
	clSetKernelArg(RescaleVolumeLinearKernel, 0, sizeof(cl_mem), &d_Interpolated_Volume);
	clSetKernelArg(RescaleVolumeLinearKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
	clSetKernelArg(RescaleVolumeLinearKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
	clSetKernelArg(RescaleVolumeLinearKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
	clSetKernelArg(RescaleVolumeLinearKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
	clSetKernelArg(RescaleVolumeLinearKernel, 5, sizeof(int), &DATA_W_INTERPOLATED);
	clSetKernelArg(RescaleVolumeLinearKernel, 6, sizeof(int), &DATA_H_INTERPOLATED);
	clSetKernelArg(RescaleVolumeLinearKernel, 7, sizeof(int), &DATA_D_INTERPOLATED);		

	// Make sure that the interpolated EPI volume has the same number of voxels as the new volume in each direction
	int x_diff = DATA_W_INTERPOLATED - NEW_DATA_W;
	int y_diff = DATA_H_INTERPOLATED - NEW_DATA_H;
	int z_diff = DATA_D_INTERPOLATED - NEW_DATA_D;

	clSetKernelArg(CopyVolumeToNewKernel, 0, sizeof(cl_mem), &d_New_Volumes);
	clSetKernelArg(CopyVolumeToNewKernel, 1, sizeof(cl_mem), &d_Interpolated_Volume);
	clSetKernelArg(CopyVolumeToNewKernel, 2, sizeof(int), &NEW_DATA_W);
	clSetKernelArg(CopyVolumeToNewKernel, 3, sizeof(int), &NEW_DATA_H);
	clSetKernelArg(CopyVolumeToNewKernel, 4, sizeof(int), &NEW_DATA_D);	
	clSetKernelArg(CopyVolumeToNewKernel, 5, sizeof(int), &DATA_W_INTERPOLATED);
	clSetKernelArg(CopyVolumeToNewKernel, 6, sizeof(int), &DATA_H_INTERPOLATED);
	clSetKernelArg(CopyVolumeToNewKernel, 7, sizeof(int), &DATA_D_INTERPOLATED);	
	clSetKernelArg(CopyVolumeToNewKernel, 8, sizeof(int), &x_diff);
	clSetKernelArg(CopyVolumeToNewKernel, 9, sizeof(int), &y_diff);
	clSetKernelArg(CopyVolumeToNewKernel, 10, sizeof(int), &z_diff);
	clSetKernelArg(CopyVolumeToNewKernel, 11, sizeof(int), &MM_Z_CUT);
	clSetKernelArg(CopyVolumeToNewKernel, 12, sizeof(float), &NEW_VOXEL_SIZE_Z);
	
	// Set all values to zero
	SetMemory(d_New_Volumes, 110.3f, DATA_W * DATA_H * DATA_D * NUMBER_OF_VOLUMES);
	
	for (int volume = 0; volume < NUMBER_OF_VOLUMES; volume++)
	{
		// Copy the current volume to an image to interpolate from
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {DATA_W, DATA_H, DATA_D};
		clEnqueueCopyBufferToImage(commandQueue, d_Volumes, d_Volume_Texture, volume * DATA_W * DATA_H * DATA_D * sizeof(float), origin, region, 0, NULL, NULL);
		
		// Rescale current volume to the same voxel size as the new volume
		error = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeLinearKernel, 3, NULL, globalWorkSizeRescaleVolumeLinear, localWorkSizeRescaleVolumeLinear, 0, NULL, NULL);
		clFinish(commandQueue);	
		
		clSetKernelArg(CopyVolumeToNewKernel, 13, sizeof(int), &volume);
	
		error = clEnqueueNDRangeKernel(commandQueue, CopyVolumeToNewKernel, 3, NULL, globalWorkSizeCopyVolumeToNew, localWorkSizeCopyVolumeToNew, 0, NULL, NULL);
		clFinish(commandQueue);	
	}

	clReleaseMemObject(d_Interpolated_Volume);
	clReleaseMemObject(d_Volume_Texture);
}

void BROCCOLI_LIB::InvertAffineRegistrationParameters(float* h_Inverse_Parameters, float* h_Parameters)
{
	// Put parameters in 4 x 4 affine transformation matrix
	
	// (p1 p2 p3 tx)
	// (p4 p5 p6 ty)
	// (p7 p8 p9 tz)
	// (0  0  0  1 )

	float Affine_Matrix[16], Inverse_Affine_Matrix[16];
	
	// Add ones in the diagonal, to get a transformation matrix
	// First row
	Affine_Matrix[0] = h_Parameters[3] + 1.0f;
	Affine_Matrix[1] = h_Parameters[4];
	Affine_Matrix[2] = h_Parameters[5];
	Affine_Matrix[3] = h_Parameters[0];

	// Second row
	Affine_Matrix[4] = h_Parameters[6];
	Affine_Matrix[5] = h_Parameters[7] + 1.0f;
	Affine_Matrix[6] = h_Parameters[8];
	Affine_Matrix[7] = h_Parameters[1];

	// Third row
	Affine_Matrix[8]  = h_Parameters[9];
	Affine_Matrix[9]  = h_Parameters[10];
	Affine_Matrix[10] = h_Parameters[11] + 1.0f;
	Affine_Matrix[11] = h_Parameters[2];

	// Fourth row
	Affine_Matrix[12] = 0.0f;
	Affine_Matrix[13] = 0.0f;
	Affine_Matrix[14] = 0.0f;
	Affine_Matrix[15] = 1.0f;

	// Invert the affine transformation matrix, th get the inverse parameters
	InvertMatrix(Inverse_Affine_Matrix, Affine_Matrix, 4);

	// Subtract ones in the diagonal
	// First row
	h_Inverse_Parameters[0] = Inverse_Affine_Matrix[3];
	h_Inverse_Parameters[1] = Inverse_Affine_Matrix[7];
	h_Inverse_Parameters[2] = Inverse_Affine_Matrix[11];

	// Second row
	h_Inverse_Parameters[3] = Inverse_Affine_Matrix[0] - 1.0f;
	h_Inverse_Parameters[4] = Inverse_Affine_Matrix[1];
	h_Inverse_Parameters[5] = Inverse_Affine_Matrix[2];

	// Third row
	h_Inverse_Parameters[6] = Inverse_Affine_Matrix[4];
	h_Inverse_Parameters[7] = Inverse_Affine_Matrix[5] - 1.0f;
	h_Inverse_Parameters[8] = Inverse_Affine_Matrix[6];

	// Fourth row
	h_Inverse_Parameters[9] = Inverse_Affine_Matrix[8];
	h_Inverse_Parameters[10] = Inverse_Affine_Matrix[9];
	h_Inverse_Parameters[11] = Inverse_Affine_Matrix[10] - 1.0f;
}

// Performs skullstrip by multiplying aligned T1 volume with MNI brain mask
void BROCCOLI_LIB::PerformSkullstrip(cl_mem d_Skullstripped_T1_Volume, cl_mem d_T1_Volume, cl_mem d_Transformed_MNI_Brain_Mask, int DATA_W, int DATA_H, int DATA_D)
{
	SetGlobalAndLocalWorkSizesMultiplyVolumes(DATA_W, DATA_H, DATA_D);

	clSetKernelArg(MultiplyVolumesKernel, 0, sizeof(cl_mem), &d_Skullstripped_T1_Volume);
	clSetKernelArg(MultiplyVolumesKernel, 1, sizeof(cl_mem), &d_T1_Volume);
	clSetKernelArg(MultiplyVolumesKernel, 2, sizeof(cl_mem), &d_Transformed_MNI_Brain_Mask);
	clSetKernelArg(MultiplyVolumesKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(MultiplyVolumesKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(MultiplyVolumesKernel, 5, sizeof(int), &DATA_D);

	runKernelErrorMultiplyVolumes = clEnqueueNDRangeKernel(commandQueue, MultiplyVolumesKernel, 3, NULL, globalWorkSizeMultiplyVolumes, localWorkSizeMultiplyVolumes, 0, NULL, NULL);
	clFinish(commandQueue);	
}



void BROCCOLI_LIB::PerformRegistrationEPIT1Wrapper()
{
	// Allocate memory for EPI volume, T1 volume and EPI volume of T1 size
	d_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);	
	d_T1_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	
	// Copy data to EPI volume and T1 volume
	clEnqueueWriteBuffer(commandQueue, d_EPI_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);
	
	// Interpolate EPI volume to T1 resolution and make sure it has the same size
	ChangeEPIVolumeResolutionAndSize(d_T1_EPI_Volume, d_EPI_Volume, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z);       
	
	// Copy the EPI T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_Interpolated_EPI_Volume, 0, NULL, NULL);

	// Do the registration between EPI and T1 with several scales, rigid
	AlignTwoVolumesSeveralScales(h_Registration_Parameters_EPI_T1_Affine, h_Rotations, d_T1_EPI_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE);
	
	// Copy the aligned EPI volume to host
	clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_Aligned_EPI_Volume, 0, NULL, NULL);
	
	// Get translations 
	h_Registration_Parameters_EPI_T1_Out[0] = h_Registration_Parameters_EPI_T1_Affine[0];
	h_Registration_Parameters_EPI_T1_Out[1] = h_Registration_Parameters_EPI_T1_Affine[1];
	h_Registration_Parameters_EPI_T1_Out[2] = h_Registration_Parameters_EPI_T1_Affine[2];

	// Get rotations
	h_Registration_Parameters_EPI_T1_Out[3] = h_Rotations[0];
	h_Registration_Parameters_EPI_T1_Out[4] = h_Rotations[1];
	h_Registration_Parameters_EPI_T1_Out[5] = h_Rotations[2];

	// Cleanup
	clReleaseMemObject(d_EPI_Volume);
	clReleaseMemObject(d_T1_Volume);
	clReleaseMemObject(d_T1_EPI_Volume);	
}

void BROCCOLI_LIB::PerformRegistrationEPIT1()
{
	// Interpolate EPI volume to T1 resolution and make sure it has the same size, 
	// the registration is performed to the skullstripped T1 volume, which has MNI size
	ChangeEPIVolumeResolutionAndSize(d_T1_EPI_Volume, d_EPI_Volume, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z);       
	
	// Do the registration between EPI and skullstripped T1 with several scales, rigid
	AlignTwoVolumesSeveralScales(h_Registration_Parameters_EPI_T1_Affine, h_Rotations, d_T1_EPI_Volume, d_Skullstripped_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION, RIGID, NO_OVERWRITE);
	
	// Get translations 
	h_Registration_Parameters_EPI_T1[0] = h_Registration_Parameters_EPI_T1_Affine[0];
	h_Registration_Parameters_EPI_T1[1] = h_Registration_Parameters_EPI_T1_Affine[1];
	h_Registration_Parameters_EPI_T1[2] = h_Registration_Parameters_EPI_T1_Affine[2];

	// Get rotations
	h_Registration_Parameters_EPI_T1[3] = h_Rotations[0];
	h_Registration_Parameters_EPI_T1[4] = h_Rotations[1];
	h_Registration_Parameters_EPI_T1[5] = h_Rotations[2];
}

void BROCCOLI_LIB::PerformRegistrationT1MNIWrapper()
{
	// Allocate memory for T1 volume, MNI volume and T1 volume of MNI size
	d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);	
	d_MNI_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	
	// Copy data to T1 volume and MNI volume
	clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Volume , 0, NULL, NULL);

	// Interpolate T1 volume to MNI resolution and make sure it has the same size
	ChangeT1VolumeResolutionAndSize(d_MNI_T1_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z);       
	
	// Copy the MNI T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Interpolated_T1_Volume, 0, NULL, NULL);

	// Do the registration between T1 and MNI with several scales
	AlignTwoVolumesSeveralScales(h_Registration_Parameters_T1_MNI_Out, h_Rotations, d_MNI_T1_Volume, d_MNI_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE);
	
	// Copy the aligned T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume, 0, NULL, NULL);
				
	// Allocate memory for skullstripped T1 volume and MNI brain mask
	d_Skullstripped_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);	

	// Copy MNI brain mask to device
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

	// Calculate inverse transform between T1 and MNI
	InvertAffineRegistrationParameters(h_Inverse_Registration_Parameters, h_Registration_Parameters_T1_MNI_Out);

	// Now apply the inverse transformation between MNI and T1, to transform MNI brain mask to T1 space
	TransformVolume(d_MNI_Brain_Mask, h_Inverse_Registration_Parameters, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NEAREST);

	// Interpolate T1 volume to MNI resolution and make sure it has the same size (again since we overwrote the volume previously, to be able to copy it to the host)
	ChangeT1VolumeResolutionAndSize(d_MNI_T1_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z);       

	// Create skullstripped volume, by multiplying original T1 volume with transformed MNI brain mask 
	PerformSkullstrip(d_Skullstripped_T1_Volume, d_MNI_T1_Volume, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	// Copy the skullstripped T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_Skullstripped_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Skullstripped_T1_Volume, 0, NULL, NULL);

	// Cleanup	
	clReleaseMemObject(d_T1_Volume);
	clReleaseMemObject(d_MNI_Volume);
	clReleaseMemObject(d_MNI_T1_Volume);
	clReleaseMemObject(d_Skullstripped_T1_Volume);
	clReleaseMemObject(d_MNI_Brain_Mask);
}

// Performs registration between one high resolution T1 volume and a high resolution MNI volume (brain template)
void BROCCOLI_LIB::PerformRegistrationT1MNI()
{
	// Interpolate T1 volume to MNI resolution and make sure it has the same size
	ChangeT1VolumeResolutionAndSize(d_MNI_T1_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z);       
	
	// Do the registration between T1 and MNI with several scales (we do not need the aligned T1 volume so do not overwrite)
	AlignTwoVolumesSeveralScales(h_Registration_Parameters_T1_MNI, h_Rotations, d_MNI_T1_Volume, d_MNI_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION, AFFINE, NO_OVERWRITE);
						
	// Calculate inverse transform between T1 and MNI
	InvertAffineRegistrationParameters(h_Inverse_Registration_Parameters, h_Registration_Parameters_T1_MNI);
	
	// Now apply the inverse transformation between MNI and T1, to transform MNI brain mask to T1 space
	TransformVolume(d_MNI_Brain_Mask, h_Inverse_Registration_Parameters, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NEAREST);

	// Create skullstripped volume, by multiplying original T1 volume with transformed MNI brain mask
	PerformSkullstrip(d_Skullstripped_T1_Volume, d_MNI_T1_Volume, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
}

void BROCCOLI_LIB::TransformVolume(cl_mem d_Volume, float* h_Registration_Parameters_, int DATA_W, int DATA_H, int DATA_D, int INTERPOLATION_MODE)
{
	// Allocate constant memory
	cl_mem c_Parameters = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), NULL, &createBufferErrorRegistrationParameters);

	// Copy parameter vector to constant memory
	clEnqueueWriteBuffer(commandQueue, c_Parameters, CL_TRUE, 0, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), h_Registration_Parameters_, 0, NULL, NULL);

	// Allocate memory for texture
	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_INTENSITY;
	cl_mem d_Volume_Texture = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, DATA_W, DATA_H, DATA_D, 0, 0, NULL, NULL);

	// Copy volume to texture
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {DATA_W, DATA_H, DATA_D};
	clEnqueueCopyBufferToImage(commandQueue, d_Volume, d_Volume_Texture, 0, origin, region, 0, NULL, NULL);
	
	SetGlobalAndLocalWorkSizesInterpolateVolume(DATA_W, DATA_H, DATA_D);

	int volume = 0;

	// Interpolate to get the transformed volume
	if (INTERPOLATION_MODE == LINEAR)
	{
		clSetKernelArg(InterpolateVolumeLinearKernel, 0, sizeof(cl_mem), &d_Volume);
		clSetKernelArg(InterpolateVolumeLinearKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
		clSetKernelArg(InterpolateVolumeLinearKernel, 2, sizeof(cl_mem), &c_Parameters);
		clSetKernelArg(InterpolateVolumeLinearKernel, 3, sizeof(int), &DATA_W);
		clSetKernelArg(InterpolateVolumeLinearKernel, 4, sizeof(int), &DATA_H);
		clSetKernelArg(InterpolateVolumeLinearKernel, 5, sizeof(int), &DATA_D);	
		clSetKernelArg(InterpolateVolumeLinearKernel, 6, sizeof(int), &volume);	
		runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearKernel, 3, NULL, globalWorkSizeInterpolateVolumeLinear, localWorkSizeInterpolateVolumeLinear, 0, NULL, NULL);
		clFinish(commandQueue);	
	}
	else if (INTERPOLATION_MODE == NEAREST)
	{
		clSetKernelArg(InterpolateVolumeNearestKernel, 0, sizeof(cl_mem), &d_Volume);
		clSetKernelArg(InterpolateVolumeNearestKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
		clSetKernelArg(InterpolateVolumeNearestKernel, 2, sizeof(cl_mem), &c_Parameters);
		clSetKernelArg(InterpolateVolumeNearestKernel, 3, sizeof(int), &DATA_W);
		clSetKernelArg(InterpolateVolumeNearestKernel, 4, sizeof(int), &DATA_H);
		clSetKernelArg(InterpolateVolumeNearestKernel, 5, sizeof(int), &DATA_D);
		clSetKernelArg(InterpolateVolumeNearestKernel, 6, sizeof(int), &volume);	
		runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeNearestKernel, 3, NULL, globalWorkSizeInterpolateVolumeNearest, localWorkSizeInterpolateVolumeNearest, 0, NULL, NULL);
		clFinish(commandQueue);	
	}

	clReleaseMemObject(d_Volume_Texture);
	clReleaseMemObject(c_Parameters);
}

void BROCCOLI_LIB::TransformVolumes(cl_mem d_Volumes, float* h_Registration_Parameters_, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_VOLUMES, int INTERPOLATION_MODE)
{
	// Allocate constant memory
	cl_mem c_Parameters = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), NULL, &createBufferErrorRegistrationParameters);

	// Copy parameter vector to constant memory
	clEnqueueWriteBuffer(commandQueue, c_Parameters, CL_TRUE, 0, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), h_Registration_Parameters_, 0, NULL, NULL);

	// Allocate memory for texture
	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_INTENSITY;
	cl_mem d_Volume_Texture = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, DATA_W, DATA_H, DATA_D, 0, 0, NULL, NULL);

	SetGlobalAndLocalWorkSizesInterpolateVolume(DATA_W, DATA_H, DATA_D);

	for (int volume = 0; volume < NUMBER_OF_VOLUMES; volume++)
	{
		// Copy current volume to texture
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {DATA_W, DATA_H, DATA_D};
		clEnqueueCopyBufferToImage(commandQueue, d_Volumes, d_Volume_Texture, volume * DATA_W * DATA_H * DATA_D * sizeof(float), origin, region, 0, NULL, NULL);
		
		// Interpolate to get the transformed volume
		if (INTERPOLATION_MODE == LINEAR)
		{
			clSetKernelArg(InterpolateVolumeLinearKernel, 0, sizeof(cl_mem), &d_Volumes);
			clSetKernelArg(InterpolateVolumeLinearKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
			clSetKernelArg(InterpolateVolumeLinearKernel, 2, sizeof(cl_mem), &c_Parameters);
			clSetKernelArg(InterpolateVolumeLinearKernel, 3, sizeof(int), &DATA_W);
			clSetKernelArg(InterpolateVolumeLinearKernel, 4, sizeof(int), &DATA_H);
			clSetKernelArg(InterpolateVolumeLinearKernel, 5, sizeof(int), &DATA_D);	
			clSetKernelArg(InterpolateVolumeLinearKernel, 6, sizeof(int), &volume);	
			runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearKernel, 3, NULL, globalWorkSizeInterpolateVolumeLinear, localWorkSizeInterpolateVolumeLinear, 0, NULL, NULL);
			clFinish(commandQueue);	
		}
		else if (INTERPOLATION_MODE == NEAREST)
		{
			clSetKernelArg(InterpolateVolumeNearestKernel, 0, sizeof(cl_mem), &d_Volumes);
			clSetKernelArg(InterpolateVolumeNearestKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
			clSetKernelArg(InterpolateVolumeNearestKernel, 2, sizeof(cl_mem), &c_Parameters);
			clSetKernelArg(InterpolateVolumeNearestKernel, 3, sizeof(int), &DATA_W);
			clSetKernelArg(InterpolateVolumeNearestKernel, 4, sizeof(int), &DATA_H);
			clSetKernelArg(InterpolateVolumeNearestKernel, 5, sizeof(int), &DATA_D);	
			clSetKernelArg(InterpolateVolumeNearestKernel, 6, sizeof(int), &volume);	
			runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeNearestKernel, 3, NULL, globalWorkSizeInterpolateVolumeNearest, localWorkSizeInterpolateVolumeNearest, 0, NULL, NULL);
			clFinish(commandQueue);	
		}
	}

	clReleaseMemObject(d_Volume_Texture);
	clReleaseMemObject(c_Parameters);
}

void BROCCOLI_LIB::PerformFirstLevelAnalysisWrapper()
{
	//------------------------

	// Allocate memory on device
	d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);	
	d_MNI_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);	
	d_Skullstripped_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);	

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

	PerformRegistrationT1MNI();

	h_Registration_Parameters_T1_MNI_Out[0] = h_Registration_Parameters_T1_MNI[0];
	h_Registration_Parameters_T1_MNI_Out[1] = h_Registration_Parameters_T1_MNI[1];
	h_Registration_Parameters_T1_MNI_Out[2] = h_Registration_Parameters_T1_MNI[2];
	h_Registration_Parameters_T1_MNI_Out[3] = h_Registration_Parameters_T1_MNI[3];
	h_Registration_Parameters_T1_MNI_Out[4] = h_Registration_Parameters_T1_MNI[4];
	h_Registration_Parameters_T1_MNI_Out[5] = h_Registration_Parameters_T1_MNI[5];
	h_Registration_Parameters_T1_MNI_Out[6] = h_Registration_Parameters_T1_MNI[6];
	h_Registration_Parameters_T1_MNI_Out[7] = h_Registration_Parameters_T1_MNI[7];
	h_Registration_Parameters_T1_MNI_Out[8] = h_Registration_Parameters_T1_MNI[8];
	h_Registration_Parameters_T1_MNI_Out[9] = h_Registration_Parameters_T1_MNI[9];
	h_Registration_Parameters_T1_MNI_Out[10] = h_Registration_Parameters_T1_MNI[10];
	h_Registration_Parameters_T1_MNI_Out[11] = h_Registration_Parameters_T1_MNI[11];

	// Cleanup
	clReleaseMemObject(d_T1_Volume);
	clReleaseMemObject(d_MNI_Volume);
	

	//------------------------

	// Allocate memory on device
	d_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_T1_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	
	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_EPI_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);
		
	PerformRegistrationEPIT1();
	
	h_Registration_Parameters_EPI_T1_Out[0] = h_Registration_Parameters_EPI_T1[0];
	h_Registration_Parameters_EPI_T1_Out[1] = h_Registration_Parameters_EPI_T1[1];
	h_Registration_Parameters_EPI_T1_Out[2] = h_Registration_Parameters_EPI_T1[2];
	h_Registration_Parameters_EPI_T1_Out[3] = h_Registration_Parameters_EPI_T1[3];
	h_Registration_Parameters_EPI_T1_Out[4] = h_Registration_Parameters_EPI_T1[4];
	h_Registration_Parameters_EPI_T1_Out[5] = h_Registration_Parameters_EPI_T1[5];
	
	clReleaseMemObject(d_MNI_T1_Volume);
	clReleaseMemObject(d_Skullstripped_T1_Volume);
	clReleaseMemObject(d_EPI_Volume);
	clReleaseMemObject(d_T1_EPI_Volume);

	//------------------------

	
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Motion_Corrected_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);
	

	
	//h_Motion_Parameters = (float*)malloc(EPI_DATA_T * 6 * sizeof(float));

	PerformMotionCorrection();
		
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		for (int p = 0; p < 6; p++)
		{
			h_Motion_Parameters_Out[t + p * EPI_DATA_T] = h_Motion_Parameters[t + p * EPI_DATA_T];
		}
	}
	
	clEnqueueReadBuffer(commandQueue, d_Motion_Corrected_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Motion_Corrected_fMRI_Volumes, 0, NULL, NULL);


	//-------------------------------

	d_Smoothed_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Allocate memory for smoothing filters
	c_Smoothing_Filter_X = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Y = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Z = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	
	// Copy smoothing filters to constant memory
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_X, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_X , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Y, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Y , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Z, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Z , 0, NULL, NULL);
	
	PerformSmoothing(d_Smoothed_fMRI_Volumes, d_Motion_Corrected_fMRI_Volumes, c_Smoothing_Filter_X, c_Smoothing_Filter_Y, c_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	
	clEnqueueReadBuffer(commandQueue, d_Smoothed_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Smoothed_fMRI_Volumes, 0, NULL, NULL);

	clReleaseMemObject(c_Smoothing_Filter_X);
	clReleaseMemObject(c_Smoothing_Filter_Y);
	clReleaseMemObject(c_Smoothing_Filter_Z);


	//-------------------------------	

	d_Mask = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);	
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_REGRESSORS * sizeof(float), NULL, NULL);	
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Beta_Contrasts = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);	
	d_Residuals = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	
	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM , 0, NULL, NULL);


	CalculateStatisticalMapsGLMFirstLevel(d_Smoothed_fMRI_Volumes);

	// Allocate memory on device
	d_Beta_Volumes_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_REGRESSORS * sizeof(float), NULL, &createBufferErrorBetaVolumesMNI);
	
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);


	TransformFirstLevelResultsToMNI();

	// Copy data to host
	clEnqueueReadBuffer(commandQueue, d_Beta_Volumes_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_REGRESSORS * sizeof(float), h_Beta_Volumes, 0, NULL, NULL);	
	//clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_REGRESSORS * sizeof(float), h_Beta_Volumes, 0, NULL, NULL);	
	//clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Residuals, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);

	clReleaseMemObject(d_MNI_Brain_Mask);


	//free(h_Motion_Parameters);
	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Motion_Corrected_fMRI_Volumes);
	clReleaseMemObject(d_Smoothed_fMRI_Volumes);

	clReleaseMemObject(d_Mask);
	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);

	

	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Beta_Contrasts);
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);

	clReleaseMemObject(d_Beta_Volumes_MNI);

	//CalculateSlicesPreprocessedfMRIData();
}

void BROCCOLI_LIB::TransformFirstLevelResultsToMNI()
{
	ChangeVolumesResolutionAndSize(d_Beta_Volumes_MNI, d_Beta_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_REGRESSORS, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT);       	
		
	h_Registration_Parameters_EPI_MNI[0] = h_Registration_Parameters_EPI_T1_Affine[0] + h_Registration_Parameters_T1_MNI[0]; 
	h_Registration_Parameters_EPI_MNI[1] = h_Registration_Parameters_EPI_T1_Affine[1] + h_Registration_Parameters_T1_MNI[1]; 
	h_Registration_Parameters_EPI_MNI[2] = h_Registration_Parameters_EPI_T1_Affine[2] + h_Registration_Parameters_T1_MNI[2]; 

	h_Registration_Parameters_EPI_MNI[3] = h_Registration_Parameters_EPI_T1_Affine[3] + h_Registration_Parameters_T1_MNI[3];
	h_Registration_Parameters_EPI_MNI[4] = h_Registration_Parameters_EPI_T1_Affine[4] + h_Registration_Parameters_T1_MNI[4];
	h_Registration_Parameters_EPI_MNI[5] = h_Registration_Parameters_EPI_T1_Affine[5] + h_Registration_Parameters_T1_MNI[5];
	h_Registration_Parameters_EPI_MNI[6] = h_Registration_Parameters_EPI_T1_Affine[6] + h_Registration_Parameters_T1_MNI[6];
	h_Registration_Parameters_EPI_MNI[7] = h_Registration_Parameters_EPI_T1_Affine[7] + h_Registration_Parameters_T1_MNI[7];
	h_Registration_Parameters_EPI_MNI[8] = h_Registration_Parameters_EPI_T1_Affine[8] + h_Registration_Parameters_T1_MNI[8];
	h_Registration_Parameters_EPI_MNI[9] = h_Registration_Parameters_EPI_T1_Affine[9] + h_Registration_Parameters_T1_MNI[9];
	h_Registration_Parameters_EPI_MNI[10] = h_Registration_Parameters_EPI_T1_Affine[10] + h_Registration_Parameters_T1_MNI[10];
	h_Registration_Parameters_EPI_MNI[11] = h_Registration_Parameters_EPI_T1_Affine[11] + h_Registration_Parameters_T1_MNI[11];

	TransformVolumes(d_Beta_Volumes_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_REGRESSORS, LINEAR);	

	SetGlobalAndLocalWorkSizesMultiplyVolumes(MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	clSetKernelArg(MultiplyVolumesOverwriteKernel, 0, sizeof(cl_mem), &d_Beta_Volumes_MNI);
	clSetKernelArg(MultiplyVolumesOverwriteKernel, 1, sizeof(cl_mem), &d_MNI_Brain_Mask);
	clSetKernelArg(MultiplyVolumesOverwriteKernel, 2, sizeof(int), &MNI_DATA_W);
	clSetKernelArg(MultiplyVolumesOverwriteKernel, 3, sizeof(int), &MNI_DATA_H);
	clSetKernelArg(MultiplyVolumesOverwriteKernel, 4, sizeof(int), &MNI_DATA_D);
	
	runKernelErrorMultiplyVolumes = clEnqueueNDRangeKernel(commandQueue, MultiplyVolumesOverwriteKernel, 3, NULL, globalWorkSizeMultiplyVolumes, localWorkSizeMultiplyVolumes, 0, NULL, NULL);
	clFinish(commandQueue);	
}

// Performs slice timing correction of an fMRI dataset
void BROCCOLI_LIB::PerformSliceTimingCorrection()
{
	
}


void BROCCOLI_LIB::PerformMotionCorrectionWrapper()
{
	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Motion_Corrected_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Copy volumes to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

	clFinish(commandQueue);

	// Setup all parameters and allocate memory on device
	AlignTwoVolumesSetup(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Set the first volume as the reference volume
	clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Reference_Volume, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

	// Copy the first volume to the corrected volumes
	clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Motion_Corrected_fMRI_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);	

	// Translations
	h_Motion_Parameters_Out[0 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters_Out[1 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters_Out[2 * EPI_DATA_T] = 0.0f;

	// Rotations
	h_Motion_Parameters_Out[3 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters_Out[4 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters_Out[5 * EPI_DATA_T] = 0.0f;

	// Run the registration for each volume
	for (int t = 1; t < EPI_DATA_T; t++)
	{
		// Set a new volume to be aligned
		clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Aligned_Volume, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

		// Also copy the same volume to an image to interpolate from
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {EPI_DATA_W, EPI_DATA_H, EPI_DATA_D};
		clEnqueueCopyBufferToImage(commandQueue, d_fMRI_Volumes, d_Original_Volume, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), origin, region, 0, NULL, NULL);
		
		// Do rigid registration with only one scale
		AlignTwoVolumes(h_Registration_Parameters_Motion_Correction, h_Rotations, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION, RIGID);

		// Copy the corrected volume to the corrected volumes
		clEnqueueCopyBuffer(commandQueue, d_Aligned_Volume, d_Motion_Corrected_fMRI_Volumes, 0, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);
	
		// Write the total parameter vector to host

		// Translations
		h_Motion_Parameters_Out[t + 0 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[0];
		h_Motion_Parameters_Out[t + 1 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[1];
		h_Motion_Parameters_Out[t + 2 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[2];

		// Rotations
		h_Motion_Parameters_Out[t + 3 * EPI_DATA_T] = h_Rotations[0];
		h_Motion_Parameters_Out[t + 4 * EPI_DATA_T] = h_Rotations[1];
		h_Motion_Parameters_Out[t + 5 * EPI_DATA_T] = h_Rotations[2];
	}
		
	// Copy all corrected volumes to host
	clEnqueueReadBuffer(commandQueue, d_Motion_Corrected_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Motion_Corrected_fMRI_Volumes, 0, NULL, NULL);

	// Cleanup allocated memory
	AlignTwoVolumesCleanup();

	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Motion_Corrected_fMRI_Volumes);
}

// Performs motion correction of an fMRI dataset
void BROCCOLI_LIB::PerformMotionCorrection()
{
	// Setup all parameters and allocate memory on host
	AlignTwoVolumesSetup(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Set the first volume as the reference volume
	clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Reference_Volume, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

	// Copy the first volume to the corrected volumes
	clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Motion_Corrected_fMRI_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);
	
	// Translations
	h_Motion_Parameters[0 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters[1 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters[2 * EPI_DATA_T] = 0.0f;

	// Rotations
	h_Motion_Parameters[3 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters[4 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters[5 * EPI_DATA_T] = 0.0f;

	// Run the registration for each volume
	for (int t = 1; t < EPI_DATA_T; t++)
	{
		// Set a new volume to be aligned
		clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Aligned_Volume, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

		// Also copy the same volume to an image to interpolate from
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {EPI_DATA_W, EPI_DATA_H, EPI_DATA_D};
		clEnqueueCopyBufferToImage(commandQueue, d_fMRI_Volumes, d_Original_Volume, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), origin, region, 0, NULL, NULL);
		
		// Do rigid registration with only one scale
		AlignTwoVolumes(h_Registration_Parameters_Motion_Correction, h_Rotations, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION, RIGID);

		// Copy the corrected volume to the corrected volumes
		clEnqueueCopyBuffer(commandQueue, d_Aligned_Volume, d_Motion_Corrected_fMRI_Volumes, 0, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);
	
		// Write the total parameter vector to host
		
		// Translations
		h_Motion_Parameters[t + 0 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[0];
		h_Motion_Parameters[t + 1 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[1];
		h_Motion_Parameters[t + 2 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[2];

		// Rotations
		h_Motion_Parameters[t + 3 * EPI_DATA_T] = h_Rotations[0];
		h_Motion_Parameters[t + 4 * EPI_DATA_T] = h_Rotations[1];
		h_Motion_Parameters[t + 5 * EPI_DATA_T] = h_Rotations[2];
	}
		
	// Cleanup allocated memory
	AlignTwoVolumesCleanup();
}





void BROCCOLI_LIB::PerformSmoothingWrapper()
{
	SetGlobalAndLocalWorkSizesSeparableConvolution(EPI_DATA_W,EPI_DATA_H,EPI_DATA_D);

	// Allocate memory for smoothing filters
	c_Smoothing_Filter_X = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Y = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Z = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);

	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Smoothed_fMRI_Volumes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Copy volumes to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

	// Allocate temporary memory
	cl_mem d_Convolved_Rows = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Convolved_Columns = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	// Copy smoothing filters to constant memory
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_X, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_X , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Y, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Y , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Z, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Z , 0, NULL, NULL);

	// Set arguments for the kernels
	clSetKernelArg(SeparableConvolutionRowsKernel, 0, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionRowsKernel, 1, sizeof(cl_mem), &d_fMRI_Volumes);
	clSetKernelArg(SeparableConvolutionRowsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_Y);
	clSetKernelArg(SeparableConvolutionRowsKernel, 4, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(SeparableConvolutionRowsKernel, 5, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(SeparableConvolutionRowsKernel, 6, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(SeparableConvolutionRowsKernel, 7, sizeof(int), &EPI_DATA_T);

	clSetKernelArg(SeparableConvolutionColumnsKernel, 0, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 1, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_X);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 4, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 5, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 6, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 7, sizeof(int), &EPI_DATA_T);

	clSetKernelArg(SeparableConvolutionRodsKernel, 0, sizeof(cl_mem), &d_Smoothed_fMRI_Volumes);
	clSetKernelArg(SeparableConvolutionRodsKernel, 1, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionRodsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_Z);
    clSetKernelArg(SeparableConvolutionRodsKernel, 4, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(SeparableConvolutionRodsKernel, 5, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(SeparableConvolutionRodsKernel, 6, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(SeparableConvolutionRodsKernel, 7, sizeof(int), &EPI_DATA_T);

	// Loop over volumes
	for (int v = 0; v < EPI_DATA_T; v++)
	{		
		clSetKernelArg(SeparableConvolutionRowsKernel, 3, sizeof(int), &v);
		kernel_error = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRowsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRows, localWorkSizeSeparableConvolutionRows, 0, NULL, NULL);
		clFinish(commandQueue);
			
		clSetKernelArg(SeparableConvolutionColumnsKernel, 3, sizeof(int), &v);
		kernel_error = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionColumnsKernel, 3, NULL, globalWorkSizeSeparableConvolutionColumns, localWorkSizeSeparableConvolutionColumns, 0, NULL, NULL);
		clFinish(commandQueue);
		
		clSetKernelArg(SeparableConvolutionRodsKernel, 3, sizeof(int), &v);
		kernel_error = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRodsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRods, localWorkSizeSeparableConvolutionRods, 0, NULL, NULL);
		clFinish(commandQueue);	
	}
	
	// Copy result back to host
	clEnqueueReadBuffer(commandQueue, d_Smoothed_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Result, 0, NULL, NULL);
	
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
void BROCCOLI_LIB::PerformSmoothing(cl_mem d_Smoothed_Volumes, cl_mem d_Volumes, cl_mem c_Smoothing_Filter_X, cl_mem c_Smoothing_Filter_Y, cl_mem c_Smoothing_Filter_Z, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
	SetGlobalAndLocalWorkSizesSeparableConvolution(DATA_W,DATA_H,DATA_D);

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
	for (int v = 0; v < DATA_T; v++)
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

// Performs smoothing of a number of volumes, overwrites data
void BROCCOLI_LIB::PerformSmoothing(cl_mem d_Volumes, cl_mem c_Smoothing_Filter_X, cl_mem c_Smoothing_Filter_Y, cl_mem c_Smoothing_Filter_Z, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
	SetGlobalAndLocalWorkSizesSeparableConvolution(DATA_W,DATA_H,DATA_D);

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

	clSetKernelArg(SeparableConvolutionRodsKernel, 0, sizeof(cl_mem), &d_Volumes);
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
void BROCCOLI_LIB::PerformFirstLevelAnalysis()
{
	PerformRegistrationT1MNI();
	PerformRegistrationEPIT1();
	//PerformSliceTimingCorrection();
	PerformMotionCorrection();
	//PerformSmoothing();	
	//PerformDetrending();
	CalculateStatisticalMapsGLMFirstLevel(d_Smoothed_fMRI_Volumes);

	//CalculateSlicesPreprocessedfMRIData();
}


void BROCCOLI_LIB::PerformGLMWrapper()
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Mask = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);	
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_REGRESSORS * sizeof(float), NULL, NULL);	
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Beta_Contrasts = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);	
	d_Residuals = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	
	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Mask , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM , 0, NULL, NULL);

	clFinish(commandQueue);

	// Calculate beta values
	clSetKernelArg(CalculateBetaValuesGLMKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 1, sizeof(cl_mem), &d_fMRI_Volumes);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_GLM);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 4, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 5, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 6, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 7, sizeof(int), &EPI_DATA_T);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 8, sizeof(int), &NUMBER_OF_REGRESSORS);
	//clSetKernelArg(CalculateBetaValuesGLMKernel, 4, sizeof(cl_mem), &c_Censor);
	//kernel_error = clEnqueueNDRangeKernel(commandQueue, CalculateBetaValuesGLMKernel, 3, NULL, globalWorkSizeCalculateBetaValuesGLM, localWorkSizeCalculateBetaValuesGLM, 0, NULL, NULL);
	kernel_error = clEnqueueNDRangeKernel(commandQueue, CalculateBetaValuesGLMKernel, 3, NULL, globalWorkSizeCalculateBetaValuesGLM, localWorkSizeCalculateBetaValuesGLM, 0, NULL, &event);
	clFinish(commandQueue);
	
	
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	convolution_time += time_end - time_start;

	// Calculate t-values and residuals
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 1, sizeof(cl_mem), &d_Beta_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 2, sizeof(cl_mem), &d_Residuals);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 3, sizeof(cl_mem), &d_Residual_Variances);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 4, sizeof(cl_mem), &d_fMRI_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 5, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 6, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 7, sizeof(cl_mem), &c_X_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 8, sizeof(cl_mem), &c_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 9, sizeof(cl_mem), &c_ctxtxc_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 10, sizeof(int),   &EPI_DATA_W);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 11, sizeof(int),   &EPI_DATA_H);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 12, sizeof(int),   &EPI_DATA_D);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 13, sizeof(int),   &EPI_DATA_T);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 14, sizeof(int),   &NUMBER_OF_REGRESSORS);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 15, sizeof(int),   &NUMBER_OF_CONTRASTS);
	//clSetKernelArg(CalculateStatisticalMapsGLMKernel, 10, sizeof(cl_mem), &c_Censor);
	//kernel_error = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	kernel_error = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, &event);
	clFinish(commandQueue);

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	convolution_time += time_end - time_start;

	clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_REGRESSORS * sizeof(float), h_Beta_Volumes, 0, NULL, NULL);	
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Residuals, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);

	// Estimate auto correlation from residuals

	// Remove auto correlation from regressors and data

	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Mask);
	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);

	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Beta_Contrasts);
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);
}

// Calculates a statistical map for first level analysis
void BROCCOLI_LIB::CalculateStatisticalMapsGLMFirstLevel(cl_mem d_Volumes)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);	

	// Calculate beta values
	clSetKernelArg(CalculateBetaValuesGLMKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_GLM);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 4, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 5, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 6, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 7, sizeof(int), &EPI_DATA_T);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 8, sizeof(int), &NUMBER_OF_REGRESSORS);
	//clSetKernelArg(CalculateBetaValuesGLMKernel, 4, sizeof(cl_mem), &c_Censor);
	//kernel_error = clEnqueueNDRangeKernel(commandQueue, CalculateBetaValuesGLMKernel, 3, NULL, globalWorkSizeCalculateBetaValuesGLM, localWorkSizeCalculateBetaValuesGLM, 0, NULL, NULL);
	kernel_error = clEnqueueNDRangeKernel(commandQueue, CalculateBetaValuesGLMKernel, 3, NULL, globalWorkSizeCalculateBetaValuesGLM, localWorkSizeCalculateBetaValuesGLM, 0, NULL, &event);
	clFinish(commandQueue);
	
	
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	convolution_time += time_end - time_start;

	// Calculate t-values and residuals
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 1, sizeof(cl_mem), &d_Beta_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 2, sizeof(cl_mem), &d_Residuals);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 3, sizeof(cl_mem), &d_Residual_Variances);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 4, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 5, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 6, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 7, sizeof(cl_mem), &c_X_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 8, sizeof(cl_mem), &c_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 9, sizeof(cl_mem), &c_ctxtxc_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 10, sizeof(int),   &EPI_DATA_W);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 11, sizeof(int),   &EPI_DATA_H);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 12, sizeof(int),   &EPI_DATA_D);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 13, sizeof(int),   &EPI_DATA_T);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 14, sizeof(int),   &NUMBER_OF_REGRESSORS);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 15, sizeof(int),   &NUMBER_OF_CONTRASTS);
	//clSetKernelArg(CalculateStatisticalMapsGLMKernel, 10, sizeof(cl_mem), &c_Censor);
	//kernel_error = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	kernel_error = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, &event);
	clFinish(commandQueue);

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	convolution_time += time_end - time_start;

	
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
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 2, sizeof(cl_mem), &d_Residuals);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 3, sizeof(cl_mem), &d_Residual_Variances);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 4, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 5, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 6, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 7, sizeof(cl_mem), &c_X_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 8, sizeof(cl_mem), &c_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMKernel, 9, sizeof(cl_mem), &c_ctxtxc_GLM);
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
        CalculateStatisticalMapPermutation();
		h_Maximum_Test_Values[p] = FindMaxTestvaluePermutation();  
    }

	
    // Sort the maximum test values
	
	// Find the threshold for the significance level
	
	NUMBER_OF_SIGNIFICANTLY_ACTIVE_VOXELS = 0;
	for (int i = 0; i < EPI_DATA_W * EPI_DATA_H * EPI_DATA_D; i++)
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
        CalculateStatisticalMapPermutation();
		h_Maximum_Test_Values[p] = FindMaxTestvaluePermutation();  
    }

	
    // Sort the maximum test values
	
	// Find the threshold for the significance level
	
	NUMBER_OF_SIGNIFICANTLY_ACTIVE_VOXELS = 0;
	for (int i = 0; i < MNI_DATA_W * MNI_DATA_H * MNI_DATA_D; i++)
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
        for (int i = 0; i < EPI_DATA_T; i++)
        {			
            h_Permutation_Matrix[i + p * EPI_DATA_T] = (unsigned short int)i;
        }

		// Generate random number and switch position of two existing numbers
        for (int i = 0; i < EPI_DATA_T; i++)
        {			
            int j = rand() % (EPI_DATA_T - i) + i;
            unsigned short int temp = h_Permutation_Matrix[j + p * EPI_DATA_T];
            h_Permutation_Matrix[j + p * EPI_DATA_T] = h_Permutation_Matrix[i + p * EPI_DATA_T];
            h_Permutation_Matrix[i + p * EPI_DATA_T] = temp;
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

	PerformSmoothing(d_Smoothed_AR1_Estimates, d_AR1_Estimates, c_AR_Smoothing_Filter_X, c_AR_Smoothing_Filter_Y, c_AR_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
	PerformSmoothing(d_Smoothed_AR2_Estimates, d_AR2_Estimates, c_AR_Smoothing_Filter_X, c_AR_Smoothing_Filter_Y, c_AR_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
	PerformSmoothing(d_Smoothed_AR3_Estimates, d_AR3_Estimates, c_AR_Smoothing_Filter_X, c_AR_Smoothing_Filter_Y, c_AR_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
	PerformSmoothing(d_Smoothed_AR4_Estimates, d_AR4_Estimates, c_AR_Smoothing_Filter_X, c_AR_Smoothing_Filter_Y, c_AR_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
	
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

void BROCCOLI_LIB::CalculateStatisticalMapPermutation()
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

            unsigned short int t = h_Permutation_Matrix[j + p * NUMBER_OF_SUBJECTS];
            h_Permutation_Matrix[j + p * NUMBER_OF_SUBJECTS] = h_Permutation_Matrix[i + p * NUMBER_OF_SUBJECTS];
            h_Permutation_Matrix[i + p * NUMBER_OF_SUBJECTS] = t;
        }
    }
}






// Read functions, public


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
			data[i] = current_value;
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
			data[i] = current_value;
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

/*
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

	X_SLICE_LOCATION_fMRI_DATA = EPI_DATA_W / 2;
	Y_SLICE_LOCATION_fMRI_DATA = EPI_DATA_H / 2;
	Z_SLICE_LOCATION_fMRI_DATA = EPI_DATA_D / 2;
	TIMEPOINT_fMRI_DATA = 0;
	
	h_fMRI_Volumes = (float*)malloc(DATA_SIZE_FMRI_VOLUMES);
	h_Motion_Corrected_fMRI_Volumes = (float*)malloc(DATA_SIZE_FMRI_VOLUMES);
	h_Smoothed_fMRI_Volumes = (float*)malloc(DATA_SIZE_FMRI_VOLUMES);
	h_Detrended_fMRI_Volumes = (float*)malloc(DATA_SIZE_FMRI_VOLUMES);
	
	h_X_Detrend = (float*)malloc(DATA_SIZE_DETRENDING);
	h_xtxxt_Detrend = (float*)malloc(DATA_SIZE_DETRENDING);	

	h_X_GLM = (float*)malloc(DATA_SIZE_TEMPORAL_BASIS_FUNCTIONS);
	h_xtxxt_GLM = (float*)malloc(DATA_SIZE_TEMPORAL_BASIS_FUNCTIONS);
	h_Contrasts = (float*)malloc(sizeof(float) * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS * NUMBER_OF_CONTRASTS);
	
	//h_Activity_Volume = (float*)malloc(DATA_SIZE_fMRI_VOLUME);
	
	host_pointers[fMRI_VOLUMES] = (void*)h_fMRI_Volumes;
	host_pointers[MOTION_CORRECTED_VOLUMES] = (void*)h_Motion_Corrected_fMRI_Volumes;
	host_pointers[SMOOTHED1] = (void*)h_Smoothed_fMRI_Volumes;
	host_pointers[DETRENDED1] = (void*)h_Detrended_fMRI_Volumes;
	host_pointers[XDETREND1] = (void*)h_X_Detrend;
	host_pointers[XDETREND2] = (void*)h_xtxxt_Detrend;
	host_pointers[XGLM1] = (void*)h_X_GLM;
	host_pointers[XGLM2] = (void*)h_xtxxt_GLM;
	host_pointers[CONTRAST_VECTOR] = (void*)h_Contrasts;
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
*/

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
			current_value = data[i];
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
			current_value = data[i];
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
	float max = CalculateMax(h_fMRI_Volumes, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T);
	float min = CalculateMin(h_fMRI_Volumes, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T);
	float adder = -min;
	float multiplier = 255.0f / (max + adder);

	for (int x = 0; x < EPI_DATA_W; x++)
	{
		for (int y = 0; y < EPI_DATA_H; y++)
		{
			z_slice_fMRI_data[x + y * EPI_DATA_W] = (unsigned char)((h_fMRI_Volumes[x + y * EPI_DATA_W + Z_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W * EPI_DATA_H + TIMEPOINT_fMRI_DATA * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] + adder) * multiplier);
		}
	}

	for (int x = 0; x < EPI_DATA_W; x++)
	{
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			int inv_z = EPI_DATA_D - 1 - z;
			y_slice_fMRI_data[x + inv_z * EPI_DATA_W] = (unsigned char)((h_fMRI_Volumes[x + Y_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + TIMEPOINT_fMRI_DATA * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] + adder) * multiplier);
		}
	}

	for (int y = 0; y < EPI_DATA_H; y++)
	{
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			int inv_z = EPI_DATA_D - 1 - z;
			x_slice_fMRI_data[y + inv_z * EPI_DATA_H] = (unsigned char)((h_fMRI_Volumes[X_SLICE_LOCATION_fMRI_DATA + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + TIMEPOINT_fMRI_DATA * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] + adder) * multiplier);
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

	float max = CalculateMax(pointer, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T);
	float min = CalculateMin(pointer, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T);
	float adder = -min;
	float multiplier = 255.0f / (max + adder);


	for (int x = 0; x < EPI_DATA_W; x++)
	{
		for (int y = 0; y < EPI_DATA_H; y++)
		{
			z_slice_preprocessed_fMRI_data[x + y * EPI_DATA_W] = (unsigned char)((pointer[x + y * EPI_DATA_W + Z_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W * EPI_DATA_H + TIMEPOINT_fMRI_DATA * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] + adder) * multiplier);
		}
	}

	for (int x = 0; x < EPI_DATA_W; x++)
	{
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			int inv_z = EPI_DATA_D - 1 - z;
			y_slice_preprocessed_fMRI_data[x + inv_z * EPI_DATA_W] = (unsigned char)((pointer[x + Y_SLICE_LOCATION_fMRI_DATA * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + TIMEPOINT_fMRI_DATA * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] + adder) * multiplier);
		}
	}

	for (int y = 0; y < EPI_DATA_H; y++)
	{
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			int inv_z = EPI_DATA_D - 1 - z;
			x_slice_preprocessed_fMRI_data[y + inv_z * EPI_DATA_H] = (unsigned char)((pointer[X_SLICE_LOCATION_fMRI_DATA + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + TIMEPOINT_fMRI_DATA * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] + adder) * multiplier);
		}
	}
}

/*
void BROCCOLI_LIB::Convert4FloatToFloat4(float4* floats, float* float_1, float* float_2, float* float_3, float* float_4, int N)
{
	for (int i = 0; i < N; i++)
	{
		floats[i] = float_1[i];
		floats[i] = float_2[i];
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
		floats[i] = float_1[i];
		floats[i] = float_2[i];
	}
}
*/

/*
void BROCCOLI_LIB::ConvertRealToComplex(Complex* complex_data, float* real_data, int N)
{
	for (int i = 0; i < N; i++)
	{
		complex_data[i] = real_data[i];
		complex_data[i] = 0.0f;
	}
}
*/

/*
void BROCCOLI_LIB::ExtractRealData(float* real_data, Complex* complex_data, int N)
{
	for (int i = 0; i < N; i++)
	{
		real_data[i] = complex_data[i];
	}
}
*/


void BROCCOLI_LIB::InvertMatrix(float* inverse_matrix, float* matrix, int N)
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
        piv[i] = (float)i;
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
            int t = (int)piv[p]; piv[p] = piv[k]; piv[k] = (float)t;
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
        int current_row = (int)piv[i];
        
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

void BROCCOLI_LIB::CalculateMatrixSquareRoot(float* sqrt_matrix, float* matrix, int N)
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
		InvertMatrix(tempinv, sqrt_matrix, N);
		sqrt_matrix[0] = 0.5f * sqrt_matrix[0] + 0.5f * (matrix[0] * tempinv[0] + matrix[1] * tempinv[2]);
		sqrt_matrix[1] = 0.5f * sqrt_matrix[1] + 0.5f * (matrix[0] * tempinv[1] + matrix[1] * tempinv[3]);
		sqrt_matrix[2] = 0.5f * sqrt_matrix[2] + 0.5f * (matrix[2] * tempinv[0] + matrix[3] * tempinv[2]);
		sqrt_matrix[3] = 0.5f * sqrt_matrix[3] + 0.5f * (matrix[2] * tempinv[1] + matrix[3] * tempinv[3]);
	}

	free(tempinv);
}

void BROCCOLI_LIB::SolveEquationSystem(float* h_A_matrix, float* h_inverse_A_matrix, float* h_h_vector, float* h_Parameter_Vector, int N)
{
	InvertMatrix(h_inverse_A_matrix, h_A_matrix, N);

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
	float offset = -((float)EPI_DATA_T - 1.0f)/2.0f;
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		h_X_Detrend[t + 0 * EPI_DATA_T] = 1.0f;
		h_X_Detrend[t + 1 * EPI_DATA_T] = offset + (float)t;
	}

	// X^2 and X^3
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		h_X_Detrend[t + 2 * EPI_DATA_T] = h_X_Detrend[t + 1 * EPI_DATA_T] * h_X_Detrend[t + 1 * EPI_DATA_T];
		h_X_Detrend[t + 3 * EPI_DATA_T] = h_X_Detrend[t + 1 * EPI_DATA_T] * h_X_Detrend[t + 1 * EPI_DATA_T] * h_X_Detrend[t + 1 * EPI_DATA_T];
	}

	// Normalize

	// 1
	float norm = 0.0f;
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		norm += h_X_Detrend[t + 0 * EPI_DATA_T] * h_X_Detrend[t + 0 * EPI_DATA_T];
	}
	norm = sqrt(norm);
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		h_X_Detrend[t + 0 * EPI_DATA_T] /= norm;
	}

	// X
	norm = 0.0f;
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		norm += h_X_Detrend[t + 1 * EPI_DATA_T] * h_X_Detrend[t + 1 * EPI_DATA_T];
	}
	norm = sqrt(norm);
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		h_X_Detrend[t + 1 * EPI_DATA_T] /= norm;
	}

	// X^2
	norm = 0.0f;
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		norm += h_X_Detrend[t + 2 * EPI_DATA_T] * h_X_Detrend[t + 2 * EPI_DATA_T];
	}
	norm = sqrt(norm);
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		h_X_Detrend[t + 2 * EPI_DATA_T] /= norm;
	}

	// X^3
	norm = 0.0f;
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		norm += h_X_Detrend[t + 3 * EPI_DATA_T] * h_X_Detrend[t + 3 * EPI_DATA_T];
	}
	norm = sqrt(norm);
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		h_X_Detrend[t + 3 * EPI_DATA_T] /= norm;
	}

	// Calculate X_Detrend'*X_Detrend
	float xtx[16];
	float inv_xtx[16];

	for (int i = 0; i < NUMBER_OF_DETRENDING_BASIS_FUNCTIONS; i++)
	{
		for (int j = 0; j < NUMBER_OF_DETRENDING_BASIS_FUNCTIONS; j++)
		{
			xtx[i + j * NUMBER_OF_DETRENDING_BASIS_FUNCTIONS] = 0.0f;
			for (int t = 0; t < EPI_DATA_T; t++)
			{
				xtx[i + j * NUMBER_OF_DETRENDING_BASIS_FUNCTIONS] += h_X_Detrend[t + i * EPI_DATA_T] * h_X_Detrend[t + j * EPI_DATA_T];
			}
		}
	}

	// Calculate inverse of X_Detrend'*X_Detrend
	InvertMatrix(inv_xtx, xtx, NUMBER_OF_DETRENDING_BASIS_FUNCTIONS);

	// Calculate inv(X_Detrend'*X_Detrend)*X_Detrend'
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		h_xtxxt_Detrend[t + 0 * EPI_DATA_T] = inv_xtx[0] * h_X_Detrend[t + 0 * EPI_DATA_T] + inv_xtx[1] * h_X_Detrend[t + 1 * EPI_DATA_T] + inv_xtx[2] * h_X_Detrend[t + 2 * EPI_DATA_T] + inv_xtx[3] * h_X_Detrend[t + 3 * EPI_DATA_T];
		h_xtxxt_Detrend[t + 1 * EPI_DATA_T] = inv_xtx[4] * h_X_Detrend[t + 0 * EPI_DATA_T] + inv_xtx[5] * h_X_Detrend[t + 1 * EPI_DATA_T] + inv_xtx[6] * h_X_Detrend[t + 2 * EPI_DATA_T] + inv_xtx[7] * h_X_Detrend[t + 3 * EPI_DATA_T];
		h_xtxxt_Detrend[t + 2 * EPI_DATA_T] = inv_xtx[8] * h_X_Detrend[t + 0 * EPI_DATA_T] + inv_xtx[9] * h_X_Detrend[t + 1 * EPI_DATA_T] + inv_xtx[10] * h_X_Detrend[t + 2 * EPI_DATA_T] + inv_xtx[11] * h_X_Detrend[t + 3 * EPI_DATA_T];
		h_xtxxt_Detrend[t + 3 * EPI_DATA_T] = inv_xtx[12] * h_X_Detrend[t + 0 * EPI_DATA_T] + inv_xtx[13] * h_X_Detrend[t + 1 * EPI_DATA_T] + inv_xtx[14] * h_X_Detrend[t + 2 * EPI_DATA_T] + inv_xtx[15] * h_X_Detrend[t + 3 * EPI_DATA_T];
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


void BROCCOLI_LIB::Cross(double *z, const double *x, const double *y) 
{
	z[0] = x[1]*y[2]-x[2]*y[1];
	z[1] = -(x[0]*y[2]-x[2]*y[0]);
	z[2] = x[0]*y[1]-x[1]*y[0];
}

void BROCCOLI_LIB::Sort3(double *x) 
{
	double tmp;

	if (x[0] < x[1]) 
	{
		tmp = x[0];
		x[0] = x[1];
		x[1] = tmp;
	}
	if (x[1] < x[2]) 
	{
		if (x[0] < x[2]) 
		{
			tmp = x[2];
			x[2] = x[1];
			x[1] = x[0];
			x[0] = tmp;
		}
		else 
		{
			tmp = x[1];
			x[1] = x[2];
			x[2] = tmp;
		}
	}
}

void BROCCOLI_LIB::Unit3(double *x) 
{
	double tmp = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
	x[0] /= tmp;
	x[1] /= tmp;
	x[2] /= tmp;
}

void BROCCOLI_LIB::LDUBSolve3(double *x, const double *y, const double *LDU, const int *P) 
{
	x[P[2]] = y[2];
	x[P[1]] = y[1] - LDU[3*P[2]+1]*x[P[2]];
	x[P[0]] = y[0] - LDU[3*P[2]+0]*x[P[2]] - LDU[3*P[1]+0]*x[P[1]];
}

void BROCCOLI_LIB::MatMul3x3(double *C, const double *A, const double *B) 
{
	C[3*0+0] = A[3*0+0]*B[3*0+0] + A[3*1+0]*B[3*0+1] + A[3*2+0]*B[3*0+2];
	C[3*1+0] = A[3*0+0]*B[3*1+0] + A[3*1+0]*B[3*1+1] + A[3*2+0]*B[3*1+2];
	C[3*2+0] = A[3*0+0]*B[3*2+0] + A[3*1+0]*B[3*2+1] + A[3*2+0]*B[3*2+2];

	C[3*0+1] = A[3*0+1]*B[3*0+0] + A[3*1+1]*B[3*0+1] + A[3*2+1]*B[3*0+2];
	C[3*1+1] = A[3*0+1]*B[3*1+0] + A[3*1+1]*B[3*1+1] + A[3*2+1]*B[3*1+2];
	C[3*2+1] = A[3*0+1]*B[3*2+0] + A[3*1+1]*B[3*2+1] + A[3*2+1]*B[3*2+2];

	C[3*0+2] = A[3*0+2]*B[3*0+0] + A[3*1+2]*B[3*0+1] + A[3*2+2]*B[3*0+2];
	C[3*1+2] = A[3*0+2]*B[3*1+0] + A[3*1+2]*B[3*1+1] + A[3*2+2]*B[3*1+2];
	C[3*2+2] = A[3*0+2]*B[3*2+0] + A[3*1+2]*B[3*2+1] + A[3*2+2]*B[3*2+2];
}

void BROCCOLI_LIB::MatVec3(double *y, const double *A, const double *x) 
{
	y[0] = A[3*0+0]*x[0] + A[3*1+0]*x[1] + A[3*2+0]*x[2];
	y[1] = A[3*0+1]*x[0] + A[3*1+1]*x[1] + A[3*2+1]*x[2];
	y[2] = A[3*0+2]*x[0] + A[3*1+2]*x[1] + A[3*2+2]*x[2];
}

void BROCCOLI_LIB::A_Transpose_A3x3(double *AA, const double *A) 
{
	AA[3*0+0] = A[3*0+0]*A[3*0+0] + A[3*0+1]*A[3*0+1] + A[3*0+2]*A[3*0+2];
	AA[3*1+0] = A[3*0+0]*A[3*1+0] + A[3*0+1]*A[3*1+1] + A[3*0+2]*A[3*1+2];
	AA[3*2+0] = A[3*0+0]*A[3*2+0] + A[3*0+1]*A[3*2+1] + A[3*0+2]*A[3*2+2];

	AA[3*0+1] = AA[3*1+0];
	AA[3*1+1] = A[3*1+0]*A[3*1+0] + A[3*1+1]*A[3*1+1] + A[3*1+2]*A[3*1+2];
	AA[3*2+1] = A[3*1+0]*A[3*2+0] + A[3*1+1]*A[3*2+1] + A[3*1+2]*A[3*2+2];

	AA[3*0+2] = AA[3*2+0];
	AA[3*1+2] = AA[3*2+1];
	AA[3*2+2] = A[3*2+0]*A[3*2+0] + A[3*2+1]*A[3*2+1] + A[3*2+2]*A[3*2+2];
}

void BROCCOLI_LIB::A_A_Transpose3x3(double *AA, const double *A) 
{
	AA[3*0+0] = A[3*0+0]*A[3*0+0] + A[3*1+0]*A[3*1+0] + A[3*2+0]*A[3*2+0];
	AA[3*1+0] = A[3*0+0]*A[3*0+1] + A[3*1+0]*A[3*1+1] + A[3*2+0]*A[3*2+1];
	AA[3*2+0] = A[3*0+0]*A[3*0+2] + A[3*1+0]*A[3*1+2] + A[3*2+0]*A[3*2+2];

	AA[3*0+1] = AA[3*1+0];
	AA[3*1+1] = A[3*0+1]*A[3*0+1] + A[3*1+1]*A[3*1+1] + A[3*2+1]*A[3*2+1];
	AA[3*2+1] = A[3*0+1]*A[3*0+2] + A[3*1+1]*A[3*1+2] + A[3*2+1]*A[3*2+2];

	AA[3*0+2] = AA[3*2+0];
	AA[3*1+2] = AA[3*2+1];
	AA[3*2+2] = A[3*0+2]*A[3*0+2] + A[3*1+2]*A[3*1+2] + A[3*2+2]*A[3*2+2];
}

void BROCCOLI_LIB::Transpose3x3(double *A) 
{
	double tmp;

	tmp = A[3*1+0];
	A[3*1+0] = A[3*0+1];
	A[3*0+1] = tmp;

	tmp = A[3*2+0];
	A[3*2+0] = A[3*0+2];
	A[3*0+2] = tmp;

	tmp = A[3*2+1];
	A[3*2+1] = A[3*1+2];
	A[3*1+2] = tmp;
}

double BROCCOLI_LIB::cbrt(double x) 
{
  if (fabs(x) < DBL_EPSILON) return 0.0;

  if (x > 0.0) return pow(x, 1.0/3.0);

  return -pow(-x, 1.0/3.0);

}

void BROCCOLI_LIB::SolveCubic(double *c) 
{
	const double sq3d2 = 0.86602540378443864676, c2d3 = c[2]/3, c2sq = c[2]*c[2], Q = (3*c[1]-c2sq)/9, R = (c[2]*(9*c[1]-2*c2sq)-27*c[0])/54;
	double tmp, t, sint, cost;

	if (Q < 0) 
	{
		/* 
		 * Instead of computing
		 * c_0 = A cos(t) - B
		 * c_1 = A cos(t + 2 pi/3) - B
		 * c_2 = A cos(t + 4 pi/3) - B
		 * Use cos(a+b) = cos(a) cos(b) - sin(a) sin(b)
		 * Keeps t small and eliminates 1 function call.
		 * cos(2 pi/3) = cos(4 pi/3) = -0.5
		 * sin(2 pi/3) = sqrt(3)/2
		 * sin(4 pi/3) = -sqrt(3)/2
		 */

		tmp = 2*sqrt(-Q);
		t = acos(R/sqrt(-Q*Q*Q))/3;
		cost = tmp*cos(t);
		sint = tmp*sin(t);

		c[0] = cost - c2d3;

		cost = -0.5*cost - c2d3;
		sint = sq3d2*sint;

		c[1] = cost - sint;
		c[2] = cost + sint;
	}
	else 
	{
		tmp = cbrt(R);
		c[0] = -c2d3 + 2*tmp;
		c[1] = c[2] = -c2d3 - tmp;
	}
}

void BROCCOLI_LIB::LDU3(double *A, int *P) 
{
	int tmp;

	P[1] = 1;
	P[2] = 2;

	P[0] = fabs(A[3*1+0]) > fabs(A[3*0+0]) ? 
		(fabs(A[3*2+0]) > fabs(A[3*1+0]) ? 2 : 1) : 
		(fabs(A[3*2+0]) > fabs(A[3*0+0]) ? 2 : 0);
	P[P[0]] = 0;

	if (fabs(A[3*P[2]+1]) > fabs(A[3*P[1]+1])) 
	{
		tmp = P[1];
		P[1] = P[2];
		P[2] = tmp;
	}

	if (A[3*P[0]+0] != 0) 
	{
		A[3*P[1]+0] = A[3*P[1]+0]/A[3*P[0]+0];
		A[3*P[2]+0] = A[3*P[2]+0]/A[3*P[0]+0];
		A[3*P[0]+1] = A[3*P[0]+1]/A[3*P[0]+0];
		A[3*P[0]+2] = A[3*P[0]+2]/A[3*P[0]+0];
	}

	A[3*P[1]+1] = A[3*P[1]+1] - A[3*P[0]+1]*A[3*P[1]+0]*A[3*P[0]+0];

	if (A[3*P[1]+1] != 0) 
	{
		A[3*P[2]+1] = (A[3*P[2]+1] - A[3*P[0]+1]*A[3*P[2]+0]*A[3*P[0]+0])/A[3*P[1]+1];
		A[3*P[1]+2] = (A[3*P[1]+2] - A[3*P[0]+2]*A[3*P[1]+0]*A[3*P[0]+0])/A[3*P[1]+1];
	}

	A[3*P[2]+2] = A[3*P[2]+2] - A[3*P[0]+2]*A[3*P[2]+0]*A[3*P[0]+0] - A[3*P[1]+2]*A[3*P[2]+1]*A[3*P[1]+1];

}

void BROCCOLI_LIB::SVD3x3(double *U, double *S, double *V, const double *A) 
{
	const double thr = 1e-10;
	int P[3], k;
	double y[3], AA[3][3], LDU[3][3];

	/*
	 * Steps:
	 * 1) Use eigendecomposition on A^T A to compute V.
	 * Since A = U S V^T then A^T A = V S^T S V^T with D = S^T S and V the 
	 * eigenvalues and eigenvectors respectively (V is orthogonal).
	 * 2) Compute U from A and V.
	 * 3) Normalize columns of U and V and root the eigenvalues to obtain 
	 * the singular values.
	 */

	/* Compute AA = A^T A */
	A_Transpose_A3x3((double *)AA, A);

	/* Form the monic characteristic polynomial */
	S[2] = -AA[0][0] - AA[1][1] - AA[2][2];
	S[1] = AA[0][0]*AA[1][1] + AA[2][2]*AA[0][0] + AA[2][2]*AA[1][1] - 
		AA[2][1]*AA[1][2] - AA[2][0]*AA[0][2] - AA[1][0]*AA[0][1];
	S[0] = AA[2][1]*AA[1][2]*AA[0][0] + AA[2][0]*AA[0][2]*AA[1][1] + AA[1][0]*AA[0][1]*AA[2][2] -
		AA[0][0]*AA[1][1]*AA[2][2] - AA[1][0]*AA[2][1]*AA[0][2] - AA[2][0]*AA[0][1]*AA[1][2];

	/* Solve the cubic equation. */
	SolveCubic(S);

	/* All roots should be positive */
	if (S[0] < 0)
		S[0] = 0;
	if (S[1] < 0)
		S[1] = 0;
	if (S[2] < 0)
		S[2] = 0;

	/* Sort from greatest to least */
	Sort3(S);

	/* Form the eigenvector system for the first (largest) eigenvalue */
	memcpy(LDU,AA,sizeof(LDU));
	LDU[0][0] -= S[0];
	LDU[1][1] -= S[0];
	LDU[2][2] -= S[0];

	/* Perform LDUP decomposition */
	LDU3((double *)LDU, P);

	/* 
	 * Write LDU = AA-I*lambda.  Then an eigenvector can be
	 * found by solving LDU x = LD y = L z = 0
	 * L is invertible, so L z = 0 implies z = 0
	 * D is singular since det(AA-I*lambda) = 0 and so 
	 * D y = z = 0 has a non-unique solution.
	 * Pick k so that D_kk = 0 and set y = e_k, the k'th column
	 * of the identity matrix.
	 * U is invertible so U x = y has a unique solution for a given y.
	 * The solution for U x = y is an eigenvector.
	 */

	/* Pick the component of D nearest to 0 */
	y[0] = y[1] = y[2] = 0;
	k = fabs(LDU[P[1]][1]) < fabs(LDU[P[0]][0]) ?
		(fabs(LDU[P[2]][2]) < fabs(LDU[P[1]][1]) ? 2 : 1) :
		(fabs(LDU[P[2]][2]) < fabs(LDU[P[0]][0]) ? 2 : 0);
	y[k] = 1;

	/* Do a backward solve for the eigenvector */
	LDUBSolve3(V+(3*0+0), y, (double *)LDU, P);

	/* Form the eigenvector system for the last (smallest) eigenvalue */
	memcpy(LDU,AA,sizeof(LDU));
	LDU[0][0] -= S[2];
	LDU[1][1] -= S[2];
	LDU[2][2] -= S[2];

	/* Perform LDUP decomposition */
	LDU3((double *)LDU, P);

	/* 
	 * NOTE: The arrangement of the ternary operator output is IMPORTANT!
	 * It ensures a different system is solved if there are 3 repeat eigenvalues.
	 */

	/* Pick the component of D nearest to 0 */
	y[0] = y[1] = y[2] = 0;
	k = fabs(LDU[P[0]][0]) < fabs(LDU[P[2]][2]) ?
		(fabs(LDU[P[0]][0]) < fabs(LDU[P[1]][1]) ? 0 : 1) :
		(fabs(LDU[P[1]][1]) < fabs(LDU[P[2]][2]) ? 1 : 2);
	y[k] = 1;

	/* Do a backward solve for the eigenvector */
	LDUBSolve3(V+(3*2+0), y, (double *)LDU, P);

	 /* The remaining column must be orthogonal (AA is symmetric) */
	Cross(V+(3*1+0), V+(3*2+0), V+(3*0+0));

	/* Count the rank */
	k = (S[0] > thr) + (S[1] > thr) + (S[2] > thr);

	switch (k) 
	{
		case 0:
			/*
			 * Zero matrix. 
			 * Since V is already orthogonal, just copy it into U.
			 */
			memcpy(U,V,9*sizeof(double));
			break;
		case 1:
			/* 
			 * The first singular value is non-zero.
			 * Since A = U S V^T, then A V = U S.
			 * A V_1 = S_11 U_1 is non-zero. Here V_1 and U_1 are
			 * column vectors. Since V_1 is known, we may compute
			 * U_1 = A V_1.  The S_11 factor is not important as
			 * U_1 will be normalized later.
			 */
			MatVec3(U+(3*0+0), A, V+(3*0+0));

			/* 
			 * The other columns of U do not contribute to the expansion
			 * and we may arbitrarily choose them (but they do need to be
			 * orthogonal). To ensure the first cross product does not fail,
			 * pick k so that U_k1 is nearest 0 and then cross with e_k to
			 * obtain an orthogonal vector to U_1.
			 */
			y[0] = y[1] = y[2] = 0;
			k = fabs(U[3*0+0]) < fabs(U[3*0+2]) ?
				(fabs(U[3*0+0]) < fabs(U[3*0+1]) ? 0 : 1) :
				(fabs(U[3*0+1]) < fabs(U[3*0+2]) ? 1 : 2);
			y[k] = 1;

			Cross(U+(3*1+0), y, U+(3*0+0));

			/* Cross the first two to obtain the remaining column */
			Cross(U+(3*2+0), U+(3*0+0), U+(3*1+0));
			break;
		case 2:
			/*
			 * The first two singular values are non-zero.
			 * Compute U_1 = A V_1 and U_2 = A V_2. See case 1
			 * for more information.
			 */
			MatVec3(U+(3*0+0), A, V+(3*0+0));
			MatVec3(U+(3*1+0), A, V+(3*1+0));

			/* Cross the first two to obtain the remaining column */
			Cross(U+(3*2+0), U+(3*0+0), U+(3*1+0));
			break;
		case 3:
			/*
			 * All singular values are non-zero.
			 * We may compute U = A V. See case 1 for more information.
			 */
			MatMul3x3(U, A, V);
			break;
	}

	/* Normalize the columns of U and V */
	Unit3(V+(3*0+0));
	Unit3(V+(3*1+0));
	Unit3(V+(3*2+0));

	Unit3(U+(3*0+0));
	Unit3(U+(3*1+0));
	Unit3(U+(3*2+0));

	/* S was initially the eigenvalues of A^T A = V S^T S V^T which are squared. */
	S[0] = sqrt(S[0]);
	S[1] = sqrt(S[1]);
	S[2] = sqrt(S[2]);
}
