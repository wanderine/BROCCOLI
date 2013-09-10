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


#include <Dense>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
//#include <ifstream>
#include <string>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <vector>
#include <math.h>
#include <cfloat>

#include <opencl.h>

//#include <shrUtils.h>
//#include <shrQATest.h>
#include "broccoli_lib.h"

//#include "nifti1.h"
//#include "nifti1_io.h"

#include <cstdlib>




// public
float round( float d )
{
    return floor( d + 0.5f );
}

// Constructors

BROCCOLI_LIB::BROCCOLI_LIB()
{	
	OPENCL_INITIATED = 0;
	SetStartValues();
	ResetAllPointers();
}

BROCCOLI_LIB::BROCCOLI_LIB(cl_uint platform, cl_uint device)
{
	SetStartValues();
	OPENCL_INITIATED = 0;
	OpenCLInitiate(platform,device);	
	ResetAllPointers();
	//AllocateMemory();
	//ReadImageRegistrationFilters();
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
	EPI_Smoothing_FWHM = 8.0f;
	AR_Smoothing_FWHM = 8.0f;

	programBinarySize = 0;
	writtenElements = 0;

	BETA_SPACE = EPI;

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

	IMAGE_REGISTRATION_FILTER_SIZE = 7;
	NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION = 5;
	NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION = 5;
	NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS = 30;
	
	SMOOTHING_FILTER_SIZE = 9;
	
	NUMBER_OF_DETRENDING_REGRESSORS = 4;
	NUMBER_OF_MOTION_REGRESSORS = 6;

	SEGMENTATION_THRESHOLD = 600.0f;
	NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS = 2;
	NUMBER_OF_PERIODS = 4;
	PERIOD_TIME = 20;

	int DATA_SIZE_QUADRATURE_FILTER_REAL = sizeof(float) * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE;
	//int DATA_SIZE_QUADRATURE_FILTER_COMPLEX = sizeof(Complex) * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE;

	int DATA_SIZE_SMOOTHING_FILTER_GLM = sizeof(float) * SMOOTHING_FILTER_SIZE;

	error = 0;

	NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS = 12;

	convolution_time = 0.0;

	for (int i = 0; i < 50; i++)
	{
		OpenCLCreateBufferErrors[i] = 0;
		OpenCLRunKernelErrors[i] = 0;
		OpenCLCreateKernelErrors[i] = 0;
	}

	createBufferErrorAlignedVolume = 0;
	createBufferErrorReferenceVolume = 0;
	createBufferErrorq11Real = 0;
	createBufferErrorq11Imag = 0; 
	createBufferErrorq12Real = 0;
	createBufferErrorq12Imag = 0;
	createBufferErrorq13Real = 0;
	createBufferErrorq13Imag = 0;
	createBufferErrorq21Real = 0;
	createBufferErrorq21Imag = 0;
	createBufferErrorq22Real = 0;
	createBufferErrorq22Imag = 0;
	createBufferErrorq23Real = 0;
	createBufferErrorq23Imag = 0;
	createBufferErrorPhaseDifferences = 0;
	createBufferErrorPhaseCertainties = 0;
	createBufferErrorPhaseGradients = 0;
	createBufferErrorAMatrix = 0;
	createBufferErrorHVector = 0;
	createBufferErrorAMatrix2DValues = 0;
	createBufferErrorAMatrix1DValues = 0;
	createBufferErrorHVector2DValues = 0;
	createBufferErrorHVector1DValues = 0;
	createBufferErrorQuadratureFilter1Real = 0;
	createBufferErrorQuadratureFilter1Imag = 0;
	createBufferErrorQuadratureFilter2Real = 0;
	createBufferErrorQuadratureFilter2Imag = 0;
	createBufferErrorQuadratureFilter3Real = 0;
	createBufferErrorQuadratureFilter3Imag = 0;   
	createBufferErrorRegistrationParameters = 0;
	createBufferErrorBetaVolumesMNI = 0;
	createBufferErrorStatisticalMapsMNI = 0;
	createBufferErrorResidualVariancesMNI = 0;

	runKernelErrorNonseparableConvolution3DComplexThreeFilters = 0;
	runKernelErrorMemset = 0;
	runKernelErrorCalculatePhaseDifferencesAndCertainties = 0;
	runKernelErrorCalculatePhaseGradientsX = 0;
	runKernelErrorCalculatePhaseGradientsY = 0;
	runKernelErrorCalculatePhaseGradientsZ = 0;
	runKernelErrorCalculateAMatrixAndHVector2DValuesX = 0;
	runKernelErrorCalculateAMatrixAndHVector2DValuesY = 0;
	runKernelErrorCalculateAMatrixAndHVector2DValuesZ = 0;
	runKernelErrorCalculateAMatrix1DValues = 0;
	runKernelErrorCalculateHVector1DValues = 0;
	runKernelErrorCalculateAMatrix = 0;
	runKernelErrorCalculateHVector = 0;
	runKernelErrorInterpolateVolume = 0;
	runKernelErrorCalculateBetaValuesGLM = 0;
	runKernelErrorCalculateStatisticalMapsGLM = 0;
    runKernelErrorRescaleVolume = 0;
	runKernelErrorCopyVolume = 0;
	runKernelErrorEstimateAR4Models = 0;
	runKernelErrorApplyAR4Whitening = 0;

	createKernelErrorMemset = 0;
	createKernelErrorSeparableConvolutionRows = 0;
	createKernelErrorSeparableConvolutionColumns = 0;
	createKernelErrorSeparableConvolutionRods = 0;
	createKernelErrorNonseparableConvolution3DComplexThreeFilters = 0;
	createKernelErrorCalculatePhaseDifferencesAndCertainties = 0;
	createKernelErrorCalculatePhaseGradientsX = 0;
	createKernelErrorCalculatePhaseGradientsY = 0;
	createKernelErrorCalculatePhaseGradientsZ = 0;
	createKernelErrorCalculateAMatrixAndHVector2DValuesX = 0;
	createKernelErrorCalculateAMatrixAndHVector2DValuesY = 0;
	createKernelErrorCalculateAMatrixAndHVector2DValuesZ = 0;
	createKernelErrorCalculateAMatrix1DValues = 0;
	createKernelErrorCalculateHVector1DValues = 0;
	createKernelErrorCalculateAMatrix = 0;
	createKernelErrorCalculateHVector = 0;
	createKernelErrorInterpolateVolumeNearestParametric = 0;
	createKernelErrorInterpolateVolumeLinearParametric = 0;
	createKernelErrorInterpolateVolumeCubicParametric = 0;
	createKernelErrorInterpolateVolumeNearestNonParametric = 0;
	createKernelErrorInterpolateVolumeLinearNonParametric = 0;
	createKernelErrorInterpolateVolumeCubicNonParametric = 0;
	createKernelErrorRescaleVolumeNearest = 0;
	createKernelErrorRescaleVolumeLinear = 0;
	createKernelErrorRescaleVolumeCubic = 0;
	createKernelErrorCopyT1VolumeToMNI = 0;
	createKernelErrorCopyEPIVolumeToT1 = 0;
	createKernelErrorCopyVolumeToNew = 0;
	createKernelErrorMultiplyVolume = 0;
	createKernelErrorMultiplyVolumes = 0;
	createKernelErrorMultiplyVolumesOverwrite = 0;
	createKernelErrorAddVolume = 0;
	createKernelErrorAddVolumes = 0;
	createKernelErrorAddVolumesOverwrite = 0;	
	createKernelErrorCalculateMagnitudes = 0;
	createKernelErrorCalculateColumnSums = 0;
	createKernelErrorCalculateRowSums = 0;
	createKernelErrorCalculateColumnMaxs = 0;
	createKernelErrorCalculateRowMaxs = 0;
	createKernelErrorCalculateBetaValuesGLM = 0;
	createKernelErrorCalculateStatisticalMapsGLMTTest = 0;
	createKernelErrorCalculateStatisticalMapsGLMFTest = 0;
	createKernelErrorEstimateAR4Models = 0;
	createKernelErrorApplyWhiteningAR4 = 0;
	createKernelErrorRemoveLinearFit = 0;
	createKernelErrorGeneratePermutedVolumesFirstLevel = 0;
	createKernelErrorGeneratePermutedVolumesSecondLevel = 0;


	getPlatformIDsError = 0;
	getDeviceIDsError = 0;		
	createContextError = 0;
	getContextInfoError = 0;
	createCommandQueueError = 0;
	createProgramError = 0;
	buildProgramError = 0;
	getProgramBuildInfoError = 0;
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

int BROCCOLI_LIB::GetOpenCLInitiated()
{
	return OPENCL_INITIATED;
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
	clGetPlatformIDs (platformIdCount, platformIds.data(), NULL);

	// Loop over platforms
	for (uint i = 0; i < platformIdCount; i++) 
    {
		device_info.append("---------------------------------------------");
		device_info.append("\n");
		device_info.append("Platform number: ");
		temp_stream.str("");
		temp_stream.clear();
		temp_stream << i;
		device_info.append(temp_stream.str());
		device_info.append("\n");
		device_info.append("---------------------------------------------");
		device_info.append("\n");

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
		free(value);

		device_info.append("---------------------------------------------");
		device_info.append("\n");
		device_info.append("\n");

		// Get devices for each platform
		cl_uint deviceIdCount = 0;
		clGetDeviceIDs (platformIds[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceIdCount);
		std::vector<cl_device_id> deviceIds (deviceIdCount);
		clGetDeviceIDs (platformIds[i], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), NULL);

		// Get information for for each device and save as a long string
		for (uint j = 0; j < deviceIdCount; j++) 
		{
			device_info.append("---------------------------------------------");
			device_info.append("\n");
			device_info.append("Device number: ");
			temp_stream.str("");
			temp_stream.clear();
			temp_stream << j;
			device_info.append(temp_stream.str());
			device_info.append("\n");
			device_info.append("---------------------------------------------");
			device_info.append("\n");

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
            
			// Get device extensions
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_EXTENSIONS, 0, NULL, &valueSize);
			value = (char*) malloc(valueSize);
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_EXTENSIONS, valueSize, value, NULL);            
			device_info.append("Device extensions: ");
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
            
			// Get global memory cache size
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(memorySize), &memorySize, NULL);            
			device_info.append("Global memory cache size in KB: ");
			temp_stream.str("");
			temp_stream.clear();
			temp_stream << memorySize/ (1024);            
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

cl_int BROCCOLI_LIB::CreateProgramFromBinary(cl_program& program, cl_context context, cl_device_id device, std::string filename)
{
	// Get device name and remove spaces, add to filename
	char* value;
	size_t valueSize;
	std::string device_name;
	clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
	value = (char*) malloc(valueSize);
	clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);            
	filename.append("_");
	device_name = value;
	device_name.erase(std::remove (device_name.begin(), device_name.end(), ' '), device_name.end());
	filename.append(device_name);			
	filename.append(".bin");
	free(value);

	FILE* fp = fopen(filename.c_str(), "rb");
	if (fp == NULL)
	{
		program = NULL;
		return -1;
	}

	// Determine the size of the binary
	size_t binarySize;
	fseek(fp, 0, SEEK_END);
	binarySize = ftell(fp);
	rewind(fp);

	// Load binary from disk
	unsigned char* programBinary = new unsigned char[binarySize];
	fread(programBinary, 1, binarySize, fp);
	fclose(fp);

	cl_int binaryStatus;

	program = clCreateProgramWithBinary(context, 1, &device, &binarySize, (const unsigned char**)&programBinary, &binaryStatus, &error);
	delete [] programBinary;

	if (binaryStatus != SUCCESS)
	{
		program = NULL;
		return binaryStatus;
	}	

	if (error != SUCCESS)
	{
		program = NULL;
		return error;
	}
	else
	{
		return error;
	}	
}

bool BROCCOLI_LIB::SaveProgramBinary(cl_program program, cl_device_id device, std::string filename)
{
	// Get number of devices for program
	cl_uint numDevices = 0;
	error = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &numDevices, NULL);
	if (error != SUCCESS)
	{
		return false;
	}

	// Get device IDs
	cl_device_id* devices = new cl_device_id[numDevices];
	error = clGetProgramInfo(program, CL_PROGRAM_DEVICES, sizeof(cl_device_id) * numDevices, devices, NULL);
	if (error != SUCCESS)
	{
		// Cleanup
		delete [] devices;
		return false;
	}

	// Get size of each program binary
	size_t* programBinarySizes = new size_t[numDevices];
	error = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * numDevices, programBinarySizes, NULL);
	if (error != SUCCESS)
	{
		// Cleanup
		delete [] devices;
		delete [] programBinarySizes;
		return false;
	}

	// Allocate temporar memory
	unsigned char** programBinaries = new unsigned char*[numDevices];
	
	for (cl_uint i = 0; i < numDevices; i++)
	{
		programBinaries[i] = new unsigned char[programBinarySizes[i]];
	}

	// Get all program binaries
	error = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*) * numDevices, programBinaries, NULL);
	if (error != SUCCESS)
	{
		// Cleanup
		delete [] devices;
		delete [] programBinarySizes;
		for (cl_uint i = 0; i < numDevices; i++)
		{
			delete [] programBinaries[i];
		}
		delete [] programBinaries;
		return false;
	}

	for (cl_uint i = 0; i < numDevices; i++)
	{
		if (devices[i] == device)
		{
			// Get device name and remove spaces, add to filename
			char* value;
			size_t valueSize;
			std::string device_name;
			clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
			value = (char*) malloc(valueSize);
			clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);            
			filename.append("_");
			device_name = value;
			device_name.erase(std::remove (device_name.begin(), device_name.end(), ' '), device_name.end());
			filename.append(device_name);			
			filename.append(".bin");
			free(value);

			FILE* fp = fopen(filename.c_str(), "wb");
			if (fp != NULL)
			{
				programBinarySize = programBinarySizes[i];
				writtenElements = fwrite(programBinaries[i], 1, programBinarySizes[i], fp);
				fclose(fp);
				break;				
			}
			else
			{
				return false;
			}
		}
	}

	// Cleanup
	delete [] devices;
	delete [] programBinarySizes;
	for (cl_uint i = 0; i < numDevices; i++)
	{
		delete [] programBinaries[i];
	}
	delete [] programBinaries;
	return true;
}



void BROCCOLI_LIB::OpenCLInitiate(cl_uint OPENCL_PLATFORM, cl_uint OPENCL_DEVICE)
{
	char* value;
	size_t valueSize;
	cl_device_id *clDevices;

  	// Get number of platforms
	cl_uint platformIdCount = 0;
	getPlatformIDsError = clGetPlatformIDs (0, NULL, &platformIdCount);

	if (getPlatformIDsError == SUCCESS)
	{
		// Get platform IDs
		std::vector<cl_platform_id> platformIds(platformIdCount);
		getPlatformIDsError = clGetPlatformIDs(platformIdCount, platformIds.data(), NULL);              

		if (getPlatformIDsError == SUCCESS)
		{	
			// Check if the requested platform exists
			if ((OPENCL_PLATFORM >= 0) &&  (OPENCL_PLATFORM < platformIdCount))
			{
				// Create context
				const cl_context_properties contextProperties [] =
				{
					CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[OPENCL_PLATFORM]), 0, 0
				};

				// Get number of devices for current platform
				cl_uint deviceIdCount = 0;
				getDeviceIDsError = clGetDeviceIDs (platformIds[OPENCL_PLATFORM], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceIdCount);
	
				if (getDeviceIDsError == SUCCESS)
				{
					// Check if the requested device exists
					if ((OPENCL_DEVICE >= 0) &&  (OPENCL_DEVICE < deviceIdCount))
					{
						// Get device IDs for current platform
						std::vector<cl_device_id> deviceIds(deviceIdCount);
						getDeviceIDsError = clGetDeviceIDs(platformIds[OPENCL_PLATFORM], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), NULL);

						if (getDeviceIDsError == SUCCESS)
						{
							// Create context
							context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), NULL, NULL, &createContextError);

							if (createContextError == SUCCESS)
							{
								// Get size of context info
								getContextInfoError = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &valueSize);

								if (getContextInfoError == SUCCESS)
								{
									// Get context info
									clDevices = (cl_device_id *) malloc(valueSize);
									getContextInfoError = clGetContextInfo(context, CL_CONTEXT_DEVICES, valueSize, clDevices, NULL);

									if (getContextInfoError == SUCCESS)
									{
										// Get size of name of current platform
										clGetPlatformInfo(platformIds[OPENCL_PLATFORM], CL_PLATFORM_NAME, 0, NULL, &valueSize);
										value = (char*) malloc(valueSize);
										// Get name of current platform
										clGetPlatformInfo(platformIds[OPENCL_PLATFORM], CL_PLATFORM_NAME, valueSize, value, NULL);
										std::string vendor_string(value);
										free(value);

										// Figure out the vendor
										size_t npos = vendor_string.find("NVIDIA");
										size_t ipos = vendor_string.find("Intel");
										size_t apos = vendor_string.find("AMD");

										if (npos != std::string::npos)
										{
											VENDOR = NVIDIA;
											binaryFilename = "broccoli_lib_kernel_Nvidia";
										}
										else if (ipos != std::string::npos)
										{
											VENDOR = INTEL;
											binaryFilename = "broccoli_lib_kernel_Intel";
										}
										else if (apos != std::string::npos)
										{
											VENDOR = AMD;
											binaryFilename = "broccoli_lib_kernel_AMD";
										}

										// Create a command queue
										commandQueue = clCreateCommandQueue(context, deviceIds[OPENCL_DEVICE], CL_QUEUE_PROFILING_ENABLE, &createCommandQueueError);

										if (createCommandQueueError == SUCCESS)
										{
											// First try to compile from binary file
											createProgramError = CreateProgramFromBinary(program, context, deviceIds[OPENCL_DEVICE], binaryFilename);
											buildProgramError = clBuildProgram(program, deviceIdCount, deviceIds.data(), NULL, NULL, NULL);

											// Otherwise compile from source code
											if (buildProgramError != SUCCESS)
											{
												// Read the kernel code from file
												std::fstream kernelFile("broccoli_lib_kernel.cpp",std::ios::in);
												std::ostringstream oss;
												oss << kernelFile.rdbuf();
												std::string src = oss.str();
												const char *srcstr = src.c_str();

												// Create program and build the code
												program = clCreateProgramWithSource(context, 1, (const char**)&srcstr , NULL, &createProgramError);
												buildProgramError = clBuildProgram(program, deviceIdCount, deviceIds.data(), NULL, NULL, NULL);

												// Save to binary file
												if (buildProgramError == SUCCESS)
												{
													SaveProgramBinary(program,deviceIds[OPENCL_DEVICE],binaryFilename);
												}
											}

											if (buildProgramError == SUCCESS)
											{
												// Get size of build info
												valueSize = 0;
												getProgramBuildInfoError = clGetProgramBuildInfo(program, deviceIds[OPENCL_DEVICE], CL_PROGRAM_BUILD_LOG, 0, NULL, &valueSize);

												if (getProgramBuildInfoError == SUCCESS)
												{
													// Get build info
													value = (char*)malloc(valueSize);
													getProgramBuildInfoError = clGetProgramBuildInfo(program, deviceIds[OPENCL_DEVICE], CL_PROGRAM_BUILD_LOG, valueSize, value, NULL);

													if (getProgramBuildInfoError == SUCCESS)
													{
														build_info.append(value);

														// Create kernels

														MemsetKernel = clCreateKernel(program,"Memset",&createKernelErrorMemset);
														MemsetFloat2Kernel = clCreateKernel(program,"MemsetFloat2",&createKernelErrorMemsetFloat2);
														//MemsetDoubleKernel = clCreateKernel(program,"MemsetDouble",&createKernelErrorMemset);

														if ( (VENDOR == NVIDIA) || (VENDOR == INTEL))
														{
															NonseparableConvolution3DComplexThreeFiltersKernel = clCreateKernel(program,"Nonseparable3DConvolutionComplexThreeQuadratureFilters",&createKernelErrorNonseparableConvolution3DComplexThreeFilters);
															SeparableConvolutionRowsKernel = clCreateKernel(program,"SeparableConvolutionRows",&createKernelErrorSeparableConvolutionRows);
															SeparableConvolutionColumnsKernel = clCreateKernel(program,"SeparableConvolutionColumns",&createKernelErrorSeparableConvolutionColumns);
															SeparableConvolutionRodsKernel = clCreateKernel(program,"SeparableConvolutionRods",&createKernelErrorSeparableConvolutionRods);
														}
														else if (VENDOR == AMD)
														{
															NonseparableConvolution3DComplexThreeFiltersKernel = clCreateKernel(program,"Nonseparable3DConvolutionComplexThreeQuadratureFiltersAMD",&createKernelErrorNonseparableConvolution3DComplexThreeFilters);
															SeparableConvolutionRowsKernel = clCreateKernel(program,"SeparableConvolutionRowsAMD",&createKernelErrorSeparableConvolutionRows);
															SeparableConvolutionColumnsKernel = clCreateKernel(program,"SeparableConvolutionColumnsAMD",&createKernelErrorSeparableConvolutionColumns);
															SeparableConvolutionRodsKernel = clCreateKernel(program,"SeparableConvolutionRodsAMD",&createKernelErrorSeparableConvolutionRods);
														}

														// Kernels for parametric registration
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
												
														// Kernels for non-parametric registration
														//CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel = clCreateKernel(program, "CalculatePhaseDifferencesCertaintiesAndTensorComponents", &createKernelErrorCalculatePhaseDifferencesCertaintiesAndTensorComponents);
														CalculateTensorComponentsKernel = clCreateKernel(program, "CalculateTensorComponents", &createKernelErrorCalculateTensorComponents);
														CalculateTensorNormsKernel = clCreateKernel(program, "CalculateTensorNorms", &createKernelErrorCalculateTensorNorms);
														CalculateAMatricesAndHVectorsKernel = clCreateKernel(program, "CalculateAMatricesAndHVectors", &createKernelErrorCalculateAMatricesAndHVectors);
														//CalculateDisplacementAndCertaintyUpdateKernel = clCreateKernel(program, "CalculateDisplacementAndCertaintyUpdate", &createKernelErrorCalculateDisplacementAndCertaintyUpdate);
														CalculateDisplacementUpdateKernel = clCreateKernel(program, "CalculateDisplacementUpdate", &createKernelErrorCalculateDisplacementUpdate);

														CalculateMagnitudesKernel = clCreateKernel(program,"CalculateMagnitudes",&createKernelErrorCalculateMagnitudes);
														CalculateColumnSumsKernel = clCreateKernel(program,"CalculateColumnSums",&createKernelErrorCalculateColumnSums);
														CalculateRowSumsKernel = clCreateKernel(program,"CalculateRowSums",&createKernelErrorCalculateRowSums);
														CalculateColumnMaxsKernel = clCreateKernel(program,"CalculateColumnMaxs",&createKernelErrorCalculateColumnMaxs);
														CalculateRowMaxsKernel = clCreateKernel(program,"CalculateRowMaxs",&createKernelErrorCalculateRowMaxs);
														ThresholdVolumeKernel = clCreateKernel(program,"ThresholdVolume",&createKernelErrorThresholdVolume);

														InterpolateVolumeNearestParametricKernel = clCreateKernel(program,"InterpolateVolumeNearestParametric",&createKernelErrorInterpolateVolumeNearestParametric);
														InterpolateVolumeLinearParametricKernel = clCreateKernel(program,"InterpolateVolumeLinearParametric",&createKernelErrorInterpolateVolumeLinearParametric);
														InterpolateVolumeCubicParametricKernel = clCreateKernel(program,"InterpolateVolumeCubicParametric",&createKernelErrorInterpolateVolumeCubicParametric);
														InterpolateVolumeNearestNonParametricKernel = clCreateKernel(program,"InterpolateVolumeNearestNonParametric",&createKernelErrorInterpolateVolumeNearestNonParametric);
														InterpolateVolumeLinearNonParametricKernel = clCreateKernel(program,"InterpolateVolumeLinearNonParametric",&createKernelErrorInterpolateVolumeLinearNonParametric);
														InterpolateVolumeCubicNonParametricKernel = clCreateKernel(program,"InterpolateVolumeCubicNonParametric",&createKernelErrorInterpolateVolumeCubicNonParametric);

														RescaleVolumeLinearKernel = clCreateKernel(program,"RescaleVolumeLinear",&createKernelErrorRescaleVolumeLinear);
														RescaleVolumeCubicKernel = clCreateKernel(program,"RescaleVolumeCubic",&createKernelErrorRescaleVolumeCubic);
														CopyT1VolumeToMNIKernel = clCreateKernel(program,"CopyT1VolumeToMNI",&createKernelErrorCopyT1VolumeToMNI);
														CopyEPIVolumeToT1Kernel = clCreateKernel(program,"CopyEPIVolumeToT1",&createKernelErrorCopyEPIVolumeToT1);
														CopyVolumeToNewKernel = clCreateKernel(program,"CopyVolumeToNew",&createKernelErrorCopyVolumeToNew);

														MultiplyVolumeKernel = clCreateKernel(program,"MultiplyVolume",&createKernelErrorMultiplyVolume);
														MultiplyVolumesKernel = clCreateKernel(program,"MultiplyVolumes",&createKernelErrorMultiplyVolumes);
														MultiplyVolumesOverwriteKernel = clCreateKernel(program,"MultiplyVolumesOverwrite",&createKernelErrorMultiplyVolumesOverwrite);
														AddVolumeKernel = clCreateKernel(program,"AddVolume",&createKernelErrorAddVolume);
														AddVolumesKernel = clCreateKernel(program,"AddVolumes",&createKernelErrorAddVolumes);
														AddVolumesOverwriteKernel = clCreateKernel(program,"AddVolumesOverwrite",&createKernelErrorAddVolumesOverwrite);
												
														CalculateBetaValuesGLMKernel = clCreateKernel(program,"CalculateBetaValuesGLM",&createKernelErrorCalculateBetaValuesGLM);
														CalculateStatisticalMapsGLMTTestKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMTTest",&createKernelErrorCalculateStatisticalMapsGLMTTest);
														CalculateStatisticalMapsGLMFTestKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMFTest",&createKernelErrorCalculateStatisticalMapsGLMFTest);
														CalculateStatisticalMapsGLMPermutationKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMPermutation",&createKernelErrorCalculateStatisticalMapsGLMPermutation);
														EstimateAR4ModelsKernel = clCreateKernel(program,"EstimateAR4Models",&createKernelErrorEstimateAR4Models);
														ApplyWhiteningAR4Kernel = clCreateKernel(program,"ApplyWhiteningAR4",&createKernelErrorApplyWhiteningAR4);
														GeneratePermutedVolumesFirstLevelKernel = clCreateKernel(program,"GeneratePermutedVolumesFirstLevel",&createKernelErrorGeneratePermutedVolumesFirstLevel);
														GeneratePermutedVolumesSecondLevelKernel = clCreateKernel(program,"GeneratePermutedVolumesSecondLevel",&createKernelErrorGeneratePermutedVolumesSecondLevel);
														RemoveLinearFitKernel = clCreateKernel(program,"RemoveLinearFit",&createKernelErrorRemoveLinearFit);

														RemoveMeanKernel = clCreateKernel(program,"RemoveMean",&createKernelErrorRemoveLinearFit);

														OPENCL_INITIATED = 1;
													}
													free(value);
												}
											}
										}
									}
									free(clDevices);
								}
							}
						}
					}
				}
			}
		}
	}
}

void BROCCOLI_LIB::OpenCLCleanup()
{
	if (OPENCL_INITIATED == 1)
	{		
		if (MemsetKernel != NULL)
		{
			clReleaseKernel(MemsetKernel);
		}
		if (NonseparableConvolution3DComplexThreeFiltersKernel != NULL)
		{
			clReleaseKernel(NonseparableConvolution3DComplexThreeFiltersKernel);		
		}
		if (SeparableConvolutionRowsKernel != NULL)
		{
			clReleaseKernel(SeparableConvolutionRowsKernel);
		}
		if (SeparableConvolutionColumnsKernel != NULL)
		{
			clReleaseKernel(SeparableConvolutionColumnsKernel);
		}
		if (SeparableConvolutionRodsKernel != NULL)
		{
			clReleaseKernel(SeparableConvolutionRodsKernel);
		}
		if (CalculatePhaseDifferencesAndCertaintiesKernel != NULL)
		{
			clReleaseKernel(CalculatePhaseDifferencesAndCertaintiesKernel);
		}
		if (CalculatePhaseGradientsXKernel != NULL)
		{
			clReleaseKernel(CalculatePhaseGradientsXKernel);
		}
		if (CalculatePhaseGradientsYKernel != NULL)
		{
			clReleaseKernel(CalculatePhaseGradientsYKernel);
		}
		if (CalculatePhaseGradientsZKernel != NULL)
		{
			clReleaseKernel(CalculatePhaseGradientsZKernel);
		}
		if (CalculateAMatrixAndHVector2DValuesXKernel != NULL)
		{
			clReleaseKernel(CalculateAMatrixAndHVector2DValuesXKernel);
		}
		if (CalculateAMatrixAndHVector2DValuesYKernel != NULL)
		{
			clReleaseKernel(CalculateAMatrixAndHVector2DValuesYKernel);
		}
		if (CalculateAMatrixAndHVector2DValuesZKernel != NULL)
		{
			clReleaseKernel(CalculateAMatrixAndHVector2DValuesZKernel);
		}
		if (CalculateAMatrix1DValuesKernel != NULL)
		{
			clReleaseKernel(CalculateAMatrix1DValuesKernel);
		}
		if (CalculateHVector1DValuesKernel != NULL)
		{
			clReleaseKernel(CalculateHVector1DValuesKernel);
		}
		if (CalculateAMatrixKernel != NULL)
		{
			clReleaseKernel(CalculateAMatrixKernel);
		}
		if (CalculateHVectorKernel != NULL)
		{
			clReleaseKernel(CalculateHVectorKernel);
		}
		if (CalculateMagnitudesKernel != NULL)
		{
			clReleaseKernel(CalculateMagnitudesKernel);
		}
		if (CalculateColumnSumsKernel != NULL)
		{
			clReleaseKernel(CalculateColumnSumsKernel);
		}
		if (CalculateRowSumsKernel != NULL)
		{
			clReleaseKernel(CalculateRowSumsKernel);
		}
		if (CalculateColumnMaxsKernel != NULL)
		{
			clReleaseKernel(CalculateColumnMaxsKernel);
		}
		if (CalculateRowMaxsKernel != NULL)
		{
			clReleaseKernel(CalculateRowMaxsKernel);
		}
		if (ThresholdVolumeKernel != NULL)
		{
			clReleaseKernel(ThresholdVolumeKernel);
		}
		if (InterpolateVolumeNearestParametricKernel != NULL)
		{
			clReleaseKernel(InterpolateVolumeNearestParametricKernel);
		}
		if (InterpolateVolumeLinearParametricKernel != NULL)
		{
			clReleaseKernel(InterpolateVolumeLinearParametricKernel);
		}
		if (InterpolateVolumeCubicParametricKernel != NULL)
		{
			clReleaseKernel(InterpolateVolumeCubicParametricKernel);
		}
		if (InterpolateVolumeNearestNonParametricKernel != NULL)
		{
			clReleaseKernel(InterpolateVolumeNearestNonParametricKernel);
		}
		if (InterpolateVolumeLinearNonParametricKernel != NULL)
		{
			clReleaseKernel(InterpolateVolumeLinearNonParametricKernel);
		}
		if (InterpolateVolumeCubicNonParametricKernel != NULL)
		{
			clReleaseKernel(InterpolateVolumeCubicNonParametricKernel);
		}

		//clReleaseKernel(RescaleVolumeNearestKernel);
		if (RescaleVolumeLinearKernel != NULL)
		{
			clReleaseKernel(RescaleVolumeLinearKernel);
		}
		if (RescaleVolumeCubicKernel != NULL)
		{
			clReleaseKernel(RescaleVolumeCubicKernel);
		}
		if (CopyT1VolumeToMNIKernel != NULL)
		{
			clReleaseKernel(CopyT1VolumeToMNIKernel);
		}
		if (CopyEPIVolumeToT1Kernel != NULL)
		{
			clReleaseKernel(CopyEPIVolumeToT1Kernel);
		}
		if (CopyVolumeToNewKernel != NULL)
		{
			clReleaseKernel(CopyVolumeToNewKernel);
		}
		if (MultiplyVolumesKernel != NULL)
		{
			clReleaseKernel(MultiplyVolumesKernel);
		}
		if (MultiplyVolumesOverwriteKernel != NULL)
		{
			clReleaseKernel(MultiplyVolumesOverwriteKernel);
		}
		if (CalculateBetaValuesGLMKernel != NULL)
		{
			clReleaseKernel(CalculateBetaValuesGLMKernel);
		}
		if (CalculateStatisticalMapsGLMTTestKernel != NULL)
		{
			clReleaseKernel(CalculateStatisticalMapsGLMTTestKernel);
		}
		if (CalculateStatisticalMapsGLMFTestKernel != NULL)
		{
			clReleaseKernel(CalculateStatisticalMapsGLMFTestKernel);
		}
		if (EstimateAR4ModelsKernel != NULL)
		{
			clReleaseKernel(EstimateAR4ModelsKernel);
		}
		if (ApplyWhiteningAR4Kernel != NULL)
		{
			clReleaseKernel(ApplyWhiteningAR4Kernel);
		}
	
		if (program != NULL)
		{
			clReleaseProgram(program);    
		}
		if (commandQueue != NULL)
		{
			clReleaseCommandQueue(commandQueue);
		}
		if (context != NULL)
		{
			clReleaseContext(context);		
		}
	}
}


void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesSeparableConvolution(int DATA_W, int DATA_H, int DATA_D)
{		
	if ( (VENDOR == NVIDIA) || (VENDOR == INTEL) )
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
	else if (VENDOR == AMD)
	{
		//----------------------------------
		// Separable convolution rows
		//----------------------------------

		localWorkSizeSeparableConvolutionRows[0] = 32;
		localWorkSizeSeparableConvolutionRows[1] = 8;
		localWorkSizeSeparableConvolutionRows[2] = 1;

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
		localWorkSizeSeparableConvolutionColumns[2] = 1;

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
		localWorkSizeSeparableConvolutionRods[1] = 1;
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
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesMemset(int N)
{	
	localWorkSizeMemset[0] = 256;
	localWorkSizeMemset[1] = 1;
	localWorkSizeMemset[2] = 1;

	xBlocks = (size_t)ceil((float)(N) / (float)localWorkSizeMemset[0]);
	
	globalWorkSizeMemset[0] = xBlocks * localWorkSizeMemset[0];
	globalWorkSizeMemset[1] = 1;
	globalWorkSizeMemset[2] = 1;
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesNonSeparableConvolution(int DATA_W, int DATA_H, int DATA_D)
{	
	if ( (VENDOR == NVIDIA) || (VENDOR == INTEL) )
	{
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
	else if (VENDOR == AMD)
	{
		localWorkSizeNonseparableConvolution3DComplex[0] = 16;
		localWorkSizeNonseparableConvolution3DComplex[1] = 16;
		localWorkSizeNonseparableConvolution3DComplex[2] = 1;

		// Calculate how many blocks are required
		xBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_X_CONVOLUTION_2D_AMD);
		yBlocks = (size_t)ceil((float)DATA_H / (float)VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D_AMD);
		zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeNonseparableConvolution3DComplex[2]);

		// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
		globalWorkSizeNonseparableConvolution3DComplex[0] = xBlocks * localWorkSizeNonseparableConvolution3DComplex[0];
		globalWorkSizeNonseparableConvolution3DComplex[1] = yBlocks * localWorkSizeNonseparableConvolution3DComplex[1];
		globalWorkSizeNonseparableConvolution3DComplex[2] = zBlocks * localWorkSizeNonseparableConvolution3DComplex[2];
	}
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesImageRegistration(int DATA_W, int DATA_H, int DATA_D)
{	
	//----------------------------------
	// Phase differences and certainties
	//----------------------------------

	localWorkSizeCalculatePhaseDifferencesAndCertainties[0] = 16;
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

	localWorkSizeCalculatePhaseGradients[0] = 16;
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
	// Tensor norms
	//----------------------------------

	localWorkSizeCalculateTensorNorms[0] = 16;
	localWorkSizeCalculateTensorNorms[1] = 16;
	localWorkSizeCalculateTensorNorms[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculateTensorNorms[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculateTensorNorms[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateTensorNorms[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculateTensorNorms[0] = xBlocks * localWorkSizeCalculateTensorNorms[0];
	globalWorkSizeCalculateTensorNorms[1] = yBlocks * localWorkSizeCalculateTensorNorms[1];
	globalWorkSizeCalculateTensorNorms[2] = zBlocks * localWorkSizeCalculateTensorNorms[2];

	//----------------------------------
	// Displacement
	//----------------------------------

	localWorkSizeCalculateDisplacementAndCertaintyUpdate[0] = 16;
	localWorkSizeCalculateDisplacementAndCertaintyUpdate[1] = 16;
	localWorkSizeCalculateDisplacementAndCertaintyUpdate[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculateDisplacementAndCertaintyUpdate[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculateDisplacementAndCertaintyUpdate[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateDisplacementAndCertaintyUpdate[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculateDisplacementAndCertaintyUpdate[0] = xBlocks * localWorkSizeCalculateDisplacementAndCertaintyUpdate[0];
	globalWorkSizeCalculateDisplacementAndCertaintyUpdate[1] = yBlocks * localWorkSizeCalculateDisplacementAndCertaintyUpdate[1];
	globalWorkSizeCalculateDisplacementAndCertaintyUpdate[2] = zBlocks * localWorkSizeCalculateDisplacementAndCertaintyUpdate[2];

	//----------------------------------
	// A-matrices and h-vectors
	//----------------------------------

	localWorkSizeCalculateAMatricesAndHVectors[0] = 16;
	localWorkSizeCalculateAMatricesAndHVectors[1] = 16;
	localWorkSizeCalculateAMatricesAndHVectors[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculateAMatricesAndHVectors[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculateAMatricesAndHVectors[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateAMatricesAndHVectors[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculateAMatricesAndHVectors[0] = xBlocks * localWorkSizeCalculateAMatricesAndHVectors[0];
	globalWorkSizeCalculateAMatricesAndHVectors[1] = yBlocks * localWorkSizeCalculateAMatricesAndHVectors[1];
	globalWorkSizeCalculateAMatricesAndHVectors[2] = zBlocks * localWorkSizeCalculateAMatricesAndHVectors[2];

	
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
	localWorkSizeRescaleVolumeLinear[0] = 16;
	localWorkSizeRescaleVolumeLinear[1] = 16;
	localWorkSizeRescaleVolumeLinear[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeRescaleVolumeLinear[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeRescaleVolumeLinear[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeRescaleVolumeLinear[2]);

	globalWorkSizeRescaleVolumeLinear[0] = xBlocks * localWorkSizeRescaleVolumeLinear[0];
	globalWorkSizeRescaleVolumeLinear[1] = yBlocks * localWorkSizeRescaleVolumeLinear[1];
	globalWorkSizeRescaleVolumeLinear[2] = zBlocks * localWorkSizeRescaleVolumeLinear[2];

	localWorkSizeRescaleVolumeCubic[0] = 16;
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
	localWorkSizeInterpolateVolumeNearest[0] = 16;
	localWorkSizeInterpolateVolumeNearest[1] = 16;
	localWorkSizeInterpolateVolumeNearest[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeInterpolateVolumeNearest[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeInterpolateVolumeNearest[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeInterpolateVolumeNearest[2]);

	globalWorkSizeInterpolateVolumeNearest[0] = xBlocks * localWorkSizeInterpolateVolumeNearest[0];
	globalWorkSizeInterpolateVolumeNearest[1] = yBlocks * localWorkSizeInterpolateVolumeNearest[1];
	globalWorkSizeInterpolateVolumeNearest[2] = zBlocks * localWorkSizeInterpolateVolumeNearest[2];

	localWorkSizeInterpolateVolumeLinear[0] = 16;
	localWorkSizeInterpolateVolumeLinear[1] = 16;
	localWorkSizeInterpolateVolumeLinear[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeInterpolateVolumeLinear[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeInterpolateVolumeLinear[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeInterpolateVolumeLinear[2]);

	globalWorkSizeInterpolateVolumeLinear[0] = xBlocks * localWorkSizeInterpolateVolumeLinear[0];
	globalWorkSizeInterpolateVolumeLinear[1] = yBlocks * localWorkSizeInterpolateVolumeLinear[1];
	globalWorkSizeInterpolateVolumeLinear[2] = zBlocks * localWorkSizeInterpolateVolumeLinear[2];

	localWorkSizeInterpolateVolumeCubic[0] = 16;
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
	localWorkSizeCopyVolumeToNew[0] = 16;
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
	localWorkSizeMultiplyVolumes[0] = 16;
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

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesAddVolumes(int DATA_W, int DATA_H, int DATA_D)
{	
	localWorkSizeAddVolumes[0] = 16;
	localWorkSizeAddVolumes[1] = 16;
	localWorkSizeAddVolumes[2] = 1;
	
	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeAddVolumes[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeAddVolumes[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeAddVolumes[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeAddVolumes[0] = xBlocks * localWorkSizeAddVolumes[0];
	globalWorkSizeAddVolumes[1] = yBlocks * localWorkSizeAddVolumes[1];
	globalWorkSizeAddVolumes[2] = zBlocks * localWorkSizeAddVolumes[2];
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesCalculateSum(int DATA_W, int DATA_H, int DATA_D)
{
	localWorkSizeCalculateColumnSums[0] = 16;
	localWorkSizeCalculateColumnSums[1] = 16;
	localWorkSizeCalculateColumnSums[2] = 1;
	
	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculateColumnSums[0]);
	yBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateColumnSums[1]);
	zBlocks = 1;

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculateColumnSums[0] = xBlocks * localWorkSizeCalculateColumnSums[0];
	globalWorkSizeCalculateColumnSums[1] = yBlocks * localWorkSizeCalculateColumnSums[1];
	globalWorkSizeCalculateColumnSums[2] = 1;
	
	localWorkSizeCalculateRowSums[0] = 32;
	localWorkSizeCalculateRowSums[1] = 1;
	localWorkSizeCalculateRowSums[2] = 1;
	
	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateRowSums[0]);
	yBlocks = 1;
	zBlocks = 1;

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculateRowSums[0] = xBlocks * localWorkSizeCalculateRowSums[0];
	globalWorkSizeCalculateRowSums[1] = 1;
	globalWorkSizeCalculateRowSums[2] = 1;
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesCalculateMax(int DATA_W, int DATA_H, int DATA_D)
{
	localWorkSizeCalculateColumnMaxs[0] = 16;
	localWorkSizeCalculateColumnMaxs[1] = 16;
	localWorkSizeCalculateColumnMaxs[2] = 1;
	
	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculateColumnMaxs[0]);
	yBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateColumnMaxs[1]);
	zBlocks = 1;

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculateColumnMaxs[0] = xBlocks * localWorkSizeCalculateColumnMaxs[0];
	globalWorkSizeCalculateColumnMaxs[1] = yBlocks * localWorkSizeCalculateColumnMaxs[1];
	globalWorkSizeCalculateColumnMaxs[2] = 1;
	
	localWorkSizeCalculateRowMaxs[0] = 32;
	localWorkSizeCalculateRowMaxs[1] = 1;
	localWorkSizeCalculateRowMaxs[2] = 1;
	
	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateRowMaxs[0]);
	yBlocks = 1;
	zBlocks = 1;

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculateRowMaxs[0] = xBlocks * localWorkSizeCalculateRowMaxs[0];
	globalWorkSizeCalculateRowMaxs[1] = 1;
	globalWorkSizeCalculateRowMaxs[2] = 1;
}


void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesCalculateMagnitudes(int DATA_W, int DATA_H, int DATA_D)
{	
	localWorkSizeCalculateMagnitudes[0] = 16;
	localWorkSizeCalculateMagnitudes[1] = 16;
	localWorkSizeCalculateMagnitudes[2] = 1;
	
	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculateMagnitudes[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculateMagnitudes[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateMagnitudes[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculateMagnitudes[0] = xBlocks * localWorkSizeCalculateMagnitudes[0];
	globalWorkSizeCalculateMagnitudes[1] = yBlocks * localWorkSizeCalculateMagnitudes[1];
	globalWorkSizeCalculateMagnitudes[2] = zBlocks * localWorkSizeCalculateMagnitudes[2];	
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesThresholdVolume(int DATA_W, int DATA_H, int DATA_D)
{		
	localWorkSizeThresholdVolume[0] = 16;
	localWorkSizeThresholdVolume[1] = 16;
	localWorkSizeThresholdVolume[2] = 1;
	
	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeThresholdVolume[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeThresholdVolume[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeThresholdVolume[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeThresholdVolume[0] = xBlocks * localWorkSizeThresholdVolume[0];
	globalWorkSizeThresholdVolume[1] = yBlocks * localWorkSizeThresholdVolume[1];
	globalWorkSizeThresholdVolume[2] = zBlocks * localWorkSizeThresholdVolume[2];	
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesStatisticalCalculations(int DATA_W, int DATA_H, int DATA_D)
{	
	localWorkSizeCalculateBetaValuesGLM[0] = 16;
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

	localWorkSizeCalculateStatisticalMapsGLM[0] = 16;
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


	localWorkSizeEstimateAR4Models[0] = 16;
	localWorkSizeEstimateAR4Models[1] = 16;
	localWorkSizeEstimateAR4Models[2] = 1;
	
	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeEstimateAR4Models[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeEstimateAR4Models[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeEstimateAR4Models[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeEstimateAR4Models[0] = xBlocks * localWorkSizeEstimateAR4Models[0];
	globalWorkSizeEstimateAR4Models[1] = yBlocks * localWorkSizeEstimateAR4Models[1];
	globalWorkSizeEstimateAR4Models[2] = zBlocks * localWorkSizeEstimateAR4Models[2];

	localWorkSizeApplyWhiteningAR4[0] = 16;
	localWorkSizeApplyWhiteningAR4[1] = 16;
	localWorkSizeApplyWhiteningAR4[2] = 1;
	
	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeApplyWhiteningAR4[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeApplyWhiteningAR4[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeApplyWhiteningAR4[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeApplyWhiteningAR4[0] = xBlocks * localWorkSizeApplyWhiteningAR4[0];
	globalWorkSizeApplyWhiteningAR4[1] = yBlocks * localWorkSizeApplyWhiteningAR4[1];
	globalWorkSizeApplyWhiteningAR4[2] = zBlocks * localWorkSizeApplyWhiteningAR4[2];

	localWorkSizeGeneratePermutedVolumesFirstLevel[0] = 16;
	localWorkSizeGeneratePermutedVolumesFirstLevel[1] = 16;
	localWorkSizeGeneratePermutedVolumesFirstLevel[2] = 1;
	
	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeGeneratePermutedVolumesFirstLevel[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeGeneratePermutedVolumesFirstLevel[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeGeneratePermutedVolumesFirstLevel[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeGeneratePermutedVolumesFirstLevel[0] = xBlocks * localWorkSizeGeneratePermutedVolumesFirstLevel[0];
	globalWorkSizeGeneratePermutedVolumesFirstLevel[1] = yBlocks * localWorkSizeGeneratePermutedVolumesFirstLevel[1];
	globalWorkSizeGeneratePermutedVolumesFirstLevel[2] = zBlocks * localWorkSizeGeneratePermutedVolumesFirstLevel[2];

	localWorkSizeGeneratePermutedVolumesSecondLevel[0] = 16;
	localWorkSizeGeneratePermutedVolumesSecondLevel[1] = 16;
	localWorkSizeGeneratePermutedVolumesSecondLevel[2] = 1;
	
	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeGeneratePermutedVolumesSecondLevel[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeGeneratePermutedVolumesSecondLevel[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeGeneratePermutedVolumesSecondLevel[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeGeneratePermutedVolumesSecondLevel[0] = xBlocks * localWorkSizeGeneratePermutedVolumesSecondLevel[0];
	globalWorkSizeGeneratePermutedVolumesSecondLevel[1] = yBlocks * localWorkSizeGeneratePermutedVolumesSecondLevel[1];
	globalWorkSizeGeneratePermutedVolumesSecondLevel[2] = zBlocks * localWorkSizeGeneratePermutedVolumesSecondLevel[2];

	localWorkSizeRemoveLinearFit[0] = 16;
	localWorkSizeRemoveLinearFit[1] = 16;
	localWorkSizeRemoveLinearFit[2] = 1;
	
	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeRemoveLinearFit[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeRemoveLinearFit[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeRemoveLinearFit[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeRemoveLinearFit[0] = xBlocks * localWorkSizeRemoveLinearFit[0];
	globalWorkSizeRemoveLinearFit[1] = yBlocks * localWorkSizeRemoveLinearFit[1];
	globalWorkSizeRemoveLinearFit[2] = zBlocks * localWorkSizeRemoveLinearFit[2];
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

void BROCCOLI_LIB::SetInputMNIBrainVolume(float* data)
{
	h_MNI_Brain_Volume = data;
}


void BROCCOLI_LIB::SetInputMNIBrainMask(float* data)
{
	h_MNI_Brain_Mask = data;
}

void BROCCOLI_LIB::SetMask(float* data)
{
	h_Mask = data;
}

void BROCCOLI_LIB::SetNumberOfGLMRegressors(int N)
{
	NUMBER_OF_GLM_REGRESSORS = N;
}

void BROCCOLI_LIB::SetNumberOfDetrendingRegressors(int N)
{
	NUMBER_OF_DETRENDING_REGRESSORS = N;
}

void BROCCOLI_LIB::SetNumberOfContrasts(int N)
{
	NUMBER_OF_CONTRASTS = N;
}

void BROCCOLI_LIB::SetDesignMatrix(float* data1, float* data2)
{
	h_X_GLM_In = data1;
	h_xtxxt_GLM_In = data2;
}

void BROCCOLI_LIB::SetOutputDesignMatrix(float* data1, float* data2)
{
	h_X_GLM_Out = data1;
	h_xtxxt_GLM_Out = data2;
}

void BROCCOLI_LIB::SetGLMScalars(float* data)
{
	h_ctxtxc_GLM_In = data;
}

void BROCCOLI_LIB::SetContrasts(float* data)
{
	h_Contrasts_In = data;
}

void BROCCOLI_LIB::SetNumberOfPermutations(int N)
{
	NUMBER_OF_PERMUTATIONS = N;
}

void BROCCOLI_LIB::SetSmoothingFilters(float* Smoothing_Filter_X, float* Smoothing_Filter_Y, float* Smoothing_Filter_Z)
{
	h_Smoothing_Filter_X_In = Smoothing_Filter_X;
	h_Smoothing_Filter_Y_In = Smoothing_Filter_Y;
	h_Smoothing_Filter_Z_In = Smoothing_Filter_Z;
}

void BROCCOLI_LIB::SetImageRegistrationFilterSize(int N) 
{
	IMAGE_REGISTRATION_FILTER_SIZE = N;
}

//void BROCCOLI_LIB::SetParametricImageRegistrationFilters(cl_float2* qf1, cl_float2* qf2, cl_float2* qf3)    
void BROCCOLI_LIB::SetParametricImageRegistrationFilters(float* qf1r, float* qf1i, float* qf2r, float* qf2i, float* qf3r, float* qf3i)
{
	/*
	h_Quadrature_Filter_1_Parametric_Registration = qf1;
	h_Quadrature_Filter_2_Parametric_Registration = qf2;
	h_Quadrature_Filter_3_Parametric_Registration = qf3;
	*/

	h_Quadrature_Filter_1_Parametric_Registration_Real = qf1r;
	h_Quadrature_Filter_1_Parametric_Registration_Imag = qf1i;

	h_Quadrature_Filter_2_Parametric_Registration_Real = qf2r;
	h_Quadrature_Filter_2_Parametric_Registration_Imag = qf2i;

	h_Quadrature_Filter_3_Parametric_Registration_Real = qf3r;
	h_Quadrature_Filter_3_Parametric_Registration_Imag = qf3i;
}

//void BROCCOLI_LIB::SetNonParametricImageRegistrationFilters(cl_float2* qf1, cl_float2* qf2, cl_float2* qf3, cl_float2* qf4, cl_float2* qf5, cl_float2* qf6)    
void BROCCOLI_LIB::SetNonParametricImageRegistrationFilters(float* qf1r, float* qf1i, float* qf2r, float* qf2i, float* qf3r, float* qf3i, float* qf4r, float* qf4i, float* qf5r, float* qf5i, float* qf6r, float* qf6i) 
{
	/*
	h_Quadrature_Filter_1_NonParametric_Registration = qf1;
	h_Quadrature_Filter_2_NonParametric_Registration = qf2;
	h_Quadrature_Filter_3_NonParametric_Registration = qf3;
	h_Quadrature_Filter_4_NonParametric_Registration = qf4;
	h_Quadrature_Filter_5_NonParametric_Registration = qf5;
	h_Quadrature_Filter_6_NonParametric_Registration = qf6;
	*/

	h_Quadrature_Filter_1_NonParametric_Registration_Real = qf1r;
	h_Quadrature_Filter_1_NonParametric_Registration_Imag = qf1i;
	h_Quadrature_Filter_2_NonParametric_Registration_Real = qf2r;
	h_Quadrature_Filter_2_NonParametric_Registration_Imag = qf2i;
	h_Quadrature_Filter_3_NonParametric_Registration_Real = qf3r;
	h_Quadrature_Filter_3_NonParametric_Registration_Imag = qf3i;
	h_Quadrature_Filter_4_NonParametric_Registration_Real = qf4r;
	h_Quadrature_Filter_4_NonParametric_Registration_Imag = qf4i;
	h_Quadrature_Filter_5_NonParametric_Registration_Real = qf5r;
	h_Quadrature_Filter_5_NonParametric_Registration_Imag = qf5i;
	h_Quadrature_Filter_6_NonParametric_Registration_Real = qf6r;
	h_Quadrature_Filter_6_NonParametric_Registration_Imag = qf6i;
}

void SetNonParametricImageRegistrationFilters(float* qf1r, float* qf1i, float* qf2r, float* qf2i, float* q3r, float* q3i, float* qf4r, float* qf4i, float* qf5r, float* qf5i, float* q6r, float* q6i);

void BROCCOLI_LIB::SetProjectionTensorMatrixFirstFilter(float m11, float m12, float m13, float m22, float m23, float m33)
{
	M11_1 = m11;
	M12_1 = m12;
	M13_1 = m13;
	M22_1 = m22;
	M23_1 = m23;
	M33_1 = m33;
}

void BROCCOLI_LIB::SetProjectionTensorMatrixSecondFilter(float m11, float m12, float m13, float m22, float m23, float m33)
{
	M11_2 = m11;
	M12_2 = m12;
	M13_2 = m13;
	M22_2 = m22;
	M23_2 = m23;
	M33_2 = m33;
}

void BROCCOLI_LIB::SetProjectionTensorMatrixThirdFilter(float m11, float m12, float m13, float m22, float m23, float m33)
{
	M11_3 = m11;
	M12_3 = m12;
	M13_3 = m13;
	M22_3 = m22;
	M23_3 = m23;
	M33_3 = m33;
}

void BROCCOLI_LIB::SetProjectionTensorMatrixFourthFilter(float m11, float m12, float m13, float m22, float m23, float m33)
{
	M11_4 = m11;
	M12_4 = m12;
	M13_4 = m13;
	M22_4 = m22;
	M23_4 = m23;
	M33_4 = m33;
}

void BROCCOLI_LIB::SetProjectionTensorMatrixFifthFilter(float m11, float m12, float m13, float m22, float m23, float m33)
{
	M11_5 = m11;
	M12_5 = m12;
	M13_5 = m13;
	M22_5 = m22;
	M23_5 = m23;
	M33_5 = m33;
}

void BROCCOLI_LIB::SetProjectionTensorMatrixSixthFilter(float m11, float m12, float m13, float m22, float m23, float m33)
{
	M11_6 = m11;
	M12_6 = m12;
	M13_6 = m13;
	M22_6 = m22;
	M23_6 = m23;
	M33_6 = m33;
}

void BROCCOLI_LIB::SetFilterDirections(float* x, float* y, float* z)
{
	h_Filter_Directions_X = x;
	h_Filter_Directions_Y = y;
	h_Filter_Directions_Z = z;
}

void BROCCOLI_LIB::SetNumberOfIterationsForParametricImageRegistration(int N)
{
	NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION = N;
}

void BROCCOLI_LIB::SetNumberOfIterationsForNonParametricImageRegistration(int N)
{
	NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION = N;
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

void BROCCOLI_LIB::SetBetaSpace(int space)
{
	BETA_SPACE = space;
}

void BROCCOLI_LIB::SetInterpolationMode(int mode)
{
	INTERPOLATION_MODE = mode;
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

void BROCCOLI_LIB::SetNumberOfBasisFunctionsDetrending(int N)
{
	NUMBER_OF_DETRENDING_REGRESSORS = N;
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

void BROCCOLI_LIB::SetEPISmoothingAmount(float mm)
{
	EPI_Smoothing_FWHM = mm;
}

void BROCCOLI_LIB::SetARSmoothingAmount(float mm)
{
	AR_Smoothing_FWHM = mm;
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

void BROCCOLI_LIB::SetOutputEPIMNIRegistrationParameters(float* output)
{
	h_Registration_Parameters_EPI_MNI_Out = output;
}

void BROCCOLI_LIB::SetOutputQuadratureFilterResponses(cl_float2* qfr1, cl_float2* qfr2, cl_float2* qfr3, cl_float2* qfr4, cl_float2* qfr5, cl_float2* qfr6)
{
	h_Quadrature_Filter_Response_1 = qfr1;
	h_Quadrature_Filter_Response_2 = qfr2;
	h_Quadrature_Filter_Response_3 = qfr3;	
	h_Quadrature_Filter_Response_4 = qfr4;	
	h_Quadrature_Filter_Response_5 = qfr5;	
	h_Quadrature_Filter_Response_6 = qfr6;	
}

void BROCCOLI_LIB::SetOutputTensorComponents(float* t11, float* t12, float* t13, float* t22, float* t23, float* t33)
{
	h_t11 = t11;
	h_t12 = t12;
	h_t13 = t13;
	h_t22 = t22;
	h_t23 = t23;
	h_t33 = t33;
}

void BROCCOLI_LIB::SetOutputDisplacementField(float* x, float* y, float* z)
{
	h_Displacement_Field_X = x;
	h_Displacement_Field_Y = y;
	h_Displacement_Field_Z = z;
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

void BROCCOLI_LIB::SetOutputAlignedT1VolumeNonParametric(float* aligned)
{
	h_Aligned_T1_Volume_NonParametric = aligned;
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

void BROCCOLI_LIB::SetOutputDetrendedfMRIVolumes(float* detrended)
{
	h_Detrended_fMRI_Volumes = detrended;
}

void BROCCOLI_LIB::SetOutputWhitenedfMRIVolumes(float* whitened)
{
	h_Whitened_fMRI_Volumes = whitened;
}

void BROCCOLI_LIB::SetOutputPermutedfMRIVolumes(float* permuted)
{
	h_Permuted_fMRI_Volumes = permuted;
}

void BROCCOLI_LIB::SetOutputDownsampledVolume(float* downsampled)
{
	h_Downsampled_Volume = downsampled;
}

void BROCCOLI_LIB::SetOutputAREstimates(float* ar1, float* ar2, float* ar3, float* ar4)
{
	h_AR1_Estimates = ar1;
	h_AR2_Estimates = ar2;
	h_AR3_Estimates = ar3;
	h_AR4_Estimates = ar4;
}

void BROCCOLI_LIB::SetOutputSliceSums(float* output)
{
	h_Slice_Sums = output;
}

void BROCCOLI_LIB::SetOutputTopSlice(float* output)
{
	h_Top_Slice = output;
}

void BROCCOLI_LIB::SetOutputPermutationDistribution(float* output)
{
	h_Permutation_Distribution = output;
}

void BROCCOLI_LIB::SetOutputAMatrix(float* a)
{
	h_A_Matrix_Out = a;
}

void BROCCOLI_LIB::SetOutputHVector(float* h)
{
	h_h_Vector_Out = h;
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

int BROCCOLI_LIB::GetProgramBinarySize()
{
	return programBinarySize;
}

int BROCCOLI_LIB::GetWrittenElements()
{
	return writtenElements;
}

int BROCCOLI_LIB::GetOpenCLPlatformIDsError()
{
	return getPlatformIDsError;
}

int BROCCOLI_LIB::GetOpenCLDeviceIDsError()	
{
	return getDeviceIDsError;
}

int BROCCOLI_LIB::GetOpenCLCreateContextError()
{
	return createContextError;
}

int BROCCOLI_LIB::GetOpenCLContextInfoError()
{
	return getContextInfoError;
}

int BROCCOLI_LIB::GetOpenCLCreateCommandQueueError()
{
	return createCommandQueueError;
}

int BROCCOLI_LIB::GetOpenCLCreateProgramError()
{
	return createProgramError;
}

int BROCCOLI_LIB::GetOpenCLBuildProgramError()
{
	return buildProgramError;
}

int BROCCOLI_LIB::GetOpenCLProgramBuildInfoError()
{
	return getProgramBuildInfoError;
}





int* BROCCOLI_LIB::GetOpenCLCreateKernelErrors()
{	
	OpenCLCreateKernelErrors[0] = createKernelErrorMemset;
	OpenCLCreateKernelErrors[1] = createKernelErrorSeparableConvolutionRows;
	OpenCLCreateKernelErrors[2] = createKernelErrorSeparableConvolutionColumns;
	OpenCLCreateKernelErrors[3] = createKernelErrorSeparableConvolutionRods;
	OpenCLCreateKernelErrors[4] = createKernelErrorNonseparableConvolution3DComplexThreeFilters;
	OpenCLCreateKernelErrors[5] = createKernelErrorCalculatePhaseDifferencesAndCertainties;
	OpenCLCreateKernelErrors[6] = createKernelErrorCalculatePhaseGradientsX;
	OpenCLCreateKernelErrors[7] = createKernelErrorCalculatePhaseGradientsY;
	OpenCLCreateKernelErrors[8] = createKernelErrorCalculatePhaseGradientsZ;
	OpenCLCreateKernelErrors[9] = createKernelErrorCalculateAMatrixAndHVector2DValuesX;
	OpenCLCreateKernelErrors[10] = createKernelErrorCalculateAMatrixAndHVector2DValuesY;
	OpenCLCreateKernelErrors[11] = createKernelErrorCalculateAMatrixAndHVector2DValuesZ;
	OpenCLCreateKernelErrors[12] = createKernelErrorCalculateAMatrix1DValues;
	OpenCLCreateKernelErrors[13] = createKernelErrorCalculateHVector1DValues;
	OpenCLCreateKernelErrors[14] = createKernelErrorCalculateAMatrix;
	OpenCLCreateKernelErrors[15] = createKernelErrorCalculateHVector;
	OpenCLCreateKernelErrors[16] = createKernelErrorInterpolateVolumeNearestParametric;
	OpenCLCreateKernelErrors[17] = createKernelErrorInterpolateVolumeLinearParametric;
	OpenCLCreateKernelErrors[18] = createKernelErrorInterpolateVolumeCubicParametric;
	OpenCLCreateKernelErrors[19] = createKernelErrorInterpolateVolumeNearestNonParametric;
	OpenCLCreateKernelErrors[20] = createKernelErrorInterpolateVolumeLinearNonParametric;
	OpenCLCreateKernelErrors[21] = createKernelErrorInterpolateVolumeCubicNonParametric;
	OpenCLCreateKernelErrors[22] = createKernelErrorRescaleVolumeNearest;
	OpenCLCreateKernelErrors[23] = createKernelErrorRescaleVolumeLinear;
	OpenCLCreateKernelErrors[24] = createKernelErrorRescaleVolumeCubic;
	OpenCLCreateKernelErrors[25] = createKernelErrorCopyT1VolumeToMNI;
	OpenCLCreateKernelErrors[26] = createKernelErrorCopyEPIVolumeToT1;
	OpenCLCreateKernelErrors[27] = createKernelErrorCopyVolumeToNew;
	OpenCLCreateKernelErrors[28] = createKernelErrorMultiplyVolume;
	OpenCLCreateKernelErrors[29] = createKernelErrorMultiplyVolumes;
	OpenCLCreateKernelErrors[30] = createKernelErrorMultiplyVolumesOverwrite;
	OpenCLCreateKernelErrors[31] = createKernelErrorAddVolume;
	OpenCLCreateKernelErrors[32] = createKernelErrorAddVolumes;
	OpenCLCreateKernelErrors[33] = createKernelErrorAddVolumesOverwrite;
	OpenCLCreateKernelErrors[34] = createKernelErrorCalculateMagnitudes;
	OpenCLCreateKernelErrors[35] = createKernelErrorCalculateColumnSums;
	OpenCLCreateKernelErrors[36] = createKernelErrorCalculateRowSums;
	OpenCLCreateKernelErrors[37] = createKernelErrorCalculateColumnMaxs;
	OpenCLCreateKernelErrors[38] = createKernelErrorCalculateRowMaxs;
	OpenCLCreateKernelErrors[39] = createKernelErrorCalculateBetaValuesGLM;
	OpenCLCreateKernelErrors[40] = createKernelErrorCalculateStatisticalMapsGLMTTest;
	OpenCLCreateKernelErrors[41] = createKernelErrorCalculateStatisticalMapsGLMFTest;
	OpenCLCreateKernelErrors[42] = createKernelErrorEstimateAR4Models;
	OpenCLCreateKernelErrors[43] = createKernelErrorApplyWhiteningAR4;
	OpenCLCreateKernelErrors[44] = createKernelErrorGeneratePermutedVolumesFirstLevel;
	OpenCLCreateKernelErrors[45] = createKernelErrorGeneratePermutedVolumesSecondLevel;
	OpenCLCreateKernelErrors[46] = createKernelErrorRemoveLinearFit;



	return OpenCLCreateKernelErrors;
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
	OpenCLCreateBufferErrors[31] = createBufferErrorStatisticalMapsMNI;
	OpenCLCreateBufferErrors[32] = createBufferErrorResidualVariancesMNI;

    return OpenCLCreateBufferErrors;
}

int* BROCCOLI_LIB::GetOpenCLRunKernelErrors()
{
    OpenCLRunKernelErrors[0] = runKernelErrorNonseparableConvolution3DComplexThreeFilters;
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
	OpenCLRunKernelErrors[14] = runKernelErrorCalculateBetaValuesGLM;
	OpenCLRunKernelErrors[15] = runKernelErrorCalculateStatisticalMapsGLM;
    OpenCLRunKernelErrors[16] =	runKernelErrorRescaleVolume;
	OpenCLRunKernelErrors[17] = runKernelErrorCopyVolume;
	OpenCLRunKernelErrors[18] = runKernelErrorEstimateAR4Models;
	OpenCLRunKernelErrors[19] = runKernelErrorApplyAR4Whitening;

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









// Processing



// Preprocessing


// Copy a slice of the quadrature filters to constant memory
void BROCCOLI_LIB::CopyThreeQuadratureFiltersToConstantMemory(cl_mem c_Filter_1_Real, 
	                                                          cl_mem c_Filter_1_Imag, 
															  cl_mem c_Filter_2_Real, 
															  cl_mem c_Filter_2_Imag, 
															  cl_mem c_Filter_3_Real, 
															  cl_mem c_Filter_3_Imag, 
															  float* h_Filter_1_Real, 
															  float* h_Filter_1_Imag, 
															  float* h_Filter_2_Real, 
															  float* h_Filter_2_Imag, 
															  float* h_Filter_3_Real, 
															  float* h_Filter_3_Imag, 
															  int z, 
															  int FILTER_SIZE)
{	
	/*
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_1_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_1_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_1_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_1_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_2_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_2_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_2_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_2_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_3_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_3_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_3_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_3_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);	
	*/

	/*
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_1_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_1_Parametric_Registration_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_1_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_1_Parametric_Registration_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_2_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_2_Parametric_Registration_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_2_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_2_Parametric_Registration_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_3_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_3_Parametric_Registration_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Quadrature_Filter_3_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Quadrature_Filter_3_Parametric_Registration_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);	
	*/

	clEnqueueWriteBuffer(commandQueue, c_Filter_1_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Filter_1_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Filter_1_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Filter_1_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Filter_2_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Filter_2_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Filter_2_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Filter_2_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Filter_3_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Filter_3_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Filter_3_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Filter_3_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);	
}


void BROCCOLI_LIB::NonseparableConvolution3D(cl_mem d_q1, cl_mem d_q2, cl_mem d_q3, cl_mem d_Volume, cl_mem c_Filter_1_Real, cl_mem c_Filter_1_Imag, cl_mem c_Filter_2_Real, cl_mem c_Filter_2_Imag, cl_mem c_Filter_3_Real, cl_mem c_Filter_3_Imag, float* h_Filter_1_Real, float* h_Filter_1_Imag, float* h_Filter_2_Real, float* h_Filter_2_Imag, float* h_Filter_3_Real, float* h_Filter_3_Imag, int DATA_W, int DATA_H, int DATA_D)
{
	SetGlobalAndLocalWorkSizesNonSeparableConvolution(DATA_W, DATA_H, DATA_D);

	clSetKernelArg(NonseparableConvolution3DComplexThreeFiltersKernel, 0, sizeof(cl_mem), &d_q1);
	clSetKernelArg(NonseparableConvolution3DComplexThreeFiltersKernel, 1, sizeof(cl_mem), &d_q2);
	clSetKernelArg(NonseparableConvolution3DComplexThreeFiltersKernel, 2, sizeof(cl_mem), &d_q3);
	clSetKernelArg(NonseparableConvolution3DComplexThreeFiltersKernel, 3, sizeof(cl_mem), &d_Volume);
	clSetKernelArg(NonseparableConvolution3DComplexThreeFiltersKernel, 4, sizeof(cl_mem), &c_Filter_1_Real);
	clSetKernelArg(NonseparableConvolution3DComplexThreeFiltersKernel, 5, sizeof(cl_mem), &c_Filter_1_Imag);
	clSetKernelArg(NonseparableConvolution3DComplexThreeFiltersKernel, 6, sizeof(cl_mem), &c_Filter_2_Real);
	clSetKernelArg(NonseparableConvolution3DComplexThreeFiltersKernel, 7, sizeof(cl_mem), &c_Filter_2_Imag);
	clSetKernelArg(NonseparableConvolution3DComplexThreeFiltersKernel, 8, sizeof(cl_mem), &c_Filter_3_Real);
	clSetKernelArg(NonseparableConvolution3DComplexThreeFiltersKernel, 9, sizeof(cl_mem), &c_Filter_3_Imag);	
	clSetKernelArg(NonseparableConvolution3DComplexThreeFiltersKernel, 11, sizeof(int), &DATA_W);
	clSetKernelArg(NonseparableConvolution3DComplexThreeFiltersKernel, 12, sizeof(int), &DATA_H);
	clSetKernelArg(NonseparableConvolution3DComplexThreeFiltersKernel, 13, sizeof(int), &DATA_D);

	// Reset filter responses
	SetMemoryFloat2(d_q1, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemoryFloat2(d_q2, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemoryFloat2(d_q3, 0.0f, DATA_W * DATA_H * DATA_D);
	
	// Do 3D convolution by summing 2D convolutions
	int z_offset = -(IMAGE_REGISTRATION_FILTER_SIZE - 1)/2;
	for (int zz = IMAGE_REGISTRATION_FILTER_SIZE -1; zz >= 0; zz--)
	{
		CopyThreeQuadratureFiltersToConstantMemory(c_Filter_1_Real, c_Filter_1_Imag, c_Filter_2_Real, c_Filter_2_Imag, c_Filter_3_Real, c_Filter_3_Imag, h_Filter_1_Real, h_Filter_1_Imag, h_Filter_2_Real, h_Filter_2_Imag, h_Filter_3_Real, h_Filter_3_Imag, zz, IMAGE_REGISTRATION_FILTER_SIZE);

		clSetKernelArg(NonseparableConvolution3DComplexThreeFiltersKernel, 10, sizeof(int), &z_offset);
		runKernelErrorNonseparableConvolution3DComplexThreeFilters = clEnqueueNDRangeKernel(commandQueue, NonseparableConvolution3DComplexThreeFiltersKernel, 3, NULL, globalWorkSizeNonseparableConvolution3DComplex, localWorkSizeNonseparableConvolution3DComplex, 0, NULL, NULL);

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

void BROCCOLI_LIB::SetMemoryFloat2(cl_mem memory, float value, int N)
{
	SetGlobalAndLocalWorkSizesMemset(N);
	clSetKernelArg(MemsetFloat2Kernel, 0, sizeof(cl_mem), &memory);
	clSetKernelArg(MemsetFloat2Kernel, 1, sizeof(float), &value);
	clSetKernelArg(MemsetFloat2Kernel, 2, sizeof(int), &N);		
	runKernelErrorMemset = clEnqueueNDRangeKernel(commandQueue, MemsetFloat2Kernel, 1, NULL, globalWorkSizeMemset, localWorkSizeMemset, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::SetMemoryDouble(cl_mem memory, double value, int N)
{
	SetGlobalAndLocalWorkSizesMemset(N);
	clSetKernelArg(MemsetDoubleKernel, 0, sizeof(cl_mem), &memory);
	clSetKernelArg(MemsetDoubleKernel, 1, sizeof(double), &value);
	clSetKernelArg(MemsetDoubleKernel, 2, sizeof(int), &N);		
	runKernelErrorMemset = clEnqueueNDRangeKernel(commandQueue, MemsetDoubleKernel, 1, NULL, globalWorkSizeMemset, localWorkSizeMemset, 0, NULL, NULL);
	clFinish(commandQueue);
}

// This function is used by all parametric registration functions, to setup necessary parameters
void BROCCOLI_LIB::AlignTwoVolumesParametricSetup(int DATA_W, int DATA_H, int DATA_D)
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

	/*
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
	*/

	d_q11 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq11Real);
	d_q12 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq12Real);
	d_q13 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq13Real);
	
	d_q21 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq21Real);
	d_q22 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq22Real);
	d_q23 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq23Real);
	

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

	
	clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 0, sizeof(cl_mem), &d_Phase_Differences);
	clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 1, sizeof(cl_mem), &d_Phase_Certainties);
	clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 4, sizeof(int), &DATA_W);
	clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 5, sizeof(int), &DATA_H);
	clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 6, sizeof(int), &DATA_D);
			
	clSetKernelArg(CalculatePhaseGradientsXKernel, 0, sizeof(cl_mem), &d_Phase_Gradients);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 1, sizeof(cl_mem), &d_q11);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 2, sizeof(cl_mem), &d_q21);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(CalculatePhaseGradientsXKernel, 5, sizeof(int), &DATA_D);
		
	clSetKernelArg(CalculatePhaseGradientsYKernel, 0, sizeof(cl_mem), &d_Phase_Gradients);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 1, sizeof(cl_mem), &d_q12);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 2, sizeof(cl_mem), &d_q22);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(CalculatePhaseGradientsYKernel, 5, sizeof(int), &DATA_D);
	
	clSetKernelArg(CalculatePhaseGradientsZKernel, 0, sizeof(cl_mem), &d_Phase_Gradients);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 1, sizeof(cl_mem), &d_q13);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 2, sizeof(cl_mem), &d_q23);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(CalculatePhaseGradientsZKernel, 5, sizeof(int), &DATA_D);
		
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

	clSetKernelArg(InterpolateVolumeNearestParametricKernel, 0, sizeof(cl_mem), &d_Aligned_Volume);
	clSetKernelArg(InterpolateVolumeNearestParametricKernel, 1, sizeof(cl_mem), &d_Original_Volume);
	clSetKernelArg(InterpolateVolumeNearestParametricKernel, 2, sizeof(cl_mem), &c_Registration_Parameters);
	clSetKernelArg(InterpolateVolumeNearestParametricKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(InterpolateVolumeNearestParametricKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(InterpolateVolumeNearestParametricKernel, 5, sizeof(int), &DATA_D);	
	clSetKernelArg(InterpolateVolumeNearestParametricKernel, 6, sizeof(int), &volume);	

	clSetKernelArg(InterpolateVolumeLinearParametricKernel, 0, sizeof(cl_mem), &d_Aligned_Volume);
	clSetKernelArg(InterpolateVolumeLinearParametricKernel, 1, sizeof(cl_mem), &d_Original_Volume);
	clSetKernelArg(InterpolateVolumeLinearParametricKernel, 2, sizeof(cl_mem), &c_Registration_Parameters);
	clSetKernelArg(InterpolateVolumeLinearParametricKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(InterpolateVolumeLinearParametricKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(InterpolateVolumeLinearParametricKernel, 5, sizeof(int), &DATA_D);	
	clSetKernelArg(InterpolateVolumeLinearParametricKernel, 6, sizeof(int), &volume);	
	
	clSetKernelArg(InterpolateVolumeCubicParametricKernel, 0, sizeof(cl_mem), &d_Aligned_Volume);
	clSetKernelArg(InterpolateVolumeCubicParametricKernel, 1, sizeof(cl_mem), &d_Original_Volume);
	clSetKernelArg(InterpolateVolumeCubicParametricKernel, 2, sizeof(cl_mem), &c_Registration_Parameters);
	clSetKernelArg(InterpolateVolumeCubicParametricKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(InterpolateVolumeCubicParametricKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(InterpolateVolumeCubicParametricKernel, 5, sizeof(int), &DATA_D);	
	clSetKernelArg(InterpolateVolumeCubicParametricKernel, 6, sizeof(int), &volume);	
}





// This function is the foundation for all the parametric image registration functions
void BROCCOLI_LIB::AlignTwoVolumesParametric(float *h_Registration_Parameters_Align_Two_Volumes, float* h_Rotations, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_ITERATIONS, int ALIGNMENT_TYPE, int INTERPOLATION_MODE)
{
	// Calculate the filter responses for the reference volume (only needed once)
	NonseparableConvolution3D(d_q11, d_q12, d_q13, d_Reference_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_1_Parametric_Registration_Real, h_Quadrature_Filter_1_Parametric_Registration_Imag, h_Quadrature_Filter_2_Parametric_Registration_Real, h_Quadrature_Filter_2_Parametric_Registration_Imag, h_Quadrature_Filter_3_Parametric_Registration_Real, h_Quadrature_Filter_3_Parametric_Registration_Imag, DATA_W, DATA_H, DATA_D);


	//clEnqueueReadBuffer(commandQueue, d_q11, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_1, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_q12, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_2, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_q13, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_3, 0, NULL, NULL);



	// Reset the parameter vector
	for (int p = 0; p < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; p++)
	{
		h_Registration_Parameters_Align_Two_Volumes[p] = 0.0f;
		h_Registration_Parameters[p] = 0.0f;
	}
	
	// Run the registration algorithm for a number of iterations
	for (int it = 0; it < NUMBER_OF_ITERATIONS; it++)
	{
		NonseparableConvolution3D(d_q21, d_q22, d_q23, d_Aligned_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_1_Parametric_Registration_Real, h_Quadrature_Filter_1_Parametric_Registration_Imag, h_Quadrature_Filter_2_Parametric_Registration_Real, h_Quadrature_Filter_2_Parametric_Registration_Imag, h_Quadrature_Filter_3_Parametric_Registration_Real, h_Quadrature_Filter_3_Parametric_Registration_Imag, DATA_W, DATA_H, DATA_D);

		//clEnqueueReadBuffer(commandQueue, d_q21, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_1, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_q22, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_2, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_q23, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_3, 0, NULL, NULL);

		// Calculate phase differences, certainties and phase gradients in the X direction
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 2, sizeof(cl_mem), &d_q11);
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 3, sizeof(cl_mem), &d_q21);
		runKernelErrorCalculatePhaseDifferencesAndCertainties = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesAndCertaintiesKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);
		clFinish(commandQueue);


		runKernelErrorCalculatePhaseGradientsX = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseGradientsXKernel, 3, NULL, globalWorkSizeCalculatePhaseGradients, localWorkSizeCalculatePhaseGradients, 0, NULL, NULL);
		clFinish(commandQueue);


		// Calculate values for the A-matrix and h-vector in the X direction
		runKernelErrorCalculateAMatrixAndHVector2DValuesX = clEnqueueNDRangeKernel(commandQueue, CalculateAMatrixAndHVector2DValuesXKernel, 3, NULL, globalWorkSizeCalculateAMatrixAndHVector2DValuesX, localWorkSizeCalculateAMatrixAndHVector2DValuesX, 0, NULL, NULL);
		clFinish(commandQueue);
		
		// Calculate phase differences, certainties and phase gradients in the Y direction
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 2, sizeof(cl_mem), &d_q12);
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 3, sizeof(cl_mem), &d_q22);
		runKernelErrorCalculatePhaseDifferencesAndCertainties = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesAndCertaintiesKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);
		clFinish(commandQueue);

		runKernelErrorCalculatePhaseGradientsY = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseGradientsYKernel, 3, NULL, globalWorkSizeCalculatePhaseGradients, localWorkSizeCalculatePhaseGradients, 0, NULL, NULL);
		clFinish(commandQueue);
		
		// Calculate values for the A-matrix and h-vector in the Y direction
		runKernelErrorCalculateAMatrixAndHVector2DValuesY = clEnqueueNDRangeKernel(commandQueue, CalculateAMatrixAndHVector2DValuesYKernel, 3, NULL, globalWorkSizeCalculateAMatrixAndHVector2DValuesY, localWorkSizeCalculateAMatrixAndHVector2DValuesY, 0, NULL, NULL);
		clFinish(commandQueue);
		
		// Calculate phase differences, certainties and phase gradients in the Z direction
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 2, sizeof(cl_mem), &d_q13);
		clSetKernelArg(CalculatePhaseDifferencesAndCertaintiesKernel, 3, sizeof(cl_mem), &d_q23);
		runKernelErrorCalculatePhaseDifferencesAndCertainties = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesAndCertaintiesKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);
		clFinish(commandQueue);

		runKernelErrorCalculatePhaseGradientsZ = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseGradientsZKernel, 3, NULL, globalWorkSizeCalculatePhaseGradients, localWorkSizeCalculatePhaseGradients, 0, NULL, NULL);
		clFinish(commandQueue);

		//clEnqueueReadBuffer(commandQueue, d_Phase_Differences, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Differences, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_Phase_Gradients, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Gradients, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_Phase_Certainties, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Certainties, 0, NULL, NULL);

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
			//h_h_Vector_Out[j] = h_h_Vector[j];
			for (int i = 0; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; i++)
			{
				h_A_Matrix[j + i*NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS] = h_A_Matrix[i + j*NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS];
				//h_A_Matrix_Out[j + i*NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS] = h_A_Matrix[i + j*NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS];
			}
		}


		// Solve the equation system A * p = h to obtain the parameter vector
		SolveEquationSystem(h_Registration_Parameters, h_A_Matrix, h_h_Vector, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS);

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
			//RemoveTransformationScaling(h_Registration_Parameters);


			//for (int i = 0; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; i++)
			//{
			//	h_Registration_Parameters_Align_Two_Volumes[i] += h_Registration_Parameters[i];
			//}

			//RemoveTransformationScaling(h_Registration_Parameters_Align_Two_Volumes);


			AddAffineRegistrationParameters(h_Registration_Parameters_Align_Two_Volumes,h_Registration_Parameters);

			RemoveTransformationScaling(h_Registration_Parameters_Align_Two_Volumes);

		}
		// Keep all parameters
		else if (ALIGNMENT_TYPE == AFFINE)
		{

			//for (int i = 0; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; i++)
			//{
			//	h_Registration_Parameters_Align_Two_Volumes[i] += h_Registration_Parameters[i];
			//}


			AddAffineRegistrationParameters(h_Registration_Parameters_Align_Two_Volumes,h_Registration_Parameters);
		}

		/*
		for (int j = 0; j < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; j++)
		{
			h_h_Vector_Out[j] = h_Registration_Parameters[j];
		}
		*/

		// Copy parameter vector to constant memory
		clEnqueueWriteBuffer(commandQueue, c_Registration_Parameters, CL_TRUE, 0, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), h_Registration_Parameters_Align_Two_Volumes, 0, NULL, NULL);

		// Interpolate to get the new volume
		runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeLinear, localWorkSizeInterpolateVolumeLinear, 0, NULL, NULL);

		clFinish(commandQueue);
	}

	if (ALIGNMENT_TYPE == RIGID)
	{
		CalculateRotationAnglesFromRotationMatrix(h_Rotations, h_Registration_Parameters_Align_Two_Volumes);
	}
}





// This function is used by all parametric registration functions, to setup necessary parameters
void BROCCOLI_LIB::AlignTwoVolumesNonParametricSetup(int DATA_W, int DATA_H, int DATA_D)
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

	d_q11 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq11);
	d_q12 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq12);
	d_q13 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq13);
	d_q14 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq14);
	d_q15 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq15);
	d_q16 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq16);

	d_q21 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq21);
	d_q22 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq22);
	d_q23 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq23);
	d_q24 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq24);
	d_q25 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq25);
	d_q26 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(cl_float2), NULL, &createBufferErrorq26);

	d_t11 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrort11);
	d_t12 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrort12);
	d_t13 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrort13);
	d_t22 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrort22);
	d_t23 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrort23);
	d_t33 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrort33);

	d_a11 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrort11);
	d_a12 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrort12);
	d_a13 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrort13);
	d_a22 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrort22);
	d_a23 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrort23);
	d_a33 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrort33);

	d_h1 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrort11);
	d_h2 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrort12);
	d_h3 = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrort13);

	//d_Phase_Differences = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float) * NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION, NULL, &createBufferErrorPhaseDifferences);
	//d_Phase_Certainties = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float) * NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION, NULL, &createBufferErrorPhaseCertainties);


	d_Update_Displacement_Field_X = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);
	d_Update_Displacement_Field_Y = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);
	d_Update_Displacement_Field_Z = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);

	d_Temp_Displacement_Field_X = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);
	d_Temp_Displacement_Field_Y = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);
	d_Temp_Displacement_Field_Z = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);

	//d_Update_Certainty = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);



	// Allocate constant memory

	/*
	c_Quadrature_Filter_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(cl_float2), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Quadrature_Filter_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(cl_float2), NULL, &createBufferErrorQuadratureFilter2Real);
	c_Quadrature_Filter_3 = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(cl_float2), NULL, &createBufferErrorQuadratureFilter3Real);
	c_Quadrature_Filter_4 = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(cl_float2), NULL, &createBufferErrorQuadratureFilter4Real);
	c_Quadrature_Filter_5 = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(cl_float2), NULL, &createBufferErrorQuadratureFilter5Real);
	c_Quadrature_Filter_6 = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(cl_float2), NULL, &createBufferErrorQuadratureFilter6Real);
	*/

	c_Quadrature_Filter_1_Real = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Quadrature_Filter_1_Imag = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Quadrature_Filter_2_Real = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Quadrature_Filter_2_Imag = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Quadrature_Filter_3_Real = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Quadrature_Filter_3_Imag = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Quadrature_Filter_4_Real = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Quadrature_Filter_4_Imag = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Quadrature_Filter_5_Real = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Quadrature_Filter_5_Imag = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Quadrature_Filter_6_Real = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Quadrature_Filter_6_Imag = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_REGISTRATION_FILTER_SIZE * IMAGE_REGISTRATION_FILTER_SIZE * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);

	c_Filter_Directions_X = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Filter_Directions_Y = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Filter_Directions_Z = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);

	clEnqueueWriteBuffer(commandQueue, c_Filter_Directions_X, CL_TRUE, 0, NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION * sizeof(float), h_Filter_Directions_X, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Filter_Directions_Y, CL_TRUE, 0, NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION * sizeof(float), h_Filter_Directions_Y, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Filter_Directions_Z, CL_TRUE, 0, NUMBER_OF_FILTERS_FOR_NONPARAMETRIC_REGISTRATION * sizeof(float), h_Filter_Directions_Z, 0, NULL, NULL);

	// Set all kernel arguments


	/*
	clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 0, sizeof(cl_mem), &d_Phase_Differences);
	clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 1, sizeof(cl_mem), &d_Phase_Certainties);
	clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 2, sizeof(cl_mem), &d_t11);
	clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 3, sizeof(cl_mem), &d_t12);
	clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 4, sizeof(cl_mem), &d_t13);
	clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 5, sizeof(cl_mem), &d_t22);
	clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 6, sizeof(cl_mem), &d_t23);
	clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 7, sizeof(cl_mem), &d_t33);
	clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 16, sizeof(int), &DATA_W);
	clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 17, sizeof(int), &DATA_H);
	clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 18, sizeof(int), &DATA_D);
	*/

	clSetKernelArg(CalculateTensorComponentsKernel, 0, sizeof(cl_mem), &d_t11);
	clSetKernelArg(CalculateTensorComponentsKernel, 1, sizeof(cl_mem), &d_t12);
	clSetKernelArg(CalculateTensorComponentsKernel, 2, sizeof(cl_mem), &d_t13);
	clSetKernelArg(CalculateTensorComponentsKernel, 3, sizeof(cl_mem), &d_t22);
	clSetKernelArg(CalculateTensorComponentsKernel, 4, sizeof(cl_mem), &d_t23);
	clSetKernelArg(CalculateTensorComponentsKernel, 5, sizeof(cl_mem), &d_t33);
	clSetKernelArg(CalculateTensorComponentsKernel, 14, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateTensorComponentsKernel, 15, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateTensorComponentsKernel, 16, sizeof(int), &DATA_D);

	clSetKernelArg(CalculateTensorNormsKernel, 0, sizeof(cl_mem), &d_a11);
	clSetKernelArg(CalculateTensorNormsKernel, 1, sizeof(cl_mem), &d_t11);
	clSetKernelArg(CalculateTensorNormsKernel, 2, sizeof(cl_mem), &d_t12);
	clSetKernelArg(CalculateTensorNormsKernel, 3, sizeof(cl_mem), &d_t13);
	clSetKernelArg(CalculateTensorNormsKernel, 4, sizeof(cl_mem), &d_t22);
	clSetKernelArg(CalculateTensorNormsKernel, 5, sizeof(cl_mem), &d_t23);
	clSetKernelArg(CalculateTensorNormsKernel, 6, sizeof(cl_mem), &d_t33);
	clSetKernelArg(CalculateTensorNormsKernel, 7, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateTensorNormsKernel, 8, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateTensorNormsKernel, 9, sizeof(int), &DATA_D);

	/*
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 0, sizeof(cl_mem), &d_a11);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 1, sizeof(cl_mem), &d_a12);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 2, sizeof(cl_mem), &d_a13);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 3, sizeof(cl_mem), &d_a22);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 4, sizeof(cl_mem), &d_a23);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 5, sizeof(cl_mem), &d_a33);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 6, sizeof(cl_mem), &d_h1);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 7, sizeof(cl_mem), &d_h2);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 8, sizeof(cl_mem), &d_h3);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 9, sizeof(cl_mem), &d_Phase_Differences);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 10, sizeof(cl_mem), &d_Phase_Certainties);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 11, sizeof(cl_mem), &d_t11);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 12, sizeof(cl_mem), &d_t12);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 13, sizeof(cl_mem), &d_t13);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 14, sizeof(cl_mem), &d_t22);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 15, sizeof(cl_mem), &d_t23);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 16, sizeof(cl_mem), &d_t33);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 17, sizeof(cl_mem), &c_Filter_Directions_X);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 18, sizeof(cl_mem), &c_Filter_Directions_Y);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 19, sizeof(cl_mem), &c_Filter_Directions_Z);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 20, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 21, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 22, sizeof(int), &DATA_D);
	*/

	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 0, sizeof(cl_mem), &d_a11);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 1, sizeof(cl_mem), &d_a12);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 2, sizeof(cl_mem), &d_a13);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 3, sizeof(cl_mem), &d_a22);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 4, sizeof(cl_mem), &d_a23);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 5, sizeof(cl_mem), &d_a33);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 6, sizeof(cl_mem), &d_h1);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 7, sizeof(cl_mem), &d_h2);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 8, sizeof(cl_mem), &d_h3);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 11, sizeof(cl_mem), &d_t11);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 12, sizeof(cl_mem), &d_t12);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 13, sizeof(cl_mem), &d_t13);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 14, sizeof(cl_mem), &d_t22);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 15, sizeof(cl_mem), &d_t23);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 16, sizeof(cl_mem), &d_t33);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 17, sizeof(cl_mem), &c_Filter_Directions_X);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 18, sizeof(cl_mem), &c_Filter_Directions_Y);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 19, sizeof(cl_mem), &c_Filter_Directions_Z);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 20, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 21, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 22, sizeof(int), &DATA_D);

	/*
	clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 0, sizeof(cl_mem), &d_Temp_Displacement_Field_X);
	clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 1, sizeof(cl_mem), &d_Temp_Displacement_Field_Y);
	clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 2, sizeof(cl_mem), &d_Temp_Displacement_Field_Z);
	clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 3, sizeof(cl_mem), &d_Update_Certainty);
	clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 4, sizeof(cl_mem), &d_a11);
	clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 5, sizeof(cl_mem), &d_a12);
	clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 6, sizeof(cl_mem), &d_a13);
	clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 7, sizeof(cl_mem), &d_a22);
	clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 8, sizeof(cl_mem), &d_a23);
	clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 9, sizeof(cl_mem), &d_a33);
	clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 10, sizeof(cl_mem), &d_h1);
	clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 11, sizeof(cl_mem), &d_h2);
	clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 12, sizeof(cl_mem), &d_h3);
	clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 13, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 14, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateDisplacementAndCertaintyUpdateKernel, 15, sizeof(int), &DATA_D);
	*/

	clSetKernelArg(CalculateDisplacementUpdateKernel, 0, sizeof(cl_mem), &d_Temp_Displacement_Field_X);
	clSetKernelArg(CalculateDisplacementUpdateKernel, 1, sizeof(cl_mem), &d_Temp_Displacement_Field_Y);
	clSetKernelArg(CalculateDisplacementUpdateKernel, 2, sizeof(cl_mem), &d_Temp_Displacement_Field_Z);
	clSetKernelArg(CalculateDisplacementUpdateKernel, 3, sizeof(cl_mem), &d_a11);
	clSetKernelArg(CalculateDisplacementUpdateKernel, 4, sizeof(cl_mem), &d_a12);
	clSetKernelArg(CalculateDisplacementUpdateKernel, 5, sizeof(cl_mem), &d_a13);
	clSetKernelArg(CalculateDisplacementUpdateKernel, 6, sizeof(cl_mem), &d_a22);
	clSetKernelArg(CalculateDisplacementUpdateKernel, 7, sizeof(cl_mem), &d_a23);
	clSetKernelArg(CalculateDisplacementUpdateKernel, 8, sizeof(cl_mem), &d_a33);
	clSetKernelArg(CalculateDisplacementUpdateKernel, 9, sizeof(cl_mem), &d_h1);
	clSetKernelArg(CalculateDisplacementUpdateKernel, 10, sizeof(cl_mem), &d_h2);
	clSetKernelArg(CalculateDisplacementUpdateKernel, 11, sizeof(cl_mem), &d_h3);
	clSetKernelArg(CalculateDisplacementUpdateKernel, 12, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateDisplacementUpdateKernel, 13, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateDisplacementUpdateKernel, 14, sizeof(int), &DATA_D);

	int volume = 0;


	/*
	clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 0, sizeof(cl_mem), &d_Aligned_Volume);
	clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 1, sizeof(cl_mem), &d_Original_Volume);
	clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 2, sizeof(cl_mem), &d_Displacement_Field);
	clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 5, sizeof(int), &DATA_D);
	clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 6, sizeof(int), &volume);
	*/


	clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 0, sizeof(cl_mem), &d_Aligned_Volume);
	clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 1, sizeof(cl_mem), &d_Original_Volume);

	clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 8, sizeof(int), &volume);

	/*
	clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 0, sizeof(cl_mem), &d_Aligned_Volume);
	clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 1, sizeof(cl_mem), &d_Original_Volume);
	clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 2, sizeof(cl_mem), &d_Displacement_Field);
	clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 5, sizeof(int), &DATA_D);
	clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 6, sizeof(int), &volume);
	*/
}

void BROCCOLI_LIB::CalculateTensorMagnitude(cl_mem d_Tensor_Magnitudes, cl_mem d_Volume, int DATA_W, int DATA_H, int DATA_D)
{
	AlignTwoVolumesNonParametricSetup(DATA_W,DATA_H,DATA_D);

	NonseparableConvolution3D(d_q21, d_q22, d_q23, d_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_1_NonParametric_Registration_Real, h_Quadrature_Filter_1_NonParametric_Registration_Imag, h_Quadrature_Filter_2_NonParametric_Registration_Real, h_Quadrature_Filter_2_NonParametric_Registration_Imag, h_Quadrature_Filter_3_NonParametric_Registration_Real, h_Quadrature_Filter_3_NonParametric_Registration_Imag, DATA_W, DATA_H, DATA_D);
	NonseparableConvolution3D(d_q24, d_q25, d_q26, d_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_4_NonParametric_Registration_Real, h_Quadrature_Filter_4_NonParametric_Registration_Imag, h_Quadrature_Filter_5_NonParametric_Registration_Real, h_Quadrature_Filter_5_NonParametric_Registration_Imag, h_Quadrature_Filter_6_NonParametric_Registration_Real, h_Quadrature_Filter_6_NonParametric_Registration_Imag, DATA_W, DATA_H, DATA_D);

	SetMemory(d_t11, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_t12, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_t13, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_t22, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_t23, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_t33, 0.0f, DATA_W * DATA_H * DATA_D);

	clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q11);
	clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q21);
	clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_1);
	clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_1);
	clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_1);
	clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_1);
	clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_1);
	clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_1);
	runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

	clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q12);
	clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q22);
	clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_2);
	clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_2);
	clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_2);
	clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_2);
	clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_2);
	clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_2);
	runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

	clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q13);
	clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q23);
	clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_3);
	clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_3);
	clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_3);
	clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_3);
	clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_3);
	clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_3);
	runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

	clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q14);
	clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q24);
	clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_4);
	clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_4);
	clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_4);
	clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_4);
	clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_4);
	clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_4);
	runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

	clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q15);
	clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q25);
	clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_5);
	clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_5);
	clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_5);
	clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_5);
	clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_5);
	clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_5);
	runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

	clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q16);
	clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q26);
	clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_6);
	clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_6);
	clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_6);
	clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_6);
	clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_6);
	clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_6);
	runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

	clSetKernelArg(CalculateTensorNormsKernel, 0, sizeof(cl_mem), &d_Tensor_Magnitudes);
	clSetKernelArg(CalculateTensorNormsKernel, 1, sizeof(cl_mem), &d_t11);
	clSetKernelArg(CalculateTensorNormsKernel, 2, sizeof(cl_mem), &d_t12);
	clSetKernelArg(CalculateTensorNormsKernel, 3, sizeof(cl_mem), &d_t13);
	clSetKernelArg(CalculateTensorNormsKernel, 4, sizeof(cl_mem), &d_t22);
	clSetKernelArg(CalculateTensorNormsKernel, 5, sizeof(cl_mem), &d_t23);
	clSetKernelArg(CalculateTensorNormsKernel, 6, sizeof(cl_mem), &d_t33);
	clSetKernelArg(CalculateTensorNormsKernel, 7, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateTensorNormsKernel, 8, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateTensorNormsKernel, 9, sizeof(int), &DATA_D);
	runKernelErrorCalculateTensorNorms = clEnqueueNDRangeKernel(commandQueue, CalculateTensorNormsKernel, 3, NULL, globalWorkSizeCalculateTensorNorms, localWorkSizeCalculateTensorNorms, 0, NULL, NULL);

	AlignTwoVolumesNonParametricCleanup();
}

// This function is the foundation for all the non-parametric image registration functions
void BROCCOLI_LIB::AlignTwoVolumesNonParametric(int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_ITERATIONS, int INTERPOLATION_MODE)
{
	// Calculate the filter responses for the reference volume (only needed once)
	//NonseparableConvolution3D(d_q11, d_q12, d_q13, d_q14, d_q15, d_q16, d_Reference_Volume, DATA_W, DATA_H, DATA_D);
	NonseparableConvolution3D(d_q11, d_q12, d_q13, d_Reference_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_1_NonParametric_Registration_Real, h_Quadrature_Filter_1_NonParametric_Registration_Imag, h_Quadrature_Filter_2_NonParametric_Registration_Real, h_Quadrature_Filter_2_NonParametric_Registration_Imag, h_Quadrature_Filter_3_NonParametric_Registration_Real, h_Quadrature_Filter_3_NonParametric_Registration_Imag, DATA_W, DATA_H, DATA_D);
	NonseparableConvolution3D(d_q14, d_q15, d_q16, d_Reference_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_4_NonParametric_Registration_Real, h_Quadrature_Filter_4_NonParametric_Registration_Imag, h_Quadrature_Filter_5_NonParametric_Registration_Real, h_Quadrature_Filter_5_NonParametric_Registration_Imag, h_Quadrature_Filter_6_NonParametric_Registration_Real, h_Quadrature_Filter_6_NonParametric_Registration_Imag, DATA_W, DATA_H, DATA_D);

	//clEnqueueReadBuffer(commandQueue, d_q11, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_1, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_q12, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_2, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_q13, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_3, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_q14, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_4, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_q15, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_5, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_q16, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_6, 0, NULL, NULL);

	SetMemory(d_Update_Displacement_Field_X, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_Update_Displacement_Field_Y, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_Update_Displacement_Field_Z, 0.0f, DATA_W * DATA_H * DATA_D);

	int zero, one, two, three, four, five;
	zero = 0; one = 1; two = 2; three = 3; four = 4; five = 5;

	// Run the registration algorithm for a number of iterations
	for (int it = 0; it < NUMBER_OF_ITERATIONS; it++)
	{
		//NonseparableConvolution3D(d_q21, d_q22, d_q23, d_q24, d_q25, d_q26, d_Aligned_Volume, DATA_W, DATA_H, DATA_D);
		NonseparableConvolution3D(d_q21, d_q22, d_q23, d_Aligned_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_1_NonParametric_Registration_Real, h_Quadrature_Filter_1_NonParametric_Registration_Imag, h_Quadrature_Filter_2_NonParametric_Registration_Real, h_Quadrature_Filter_2_NonParametric_Registration_Imag, h_Quadrature_Filter_3_NonParametric_Registration_Real, h_Quadrature_Filter_3_NonParametric_Registration_Imag, DATA_W, DATA_H, DATA_D);
		NonseparableConvolution3D(d_q24, d_q25, d_q26, d_Aligned_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_4_NonParametric_Registration_Real, h_Quadrature_Filter_4_NonParametric_Registration_Imag, h_Quadrature_Filter_5_NonParametric_Registration_Real, h_Quadrature_Filter_5_NonParametric_Registration_Imag, h_Quadrature_Filter_6_NonParametric_Registration_Real, h_Quadrature_Filter_6_NonParametric_Registration_Imag, DATA_W, DATA_H, DATA_D);

		//clEnqueueReadBuffer(commandQueue, d_q21, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_1, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_q22, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_2, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_q23, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_3, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_q24, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_4, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_q25, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_5, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_q26, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_6, 0, NULL, NULL);

		SetMemory(d_t11, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_t12, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_t13, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_t22, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_t23, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_t33, 0.0f, DATA_W * DATA_H * DATA_D);

		SetMemory(d_a11, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_a12, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_a13, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_a22, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_a23, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_a33, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_h1, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_h2, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_h3, 0.0f, DATA_W * DATA_H * DATA_D);
		/*
		// Calculate phase differences, certainties and tensor components, first quadrature filter
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 8, sizeof(cl_mem), &d_q11);
  	    clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 9, sizeof(cl_mem), &d_q21);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 10, sizeof(float), &M11_1);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 11, sizeof(float), &M12_1);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 12, sizeof(float), &M13_1);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 13, sizeof(float), &M22_1);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 14, sizeof(float), &M23_1);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 15, sizeof(float), &M33_1);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 19, sizeof(int), &zero);
		runKernelErrorCalculatePhaseDifferencesCertaintiesAndTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		// Calculate phase differences, certainties and tensor components, second quadrature filter
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 8, sizeof(cl_mem), &d_q12);
  	    clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 9, sizeof(cl_mem), &d_q22);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 10, sizeof(float), &M11_2);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 11, sizeof(float), &M12_2);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 12, sizeof(float), &M13_2);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 13, sizeof(float), &M22_2);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 14, sizeof(float), &M23_2);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 15, sizeof(float), &M33_2);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 19, sizeof(int), &one);
		runKernelErrorCalculatePhaseDifferencesCertaintiesAndTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		// Calculate phase differences, certainties and tensor components, third quadrature filter
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 8, sizeof(cl_mem), &d_q13);
  	    clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 9, sizeof(cl_mem), &d_q23);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 10, sizeof(float), &M11_3);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 11, sizeof(float), &M12_3);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 12, sizeof(float), &M13_3);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 13, sizeof(float), &M22_3);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 14, sizeof(float), &M23_3);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 15, sizeof(float), &M33_3);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 19, sizeof(int), &two);
		runKernelErrorCalculatePhaseDifferencesCertaintiesAndTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		// Calculate phase differences, certainties and tensor components, fourth quadrature filter
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 8, sizeof(cl_mem), &d_q14);
  	    clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 9, sizeof(cl_mem), &d_q24);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 10, sizeof(float), &M11_4);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 11, sizeof(float), &M12_4);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 12, sizeof(float), &M13_4);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 13, sizeof(float), &M22_4);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 14, sizeof(float), &M23_4);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 15, sizeof(float), &M33_4);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 19, sizeof(int), &three);
		runKernelErrorCalculatePhaseDifferencesCertaintiesAndTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		// Calculate phase differences, certainties and tensor components, fifth quadrature filter
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 8, sizeof(cl_mem), &d_q15);
  	    clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 9, sizeof(cl_mem), &d_q25);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 10, sizeof(float), &M11_5);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 11, sizeof(float), &M12_5);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 12, sizeof(float), &M13_5);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 13, sizeof(float), &M22_5);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 14, sizeof(float), &M23_5);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 15, sizeof(float), &M33_5);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 19, sizeof(int), &four);
		runKernelErrorCalculatePhaseDifferencesCertaintiesAndTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		// Calculate phase differences, certainties and tensor components, sixth quadrature filter
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 8, sizeof(cl_mem), &d_q16);
  	    clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 9, sizeof(cl_mem), &d_q26);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 10, sizeof(float), &M11_6);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 11, sizeof(float), &M12_6);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 12, sizeof(float), &M13_6);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 13, sizeof(float), &M22_6);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 14, sizeof(float), &M23_6);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 15, sizeof(float), &M33_6);
		clSetKernelArg(CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 19, sizeof(int), &five);
		runKernelErrorCalculatePhaseDifferencesCertaintiesAndTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculatePhaseDifferencesCertaintiesAndTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);
		*/

		clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q11);
  	    clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q21);
		clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_1);
		clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_1);
		clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_1);
		clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_1);
		clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_1);
		clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_1);
		runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q12);
  	    clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q22);
		clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_2);
		clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_2);
		clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_2);
		clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_2);
		clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_2);
		clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_2);
		runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q13);
  	    clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q23);
		clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_3);
		clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_3);
		clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_3);
		clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_3);
		clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_3);
		clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_3);
		runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q14);
  	    clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q24);
		clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_4);
		clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_4);
		clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_4);
		clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_4);
		clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_4);
		clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_4);
		runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q15);
  	    clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q25);
		clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_5);
		clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_5);
		clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_5);
		clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_5);
		clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_5);
		clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_5);
		runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q16);
  	    clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q26);
		clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_6);
		clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_6);
		clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_6);
		clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_6);
		clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_6);
		clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_6);
		runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		/*
		clEnqueueReadBuffer(commandQueue, d_t11, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t11, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_t12, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t12, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_t13, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t13, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_t22, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t22, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_t23, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t23, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_t33, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t33, 0, NULL, NULL);
		*/

		// Calculate tensor norms
		runKernelErrorCalculateTensorNorms = clEnqueueNDRangeKernel(commandQueue, CalculateTensorNormsKernel, 3, NULL, globalWorkSizeCalculateTensorNorms, localWorkSizeCalculateTensorNorms, 0, NULL, NULL);



		// Smooth tensor components
		CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, 2.25);
		//PerformSmoothing(d_Smoothed_Tensor_Norms, d_Tensor_Norms, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		//PerformSmoothingNormalized(d_t11, d_Tensor_Norms, d_Smoothed_Tensor_Norms, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		//PerformSmoothingNormalized(d_t12, d_Tensor_Norms, d_Smoothed_Tensor_Norms, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		//PerformSmoothingNormalized(d_t13, d_Tensor_Norms, d_Smoothed_Tensor_Norms, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		//PerformSmoothingNormalized(d_t22, d_Tensor_Norms, d_Smoothed_Tensor_Norms, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		//PerformSmoothingNormalized(d_t23, d_Tensor_Norms, d_Smoothed_Tensor_Norms, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		//PerformSmoothingNormalized(d_t33, d_Tensor_Norms, d_Smoothed_Tensor_Norms, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);

		PerformSmoothing(d_t11, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_t12, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_t13, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_t22, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_t23, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_t33, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);

		//clEnqueueReadBuffer(commandQueue, d_Tensor_Norms, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t11, 0, NULL, NULL);


		//clEnqueueReadBuffer(commandQueue, d_t11, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Differences, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_t22, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Certainties, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_t33, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Gradients, 0, NULL, NULL);

		// Calculate tensor norms
		runKernelErrorCalculateTensorNorms = clEnqueueNDRangeKernel(commandQueue, CalculateTensorNormsKernel, 3, NULL, globalWorkSizeCalculateTensorNorms, localWorkSizeCalculateTensorNorms, 0, NULL, NULL);

		//clEnqueueReadBuffer(commandQueue, d_Tensor_Norms, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t11, 0, NULL, NULL);

		//clEnqueueReadBuffer(commandQueue, d_Tensor_Norms, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Differences, 0, NULL, NULL);


		// Find max norm (tensor norms are saved in d_a11, to save some memory)
		float max_norm = CalculateMax(d_a11, DATA_W, DATA_H, DATA_D);
		//float max_norm = 4.1f;

		// Normalize tensor components


		MultiplyVolume(d_t11, 1.0f/max_norm, DATA_W, DATA_H, DATA_D);
		MultiplyVolume(d_t12, 1.0f/max_norm, DATA_W, DATA_H, DATA_D);
		MultiplyVolume(d_t13, 1.0f/max_norm, DATA_W, DATA_H, DATA_D);
		MultiplyVolume(d_t22, 1.0f/max_norm, DATA_W, DATA_H, DATA_D);
		MultiplyVolume(d_t23, 1.0f/max_norm, DATA_W, DATA_H, DATA_D);
		MultiplyVolume(d_t33, 1.0f/max_norm, DATA_W, DATA_H, DATA_D);




		//clEnqueueReadBuffer(commandQueue, d_t11, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t11, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_t12, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t12, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_t13, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t13, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_t22, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t22, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_t23, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t23, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_t33, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t33, 0, NULL, NULL);



		// Calculate A-matrices and h-vectors
		//runKernelErrorCalculateAMatricesAndHVectors = clEnqueueNDRangeKernel(commandQueue, CalculateAMatricesAndHVectorsKernel, 3, NULL, globalWorkSizeCalculateAMatricesAndHVectors, localWorkSizeCalculateAMatricesAndHVectors, 0, NULL, NULL);

		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 9, sizeof(cl_mem), &d_q11);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 10, sizeof(cl_mem), &d_q21);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 23, sizeof(int), &zero);
		runKernelErrorCalculateAMatricesAndHVectors = clEnqueueNDRangeKernel(commandQueue, CalculateAMatricesAndHVectorsKernel, 3, NULL, globalWorkSizeCalculateAMatricesAndHVectors, localWorkSizeCalculateAMatricesAndHVectors, 0, NULL, NULL);

		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 9, sizeof(cl_mem), &d_q12);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 10, sizeof(cl_mem), &d_q22);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 23, sizeof(int), &one);
		runKernelErrorCalculateAMatricesAndHVectors = clEnqueueNDRangeKernel(commandQueue, CalculateAMatricesAndHVectorsKernel, 3, NULL, globalWorkSizeCalculateAMatricesAndHVectors, localWorkSizeCalculateAMatricesAndHVectors, 0, NULL, NULL);

		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 9, sizeof(cl_mem), &d_q13);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 10, sizeof(cl_mem), &d_q23);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 23, sizeof(int), &two);
		runKernelErrorCalculateAMatricesAndHVectors = clEnqueueNDRangeKernel(commandQueue, CalculateAMatricesAndHVectorsKernel, 3, NULL, globalWorkSizeCalculateAMatricesAndHVectors, localWorkSizeCalculateAMatricesAndHVectors, 0, NULL, NULL);

		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 9, sizeof(cl_mem), &d_q14);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 10, sizeof(cl_mem), &d_q24);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 23, sizeof(int), &three);
		runKernelErrorCalculateAMatricesAndHVectors = clEnqueueNDRangeKernel(commandQueue, CalculateAMatricesAndHVectorsKernel, 3, NULL, globalWorkSizeCalculateAMatricesAndHVectors, localWorkSizeCalculateAMatricesAndHVectors, 0, NULL, NULL);

		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 9, sizeof(cl_mem), &d_q15);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 10, sizeof(cl_mem), &d_q25);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 23, sizeof(int), &four);
		runKernelErrorCalculateAMatricesAndHVectors = clEnqueueNDRangeKernel(commandQueue, CalculateAMatricesAndHVectorsKernel, 3, NULL, globalWorkSizeCalculateAMatricesAndHVectors, localWorkSizeCalculateAMatricesAndHVectors, 0, NULL, NULL);

		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 9, sizeof(cl_mem), &d_q16);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 10, sizeof(cl_mem), &d_q26);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 23, sizeof(int), &five);
		runKernelErrorCalculateAMatricesAndHVectors = clEnqueueNDRangeKernel(commandQueue, CalculateAMatricesAndHVectorsKernel, 3, NULL, globalWorkSizeCalculateAMatricesAndHVectors, localWorkSizeCalculateAMatricesAndHVectors, 0, NULL, NULL);


		/*
		clEnqueueReadBuffer(commandQueue, d_h1, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Differences, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_h2, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Certainties, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_h3, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Gradients, 0, NULL, NULL);
		*/

		// Smooth components of A-matrices and h-vectors
		CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, 2.25);
		PerformSmoothing(d_a11, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_a12, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_a13, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_a22, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_a23, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_a33, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_h1, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_h2, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_h3, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);

		/*
		clEnqueueReadBuffer(commandQueue, d_a11, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t11, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_a12, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t12, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_a13, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t13, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_a22, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t22, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_a23, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t23, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_a33, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t33, 0, NULL, NULL);
		*/

		//runKernelErrorCalculateDisplacementAndCertaintyUpdate = clEnqueueNDRangeKernel(commandQueue, CalculateDisplacementAndCertaintyUpdateKernel, 3, NULL, globalWorkSizeCalculateDisplacementAndCertaintyUpdate, localWorkSizeCalculateDisplacementAndCertaintyUpdate, 0, NULL, NULL);
		runKernelErrorCalculateDisplacementUpdate = clEnqueueNDRangeKernel(commandQueue, CalculateDisplacementUpdateKernel, 3, NULL, globalWorkSizeCalculateDisplacementAndCertaintyUpdate, localWorkSizeCalculateDisplacementAndCertaintyUpdate, 0, NULL, NULL);

		//clEnqueueReadBuffer(commandQueue, d_Update_Displacement_Field_X, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Differences, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_Update_Displacement_Field_Y, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Certainties, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_Update_Displacement_Field_Z, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Gradients, 0, NULL, NULL);


		//MultiplyVolume(d_Update_Displacement_Field_X, 0.75f, DATA_W, DATA_H, DATA_D);
		//MultiplyVolume(d_Update_Displacement_Field_Y, 0.75f, DATA_W, DATA_H, DATA_D);
		//MultiplyVolume(d_Update_Displacement_Field_Z, 0.75f, DATA_W, DATA_H, DATA_D);


		/*
		clEnqueueReadBuffer(commandQueue, d_Update_Displacement_Field_X, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t11, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Update_Displacement_Field_Y, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t12, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Update_Displacement_Field_Z, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t13, 0, NULL, NULL);
		*/

		CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, 2.25);
		PerformSmoothing(d_Temp_Displacement_Field_X, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_Temp_Displacement_Field_Y, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_Temp_Displacement_Field_Z, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);

		AddVolumes(d_Update_Displacement_Field_X, d_Temp_Displacement_Field_X, DATA_W, DATA_H, DATA_D);
		AddVolumes(d_Update_Displacement_Field_Y, d_Temp_Displacement_Field_Y, DATA_W, DATA_H, DATA_D);
		AddVolumes(d_Update_Displacement_Field_Z, d_Temp_Displacement_Field_Z, DATA_W, DATA_H, DATA_D);

		//PerformSmoothing(d_Update_Displacement_Field_X, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		//PerformSmoothing(d_Update_Displacement_Field_Y, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		//PerformSmoothing(d_Update_Displacement_Field_Z, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);

		//clEnqueueReadBuffer(commandQueue, d_Update_Displacement_Field_X, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Displacement_Field_X, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_Update_Displacement_Field_Y, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Displacement_Field_Y, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_Update_Displacement_Field_Z, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Displacement_Field_Z, 0, NULL, NULL);


		//AddVolumes(d_Total_Displacement_Field_X, d_Update_Displacement_Field_X, DATA_W, DATA_H, DATA_D);
		//AddVolumes(d_Total_Displacement_Field_Y, d_Update_Displacement_Field_Y, DATA_W, DATA_H, DATA_D);
		//AddVolumes(d_Total_Displacement_Field_Z, d_Update_Displacement_Field_Z, DATA_W, DATA_H, DATA_D);


		// Interpolate to get the new volume
		clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 2, sizeof(cl_mem), &d_Update_Displacement_Field_X);
		clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 3, sizeof(cl_mem), &d_Update_Displacement_Field_Y);
		clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 4, sizeof(cl_mem), &d_Update_Displacement_Field_Z);
		runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearNonParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeLinear, localWorkSizeInterpolateVolumeLinear, 0, NULL, NULL);
		clFinish(commandQueue);

	}


	//clEnqueueReadBuffer(commandQueue, d_Aligned_Volume, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Aligned_T1_Volume_NonParametric, 0, NULL, NULL);
}

void BROCCOLI_LIB::AlignTwoVolumesNonParametricCleanup()
{
	// Free all the allocated memory on the device

	clReleaseMemObject(d_Original_Volume);
	clReleaseMemObject(d_Reference_Volume);
	clReleaseMemObject(d_Aligned_Volume);

	clReleaseMemObject(d_q11);
	clReleaseMemObject(d_q12);
	clReleaseMemObject(d_q13);
	clReleaseMemObject(d_q14);
	clReleaseMemObject(d_q15);
	clReleaseMemObject(d_q16);


	clReleaseMemObject(d_q21);
	clReleaseMemObject(d_q22);
	clReleaseMemObject(d_q23);
	clReleaseMemObject(d_q24);
	clReleaseMemObject(d_q25);
	clReleaseMemObject(d_q26);


	//clReleaseMemObject(d_Phase_Differences);
	//clReleaseMemObject(d_Phase_Certainties);

	/*
	clReleaseMemObject(c_Quadrature_Filter_1);
	clReleaseMemObject(c_Quadrature_Filter_2);
	clReleaseMemObject(c_Quadrature_Filter_3);
	clReleaseMemObject(c_Quadrature_Filter_4);
	clReleaseMemObject(c_Quadrature_Filter_5);
	clReleaseMemObject(c_Quadrature_Filter_6);
	*/

	clReleaseMemObject(c_Quadrature_Filter_1_Real);
	clReleaseMemObject(c_Quadrature_Filter_1_Imag);
	clReleaseMemObject(c_Quadrature_Filter_2_Real);
	clReleaseMemObject(c_Quadrature_Filter_2_Imag);
	clReleaseMemObject(c_Quadrature_Filter_3_Real);
	clReleaseMemObject(c_Quadrature_Filter_3_Imag);
	clReleaseMemObject(c_Quadrature_Filter_4_Real);
	clReleaseMemObject(c_Quadrature_Filter_4_Imag);
	clReleaseMemObject(c_Quadrature_Filter_5_Real);
	clReleaseMemObject(c_Quadrature_Filter_5_Imag);
	clReleaseMemObject(c_Quadrature_Filter_6_Real);
	clReleaseMemObject(c_Quadrature_Filter_6_Imag);

	clReleaseMemObject(c_Filter_Directions_X);
	clReleaseMemObject(c_Filter_Directions_Y);
	clReleaseMemObject(c_Filter_Directions_Z);

	clReleaseMemObject(d_t11);
	clReleaseMemObject(d_t12);
	clReleaseMemObject(d_t13);
	clReleaseMemObject(d_t22);
	clReleaseMemObject(d_t23);
	clReleaseMemObject(d_t33);

	clReleaseMemObject(d_a11);
	clReleaseMemObject(d_a12);
	clReleaseMemObject(d_a13);
	clReleaseMemObject(d_a22);
	clReleaseMemObject(d_a23);
	clReleaseMemObject(d_a33);

	clReleaseMemObject(d_h1);
	clReleaseMemObject(d_h2);
	clReleaseMemObject(d_h3);

	clReleaseMemObject(d_Update_Displacement_Field_X);
	clReleaseMemObject(d_Update_Displacement_Field_Y);
	clReleaseMemObject(d_Update_Displacement_Field_Z);
	//clReleaseMemObject(d_Update_Certainty);

	clReleaseMemObject(d_Temp_Displacement_Field_X);
	clReleaseMemObject(d_Temp_Displacement_Field_Y);
	clReleaseMemObject(d_Temp_Displacement_Field_Z);
}




void BROCCOLI_LIB::RemoveTransformationScaling(float* h_Registration_Parameters)
{
	Eigen::MatrixXd TransformationMatrix(3,3);

	// Make a copy of transformation matrix parameters
	TransformationMatrix(0,0) = (double)h_Registration_Parameters[3];
	TransformationMatrix(0,1) = (double)h_Registration_Parameters[4];
	TransformationMatrix(0,2) = (double)h_Registration_Parameters[5];
	TransformationMatrix(1,0) = (double)h_Registration_Parameters[6];
	TransformationMatrix(1,1) = (double)h_Registration_Parameters[7];
	TransformationMatrix(1,2) = (double)h_Registration_Parameters[8];
	TransformationMatrix(2,0) = (double)h_Registration_Parameters[9];
	TransformationMatrix(2,1) = (double)h_Registration_Parameters[10];
	TransformationMatrix(2,2) = (double)h_Registration_Parameters[11];

	// Add one to diagonal
	TransformationMatrix(0,0) += 1.0;
	TransformationMatrix(1,1) += 1.0;
	TransformationMatrix(2,2) += 1.0;

	// Calculate SVD
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(TransformationMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

	// Calculate transformation matrix without scaling (i.e. singular values = ones)
	Eigen::MatrixXd TransformationMatrixWithoutScaling = svd.matrixU() * svd.matrixV().transpose();

	// Remove one from diagonal
	TransformationMatrixWithoutScaling(0,0) -= 1.0;
	TransformationMatrixWithoutScaling(1,1) -= 1.0;
	TransformationMatrixWithoutScaling(2,2) -= 1.0;

	h_Registration_Parameters[3] = (float)TransformationMatrixWithoutScaling(0,0);
	h_Registration_Parameters[4] = (float)TransformationMatrixWithoutScaling(0,1);
	h_Registration_Parameters[5] = (float)TransformationMatrixWithoutScaling(0,2);
	h_Registration_Parameters[6] = (float)TransformationMatrixWithoutScaling(1,0);
	h_Registration_Parameters[7] = (float)TransformationMatrixWithoutScaling(1,1);
	h_Registration_Parameters[8] = (float)TransformationMatrixWithoutScaling(1,2);
	h_Registration_Parameters[9] = (float)TransformationMatrixWithoutScaling(2,0);
	h_Registration_Parameters[10] = (float)TransformationMatrixWithoutScaling(2,1);
	h_Registration_Parameters[11] = (float)TransformationMatrixWithoutScaling(2,2);
}



// Calculate Euler rotation angles from a rotation matrix
void BROCCOLI_LIB::CalculateRotationAnglesFromRotationMatrix(float* h_Rotations, float* h_Registration_Parameters)
{
	double h_Transformation_Matrix[9];
	double c1, c2, s1;
	double angle1, angle2, angle3;

	// Make a copy of transformation matrix parameters
	for (int i = 3; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; i++)
	{
		h_Transformation_Matrix[i-3] = (double)h_Registration_Parameters[i];
	}

	// Add ones in the diagonal
	h_Transformation_Matrix[0] += 1.0;
	h_Transformation_Matrix[4] += 1.0;
	h_Transformation_Matrix[8] += 1.0;

	// Calculate rotation angles

	// (p0  p1  p2)
	// (p3  p4  p5)
 	// (p6  p7  p8)

	/* Matlab equivalent
	angle1 = atan2(p_matrix(2,3),p_matrix(3,3))*180/pi;
	c2 = sqrt(p_matrix(1,1)^2 + p_matrix(1,2)^2);
	angle2 = atan2(-p_matrix(1,3),c2)*180/pi;
	s1 = sind(angle1);
	c1 = cosd(angle1);
	angle3 = atan2(s1*p_matrix(3,1)-c1*p_matrix(2,1),c1*p_matrix(2,2)-s1*p_matrix(3,2))*180/pi;
	rotations = [angle1, angle2, angle3];
	*/

	// Minus signs since the transformation matrix is transposed compared to Matlab

	angle1 = -atan2(h_Transformation_Matrix[5], h_Transformation_Matrix[8]) * 180.0/PI;
	c2 = sqrt(h_Transformation_Matrix[0] * h_Transformation_Matrix[0] + h_Transformation_Matrix[1] * h_Transformation_Matrix[1]);
	angle2 = -atan2(-h_Transformation_Matrix[2],c2)*180.0/PI;
	s1 = sin(angle1*PI/180.0);
	c1 = cos(angle1*PI/180.0);
	angle3 = -atan2(s1*h_Transformation_Matrix[6] - c1*h_Transformation_Matrix[3],c1*h_Transformation_Matrix[4] - s1*h_Transformation_Matrix[7])*180.0/PI;

	h_Rotations[0] = (float)angle1;
	h_Rotations[1] = (float)angle2;
	h_Rotations[2] = (float)angle3;
}

void BROCCOLI_LIB::ChangeVolumeSize(cl_mem d_Changed_Volume, cl_mem d_Original_Volume_, int ORIGINAL_DATA_W, int ORIGINAL_DATA_H, int ORIGINAL_DATA_D, int NEW_DATA_W, int NEW_DATA_H, int NEW_DATA_D, int INTERPOLATION_MODE)
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

	if (INTERPOLATION_MODE == LINEAR)
	{
		clSetKernelArg(RescaleVolumeLinearKernel, 0, sizeof(cl_mem), &d_Changed_Volume);
		clSetKernelArg(RescaleVolumeLinearKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
		clSetKernelArg(RescaleVolumeLinearKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
		clSetKernelArg(RescaleVolumeLinearKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
		clSetKernelArg(RescaleVolumeLinearKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
		clSetKernelArg(RescaleVolumeLinearKernel, 5, sizeof(int), &NEW_DATA_W);
		clSetKernelArg(RescaleVolumeLinearKernel, 6, sizeof(int), &NEW_DATA_H);
		clSetKernelArg(RescaleVolumeLinearKernel, 7, sizeof(int), &NEW_DATA_D);

		runKernelErrorRescaleVolume = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeLinearKernel, 3, NULL, globalWorkSizeRescaleVolumeLinear, localWorkSizeRescaleVolumeLinear, 0, NULL, NULL);
		clFinish(commandQueue);
	}
	else if (INTERPOLATION_MODE == CUBIC)
	{
		clSetKernelArg(RescaleVolumeCubicKernel, 0, sizeof(cl_mem), &d_Changed_Volume);
		clSetKernelArg(RescaleVolumeCubicKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
		clSetKernelArg(RescaleVolumeCubicKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
		clSetKernelArg(RescaleVolumeCubicKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
		clSetKernelArg(RescaleVolumeCubicKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
		clSetKernelArg(RescaleVolumeCubicKernel, 5, sizeof(int), &NEW_DATA_W);
		clSetKernelArg(RescaleVolumeCubicKernel, 6, sizeof(int), &NEW_DATA_H);
		clSetKernelArg(RescaleVolumeCubicKernel, 7, sizeof(int), &NEW_DATA_D);

		runKernelErrorRescaleVolume = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeCubicKernel, 3, NULL, globalWorkSizeRescaleVolumeCubic, localWorkSizeRescaleVolumeCubic, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	clReleaseMemObject(d_Volume_Texture);
}

void BROCCOLI_LIB::ChangeVolumeSize(cl_mem& d_Original_Volume, int ORIGINAL_DATA_W, int ORIGINAL_DATA_H, int ORIGINAL_DATA_D, int NEW_DATA_W, int NEW_DATA_H, int NEW_DATA_D, int INTERPOLATION_MODE)
{
	// Create a 3D image (texture) for fast interpolation
	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_INTENSITY;
	cl_mem d_Volume_Texture = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, ORIGINAL_DATA_W, ORIGINAL_DATA_H, ORIGINAL_DATA_D, 0, 0, NULL, NULL);

	// Copy the T1 volume to an image to interpolate from
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {ORIGINAL_DATA_W, ORIGINAL_DATA_H, ORIGINAL_DATA_D};
	clEnqueueCopyBufferToImage(commandQueue, d_Original_Volume, d_Volume_Texture, 0, origin, region, 0, NULL, NULL);

	// Throw away old volume and make a new one of the new size
	clReleaseMemObject(d_Original_Volume);
	d_Original_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  NEW_DATA_W * NEW_DATA_H * NEW_DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);

	float VOXEL_DIFFERENCE_X = (float)(ORIGINAL_DATA_W-1)/(float)(NEW_DATA_W-1);
	float VOXEL_DIFFERENCE_Y = (float)(ORIGINAL_DATA_H-1)/(float)(NEW_DATA_H-1);
	float VOXEL_DIFFERENCE_Z = (float)(ORIGINAL_DATA_D-1)/(float)(NEW_DATA_D-1);

	SetGlobalAndLocalWorkSizesRescaleVolume(NEW_DATA_W, NEW_DATA_H, NEW_DATA_D);

	if (INTERPOLATION_MODE == LINEAR)
	{
		clSetKernelArg(RescaleVolumeLinearKernel, 0, sizeof(cl_mem), &d_Original_Volume);
		clSetKernelArg(RescaleVolumeLinearKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
		clSetKernelArg(RescaleVolumeLinearKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
		clSetKernelArg(RescaleVolumeLinearKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
		clSetKernelArg(RescaleVolumeLinearKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
		clSetKernelArg(RescaleVolumeLinearKernel, 5, sizeof(int), &NEW_DATA_W);
		clSetKernelArg(RescaleVolumeLinearKernel, 6, sizeof(int), &NEW_DATA_H);
		clSetKernelArg(RescaleVolumeLinearKernel, 7, sizeof(int), &NEW_DATA_D);

		runKernelErrorRescaleVolume = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeLinearKernel, 3, NULL, globalWorkSizeRescaleVolumeLinear, localWorkSizeRescaleVolumeLinear, 0, NULL, NULL);
		clFinish(commandQueue);
	}
	else if (INTERPOLATION_MODE == CUBIC)
	{
		clSetKernelArg(RescaleVolumeCubicKernel, 0, sizeof(cl_mem), &d_Original_Volume);
		clSetKernelArg(RescaleVolumeCubicKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
		clSetKernelArg(RescaleVolumeCubicKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
		clSetKernelArg(RescaleVolumeCubicKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
		clSetKernelArg(RescaleVolumeCubicKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
		clSetKernelArg(RescaleVolumeCubicKernel, 5, sizeof(int), &NEW_DATA_W);
		clSetKernelArg(RescaleVolumeCubicKernel, 6, sizeof(int), &NEW_DATA_H);
		clSetKernelArg(RescaleVolumeCubicKernel, 7, sizeof(int), &NEW_DATA_D);

		runKernelErrorRescaleVolume = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeCubicKernel, 3, NULL, globalWorkSizeRescaleVolumeCubic, localWorkSizeRescaleVolumeCubic, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	clReleaseMemObject(d_Volume_Texture);
}

// Runs parametric registration over several scales, COARSEST_SCALE should be 8, 4, 2 or 1
void BROCCOLI_LIB::AlignTwoVolumesParametricSeveralScales(float *h_Registration_Parameters_Align_Two_Volumes_Several_Scales, float* h_Rotations, cl_mem d_Original_Aligned_Volume, cl_mem d_Original_Reference_Volume, int DATA_W, int DATA_H, int DATA_D, int COARSEST_SCALE, int NUMBER_OF_ITERATIONS, int ALIGNMENT_TYPE, int OVERWRITE, int INTERPOLATION_MODE)
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
	CURRENT_DATA_W = (int)round((float)DATA_W/((float)COARSEST_SCALE));
	CURRENT_DATA_H = (int)round((float)DATA_H/((float)COARSEST_SCALE));
	CURRENT_DATA_D = (int)round((float)DATA_D/((float)COARSEST_SCALE));

	// Setup all parameters and allocate memory on host
	AlignTwoVolumesParametricSetup(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);

	// Change size of original volumes to current scale
	ChangeVolumeSize(d_Aligned_Volume, d_Original_Aligned_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, INTERPOLATION_MODE);
	ChangeVolumeSize(d_Reference_Volume, d_Original_Reference_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, INTERPOLATION_MODE);

	// Copy volume to be aligned to an image (texture)
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D};
	clEnqueueCopyBufferToImage(commandQueue, d_Aligned_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);
	
	// Loop registration over scales
	for (int current_scale = COARSEST_SCALE; current_scale >= 1; current_scale = current_scale/2)
	//for (int current_scale = COARSEST_SCALE; current_scale >= COARSEST_SCALE; current_scale = current_scale/2)
	{
		if (current_scale == 1)
		{
			AlignTwoVolumesParametric(h_Registration_Parameters_Temp, h_Rotations_Temp, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, (int)ceil((float)NUMBER_OF_ITERATIONS/10.0f), ALIGNMENT_TYPE, INTERPOLATION_MODE);
			//AlignTwoVolumesParametric(h_Registration_Parameters_Temp, h_Rotations_Temp, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, NUMBER_OF_ITERATIONS, ALIGNMENT_TYPE, INTERPOLATION_MODE);
		}
		else
		{
			AlignTwoVolumesParametric(h_Registration_Parameters_Temp, h_Rotations_Temp, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, NUMBER_OF_ITERATIONS, ALIGNMENT_TYPE, INTERPOLATION_MODE);
		}

		if (current_scale != 1)
		{
			h_Rotations[0] += h_Rotations_Temp[0];
			h_Rotations[1] += h_Rotations_Temp[1];
			h_Rotations[2] += h_Rotations_Temp[2];

			// Multiply the transformations by a factor 2 for the next scale and add to previous parameters
			//h_Registration_Parameters_Align_Two_Volumes_Several_Scales = h_Registration_Parameters_Align_Two_Volumes_Several_Scales*2 + h_Registration_Parameters_Temp*2
			AddAffineRegistrationParametersNextScale(h_Registration_Parameters_Align_Two_Volumes_Several_Scales,h_Registration_Parameters_Temp);

			// Clean up before the next scale
			AlignTwoVolumesParametricCleanup();

			// Prepare for the next scale
			CURRENT_DATA_W = (int)round((float)DATA_W/((float)current_scale/2.0f));
			CURRENT_DATA_H = (int)round((float)DATA_H/((float)current_scale/2.0f));
			CURRENT_DATA_D = (int)round((float)DATA_D/((float)current_scale/2.0f));

			// Setup all parameters and allocate memory on host
			AlignTwoVolumesParametricSetup(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);

			// Change size of original volumes to current scale
			ChangeVolumeSize(d_Aligned_Volume, d_Original_Aligned_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, INTERPOLATION_MODE);
			ChangeVolumeSize(d_Reference_Volume, d_Original_Reference_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, INTERPOLATION_MODE);

			// Copy volume to be aligned to an image (texture)
			size_t origin[3] = {0, 0, 0};
			size_t region[3] = {CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D};
			clEnqueueCopyBufferToImage(commandQueue, d_Aligned_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);

			// Copy incremented parameter vector to constant memory
			clEnqueueWriteBuffer(commandQueue, c_Registration_Parameters, CL_TRUE, 0, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), h_Registration_Parameters_Align_Two_Volumes_Several_Scales, 0, NULL, NULL);

			// Apply transformation to next scale
			if (INTERPOLATION_MODE == LINEAR)
			{
				runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeLinear, localWorkSizeInterpolateVolumeLinear, 0, NULL, NULL);
				clFinish(commandQueue);
			}
			else if (INTERPOLATION_MODE == CUBIC)
			{
				runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeCubicParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeCubic, localWorkSizeInterpolateVolumeCubic, 0, NULL, NULL);
				clFinish(commandQueue);
			}

			// Copy transformed volume back to image (texture)
			clEnqueueCopyBufferToImage(commandQueue, d_Aligned_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);
		}
		else // Last scale, nothing more to do
		{
			// Clean up
			AlignTwoVolumesParametricCleanup();

			// Calculate final registration parameters
			AddAffineRegistrationParameters(h_Registration_Parameters_Align_Two_Volumes_Several_Scales,h_Registration_Parameters_Temp);


			h_Rotations[0] += h_Rotations_Temp[0];
			h_Rotations[1] += h_Rotations_Temp[1];
			h_Rotations[2] += h_Rotations_Temp[2];

			if (OVERWRITE == DO_OVERWRITE)
			{
				// Transform the original volume once with the final registration parameters, to remove effects of several interpolations
				TransformVolumeParametric(d_Original_Aligned_Volume, h_Registration_Parameters_Align_Two_Volumes_Several_Scales, DATA_W, DATA_H, DATA_D, INTERPOLATION_MODE);
			}
		}
	}
}

// Runs non-parametric registration over several scales, COARSEST_SCALE should be 8, 4, 2 or 1
void BROCCOLI_LIB::AlignTwoVolumesNonParametricSeveralScales(cl_mem d_Original_Aligned_Volume, cl_mem d_Original_Reference_Volume, int DATA_W, int DATA_H, int DATA_D, int COARSEST_SCALE, int NUMBER_OF_ITERATIONS, int OVERWRITE, int INTERPOLATION_MODE, int KEEP)
{
	// Calculate volume size for coarsest scale
	CURRENT_DATA_W = (int)round((float)DATA_W/((float)COARSEST_SCALE));
	CURRENT_DATA_H = (int)round((float)DATA_H/((float)COARSEST_SCALE));
	CURRENT_DATA_D = (int)round((float)DATA_D/((float)COARSEST_SCALE));

	int PREVIOUS_DATA_W = CURRENT_DATA_W;
	int PREVIOUS_DATA_H = CURRENT_DATA_H;
	int PREVIOUS_DATA_D = CURRENT_DATA_D;

	// Setup all parameters and allocate memory on host
	AlignTwoVolumesNonParametricSetup(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);

	// Change size of original volumes to current scale
	ChangeVolumeSize(d_Aligned_Volume, d_Original_Aligned_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, INTERPOLATION_MODE);
	ChangeVolumeSize(d_Reference_Volume, d_Original_Reference_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, INTERPOLATION_MODE);

	// Copy volume to be aligned to an image (texture)
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D};
	clEnqueueCopyBufferToImage(commandQueue, d_Aligned_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);
	
	d_Total_Displacement_Field_X = clCreateBuffer(context, CL_MEM_READ_WRITE,  CURRENT_DATA_W * CURRENT_DATA_H * CURRENT_DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);
	d_Total_Displacement_Field_Y = clCreateBuffer(context, CL_MEM_READ_WRITE,  CURRENT_DATA_W * CURRENT_DATA_H * CURRENT_DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);
	d_Total_Displacement_Field_Z = clCreateBuffer(context, CL_MEM_READ_WRITE,  CURRENT_DATA_W * CURRENT_DATA_H * CURRENT_DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);

	// Set total displacement field to 0
	SetMemory(d_Total_Displacement_Field_X, 0.0f, CURRENT_DATA_W * CURRENT_DATA_H * CURRENT_DATA_D);
	SetMemory(d_Total_Displacement_Field_Y, 0.0f, CURRENT_DATA_W * CURRENT_DATA_H * CURRENT_DATA_D);
	SetMemory(d_Total_Displacement_Field_Z, 0.0f, CURRENT_DATA_W * CURRENT_DATA_H * CURRENT_DATA_D);

	// Loop registration over scales
	for (int current_scale = COARSEST_SCALE; current_scale >= 1; current_scale = current_scale/2)
	{
		if (current_scale == 1)
		{
			AlignTwoVolumesNonParametric(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, (int)ceil((float)NUMBER_OF_ITERATIONS/2.0f), INTERPOLATION_MODE);
		}
		else
		{
			AlignTwoVolumesNonParametric(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, NUMBER_OF_ITERATIONS, INTERPOLATION_MODE);
		}

		if (current_scale != 1)
		{
			// Add found displacement field to total displacement field
			AddVolumes(d_Total_Displacement_Field_X, d_Update_Displacement_Field_X, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);
			AddVolumes(d_Total_Displacement_Field_Y, d_Update_Displacement_Field_Y, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);
			AddVolumes(d_Total_Displacement_Field_Z, d_Update_Displacement_Field_Z, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);

			// Clean up before the next scale
			AlignTwoVolumesNonParametricCleanup();

			// Prepare for the next scale (the previous scale was current scale, so the next scale is times 2)
			CURRENT_DATA_W = (int)round((float)DATA_W/((float)current_scale/2.0f));
			CURRENT_DATA_H = (int)round((float)DATA_H/((float)current_scale/2.0f));
			CURRENT_DATA_D = (int)round((float)DATA_D/((float)current_scale/2.0f));

			float scale_factor = 2.0f;

			// Setup all parameters and allocate memory on host
			AlignTwoVolumesNonParametricSetup(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);

			// Change size of original volumes to current scale
			ChangeVolumeSize(d_Aligned_Volume, d_Original_Aligned_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, INTERPOLATION_MODE);
			ChangeVolumeSize(d_Reference_Volume, d_Original_Reference_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, INTERPOLATION_MODE);

			// Copy volume to be aligned to an image (texture)
			size_t origin[3] = {0, 0, 0};
			size_t region[3] = {CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D};
			clEnqueueCopyBufferToImage(commandQueue, d_Aligned_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);

			// Rescale the displacement field to the current volume size
			ChangeVolumeSize(d_Total_Displacement_Field_X, PREVIOUS_DATA_W, PREVIOUS_DATA_H, PREVIOUS_DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, INTERPOLATION_MODE);
			ChangeVolumeSize(d_Total_Displacement_Field_Y, PREVIOUS_DATA_W, PREVIOUS_DATA_H, PREVIOUS_DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, INTERPOLATION_MODE);
			ChangeVolumeSize(d_Total_Displacement_Field_Z, PREVIOUS_DATA_W, PREVIOUS_DATA_H, PREVIOUS_DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, INTERPOLATION_MODE);

			// Multiply each motion vector with the scale factor, to compensate for the new resolution
			MultiplyVolume(d_Total_Displacement_Field_X, scale_factor, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);
			MultiplyVolume(d_Total_Displacement_Field_Y, scale_factor, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);
			MultiplyVolume(d_Total_Displacement_Field_Z, scale_factor, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);

			PREVIOUS_DATA_W = CURRENT_DATA_W;
			PREVIOUS_DATA_H = CURRENT_DATA_H;
			PREVIOUS_DATA_D = CURRENT_DATA_D;

			// Apply transformation to next scale
			if (INTERPOLATION_MODE == LINEAR)
			{
				clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 2, sizeof(cl_mem), &d_Total_Displacement_Field_X);
				clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 3, sizeof(cl_mem), &d_Total_Displacement_Field_Y);
				clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 4, sizeof(cl_mem), &d_Total_Displacement_Field_Z);
				runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearNonParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeLinear, localWorkSizeInterpolateVolumeLinear, 0, NULL, NULL);
				clFinish(commandQueue);
			}
			else if (INTERPOLATION_MODE == CUBIC)
			{
				runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeCubicNonParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeCubic, localWorkSizeInterpolateVolumeCubic, 0, NULL, NULL);
				clFinish(commandQueue);
			}

			// Copy transformed volume back to image (texture)
			clEnqueueCopyBufferToImage(commandQueue, d_Aligned_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);
		}
		else // Last scale, nothing more to do
		{
			AddVolumes(d_Total_Displacement_Field_X, d_Update_Displacement_Field_X, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);
			AddVolumes(d_Total_Displacement_Field_Y, d_Update_Displacement_Field_Y, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);
			AddVolumes(d_Total_Displacement_Field_Z, d_Update_Displacement_Field_Z, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);

			if (OVERWRITE == DO_OVERWRITE)
			{
				// Transform the original volume once with the final total displacement field, to remove effects of several interpolations
				TransformVolumeNonParametric(d_Original_Aligned_Volume, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, DATA_W, DATA_H, DATA_D, INTERPOLATION_MODE);
			}

			// Clean up
			AlignTwoVolumesNonParametricCleanup();
		}
	}

	//clEnqueueReadBuffer(commandQueue, d_Total_Displacement_Field_X, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Displacement_Field_X, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_Total_Displacement_Field_Y, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Displacement_Field_Y, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_Total_Displacement_Field_Z, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Displacement_Field_Z, 0, NULL, NULL);

	if (KEEP == 0)
	{
		// Clean up
		clReleaseMemObject(d_Total_Displacement_Field_X);
		clReleaseMemObject(d_Total_Displacement_Field_Y);
		clReleaseMemObject(d_Total_Displacement_Field_Z);
	}
}



// This function is used by all registration functions, to cleanup
void BROCCOLI_LIB::AlignTwoVolumesParametricCleanup()
{
	// Free all the allocated memory on the device

	clReleaseMemObject(d_Original_Volume);
	clReleaseMemObject(d_Reference_Volume);
	clReleaseMemObject(d_Aligned_Volume);

	/*
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
	*/

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


	clReleaseMemObject(c_Quadrature_Filter_1_Real);
	clReleaseMemObject(c_Quadrature_Filter_1_Imag);
	clReleaseMemObject(c_Quadrature_Filter_2_Real);
	clReleaseMemObject(c_Quadrature_Filter_2_Imag);
	clReleaseMemObject(c_Quadrature_Filter_3_Real);
	clReleaseMemObject(c_Quadrature_Filter_3_Imag);
	
	clReleaseMemObject(c_Registration_Parameters);
}

int mymax(int a, int b)
{
	if (a > b)
		return a;
	else
		return b;
}

float mymax(float a, float b)
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

	if (INTERPOLATION_MODE == LINEAR)
	{
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
	}
	else if (INTERPOLATION_MODE == CUBIC)
	{
		clSetKernelArg(RescaleVolumeCubicKernel, 0, sizeof(cl_mem), &d_Interpolated_T1_Volume);
		clSetKernelArg(RescaleVolumeCubicKernel, 1, sizeof(cl_mem), &d_T1_Volume_Texture);
		clSetKernelArg(RescaleVolumeCubicKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
		clSetKernelArg(RescaleVolumeCubicKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
		clSetKernelArg(RescaleVolumeCubicKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
		clSetKernelArg(RescaleVolumeCubicKernel, 5, sizeof(int), &T1_DATA_W_INTERPOLATED);
		clSetKernelArg(RescaleVolumeCubicKernel, 6, sizeof(int), &T1_DATA_H_INTERPOLATED);
		clSetKernelArg(RescaleVolumeCubicKernel, 7, sizeof(int), &T1_DATA_D_INTERPOLATED);

		error = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeCubicKernel, 3, NULL, globalWorkSizeRescaleVolumeCubic, localWorkSizeRescaleVolumeCubic, 0, NULL, NULL);
		clFinish(commandQueue);
	}

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


void BROCCOLI_LIB::ChangeT1VolumeResolutionAndSize(cl_mem d_MNI_T1_Volume, cl_mem d_T1_Volume, int T1_DATA_W, int T1_DATA_H, int T1_DATA_D, int MNI_DATA_W, int MNI_DATA_H, int MNI_DATA_D, float T1_VOXEL_SIZE_X, float T1_VOXEL_SIZE_Y, float T1_VOXEL_SIZE_Z, float MNI_VOXEL_SIZE_X, float MNI_VOXEL_SIZE_Y, float MNI_VOXEL_SIZE_Z, int INTERPOLATION_MODE, int MNI_WITH_SKULL)
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

	if (INTERPOLATION_MODE == LINEAR)
	{
		clSetKernelArg(RescaleVolumeLinearKernel, 0, sizeof(cl_mem), &d_Interpolated_T1_Volume);
		clSetKernelArg(RescaleVolumeLinearKernel, 1, sizeof(cl_mem), &d_T1_Volume_Texture);
		clSetKernelArg(RescaleVolumeLinearKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
		clSetKernelArg(RescaleVolumeLinearKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
		clSetKernelArg(RescaleVolumeLinearKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
		clSetKernelArg(RescaleVolumeLinearKernel, 5, sizeof(int), &T1_DATA_W_INTERPOLATED);
		clSetKernelArg(RescaleVolumeLinearKernel, 6, sizeof(int), &T1_DATA_H_INTERPOLATED);
		clSetKernelArg(RescaleVolumeLinearKernel, 7, sizeof(int), &T1_DATA_D_INTERPOLATED);

		// Interpolate T1 volume to the same voxel size as the MNI volume
		runKernelErrorRescaleVolume = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeLinearKernel, 3, NULL, globalWorkSizeRescaleVolumeLinear, localWorkSizeRescaleVolumeLinear, 0, NULL, NULL);
		clFinish(commandQueue);
	}
	else if (INTERPOLATION_MODE == CUBIC)
	{
		clSetKernelArg(RescaleVolumeCubicKernel, 0, sizeof(cl_mem), &d_Interpolated_T1_Volume);
		clSetKernelArg(RescaleVolumeCubicKernel, 1, sizeof(cl_mem), &d_T1_Volume_Texture);
		clSetKernelArg(RescaleVolumeCubicKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
		clSetKernelArg(RescaleVolumeCubicKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
		clSetKernelArg(RescaleVolumeCubicKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
		clSetKernelArg(RescaleVolumeCubicKernel, 5, sizeof(int), &T1_DATA_W_INTERPOLATED);
		clSetKernelArg(RescaleVolumeCubicKernel, 6, sizeof(int), &T1_DATA_H_INTERPOLATED);
		clSetKernelArg(RescaleVolumeCubicKernel, 7, sizeof(int), &T1_DATA_D_INTERPOLATED);

		// Interpolate T1 volume to the same voxel size as the MNI volume
		runKernelErrorRescaleVolume = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeCubicKernel, 3, NULL, globalWorkSizeRescaleVolumeCubic, localWorkSizeRescaleVolumeCubic, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	// Now make sure that the interpolated T1 volume has the same number of voxels as the MNI volume in each direction
	int x_diff = T1_DATA_W_INTERPOLATED - MNI_DATA_W;
	int y_diff = T1_DATA_H_INTERPOLATED - MNI_DATA_H;
	int z_diff = T1_DATA_D_INTERPOLATED - MNI_DATA_D;

	// Set all values to zero
	SetMemory(d_MNI_T1_Volume, 0.0f, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D);

	// Make an initial move to MNI size
	SetGlobalAndLocalWorkSizesCopyVolumeToNew(mymax(MNI_DATA_W,T1_DATA_W_INTERPOLATED),mymax(MNI_DATA_H,T1_DATA_H_INTERPOLATED),mymax(MNI_DATA_D,T1_DATA_D_INTERPOLATED));

	int MM_T1_Z_CUT_ = MM_T1_Z_CUT / (int)MNI_VOXEL_SIZE_X;

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
	clSetKernelArg(CopyT1VolumeToMNIKernel, 11, sizeof(int), &MM_T1_Z_CUT_);
	clSetKernelArg(CopyT1VolumeToMNIKernel, 12, sizeof(float), &MNI_VOXEL_SIZE_Z);

	runKernelErrorCopyVolume = clEnqueueNDRangeKernel(commandQueue, CopyT1VolumeToMNIKernel, 3, NULL, globalWorkSizeCopyVolumeToNew, localWorkSizeCopyVolumeToNew, 0, NULL, NULL);
	clFinish(commandQueue);
	
	// Check where the top of the brain is in the moved T1 volume
	int top_slice;
	CalculateTopBrainSlice(top_slice, d_MNI_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, MM_T1_Z_CUT_);
	
	int diff;
	if (MNI_VOXEL_SIZE_X == 1.0f)
	{
        if (MNI_WITH_SKULL == 1)
		{
			diff = top_slice - 172;
		}
		else
		{
			diff = top_slice - 150;
		}
    }
	else if (MNI_VOXEL_SIZE_X == 2.0f)
	{
		if (MNI_WITH_SKULL == 1)
		{
			diff = top_slice - 85;
		}
		else
		{
			diff = top_slice - 75;
		}
    }

	// Make final move to MNI size, only move half the distance since T1 brains normally are smaller than the MNI brain
	MM_T1_Z_CUT_ = (int)((float)MM_T1_Z_CUT / MNI_VOXEL_SIZE_X) + (int)round((float)diff/1.5f);

	//*h_Top_Slice = top_slice;
	//*h_Top_Slice = MM_T1_Z_CUT;
	//*h_Top_Slice = top_slice;

	// Set all values to zero again
	SetMemory(d_MNI_T1_Volume, 0.0f, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D);

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
	clSetKernelArg(CopyT1VolumeToMNIKernel, 11, sizeof(int), &MM_T1_Z_CUT_); // ? streck eller inte?
	clSetKernelArg(CopyT1VolumeToMNIKernel, 12, sizeof(float), &MNI_VOXEL_SIZE_Z);

	runKernelErrorCopyVolume = clEnqueueNDRangeKernel(commandQueue, CopyT1VolumeToMNIKernel, 3, NULL, globalWorkSizeCopyVolumeToNew, localWorkSizeCopyVolumeToNew, 0, NULL, NULL);
	clFinish(commandQueue);

	clReleaseMemObject(d_Interpolated_T1_Volume);
	clReleaseMemObject(d_T1_Volume_Texture);
}



void BROCCOLI_LIB::CalculateTopBrainSlice(int& slice, cl_mem d_Volume, int DATA_W, int DATA_H, int DATA_D, int z_cut)
{
	SetGlobalAndLocalWorkSizesCalculateMagnitudes(DATA_W, DATA_H, DATA_D);
	SetGlobalAndLocalWorkSizesCalculateSum(DATA_W, DATA_H, DATA_D);

	AlignTwoVolumesParametricSetup(DATA_W, DATA_H, DATA_D);

	// Apply quadrature filters to brain volume
	NonseparableConvolution3D(d_q21, d_q22, d_q23, d_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_1_Parametric_Registration_Real, h_Quadrature_Filter_1_Parametric_Registration_Imag, h_Quadrature_Filter_2_Parametric_Registration_Real, h_Quadrature_Filter_2_Parametric_Registration_Imag, h_Quadrature_Filter_3_Parametric_Registration_Real, h_Quadrature_Filter_3_Parametric_Registration_Imag, DATA_W, DATA_H, DATA_D);

	cl_mem d_Magnitudes = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Column_Sums = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_H * DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Sums = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_D * sizeof(float), NULL, NULL);

	// Calculate filter magnitudes
	clSetKernelArg(CalculateMagnitudesKernel, 0, sizeof(cl_mem), &d_Magnitudes);
	clSetKernelArg(CalculateMagnitudesKernel, 1, sizeof(cl_mem), &d_q23);
	clSetKernelArg(CalculateMagnitudesKernel, 2, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateMagnitudesKernel, 3, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateMagnitudesKernel, 4, sizeof(int), &DATA_D);

	runKernelErrorCalculateMagnitudes = clEnqueueNDRangeKernel(commandQueue, CalculateMagnitudesKernel, 3, NULL, globalWorkSizeCalculateMagnitudes, localWorkSizeCalculateMagnitudes, 0, NULL, NULL);
	clFinish(commandQueue);

	// Calculate sum of filter response magnitudes for each slice
	clSetKernelArg(CalculateColumnSumsKernel, 0, sizeof(cl_mem), &d_Column_Sums);
	clSetKernelArg(CalculateColumnSumsKernel, 1, sizeof(cl_mem), &d_Magnitudes);
	clSetKernelArg(CalculateColumnSumsKernel, 2, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateColumnSumsKernel, 3, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateColumnSumsKernel, 4, sizeof(int), &DATA_D);

	runKernelErrorCalculateColumnSums = clEnqueueNDRangeKernel(commandQueue, CalculateColumnSumsKernel, 2, NULL, globalWorkSizeCalculateColumnSums, localWorkSizeCalculateColumnSums, 0, NULL, NULL);
	clFinish(commandQueue);

	clSetKernelArg(CalculateRowSumsKernel, 0, sizeof(cl_mem), &d_Sums);
	clSetKernelArg(CalculateRowSumsKernel, 1, sizeof(cl_mem), &d_Column_Sums);
	clSetKernelArg(CalculateRowSumsKernel, 2, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateRowSumsKernel, 3, sizeof(int), &DATA_D);

	runKernelErrorCalculateRowSums = clEnqueueNDRangeKernel(commandQueue, CalculateRowSumsKernel, 2, NULL, globalWorkSizeCalculateRowSums, localWorkSizeCalculateRowSums, 0, NULL, NULL);
	clFinish(commandQueue);

	// Copy slice sums to host
	float* h_Sums = (float*)malloc(DATA_D * sizeof(float));
	float* h_Derivatives = (float*)malloc(DATA_D * sizeof(float));
	clEnqueueReadBuffer(commandQueue, d_Sums, CL_TRUE, 0, DATA_D * sizeof(float), h_Sums, 0, NULL, NULL);

	// Fix sums to remove unwanted derivatives
	for (int z = DATA_D - 1; z >= (DATA_D - z_cut - 4 - 1); z--)
	{
		h_Sums[z] = h_Sums[DATA_D - 1 - z_cut - 4 - 1];
	}
	for (int z = (int)round((float)DATA_D/2.0); z >= 0; z--)
	{
		h_Sums[z] = 0.0f;
	}

	for (int z = 0; z < DATA_D; z++)
	{
		//h_Slice_Sums[z] = h_Sums[z];
	}


	// Reset all derivatives
	for (int z = 0; z < DATA_D; z++)
	{
		h_Derivatives[z] = 0.0f;
	}

	// Calculate derivative of sums
	for (int z = 1; z < DATA_D; z++)
	{
		h_Derivatives[z] = h_Sums[z-1] - h_Sums[z];
	}

	// Find max derivative
	float max_slope = -10000.0f;
	for (int z = 0; z < DATA_D; z++)
	{
		if (h_Derivatives[z] > max_slope)
		{
			max_slope = h_Derivatives[z];
			slice = z;
		}
	}

	AlignTwoVolumesParametricCleanup();

	clReleaseMemObject(d_Magnitudes);
	clReleaseMemObject(d_Column_Sums);
	clReleaseMemObject(d_Sums);
	free(h_Sums);
	free(h_Derivatives);
}





void BROCCOLI_LIB::ChangeEPIVolumeResolutionAndSize(cl_mem d_T1_EPI_Volume, cl_mem d_EPI_Volume, int EPI_DATA_W, int EPI_DATA_H, int EPI_DATA_D, int T1_DATA_W, int T1_DATA_H, int T1_DATA_D, float EPI_VOXEL_SIZE_X, float EPI_VOXEL_SIZE_Y, float EPI_VOXEL_SIZE_Z, float T1_VOXEL_SIZE_X, float T1_VOXEL_SIZE_Y, float T1_VOXEL_SIZE_Z, int INTERPOLATION_MODE)
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

	if (INTERPOLATION_MODE == LINEAR)
	{
		clSetKernelArg(RescaleVolumeLinearKernel, 0, sizeof(cl_mem), &d_Interpolated_EPI_Volume);
		clSetKernelArg(RescaleVolumeLinearKernel, 1, sizeof(cl_mem), &d_EPI_Volume_Texture);
		clSetKernelArg(RescaleVolumeLinearKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
		clSetKernelArg(RescaleVolumeLinearKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
		clSetKernelArg(RescaleVolumeLinearKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
		clSetKernelArg(RescaleVolumeLinearKernel, 5, sizeof(int), &EPI_DATA_W_INTERPOLATED);
		clSetKernelArg(RescaleVolumeLinearKernel, 6, sizeof(int), &EPI_DATA_H_INTERPOLATED);
		clSetKernelArg(RescaleVolumeLinearKernel, 7, sizeof(int), &EPI_DATA_D_INTERPOLATED);

		// Interpolate EPI volume to the same voxel size as the new volume
		runKernelErrorRescaleVolume = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeLinearKernel, 3, NULL, globalWorkSizeRescaleVolumeLinear, localWorkSizeRescaleVolumeLinear, 0, NULL, NULL);
		clFinish(commandQueue);
	}
	else if (INTERPOLATION_MODE == CUBIC)
	{
		clSetKernelArg(RescaleVolumeCubicKernel, 0, sizeof(cl_mem), &d_Interpolated_EPI_Volume);
		clSetKernelArg(RescaleVolumeCubicKernel, 1, sizeof(cl_mem), &d_EPI_Volume_Texture);
		clSetKernelArg(RescaleVolumeCubicKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
		clSetKernelArg(RescaleVolumeCubicKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
		clSetKernelArg(RescaleVolumeCubicKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
		clSetKernelArg(RescaleVolumeCubicKernel, 5, sizeof(int), &EPI_DATA_W_INTERPOLATED);
		clSetKernelArg(RescaleVolumeCubicKernel, 6, sizeof(int), &EPI_DATA_H_INTERPOLATED);
		clSetKernelArg(RescaleVolumeCubicKernel, 7, sizeof(int), &EPI_DATA_D_INTERPOLATED);

		// Interpolate EPI volume to the same voxel size as the new volume
		runKernelErrorRescaleVolume = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeCubicKernel, 3, NULL, globalWorkSizeRescaleVolumeCubic, localWorkSizeRescaleVolumeCubic, 0, NULL, NULL);
		clFinish(commandQueue);
	}

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

	runKernelErrorCopyVolume = clEnqueueNDRangeKernel(commandQueue, CopyEPIVolumeToT1Kernel, 3, NULL, globalWorkSizeCopyVolumeToNew, localWorkSizeCopyVolumeToNew, 0, NULL, NULL);
	clFinish(commandQueue);

	clReleaseMemObject(d_Interpolated_EPI_Volume);
	clReleaseMemObject(d_EPI_Volume_Texture);
}

void BROCCOLI_LIB::ChangeVolumesResolutionAndSize(cl_mem d_New_Volumes, cl_mem d_Volumes, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_VOLUMES, int NEW_DATA_W, int NEW_DATA_H, int NEW_DATA_D, float VOXEL_SIZE_X, float VOXEL_SIZE_Y, float VOXEL_SIZE_Z, float NEW_VOXEL_SIZE_X, float NEW_VOXEL_SIZE_Y, float NEW_VOXEL_SIZE_Z, int MM_Z_CUT, int INTERPOLATION_MODE)
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

	if (INTERPOLATION_MODE == LINEAR)
	{
		clSetKernelArg(RescaleVolumeLinearKernel, 0, sizeof(cl_mem), &d_Interpolated_Volume);
		clSetKernelArg(RescaleVolumeLinearKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
		clSetKernelArg(RescaleVolumeLinearKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
		clSetKernelArg(RescaleVolumeLinearKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
		clSetKernelArg(RescaleVolumeLinearKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
		clSetKernelArg(RescaleVolumeLinearKernel, 5, sizeof(int), &DATA_W_INTERPOLATED);
		clSetKernelArg(RescaleVolumeLinearKernel, 6, sizeof(int), &DATA_H_INTERPOLATED);
		clSetKernelArg(RescaleVolumeLinearKernel, 7, sizeof(int), &DATA_D_INTERPOLATED);
	}
	else if (INTERPOLATION_MODE == CUBIC)
	{
		clSetKernelArg(RescaleVolumeCubicKernel, 0, sizeof(cl_mem), &d_Interpolated_Volume);
		clSetKernelArg(RescaleVolumeCubicKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
		clSetKernelArg(RescaleVolumeCubicKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
		clSetKernelArg(RescaleVolumeCubicKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
		clSetKernelArg(RescaleVolumeCubicKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
		clSetKernelArg(RescaleVolumeCubicKernel, 5, sizeof(int), &DATA_W_INTERPOLATED);
		clSetKernelArg(RescaleVolumeCubicKernel, 6, sizeof(int), &DATA_H_INTERPOLATED);
		clSetKernelArg(RescaleVolumeCubicKernel, 7, sizeof(int), &DATA_D_INTERPOLATED);
	}

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
	//clSetKernelArg(CopyVolumeToNewKernel, 11, sizeof(int), &MM_EPI_Z_CUT);
	clSetKernelArg(CopyVolumeToNewKernel, 12, sizeof(float), &NEW_VOXEL_SIZE_Z);
	
	// Set all values to zero
	SetMemory(d_New_Volumes, 0.0f, DATA_W * DATA_H * DATA_D * NUMBER_OF_VOLUMES);
	
	for (int volume = 0; volume < NUMBER_OF_VOLUMES; volume++)
	{
		// Copy the current volume to an image to interpolate from
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {DATA_W, DATA_H, DATA_D};
		clEnqueueCopyBufferToImage(commandQueue, d_Volumes, d_Volume_Texture, volume * DATA_W * DATA_H * DATA_D * sizeof(float), origin, region, 0, NULL, NULL);

		// Rescale current volume to the same voxel size as the new volume
		if (INTERPOLATION_MODE == LINEAR)
		{
			runKernelErrorRescaleVolume = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeLinearKernel, 3, NULL, globalWorkSizeRescaleVolumeLinear, localWorkSizeRescaleVolumeLinear, 0, NULL, NULL);
			clFinish(commandQueue);
		}
		else if (INTERPOLATION_MODE == CUBIC)
		{
			runKernelErrorRescaleVolume = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeCubicKernel, 3, NULL, globalWorkSizeRescaleVolumeCubic, localWorkSizeRescaleVolumeCubic, 0, NULL, NULL);
			clFinish(commandQueue);
		}

		clSetKernelArg(CopyVolumeToNewKernel, 13, sizeof(int), &volume);
	
		runKernelErrorCopyVolume = clEnqueueNDRangeKernel(commandQueue, CopyVolumeToNewKernel, 3, NULL, globalWorkSizeCopyVolumeToNew, localWorkSizeCopyVolumeToNew, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	clReleaseMemObject(d_Interpolated_Volume);
	clReleaseMemObject(d_Volume_Texture);
}




void BROCCOLI_LIB::InvertAffineRegistrationParameters(float* h_Inverse_Parameters, float* h_Parameters)
{
	// Put parameters in 4 x 4 affine transformation matrix

	// (p3 p4  p5  tx)
	// (p6 p7  p8  ty)
	// (p9 p10 p11 tz)
	// (0  0   0   1 )

	Eigen::MatrixXd Affine_Matrix(4,4);

	// Add ones in the diagonal, to get a transformation matrix
	// First row
	Affine_Matrix(0,0) = (double)(h_Parameters[3] + 1.0f);
	Affine_Matrix(0,1) = (double)h_Parameters[4];
	Affine_Matrix(0,2) = (double)h_Parameters[5];
	Affine_Matrix(0,3) = (double)h_Parameters[0];

	// Second row
	Affine_Matrix(1,0) = (double)h_Parameters[6];
	Affine_Matrix(1,1) = (double)(h_Parameters[7] + 1.0f);
	Affine_Matrix(1,2) = (double)h_Parameters[8];
	Affine_Matrix(1,3) = (double)h_Parameters[1];

	// Third row
	Affine_Matrix(2,0)  = (double)h_Parameters[9];
	Affine_Matrix(2,1)  = (double)h_Parameters[10];
	Affine_Matrix(2,2) = (double)(h_Parameters[11] + 1.0f);
	Affine_Matrix(2,3) = (double)h_Parameters[2];

	// Fourth row
	Affine_Matrix(3,0) = 0.0;
	Affine_Matrix(3,1) = 0.0;
	Affine_Matrix(3,2) = 0.0;
	Affine_Matrix(3,3) = 1.0;

	Affine_Matrix.inverse();

	// Subtract ones in the diagonal

	// Translation parameters
	h_Inverse_Parameters[0] = (float)Affine_Matrix(0,3);
	h_Inverse_Parameters[1] = (float)Affine_Matrix(1,3);
	h_Inverse_Parameters[2] = (float)Affine_Matrix(2,3);

	// First row
	h_Inverse_Parameters[3] = (float)(Affine_Matrix(0,0) - 1.0);
	h_Inverse_Parameters[4] = (float)Affine_Matrix(0,1);
	h_Inverse_Parameters[5] = (float)Affine_Matrix(0,2);

	// Second row
	h_Inverse_Parameters[6] = (float)Affine_Matrix(1,0);
	h_Inverse_Parameters[7] = (float)(Affine_Matrix(1,1) - 1.0);
	h_Inverse_Parameters[8] = (float)Affine_Matrix(1,2);

	// Third row
	h_Inverse_Parameters[9] = (float)Affine_Matrix(2,0);
	h_Inverse_Parameters[10] = (float)Affine_Matrix(2,1);
	h_Inverse_Parameters[11] = (float)(Affine_Matrix(2,2) - 1.0);
}



void BROCCOLI_LIB::AddAffineRegistrationParameters(float* h_Old_Parameters, float* h_New_Parameters)
{
	// Put parameters in 4 x 4 affine transformation matrix

	// (p3 p4  p5  tx)
	// (p6 p7  p8  ty)
	// (p9 p10 p11 tz)
	// (0  0   0   1 )

	Eigen::MatrixXd Old_Affine_Matrix(4,4), New_Affine_Matrix(4,4), Total_Affine_Matrix(4,4);

	// Add ones in the diagonal, to get a transformation matrix
	// First row
	Old_Affine_Matrix(0,0) = (double)(h_Old_Parameters[3] + 1.0f);
	Old_Affine_Matrix(0,1) = (double)h_Old_Parameters[4];
	Old_Affine_Matrix(0,2) = (double)h_Old_Parameters[5];
	Old_Affine_Matrix(0,3) = (double)h_Old_Parameters[0];

	// Second row
	Old_Affine_Matrix(1,0) = (double)h_Old_Parameters[6];
	Old_Affine_Matrix(1,1) = (double)(h_Old_Parameters[7] + 1.0f);
	Old_Affine_Matrix(1,2) = (double)h_Old_Parameters[8];
	Old_Affine_Matrix(1,3) = (double)h_Old_Parameters[1];

	// Third row
	Old_Affine_Matrix(2,0)  = (double)h_Old_Parameters[9];
	Old_Affine_Matrix(2,1)  = (double)h_Old_Parameters[10];
	Old_Affine_Matrix(2,2) = (double)(h_Old_Parameters[11] + 1.0f);
	Old_Affine_Matrix(2,3) = (double)h_Old_Parameters[2];

	// Fourth row
	Old_Affine_Matrix(3,0) = 0.0;
	Old_Affine_Matrix(3,1) = 0.0;
	Old_Affine_Matrix(3,2) = 0.0;
	Old_Affine_Matrix(3,3) = 1.0;

	// Add ones in the diagonal, to get a transformation matrix
	// First row
	New_Affine_Matrix(0,0) = (double)(h_New_Parameters[3] + 1.0f);
	New_Affine_Matrix(0,1) = (double)h_New_Parameters[4];
	New_Affine_Matrix(0,2) = (double)h_New_Parameters[5];
	New_Affine_Matrix(0,3) = (double)h_New_Parameters[0];

	// Second row
	New_Affine_Matrix(1,0) = (double)h_New_Parameters[6];
	New_Affine_Matrix(1,1) = (double)(h_New_Parameters[7] + 1.0f);
	New_Affine_Matrix(1,2) = (double)h_New_Parameters[8];
	New_Affine_Matrix(1,3) = (double)h_New_Parameters[1];

	// Third row
	New_Affine_Matrix(2,0)  = (double)h_New_Parameters[9];
	New_Affine_Matrix(2,1)  = (double)h_New_Parameters[10];
	New_Affine_Matrix(2,2) = (double)(h_New_Parameters[11] + 1.0f);
	New_Affine_Matrix(2,3) = (double)h_New_Parameters[2];

	// Fourth row
	New_Affine_Matrix(3,0) = 0.0;
	New_Affine_Matrix(3,1) = 0.0;
	New_Affine_Matrix(3,2) = 0.0;
	New_Affine_Matrix(3,3) = 1.0;

	// Multiply the two matrices
	Total_Affine_Matrix = New_Affine_Matrix * Old_Affine_Matrix;

	// Subtract ones in the diagonal

	// Translation parameters
	h_Old_Parameters[0] = (float)Total_Affine_Matrix(0,3);
	h_Old_Parameters[1] = (float)Total_Affine_Matrix(1,3);
	h_Old_Parameters[2] = (float)Total_Affine_Matrix(2,3);

	// First row
	h_Old_Parameters[3] = (float)(Total_Affine_Matrix(0,0) - 1.0);
	h_Old_Parameters[4] = (float)Total_Affine_Matrix(0,1);
	h_Old_Parameters[5] = (float)Total_Affine_Matrix(0,2);

	// Second row
	h_Old_Parameters[6] = (float)Total_Affine_Matrix(1,0);
	h_Old_Parameters[7] = (float)(Total_Affine_Matrix(1,1) - 1.0);
	h_Old_Parameters[8] = (float)Total_Affine_Matrix(1,2);

	// Third row
	h_Old_Parameters[9] = (float)Total_Affine_Matrix(2,0);
	h_Old_Parameters[10] = (float)Total_Affine_Matrix(2,1);
	h_Old_Parameters[11] = (float)(Total_Affine_Matrix(2,2) - 1.0);
}

void BROCCOLI_LIB::AddAffineRegistrationParametersNextScale(float* h_Old_Parameters, float* h_New_Parameters)
{
	// Put parameters in 4 x 4 affine transformation matrix

	// (p3 p4  p5  tx)
	// (p6 p7  p8  ty)
	// (p9 p10 p11 tz)
	// (0  0   0   1 )

	Eigen::MatrixXd Old_Affine_Matrix(4,4), New_Affine_Matrix(4,4), Total_Affine_Matrix(4,4);

	// Add ones in the diagonal, to get a transformation matrix
	// First row
	Old_Affine_Matrix(0,0) = (double)(h_Old_Parameters[3] + 1.0f);
	Old_Affine_Matrix(0,1) = (double)h_Old_Parameters[4];
	Old_Affine_Matrix(0,2) = (double)h_Old_Parameters[5];
	Old_Affine_Matrix(0,3) = (double)(h_Old_Parameters[0] * 2.0f);

	// Second row
	Old_Affine_Matrix(1,0) = (double)h_Old_Parameters[6];
	Old_Affine_Matrix(1,1) = (double)(h_Old_Parameters[7] + 1.0f);
	Old_Affine_Matrix(1,2) = (double)h_Old_Parameters[8];
	Old_Affine_Matrix(1,3) = (double)(h_Old_Parameters[1] * 2.0f);

	// Third row
	Old_Affine_Matrix(2,0)  = (double)h_Old_Parameters[9];
	Old_Affine_Matrix(2,1)  = (double)h_Old_Parameters[10];
	Old_Affine_Matrix(2,2) = (double)(h_Old_Parameters[11] + 1.0f);
	Old_Affine_Matrix(2,3) = (double)(h_Old_Parameters[2] * 2.0f);

	// Fourth row
	Old_Affine_Matrix(3,0) = 0.0;
	Old_Affine_Matrix(3,1) = 0.0;
	Old_Affine_Matrix(3,2) = 0.0;
	Old_Affine_Matrix(3,3) = 1.0;

	// Add ones in the diagonal, to get a transformation matrix
	// First row
	New_Affine_Matrix(0,0) = (double)(h_New_Parameters[3] + 1.0f);
	New_Affine_Matrix(0,1) = (double)h_New_Parameters[4];
	New_Affine_Matrix(0,2) = (double)h_New_Parameters[5];
	New_Affine_Matrix(0,3) = (double)(h_New_Parameters[0] * 2.0f);

	// Second row
	New_Affine_Matrix(1,0) = (double)h_New_Parameters[6];
	New_Affine_Matrix(1,1) = (double)(h_New_Parameters[7] + 1.0f);
	New_Affine_Matrix(1,2) = (double)h_New_Parameters[8];
	New_Affine_Matrix(1,3) = (double)(h_New_Parameters[1] * 2.0f);

	// Third row
	New_Affine_Matrix(2,0)  = (double)h_New_Parameters[9];
	New_Affine_Matrix(2,1)  = (double)h_New_Parameters[10];
	New_Affine_Matrix(2,2) = (double)(h_New_Parameters[11] + 1.0f);
	New_Affine_Matrix(2,3) = (double)(h_New_Parameters[2] * 2.0f);

	// Fourth row
	New_Affine_Matrix(3,0) = 0.0;
	New_Affine_Matrix(3,1) = 0.0;
	New_Affine_Matrix(3,2) = 0.0;
	New_Affine_Matrix(3,3) = 1.0;

	// Multiply the two matrices
	Total_Affine_Matrix = New_Affine_Matrix * Old_Affine_Matrix;

	// Subtract ones in the diagonal

	// Translation parameters
	h_Old_Parameters[0] = (float)Total_Affine_Matrix(0,3);
	h_Old_Parameters[1] = (float)Total_Affine_Matrix(1,3);
	h_Old_Parameters[2] = (float)Total_Affine_Matrix(2,3);

	// First row
	h_Old_Parameters[3] = (float)(Total_Affine_Matrix(0,0) - 1.0);
	h_Old_Parameters[4] = (float)Total_Affine_Matrix(0,1);
	h_Old_Parameters[5] = (float)Total_Affine_Matrix(0,2);

	// Second row
	h_Old_Parameters[6] = (float)Total_Affine_Matrix(1,0);
	h_Old_Parameters[7] = (float)(Total_Affine_Matrix(1,1) - 1.0);
	h_Old_Parameters[8] = (float)Total_Affine_Matrix(1,2);

	// Third row
	h_Old_Parameters[9] = (float)Total_Affine_Matrix(2,0);
	h_Old_Parameters[10] = (float)Total_Affine_Matrix(2,1);
	h_Old_Parameters[11] = (float)(Total_Affine_Matrix(2,2) - 1.0);
}



void BROCCOLI_LIB::AddAffineRegistrationParameters(float* h_Resulting_Parameters, float* h_Old_Parameters, float* h_New_Parameters)
{
	// Put parameters in 4 x 4 affine transformation matrix

	// (p3 p4  p5  tx)
	// (p6 p7  p8  ty)
	// (p9 p10 p11 tz)
	// (0  0   0   1 )

	Eigen::MatrixXd Old_Affine_Matrix(4,4), New_Affine_Matrix(4,4), Total_Affine_Matrix(4,4);

	// Add ones in the diagonal, to get a transformation matrix
	// First row
	Old_Affine_Matrix(0,0) = (double)(h_Old_Parameters[3] + 1.0f);
	Old_Affine_Matrix(0,1) = (double)h_Old_Parameters[4];
	Old_Affine_Matrix(0,2) = (double)h_Old_Parameters[5];
	Old_Affine_Matrix(0,3) = (double)h_Old_Parameters[0];

	// Second row
	Old_Affine_Matrix(1,0) = (double)h_Old_Parameters[6];
	Old_Affine_Matrix(1,1) = (double)(h_Old_Parameters[7] + 1.0f);
	Old_Affine_Matrix(1,2) = (double)h_Old_Parameters[8];
	Old_Affine_Matrix(1,3) = (double)h_Old_Parameters[1];

	// Third row
	Old_Affine_Matrix(2,0)  = (double)h_Old_Parameters[9];
	Old_Affine_Matrix(2,1)  = (double)h_Old_Parameters[10];
	Old_Affine_Matrix(2,2) = (double)(h_Old_Parameters[11] + 1.0f);
	Old_Affine_Matrix(2,3) = (double)h_Old_Parameters[2];

	// Fourth row
	Old_Affine_Matrix(3,0) = 0.0;
	Old_Affine_Matrix(3,1) = 0.0;
	Old_Affine_Matrix(3,2) = 0.0;
	Old_Affine_Matrix(3,3) = 1.0;

	// Add ones in the diagonal, to get a transformation matrix
	// First row
	New_Affine_Matrix(0,0) = (double)(h_New_Parameters[3] + 1.0f);
	New_Affine_Matrix(0,1) = (double)h_New_Parameters[4];
	New_Affine_Matrix(0,2) = (double)h_New_Parameters[5];
	New_Affine_Matrix(0,3) = (double)h_New_Parameters[0];

	// Second row
	New_Affine_Matrix(1,0) = (double)h_New_Parameters[6];
	New_Affine_Matrix(1,1) = (double)(h_New_Parameters[7] + 1.0f);
	New_Affine_Matrix(1,2) = (double)h_New_Parameters[8];
	New_Affine_Matrix(1,3) = (double)h_New_Parameters[1];

	// Third row
	New_Affine_Matrix(2,0)  = (double)h_New_Parameters[9];
	New_Affine_Matrix(2,1)  = (double)h_New_Parameters[10];
	New_Affine_Matrix(2,2) = (double)(h_New_Parameters[11] + 1.0f);
	New_Affine_Matrix(2,3) = (double)h_New_Parameters[2];

	// Fourth row
	New_Affine_Matrix(3,0) = 0.0;
	New_Affine_Matrix(3,1) = 0.0;
	New_Affine_Matrix(3,2) = 0.0;
	New_Affine_Matrix(3,3) = 1.0;

	// Multiply the two matrices
	Total_Affine_Matrix = New_Affine_Matrix * Old_Affine_Matrix;

	// Subtract ones in the diagonal
	// First row
	h_Resulting_Parameters[0] = (float)Total_Affine_Matrix(0,3);
	h_Resulting_Parameters[1] = (float)Total_Affine_Matrix(1,3);
	h_Resulting_Parameters[2] = (float)Total_Affine_Matrix(2,3);

	// Second row
	h_Resulting_Parameters[3] = (float)(Total_Affine_Matrix(0,0) - 1.0);
	h_Resulting_Parameters[4] = (float)Total_Affine_Matrix(0,1);
	h_Resulting_Parameters[5] = (float)Total_Affine_Matrix(0,2);

	// Third row
	h_Resulting_Parameters[6] = (float)Total_Affine_Matrix(1,0);
	h_Resulting_Parameters[7] = (float)(Total_Affine_Matrix(1,1) - 1.0);
	h_Resulting_Parameters[8] = (float)Total_Affine_Matrix(1,2);

	// Fourth row
	h_Resulting_Parameters[9] = (float)Total_Affine_Matrix(2,0);
	h_Resulting_Parameters[10] = (float)Total_Affine_Matrix(2,1);
	h_Resulting_Parameters[11] = (float)(Total_Affine_Matrix(2,2) - 1.0);
}


void BROCCOLI_LIB::MultiplyVolume(cl_mem d_Volume, float factor, int DATA_W, int DATA_H, int DATA_D)
{
	SetGlobalAndLocalWorkSizesMultiplyVolumes(DATA_W, DATA_H, DATA_D);

	clSetKernelArg(MultiplyVolumeKernel, 0, sizeof(cl_mem), &d_Volume);
	clSetKernelArg(MultiplyVolumeKernel, 1, sizeof(float), &factor);
	clSetKernelArg(MultiplyVolumeKernel, 2, sizeof(int), &DATA_W);
	clSetKernelArg(MultiplyVolumeKernel, 3, sizeof(int), &DATA_H);
	clSetKernelArg(MultiplyVolumeKernel, 4, sizeof(int), &DATA_D);

	runKernelErrorMultiplyVolumes = clEnqueueNDRangeKernel(commandQueue, MultiplyVolumeKernel, 3, NULL, globalWorkSizeMultiplyVolumes, localWorkSizeMultiplyVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::MultiplyVolumes(cl_mem d_Result, cl_mem d_Volume_1, cl_mem d_Volume_2, int DATA_W, int DATA_H, int DATA_D)
{
	SetGlobalAndLocalWorkSizesMultiplyVolumes(DATA_W, DATA_H, DATA_D);

	clSetKernelArg(MultiplyVolumesKernel, 0, sizeof(cl_mem), &d_Result);
	clSetKernelArg(MultiplyVolumesKernel, 1, sizeof(cl_mem), &d_Volume_1);
	clSetKernelArg(MultiplyVolumesKernel, 2, sizeof(cl_mem), &d_Volume_2);
	clSetKernelArg(MultiplyVolumesKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(MultiplyVolumesKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(MultiplyVolumesKernel, 5, sizeof(int), &DATA_D);

	runKernelErrorMultiplyVolumes = clEnqueueNDRangeKernel(commandQueue, MultiplyVolumesKernel, 3, NULL, globalWorkSizeMultiplyVolumes, localWorkSizeMultiplyVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::MultiplyVolumes(cl_mem d_Volume_1, cl_mem d_Volume_2, int DATA_W, int DATA_H, int DATA_D)
{
	SetGlobalAndLocalWorkSizesMultiplyVolumes(DATA_W, DATA_H, DATA_D);

	int zero = 0;
	clSetKernelArg(MultiplyVolumesOverwriteKernel, 0, sizeof(cl_mem), &d_Volume_1);
	clSetKernelArg(MultiplyVolumesOverwriteKernel, 1, sizeof(cl_mem), &d_Volume_2);
	clSetKernelArg(MultiplyVolumesOverwriteKernel, 2, sizeof(int), &DATA_W);
	clSetKernelArg(MultiplyVolumesOverwriteKernel, 3, sizeof(int), &DATA_H);
	clSetKernelArg(MultiplyVolumesOverwriteKernel, 4, sizeof(int), &DATA_D);
	clSetKernelArg(MultiplyVolumesOverwriteKernel, 5, sizeof(int), &zero);

	runKernelErrorMultiplyVolumes = clEnqueueNDRangeKernel(commandQueue, MultiplyVolumesOverwriteKernel, 3, NULL, globalWorkSizeMultiplyVolumes, localWorkSizeMultiplyVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::MultiplyVolumes(cl_mem d_Volume_1, cl_mem d_Volume_2, int DATA_W, int DATA_H, int DATA_D, int VOLUMES)
{
	SetGlobalAndLocalWorkSizesMultiplyVolumes(DATA_W, DATA_H, DATA_D);

	for (int v = 0; v < VOLUMES; v++)
	{
		clSetKernelArg(MultiplyVolumesOverwriteKernel, 0, sizeof(cl_mem), &d_Volume_1);
		clSetKernelArg(MultiplyVolumesOverwriteKernel, 1, sizeof(cl_mem), &d_Volume_2);
		clSetKernelArg(MultiplyVolumesOverwriteKernel, 2, sizeof(int), &DATA_W);
		clSetKernelArg(MultiplyVolumesOverwriteKernel, 3, sizeof(int), &DATA_H);
		clSetKernelArg(MultiplyVolumesOverwriteKernel, 4, sizeof(int), &DATA_D);
		clSetKernelArg(MultiplyVolumesOverwriteKernel, 5, sizeof(int), &v);

		runKernelErrorMultiplyVolumes = clEnqueueNDRangeKernel(commandQueue, MultiplyVolumesOverwriteKernel, 3, NULL, globalWorkSizeMultiplyVolumes, localWorkSizeMultiplyVolumes, 0, NULL, NULL);
		clFinish(commandQueue);
	}
}

void BROCCOLI_LIB::AddVolume(cl_mem d_Volume, float value, int DATA_W, int DATA_H, int DATA_D)
{
	SetGlobalAndLocalWorkSizesAddVolumes(DATA_W, DATA_H, DATA_D);

	clSetKernelArg(AddVolumeKernel, 0, sizeof(cl_mem), &d_Volume);
	clSetKernelArg(AddVolumeKernel, 1, sizeof(float), &value);
	clSetKernelArg(AddVolumeKernel, 2, sizeof(int), &DATA_W);
	clSetKernelArg(AddVolumeKernel, 3, sizeof(int), &DATA_H);
	clSetKernelArg(AddVolumeKernel, 4, sizeof(int), &DATA_D);

	runKernelErrorAddVolumes = clEnqueueNDRangeKernel(commandQueue, AddVolumeKernel, 3, NULL, globalWorkSizeAddVolumes, localWorkSizeAddVolumes, 0, NULL, NULL);
	clFinish(commandQueue);	
}

void BROCCOLI_LIB::AddVolumes(cl_mem d_Result, cl_mem d_Volume_1, cl_mem d_Volume_2, int DATA_W, int DATA_H, int DATA_D)
{
	SetGlobalAndLocalWorkSizesAddVolumes(DATA_W, DATA_H, DATA_D);

	clSetKernelArg(AddVolumesKernel, 0, sizeof(cl_mem), &d_Result);
	clSetKernelArg(AddVolumesKernel, 1, sizeof(cl_mem), &d_Volume_1);
	clSetKernelArg(AddVolumesKernel, 2, sizeof(cl_mem), &d_Volume_2);
	clSetKernelArg(AddVolumesKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(AddVolumesKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(AddVolumesKernel, 5, sizeof(int), &DATA_D);

	runKernelErrorAddVolumes = clEnqueueNDRangeKernel(commandQueue, AddVolumesKernel, 3, NULL, globalWorkSizeAddVolumes, localWorkSizeAddVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::AddVolumes(cl_mem d_Volume_1, cl_mem d_Volume_2, int DATA_W, int DATA_H, int DATA_D)
{
	SetGlobalAndLocalWorkSizesAddVolumes(DATA_W, DATA_H, DATA_D);

	clSetKernelArg(AddVolumesOverwriteKernel, 0, sizeof(cl_mem), &d_Volume_1);
	clSetKernelArg(AddVolumesOverwriteKernel, 1, sizeof(cl_mem), &d_Volume_2);
	clSetKernelArg(AddVolumesOverwriteKernel, 2, sizeof(int), &DATA_W);
	clSetKernelArg(AddVolumesOverwriteKernel, 3, sizeof(int), &DATA_H);
	clSetKernelArg(AddVolumesOverwriteKernel, 4, sizeof(int), &DATA_D);

	runKernelErrorAddVolumes = clEnqueueNDRangeKernel(commandQueue, AddVolumesOverwriteKernel, 3, NULL, globalWorkSizeAddVolumes, localWorkSizeAddVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

/*
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
	ChangeEPIVolumeResolutionAndSize(d_T1_EPI_Volume, d_EPI_Volume, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, INTERPOLATION_MODE);

	// Copy the EPI T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_Interpolated_EPI_Volume, 0, NULL, NULL);

	// Do the registration between EPI and T1 with several scales, rigid
	AlignTwoVolumesParametricSeveralScales(h_Registration_Parameters_EPI_T1_Affine, h_Rotations, d_T1_EPI_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);

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
*/

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
	ChangeEPIVolumeResolutionAndSize(d_T1_EPI_Volume, d_EPI_Volume, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, INTERPOLATION_MODE);
	
	// Copy the EPI T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_Interpolated_EPI_Volume, 0, NULL, NULL);

	//d_Tensor_Magnitude_T1 = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	//d_Tensor_Magnitude_T1_EPI = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);

	// Apply filters for non-parametric registration and estimate tensor magnitude for T1 volume
	//CalculateTensorMagnitude(d_Tensor_Magnitude_T1, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D);

	// Apply filters for non-parametric registration and estimate tensor magnitude for T1 EPI volume
	//CalculateTensorMagnitude(d_Tensor_Magnitude_T1_EPI, d_T1_EPI_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D);

	//clEnqueueReadBuffer(commandQueue, d_Tensor_Magnitude_T1_EPI, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_Interpolated_EPI_Volume, 0, NULL, NULL);

	// Do the registration between EPI and T1 with several scales, rigid
	//AlignTwoVolumesParametricSeveralScales(h_Registration_Parameters_EPI_T1_Affine, h_Rotations, d_Tensor_Magnitude_T1_EPI, d_Tensor_Magnitude_T1, T1_DATA_W, T1_DATA_H, T1_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION, TRANSLATION, DO_OVERWRITE, INTERPOLATION_MODE);

	AlignTwoVolumesParametricSeveralScales(h_Registration_Parameters_EPI_T1_Affine, h_Rotations, d_T1_EPI_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION, TRANSLATION, DO_OVERWRITE, INTERPOLATION_MODE);
	//AlignTwoVolumesParametricSeveralScales(h_Registration_Parameters_EPI_T1_Affine, h_Rotations, d_T1_EPI_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);

	//AlignTwoVolumesParametricSeveralScales(h_Registration_Parameters_EPI_T1_Affine, h_Rotations, d_T1_EPI_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);

	//h_Registration_Parameters_EPI_T1_Affine[2] -= 3.5f;

	// Apply transformation to interpolated EPI volume
	//TransformVolumeParametric(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, INTERPOLATION_MODE);

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

	//clReleaseMemObject(d_Tensor_Magnitude_T1);
	//clReleaseMemObject(d_Tensor_Magnitude_T1_EPI);
}



void BROCCOLI_LIB::PerformRegistrationEPIT1()
{
	// Interpolate EPI volume to T1 resolution and make sure it has the same size,
	// the registration is performed to the skullstripped T1 volume, which has MNI size
	ChangeEPIVolumeResolutionAndSize(d_T1_EPI_Volume, d_EPI_Volume, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, INTERPOLATION_MODE);

	// Do the registration between EPI and skullstripped T1 with several scales, rigid
	AlignTwoVolumesParametricSeveralScales(h_Registration_Parameters_EPI_T1_Affine, h_Rotations, d_T1_EPI_Volume, d_Skullstripped_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION, TRANSLATION, NO_OVERWRITE, INTERPOLATION_MODE);

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
	ChangeT1VolumeResolutionAndSize(d_MNI_T1_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, INTERPOLATION_MODE, NOT_SKULL_STRIPPED);

	clReleaseMemObject(d_T1_Volume);

	// Copy the MNI T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Interpolated_T1_Volume, 0, NULL, NULL);

	// Do the registration between T1 and MNI with several scales (with skull)
	AlignTwoVolumesParametricSeveralScales(h_Registration_Parameters_T1_MNI_Out, h_Rotations, d_MNI_T1_Volume, d_MNI_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION, AFFINE, NO_OVERWRITE, INTERPOLATION_MODE);

	clReleaseMemObject(d_MNI_Volume);

	// Copy the aligned T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume, 0, NULL, NULL);

	// Allocate memory for MNI brain mask
	d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy MNI brain mask to device
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

	// Calculate inverse transform between T1 and MNI
	InvertAffineRegistrationParameters(h_Inverse_Registration_Parameters, h_Registration_Parameters_T1_MNI_Out);

	// Now apply the inverse transformation between MNI and T1, to transform MNI brain mask to original T1 space
	TransformVolumeParametric(d_MNI_Brain_Mask, h_Inverse_Registration_Parameters, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NEAREST);

	// Create skullstripped volume, by multiplying original T1 volume with transformed MNI brain mask
	MultiplyVolumes(d_MNI_T1_Volume, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);


	d_MNI_Brain_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Volume , 0, NULL, NULL);


	// Now align skullstripped volume with MNI brain
	AlignTwoVolumesParametricSeveralScales(h_Registration_Parameters_T1_MNI_Out, h_Rotations, d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION, AFFINE, NO_OVERWRITE, INTERPOLATION_MODE);

	// Copy MNI brain mask to device again
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

	// Calculate inverse transform between T1 and MNI (to get better skullstrip)
	InvertAffineRegistrationParameters(h_Inverse_Registration_Parameters, h_Registration_Parameters_T1_MNI_Out);
	
	// Apply inverse transform
	TransformVolumeParametric(d_MNI_Brain_Mask, h_Inverse_Registration_Parameters, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NEAREST);

	// Multiply inverse transformed mask with original volume (to get better skullstrip)
	MultiplyVolumes(d_MNI_T1_Volume, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	// Copy the skullstripped T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Skullstripped_T1_Volume, 0, NULL, NULL);

	// Apply forward transform to skullstripped volume
	TransformVolumeParametric(d_MNI_T1_Volume, h_Registration_Parameters_T1_MNI_Out, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, LINEAR);

	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume, 0, NULL, NULL);

	// Perform non-parametric registration between tramsformed skullstripped volume and MNI brain volume
	AlignTwoVolumesNonParametricSeveralScales(d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION, DO_OVERWRITE, INTERPOLATION_MODE, DISCARD_DISPLACEMENT_FIELD);

	// Copy the aligned T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_NonParametric, 0, NULL, NULL);

	// Cleanup

	clReleaseMemObject(d_MNI_Brain_Volume);
	clReleaseMemObject(d_MNI_T1_Volume);
	clReleaseMemObject(d_MNI_Brain_Mask);
}

void BROCCOLI_LIB::PerformRegistrationT1MNINoSkullstripWrapper()
{
	// Allocate memory for T1 volume, MNI volume and T1 volume of MNI size
	d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Brain_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to T1 volume and MNI volume
	clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Volume , 0, NULL, NULL);

	// Interpolate T1 volume to MNI resolution and make sure it has the same size
	ChangeT1VolumeResolutionAndSize(d_MNI_T1_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, INTERPOLATION_MODE, SKULL_STRIPPED);

	clReleaseMemObject(d_T1_Volume);

	// Copy the MNI T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Interpolated_T1_Volume, 0, NULL, NULL);

	// Do the registration between T1 and MNI with several scales (without skull)
	AlignTwoVolumesParametricSeveralScales(h_Registration_Parameters_T1_MNI_Out, h_Rotations, d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);

	// Copy the aligned T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume, 0, NULL, NULL);

	// Perform non-parametric registration between tramsformed skullstripped volume and MNI brain volume
	AlignTwoVolumesNonParametricSeveralScales(d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION, DO_OVERWRITE, INTERPOLATION_MODE, DISCARD_DISPLACEMENT_FIELD);

	// Copy the aligned T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_NonParametric, 0, NULL, NULL);

	// Cleanup

	clReleaseMemObject(d_MNI_Brain_Volume);
	clReleaseMemObject(d_MNI_T1_Volume);
}

// Performs registration between one high resolution T1 volume and a high resolution MNI volume (brain template)
void BROCCOLI_LIB::PerformRegistrationT1MNI()
{
	// Interpolate T1 volume to MNI resolution and make sure it has the same size
	ChangeT1VolumeResolutionAndSize(d_MNI_T1_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, INTERPOLATION_MODE, NOT_SKULL_STRIPPED);

	// Do the registration between T1 and MNI with several scales (we do not need the aligned T1 volume so do not overwrite)
	AlignTwoVolumesParametricSeveralScales(h_Registration_Parameters_T1_MNI, h_Rotations, d_MNI_T1_Volume, d_MNI_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION, AFFINE, NO_OVERWRITE, INTERPOLATION_MODE);

	// Calculate inverse transform between T1 and MNI
	InvertAffineRegistrationParameters(h_Inverse_Registration_Parameters, h_Registration_Parameters_T1_MNI);

	// Now apply the inverse transformation between MNI and T1, to transform MNI brain mask to T1 space
	TransformVolumeParametric(d_MNI_Brain_Mask, h_Inverse_Registration_Parameters, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NEAREST);

	// Create skullstripped volume, by multiplying original T1 volume with transformed MNI brain mask
	MultiplyVolumes(d_Skullstripped_T1_Volume, d_MNI_T1_Volume, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
}

void BROCCOLI_LIB::PerformRegistrationT1MNINoSkullstrip()
{
	// Interpolate T1 volume to MNI resolution and make sure it has the same size
	ChangeT1VolumeResolutionAndSize(d_MNI_T1_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, INTERPOLATION_MODE, SKULL_STRIPPED);
	ChangeT1VolumeResolutionAndSize(d_Skullstripped_T1_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, INTERPOLATION_MODE, SKULL_STRIPPED);

	// Do parametric registration between T1 and MNI with several scales (without skull)
	AlignTwoVolumesParametricSeveralScales(h_Registration_Parameters_T1_MNI, h_Rotations, d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);

	// Perform non-parametric registration between registered skullstripped volume and MNI brain volume
	AlignTwoVolumesNonParametricSeveralScales(d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION, NO_OVERWRITE, INTERPOLATION_MODE, KEEP_DISPLACEMENT_FIELD);
}


void BROCCOLI_LIB::TransformVolumeParametric(cl_mem d_Volume, float* h_Registration_Parameters_, int DATA_W, int DATA_H, int DATA_D, int INTERPOLATION_MODE)
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
		clSetKernelArg(InterpolateVolumeLinearParametricKernel, 0, sizeof(cl_mem), &d_Volume);
		clSetKernelArg(InterpolateVolumeLinearParametricKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
		clSetKernelArg(InterpolateVolumeLinearParametricKernel, 2, sizeof(cl_mem), &c_Parameters);
		clSetKernelArg(InterpolateVolumeLinearParametricKernel, 3, sizeof(int), &DATA_W);
		clSetKernelArg(InterpolateVolumeLinearParametricKernel, 4, sizeof(int), &DATA_H);
		clSetKernelArg(InterpolateVolumeLinearParametricKernel, 5, sizeof(int), &DATA_D);
		clSetKernelArg(InterpolateVolumeLinearParametricKernel, 6, sizeof(int), &volume);
		runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeLinear, localWorkSizeInterpolateVolumeLinear, 0, NULL, NULL);
		clFinish(commandQueue);
	}
	else if (INTERPOLATION_MODE == CUBIC)
	{
		clSetKernelArg(InterpolateVolumeCubicParametricKernel, 0, sizeof(cl_mem), &d_Volume);
		clSetKernelArg(InterpolateVolumeCubicParametricKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
		clSetKernelArg(InterpolateVolumeCubicParametricKernel, 2, sizeof(cl_mem), &c_Parameters);
		clSetKernelArg(InterpolateVolumeCubicParametricKernel, 3, sizeof(int), &DATA_W);
		clSetKernelArg(InterpolateVolumeCubicParametricKernel, 4, sizeof(int), &DATA_H);
		clSetKernelArg(InterpolateVolumeCubicParametricKernel, 5, sizeof(int), &DATA_D);
		clSetKernelArg(InterpolateVolumeCubicParametricKernel, 6, sizeof(int), &volume);
		runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeCubicParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeCubic, localWorkSizeInterpolateVolumeCubic, 0, NULL, NULL);
		clFinish(commandQueue);
	}
	else if (INTERPOLATION_MODE == NEAREST)
	{
		clSetKernelArg(InterpolateVolumeNearestParametricKernel, 0, sizeof(cl_mem), &d_Volume);
		clSetKernelArg(InterpolateVolumeNearestParametricKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
		clSetKernelArg(InterpolateVolumeNearestParametricKernel, 2, sizeof(cl_mem), &c_Parameters);
		clSetKernelArg(InterpolateVolumeNearestParametricKernel, 3, sizeof(int), &DATA_W);
		clSetKernelArg(InterpolateVolumeNearestParametricKernel, 4, sizeof(int), &DATA_H);
		clSetKernelArg(InterpolateVolumeNearestParametricKernel, 5, sizeof(int), &DATA_D);
		clSetKernelArg(InterpolateVolumeNearestParametricKernel, 6, sizeof(int), &volume);
		runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeNearestParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeNearest, localWorkSizeInterpolateVolumeNearest, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	clReleaseMemObject(d_Volume_Texture);
	clReleaseMemObject(c_Parameters);
}

void BROCCOLI_LIB::TransformVolumesParametric(cl_mem d_Volumes, float* h_Registration_Parameters_, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_VOLUMES, int INTERPOLATION_MODE)
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
			clSetKernelArg(InterpolateVolumeLinearParametricKernel, 0, sizeof(cl_mem), &d_Volumes);
			clSetKernelArg(InterpolateVolumeLinearParametricKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
			clSetKernelArg(InterpolateVolumeLinearParametricKernel, 2, sizeof(cl_mem), &c_Parameters);
			clSetKernelArg(InterpolateVolumeLinearParametricKernel, 3, sizeof(int), &DATA_W);
			clSetKernelArg(InterpolateVolumeLinearParametricKernel, 4, sizeof(int), &DATA_H);
			clSetKernelArg(InterpolateVolumeLinearParametricKernel, 5, sizeof(int), &DATA_D);
			clSetKernelArg(InterpolateVolumeLinearParametricKernel, 6, sizeof(int), &volume);
			runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeLinear, localWorkSizeInterpolateVolumeLinear, 0, NULL, NULL);
			clFinish(commandQueue);
		}
		else if (INTERPOLATION_MODE == CUBIC)
		{
			clSetKernelArg(InterpolateVolumeCubicParametricKernel, 0, sizeof(cl_mem), &d_Volumes);
			clSetKernelArg(InterpolateVolumeCubicParametricKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
			clSetKernelArg(InterpolateVolumeCubicParametricKernel, 2, sizeof(cl_mem), &c_Parameters);
			clSetKernelArg(InterpolateVolumeCubicParametricKernel, 3, sizeof(int), &DATA_W);
			clSetKernelArg(InterpolateVolumeCubicParametricKernel, 4, sizeof(int), &DATA_H);
			clSetKernelArg(InterpolateVolumeCubicParametricKernel, 5, sizeof(int), &DATA_D);
			clSetKernelArg(InterpolateVolumeCubicParametricKernel, 6, sizeof(int), &volume);
			runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeCubicParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeCubic, localWorkSizeInterpolateVolumeCubic, 0, NULL, NULL);
			clFinish(commandQueue);
		}
		else if (INTERPOLATION_MODE == NEAREST)
		{
			clSetKernelArg(InterpolateVolumeNearestParametricKernel, 0, sizeof(cl_mem), &d_Volumes);
			clSetKernelArg(InterpolateVolumeNearestParametricKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
			clSetKernelArg(InterpolateVolumeNearestParametricKernel, 2, sizeof(cl_mem), &c_Parameters);
			clSetKernelArg(InterpolateVolumeNearestParametricKernel, 3, sizeof(int), &DATA_W);
			clSetKernelArg(InterpolateVolumeNearestParametricKernel, 4, sizeof(int), &DATA_H);
			clSetKernelArg(InterpolateVolumeNearestParametricKernel, 5, sizeof(int), &DATA_D);
			clSetKernelArg(InterpolateVolumeNearestParametricKernel, 6, sizeof(int), &volume);
			runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeNearestParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeNearest, localWorkSizeInterpolateVolumeNearest, 0, NULL, NULL);
			clFinish(commandQueue);
		}
	}

	clReleaseMemObject(d_Volume_Texture);
	clReleaseMemObject(c_Parameters);
}

void BROCCOLI_LIB::TransformVolumeNonParametric(cl_mem d_Volume, cl_mem d_Displacement_Field_X, cl_mem d_Displacement_Field_Y, cl_mem d_Displacement_Field_Z, int DATA_W, int DATA_H, int DATA_D, int INTERPOLATION_MODE)
{
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
		clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 0, sizeof(cl_mem), &d_Volume);
		clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
		clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 2, sizeof(cl_mem), &d_Displacement_Field_X);
		clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 3, sizeof(cl_mem), &d_Displacement_Field_Y);
		clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 4, sizeof(cl_mem), &d_Displacement_Field_Z);
		clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 5, sizeof(int), &DATA_W);
		clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 6, sizeof(int), &DATA_H);
		clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 7, sizeof(int), &DATA_D);
		clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 8, sizeof(int), &volume);
		runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearNonParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeLinear, localWorkSizeInterpolateVolumeLinear, 0, NULL, NULL);
		clFinish(commandQueue);
	}
	else if (INTERPOLATION_MODE == CUBIC)
	{
		clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 0, sizeof(cl_mem), &d_Volume);
		clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
		clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 2, sizeof(cl_mem), &d_Displacement_Field_X);
		clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 3, sizeof(cl_mem), &d_Displacement_Field_Y);
		clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 4, sizeof(cl_mem), &d_Displacement_Field_Z);
		clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 5, sizeof(int), &DATA_W);
		clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 6, sizeof(int), &DATA_H);
		clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 7, sizeof(int), &DATA_D);
		clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 8, sizeof(int), &volume);
		runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeCubicNonParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeCubic, localWorkSizeInterpolateVolumeCubic, 0, NULL, NULL);
		clFinish(commandQueue);
	}
	else if (INTERPOLATION_MODE == NEAREST)
	{
		clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 0, sizeof(cl_mem), &d_Volume);
		clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
		clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 2, sizeof(cl_mem), &d_Displacement_Field_X);
		clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 3, sizeof(cl_mem), &d_Displacement_Field_Y);
		clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 4, sizeof(cl_mem), &d_Displacement_Field_Z);
		clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 5, sizeof(int), &DATA_W);
		clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 6, sizeof(int), &DATA_H);
		clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 7, sizeof(int), &DATA_D);
		clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 8, sizeof(int), &volume);
		runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeNearestNonParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeNearest, localWorkSizeInterpolateVolumeNearest, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	clReleaseMemObject(d_Volume_Texture);
}




void BROCCOLI_LIB::TransformVolumesNonParametric(cl_mem d_Volumes, cl_mem d_Displacement_Field_X, cl_mem d_Displacement_Field_Y, cl_mem d_Displacement_Field_Z, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_VOLUMES, int INTERPOLATION_MODE)
{
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
			clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 0, sizeof(cl_mem), &d_Volumes);
			clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
			clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 2, sizeof(cl_mem), &d_Displacement_Field_X);
			clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 3, sizeof(cl_mem), &d_Displacement_Field_Y);
			clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 4, sizeof(cl_mem), &d_Displacement_Field_Z);
			clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 5, sizeof(int), &DATA_W);
			clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 6, sizeof(int), &DATA_H);
			clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 7, sizeof(int), &DATA_D);
			clSetKernelArg(InterpolateVolumeLinearNonParametricKernel, 8, sizeof(int), &volume);
			runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearNonParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeLinear, localWorkSizeInterpolateVolumeLinear, 0, NULL, NULL);
			clFinish(commandQueue);
		}
		else if (INTERPOLATION_MODE == CUBIC)
		{
			clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 0, sizeof(cl_mem), &d_Volumes);
			clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
			clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 2, sizeof(cl_mem), &d_Displacement_Field_X);
			clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 3, sizeof(cl_mem), &d_Displacement_Field_Y);
			clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 4, sizeof(cl_mem), &d_Displacement_Field_Z);
			clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 5, sizeof(int), &DATA_W);
			clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 6, sizeof(int), &DATA_H);
			clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 7, sizeof(int), &DATA_D);
			clSetKernelArg(InterpolateVolumeCubicNonParametricKernel, 8, sizeof(int), &volume);
			runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeCubicNonParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeCubic, localWorkSizeInterpolateVolumeCubic, 0, NULL, NULL);
			clFinish(commandQueue);
		}
		else if (INTERPOLATION_MODE == NEAREST)
		{
			clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 0, sizeof(cl_mem), &d_Volumes);
			clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
			clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 2, sizeof(cl_mem), &d_Displacement_Field_X);
			clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 3, sizeof(cl_mem), &d_Displacement_Field_Y);
			clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 4, sizeof(cl_mem), &d_Displacement_Field_Z);
			clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 5, sizeof(int), &DATA_W);
			clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 6, sizeof(int), &DATA_H);
			clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 7, sizeof(int), &DATA_D);
			clSetKernelArg(InterpolateVolumeNearestNonParametricKernel, 8, sizeof(int), &volume);
			runKernelErrorInterpolateVolume = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeNearestNonParametricKernel, 3, NULL, globalWorkSizeInterpolateVolumeNearest, localWorkSizeInterpolateVolumeNearest, 0, NULL, NULL);
			clFinish(commandQueue);
		}
	}

	clReleaseMemObject(d_Volume_Texture);
}

void BROCCOLI_LIB::PerformFirstLevelAnalysisWrapper()
{
	//------------------------

	// Allocate memory on device
	d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Brain_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Skullstripped_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

	PerformRegistrationT1MNINoSkullstrip();
	//PerformRegistrationT1MNI();

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
	clReleaseMemObject(d_MNI_Brain_Volume);
	//clReleaseMemObject(d_MNI_Brain_Mask);



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

	PerformMotionCorrection();

	for (int t = 0; t < EPI_DATA_T; t++)
	{
		for (int p = 0; p < 6; p++)
		{
			h_Motion_Parameters_Out[t + p * EPI_DATA_T] = h_Motion_Parameters[t + p * EPI_DATA_T];
		}
	}

	clEnqueueReadBuffer(commandQueue, d_Motion_Corrected_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Motion_Corrected_fMRI_Volumes, 0, NULL, NULL);



	//------------------------

	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Smoothed_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	SegmentEPIData();

	//-------------------------------

	d_Smoothed_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, EPI_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	//PerformSmoothing(d_Smoothed_fMRI_Volumes, d_Motion_Corrected_fMRI_Volumes, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	PerformSmoothing(d_Smoothed_EPI_Mask, d_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
	PerformSmoothingNormalized(d_Smoothed_fMRI_Volumes, d_Motion_Corrected_fMRI_Volumes, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	clEnqueueReadBuffer(commandQueue, d_Smoothed_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Smoothed_fMRI_Volumes, 0, NULL, NULL);


	//-------------------------------

	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS;

	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Beta_Contrasts = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Residuals = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device

	//clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM_In, 0, NULL, NULL);
	//clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM_In, 0, NULL, NULL);
	//clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts_In, 0, NULL, NULL);
	//clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In, 0, NULL, NULL);

	d_AR1_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR2_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR3_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR4_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Whitened_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	//SetMemory(d_EPI_Mask, 1.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);


	h_X_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
	h_xtxxt_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
	h_Contrasts = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float));
	h_ctxtxc_GLM = (float*)malloc(NUMBER_OF_CONTRASTS * sizeof(float));

	SetupStatisticalAnalysisRegressors(EPI_DATA_T);

	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM , 0, NULL, NULL);

	for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
	{
		for (int t = 0; t < EPI_DATA_T; t++)
		{
			h_X_GLM_Out[t + r * EPI_DATA_T] = h_X_GLM[t + r * EPI_DATA_T];
			h_xtxxt_GLM_Out[t + r * EPI_DATA_T] = h_xtxxt_GLM[t + r * EPI_DATA_T];
		}
	}

	free(h_X_GLM);
	free(h_xtxxt_GLM);
	free(h_Contrasts);
	free(h_ctxtxc_GLM);

	CalculateStatisticalMapsGLMFirstLevel(d_Smoothed_fMRI_Volumes);

	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR1_Estimates, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR2_Estimates, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR3_Estimates, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR4_Estimates, 0, NULL, NULL);

	clReleaseMemObject(d_AR1_Estimates);
	clReleaseMemObject(d_AR2_Estimates);
	clReleaseMemObject(d_AR3_Estimates);
	clReleaseMemObject(d_AR4_Estimates);
	clReleaseMemObject(d_Whitened_fMRI_Volumes);

	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_MNI,h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_T1_MNI);


	h_Registration_Parameters_EPI_MNI_Out[0] = h_Registration_Parameters_EPI_MNI[0];
	h_Registration_Parameters_EPI_MNI_Out[1] = h_Registration_Parameters_EPI_MNI[1];
	h_Registration_Parameters_EPI_MNI_Out[2] = h_Registration_Parameters_EPI_MNI[2];
	h_Registration_Parameters_EPI_MNI_Out[3] = h_Registration_Parameters_EPI_MNI[3];
	h_Registration_Parameters_EPI_MNI_Out[4] = h_Registration_Parameters_EPI_MNI[4];
	h_Registration_Parameters_EPI_MNI_Out[5] = h_Registration_Parameters_EPI_MNI[5];
	h_Registration_Parameters_EPI_MNI_Out[6] = h_Registration_Parameters_EPI_MNI[6];
	h_Registration_Parameters_EPI_MNI_Out[7] = h_Registration_Parameters_EPI_MNI[7];
	h_Registration_Parameters_EPI_MNI_Out[8] = h_Registration_Parameters_EPI_MNI[8];
	h_Registration_Parameters_EPI_MNI_Out[9] = h_Registration_Parameters_EPI_MNI[9];
	h_Registration_Parameters_EPI_MNI_Out[10] = h_Registration_Parameters_EPI_MNI[10];
	h_Registration_Parameters_EPI_MNI_Out[11] = h_Registration_Parameters_EPI_MNI[11];

	// Copy data to host
	if (BETA_SPACE == MNI)
	{
		// Allocate memory on device
		d_Beta_Volumes_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, &createBufferErrorBetaVolumesMNI);
		d_Statistical_Maps_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, &createBufferErrorStatisticalMapsMNI);
		d_Residual_Variances_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, &createBufferErrorResidualVariancesMNI);

		clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

		TransformFirstLevelResultsToMNI();

		clEnqueueReadBuffer(commandQueue, d_Beta_Volumes_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Statistical_Maps_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Residual_Variances_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);

		clReleaseMemObject(d_Beta_Volumes_MNI);
		clReleaseMemObject(d_Statistical_Maps_MNI);
		clReleaseMemObject(d_Residual_Variances_MNI);
	}
	else if (BETA_SPACE == EPI)
	{
		clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);
	}

	clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Residuals, 0, NULL, NULL);

	clReleaseMemObject(d_MNI_Brain_Mask);

	clReleaseMemObject(d_Total_Displacement_Field_X);
	clReleaseMemObject(d_Total_Displacement_Field_Y);
	clReleaseMemObject(d_Total_Displacement_Field_Z);


	//free(h_Motion_Parameters);
	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Motion_Corrected_fMRI_Volumes);
	clReleaseMemObject(d_Smoothed_fMRI_Volumes);

	clReleaseMemObject(d_EPI_Mask);
	clReleaseMemObject(d_Smoothed_EPI_Mask);

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

void BROCCOLI_LIB::TransformFirstLevelResultsToMNI()
{
	ChangeVolumesResolutionAndSize(d_Beta_Volumes_MNI, d_Beta_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_TOTAL_GLM_REGRESSORS, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	TransformVolumesParametric(d_Beta_Volumes_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_TOTAL_GLM_REGRESSORS, INTERPOLATION_MODE);
	TransformVolumesNonParametric(d_Beta_Volumes_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_TOTAL_GLM_REGRESSORS, INTERPOLATION_MODE);

	ChangeVolumesResolutionAndSize(d_Statistical_Maps_MNI, d_Statistical_Maps, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	TransformVolumesParametric(d_Statistical_Maps_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);
	TransformVolumesNonParametric(d_Statistical_Maps_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);

	ChangeVolumesResolutionAndSize(d_Residual_Variances_MNI, d_Residual_Variances, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	TransformVolumesParametric(d_Residual_Variances_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	TransformVolumesNonParametric(d_Residual_Variances_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	
	//MultiplyVolumes(d_Beta_Volumes_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_GLM_REGRESSORS);
	//MultiplyVolumes(d_Statistical_Maps_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS);
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
	AlignTwoVolumesParametricSetup(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

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
		AlignTwoVolumesParametric(h_Registration_Parameters_Motion_Correction, h_Rotations, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION, RIGID, INTERPOLATION_MODE);
		
		// Copy the corrected volume to the corrected volumes
		clEnqueueCopyBuffer(commandQueue, d_Aligned_Volume, d_Motion_Corrected_fMRI_Volumes, 0, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

		// Write the total parameter vector to host

		// Translations
		h_Motion_Parameters_Out[t + 0 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[0]; // * EPI_VOXEL_SIZE_X;
		h_Motion_Parameters_Out[t + 1 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[1]; // * EPI_VOXEL_SIZE_Y;
		h_Motion_Parameters_Out[t + 2 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[2]; // * EPI_VOXEL_SIZE_Z;

		// Rotations
		h_Motion_Parameters_Out[t + 3 * EPI_DATA_T] = h_Rotations[0];
		h_Motion_Parameters_Out[t + 4 * EPI_DATA_T] = h_Rotations[1];
		h_Motion_Parameters_Out[t + 5 * EPI_DATA_T] = h_Rotations[2];
	}

	// Copy all corrected volumes to host
	clEnqueueReadBuffer(commandQueue, d_Motion_Corrected_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Motion_Corrected_fMRI_Volumes, 0, NULL, NULL);

	// Cleanup allocated memory
	AlignTwoVolumesParametricCleanup();

	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Motion_Corrected_fMRI_Volumes);
}

void BROCCOLI_LIB::PerformMotionCorrectionWrapperSeveralScales()
{
	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Motion_Corrected_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Copy volumes to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);
	clFinish(commandQueue);

	cl_mem d_Current_fMRI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);
	cl_mem d_Current_Reference_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);

	// Set the first volume as the reference volume
	clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Current_Reference_Volume, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

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
		clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Current_fMRI_Volume, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

		// Align to reference volume using 2 scales
		AlignTwoVolumesParametricSeveralScales(h_Registration_Parameters_Motion_Correction, h_Rotations, d_Current_fMRI_Volume, d_Current_Reference_Volume, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);

		// Copy the corrected volume to the corrected volumes
		clEnqueueCopyBuffer(commandQueue, d_Current_fMRI_Volume, d_Motion_Corrected_fMRI_Volumes, 0, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

		// Write the total parameter vector to host

		// Translations
		h_Motion_Parameters_Out[t + 0 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[0]; // * EPI_VOXEL_SIZE_X;
		h_Motion_Parameters_Out[t + 1 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[1]; // * EPI_VOXEL_SIZE_Y;
		h_Motion_Parameters_Out[t + 2 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[2]; // * EPI_VOXEL_SIZE_Z;

		// Rotations
		h_Motion_Parameters_Out[t + 3 * EPI_DATA_T] = h_Rotations[0];
		h_Motion_Parameters_Out[t + 4 * EPI_DATA_T] = h_Rotations[1];
		h_Motion_Parameters_Out[t + 5 * EPI_DATA_T] = h_Rotations[2];
	}

	// Copy all corrected volumes to host
	clEnqueueReadBuffer(commandQueue, d_Motion_Corrected_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Motion_Corrected_fMRI_Volumes, 0, NULL, NULL);

	// Cleanup allocated memory
	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Motion_Corrected_fMRI_Volumes);
	clReleaseMemObject(d_Current_fMRI_Volume);
	clReleaseMemObject(d_Current_Reference_Volume);
}


// Performs motion correction of an fMRI dataset
void BROCCOLI_LIB::PerformMotionCorrection()
{
	// Setup all parameters and allocate memory on host
	AlignTwoVolumesParametricSetup(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

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

		// Also copy the same volume to an image (texture) to interpolate from
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {EPI_DATA_W, EPI_DATA_H, EPI_DATA_D};
		clEnqueueCopyBufferToImage(commandQueue, d_fMRI_Volumes, d_Original_Volume, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), origin, region, 0, NULL, NULL);

		// Do rigid registration with only one scale
		AlignTwoVolumesParametric(h_Registration_Parameters_Motion_Correction, h_Rotations, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION, RIGID, INTERPOLATION_MODE);

		// Copy the corrected volume to the corrected volumes
		clEnqueueCopyBuffer(commandQueue, d_Aligned_Volume, d_Motion_Corrected_fMRI_Volumes, 0, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

		// Write the total parameter vector to host

		// Translations (in mm)
		h_Motion_Parameters[t + 0 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[0] * EPI_VOXEL_SIZE_X;
		h_Motion_Parameters[t + 1 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[1] * EPI_VOXEL_SIZE_Y;
		h_Motion_Parameters[t + 2 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[2] * EPI_VOXEL_SIZE_Z;

		// Rotations
		h_Motion_Parameters[t + 3 * EPI_DATA_T] = h_Rotations[0];
		h_Motion_Parameters[t + 4 * EPI_DATA_T] = h_Rotations[1];
		h_Motion_Parameters[t + 5 * EPI_DATA_T] = h_Rotations[2];
	}

	// Cleanup allocated memory
	AlignTwoVolumesParametricCleanup();
}

float BROCCOLI_LIB::CalculateSum(cl_mem d_Volume, int DATA_W, int DATA_H, int DATA_D)
{
	SetGlobalAndLocalWorkSizesCalculateSum(DATA_W, DATA_H, DATA_D);

	cl_mem d_Column_Sums = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_H * DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Sums = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_D * sizeof(float), NULL, NULL);

	// Calculate sum of filter smoothed EPI for each slice
	clSetKernelArg(CalculateColumnSumsKernel, 0, sizeof(cl_mem), &d_Column_Sums);
	clSetKernelArg(CalculateColumnSumsKernel, 1, sizeof(cl_mem), &d_Volume);
	clSetKernelArg(CalculateColumnSumsKernel, 2, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateColumnSumsKernel, 3, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateColumnSumsKernel, 4, sizeof(int), &DATA_D);

	runKernelErrorCalculateColumnSums = clEnqueueNDRangeKernel(commandQueue, CalculateColumnSumsKernel, 2, NULL, globalWorkSizeCalculateColumnSums, localWorkSizeCalculateColumnSums, 0, NULL, NULL);
	clFinish(commandQueue);

	clSetKernelArg(CalculateRowSumsKernel, 0, sizeof(cl_mem), &d_Sums);
	clSetKernelArg(CalculateRowSumsKernel, 1, sizeof(cl_mem), &d_Column_Sums);
	clSetKernelArg(CalculateRowSumsKernel, 2, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateRowSumsKernel, 3, sizeof(int), &DATA_D);
	
	runKernelErrorCalculateRowSums = clEnqueueNDRangeKernel(commandQueue, CalculateRowSumsKernel, 2, NULL, globalWorkSizeCalculateRowSums, localWorkSizeCalculateRowSums, 0, NULL, NULL);
	clFinish(commandQueue);

	// Copy slice maxs to host
	float* h_Sums = (float*)malloc(DATA_D * sizeof(float));
	clEnqueueReadBuffer(commandQueue, d_Sums, CL_TRUE, 0, DATA_D * sizeof(float), h_Sums, 0, NULL, NULL);

	float sum = 0.0f;
	for (int z = 0; z < DATA_D; z++)
	{
		sum += h_Sums[z];
	}
	free(h_Sums);

	clReleaseMemObject(d_Column_Sums);
	clReleaseMemObject(d_Sums);

	return sum;
}

float BROCCOLI_LIB::CalculateMax(cl_mem d_Volume, int DATA_W, int DATA_H, int DATA_D)
{
	SetGlobalAndLocalWorkSizesCalculateMax(DATA_W, DATA_H, DATA_D);

	cl_mem d_Column_Maxs = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_H * DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Maxs = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_D * sizeof(float), NULL, NULL);

	// Calculate sum of filter smoothed EPI for each slice
	clSetKernelArg(CalculateColumnMaxsKernel, 0, sizeof(cl_mem), &d_Column_Maxs);
	clSetKernelArg(CalculateColumnMaxsKernel, 1, sizeof(cl_mem), &d_Volume);
	clSetKernelArg(CalculateColumnMaxsKernel, 2, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateColumnMaxsKernel, 3, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateColumnMaxsKernel, 4, sizeof(int), &DATA_D);

	runKernelErrorCalculateColumnMaxs = clEnqueueNDRangeKernel(commandQueue, CalculateColumnMaxsKernel, 2, NULL, globalWorkSizeCalculateColumnMaxs, localWorkSizeCalculateColumnMaxs, 0, NULL, NULL);
	clFinish(commandQueue);

	clSetKernelArg(CalculateRowMaxsKernel, 0, sizeof(cl_mem), &d_Maxs);
	clSetKernelArg(CalculateRowMaxsKernel, 1, sizeof(cl_mem), &d_Column_Maxs);
	clSetKernelArg(CalculateRowMaxsKernel, 2, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateRowMaxsKernel, 3, sizeof(int), &DATA_D);
	
	runKernelErrorCalculateRowMaxs = clEnqueueNDRangeKernel(commandQueue, CalculateRowMaxsKernel, 2, NULL, globalWorkSizeCalculateRowMaxs, localWorkSizeCalculateRowMaxs, 0, NULL, NULL);
	clFinish(commandQueue);

	// Copy slice maxs to host
	float* h_Maxs = (float*)malloc(DATA_D * sizeof(float));
	clEnqueueReadBuffer(commandQueue, d_Maxs, CL_TRUE, 0, DATA_D * sizeof(float), h_Maxs, 0, NULL, NULL);

	float max = -1000.0f;
	for (int z = 0; z < DATA_D; z++)
	{
		max = mymax(max, h_Maxs[z]);
	}
	free(h_Maxs);

	clReleaseMemObject(d_Column_Maxs);
	clReleaseMemObject(d_Maxs);

	return max;
}

void BROCCOLI_LIB::ThresholdVolume(cl_mem d_Thresholded_Volume, cl_mem d_Volume, float threshold, int DATA_W, int DATA_H, int DATA_D)
{
	SetGlobalAndLocalWorkSizesThresholdVolume(DATA_W, DATA_H, DATA_D);

	clSetKernelArg(ThresholdVolumeKernel, 0, sizeof(cl_mem), &d_Thresholded_Volume);
	clSetKernelArg(ThresholdVolumeKernel, 1, sizeof(cl_mem), &d_Volume);
	clSetKernelArg(ThresholdVolumeKernel, 2, sizeof(float), &threshold);
	clSetKernelArg(ThresholdVolumeKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(ThresholdVolumeKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(ThresholdVolumeKernel, 5, sizeof(int), &DATA_D);

	runKernelErrorThresholdVolume = clEnqueueNDRangeKernel(commandQueue, ThresholdVolumeKernel, 3, NULL, globalWorkSizeThresholdVolume, localWorkSizeThresholdVolume, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::SegmentEPIData()
{
	cl_mem d_EPI = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Smoothed_EPI = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_EPI, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, 4.0, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	PerformSmoothing(d_Smoothed_EPI, d_EPI, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

	float sum = CalculateSum(d_Smoothed_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	float threshold = 0.9f * sum / ((float) EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);

	ThresholdVolume(d_EPI_Mask, d_Smoothed_EPI, threshold, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	clReleaseMemObject(d_EPI);
	clReleaseMemObject(d_Smoothed_EPI);
}

void BROCCOLI_LIB::CreateSmoothingFilters(float* Smoothing_Filter_X, float* Smoothing_Filter_Y, float* Smoothing_Filter_Z, int size, float smoothing_FWHM, float voxel_size_x, float voxel_size_y, float voxel_size_z)
{
	int halfSize = (size - 1) / 2;
	double sigma_x = (double)smoothing_FWHM / 2.354 / (double)voxel_size_x;
	double sigma_y = (double)smoothing_FWHM / 2.354 / (double)voxel_size_y;
	double sigma_z = (double)smoothing_FWHM / 2.354 / (double)voxel_size_z;

	double sigma_x2 = 2.0 * sigma_x * sigma_x;
	double sigma_y2 = 2.0 * sigma_y * sigma_y;
	double sigma_z2 = 2.0 * sigma_z * sigma_z;

	double u;
	float sumX, sumY, sumZ;
	sumX = 0.0f;
	sumY = 0.0f;
	sumZ = 0.0f;

	for (int i = 0; i < size; i++)
	{
		u = (double)(i - halfSize);
		Smoothing_Filter_X[i] = (float)exp(-pow(u,2.0) / sigma_x2);
		Smoothing_Filter_Y[i] = (float)exp(-pow(u,2.0) / sigma_y2);
		Smoothing_Filter_Z[i] = (float)exp(-pow(u,2.0) / sigma_z2);
		sumX += Smoothing_Filter_X[i];
		sumY += Smoothing_Filter_Y[i];
		sumZ += Smoothing_Filter_Z[i];
	}
	for (int i = 0; i < size; i++)
	{
		Smoothing_Filter_X[i] /= sumX;
		Smoothing_Filter_Y[i] /= sumY;
		Smoothing_Filter_Z[i] /= sumZ;
	}
}

void BROCCOLI_LIB::CreateSmoothingFilters(float* Smoothing_Filter_X, float* Smoothing_Filter_Y, float* Smoothing_Filter_Z, int size, double sigma)
{
	int halfSize = (size - 1) / 2;

	double sigma_2 = 2.0 * sigma * sigma;

	double u;
	float sum;
	sum = 0.0f;

	for (int i = 0; i < size; i++)
	{
		u = (double)(i - halfSize);
		Smoothing_Filter_X[i] = (float)exp(-pow(u,2.0) / sigma_2);
		Smoothing_Filter_Y[i] = (float)exp(-pow(u,2.0) / sigma_2);
		Smoothing_Filter_Z[i] = (float)exp(-pow(u,2.0) / sigma_2);
		sum += Smoothing_Filter_X[i];
	}
	for (int i = 0; i < size; i++)
	{
		Smoothing_Filter_X[i] /= sum;
		Smoothing_Filter_Y[i] /= sum;
		Smoothing_Filter_Z[i] /= sum;
	}
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

	d_Certainty = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Smoothed_Certainty = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	SetMemory(d_Certainty, 1.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_Smoothed_Certainty, 1.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);

	// Copy volumes to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

	// Allocate temporary memory
	cl_mem d_Convolved_Rows = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Convolved_Columns = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, EPI_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);

	// Copy smoothing filters to constant memory
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_X, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_X , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Y, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Y , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Z, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Z , 0, NULL, NULL);

	// Set arguments for the kernels
	clSetKernelArg(SeparableConvolutionRowsKernel, 0, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionRowsKernel, 1, sizeof(cl_mem), &d_fMRI_Volumes);
	clSetKernelArg(SeparableConvolutionRowsKernel, 2, sizeof(cl_mem), &d_Certainty);
	clSetKernelArg(SeparableConvolutionRowsKernel, 3, sizeof(cl_mem), &c_Smoothing_Filter_Y);
	clSetKernelArg(SeparableConvolutionRowsKernel, 5, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(SeparableConvolutionRowsKernel, 6, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(SeparableConvolutionRowsKernel, 7, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(SeparableConvolutionRowsKernel, 8, sizeof(int), &EPI_DATA_T);

	clSetKernelArg(SeparableConvolutionColumnsKernel, 0, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 1, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_X);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 4, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 5, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 6, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 7, sizeof(int), &EPI_DATA_T);

	clSetKernelArg(SeparableConvolutionRodsKernel, 0, sizeof(cl_mem), &d_Smoothed_fMRI_Volumes);
	clSetKernelArg(SeparableConvolutionRodsKernel, 1, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionRodsKernel, 2, sizeof(cl_mem), &d_Smoothed_Certainty);
	clSetKernelArg(SeparableConvolutionRodsKernel, 3, sizeof(cl_mem), &c_Smoothing_Filter_Z);
    clSetKernelArg(SeparableConvolutionRodsKernel, 5, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(SeparableConvolutionRodsKernel, 6, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(SeparableConvolutionRodsKernel, 7, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(SeparableConvolutionRodsKernel, 8, sizeof(int), &EPI_DATA_T);

	// Loop over volumes
	for (int v = 0; v < EPI_DATA_T; v++)
	{
		clSetKernelArg(SeparableConvolutionRowsKernel, 4, sizeof(int), &v);
		runKernelErrorSeparableConvolutionRows = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRowsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRows, localWorkSizeSeparableConvolutionRows, 0, NULL, NULL);
		clFinish(commandQueue);

		clSetKernelArg(SeparableConvolutionColumnsKernel, 3, sizeof(int), &v);
		runKernelErrorSeparableConvolutionColumns = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionColumnsKernel, 3, NULL, globalWorkSizeSeparableConvolutionColumns, localWorkSizeSeparableConvolutionColumns, 0, NULL, NULL);
		clFinish(commandQueue);
		
		clSetKernelArg(SeparableConvolutionRodsKernel, 4, sizeof(int), &v);
		runKernelErrorSeparableConvolutionRods = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRodsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRods, localWorkSizeSeparableConvolutionRods, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	// Copy result back to host
	clEnqueueReadBuffer(commandQueue, d_Smoothed_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Smoothed_fMRI_Volumes, 0, NULL, NULL);
	
	// Free memory
	clReleaseMemObject(d_Convolved_Rows);
	clReleaseMemObject(d_Convolved_Columns);

	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Smoothed_fMRI_Volumes);

	clReleaseMemObject(d_Certainty);
	clReleaseMemObject(d_Smoothed_Certainty);

	clReleaseMemObject(c_Smoothing_Filter_X);
	clReleaseMemObject(c_Smoothing_Filter_Y);
	clReleaseMemObject(c_Smoothing_Filter_Z);
}

// Performs smoothing of a number of volumes
void BROCCOLI_LIB::PerformSmoothing(cl_mem d_Smoothed_Volumes, cl_mem d_Volumes, float* h_Smoothing_Filter_X, float* h_Smoothing_Filter_Y, float* h_Smoothing_Filter_Z, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
	SetGlobalAndLocalWorkSizesSeparableConvolution(DATA_W,DATA_H,DATA_D);

	// Allocate memory for smoothing filters
	c_Smoothing_Filter_X = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Y = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Z = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);

	// Copy smoothing filters to constant memory
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_X, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_X , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Y, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Y , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Z, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Z , 0, NULL, NULL);

	// Allocate temporary memory
	cl_mem d_Convolved_Rows = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Convolved_Columns = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);

	cl_mem d_Certainty = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);

	SetMemory(d_Certainty, 1.0f, DATA_W * DATA_H * DATA_D);
	
	// Set arguments for the kernels
	clSetKernelArg(SeparableConvolutionRowsKernel, 0, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionRowsKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(SeparableConvolutionRowsKernel, 2, sizeof(cl_mem), &d_Certainty);
	clSetKernelArg(SeparableConvolutionRowsKernel, 3, sizeof(cl_mem), &c_Smoothing_Filter_Y);
	clSetKernelArg(SeparableConvolutionRowsKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionRowsKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionRowsKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionRowsKernel, 8, sizeof(int), &DATA_T);

	clSetKernelArg(SeparableConvolutionColumnsKernel, 0, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 1, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_X);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 4, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 5, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 6, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 7, sizeof(int), &DATA_T);

	clSetKernelArg(SeparableConvolutionRodsKernel, 0, sizeof(cl_mem), &d_Smoothed_Volumes);
	clSetKernelArg(SeparableConvolutionRodsKernel, 1, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionRodsKernel, 2, sizeof(cl_mem), &d_Certainty);
	clSetKernelArg(SeparableConvolutionRodsKernel, 3, sizeof(cl_mem), &c_Smoothing_Filter_Z);
	clSetKernelArg(SeparableConvolutionRodsKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionRodsKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionRodsKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionRodsKernel, 8, sizeof(int), &DATA_T);

	// Loop over volumes
	for (int v = 0; v < DATA_T; v++)
	{
		clSetKernelArg(SeparableConvolutionRowsKernel, 4, sizeof(int), &v);
		runKernelErrorSeparableConvolutionRows = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRowsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRows, localWorkSizeSeparableConvolutionRows, 0, NULL, NULL);
		clFinish(commandQueue);

		clSetKernelArg(SeparableConvolutionColumnsKernel, 3, sizeof(int), &v);
		runKernelErrorSeparableConvolutionColumns = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionColumnsKernel, 3, NULL, globalWorkSizeSeparableConvolutionColumns, localWorkSizeSeparableConvolutionColumns, 0, NULL, NULL);
		clFinish(commandQueue);
	
		clSetKernelArg(SeparableConvolutionRodsKernel, 4, sizeof(int), &v);
		runKernelErrorSeparableConvolutionRods = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRodsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRods, localWorkSizeSeparableConvolutionRods, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	// Free temporary memory
	clReleaseMemObject(c_Smoothing_Filter_X);
	clReleaseMemObject(c_Smoothing_Filter_Y);
	clReleaseMemObject(c_Smoothing_Filter_Z);

	clReleaseMemObject(d_Convolved_Rows);
	clReleaseMemObject(d_Convolved_Columns);

	clReleaseMemObject(d_Certainty);
}

// Performs smoothing of a number of volumes, normalized
void BROCCOLI_LIB::PerformSmoothingNormalized(cl_mem d_Smoothed_Volumes, cl_mem d_Volumes, cl_mem d_Certainty, cl_mem d_Smoothed_Certainty, float* h_Smoothing_Filter_X, float* h_Smoothing_Filter_Y, float* h_Smoothing_Filter_Z, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
	SetGlobalAndLocalWorkSizesSeparableConvolution(DATA_W,DATA_H,DATA_D);

	// Allocate memory for smoothing filters
	c_Smoothing_Filter_X = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Y = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Z = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	
	// Copy smoothing filters to constant memory
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_X, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_X , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Y, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Y , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Z, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Z , 0, NULL, NULL);

	// Allocate temporary memory
	cl_mem d_Convolved_Rows = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Convolved_Columns = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);

	// Set arguments for the kernels
	clSetKernelArg(SeparableConvolutionRowsKernel, 0, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionRowsKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(SeparableConvolutionRowsKernel, 2, sizeof(cl_mem), &d_Certainty);
	clSetKernelArg(SeparableConvolutionRowsKernel, 3, sizeof(cl_mem), &c_Smoothing_Filter_Y);
	clSetKernelArg(SeparableConvolutionRowsKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionRowsKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionRowsKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionRowsKernel, 8, sizeof(int), &DATA_T);

	clSetKernelArg(SeparableConvolutionColumnsKernel, 0, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 1, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_X);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 4, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 5, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 6, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 7, sizeof(int), &DATA_T);

	clSetKernelArg(SeparableConvolutionRodsKernel, 0, sizeof(cl_mem), &d_Smoothed_Volumes);
	clSetKernelArg(SeparableConvolutionRodsKernel, 1, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionRodsKernel, 2, sizeof(cl_mem), &d_Smoothed_Certainty);
	clSetKernelArg(SeparableConvolutionRodsKernel, 3, sizeof(cl_mem), &c_Smoothing_Filter_Z);
	clSetKernelArg(SeparableConvolutionRodsKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionRodsKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionRodsKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionRodsKernel, 8, sizeof(int), &DATA_T);


	// Loop over volumes
	for (int v = 0; v < DATA_T; v++)
	{
		clSetKernelArg(SeparableConvolutionRowsKernel, 4, sizeof(int), &v);
		runKernelErrorSeparableConvolutionRows = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRowsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRows, localWorkSizeSeparableConvolutionRows, 0, NULL, NULL);
		clFinish(commandQueue);

		clSetKernelArg(SeparableConvolutionColumnsKernel, 3, sizeof(int), &v);
		runKernelErrorSeparableConvolutionColumns = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionColumnsKernel, 3, NULL, globalWorkSizeSeparableConvolutionColumns, localWorkSizeSeparableConvolutionColumns, 0, NULL, NULL);
		clFinish(commandQueue);

		clSetKernelArg(SeparableConvolutionRodsKernel, 4, sizeof(int), &v);
		runKernelErrorSeparableConvolutionRods = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRodsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRods, localWorkSizeSeparableConvolutionRods, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	// Free temporary memory
	clReleaseMemObject(c_Smoothing_Filter_X);
	clReleaseMemObject(c_Smoothing_Filter_Y);
	clReleaseMemObject(c_Smoothing_Filter_Z);

	clReleaseMemObject(d_Convolved_Rows);
	clReleaseMemObject(d_Convolved_Columns);
}

// Performs smoothing of a number of volumes, overwrites data
void BROCCOLI_LIB::PerformSmoothing(cl_mem d_Volumes, float* h_Smoothing_Filter_X, float* h_Smoothing_Filter_Y, float* h_Smoothing_Filter_Z, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
	SetGlobalAndLocalWorkSizesSeparableConvolution(DATA_W,DATA_H,DATA_D);

	// Allocate memory for smoothing filters
	c_Smoothing_Filter_X = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Y = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Z = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);

	// Copy smoothing filters to constant memory
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_X, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_X , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Y, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Y , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Z, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Z , 0, NULL, NULL);

	// Allocate temporary memory
	cl_mem d_Convolved_Rows = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Convolved_Columns = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);

	cl_mem d_Certainty = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);
	
	SetMemory(d_Certainty, 1.0f, DATA_W * DATA_H * DATA_D);

	// Set arguments for the kernels
	clSetKernelArg(SeparableConvolutionRowsKernel, 0, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionRowsKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(SeparableConvolutionRowsKernel, 2, sizeof(cl_mem), &d_Certainty);
	clSetKernelArg(SeparableConvolutionRowsKernel, 3, sizeof(cl_mem), &c_Smoothing_Filter_Y);
	clSetKernelArg(SeparableConvolutionRowsKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionRowsKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionRowsKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionRowsKernel, 8, sizeof(int), &DATA_T);

	clSetKernelArg(SeparableConvolutionColumnsKernel, 0, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 1, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_X);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 4, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 5, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 6, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 7, sizeof(int), &DATA_T);

	clSetKernelArg(SeparableConvolutionRodsKernel, 0, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(SeparableConvolutionRodsKernel, 1, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionRodsKernel, 2, sizeof(cl_mem), &d_Certainty);
	clSetKernelArg(SeparableConvolutionRodsKernel, 3, sizeof(cl_mem), &c_Smoothing_Filter_Z);
	clSetKernelArg(SeparableConvolutionRodsKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionRodsKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionRodsKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionRodsKernel, 8, sizeof(int), &DATA_T);

	// Loop over volumes
	for (int v = 0; v < DATA_T; v++)
	{
		clSetKernelArg(SeparableConvolutionRowsKernel, 4, sizeof(int), &v);
		runKernelErrorSeparableConvolutionRows = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRowsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRows, localWorkSizeSeparableConvolutionRows, 0, NULL, NULL);
		clFinish(commandQueue);

		clSetKernelArg(SeparableConvolutionColumnsKernel, 3, sizeof(int), &v);
		runKernelErrorSeparableConvolutionColumns = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionColumnsKernel, 3, NULL, globalWorkSizeSeparableConvolutionColumns, localWorkSizeSeparableConvolutionColumns, 0, NULL, NULL);
		clFinish(commandQueue);

		clSetKernelArg(SeparableConvolutionRodsKernel, 4, sizeof(int), &v);
		runKernelErrorSeparableConvolutionRods = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRodsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRods, localWorkSizeSeparableConvolutionRods, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	// Free temporary memory
	clReleaseMemObject(c_Smoothing_Filter_X);
	clReleaseMemObject(c_Smoothing_Filter_Y);
	clReleaseMemObject(c_Smoothing_Filter_Z);

	clReleaseMemObject(d_Convolved_Rows);
	clReleaseMemObject(d_Convolved_Columns);

	clReleaseMemObject(d_Certainty);
}

// Performs smoothing of a number of volumes, overwrites data, normalized
void BROCCOLI_LIB::PerformSmoothingNormalized(cl_mem d_Volumes, cl_mem d_Certainty, cl_mem d_Smoothed_Certainty, float* h_Smoothing_Filter_X, float* h_Smoothing_Filter_Y, float* h_Smoothing_Filter_Z, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
	SetGlobalAndLocalWorkSizesSeparableConvolution(DATA_W,DATA_H,DATA_D);

	// Allocate memory for smoothing filters
	c_Smoothing_Filter_X = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Y = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Z = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	
	// Copy smoothing filters to constant memory
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_X, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_X , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Y, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Y , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Z, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Z , 0, NULL, NULL);

	// Allocate temporary memory
	cl_mem d_Convolved_Rows = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Convolved_Columns = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);
	
	// Set arguments for the kernels
	clSetKernelArg(SeparableConvolutionRowsKernel, 0, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionRowsKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(SeparableConvolutionRowsKernel, 2, sizeof(cl_mem), &d_Certainty);
	clSetKernelArg(SeparableConvolutionRowsKernel, 3, sizeof(cl_mem), &c_Smoothing_Filter_Y);
	clSetKernelArg(SeparableConvolutionRowsKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionRowsKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionRowsKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionRowsKernel, 8, sizeof(int), &DATA_T);

	clSetKernelArg(SeparableConvolutionColumnsKernel, 0, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 1, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_X);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 4, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 5, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 6, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 7, sizeof(int), &DATA_T);

	clSetKernelArg(SeparableConvolutionRodsKernel, 0, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(SeparableConvolutionRodsKernel, 1, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionRodsKernel, 2, sizeof(cl_mem), &d_Smoothed_Certainty);
	clSetKernelArg(SeparableConvolutionRodsKernel, 3, sizeof(cl_mem), &c_Smoothing_Filter_Z);
	clSetKernelArg(SeparableConvolutionRodsKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionRodsKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionRodsKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionRodsKernel, 8, sizeof(int), &DATA_T);

	// Loop over volumes
	for (int v = 0; v < DATA_T; v++)
	{
		clSetKernelArg(SeparableConvolutionRowsKernel, 4, sizeof(int), &v);
		runKernelErrorSeparableConvolutionRows = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRowsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRows, localWorkSizeSeparableConvolutionRows, 0, NULL, NULL);
		clFinish(commandQueue);

		clSetKernelArg(SeparableConvolutionColumnsKernel, 3, sizeof(int), &v);
		runKernelErrorSeparableConvolutionColumns = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionColumnsKernel, 3, NULL, globalWorkSizeSeparableConvolutionColumns, localWorkSizeSeparableConvolutionColumns, 0, NULL, NULL);
		clFinish(commandQueue);

		clSetKernelArg(SeparableConvolutionRodsKernel, 4, sizeof(int), &v);
		runKernelErrorSeparableConvolutionRods = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRodsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRods, localWorkSizeSeparableConvolutionRods, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	// Free temporary memory
	clReleaseMemObject(c_Smoothing_Filter_X);
	clReleaseMemObject(c_Smoothing_Filter_Y);
	clReleaseMemObject(c_Smoothing_Filter_Z);

	clReleaseMemObject(d_Convolved_Rows);
	clReleaseMemObject(d_Convolved_Columns);
}

// Performs detrending of an fMRI dataset
void BROCCOLI_LIB::PerformDetrending(cl_mem d_Detrended_Volumes, cl_mem d_Volumes, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
	// Allocate host memory
	h_X_Detrend = (float*)malloc(NUMBER_OF_DETRENDING_REGRESSORS * DATA_T * sizeof(float));
	h_xtxxt_Detrend = (float*)malloc(NUMBER_OF_DETRENDING_REGRESSORS * DATA_T * sizeof(float));
	
	// Setup regressors for mean, linear, quadratic and cubic trends
	SetupDetrendingRegressors(DATA_T);

	// Allocate constant memory on device
	c_X_Detrend = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_DETRENDING_REGRESSORS * DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_Detrend = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_DETRENDING_REGRESSORS * DATA_T * sizeof(float), NULL, NULL);
	
	// Copy data to constant memory
	clEnqueueWriteBuffer(commandQueue, c_X_Detrend, CL_TRUE, 0, NUMBER_OF_DETRENDING_REGRESSORS * DATA_T * sizeof(float), h_X_Detrend , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_Detrend, CL_TRUE, 0, NUMBER_OF_DETRENDING_REGRESSORS * DATA_T * sizeof(float), h_xtxxt_Detrend , 0, NULL, NULL);

	SetGlobalAndLocalWorkSizesStatisticalCalculations(DATA_W, DATA_H, DATA_D);

	h_Censored_Timepoints = (float*)malloc(EPI_DATA_T * sizeof(float));
	c_Censored_Timepoints = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(float), NULL, NULL);

	for (int t = 0; t < EPI_DATA_T; t++)
	{
		h_Censored_Timepoints[t] = 1.0f;
	}
	clEnqueueWriteBuffer(commandQueue, c_Censored_Timepoints, CL_TRUE, 0, EPI_DATA_T * sizeof(float), h_Censored_Timepoints , 0, NULL, NULL);


	// First estimate beta weights
	clSetKernelArg(CalculateBetaValuesGLMKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_Detrend);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 4, sizeof(cl_mem), &c_Censored_Timepoints);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 8, sizeof(int), &DATA_T);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 9, sizeof(int), &NUMBER_OF_DETRENDING_REGRESSORS);

	runKernelErrorCalculateBetaValuesGLM = clEnqueueNDRangeKernel(commandQueue, CalculateBetaValuesGLMKernel, 3, NULL, globalWorkSizeCalculateBetaValuesGLM, localWorkSizeCalculateBetaValuesGLM, 0, NULL, NULL);
	clFinish(commandQueue);



	// Then remove linear fit
	clSetKernelArg(RemoveLinearFitKernel, 0, sizeof(cl_mem), &d_Detrended_Volumes);
	clSetKernelArg(RemoveLinearFitKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(RemoveLinearFitKernel, 2, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(RemoveLinearFitKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(RemoveLinearFitKernel, 4, sizeof(cl_mem), &c_X_Detrend);
	clSetKernelArg(RemoveLinearFitKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(RemoveLinearFitKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(RemoveLinearFitKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(RemoveLinearFitKernel, 8, sizeof(int), &DATA_T);
	clSetKernelArg(RemoveLinearFitKernel, 9, sizeof(int), &NUMBER_OF_DETRENDING_REGRESSORS);

	runKernelErrorRemoveLinearFit = clEnqueueNDRangeKernel(commandQueue, RemoveLinearFitKernel, 3, NULL, globalWorkSizeRemoveLinearFit, localWorkSizeRemoveLinearFit, 0, NULL, NULL);
	clFinish(commandQueue);

	// Free host memory
	free(h_Censored_Timepoints);
	free(h_X_Detrend);
	free(h_xtxxt_Detrend);

	// Free constant memory
	clReleaseMemObject(c_Censored_Timepoints);
	clReleaseMemObject(c_X_Detrend);
	clReleaseMemObject(c_xtxxt_Detrend);
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
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Smoothed_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	//------------------------------------

	//NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS;
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS;

	//h_X_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
	//h_xtxxt_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
	//h_Contrasts = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float));
	//h_ctxtxc_GLM = (float*)malloc(NUMBER_OF_CONTRASTS * sizeof(float));

	//SetupStatisticalAnalysisRegressors(EPI_DATA_T);

	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	//c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts_In , 0, NULL, NULL);
	//clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In , 0, NULL, NULL);

	/*
	for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
	{
		for (int t = 0; t < EPI_DATA_T; t++)
		{
			h_X_GLM_Out[t + r * EPI_DATA_T] = h_X_GLM[t + r * EPI_DATA_T];
			h_xtxxt_GLM_Out[t + r * EPI_DATA_T] = h_xtxxt_GLM[t + r * EPI_DATA_T];
		}
	}

	free(h_X_GLM);
	free(h_xtxxt_GLM);
	free(h_Contrasts);
	free(h_ctxtxc_GLM);
	 */


	//------------------------------------


	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	//d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * 1 * sizeof(float), NULL, NULL);
	d_Beta_Contrasts = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Residuals = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	
	d_AR1_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR2_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR3_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR4_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Whitened_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts_In , 0, NULL, NULL);
	//clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In , 0, NULL, NULL);

	clFinish(commandQueue);

	SegmentEPIData();
	
	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, AR_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	PerformSmoothing(d_Smoothed_EPI_Mask, d_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
	
	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, EPI_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	//PerformSmoothingNormalized(d_fMRI_Volumes, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	
	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, AR_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	
	h_Censored_Timepoints = (float*)malloc(EPI_DATA_T * sizeof(float));
	c_Censored_Timepoints = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(float), NULL, NULL);

	// Start with all timepoints
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		h_Censored_Timepoints[t] = 1.0f;
	}
	clEnqueueWriteBuffer(commandQueue, c_Censored_Timepoints, CL_TRUE, 0, EPI_DATA_T * sizeof(float), h_Censored_Timepoints , 0, NULL, NULL);


	int NUMBER_OF_INVALID_TIMEPOINTS = 0;

	for (int it = 0; it < 1; it++)
	{
		// Calculate beta values
		clSetKernelArg(CalculateBetaValuesGLMKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 1, sizeof(cl_mem), &d_fMRI_Volumes);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_GLM);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 4, sizeof(cl_mem), &c_Censored_Timepoints);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 5, sizeof(int), &EPI_DATA_W);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 6, sizeof(int), &EPI_DATA_H);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 7, sizeof(int), &EPI_DATA_D);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 8, sizeof(int), &EPI_DATA_T);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 9, sizeof(int), &NUMBER_OF_GLM_REGRESSORS);
		runKernelErrorCalculateBetaValuesGLM = clEnqueueNDRangeKernel(commandQueue, CalculateBetaValuesGLMKernel, 3, NULL, globalWorkSizeCalculateBetaValuesGLM, localWorkSizeCalculateBetaValuesGLM, 0, NULL, NULL);
		clFinish(commandQueue);

		// Calculate t-values and residuals
		/*
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 1, sizeof(cl_mem), &d_Beta_Contrasts);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 2, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 3, sizeof(cl_mem), &d_Residual_Variances);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 4, sizeof(cl_mem), &d_fMRI_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 5, sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 6, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 7, sizeof(cl_mem), &c_X_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 8, sizeof(cl_mem), &c_Contrasts);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 9, sizeof(cl_mem), &c_ctxtxc_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 10, sizeof(cl_mem), &c_Censored_Timepoints);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 11, sizeof(int),   &EPI_DATA_W);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 12, sizeof(int),   &EPI_DATA_H);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 13, sizeof(int),   &EPI_DATA_D);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 14, sizeof(int),   &EPI_DATA_T);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 15, sizeof(int),   &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 16, sizeof(int),   &NUMBER_OF_CONTRASTS);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 17, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorCalculateStatisticalMapsGLM = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMTTestKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
		clFinish(commandQueue);
		*/


		clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 1, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 2, sizeof(cl_mem), &d_Residual_Variances);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 3, sizeof(cl_mem), &d_fMRI_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 4, sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 5, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 6, sizeof(cl_mem), &c_X_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 7, sizeof(cl_mem), &c_Contrasts);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 8, sizeof(cl_mem), &c_ctxtxc_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 9, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 10, sizeof(int),   &EPI_DATA_H);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 11, sizeof(int),   &EPI_DATA_D);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 12, sizeof(int),   &EPI_DATA_T);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 13, sizeof(int),   &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 14, sizeof(int),   &NUMBER_OF_CONTRASTS);
		runKernelErrorCalculateStatisticalMapsGLM = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMFTestKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
		clFinish(commandQueue);


		/*
		// Estimate auto correlation from residuals
		clSetKernelArg(EstimateAR4ModelsKernel, 0, sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 1, sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 2, sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 3, sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 4, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(EstimateAR4ModelsKernel, 5, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(EstimateAR4ModelsKernel, 6, sizeof(int),   &EPI_DATA_W);
		clSetKernelArg(EstimateAR4ModelsKernel, 7, sizeof(int),   &EPI_DATA_H);
		clSetKernelArg(EstimateAR4ModelsKernel, 8, sizeof(int),   &EPI_DATA_D);
		clSetKernelArg(EstimateAR4ModelsKernel, 9, sizeof(int),   &EPI_DATA_T);
		runKernelErrorEstimateAR4Models = clEnqueueNDRangeKernel(commandQueue, EstimateAR4ModelsKernel, 3, NULL, globalWorkSizeEstimateAR4Models, localWorkSizeEstimateAR4Models, 0, NULL, NULL);

		// Smooth AR estimates
		PerformSmoothingNormalized(d_AR1_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR2_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR3_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR4_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

		// Remove auto correlation from data
		clSetKernelArg(ApplyWhiteningAR4Kernel, 0, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 1, sizeof(cl_mem), &d_fMRI_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 2, sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 3, sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 4, sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 5, sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 6, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 7, sizeof(int),   &EPI_DATA_W);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 8, sizeof(int),   &EPI_DATA_H);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 9, sizeof(int),   &EPI_DATA_D);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 10, sizeof(int),  &EPI_DATA_T);
		runKernelErrorApplyAR4Whitening = clEnqueueNDRangeKernel(commandQueue, ApplyWhiteningAR4Kernel, 3, NULL, globalWorkSizeApplyWhiteningAR4, localWorkSizeApplyWhiteningAR4, 0, NULL, NULL);
		*/

		clEnqueueCopyBuffer(commandQueue, d_Whitened_fMRI_Volumes, d_fMRI_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), 0, NULL, NULL);
	}

	clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * 1 * sizeof(float), h_Statistical_Maps, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Residuals, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);

	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR1_Estimates, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR2_Estimates, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR3_Estimates, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR4_Estimates, 0, NULL, NULL);
	
	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_EPI_Mask);
	clReleaseMemObject(d_Smoothed_EPI_Mask);

	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);

	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Beta_Contrasts);
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);

	clReleaseMemObject(d_AR1_Estimates);
	clReleaseMemObject(d_AR2_Estimates);
	clReleaseMemObject(d_AR3_Estimates);
	clReleaseMemObject(d_AR4_Estimates);
	clReleaseMemObject(d_Whitened_fMRI_Volumes);


	free(h_Censored_Timepoints);
	clReleaseMemObject(c_Censored_Timepoints);
}

// Calculates a statistical map for first level analysis
void BROCCOLI_LIB::CalculateStatisticalMapsGLMFirstLevel(cl_mem d_Volumes)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, AR_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	PerformSmoothing(d_Smoothed_EPI_Mask, d_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);



	h_Censored_Timepoints = (float*)malloc(EPI_DATA_T * sizeof(float));
	c_Censored_Timepoints = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(float), NULL, NULL);

	// Start with all timepoints
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		h_Censored_Timepoints[t] = 1.0f;
	}
	clEnqueueWriteBuffer(commandQueue, c_Censored_Timepoints, CL_TRUE, 0, EPI_DATA_T * sizeof(float), h_Censored_Timepoints , 0, NULL, NULL);

	int NUMBER_OF_INVALID_TIMEPOINTS = 0;

	clSetKernelArg(RemoveMeanKernel, 0, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(RemoveMeanKernel, 1, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(RemoveMeanKernel, 2, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(RemoveMeanKernel, 3, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(RemoveMeanKernel, 4, sizeof(int), &EPI_DATA_T);
	clEnqueueNDRangeKernel(commandQueue, RemoveMeanKernel, 3, NULL, globalWorkSizeCalculateBetaValuesGLM, localWorkSizeCalculateBetaValuesGLM, 0, NULL, NULL);

	clEnqueueCopyBuffer(commandQueue, d_Volumes, d_Whitened_fMRI_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), 0, NULL, NULL);

	// Cochrane-Orcutt procedure
	for (int it = 0; it < 3; it++)
	{
		/*
		for (int t = 0; t < (it*4); t++)
		{
			h_Censored_Timepoints[t] = 0.0f;
		}
		clEnqueueWriteBuffer(commandQueue, c_Censored_Timepoints, CL_TRUE, 0, EPI_DATA_T * sizeof(float), h_Censored_Timepoints , 0, NULL, NULL);

		int NUMBER_OF_INVALID_TIMEPOINTS = it*4;
		*/

		// Calculate beta values
		clSetKernelArg(CalculateBetaValuesGLMKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 1, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_GLM);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 4, sizeof(cl_mem), &c_Censored_Timepoints);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 5, sizeof(int), &EPI_DATA_W);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 6, sizeof(int), &EPI_DATA_H);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 7, sizeof(int), &EPI_DATA_D);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 8, sizeof(int), &EPI_DATA_T);
		clSetKernelArg(CalculateBetaValuesGLMKernel, 9, sizeof(int), &NUMBER_OF_TOTAL_GLM_REGRESSORS);

		runKernelErrorCalculateBetaValuesGLM = clEnqueueNDRangeKernel(commandQueue, CalculateBetaValuesGLMKernel, 3, NULL, globalWorkSizeCalculateBetaValuesGLM, localWorkSizeCalculateBetaValuesGLM, 0, NULL, NULL);
		clFinish(commandQueue);

		// Calculate t-values and residuals
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 1, sizeof(cl_mem), &d_Beta_Contrasts);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 2, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 3, sizeof(cl_mem), &d_Residual_Variances);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 4, sizeof(cl_mem), &d_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 5, sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 6, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 7, sizeof(cl_mem), &c_X_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 8, sizeof(cl_mem), &c_Contrasts);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 9, sizeof(cl_mem), &c_ctxtxc_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 10, sizeof(cl_mem), &c_Censored_Timepoints);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 11, sizeof(int),   &EPI_DATA_W);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 12, sizeof(int),   &EPI_DATA_H);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 13, sizeof(int),   &EPI_DATA_D);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 14, sizeof(int),   &EPI_DATA_T);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 15, sizeof(int),   &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 16, sizeof(int),   &NUMBER_OF_CONTRASTS);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 17, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);

		runKernelErrorCalculateStatisticalMapsGLM = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMTTestKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
		clFinish(commandQueue);

		//PerformWhitening(d_Whitened_fMRI_Volumes, d_Residuals);


		// Estimate auto correlation from residuals
		clSetKernelArg(EstimateAR4ModelsKernel, 0, sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 1, sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 2, sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 3, sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 4, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(EstimateAR4ModelsKernel, 5, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(EstimateAR4ModelsKernel, 6, sizeof(int),   &EPI_DATA_W);
		clSetKernelArg(EstimateAR4ModelsKernel, 7, sizeof(int),   &EPI_DATA_H);
		clSetKernelArg(EstimateAR4ModelsKernel, 8, sizeof(int),   &EPI_DATA_D);
		clSetKernelArg(EstimateAR4ModelsKernel, 9, sizeof(int),   &EPI_DATA_T);
		clSetKernelArg(EstimateAR4ModelsKernel, 10, sizeof(int),  &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorEstimateAR4Models = clEnqueueNDRangeKernel(commandQueue, EstimateAR4ModelsKernel, 3, NULL, globalWorkSizeEstimateAR4Models, localWorkSizeEstimateAR4Models, 0, NULL, NULL);

		// Smooth auto correlation estimates
		PerformSmoothingNormalized(d_AR1_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR2_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR3_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR4_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);


		// Remove auto correlation from data
		clSetKernelArg(ApplyWhiteningAR4Kernel, 0, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 1, sizeof(cl_mem), &d_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 2, sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 3, sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 4, sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 5, sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 6, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 7, sizeof(int),   &EPI_DATA_W);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 8, sizeof(int),   &EPI_DATA_H);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 9, sizeof(int),   &EPI_DATA_D);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 10, sizeof(int),  &EPI_DATA_T);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 11, sizeof(int),  &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorApplyAR4Whitening = clEnqueueNDRangeKernel(commandQueue, ApplyWhiteningAR4Kernel, 3, NULL, globalWorkSizeApplyWhiteningAR4, localWorkSizeApplyWhiteningAR4, 0, NULL, NULL);


		// Copy whitened volumes back to volumes
		//clEnqueueCopyBuffer(commandQueue, d_Whitened_fMRI_Volumes, d_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), 0, NULL, NULL);
		//clEnqueueCopyBuffer(commandQueue, d_Whitened_fMRI_Volumes, d_Residuals, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), 0, NULL, NULL);
	}


	free(h_Censored_Timepoints);
	clReleaseMemObject(c_Censored_Timepoints);
}

void BROCCOLI_LIB::CalculateStatisticalMapsGLMFirstLevelPermutation(cl_mem d_Volumes)
{
	for (int it = 0; it < 1; it++)
	{
		clSetKernelArg(CalculateStatisticalMapsGLMPermutationKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
		clSetKernelArg(CalculateStatisticalMapsGLMPermutationKernel, 1, sizeof(cl_mem), &d_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMPermutationKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateStatisticalMapsGLMPermutationKernel, 3, sizeof(cl_mem), &c_xtxxt_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMPermutationKernel, 4, sizeof(cl_mem), &c_X_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMPermutationKernel, 5, sizeof(cl_mem), &c_Contrasts);
		clSetKernelArg(CalculateStatisticalMapsGLMPermutationKernel, 6, sizeof(cl_mem), &c_ctxtxc_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMPermutationKernel, 7, sizeof(int), &EPI_DATA_W);
		clSetKernelArg(CalculateStatisticalMapsGLMPermutationKernel, 8, sizeof(int), &EPI_DATA_H);
		clSetKernelArg(CalculateStatisticalMapsGLMPermutationKernel, 9, sizeof(int), &EPI_DATA_D);
		clSetKernelArg(CalculateStatisticalMapsGLMPermutationKernel, 10, sizeof(int), &EPI_DATA_T);
		clSetKernelArg(CalculateStatisticalMapsGLMPermutationKernel, 11, sizeof(int), &NUMBER_OF_GLM_REGRESSORS);
		clSetKernelArg(CalculateStatisticalMapsGLMPermutationKernel, 12, sizeof(int), &NUMBER_OF_CONTRASTS);
		runKernelErrorCalculateStatisticalMapsGLM = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMPermutationKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
		clFinish(commandQueue);
	

		/*
		// Estimate auto correlation from residuals
		clSetKernelArg(EstimateAR4ModelsKernel, 0, sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 1, sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 2, sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 3, sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 4, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(EstimateAR4ModelsKernel, 5, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(EstimateAR4ModelsKernel, 6, sizeof(int),   &EPI_DATA_W);
		clSetKernelArg(EstimateAR4ModelsKernel, 7, sizeof(int),   &EPI_DATA_H);
		clSetKernelArg(EstimateAR4ModelsKernel, 8, sizeof(int),   &EPI_DATA_D);
		clSetKernelArg(EstimateAR4ModelsKernel, 9, sizeof(int),   &EPI_DATA_T);
		runKernelErrorEstimateAR4Models = clEnqueueNDRangeKernel(commandQueue, EstimateAR4ModelsKernel, 3, NULL, globalWorkSizeEstimateAR4Models, localWorkSizeEstimateAR4Models, 0, NULL, NULL);

		// Smooth auto correlation estimates
		PerformSmoothingNormalized(d_AR1_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR2_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR3_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR4_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

		// Remove auto correlation from data
		clSetKernelArg(ApplyWhiteningAR4Kernel, 0, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 1, sizeof(cl_mem), &d_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 2, sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 3, sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 4, sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 5, sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 6, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 7, sizeof(int),   &EPI_DATA_W);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 8, sizeof(int),   &EPI_DATA_H);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 9, sizeof(int),   &EPI_DATA_D);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 10, sizeof(int),  &EPI_DATA_T);
		runKernelErrorApplyAR4Whitening = clEnqueueNDRangeKernel(commandQueue, ApplyWhiteningAR4Kernel, 3, NULL, globalWorkSizeApplyWhiteningAR4, localWorkSizeApplyWhiteningAR4, 0, NULL, NULL);

		// Copy whitened volumes back to volumes
		clEnqueueCopyBuffer(commandQueue, d_Whitened_fMRI_Volumes, d_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), 0, NULL, NULL);
		*/
	}
}

// Calculates a statistical map for second level analysis
void BROCCOLI_LIB::CalculateStatisticalMapsGLMSecondLevel(cl_mem d_Volumes)
{
	// Calculate beta values
	clSetKernelArg(CalculateBetaValuesGLMKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 2, sizeof(cl_mem), &d_Group_Mask);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_GLM);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 4, sizeof(int), &MNI_DATA_W);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 5, sizeof(int), &MNI_DATA_H);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 6, sizeof(int), &MNI_DATA_D);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 7, sizeof(int), &NUMBER_OF_SUBJECTS);
	clSetKernelArg(CalculateBetaValuesGLMKernel, 8, sizeof(int), &NUMBER_OF_GLM_REGRESSORS);
	//clSetKernelArg(CalculateBetaValuesGLMKernel, 4, sizeof(cl_mem), &c_Censor);
	runKernelErrorCalculateBetaValuesGLM = clEnqueueNDRangeKernel(commandQueue, CalculateBetaValuesGLMKernel, 3, NULL, globalWorkSizeCalculateBetaValuesGLM, localWorkSizeCalculateBetaValuesGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Calculate t-values and residuals
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 1, sizeof(cl_mem), &d_Beta_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 2, sizeof(cl_mem), &d_Residuals);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 3, sizeof(cl_mem), &d_Residual_Variances);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 4, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 5, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 6, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 7, sizeof(cl_mem), &c_X_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 8, sizeof(cl_mem), &c_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 9, sizeof(cl_mem), &c_ctxtxc_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 10, sizeof(int),   &MNI_DATA_W);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 11, sizeof(int),   &MNI_DATA_H);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 12, sizeof(int),   &MNI_DATA_D);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 13, sizeof(int),   &NUMBER_OF_SUBJECTS);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 14, sizeof(int),   &NUMBER_OF_GLM_REGRESSORS);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 15, sizeof(int),   &NUMBER_OF_CONTRASTS);
	//clSetKernelArg(CalculateStatisticalMapsGLMKernel, 10, sizeof(cl_mem), &c_Censor);
	runKernelErrorCalculateStatisticalMapsGLM = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMTTestKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::CalculatePermutationTestThresholdFirstLevelWrapper()
{
	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Detrended_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Whitened_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Permuted_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Smoothed_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	
	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_Permutation_Vector = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(unsigned short int), NULL, NULL);

	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Beta_Contrasts = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Residuals = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	
	d_AR1_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR2_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR3_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR4_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM_In, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In , 0, NULL, NULL);
	clFinish(commandQueue);

	SegmentEPIData();

	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, AR_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	PerformSmoothing(d_Smoothed_EPI_Mask, d_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, EPI_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	PerformSmoothingNormalized(d_fMRI_Volumes, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

	h_Permutation_Matrix = (unsigned short int*)malloc(NUMBER_OF_PERMUTATIONS * EPI_DATA_T * sizeof(unsigned short int));
	CalculatePermutationTestThresholdFirstLevel(d_fMRI_Volumes);
	free(h_Permutation_Matrix);

	clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Residuals, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);

	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR1_Estimates, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR2_Estimates, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR3_Estimates, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR4_Estimates, 0, NULL, NULL);

	clEnqueueReadBuffer(commandQueue, d_Detrended_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Detrended_fMRI_Volumes, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Whitened_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Whitened_fMRI_Volumes, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Permuted_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Permuted_fMRI_Volumes, 0, NULL, NULL);

	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Detrended_fMRI_Volumes);
	clReleaseMemObject(d_Whitened_fMRI_Volumes);
	clReleaseMemObject(d_Permuted_fMRI_Volumes);

	clReleaseMemObject(d_EPI_Mask);
	clReleaseMemObject(d_Smoothed_EPI_Mask);

	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);
	clReleaseMemObject(c_Permutation_Vector);

	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Beta_Contrasts);
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);

	clReleaseMemObject(d_AR1_Estimates);
	clReleaseMemObject(d_AR2_Estimates);
	clReleaseMemObject(d_AR3_Estimates);
	clReleaseMemObject(d_AR4_Estimates);
}

void BROCCOLI_LIB::PerformWhiteningPriorPermutations(cl_mem d_Whitened_Volumes, cl_mem d_Volumes)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, AR_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	
	int NUMBER_OF_INVALID_TIMEPOINTS = 0;

	cl_mem d_Total_AR1_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Total_AR2_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Total_AR3_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Total_AR4_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	clEnqueueCopyBuffer(commandQueue, d_Volumes, d_Whitened_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), 0, NULL, NULL);

	SetMemory(d_Total_AR1_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_Total_AR2_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_Total_AR3_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_Total_AR4_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);

	for (int it = 0; it < 3; it++)
	{
		// Estimate auto correlation
		clSetKernelArg(EstimateAR4ModelsKernel, 0, sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 1, sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 2, sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 3, sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 4, sizeof(cl_mem), &d_Whitened_Volumes);
		clSetKernelArg(EstimateAR4ModelsKernel, 5, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(EstimateAR4ModelsKernel, 6, sizeof(int),   &EPI_DATA_W);
		clSetKernelArg(EstimateAR4ModelsKernel, 7, sizeof(int),   &EPI_DATA_H);
		clSetKernelArg(EstimateAR4ModelsKernel, 8, sizeof(int),   &EPI_DATA_D);
		clSetKernelArg(EstimateAR4ModelsKernel, 9, sizeof(int),   &EPI_DATA_T);
		clSetKernelArg(EstimateAR4ModelsKernel, 10, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorEstimateAR4Models = clEnqueueNDRangeKernel(commandQueue, EstimateAR4ModelsKernel, 3, NULL, globalWorkSizeEstimateAR4Models, localWorkSizeEstimateAR4Models, 0, NULL, NULL);

		// Smooth AR estimates
		PerformSmoothingNormalized(d_AR1_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR2_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR3_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR4_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

		AddVolumes(d_Total_AR1_Estimates, d_AR1_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
		AddVolumes(d_Total_AR2_Estimates, d_AR2_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
		AddVolumes(d_Total_AR3_Estimates, d_AR3_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
		AddVolumes(d_Total_AR4_Estimates, d_AR4_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

		// Remove auto correlation from data
		clSetKernelArg(ApplyWhiteningAR4Kernel, 0, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 1, sizeof(cl_mem), &d_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 2, sizeof(cl_mem), &d_Total_AR1_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 3, sizeof(cl_mem), &d_Total_AR2_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 4, sizeof(cl_mem), &d_Total_AR3_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 5, sizeof(cl_mem), &d_Total_AR4_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 6, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 7, sizeof(int),   &EPI_DATA_W);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 8, sizeof(int),   &EPI_DATA_H);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 9, sizeof(int),   &EPI_DATA_D);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 10, sizeof(int),  &EPI_DATA_T);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 11, sizeof(int),  &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorApplyAR4Whitening = clEnqueueNDRangeKernel(commandQueue, ApplyWhiteningAR4Kernel, 3, NULL, globalWorkSizeApplyWhiteningAR4, localWorkSizeApplyWhiteningAR4, 0, NULL, NULL);
	}

	clEnqueueCopyBuffer(commandQueue, d_Total_AR1_Estimates, d_AR1_Estimates, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);
	clEnqueueCopyBuffer(commandQueue, d_Total_AR2_Estimates, d_AR2_Estimates, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);
	clEnqueueCopyBuffer(commandQueue, d_Total_AR3_Estimates, d_AR3_Estimates, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);
	clEnqueueCopyBuffer(commandQueue, d_Total_AR4_Estimates, d_AR4_Estimates, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

	clReleaseMemObject(d_Total_AR1_Estimates);
	clReleaseMemObject(d_Total_AR2_Estimates);
	clReleaseMemObject(d_Total_AR3_Estimates);
	clReleaseMemObject(d_Total_AR4_Estimates);
}

// Calculates a significance threshold for a single subject
void BROCCOLI_LIB::CalculatePermutationTestThresholdFirstLevel(cl_mem d_fMRI_Volumes)
{
	SetupParametersPermutationSingleSubject();
	GeneratePermutationMatrixFirstLevel();

	// Make the timeseries white prior to the random permutations
	//CreateBOLDRegressedVolumes();
	PerformDetrending(d_Detrended_fMRI_Volumes, d_fMRI_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	PerformWhiteningPriorPermutations(d_Whitened_fMRI_Volumes, d_Detrended_fMRI_Volumes);

    // Loop over all the permutations, save the maximum test value from each permutation
    for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
    {
        GeneratePermutedVolumesFirstLevel(d_Permuted_fMRI_Volumes, d_Whitened_fMRI_Volumes, p);
        //PerformSmoothing(d_Smoothed_fMRI_Volumes);
        //PerformDetrendingPermutation();
        //PerformWhiteningPermutation();
        CalculateStatisticalMapsGLMFirstLevelPermutation(d_Permuted_fMRI_Volumes);
		h_Permutation_Distribution[p] = CalculateMax(d_Statistical_Maps, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
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
void BROCCOLI_LIB::CalculatePermutationTestThresholdSecondLevel()
{
	GeneratePermutationMatrixSecondLevel();

    // Loop over all the permutations, save the maximum test value from each permutation
    for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
    {
         // Copy a new permutation vector to constant memory

        //GeneratePermutedVolumesSecondLevel();
        //CalculateStatisticalMapPermutation();
		//h_Maximum_Test_Values[p] = FindMaxTestvaluePermutation();
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
void BROCCOLI_LIB::GeneratePermutationMatrixFirstLevel()
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


void BROCCOLI_LIB::CreateBOLDRegressedVolumes()
{
	
}



void BROCCOLI_LIB::GeneratePermutedVolumesFirstLevel(cl_mem d_Permuted_fMRI_Volumes, cl_mem d_Whitened_fMRI_Volumes, int permutation)
{
	clEnqueueWriteBuffer(commandQueue, c_Permutation_Vector, CL_TRUE, 0, EPI_DATA_T * sizeof(unsigned short int), &h_Permutation_Matrix[permutation * EPI_DATA_T] , 0, NULL, NULL);

	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 0, sizeof(cl_mem), &d_Permuted_fMRI_Volumes);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 1, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 2, sizeof(cl_mem), &d_AR1_Estimates);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 3, sizeof(cl_mem), &d_AR2_Estimates);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 4, sizeof(cl_mem), &d_AR3_Estimates);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 5, sizeof(cl_mem), &d_AR4_Estimates);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 6, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 7, sizeof(cl_mem), &c_Permutation_Vector);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 8, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 9, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 10, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 11, sizeof(int), &EPI_DATA_T);
	
	clEnqueueNDRangeKernel(commandQueue, GeneratePermutedVolumesFirstLevelKernel, 3, NULL, globalWorkSizeGeneratePermutedVolumesFirstLevel, localWorkSizeGeneratePermutedVolumesFirstLevel, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::GeneratePermutedVolumesSecondLevel(cl_mem d_Permuted_Volumes, cl_mem d_Volumes, int permutation)
{
	clEnqueueWriteBuffer(commandQueue, c_Permutation_Vector, CL_TRUE, 0, NUMBER_OF_SUBJECTS * sizeof(float), &h_Permutation_Matrix[permutation * NUMBER_OF_SUBJECTS] , 0, NULL, NULL);

	clSetKernelArg(GeneratePermutedVolumesSecondLevelKernel, 0, sizeof(cl_mem), &d_Permuted_fMRI_Volumes);
	clSetKernelArg(GeneratePermutedVolumesSecondLevelKernel, 1, sizeof(cl_mem), &d_fMRI_Volumes);
	clSetKernelArg(GeneratePermutedVolumesSecondLevelKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(GeneratePermutedVolumesSecondLevelKernel, 3, sizeof(cl_mem), &c_Permutation_Vector);
	clSetKernelArg(GeneratePermutedVolumesSecondLevelKernel, 4, sizeof(int), &MNI_DATA_W);
	clSetKernelArg(GeneratePermutedVolumesSecondLevelKernel, 5, sizeof(int), &MNI_DATA_H);
	clSetKernelArg(GeneratePermutedVolumesSecondLevelKernel, 6, sizeof(int), &MNI_DATA_D);
	clSetKernelArg(GeneratePermutedVolumesSecondLevelKernel, 7, sizeof(int), &NUMBER_OF_SUBJECTS);
	
	clEnqueueNDRangeKernel(commandQueue, GeneratePermutedVolumesSecondLevelKernel, 3, NULL, globalWorkSizeGeneratePermutedVolumesSecondLevel, localWorkSizeGeneratePermutedVolumesSecondLevel, 0, NULL, NULL);
	clFinish(commandQueue);
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
void BROCCOLI_LIB::GeneratePermutationMatrixSecondLevel()
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
	std::fstream file(filename.c_str(), std::ios::in | std::ios::binary);
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
	std::fstream file(filename.c_str(), std::ios::in | std::ios::binary);
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
	std::fstream file(filename.c_str(), std::ios::in | std::ios::binary);
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
	std::fstream file(filename.c_str(), std::ios::in | std::ios::binary);
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
	std::fstream file(filename.c_str(), std::ios::in | std::ios::binary);
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
	std::fstream file(filename.c_str(), std::ios::in | std::ios::binary);
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
	//out << SMOOTHING_AMOUNT_MM;
	//mm_string = out.str();

	//std::string filename_GLM = filename_GLM_filter + mm_string + "mm.raw";
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
	std::fstream file(filename.c_str(), std::ios::out | std::ios::binary);
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
	std::fstream file(filename.c_str(), std::ios::out | std::ios::binary);
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
	std::fstream file(filename.c_str(), std::ios::out | std::ios::binary);
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

	/*
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
	*/

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







void BROCCOLI_LIB::SolveEquationSystem(float* h_Parameter_Vector, float* h_A_Matrix, float* h_h_Vector, int N)
{
	Eigen::MatrixXd A(12,12);
	Eigen::VectorXd h(12,1);

	// Make a double version
	for (int i = 0; i < N; i++)
	{
		h(i) = (double)h_h_Vector[i];

		for (int j = 0; j < N; j++)
		{
			A(i,j) = (double)h_A_Matrix[i + j*N];

			if (i == j)
			{
				A(i,j) += 0.001;
			}
		}
	}

	Eigen::VectorXd x = A.fullPivHouseholderQr().solve(h);
	relativeErrorEquationSystemSolution = (A*x - h).norm() / h.norm(); // norm() is L2 norm

    for (int i = 0; i < N; i++)
	{
		h_Parameter_Vector[i] = (float)x(i);
	}
}




void BROCCOLI_LIB::SetupDetrendingRegressors(int N)
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

	Eigen::VectorXd Ones(N,1);
	Eigen::VectorXd Linear(N,1);
	Eigen::VectorXd Quadratic(N,1);
	Eigen::VectorXd Cubic(N,1);

	// 1 and X
	double offset = -((double)N - 1.0f)/2.0f;
	for (int t = 0; t < N; t++)
	{
		Ones(t) = 1.0;
		Linear(t) = offset + (double)t;
	}

	Quadratic = Linear.cwiseProduct(Linear);
	Cubic = Linear.cwiseProduct(Linear);
	Cubic = Cubic.cwiseProduct(Linear);

	// Normalize
	Ones.normalize();
	Linear.normalize();
	Quadratic.normalize();
	Cubic.normalize();

	// Setup total detrending design matrix
	Eigen::MatrixXd X(N,4);
	for (int i = 0; i < N; i++)
	{
		X(i,0) = Ones(i);
		X(i,1) = Linear(i);
		X(i,2) = Quadratic(i);
		X(i,3) = Cubic(i);
	}

	Eigen::MatrixXd xtx(4,4);
	xtx = X.transpose() * X;
	Eigen::MatrixXd inv_xtx = xtx.inverse();
	Eigen::MatrixXd xtxxt = inv_xtx * X.transpose();

	// Finally store regressors in ordinary arrays
	for (int i = 0; i < N; i++)
	{
		h_X_Detrend[i + 0 * N] = (float)Ones(i);
		h_X_Detrend[i + 1 * N] = (float)Linear(i);
		h_X_Detrend[i + 2 * N] = (float)Quadratic(i);
		h_X_Detrend[i + 3 * N] = (float)Cubic(i);

		h_xtxxt_Detrend[i + 0 * N] = (float)xtxxt(0,i);
		h_xtxxt_Detrend[i + 1 * N] = (float)xtxxt(1,i);
		h_xtxxt_Detrend[i + 2 * N] = (float)xtxxt(2,i);
		h_xtxxt_Detrend[i + 3 * N] = (float)xtxxt(3,i);
	}
}

void BROCCOLI_LIB::SetupStatisticalAnalysisRegressors(int N)
{
	// Calculate total number of regressors
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS;

	// Create detrending regressors
	Eigen::VectorXd Ones(N,1);
	Eigen::VectorXd Linear(N,1);
	Eigen::VectorXd Quadratic(N,1);
	Eigen::VectorXd Cubic(N,1);

	// 1 and X
	float offset = -((float)N - 1.0f)/2.0f;
	for (int t = 0; t < N; t++)
	{
		Ones(t) = 1.0;
		Linear(t) = offset + (double)t;
	}

	Quadratic = Linear.cwiseProduct(Linear);
	Cubic = Linear.cwiseProduct(Linear);
	Cubic = Cubic.cwiseProduct(Linear);

	// Normalize
	Ones.normalize();
	Linear.normalize();
	Quadratic.normalize();
	Cubic.normalize();

	// Setup total design matrix
	Eigen::MatrixXd X(N,NUMBER_OF_TOTAL_GLM_REGRESSORS);

	for (int i = 0; i < N; i++)
	{
		// Regressors for paradigms
		for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
		{
			X(i,r) = (double)h_X_GLM_In[i + r * N];
		}

		// Regressors for motion
		X(i,NUMBER_OF_GLM_REGRESSORS + 0) = h_Motion_Parameters[i + 0 * N];
		X(i,NUMBER_OF_GLM_REGRESSORS + 1) = h_Motion_Parameters[i + 1 * N];
		X(i,NUMBER_OF_GLM_REGRESSORS + 2) = h_Motion_Parameters[i + 2 * N];
		X(i,NUMBER_OF_GLM_REGRESSORS + 3) = h_Motion_Parameters[i + 3 * N];
		X(i,NUMBER_OF_GLM_REGRESSORS + 4) = h_Motion_Parameters[i + 4 * N];
		X(i,NUMBER_OF_GLM_REGRESSORS + 5) = h_Motion_Parameters[i + 5 * N];

		// Regressors for detrending
		X(i,NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 0) = Ones(i);
		X(i,NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 1) = Linear(i);
		X(i,NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 2) = Quadratic(i);
		X(i,NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 3) = Cubic(i);
	}

	Eigen::MatrixXd xtx(NUMBER_OF_TOTAL_GLM_REGRESSORS,NUMBER_OF_TOTAL_GLM_REGRESSORS);
	xtx = X.transpose() * X;
	Eigen::MatrixXd inv_xtx = xtx.inverse();
	Eigen::MatrixXd xtxxt = inv_xtx * X.transpose();

	// Finally store regressors in ordinary arrays
	for (int i = 0; i < N; i++)
	{
		for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
		{
			h_X_GLM[i + r * N] = X(i,r);
		}

		h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS + 0) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS + 0);
		h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS + 1) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS + 1);
		h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS + 2) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS + 2);
		h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS + 3) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS + 3);
		h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS + 4) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS + 4);
		h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS + 5) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS + 5);

		h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 0) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 0);
		h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 1) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 1);
		h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 2) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 2);
		h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 3) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 3);

		for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
		{
			h_xtxxt_GLM[i + r * N] = (float)xtxxt(r,i);
		}

		h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS + 0) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS + 0,i);
		h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS + 1) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS + 1,i);
		h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS + 2) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS + 2,i);
		h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS + 3) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS + 3,i);
		h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS + 4) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS + 4,i);
		h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS + 5) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS + 5,i);

		h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 0) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 0,i);
		h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 1) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 1,i);
		h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 2) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 2,i);
		h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 3) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS + 3,i);
	}

	// Now update the contrast vectors also
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		Eigen::VectorXd Contrast(NUMBER_OF_TOTAL_GLM_REGRESSORS);

		// Copy contrasts for paradigm regressors
		for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
		{
			Contrast(r) = (double)h_Contrasts_In[c + r * NUMBER_OF_CONTRASTS];
			h_Contrasts[NUMBER_OF_TOTAL_GLM_REGRESSORS * c + r] = (float)Contrast(r);
		}

		// Set all other contrasts to 0
		for (int r = NUMBER_OF_GLM_REGRESSORS; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
		{
			Contrast(r) = 0.0;
			h_Contrasts[NUMBER_OF_TOTAL_GLM_REGRESSORS * c + r] = (float)Contrast(r);
		}

		Eigen::VectorXd scalar = Contrast.transpose() * inv_xtx * Contrast;
		h_ctxtxc_GLM[c] = scalar(0);
	}
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




