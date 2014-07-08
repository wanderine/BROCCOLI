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


#include <limits>
#include <Dense>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <vector>
#include <math.h>
#include <cfloat>

#include <limits.h>
#include <unistd.h>

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

void debugVolumeInfo(const char* name, int W, int H, int D, int T, float* volume)
{
	#ifndef NDEBUG
    printf("%s sizes: %d, %d, %d, %d => %d\n", name, W, H, D, T, W*H*D*T);

    float maxF = 0;
    int maxI = 0;

    for (int i = 0; i < W*H*D*T; ++i)
    {
        if (volume[i] > maxF)
        {
            maxI = i;
            maxF = volume[i];
      //      printf("New maximum: %g at %d of %d\n", maxF, maxI, W*H*D);
      //      printf("At positions (%d, %d, %d, %d)\n", maxI % W, (maxI/W)%H, (maxI/W/H)%D, (maxI/W/H/D));
        }
    }

    printf("%s maximum element: %d => %f\n", name, maxI, maxF);
    printf("%s maximum element at (%d, %d, %d, %d)\n", name, maxI % W, (maxI/W)%H, (maxI/W/H)%D, (maxI/W/H/D));
	#endif
}

void debugVolumeInfo(const char* name, int W, int H, int D, float* volume)
{
    debugVolumeInfo(name, W, H, D, 1, volume);
}

// Constructors

BROCCOLI_LIB::BROCCOLI_LIB()
{	
	OPENCL_INITIATED = 0;
	SetStartValues();
}

BROCCOLI_LIB::BROCCOLI_LIB(cl_uint platform, cl_uint device)
{
	SetStartValues();
	OPENCL_INITIATED = 0;
	SUCCESSFUL_INITIALIZATION = OpenCLInitiate(platform,device);
}

BROCCOLI_LIB::BROCCOLI_LIB(cl_uint platform, cl_uint device, int wrapper)
{
	SetStartValues();
	WRAPPER = wrapper;
	OPENCL_INITIATED = 0;
	SUCCESSFUL_INITIALIZATION = OpenCLInitiate(platform,device);
}

// Destructor
BROCCOLI_LIB::~BROCCOLI_LIB()
{
	OpenCLCleanup();
}

void BROCCOLI_LIB::SetDebug(bool debug)
{
	DEBUG = debug;
}

void BROCCOLI_LIB::SetPrint(bool print)
{
	PRINT = print;
}

void BROCCOLI_LIB::SetDoAllPermutations(bool doall)
{
	DO_ALL_PERMUTATIONS = doall;
}

void BROCCOLI_LIB::SetRawRegressors(bool raw)
{
	RAW_REGRESSORS = raw;
}

void BROCCOLI_LIB::SetDoSkullstrip(bool doskullstrip)
{
	DO_SKULLSTRIP = doskullstrip;
}

void BROCCOLI_LIB::SetWrapper(int wrapper)
{
	WRAPPER = wrapper;
}

//void BROCCOLI_LIB::SetSaveDisplacementField(bool save)
//{
//	DEBUG = debug;
//}

// Set some default values
void BROCCOLI_LIB::SetStartValues()
{
	INITIALIZATION_ERROR = "";
	OPENCL_ERROR = "";

	DEBUG = false;
	WRAPPER = -1;
	PRINT = true;
	DO_ALL_PERMUTATIONS = false;

	WRITE_INTERPOLATED_T1 = false;
	WRITE_ALIGNED_T1_MNI_LINEAR = false;
	WRITE_ALIGNED_T1_MNI_NONLINEAR = false;
	DO_SKULLSTRIP = false;

	WRITE_ALIGNED_EPI_T1 = false;
	WRITE_ALIGNED_EPI_MNI = false;

	WRITE_EPI_MASK = false;
	WRITE_SLICETIMING_CORRECTED = false;
	WRITE_MOTION_CORRECTED = false;
	WRITE_SMOOTHED = false;

	WRITE_ACTIVITY_EPI = false;
	WRITE_DESIGNMATRIX = false;
	WRITE_AR_ESTIMATES_EPI = false;
	WRITE_AR_ESTIMATES_MNI = false;

	WRITE_UNWHITENED_RESULTS = false;

	EPI_Smoothing_FWHM = 8.0f;
	AR_Smoothing_FWHM = 8.0f;

	programBinarySize = 0;
	writtenElements = 0;

	BETA_SPACE = EPI;

	SLICE_ORDER = DOWN;

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
	SIGNIFICANCE_LEVEL = 0.05f;
	SIGNIFICANCE_THRESHOLD = 0;

	IMAGE_REGISTRATION_FILTER_SIZE = 7;
	NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION = 10;
	NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION = 10;
	NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS = 30;
	
	SMOOTHING_FILTER_SIZE = 9;
	
	NUMBER_OF_DETRENDING_REGRESSORS = 4;
	NUMBER_OF_MOTION_REGRESSORS = 6;

	REGRESS_MOTION = 1;
	REGRESS_CONFOUNDS = 0;
	PERMUTE_FIRST_LEVEL = false;


	NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS = 12;

	TSIGMA = 5.0;
	ESIGMA = 5.0;
	DSIGMA = 5.0;

	convolution_time = 0.0;

	error = 0;

	NUMBER_OF_OPENCL_KERNELS = 76;

	commandQueue = NULL;
	program = NULL;
	context = NULL;

	// Reset kernels and errors
	for (int i = 0; i < NUMBER_OF_OPENCL_KERNELS; i++)
	{
		OpenCLKernels[i] = NULL;
		OpenCLRunKernelErrors[i] = 0;
		OpenCLCreateKernelErrors[i] = 0;
		OpenCLCreateBufferErrors[i] = 0;
	}

	// Reset create buffer errors, 33
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

	// Reset create kernel errors
	createKernelErrorNonseparableConvolution3DComplexThreeFilters = 0;
	createKernelErrorSeparableConvolutionRows = 0;
	createKernelErrorSeparableConvolutionColumns = 0;
	createKernelErrorSeparableConvolutionRods = 0;
	createKernelErrorSliceTimingCorrection = 0;
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
	createKernelErrorCalculateTensorComponents = 0;
	createKernelErrorCalculateTensorNorms = 0;
	createKernelErrorCalculateAMatricesAndHVectors = 0;
	createKernelErrorCalculateDisplacementUpdate = 0;
	createKernelErrorCalculateMagnitudes = 0;
	createKernelErrorCalculateColumnSums = 0;
	createKernelErrorCalculateRowSums = 0;
	createKernelErrorCalculateColumnMaxs = 0;
	createKernelErrorCalculateRowMaxs = 0;
	createKernelErrorCalculateMaxAtomic = 0;
	createKernelErrorThresholdVolume = 0;
	createKernelErrorInterpolateVolumeNearestLinear = 0;
	createKernelErrorInterpolateVolumeLinearLinear = 0;
	createKernelErrorInterpolateVolumeCubicLinear = 0;
	createKernelErrorInterpolateVolumeNearestNonLinear = 0;
	createKernelErrorInterpolateVolumeLinearNonLinear = 0;
	createKernelErrorInterpolateVolumeCubicNonLinear = 0;
	createKernelErrorRescaleVolumeLinear = 0;
	createKernelErrorRescaleVolumeCubic = 0;
	createKernelErrorRescaleVolumeNearest = 0;
	createKernelErrorCopyT1VolumeToMNI = 0;
	createKernelErrorCopyEPIVolumeToT1 = 0;
	createKernelErrorCopyVolumeToNew = 0;
	createKernelErrorMemset = 0;
	createKernelErrorMemsetInt = 0;
	createKernelErrorMemsetFloat2 = 0;
	createKernelErrorMultiplyVolume = 0;
	createKernelErrorMultiplyVolumes = 0;
	createKernelErrorMultiplyVolumesOverwrite = 0;
	createKernelErrorAddVolume = 0;
	createKernelErrorAddVolumes = 0;
	createKernelErrorAddVolumesOverwrite = 0;
	createKernelErrorRemoveMean = 0;
	createKernelErrorSetStartClusterIndices = 0;
	createKernelErrorClusterizeScan = 0;
	createKernelErrorClusterizeRelabel = 0;
	createKernelErrorCalculateClusterSizes = 0;
	createKernelErrorCalculateClusterMasses = 0;
	createKernelErrorCalculateLargestCluster = 0;
	createKernelErrorCalculateTFCEValues = 0;
	createKernelErrorCalculateBetaWeightsGLM = 0;
	createKernelErrorCalculateBetaWeightsGLMFirstLevel = 0;
	createKernelErrorCalculateGLMResiduals = 0;
	createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel = 0;
	createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevel = 0;
	createKernelErrorCalculateStatisticalMapsGLMTTest = 0;
	createKernelErrorCalculateStatisticalMapsGLMFTest = 0;
	createKernelErrorCalculateStatisticalMapsGLMBayesian = 0;
	createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelPermutation = 0;
	createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelPermutation = 0;
	createKernelErrorCalculateStatisticalMapsGLMTTestSecondLevelPermutation = 0;
	createKernelErrorCalculateStatisticalMapsGLMFTestSecondLevelPermutation = 0;
	createKernelErrorCalculateStatisticalMapsMeanSecondLevelPermutation = 0;
	createKernelErrorEstimateAR4Models = 0;
	createKernelErrorApplyWhiteningAR4 = 0;
	createKernelErrorGeneratePermutedVolumesFirstLevel = 0;
	createKernelErrorRemoveLinearFit = 0;
	createKernelErrorCalculatePermutationPValuesVoxelLevelInference = 0;
	createKernelErrorCalculatePermutationPValuesClusterLevelInference = 0;

	// Reset run kernel errors
	runKernelErrorNonseparableConvolution3DComplexThreeFilters = 0;
	runKernelErrorSeparableConvolutionRows = 0;
	runKernelErrorSeparableConvolutionColumns = 0;
	runKernelErrorSeparableConvolutionRods = 0;
	runKernelErrorSliceTimingCorrection = 0;
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
	runKernelErrorCalculateTensorComponents = 0;
	runKernelErrorCalculateTensorNorms = 0;
	runKernelErrorCalculateAMatricesAndHVectors = 0;
	runKernelErrorCalculateDisplacementUpdate = 0;
	runKernelErrorCalculateMagnitudes = 0;
	runKernelErrorCalculateColumnSums = 0;
	runKernelErrorCalculateRowSums = 0;
	runKernelErrorCalculateColumnMaxs = 0;
	runKernelErrorCalculateRowMaxs = 0;
	runKernelErrorCalculateMaxAtomic = 0;
	runKernelErrorThresholdVolume = 0;
	runKernelErrorInterpolateVolumeNearestLinear = 0;
	runKernelErrorInterpolateVolumeLinearLinear = 0;
	runKernelErrorInterpolateVolumeCubicLinear = 0;
	runKernelErrorInterpolateVolumeNearestNonLinear = 0;
	runKernelErrorInterpolateVolumeLinearNonLinear = 0;
	runKernelErrorInterpolateVolumeCubicNonLinear = 0;
	runKernelErrorRescaleVolumeLinear = 0;
	runKernelErrorRescaleVolumeCubic = 0;
	runKernelErrorRescaleVolumeNearest = 0;
	runKernelErrorCopyT1VolumeToMNI = 0;
	runKernelErrorCopyEPIVolumeToT1 = 0;
	runKernelErrorCopyVolumeToNew = 0;
	runKernelErrorMemset = 0;
	runKernelErrorMemsetInt = 0;
	runKernelErrorMemsetFloat2 = 0;
	runKernelErrorMultiplyVolume = 0;
	runKernelErrorMultiplyVolumes = 0;
	runKernelErrorMultiplyVolumesOverwrite = 0;
	runKernelErrorAddVolume = 0;
	runKernelErrorAddVolumes = 0;
	runKernelErrorAddVolumesOverwrite = 0;
	runKernelErrorRemoveMean = 0;
	runKernelErrorSetStartClusterIndices = 0;
	runKernelErrorClusterizeScan = 0;
	runKernelErrorClusterizeRelabel = 0;
	runKernelErrorCalculateClusterSizes = 0;
	runKernelErrorCalculateClusterMasses = 0;
	runKernelErrorCalculateLargestCluster = 0;
	runKernelErrorCalculateTFCEValues = 0;
	runKernelErrorCalculateBetaWeightsGLM = 0;
	runKernelErrorCalculateBetaWeightsGLMFirstLevel = 0;
	runKernelErrorCalculateGLMResiduals = 0;
	runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel = 0;
	runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel = 0;
	runKernelErrorCalculateStatisticalMapsGLMTTest = 0;
	runKernelErrorCalculateStatisticalMapsGLMFTest = 0;
	runKernelErrorCalculateStatisticalMapsGLMBayesian = 0;
	runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelPermutation = 0;
	runKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelPermutation = 0;
	runKernelErrorCalculateStatisticalMapsGLMTTestSecondLevelPermutation = 0;
	runKernelErrorCalculateStatisticalMapsGLMFTestSecondLevelPermutation = 0;
	runKernelErrorCalculateStatisticalMapsMeanSecondLevelPermutation = 0;
	runKernelErrorEstimateAR4Models = 0;
	runKernelErrorApplyWhiteningAR4 = 0;
	runKernelErrorGeneratePermutedVolumesFirstLevel = 0;
	runKernelErrorRemoveLinearFit = 0;
	runKernelErrorCalculatePermutationPValuesVoxelLevelInference = 0;
	runKernelErrorCalculatePermutationPValuesClusterLevelInference = 0;

	getPlatformIDsError = 0;
	getDeviceIDsError = 0;		
	createContextError = 0;
	getContextInfoError = 0;
	createCommandQueueError = 0;
	createProgramError = 0;
	buildProgramError = 0;
	getProgramBuildInfoError = 0;
}



bool BROCCOLI_LIB::GetOpenCLInitiated()
{
	return SUCCESSFUL_INITIALIZATION;
}

int BROCCOLI_LIB::GetNumberOfOpenCLKernels()
{
	return NUMBER_OF_OPENCL_KERNELS;
}

// Returns information about available platforms and devices
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

// Creates an OpenCL program from a binary file
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

// Saves a compiled program to a binary file
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

	// Allocate temporary memory
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

	// Loop over devices
	for (cl_uint i = 0; i < numDevices; i++)
	{
		// Only save the binary for the requested device
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
			// Remove spaces
			device_name.erase(std::remove (device_name.begin(), device_name.end(), ' '), device_name.end());
			filename.append(device_name);			
			filename.append(".bin");
			free(value);

			// Write binary to file
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

std::string BROCCOLI_LIB::Getexepath()
{
  char result[ PATH_MAX ];
  ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );
  return std::string( result, (count > 0) ? count : 0 );
}

/*
 * Old version of OpenCLInitiate, hard to read and lacks some error checking
 *

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

				// Get number of devices for selected platform
				cl_uint deviceIdCount = 0;
				getDeviceIDsError = clGetDeviceIDs(platformIds[OPENCL_PLATFORM], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceIdCount);
	
				if (getDeviceIDsError == SUCCESS)
				{
					// Get device IDs for selected platform
					std::vector<cl_device_id> deviceIds(deviceIdCount);
					getDeviceIDsError = clGetDeviceIDs(platformIds[OPENCL_PLATFORM], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), NULL);

					// Check if the requested device exists
					if ((OPENCL_DEVICE >= 0) &&  (OPENCL_DEVICE < deviceIdCount))
					{
						if (getDeviceIDsError == SUCCESS)
						{
							// Create context for selected device
							//context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), NULL, NULL, &createContextError);
							context = clCreateContext(contextProperties, 1, &deviceIds[OPENCL_DEVICE], NULL, NULL, &createContextError);

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

										binaryFilename = "broccoli_lib_kernel_unknown";
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
										else if (WRAPPER == BASH)
										{
											printf("\nUnsupported OpenCL vendor!\n\n");
										}

										// Create a command queue for the selected device
										commandQueue = clCreateCommandQueue(context, deviceIds[OPENCL_DEVICE], CL_QUEUE_PROFILING_ENABLE, &createCommandQueueError);

										if (createCommandQueueError == SUCCESS)
										{
											// Support for running functions from any folder
											//std::string kernelFileName = Getexepath();
											//kernelFileName.erase(kernelFileName.end()-16, kernelFileName.end());
											//kernelFileName.append(binaryFilename);

											std::string kernelFileName;
											kernelFileName.append(binaryFilename);

											// First try to compile from binary file for the selected device
											//createProgramError = CreateProgramFromBinary(program, context, deviceIds[OPENCL_DEVICE], kernelFileName);
											createProgramError = CreateProgramFromBinary(program, context, deviceIds[OPENCL_DEVICE], binaryFilename);
											//buildProgramError = clBuildProgram(program, deviceIdCount, deviceIds.data(), NULL, NULL, NULL);

											if (VENDOR == NVIDIA)
											{
												buildProgramError = clBuildProgram(program, 1, &deviceIds[OPENCL_DEVICE], "-cl-nv-verbose", NULL, NULL);
											}
											else
											{
												//buildProgramError = clBuildProgram(program, 1, &deviceIds[OPENCL_DEVICE], "-cl-opt-disable", NULL, NULL);
												buildProgramError = clBuildProgram(program, 1, &deviceIds[OPENCL_DEVICE], NULL, NULL, NULL);
											}

											// Otherwise compile from source code
											if (buildProgramError != SUCCESS)
											{
												// Read the kernel code from file
												//std::string kernelFileName = Getexepath();
												//kernelFileName.erase(kernelFileName.end()-16, kernelFileName.end());

												//std::string kernelFileName;
												//kernelFileName.append("broccoli_lib_kernel.cpp");
												//std::fstream kernelFile(kernelFileName.c_str(),std::ios::in);
												std::fstream kernelFile("broccoli_lib_kernel.cpp",std::ios::in);
												std::ostringstream oss;
												oss << kernelFile.rdbuf();
												std::string src = oss.str();
												const char *srcstr = src.c_str();

												// Create program and build the code for the selected device
												program = clCreateProgramWithSource(context, 1, (const char**)&srcstr , NULL, &createProgramError);
												//buildProgramError = clBuildProgram(program, deviceIdCount, deviceIds.data(), NULL, NULL, NULL);

												if (VENDOR == NVIDIA)
												{
													buildProgramError = clBuildProgram(program, 1, &deviceIds[OPENCL_DEVICE], "-cl-nv-verbose", NULL, NULL);
												}
												else
												{
													buildProgramError = clBuildProgram(program, 1, &deviceIds[OPENCL_DEVICE], NULL, NULL, NULL);
												}

												// If successful build, save to binary file
												if (buildProgramError == SUCCESS)
												{
													SaveProgramBinary(program,deviceIds[OPENCL_DEVICE],binaryFilename);
												}
											}

											// Always get build info

											// Get size of build info
											valueSize = 0;
											getProgramBuildInfoError = clGetProgramBuildInfo(program, deviceIds[OPENCL_DEVICE], CL_PROGRAM_BUILD_LOG, 0, NULL, &valueSize);

											// Get build info
											if (getProgramBuildInfoError == SUCCESS)
											{
												value = (char*)malloc(valueSize);
												getProgramBuildInfoError = clGetProgramBuildInfo(program, deviceIds[OPENCL_DEVICE], CL_PROGRAM_BUILD_LOG, valueSize, value, NULL);

												if (getProgramBuildInfoError == SUCCESS)
												{
													build_info.append(value);
												}
												else if (WRAPPER == BASH)
												{
													printf("\nUnable to get OpenCL build info! \n\n");
												}
												free(value);
											}
											else if (WRAPPER == BASH)
											{
												printf("\nUnable to get size of OpenCL build info!\n\n");
											}

											if (buildProgramError == SUCCESS)
											{
												// Create kernels

												// Convolution kernels
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

												OpenCLKernels[0] = NonseparableConvolution3DComplexThreeFiltersKernel;
												OpenCLKernels[1] = SeparableConvolutionRowsKernel;
												OpenCLKernels[2] = SeparableConvolutionColumnsKernel;
												OpenCLKernels[3] = SeparableConvolutionRodsKernel;

												SliceTimingCorrectionKernel = clCreateKernel(program,"SliceTimingCorrection",&createKernelErrorSliceTimingCorrection);

												OpenCLKernels[4] = SliceTimingCorrectionKernel;

												// Kernels for Linear registration
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

												OpenCLKernels[5] = CalculatePhaseDifferencesAndCertaintiesKernel;
												OpenCLKernels[6] = CalculatePhaseGradientsXKernel;
												OpenCLKernels[7] = CalculatePhaseGradientsYKernel;
												OpenCLKernels[8] = CalculatePhaseGradientsZKernel;
												OpenCLKernels[9] = CalculateAMatrixAndHVector2DValuesXKernel;
												OpenCLKernels[10] = CalculateAMatrixAndHVector2DValuesYKernel;
												OpenCLKernels[11] = CalculateAMatrixAndHVector2DValuesZKernel;
												OpenCLKernels[12] = CalculateAMatrix1DValuesKernel;
												OpenCLKernels[13] = CalculateHVector1DValuesKernel;
												OpenCLKernels[14] = CalculateAMatrixKernel;
												OpenCLKernels[15] = CalculateHVectorKernel;

												// Kernels for non-Linear registration
												CalculateTensorComponentsKernel = clCreateKernel(program, "CalculateTensorComponents", &createKernelErrorCalculateTensorComponents);
												CalculateTensorNormsKernel = clCreateKernel(program, "CalculateTensorNorms", &createKernelErrorCalculateTensorNorms);
												CalculateAMatricesAndHVectorsKernel = clCreateKernel(program, "CalculateAMatricesAndHVectors", &createKernelErrorCalculateAMatricesAndHVectors);
												CalculateDisplacementUpdateKernel = clCreateKernel(program, "CalculateDisplacementUpdate", &createKernelErrorCalculateDisplacementUpdate);
												AddLinearAndNonLinearDisplacementKernel = clCreateKernel(program, "AddLinearAndNonLinearDisplacement", &createKernelErrorAddLinearAndNonLinearDisplacement);

												OpenCLKernels[16] = CalculateTensorComponentsKernel;
												OpenCLKernels[17] = CalculateTensorNormsKernel;
												OpenCLKernels[18] = CalculateAMatricesAndHVectorsKernel;
												OpenCLKernels[19] = CalculateDisplacementUpdateKernel;
												OpenCLKernels[20] = AddLinearAndNonLinearDisplacementKernel;

												CalculateMagnitudesKernel = clCreateKernel(program,"CalculateMagnitudes",&createKernelErrorCalculateMagnitudes);
												CalculateColumnSumsKernel = clCreateKernel(program,"CalculateColumnSums",&createKernelErrorCalculateColumnSums);
												CalculateRowSumsKernel = clCreateKernel(program,"CalculateRowSums",&createKernelErrorCalculateRowSums);
												CalculateColumnMaxsKernel = clCreateKernel(program,"CalculateColumnMaxs",&createKernelErrorCalculateColumnMaxs);
												CalculateRowMaxsKernel = clCreateKernel(program,"CalculateRowMaxs",&createKernelErrorCalculateRowMaxs);
												CalculateMaxAtomicKernel = clCreateKernel(program,"CalculateMaxAtomic",&createKernelErrorCalculateMaxAtomic);
												ThresholdVolumeKernel = clCreateKernel(program,"ThresholdVolume",&createKernelErrorThresholdVolume);

												OpenCLKernels[21] = CalculateMagnitudesKernel;
												OpenCLKernels[22] = CalculateColumnSumsKernel;
												OpenCLKernels[23] = CalculateRowSumsKernel;
												OpenCLKernels[24] = CalculateColumnMaxsKernel;
												OpenCLKernels[25] = CalculateRowMaxsKernel;
												OpenCLKernels[26] = CalculateMaxAtomicKernel;
												OpenCLKernels[27] = ThresholdVolumeKernel;

												// Interpolation kernels
												InterpolateVolumeNearestLinearKernel = clCreateKernel(program,"InterpolateVolumeNearestLinear",&createKernelErrorInterpolateVolumeNearestLinear);
												InterpolateVolumeLinearLinearKernel = clCreateKernel(program,"InterpolateVolumeLinearLinear",&createKernelErrorInterpolateVolumeLinearLinear);
												InterpolateVolumeCubicLinearKernel = clCreateKernel(program,"InterpolateVolumeCubicLinear",&createKernelErrorInterpolateVolumeCubicLinear);
												InterpolateVolumeNearestNonLinearKernel = clCreateKernel(program,"InterpolateVolumeNearestNonLinear",&createKernelErrorInterpolateVolumeNearestNonLinear);
												InterpolateVolumeLinearNonLinearKernel = clCreateKernel(program,"InterpolateVolumeLinearNonLinear",&createKernelErrorInterpolateVolumeLinearNonLinear);
												InterpolateVolumeCubicNonLinearKernel = clCreateKernel(program,"InterpolateVolumeCubicNonLinear",&createKernelErrorInterpolateVolumeCubicNonLinear);

												OpenCLKernels[28] = InterpolateVolumeNearestLinearKernel;
												OpenCLKernels[29] = InterpolateVolumeLinearLinearKernel;
												OpenCLKernels[30] = InterpolateVolumeCubicLinearKernel;
												OpenCLKernels[31] = InterpolateVolumeNearestNonLinearKernel;
												OpenCLKernels[32] = InterpolateVolumeLinearNonLinearKernel;
												OpenCLKernels[33] = InterpolateVolumeCubicNonLinearKernel;

												RescaleVolumeLinearKernel = clCreateKernel(program,"RescaleVolumeLinear",&createKernelErrorRescaleVolumeLinear);
												RescaleVolumeCubicKernel = clCreateKernel(program,"RescaleVolumeCubic",&createKernelErrorRescaleVolumeCubic);
												RescaleVolumeNearestKernel = clCreateKernel(program,"RescaleVolumeNearest",&createKernelErrorRescaleVolumeNearest);

												OpenCLKernels[34] = RescaleVolumeLinearKernel;
												OpenCLKernels[35] = RescaleVolumeCubicKernel;
												OpenCLKernels[36] = RescaleVolumeNearestKernel;

												CopyT1VolumeToMNIKernel = clCreateKernel(program,"CopyT1VolumeToMNI",&createKernelErrorCopyT1VolumeToMNI);
												CopyEPIVolumeToT1Kernel = clCreateKernel(program,"CopyEPIVolumeToT1",&createKernelErrorCopyEPIVolumeToT1);
												CopyVolumeToNewKernel = clCreateKernel(program,"CopyVolumeToNew",&createKernelErrorCopyVolumeToNew);

												OpenCLKernels[37] = CopyT1VolumeToMNIKernel;
												OpenCLKernels[38] = CopyEPIVolumeToT1Kernel;
												OpenCLKernels[39] = CopyVolumeToNewKernel;

												// Help kernels
												MemsetKernel = clCreateKernel(program,"Memset",&createKernelErrorMemset);
												MemsetIntKernel = clCreateKernel(program,"MemsetInt",&createKernelErrorMemsetInt);
												MemsetFloat2Kernel = clCreateKernel(program,"MemsetFloat2",&createKernelErrorMemsetFloat2);
												MultiplyVolumeKernel = clCreateKernel(program,"MultiplyVolume",&createKernelErrorMultiplyVolume);
												MultiplyVolumesKernel = clCreateKernel(program,"MultiplyVolumes",&createKernelErrorMultiplyVolumes);
												MultiplyVolumesOverwriteKernel = clCreateKernel(program,"MultiplyVolumesOverwrite",&createKernelErrorMultiplyVolumesOverwrite);
												AddVolumeKernel = clCreateKernel(program,"AddVolume",&createKernelErrorAddVolume);
												AddVolumesKernel = clCreateKernel(program,"AddVolumes",&createKernelErrorAddVolumes);
												AddVolumesOverwriteKernel = clCreateKernel(program,"AddVolumesOverwrite",&createKernelErrorAddVolumesOverwrite);
												RemoveMeanKernel = clCreateKernel(program,"RemoveMean",&createKernelErrorRemoveMean);
												SetStartClusterIndicesKernel = clCreateKernel(program,"SetStartClusterIndicesKernel",&createKernelErrorSetStartClusterIndices);
												ClusterizeScanKernel = clCreateKernel(program,"ClusterizeScan",&createKernelErrorClusterizeScan);
												ClusterizeRelabelKernel = clCreateKernel(program,"ClusterizeRelabel",&createKernelErrorClusterizeRelabel);
												CalculateClusterSizesKernel = clCreateKernel(program,"CalculateClusterSizes",&createKernelErrorCalculateClusterSizes);
												CalculateLargestClusterKernel = clCreateKernel(program,"CalculateLargestCluster",&createKernelErrorCalculateLargestCluster);


												OpenCLKernels[40] = MemsetKernel;
												OpenCLKernels[41] = MemsetIntKernel;
												OpenCLKernels[42] = MemsetFloat2Kernel;
												OpenCLKernels[43] = MultiplyVolumeKernel;
												OpenCLKernels[44] = MultiplyVolumesKernel;
												OpenCLKernels[45] = MultiplyVolumesOverwriteKernel;
												OpenCLKernels[46] = AddVolumeKernel;
												OpenCLKernels[47] = AddVolumesKernel;
												OpenCLKernels[48] = AddVolumesOverwriteKernel;
												OpenCLKernels[49] = RemoveMeanKernel;
												OpenCLKernels[50] = SetStartClusterIndicesKernel;
												OpenCLKernels[51] = ClusterizeScanKernel;
												OpenCLKernels[52] = ClusterizeRelabelKernel;
												OpenCLKernels[53] = CalculateClusterSizesKernel;
												OpenCLKernels[54] = CalculateLargestClusterKernel;

												// Statistical kernels
												CalculateBetaWeightsGLMKernel = clCreateKernel(program,"CalculateBetaWeightsGLM",&createKernelErrorCalculateBetaWeightsGLM);
												CalculateBetaWeightsGLMFirstLevelKernel = clCreateKernel(program,"CalculateBetaWeightsGLMFirstLevel",&createKernelErrorCalculateBetaWeightsGLMFirstLevel);
												CalculateGLMResidualsKernel = clCreateKernel(program,"CalculateGLMResiduals",&createKernelErrorCalculateGLMResiduals);
												CalculateStatisticalMapsGLMTTestFirstLevelKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMTTestFirstLevel",&createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel);
												CalculateStatisticalMapsGLMFTestFirstLevelKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMFTestFirstLevel",&createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevel);
												CalculateStatisticalMapsGLMTTestKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMTTest",&createKernelErrorCalculateStatisticalMapsGLMTTest);
												CalculateStatisticalMapsGLMFTestKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMFTest",&createKernelErrorCalculateStatisticalMapsGLMFTest);
												CalculateStatisticalMapsGLMBayesianKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMBayesian",&createKernelErrorCalculateStatisticalMapsGLMBayesian);
												CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMTTestFirstLevelPermutation",&createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelPermutation);
												CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMFTestFirstLevelPermutation",&createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelPermutation);
												CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMTTestSecondLevelPermutation",&createKernelErrorCalculateStatisticalMapsGLMTTestSecondLevelPermutation);
												CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMFTestSecondLevelPermutation",&createKernelErrorCalculateStatisticalMapsGLMFTestSecondLevelPermutation);
												EstimateAR4ModelsKernel = clCreateKernel(program,"EstimateAR4Models",&createKernelErrorEstimateAR4Models);
												ApplyWhiteningAR4Kernel = clCreateKernel(program,"ApplyWhiteningAR4",&createKernelErrorApplyWhiteningAR4);
												GeneratePermutedVolumesFirstLevelKernel = clCreateKernel(program,"GeneratePermutedVolumesFirstLevel",&createKernelErrorGeneratePermutedVolumesFirstLevel);
												RemoveLinearFitKernel = clCreateKernel(program,"RemoveLinearFit",&createKernelErrorRemoveLinearFit);

												OpenCLKernels[55] = CalculateBetaWeightsGLMKernel;
												OpenCLKernels[56] = CalculateBetaWeightsGLMFirstLevelKernel;
												OpenCLKernels[57] = CalculateGLMResidualsKernel;
												OpenCLKernels[58] = CalculateStatisticalMapsGLMTTestFirstLevelKernel;
												OpenCLKernels[59] = CalculateStatisticalMapsGLMFTestFirstLevelKernel;
												OpenCLKernels[60] = CalculateStatisticalMapsGLMTTestKernel;
												OpenCLKernels[61] = CalculateStatisticalMapsGLMFTestKernel;
												OpenCLKernels[62] = CalculateStatisticalMapsGLMBayesianKernel;
												OpenCLKernels[63] = CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel;
												OpenCLKernels[64] = CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel;
												OpenCLKernels[65] = CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel;
												OpenCLKernels[66] = CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel;
												OpenCLKernels[67] = EstimateAR4ModelsKernel;
												OpenCLKernels[68] = ApplyWhiteningAR4Kernel;
												OpenCLKernels[69] = GeneratePermutedVolumesFirstLevelKernel;
												OpenCLKernels[70] = RemoveLinearFitKernel;

												OPENCL_INITIATED = 1;
											}
											else if (WRAPPER == BASH)
											{
												printf("\nUnable to build OpenCL program. Aborting! \n\n");
											}
										}
										else if (WRAPPER == BASH)
										{
											printf("\nUnable to create an OpenCL command queue. Aborting! \n\n");
										}
									}
									else if (WRAPPER == BASH)
									{
										printf("\nUnable to get OpenCL context info. Aborting! \n\n");
									}
									free(clDevices);
								}
								else if (WRAPPER == BASH)
								{
									printf("\nUnable to get size of OpenCL context info. Aborting! \n\n");
								}
							}
							else if (WRAPPER == BASH)
							{
								printf("\nUnable to create an OpenCL context. Aborting! \n\n");
							}
						}
						else if (WRAPPER == BASH)
						{
							printf("\nUnable to get OpenCL device id's for the specified platform. Aborting! \n\n");
						}
					}
					else if (WRAPPER == BASH)
					{
						printf("\nYou tried to use the invalid OpenCL device %i, valid devices for the selected platform are 0 <= device < %i. Aborting! \n\n",OPENCL_DEVICE,deviceIdCount);
					}
				}
				else if (WRAPPER == BASH)
				{
					printf("\nUnable to get number of OpenCL devices for the specified platform. Aborting! \n\n");
				}
			}
			else if (WRAPPER == BASH)
			{
				printf("\nYou tried to use the invalid OpenCL platform %i, valid platforms are 0 <= platform < %i. Aborting! \n\n",OPENCL_PLATFORM,platformIdCount);
			}
		}
		else if (WRAPPER == BASH)
		{
			printf("\nUnable to get OpenCL platform id's. Aborting! \n\n");
		}
	}
	else if (WRAPPER == BASH)
	{
		printf("\nUnable to get number of OpenCL platforms. Aborting! \n\n");
	}
}
*/

bool BROCCOLI_LIB::OpenCLInitiate(cl_uint OPENCL_PLATFORM, cl_uint OPENCL_DEVICE)
{
	char* value;
	size_t valueSize;
	cl_device_id *clDevices;

  	// Get number of platforms
	cl_uint platformIdCount = 0;
	error = clGetPlatformIDs (0, NULL, &platformIdCount);

	if (error != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable to get number of OpenCL platforms.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);
		return false;
	}

	// Check if the requested platform exists
	if (OPENCL_PLATFORM >= platformIdCount)
	{
		INITIALIZATION_ERROR = "You tried to use an invalid OpenCL platform";
		OPENCL_ERROR = "";
		return false;
	}

	// Get platform IDs
	std::vector<cl_platform_id> platformIds(platformIdCount);
	error = clGetPlatformIDs(platformIdCount, platformIds.data(), NULL);

	if (error != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable to get OpenCL platform id's.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);
		return false;
	}

	// Create context
	const cl_context_properties contextProperties [] =
	{
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[OPENCL_PLATFORM]), 0, 0
	};

	// Get number of devices for selected platform
	cl_uint deviceIdCount = 0;
	error = clGetDeviceIDs(platformIds[OPENCL_PLATFORM], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceIdCount);

	if (error != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable to get number of OpenCL devices for the specified platform.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);
		return false;
	}

	// Check if the requested device exists
	if (OPENCL_DEVICE >= deviceIdCount)
	{
		INITIALIZATION_ERROR = "You tried to use an invalid device for the specified OpenCL platform";
		OPENCL_ERROR = "";
		return false;
	}

	// Get device IDs for selected platform
	std::vector<cl_device_id> deviceIds(deviceIdCount);
	error = clGetDeviceIDs(platformIds[OPENCL_PLATFORM], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), NULL);

	if (error != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable to get device IDs for the specified platform.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);
		return false;
	}

	// Create context for selected device
	context = clCreateContext(contextProperties, 1, &deviceIds[OPENCL_DEVICE], NULL, NULL, &error);

	if (error != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable to create an OpenCL context.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);
		return false;
	}

	// Get size of context info
	error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &valueSize);

	if (error != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable to get size of OpenCL context info.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);
		return false;
	}

	// Get context info
	clDevices = (cl_device_id *) malloc(valueSize);
	error = clGetContextInfo(context, CL_CONTEXT_DEVICES, valueSize, clDevices, NULL);

	if (error != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable to get OpenCL context info.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);
		free(clDevices);
		return false;
	}

	// Get size of name of current platform
	error = clGetPlatformInfo(platformIds[OPENCL_PLATFORM], CL_PLATFORM_NAME, 0, NULL, &valueSize);

	if (error != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable to get size of platform name.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);
		return false;
	}

	// Get name of current platform
	value = (char*) malloc(valueSize);
	error = clGetPlatformInfo(platformIds[OPENCL_PLATFORM], CL_PLATFORM_NAME, valueSize, value, NULL);

	if (error != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable to get name of specified OpenCL platform.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);
		free(value);
		return false;
	}

	// Convert to string
	std::string vendor_string(value);
	free(value);

	// Figure out the vendor
	size_t nvidiaPos = vendor_string.find("NVIDIA");
	size_t intelPos = vendor_string.find("Intel");
	size_t amdPos = vendor_string.find("AMD");
	size_t applePos = vendor_string.find("Apple");

	binaryFilename = "";
	if (nvidiaPos != std::string::npos)
	{
		VENDOR = NVIDIA;
		binaryFilename = "broccoli_lib_kernel_Nvidia";
	}
	else if (intelPos != std::string::npos)
	{
		VENDOR = INTEL;
		binaryFilename = "broccoli_lib_kernel_Intel";
	}
	else if (amdPos != std::string::npos)
	{
		VENDOR = AMD;
		binaryFilename = "broccoli_lib_kernel_AMD";
	}
	else if (applePos != std::string::npos)
	{
		VENDOR = APPLE;
		binaryFilename = "broccoli_lib_kernel_Apple";
	}
	else
	{
		INITIALIZATION_ERROR = "Unsupported OpenCL vendor.";
		OPENCL_ERROR = "";
		return false;
	}

	// Create a command queue for the selected device
	commandQueue = clCreateCommandQueue(context, deviceIds[OPENCL_DEVICE], CL_QUEUE_PROFILING_ENABLE, &error);

	if (error != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable to create a command queue for the selected device and platform.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);
		return false;
	}

	// Support for running BROCCOLI from any directory
	//std::string kernelFileName = Getexepath();
	//kernelFileName.erase(kernelFileName.end()-16, kernelFileName.end());
	//kernelFileName.append(binaryFilename);

	std::string kernelFileName;
	kernelFileName.append(binaryFilename);

	// First try to compile from binary file for the selected device
	error = CreateProgramFromBinary(program, context, deviceIds[OPENCL_DEVICE], binaryFilename);
	//error = CreateProgramFromBinary(program, context, deviceIds[OPENCL_DEVICE], kernelFileName);

	// Build program for selected device
	if (VENDOR == NVIDIA)
	{
		error = clBuildProgram(program, 1, &deviceIds[OPENCL_DEVICE], "-cl-nv-verbose", NULL, NULL);
	}
	else
	{
		error = clBuildProgram(program, 1, &deviceIds[OPENCL_DEVICE], NULL, NULL, NULL);
	}

	// Otherwise compile from source code
	if (error != SUCCESS)
	{
		// Check if kernel file exists
		std::ifstream file("broccoli_lib_kernel.cpp");
		if ( !file.good() )
		{
			INITIALIZATION_ERROR = "Unable to open broccoli_lib_kernel.cpp.";
			OPENCL_ERROR = "";
			return false;
		}

		// Support for running BROCCOLI from any directory
		//std::string kernelFileName = Getexepath();
		//kernelFileName.erase(kernelFileName.end()-16, kernelFileName.end());
		//kernelFileName.append("broccoli_lib_kernel.cpp");
		//std::fstream kernelFile(kernelFileName.c_str(),std::ios::in);

		// Read the kernel code from file
		std::fstream kernelFile("broccoli_lib_kernel.cpp",std::ios::in);

		std::ostringstream oss;
		oss << kernelFile.rdbuf();
		std::string src = oss.str();
		const char *srcstr = src.c_str();

		// Create program and build the code for the selected device
		program = clCreateProgramWithSource(context, 1, (const char**)&srcstr , NULL, &error);

		if (error != SUCCESS)
		{
			INITIALIZATION_ERROR = "Unable to create program with source.";
			OPENCL_ERROR = GetOpenCLErrorMessage(error);
			return false;
		}

		if (VENDOR == NVIDIA)
		{
			buildProgramError = clBuildProgram(program, 1, &deviceIds[OPENCL_DEVICE], "-cl-nv-verbose", NULL, NULL);
		}
		else
		{
			buildProgramError = clBuildProgram(program, 1, &deviceIds[OPENCL_DEVICE], NULL, NULL, NULL);
		}

		// If successful build, save to binary file
		if (buildProgramError == SUCCESS)
		{
			SaveProgramBinary(program,deviceIds[OPENCL_DEVICE],binaryFilename);
		}
	}

	// Always get build info

	// Get size of build info
	valueSize = 0;
	error = clGetProgramBuildInfo(program, deviceIds[OPENCL_DEVICE], CL_PROGRAM_BUILD_LOG, 0, NULL, &valueSize);

	if (error != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable to get size of build info.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);
		return false;
	}

	value = (char*)malloc(valueSize);
	error = clGetProgramBuildInfo(program, deviceIds[OPENCL_DEVICE], CL_PROGRAM_BUILD_LOG, valueSize, value, NULL);

	if (error != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable to get build info.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);
		free(value);
		return false;
	}

	build_info.append(value);
	free(value);

	if (buildProgramError != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable build OpenCL program from binary or source.";
		OPENCL_ERROR = GetOpenCLErrorMessage(buildProgramError);
		return false;
	}


	// Create kernels

	// Convolution kernels
	if ( (VENDOR == NVIDIA) || (VENDOR == INTEL) || (VENDOR == APPLE) )
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

	OpenCLKernels[0] = NonseparableConvolution3DComplexThreeFiltersKernel;
	OpenCLKernels[1] = SeparableConvolutionRowsKernel;
	OpenCLKernels[2] = SeparableConvolutionColumnsKernel;
	OpenCLKernels[3] = SeparableConvolutionRodsKernel;

	SliceTimingCorrectionKernel = clCreateKernel(program,"SliceTimingCorrection",&createKernelErrorSliceTimingCorrection);

	OpenCLKernels[4] = SliceTimingCorrectionKernel;

	// Kernels for Linear registration
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

	OpenCLKernels[5] = CalculatePhaseDifferencesAndCertaintiesKernel;
	OpenCLKernels[6] = CalculatePhaseGradientsXKernel;
	OpenCLKernels[7] = CalculatePhaseGradientsYKernel;
	OpenCLKernels[8] = CalculatePhaseGradientsZKernel;
	OpenCLKernels[9] = CalculateAMatrixAndHVector2DValuesXKernel;
	OpenCLKernels[10] = CalculateAMatrixAndHVector2DValuesYKernel;
	OpenCLKernels[11] = CalculateAMatrixAndHVector2DValuesZKernel;
	OpenCLKernels[12] = CalculateAMatrix1DValuesKernel;
	OpenCLKernels[13] = CalculateHVector1DValuesKernel;
	OpenCLKernels[14] = CalculateAMatrixKernel;
	OpenCLKernels[15] = CalculateHVectorKernel;

	// Kernels for non-Linear registration
	CalculateTensorComponentsKernel = clCreateKernel(program, "CalculateTensorComponents", &createKernelErrorCalculateTensorComponents);
	CalculateTensorNormsKernel = clCreateKernel(program, "CalculateTensorNorms", &createKernelErrorCalculateTensorNorms);
	CalculateAMatricesAndHVectorsKernel = clCreateKernel(program, "CalculateAMatricesAndHVectors", &createKernelErrorCalculateAMatricesAndHVectors);
	CalculateDisplacementUpdateKernel = clCreateKernel(program, "CalculateDisplacementUpdate", &createKernelErrorCalculateDisplacementUpdate);
	AddLinearAndNonLinearDisplacementKernel = clCreateKernel(program, "AddLinearAndNonLinearDisplacement", &createKernelErrorAddLinearAndNonLinearDisplacement);

	OpenCLKernels[16] = CalculateTensorComponentsKernel;
	OpenCLKernels[17] = CalculateTensorNormsKernel;
	OpenCLKernels[18] = CalculateAMatricesAndHVectorsKernel;
	OpenCLKernels[19] = CalculateDisplacementUpdateKernel;
	OpenCLKernels[20] = AddLinearAndNonLinearDisplacementKernel;

	CalculateMagnitudesKernel = clCreateKernel(program,"CalculateMagnitudes",&createKernelErrorCalculateMagnitudes);
	CalculateColumnSumsKernel = clCreateKernel(program,"CalculateColumnSums",&createKernelErrorCalculateColumnSums);
	CalculateRowSumsKernel = clCreateKernel(program,"CalculateRowSums",&createKernelErrorCalculateRowSums);
	CalculateColumnMaxsKernel = clCreateKernel(program,"CalculateColumnMaxs",&createKernelErrorCalculateColumnMaxs);
	CalculateRowMaxsKernel = clCreateKernel(program,"CalculateRowMaxs",&createKernelErrorCalculateRowMaxs);
	CalculateMaxAtomicKernel = clCreateKernel(program,"CalculateMaxAtomic",&createKernelErrorCalculateMaxAtomic);
	ThresholdVolumeKernel = clCreateKernel(program,"ThresholdVolume",&createKernelErrorThresholdVolume);

	OpenCLKernels[21] = CalculateMagnitudesKernel;
	OpenCLKernels[22] = CalculateColumnSumsKernel;
	OpenCLKernels[23] = CalculateRowSumsKernel;
	OpenCLKernels[24] = CalculateColumnMaxsKernel;
	OpenCLKernels[25] = CalculateRowMaxsKernel;
	OpenCLKernels[26] = CalculateMaxAtomicKernel;
	OpenCLKernels[27] = ThresholdVolumeKernel;

	// Interpolation kernels
	InterpolateVolumeNearestLinearKernel = clCreateKernel(program,"InterpolateVolumeNearestLinear",&createKernelErrorInterpolateVolumeNearestLinear);
	InterpolateVolumeLinearLinearKernel = clCreateKernel(program,"InterpolateVolumeLinearLinear",&createKernelErrorInterpolateVolumeLinearLinear);
	InterpolateVolumeCubicLinearKernel = clCreateKernel(program,"InterpolateVolumeCubicLinear",&createKernelErrorInterpolateVolumeCubicLinear);
	InterpolateVolumeNearestNonLinearKernel = clCreateKernel(program,"InterpolateVolumeNearestNonLinear",&createKernelErrorInterpolateVolumeNearestNonLinear);
	InterpolateVolumeLinearNonLinearKernel = clCreateKernel(program,"InterpolateVolumeLinearNonLinear",&createKernelErrorInterpolateVolumeLinearNonLinear);
	InterpolateVolumeCubicNonLinearKernel = clCreateKernel(program,"InterpolateVolumeCubicNonLinear",&createKernelErrorInterpolateVolumeCubicNonLinear);

	OpenCLKernels[28] = InterpolateVolumeNearestLinearKernel;
	OpenCLKernels[29] = InterpolateVolumeLinearLinearKernel;
	OpenCLKernels[30] = InterpolateVolumeCubicLinearKernel;
	OpenCLKernels[31] = InterpolateVolumeNearestNonLinearKernel;
	OpenCLKernels[32] = InterpolateVolumeLinearNonLinearKernel;
	OpenCLKernels[33] = InterpolateVolumeCubicNonLinearKernel;

	RescaleVolumeLinearKernel = clCreateKernel(program,"RescaleVolumeLinear",&createKernelErrorRescaleVolumeLinear);
	RescaleVolumeCubicKernel = clCreateKernel(program,"RescaleVolumeCubic",&createKernelErrorRescaleVolumeCubic);
	RescaleVolumeNearestKernel = clCreateKernel(program,"RescaleVolumeNearest",&createKernelErrorRescaleVolumeNearest);

	OpenCLKernels[34] = RescaleVolumeLinearKernel;
	OpenCLKernels[35] = RescaleVolumeCubicKernel;
	OpenCLKernels[36] = RescaleVolumeNearestKernel;

	CopyT1VolumeToMNIKernel = clCreateKernel(program,"CopyT1VolumeToMNI",&createKernelErrorCopyT1VolumeToMNI);
	CopyEPIVolumeToT1Kernel = clCreateKernel(program,"CopyEPIVolumeToT1",&createKernelErrorCopyEPIVolumeToT1);
	CopyVolumeToNewKernel = clCreateKernel(program,"CopyVolumeToNew",&createKernelErrorCopyVolumeToNew);

	OpenCLKernels[37] = CopyT1VolumeToMNIKernel;
	OpenCLKernels[38] = CopyEPIVolumeToT1Kernel;
	OpenCLKernels[39] = CopyVolumeToNewKernel;

	// Help kernels
	MemsetKernel = clCreateKernel(program,"Memset",&createKernelErrorMemset);
	MemsetIntKernel = clCreateKernel(program,"MemsetInt",&createKernelErrorMemsetInt);
	MemsetFloat2Kernel = clCreateKernel(program,"MemsetFloat2",&createKernelErrorMemsetFloat2);
	MultiplyVolumeKernel = clCreateKernel(program,"MultiplyVolume",&createKernelErrorMultiplyVolume);
	MultiplyVolumesKernel = clCreateKernel(program,"MultiplyVolumes",&createKernelErrorMultiplyVolumes);
	MultiplyVolumesOverwriteKernel = clCreateKernel(program,"MultiplyVolumesOverwrite",&createKernelErrorMultiplyVolumesOverwrite);
	AddVolumeKernel = clCreateKernel(program,"AddVolume",&createKernelErrorAddVolume);
	AddVolumesKernel = clCreateKernel(program,"AddVolumes",&createKernelErrorAddVolumes);
	AddVolumesOverwriteKernel = clCreateKernel(program,"AddVolumesOverwrite",&createKernelErrorAddVolumesOverwrite);
	RemoveMeanKernel = clCreateKernel(program,"RemoveMean",&createKernelErrorRemoveMean);
	SetStartClusterIndicesKernel = clCreateKernel(program,"SetStartClusterIndicesKernel",&createKernelErrorSetStartClusterIndices);
	ClusterizeScanKernel = clCreateKernel(program,"ClusterizeScan",&createKernelErrorClusterizeScan);
	ClusterizeRelabelKernel = clCreateKernel(program,"ClusterizeRelabel",&createKernelErrorClusterizeRelabel);
	CalculateClusterSizesKernel = clCreateKernel(program,"CalculateClusterSizes",&createKernelErrorCalculateClusterSizes);
	CalculateClusterMassesKernel = clCreateKernel(program,"CalculateClusterMasses",&createKernelErrorCalculateClusterMasses);
	CalculateLargestClusterKernel = clCreateKernel(program,"CalculateLargestCluster",&createKernelErrorCalculateLargestCluster);
	CalculateTFCEValuesKernel = clCreateKernel(program,"CalculateTFCEValues",&createKernelErrorCalculateTFCEValues);

	OpenCLKernels[40] = MemsetKernel;
	OpenCLKernels[41] = MemsetIntKernel;
	OpenCLKernels[42] = MemsetFloat2Kernel;
	OpenCLKernels[43] = MultiplyVolumeKernel;
	OpenCLKernels[44] = MultiplyVolumesKernel;
	OpenCLKernels[45] = MultiplyVolumesOverwriteKernel;
	OpenCLKernels[46] = AddVolumeKernel;
	OpenCLKernels[47] = AddVolumesKernel;
	OpenCLKernels[48] = AddVolumesOverwriteKernel;
	OpenCLKernels[49] = RemoveMeanKernel;
	OpenCLKernels[50] = SetStartClusterIndicesKernel;
	OpenCLKernels[51] = ClusterizeScanKernel;
	OpenCLKernels[52] = ClusterizeRelabelKernel;
	OpenCLKernels[53] = CalculateClusterSizesKernel;
	OpenCLKernels[54] = CalculateClusterMassesKernel;
	OpenCLKernels[55] = CalculateLargestClusterKernel;
	OpenCLKernels[56] = CalculateTFCEValuesKernel;

	// Statistical kernels
	CalculateBetaWeightsGLMKernel = clCreateKernel(program,"CalculateBetaWeightsGLM",&createKernelErrorCalculateBetaWeightsGLM);
	CalculateBetaWeightsGLMFirstLevelKernel = clCreateKernel(program,"CalculateBetaWeightsGLMFirstLevel",&createKernelErrorCalculateBetaWeightsGLMFirstLevel);
	CalculateGLMResidualsKernel = clCreateKernel(program,"CalculateGLMResiduals",&createKernelErrorCalculateGLMResiduals);
	CalculateStatisticalMapsGLMTTestFirstLevelKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMTTestFirstLevel",&createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel);
	CalculateStatisticalMapsGLMFTestFirstLevelKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMFTestFirstLevel",&createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevel);
	CalculateStatisticalMapsGLMTTestKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMTTest",&createKernelErrorCalculateStatisticalMapsGLMTTest);
	CalculateStatisticalMapsGLMFTestKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMFTest",&createKernelErrorCalculateStatisticalMapsGLMFTest);
	CalculateStatisticalMapsGLMBayesianKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMBayesian",&createKernelErrorCalculateStatisticalMapsGLMBayesian);
	CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMTTestFirstLevelPermutation",&createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelPermutation);
	CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMFTestFirstLevelPermutation",&createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelPermutation);
	CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMTTestSecondLevelPermutation",&createKernelErrorCalculateStatisticalMapsGLMTTestSecondLevelPermutation);
	CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel = clCreateKernel(program,"CalculateStatisticalMapsGLMFTestSecondLevelPermutation",&createKernelErrorCalculateStatisticalMapsGLMFTestSecondLevelPermutation);
	CalculateStatisticalMapsMeanSecondLevelPermutationKernel = clCreateKernel(program,"CalculateStatisticalMapsMeanSecondLevelPermutation",&createKernelErrorCalculateStatisticalMapsMeanSecondLevelPermutation);
	EstimateAR4ModelsKernel = clCreateKernel(program,"EstimateAR4Models",&createKernelErrorEstimateAR4Models);
	ApplyWhiteningAR4Kernel = clCreateKernel(program,"ApplyWhiteningAR4",&createKernelErrorApplyWhiteningAR4);
	GeneratePermutedVolumesFirstLevelKernel = clCreateKernel(program,"GeneratePermutedVolumesFirstLevel",&createKernelErrorGeneratePermutedVolumesFirstLevel);
	RemoveLinearFitKernel = clCreateKernel(program,"RemoveLinearFit",&createKernelErrorRemoveLinearFit);
	CalculatePermutationPValuesVoxelLevelInferenceKernel = clCreateKernel(program,"CalculatePermutationPValuesVoxelLevelInference",&createKernelErrorCalculatePermutationPValuesVoxelLevelInference);
	CalculatePermutationPValuesClusterLevelInferenceKernel = clCreateKernel(program,"CalculatePermutationPValuesClusterLevelInference",&createKernelErrorCalculatePermutationPValuesClusterLevelInference);

	OpenCLKernels[57] = CalculateBetaWeightsGLMKernel;
	OpenCLKernels[58] = CalculateBetaWeightsGLMFirstLevelKernel;
	OpenCLKernels[59] = CalculateGLMResidualsKernel;
	OpenCLKernels[60] = CalculateStatisticalMapsGLMTTestFirstLevelKernel;
	OpenCLKernels[61] = CalculateStatisticalMapsGLMFTestFirstLevelKernel;
	OpenCLKernels[62] = CalculateStatisticalMapsGLMTTestKernel;
	OpenCLKernels[63] = CalculateStatisticalMapsGLMFTestKernel;
	OpenCLKernels[64] = CalculateStatisticalMapsGLMBayesianKernel;
	OpenCLKernels[65] = CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel;
	OpenCLKernels[66] = CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel;
	OpenCLKernels[67] = CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel;
	OpenCLKernels[68] = CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel;
	OpenCLKernels[69] = CalculateStatisticalMapsMeanSecondLevelPermutationKernel;
	OpenCLKernels[70] = EstimateAR4ModelsKernel;
	OpenCLKernels[71] = ApplyWhiteningAR4Kernel;
	OpenCLKernels[72] = GeneratePermutedVolumesFirstLevelKernel;
	OpenCLKernels[73] = RemoveLinearFitKernel;
	OpenCLKernels[74] = CalculatePermutationPValuesVoxelLevelInferenceKernel;
	OpenCLKernels[75] = CalculatePermutationPValuesClusterLevelInferenceKernel;

	OPENCL_INITIATED = 1;

	// Set all create kernel errors into an array
	GetOpenCLCreateKernelErrors();

	// Check all create kernel errors
	bool ALL_KERNELS_OK = true;
	for (int i = 0; i < NUMBER_OF_OPENCL_KERNELS; i++)
	{
		if (OpenCLCreateKernelErrors[i] != SUCCESS)
		{
			ALL_KERNELS_OK = false;
		}
	}

	if (!ALL_KERNELS_OK)
	{
		INITIALIZATION_ERROR = "One or several kernels were not created.";
		OPENCL_ERROR = "";
		return false;
	}
	else
	{
		INITIALIZATION_ERROR = "";
		OPENCL_ERROR = "";
		return true;
	}
}


// Cleans up all the OpenCL variables when the BROCCOLI instance is destroyed
void BROCCOLI_LIB::OpenCLCleanup()
{
	if (OPENCL_INITIATED == 1)
	{
		// Release all kernels
		for (int k = 0; k < NUMBER_OF_OPENCL_KERNELS; k++)
		{
			cl_kernel kernel = OpenCLKernels[k];

			if (kernel != NULL)
			{
				clReleaseKernel(kernel);
			}
		}

		// Release program, command queue and context
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
	if ( (VENDOR == NVIDIA) || (VENDOR == INTEL) || (VENDOR == APPLE) )
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
	if ( (VENDOR == NVIDIA) || (VENDOR == INTEL) || (VENDOR == APPLE) )
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

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesClusterize(int DATA_W, int DATA_H, int DATA_D)
{
	localWorkSizeClusterize[0] = 16;
	localWorkSizeClusterize[1] = 16;
	localWorkSizeClusterize[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeClusterize[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeClusterize[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeClusterize[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeClusterize[0] = xBlocks * localWorkSizeClusterize[0];
	globalWorkSizeClusterize[1] = yBlocks * localWorkSizeClusterize[1];
	globalWorkSizeClusterize[2] = zBlocks * localWorkSizeClusterize[2];
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesInterpolateVolume(int DATA_W, int DATA_H, int DATA_D)
{
	localWorkSizeInterpolateVolume[0] = 16;
	localWorkSizeInterpolateVolume[1] = 16;
	localWorkSizeInterpolateVolume[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeInterpolateVolume[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeInterpolateVolume[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeInterpolateVolume[2]);

	globalWorkSizeInterpolateVolume[0] = xBlocks * localWorkSizeInterpolateVolume[0];
	globalWorkSizeInterpolateVolume[1] = yBlocks * localWorkSizeInterpolateVolume[1];
	globalWorkSizeInterpolateVolume[2] = zBlocks * localWorkSizeInterpolateVolume[2];
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
	localWorkSizeCalculateMaxAtomic[0] = 32;
	localWorkSizeCalculateMaxAtomic[1] = 8;
	localWorkSizeCalculateMaxAtomic[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculateMaxAtomic[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculateMaxAtomic[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateMaxAtomic[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculateMaxAtomic[0] = xBlocks * localWorkSizeCalculateMaxAtomic[0];
	globalWorkSizeCalculateMaxAtomic[1] = yBlocks * localWorkSizeCalculateMaxAtomic[1];
	globalWorkSizeCalculateMaxAtomic[2] = zBlocks * localWorkSizeCalculateMaxAtomic[2];


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
	localWorkSizeCalculateBetaWeightsGLM[0] = 32;
	localWorkSizeCalculateBetaWeightsGLM[1] = 8;
	localWorkSizeCalculateBetaWeightsGLM[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculateBetaWeightsGLM[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculateBetaWeightsGLM[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateBetaWeightsGLM[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculateBetaWeightsGLM[0] = xBlocks * localWorkSizeCalculateBetaWeightsGLM[0];
	globalWorkSizeCalculateBetaWeightsGLM[1] = yBlocks * localWorkSizeCalculateBetaWeightsGLM[1];
	globalWorkSizeCalculateBetaWeightsGLM[2] = zBlocks * localWorkSizeCalculateBetaWeightsGLM[2];

	localWorkSizeCalculateStatisticalMapsGLM[0] = 32;
	localWorkSizeCalculateStatisticalMapsGLM[1] = 8;
	localWorkSizeCalculateStatisticalMapsGLM[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculateStatisticalMapsGLM[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculateStatisticalMapsGLM[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateStatisticalMapsGLM[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculateStatisticalMapsGLM[0] = xBlocks * localWorkSizeCalculateStatisticalMapsGLM[0];
	globalWorkSizeCalculateStatisticalMapsGLM[1] = yBlocks * localWorkSizeCalculateStatisticalMapsGLM[1];
	globalWorkSizeCalculateStatisticalMapsGLM[2] = zBlocks * localWorkSizeCalculateStatisticalMapsGLM[2];


	localWorkSizeEstimateAR4Models[0] = 32;
	localWorkSizeEstimateAR4Models[1] = 8;
	localWorkSizeEstimateAR4Models[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeEstimateAR4Models[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeEstimateAR4Models[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeEstimateAR4Models[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeEstimateAR4Models[0] = xBlocks * localWorkSizeEstimateAR4Models[0];
	globalWorkSizeEstimateAR4Models[1] = yBlocks * localWorkSizeEstimateAR4Models[1];
	globalWorkSizeEstimateAR4Models[2] = zBlocks * localWorkSizeEstimateAR4Models[2];

	localWorkSizeApplyWhiteningAR4[0] = 32;
	localWorkSizeApplyWhiteningAR4[1] = 8;
	localWorkSizeApplyWhiteningAR4[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeApplyWhiteningAR4[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeApplyWhiteningAR4[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeApplyWhiteningAR4[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeApplyWhiteningAR4[0] = xBlocks * localWorkSizeApplyWhiteningAR4[0];
	globalWorkSizeApplyWhiteningAR4[1] = yBlocks * localWorkSizeApplyWhiteningAR4[1];
	globalWorkSizeApplyWhiteningAR4[2] = zBlocks * localWorkSizeApplyWhiteningAR4[2];

	localWorkSizeGeneratePermutedVolumesFirstLevel[0] = 32;
	localWorkSizeGeneratePermutedVolumesFirstLevel[1] = 8;
	localWorkSizeGeneratePermutedVolumesFirstLevel[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeGeneratePermutedVolumesFirstLevel[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeGeneratePermutedVolumesFirstLevel[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeGeneratePermutedVolumesFirstLevel[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeGeneratePermutedVolumesFirstLevel[0] = xBlocks * localWorkSizeGeneratePermutedVolumesFirstLevel[0];
	globalWorkSizeGeneratePermutedVolumesFirstLevel[1] = yBlocks * localWorkSizeGeneratePermutedVolumesFirstLevel[1];
	globalWorkSizeGeneratePermutedVolumesFirstLevel[2] = zBlocks * localWorkSizeGeneratePermutedVolumesFirstLevel[2];

	localWorkSizeRemoveLinearFit[0] = 32;
	localWorkSizeRemoveLinearFit[1] = 8;
	localWorkSizeRemoveLinearFit[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeRemoveLinearFit[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeRemoveLinearFit[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeRemoveLinearFit[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeRemoveLinearFit[0] = xBlocks * localWorkSizeRemoveLinearFit[0];
	globalWorkSizeRemoveLinearFit[1] = yBlocks * localWorkSizeRemoveLinearFit[1];
	globalWorkSizeRemoveLinearFit[2] = zBlocks * localWorkSizeRemoveLinearFit[2];

	localWorkSizeCalculatePermutationPValues[0] = 32;
	localWorkSizeCalculatePermutationPValues[1] = 8;
	localWorkSizeCalculatePermutationPValues[2] = 1;

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculatePermutationPValues[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculatePermutationPValues[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculatePermutationPValues[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculatePermutationPValues[0] = xBlocks * localWorkSizeCalculatePermutationPValues[0];
	globalWorkSizeCalculatePermutationPValues[1] = yBlocks * localWorkSizeCalculatePermutationPValues[1];
	globalWorkSizeCalculatePermutationPValues[2] = zBlocks * localWorkSizeCalculatePermutationPValues[2];
}








// Set functions for GUI / Wrappers


void BROCCOLI_LIB::SetInputfMRIVolumes(float* data)
{
	h_fMRI_Volumes = data;
    //    debugVolumeInfo("fMRI", EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, data);
}

void BROCCOLI_LIB::SetInputEPIVolume(float* data)
{
	h_EPI_Volume = data;
    //    debugVolumeInfo("EPI", EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, data);
}

void BROCCOLI_LIB::SetInputT1Volume(float* data)
{
	h_T1_Volume = data;
    //    debugVolumeInfo("T1", T1_DATA_W, T1_DATA_H, T1_DATA_D, data);
}

void BROCCOLI_LIB::SetInputMNIVolume(float* data)
{
	h_MNI_Volume = data;
    //    debugVolumeInfo("MNI", MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, data);
}

void BROCCOLI_LIB::SetInputMNIBrainVolume(float* data)
{
	h_MNI_Brain_Volume = data;
    //    debugVolumeInfo("MNI Brain", MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, data);
}


void BROCCOLI_LIB::SetInputMNIBrainMask(float* data)
{
	h_MNI_Brain_Mask = data;
    //    debugVolumeInfo("MNI Brain Mask", MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, data);
}

void BROCCOLI_LIB::SetInputFirstLevelResults(float* data)
{
	h_First_Level_Results = data;
}

void BROCCOLI_LIB::SetNumberOfSubjects(int N)
{
	NUMBER_OF_SUBJECTS = N;
}

void BROCCOLI_LIB::SetMask(float* data)
{
	h_Mask = data;
}

void BROCCOLI_LIB::SetEPIMask(float* data)
{
	h_EPI_Mask = data;
}

void BROCCOLI_LIB::SetSmoothedEPIMask(float* data)
{
	h_Smoothed_EPI_Mask = data;
}


void BROCCOLI_LIB::SetTemporalDerivatives(int N)
{
	USE_TEMPORAL_DERIVATIVES = N;
}

void BROCCOLI_LIB::SetRegressConfounds(int R)
{
	REGRESS_CONFOUNDS = R;
}

void BROCCOLI_LIB::SetPermuteFirstLevel(bool value)
{
	PERMUTE_FIRST_LEVEL = value;
}

void BROCCOLI_LIB::SetConfoundRegressors(float* regressors)
{
	h_X_GLM_Confounds = regressors;
}

void BROCCOLI_LIB::SetRegressMotion(int R)
{
	REGRESS_MOTION = R;
}

void BROCCOLI_LIB::SetNumberOfConfoundRegressors(int N)
{
	NUMBER_OF_CONFOUND_REGRESSORS = N;
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

void BROCCOLI_LIB::SetPermutationMatrix(unsigned short int* matrix)
{
	h_Permutation_Matrix = matrix;
}

void BROCCOLI_LIB::SetSignMatrix(float* matrix)
{
	h_Sign_Matrix = matrix;
}

void BROCCOLI_LIB::SetPermutationFileUsage(bool use)
{
	USE_PERMUTATION_FILE = use;
}

void BROCCOLI_LIB::SetOutputDesignMatrix(float* data1, float* data2)
{
	h_X_GLM_Out = data1;
	h_xtxxt_GLM_Out = data2;
}

void BROCCOLI_LIB::SetOutputWhitenedModels(float* whitened_models)
{
	h_Whitened_Models = whitened_models;
}

void BROCCOLI_LIB::SetOutputClusterIndices(int* data)
{
	h_Cluster_Indices = data;
}

void BROCCOLI_LIB::SetOutputLargestCluster(int* size)
{
	h_Largest_Cluster = size;
}

void BROCCOLI_LIB::SetOutputEPIMask(float* data)
{
	h_EPI_Mask = data;
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

void BROCCOLI_LIB::SetNumberOfMCMCIterations(int N)
{
	NUMBER_OF_MCMC_ITERATIONS = N;
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

//void BROCCOLI_LIB::SetLinearImageRegistrationFilters(cl_float2* qf1, cl_float2* qf2, cl_float2* qf3)
void BROCCOLI_LIB::SetLinearImageRegistrationFilters(float* qf1r, float* qf1i, float* qf2r, float* qf2i, float* qf3r, float* qf3i)
{
	h_Quadrature_Filter_1_Linear_Registration_Real = qf1r;
	h_Quadrature_Filter_1_Linear_Registration_Imag = qf1i;

	h_Quadrature_Filter_2_Linear_Registration_Real = qf2r;
	h_Quadrature_Filter_2_Linear_Registration_Imag = qf2i;

	h_Quadrature_Filter_3_Linear_Registration_Real = qf3r;
	h_Quadrature_Filter_3_Linear_Registration_Imag = qf3i;
}

//void BROCCOLI_LIB::SetNonLinearImageRegistrationFilters(cl_float2* qf1, cl_float2* qf2, cl_float2* qf3, cl_float2* qf4, cl_float2* qf5, cl_float2* qf6)
void BROCCOLI_LIB::SetNonLinearImageRegistrationFilters(float* qf1r, float* qf1i, float* qf2r, float* qf2i, float* qf3r, float* qf3i, float* qf4r, float* qf4i, float* qf5r, float* qf5i, float* qf6r, float* qf6i)
{
	h_Quadrature_Filter_1_NonLinear_Registration_Real = qf1r;
	h_Quadrature_Filter_1_NonLinear_Registration_Imag = qf1i;
	h_Quadrature_Filter_2_NonLinear_Registration_Real = qf2r;
	h_Quadrature_Filter_2_NonLinear_Registration_Imag = qf2i;
	h_Quadrature_Filter_3_NonLinear_Registration_Real = qf3r;
	h_Quadrature_Filter_3_NonLinear_Registration_Imag = qf3i;
	h_Quadrature_Filter_4_NonLinear_Registration_Real = qf4r;
	h_Quadrature_Filter_4_NonLinear_Registration_Imag = qf4i;
	h_Quadrature_Filter_5_NonLinear_Registration_Real = qf5r;
	h_Quadrature_Filter_5_NonLinear_Registration_Imag = qf5i;
	h_Quadrature_Filter_6_NonLinear_Registration_Real = qf6r;
	h_Quadrature_Filter_6_NonLinear_Registration_Imag = qf6i;
}

void SetNonLinearImageRegistrationFilters(float* qf1r, float* qf1i, float* qf2r, float* qf2i, float* q3r, float* q3i, float* qf4r, float* qf4i, float* qf5r, float* qf5i, float* q6r, float* q6i);

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

void BROCCOLI_LIB::SetNumberOfIterationsForLinearImageRegistration(int N)
{
	NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION = N;
}

void BROCCOLI_LIB::SetNumberOfIterationsForNonLinearImageRegistration(int N)
{
	NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION = N;
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

void BROCCOLI_LIB::SetStatisticalTest(int test)
{
	STATISTICAL_TEST = test;
}


void BROCCOLI_LIB::SetInferenceMode(int mode)
{
	INFERENCE_MODE = mode;
}

void BROCCOLI_LIB::SetClusterDefiningThreshold(float threshold)
{
	CLUSTER_DEFINING_THRESHOLD = threshold;
}

void BROCCOLI_LIB::SetInterpolationMode(int mode)
{
	INTERPOLATION_MODE = mode;
}

void BROCCOLI_LIB::SetTsigma(float sigma)
{
	TSIGMA = sigma;
}

void BROCCOLI_LIB::SetEsigma(float sigma)
{
	ESIGMA = sigma;
}

void BROCCOLI_LIB::SetDsigma(float sigma)
{
	DSIGMA = sigma;
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

void BROCCOLI_LIB::SetSignificanceLevel(float value)
{
	SIGNIFICANCE_LEVEL = value;
}

void BROCCOLI_LIB::SetSaveInterpolatedT1(bool value)
{
	WRITE_INTERPOLATED_T1 = value;
}

void BROCCOLI_LIB::SetSaveAlignedT1MNILinear(bool value)
{
	WRITE_ALIGNED_T1_MNI_LINEAR = value;
}

void BROCCOLI_LIB::SetSaveAlignedT1MNINonLinear(bool value)
{
	WRITE_ALIGNED_T1_MNI_NONLINEAR = value;
}

void BROCCOLI_LIB::SetSaveAlignedEPIT1(bool value)
{
	WRITE_ALIGNED_EPI_T1 = value;
}

void BROCCOLI_LIB::SetSaveAlignedEPIMNI(bool value)
{
	WRITE_ALIGNED_EPI_MNI = value;
}

void BROCCOLI_LIB::SetSaveEPIMask(bool value)
{
	WRITE_EPI_MASK = value;
}

void BROCCOLI_LIB::SetSaveSliceTimingCorrected(bool value)
{
	WRITE_SLICETIMING_CORRECTED = value;
}

void BROCCOLI_LIB::SetSaveMotionCorrected(bool value)
{
	WRITE_MOTION_CORRECTED = value;
}

void BROCCOLI_LIB::SetSaveSmoothed(bool value)
{
	WRITE_SMOOTHED = value;
}

void BROCCOLI_LIB::SetSaveActivityEPI(bool value)
{
	WRITE_ACTIVITY_EPI = value;
}

void BROCCOLI_LIB::SetSaveActivityT1(bool value)
{
	WRITE_ACTIVITY_T1 = value;
}

void BROCCOLI_LIB::SetSaveDesignMatrix(bool value)
{
	WRITE_DESIGNMATRIX = value;
}

void BROCCOLI_LIB::SetSaveAREstimatesEPI(bool value)
{
	WRITE_AR_ESTIMATES_EPI = value;
}

void BROCCOLI_LIB::SetSaveAREstimatesT1(bool value)
{
	WRITE_AR_ESTIMATES_T1 = value;
}

void BROCCOLI_LIB::SetSaveAREstimatesMNI(bool value)
{
	WRITE_AR_ESTIMATES_MNI = value;
}

void BROCCOLI_LIB::SetSaveUnwhitenedResults(bool value)
{
	WRITE_UNWHITENED_RESULTS = value;
}

void BROCCOLI_LIB::SetSmoothingType(int type)
{
	SMOOTHING_TYPE = type;
}


void BROCCOLI_LIB::SetEPISmoothingAmount(float mm)
{
	EPI_Smoothing_FWHM = mm;
}

void BROCCOLI_LIB::SetARSmoothingAmount(float mm)
{
	AR_Smoothing_FWHM = mm;
}






void BROCCOLI_LIB::SetOutputBetaVolumesEPI(float* data)
{
	h_Beta_Volumes_EPI = data;
}

void BROCCOLI_LIB::SetOutputBetaVolumesT1(float* data)
{
	h_Beta_Volumes_T1 = data;
}

void BROCCOLI_LIB::SetOutputBetaVolumesMNI(float* data)
{
	h_Beta_Volumes_MNI = data;
}

void BROCCOLI_LIB::SetOutputBetaVolumesNoWhiteningEPI(float* data)
{
	h_Beta_Volumes_No_Whitening_EPI = data;
}

void BROCCOLI_LIB::SetOutputBetaVolumesNoWhiteningT1(float* data)
{
	h_Beta_Volumes_No_Whitening_T1 = data;
}

void BROCCOLI_LIB::SetOutputBetaVolumesNoWhiteningMNI(float* data)
{
	h_Beta_Volumes_No_Whitening_MNI = data;
}

void BROCCOLI_LIB::SetOutputContrastVolumesEPI(float* data)
{
	h_Contrast_Volumes_EPI = data;
}

void BROCCOLI_LIB::SetOutputContrastVolumesT1(float* data)
{
	h_Contrast_Volumes_T1 = data;
}

void BROCCOLI_LIB::SetOutputContrastVolumesMNI(float* data)
{
	h_Contrast_Volumes_MNI = data;
}

void BROCCOLI_LIB::SetOutputContrastVolumesNoWhiteningEPI(float* data)
{
	h_Contrast_Volumes_No_Whitening_EPI = data;
}

void BROCCOLI_LIB::SetOutputContrastVolumesNoWhiteningT1(float* data)
{
	h_Contrast_Volumes_No_Whitening_T1 = data;
}

void BROCCOLI_LIB::SetOutputContrastVolumesNoWhiteningMNI(float* data)
{
	h_Contrast_Volumes_No_Whitening_MNI = data;
}

void BROCCOLI_LIB::SetOutputStatisticalMapsEPI(float* data)
{
	h_Statistical_Maps_EPI = data;
}

void BROCCOLI_LIB::SetOutputStatisticalMapsT1(float* data)
{
	h_Statistical_Maps_T1 = data;
}

void BROCCOLI_LIB::SetOutputStatisticalMapsMNI(float* data)
{
	h_Statistical_Maps_MNI = data;
}

void BROCCOLI_LIB::SetOutputStatisticalMapsNoWhiteningEPI(float* data)
{
	h_Statistical_Maps_No_Whitening_EPI = data;
}

void BROCCOLI_LIB::SetOutputStatisticalMapsNoWhiteningT1(float* data)
{
	h_Statistical_Maps_No_Whitening_T1 = data;
}

void BROCCOLI_LIB::SetOutputStatisticalMapsNoWhiteningMNI(float* data)
{
	h_Statistical_Maps_No_Whitening_MNI = data;
}

void BROCCOLI_LIB::SetOutputResiduals(float* data)
{
	h_Residuals = data;
}

void BROCCOLI_LIB::SetOutputResidualVariances(float* data)
{
	h_Residual_Variances = data;
}

void BROCCOLI_LIB::SetOutputPValuesEPI(float* data)
{
	h_P_Values_EPI = data;
}

void BROCCOLI_LIB::SetOutputPValuesT1(float* data)
{
	h_P_Values_T1 = data;
}

void BROCCOLI_LIB::SetOutputPValuesMNI(float* data)
{
	h_P_Values_MNI = data;
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

void BROCCOLI_LIB::SetOutputQuadratureFilterResponses(cl_float2* qfr1, cl_float2* qfr2, cl_float2* qfr3)
{
	h_Quadrature_Filter_Response_1 = qfr1;
	h_Quadrature_Filter_Response_2 = qfr2;
	h_Quadrature_Filter_Response_3 = qfr3;
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

void BROCCOLI_LIB::SetOutputAlignedT1VolumeLinear(float* aligned)
{
	h_Aligned_T1_Volume_Linear = aligned;
}

void BROCCOLI_LIB::SetOutputAlignedT1VolumeNonLinear(float* aligned)
{
	h_Aligned_T1_Volume_NonLinear = aligned;
}

void BROCCOLI_LIB::SetOutputAlignedEPIVolumeT1(float* aligned)
{
	h_Aligned_EPI_Volume_T1 = aligned;
}

void BROCCOLI_LIB::SetOutputAlignedEPIVolumeMNI(float* aligned)
{
	h_Aligned_EPI_Volume_MNI = aligned;
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


void BROCCOLI_LIB::SetOutputSliceTimingCorrectedfMRIVolumes(float* slice_timing_corrected)
{
	h_Slice_Timing_Corrected_fMRI_Volumes = slice_timing_corrected;
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

void BROCCOLI_LIB::SetOutputPermutedFirstLevelResults(float* permuted)
{
	h_Permuted_First_Level_Results = permuted;
}

void BROCCOLI_LIB::SetOutputDownsampledVolume(float* downsampled)
{
	h_Downsampled_Volume = downsampled;
}

void BROCCOLI_LIB::SetOutputAREstimatesEPI(float* ar1, float* ar2, float* ar3, float* ar4)
{
	h_AR1_Estimates_EPI = ar1;
	h_AR2_Estimates_EPI = ar2;
	h_AR3_Estimates_EPI = ar3;
	h_AR4_Estimates_EPI = ar4;
}

void BROCCOLI_LIB::SetOutputAREstimatesT1(float* ar1, float* ar2, float* ar3, float* ar4)
{
	h_AR1_Estimates_T1 = ar1;
	h_AR2_Estimates_T1 = ar2;
	h_AR3_Estimates_T1 = ar3;
	h_AR4_Estimates_T1 = ar4;
}

void BROCCOLI_LIB::SetOutputAREstimatesMNI(float* ar1, float* ar2, float* ar3, float* ar4)
{
	h_AR1_Estimates_MNI = ar1;
	h_AR2_Estimates_MNI = ar2;
	h_AR3_Estimates_MNI = ar3;
	h_AR4_Estimates_MNI = ar4;
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

const char* BROCCOLI_LIB::GetOpenCLErrorMessage(int error)
{
	switch (error)
	{
		case 0:
			return "CL_SUCCESS";
			break;
		case -1:
			return "CL_DEVICE_NOT_FOUND";
			break;
		case -2:
			return "CL_DEVICE_NOT_AVAILABLE";
			break;
		case -3:
			return "CL_COMPILER_NOT_AVAILABLE";
			break;
		case -4:
			return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
			break;
		case -5:
			return "CL_OUT_OF_RESOURCES";
			break;
		case -6:
			return "CL_OUT_OF_HOST_MEMORY";
			break;
		case -7:
			return "CL_PROFILING_INFO_NOT_AVAILABLE";
			break;
		case -8:
			return "CL_MEM_COPY_OVERLAP";
			break;
		case -9:
			return "CL_IMAGE_FORMAT_MISMATCH";
			break;
		case -10:
			return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
			break;
		case -11:
			return "CL_BUILD_PROGRAM_FAILURE";
			break;
		case -12:
			return "CL_MAP_FAILURE";
			break;

		case -30:
			return "CL_INVALID_VALUE";
			break;
		case -31:
			return "CL_INVALID_DEVICE_TYPE";
			break;
		case -32:
			return "CL_INVALID_PLATFORM";
			break;
		case -33:
			return "CL_INVALID_DEVICE";
			break;
		case -34:
			return "CL_INVALID_CONTEXT";
			break;
		case -35:
			return "CL_INVALID_QUEUE_PROPERTIES";
			break;
		case -36:
			return "CL_INVALID_COMMAND_QUEUE";
			break;
		case -37:
			return "CL_INVALID_HOST_PTR";
			break;
		case -38:
			return "CL_INVALID_MEM_OBJECT";
			break;
		case -39:
			return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
			break;
		case -40:
			return "CL_INVALID_IMAGE_SIZE";
			break;
		case -41:
			return "CL_INVALID_SAMPLER";
			break;
		case -42:
			return "CL_INVALID_BINARY";
			break;
		case -43:
			return "CL_INVALID_BUILD_OPTIONS";
			break;
		case -44:
			return "CL_INVALID_PROGRAM";
			break;
		case -45:
			return "CL_INVALID_PROGRAM_EXECUTABLE";
			break;
		case -46:
			return "CL_INVALID_KERNEL_NAME";
			break;
		case -47:
			return "CL_INVALID_KERNEL_DEFINITION";
			break;
		case -48:
			return "CL_INVALID_KERNEL";
			break;
		case -49:
			return "CL_INVALID_ARG_INDEX";
			break;
		case -50:
			return "CL_INVALID_ARG_VALUE";
			break;
		case -51:
			return "CL_INVALID_ARG_SIZE";
			break;
		case -52:
			return "CL_INVALID_KERNEL_ARGS";
			break;
		case -53:
			return "CL_INVALID_WORK_DIMENSION";
			break;
		case -54:
			return "CL_INVALID_WORK_GROUP_SIZE";
			break;
		case -55:
			return "CL_INVALID_WORK_ITEM_SIZE";
			break;
		case -56:
			return "CL_INVALID_GLOBAL_OFFSET";
			break;
		case -57:
			return "CL_INVALID_EVENT_WAIT_LIST";
			break;
		case -58:
			return "CL_INVALID_EVENT";
			break;
		case -59:
			return "CL_INVALID_OPERATION";
			break;
		case -60:
			return "CL_INVALID_GL_OBJECT";
			break;
		case -61:
			return "CL_INVALID_BUFFER_SIZE";
			break;
		case -62:
			return "CL_INVALID_MIP_LEVEL";
			break;
		case -63:
			return "CL_INVALID_GLOBAL_WORK_SIZE";
			break;

		default:
			return "Unrecognized OpenCL error message";
	}



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

const char* BROCCOLI_LIB::GetOpenCLInitializationError()
{
	return INITIALIZATION_ERROR;
}

const char* BROCCOLI_LIB::GetOpenCLError()
{
	return OPENCL_ERROR;
}


int* BROCCOLI_LIB::GetOpenCLCreateKernelErrors()
{
	OpenCLCreateKernelErrors[0] = createKernelErrorNonseparableConvolution3DComplexThreeFilters;
	OpenCLCreateKernelErrors[1] = createKernelErrorSeparableConvolutionRows;
	OpenCLCreateKernelErrors[2] = createKernelErrorSeparableConvolutionColumns;
	OpenCLCreateKernelErrors[3] = createKernelErrorSeparableConvolutionRods;
	OpenCLCreateKernelErrors[4] = createKernelErrorSliceTimingCorrection;
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
	OpenCLCreateKernelErrors[16] = createKernelErrorCalculateTensorComponents;
	OpenCLCreateKernelErrors[17] = createKernelErrorCalculateTensorNorms;
	OpenCLCreateKernelErrors[18] = createKernelErrorCalculateAMatricesAndHVectors;
	OpenCLCreateKernelErrors[19] = createKernelErrorCalculateDisplacementUpdate;
	OpenCLCreateKernelErrors[20] = createKernelErrorAddLinearAndNonLinearDisplacement;
	OpenCLCreateKernelErrors[21] = createKernelErrorCalculateMagnitudes;
	OpenCLCreateKernelErrors[22] = createKernelErrorCalculateColumnSums;
	OpenCLCreateKernelErrors[23] = createKernelErrorCalculateRowSums;
	OpenCLCreateKernelErrors[24] = createKernelErrorCalculateColumnMaxs;
	OpenCLCreateKernelErrors[25] = createKernelErrorCalculateRowMaxs;
	OpenCLCreateKernelErrors[26] = createKernelErrorCalculateMaxAtomic;
	OpenCLCreateKernelErrors[27] = createKernelErrorThresholdVolume;
	OpenCLCreateKernelErrors[28] = createKernelErrorInterpolateVolumeNearestLinear;
	OpenCLCreateKernelErrors[29] = createKernelErrorInterpolateVolumeLinearLinear;
	OpenCLCreateKernelErrors[30] = createKernelErrorInterpolateVolumeCubicLinear;
	OpenCLCreateKernelErrors[31] = createKernelErrorInterpolateVolumeNearestNonLinear;
	OpenCLCreateKernelErrors[32] = createKernelErrorInterpolateVolumeLinearNonLinear;
	OpenCLCreateKernelErrors[33] = createKernelErrorInterpolateVolumeCubicNonLinear;
	OpenCLCreateKernelErrors[34] = createKernelErrorRescaleVolumeLinear;
	OpenCLCreateKernelErrors[35] = createKernelErrorRescaleVolumeCubic;
	OpenCLCreateKernelErrors[36] = createKernelErrorRescaleVolumeNearest;
	OpenCLCreateKernelErrors[37] = createKernelErrorCopyT1VolumeToMNI;
	OpenCLCreateKernelErrors[38] = createKernelErrorCopyEPIVolumeToT1;
	OpenCLCreateKernelErrors[39] = createKernelErrorCopyVolumeToNew;
	OpenCLCreateKernelErrors[40] = createKernelErrorMemset;
	OpenCLCreateKernelErrors[41] = createKernelErrorMemsetInt;
	OpenCLCreateKernelErrors[42] = createKernelErrorMemsetFloat2;
	OpenCLCreateKernelErrors[43] = createKernelErrorMultiplyVolume;
	OpenCLCreateKernelErrors[44] = createKernelErrorMultiplyVolumes;
	OpenCLCreateKernelErrors[45] = createKernelErrorMultiplyVolumesOverwrite;
	OpenCLCreateKernelErrors[46] = createKernelErrorAddVolume;
	OpenCLCreateKernelErrors[47] = createKernelErrorAddVolumes;
	OpenCLCreateKernelErrors[48] = createKernelErrorAddVolumesOverwrite;
	OpenCLCreateKernelErrors[49] = createKernelErrorRemoveMean;
	OpenCLCreateKernelErrors[50] = createKernelErrorSetStartClusterIndices;
	OpenCLCreateKernelErrors[51] = createKernelErrorClusterizeScan;
	OpenCLCreateKernelErrors[52] = createKernelErrorClusterizeRelabel;
	OpenCLCreateKernelErrors[53] = createKernelErrorCalculateClusterSizes;
	OpenCLCreateKernelErrors[54] = createKernelErrorCalculateClusterMasses;
	OpenCLCreateKernelErrors[55] = createKernelErrorCalculateLargestCluster;
	OpenCLCreateKernelErrors[56] = createKernelErrorCalculateTFCEValues;
	OpenCLCreateKernelErrors[57] = createKernelErrorCalculateBetaWeightsGLM;
	OpenCLCreateKernelErrors[58] = createKernelErrorCalculateBetaWeightsGLMFirstLevel;
	OpenCLCreateKernelErrors[59] = createKernelErrorCalculateGLMResiduals;
	OpenCLCreateKernelErrors[60] = createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel;
	OpenCLCreateKernelErrors[61] = createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevel;
	OpenCLCreateKernelErrors[62] = createKernelErrorCalculateStatisticalMapsGLMTTest;
	OpenCLCreateKernelErrors[63] = createKernelErrorCalculateStatisticalMapsGLMFTest;
	OpenCLCreateKernelErrors[64] = createKernelErrorCalculateStatisticalMapsGLMBayesian;
	OpenCLCreateKernelErrors[65] = createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelPermutation;
	OpenCLCreateKernelErrors[66] = createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelPermutation;
	OpenCLCreateKernelErrors[67] = createKernelErrorCalculateStatisticalMapsGLMTTestSecondLevelPermutation;
	OpenCLCreateKernelErrors[68] = createKernelErrorCalculateStatisticalMapsGLMFTestSecondLevelPermutation;
	OpenCLCreateKernelErrors[69] = createKernelErrorCalculateStatisticalMapsMeanSecondLevelPermutation;
	OpenCLCreateKernelErrors[70] = createKernelErrorEstimateAR4Models;
	OpenCLCreateKernelErrors[71] = createKernelErrorApplyWhiteningAR4;
	OpenCLCreateKernelErrors[72] = createKernelErrorGeneratePermutedVolumesFirstLevel;
	OpenCLCreateKernelErrors[73] = createKernelErrorRemoveLinearFit;
	OpenCLCreateKernelErrors[74] = createKernelErrorCalculatePermutationPValuesVoxelLevelInference;
	OpenCLCreateKernelErrors[75] = createKernelErrorCalculatePermutationPValuesClusterLevelInference;

	return OpenCLCreateKernelErrors;
}

int* BROCCOLI_LIB::GetOpenCLRunKernelErrors()
{
	OpenCLRunKernelErrors[0] = runKernelErrorNonseparableConvolution3DComplexThreeFilters;
	OpenCLRunKernelErrors[1] = runKernelErrorSeparableConvolutionRows;
	OpenCLRunKernelErrors[2] = runKernelErrorSeparableConvolutionColumns;
	OpenCLRunKernelErrors[3] = runKernelErrorSeparableConvolutionRods;
	OpenCLRunKernelErrors[4] = runKernelErrorSliceTimingCorrection;
	OpenCLRunKernelErrors[5] = runKernelErrorCalculatePhaseDifferencesAndCertainties;
	OpenCLRunKernelErrors[6] = runKernelErrorCalculatePhaseGradientsX;
	OpenCLRunKernelErrors[7] = runKernelErrorCalculatePhaseGradientsY;
	OpenCLRunKernelErrors[8] = runKernelErrorCalculatePhaseGradientsZ;
	OpenCLRunKernelErrors[9] = runKernelErrorCalculateAMatrixAndHVector2DValuesX;
	OpenCLRunKernelErrors[10] = runKernelErrorCalculateAMatrixAndHVector2DValuesY;
	OpenCLRunKernelErrors[11] = runKernelErrorCalculateAMatrixAndHVector2DValuesZ;
	OpenCLRunKernelErrors[12] = runKernelErrorCalculateAMatrix1DValues;
	OpenCLRunKernelErrors[13] = runKernelErrorCalculateHVector1DValues;
	OpenCLRunKernelErrors[14] = runKernelErrorCalculateAMatrix;
	OpenCLRunKernelErrors[15] = runKernelErrorCalculateHVector;
	OpenCLRunKernelErrors[16] = runKernelErrorCalculateTensorComponents;
	OpenCLRunKernelErrors[17] = runKernelErrorCalculateTensorNorms;
	OpenCLRunKernelErrors[18] = runKernelErrorCalculateAMatricesAndHVectors;
	OpenCLRunKernelErrors[19] = runKernelErrorCalculateDisplacementUpdate;
	OpenCLRunKernelErrors[20] = runKernelErrorAddLinearAndNonLinearDisplacement;
	OpenCLRunKernelErrors[21] = runKernelErrorCalculateMagnitudes;
	OpenCLRunKernelErrors[22] = runKernelErrorCalculateColumnSums;
	OpenCLRunKernelErrors[23] = runKernelErrorCalculateRowSums;
	OpenCLRunKernelErrors[24] = runKernelErrorCalculateColumnMaxs;
	OpenCLRunKernelErrors[25] = runKernelErrorCalculateRowMaxs;
	OpenCLRunKernelErrors[26] = runKernelErrorCalculateMaxAtomic;
	OpenCLRunKernelErrors[27] = runKernelErrorThresholdVolume;
	OpenCLRunKernelErrors[28] = runKernelErrorInterpolateVolumeNearestLinear;
	OpenCLRunKernelErrors[29] = runKernelErrorInterpolateVolumeLinearLinear;
	OpenCLRunKernelErrors[30] = runKernelErrorInterpolateVolumeCubicLinear;
	OpenCLRunKernelErrors[31] = runKernelErrorInterpolateVolumeNearestNonLinear;
	OpenCLRunKernelErrors[32] = runKernelErrorInterpolateVolumeLinearNonLinear;
	OpenCLRunKernelErrors[33] = runKernelErrorInterpolateVolumeCubicNonLinear;
	OpenCLRunKernelErrors[34] = runKernelErrorRescaleVolumeLinear;
	OpenCLRunKernelErrors[35] = runKernelErrorRescaleVolumeCubic;
	OpenCLRunKernelErrors[36] = runKernelErrorRescaleVolumeNearest;
	OpenCLRunKernelErrors[37] = runKernelErrorCopyT1VolumeToMNI;
	OpenCLRunKernelErrors[38] = runKernelErrorCopyEPIVolumeToT1;
	OpenCLRunKernelErrors[39] = runKernelErrorCopyVolumeToNew;
	OpenCLRunKernelErrors[40] = runKernelErrorMemset;
	OpenCLRunKernelErrors[41] = runKernelErrorMemsetInt;
	OpenCLRunKernelErrors[42] = runKernelErrorMemsetFloat2;
	OpenCLRunKernelErrors[43] = runKernelErrorMultiplyVolume;
	OpenCLRunKernelErrors[44] = runKernelErrorMultiplyVolumes;
	OpenCLRunKernelErrors[45] = runKernelErrorMultiplyVolumesOverwrite;
	OpenCLRunKernelErrors[46] = runKernelErrorAddVolume;
	OpenCLRunKernelErrors[47] = runKernelErrorAddVolumes;
	OpenCLRunKernelErrors[48] = runKernelErrorAddVolumesOverwrite;
	OpenCLRunKernelErrors[49] = runKernelErrorRemoveMean;
	OpenCLRunKernelErrors[50] = runKernelErrorSetStartClusterIndices;
	OpenCLRunKernelErrors[51] = runKernelErrorClusterizeScan;
	OpenCLRunKernelErrors[52] = runKernelErrorClusterizeRelabel;
	OpenCLRunKernelErrors[53] = runKernelErrorCalculateClusterSizes;
	OpenCLRunKernelErrors[54] = runKernelErrorCalculateClusterMasses;
	OpenCLRunKernelErrors[55] = runKernelErrorCalculateLargestCluster;
	OpenCLRunKernelErrors[56] = runKernelErrorCalculateTFCEValues;
	OpenCLRunKernelErrors[57] = runKernelErrorCalculateBetaWeightsGLM;
	OpenCLRunKernelErrors[58] = runKernelErrorCalculateBetaWeightsGLMFirstLevel;
	OpenCLRunKernelErrors[59] = runKernelErrorCalculateGLMResiduals;
	OpenCLRunKernelErrors[60] = runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel;
	OpenCLRunKernelErrors[61] = runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel;
	OpenCLRunKernelErrors[62] = runKernelErrorCalculateStatisticalMapsGLMTTest;
	OpenCLRunKernelErrors[63] = runKernelErrorCalculateStatisticalMapsGLMFTest;
	OpenCLRunKernelErrors[64] = runKernelErrorCalculateStatisticalMapsGLMBayesian;
	OpenCLRunKernelErrors[65] = runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelPermutation;
	OpenCLRunKernelErrors[66] = runKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelPermutation;
	OpenCLRunKernelErrors[67] = runKernelErrorCalculateStatisticalMapsGLMTTestSecondLevelPermutation;
	OpenCLRunKernelErrors[68] = runKernelErrorCalculateStatisticalMapsGLMFTestSecondLevelPermutation;
	OpenCLRunKernelErrors[69] = runKernelErrorCalculateStatisticalMapsMeanSecondLevelPermutation;
	OpenCLRunKernelErrors[70] = runKernelErrorEstimateAR4Models;
	OpenCLRunKernelErrors[71] = runKernelErrorApplyWhiteningAR4;
	OpenCLRunKernelErrors[72] = runKernelErrorGeneratePermutedVolumesFirstLevel;
	OpenCLRunKernelErrors[73] = runKernelErrorRemoveLinearFit;
	OpenCLRunKernelErrors[74] = runKernelErrorCalculatePermutationPValuesVoxelLevelInference;
	OpenCLRunKernelErrors[75] = runKernelErrorCalculatePermutationPValuesClusterLevelInference;

	return OpenCLRunKernelErrors;
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


// Returns the significance threshold calculated with a permutation test
float BROCCOLI_LIB::GetSignificanceThreshold()
{
	return SIGNIFICANCE_THRESHOLD;
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
	clEnqueueWriteBuffer(commandQueue, c_Filter_1_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Filter_1_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Filter_1_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Filter_1_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Filter_2_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Filter_2_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Filter_2_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Filter_2_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Filter_3_Real, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Filter_3_Real[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Filter_3_Imag, CL_TRUE, 0, FILTER_SIZE * FILTER_SIZE * sizeof(float), &h_Filter_3_Imag[z * FILTER_SIZE * FILTER_SIZE], 0, NULL, NULL);
}

// Performs non-separable convolution in 3D, for three complex valued (quadrature) filters
void BROCCOLI_LIB::NonseparableConvolution3D(cl_mem d_q1,
		                                     cl_mem d_q2,
		                                     cl_mem d_q3,
		                                     cl_mem d_Volume,
		                                     cl_mem c_Filter_1_Real,
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
		                                     int DATA_W,
		                                     int DATA_H,
		                                     int DATA_D)
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

void BROCCOLI_LIB::SetMemoryInt(cl_mem memory, int value, int N)
{
	SetGlobalAndLocalWorkSizesMemset(N);
	clSetKernelArg(MemsetIntKernel, 0, sizeof(cl_mem), &memory);
	clSetKernelArg(MemsetIntKernel, 1, sizeof(int), &value);
	clSetKernelArg(MemsetIntKernel, 2, sizeof(int), &N);
	runKernelErrorMemsetInt = clEnqueueNDRangeKernel(commandQueue, MemsetIntKernel, 1, NULL, globalWorkSizeMemset, localWorkSizeMemset, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::SetMemoryFloat2(cl_mem memory, float value, int N)
{
	SetGlobalAndLocalWorkSizesMemset(N);
	clSetKernelArg(MemsetFloat2Kernel, 0, sizeof(cl_mem), &memory);
	clSetKernelArg(MemsetFloat2Kernel, 1, sizeof(float), &value);
	clSetKernelArg(MemsetFloat2Kernel, 2, sizeof(int), &N);
	runKernelErrorMemsetFloat2 = clEnqueueNDRangeKernel(commandQueue, MemsetFloat2Kernel, 1, NULL, globalWorkSizeMemset, localWorkSizeMemset, 0, NULL, NULL);
	clFinish(commandQueue);
}


// This function is used by all Linear registration functions, to setup necessary parameters
void BROCCOLI_LIB::AlignTwoVolumesLinearSetup(int DATA_W, int DATA_H, int DATA_D)
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

	clSetKernelArg(InterpolateVolumeNearestLinearKernel, 0, sizeof(cl_mem), &d_Aligned_Volume);
	clSetKernelArg(InterpolateVolumeNearestLinearKernel, 1, sizeof(cl_mem), &d_Original_Volume);
	clSetKernelArg(InterpolateVolumeNearestLinearKernel, 2, sizeof(cl_mem), &c_Registration_Parameters);
	clSetKernelArg(InterpolateVolumeNearestLinearKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(InterpolateVolumeNearestLinearKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(InterpolateVolumeNearestLinearKernel, 5, sizeof(int), &DATA_D);
	clSetKernelArg(InterpolateVolumeNearestLinearKernel, 6, sizeof(int), &volume);

	clSetKernelArg(InterpolateVolumeLinearLinearKernel, 0, sizeof(cl_mem), &d_Aligned_Volume);
	clSetKernelArg(InterpolateVolumeLinearLinearKernel, 1, sizeof(cl_mem), &d_Original_Volume);
	clSetKernelArg(InterpolateVolumeLinearLinearKernel, 2, sizeof(cl_mem), &c_Registration_Parameters);
	clSetKernelArg(InterpolateVolumeLinearLinearKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(InterpolateVolumeLinearLinearKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(InterpolateVolumeLinearLinearKernel, 5, sizeof(int), &DATA_D);
	clSetKernelArg(InterpolateVolumeLinearLinearKernel, 6, sizeof(int), &volume);

	clSetKernelArg(InterpolateVolumeCubicLinearKernel, 0, sizeof(cl_mem), &d_Aligned_Volume);
	clSetKernelArg(InterpolateVolumeCubicLinearKernel, 1, sizeof(cl_mem), &d_Original_Volume);
	clSetKernelArg(InterpolateVolumeCubicLinearKernel, 2, sizeof(cl_mem), &c_Registration_Parameters);
	clSetKernelArg(InterpolateVolumeCubicLinearKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(InterpolateVolumeCubicLinearKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(InterpolateVolumeCubicLinearKernel, 5, sizeof(int), &DATA_D);
	clSetKernelArg(InterpolateVolumeCubicLinearKernel, 6, sizeof(int), &volume);
}





// This function is the foundation for all the Linear image registration functions
void BROCCOLI_LIB::AlignTwoVolumesLinear(float *h_Registration_Parameters_Align_Two_Volumes,
		                                     float* h_Rotations,
		                                     int DATA_W,
		                                     int DATA_H,
		                                     int DATA_D,
		                                     int NUMBER_OF_ITERATIONS,
		                                     int ALIGNMENT_TYPE,
		                                     int INTERPOLATION_MODE)
{
	// Calculate the filter responses for the reference volume (only needed once)
	NonseparableConvolution3D(d_q11, d_q12, d_q13, d_Reference_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_1_Linear_Registration_Real, h_Quadrature_Filter_1_Linear_Registration_Imag, h_Quadrature_Filter_2_Linear_Registration_Real, h_Quadrature_Filter_2_Linear_Registration_Imag, h_Quadrature_Filter_3_Linear_Registration_Real, h_Quadrature_Filter_3_Linear_Registration_Imag, DATA_W, DATA_H, DATA_D);

	if (DEBUG)
	{
		clEnqueueReadBuffer(commandQueue, d_q11, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_1, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_q12, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_2, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_q13, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_3, 0, NULL, NULL);
	}

	// Reset the parameter vector
	for (int p = 0; p < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; p++)
	{
		h_Registration_Parameters_Align_Two_Volumes[p] = 0.0f;
		h_Registration_Parameters[p] = 0.0f;
	}

	// Run the registration algorithm for a number of iterations
	for (int it = 0; it < NUMBER_OF_ITERATIONS; it++)
	{
		// Calculate the filter responses for the altered volume
		NonseparableConvolution3D(d_q21, d_q22, d_q23, d_Aligned_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_1_Linear_Registration_Real, h_Quadrature_Filter_1_Linear_Registration_Imag, h_Quadrature_Filter_2_Linear_Registration_Real, h_Quadrature_Filter_2_Linear_Registration_Imag, h_Quadrature_Filter_3_Linear_Registration_Real, h_Quadrature_Filter_3_Linear_Registration_Imag, DATA_W, DATA_H, DATA_D);

		/*
		if ( DEBUG && (it == 0))
		{
			clEnqueueReadBuffer(commandQueue, d_q21, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_1, 0, NULL, NULL);
			clEnqueueReadBuffer(commandQueue, d_q22, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_2, 0, NULL, NULL);
			clEnqueueReadBuffer(commandQueue, d_q23, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_3, 0, NULL, NULL);
		}
		*/

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

		if ( DEBUG && (it == 0) )
		{
			clEnqueueReadBuffer(commandQueue, d_Phase_Differences, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Differences, 0, NULL, NULL);
			clEnqueueReadBuffer(commandQueue, d_Phase_Gradients, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Gradients, 0, NULL, NULL);
			clEnqueueReadBuffer(commandQueue, d_Phase_Certainties, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Certainties, 0, NULL, NULL);
		}

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

		// Mirror the matrix values to get full matrix
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
			RemoveTransformationScaling(h_Registration_Parameters);
			AddAffineRegistrationParameters(h_Registration_Parameters_Align_Two_Volumes,h_Registration_Parameters);
		}
		// Keep all parameters
		else if (ALIGNMENT_TYPE == AFFINE)
		{
			AddAffineRegistrationParameters(h_Registration_Parameters_Align_Two_Volumes,h_Registration_Parameters);
		}

		// Copy parameter vector to constant memory
		clEnqueueWriteBuffer(commandQueue, c_Registration_Parameters, CL_TRUE, 0, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), h_Registration_Parameters_Align_Two_Volumes, 0, NULL, NULL);

		// Interpolate to get the new volume
		runKernelErrorInterpolateVolumeLinearLinear = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);

		clFinish(commandQueue);
	}

	// Convert rotation matrix to rotation angles
	if (ALIGNMENT_TYPE == RIGID)
	{
		CalculateRotationAnglesFromRotationMatrix(h_Rotations, h_Registration_Parameters_Align_Two_Volumes);
	}
}


// This function is used by all non-Linear registration functions, to setup necessary parameters
void BROCCOLI_LIB::AlignTwoVolumesNonLinearSetup(int DATA_W, int DATA_H, int DATA_D)
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

	//d_Phase_Differences = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float) * NUMBER_OF_FILTERS_FOR_NONLinear_REGISTRATION, NULL, &createBufferErrorPhaseDifferences);
	//d_Phase_Certainties = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float) * NUMBER_OF_FILTERS_FOR_NONLinear_REGISTRATION, NULL, &createBufferErrorPhaseCertainties);


	d_Update_Displacement_Field_X = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);
	d_Update_Displacement_Field_Y = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);
	d_Update_Displacement_Field_Z = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);

	d_Temp_Displacement_Field_X = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);
	d_Temp_Displacement_Field_Y = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);
	d_Temp_Displacement_Field_Z = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);

	//d_Update_Certainty = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W * DATA_H * DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);



	// Allocate constant memory

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

	c_Filter_Directions_X = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Filter_Directions_Y = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);
	c_Filter_Directions_Z = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION * sizeof(float), NULL, &createBufferErrorQuadratureFilter1Real);

	clEnqueueWriteBuffer(commandQueue, c_Filter_Directions_X, CL_TRUE, 0, NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION * sizeof(float), h_Filter_Directions_X, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Filter_Directions_Y, CL_TRUE, 0, NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION * sizeof(float), h_Filter_Directions_Y, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Filter_Directions_Z, CL_TRUE, 0, NUMBER_OF_FILTERS_FOR_NONLINEAR_REGISTRATION * sizeof(float), h_Filter_Directions_Z, 0, NULL, NULL);

	// Set all kernel arguments

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
	clSetKernelArg(InterpolateVolumeNearestNonLinearKernel, 0, sizeof(cl_mem), &d_Aligned_Volume);
	clSetKernelArg(InterpolateVolumeNearestNonLinearKernel, 1, sizeof(cl_mem), &d_Original_Volume);
	clSetKernelArg(InterpolateVolumeNearestNonLinearKernel, 2, sizeof(cl_mem), &d_Displacement_Field);
	clSetKernelArg(InterpolateVolumeNearestNonLinearKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(InterpolateVolumeNearestNonLinearKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(InterpolateVolumeNearestNonLinearKernel, 5, sizeof(int), &DATA_D);
	clSetKernelArg(InterpolateVolumeNearestNonLinearKernel, 6, sizeof(int), &volume);
	*/


	clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 0, sizeof(cl_mem), &d_Aligned_Volume);
	clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 1, sizeof(cl_mem), &d_Original_Volume);

	clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 8, sizeof(int), &volume);

	/*
	clSetKernelArg(InterpolateVolumeCubicNonLinearKernel, 0, sizeof(cl_mem), &d_Aligned_Volume);
	clSetKernelArg(InterpolateVolumeCubicNonLinearKernel, 1, sizeof(cl_mem), &d_Original_Volume);
	clSetKernelArg(InterpolateVolumeCubicNonLinearKernel, 2, sizeof(cl_mem), &d_Displacement_Field);
	clSetKernelArg(InterpolateVolumeCubicNonLinearKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(InterpolateVolumeCubicNonLinearKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(InterpolateVolumeCubicNonLinearKernel, 5, sizeof(int), &DATA_D);
	clSetKernelArg(InterpolateVolumeCubicNonLinearKernel, 6, sizeof(int), &volume);
	*/
}

// Takes a volume, applies 6 quadrature filters, calculates the 3D structure tensor, finally calculates magnitude of tensor
void BROCCOLI_LIB::CalculateTensorMagnitude(cl_mem d_Tensor_Magnitudes, cl_mem d_Volume, int DATA_W, int DATA_H, int DATA_D)
{
	AlignTwoVolumesNonLinearSetup(DATA_W,DATA_H,DATA_D);

	NonseparableConvolution3D(d_q21, d_q22, d_q23, d_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_1_NonLinear_Registration_Real, h_Quadrature_Filter_1_NonLinear_Registration_Imag, h_Quadrature_Filter_2_NonLinear_Registration_Real, h_Quadrature_Filter_2_NonLinear_Registration_Imag, h_Quadrature_Filter_3_NonLinear_Registration_Real, h_Quadrature_Filter_3_NonLinear_Registration_Imag, DATA_W, DATA_H, DATA_D);
	NonseparableConvolution3D(d_q24, d_q25, d_q26, d_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_4_NonLinear_Registration_Real, h_Quadrature_Filter_4_NonLinear_Registration_Imag, h_Quadrature_Filter_5_NonLinear_Registration_Real, h_Quadrature_Filter_5_NonLinear_Registration_Imag, h_Quadrature_Filter_6_NonLinear_Registration_Real, h_Quadrature_Filter_6_NonLinear_Registration_Imag, DATA_W, DATA_H, DATA_D);

	SetMemory(d_t11, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_t12, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_t13, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_t22, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_t23, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_t33, 0.0f, DATA_W * DATA_H * DATA_D);

	clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q11);
	clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q21);
	clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float),  &M11_1);
	clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float),  &M12_1);
	clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_1);
	clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_1);
	clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_1);
	clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_1);
	runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

	clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q12);
	clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q22);
	clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float),  &M11_2);
	clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float),  &M12_2);
	clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_2);
	clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_2);
	clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_2);
	clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_2);
	runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

	clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q13);
	clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q23);
	clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float),  &M11_3);
	clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float),  &M12_3);
	clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_3);
	clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_3);
	clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_3);
	clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_3);
	runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

	clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q14);
	clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q24);
	clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float),  &M11_4);
	clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float),  &M12_4);
	clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_4);
	clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_4);
	clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_4);
	clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_4);
	runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

	clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q15);
	clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q25);
	clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float),  &M11_5);
	clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float),  &M12_5);
	clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_5);
	clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_5);
	clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_5);
	clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_5);
	runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

	clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q16);
	clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q26);
	clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float),  &M11_6);
	clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float),  &M12_6);
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

	AlignTwoVolumesNonLinearCleanup();
}

// This function is the foundation for all the non-Linear image registration functions
void BROCCOLI_LIB::AlignTwoVolumesNonLinear(int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_ITERATIONS, int INTERPOLATION_MODE)
{
	// Calculate the filter responses for the reference volume (only needed once), calculate three complex valued filter responses at a time
	NonseparableConvolution3D(d_q11, d_q12, d_q13, d_Reference_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_1_NonLinear_Registration_Real, h_Quadrature_Filter_1_NonLinear_Registration_Imag, h_Quadrature_Filter_2_NonLinear_Registration_Real, h_Quadrature_Filter_2_NonLinear_Registration_Imag, h_Quadrature_Filter_3_NonLinear_Registration_Real, h_Quadrature_Filter_3_NonLinear_Registration_Imag, DATA_W, DATA_H, DATA_D);
	NonseparableConvolution3D(d_q14, d_q15, d_q16, d_Reference_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_4_NonLinear_Registration_Real, h_Quadrature_Filter_4_NonLinear_Registration_Imag, h_Quadrature_Filter_5_NonLinear_Registration_Real, h_Quadrature_Filter_5_NonLinear_Registration_Imag, h_Quadrature_Filter_6_NonLinear_Registration_Real, h_Quadrature_Filter_6_NonLinear_Registration_Imag, DATA_W, DATA_H, DATA_D);

	//clEnqueueReadBuffer(commandQueue, d_q11, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_1, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_q12, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_2, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_q13, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_3, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_q14, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_4, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_q15, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_5, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_q16, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_6, 0, NULL, NULL);

	// Reset displacement field
	SetMemory(d_Update_Displacement_Field_X, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_Update_Displacement_Field_Y, 0.0f, DATA_W * DATA_H * DATA_D);
	SetMemory(d_Update_Displacement_Field_Z, 0.0f, DATA_W * DATA_H * DATA_D);

	int zero, one, two, three, four, five;
	zero = 0; one = 1; two = 2; three = 3; four = 4; five = 5;

	// Run the registration algorithm for a number of iterations
	for (int it = 0; it < NUMBER_OF_ITERATIONS; it++)
	{
		// Calculate the filter responses for the aligned volume, calculate three complex valued filter responses at a time
		NonseparableConvolution3D(d_q21, d_q22, d_q23, d_Aligned_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_1_NonLinear_Registration_Real, h_Quadrature_Filter_1_NonLinear_Registration_Imag, h_Quadrature_Filter_2_NonLinear_Registration_Real, h_Quadrature_Filter_2_NonLinear_Registration_Imag, h_Quadrature_Filter_3_NonLinear_Registration_Real, h_Quadrature_Filter_3_NonLinear_Registration_Imag, DATA_W, DATA_H, DATA_D);
		NonseparableConvolution3D(d_q24, d_q25, d_q26, d_Aligned_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_4_NonLinear_Registration_Real, h_Quadrature_Filter_4_NonLinear_Registration_Imag, h_Quadrature_Filter_5_NonLinear_Registration_Real, h_Quadrature_Filter_5_NonLinear_Registration_Imag, h_Quadrature_Filter_6_NonLinear_Registration_Real, h_Quadrature_Filter_6_NonLinear_Registration_Imag, DATA_W, DATA_H, DATA_D);

		//clEnqueueReadBuffer(commandQueue, d_q21, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_1, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_q22, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_2, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_q23, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_3, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_q24, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_4, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_q25, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_5, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_q26, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(cl_float2), h_Quadrature_Filter_Response_6, 0, NULL, NULL);

		// Reset tensor components
		SetMemory(d_t11, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_t12, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_t13, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_t22, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_t23, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_t33, 0.0f, DATA_W * DATA_H * DATA_D);

		// Reset equation system
		SetMemory(d_a11, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_a12, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_a13, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_a22, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_a23, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_a33, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_h1, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_h2, 0.0f, DATA_W * DATA_H * DATA_D);
		SetMemory(d_h3, 0.0f, DATA_W * DATA_H * DATA_D);

		// Calculate tensor components by summing over 6 quadrature filters

		// First filter
		clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q11);
  	    clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q21);
		clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_1);
		clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_1);
		clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_1);
		clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_1);
		clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_1);
		clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_1);
		runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		// Second filter
		clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q12);
  	    clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q22);
		clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_2);
		clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_2);
		clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_2);
		clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_2);
		clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_2);
		clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_2);
		runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		// Third filter
		clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q13);
  	    clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q23);
		clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_3);
		clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_3);
		clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_3);
		clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_3);
		clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_3);
		clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_3);
		runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		// Fourth filter
		clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q14);
  	    clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q24);
		clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_4);
		clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_4);
		clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_4);
		clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_4);
		clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_4);
		clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_4);
		runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		// Fifth filter
		clSetKernelArg(CalculateTensorComponentsKernel, 6, sizeof(cl_mem), &d_q15);
  	    clSetKernelArg(CalculateTensorComponentsKernel, 7, sizeof(cl_mem), &d_q25);
		clSetKernelArg(CalculateTensorComponentsKernel, 8, sizeof(float), &M11_5);
		clSetKernelArg(CalculateTensorComponentsKernel, 9, sizeof(float), &M12_5);
		clSetKernelArg(CalculateTensorComponentsKernel, 10, sizeof(float), &M13_5);
		clSetKernelArg(CalculateTensorComponentsKernel, 11, sizeof(float), &M22_5);
		clSetKernelArg(CalculateTensorComponentsKernel, 12, sizeof(float), &M23_5);
		clSetKernelArg(CalculateTensorComponentsKernel, 13, sizeof(float), &M33_5);
		runKernelErrorCalculateTensorComponents = clEnqueueNDRangeKernel(commandQueue, CalculateTensorComponentsKernel, 3, NULL, globalWorkSizeCalculatePhaseDifferencesAndCertainties, localWorkSizeCalculatePhaseDifferencesAndCertainties, 0, NULL, NULL);

		// Sixth filter
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
		//CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, 2.25);
		CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, TSIGMA);
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



		// Calculate A-matrices and h-vectors, by summing over 6 quadrature filters

		// First filter
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 9, sizeof(cl_mem), &d_q11);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 10, sizeof(cl_mem), &d_q21);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 23, sizeof(int), &zero);
		runKernelErrorCalculateAMatricesAndHVectors = clEnqueueNDRangeKernel(commandQueue, CalculateAMatricesAndHVectorsKernel, 3, NULL, globalWorkSizeCalculateAMatricesAndHVectors, localWorkSizeCalculateAMatricesAndHVectors, 0, NULL, NULL);

		// Second filter
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 9, sizeof(cl_mem), &d_q12);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 10, sizeof(cl_mem), &d_q22);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 23, sizeof(int), &one);
		runKernelErrorCalculateAMatricesAndHVectors = clEnqueueNDRangeKernel(commandQueue, CalculateAMatricesAndHVectorsKernel, 3, NULL, globalWorkSizeCalculateAMatricesAndHVectors, localWorkSizeCalculateAMatricesAndHVectors, 0, NULL, NULL);

		// Third filter
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 9, sizeof(cl_mem), &d_q13);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 10, sizeof(cl_mem), &d_q23);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 23, sizeof(int), &two);
		runKernelErrorCalculateAMatricesAndHVectors = clEnqueueNDRangeKernel(commandQueue, CalculateAMatricesAndHVectorsKernel, 3, NULL, globalWorkSizeCalculateAMatricesAndHVectors, localWorkSizeCalculateAMatricesAndHVectors, 0, NULL, NULL);

		// Fourth filter
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 9, sizeof(cl_mem), &d_q14);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 10, sizeof(cl_mem), &d_q24);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 23, sizeof(int), &three);
		runKernelErrorCalculateAMatricesAndHVectors = clEnqueueNDRangeKernel(commandQueue, CalculateAMatricesAndHVectorsKernel, 3, NULL, globalWorkSizeCalculateAMatricesAndHVectors, localWorkSizeCalculateAMatricesAndHVectors, 0, NULL, NULL);

		// Fifth filter
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 9, sizeof(cl_mem), &d_q15);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 10, sizeof(cl_mem), &d_q25);
		clSetKernelArg(CalculateAMatricesAndHVectorsKernel, 23, sizeof(int), &four);
		runKernelErrorCalculateAMatricesAndHVectors = clEnqueueNDRangeKernel(commandQueue, CalculateAMatricesAndHVectorsKernel, 3, NULL, globalWorkSizeCalculateAMatricesAndHVectors, localWorkSizeCalculateAMatricesAndHVectors, 0, NULL, NULL);

		// Sixth filter
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
		//CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, 2.25);
		CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, ESIGMA);
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

		// Calculate the best displacement vector in each voxel
		runKernelErrorCalculateDisplacementUpdate = clEnqueueNDRangeKernel(commandQueue, CalculateDisplacementUpdateKernel, 3, NULL, globalWorkSizeCalculateDisplacementAndCertaintyUpdate, localWorkSizeCalculateDisplacementAndCertaintyUpdate, 0, NULL, NULL);

		//clEnqueueReadBuffer(commandQueue, d_Update_Displacement_Field_X, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Differences, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_Update_Displacement_Field_Y, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Certainties, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_Update_Displacement_Field_Z, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Phase_Gradients, 0, NULL, NULL);


		/*
		clEnqueueReadBuffer(commandQueue, d_Update_Displacement_Field_X, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t11, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Update_Displacement_Field_Y, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t12, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Update_Displacement_Field_Z, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_t13, 0, NULL, NULL);
		*/

		// Smooth the displacement field
		//CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, 2.25);
		CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, DSIGMA);
		PerformSmoothing(d_Temp_Displacement_Field_X, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_Temp_Displacement_Field_Y, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);
		PerformSmoothing(d_Temp_Displacement_Field_Z, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, DATA_W, DATA_H, DATA_D, 1);

		// Increment total displacement field
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
		clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 2, sizeof(cl_mem), &d_Update_Displacement_Field_X);
		clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 3, sizeof(cl_mem), &d_Update_Displacement_Field_Y);
		clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 4, sizeof(cl_mem), &d_Update_Displacement_Field_Z);
		runKernelErrorInterpolateVolumeLinearNonLinear = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearNonLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
		clFinish(commandQueue);

	}


	//clEnqueueReadBuffer(commandQueue, d_Aligned_Volume, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Aligned_T1_Volume_NonLinear, 0, NULL, NULL);
}

void BROCCOLI_LIB::AlignTwoVolumesNonLinearCleanup()
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



// Removes scaling from a transformation matrix, by doing a SVD and setting all singular values to 0
void BROCCOLI_LIB::RemoveTransformationScaling(float* h_Registration_Parameters)
{
	Eigen::MatrixXd TransformationMatrix(3,3);

	// Make a copy of transformation matrix parameters and put into Eigen matrix
	TransformationMatrix(0,0) = (double)h_Registration_Parameters[3];
	TransformationMatrix(0,1) = (double)h_Registration_Parameters[4];
	TransformationMatrix(0,2) = (double)h_Registration_Parameters[5];
	TransformationMatrix(1,0) = (double)h_Registration_Parameters[6];
	TransformationMatrix(1,1) = (double)h_Registration_Parameters[7];
	TransformationMatrix(1,2) = (double)h_Registration_Parameters[8];
	TransformationMatrix(2,0) = (double)h_Registration_Parameters[9];
	TransformationMatrix(2,1) = (double)h_Registration_Parameters[10];
	TransformationMatrix(2,2) = (double)h_Registration_Parameters[11];

	// Add one to diagonal (since the registration algorithm only estimates the difference between the two volumes)
	TransformationMatrix(0,0) += 1.0;
	TransformationMatrix(1,1) += 1.0;
	TransformationMatrix(2,2) += 1.0;

	// Calculate SVD, and calculate U and V matrices
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(TransformationMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

	// Calculate transformation matrix without scaling (i.e. singular values = ones)
	Eigen::MatrixXd TransformationMatrixWithoutScaling = svd.matrixU() * svd.matrixV().transpose();

	// Remove one from diagonal
	TransformationMatrixWithoutScaling(0,0) -= 1.0;
	TransformationMatrixWithoutScaling(1,1) -= 1.0;
	TransformationMatrixWithoutScaling(2,2) -= 1.0;

	// Put back transformation matrix to array
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

	// Make a double copy of transformation matrix parameters
	for (int i = 3; i < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; i++)
	{
		h_Transformation_Matrix[i-3] = (double)h_Registration_Parameters[i];
	}

	// Add ones in the diagonal (since the registration algorithm only estimates the difference between the two volumes)
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

// Changes volume size out of place
void BROCCOLI_LIB::ChangeVolumeSize(cl_mem d_Changed_Volume, cl_mem d_Original_Volume_, int ORIGINAL_DATA_W, int ORIGINAL_DATA_H, int ORIGINAL_DATA_D, int NEW_DATA_W, int NEW_DATA_H, int NEW_DATA_D, int INTERPOLATION_MODE)
{
	// Create a 3D image (texture) for fast interpolation
	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_INTENSITY;
	cl_mem d_Volume_Texture = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, ORIGINAL_DATA_W, ORIGINAL_DATA_H, ORIGINAL_DATA_D, 0, 0, NULL, NULL);

	// Copy the volume to an image to interpolate from
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {ORIGINAL_DATA_W, ORIGINAL_DATA_H, ORIGINAL_DATA_D};
	clEnqueueCopyBufferToImage(commandQueue, d_Original_Volume_, d_Volume_Texture, 0, origin, region, 0, NULL, NULL);

	// Calculate how to interpolate (up or down)
	float VOXEL_DIFFERENCE_X = (float)(ORIGINAL_DATA_W-1)/(float)(NEW_DATA_W-1);
	float VOXEL_DIFFERENCE_Y = (float)(ORIGINAL_DATA_H-1)/(float)(NEW_DATA_H-1);
	float VOXEL_DIFFERENCE_Z = (float)(ORIGINAL_DATA_D-1)/(float)(NEW_DATA_D-1);

	SetGlobalAndLocalWorkSizesInterpolateVolume(NEW_DATA_W, NEW_DATA_H, NEW_DATA_D);

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

		runKernelErrorRescaleVolumeLinear = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
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

		runKernelErrorRescaleVolumeCubic = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeCubicKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	clReleaseMemObject(d_Volume_Texture);
}

// Changes volume size in place
void BROCCOLI_LIB::ChangeVolumeSize(cl_mem& d_Original_Volume, int ORIGINAL_DATA_W, int ORIGINAL_DATA_H, int ORIGINAL_DATA_D, int NEW_DATA_W, int NEW_DATA_H, int NEW_DATA_D, int INTERPOLATION_MODE)
{
	// Create a 3D image (texture) for fast interpolation
	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_INTENSITY;
	cl_mem d_Volume_Texture = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, ORIGINAL_DATA_W, ORIGINAL_DATA_H, ORIGINAL_DATA_D, 0, 0, NULL, NULL);

	// Copy the volume to an image to interpolate from
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {ORIGINAL_DATA_W, ORIGINAL_DATA_H, ORIGINAL_DATA_D};
	clEnqueueCopyBufferToImage(commandQueue, d_Original_Volume, d_Volume_Texture, 0, origin, region, 0, NULL, NULL);

	// Throw away old volume and make a new one of the new size
	clReleaseMemObject(d_Original_Volume);
	d_Original_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  NEW_DATA_W * NEW_DATA_H * NEW_DATA_D * sizeof(float), NULL, &createBufferErrorPhaseCertainties);

	// Calculate how to interpolate (up or down)
	float VOXEL_DIFFERENCE_X = (float)(ORIGINAL_DATA_W-1)/(float)(NEW_DATA_W-1);
	float VOXEL_DIFFERENCE_Y = (float)(ORIGINAL_DATA_H-1)/(float)(NEW_DATA_H-1);
	float VOXEL_DIFFERENCE_Z = (float)(ORIGINAL_DATA_D-1)/(float)(NEW_DATA_D-1);

	SetGlobalAndLocalWorkSizesInterpolateVolume(NEW_DATA_W, NEW_DATA_H, NEW_DATA_D);

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

		runKernelErrorRescaleVolumeLinear = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
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

		runKernelErrorRescaleVolumeCubic = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeCubicKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	clReleaseMemObject(d_Volume_Texture);
}

// Runs Linear registration over several scales, COARSEST_SCALE should be 8, 4, 2 or 1
void BROCCOLI_LIB::AlignTwoVolumesLinearSeveralScales(float *h_Registration_Parameters_Align_Two_Volumes_Several_Scales,
                                                          float* h_Rotations,
                                                          cl_mem d_Original_Aligned_Volume,
														  cl_mem d_Original_Reference_Volume,
														  int DATA_W,
														  int DATA_H,
														  int DATA_D,
														  int COARSEST_SCALE,
														  int NUMBER_OF_ITERATIONS,
														  int ALIGNMENT_TYPE,
														  int OVERWRITE,
														  int INTERPOLATION_MODE)
{
	// Reset parameter vectors
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
	AlignTwoVolumesLinearSetup(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);

	// Change size of original volumes to current scale
	ChangeVolumeSize(d_Aligned_Volume, d_Original_Aligned_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, INTERPOLATION_MODE);
	ChangeVolumeSize(d_Reference_Volume, d_Original_Reference_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, INTERPOLATION_MODE);

	// Copy volume to be aligned to an image (texture)
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D};
	clEnqueueCopyBufferToImage(commandQueue, d_Aligned_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);

	// Loop registration over scales
	for (int current_scale = COARSEST_SCALE; current_scale >= 1; current_scale = current_scale/2)
	{
		// Less iterations on finest scale
		if (current_scale == 1)
		{
			AlignTwoVolumesLinear(h_Registration_Parameters_Temp, h_Rotations_Temp, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, (int)ceil((float)NUMBER_OF_ITERATIONS/5.0f), ALIGNMENT_TYPE, INTERPOLATION_MODE);
		}
		else
		{
			AlignTwoVolumesLinear(h_Registration_Parameters_Temp, h_Rotations_Temp, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, NUMBER_OF_ITERATIONS, ALIGNMENT_TYPE, INTERPOLATION_MODE);
		}

		// Not last scale
		if (current_scale != 1)
		{
			h_Rotations[0] += h_Rotations_Temp[0];
			h_Rotations[1] += h_Rotations_Temp[1];
			h_Rotations[2] += h_Rotations_Temp[2];

			// Multiply the transformations by a factor 2 for the next scale and add to previous parameters
			AddAffineRegistrationParametersNextScale(h_Registration_Parameters_Align_Two_Volumes_Several_Scales,h_Registration_Parameters_Temp);

			// Clean up before the next scale
			AlignTwoVolumesLinearCleanup();

			// Prepare for the next scale  (the previous scale was current scale, so the next scale is times 2)
			CURRENT_DATA_W = (int)round((float)DATA_W/((float)current_scale/2.0f));
			CURRENT_DATA_H = (int)round((float)DATA_H/((float)current_scale/2.0f));
			CURRENT_DATA_D = (int)round((float)DATA_D/((float)current_scale/2.0f));

			// Setup all parameters and allocate memory on host
			AlignTwoVolumesLinearSetup(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);

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
				runKernelErrorInterpolateVolumeLinearLinear = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
				clFinish(commandQueue);
			}
			else if (INTERPOLATION_MODE == CUBIC)
			{
				runKernelErrorInterpolateVolumeCubicLinear = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeCubicLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
				clFinish(commandQueue);
			}

			// Copy transformed volume back to image (texture)
			clEnqueueCopyBufferToImage(commandQueue, d_Aligned_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);
		}
		else // Last scale, nothing more to do
		{
			// Clean up
			AlignTwoVolumesLinearCleanup();

			// Calculate final registration parameters
			AddAffineRegistrationParameters(h_Registration_Parameters_Align_Two_Volumes_Several_Scales,h_Registration_Parameters_Temp);

			h_Rotations[0] += h_Rotations_Temp[0];
			h_Rotations[1] += h_Rotations_Temp[1];
			h_Rotations[2] += h_Rotations_Temp[2];

			if (OVERWRITE == DO_OVERWRITE)
			{
				// Transform the original volume once with the final registration parameters, to remove effects of several interpolations
				TransformVolumesLinear(d_Original_Aligned_Volume, h_Registration_Parameters_Align_Two_Volumes_Several_Scales, DATA_W, DATA_H, DATA_D, 1, INTERPOLATION_MODE);
			}
		}
	}
}

// Runs non-Linear registration over several scales, COARSEST_SCALE should be 8, 4, 2 or 1
void BROCCOLI_LIB::AlignTwoVolumesNonLinearSeveralScales(cl_mem d_Original_Aligned_Volume,
		                                                     cl_mem d_Original_Reference_Volume,
		                                                     int DATA_W,
		                                                     int DATA_H,
		                                                     int DATA_D,
		                                                     int COARSEST_SCALE,
		                                                     int NUMBER_OF_ITERATIONS,
		                                                     int OVERWRITE,
		                                                     int INTERPOLATION_MODE,
		                                                     int KEEP)
{
	// Calculate volume size for coarsest scale
	CURRENT_DATA_W = (int)round((float)DATA_W/((float)COARSEST_SCALE));
	CURRENT_DATA_H = (int)round((float)DATA_H/((float)COARSEST_SCALE));
	CURRENT_DATA_D = (int)round((float)DATA_D/((float)COARSEST_SCALE));

	int PREVIOUS_DATA_W = CURRENT_DATA_W;
	int PREVIOUS_DATA_H = CURRENT_DATA_H;
	int PREVIOUS_DATA_D = CURRENT_DATA_D;

	// Setup all parameters and allocate memory on host
	AlignTwoVolumesNonLinearSetup(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);

	// Change size of original volumes to current scale
	ChangeVolumeSize(d_Aligned_Volume, d_Original_Aligned_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, INTERPOLATION_MODE);
	ChangeVolumeSize(d_Reference_Volume, d_Original_Reference_Volume, DATA_W, DATA_H, DATA_D, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, INTERPOLATION_MODE);

	// Copy volume to be aligned to an image (texture)
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D};
	clEnqueueCopyBufferToImage(commandQueue, d_Aligned_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);

	// Allocate memory for total displacement field, done separately as we release memory for each new scale
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
		// Less iterations on finest scale
		if (current_scale == 1)
		{
			AlignTwoVolumesNonLinear(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, (int)ceil((float)NUMBER_OF_ITERATIONS/2.0f), INTERPOLATION_MODE);
		}
		else
		{
			AlignTwoVolumesNonLinear(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D, NUMBER_OF_ITERATIONS, INTERPOLATION_MODE);
		}

		// Not last scale
		if (current_scale != 1)
		{
			// Add found displacement field to total displacement field
			AddVolumes(d_Total_Displacement_Field_X, d_Update_Displacement_Field_X, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);
			AddVolumes(d_Total_Displacement_Field_Y, d_Update_Displacement_Field_Y, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);
			AddVolumes(d_Total_Displacement_Field_Z, d_Update_Displacement_Field_Z, CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);

			// Clean up before the next scale
			AlignTwoVolumesNonLinearCleanup();

			// Prepare for the next scale (the previous scale was current scale, so the next scale is times 2)
			CURRENT_DATA_W = (int)round((float)DATA_W/((float)current_scale/2.0f));
			CURRENT_DATA_H = (int)round((float)DATA_H/((float)current_scale/2.0f));
			CURRENT_DATA_D = (int)round((float)DATA_D/((float)current_scale/2.0f));

			float scale_factor = 2.0f;

			// Setup all parameters and allocate memory on host
			AlignTwoVolumesNonLinearSetup(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);

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
				clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 2, sizeof(cl_mem), &d_Total_Displacement_Field_X);
				clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 3, sizeof(cl_mem), &d_Total_Displacement_Field_Y);
				clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 4, sizeof(cl_mem), &d_Total_Displacement_Field_Z);
				runKernelErrorInterpolateVolumeLinearNonLinear = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearNonLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
				clFinish(commandQueue);
			}
			else if (INTERPOLATION_MODE == CUBIC)
			{
				// Not implemented yet
				runKernelErrorInterpolateVolumeCubicNonLinear = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeCubicNonLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
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
				TransformVolumesNonLinear(d_Original_Aligned_Volume, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, DATA_W, DATA_H, DATA_D, 1, INTERPOLATION_MODE);
			}

			// Clean up
			AlignTwoVolumesNonLinearCleanup();
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



// This function is used by all registration functions, to cleanup allocated memory
void BROCCOLI_LIB::AlignTwoVolumesLinearCleanup()
{
	// Free all the allocated memory on the device

	clReleaseMemObject(d_Original_Volume);
	clReleaseMemObject(d_Reference_Volume);
	clReleaseMemObject(d_Aligned_Volume);

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

int mymin(int a, int b)
{
	if (a < b)
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

	SetGlobalAndLocalWorkSizesInterpolateVolume(T1_DATA_W_INTERPOLATED, T1_DATA_H_INTERPOLATED, T1_DATA_D_INTERPOLATED);

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
		error = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
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

		error = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeCubicKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
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

	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_Linear, 0, NULL, NULL);

	clReleaseMemObject(d_T1_Volume);
	clReleaseMemObject(d_Interpolated_T1_Volume);
	clReleaseMemObject(d_MNI_T1_Volume);
	clReleaseMemObject(d_T1_Volume_Texture);
}

void BROCCOLI_LIB::ChangeVolumesResolutionAndSize(cl_mem d_New_Volumes,
		                                          cl_mem d_Volumes,
		                                          int DATA_W,
		                                          int DATA_H,
		                                          int DATA_D,
		                                          int NUMBER_OF_VOLUMES,
		                                          int NEW_DATA_W,
		                                          int NEW_DATA_H,
		                                          int NEW_DATA_D,
		                                          float VOXEL_SIZE_X,
		                                          float VOXEL_SIZE_Y,
		                                          float VOXEL_SIZE_Z,
		                                          float NEW_VOXEL_SIZE_X,
		                                          float NEW_VOXEL_SIZE_Y,
		                                          float NEW_VOXEL_SIZE_Z,
		                                          int MM_Z_CUT,
		                                          int INTERPOLATION_MODE)
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

	SetGlobalAndLocalWorkSizesInterpolateVolume(DATA_W_INTERPOLATED, DATA_H_INTERPOLATED, DATA_D_INTERPOLATED);

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
	else if (INTERPOLATION_MODE == NEAREST)
	{
		clSetKernelArg(RescaleVolumeNearestKernel, 0, sizeof(cl_mem), &d_Interpolated_Volume);
		clSetKernelArg(RescaleVolumeNearestKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
		clSetKernelArg(RescaleVolumeNearestKernel, 2, sizeof(float), &VOXEL_DIFFERENCE_X);
		clSetKernelArg(RescaleVolumeNearestKernel, 3, sizeof(float), &VOXEL_DIFFERENCE_Y);
		clSetKernelArg(RescaleVolumeNearestKernel, 4, sizeof(float), &VOXEL_DIFFERENCE_Z);
		clSetKernelArg(RescaleVolumeNearestKernel, 5, sizeof(int), &DATA_W_INTERPOLATED);
		clSetKernelArg(RescaleVolumeNearestKernel, 6, sizeof(int), &DATA_H_INTERPOLATED);
		clSetKernelArg(RescaleVolumeNearestKernel, 7, sizeof(int), &DATA_D_INTERPOLATED);
	}

	// Make sure that the interpolated volume has the same number of voxels as the new volume in each direction
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

	SetMemory(d_New_Volumes, 0.0f, NEW_DATA_W * NEW_DATA_H * NEW_DATA_D * NUMBER_OF_VOLUMES);

	for (int volume = 0; volume < NUMBER_OF_VOLUMES; volume++)
	{
		SetMemory(d_Interpolated_Volume, 0.0f, DATA_W_INTERPOLATED * DATA_H_INTERPOLATED * DATA_D_INTERPOLATED);

		// Copy the current volume to an image to interpolate from
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {DATA_W, DATA_H, DATA_D};
		clEnqueueCopyBufferToImage(commandQueue, d_Volumes, d_Volume_Texture, volume * DATA_W * DATA_H * DATA_D * sizeof(float), origin, region, 0, NULL, NULL);

		// Rescale current volume to the same voxel size as the new volume
		if (INTERPOLATION_MODE == LINEAR)
		{
			runKernelErrorRescaleVolumeLinear = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
			clFinish(commandQueue);
		}
		else if (INTERPOLATION_MODE == CUBIC)
		{
			runKernelErrorRescaleVolumeCubic = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeCubicKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
			clFinish(commandQueue);
		}
		else if (INTERPOLATION_MODE == NEAREST)
		{
			runKernelErrorRescaleVolumeNearest = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeNearestKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
			clFinish(commandQueue);
		}

		clSetKernelArg(CopyVolumeToNewKernel, 13, sizeof(int), &volume);

		runKernelErrorCopyVolumeToNew = clEnqueueNDRangeKernel(commandQueue, CopyVolumeToNewKernel, 3, NULL, globalWorkSizeCopyVolumeToNew, localWorkSizeCopyVolumeToNew, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	clReleaseMemObject(d_Interpolated_Volume);
	clReleaseMemObject(d_Volume_Texture);
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

	SetGlobalAndLocalWorkSizesInterpolateVolume(T1_DATA_W_INTERPOLATED, T1_DATA_H_INTERPOLATED, T1_DATA_D_INTERPOLATED);

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
		runKernelErrorRescaleVolumeLinear = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
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
		runKernelErrorRescaleVolumeCubic = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeCubicKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
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

	runKernelErrorCopyT1VolumeToMNI = clEnqueueNDRangeKernel(commandQueue, CopyT1VolumeToMNIKernel, 3, NULL, globalWorkSizeCopyVolumeToNew, localWorkSizeCopyVolumeToNew, 0, NULL, NULL);
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

	runKernelErrorCopyT1VolumeToMNI = clEnqueueNDRangeKernel(commandQueue, CopyT1VolumeToMNIKernel, 3, NULL, globalWorkSizeCopyVolumeToNew, localWorkSizeCopyVolumeToNew, 0, NULL, NULL);
	clFinish(commandQueue);

	clReleaseMemObject(d_Interpolated_T1_Volume);
	clReleaseMemObject(d_T1_Volume_Texture);
}



void BROCCOLI_LIB::CalculateTopBrainSlice(int& slice, cl_mem d_Volume, int DATA_W, int DATA_H, int DATA_D, int z_cut)
{
	SetGlobalAndLocalWorkSizesCalculateMagnitudes(DATA_W, DATA_H, DATA_D);
	SetGlobalAndLocalWorkSizesCalculateSum(DATA_W, DATA_H, DATA_D);

	AlignTwoVolumesLinearSetup(DATA_W, DATA_H, DATA_D);

	// Apply quadrature filters to brain volume
	NonseparableConvolution3D(d_q21, d_q22, d_q23, d_Volume, c_Quadrature_Filter_1_Real, c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_3_Imag, h_Quadrature_Filter_1_Linear_Registration_Real, h_Quadrature_Filter_1_Linear_Registration_Imag, h_Quadrature_Filter_2_Linear_Registration_Real, h_Quadrature_Filter_2_Linear_Registration_Imag, h_Quadrature_Filter_3_Linear_Registration_Real, h_Quadrature_Filter_3_Linear_Registration_Imag, DATA_W, DATA_H, DATA_D);

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
	float max_slope = std::numeric_limits<float>::min();
	for (int z = 0; z < DATA_D; z++)
	{
		if (h_Derivatives[z] > max_slope)
		{
			max_slope = h_Derivatives[z];
			slice = z;
		}
	}

	AlignTwoVolumesLinearCleanup();

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

	SetGlobalAndLocalWorkSizesInterpolateVolume(EPI_DATA_W_INTERPOLATED, EPI_DATA_H_INTERPOLATED, EPI_DATA_D_INTERPOLATED);

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
		runKernelErrorRescaleVolumeLinear = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
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
		runKernelErrorRescaleVolumeCubic = clEnqueueNDRangeKernel(commandQueue, RescaleVolumeCubicKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
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

	runKernelErrorCopyEPIVolumeToT1 = clEnqueueNDRangeKernel(commandQueue, CopyEPIVolumeToT1Kernel, 3, NULL, globalWorkSizeCopyVolumeToNew, localWorkSizeCopyVolumeToNew, 0, NULL, NULL);
	clFinish(commandQueue);

	clReleaseMemObject(d_Interpolated_EPI_Volume);
	clReleaseMemObject(d_EPI_Volume_Texture);
}





// Inverts parameters for affine transformation
void BROCCOLI_LIB::InvertAffineRegistrationParameters(float* h_Inverse_Parameters, float* h_Parameters)
{
	// Put parameters in 4 x 4 affine transformation matrix

	// (p3 p4  p5  tx)
	// (p6 p7  p8  ty)
	// (p9 p10 p11 tz)
	// (0  0   0   1 )

	Eigen::MatrixXd Affine_Matrix(4,4);

	// Put values into an Eigen matrix, and convert to double
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

	// Calculate inverse
	Affine_Matrix.inverse();

	// Subtract ones in the diagonal

	// Put back translation parameters into array
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


// Adds two affine transformations, by a matrix multiplication
void BROCCOLI_LIB::AddAffineRegistrationParameters(float* h_Old_Parameters, float* h_New_Parameters)
{
	// Put parameters in 4 x 4 affine transformation matrix

	// (p3 p4  p5  tx)
	// (p6 p7  p8  ty)
	// (p9 p10 p11 tz)
	// (0  0   0   1 )

	Eigen::MatrixXd Old_Affine_Matrix(4,4), New_Affine_Matrix(4,4), Total_Affine_Matrix(4,4);

	// Put values into an Eigen matrix, and convert to double
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

	// Put values into an Eigen matrix, and convert to double
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

	// Put values back into array
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


// Adds two sets of affine registration parameters for the next registration scale
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


// Adds two sets of affine registration parameters, by doing a matrix multiplication, saves new parameters
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

// Multiplies all values in a volume with a factor
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

// Multiplies two volumes and saves result in a third volume
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

// Multiplies two volumes and overwrites first volume
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

// Multiplies two volumes and overwrites first volume
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

// Adds a value to each element in a volume
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

// Adds two volumes and saves as a third volume
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

// Adds two volumes and overwrites the first volume
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

void BROCCOLI_LIB::PerformRegistrationEPIT1Wrapper()
{
	// Reset total registration parameters
	for (int p = 0; p < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; p++)
	{
		h_Registration_Parameters_EPI_T1_Affine[p] = 0.0f;
	}

	// Allocate memory for EPI volume, T1 volume and EPI volume of T1 size
	d_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	d_T1_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to EPI volume and T1 volume
	clEnqueueWriteBuffer(commandQueue, d_EPI_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);

	// Make a segmentation of the EPI volume first
	SegmentEPIData(d_EPI_Volume);

	// Interpolate EPI volume to T1 resolution and make sure it has the same size
	ChangeEPIVolumeResolutionAndSize(d_T1_EPI_Volume, d_EPI_Volume, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, INTERPOLATION_MODE);

	// Copy the interpolated EPI T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_Interpolated_EPI_Volume, 0, NULL, NULL);

	// Calculate tensor magnitudes
	cl_mem d_T1_EPI_Tensor_Magnitude = clCreateBuffer(context, CL_MEM_READ_WRITE, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_T1_Tensor_Magnitude = clCreateBuffer(context, CL_MEM_READ_WRITE, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);

	CalculateTensorMagnitude(d_T1_EPI_Tensor_Magnitude, d_T1_EPI_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D);
	CalculateTensorMagnitude(d_T1_Tensor_Magnitude, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D);

	// Do the registration between EPI and skullstripped T1 with several scales, first translation
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Translation, h_Rotations, d_T1_EPI_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, TRANSLATION, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Tensor_Magnitude, h_Registration_Parameters_EPI_T1_Translation, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Translation);

	// Do the registration between EPI and skullstripped T1 with several scales, first translation
	//AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, COARSEST_SCALE_EPI_T1/2, NUMBER_OF_ITERATIONS_FOR_Linear_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);
	//TransformVolumesLinear(d_T1_EPI_Tensor_Magnitude, h_Registration_Parameters_EPI_T1_Rigid, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	//AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Rigid);


	// Rigid with tensor magnitudes
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Tensor_Magnitude, d_T1_Tensor_Magnitude, T1_DATA_W, T1_DATA_H, T1_DATA_D, 2, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Rigid, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Rigid);


	// Affine with tensor magnitudes
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Tensor_Magnitude, d_T1_Tensor_Magnitude, T1_DATA_W, T1_DATA_H, T1_DATA_D, 2, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Rigid, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Rigid);


	// Copy the aligned EPI volume to host
	clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_Aligned_EPI_Volume_T1, 0, NULL, NULL);

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

	clReleaseMemObject(d_T1_EPI_Tensor_Magnitude);
	clReleaseMemObject(d_T1_Tensor_Magnitude);
}


// Not fully optimized, T1 is of MNI size
void BROCCOLI_LIB::PerformRegistrationEPIT1()
{
	// Reset total registration parameters
	for (int p = 0; p < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; p++)
	{
		h_Registration_Parameters_EPI_T1_Affine[p] = 0.0f;
	}

	// Make a segmentation of the EPI volume first
	//SegmentEPIData(d_EPI_Volume);

	// Interpolate EPI volume to T1 resolution and make sure it has the same size,
	// the registration is performed to the skullstripped T1 volume, which has MNI size
	ChangeEPIVolumeResolutionAndSize(d_T1_EPI_Volume, d_EPI_Volume, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, INTERPOLATION_MODE);

	cl_mem d_T1_EPI_Tensor_Magnitude = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Skullstripped_T1_Tensor_Magnitude = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	CalculateTensorMagnitude(d_T1_EPI_Tensor_Magnitude, d_T1_EPI_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
	CalculateTensorMagnitude(d_Skullstripped_T1_Tensor_Magnitude, d_Skullstripped_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	// Do the registration between EPI and skullstripped T1 with several scales, first translation
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Translation, h_Rotations, d_T1_EPI_Volume, d_Skullstripped_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, TRANSLATION, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Tensor_Magnitude, h_Registration_Parameters_EPI_T1_Translation, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Translation);

	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Volume, d_Skullstripped_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Tensor_Magnitude, h_Registration_Parameters_EPI_T1_Rigid, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Rigid);

	//AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Translation, h_Rotations, d_T1_EPI_Volume, d_Skullstripped_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_Linear_IMAGE_REGISTRATION, TRANSLATION, DO_OVERWRITE, INTERPOLATION_MODE);
	//AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Translation, h_Rotations, d_T1_EPI_Tensor_Magnitude, d_Skullstripped_T1_Tensor_Magnitude, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_EPI_T1/2, NUMBER_OF_ITERATIONS_FOR_Linear_IMAGE_REGISTRATION, TRANSLATION, DO_OVERWRITE, INTERPOLATION_MODE);

	// Do the registration between EPI and skullstripped T1 with several scales, now rigid
	//AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Volume, d_Skullstripped_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_Linear_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);

	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Tensor_Magnitude, d_Skullstripped_T1_Tensor_Magnitude, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 2, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Rigid, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Rigid);

	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Tensor_Magnitude, d_Skullstripped_T1_Tensor_Magnitude, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 2, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Rigid, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Rigid);


	//ChangeEPIVolumeResolutionAndSize(d_T1_EPI_Volume, d_EPI_Volume, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, INTERPOLATION_MODE);
	//TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Affine, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

	//AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Volume, d_Skullstripped_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_EPI_T1/2, NUMBER_OF_ITERATIONS_FOR_Linear_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);

	// Get translations
	//h_Registration_Parameters_EPI_T1[0] = h_Registration_Parameters_EPI_T1_Affine[0];
	//h_Registration_Parameters_EPI_T1[1] = h_Registration_Parameters_EPI_T1_Affine[1];
	//h_Registration_Parameters_EPI_T1[2] = h_Registration_Parameters_EPI_T1_Affine[2];

	//AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine, h_Registration_Parameters_EPI_T1_Rigid, h_Registration_Parameters_EPI_T1_Translation);
	//AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine, h_Registration_Parameters_EPI_T1_Translation, h_Registration_Parameters_EPI_T1_Rigid);

	//TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Affine, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);



	h_Registration_Parameters_EPI_T1[0] = h_Registration_Parameters_EPI_T1_Translation[0] + h_Registration_Parameters_EPI_T1_Rigid[0];
	h_Registration_Parameters_EPI_T1[1] = h_Registration_Parameters_EPI_T1_Translation[1] + h_Registration_Parameters_EPI_T1_Rigid[1];
	h_Registration_Parameters_EPI_T1[2] = h_Registration_Parameters_EPI_T1_Translation[2] + h_Registration_Parameters_EPI_T1_Rigid[2];

	// Get rotations
	h_Registration_Parameters_EPI_T1[3] = h_Rotations[0];
	h_Registration_Parameters_EPI_T1[4] = h_Rotations[1];
	h_Registration_Parameters_EPI_T1[5] = h_Rotations[2];

	clReleaseMemObject(d_T1_EPI_Tensor_Magnitude);
	clReleaseMemObject(d_Skullstripped_T1_Tensor_Magnitude);
}


void BROCCOLI_LIB::PerformRegistrationEPIT1_()
{
	// Reset total registration parameters
	for (int p = 0; p < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; p++)
	{
		h_Registration_Parameters_EPI_T1_Affine[p] = 0.0f;
	}

	// Make a segmentation of the EPI volume first
	//SegmentEPIData(d_EPI_Volume);

	// Interpolate EPI volume to T1 resolution and make sure it has the same size,
	ChangeEPIVolumeResolutionAndSize(d_T1_EPI_Volume_, d_EPI_Volume, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, INTERPOLATION_MODE);

	cl_mem d_T1_EPI_Tensor_Magnitude = clCreateBuffer(context, CL_MEM_READ_WRITE, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Skullstripped_T1_Tensor_Magnitude = clCreateBuffer(context, CL_MEM_READ_WRITE, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);

	CalculateTensorMagnitude(d_T1_EPI_Tensor_Magnitude, d_T1_EPI_Volume_, T1_DATA_W, T1_DATA_H, T1_DATA_D);
	CalculateTensorMagnitude(d_Skullstripped_T1_Tensor_Magnitude, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D);

	// Do the registration between EPI and skullstripped T1 with several scales, first translation
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Translation, h_Rotations, d_T1_EPI_Volume_, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, TRANSLATION, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Tensor_Magnitude, h_Registration_Parameters_EPI_T1_Translation, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Translation);

	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Volume_, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Tensor_Magnitude, h_Registration_Parameters_EPI_T1_Rigid, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Rigid);

	//AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Translation, h_Rotations, d_T1_EPI_Volume, d_Skullstripped_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_Linear_IMAGE_REGISTRATION, TRANSLATION, DO_OVERWRITE, INTERPOLATION_MODE);
	//AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Translation, h_Rotations, d_T1_EPI_Tensor_Magnitude, d_Skullstripped_T1_Tensor_Magnitude, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_EPI_T1/2, NUMBER_OF_ITERATIONS_FOR_Linear_IMAGE_REGISTRATION, TRANSLATION, DO_OVERWRITE, INTERPOLATION_MODE);

	// Do the registration between EPI and skullstripped T1 with several scales, now rigid
	//AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Volume, d_Skullstripped_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_Linear_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);

	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Tensor_Magnitude, d_Skullstripped_T1_Tensor_Magnitude, T1_DATA_W, T1_DATA_H, T1_DATA_D, 2, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Rigid, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Rigid);

	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Tensor_Magnitude, d_Skullstripped_T1_Tensor_Magnitude, T1_DATA_W, T1_DATA_H, T1_DATA_D, 2, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Rigid, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Rigid);


	//ChangeEPIVolumeResolutionAndSize(d_T1_EPI_Volume, d_EPI_Volume, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, INTERPOLATION_MODE);
	//TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Affine, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

	//AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Volume, d_Skullstripped_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_EPI_T1/2, NUMBER_OF_ITERATIONS_FOR_Linear_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);

	// Get translations
	//h_Registration_Parameters_EPI_T1[0] = h_Registration_Parameters_EPI_T1_Affine[0];
	//h_Registration_Parameters_EPI_T1[1] = h_Registration_Parameters_EPI_T1_Affine[1];
	//h_Registration_Parameters_EPI_T1[2] = h_Registration_Parameters_EPI_T1_Affine[2];

	//AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine, h_Registration_Parameters_EPI_T1_Rigid, h_Registration_Parameters_EPI_T1_Translation);
	//AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine, h_Registration_Parameters_EPI_T1_Translation, h_Registration_Parameters_EPI_T1_Rigid);

	//TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Affine, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);



//	h_Registration_Parameters_EPI_T1[0] = h_Registration_Parameters_EPI_T1_Translation[0] + h_Registration_Parameters_EPI_T1_Rigid[0];
//	h_Registration_Parameters_EPI_T1[1] = h_Registration_Parameters_EPI_T1_Translation[1] + h_Registration_Parameters_EPI_T1_Rigid[1];
//	h_Registration_Parameters_EPI_T1[2] = h_Registration_Parameters_EPI_T1_Translation[2] + h_Registration_Parameters_EPI_T1_Rigid[2];

	// Get rotations
//	h_Registration_Parameters_EPI_T1[3] = h_Rotations[0];
//	h_Registration_Parameters_EPI_T1[4] = h_Rotations[1];
//	h_Registration_Parameters_EPI_T1[5] = h_Rotations[2];

	clReleaseMemObject(d_T1_EPI_Tensor_Magnitude);
	clReleaseMemObject(d_Skullstripped_T1_Tensor_Magnitude);
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
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_T1_MNI_Out, h_Rotations, d_MNI_T1_Volume, d_MNI_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, NO_OVERWRITE, INTERPOLATION_MODE);

	clReleaseMemObject(d_MNI_Volume);

	// Copy the aligned T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_Linear, 0, NULL, NULL);

	// Allocate memory for MNI brain mask
	d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy MNI brain mask to device
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

	// Calculate inverse transform between T1 and MNI
	InvertAffineRegistrationParameters(h_Inverse_Registration_Parameters, h_Registration_Parameters_T1_MNI_Out);

	// Now apply the inverse transformation between MNI and T1, to transform MNI brain mask to original T1 space
	TransformVolumesLinear(d_MNI_Brain_Mask, h_Inverse_Registration_Parameters, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, NEAREST);

	// Create skullstripped volume, by multiplying original T1 volume with transformed MNI brain mask
	MultiplyVolumes(d_MNI_T1_Volume, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);


	d_MNI_Brain_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Volume , 0, NULL, NULL);


	// Now align skullstripped volume with MNI brain
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_T1_MNI_Out, h_Rotations, d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, NO_OVERWRITE, INTERPOLATION_MODE);

	// Copy MNI brain mask to device again
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

	// Calculate inverse transform between T1 and MNI (to get better skullstrip)
	InvertAffineRegistrationParameters(h_Inverse_Registration_Parameters, h_Registration_Parameters_T1_MNI_Out);

	// Apply inverse transform
	TransformVolumesLinear(d_MNI_Brain_Mask, h_Inverse_Registration_Parameters, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, NEAREST);

	// Multiply inverse transformed mask with original volume (to get better skullstrip)
	MultiplyVolumes(d_MNI_T1_Volume, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	// Copy the skullstripped T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Skullstripped_T1_Volume, 0, NULL, NULL);

	// Apply forward transform to skullstripped volume
	TransformVolumesLinear(d_MNI_T1_Volume, h_Registration_Parameters_T1_MNI_Out, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_Linear, 0, NULL, NULL);

	// Perform non-Linear registration between tramsformed skullstripped volume and MNI brain volume
	AlignTwoVolumesNonLinearSeveralScales(d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION, DO_OVERWRITE, INTERPOLATION_MODE, DISCARD_DISPLACEMENT_FIELD);

	// Copy the aligned T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_NonLinear, 0, NULL, NULL);

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

	// Do Linear registration between T1 and MNI with several scales (without skull)
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_T1_MNI_Out, h_Rotations, d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);

	// Copy the aligned T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_Linear, 0, NULL, NULL);


	// Perform non-Linear registration between tramsformed skullstripped volume and MNI brain volume
	//if (NUMBER_OF_ITERATIONS_FOR_NONLinear_IMAGE_REGISTRATION > 0)
	//{
		AlignTwoVolumesNonLinearSeveralScales(d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION, DO_OVERWRITE, INTERPOLATION_MODE, DISCARD_DISPLACEMENT_FIELD);
	//}

	// Copy the aligned T1 volume to host
	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_NonLinear, 0, NULL, NULL);

	// Cleanup
	clReleaseMemObject(d_MNI_Brain_Volume);
	clReleaseMemObject(d_MNI_T1_Volume);
}

void BROCCOLI_LIB::PerformRegistrationTwoVolumesWrapper()
{
	// Allocate memory for input volume, input volume of reference size and referencec volume
	cl_mem d_Input_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Reference_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Input_Volume_Reference_Size = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
    clEnqueueWriteBuffer(commandQueue, d_Input_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);
    clEnqueueWriteBuffer(commandQueue, d_Reference_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Volume , 0, NULL, NULL);

    // Change resolution and size of input volume
    ChangeVolumesResolutionAndSize(d_Input_Volume_Reference_Size, d_Input_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_T1_Z_CUT, INTERPOLATION_MODE);

	// Copy the interpolated volume to host
	clEnqueueReadBuffer(commandQueue, d_Input_Volume_Reference_Size, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Interpolated_T1_Volume, 0, NULL, NULL);

	// Do Linear registration between the two volumes
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_T1_MNI_Out, h_Rotations, d_Input_Volume_Reference_Size, d_Reference_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);

	// Copy the linearly aligned volume to host
	clEnqueueReadBuffer(commandQueue, d_Input_Volume_Reference_Size, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_Linear, 0, NULL, NULL);

	// Perform non-Linear registration between tramsformed skullstripped volume and MNI brain volume
	if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
	{
		AlignTwoVolumesNonLinearSeveralScales(d_Input_Volume_Reference_Size, d_Reference_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION, DO_OVERWRITE, INTERPOLATION_MODE, KEEP_DISPLACEMENT_FIELD);
	}

	CreateCombinedDisplacementField(h_Registration_Parameters_T1_MNI_Out,d_Total_Displacement_Field_X,d_Total_Displacement_Field_Y,d_Total_Displacement_Field_Z,MNI_DATA_W,MNI_DATA_H,MNI_DATA_D);

	// Copy the non-linearly aligned volume to host
	clEnqueueReadBuffer(commandQueue, d_Input_Volume_Reference_Size, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_NonLinear, 0, NULL, NULL);

	if (DO_SKULLSTRIP)
	{
		// Copy brain mask from host
		cl_mem d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
		clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask, 0, NULL, NULL);

		// Multiply the non-linearly aligned volume with the brain mask
		MultiplyVolumes(d_Input_Volume_Reference_Size, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

		// Copy the skullstripped volume to host
		clEnqueueReadBuffer(commandQueue, d_Input_Volume_Reference_Size, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Skullstripped_T1_Volume, 0, NULL, NULL);

		clReleaseMemObject(d_MNI_Brain_Mask);
	}

	// Copy the displacement field to host
	clEnqueueReadBuffer(commandQueue, d_Total_Displacement_Field_X, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Displacement_Field_X, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Total_Displacement_Field_Y, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Displacement_Field_Y, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Total_Displacement_Field_Z, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Displacement_Field_Z, 0, NULL, NULL);

	// Cleanup
	clReleaseMemObject(d_Input_Volume);
	clReleaseMemObject(d_Input_Volume_Reference_Size);
	clReleaseMemObject(d_Reference_Volume);

	clReleaseMemObject(d_Total_Displacement_Field_X);
	clReleaseMemObject(d_Total_Displacement_Field_Y);
	clReleaseMemObject(d_Total_Displacement_Field_Z);
}

void BROCCOLI_LIB::CreateCombinedDisplacementField(float* h_Registration_Parameters_,
		                                           cl_mem d_Displacement_Field_X,
		                                           cl_mem d_Displacement_Field_Y,
		                                           cl_mem d_Displacement_Field_Z,
		                                           int DATA_W,
		                                           int DATA_H,
		                                           int DATA_D)
{
	SetGlobalAndLocalWorkSizesInterpolateVolume(DATA_W, DATA_H, DATA_D);

	// Allocate memory for linear registration parameters
	c_Registration_Parameters = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), NULL, &createBufferErrorRegistrationParameters);

	// Copy linear registration parameters to device
	clEnqueueWriteBuffer(commandQueue, c_Registration_Parameters, CL_TRUE, 0, NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float), h_Registration_Parameters_, 0, NULL, NULL);

	// Set all kernel arguments
	clSetKernelArg(AddLinearAndNonLinearDisplacementKernel, 0, sizeof(cl_mem), &d_Displacement_Field_X);
	clSetKernelArg(AddLinearAndNonLinearDisplacementKernel, 1, sizeof(cl_mem), &d_Displacement_Field_Y);
	clSetKernelArg(AddLinearAndNonLinearDisplacementKernel, 2, sizeof(cl_mem), &d_Displacement_Field_Z);
	clSetKernelArg(AddLinearAndNonLinearDisplacementKernel, 3, sizeof(cl_mem), &c_Registration_Parameters);
	clSetKernelArg(AddLinearAndNonLinearDisplacementKernel, 4, sizeof(int),    &DATA_W);
	clSetKernelArg(AddLinearAndNonLinearDisplacementKernel, 5, sizeof(int),    &DATA_H);
	clSetKernelArg(AddLinearAndNonLinearDisplacementKernel, 6, sizeof(int),    &DATA_D);

	runKernelErrorAddLinearAndNonLinearDisplacement = clEnqueueNDRangeKernel(commandQueue, AddLinearAndNonLinearDisplacementKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);

	clReleaseMemObject(c_Registration_Parameters);
}

void BROCCOLI_LIB::TransformVolumesNonLinearWrapper()
{
	// Allocate memory for volume and displacement field
	cl_mem d_Input_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Input_Volume_Reference_Size = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Total_Displacement_Field_X = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Total_Displacement_Field_Y = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Total_Displacement_Field_Z = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_Input_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_Total_Displacement_Field_X, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Displacement_Field_X , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_Total_Displacement_Field_Y, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Displacement_Field_Y , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_Total_Displacement_Field_Z, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Displacement_Field_Z , 0, NULL, NULL);

	// Change resolution and size of input volume
	ChangeVolumesResolutionAndSize(d_Input_Volume_Reference_Size, d_Input_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_T1_Z_CUT, INTERPOLATION_MODE);

	// Apply the transformation
	TransformVolumesNonLinear(d_Input_Volume_Reference_Size, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

	// Copy the transformed volume to host
	clEnqueueReadBuffer(commandQueue, d_Input_Volume_Reference_Size, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Interpolated_T1_Volume, 0, NULL, NULL);

	// Release memory
	clReleaseMemObject(d_Input_Volume);
	clReleaseMemObject(d_Input_Volume_Reference_Size);
	clReleaseMemObject(d_Total_Displacement_Field_X);
	clReleaseMemObject(d_Total_Displacement_Field_Y);
	clReleaseMemObject(d_Total_Displacement_Field_Z);
}



// Performs registration between one high resolution T1 volume and a high resolution MNI volume (brain template)
void BROCCOLI_LIB::PerformRegistrationT1MNI()
{
	// Interpolate T1 volume to MNI resolution and make sure it has the same size
	ChangeT1VolumeResolutionAndSize(d_MNI_T1_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, INTERPOLATION_MODE, NOT_SKULL_STRIPPED);

	// Do the registration between T1 and MNI with several scales
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_T1_MNI, h_Rotations, d_MNI_T1_Volume, d_MNI_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, NO_OVERWRITE, INTERPOLATION_MODE);

	// Calculate inverse transform between T1 and MNI
	InvertAffineRegistrationParameters(h_Inverse_Registration_Parameters, h_Registration_Parameters_T1_MNI);

	// Now apply the inverse transformation between MNI and T1, to transform MNI brain mask to T1 space
	TransformVolumesLinear(d_MNI_Brain_Mask, h_Inverse_Registration_Parameters, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, NEAREST);

	// Create skullstripped volume, by multiplying original T1 volume with transformed MNI brain mask
	MultiplyVolumes(d_Skullstripped_T1_Volume, d_MNI_T1_Volume, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	//----------

	// Now align skullstripped volume with MNI template without skull
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_T1_MNI, h_Rotations, d_Skullstripped_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, NO_OVERWRITE, INTERPOLATION_MODE);

	// Copy MNI brain mask to device again
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

	// Calculate inverse transform between T1 and MNI (to get better skullstrip)
	InvertAffineRegistrationParameters(h_Inverse_Registration_Parameters, h_Registration_Parameters_T1_MNI);

	// Apply inverse transform to mask
	TransformVolumesLinear(d_MNI_Brain_Mask, h_Inverse_Registration_Parameters, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, NEAREST);

	// Multiply inverse transformed mask with original volume (to get better skullstrip)
	MultiplyVolumes(d_MNI_T1_Volume, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	// Apply forward transform to skullstripped volume
	TransformVolumesLinear(d_MNI_T1_Volume, h_Registration_Parameters_T1_MNI_Out, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

	// Finally perform non-Linear registration between transformed skullstripped volume and MNI brain volume
	AlignTwoVolumesNonLinearSeveralScales(d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION, DO_OVERWRITE, INTERPOLATION_MODE, DISCARD_DISPLACEMENT_FIELD);
}

// Performs registration between one high resolution skullstripped T1 volume and a high resolution skullstripped MNI volume (brain template)
void BROCCOLI_LIB::PerformRegistrationT1MNINoSkullstrip()
{
	// Interpolate T1 volume to MNI resolution and make sure it has the same size
	ChangeT1VolumeResolutionAndSize(d_MNI_T1_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, INTERPOLATION_MODE, SKULL_STRIPPED);
	ChangeT1VolumeResolutionAndSize(d_Skullstripped_T1_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, INTERPOLATION_MODE, SKULL_STRIPPED);

	if (WRITE_INTERPOLATED_T1)
	{
		clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Interpolated_T1_Volume, 0, NULL, NULL);
	}

	// Do Linear registration between T1 and MNI with several scales (without skull)
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_T1_MNI, h_Rotations, d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);

	if (WRITE_ALIGNED_T1_MNI_LINEAR)
	{
		clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_Linear, 0, NULL, NULL);
	}

	// Perform non-Linear registration between registered skullstripped volume and MNI brain volume
	AlignTwoVolumesNonLinearSeveralScales(d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION, DO_OVERWRITE, INTERPOLATION_MODE, KEEP_DISPLACEMENT_FIELD);

	if (WRITE_ALIGNED_T1_MNI_NONLINEAR)
	{
		clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_NonLinear, 0, NULL, NULL);
	}

	//ChangeT1VolumeResolutionAndSize(d_MNI_T1_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, INTERPOLATION_MODE, SKULL_STRIPPED);
}



// Transforms volumes Linearally by applying a parameter vector
void BROCCOLI_LIB::TransformVolumesLinear(cl_mem d_Volumes,
		                                      float* h_Registration_Parameters_,
		                                      int DATA_W,
		                                      int DATA_H,
		                                      int DATA_D,
		                                      int NUMBER_OF_VOLUMES,
		                                      int INTERPOLATION_MODE)
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

	// Loop over volumes
	for (int volume = 0; volume < NUMBER_OF_VOLUMES; volume++)
	{
		// Copy current volume to texture
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {DATA_W, DATA_H, DATA_D};
		clEnqueueCopyBufferToImage(commandQueue, d_Volumes, d_Volume_Texture, volume * DATA_W * DATA_H * DATA_D * sizeof(float), origin, region, 0, NULL, NULL);

		// Interpolate to get the transformed volume
		if (INTERPOLATION_MODE == LINEAR)
		{
			clSetKernelArg(InterpolateVolumeLinearLinearKernel, 0, sizeof(cl_mem), &d_Volumes);
			clSetKernelArg(InterpolateVolumeLinearLinearKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
			clSetKernelArg(InterpolateVolumeLinearLinearKernel, 2, sizeof(cl_mem), &c_Parameters);
			clSetKernelArg(InterpolateVolumeLinearLinearKernel, 3, sizeof(int), &DATA_W);
			clSetKernelArg(InterpolateVolumeLinearLinearKernel, 4, sizeof(int), &DATA_H);
			clSetKernelArg(InterpolateVolumeLinearLinearKernel, 5, sizeof(int), &DATA_D);
			clSetKernelArg(InterpolateVolumeLinearLinearKernel, 6, sizeof(int), &volume);
			runKernelErrorInterpolateVolumeLinearLinear = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
			clFinish(commandQueue);
		}
		else if (INTERPOLATION_MODE == CUBIC)
		{
			clSetKernelArg(InterpolateVolumeCubicLinearKernel, 0, sizeof(cl_mem), &d_Volumes);
			clSetKernelArg(InterpolateVolumeCubicLinearKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
			clSetKernelArg(InterpolateVolumeCubicLinearKernel, 2, sizeof(cl_mem), &c_Parameters);
			clSetKernelArg(InterpolateVolumeCubicLinearKernel, 3, sizeof(int), &DATA_W);
			clSetKernelArg(InterpolateVolumeCubicLinearKernel, 4, sizeof(int), &DATA_H);
			clSetKernelArg(InterpolateVolumeCubicLinearKernel, 5, sizeof(int), &DATA_D);
			clSetKernelArg(InterpolateVolumeCubicLinearKernel, 6, sizeof(int), &volume);
			runKernelErrorInterpolateVolumeCubicLinear = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeCubicLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
			clFinish(commandQueue);
		}
		else if (INTERPOLATION_MODE == NEAREST)
		{
			clSetKernelArg(InterpolateVolumeNearestLinearKernel, 0, sizeof(cl_mem), &d_Volumes);
			clSetKernelArg(InterpolateVolumeNearestLinearKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
			clSetKernelArg(InterpolateVolumeNearestLinearKernel, 2, sizeof(cl_mem), &c_Parameters);
			clSetKernelArg(InterpolateVolumeNearestLinearKernel, 3, sizeof(int), &DATA_W);
			clSetKernelArg(InterpolateVolumeNearestLinearKernel, 4, sizeof(int), &DATA_H);
			clSetKernelArg(InterpolateVolumeNearestLinearKernel, 5, sizeof(int), &DATA_D);
			clSetKernelArg(InterpolateVolumeNearestLinearKernel, 6, sizeof(int), &volume);
			runKernelErrorInterpolateVolumeNearestLinear = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeNearestLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
			clFinish(commandQueue);
		}
	}

	clReleaseMemObject(d_Volume_Texture);
	clReleaseMemObject(c_Parameters);
}




// Transforms volumes non-Linearally by applying a displacement field
void BROCCOLI_LIB::TransformVolumesNonLinear(cl_mem d_Volumes,
		                                         cl_mem d_Displacement_Field_X,
		                                         cl_mem d_Displacement_Field_Y,
		                                         cl_mem d_Displacement_Field_Z,
		                                         int DATA_W,
		                                         int DATA_H,
		                                         int DATA_D,
		                                         int NUMBER_OF_VOLUMES,
		                                         int INTERPOLATION_MODE)
{
	// Allocate memory for texture
	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_INTENSITY;
	cl_mem d_Volume_Texture = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, DATA_W, DATA_H, DATA_D, 0, 0, NULL, NULL);

	SetGlobalAndLocalWorkSizesInterpolateVolume(DATA_W, DATA_H, DATA_D);

	// Transform all volumes
	for (int volume = 0; volume < NUMBER_OF_VOLUMES; volume++)
	{
		// Copy current volume to texture
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {DATA_W, DATA_H, DATA_D};
		clEnqueueCopyBufferToImage(commandQueue, d_Volumes, d_Volume_Texture, volume * DATA_W * DATA_H * DATA_D * sizeof(float), origin, region, 0, NULL, NULL);

		// Interpolate to get the transformed volume
		if (INTERPOLATION_MODE == LINEAR)
		{
			clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 0, sizeof(cl_mem), &d_Volumes);
			clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
			clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 2, sizeof(cl_mem), &d_Displacement_Field_X);
			clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 3, sizeof(cl_mem), &d_Displacement_Field_Y);
			clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 4, sizeof(cl_mem), &d_Displacement_Field_Z);
			clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 5, sizeof(int), &DATA_W);
			clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 6, sizeof(int), &DATA_H);
			clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 7, sizeof(int), &DATA_D);
			clSetKernelArg(InterpolateVolumeLinearNonLinearKernel, 8, sizeof(int), &volume);
			runKernelErrorInterpolateVolumeLinearNonLinear = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeLinearNonLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
			clFinish(commandQueue);
		}
		else if (INTERPOLATION_MODE == CUBIC)
		{
			clSetKernelArg(InterpolateVolumeCubicNonLinearKernel, 0, sizeof(cl_mem), &d_Volumes);
			clSetKernelArg(InterpolateVolumeCubicNonLinearKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
			clSetKernelArg(InterpolateVolumeCubicNonLinearKernel, 2, sizeof(cl_mem), &d_Displacement_Field_X);
			clSetKernelArg(InterpolateVolumeCubicNonLinearKernel, 3, sizeof(cl_mem), &d_Displacement_Field_Y);
			clSetKernelArg(InterpolateVolumeCubicNonLinearKernel, 4, sizeof(cl_mem), &d_Displacement_Field_Z);
			clSetKernelArg(InterpolateVolumeCubicNonLinearKernel, 5, sizeof(int), &DATA_W);
			clSetKernelArg(InterpolateVolumeCubicNonLinearKernel, 6, sizeof(int), &DATA_H);
			clSetKernelArg(InterpolateVolumeCubicNonLinearKernel, 7, sizeof(int), &DATA_D);
			clSetKernelArg(InterpolateVolumeCubicNonLinearKernel, 8, sizeof(int), &volume);
			runKernelErrorInterpolateVolumeCubicNonLinear = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeCubicNonLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
			clFinish(commandQueue);
		}
		else if (INTERPOLATION_MODE == NEAREST)
		{
			clSetKernelArg(InterpolateVolumeNearestNonLinearKernel, 0, sizeof(cl_mem), &d_Volumes);
			clSetKernelArg(InterpolateVolumeNearestNonLinearKernel, 1, sizeof(cl_mem), &d_Volume_Texture);
			clSetKernelArg(InterpolateVolumeNearestNonLinearKernel, 2, sizeof(cl_mem), &d_Displacement_Field_X);
			clSetKernelArg(InterpolateVolumeNearestNonLinearKernel, 3, sizeof(cl_mem), &d_Displacement_Field_Y);
			clSetKernelArg(InterpolateVolumeNearestNonLinearKernel, 4, sizeof(cl_mem), &d_Displacement_Field_Z);
			clSetKernelArg(InterpolateVolumeNearestNonLinearKernel, 5, sizeof(int), &DATA_W);
			clSetKernelArg(InterpolateVolumeNearestNonLinearKernel, 6, sizeof(int), &DATA_H);
			clSetKernelArg(InterpolateVolumeNearestNonLinearKernel, 7, sizeof(int), &DATA_D);
			clSetKernelArg(InterpolateVolumeNearestNonLinearKernel, 8, sizeof(int), &volume);
			runKernelErrorInterpolateVolumeNearestNonLinear = clEnqueueNDRangeKernel(commandQueue, InterpolateVolumeNearestNonLinearKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
			clFinish(commandQueue);
		}
	}

	clReleaseMemObject(d_Volume_Texture);
}

void BROCCOLI_LIB::PerformFirstLevelAnalysisWrapper()
{
	//------------------------
	// T1-MNI registration
	//------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("\nAligning T1 to MNI\n");
	}

	// Allocate memory on device
	d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	//d_MNI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Brain_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Skullstripped_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);
	//clEnqueueWriteBuffer(commandQueue, d_MNI_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Volume , 0, NULL, NULL);
	//clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

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
	//clReleaseMemObject(d_MNI_Volume);
	clReleaseMemObject(d_MNI_Brain_Volume);

	//clReleaseMemObject(d_MNI_T1_Volume);
	//clReleaseMemObject(d_Skullstripped_T1_Volume);
	//clReleaseMemObject(d_MNI_Brain_Mask);

	//------------------------
	// fMRI-T1 registration
	//------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("Aligning fMRI to T1\n");
	}

	// Allocate memory on device
	d_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_T1_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_EPI_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

	PerformRegistrationEPIT1();

	if (WRITE_ALIGNED_EPI_T1)
	{
		clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_EPI_Volume_T1, 0, NULL, NULL);
	}

	if (WRITE_ALIGNED_EPI_MNI)
	{
		TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_T1_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesNonLinear(d_T1_EPI_Volume, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_EPI_Volume_MNI, 0, NULL, NULL);
	}

	clReleaseMemObject(d_MNI_T1_Volume);
	clReleaseMemObject(d_Skullstripped_T1_Volume);
	clReleaseMemObject(d_T1_EPI_Volume);

	h_Registration_Parameters_EPI_T1_Out[0] = h_Registration_Parameters_EPI_T1[0];
	h_Registration_Parameters_EPI_T1_Out[1] = h_Registration_Parameters_EPI_T1[1];
	h_Registration_Parameters_EPI_T1_Out[2] = h_Registration_Parameters_EPI_T1[2];
	h_Registration_Parameters_EPI_T1_Out[3] = h_Registration_Parameters_EPI_T1[3];
	h_Registration_Parameters_EPI_T1_Out[4] = h_Registration_Parameters_EPI_T1[4];
	h_Registration_Parameters_EPI_T1_Out[5] = h_Registration_Parameters_EPI_T1[5];

	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_MNI,h_Registration_Parameters_T1_MNI,h_Registration_Parameters_EPI_T1_Affine);

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

	//------------------------

	// Allocate memory for fMRI volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

	//------------------------
	// Slice timing correction
	//------------------------

	//d_Slice_Timing_Corrected_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	//PerformSliceTimingCorrection();

	if (WRITE_SLICETIMING_CORRECTED)
	{
		clEnqueueReadBuffer(commandQueue, d_Slice_Timing_Corrected_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Slice_Timing_Corrected_fMRI_Volumes, 0, NULL, NULL);
	}

	//------------------------
	// Motion correction
	//------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("Performing motion correction\n");
	}

	d_Motion_Corrected_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	PerformMotionCorrection(d_fMRI_Volumes);

	//PerformMotionCorrection(d_Slice_Timing_Corrected_fMRI_Volumes);
	//clReleaseMemObject(d_Slice_Timing_Corrected_fMRI_Volumes);

	// Copy motion parameters
	for (int t = 0; t < EPI_DATA_T; t++)
	{
		for (int p = 0; p < 6; p++)
		{
			h_Motion_Parameters_Out[t + p * EPI_DATA_T] = h_Motion_Parameters[t + p * EPI_DATA_T];
		}
	}

	if (WRITE_MOTION_CORRECTED)
	{
		clEnqueueReadBuffer(commandQueue, d_Motion_Corrected_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Motion_Corrected_fMRI_Volumes, 0, NULL, NULL);
	}

	//------------------------
	// Segment EPI data
	//------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("Segmenting EPI data\n");
	}

	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Smoothed_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	SegmentEPIData();

	if (WRITE_EPI_MASK)
	{
		clEnqueueReadBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask, 0, NULL, NULL);
	}

	// Now safe to free original data
	clReleaseMemObject(d_fMRI_Volumes);

	//------------------------
	// Smoothing
	//------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("Performing smoothing\n");
	}

	d_Smoothed_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, EPI_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	//PerformSmoothing(d_Smoothed_fMRI_Volumes, d_Motion_Corrected_fMRI_Volumes, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	PerformSmoothing(d_Smoothed_EPI_Mask, d_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
	PerformSmoothingNormalized(d_Smoothed_fMRI_Volumes, d_Motion_Corrected_fMRI_Volumes, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

	// Now safe to remove motion corrected volumes
	clReleaseMemObject(d_Motion_Corrected_fMRI_Volumes);

	if (WRITE_SMOOTHED)
	{
		clEnqueueReadBuffer(commandQueue, d_Smoothed_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Smoothed_fMRI_Volumes, 0, NULL, NULL);
	}


	//------------------------
	// GLM
	//------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("Performing statistical analysis\n");
	}

	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + NUMBER_OF_CONFOUND_REGRESSORS*REGRESS_CONFOUNDS;

	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Contrast_Volumes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
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

	h_X_GLM_With_Temporal_Derivatives = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * 2 * EPI_DATA_T * sizeof(float));
	h_X_GLM_Convolved = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * (USE_TEMPORAL_DERIVATIVES+1) * EPI_DATA_T * sizeof(float));


	SetupTTestFirstLevel(EPI_DATA_T);

	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM , 0, NULL, NULL);

	if (WRITE_DESIGNMATRIX)
	{
		for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
		{
			for (int t = 0; t < EPI_DATA_T; t++)
			{
				h_X_GLM_Out[t + r * EPI_DATA_T] = h_X_GLM[t + r * EPI_DATA_T];
				h_xtxxt_GLM_Out[t + r * EPI_DATA_T] = h_xtxxt_GLM[t + r * EPI_DATA_T];
			}
		}
	}

	CalculateStatisticalMapsGLMTTestFirstLevel(d_Smoothed_fMRI_Volumes,3);

	// Copy data in EPI space to host

	if (WRITE_ACTIVITY_EPI)
	{
		clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_EPI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Contrast_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrast_Volumes_EPI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_EPI, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);
	}

	if (WRITE_AR_ESTIMATES_EPI)
	{
		clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);
	}

	//------------------------------------------------
	// Transform results to T1 space and copy to host
	//------------------------------------------------

	if (WRITE_ACTIVITY_T1)
	{
		if ((WRAPPER == BASH) && PRINT)
		{
			printf("Transforming results to T1\n");
		}

		d_T1_EPI_Volume_ = clCreateBuffer(context, CL_MEM_READ_WRITE, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
		PerformRegistrationEPIT1_();
		clReleaseMemObject(d_T1_EPI_Volume_);

		// Allocate memory on device
		d_Beta_Volumes_T1 = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, &createBufferErrorBetaVolumesT1);
		d_Contrast_Volumes_T1 = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, &createBufferErrorContrastVolumesT1);
		d_Statistical_Maps_T1 = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, &createBufferErrorStatisticalMapsT1);
		//d_Residual_Variances_T1 = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, &createBufferErrorResidualVariancesT1);

		if (WRITE_AR_ESTIMATES_T1)
		{
			d_AR1_Estimates_T1 = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, &createBufferErrorAREstimatesT1);
			d_AR2_Estimates_T1 = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, &createBufferErrorAREstimatesT1);
			d_AR3_Estimates_T1 = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, &createBufferErrorAREstimatesT1);
			d_AR4_Estimates_T1 = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, &createBufferErrorAREstimatesT1);
		}

		// Apply transformation
		TransformFirstLevelResultsToT1();

		// Copy to host
		clEnqueueReadBuffer(commandQueue, d_Beta_Volumes_T1, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_T1, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Contrast_Volumes_T1, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrast_Volumes_T1, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Statistical_Maps_T1, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_T1, 0, NULL, NULL);
		//clEnqueueReadBuffer(commandQueue, d_Residual_Variances_T1, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_Residual_Variances_T1, 0, NULL, NULL);

		// Free memory
		clReleaseMemObject(d_Beta_Volumes_T1);
		clReleaseMemObject(d_Contrast_Volumes_T1);
		clReleaseMemObject(d_Statistical_Maps_T1);
		clReleaseMemObject(d_Residual_Variances_T1);

		if (WRITE_AR_ESTIMATES_T1)
		{
			clReleaseMemObject(d_AR1_Estimates_T1);
			clReleaseMemObject(d_AR2_Estimates_T1);
			clReleaseMemObject(d_AR3_Estimates_T1);
			clReleaseMemObject(d_AR4_Estimates_T1);
		}
	}
	clReleaseMemObject(d_EPI_Volume);
	clReleaseMemObject(d_T1_Volume);


	//------------------------------------------------
	// Transform results to MNI space and copy to host
	//------------------------------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("Transforming results to MNI\n");
	}

	// Allocate memory on device
	d_Beta_Volumes_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, &createBufferErrorBetaVolumesMNI);
	d_Contrast_Volumes_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, &createBufferErrorContrastVolumesMNI);
	d_Statistical_Maps_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, &createBufferErrorStatisticalMapsMNI);
	d_Residual_Variances_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, &createBufferErrorResidualVariancesMNI);

	if (WRITE_AR_ESTIMATES_MNI)
	{
		d_AR1_Estimates_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, &createBufferErrorAREstimatesMNI);
		d_AR2_Estimates_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, &createBufferErrorAREstimatesMNI);
		d_AR3_Estimates_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, &createBufferErrorAREstimatesMNI);
		d_AR4_Estimates_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, &createBufferErrorAREstimatesMNI);
	}

	//clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

	SetMemory(d_MNI_Brain_Mask, 1.0f, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D);

	// Apply transformations
	TransformFirstLevelResultsToMNI();

	// Copy to host
	clEnqueueReadBuffer(commandQueue, d_Beta_Volumes_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_MNI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Contrast_Volumes_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrast_Volumes_MNI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_Residual_Variances_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Residual_Variances_MNI, 0, NULL, NULL);

	if (WRITE_AR_ESTIMATES_MNI)
	{
		clEnqueueReadBuffer(commandQueue, d_AR1_Estimates_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_AR1_Estimates_MNI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_AR2_Estimates_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_AR2_Estimates_MNI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_AR3_Estimates_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_AR3_Estimates_MNI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_AR4_Estimates_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_AR4_Estimates_MNI, 0, NULL, NULL);
	}


	if (WRITE_UNWHITENED_RESULTS)
	{
		// Now calculate maps without whitening
		CalculateStatisticalMapsGLMTTestFirstLevel(d_Smoothed_fMRI_Volumes,0);

		// Copy data to host
		if (WRITE_ACTIVITY_EPI)
		{
			clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_No_Whitening_EPI, 0, NULL, NULL);
			clEnqueueReadBuffer(commandQueue, d_Contrast_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrast_Volumes_No_Whitening_EPI, 0, NULL, NULL);
			clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_No_Whitening_EPI, 0, NULL, NULL);
			//clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);
		}

		// Apply transformations
		TransformFirstLevelResultsToMNI();

		// Copy to host
		clEnqueueReadBuffer(commandQueue, d_Beta_Volumes_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_No_Whitening_MNI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Contrast_Volumes_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrast_Volumes_No_Whitening_MNI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Statistical_Maps_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_No_Whitening_MNI, 0, NULL, NULL);
	}


	clReleaseMemObject(d_Beta_Volumes_MNI);
	clReleaseMemObject(d_Contrast_Volumes_MNI);
	clReleaseMemObject(d_Statistical_Maps_MNI);
	clReleaseMemObject(d_Residual_Variances_MNI);

	if (WRITE_AR_ESTIMATES_MNI)
	{
		clReleaseMemObject(d_AR1_Estimates_MNI);
		clReleaseMemObject(d_AR2_Estimates_MNI);
		clReleaseMemObject(d_AR3_Estimates_MNI);
		clReleaseMemObject(d_AR4_Estimates_MNI);
	}


	//------------------------
	// Run permutation test
	//------------------------

	if (PERMUTE_FIRST_LEVEL)
	{
		if ((WRAPPER == BASH) && PRINT)
		{
			printf("Running permutation test\n");
		}

		// Allocate temporary memory
		d_Detrended_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
		d_Permuted_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
		d_Cluster_Indices = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int), NULL, NULL);
		d_Cluster_Sizes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int), NULL, NULL);
		d_TFCE_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int), NULL, NULL);
		d_P_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
		c_Permutation_Vector = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(unsigned short int), NULL, NULL);
		c_Permutation_Distribution = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_PERMUTATIONS * sizeof(float), NULL, NULL);

		ApplyPermutationTestFirstLevel(d_Smoothed_fMRI_Volumes); // Right now no smoothing in each permutation

		// Calculate activity map without Cochrane-Orcutt
		CalculateStatisticalMapsGLMTTestFirstLevel(d_Smoothed_fMRI_Volumes,0);

		// Clusterize original statistical maps
		if (INFERENCE_MODE != VOXEL)
		{
			ClusterizeOpenCL(d_Cluster_Indices, d_Cluster_Sizes, MAX_CLUSTER, d_Statistical_Maps, CLUSTER_DEFINING_THRESHOLD, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
		}

		// Calculate permutation p-values
		CalculatePermutationPValues(d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

		// Free temporary memory
		clReleaseMemObject(d_Detrended_fMRI_Volumes);
		clReleaseMemObject(d_Permuted_fMRI_Volumes);
		clReleaseMemObject(d_Cluster_Indices);
		clReleaseMemObject(d_Cluster_Sizes);
		clReleaseMemObject(d_TFCE_Values);
		clReleaseMemObject(c_Permutation_Vector);
		clReleaseMemObject(c_Permutation_Distribution);
	}

	free(h_X_GLM);
	free(h_xtxxt_GLM);
	free(h_Contrasts);
	free(h_ctxtxc_GLM);
	free(h_X_GLM_With_Temporal_Derivatives);
	free(h_X_GLM_Convolved);

	clReleaseMemObject(d_Whitened_fMRI_Volumes);

	if (WRITE_ACTIVITY_EPI && PERMUTE_FIRST_LEVEL)
	{
		clEnqueueReadBuffer(commandQueue, d_P_Values, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_P_Values_EPI, 0, NULL, NULL);
	}

	if (PERMUTE_FIRST_LEVEL)
	{
		d_P_Values_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, &createBufferErrorStatisticalMapsMNI);
		TransformPValuesToMNI();
		clEnqueueReadBuffer(commandQueue, d_P_Values_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_P_Values_MNI, 0, NULL, NULL);
		clReleaseMemObject(d_P_Values_MNI);
	}


	clReleaseMemObject(d_MNI_Brain_Mask);

	clReleaseMemObject(d_Total_Displacement_Field_X);
	clReleaseMemObject(d_Total_Displacement_Field_Y);
	clReleaseMemObject(d_Total_Displacement_Field_Z);

	//free(h_Motion_Parameters);
	clReleaseMemObject(d_Smoothed_fMRI_Volumes);

	clReleaseMemObject(d_EPI_Mask);
	clReleaseMemObject(d_Smoothed_EPI_Mask);

	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);

	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Contrast_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);

	if (PERMUTE_FIRST_LEVEL)
	{
		clReleaseMemObject(d_P_Values);
	}

	clReleaseMemObject(d_AR1_Estimates);
	clReleaseMemObject(d_AR2_Estimates);
	clReleaseMemObject(d_AR3_Estimates);
	clReleaseMemObject(d_AR4_Estimates);
}


void BROCCOLI_LIB::PerformFirstLevelAnalysisBayesianWrapper()
{
	//------------------------
	// T1-MNI registration
	//------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("\nAligning T1 to MNI\n");
	}

	// Allocate memory on device
	d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Brain_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Skullstripped_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);
	//clEnqueueWriteBuffer(commandQueue, d_MNI_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Volume , 0, NULL, NULL);
	//clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

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




	//clReleaseMemObject(d_MNI_T1_Volume);
	//clReleaseMemObject(d_Skullstripped_T1_Volume);
	//clReleaseMemObject(d_MNI_Brain_Mask);




	//------------------------
	// fMRI-T1 registration
	//------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("Aligning fMRI to T1\n");
	}

	// Allocate memory on device
	d_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_T1_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_EPI_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

	PerformRegistrationEPIT1();

	if (WRITE_ALIGNED_EPI_T1)
	{
		clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_EPI_Volume_T1, 0, NULL, NULL);
	}

	if (WRITE_ALIGNED_EPI_MNI)
	{
		TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_T1_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesNonLinear(d_T1_EPI_Volume, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_EPI_Volume_MNI, 0, NULL, NULL);
	}

	h_Registration_Parameters_EPI_T1_Out[0] = h_Registration_Parameters_EPI_T1[0];
	h_Registration_Parameters_EPI_T1_Out[1] = h_Registration_Parameters_EPI_T1[1];
	h_Registration_Parameters_EPI_T1_Out[2] = h_Registration_Parameters_EPI_T1[2];
	h_Registration_Parameters_EPI_T1_Out[3] = h_Registration_Parameters_EPI_T1[3];
	h_Registration_Parameters_EPI_T1_Out[4] = h_Registration_Parameters_EPI_T1[4];
	h_Registration_Parameters_EPI_T1_Out[5] = h_Registration_Parameters_EPI_T1[5];

	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_MNI,h_Registration_Parameters_T1_MNI,h_Registration_Parameters_EPI_T1_Affine);

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

	clReleaseMemObject(d_MNI_T1_Volume);
	clReleaseMemObject(d_Skullstripped_T1_Volume);
	clReleaseMemObject(d_EPI_Volume);
	clReleaseMemObject(d_T1_EPI_Volume);

	//------------------------
	// Slice timing correction
	//------------------------

	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	d_Slice_Timing_Corrected_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);


	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

	//PerformSliceTimingCorrection();

	if (WRITE_SLICETIMING_CORRECTED)
	{
		clEnqueueReadBuffer(commandQueue, d_Slice_Timing_Corrected_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Slice_Timing_Corrected_fMRI_Volumes, 0, NULL, NULL);
	}

	//------------------------
	// Motion correction
	//------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("Performing motion correction\n");
	}

	d_Motion_Corrected_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	PerformMotionCorrection(d_fMRI_Volumes);

	for (int t = 0; t < EPI_DATA_T; t++)
	{
		for (int p = 0; p < 6; p++)
		{
			h_Motion_Parameters_Out[t + p * EPI_DATA_T] = h_Motion_Parameters[t + p * EPI_DATA_T];
		}
	}

	if (WRITE_MOTION_CORRECTED)
	{
		clEnqueueReadBuffer(commandQueue, d_Motion_Corrected_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Motion_Corrected_fMRI_Volumes, 0, NULL, NULL);
	}

	//------------------------
	// Segment EPI data
	//------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("Segmenting EPI data\n");
	}

	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Smoothed_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	SegmentEPIData();

	if (WRITE_EPI_MASK)
	{
		clEnqueueReadBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask, 0, NULL, NULL);
	}

	//------------------------
	// Smoothing
	//------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("Performing smoothing\n");
	}

	d_Smoothed_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, EPI_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	//PerformSmoothing(d_Smoothed_fMRI_Volumes, d_Motion_Corrected_fMRI_Volumes, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	PerformSmoothing(d_Smoothed_EPI_Mask, d_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
	PerformSmoothingNormalized(d_Smoothed_fMRI_Volumes, d_Motion_Corrected_fMRI_Volumes, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

	if (WRITE_SMOOTHED)
	{
		clEnqueueReadBuffer(commandQueue, d_Smoothed_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Smoothed_fMRI_Volumes, 0, NULL, NULL);
	}


	//------------------------
	// GLM
	//------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("Performing statistical analysis\n");
	}

	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + NUMBER_OF_CONFOUND_REGRESSORS*REGRESS_CONFOUNDS;

	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * 10 * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * 6 * sizeof(float), NULL, NULL);
	d_AR1_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);


	// Copy data to device

	//clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM_In, 0, NULL, NULL);
	//clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM_In, 0, NULL, NULL);
	//clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts_In, 0, NULL, NULL);
	//clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In, 0, NULL, NULL);


	h_X_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
	h_xtxxt_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
	h_Contrasts = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float));
	h_ctxtxc_GLM = (float*)malloc(NUMBER_OF_CONTRASTS * sizeof(float));

	h_X_GLM_With_Temporal_Derivatives = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * 2 * EPI_DATA_T * sizeof(float));
	h_X_GLM_Convolved = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * (USE_TEMPORAL_DERIVATIVES+1) * EPI_DATA_T * sizeof(float));


	SetupTTestFirstLevel(EPI_DATA_T);

	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM , 0, NULL, NULL);

	if (WRITE_DESIGNMATRIX)
	{
		for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
		{
			for (int t = 0; t < EPI_DATA_T; t++)
			{
				h_X_GLM_Out[t + r * EPI_DATA_T] = h_X_GLM[t + r * EPI_DATA_T];
				h_xtxxt_GLM_Out[t + r * EPI_DATA_T] = h_xtxxt_GLM[t + r * EPI_DATA_T];
			}
		}
	}

	CalculateStatisticalMapsGLMBayesianFirstLevel(d_Smoothed_fMRI_Volumes);

	// Copy data to host
	if (WRITE_ACTIVITY_EPI)
	{
		clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * 2 * sizeof(float), h_Beta_Volumes_EPI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * 6 * sizeof(float), h_Statistical_Maps_EPI, 0, NULL, NULL);
	}

	//------------------------------------------------
	// Transform results to MNI space and copy to host
	//------------------------------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("Transforming results to MNI\n");
	}

	// Allocate memory on device
	d_Beta_Volumes_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * 2 * sizeof(float), NULL, &createBufferErrorBetaVolumesMNI);
	d_Statistical_Maps_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * 6 * sizeof(float), NULL, &createBufferErrorStatisticalMapsMNI);

	if (WRITE_AR_ESTIMATES_MNI)
	{
		d_AR1_Estimates_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, &createBufferErrorAREstimatesMNI);
	}

	SetMemory(d_MNI_Brain_Mask, 1.0f, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D);

	TransformBayesianFirstLevelResultsToMNI();

	clEnqueueReadBuffer(commandQueue, d_Beta_Volumes_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * 2 * sizeof(float), h_Beta_Volumes_MNI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * 6 * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);

	if (WRITE_AR_ESTIMATES_MNI)
	{
		clEnqueueReadBuffer(commandQueue, d_AR1_Estimates_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_AR1_Estimates_MNI, 0, NULL, NULL);
	}

	if (WRITE_AR_ESTIMATES_EPI)
	{
		clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
	}

	clReleaseMemObject(d_Beta_Volumes_MNI);
	clReleaseMemObject(d_Statistical_Maps_MNI);

	if (WRITE_AR_ESTIMATES_MNI)
	{
		clReleaseMemObject(d_AR1_Estimates_MNI);
	}

	free(h_X_GLM);
	free(h_xtxxt_GLM);
	free(h_Contrasts);
	free(h_ctxtxc_GLM);
	free(h_X_GLM_With_Temporal_Derivatives);
	free(h_X_GLM_Convolved);

	clReleaseMemObject(d_MNI_Brain_Mask);

	clReleaseMemObject(d_Total_Displacement_Field_X);
	clReleaseMemObject(d_Total_Displacement_Field_Y);
	clReleaseMemObject(d_Total_Displacement_Field_Z);

	//free(h_Motion_Parameters);
	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Slice_Timing_Corrected_fMRI_Volumes);
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

	clReleaseMemObject(d_AR1_Estimates);
}

// Old version
/*
void BROCCOLI_LIB::PerformFirstLevelAnalysisBayesianWrapper()
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

	clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_NonLinear, 0, NULL, NULL);

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

	TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_T1_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	TransformVolumesNonLinear(d_T1_EPI_Volume, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_EPI_Volume_MNI, 0, NULL, NULL);

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

	//NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS*2 + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS;
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + NUMBER_OF_CONFOUND_REGRESSORS*REGRESS_CONFOUNDS;

	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
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

	h_X_GLM_With_Temporal_Derivatives = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * 2 * EPI_DATA_T * sizeof(float));
	h_X_GLM_Convolved = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * (USE_TEMPORAL_DERIVATIVES+1) * EPI_DATA_T * sizeof(float));


	SetupTTestFirstLevel(EPI_DATA_T);

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



	CalculateStatisticalMapsGLMBayesianFirstLevel(d_Smoothed_fMRI_Volumes);

	free(h_X_GLM);
	free(h_xtxxt_GLM);
	free(h_Contrasts);
	free(h_ctxtxc_GLM);
	free(h_X_GLM_With_Temporal_Derivatives);
	free(h_X_GLM_Convolved);

	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);

	clReleaseMemObject(d_AR1_Estimates);
	clReleaseMemObject(d_AR2_Estimates);
	clReleaseMemObject(d_AR3_Estimates);
	clReleaseMemObject(d_AR4_Estimates);
	clReleaseMemObject(d_Whitened_fMRI_Volumes);

	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_MNI,h_Registration_Parameters_T1_MNI,h_Registration_Parameters_EPI_T1_Affine);

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

	// Transform results to MNI space and copy to host
	if (BETA_SPACE == MNI)
	{
		// Allocate memory on device
		d_Beta_Volumes_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, &createBufferErrorBetaVolumesMNI);
		d_Statistical_Maps_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, &createBufferErrorStatisticalMapsMNI);
		d_Residual_Variances_MNI = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, &createBufferErrorResidualVariancesMNI);

		clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

		TransformFirstLevelResultsToMNI();

		clEnqueueReadBuffer(commandQueue, d_Beta_Volumes_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_MNI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Statistical_Maps_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Residual_Variances_MNI, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);

		clReleaseMemObject(d_Beta_Volumes_MNI);
		clReleaseMemObject(d_Statistical_Maps_MNI);
		clReleaseMemObject(d_Residual_Variances_MNI);
	}
	// Copy data to host
	else if (BETA_SPACE == EPI)
	{
		clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_EPI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_EPI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);

		clEnqueueReadBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask, 0, NULL, NULL);

		//Clusterize(h_Cluster_Indices, MAX_CLUSTER_SIZE, MAX_CLUSTER_MASS, NUMBER_OF_CLUSTERS, h_Statistical_Maps, 2.0f, h_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, CALCULATE_VOXEL_LABELS, CALCULATE_CLUSTER_MASS);
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
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);

}
*/

// Transforms results from EPI space to MNI space
void BROCCOLI_LIB::TransformFirstLevelResultsToMNI()
{
	ChangeVolumesResolutionAndSize(d_Beta_Volumes_MNI, d_Beta_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_TOTAL_GLM_REGRESSORS, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	TransformVolumesLinear(d_Beta_Volumes_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_TOTAL_GLM_REGRESSORS, INTERPOLATION_MODE);
	TransformVolumesNonLinear(d_Beta_Volumes_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_TOTAL_GLM_REGRESSORS, INTERPOLATION_MODE);

	ChangeVolumesResolutionAndSize(d_Contrast_Volumes_MNI, d_Contrast_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	TransformVolumesLinear(d_Contrast_Volumes_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);
	TransformVolumesNonLinear(d_Contrast_Volumes_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);

	ChangeVolumesResolutionAndSize(d_Statistical_Maps_MNI, d_Statistical_Maps, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	TransformVolumesLinear(d_Statistical_Maps_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);
	TransformVolumesNonLinear(d_Statistical_Maps_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);

	ChangeVolumesResolutionAndSize(d_Residual_Variances_MNI, d_Residual_Variances, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	TransformVolumesLinear(d_Residual_Variances_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	TransformVolumesNonLinear(d_Residual_Variances_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

	MultiplyVolumes(d_Beta_Volumes_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_GLM_REGRESSORS);
	MultiplyVolumes(d_Contrast_Volumes_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS);
	MultiplyVolumes(d_Statistical_Maps_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS);
	MultiplyVolumes(d_Residual_Variances_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1);

	if (WRITE_AR_ESTIMATES_MNI)
	{
		ChangeVolumesResolutionAndSize(d_AR1_Estimates_MNI, d_AR1_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
		TransformVolumesLinear(d_AR1_Estimates_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesNonLinear(d_AR1_Estimates_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

		ChangeVolumesResolutionAndSize(d_AR2_Estimates_MNI, d_AR2_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
		TransformVolumesLinear(d_AR2_Estimates_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesNonLinear(d_AR2_Estimates_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

		ChangeVolumesResolutionAndSize(d_AR3_Estimates_MNI, d_AR3_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
		TransformVolumesLinear(d_AR3_Estimates_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesNonLinear(d_AR3_Estimates_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

		ChangeVolumesResolutionAndSize(d_AR4_Estimates_MNI, d_AR4_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
		TransformVolumesLinear(d_AR4_Estimates_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesNonLinear(d_AR4_Estimates_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	}
}

// Transforms results from EPI space to T1 space
void BROCCOLI_LIB::TransformFirstLevelResultsToT1()
{
	ChangeVolumesResolutionAndSize(d_Beta_Volumes_T1, d_Beta_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_TOTAL_GLM_REGRESSORS, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	TransformVolumesLinear(d_Beta_Volumes_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, NUMBER_OF_TOTAL_GLM_REGRESSORS, INTERPOLATION_MODE);

	ChangeVolumesResolutionAndSize(d_Contrast_Volumes_T1, d_Contrast_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	TransformVolumesLinear(d_Contrast_Volumes_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);

	ChangeVolumesResolutionAndSize(d_Statistical_Maps_T1, d_Statistical_Maps, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	TransformVolumesLinear(d_Statistical_Maps_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);

	//ChangeVolumesResolutionAndSize(d_Residual_Variances_T1, d_Residual_Variances, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	//TransformVolumesLinear(d_Residual_Variances_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);

	//MultiplyVolumes(d_Beta_Volumes_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_GLM_REGRESSORS);
	//MultiplyVolumes(d_Contrast_Volumes_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS);
	//MultiplyVolumes(d_Statistical_Maps_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS);
	//MultiplyVolumes(d_Residual_Variances_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1);

	if (WRITE_AR_ESTIMATES_T1)
	{
		ChangeVolumesResolutionAndSize(d_AR1_Estimates_T1, d_AR1_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
		TransformVolumesLinear(d_AR1_Estimates_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);

		ChangeVolumesResolutionAndSize(d_AR2_Estimates_T1, d_AR2_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
		TransformVolumesLinear(d_AR2_Estimates_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);

		ChangeVolumesResolutionAndSize(d_AR3_Estimates_T1, d_AR3_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
		TransformVolumesLinear(d_AR3_Estimates_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);

		ChangeVolumesResolutionAndSize(d_AR4_Estimates_T1, d_AR4_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
		TransformVolumesLinear(d_AR4_Estimates_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	}
}

// Transforms Bayesian results from EPI space to MNI space
void BROCCOLI_LIB::TransformBayesianFirstLevelResultsToMNI()
{
	ChangeVolumesResolutionAndSize(d_Beta_Volumes_MNI, d_Beta_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 2, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	TransformVolumesLinear(d_Beta_Volumes_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 2, INTERPOLATION_MODE);
	TransformVolumesNonLinear(d_Beta_Volumes_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 2, INTERPOLATION_MODE);

	ChangeVolumesResolutionAndSize(d_Statistical_Maps_MNI, d_Statistical_Maps, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 6, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	TransformVolumesLinear(d_Statistical_Maps_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 6, INTERPOLATION_MODE);
	TransformVolumesNonLinear(d_Statistical_Maps_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 6, INTERPOLATION_MODE);

	MultiplyVolumes(d_Beta_Volumes_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 2);
	MultiplyVolumes(d_Statistical_Maps_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 6);

	if (WRITE_AR_ESTIMATES_MNI)
	{
		ChangeVolumesResolutionAndSize(d_AR1_Estimates_MNI, d_AR1_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
		TransformVolumesLinear(d_AR1_Estimates_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesNonLinear(d_AR1_Estimates_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	}
}

void BROCCOLI_LIB::TransformPValuesToMNI()
{
	// Nearest neighbour interpolation for cluster inference, since all voxels in the cluster should have the same p-value
	if ( (INFERENCE_MODE == CLUSTER_EXTENT) || (INFERENCE_MODE == CLUSTER_MASS) )
	{
		ChangeVolumesResolutionAndSize(d_P_Values_MNI, d_P_Values, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, NEAREST);
		TransformVolumesLinear(d_P_Values_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS, NEAREST);
		TransformVolumesNonLinear(d_P_Values_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS, NEAREST);
	}
	// Linear interpolation otherwhise
	else
	{
		ChangeVolumesResolutionAndSize(d_P_Values_MNI, d_P_Values, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
		TransformVolumesLinear(d_P_Values_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);
		TransformVolumesNonLinear(d_P_Values_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);
	}
}


// Permutation based second level analysis
void BROCCOLI_LIB::PerformSecondLevelAnalysisWrapper()
{
	//------------------------

	// Allocate memory on device
	d_First_Level_Results = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_First_Level_Results, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_First_Level_Results, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);
	clFinish(commandQueue);

	//-------------------------------

	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS;


	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);


	c_Permutation_Vector = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_SUBJECTS * sizeof(unsigned short int), NULL, NULL);

	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Residuals = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device


	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), h_X_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), h_xtxxt_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In , 0, NULL, NULL);
	clFinish(commandQueue);



	d_Cluster_Indices = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), NULL, NULL);
	//ClusterizeOpenCL(d_Cluster_Indices, NUMBER_OF_CLUSTERS, d_Statistical_Maps, 2.0f, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
	//clEnqueueReadBuffer(commandQueue, d_Cluster_Indices, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), h_Cluster_Indices, 0, NULL, NULL);

	// Estimate null distribution
	//h_Permutation_Matrix = (unsigned short int*)malloc(NUMBER_OF_PERMUTATIONS * NUMBER_OF_SUBJECTS * sizeof(unsigned short int));



	ApplyPermutationTestSecondLevel();

	clReleaseMemObject(d_Cluster_Indices);



	CalculateStatisticalMapsGLMTTestSecondLevel(d_First_Level_Results, d_MNI_Brain_Mask);

	clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_MNI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_Residuals, 0, NULL, NULL);
	clFinish(commandQueue);

	//Clusterize(h_Cluster_Indices, MAX_CLUSTER_SIZE, MAX_CLUSTER_MASS, NUMBER_OF_CLUSTERS, h_Statistical_Maps, CLUSTER_DEFINING_THRESHOLD, h_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, CALCULATE_VOXEL_LABELS, CALCULATE_CLUSTER_MASS);




	clEnqueueReadBuffer(commandQueue, d_Permuted_First_Level_Results, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_Permuted_First_Level_Results, 0, NULL, NULL);

	//free(h_Permutation_Matrix);

	clReleaseMemObject(d_First_Level_Results);
	clReleaseMemObject(d_MNI_Brain_Mask);

	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);
	clReleaseMemObject(c_Permutation_Vector);

	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);
}




// Performs slice timing correction of an fMRI dataset
void BROCCOLI_LIB::PerformSliceTimingCorrection()
{
	SetGlobalAndLocalWorkSizesInterpolateVolume(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Allocate memory for slice differences
	c_Slice_Differences = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_D * sizeof(float), NULL, NULL);

	h_Slice_Differences = (float*)malloc(EPI_DATA_D * sizeof(float));

	// Calculate middle slice
	float middle_slice = round((float)EPI_DATA_D / 2.0f) - 1.0f;

	// Calculate slice differences
	if (SLICE_ORDER == UP)
	{
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			h_Slice_Differences[z] = ((float)z - middle_slice)/((float)EPI_DATA_D);
		}
	}
	else if (SLICE_ORDER == DOWN)
	{
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			h_Slice_Differences[z] = (middle_slice - (float)z)/((float)EPI_DATA_D);
		}
	}
	else if (SLICE_ORDER == UP_INTERLEAVED)
	{

	}
	else if (SLICE_ORDER == DOWN_INTERLEAVED)
	{

	}



	// Copy slice differences to device
	clEnqueueWriteBuffer(commandQueue, c_Slice_Differences, CL_TRUE, 0, EPI_DATA_D * sizeof(float), h_Slice_Differences, 0, NULL, NULL);

	clSetKernelArg(SliceTimingCorrectionKernel, 0, sizeof(cl_mem), &d_Slice_Timing_Corrected_fMRI_Volumes);
	clSetKernelArg(SliceTimingCorrectionKernel, 1, sizeof(cl_mem), &d_fMRI_Volumes);
	clSetKernelArg(SliceTimingCorrectionKernel, 2, sizeof(cl_mem), &c_Slice_Differences);
	clSetKernelArg(SliceTimingCorrectionKernel, 3, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(SliceTimingCorrectionKernel, 4, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(SliceTimingCorrectionKernel, 5, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(SliceTimingCorrectionKernel, 6, sizeof(int), &EPI_DATA_T);

	runKernelErrorSliceTimingCorrection = clEnqueueNDRangeKernel(commandQueue, SliceTimingCorrectionKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
	clFinish(commandQueue);

	clReleaseMemObject(c_Slice_Differences);
	free(h_Slice_Differences);
}

void BROCCOLI_LIB::PerformSliceTimingCorrectionWrapper()
{
	SetGlobalAndLocalWorkSizesInterpolateVolume(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Slice_Timing_Corrected_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Allocate memory for slice differences
	c_Slice_Differences = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_D * sizeof(float), NULL, NULL);

	// Copy volumes to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

	h_Slice_Differences = (float*)malloc(EPI_DATA_D * sizeof(float));

	// Calculate middle slice
	float middle_slice = round((float)EPI_DATA_D / 2.0f) - 1.0f;

	for (int z = 0; z < EPI_DATA_D; z++)
	{
		h_Slice_Differences[z] = ((float)z - middle_slice)/((float)EPI_DATA_D);
	}

	// Copy slice differences to device
	clEnqueueWriteBuffer(commandQueue, c_Slice_Differences, CL_TRUE, 0, EPI_DATA_D * sizeof(float), h_Slice_Differences, 0, NULL, NULL);

	clSetKernelArg(SliceTimingCorrectionKernel, 0, sizeof(cl_mem), &d_Slice_Timing_Corrected_fMRI_Volumes);
	clSetKernelArg(SliceTimingCorrectionKernel, 1, sizeof(cl_mem), &d_fMRI_Volumes);
	clSetKernelArg(SliceTimingCorrectionKernel, 2, sizeof(cl_mem), &c_Slice_Differences);
	clSetKernelArg(SliceTimingCorrectionKernel, 3, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(SliceTimingCorrectionKernel, 4, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(SliceTimingCorrectionKernel, 5, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(SliceTimingCorrectionKernel, 6, sizeof(int), &EPI_DATA_T);

	runKernelErrorSliceTimingCorrection = clEnqueueNDRangeKernel(commandQueue, SliceTimingCorrectionKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
	clFinish(commandQueue);


	// Copy all corrected volumes to host
	clEnqueueReadBuffer(commandQueue, d_Slice_Timing_Corrected_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Slice_Timing_Corrected_fMRI_Volumes, 0, NULL, NULL);

	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Slice_Timing_Corrected_fMRI_Volumes);
	clReleaseMemObject(c_Slice_Differences);
	free(h_Slice_Differences);
}

/* Old version  using a lot of memory
void BROCCOLI_LIB::PerformMotionCorrectionWrapper()
{
	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Motion_Corrected_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Copy volumes to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

	clFinish(commandQueue);

	// Setup all parameters and allocate memory on device
	AlignTwoVolumesLinearSetup(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

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
		AlignTwoVolumesLinear(h_Registration_Parameters_Motion_Correction, h_Rotations, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION, RIGID, INTERPOLATION_MODE);

		// Copy the corrected volume to the corrected volumes
		clEnqueueCopyBuffer(commandQueue, d_Aligned_Volume, d_Motion_Corrected_fMRI_Volumes, 0, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

		// Write the total parameter vector to host

		// Translations
		h_Motion_Parameters_Out[t + 0 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[0] * EPI_VOXEL_SIZE_X;
		h_Motion_Parameters_Out[t + 1 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[1] * EPI_VOXEL_SIZE_Y;
		h_Motion_Parameters_Out[t + 2 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[2] * EPI_VOXEL_SIZE_Z;

		// Rotations
		h_Motion_Parameters_Out[t + 3 * EPI_DATA_T] = h_Rotations[0];
		h_Motion_Parameters_Out[t + 4 * EPI_DATA_T] = h_Rotations[1];
		h_Motion_Parameters_Out[t + 5 * EPI_DATA_T] = h_Rotations[2];
	}

	// Copy all corrected volumes to host
	clEnqueueReadBuffer(commandQueue, d_Motion_Corrected_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Motion_Corrected_fMRI_Volumes, 0, NULL, NULL);

	// Cleanup allocated memory
	AlignTwoVolumesLinearCleanup();

	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Motion_Corrected_fMRI_Volumes);
}
*/

void BROCCOLI_LIB::PerformMotionCorrectionWrapper()
{
	// Setup all parameters and allocate memory on device
	AlignTwoVolumesLinearSetup(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Set the first volume as the reference volume
	clEnqueueWriteBuffer(commandQueue, d_Reference_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

	// Copy the first volume to the corrected volumes
	clEnqueueReadBuffer(commandQueue, d_Reference_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Motion_Corrected_fMRI_Volumes, 0, NULL, NULL);

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
		clEnqueueWriteBuffer(commandQueue, d_Aligned_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), &h_fMRI_Volumes[t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D], 0, NULL, NULL);

		// Also copy the same volume to an image to interpolate from
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {EPI_DATA_W, EPI_DATA_H, EPI_DATA_D};
		clEnqueueCopyBufferToImage(commandQueue, d_Aligned_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);

		// Do rigid registration with only one scale
		AlignTwoVolumesLinear(h_Registration_Parameters_Motion_Correction, h_Rotations, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION, RIGID, INTERPOLATION_MODE);

		// Copy the corrected volume to the corrected volumes
		clEnqueueReadBuffer(commandQueue, d_Aligned_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), &h_Motion_Corrected_fMRI_Volumes[t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D], 0, NULL, NULL);

		// Write the total parameter vector to host

		// Translations
		h_Motion_Parameters_Out[t + 0 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[0] * EPI_VOXEL_SIZE_X;
		h_Motion_Parameters_Out[t + 1 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[1] * EPI_VOXEL_SIZE_Y;
		h_Motion_Parameters_Out[t + 2 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[2] * EPI_VOXEL_SIZE_Z;

		// Rotations
		h_Motion_Parameters_Out[t + 3 * EPI_DATA_T] = h_Rotations[0];
		h_Motion_Parameters_Out[t + 4 * EPI_DATA_T] = h_Rotations[1];
		h_Motion_Parameters_Out[t + 5 * EPI_DATA_T] = h_Rotations[2];
	}

	// Cleanup allocated memory
	AlignTwoVolumesLinearCleanup();
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
		AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_Motion_Correction, h_Rotations, d_Current_fMRI_Volume, d_Current_Reference_Volume, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);

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
void BROCCOLI_LIB::PerformMotionCorrection(cl_mem d_Volumes)
{
	// Setup all parameters and allocate memory on device
	AlignTwoVolumesLinearSetup(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Set the first volume as the reference volume
	clEnqueueCopyBuffer(commandQueue, d_Volumes, d_Reference_Volume, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

	// Copy the first volume to the corrected volumes
	clEnqueueCopyBuffer(commandQueue, d_Volumes, d_Motion_Corrected_fMRI_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

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
		clEnqueueCopyBuffer(commandQueue, d_Volumes, d_Aligned_Volume, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

		// Also copy the same volume to an image (texture) to interpolate from
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {EPI_DATA_W, EPI_DATA_H, EPI_DATA_D};
		clEnqueueCopyBufferToImage(commandQueue, d_Volumes, d_Original_Volume, t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), origin, region, 0, NULL, NULL);

		// Do rigid registration with only one scale
		AlignTwoVolumesLinear(h_Registration_Parameters_Motion_Correction, h_Rotations, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION, RIGID, INTERPOLATION_MODE);

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
	AlignTwoVolumesLinearCleanup();
}

// Slow way of calculating the sum of a volume
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

// Slow way of calculating the maximum of a volume
float BROCCOLI_LIB::CalculateMax(cl_mem d_Volume, int DATA_W, int DATA_H, int DATA_D)
{
	SetGlobalAndLocalWorkSizesCalculateMax(DATA_W, DATA_H, DATA_D);

	cl_mem d_Column_Maxs = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_H * DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Maxs = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_D * sizeof(float), NULL, NULL);

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

	float max = std::numeric_limits<float>::min();
	for (int z = 0; z < DATA_D; z++)
	{
		max = mymax(max, h_Maxs[z]);
	}
	free(h_Maxs);

	clReleaseMemObject(d_Column_Maxs);
	clReleaseMemObject(d_Maxs);

	return max;
}





float BROCCOLI_LIB::CalculateMaxAtomic(cl_mem d_Volume, cl_mem d_Mask, int DATA_W, int DATA_H, int DATA_D)
{
	SetGlobalAndLocalWorkSizesCalculateMax(DATA_W, DATA_H, DATA_D);

	//cl_mem d_Max_Value = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
	cl_mem d_Max_Value = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

	//SetMemory(d_Max_Value, -10000.0f, 1);
	SetMemory(d_Max_Value, -1000000, 1);

	clSetKernelArg(CalculateMaxAtomicKernel, 0, sizeof(cl_mem), &d_Max_Value);
	clSetKernelArg(CalculateMaxAtomicKernel, 1, sizeof(cl_mem), &d_Volume);
	clSetKernelArg(CalculateMaxAtomicKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateMaxAtomicKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateMaxAtomicKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateMaxAtomicKernel, 5, sizeof(int), &DATA_D);

	runKernelErrorCalculateMaxAtomic = clEnqueueNDRangeKernel(commandQueue, CalculateMaxAtomicKernel, 3, NULL, globalWorkSizeCalculateMaxAtomic, localWorkSizeCalculateMaxAtomic, 0, NULL, NULL);
	clFinish(commandQueue);

	//float max;
	//clEnqueueReadBuffer(commandQueue, d_Max_Value, CL_TRUE, 0, sizeof(float), &max, 0, NULL, NULL);

	int max;
	clEnqueueReadBuffer(commandQueue, d_Max_Value, CL_TRUE, 0, sizeof(int), &max, 0, NULL, NULL);

	clReleaseMemObject(d_Max_Value);

	return (float)((float)max/10000.0f);
}

// Thresholds a volume
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

// Segments one volume by smoothing and a simple thresholding, uses the first fMRI volume as input
void BROCCOLI_LIB::SegmentEPIData()
{
	cl_mem d_EPI = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Smoothed_EPI = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	// Copy the first fMRI volume
	clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_EPI, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

	// Smooth the volume with a 4 mm Gaussian filter
	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, 4.0, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	PerformSmoothing(d_Smoothed_EPI, d_EPI, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

	// Calculate sum of all voxels
	float sum = CalculateSum(d_Smoothed_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Apply a threshold that is 90% of the mean voxel value
	float threshold = 0.9f * sum / ((float) EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	ThresholdVolume(d_EPI_Mask, d_Smoothed_EPI, threshold, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	clReleaseMemObject(d_EPI);
	clReleaseMemObject(d_Smoothed_EPI);
}

// Segments one fMRI volume by smoothing and a simple thresholding, uses a defined volume as input, inplace
void BROCCOLI_LIB::SegmentEPIData(cl_mem d_Volume)
{
	cl_mem d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Smoothed_EPI = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	// Smooth the volume with a 4 mm Gaussian filter
	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, 4.0, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	PerformSmoothing(d_Smoothed_EPI, d_Volume, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

	// Calculate sum of all voxels
	float sum = CalculateSum(d_Smoothed_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Apply a threshold that is 90% of the mean voxel value
	float threshold = 0.9f * sum / ((float) EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	ThresholdVolume(d_EPI_Mask, d_Smoothed_EPI, threshold, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	MultiplyVolumes(d_Volume, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	clReleaseMemObject(d_EPI_Mask);
	clReleaseMemObject(d_Smoothed_EPI);
}

// Creates Gaussian smoothing filters, as function of FWHM in mm and voxel size
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

	// Normalize
	for (int i = 0; i < size; i++)
	{
		Smoothing_Filter_X[i] /= sumX;
		Smoothing_Filter_Y[i] /= sumY;
		Smoothing_Filter_Z[i] /= sumZ;
	}
}

// Creates Gaussian smoothing filters, as function of sigma
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

	// Normalize
	for (int i = 0; i < size; i++)
	{
		Smoothing_Filter_X[i] /= sum;
		Smoothing_Filter_Y[i] /= sum;
		Smoothing_Filter_Z[i] /= sum;
	}
}

// This function only performs smoothing, and is used for testing from Matlab (or any other wrapper)
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

	if (SMOOTHING_TYPE == LOWPASS)
	{
		CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, EPI_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);

		// Copy smoothing filters to constant memory
		clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_X, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_X, 0, NULL, NULL);
		clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Y, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Y, 0, NULL, NULL);
		clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Z, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Z, 0, NULL, NULL);
	}
	else if (SMOOTHING_TYPE == RANDOM)
	{
		// Copy smoothing filters to constant memory
		clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_X, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_X_In, 0, NULL, NULL);
		clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Y, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Y_In, 0, NULL, NULL);
		clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Z, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Z_In, 0, NULL, NULL);
	}


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

	// Release memory
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

void BROCCOLI_LIB::PerformSmoothingNormalizedWrapper()
{
	SetGlobalAndLocalWorkSizesSeparableConvolution(EPI_DATA_W,EPI_DATA_H,EPI_DATA_D);

	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Smoothed_fMRI_Volumes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	d_Certainty = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Smoothed_Certainty = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_Certainty, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask, 0, NULL, NULL);
	//clEnqueueWriteBuffer(commandQueue, d_Smoothed_Certainty, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Smoothed_EPI_Mask, 0, NULL, NULL);

	if (SMOOTHING_TYPE == LOWPASS)
	{
		// Create Gaussian smoothing filters
		CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, EPI_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);

		PerformSmoothing(d_Smoothed_Certainty, d_Certainty, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

		// Do smoothing
		PerformSmoothingNormalized(d_Smoothed_fMRI_Volumes, d_fMRI_Volumes, d_Certainty, d_Smoothed_Certainty, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	}
	else if (SMOOTHING_TYPE == RANDOM)
	{
		PerformSmoothing(d_Smoothed_Certainty, d_Certainty, h_Smoothing_Filter_X_In, h_Smoothing_Filter_Y_In, h_Smoothing_Filter_Z_In, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

		// Do smoothing
		PerformSmoothingNormalized(d_Smoothed_fMRI_Volumes, d_fMRI_Volumes, d_Certainty, d_Smoothed_Certainty, h_Smoothing_Filter_X_In, h_Smoothing_Filter_Y_In, h_Smoothing_Filter_Z_In, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	}

	// Copy result back to host
	clEnqueueReadBuffer(commandQueue, d_Smoothed_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Smoothed_fMRI_Volumes, 0, NULL, NULL);

	// Release memory
	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Smoothed_fMRI_Volumes);

	clReleaseMemObject(d_Certainty);
	clReleaseMemObject(d_Smoothed_Certainty);
}


// Performs smoothing of a number of volumes
void BROCCOLI_LIB::PerformSmoothing(cl_mem d_Smoothed_Volumes,
		                            cl_mem d_Volumes,
		                            float* h_Smoothing_Filter_X,
		                            float* h_Smoothing_Filter_Y,
		                            float* h_Smoothing_Filter_Z,
		                            int DATA_W,
		                            int DATA_H,
		                            int DATA_D,
		                            int DATA_T)
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

	cl_mem d_Certainty_Temp = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);

	SetMemory(d_Certainty_Temp, 1.0f, DATA_W * DATA_H * DATA_D);

	// Set arguments for the kernels
	clSetKernelArg(SeparableConvolutionRowsKernel, 0, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionRowsKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(SeparableConvolutionRowsKernel, 2, sizeof(cl_mem), &d_Certainty_Temp);
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
	clSetKernelArg(SeparableConvolutionRodsKernel, 2, sizeof(cl_mem), &d_Certainty_Temp);
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

	clReleaseMemObject(d_Certainty_Temp);
}

// Performs smoothing of a number of volumes, normalized with certainty (brain mask)
void BROCCOLI_LIB::PerformSmoothingNormalized(cl_mem d_Smoothed_Volumes,
		                                      cl_mem d_Volumes,
		                                      cl_mem d_Certainty,
		                                      cl_mem d_Smoothed_Certainty,
		                                      float* h_Smoothing_Filter_X,
		                                      float* h_Smoothing_Filter_Y,
		                                      float* h_Smoothing_Filter_Z,
		                                      int DATA_W,
		                                      int DATA_H,
		                                      int DATA_D,
		                                      int DATA_T)
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

	MultiplyVolumes(d_Smoothed_Volumes, d_Certainty, DATA_W, DATA_H, DATA_D, DATA_T);

	// Free temporary memory
	clReleaseMemObject(c_Smoothing_Filter_X);
	clReleaseMemObject(c_Smoothing_Filter_Y);
	clReleaseMemObject(c_Smoothing_Filter_Z);

	clReleaseMemObject(d_Convolved_Rows);
	clReleaseMemObject(d_Convolved_Columns);
}

void BROCCOLI_LIB::PerformSmoothingNormalizedPermutation()
{
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
}

// Performs smoothing of a number of volumes, overwrites data
void BROCCOLI_LIB::PerformSmoothing(cl_mem d_Volumes,
		                            float* h_Smoothing_Filter_X,
		                            float* h_Smoothing_Filter_Y,
		                            float* h_Smoothing_Filter_Z,
		                            int DATA_W,
		                            int DATA_H,
		                            int DATA_D,
		                            int DATA_T)
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

	cl_mem d_Certainty_Temp = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);

	SetMemory(d_Certainty_Temp, 1.0f, DATA_W * DATA_H * DATA_D);

	// Set arguments for the kernels
	clSetKernelArg(SeparableConvolutionRowsKernel, 0, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionRowsKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(SeparableConvolutionRowsKernel, 2, sizeof(cl_mem), &d_Certainty_Temp);
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
	clSetKernelArg(SeparableConvolutionRodsKernel, 2, sizeof(cl_mem), &d_Certainty_Temp);
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

	clReleaseMemObject(d_Certainty_Temp);
}

// Performs smoothing of a number of volumes, overwrites data, normalized with certainty (brain mask)
void BROCCOLI_LIB::PerformSmoothingNormalized(cl_mem d_Volumes,
		                                      cl_mem d_Certainty,
		                                      cl_mem d_Smoothed_Certainty,
		                                      float* h_Smoothing_Filter_X,
		                                      float* h_Smoothing_Filter_Y,
		                                      float* h_Smoothing_Filter_Z,
		                                      int DATA_W,
		                                      int DATA_H,
		                                      int DATA_D,
		                                      int DATA_T)
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

	MultiplyVolumes(d_Volumes, d_Certainty, DATA_W, DATA_H, DATA_D, DATA_T);

	// Free temporary memory
	clReleaseMemObject(c_Smoothing_Filter_X);
	clReleaseMemObject(c_Smoothing_Filter_Y);
	clReleaseMemObject(c_Smoothing_Filter_Z);

	clReleaseMemObject(d_Convolved_Rows);
	clReleaseMemObject(d_Convolved_Columns);
}

// Performs detrending of an fMRI dataset (removes mean, linear trend, quadratic trend, cubic trend)
void BROCCOLI_LIB::PerformDetrending(cl_mem d_Detrended_Volumes, cl_mem d_Volumes, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
	// Allocate host memory
	h_X_Detrend = (float*)malloc(NUMBER_OF_DETRENDING_REGRESSORS * DATA_T * sizeof(float));
	h_xtxxt_Detrend = (float*)malloc(NUMBER_OF_DETRENDING_REGRESSORS * DATA_T * sizeof(float));

	// Setup regressors for mean, linear, quadratic and cubic trends
	SetupDetrendingRegressors(DATA_T);

	cl_mem d_Beta = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float), NULL, NULL);

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


	// Estimate beta weights
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 0, sizeof(cl_mem), &d_Beta);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_Detrend);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 4, sizeof(cl_mem), &c_Censored_Timepoints);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 5, sizeof(int),    &DATA_W);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 6, sizeof(int),    &DATA_H);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 7, sizeof(int),    &DATA_D);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 8, sizeof(int),    &DATA_T);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 9, sizeof(int),    &NUMBER_OF_DETRENDING_REGRESSORS);

	runKernelErrorCalculateBetaWeightsGLM = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Remove linear fit
	clSetKernelArg(RemoveLinearFitKernel, 0, sizeof(cl_mem), &d_Detrended_Volumes);
	clSetKernelArg(RemoveLinearFitKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(RemoveLinearFitKernel, 2, sizeof(cl_mem), &d_Beta);
	clSetKernelArg(RemoveLinearFitKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(RemoveLinearFitKernel, 4, sizeof(cl_mem), &c_X_Detrend);
	clSetKernelArg(RemoveLinearFitKernel, 5, sizeof(int),    &DATA_W);
	clSetKernelArg(RemoveLinearFitKernel, 6, sizeof(int),    &DATA_H);
	clSetKernelArg(RemoveLinearFitKernel, 7, sizeof(int),    &DATA_D);
	clSetKernelArg(RemoveLinearFitKernel, 8, sizeof(int),    &DATA_T);
	clSetKernelArg(RemoveLinearFitKernel, 9, sizeof(int),    &NUMBER_OF_DETRENDING_REGRESSORS);

	runKernelErrorRemoveLinearFit = clEnqueueNDRangeKernel(commandQueue, RemoveLinearFitKernel, 3, NULL, globalWorkSizeRemoveLinearFit, localWorkSizeRemoveLinearFit, 0, NULL, NULL);
	clFinish(commandQueue);

	// Free host memory
	free(h_Censored_Timepoints);
	free(h_X_Detrend);
	free(h_xtxxt_Detrend);

	// Free memory
	clReleaseMemObject(d_Beta);
	clReleaseMemObject(c_Censored_Timepoints);
	clReleaseMemObject(c_X_Detrend);
	clReleaseMemObject(c_xtxxt_Detrend);
}

// Removes the linear fit between detrending regressors (mean, linear trend, quadratic trend, cubic trend) and motion regressors
void BROCCOLI_LIB::PerformDetrendingAndMotionRegression(cl_mem d_Regressed_Volumes, cl_mem d_Volumes, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
	int NUMBER_OF_DETRENDING_AND_MOTION_REGRESSORS = 10;

	// Allocate host memory
	h_X_Detrend = (float*)malloc(NUMBER_OF_DETRENDING_AND_MOTION_REGRESSORS * DATA_T * sizeof(float));
	h_xtxxt_Detrend = (float*)malloc(NUMBER_OF_DETRENDING_AND_MOTION_REGRESSORS * DATA_T * sizeof(float));

	// Setup regressors for mean, linear, quadratic and cubic trends
	SetupDetrendingAndMotionRegressors(DATA_T);

	// Allocate constant memory on device
	c_X_Detrend = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_DETRENDING_AND_MOTION_REGRESSORS * DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_Detrend = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_DETRENDING_AND_MOTION_REGRESSORS * DATA_T * sizeof(float), NULL, NULL);

	// Copy data to constant memory
	clEnqueueWriteBuffer(commandQueue, c_X_Detrend, CL_TRUE, 0, NUMBER_OF_DETRENDING_AND_MOTION_REGRESSORS * DATA_T * sizeof(float), h_X_Detrend , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_Detrend, CL_TRUE, 0, NUMBER_OF_DETRENDING_AND_MOTION_REGRESSORS * DATA_T * sizeof(float), h_xtxxt_Detrend , 0, NULL, NULL);

	SetGlobalAndLocalWorkSizesStatisticalCalculations(DATA_W, DATA_H, DATA_D);

	h_Censored_Timepoints = (float*)malloc(EPI_DATA_T * sizeof(float));
	c_Censored_Timepoints = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(float), NULL, NULL);

	for (int t = 0; t < EPI_DATA_T; t++)
	{
		h_Censored_Timepoints[t] = 1.0f;
	}
	clEnqueueWriteBuffer(commandQueue, c_Censored_Timepoints, CL_TRUE, 0, EPI_DATA_T * sizeof(float), h_Censored_Timepoints , 0, NULL, NULL);


	// Estimate beta weights
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_Detrend);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 4, sizeof(cl_mem), &c_Censored_Timepoints);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 8, sizeof(int), &DATA_T);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 9, sizeof(int), &NUMBER_OF_DETRENDING_AND_MOTION_REGRESSORS);

	runKernelErrorCalculateBetaWeightsGLM = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Remove linear fit
	clSetKernelArg(RemoveLinearFitKernel, 0, sizeof(cl_mem), &d_Regressed_Volumes);
	clSetKernelArg(RemoveLinearFitKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(RemoveLinearFitKernel, 2, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(RemoveLinearFitKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(RemoveLinearFitKernel, 4, sizeof(cl_mem), &c_X_Detrend);
	clSetKernelArg(RemoveLinearFitKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(RemoveLinearFitKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(RemoveLinearFitKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(RemoveLinearFitKernel, 8, sizeof(int), &DATA_T);
	clSetKernelArg(RemoveLinearFitKernel, 9, sizeof(int), &NUMBER_OF_DETRENDING_AND_MOTION_REGRESSORS);

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
	//PerformMotionCorrection();
	//PerformSmoothing();
	//PerformDetrending();
	//CalculateStatisticalMapsGLMFirstLevel(d_Smoothed_fMRI_Volumes);

	//CalculateSlicesPreprocessedfMRIData();
}

// Used for testing of t-test only
void BROCCOLI_LIB::PerformGLMTTestFirstLevelWrapper()
{
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS;

	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Smoothed_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	// Allocate memory for model
	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_Censored_Timepoints = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(float), NULL, NULL);

	// Allocate memory for results
	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Residuals = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	d_AR1_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR2_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR3_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR4_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Whitened_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_Smoothed_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Smoothed_EPI_Mask , 0, NULL, NULL);

	// Copy model to constant memory
	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In , 0, NULL, NULL);
	clFinish(commandQueue);


	// Setup work sizes
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Create a mapping between voxel coordinates and brain voxel number, since we cannot store the modified GLM design matrix for all voxels, only for the brain voxels
	cl_mem d_Voxel_Numbers = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	CreateVoxelNumbers(d_Voxel_Numbers, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Allocate memory for voxel specific design matrices (sufficient to store the pseudo inverses, since we only need to estimate beta weights with the voxel-specific models, not the residuals)
	cl_mem d_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Allocate memory for voxel specific GLM scalars
	cl_mem d_GLM_Scalars = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, AR_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);

	// All timepoints are valid the first run
	NUMBER_OF_INVALID_TIMEPOINTS = 0;
	SetMemory(c_Censored_Timepoints, 1.0f, EPI_DATA_T);

	// Reset all AR parameters
	SetMemory(d_AR1_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_AR2_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_AR3_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_AR4_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);

	// Apply whitening to model (no whitening first time, so just copy regressors)
	WhitenDesignMatricesInverse(d_xtxxt_GLM, h_X_GLM_In, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS);

	// Set whitened volumes to original volumes
	clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Whitened_fMRI_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), 0, NULL, NULL);

	// Cochrane-Orcutt procedure, iterate
	for (int it = 0; it < 1; it++)
	{
		// Calculate beta values, using whitened data and the whitened voxel-specific models
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 1, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 3, sizeof(cl_mem), &d_xtxxt_GLM);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 4, sizeof(cl_mem), &d_Voxel_Numbers);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 5, sizeof(cl_mem), &c_Censored_Timepoints);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 6, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 7, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 8, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 9, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 10, sizeof(int),   &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 11, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorCalculateBetaWeightsGLMFirstLevel = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMFirstLevelKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
		clFinish(commandQueue);

		// Calculate residuals, using original data and the original model
		clSetKernelArg(CalculateGLMResidualsKernel, 0, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(CalculateGLMResidualsKernel, 1, sizeof(cl_mem), &d_fMRI_Volumes);
		clSetKernelArg(CalculateGLMResidualsKernel, 2, sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateGLMResidualsKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateGLMResidualsKernel, 4, sizeof(cl_mem), &c_X_GLM);
		clSetKernelArg(CalculateGLMResidualsKernel, 5, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(CalculateGLMResidualsKernel, 6, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(CalculateGLMResidualsKernel, 7, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(CalculateGLMResidualsKernel, 8, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(CalculateGLMResidualsKernel, 9, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		runKernelErrorCalculateGLMResiduals = clEnqueueNDRangeKernel(commandQueue, CalculateGLMResidualsKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
		clFinish(commandQueue);


		// Estimate auto correlation from residuals
		clSetKernelArg(EstimateAR4ModelsKernel, 0, sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 1, sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 2, sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 3, sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 4, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(EstimateAR4ModelsKernel, 5, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(EstimateAR4ModelsKernel, 6, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(EstimateAR4ModelsKernel, 7, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(EstimateAR4ModelsKernel, 8, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(EstimateAR4ModelsKernel, 9, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(EstimateAR4ModelsKernel, 10, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorEstimateAR4Models = clEnqueueNDRangeKernel(commandQueue, EstimateAR4ModelsKernel, 3, NULL, globalWorkSizeEstimateAR4Models, localWorkSizeEstimateAR4Models, 0, NULL, NULL);

		// Smooth auto correlation estimates
		PerformSmoothingNormalized(d_AR1_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR2_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR3_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR4_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

		// Apply whitening to data
		clSetKernelArg(ApplyWhiteningAR4Kernel, 0,  sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 1,  sizeof(cl_mem), &d_fMRI_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 2,  sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 3,  sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 4,  sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 5,  sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 6,  sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 7,  sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 8,  sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 9,  sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 10, sizeof(int),    &EPI_DATA_T);
		runKernelErrorApplyWhiteningAR4 = clEnqueueNDRangeKernel(commandQueue, ApplyWhiteningAR4Kernel, 3, NULL, globalWorkSizeApplyWhiteningAR4, localWorkSizeApplyWhiteningAR4, 0, NULL, NULL);

		// First four timepoints are now invalid
		SetMemory(c_Censored_Timepoints, 0.0f, 4);
		NUMBER_OF_INVALID_TIMEPOINTS = 4;

		// Apply whitening to model and create voxel-specific models
		WhitenDesignMatricesInverse(d_xtxxt_GLM, h_X_GLM_In, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS);
	}

	// Calculate beta values, using whitened data and the whitened voxel-specific models
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 1, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 3, sizeof(cl_mem), &d_xtxxt_GLM);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 4, sizeof(cl_mem), &d_Voxel_Numbers);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 5, sizeof(cl_mem), &c_Censored_Timepoints);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 6, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 7, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 8, sizeof(int),    &EPI_DATA_D);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 9, sizeof(int),    &EPI_DATA_T);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 10, sizeof(int),   &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 11, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);
	runKernelErrorCalculateBetaWeightsGLMFirstLevel = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMFirstLevelKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// d_xtxxt_GLM now contains X_GLM and not xtxxt_GLM ...
	WhitenDesignMatricesTTest(d_xtxxt_GLM, d_GLM_Scalars, h_X_GLM_In, h_Contrasts_In, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS, NUMBER_OF_CONTRASTS);

	// Finally calculate statistical maps using whitened model and whitened data
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 0, sizeof(cl_mem),  &d_Statistical_Maps);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 1, sizeof(cl_mem),  &d_Residuals);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 2, sizeof(cl_mem),  &d_Residual_Variances);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 3, sizeof(cl_mem),  &d_Whitened_fMRI_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 4, sizeof(cl_mem),  &d_Beta_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 5, sizeof(cl_mem),  &d_EPI_Mask);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 6, sizeof(cl_mem),  &d_xtxxt_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 7, sizeof(cl_mem),  &d_GLM_Scalars);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 8, sizeof(cl_mem),  &d_Voxel_Numbers);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 9, sizeof(cl_mem),  &c_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 10, sizeof(cl_mem), &c_Censored_Timepoints);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 11, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 12, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 13, sizeof(int),    &EPI_DATA_D);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 14, sizeof(int),    &EPI_DATA_T);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 15, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 16, sizeof(int),    &NUMBER_OF_CONTRASTS);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 17, sizeof(int),    &NUMBER_OF_INVALID_TIMEPOINTS);
	runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMTTestFirstLevelKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	MultiplyVolumes(d_AR1_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR2_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR3_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	PutWhitenedModelsIntoVolumes2(d_EPI_Mask, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, h_X_GLM_In, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS);

	// Copy results to  host
	clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Residuals, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);

	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);


	// Release memory
	clReleaseMemObject(d_xtxxt_GLM);
	clReleaseMemObject(d_GLM_Scalars);
	clReleaseMemObject(d_Voxel_Numbers);

	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_EPI_Mask);
	clReleaseMemObject(d_Smoothed_EPI_Mask);

	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);
	clReleaseMemObject(c_Censored_Timepoints);

	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);

	clReleaseMemObject(d_AR1_Estimates);
	clReleaseMemObject(d_AR2_Estimates);
	clReleaseMemObject(d_AR3_Estimates);
	clReleaseMemObject(d_AR4_Estimates);
	clReleaseMemObject(d_Whitened_fMRI_Volumes);
}


// Used for testing of F-test only
void BROCCOLI_LIB::PerformGLMFTestFirstLevelWrapper()
{
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS;

	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Smoothed_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	// Allocate memory for model
	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_Censored_Timepoints = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(float), NULL, NULL);

	// Allocate memory for results
	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Residuals = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	d_AR1_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR2_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR3_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR4_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Whitened_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_Smoothed_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Smoothed_EPI_Mask , 0, NULL, NULL);

	// Copy model to constant memory
	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In , 0, NULL, NULL);
	clFinish(commandQueue);

	// Setup work sizes
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Create a mapping between voxel coordinates and brain voxel number, since we cannot store the modified GLM design matrix for all voxels, only for the brain voxels
	cl_mem d_Voxel_Numbers = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	CreateVoxelNumbers(d_Voxel_Numbers, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Allocate memory for voxel specific design matrices (sufficient to store the pseudo inverses, since we only need to estimate beta weights with the voxel-specific models, not the residuals)
	cl_mem d_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Allocate memory for voxel specific GLM scalars
	cl_mem d_GLM_Scalars = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, AR_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);

	// All timepoints are valid the first run
	NUMBER_OF_INVALID_TIMEPOINTS = 0;
	SetMemory(c_Censored_Timepoints, 1.0f, EPI_DATA_T);

	// Reset all AR parameters
	SetMemory(d_AR1_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_AR2_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_AR3_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_AR4_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);

	// Apply whitening to model (no whitening first time, so just copy regressors)
	WhitenDesignMatricesInverse(d_xtxxt_GLM, h_X_GLM_In, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS);

	// Set whitened volumes to original volumes
	clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Whitened_fMRI_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), 0, NULL, NULL);

	// Cochrane-Orcutt procedure, iterate
	for (int it = 0; it < 1; it++)
	{
		// Calculate beta values, using whitened data and the whitened voxel-specific models
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 1, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 3, sizeof(cl_mem), &d_xtxxt_GLM);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 4, sizeof(cl_mem), &d_Voxel_Numbers);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 5, sizeof(cl_mem), &c_Censored_Timepoints);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 6, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 7, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 8, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 9, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 10, sizeof(int),   &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 11, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorCalculateBetaWeightsGLMFirstLevel = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMFirstLevelKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
		clFinish(commandQueue);

		// Calculate residuals, using original data and the original model
		clSetKernelArg(CalculateGLMResidualsKernel, 0, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(CalculateGLMResidualsKernel, 1, sizeof(cl_mem), &d_fMRI_Volumes);
		clSetKernelArg(CalculateGLMResidualsKernel, 2, sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateGLMResidualsKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateGLMResidualsKernel, 4, sizeof(cl_mem), &c_X_GLM);
		clSetKernelArg(CalculateGLMResidualsKernel, 5, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(CalculateGLMResidualsKernel, 6, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(CalculateGLMResidualsKernel, 7, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(CalculateGLMResidualsKernel, 8, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(CalculateGLMResidualsKernel, 9, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		runKernelErrorCalculateGLMResiduals = clEnqueueNDRangeKernel(commandQueue, CalculateGLMResidualsKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
		clFinish(commandQueue);

		// Estimate auto correlation from residuals
		clSetKernelArg(EstimateAR4ModelsKernel, 0, sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 1, sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 2, sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 3, sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 4, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(EstimateAR4ModelsKernel, 5, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(EstimateAR4ModelsKernel, 6, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(EstimateAR4ModelsKernel, 7, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(EstimateAR4ModelsKernel, 8, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(EstimateAR4ModelsKernel, 9, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(EstimateAR4ModelsKernel, 10, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorEstimateAR4Models = clEnqueueNDRangeKernel(commandQueue, EstimateAR4ModelsKernel, 3, NULL, globalWorkSizeEstimateAR4Models, localWorkSizeEstimateAR4Models, 0, NULL, NULL);

		// Smooth auto correlation estimates
		PerformSmoothingNormalized(d_AR1_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR2_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR3_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR4_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

		// Apply whitening to data
		clSetKernelArg(ApplyWhiteningAR4Kernel, 0,  sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 1,  sizeof(cl_mem), &d_fMRI_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 2,  sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 3,  sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 4,  sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 5,  sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 6,  sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 7,  sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 8,  sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 9,  sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 10, sizeof(int),    &EPI_DATA_T);
		runKernelErrorApplyWhiteningAR4 = clEnqueueNDRangeKernel(commandQueue, ApplyWhiteningAR4Kernel, 3, NULL, globalWorkSizeApplyWhiteningAR4, localWorkSizeApplyWhiteningAR4, 0, NULL, NULL);

		// First four timepoints are now invalid
		SetMemory(c_Censored_Timepoints, 0.0f, 4);
		NUMBER_OF_INVALID_TIMEPOINTS = 4;

		// Apply whitening to model and create voxel-specific models
		WhitenDesignMatricesInverse(d_xtxxt_GLM, h_X_GLM_In, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS);
	}

	// Calculate beta values, using whitened data and the whitened voxel-specific models
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 1, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 3, sizeof(cl_mem), &d_xtxxt_GLM);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 4, sizeof(cl_mem), &d_Voxel_Numbers);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 5, sizeof(cl_mem), &c_Censored_Timepoints);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 6, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 7, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 8, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 9, sizeof(int), &EPI_DATA_T);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 10, sizeof(int), &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 11, sizeof(int), &NUMBER_OF_INVALID_TIMEPOINTS);
	runKernelErrorCalculateBetaWeightsGLMFirstLevel = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMFirstLevelKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// d_xtxxt_GLM now contains X_GLM and not xtxxt_GLM ...
	WhitenDesignMatricesFTest(d_xtxxt_GLM, d_GLM_Scalars, h_X_GLM_In, h_Contrasts_In, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS, NUMBER_OF_CONTRASTS);

	// Finally calculate statistical maps using whitened model and whitened data
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 0, sizeof(cl_mem),  &d_Statistical_Maps);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 1, sizeof(cl_mem),  &d_Residuals);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 2, sizeof(cl_mem),  &d_Residual_Variances);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 3, sizeof(cl_mem),  &d_Whitened_fMRI_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 4, sizeof(cl_mem),  &d_Beta_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 5, sizeof(cl_mem),  &d_EPI_Mask);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 6, sizeof(cl_mem),  &d_xtxxt_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 7, sizeof(cl_mem),  &d_GLM_Scalars);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 8, sizeof(cl_mem),  &d_Voxel_Numbers);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 9, sizeof(cl_mem),  &c_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 10, sizeof(cl_mem), &c_Censored_Timepoints);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 11, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 12, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 13, sizeof(int),    &EPI_DATA_D);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 14, sizeof(int),    &EPI_DATA_T);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 15, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 16, sizeof(int),    &NUMBER_OF_CONTRASTS);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 17, sizeof(int),    &NUMBER_OF_INVALID_TIMEPOINTS);
	runKernelErrorCalculateStatisticalMapsGLMFTestFirstLevel = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMFTestFirstLevelKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	MultiplyVolumes(d_AR1_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR2_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR3_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	PutWhitenedModelsIntoVolumes2(d_EPI_Mask, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, h_X_GLM_In, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS);

	// Copy results to  host
	clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Statistical_Maps_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Residuals, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);

	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);


	// Release memory
	clReleaseMemObject(d_xtxxt_GLM);
	clReleaseMemObject(d_Voxel_Numbers);
	clReleaseMemObject(d_GLM_Scalars);

	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_EPI_Mask);
	clReleaseMemObject(d_Smoothed_EPI_Mask);

	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);
	clReleaseMemObject(c_Censored_Timepoints);

	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);

	clReleaseMemObject(d_AR1_Estimates);
	clReleaseMemObject(d_AR2_Estimates);
	clReleaseMemObject(d_AR3_Estimates);
	clReleaseMemObject(d_AR4_Estimates);
	clReleaseMemObject(d_Whitened_fMRI_Volumes);
}

void BROCCOLI_LIB::PerformGLMTTestFirstLevelPermutationWrapper()
{
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS;

	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Detrended_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Whitened_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Permuted_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	d_Cluster_Indices = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int), NULL, NULL);

	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Smoothed_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_Permutation_Vector = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(unsigned short int), NULL, NULL);

	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

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

	clEnqueueWriteBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask , 0, NULL, NULL);

	//SegmentEPIData();
	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, EPI_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	PerformSmoothing(d_fMRI_Volumes, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

	//h_Permutation_Matrix = (unsigned short int*)malloc(NUMBER_OF_PERMUTATIONS * EPI_DATA_T * sizeof(unsigned short int));
	ApplyPermutationTestFirstLevel(d_fMRI_Volumes);
	//free(h_Permutation_Matrix);

	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_EPI, 0, NULL, NULL);

	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);

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

	clReleaseMemObject(d_Statistical_Maps);

	clReleaseMemObject(d_AR1_Estimates);
	clReleaseMemObject(d_AR2_Estimates);
	clReleaseMemObject(d_AR3_Estimates);
	clReleaseMemObject(d_AR4_Estimates);

	clReleaseMemObject(d_Cluster_Indices);
}




void BROCCOLI_LIB::PerformGLMFTestFirstLevelPermutationWrapper()
{
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS;

	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Detrended_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Whitened_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_Permuted_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);

	d_Cluster_Indices = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int), NULL, NULL);

	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Smoothed_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_Permutation_Vector = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(unsigned short int), NULL, NULL);

	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	d_AR1_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR2_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR3_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR4_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM_In, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In , 0, NULL, NULL);
	clFinish(commandQueue);

	clEnqueueWriteBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask , 0, NULL, NULL);

	//SegmentEPIData();

	// Smooth mask, for normalized convolution
	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, AR_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	PerformSmoothing(d_Smoothed_EPI_Mask, d_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

	//h_Permutation_Matrix = (unsigned short int*)malloc(NUMBER_OF_PERMUTATIONS * EPI_DATA_T * sizeof(unsigned short int));
	ApplyPermutationTestFirstLevel(d_fMRI_Volumes);
	//free(h_Permutation_Matrix);

	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Statistical_Maps_EPI, 0, NULL, NULL);

	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);

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

	clReleaseMemObject(d_Statistical_Maps);

	clReleaseMemObject(d_AR1_Estimates);
	clReleaseMemObject(d_AR2_Estimates);
	clReleaseMemObject(d_AR3_Estimates);
	clReleaseMemObject(d_AR4_Estimates);

	clReleaseMemObject(d_Cluster_Indices);
}

// Used for testing of t-test only
void BROCCOLI_LIB::PerformGLMTTestSecondLevelWrapper()
{
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS;

	// Allocate memory for volumes
	d_First_Level_Results = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Allocate memory for model
	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_Censored_Volumes = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);

	// Allocate memory for results
	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Residuals = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_First_Level_Results, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_First_Level_Results , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Mask , 0, NULL, NULL);

	// Copy model to constant memory
	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), h_X_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), h_xtxxt_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In , 0, NULL, NULL);
	clFinish(commandQueue);

	SetGlobalAndLocalWorkSizesStatisticalCalculations(MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	int NUMBER_OF_INVALID_VOLUMES = 0;
	SetMemory(c_Censored_Volumes, 1.0f, NUMBER_OF_SUBJECTS);

	// Calculate beta weights
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 1, sizeof(cl_mem), &d_First_Level_Results);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 2, sizeof(cl_mem), &d_MNI_Brain_Mask);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_GLM);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 4, sizeof(cl_mem), &c_Censored_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 5, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 6, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 7, sizeof(int),    &MNI_DATA_D);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 8, sizeof(int),    &NUMBER_OF_SUBJECTS);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 9, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	runKernelErrorCalculateBetaWeightsGLM = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Calculate t-values and residuals
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 0, sizeof(cl_mem),  &d_Statistical_Maps);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 1, sizeof(cl_mem),  &d_Residuals);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 2, sizeof(cl_mem),  &d_Residual_Variances);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 3, sizeof(cl_mem),  &d_First_Level_Results);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 4, sizeof(cl_mem),  &d_Beta_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 5, sizeof(cl_mem),  &d_MNI_Brain_Mask);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 6, sizeof(cl_mem),  &c_X_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 7, sizeof(cl_mem),  &c_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 8, sizeof(cl_mem),  &c_ctxtxc_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 9, sizeof(cl_mem),  &c_Censored_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 10, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 11, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 12, sizeof(int),    &MNI_DATA_D);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 13, sizeof(int),    &NUMBER_OF_SUBJECTS);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 14, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 15, sizeof(int),    &NUMBER_OF_CONTRASTS);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 16, sizeof(int),    &NUMBER_OF_INVALID_VOLUMES);
	runKernelErrorCalculateStatisticalMapsGLMTTest = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMTTestKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Copy results to  host
	clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_MNI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_Residuals, 0, NULL, NULL);

	// Release memory
	clReleaseMemObject(d_First_Level_Results);
	clReleaseMemObject(d_MNI_Brain_Mask);

	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);
	clReleaseMemObject(c_Censored_Volumes);

	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);
}

void BROCCOLI_LIB::PerformMeanSecondLevelPermutationWrapper()
{
	NUMBER_OF_TOTAL_GLM_REGRESSORS = 1;

	// Allocate memory for volumes
	d_First_Level_Results = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Cluster_Indices = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(unsigned int), NULL, NULL);
	d_Cluster_Sizes = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(unsigned int), NULL, NULL);
	d_TFCE_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Allocate memory for model
	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_Permutation_Vector = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_SUBJECTS * sizeof(unsigned short int), NULL, NULL);
	c_Sign_Vector = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);

	// Allocate memory for results
	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Residuals = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	c_Permutation_Distribution = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_PERMUTATIONS * sizeof(float), NULL, NULL);
	d_P_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_First_Level_Results, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_First_Level_Results , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

	// Copy model to constant memory
	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), h_X_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), h_xtxxt_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In , 0, NULL, NULL);
	clFinish(commandQueue);

	// Set permutation vector to not permute anything
	unsigned short int temp[NUMBER_OF_SUBJECTS];
	for (int i = 0; i < NUMBER_OF_SUBJECTS; i++)
	{
		temp[i] = (unsigned short int)i;
	}
	clEnqueueWriteBuffer(commandQueue, c_Permutation_Vector, CL_TRUE, 0, NUMBER_OF_SUBJECTS * sizeof(unsigned short int), temp , 0, NULL, NULL);

	// Run the actual permutation test
	ApplyPermutationTestSecondLevel();

	CalculateStatisticalMapsGLMTTestSecondLevel(d_First_Level_Results, d_MNI_Brain_Mask);

	if (INFERENCE_MODE != TFCE)
	{
		ClusterizeOpenCL(d_Cluster_Indices, d_Cluster_Sizes, MAX_CLUSTER, d_Statistical_Maps, CLUSTER_DEFINING_THRESHOLD, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
	}
	else
	{
		//ClusterizeOpenCLTFCE(d_Cluster_Indices, d_Cluster_Sizes, MAX_CLUSTER, d_Statistical_Maps, CLUSTER_DEFINING_THRESHOLD, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
	}

	CalculatePermutationPValues(d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	// Copy results to  host
	//clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_Residuals, 0, NULL, NULL);

	clEnqueueReadBuffer(commandQueue, d_P_Values, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_P_Values_MNI, 0, NULL, NULL);

	//Clusterize(h_Cluster_Indices, MAX_CLUSTER_SIZE, MAX_CLUSTER_MASS, NUMBER_OF_CLUSTERS, h_Statistical_Maps, CLUSTER_DEFINING_THRESHOLD, h_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, CALCULATE_VOXEL_LABELS, CALCULATE_CLUSTER_MASS);

	// Release memory
	clReleaseMemObject(d_First_Level_Results);
	clReleaseMemObject(d_MNI_Brain_Mask);
	clReleaseMemObject(d_Cluster_Indices);
	clReleaseMemObject(d_Cluster_Sizes);
	clReleaseMemObject(d_TFCE_Values);

	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);
	clReleaseMemObject(c_Permutation_Vector);
	clReleaseMemObject(c_Sign_Vector);

	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);
	clReleaseMemObject(c_Permutation_Distribution);
	clReleaseMemObject(d_P_Values);
}


void BROCCOLI_LIB::PerformGLMTTestSecondLevelPermutationWrapper()
{
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS;

	// Allocate memory for volumes
	d_First_Level_Results = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Cluster_Indices = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), NULL, NULL);
	d_Cluster_Sizes = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), NULL, NULL);
	d_TFCE_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Allocate memory for model
	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_Permutation_Vector = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_SUBJECTS * sizeof(unsigned short int), NULL, NULL);

	// Allocate memory for results
	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Residuals = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	c_Permutation_Distribution = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_PERMUTATIONS * sizeof(float), NULL, NULL);
	d_P_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_First_Level_Results, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_First_Level_Results , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

	// Copy model to constant memory
	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), h_X_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), h_xtxxt_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In , 0, NULL, NULL);
	clFinish(commandQueue);

	// Run the actual permutation test
	ApplyPermutationTestSecondLevel();

	CalculateStatisticalMapsGLMTTestSecondLevel(d_First_Level_Results, d_MNI_Brain_Mask);

	if (INFERENCE_MODE != TFCE)
	{
		ClusterizeOpenCL(d_Cluster_Indices, d_Cluster_Sizes, MAX_CLUSTER, d_Statistical_Maps, CLUSTER_DEFINING_THRESHOLD, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
	}
	CalculatePermutationPValues(d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	// Copy results to  host
	//clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_MNI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_Residuals, 0, NULL, NULL);

	clEnqueueReadBuffer(commandQueue, d_P_Values, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_P_Values_MNI, 0, NULL, NULL);

	//Clusterize(h_Cluster_Indices, MAX_CLUSTER_SIZE, MAX_CLUSTER_MASS, NUMBER_OF_CLUSTERS, h_Statistical_Maps, CLUSTER_DEFINING_THRESHOLD, h_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, CALCULATE_VOXEL_LABELS, CALCULATE_CLUSTER_MASS);

	// Release memory
	clReleaseMemObject(d_First_Level_Results);
	clReleaseMemObject(d_MNI_Brain_Mask);
	clReleaseMemObject(d_Cluster_Indices);
	clReleaseMemObject(d_Cluster_Sizes);
	clReleaseMemObject(d_TFCE_Values);

	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);
	clReleaseMemObject(c_Permutation_Vector);

	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);
	clReleaseMemObject(c_Permutation_Distribution);
	clReleaseMemObject(d_P_Values);
}

// Used for testing of F-test only
void BROCCOLI_LIB::PerformGLMFTestSecondLevelWrapper()
{
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS;

	// Allocate memory for volumes
	d_First_Level_Results = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Allocate memory for model
	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_Censored_Volumes = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);

	// Allocate memory for results
	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Residuals = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_First_Level_Results, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_First_Level_Results , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Mask, 0, NULL, NULL);

	// Copy model to constant memory
	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), h_X_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), h_xtxxt_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In , 0, NULL, NULL);
	clFinish(commandQueue);

	SetGlobalAndLocalWorkSizesStatisticalCalculations(MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	int NUMBER_OF_INVALID_VOLUMES = 0;
	SetMemory(c_Censored_Volumes, 1.0f, NUMBER_OF_SUBJECTS);

	// Calculate beta weights
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 1, sizeof(cl_mem), &d_First_Level_Results);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 2, sizeof(cl_mem), &d_MNI_Brain_Mask);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_GLM);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 4, sizeof(cl_mem), &c_Censored_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 5, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 6, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 7, sizeof(int),    &MNI_DATA_D);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 8, sizeof(int),    &NUMBER_OF_SUBJECTS);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 9, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	runKernelErrorCalculateBetaWeightsGLM = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Calculate F-values and residuals
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 0, sizeof(cl_mem),  &d_Statistical_Maps);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 1, sizeof(cl_mem),  &d_Residuals);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 2, sizeof(cl_mem),  &d_Residual_Variances);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 3, sizeof(cl_mem),  &d_First_Level_Results);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 4, sizeof(cl_mem),  &d_Beta_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 5, sizeof(cl_mem),  &d_MNI_Brain_Mask);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 6, sizeof(cl_mem),  &c_X_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 7, sizeof(cl_mem),  &c_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 8, sizeof(cl_mem),  &c_ctxtxc_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 9, sizeof(cl_mem),  &c_Censored_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 10, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 11, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 12, sizeof(int),    &MNI_DATA_D);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 13, sizeof(int),    &NUMBER_OF_SUBJECTS);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 14, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 15, sizeof(int),    &NUMBER_OF_CONTRASTS);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 16, sizeof(int),    &NUMBER_OF_INVALID_VOLUMES);
	runKernelErrorCalculateStatisticalMapsGLMFTest = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMFTestKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Copy results to  host
	clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_MNI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_Residuals, 0, NULL, NULL);

	// Release memory
	clReleaseMemObject(d_First_Level_Results);
	clReleaseMemObject(d_MNI_Brain_Mask);

	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);
	clReleaseMemObject(c_Censored_Volumes);

	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);
}

// Used for testing of F-test only
void BROCCOLI_LIB::PerformGLMFTestSecondLevelPermutationWrapper()
{
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS;

	// Allocate memory for volumes
	d_First_Level_Results = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Cluster_Indices = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), NULL, NULL);
	d_Cluster_Sizes = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), NULL, NULL);
	d_TFCE_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Allocate memory for model
	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_Permutation_Vector = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_SUBJECTS * sizeof(unsigned short int), NULL, NULL);

	// Allocate memory for results
	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Residuals = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_First_Level_Results, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_First_Level_Results , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask, 0, NULL, NULL);

	// Copy model to constant memory
	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), h_X_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float), h_xtxxt_GLM_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts_In , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM_In , 0, NULL, NULL);
	clFinish(commandQueue);

	// Run the actual permutation test
	ApplyPermutationTestSecondLevel();


	CalculateStatisticalMapsGLMFTestSecondLevel(d_First_Level_Results, d_MNI_Brain_Mask);

	// Copy results to  host
	clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_MNI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_Residuals, 0, NULL, NULL);

	//Clusterize(h_Cluster_Indices, MAX_CLUSTER_SIZE, MAX_CLUSTER_MASS, NUMBER_OF_CLUSTERS, h_Statistical_Maps, CLUSTER_DEFINING_THRESHOLD, h_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, CALCULATE_VOXEL_LABELS, CALCULATE_CLUSTER_MASS);

	// Release memory
	clReleaseMemObject(d_First_Level_Results);
	clReleaseMemObject(d_MNI_Brain_Mask);
	clReleaseMemObject(d_Cluster_Indices);
	clReleaseMemObject(d_Cluster_Sizes);
	clReleaseMemObject(d_TFCE_Values);

	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);
	clReleaseMemObject(c_Permutation_Vector);

	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);
}

// Generates a number (index) for each brain voxel, for storing design matrices for brain voxels only
void BROCCOLI_LIB::CreateVoxelNumbers(cl_mem d_Voxel_Numbers, cl_mem d_Mask, int DATA_W, int DATA_H, int DATA_D)
{
	float* h_Voxel_Numbers = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));
	float* h_Mask = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));

	clEnqueueReadBuffer(commandQueue, d_Mask, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Mask, 0, NULL, NULL);

	float voxel_number = 0.0f;
	for (int z = 0; z < DATA_D; z++)
	{
		for (int y = 0; y < DATA_H; y++)
		{
			for (int x = 0; x < DATA_W; x++)
			{
				h_Voxel_Numbers[x + y * DATA_W + z * DATA_W * DATA_H] = 0.0f;
				if ( h_Mask[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
				{
					h_Voxel_Numbers[x + y * DATA_W + z * DATA_W * DATA_H] = voxel_number;
					voxel_number += 1.0f;
				}
			}
		}
	}

	NUMBER_OF_BRAIN_VOXELS = (int)voxel_number;

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_Voxel_Numbers, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Voxel_Numbers, 0, NULL, NULL);

	free(h_Voxel_Numbers);
	free(h_Mask);
}


// Applies whitening to design matrix, different for each voxel, saves the pseudo inverse
void BROCCOLI_LIB::WhitenDesignMatricesInverse(cl_mem d_xtxxt_GLM,
		                                       float* h_X_GLM,
		                                       cl_mem d_AR1_Estimates,
		                                       cl_mem d_AR2_Estimates,
		                                       cl_mem d_AR3_Estimates,
		                                       cl_mem d_AR4_Estimates,
		                                       cl_mem d_Mask,
		                                       int DATA_W,
		                                       int DATA_H,
		                                       int DATA_D,
		                                       int DATA_T,
		                                       int NUMBER_OF_REGRESSORS,
		                                       int NUMBER_OF_INVALID_TIMEPOINTS)
{
	float* h_Mask = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));
	float* h_xtxxt_GLM_ = (float*)malloc(NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float));

	clEnqueueReadBuffer(commandQueue, d_Mask, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Mask, 0, NULL, NULL);

	// Copy AR parameters to host
	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);

	// Loop over voxels
	int voxel_number = 0;
	for (int z = 0; z < DATA_D; z++)
	{
		for (int y = 0; y < DATA_H; y++)
		{
			for (int x = 0; x < DATA_W; x++)
			{
				if ( h_Mask[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
				{
					Eigen::MatrixXd X(DATA_T,NUMBER_OF_REGRESSORS);

					// Get AR parameters for current voxel
					float AR1 = h_AR1_Estimates_EPI[x + y * DATA_W + z * DATA_W * DATA_H];
					float AR2 = h_AR2_Estimates_EPI[x + y * DATA_W + z * DATA_W * DATA_H];
					float AR3 = h_AR3_Estimates_EPI[x + y * DATA_W + z * DATA_W * DATA_H];
					float AR4 = h_AR4_Estimates_EPI[x + y * DATA_W + z * DATA_W * DATA_H];

					float old_value_1, old_value_2, old_value_3, old_value_4, old_value_5;

					// Whiten original regressors
					for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
					{
					    old_value_1 = h_X_GLM[0 + r * DATA_T];
						X(0,r) = old_value_1;
						old_value_2 = h_X_GLM[1 + r * DATA_T];
						X(1,r) = old_value_2  - AR1 * old_value_1;
						old_value_3 = h_X_GLM[2 + r * DATA_T];
						X(2,r) = old_value_3 - AR1 * old_value_2 - AR2 * old_value_1;
						old_value_4 = h_X_GLM[3 + r * DATA_T];
						X(3,r) = old_value_4 - AR1 * old_value_3 - AR2 * old_value_2 - AR3 * old_value_1;

						for (int t = 4; t < DATA_T; t++)
						{
							old_value_5 = h_X_GLM[t + r * DATA_T];
							X(t,r) = old_value_5 - AR1 * old_value_4 - AR2 * old_value_3 - AR3 * old_value_2 - AR4 * old_value_1;

							// Save old values
							old_value_1 = old_value_2;
							old_value_2 = old_value_3;
							old_value_3 = old_value_4;
							old_value_4 = old_value_5;
						}
					}

					// Set invalid timepoints to 0 in the design matrix, since they affect the pseudo inverse
					for (int t = 0; t < NUMBER_OF_INVALID_TIMEPOINTS; t++)
					{
						for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
						{
							X(t,r) = 0.0;
						}
					}

					// Calculate pseudo inverse in an ugly way
					Eigen::MatrixXd xtx(NUMBER_OF_REGRESSORS,NUMBER_OF_REGRESSORS);
					xtx = X.transpose() * X;
					Eigen::MatrixXd inv_xtx = xtx.inverse();
					Eigen::MatrixXd xtxxt = inv_xtx * X.transpose();

					// Put whitened regressors into specific format, to copy to GPU
					// (takes too much memory to store regressors for all voxels, so only store for brain voxels)
					for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
					{
						for (int t = 0; t < DATA_T; t++)
						{
							h_xtxxt_GLM_[voxel_number * NUMBER_OF_REGRESSORS * DATA_T + r * DATA_T + t] = xtxxt(r,t);
						}
					}
					voxel_number++;
				}
			}
		}
	}

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float), h_xtxxt_GLM_, 0, NULL, NULL);

	free(h_Mask);
	free(h_xtxxt_GLM_);
}


// Applies whitening to design matrix, different for each voxel, saves the whitened matrix
void BROCCOLI_LIB::WhitenDesignMatricesTTest(cl_mem d_X_GLM,
		                                	 cl_mem d_GLM_Scalars,
		                                	 float* h_X_GLM,
		                                	 float* h_Contrasts,
		                                	 cl_mem d_AR1_Estimates,
		                                	 cl_mem d_AR2_Estimates,
		                                	 cl_mem d_AR3_Estimates,
		                                	 cl_mem d_AR4_Estimates,
		                                	 cl_mem d_Mask,
		                                	 int DATA_W,
		                                	 int DATA_H,
		                                	 int DATA_D,
		                                	 int DATA_T,
		                                	 int NUMBER_OF_REGRESSORS,
		                                	 int NUMBER_OF_INVALID_TIMEPOINTS,
		                                	 int NUMBER_OF_CONTRASTS)
{
	float* h_Mask = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));
	float* h_X_GLM_ = (float*)malloc(NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float));
	float* h_GLM_Scalars = (float*)malloc(DATA_W * DATA_H * DATA_D * NUMBER_OF_CONTRASTS * sizeof(float));

	clEnqueueReadBuffer(commandQueue, d_Mask, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Mask, 0, NULL, NULL);

	// Copy AR parameters to host
	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);

	// Loop over voxels
	int voxel_number = 0;
	for (int z = 0; z < DATA_D; z++)
	{
		for (int y = 0; y < DATA_H; y++)
		{
			for (int x = 0; x < DATA_W; x++)
			{
				if ( h_Mask[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
				{
					Eigen::MatrixXd X(DATA_T,NUMBER_OF_REGRESSORS);

					// Get AR parameters for current voxel
					float AR1 = h_AR1_Estimates_EPI[x + y * DATA_W + z * DATA_W * DATA_H];
					float AR2 = h_AR2_Estimates_EPI[x + y * DATA_W + z * DATA_W * DATA_H];
					float AR3 = h_AR3_Estimates_EPI[x + y * DATA_W + z * DATA_W * DATA_H];
					float AR4 = h_AR4_Estimates_EPI[x + y * DATA_W + z * DATA_W * DATA_H];

					float old_value_1, old_value_2, old_value_3, old_value_4, old_value_5;

					// Whiten original regressors
					for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
					{
					    old_value_1 = h_X_GLM[0 + r * DATA_T];
						X(0,r) = old_value_1;
						old_value_2 = h_X_GLM[1 + r * DATA_T];
						X(1,r) = old_value_2  - AR1 * old_value_1;
						old_value_3 = h_X_GLM[2 + r * DATA_T];
						X(2,r) = old_value_3 - AR1 * old_value_2 - AR2 * old_value_1;
						old_value_4 = h_X_GLM[3 + r * DATA_T];
						X(3,r) = old_value_4 - AR1 * old_value_3 - AR2 * old_value_2 - AR3 * old_value_1;

						for (int t = 4; t < DATA_T; t++)
						{
							old_value_5 = h_X_GLM[t + r * DATA_T];
							X(t,r) = old_value_5 - AR1 * old_value_4 - AR2 * old_value_3 - AR3 * old_value_2 - AR4 * old_value_1;

							// Save old values
							old_value_1 = old_value_2;
							old_value_2 = old_value_3;
							old_value_3 = old_value_4;
							old_value_4 = old_value_5;
						}
					}

					// Set invalid timepoints to 0 in the design matrix
					for (int t = 0; t < NUMBER_OF_INVALID_TIMEPOINTS; t++)
					{
						for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
						{
							X(t,r) = 0.0;
						}
					}

					// Calculate contrast scalars
					Eigen::MatrixXd xtx(NUMBER_OF_REGRESSORS,NUMBER_OF_REGRESSORS);
					xtx = X.transpose() * X;
					Eigen::MatrixXd inv_xtx = xtx.inverse();

					for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
					{
						Eigen::MatrixXd Contrast(NUMBER_OF_REGRESSORS,1);

						for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
						{
							Contrast(r) = (double)h_Contrasts[NUMBER_OF_REGRESSORS * c + r];
						}

						Eigen::MatrixXd GLM_scalar = Contrast.transpose() * inv_xtx * Contrast;
						h_GLM_Scalars[x + y * DATA_W + z * DATA_W * DATA_H + c * DATA_W * DATA_H * DATA_D] = GLM_scalar(0);
					}

					// Put whitened regressors into specific format, to copy to GPU
					// (takes too much memory to store regressors for all voxels, so only store for brain voxels)
					for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
					{
						for (int t = 0; t < DATA_T; t++)
						{
							h_X_GLM_[voxel_number * NUMBER_OF_REGRESSORS * DATA_T + r * DATA_T + t] = X(t,r);
						}
					}
					voxel_number++;
				}
			}
		}
	}

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_X_GLM, CL_TRUE, 0, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float), h_X_GLM_, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_GLM_Scalars, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_GLM_Scalars, 0, NULL, NULL);

	free(h_Mask);
	free(h_X_GLM_);
	free(h_GLM_Scalars);
}


// Applies whitening to design matrix, different for each voxel, saves the whitened matrix
void BROCCOLI_LIB::WhitenDesignMatricesFTest(cl_mem d_X_GLM,
		                                	 cl_mem d_GLM_Scalars,
		                                	 float* h_X_GLM,
		                                	 float* h_Contrasts,
		                                	 cl_mem d_AR1_Estimates,
		                                	 cl_mem d_AR2_Estimates,
		                                	 cl_mem d_AR3_Estimates,
		                                	 cl_mem d_AR4_Estimates,
		                                	 cl_mem d_Mask,
		                                	 int DATA_W,
		                                	 int DATA_H,
		                                	 int DATA_D,
		                                	 int DATA_T,
		                                	 int NUMBER_OF_REGRESSORS,
		                                	 int NUMBER_OF_INVALID_TIMEPOINTS,
		                                	 int NUMBER_OF_CONTRASTS)
{
	float* h_Mask = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));
	float* h_X_GLM_ = (float*)malloc(NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float));
	float* h_GLM_Scalars = (float*)malloc(DATA_W * DATA_H * DATA_D * NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float));

	clEnqueueReadBuffer(commandQueue, d_Mask, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Mask, 0, NULL, NULL);

	// Copy AR parameters to host
	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);

	// Insert contrast into eigen variable
	Eigen::MatrixXd Contrasts(NUMBER_OF_CONTRASTS,NUMBER_OF_REGRESSORS);

	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
		{
			Contrasts(c,r) = (double)h_Contrasts[NUMBER_OF_REGRESSORS * c + r];
		}
	}

	// Loop over voxels
	int voxel_number = 0;
	for (int z = 0; z < DATA_D; z++)
	{
		for (int y = 0; y < DATA_H; y++)
		{
			for (int x = 0; x < DATA_W; x++)
			{
				if ( h_Mask[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
				{
					Eigen::MatrixXd X(DATA_T,NUMBER_OF_REGRESSORS);

					// Get AR parameters for current voxel
					float AR1 = h_AR1_Estimates_EPI[x + y * DATA_W + z * DATA_W * DATA_H];
					float AR2 = h_AR2_Estimates_EPI[x + y * DATA_W + z * DATA_W * DATA_H];
					float AR3 = h_AR3_Estimates_EPI[x + y * DATA_W + z * DATA_W * DATA_H];
					float AR4 = h_AR4_Estimates_EPI[x + y * DATA_W + z * DATA_W * DATA_H];

					float old_value_1, old_value_2, old_value_3, old_value_4, old_value_5;

					// Whiten original regressors
					for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
					{
					    old_value_1 = h_X_GLM[0 + r * DATA_T];
						X(0,r) = old_value_1;
						old_value_2 = h_X_GLM[1 + r * DATA_T];
						X(1,r) = old_value_2  - AR1 * old_value_1;
						old_value_3 = h_X_GLM[2 + r * DATA_T];
						X(2,r) = old_value_3 - AR1 * old_value_2 - AR2 * old_value_1;
						old_value_4 = h_X_GLM[3 + r * DATA_T];
						X(3,r) = old_value_4 - AR1 * old_value_3 - AR2 * old_value_2 - AR3 * old_value_1;

						for (int t = 4; t < DATA_T; t++)
						{
							old_value_5 = h_X_GLM[t + r * DATA_T];
							X(t,r) = old_value_5 - AR1 * old_value_4 - AR2 * old_value_3 - AR3 * old_value_2 - AR4 * old_value_1;

							// Save old values
							old_value_1 = old_value_2;
							old_value_2 = old_value_3;
							old_value_3 = old_value_4;
							old_value_4 = old_value_5;
						}
					}

					// Set invalid timepoints to 0 in the design matrix
					for (int t = 0; t < NUMBER_OF_INVALID_TIMEPOINTS; t++)
					{
						for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
						{
							X(t,r) = 0.0;
						}
					}

					// Calculate contrast values

					Eigen::MatrixXd xtx(NUMBER_OF_REGRESSORS,NUMBER_OF_REGRESSORS);
					xtx = X.transpose() * X;
					Eigen::MatrixXd inv_xtx = xtx.inverse();

					Eigen::MatrixXd temp = Contrasts * inv_xtx * Contrasts.transpose();
					Eigen::MatrixXd ctxtxc = temp.inverse();

					for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
					{
						for (int cc = 0; cc < NUMBER_OF_CONTRASTS; cc++)
						{
							h_GLM_Scalars[x + y * DATA_W + z * DATA_W * DATA_H + (cc + c * NUMBER_OF_CONTRASTS) * DATA_W * DATA_H * DATA_D] = (float)ctxtxc(c,cc);
						}
					}

					// Put whitened regressors into specific format, to copy to GPU
					// (takes too much memory to store regressors for all voxels, so only store for brain voxels)
					for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
					{
						for (int t = 0; t < DATA_T; t++)
						{
							h_X_GLM_[voxel_number * NUMBER_OF_REGRESSORS * DATA_T + r * DATA_T + t] = X(t,r);
						}
					}
					voxel_number++;
				}
			}
		}
	}

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_X_GLM, CL_TRUE, 0, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float), h_X_GLM_, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_GLM_Scalars, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float), h_GLM_Scalars, 0, NULL, NULL);

	free(h_Mask);
	free(h_X_GLM_);
	free(h_GLM_Scalars);
}



// Calculates a statistical map for first level analysis, using a Cochrane-Orcutt procedure

void BROCCOLI_LIB::CalculateStatisticalMapsGLMTTestFirstLevel(cl_mem d_Volumes, int iterations)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Create a mapping between voxel coordinates and brain voxel number, since we cannot store the modified GLM design matrix for all voxels, only for the brain voxels
	cl_mem d_Voxel_Numbers = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	CreateVoxelNumbers(d_Voxel_Numbers, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Allocate memory for voxel specific design matrices (sufficient to store the pseudo inverses, since we only need to estimate beta weights with the voxel-specific models, not the residuals)
	cl_mem d_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Allocate memory for voxel specific GLM scalars
	cl_mem d_GLM_Scalars = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, AR_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	PerformSmoothing(d_Smoothed_EPI_Mask, d_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

	// All timepoints are valid the first run
	NUMBER_OF_INVALID_TIMEPOINTS = 0;
	c_Censored_Timepoints = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(float), NULL, NULL);
	SetMemory(c_Censored_Timepoints, 1.0f, EPI_DATA_T);

	// Reset all AR parameters
	SetMemory(d_AR1_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_AR2_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_AR3_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_AR4_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);

	// Apply whitening to model (no whitening first time, so just copy regressors)
	WhitenDesignMatricesInverse(d_xtxxt_GLM, h_X_GLM, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS);

	// Set whitened volumes to original volumes
	clEnqueueCopyBuffer(commandQueue, d_Volumes, d_Whitened_fMRI_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), 0, NULL, NULL);

	// Cochrane-Orcutt procedure, iterate
	for (int it = 0; it < iterations; it++)
	{
		// Calculate beta values, using whitened data and the whitened voxel-specific models
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 0,  sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 1,  sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 2,  sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 3,  sizeof(cl_mem), &d_xtxxt_GLM);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 4,  sizeof(cl_mem), &d_Voxel_Numbers);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 5,  sizeof(cl_mem), &c_Censored_Timepoints);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 6,  sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 7,  sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 8,  sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 9,  sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 10, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 11, sizeof(int),    &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorCalculateBetaWeightsGLMFirstLevel = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMFirstLevelKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
		clFinish(commandQueue);

		// Calculate residuals, using original data and the original model
		clSetKernelArg(CalculateGLMResidualsKernel, 0, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(CalculateGLMResidualsKernel, 1, sizeof(cl_mem), &d_Volumes);
		clSetKernelArg(CalculateGLMResidualsKernel, 2, sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateGLMResidualsKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateGLMResidualsKernel, 4, sizeof(cl_mem), &c_X_GLM);
		clSetKernelArg(CalculateGLMResidualsKernel, 5, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(CalculateGLMResidualsKernel, 6, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(CalculateGLMResidualsKernel, 7, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(CalculateGLMResidualsKernel, 8, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(CalculateGLMResidualsKernel, 9, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		runKernelErrorCalculateGLMResiduals = clEnqueueNDRangeKernel(commandQueue, CalculateGLMResidualsKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
		clFinish(commandQueue);

		// Estimate auto correlation from residuals
		clSetKernelArg(EstimateAR4ModelsKernel, 0, sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 1, sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 2, sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 3, sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 4, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(EstimateAR4ModelsKernel, 5, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(EstimateAR4ModelsKernel, 6, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(EstimateAR4ModelsKernel, 7, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(EstimateAR4ModelsKernel, 8, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(EstimateAR4ModelsKernel, 9, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(EstimateAR4ModelsKernel, 10, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorEstimateAR4Models = clEnqueueNDRangeKernel(commandQueue, EstimateAR4ModelsKernel, 3, NULL, globalWorkSizeEstimateAR4Models, localWorkSizeEstimateAR4Models, 0, NULL, NULL);

		// Smooth auto correlation estimates
		PerformSmoothingNormalized(d_AR1_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR2_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR3_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR4_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

		// Apply whitening to data
		clSetKernelArg(ApplyWhiteningAR4Kernel, 0,  sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 1,  sizeof(cl_mem), &d_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 2,  sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 3,  sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 4,  sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 5,  sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 6,  sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 7,  sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 8,  sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 9,  sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 10, sizeof(int),    &EPI_DATA_T);
		runKernelErrorApplyWhiteningAR4 = clEnqueueNDRangeKernel(commandQueue, ApplyWhiteningAR4Kernel, 3, NULL, globalWorkSizeApplyWhiteningAR4, localWorkSizeApplyWhiteningAR4, 0, NULL, NULL);

		// First four timepoints are now invalid
		SetMemory(c_Censored_Timepoints, 0.0f, 4);
		NUMBER_OF_INVALID_TIMEPOINTS = 4;

		// Apply whitening to model and create voxel-specific models
		WhitenDesignMatricesInverse(d_xtxxt_GLM, h_X_GLM, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS);
	}

	// Calculate beta values, using whitened data and the whitened voxel-specific models
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 0,  sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 1,  sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 2,  sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 3,  sizeof(cl_mem), &d_xtxxt_GLM);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 4,  sizeof(cl_mem), &d_Voxel_Numbers);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 5,  sizeof(cl_mem), &c_Censored_Timepoints);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 6,  sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 7,  sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 8,  sizeof(int),    &EPI_DATA_D);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 9,  sizeof(int),    &EPI_DATA_T);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 10, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 11, sizeof(int),    &NUMBER_OF_INVALID_TIMEPOINTS);
	runKernelErrorCalculateBetaWeightsGLMFirstLevel = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMFirstLevelKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// d_xtxxt_GLM now contains X_GLM and not xtxxt_GLM ...
	WhitenDesignMatricesTTest(d_xtxxt_GLM, d_GLM_Scalars, h_X_GLM, h_Contrasts, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS, NUMBER_OF_CONTRASTS);

	// Finally calculate statistical maps using whitened model and whitened data
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 0,  sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 1,  sizeof(cl_mem), &d_Contrast_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 2,  sizeof(cl_mem), &d_Residuals);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 3,  sizeof(cl_mem), &d_Residual_Variances);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 4,  sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 5,  sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 6,  sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 7,  sizeof(cl_mem), &d_xtxxt_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 8,  sizeof(cl_mem), &d_GLM_Scalars);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 9,  sizeof(cl_mem), &d_Voxel_Numbers);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 10, sizeof(cl_mem), &c_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 11, sizeof(cl_mem), &c_Censored_Timepoints);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 12, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 13, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 14, sizeof(int),    &EPI_DATA_D);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 15, sizeof(int),    &EPI_DATA_T);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 16, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 17, sizeof(int),    &NUMBER_OF_CONTRASTS);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 18, sizeof(int),    &NUMBER_OF_INVALID_TIMEPOINTS);
	runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMTTestFirstLevelKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	MultiplyVolumes(d_AR1_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR2_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR3_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	//clEnqueueReadBuffer(commandQueue, d_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Smoothed_fMRI_Volumes, 0, NULL, NULL);

	//clEnqueueReadBuffer(commandQueue, d_Whitened_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Motion_Corrected_fMRI_Volumes, 0, NULL, NULL);

	//PutWhitenedModelsIntoVolumes(d_EPI_Mask, d_xtxxt_GLM, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS);
	//PutWhitenedModelsIntoVolumes2(d_EPI_Mask, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, h_X_GLM, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS);

	clReleaseMemObject(d_xtxxt_GLM);
	clReleaseMemObject(d_GLM_Scalars);
	clReleaseMemObject(d_Voxel_Numbers);
	clReleaseMemObject(c_Censored_Timepoints);
}

// Calculates a statistical map for first level analysis, using a Cochrane-Orcutt procedure

void BROCCOLI_LIB::CalculateStatisticalMapsGLMFTestFirstLevel(cl_mem d_Volumes, int iterations)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Create a mapping between voxel coordinates and brain voxel number, since we cannot store the modified GLM design matrix for all voxels, only for the brain voxels
	cl_mem d_Voxel_Numbers = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	CreateVoxelNumbers(d_Voxel_Numbers, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Allocate memory for voxel specific design matrices (sufficient to store the pseudo inverses, since we only need to estimate beta weights with the voxel-specific models, not the residuals)
	cl_mem d_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);

	// Allocate memory for voxel specific GLM scalars
	cl_mem d_GLM_Scalars = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, AR_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	PerformSmoothing(d_Smoothed_EPI_Mask, d_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

	// All timepoints are valid the first run
	NUMBER_OF_INVALID_TIMEPOINTS = 0;
	c_Censored_Timepoints = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(float), NULL, NULL);
	SetMemory(c_Censored_Timepoints, 1.0f, EPI_DATA_T);

	// Reset all AR parameters
	SetMemory(d_AR1_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_AR2_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_AR3_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_AR4_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);

	// Apply whitening to model (no whitening first time, so just copy regressors)
	WhitenDesignMatricesInverse(d_xtxxt_GLM, h_X_GLM, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS);

	// Set whitened volumes to original volumes
	clEnqueueCopyBuffer(commandQueue, d_Volumes, d_Whitened_fMRI_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), 0, NULL, NULL);

	// Cochrane-Orcutt procedure, iterate
	for (int it = 0; it < iterations; it++)
	{
		// Calculate beta values, using whitened data and the whitened voxel-specific models
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 1, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 3, sizeof(cl_mem), &d_xtxxt_GLM);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 4, sizeof(cl_mem), &d_Voxel_Numbers);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 5, sizeof(cl_mem), &c_Censored_Timepoints);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 6, sizeof(int), &EPI_DATA_W);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 7, sizeof(int), &EPI_DATA_H);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 8, sizeof(int), &EPI_DATA_D);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 9, sizeof(int), &EPI_DATA_T);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 10, sizeof(int), &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 11, sizeof(int), &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorCalculateBetaWeightsGLMFirstLevel = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMFirstLevelKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
		clFinish(commandQueue);

		// Calculate residuals, using original data and the original model
		clSetKernelArg(CalculateGLMResidualsKernel, 0, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(CalculateGLMResidualsKernel, 1, sizeof(cl_mem), &d_Volumes);
		clSetKernelArg(CalculateGLMResidualsKernel, 2, sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateGLMResidualsKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateGLMResidualsKernel, 4, sizeof(cl_mem), &c_X_GLM);
		clSetKernelArg(CalculateGLMResidualsKernel, 5, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(CalculateGLMResidualsKernel, 6, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(CalculateGLMResidualsKernel, 7, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(CalculateGLMResidualsKernel, 8, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(CalculateGLMResidualsKernel, 9, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		runKernelErrorCalculateGLMResiduals = clEnqueueNDRangeKernel(commandQueue, CalculateGLMResidualsKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
		clFinish(commandQueue);

		// Estimate auto correlation from residuals
		clSetKernelArg(EstimateAR4ModelsKernel, 0, sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 1, sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 2, sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 3, sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 4, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(EstimateAR4ModelsKernel, 5, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(EstimateAR4ModelsKernel, 6, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(EstimateAR4ModelsKernel, 7, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(EstimateAR4ModelsKernel, 8, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(EstimateAR4ModelsKernel, 9, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(EstimateAR4ModelsKernel, 10, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorEstimateAR4Models = clEnqueueNDRangeKernel(commandQueue, EstimateAR4ModelsKernel, 3, NULL, globalWorkSizeEstimateAR4Models, localWorkSizeEstimateAR4Models, 0, NULL, NULL);

		// Smooth auto correlation estimates
		PerformSmoothingNormalized(d_AR1_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR2_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR3_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR4_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

		// Apply whitening to data
		clSetKernelArg(ApplyWhiteningAR4Kernel, 0,  sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 1,  sizeof(cl_mem), &d_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 2,  sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 3,  sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 4,  sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 5,  sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 6,  sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 7,  sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 8,  sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 9,  sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 10, sizeof(int),    &EPI_DATA_T);
		runKernelErrorApplyWhiteningAR4 = clEnqueueNDRangeKernel(commandQueue, ApplyWhiteningAR4Kernel, 3, NULL, globalWorkSizeApplyWhiteningAR4, localWorkSizeApplyWhiteningAR4, 0, NULL, NULL);

		// First four timepoints are now invalid
		SetMemory(c_Censored_Timepoints, 0.0f, 4);
		NUMBER_OF_INVALID_TIMEPOINTS = 4;

		// Apply whitening to model and create voxel-specific models
		WhitenDesignMatricesInverse(d_xtxxt_GLM, h_X_GLM, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS);
	}

	// Calculate beta values, using whitened data and the whitened voxel-specific models
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 1, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 3, sizeof(cl_mem), &d_xtxxt_GLM);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 4, sizeof(cl_mem), &d_Voxel_Numbers);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 5, sizeof(cl_mem), &c_Censored_Timepoints);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 6, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 7, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 8, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 9, sizeof(int), &EPI_DATA_T);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 10, sizeof(int), &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(CalculateBetaWeightsGLMFirstLevelKernel, 11, sizeof(int), &NUMBER_OF_INVALID_TIMEPOINTS);
	runKernelErrorCalculateBetaWeightsGLMFirstLevel = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMFirstLevelKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// d_xtxxt_GLM now contains X_GLM and not xtxxt_GLM ...
	WhitenDesignMatricesFTest(d_xtxxt_GLM, d_GLM_Scalars, h_X_GLM, h_Contrasts, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS, NUMBER_OF_CONTRASTS);

	// Finally calculate statistical maps using whitened model and whitened data
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 0, sizeof(cl_mem),  &d_Statistical_Maps);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 1, sizeof(cl_mem),  &d_Residuals);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 2, sizeof(cl_mem),  &d_Residual_Variances);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 3, sizeof(cl_mem),  &d_Whitened_fMRI_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 4, sizeof(cl_mem),  &d_Beta_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 5, sizeof(cl_mem),  &d_EPI_Mask);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 6, sizeof(cl_mem),  &d_xtxxt_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 7, sizeof(cl_mem),  &d_GLM_Scalars);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 8, sizeof(cl_mem),  &d_Voxel_Numbers);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 9, sizeof(cl_mem),  &c_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 10, sizeof(cl_mem), &c_Censored_Timepoints);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 11, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 12, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 13, sizeof(int),    &EPI_DATA_D);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 14, sizeof(int),    &EPI_DATA_T);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 15, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 16, sizeof(int),    &NUMBER_OF_CONTRASTS);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 17, sizeof(int),    &NUMBER_OF_INVALID_TIMEPOINTS);
	runKernelErrorCalculateStatisticalMapsGLMFTestFirstLevel = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMFTestFirstLevelKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);


	MultiplyVolumes(d_AR1_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR2_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR3_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	//clEnqueueReadBuffer(commandQueue, d_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Smoothed_fMRI_Volumes, 0, NULL, NULL);

	//clEnqueueReadBuffer(commandQueue, d_Whitened_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Motion_Corrected_fMRI_Volumes, 0, NULL, NULL);

	//PutWhitenedModelsIntoVolumes(d_EPI_Mask, d_xtxxt_GLM, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS);
	//PutWhitenedModelsIntoVolumes2(d_EPI_Mask, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, h_X_GLM, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS);

	clReleaseMemObject(d_xtxxt_GLM);
	clReleaseMemObject(d_GLM_Scalars);
	clReleaseMemObject(c_Censored_Timepoints);
	clReleaseMemObject(d_Voxel_Numbers);
}

// This function currently only works for 2 regressors
void BROCCOLI_LIB::CalculateStatisticalMapsGLMBayesianFirstLevel(cl_mem d_Volumes)
{
	cl_mem d_Regressed_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	cl_mem d_Seeds = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int), NULL, NULL);

	NUMBER_OF_TOTAL_GLM_REGRESSORS = 2;

	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Remove linear fit of detrending regressors and motion regressors
	PerformDetrendingAndMotionRegression(d_Regressed_Volumes, d_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	//PerformDetrending(d_Regressed_Volumes, d_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

	float* h_X_GLM_ = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
	float* h_S00 = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float));
	float* h_S01 = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float));
	float* h_S11 = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float));

	float* h_InvOmega0 = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float));

	Eigen::MatrixXd X(EPI_DATA_T,NUMBER_OF_TOTAL_GLM_REGRESSORS);

	for (int i = 0; i < EPI_DATA_T; i++)
	{
		int r = 0;
		X(i,r) = (double)h_X_GLM[i + r * EPI_DATA_T];
		h_X_GLM_[i + 0 * EPI_DATA_T] = h_X_GLM[i + r * EPI_DATA_T];

		r = 1;
		X(i,1) = (double)h_X_GLM[i + r * EPI_DATA_T];
		h_X_GLM_[i + 1 * EPI_DATA_T] = h_X_GLM[i + r * EPI_DATA_T];
	}

	double tau = 100;
	Eigen::MatrixXd Omega0 = tau * tau * (X.transpose() * X).inverse();
	Eigen::MatrixXd InvOmega0 = Omega0.inverse();

	for (int i = 0; i < NUMBER_OF_TOTAL_GLM_REGRESSORS; i++)
	{
		for (int j = 0; j < NUMBER_OF_TOTAL_GLM_REGRESSORS; j++)
		{
			h_InvOmega0[i + j * NUMBER_OF_TOTAL_GLM_REGRESSORS] = (float)InvOmega0(i,j);

			h_S00[i + j*NUMBER_OF_TOTAL_GLM_REGRESSORS] = 0.0f;
			h_S01[i + j*NUMBER_OF_TOTAL_GLM_REGRESSORS] = 0.0f;
			h_S11[i + j*NUMBER_OF_TOTAL_GLM_REGRESSORS] = 0.0f;

			h_S00[i + j*NUMBER_OF_TOTAL_GLM_REGRESSORS] += h_X_GLM_[0 + i * EPI_DATA_T] * h_X_GLM_[0 + j * EPI_DATA_T];
			for (int t = 1; t < EPI_DATA_T; t++)
			{
				h_S00[i + j*NUMBER_OF_TOTAL_GLM_REGRESSORS] += h_X_GLM_[t + i * EPI_DATA_T] * h_X_GLM_[t + j * EPI_DATA_T];
				h_S01[i + j*NUMBER_OF_TOTAL_GLM_REGRESSORS] += h_X_GLM_[t + i * EPI_DATA_T] * h_X_GLM_[(t - 1) + j * EPI_DATA_T];
				h_S11[i + j*NUMBER_OF_TOTAL_GLM_REGRESSORS] += h_X_GLM_[(t - 1) + i * EPI_DATA_T] * h_X_GLM_[(t - 1) + j * EPI_DATA_T];
			}
		}
	}

	/*
	h_OmegaT[0] = 2.0f;
	h_OmegaT[1] = 5.0f;
	h_OmegaT[2] = 5.0f;
	h_OmegaT[3] = 13.0f;
	*/

	cl_mem c_InvOmega0 = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	cl_mem c_S00 = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	cl_mem c_S01 = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	cl_mem c_S11 = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);

	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM_, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_S00, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_S00, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_S01, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_S01, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_S11, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_S11, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_InvOmega0, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_InvOmega0, 0, NULL, NULL);
	clFinish(commandQueue);

	// Generate seeds for random number generation
	int* h_Seeds = (int*)malloc(EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int));
	for (int i = 0; i < EPI_DATA_W * EPI_DATA_H * EPI_DATA_D; i++)
	{
		h_Seeds[i] = rand();
	}
	clEnqueueWriteBuffer(commandQueue, d_Seeds, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int), h_Seeds, 0, NULL, NULL);
	clFinish(commandQueue);
	free(h_Seeds);

	// Calculate PPM(s)
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 1, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 2, sizeof(cl_mem), &d_AR1_Estimates);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 3, sizeof(cl_mem), &d_Regressed_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 4, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 5, sizeof(cl_mem), &d_Seeds);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 6, sizeof(cl_mem), &c_X_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 7, sizeof(cl_mem), &c_InvOmega0);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 8, sizeof(cl_mem), &c_S00);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 9, sizeof(cl_mem), &c_S01);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 10, sizeof(cl_mem),&c_S11);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 11, sizeof(int),   &EPI_DATA_W);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 12, sizeof(int),   &EPI_DATA_H);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 13, sizeof(int),   &EPI_DATA_D);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 14, sizeof(int),   &EPI_DATA_T);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 15, sizeof(int),   &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 16, sizeof(int),   &NUMBER_OF_MCMC_ITERATIONS);
	runKernelErrorCalculateStatisticalMapsGLMBayesian = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMBayesianKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	free(h_X_GLM_);
	free(h_S00);
	free(h_S01);
	free(h_S11);
	free(h_InvOmega0);

	clReleaseMemObject(d_Regressed_Volumes);
	clReleaseMemObject(d_Seeds);
	clReleaseMemObject(c_InvOmega0);
	clReleaseMemObject(c_S00);
	clReleaseMemObject(c_S01);
	clReleaseMemObject(c_S11);
}

void BROCCOLI_LIB::PerformBayesianFirstLevelWrapper()
{
	// Allocate memory for volumes
	d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	cl_mem d_Regressed_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
	cl_mem d_Seeds = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int), NULL, NULL);

	// Allocate memory for results
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask , 0, NULL, NULL);

	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Remove linear fit of detrending regressors
	PerformDetrending(d_Regressed_Volumes, d_fMRI_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

	NUMBER_OF_TOTAL_GLM_REGRESSORS = 2;


	float* h_X_GLM_ = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
	float* h_S00 = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float));
	float* h_S01 = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float));
	float* h_S11 = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float));

	float* h_InvOmega0 = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float));

	Eigen::MatrixXd X(EPI_DATA_T,NUMBER_OF_TOTAL_GLM_REGRESSORS);

	for (int i = 0; i < EPI_DATA_T; i++)
	{
		int r = 0;
		X(i,r) = (double)h_X_GLM_In[i + r * EPI_DATA_T];
		h_X_GLM_[i + 0 * EPI_DATA_T] = h_X_GLM_In[i + r * EPI_DATA_T];

		r = 1;
		X(i,r) = (double)h_X_GLM_In[i + r * EPI_DATA_T];
		h_X_GLM_[i + 1 * EPI_DATA_T] = h_X_GLM_In[i + r * EPI_DATA_T];
	}

	double tau = 100;
	Eigen::MatrixXd Omega0 = tau * tau * (X.transpose() * X).inverse();
	Eigen::MatrixXd InvOmega0 = Omega0.inverse();

	for (int i = 0; i < NUMBER_OF_TOTAL_GLM_REGRESSORS; i++)
	{
		for (int j = 0; j < NUMBER_OF_TOTAL_GLM_REGRESSORS; j++)
		{
			h_InvOmega0[i + j * NUMBER_OF_TOTAL_GLM_REGRESSORS] = (float)InvOmega0(i,j);

			h_S00[i + j*NUMBER_OF_TOTAL_GLM_REGRESSORS] = 0.0f;
			h_S01[i + j*NUMBER_OF_TOTAL_GLM_REGRESSORS] = 0.0f;
			h_S11[i + j*NUMBER_OF_TOTAL_GLM_REGRESSORS] = 0.0f;

			h_S00[i + j*NUMBER_OF_TOTAL_GLM_REGRESSORS] += h_X_GLM_[0 + i * EPI_DATA_T] * h_X_GLM_[0 + j * EPI_DATA_T];
			for (int t = 1; t < EPI_DATA_T; t++)
			{
				h_S00[i + j*NUMBER_OF_TOTAL_GLM_REGRESSORS] += h_X_GLM_[t + i * EPI_DATA_T] * h_X_GLM_[t + j * EPI_DATA_T];
				h_S01[i + j*NUMBER_OF_TOTAL_GLM_REGRESSORS] += h_X_GLM_[t + i * EPI_DATA_T] * h_X_GLM_[(t - 1) + j * EPI_DATA_T];
				h_S11[i + j*NUMBER_OF_TOTAL_GLM_REGRESSORS] += h_X_GLM_[(t - 1) + i * EPI_DATA_T] * h_X_GLM_[(t - 1) + j * EPI_DATA_T];
			}
		}
	}

	/*
	h_OmegaT[0] = 2.0f;
	h_OmegaT[1] = 5.0f;
	h_OmegaT[2] = 5.0f;
	h_OmegaT[3] = 13.0f;
	*/

	cl_mem c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	cl_mem c_InvOmega0 = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	cl_mem c_S00 = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	cl_mem c_S01 = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	cl_mem c_S11 = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);

	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM_, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_S00, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_S00, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_S01, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_S01, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_S11, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_S11, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_InvOmega0, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_InvOmega0, 0, NULL, NULL);
	clFinish(commandQueue);

	// Generate seeds for random number generation
	int* h_Seeds = (int*)malloc(EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int));
	for (int i = 0; i < EPI_DATA_W * EPI_DATA_H * EPI_DATA_D; i++)
	{
		h_Seeds[i] = rand();
	}
	clEnqueueWriteBuffer(commandQueue, d_Seeds, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int), h_Seeds, 0, NULL, NULL);
	free(h_Seeds);

	int NUMBER_OF_ITERATIONS = 1000;

	// Calculate PPM(s)
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 1, sizeof(cl_mem), &d_Regressed_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 3, sizeof(cl_mem), &d_Seeds);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 4, sizeof(cl_mem), &c_X_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 5, sizeof(cl_mem), &c_InvOmega0);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 6, sizeof(cl_mem), &c_S00);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 7, sizeof(cl_mem), &c_S01);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 8, sizeof(cl_mem), &c_S11);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 9, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 10, sizeof(int),   &EPI_DATA_H);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 11, sizeof(int),   &EPI_DATA_D);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 12, sizeof(int),   &EPI_DATA_T);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 13, sizeof(int),   &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 14, sizeof(int),   &NUMBER_OF_ITERATIONS);
	runKernelErrorCalculateStatisticalMapsGLMBayesian = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMBayesianKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Copy results to  host
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_EPI, 0, NULL, NULL);

	// Release memory

	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_EPI_Mask);
	clReleaseMemObject(d_Regressed_Volumes);
	clReleaseMemObject(d_Seeds);
	clReleaseMemObject(d_Statistical_Maps);

	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_InvOmega0);
	clReleaseMemObject(c_S00);
	clReleaseMemObject(c_S01);
	clReleaseMemObject(c_S11);


	free(h_X_GLM_);
	free(h_S00);
	free(h_S01);
	free(h_S11);
	free(h_InvOmega0);
}

// Puts whitened regressors for brain voxels only into real volumes, pseudo inverses
void BROCCOLI_LIB::PutWhitenedModelsIntoVolumes(cl_mem d_Mask, cl_mem d_xtxxt_GLM, int DATA_W, int DATA_H, int DATA_D, int DATA_T, int NUMBER_OF_REGRESSORS)
{
	float* h_Mask = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));
	float* h_xtxxt_GLM_ = (float*)malloc(NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float));

	clEnqueueReadBuffer(commandQueue, d_Mask, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Mask, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float), h_xtxxt_GLM_, 0, NULL, NULL);

	// Loop over voxels
	int voxel_number = 0;
	for (int z = 0; z < DATA_D; z++)
	{
		for (int y = 0; y < DATA_H; y++)
		{
			for (int x = 0; x < DATA_W; x++)
			{
				for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
				{
					for (int t = 0; t < DATA_T; t++)
					{
						h_Whitened_Models[x + y * DATA_W + z * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D + r * DATA_W * DATA_H * DATA_D * DATA_T] = 0.0f;
					}
				}

				if ( h_Mask[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
				{
					for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
					{
						for (int t = 0; t < DATA_T; t++)
						{
							h_Whitened_Models[x + y * DATA_W + z * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D + r * DATA_W * DATA_H * DATA_D * DATA_T] = h_xtxxt_GLM_[voxel_number * NUMBER_OF_REGRESSORS * DATA_T + r * DATA_T + t];
						}
					}
					voxel_number++;
				}
			}
		}
	}

	free(h_Mask);
	free(h_xtxxt_GLM_);
}

// Puts whitened regressors for brain voxels only into real volumes
void BROCCOLI_LIB::PutWhitenedModelsIntoVolumes2(cl_mem d_Mask,
		                                         cl_mem d_AR1_Estimates,
		                                         cl_mem d_AR2_Estimates,
		                                         cl_mem d_AR3_Estimates,
		                                         cl_mem d_AR4_Estimates,
		                                         float* Regressors,
		                                         int DATA_W,
		                                         int DATA_H,
		                                         int DATA_D,
		                                         int DATA_T,
		                                         int NUMBER_OF_REGRESSORS)
{
	float* h_Mask = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));
	float* h_xtxxt_GLM_ = (float*)malloc(NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float));

	clEnqueueReadBuffer(commandQueue, d_Mask, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Mask, 0, NULL, NULL);

	// Copy AR parameters to host
	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);

	// Loop over voxels
	int voxel_number = 0;
	for (int z = 0; z < DATA_D; z++)
	{
		for (int y = 0; y < DATA_H; y++)
		{
			for (int x = 0; x < DATA_W; x++)
			{
				if ( h_Mask[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
				{
					Eigen::MatrixXd X(DATA_T,NUMBER_OF_REGRESSORS);

					// Get AR parameters for current voxel
					float AR1 = h_AR1_Estimates_EPI[x + y * DATA_W + z * DATA_W * DATA_H];
					float AR2 = h_AR2_Estimates_EPI[x + y * DATA_W + z * DATA_W * DATA_H];
					float AR3 = h_AR3_Estimates_EPI[x + y * DATA_W + z * DATA_W * DATA_H];
					float AR4 = h_AR4_Estimates_EPI[x + y * DATA_W + z * DATA_W * DATA_H];

					float old_value_1, old_value_2, old_value_3, old_value_4, old_value_5;

					// Whiten original regressors
					for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
					{
					    old_value_1 = Regressors[0 + r * DATA_T];
						X(0,r) = 0;
						old_value_2 = Regressors[1 + r * DATA_T];
						X(1,r) = 0;
						old_value_3 = Regressors[2 + r * DATA_T];
						X(2,r) = 0;
						old_value_4 = Regressors[3 + r * DATA_T];
						X(3,r) = 0;

						for (int t = 4; t < DATA_T; t++)
						{
							old_value_5 = Regressors[t + r * DATA_T];
							X(t,r) = old_value_5 - AR1 * old_value_4 - AR2 * old_value_3 - AR3 * old_value_2 - AR4 * old_value_1;

							// Save old values
							old_value_1 = old_value_2;
							old_value_2 = old_value_3;
							old_value_3 = old_value_4;
							old_value_4 = old_value_5;
						}
					}

					for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
					{
						for (int t = 0; t < DATA_T; t++)
						{
							h_Whitened_Models[x + y * DATA_W + z * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D + r * DATA_W * DATA_H * DATA_D * DATA_T] = X(t,r);
						}
					}

					voxel_number++;
				}
			}
		}
	}

	free(h_Mask);
	free(h_xtxxt_GLM_);
}









// Calculates a statistical map for second level analysis
void BROCCOLI_LIB::CalculateStatisticalMapsGLMTTestSecondLevel(cl_mem d_Volumes, cl_mem d_Mask)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	int NUMBER_OF_INVALID_VOLUMES = 0;
	c_Censored_Volumes = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	SetMemory(c_Censored_Volumes, 1.0f, NUMBER_OF_SUBJECTS);

	// Calculate beta weights
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_GLM);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 4, sizeof(cl_mem), &c_Censored_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 5, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 6, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 7, sizeof(int),    &MNI_DATA_D);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 8, sizeof(int),    &NUMBER_OF_SUBJECTS);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 9, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	runKernelErrorCalculateBetaWeightsGLM = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Calculate t-values and residuals
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 0, sizeof(cl_mem),  &d_Statistical_Maps);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 1, sizeof(cl_mem),  &d_Residuals);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 2, sizeof(cl_mem),  &d_Residual_Variances);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 3, sizeof(cl_mem),  &d_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 4, sizeof(cl_mem),  &d_Beta_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 5, sizeof(cl_mem),  &d_Mask);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 6, sizeof(cl_mem),  &c_X_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 7, sizeof(cl_mem),  &c_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 8, sizeof(cl_mem),  &c_ctxtxc_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 9, sizeof(cl_mem),  &c_Censored_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 10, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 11, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 12, sizeof(int),    &MNI_DATA_D);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 13, sizeof(int),    &NUMBER_OF_SUBJECTS);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 14, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 15, sizeof(int),    &NUMBER_OF_CONTRASTS);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestKernel, 16, sizeof(int),    &NUMBER_OF_INVALID_VOLUMES);
	runKernelErrorCalculateStatisticalMapsGLMTTest = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMTTestKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);
	
	clReleaseMemObject(c_Censored_Volumes);
}



void BROCCOLI_LIB::CalculateStatisticalMapsGLMFTestSecondLevel(cl_mem d_Volumes, cl_mem d_Mask)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	int NUMBER_OF_INVALID_VOLUMES = 0;
	c_Censored_Volumes = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	SetMemory(c_Censored_Volumes, 1.0f, NUMBER_OF_SUBJECTS);

	// Calculate beta weights
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_GLM);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 4, sizeof(cl_mem), &c_Censored_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 5, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 6, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 7, sizeof(int),    &MNI_DATA_D);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 8, sizeof(int),    &NUMBER_OF_SUBJECTS);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 9, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	runKernelErrorCalculateBetaWeightsGLM = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Calculate F-values and residuals
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 1, sizeof(cl_mem), &d_Residuals);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 2, sizeof(cl_mem), &d_Residual_Variances);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 3, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 4, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 5, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 6, sizeof(cl_mem), &c_X_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 7, sizeof(cl_mem), &c_Contrasts);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 8, sizeof(cl_mem), &c_ctxtxc_GLM);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 9, sizeof(cl_mem), &c_Censored_Volumes);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 10, sizeof(int),   &MNI_DATA_W);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 11, sizeof(int),   &MNI_DATA_H);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 12, sizeof(int),   &MNI_DATA_D);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 13, sizeof(int),   &NUMBER_OF_SUBJECTS);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 14, sizeof(int),   &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 15, sizeof(int),   &NUMBER_OF_CONTRASTS);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestKernel, 16, sizeof(int),   &NUMBER_OF_INVALID_VOLUMES);
	runKernelErrorCalculateStatisticalMapsGLMFTest = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMFTestKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);
	
	clReleaseMemObject(c_Censored_Volumes);
}


void BROCCOLI_LIB::CleanupPermutationTestFirstLevel()
{
	// Free temporary memory
	clReleaseMemObject(c_Smoothing_Filter_X);
	clReleaseMemObject(c_Smoothing_Filter_Y);
	clReleaseMemObject(c_Smoothing_Filter_Z);

	clReleaseMemObject(d_Rows_Temp);
	clReleaseMemObject(d_Columns_Temp);

	clReleaseMemObject(d_Largest_Cluster);
	clReleaseMemObject(d_Updated);
}

void BROCCOLI_LIB::SetupPermutationTestFirstLevel()
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	SetGlobalAndLocalWorkSizesSeparableConvolution(EPI_DATA_W,EPI_DATA_H,EPI_DATA_D);

	// Smooth mask, for normalized convolution
	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, EPI_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	PerformSmoothing(d_Smoothed_EPI_Mask, d_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

	// Allocate memory for smoothing filters
	c_Smoothing_Filter_X = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Y = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Z = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);

	// Copy smoothing filters to constant memory
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_X, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_X , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Y, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Y , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Z, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Z , 0, NULL, NULL);

	// Allocate temporary memory for smoothing
	d_Rows_Temp = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Columns_Temp = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	// Set arguments for the smoothing kernels
	clSetKernelArg(SeparableConvolutionRowsKernel, 0, sizeof(cl_mem), &d_Rows_Temp);
	clSetKernelArg(SeparableConvolutionRowsKernel, 1, sizeof(cl_mem), &d_Permuted_fMRI_Volumes);
	clSetKernelArg(SeparableConvolutionRowsKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(SeparableConvolutionRowsKernel, 3, sizeof(cl_mem), &c_Smoothing_Filter_Y);
	clSetKernelArg(SeparableConvolutionRowsKernel, 5, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(SeparableConvolutionRowsKernel, 6, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(SeparableConvolutionRowsKernel, 7, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(SeparableConvolutionRowsKernel, 8, sizeof(int), &EPI_DATA_T);

	clSetKernelArg(SeparableConvolutionColumnsKernel, 0, sizeof(cl_mem), &d_Columns_Temp);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 1, sizeof(cl_mem), &d_Rows_Temp);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_X);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 4, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 5, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 6, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 7, sizeof(int), &EPI_DATA_T);

	clSetKernelArg(SeparableConvolutionRodsKernel, 0, sizeof(cl_mem), &d_Permuted_fMRI_Volumes);
	clSetKernelArg(SeparableConvolutionRodsKernel, 1, sizeof(cl_mem), &d_Columns_Temp);
	clSetKernelArg(SeparableConvolutionRodsKernel, 2, sizeof(cl_mem), &d_Smoothed_EPI_Mask);
	clSetKernelArg(SeparableConvolutionRodsKernel, 3, sizeof(cl_mem), &c_Smoothing_Filter_Z);
	clSetKernelArg(SeparableConvolutionRodsKernel, 5, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(SeparableConvolutionRodsKernel, 6, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(SeparableConvolutionRodsKernel, 7, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(SeparableConvolutionRodsKernel, 8, sizeof(int), &EPI_DATA_T);

	if (STATISTICAL_TEST == TTEST)
	{
		// Reset all statistical maps
		SetMemory(d_Statistical_Maps, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS);

		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel, 1, sizeof(cl_mem), &d_Permuted_fMRI_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel, 3, sizeof(cl_mem), &c_X_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel, 4, sizeof(cl_mem), &c_xtxxt_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel, 5, sizeof(cl_mem), &c_Contrasts);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel, 6, sizeof(cl_mem), &c_ctxtxc_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel, 7, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel, 8, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel, 9, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel, 10, sizeof(int),   &EPI_DATA_T);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel, 11, sizeof(int),   &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel, 12, sizeof(int),   &NUMBER_OF_CONTRASTS);
	}
	else if (STATISTICAL_TEST == FTEST)
	{
		SetMemory(d_Statistical_Maps, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);

		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel, 1, sizeof(cl_mem), &d_Permuted_fMRI_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel, 3, sizeof(cl_mem), &c_X_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel, 4, sizeof(cl_mem), &c_xtxxt_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel, 5, sizeof(cl_mem), &c_Contrasts);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel, 6, sizeof(cl_mem), &c_ctxtxc_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel, 7, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel, 8, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel, 9, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel, 10, sizeof(int),   &EPI_DATA_T);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel, 11, sizeof(int),   &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel, 12, sizeof(int),   &NUMBER_OF_CONTRASTS);
	}

	d_Largest_Cluster = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	d_Updated = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);

	SetGlobalAndLocalWorkSizesClusterize(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	clSetKernelArg(SetStartClusterIndicesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(SetStartClusterIndicesKernel, 1, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(SetStartClusterIndicesKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(SetStartClusterIndicesKernel, 3, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(SetStartClusterIndicesKernel, 4, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(SetStartClusterIndicesKernel, 5, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(SetStartClusterIndicesKernel, 6, sizeof(int),    &EPI_DATA_D);

	clSetKernelArg(ClusterizeScanKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeScanKernel, 1, sizeof(cl_mem), &d_Updated);
	clSetKernelArg(ClusterizeScanKernel, 2, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(ClusterizeScanKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(ClusterizeScanKernel, 4, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(ClusterizeScanKernel, 5, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(ClusterizeScanKernel, 6, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(ClusterizeScanKernel, 7, sizeof(int),    &EPI_DATA_D);

	clSetKernelArg(ClusterizeRelabelKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeRelabelKernel, 1, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(ClusterizeRelabelKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(ClusterizeRelabelKernel, 3, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(ClusterizeRelabelKernel, 4, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(ClusterizeRelabelKernel, 5, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(ClusterizeRelabelKernel, 6, sizeof(int),    &EPI_DATA_D);

	clSetKernelArg(CalculateClusterSizesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(CalculateClusterSizesKernel, 1, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateClusterSizesKernel, 2, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateClusterSizesKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateClusterSizesKernel, 4, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(CalculateClusterSizesKernel, 5, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(CalculateClusterSizesKernel, 6, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(CalculateClusterSizesKernel, 7, sizeof(int),    &EPI_DATA_D);

	clSetKernelArg(CalculateClusterMassesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(CalculateClusterMassesKernel, 1, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateClusterMassesKernel, 2, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateClusterMassesKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateClusterMassesKernel, 4, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(CalculateClusterMassesKernel, 5, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(CalculateClusterMassesKernel, 6, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(CalculateClusterMassesKernel, 7, sizeof(int),    &EPI_DATA_D);

	clSetKernelArg(CalculateLargestClusterKernel, 0, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateLargestClusterKernel, 1, sizeof(cl_mem), &d_Largest_Cluster);
	clSetKernelArg(CalculateLargestClusterKernel, 2, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(CalculateLargestClusterKernel, 3, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(CalculateLargestClusterKernel, 4, sizeof(int),    &EPI_DATA_D);

	clSetKernelArg(CalculateTFCEValuesKernel, 0, sizeof(cl_mem), &d_TFCE_Values);
	clSetKernelArg(CalculateTFCEValuesKernel, 1, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateTFCEValuesKernel, 2, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(CalculateTFCEValuesKernel, 3, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(CalculateTFCEValuesKernel, 4, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateTFCEValuesKernel, 5, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(CalculateTFCEValuesKernel, 6, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(CalculateTFCEValuesKernel, 7, sizeof(int),    &EPI_DATA_D);
}

void BROCCOLI_LIB::SetupPermutationTestSecondLevel(cl_mem d_Volumes, cl_mem d_Mask)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	if (STATISTICAL_TEST == GROUP_MEAN)
	{
		// Reset all statistical maps
		SetMemory(d_Statistical_Maps, 0.0f, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D);

		clSetKernelArg(CalculateStatisticalMapsMeanSecondLevelPermutationKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
		clSetKernelArg(CalculateStatisticalMapsMeanSecondLevelPermutationKernel, 1, sizeof(cl_mem), &d_Volumes);
		clSetKernelArg(CalculateStatisticalMapsMeanSecondLevelPermutationKernel, 2, sizeof(cl_mem), &d_Mask);
		clSetKernelArg(CalculateStatisticalMapsMeanSecondLevelPermutationKernel, 3, sizeof(cl_mem), &c_X_GLM);
		clSetKernelArg(CalculateStatisticalMapsMeanSecondLevelPermutationKernel, 4, sizeof(cl_mem), &c_xtxxt_GLM);
		clSetKernelArg(CalculateStatisticalMapsMeanSecondLevelPermutationKernel, 5, sizeof(cl_mem), &c_Contrasts);
		clSetKernelArg(CalculateStatisticalMapsMeanSecondLevelPermutationKernel, 6, sizeof(cl_mem), &c_ctxtxc_GLM);
		clSetKernelArg(CalculateStatisticalMapsMeanSecondLevelPermutationKernel, 7, sizeof(cl_mem), &c_Permutation_Vector);
		clSetKernelArg(CalculateStatisticalMapsMeanSecondLevelPermutationKernel, 8, sizeof(cl_mem), &c_Sign_Vector);
		clSetKernelArg(CalculateStatisticalMapsMeanSecondLevelPermutationKernel, 9, sizeof(int),    &MNI_DATA_W);
		clSetKernelArg(CalculateStatisticalMapsMeanSecondLevelPermutationKernel, 10, sizeof(int),   &MNI_DATA_H);
		clSetKernelArg(CalculateStatisticalMapsMeanSecondLevelPermutationKernel, 11, sizeof(int),   &MNI_DATA_D);
		clSetKernelArg(CalculateStatisticalMapsMeanSecondLevelPermutationKernel, 12, sizeof(int),   &NUMBER_OF_SUBJECTS);
	}
	else if (STATISTICAL_TEST == TTEST)
	{
		// Reset all statistical maps
		SetMemory(d_Statistical_Maps, 0.0f, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS);

		clSetKernelArg(CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel, 1, sizeof(cl_mem), &d_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel, 2, sizeof(cl_mem), &d_Mask);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel, 3, sizeof(cl_mem), &c_X_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel, 4, sizeof(cl_mem), &c_xtxxt_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel, 5, sizeof(cl_mem), &c_Contrasts);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel, 6, sizeof(cl_mem), &c_ctxtxc_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel, 7, sizeof(cl_mem), &c_Permutation_Vector);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel, 8, sizeof(int),    &MNI_DATA_W);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel, 9, sizeof(int),    &MNI_DATA_H);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel, 10, sizeof(int),   &MNI_DATA_D);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel, 11, sizeof(int),   &NUMBER_OF_SUBJECTS);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel, 12, sizeof(int),   &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel, 13, sizeof(int),   &NUMBER_OF_CONTRASTS);
	}
	else if (STATISTICAL_TEST == FTEST)
	{
		// Reset all statistical maps
		SetMemory(d_Statistical_Maps, 0.0f, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D);

		clSetKernelArg(CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel, 0, sizeof(cl_mem), &d_Statistical_Maps);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel, 1, sizeof(cl_mem), &d_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel, 2, sizeof(cl_mem), &d_Mask);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel, 3, sizeof(cl_mem), &c_X_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel, 4, sizeof(cl_mem), &c_xtxxt_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel, 5, sizeof(cl_mem), &c_Contrasts);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel, 6, sizeof(cl_mem), &c_ctxtxc_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel, 7, sizeof(cl_mem), &c_Permutation_Vector);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel, 8, sizeof(int),    &MNI_DATA_W);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel, 9, sizeof(int),    &MNI_DATA_H);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel, 10, sizeof(int),   &MNI_DATA_D);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel, 11, sizeof(int),   &NUMBER_OF_SUBJECTS);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel, 12, sizeof(int),   &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel, 13, sizeof(int),   &NUMBER_OF_CONTRASTS);
	}

	d_Largest_Cluster = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, NULL);
	d_Updated = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);

	SetGlobalAndLocalWorkSizesClusterize(MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	clSetKernelArg(SetStartClusterIndicesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(SetStartClusterIndicesKernel, 1, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(SetStartClusterIndicesKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(SetStartClusterIndicesKernel, 3, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(SetStartClusterIndicesKernel, 4, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(SetStartClusterIndicesKernel, 5, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(SetStartClusterIndicesKernel, 6, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(ClusterizeScanKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeScanKernel, 1, sizeof(cl_mem), &d_Updated);
	clSetKernelArg(ClusterizeScanKernel, 2, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(ClusterizeScanKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(ClusterizeScanKernel, 4, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(ClusterizeScanKernel, 5, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(ClusterizeScanKernel, 6, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(ClusterizeScanKernel, 7, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(ClusterizeRelabelKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeRelabelKernel, 1, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(ClusterizeRelabelKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(ClusterizeRelabelKernel, 3, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(ClusterizeRelabelKernel, 4, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(ClusterizeRelabelKernel, 5, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(ClusterizeRelabelKernel, 6, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(CalculateClusterSizesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(CalculateClusterSizesKernel, 1, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateClusterSizesKernel, 2, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateClusterSizesKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateClusterSizesKernel, 4, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(CalculateClusterSizesKernel, 5, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateClusterSizesKernel, 6, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateClusterSizesKernel, 7, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(CalculateClusterMassesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(CalculateClusterMassesKernel, 1, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateClusterMassesKernel, 2, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateClusterMassesKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateClusterMassesKernel, 4, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(CalculateClusterMassesKernel, 5, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateClusterMassesKernel, 6, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateClusterMassesKernel, 7, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(CalculateLargestClusterKernel, 0, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateLargestClusterKernel, 1, sizeof(cl_mem), &d_Largest_Cluster);
	clSetKernelArg(CalculateLargestClusterKernel, 2, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateLargestClusterKernel, 3, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateLargestClusterKernel, 4, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(CalculateTFCEValuesKernel, 0, sizeof(cl_mem), &d_TFCE_Values);
	clSetKernelArg(CalculateTFCEValuesKernel, 1, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateTFCEValuesKernel, 2, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(CalculateTFCEValuesKernel, 3, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(CalculateTFCEValuesKernel, 4, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateTFCEValuesKernel, 5, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateTFCEValuesKernel, 6, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateTFCEValuesKernel, 7, sizeof(int),    &MNI_DATA_D);
}

void BROCCOLI_LIB::CleanupPermutationTestSecondLevel()
{
	clReleaseMemObject(d_Largest_Cluster);
	clReleaseMemObject(d_Updated);
}

void BROCCOLI_LIB::CalculateStatisticalMapsFirstLevelPermutation()
{
	if (STATISTICAL_TEST == TTEST)
	{
		CalculateStatisticalMapsGLMTTestFirstLevelPermutation();
	}
	else if (STATISTICAL_TEST == FTEST)
	{
		CalculateStatisticalMapsGLMFTestFirstLevelPermutation();
	}
}

// Calculates a statistical t-map for second level analysis, all kernel parameters have been set in SetupPermutationTestSecondLevel
void BROCCOLI_LIB::CalculateStatisticalMapsGLMTTestFirstLevelPermutation()
{
	runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelPermutation = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);
}

// Calculates a statistical F-map for second level analysis, all kernel parameters have been set in SetupPermutationTestSecondLevel
void BROCCOLI_LIB::CalculateStatisticalMapsGLMFTestFirstLevelPermutation()
{
	runKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelPermutation = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);
}



// A small wrapper function that simply calls functions for different tests
void BROCCOLI_LIB::CalculateStatisticalMapsSecondLevelPermutation(int p)
{
   	if (STATISTICAL_TEST == GROUP_MEAN)
	{
   		// Copy a new sign vector to constant memory
	   	clEnqueueWriteBuffer(commandQueue, c_Sign_Vector, CL_TRUE, 0, NUMBER_OF_SUBJECTS * sizeof(float), &h_Sign_Matrix[p * NUMBER_OF_SUBJECTS], 0, NULL, NULL);
		CalculateStatisticalMapsMeanSecondLevelPermutation();
	}
   	else if (STATISTICAL_TEST == TTEST)
	{
   		// Copy a new permutation vector to constant memory
	   	clEnqueueWriteBuffer(commandQueue, c_Permutation_Vector, CL_TRUE, 0, NUMBER_OF_SUBJECTS * sizeof(unsigned short int), &h_Permutation_Matrix[p * NUMBER_OF_SUBJECTS], 0, NULL, NULL);
		CalculateStatisticalMapsGLMTTestSecondLevelPermutation();
	}
	else if (STATISTICAL_TEST == FTEST)
	{
		// Copy a new permutation vector to constant memory
		clEnqueueWriteBuffer(commandQueue, c_Permutation_Vector, CL_TRUE, 0, NUMBER_OF_SUBJECTS * sizeof(unsigned short int), &h_Permutation_Matrix[p * NUMBER_OF_SUBJECTS], 0, NULL, NULL);
		CalculateStatisticalMapsGLMFTestSecondLevelPermutation();
	}
}



// Calculates a mean map for second level analysis, using a sign vector to randomly flip the sign of each volume, all kernel parameters have been set in SetupPermutationTestSecondLevel
void BROCCOLI_LIB::CalculateStatisticalMapsMeanSecondLevelPermutation()
{
	runKernelErrorCalculateStatisticalMapsMeanSecondLevelPermutation = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsMeanSecondLevelPermutationKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);
}

// Calculates a statistical t-map for second level analysis, all kernel parameters have been set in SetupPermutationTestSecondLevel
void BROCCOLI_LIB::CalculateStatisticalMapsGLMTTestSecondLevelPermutation()
{
	runKernelErrorCalculateStatisticalMapsGLMTTestSecondLevelPermutation = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);
}

// Calculates a statistical F-map for second level analysis, all kernel parameters have been set in SetupPermutationTestSecondLevel
void BROCCOLI_LIB::CalculateStatisticalMapsGLMFTestSecondLevelPermutation()
{
	runKernelErrorCalculateStatisticalMapsGLMFTestSecondLevelPermutation = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
	clFinish(commandQueue);
}




// Performs whitening prior to a first level permutation test, saves AR(4) estimates
void BROCCOLI_LIB::PerformWhiteningPriorPermutations(cl_mem d_Whitened_Volumes, cl_mem d_Volumes)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Smooth mask, for normalized convolution
	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, AR_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	PerformSmoothing(d_Smoothed_EPI_Mask, d_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

	NUMBER_OF_INVALID_TIMEPOINTS = 0;

	// Allocate temporary memory
	cl_mem d_Total_AR1_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Total_AR2_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Total_AR3_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Total_AR4_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	// Reset total parameters
	SetMemory(d_Total_AR1_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_Total_AR2_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_Total_AR3_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);
	SetMemory(d_Total_AR4_Estimates, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);

	// Set whitened volumes to original volumes
	clEnqueueCopyBuffer(commandQueue, d_Volumes, d_Whitened_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), 0, NULL, NULL);

	for (int it = 0; it < 3; it++)
	{
		// Estimate auto correlation from whitened volumes
		clSetKernelArg(EstimateAR4ModelsKernel, 0, sizeof(cl_mem), &d_AR1_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 1, sizeof(cl_mem), &d_AR2_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 2, sizeof(cl_mem), &d_AR3_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 3, sizeof(cl_mem), &d_AR4_Estimates);
		clSetKernelArg(EstimateAR4ModelsKernel, 4, sizeof(cl_mem), &d_Whitened_Volumes);
		clSetKernelArg(EstimateAR4ModelsKernel, 5, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(EstimateAR4ModelsKernel, 6, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(EstimateAR4ModelsKernel, 7, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(EstimateAR4ModelsKernel, 8, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(EstimateAR4ModelsKernel, 9, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(EstimateAR4ModelsKernel, 10, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorEstimateAR4Models = clEnqueueNDRangeKernel(commandQueue, EstimateAR4ModelsKernel, 3, NULL, globalWorkSizeEstimateAR4Models, localWorkSizeEstimateAR4Models, 0, NULL, NULL);

		// Smooth AR estimates
		PerformSmoothingNormalized(d_AR1_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR2_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR3_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		PerformSmoothingNormalized(d_AR4_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

		// Add current AR estimates to total AR estimates
		AddVolumes(d_Total_AR1_Estimates, d_AR1_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
		AddVolumes(d_Total_AR2_Estimates, d_AR2_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
		AddVolumes(d_Total_AR3_Estimates, d_AR3_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
		AddVolumes(d_Total_AR4_Estimates, d_AR4_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

		// Remove auto correlation from data, using total AR estimates
		clSetKernelArg(ApplyWhiteningAR4Kernel, 0, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 1, sizeof(cl_mem), &d_Volumes);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 2, sizeof(cl_mem), &d_Total_AR1_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 3, sizeof(cl_mem), &d_Total_AR2_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 4, sizeof(cl_mem), &d_Total_AR3_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 5, sizeof(cl_mem), &d_Total_AR4_Estimates);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 6, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 7, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 8, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 9, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(ApplyWhiteningAR4Kernel, 10, sizeof(int),   &EPI_DATA_T);
		runKernelErrorApplyWhiteningAR4 = clEnqueueNDRangeKernel(commandQueue, ApplyWhiteningAR4Kernel, 3, NULL, globalWorkSizeApplyWhiteningAR4, localWorkSizeApplyWhiteningAR4, 0, NULL, NULL);

		NUMBER_OF_INVALID_TIMEPOINTS = 4;
	}

	// Copy back total AR estimates to AR estimates, since they will be used for inverse whitening to generate new fMRI data
	clEnqueueCopyBuffer(commandQueue, d_Total_AR1_Estimates, d_AR1_Estimates, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);
	clEnqueueCopyBuffer(commandQueue, d_Total_AR2_Estimates, d_AR2_Estimates, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);
	clEnqueueCopyBuffer(commandQueue, d_Total_AR3_Estimates, d_AR3_Estimates, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);
	clEnqueueCopyBuffer(commandQueue, d_Total_AR4_Estimates, d_AR4_Estimates, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);

	MultiplyVolumes(d_AR1_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR2_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR3_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Cleanup
	clReleaseMemObject(d_Total_AR1_Estimates);
	clReleaseMemObject(d_Total_AR2_Estimates);
	clReleaseMemObject(d_Total_AR3_Estimates);
	clReleaseMemObject(d_Total_AR4_Estimates);
}

//  Applies a permutation test for first level analysis
void BROCCOLI_LIB::ApplyPermutationTestFirstLevel(cl_mem d_fMRI_Volumes)
{
	if (STATISTICAL_TEST == TTEST)
	{
		NUMBER_OF_STATISTICAL_MAPS = NUMBER_OF_CONTRASTS;
	}
	else if (STATISTICAL_TEST == FTEST)
	{
		NUMBER_OF_STATISTICAL_MAPS = 1;
	}

	// Make sure all starting values are 0, for example necessary for the smoothing
	SetMemory(d_Permuted_fMRI_Volumes, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T);
	SetMemory(d_Whitened_fMRI_Volumes, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T);
	SetMemory(d_Detrended_fMRI_Volumes, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T);

	// Generate a random permutation matrix
	GeneratePermutationMatrixFirstLevel();

	// Remove mean and linear, quadratic and cubic trends
	PerformDetrending(d_Detrended_fMRI_Volumes, d_fMRI_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

	// Make the timeseries white prior to the random permutations
	PerformWhiteningPriorPermutations(d_Whitened_fMRI_Volumes, d_Detrended_fMRI_Volumes);

	// Setup parameters and memory prior to permutations, to save time in each permutation
	SetupPermutationTestFirstLevel();

	for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
	{
		if ((WRAPPER == BASH) && PRINT)
		{
			printf("Starting permutation %i \n",p+1);
		}

		// Generate new fMRI volumes, through inverse whitening and permutation
	    GeneratePermutedVolumesFirstLevel(d_Permuted_fMRI_Volumes, d_Whitened_fMRI_Volumes, p);

		// Smooth new fMRI volumes (smoothing needs to be done in each permutation, as it otherwise alters the AR parameters)
		//PerformSmoothingNormalized(d_Permuted_fMRI_Volumes, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
		//PerformSmoothingNormalizedPermutation();

		// Calculate statistical maps, for all contrasts
		CalculateStatisticalMapsFirstLevelPermutation();

		// Loop over contrasts
		//for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
		//{
			// Voxel distribution
			if (INFERENCE_MODE == VOXEL)
			{
				// Get max test value
				//h_Permutation_Distribution[p + c * NUMBER_OF_PERMUTATIONS] = CalculateMaxAtomic(d_Statistical_Maps, c, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
				h_Permutation_Distribution[p] = CalculateMaxAtomic(d_Statistical_Maps, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
			}
			// Cluster distribution, extent or mass
			else if ( (INFERENCE_MODE == CLUSTER_EXTENT) || (INFERENCE_MODE == CLUSTER_MASS) )
			{
				ClusterizeOpenCLPermutation(MAX_CLUSTER, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
				h_Permutation_Distribution[p] = MAX_CLUSTER;
				//h_Permutation_Distribution[p + c * NUMBER_OF_PERMUTATIONS] = MAX_CLUSTER;
			}
			// Threshold free cluster enhancement
			else if (INFERENCE_MODE == TFCE)
			{
				//maxActivation = CalculateMaxAtomic(d_Statistical_Maps, c, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
				float delta = 0.2846;
				//ClusterizeOpenCLTFCEPermutation(MAX_VALUE, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, maxActivation, delta);
				//h_Permutation_Distribution[p + c * NUMBER_OF_PERMUTATIONS] = MAX_VALUE;
			}
		//}
	}

	// Loop over contrasts
	//for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	//{
		// Sort the maximum test values
		//std::vector<float> max_values (h_Permutation_Distribution + c * NUMBER_OF_PERMUTATIONS, h_Permutation_Distribution + (c + 1) * NUMBER_OF_PERMUTATIONS);
		std::vector<float> max_values (h_Permutation_Distribution, h_Permutation_Distribution + NUMBER_OF_PERMUTATIONS);
		std::sort (max_values.begin(), max_values.begin() + NUMBER_OF_PERMUTATIONS);

		// Find the threshold for the specified significance level
		SIGNIFICANCE_THRESHOLD = max_values[round((1.0f - SIGNIFICANCE_LEVEL) * (float)NUMBER_OF_PERMUTATIONS)];

		if (WRAPPER == BASH)
		{
			//printf("Permutation threshold for contrast %i for a significance level of %f is %f \n",c, SIGNIFICANCE_LEVEL, SIGNIFICANCE_THRESHOLD);
			printf("Permutation threshold for a significance level of %f is %f \n",SIGNIFICANCE_LEVEL, SIGNIFICANCE_THRESHOLD);
		}
	//}

	CleanupPermutationTestFirstLevel();
}

//  Applies a permutation test for second level analysis
void BROCCOLI_LIB::ApplyPermutationTestSecondLevel()
{
	if (STATISTICAL_TEST == GROUP_MEAN)
	{
		NUMBER_OF_STATISTICAL_MAPS = 1;
	}
	else if (STATISTICAL_TEST == TTEST)
	{
		NUMBER_OF_STATISTICAL_MAPS = NUMBER_OF_CONTRASTS;
	}
	else if (STATISTICAL_TEST == FTEST)
	{
		NUMBER_OF_STATISTICAL_MAPS = 1;
	}

	// Setup parameters and memory prior to permutations, to save time in each permutation
	SetupPermutationTestSecondLevel(d_First_Level_Results, d_MNI_Brain_Mask);

	// Generate a random sign matrix, unless one is provided
	if ( (STATISTICAL_TEST == GROUP_MEAN) && (!USE_PERMUTATION_FILE) )
	{
		GenerateSignMatrixSecondLevel();
	}
	// Generate a random permutation matrix, unless one is provided
	else if (!USE_PERMUTATION_FILE)
	{
		GeneratePermutationMatrixSecondLevel();
	}

    // Loop over all the permutations, save the maximum test value from each permutation

	for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
	{
		if ((WRAPPER == BASH) && PRINT)
		{
			printf("Starting permutation %i \n",p+1);
		}

		// Calculate statistical maps
		CalculateStatisticalMapsSecondLevelPermutation(p);

		// Voxel distribution
		if (INFERENCE_MODE == VOXEL)
		{
			// Calculate max test value
			h_Permutation_Distribution[p] = CalculateMaxAtomic(d_Statistical_Maps, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
		}
		// Cluster distribution, extent or mass
		else if ( (INFERENCE_MODE == CLUSTER_EXTENT) || (INFERENCE_MODE == CLUSTER_MASS) )
		{
			ClusterizeOpenCLPermutation(MAX_CLUSTER, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
			h_Permutation_Distribution[p] = MAX_CLUSTER;
		}
		// Threshold free cluster enhancement
		else if (INFERENCE_MODE == TFCE)
		{
			maxActivation = CalculateMaxAtomic(d_Statistical_Maps, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
			float delta = 0.2846;
		    ClusterizeOpenCLTFCEPermutation(MAX_VALUE, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, maxActivation, delta);
		    h_Permutation_Distribution[p] = MAX_VALUE;
		}
	}

	std::vector<float> max_values (h_Permutation_Distribution, h_Permutation_Distribution + NUMBER_OF_PERMUTATIONS);
	std::sort (max_values.begin(), max_values.begin() + NUMBER_OF_PERMUTATIONS);

	// Find the threshold for the specified significance level
	SIGNIFICANCE_THRESHOLD = max_values[(int)(ceil((1.0f - SIGNIFICANCE_LEVEL) * (float)NUMBER_OF_PERMUTATIONS))-1];

	if (WRAPPER == BASH)
	{
		printf("Permutation threshold for a significance level of %f is %f \n",SIGNIFICANCE_LEVEL, SIGNIFICANCE_THRESHOLD);
	}

	CleanupPermutationTestSecondLevel();
}

// Calculates permutation based p-values in each voxel
void BROCCOLI_LIB::CalculatePermutationPValues(cl_mem d_Mask, int DATA_W, int DATA_H, int DATA_D)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(DATA_W, DATA_H, DATA_D);

	// Loop over contrasts
	//for (int contrast = 0; contrast < NUMBER_OF_CONTRASTS; contrast++)
	for (int contrast = 0; contrast < 1; contrast++)
	{
		// Copy max values to constant memory
		clEnqueueWriteBuffer(commandQueue, c_Permutation_Distribution, CL_TRUE, 0, NUMBER_OF_PERMUTATIONS * sizeof(float), &h_Permutation_Distribution[contrast * NUMBER_OF_PERMUTATIONS], 0, NULL, NULL);

		if ( (INFERENCE_MODE == VOXEL) || (INFERENCE_MODE == TFCE) )
		{
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 0, sizeof(cl_mem), &d_P_Values);
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 1, sizeof(cl_mem), &d_Statistical_Maps);
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 2, sizeof(cl_mem), &d_Mask);
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 3, sizeof(cl_mem), &c_Permutation_Distribution);
			//clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 4, sizeof(int),    &contrast);
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 4, sizeof(int),    &DATA_W);
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 5, sizeof(int),    &DATA_H);
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 6, sizeof(int),    &DATA_D);
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 7, sizeof(int),    &NUMBER_OF_PERMUTATIONS);
			runKernelErrorCalculatePermutationPValuesVoxelLevelInference = clEnqueueNDRangeKernel(commandQueue, CalculatePermutationPValuesVoxelLevelInferenceKernel, 3, NULL, globalWorkSizeCalculatePermutationPValues, localWorkSizeCalculatePermutationPValues, 0, NULL, NULL);
		}
		else if ( (INFERENCE_MODE == CLUSTER_EXTENT) || (INFERENCE_MODE == CLUSTER_MASS) )
		{
			clSetKernelArg(CalculatePermutationPValuesClusterLevelInferenceKernel, 0, sizeof(cl_mem), &d_P_Values);
			clSetKernelArg(CalculatePermutationPValuesClusterLevelInferenceKernel, 1, sizeof(cl_mem), &d_Cluster_Indices);
			clSetKernelArg(CalculatePermutationPValuesClusterLevelInferenceKernel, 2, sizeof(cl_mem), &d_Cluster_Sizes);
			clSetKernelArg(CalculatePermutationPValuesClusterLevelInferenceKernel, 3, sizeof(cl_mem), &d_Mask);
			clSetKernelArg(CalculatePermutationPValuesClusterLevelInferenceKernel, 4, sizeof(cl_mem), &c_Permutation_Distribution);
			//clSetKernelArg(CalculatePermutationPValuesClusterLevelInferenceKernel, 5, sizeof(int),    &contrast);
			clSetKernelArg(CalculatePermutationPValuesClusterLevelInferenceKernel, 5, sizeof(int),    &DATA_W);
			clSetKernelArg(CalculatePermutationPValuesClusterLevelInferenceKernel, 6, sizeof(int),    &DATA_H);
			clSetKernelArg(CalculatePermutationPValuesClusterLevelInferenceKernel, 7, sizeof(int),    &DATA_D);
			clSetKernelArg(CalculatePermutationPValuesClusterLevelInferenceKernel, 8, sizeof(int),    &NUMBER_OF_PERMUTATIONS);
			runKernelErrorCalculatePermutationPValuesClusterLevelInference = clEnqueueNDRangeKernel(commandQueue, CalculatePermutationPValuesClusterLevelInferenceKernel, 3, NULL, globalWorkSizeCalculatePermutationPValues, localWorkSizeCalculatePermutationPValues, 0, NULL, NULL);
		}
	}
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

		// Generate a random number and switch position of two existing numbers
        // all permutations are valid since we have whitened the data
        for (int i = 0; i < EPI_DATA_T; i++)
        {
            int j = i + rand() / (RAND_MAX / (EPI_DATA_T-i)+1);
            unsigned short int temp = h_Permutation_Matrix[j + p * EPI_DATA_T];
            h_Permutation_Matrix[j + p * EPI_DATA_T] = h_Permutation_Matrix[i + p * EPI_DATA_T];
            h_Permutation_Matrix[i + p * EPI_DATA_T] = temp;
        }
    }
}

// Generates a permutation matrix for several subjects
void BROCCOLI_LIB::GeneratePermutationMatrixSecondLevel()
{
	// Do all the possible permutations
	if (DO_ALL_PERMUTATIONS)
	{
		for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
		{
			// Generate numbers from 0 to number of subjects
			for (int i = 0; i < NUMBER_OF_SUBJECTS; i++)
			{
				h_Permutation_Matrix[i + p * NUMBER_OF_SUBJECTS] = (unsigned short int)i;
			}

			// Generate a random number and switch position of two existing numbers
			for (int i = 0; i < NUMBER_OF_SUBJECTS - 1; i++)
			{
				int j = i + rand() / (RAND_MAX / (NUMBER_OF_SUBJECTS-i)+1);

				// Check if random permutation is valid?!

				unsigned short int temp = h_Permutation_Matrix[j + p * NUMBER_OF_SUBJECTS];
				h_Permutation_Matrix[j + p * NUMBER_OF_SUBJECTS] = h_Permutation_Matrix[i + p * NUMBER_OF_SUBJECTS];
				h_Permutation_Matrix[i + p * NUMBER_OF_SUBJECTS] = temp;
			}
		}
	}
	else
	{
		for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
		{
			// Generate numbers from 0 to number of subjects
			for (int i = 0; i < NUMBER_OF_SUBJECTS; i++)
			{
				h_Permutation_Matrix[i + p * NUMBER_OF_SUBJECTS] = (unsigned short int)i;
			}

			// Generate a random number and switch position of two existing numbers
			for (int i = 0; i < NUMBER_OF_SUBJECTS - 1; i++)
			{
				int j = i + rand() / (RAND_MAX / (NUMBER_OF_SUBJECTS-i)+1);

				// Check if random permutation is valid?!

				unsigned short int temp = h_Permutation_Matrix[j + p * NUMBER_OF_SUBJECTS];
				h_Permutation_Matrix[j + p * NUMBER_OF_SUBJECTS] = h_Permutation_Matrix[i + p * NUMBER_OF_SUBJECTS];
				h_Permutation_Matrix[i + p * NUMBER_OF_SUBJECTS] = temp;
			}
		}
	}
}

// Generates a sign flipping matrix for several subjects
void BROCCOLI_LIB::GenerateSignMatrixSecondLevel()
{
	// Do all the possible sign flips
	if (DO_ALL_PERMUTATIONS)
	{
		unsigned long int p = 0;
		while (p < NUMBER_OF_PERMUTATIONS)
		{
			// Create a new set of flips
			for (int s = 0; s < NUMBER_OF_SUBJECTS; s++)
			{
				// Check if we should flip this subject or not
				if (p & (unsigned long int)(pow(2.0,(double)s)))
				{
					h_Sign_Matrix[s + p * NUMBER_OF_SUBJECTS] = -1.0f;
				}
				else
				{
					h_Sign_Matrix[s + p * NUMBER_OF_SUBJECTS] = 1.0f;
				}
			}
			p++;
		}
	}
	// Generate "more random" sign flips, compared to simply looping through 0001, 0010, 0011, 0100, 0101 etc
	else
	{
		unsigned long int p = 0;
		while (p < NUMBER_OF_PERMUTATIONS)
		{
			// Create a new set of flips
			for (int s = 0; s < NUMBER_OF_SUBJECTS; s++)
			{
				unsigned long int randNumber = rand() % 2; // Random number, 0 or 1

				// Check if we should flip this subject or not
				if (randNumber)
				{
					h_Sign_Matrix[s + p * NUMBER_OF_SUBJECTS] = -1.0f;
				}
				else
				{
					h_Sign_Matrix[s + p * NUMBER_OF_SUBJECTS] = 1.0f;
				}
			}
			p++;
		}
	}
}

// Generates new fMRI volumes for first level analysis, by inverse whitening and permutation at the same time
// (for second level analysis, the design matrix is permuted instead, as in the function randomise in FSL, so no data need to be generated)
void BROCCOLI_LIB::GeneratePermutedVolumesFirstLevel(cl_mem d_Permuted_fMRI_Volumes, cl_mem d_Whitened_fMRI_Volumes, int permutation)
{
	// Copy a new permutation vector to constant memory
	clEnqueueWriteBuffer(commandQueue, c_Permutation_Vector, CL_TRUE, 0, EPI_DATA_T * sizeof(unsigned short int), &h_Permutation_Matrix[permutation * EPI_DATA_T], 0, NULL, NULL);

	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 0, sizeof(cl_mem), &d_Permuted_fMRI_Volumes);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 1, sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 2, sizeof(cl_mem), &d_AR1_Estimates);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 3, sizeof(cl_mem), &d_AR2_Estimates);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 4, sizeof(cl_mem), &d_AR3_Estimates);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 5, sizeof(cl_mem), &d_AR4_Estimates);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 6, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 7, sizeof(cl_mem), &c_Permutation_Vector);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 8, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 9, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 10, sizeof(int),   &EPI_DATA_D);
	clSetKernelArg(GeneratePermutedVolumesFirstLevelKernel, 11, sizeof(int),   &EPI_DATA_T);

	clEnqueueNDRangeKernel(commandQueue, GeneratePermutedVolumesFirstLevelKernel, 3, NULL, globalWorkSizeGeneratePermutedVolumesFirstLevel, localWorkSizeGeneratePermutedVolumesFirstLevel, 0, NULL, NULL);
	clFinish(commandQueue);
}







// Solves an equation system using QR-factorization, used by the Linear registration algorithm
// A x = h
// A p = h
void BROCCOLI_LIB::SolveEquationSystem(float* h_Parameter_Vector, float* h_A_Matrix, float* h_h_Vector, int N)
{
	Eigen::MatrixXd A(N,N);
	Eigen::VectorXd h(N,1);

	// Make a double version of the matrix and the vector and put into Eigen variables
	for (int i = 0; i < N; i++)
	{
		h(i) = (double)h_h_Vector[i];

		for (int j = 0; j < N; j++)
		{
			A(i,j) = (double)h_A_Matrix[i + j*N];
		}
	}

	// Solve the equation system using QR in Eigen
	Eigen::VectorXd x = A.fullPivHouseholderQr().solve(h);

	// Calculate the error
	//relativeErrorEquationSystemSolution = (A*x - h).norm() / h.norm(); // norm() is L2 norm

	// Convert the solution back to floats and store in an ordinary array
    for (int i = 0; i < N; i++)
	{
		h_Parameter_Vector[i] = (float)x(i);
	}
}



// This function creates detrending regressors to be used before whitening
void BROCCOLI_LIB::SetupDetrendingRegressors(int N)
{
	Eigen::VectorXd Ones(N,1);
	Eigen::VectorXd Linear(N,1);
	Eigen::VectorXd Quadratic(N,1);
	Eigen::VectorXd Cubic(N,1);

	// Ones and linear trend
	double offset = -((double)N - 1.0)/2.0;
	for (int t = 0; t < N; t++)
	{
		Ones(t) = 1.0;
		Linear(t) = offset + (double)t;
	}

	// Calculate quadratic and cubic trends
	Quadratic = Linear.cwiseProduct(Linear);
	Cubic = Linear.cwiseProduct(Linear);
	Cubic = Cubic.cwiseProduct(Linear);

	// Normalize
	Linear = Linear / Linear.maxCoeff();
	Quadratic = Quadratic / Quadratic.maxCoeff();
	Cubic = Cubic / Cubic.maxCoeff();

	// Demean
	DemeanRegressor(Linear,N);
	DemeanRegressor(Quadratic,N);
	DemeanRegressor(Cubic,N);

	// Setup total detrending design matrix
	Eigen::MatrixXd X(N,4);
	for (int i = 0; i < N; i++)
	{
		X(i,0) = Ones(i);
		X(i,1) = Linear(i);
		X(i,2) = Quadratic(i);
		X(i,3) = Cubic(i);
	}

	// Calculate pseudo inverse (could be done with SVD instead, or QR)
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

// This function creates detrending and motion regressors to be used before whitening
void BROCCOLI_LIB::SetupDetrendingAndMotionRegressors(int N)
{
	Eigen::VectorXd Ones(N,1);
	Eigen::VectorXd Linear(N,1);
	Eigen::VectorXd Quadratic(N,1);
	Eigen::VectorXd Cubic(N,1);

	// Ones and linear trend
	double offset = -((double)N - 1.0)/2.0;
	for (int t = 0; t < N; t++)
	{
		Ones(t) = 1.0;
		Linear(t) = offset + (double)t;
	}

	// Calculate quadratic and cubic trends
	Quadratic = Linear.cwiseProduct(Linear);
	Cubic = Linear.cwiseProduct(Linear);
	Cubic = Cubic.cwiseProduct(Linear);

	// Normalize
	Linear = Linear / Linear.maxCoeff();
	Quadratic = Quadratic / Quadratic.maxCoeff();
	Cubic = Cubic / Cubic.maxCoeff();

	// Setup total detrending design matrix
	Eigen::MatrixXd X(N,10);
	for (int i = 0; i < N; i++)
	{
		X(i,0) = Ones(i);
		X(i,1) = Linear(i);
		X(i,2) = Quadratic(i);
		X(i,3) = Cubic(i);

		X(i,4) = h_Motion_Parameters[i + 0 * N];
		X(i,5) = h_Motion_Parameters[i + 1 * N];
		X(i,6) = h_Motion_Parameters[i + 2 * N];
		X(i,7) = h_Motion_Parameters[i + 3 * N];
		X(i,8) = h_Motion_Parameters[i + 4 * N];
		X(i,9) = h_Motion_Parameters[i + 5 * N];
	}

	int MEAN_REGRESSOR = 0;

	// Demean regressors
	for (int r = 0; r < 10; r++)
	{
		if (r != MEAN_REGRESSOR)
		{
			Eigen::VectorXd regressor = X.block(0,r,N,1);
			DemeanRegressor(regressor,N);
			X.block(0,r,N,1) = regressor;
		}
	}

	// Calculate pseudo inverse (could be done with SVD instead, or QR)
	Eigen::MatrixXd xtx(10,10);
	xtx = X.transpose() * X;
	Eigen::MatrixXd inv_xtx = xtx.inverse();
	Eigen::MatrixXd xtxxt = inv_xtx * X.transpose();

	// Finally store regressors in ordinary arrays
	for (int i = 0; i < N; i++)
	{
		h_X_Detrend[i + 0 * N] = (float)X(i,0);
		h_X_Detrend[i + 1 * N] = (float)X(i,1);
		h_X_Detrend[i + 2 * N] = (float)X(i,2);
		h_X_Detrend[i + 3 * N] = (float)X(i,3);

		h_X_Detrend[i + 4 * N] = (float)X(i,4);
		h_X_Detrend[i + 5 * N] = (float)X(i,5);
		h_X_Detrend[i + 6 * N] = (float)X(i,6);
		h_X_Detrend[i + 7 * N] = (float)X(i,7);
		h_X_Detrend[i + 8 * N] = (float)X(i,8);
		h_X_Detrend[i + 9 * N] = (float)X(i,9);

		h_xtxxt_Detrend[i + 0 * N] = (float)xtxxt(0,i);
		h_xtxxt_Detrend[i + 1 * N] = (float)xtxxt(1,i);
		h_xtxxt_Detrend[i + 2 * N] = (float)xtxxt(2,i);
		h_xtxxt_Detrend[i + 3 * N] = (float)xtxxt(3,i);

		h_xtxxt_Detrend[i + 4 * N] = (float)xtxxt(4,i);
		h_xtxxt_Detrend[i + 5 * N] = (float)xtxxt(5,i);
		h_xtxxt_Detrend[i + 6 * N] = (float)xtxxt(6,i);
		h_xtxxt_Detrend[i + 7 * N] = (float)xtxxt(7,i);
		h_xtxxt_Detrend[i + 8 * N] = (float)xtxxt(8,i);
		h_xtxxt_Detrend[i + 9 * N] = (float)xtxxt(9,i);

	}
}

// Demeans a regressor stored as an array
void BROCCOLI_LIB::DemeanRegressor(float* Regressor, int N)
{
	float mean = 0.0f;
	for (int t = 0; t < N; t++)
	{
		mean += Regressor[t];
	}
	mean /= (float)N;

	for (int t = 0; t < N; t++)
	{
		Regressor[t] -= mean;
	}
}

// Demeans a regressor stored as an Eigen variable
void BROCCOLI_LIB::DemeanRegressor(Eigen::VectorXd& Regressor, int N)
{
	double mean = 0.0;
	for (int t = 0; t < N; t++)
	{
		mean += Regressor(t);
	}
	mean /= (float)N;

	for (int t = 0; t < N; t++)
	{
		Regressor(t) -= mean;
	}
}

// Setups all regressors for first level analysis
Eigen::MatrixXd BROCCOLI_LIB::SetupGLMRegressorsFirstLevel(int N)
{
	// Calculate total number of regressors
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + NUMBER_OF_CONFOUND_REGRESSORS*REGRESS_CONFOUNDS;

	// Create detrending regressors
	Eigen::VectorXd Ones(N,1);
	Eigen::VectorXd Linear(N,1);
	Eigen::VectorXd Quadratic(N,1);
	Eigen::VectorXd Cubic(N,1);

	// Ones and linear trend
	float offset = -((float)N - 1.0f)/2.0f;
	for (int t = 0; t < N; t++)
	{
		Ones(t) = 1.0;
		Linear(t) = offset + (double)t;
	}

	// Calculate quadratic and cubic trends
	Quadratic = Linear.cwiseProduct(Linear);
	Cubic = Linear.cwiseProduct(Linear);
	Cubic = Cubic.cwiseProduct(Linear);

	// Normalize
	Linear = Linear / Linear.maxCoeff();
	double minn = abs(Quadratic.minCoeff());
	double maxx = Quadratic.maxCoeff();
	if (maxx > minn)
	{
		Quadratic = Quadratic / maxx;
	}
	else
	{
		Quadratic = Quadratic / minn;
	}
	Cubic = Cubic / Cubic.maxCoeff();

	// Create temporal derivatives if requested and then convolve all regressors with HRF
	if (USE_TEMPORAL_DERIVATIVES && !RAW_REGRESSORS)
	{
		GenerateRegressorTemporalDerivatives(h_X_GLM_With_Temporal_Derivatives, h_X_GLM_In, N, NUMBER_OF_GLM_REGRESSORS);
		ConvolveRegressorsWithHRF(h_X_GLM_Convolved, h_X_GLM_With_Temporal_Derivatives, N, NUMBER_OF_GLM_REGRESSORS*2);
	}
	// Convolve regressors with HRF
	else if (!RAW_REGRESSORS)
	{
		ConvolveRegressorsWithHRF(h_X_GLM_Convolved, h_X_GLM_In, N, NUMBER_OF_GLM_REGRESSORS);
	}
	// Just copy raw regressors
	else if (RAW_REGRESSORS)
	{
		// Loop over samples
		for (int i = 0; i < N; i++)
		{
			// Loop over regressors
			for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				h_X_GLM_Convolved[i + r * N] = h_X_GLM_In[i + r * N];
			}
		}
	}

	// Setup total design matrix
	Eigen::MatrixXd X(N,NUMBER_OF_TOTAL_GLM_REGRESSORS);

	// Loop over samples
	for (int i = 0; i < N; i++)
	{
		// Regressors for paradigms
		for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1); r++)
		{
			X(i,r) = (double)h_X_GLM_Convolved[i + r * N];
		}

		// Detrending regressors
		X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 0) = Ones(i);
		X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 1) = Linear(i);
		X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 2) = Quadratic(i);
		X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 3) = Cubic(i);

		if (REGRESS_MOTION)
		{
			// Motion regressors
			X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 0) = h_Motion_Parameters[i + 0 * N];
			X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 1) = h_Motion_Parameters[i + 1 * N];
			X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 2) = h_Motion_Parameters[i + 2 * N];
			X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 3) = h_Motion_Parameters[i + 3 * N];
			X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 4) = h_Motion_Parameters[i + 4 * N];
			X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 5) = h_Motion_Parameters[i + 5 * N];
		}

		if (REGRESS_CONFOUNDS)
		{
			// Confounding regressors
			for (int r = 0; r < NUMBER_OF_CONFOUND_REGRESSORS; r++)
			{
				X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + r) = (double)h_X_GLM_Confounds[i + r * N];
			}
		}
	}

	// Calculate which regressor contains only ones
	int MEAN_REGRESSOR = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1);

	// Demean regressors
	for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
	{
		if (r != MEAN_REGRESSOR)
		{
			Eigen::VectorXd regressor = X.block(0,r,N,1);
			DemeanRegressor(regressor,N);
			X.block(0,r,N,1) = regressor;
		}
	}

	// Calculate pseudo inverse (could be done with SVD instead, or QR)
	Eigen::MatrixXd xtx(NUMBER_OF_TOTAL_GLM_REGRESSORS,NUMBER_OF_TOTAL_GLM_REGRESSORS);
	xtx = X.transpose() * X;
	Eigen::MatrixXd inv_xtx = xtx.inverse();
	Eigen::MatrixXd xtxxt = inv_xtx * X.transpose();

	// Finally store regressors in ordinary arrays
	for (int i = 0; i < N; i++)
	{
		// Regressors for paradigms
		for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1); r++)
		{
			h_X_GLM[i + r * N] = X(i,r);
		}

		// Detrending regressors
		h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 0) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 0);
		h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 1) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 1);
		h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 2) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 2);
		h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 3) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 3);

		if (REGRESS_MOTION)
		{
			// Motion regressors
			h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 0) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 0);
			h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 1) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 1);
			h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 2) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 2);
			h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 3) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 3);
			h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 4) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 4);
			h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 5) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 5);
		}

		if (REGRESS_CONFOUNDS)
		{
			for (int r = 0; r < NUMBER_OF_CONFOUND_REGRESSORS; r++)
			{
				h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + r) * N] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + r);
			}
		}

		// Regressors for paradigms
		for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1); r++)
		{
			h_xtxxt_GLM[i + r * N] = (float)xtxxt(r,i);
		}

		// Detrending regressors
		h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 0) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 0,i);
		h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 1) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 1,i);
		h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 2) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 2,i);
		h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 3) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + 3,i);

		if (REGRESS_MOTION)
		{
			// Motion regressors
			h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 0) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 0,i);
			h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 1) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 1,i);
			h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 2) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 2,i);
			h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 3) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 3,i);
			h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 4) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 4,i);
			h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 5) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + 5,i);
		}

		if (REGRESS_CONFOUNDS)
		{
			// Confounding regressors
			for (int r = 0; r < NUMBER_OF_CONFOUND_REGRESSORS; r++)
			{
				h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + r) * N] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + r,i);
			}
		}
	}

	return inv_xtx;
}

// Setup variables for a t-test for first level analysis
void BROCCOLI_LIB::SetupTTestFirstLevel(int N)
{
	// Setup GLM regressors
	Eigen::MatrixXd inv_xtx = SetupGLMRegressorsFirstLevel(N);

	// Now update the contrast vectors also
	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		Eigen::VectorXd Contrast(NUMBER_OF_TOTAL_GLM_REGRESSORS);

		// Copy contrasts for paradigm regressors
		if (USE_TEMPORAL_DERIVATIVES)
		{
			// Paradigm regressors
			int rr = 0;
			for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				// Original regressor
				Contrast(rr) = (double)h_Contrasts_In[NUMBER_OF_GLM_REGRESSORS * c + r];
				h_Contrasts[NUMBER_OF_TOTAL_GLM_REGRESSORS * c + rr] = (float)Contrast(rr);

				// Temporal derivative
				Contrast(rr+1) = 0.0;
				h_Contrasts[NUMBER_OF_TOTAL_GLM_REGRESSORS * c + rr + 1] = 0.0f;

				rr += 2;
			}

			// Set all other contrasts to 0
			for (int r = NUMBER_OF_GLM_REGRESSORS*2; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
			{
				Contrast(r) = 0.0;
				h_Contrasts[NUMBER_OF_TOTAL_GLM_REGRESSORS * c + r] = 0.0f;
			}
		}
		// No temporal derivatives
		else
		{
			// Original regressors
			for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				Contrast(r) = (double)h_Contrasts_In[NUMBER_OF_GLM_REGRESSORS * c + r];
				h_Contrasts[NUMBER_OF_TOTAL_GLM_REGRESSORS * c + r] = (float)Contrast(r);
			}

			// Set all other contrasts to 0
			for (int r = NUMBER_OF_GLM_REGRESSORS; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
			{
				Contrast(r) = 0.0;
				h_Contrasts[NUMBER_OF_TOTAL_GLM_REGRESSORS * c + r] = 0.0f;
			}
		}

		// Calculate scalar constants
		Eigen::VectorXd scalar = Contrast.transpose() * inv_xtx * Contrast;
		h_ctxtxc_GLM[c] = scalar(0);
	}
}


// Setup variables for a F-test for first level analysis
void BROCCOLI_LIB::SetupFTestFirstLevel(int N)
{
	// Setup GLM regressors
	Eigen::MatrixXd inv_xtx = SetupGLMRegressorsFirstLevel(N);

	// Now update the contrasts also
	Eigen::MatrixXd Contrasts(NUMBER_OF_CONTRASTS,NUMBER_OF_TOTAL_GLM_REGRESSORS);

	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		if (USE_TEMPORAL_DERIVATIVES)
		{
			// Copy contrasts for paradigm regressors
			int rr = 0;
			for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				Contrasts(c,rr) = (double)h_Contrasts_In[NUMBER_OF_TOTAL_GLM_REGRESSORS * c + r];
				h_Contrasts[NUMBER_OF_TOTAL_GLM_REGRESSORS * c + rr] = (float)Contrasts(c,rr);

				Contrasts(c,rr+1) = 0.0;
				h_Contrasts[NUMBER_OF_TOTAL_GLM_REGRESSORS * c + rr + 1] = 0.0f;

				rr += 2;
			}

			// Set all other contrasts to 0
			for (int r = NUMBER_OF_GLM_REGRESSORS*2; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
			{
				Contrasts(c,r) = 0.0;
				h_Contrasts[NUMBER_OF_TOTAL_GLM_REGRESSORS * c + r] = 0.0f;
			}
		}
		// No temporal derivatives
		else
		{
			// Copy contrasts for paradigm regressors
			for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				Contrasts(c,r) = (double)h_Contrasts_In[NUMBER_OF_TOTAL_GLM_REGRESSORS * c + r];
				h_Contrasts[NUMBER_OF_TOTAL_GLM_REGRESSORS * c + r] = (float)Contrasts(c,r);
			}

			// Set all other contrasts to 0
			for (int r = NUMBER_OF_GLM_REGRESSORS; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
			{
				Contrasts(c,r) = 0.0;
				h_Contrasts[NUMBER_OF_TOTAL_GLM_REGRESSORS * c + r] = 0.0f;
			}
		}
	}

	// Calculate scalar constants
	Eigen::MatrixXd temp = Contrasts * inv_xtx * Contrasts.transpose();
	Eigen::MatrixXd ctxtxc = temp.inverse();

	for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		for (int cc = 0; cc < NUMBER_OF_CONTRASTS; cc++)
		{
			h_ctxtxc_GLM[c + cc  * NUMBER_OF_CONTRASTS] = ctxtxc(c,cc);
		}
	}
}

// This code was copied (with permission) from http://www.johndcook.com/Gamma.cpp
// (lgamma seems to exist for Linux compilers, but not for Visual studio)

double BROCCOLI_LIB::Gamma(double x)
{
    // Split the function domain into three intervals:
    // (0, 0.001), [0.001, 12), and (12, infinity)

    ///////////////////////////////////////////////////////////////////////////
    // First interval: (0, 0.001)
	//
	// For small x, 1/Gamma(x) has power series x + gamma x^2  - ...
	// So in this range, 1/Gamma(x) = x + gamma x^2 with error on the order of x^3.
	// The relative error over this interval is less than 6e-7.

	const double gamma = 0.577215664901532860606512090; // Euler's gamma constant

    if (x < 0.001)
        return 1.0/(x*(1.0 + gamma*x));

    ///////////////////////////////////////////////////////////////////////////
    // Second interval: [0.001, 12)

	if (x < 12.0)
    {
        // The algorithm directly approximates gamma over (1,2) and uses
        // reduction identities to reduce other arguments to this interval.

		double y = x;
        int n = 0;
        bool arg_was_less_than_one = (y < 1.0);

        // Add or subtract integers as necessary to bring y into (1,2)
        // Will correct for this below
        if (arg_was_less_than_one)
        {
            y += 1.0;
        }
        else
        {
            n = static_cast<int> (floor(y)) - 1;  // will use n later
            y -= n;
        }

        // numerator coefficients for approximation over the interval (1,2)
        static const double p[] =
        {
            -1.71618513886549492533811E+0,
             2.47656508055759199108314E+1,
            -3.79804256470945635097577E+2,
             6.29331155312818442661052E+2,
             8.66966202790413211295064E+2,
            -3.14512729688483675254357E+4,
            -3.61444134186911729807069E+4,
             6.64561438202405440627855E+4
        };

        // denominator coefficients for approximation over the interval (1,2)
        static const double q[] =
        {
            -3.08402300119738975254353E+1,
             3.15350626979604161529144E+2,
            -1.01515636749021914166146E+3,
            -3.10777167157231109440444E+3,
             2.25381184209801510330112E+4,
             4.75584627752788110767815E+3,
            -1.34659959864969306392456E+5,
            -1.15132259675553483497211E+5
        };

        double num = 0.0;
        double den = 1.0;
        int i;

        double z = y - 1;
        for (i = 0; i < 8; i++)
        {
            num = (num + p[i])*z;
            den = den*z + q[i];
        }
        double result = num/den + 1.0;

        // Apply correction if argument was not initially in (1,2)
        if (arg_was_less_than_one)
        {
            // Use identity gamma(z) = gamma(z+1)/z
            // The variable "result" now holds gamma of the original y + 1
            // Thus we use y-1 to get back the orginal y.
            result /= (y-1.0);
        }
        else
        {
            // Use the identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
            for (i = 0; i < n; i++)
                result *= y++;
        }

		return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Third interval: [12, infinity)

    if (x > 171.624)
    {
		// Correct answer too large to display. Force +infinity.
		double temp = DBL_MAX;
		return temp*2.0;
    }

    return exp(LogGamma(x));
}

// This code was copied (with permission) from http://www.johndcook.com/Gamma.cpp
// (lgamma seems to exist for Linux compilers, but not for Visual studio)

double BROCCOLI_LIB::LogGamma(double x)
{
    if (x < 12.0)
    {
        return log(fabs(Gamma(x)));
    }

	// Abramowitz and Stegun 6.1.41
    // Asymptotic series should be good to at least 11 or 12 figures
    // For error analysis, see Whittiker and Watson
    // A Course in Modern Analysis (1927), page 252

    static const double c[8] =
    {
		 1.0/12.0,
		-1.0/360.0,
		1.0/1260.0,
		-1.0/1680.0,
		1.0/1188.0,
		-691.0/360360.0,
		1.0/156.0,
		-3617.0/122400.0
    };
    double z = 1.0/(x*x);
    double sum = c[7];
    for (int i=6; i >= 0; i--)
    {
        sum *= z;
        sum += c[i];
    }
    double series = sum/x;

    static const double halfLogTwoPi = 0.91893853320467274178032973640562;
    double logGamma = (x - 0.5)*log(x) - x + halfLogTwoPi + series;
	return logGamma;
}

// Gamma probability density function
float BROCCOLI_LIB::Gpdf(double value, double shape, double scale)
{
	return (exp( (shape - 1.0) * log(value) + shape * log(scale) - scale * value - LogGamma(shape) ));
}

// Creates an HRF as in SPM
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
	HRF_LENGTH = length/downsample_factor + 1;

	hrf = (float*)malloc(sizeof(float) * HRF_LENGTH);

	for (int i = 0; i < HRF_LENGTH; i++)
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

	// Normalize
	float sum = 0.0f;
	for (int i = 0; i < HRF_LENGTH; i++)
	{
		sum += hrf[i];
	}
	for (int i = 0; i < HRF_LENGTH; i++)
	{
		hrf[i] /= sum;
	}

	free(highres_hrf);
}

// Calculate derivatives of regressors, by simple one step difference
void BROCCOLI_LIB::GenerateRegressorTemporalDerivatives(float * Regressors_With_Temporal_Derivatives, float* Regressors, int NUMBER_OF_TIMEPOINTS, int NUMBER_OF_REGRESSORS)
{
	int rr = 0;
	for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{
		for (int t = 0; t < NUMBER_OF_TIMEPOINTS; t++)
		{
			// Copy original regressor
			Regressors_With_Temporal_Derivatives[t + rr * NUMBER_OF_TIMEPOINTS] = Regressors[t + r * NUMBER_OF_TIMEPOINTS];

			// Calculate derivative and save as second regressor
			if ( ((t-1) >= 0) )
			{
				Regressors_With_Temporal_Derivatives[t + (rr + 1) * NUMBER_OF_TIMEPOINTS] = Regressors[t + r * NUMBER_OF_TIMEPOINTS] - Regressors[(t-1) + r * NUMBER_OF_TIMEPOINTS];
			}
			else
			{
				Regressors_With_Temporal_Derivatives[t + (rr + 1) * NUMBER_OF_TIMEPOINTS] = 0.0f;
			}
		}
		rr += 2;
	}
}

// Convolve regressors with a created HRF
void BROCCOLI_LIB::ConvolveRegressorsWithHRF(float* Convolved_Regressors, float* Regressors, int NUMBER_OF_TIMEPOINTS, int NUMBER_OF_REGRESSORS)
{
	CreateHRF();

	for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{
		for (int t = 0; t < NUMBER_OF_TIMEPOINTS; t++)
		{
			Convolved_Regressors[t + r * NUMBER_OF_TIMEPOINTS] = 0.0f;

			// 1D convolution
			//int offset = -(int)(((float)HRF_LENGTH - 1.0f)/2.0f);
			int offset = -(int)(((float)HRF_LENGTH - 1.0f)/1.0f);;
			for (int tt = HRF_LENGTH - 1; tt >= 0; tt--)
			{
				if ( ((t + offset) >= 0) && ((t + offset) < NUMBER_OF_TIMEPOINTS) )
				{
					Convolved_Regressors[t + r * NUMBER_OF_TIMEPOINTS] += Regressors[t + offset + r * NUMBER_OF_TIMEPOINTS] * hrf[tt];
				}
				offset++;
			}
		}
	}

	free(hrf);
}

int BROCCOLI_LIB::Calculate3DIndex(int x, int y, int z, int DATA_W, int DATA_H)
{
	return x + y * DATA_W + z * DATA_W * DATA_H;
}


// Takes a volume, thresholds it and labels each cluster, calculates cluster sizes and cluster masses, works by recursion, uses a single CPU thread
void BROCCOLI_LIB::Clusterize(int* Cluster_Indices,
		                      int& MAX_CLUSTER_SIZE,
		                      float& MAX_CLUSTER_MASS,
		                      int& NUMBER_OF_CLUSTERS,
		                      float* Data,
		                      float Threshold,
		                      float* Mask,
		                      int DATA_W,
		                      int DATA_H,
		                      int DATA_D,
		                      int GET_VOXEL_LABELS,
		                      int GET_CLUSTER_MASS)
{
	// Vector of clusters
	std::vector<std::vector<Coords3D> > clusters;

	// Keep track of labelled voxels
	int* ccMask = (int*)malloc(DATA_W * DATA_H * DATA_D * sizeof(int));
	for (int i = 0; i < (DATA_W * DATA_H * DATA_D); ++i)
	{
		ccMask[i] = 0;
		Cluster_Indices[i] = 0;
	}

	// Loop over volume voxels
	for (int z = 0; z < DATA_D; ++z)
	{
		for (int y = 0; y < DATA_H; ++y)
		{
			for (int x = 0; x < DATA_W; ++x)
			{
				// Only work with voxels inside mask that are above threshold, and have not been labelled previously
				if ( (Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] == 1.0f) && (Data[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] > Threshold ) && (ccMask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] == 0 ) )
				{
					// Start a new cluster and get a shortcut
					clusters.push_back(std::vector<Coords3D>());
					std::vector<Coords3D>& cluster = clusters.back();

					// Add first voxel to current cluster
					cluster.push_back(Coords3D(x, y, z));

					// Mark voxel as labelled
					ccMask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 1;

					// Add all voxels that are connected to current voxel,
					// note that cluster.size() changes inside the loop
					for (int i = 0; i != cluster.size(); ++i)
					{
						// Get 26-connected neighbours of the current voxel
						Neighbors3D26(neighbours, cluster[i]);

						// Loop over all neighbours
						for (int j = 0; j != 26; ++j)
						{
							// Unpack coordinates
							int const x2 = neighbours[j][X];
							int const y2 = neighbours[j][Y];
							int const z2 = neighbours[j][Z];

							// Check if neighbour is inside volume
							if ( (x2 >= 0) && (x2 < DATA_W) && (y2 >= 0) && (y2 < DATA_H) && (z2 >= 0) && (z2 < DATA_D) )
							{
								// Only work with voxels inside mask that are above threshold, and have not been labelled previously
								if ( (Mask[Calculate3DIndex(x2,y2,z2,DATA_W,DATA_H)] == 1.0f) && (Data[Calculate3DIndex(x2,y2,z2,DATA_W,DATA_H)] > Threshold ) && (ccMask[Calculate3DIndex(x2,y2,z2,DATA_W,DATA_H)] == 0) )
								{
									// Add voxel to current cluster
									cluster.push_back(neighbours[j]);

									// Mark voxel as labelled
									ccMask[Calculate3DIndex(x2,y2,z2,DATA_W,DATA_H)] = 1;
								}
							}
						}
					}
				}
			}
		}
	}

	MAX_CLUSTER_SIZE = 0;
	MAX_CLUSTER_MASS = 0.0f;
	NUMBER_OF_CLUSTERS = clusters.size();

	//Cluster_Sizes = (int*)malloc(NUMBER_OF_CLUSTERS * sizeof(int));

	// Put labels into volume
	for (int cluster = 0; cluster < NUMBER_OF_CLUSTERS; cluster++)
	{
		// Get cluster size of current cluster
		int cluster_size = clusters[cluster].size();
		//Cluster_Sizes[cluster] = cluster_size;

		if (cluster_size > MAX_CLUSTER_SIZE)
		{
			MAX_CLUSTER_SIZE = cluster_size;
		}

		// Put cluster labels into a volume
		if (GET_VOXEL_LABELS == 1)
		{
			for (int voxel = 0; voxel < cluster_size; voxel++)
			{
				// Get coordinates of current voxel in current cluster
				int x = clusters[cluster][voxel][X];
				int y = clusters[cluster][voxel][Y];
				int z = clusters[cluster][voxel][Z];

				Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = cluster + 1;
			}
		}

		// Calculate mass of cluster
		if (GET_CLUSTER_MASS == 1)
		{
			float cluster_mass = 0.0f;
			for (int voxel = 0; voxel < cluster_size; voxel++)
			{
				// Get coordinates of current voxel in current cluster
				int x = clusters[cluster][voxel][X];
				int y = clusters[cluster][voxel][Y];
				int z = clusters[cluster][voxel][Z];

				cluster_mass += Data[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];
			}

			if (cluster_mass > MAX_CLUSTER_MASS)
			{
				MAX_CLUSTER_MASS = cluster_mass;
			}
		}
	}

	// Cleanup
	free(ccMask);
}


// Parallel version of clustering
void BROCCOLI_LIB::ClusterizeOpenCL(cl_mem d_Cluster_Indices,
									cl_mem d_Cluster_Sizes,
		                            float& MAX_CLUSTER,
		                            cl_mem d_Data,
		                            float Threshold,
		                            cl_mem d_Mask,
		                            int DATA_W,
		                            int DATA_H,
		                            int DATA_D)
{
	SetGlobalAndLocalWorkSizesClusterize(DATA_W, DATA_H, DATA_D);

	cl_mem d_Largest_Cluster = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	cl_mem d_Updated = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);

	clSetKernelArg(SetStartClusterIndicesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(SetStartClusterIndicesKernel, 1, sizeof(cl_mem), &d_Data);
	clSetKernelArg(SetStartClusterIndicesKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(SetStartClusterIndicesKernel, 3, sizeof(float),  &Threshold);
	clSetKernelArg(SetStartClusterIndicesKernel, 4, sizeof(int),    &DATA_W);
	clSetKernelArg(SetStartClusterIndicesKernel, 5, sizeof(int),    &DATA_H);
	clSetKernelArg(SetStartClusterIndicesKernel, 6, sizeof(int),    &DATA_D);

	clSetKernelArg(ClusterizeScanKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeScanKernel, 1, sizeof(cl_mem), &d_Updated);
	clSetKernelArg(ClusterizeScanKernel, 2, sizeof(cl_mem), &d_Data);
	clSetKernelArg(ClusterizeScanKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(ClusterizeScanKernel, 4, sizeof(float),  &Threshold);
	clSetKernelArg(ClusterizeScanKernel, 5, sizeof(int),    &DATA_W);
	clSetKernelArg(ClusterizeScanKernel, 6, sizeof(int),    &DATA_H);
	clSetKernelArg(ClusterizeScanKernel, 7, sizeof(int),    &DATA_D);

	clSetKernelArg(ClusterizeRelabelKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeRelabelKernel, 1, sizeof(cl_mem), &d_Data);
	clSetKernelArg(ClusterizeRelabelKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(ClusterizeRelabelKernel, 3, sizeof(float),  &Threshold);
	clSetKernelArg(ClusterizeRelabelKernel, 4, sizeof(int),    &DATA_W);
	clSetKernelArg(ClusterizeRelabelKernel, 5, sizeof(int),    &DATA_H);
	clSetKernelArg(ClusterizeRelabelKernel, 6, sizeof(int),    &DATA_D);

	clSetKernelArg(CalculateClusterSizesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(CalculateClusterSizesKernel, 1, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateClusterSizesKernel, 2, sizeof(cl_mem), &d_Data);
	clSetKernelArg(CalculateClusterSizesKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateClusterSizesKernel, 4, sizeof(float),  &Threshold);
	clSetKernelArg(CalculateClusterSizesKernel, 5, sizeof(int),    &DATA_W);
	clSetKernelArg(CalculateClusterSizesKernel, 6, sizeof(int),    &DATA_H);
	clSetKernelArg(CalculateClusterSizesKernel, 7, sizeof(int),    &DATA_D);

	clSetKernelArg(CalculateClusterMassesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(CalculateClusterMassesKernel, 1, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateClusterMassesKernel, 2, sizeof(cl_mem), &d_Data);
	clSetKernelArg(CalculateClusterMassesKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateClusterMassesKernel, 4, sizeof(float),  &Threshold);
	clSetKernelArg(CalculateClusterMassesKernel, 5, sizeof(int),    &DATA_W);
	clSetKernelArg(CalculateClusterMassesKernel, 6, sizeof(int),    &DATA_H);
	clSetKernelArg(CalculateClusterMassesKernel, 7, sizeof(int),    &DATA_D);

	clSetKernelArg(CalculateLargestClusterKernel, 0, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateLargestClusterKernel, 1, sizeof(cl_mem), &d_Largest_Cluster);
	clSetKernelArg(CalculateLargestClusterKernel, 2, sizeof(int),    &DATA_W);
	clSetKernelArg(CalculateLargestClusterKernel, 3, sizeof(int),    &DATA_H);
	clSetKernelArg(CalculateLargestClusterKernel, 4, sizeof(int),    &DATA_D);

	// Set initial cluster indices, voxel 0 = 0, voxel 1 = 1 and so on
	runKernelErrorClusterizeScan = clEnqueueNDRangeKernel(commandQueue, SetStartClusterIndicesKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
	clFinish(commandQueue);

	// Loop until no more updates are done
	float UPDATED = 1.0f;
	while (UPDATED == 1.0f)
	{
		// Set updated to 0
		SetMemory(d_Updated, 0.0f, 1);

		// Run the clustering
		runKernelErrorClusterizeScan = clEnqueueNDRangeKernel(commandQueue, ClusterizeScanKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);
		runKernelErrorClusterizeRelabel = clEnqueueNDRangeKernel(commandQueue, ClusterizeRelabelKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);

		// Copy update parameter to host
		clEnqueueReadBuffer(commandQueue, d_Updated, CL_TRUE, 0, sizeof(float), &UPDATED, 0, NULL, NULL);
	}

	SetMemoryInt(d_Largest_Cluster, -100, 1);
	SetMemoryInt(d_Cluster_Sizes, 0, DATA_W * DATA_H * DATA_D);

	// Calculate the extent of each cluster
	if (INFERENCE_MODE == CLUSTER_EXTENT)
	{
		runKernelErrorCalculateClusterSizes = clEnqueueNDRangeKernel(commandQueue, CalculateClusterSizesKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);
	}
	// Calculate the mass of each cluster
	else if (INFERENCE_MODE == CLUSTER_MASS)
	{
		runKernelErrorCalculateClusterMasses = clEnqueueNDRangeKernel(commandQueue, CalculateClusterMassesKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	// Calculate size of largest cluster (extent or mass)
	runKernelErrorCalculateLargestCluster = clEnqueueNDRangeKernel(commandQueue, CalculateLargestClusterKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
	clFinish(commandQueue);

	// Copy largest cluster to host
	unsigned int Largest_Cluster;
	clEnqueueReadBuffer(commandQueue, d_Largest_Cluster, CL_TRUE, 0, sizeof(unsigned int), &Largest_Cluster, 0, NULL, NULL);

	if (INFERENCE_MODE == CLUSTER_EXTENT)
	{
		MAX_CLUSTER = (float)Largest_Cluster;
	}
	else if (INFERENCE_MODE == CLUSTER_MASS)
	{
		MAX_CLUSTER = (float)Largest_Cluster/10000.0f;
	}

	clReleaseMemObject(d_Updated);
	clReleaseMemObject(d_Largest_Cluster);
}

// Parallel clustering, optimized for permutation (for example, does not allocate or free memory in each permutation)
void BROCCOLI_LIB::ClusterizeOpenCLPermutation(float& MAX_CLUSTER, int DATA_W, int DATA_H, int DATA_D)
{
	// Set initial cluster indices, voxel 0 = 0, voxel 1 = 1 and so on
	runKernelErrorClusterizeScan = clEnqueueNDRangeKernel(commandQueue, SetStartClusterIndicesKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
	clFinish(commandQueue);

	// Loop until no more updates are done
	float UPDATED = 1.0f;
	while (UPDATED == 1.0f)
	{
		// Set updated to 0
		SetMemory(d_Updated, 0.0f, 1);

		// Run the clustering
		runKernelErrorClusterizeScan = clEnqueueNDRangeKernel(commandQueue, ClusterizeScanKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);
		runKernelErrorClusterizeRelabel = clEnqueueNDRangeKernel(commandQueue, ClusterizeRelabelKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);

		// Copy update parameter to host
		clEnqueueReadBuffer(commandQueue, d_Updated, CL_TRUE, 0, sizeof(float), &UPDATED, 0, NULL, NULL);
	}

	SetMemoryInt(d_Largest_Cluster, 0, 1);
	SetMemoryInt(d_Cluster_Sizes, 0, DATA_W * DATA_H * DATA_D);

	// Calculate the extent of each cluster
	if (INFERENCE_MODE == CLUSTER_EXTENT)
	{
		runKernelErrorCalculateClusterSizes = clEnqueueNDRangeKernel(commandQueue, CalculateClusterSizesKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);
	}
	// Calculate the mass of each cluster
	else if (INFERENCE_MODE == CLUSTER_MASS)
	{
		runKernelErrorCalculateClusterMasses = clEnqueueNDRangeKernel(commandQueue, CalculateClusterMassesKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	// Calculate size of largest cluster (extent or mass)
	runKernelErrorCalculateLargestCluster = clEnqueueNDRangeKernel(commandQueue, CalculateLargestClusterKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
	clFinish(commandQueue);

	// Copy largest cluster to host
	unsigned int Largest_Cluster;
	clEnqueueReadBuffer(commandQueue, d_Largest_Cluster, CL_TRUE, 0, sizeof(unsigned int), &Largest_Cluster, 0, NULL, NULL);

	if (INFERENCE_MODE == CLUSTER_EXTENT)
	{
		MAX_CLUSTER = (float)Largest_Cluster;
	}
	else if (INFERENCE_MODE == CLUSTER_MASS)
	{
		MAX_CLUSTER = (float)Largest_Cluster/10000.0f;
	}
}


void BROCCOLI_LIB::ClusterizeOpenCLTFCEPermutation(float& MAX_VALUE, cl_mem d_Mask, int DATA_W, int DATA_H, int DATA_D, float maxThreshold, float delta)
{
	// Reset TFCE values
	SetMemory(d_TFCE_Values, 0.0f, DATA_W * DATA_H * DATA_D);

	// Loop over thresholds
	for (float threshold = 0.0f; threshold <= maxThreshold; threshold += delta)
	{
		// Set new threshold for kernels
		clSetKernelArg(SetStartClusterIndicesKernel, 3, sizeof(float),  &threshold);
		clSetKernelArg(ClusterizeScanKernel, 4, sizeof(float), &threshold);
		clSetKernelArg(ClusterizeRelabelKernel, 3, sizeof(float),  &threshold);
		clSetKernelArg(CalculateClusterSizesKernel, 4, sizeof(float),  &threshold);
		clSetKernelArg(CalculateTFCEValuesKernel, 2, sizeof(float),  &threshold);

		// Set initial cluster indices, voxel 0 = 0, voxel 1 = 1 and so on
		runKernelErrorClusterizeScan = clEnqueueNDRangeKernel(commandQueue, SetStartClusterIndicesKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);

		// Loop until no more updates are done
		float UPDATED = 1.0f;
		while (UPDATED == 1.0f)
		{
			// Set updated to 0
			SetMemory(d_Updated, 0.0f, 1);

			// Run the clustering
			runKernelErrorClusterizeScan = clEnqueueNDRangeKernel(commandQueue, ClusterizeScanKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
			clFinish(commandQueue);
			runKernelErrorClusterizeRelabel = clEnqueueNDRangeKernel(commandQueue, ClusterizeRelabelKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
			clFinish(commandQueue);

			// Copy update parameter to host
			clEnqueueReadBuffer(commandQueue, d_Updated, CL_TRUE, 0, sizeof(float), &UPDATED, 0, NULL, NULL);
		}

		// Reset cluster sizes
		SetMemoryInt(d_Cluster_Sizes, 0, DATA_W * DATA_H * DATA_D);

		// Calculate the extent of each cluster
		runKernelErrorCalculateClusterSizes = clEnqueueNDRangeKernel(commandQueue, CalculateClusterSizesKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);

		// Calculate TFCE contributions for this threshold
		runKernelErrorCalculateTFCEValues = clEnqueueNDRangeKernel(commandQueue, CalculateTFCEValuesKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	// Find max TFCE value
	MAX_VALUE = CalculateMaxAtomic(d_TFCE_Values, d_Mask, DATA_W, DATA_H, DATA_D);
}


// Parallel version of clustering, using texture memory
void BROCCOLI_LIB::ClusterizeOpenCLWrapper()
{
	/*
	cl_mem d_Mask = clCreateBuffer(context, CL_MEM_READ_ONLY, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Data = clCreateBuffer(context, CL_MEM_READ_ONLY, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Cluster_Indices = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_First_Level_Results, 0, NULL, NULL);


	SetGlobalAndLocalWorkSizesClusterize(MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	// Create a 3D image (texture) for fast access of neighbouring indices
	cl_image_format format;
	format.image_channel_data_type = CL_SIGNED_INT32;
	format.image_channel_order = CL_INTENSITY;
	cl_mem d_Cluster_Indices_Texture = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 0, 0, NULL, NULL);

	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {MNI_DATA_W, MNI_DATA_H, MNI_DATA_D};

	cl_mem d_Updated = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
	cl_mem d_Current_Cluster = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

	SetMemoryInt(d_Current_Cluster, 0, 1);

	for (int i = 0; i < 100; i++)
	{
	clSetKernelArg(SetStartClusterIndicesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(SetStartClusterIndicesKernel, 1, sizeof(cl_mem), &d_Data);
	clSetKernelArg(SetStartClusterIndicesKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(SetStartClusterIndicesKernel, 3, sizeof(cl_mem), &d_Current_Cluster);
	clSetKernelArg(SetStartClusterIndicesKernel, 4, sizeof(float), &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(SetStartClusterIndicesKernel, 5, sizeof(int), &MNI_DATA_W);
	clSetKernelArg(SetStartClusterIndicesKernel, 6, sizeof(int), &MNI_DATA_H);
	clSetKernelArg(SetStartClusterIndicesKernel, 7, sizeof(int), &MNI_DATA_D);
	runKernelErrorClusterizeScan = clEnqueueNDRangeKernel(commandQueue, SetStartClusterIndicesKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
	clFinish(commandQueue);

	clSetKernelArg(ClusterizeKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeKernel, 1, sizeof(cl_mem), &d_Cluster_Indices_Texture);
	clSetKernelArg(ClusterizeKernel, 2, sizeof(cl_mem), &d_Updated);
	clSetKernelArg(ClusterizeKernel, 3, sizeof(cl_mem), &d_Data);
	clSetKernelArg(ClusterizeKernel, 4, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(ClusterizeKernel, 5, sizeof(float), &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(ClusterizeKernel, 6, sizeof(int), &MNI_DATA_W);
	clSetKernelArg(ClusterizeKernel, 7, sizeof(int), &MNI_DATA_H);
	clSetKernelArg(ClusterizeKernel, 8, sizeof(int), &MNI_DATA_D);


	float UPDATED = 1.0f;
	while (UPDATED == 1.0f)
	{
		// Copy the current cluster indices to a texture, for fast spatial access
		clEnqueueCopyBufferToImage(commandQueue, d_Cluster_Indices, d_Cluster_Indices_Texture, 0, origin, region, 0, NULL, NULL);
		// Set updated to 0
		SetMemory(d_Updated, 0.0f, 1);
		// Run the clustering
		runKernelErrorClusterize = clEnqueueNDRangeKernel(commandQueue, ClusterizeKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);
		// Copy update parameter to host
		clEnqueueReadBuffer(commandQueue, d_Updated, CL_TRUE, 0, sizeof(float), &UPDATED, 0, NULL, NULL);
	}
	}

	clEnqueueReadBuffer(commandQueue, d_Cluster_Indices, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), h_Cluster_Indices, 0, NULL, NULL);

	clReleaseMemObject(d_Cluster_Indices_Texture);
	clReleaseMemObject(d_Updated);
	clReleaseMemObject(d_Current_Cluster);

	clReleaseMemObject(d_Mask);
	clReleaseMemObject(d_Data);
	clReleaseMemObject(d_Cluster_Indices);
	*/
}


void BROCCOLI_LIB::ClusterizeOpenCLWrapper2()
{
	cl_mem d_Mask = clCreateBuffer(context, CL_MEM_READ_ONLY, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Data = clCreateBuffer(context, CL_MEM_READ_ONLY, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Cluster_Indices = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), NULL, NULL);
	d_Cluster_Sizes = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), NULL, NULL);
	d_Largest_Cluster = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_First_Level_Results, 0, NULL, NULL);

	SetGlobalAndLocalWorkSizesClusterize(MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	d_Updated = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);

	clSetKernelArg(ClusterizeScanKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeScanKernel, 1, sizeof(cl_mem), &d_Updated);
	clSetKernelArg(ClusterizeScanKernel, 2, sizeof(cl_mem), &d_Data);
	clSetKernelArg(ClusterizeScanKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(ClusterizeScanKernel, 4, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(ClusterizeScanKernel, 5, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(ClusterizeScanKernel, 6, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(ClusterizeScanKernel, 7, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(ClusterizeRelabelKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeRelabelKernel, 1, sizeof(cl_mem), &d_Data);
	clSetKernelArg(ClusterizeRelabelKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(ClusterizeRelabelKernel, 3, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(ClusterizeRelabelKernel, 4, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(ClusterizeRelabelKernel, 5, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(ClusterizeRelabelKernel, 6, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(CalculateClusterSizesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(CalculateClusterSizesKernel, 1, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateClusterSizesKernel, 2, sizeof(cl_mem), &d_Data);
	clSetKernelArg(CalculateClusterSizesKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateClusterSizesKernel, 4, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(CalculateClusterSizesKernel, 5, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateClusterSizesKernel, 6, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateClusterSizesKernel, 7, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(CalculateLargestClusterKernel, 0, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateLargestClusterKernel, 1, sizeof(cl_mem), &d_Largest_Cluster);
	clSetKernelArg(CalculateLargestClusterKernel, 2, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateLargestClusterKernel, 3, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateLargestClusterKernel, 4, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(SetStartClusterIndicesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(SetStartClusterIndicesKernel, 1, sizeof(cl_mem), &d_Data);
	clSetKernelArg(SetStartClusterIndicesKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(SetStartClusterIndicesKernel, 3, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(SetStartClusterIndicesKernel, 4, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(SetStartClusterIndicesKernel, 5, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(SetStartClusterIndicesKernel, 6, sizeof(int),    &MNI_DATA_D);

	for (int i = 0; i < 1000; i++)
	{



	runKernelErrorClusterizeScan = clEnqueueNDRangeKernel(commandQueue, SetStartClusterIndicesKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
	clFinish(commandQueue);



	float UPDATED = 1.0f;
	while (UPDATED == 1.0f)
	{
		// Set updated to 0
		SetMemory(d_Updated, 0.0f, 1);
		// Run the clustering
		runKernelErrorClusterizeScan = clEnqueueNDRangeKernel(commandQueue, ClusterizeScanKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);
		runKernelErrorClusterizeRelabel = clEnqueueNDRangeKernel(commandQueue, ClusterizeRelabelKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
		clFinish(commandQueue);

		// Copy update parameter to host
		clEnqueueReadBuffer(commandQueue, d_Updated, CL_TRUE, 0, sizeof(float), &UPDATED, 0, NULL, NULL);
	}

	SetMemoryInt(d_Largest_Cluster, -100, 1);
	SetMemoryInt(d_Cluster_Sizes, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D);

	runKernelErrorCalculateClusterSizes = clEnqueueNDRangeKernel(commandQueue, CalculateClusterSizesKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
	clFinish(commandQueue);
	runKernelErrorCalculateLargestCluster = clEnqueueNDRangeKernel(commandQueue, CalculateLargestClusterKernel, 3, NULL, globalWorkSizeClusterize, localWorkSizeClusterize, 0, NULL, NULL);
	clFinish(commandQueue);
	// Copy largest cluster to host
	clEnqueueReadBuffer(commandQueue, d_Largest_Cluster, CL_TRUE, 0, sizeof(int), h_Largest_Cluster, 0, NULL, NULL);

	}

	clEnqueueReadBuffer(commandQueue, d_Cluster_Indices, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), h_Cluster_Indices, 0, NULL, NULL);

	clEnqueueReadBuffer(commandQueue, d_Cluster_Sizes, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(int), h_Cluster_Indices, 0, NULL, NULL);

	clReleaseMemObject(d_Updated);

	clReleaseMemObject(d_Mask);
	clReleaseMemObject(d_Data);
	clReleaseMemObject(d_Cluster_Indices);

	clReleaseMemObject(d_Cluster_Sizes);
	clReleaseMemObject(d_Largest_Cluster);
}

void BROCCOLI_LIB::ClusterizeOpenCLWrapper3()
{
	cl_mem d_Data = clCreateBuffer(context, CL_MEM_READ_ONLY, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	float* h_Data = (float*)malloc(MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float));

	for (int i = 0; i < 1000; i++)
	{
		clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Data, 0, NULL, NULL);
		clFinish(commandQueue);
		//Clusterize(h_Cluster_Indices, MAX_CLUSTER_SIZE, MAX_CLUSTER_MASS, NUMBER_OF_CLUSTERS, h_First_Level_Results, CLUSTER_DEFINING_THRESHOLD, h_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, CALCULATE_VOXEL_LABELS, DONT_CALCULATE_CLUSTER_MASS);
		//Clusterize(h_Cluster_Indices, MAX_CLUSTER_SIZE, MAX_CLUSTER_MASS, NUMBER_OF_CLUSTERS, h_First_Level_Results, CLUSTER_DEFINING_THRESHOLD, h_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, DONT_CALCULATE_VOXEL_LABELS, DONT_CALCULATE_CLUSTER_MASS);
	}

	clReleaseMemObject(d_Data);
	free(h_Data);
}

// Small help functions

int BROCCOLI_LIB::CalculateMax(int *data, int N)
{
    int max = std::numeric_limits<int>::min();
	for (int i = 0; i < N; i++)
	{
	    if (data[i] > max)
		{
			max = data[i];
		}
	}
	return max;
}

float BROCCOLI_LIB::CalculateMax(float *data, int N)
{
    float max = std::numeric_limits<float>::min();
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
    float min = std::numeric_limits<float>::max();
	for (int i = 0; i < N; i++)
	{
	    if (data[i] < min)
		{
			min = data[i];
		}
	}
	return min;
}




