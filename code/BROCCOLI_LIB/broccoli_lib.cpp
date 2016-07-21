/*
    BROCCOLI: Software for Fast fMRI Analysis on Many-Core CPUs and GPUs
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

#include <Eigen/Eigenvalues> 
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

#include <time.h>
#include <sys/time.h>

#include <limits.h>
//#include <unistd.h>

#include <opencl.h>

#include <clBLAS.h>

#include "broccoli_lib.h"

#include <cstdlib>

#define EIGEN_DONT_PARALLELIZE

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

double GetTime()
{
    struct timeval time;
    if (gettimeofday(&time,NULL))
    {
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

float myround(float a)
{
	return floor(a + 0.5f);
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
	OPENCL_INITIATED = false;
	SetStartValues();
}

BROCCOLI_LIB::BROCCOLI_LIB(cl_uint platform, cl_uint device)
{
	SetStartValues();
	OPENCL_INITIATED = false;
	SUCCESSFUL_INITIALIZATION = OpenCLInitiate(platform,device);
}

BROCCOLI_LIB::BROCCOLI_LIB(cl_uint platform, cl_uint device, int wrapper, bool verbos)
{
	SetStartValues();
	WRAPPER = wrapper;
	VERBOS = verbos;
	OPENCL_INITIATED = false;
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

void BROCCOLI_LIB::SetVerbose(bool verbos)
{
	VERBOS = verbos;
}

void BROCCOLI_LIB::SetAllocatedHostMemory(unsigned long int allocated)
{
	allocatedHostMemory = allocated;
}

void BROCCOLI_LIB::SetDoAllPermutations(bool doall)
{
	DO_ALL_PERMUTATIONS = doall;
}

void BROCCOLI_LIB::SetRawRegressors(bool raw)
{
	RAW_REGRESSORS = raw;
}

void BROCCOLI_LIB::SetRawDesignMatrix(bool raw)
{
	RAW_DESIGNMATRIX = raw;
}

void BROCCOLI_LIB::SetCustomReferenceSlice(int slice)
{
	SLICE_CUSTOM_REF = slice;
}

void BROCCOLI_LIB::SetDoSkullstrip(bool doskullstrip)
{
	DO_SKULLSTRIP = doskullstrip;
}

void BROCCOLI_LIB::SetDoSkullstripOriginal(bool doskullstriporiginal)
{
	DO_SKULLSTRIP_ORIGINAL = doskullstriporiginal;
}

void BROCCOLI_LIB::SetCustomSliceTimes(float* times)
{
	h_Custom_Slice_Times = times;
}

void BROCCOLI_LIB::SetWrapper(int wrapper)
{
	WRAPPER = wrapper;
}

void BROCCOLI_LIB::SetApplySliceTimingCorrection(bool value)
{
	APPLY_SLICE_TIMING_CORRECTION = value;
}

void BROCCOLI_LIB::SetApplyMotionCorrection(bool value)
{
	APPLY_MOTION_CORRECTION = value;
}

void BROCCOLI_LIB::SetApplySmoothing(bool value)
{
	APPLY_SMOOTHING = value;
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

	localMemorySize = 0;
	maxThreadsPerBlock = 0;
	maxThreadsPerDimension[0] = 0;
	maxThreadsPerDimension[1] = 0;
	maxThreadsPerDimension[2] = 0;

	

	DEBUG = false;
	WRAPPER = -1;
	PRINT = true;
	VERBOS = false;
	DO_ALL_PERMUTATIONS = false;

	APPLY_SLICE_TIMING_CORRECTION = true;
	APPLY_MOTION_CORRECTION = true;
	APPLY_SMOOTHING = true;

	WRITE_INTERPOLATED_T1 = false;
	WRITE_ALIGNED_T1_MNI_LINEAR = false;
	WRITE_ALIGNED_T1_MNI_NONLINEAR = false;
	DO_SKULLSTRIP = false;

	WRITE_ALIGNED_EPI_T1 = false;
	WRITE_ALIGNED_EPI_MNI = false;

	WRITE_EPI_MASK = false;
	WRITE_MNI_MASK = false;
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
	AUTO_MASK = false;

	programBinarySize = 0;
	writtenElements = 0;

	BETA_SPACE = EPI;

	SLICE_ORDER = UNDEFINED;
	SLICE_CUSTOM_REF = 0;

	FILE_TYPE = RAW;
	DATA_TYPE = FLOAT;

	NUMBER_OF_RUNS = 1;

	EPI_DATA_W = 64;
	EPI_DATA_H = 64;
	EPI_DATA_D = 30;
	EPI_DATA_T = 100;

	EPI_VOXEL_SIZE_X = 3.00f;
	EPI_VOXEL_SIZE_Y = 3.00f;
	EPI_VOXEL_SIZE_Z = 3.00f;
	TR = 2.0f;
	
	NUMBER_OF_PERMUTATIONS = 1000;
	SIGNIFICANCE_LEVEL = 0.05f;
	SIGNIFICANCE_THRESHOLD = 0;
	STATISTICAL_TEST = 0;

	IMAGE_REGISTRATION_FILTER_SIZE = 7;
	NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION = 10;
	NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION = 10;
	NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS = 30;
	CHANGE_MOTION_CORRECTION_REFERENCE_VOLUME = false;

	SMOOTHING_FILTER_SIZE = 9;
	
	NUMBER_OF_DETRENDING_REGRESSORS = 4;
	NUMBER_OF_MOTION_REGRESSORS = 6;

    RAW_REGRESSORS = false;
    RAW_DESIGNMATRIX = false;
	BAYESIAN = false;
	REGRESS_ONLY = false;
	PREPROCESSING_ONLY = false;
	BETAS_ONLY = false;
	REGRESS_MOTION = 0;
	REGRESS_GLOBALMEAN = 0;
	REGRESS_CONFOUNDS = 0;
	PERMUTE_FIRST_LEVEL = false;
	USE_PERMUTATION_FILE = false;

	Z_SCORE = false;
	PROPORTION_OF_VARIANCE_TO_SAVE_BEFORE_ICA = 80.0f;

	NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS = 12;

	TSIGMA = 5.0;
	ESIGMA = 5.0;
	DSIGMA = 5.0;

	convolution_time = 0.0;

	error = 0;

	NUMBER_OF_OPENCL_KERNELS = 102;

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
    createKernelErrorAddLinearAndNonLinearDisplacement = 0;
    
    createKernelErrorCalculateMagnitudes = 0;
    createKernelErrorCalculateColumnSums = 0;
    createKernelErrorCalculateRowSums = 0;
    createKernelErrorCalculateColumnMaxs = 0;
    createKernelErrorCalculateRowMaxs = 0;
    createKernelErrorCalculateMaxAtomic = 0;
    createKernelErrorThresholdVolume = 0;
    createKernelErrorMemset = 0;
    createKernelErrorMemsetDouble = 0;
    createKernelErrorMemsetInt = 0;
    createKernelErrorMemsetFloat2 = 0;
    createKernelErrorIdentityMatrix = 0;
    createKernelErrorIdentityMatrixDouble = 0;
    createKernelErrorGetSubMatrix = 0;
    createKernelErrorGetSubMatrixDouble = 0;
    createKernelErrorPermuteMatrix = 0;
    createKernelErrorPermuteMatrixDouble = 0;
    createKernelErrorLogitMatrix = 0;
    createKernelErrorLogitMatrixDouble = 0;
    createKernelErrorMultiplyVolume = 0;
    createKernelErrorMultiplyVolumes = 0;
    createKernelErrorMultiplyVolumesOverwrite = 0;
    createKernelErrorMultiplyVolumesOverwriteDouble = 0;
    createKernelErrorAddVolume = 0;
    createKernelErrorAddVolumes = 0;
    createKernelErrorAddVolumesOverwrite = 0;
    createKernelErrorSubtractVolumes = 0;
    createKernelErrorSubtractVolumesOverwrite = 0;
    createKernelErrorSubtractVolumesOverwriteDouble = 0;
    createKernelErrorRemoveMean = 0;
    
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
    
    createKernelErrorSetStartClusterIndices = 0;
    createKernelErrorClusterizeScan = 0;
    createKernelErrorClusterizeRelabel = 0;
    createKernelErrorCalculateClusterSizes = 0;
    createKernelErrorCalculateClusterMasses = 0;
    createKernelErrorCalculateLargestCluster = 0;
    createKernelErrorCalculateTFCEValues = 0;
    createKernelErrorCalculatePermutationPValuesVoxelLevelInference = 0;
    createKernelErrorCalculatePermutationPValuesClusterExtentInference = 0;
    createKernelErrorCalculatePermutationPValuesClusterMassInference = 0;
    
    createKernelErrorCalculateBetaWeightsGLM = 0;
    createKernelErrorCalculateBetaWeightsGLMSlice = 0;
    createKernelErrorCalculateBetaWeightsAndContrastsGLM = 0;
    createKernelErrorCalculateBetaWeightsAndContrastsGLMSlice = 0;
    createKernelErrorCalculateBetaWeightsGLMFirstLevel = 0;
    createKernelErrorCalculateBetaWeightsGLMFirstLevelSlice = 0;
    createKernelErrorCalculateGLMResiduals = 0;
    createKernelErrorCalculateGLMResidualsSlice = 0;
    createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel = 0;
    createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevel = 0;
    createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelSlice = 0;
    createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelSlice = 0;
    createKernelErrorCalculateStatisticalMapsGLMTTest = 0;
    createKernelErrorCalculateStatisticalMapsGLMFTest = 0;
    createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelPermutation = 0;
    createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelPermutation = 0;
    createKernelErrorCalculateStatisticalMapsGLMTTestSecondLevelPermutation = 0;
    createKernelErrorCalculateStatisticalMapsGLMFTestSecondLevelPermutation = 0;
    createKernelErrorCalculateStatisticalMapsMeanSecondLevelPermutation = 0;
    createKernelErrorCalculateStatisticalMapSearchlight = 0;
    createKernelErrorTransformData = 0;
    createKernelErrorRemoveLinearFit = 0;
    createKernelErrorRemoveLinearFitSlice = 0;
    
    createKernelErrorCalculateStatisticalMapsGLMBayesian = 0;
    
    createKernelErrorEstimateAR4Models = 0;
    createKernelErrorEstimateAR4ModelsSlice = 0;
    createKernelErrorApplyWhiteningAR4 = 0;
    createKernelErrorApplyWhiteningAR4Slice = 0;
    createKernelErrorGeneratePermutedVolumesFirstLevel = 0;
    
    
    
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
    runKernelErrorAddLinearAndNonLinearDisplacement = 0;
    
    runKernelErrorCalculateMagnitudes = 0;
    runKernelErrorCalculateColumnSums = 0;
    runKernelErrorCalculateRowSums = 0;
    runKernelErrorCalculateColumnMaxs = 0;
    runKernelErrorCalculateRowMaxs = 0;
    runKernelErrorCalculateMaxAtomic = 0;
    runKernelErrorThresholdVolume = 0;
    runKernelErrorMemset = 0;
    runKernelErrorMemsetDouble = 0;
    runKernelErrorMemsetInt = 0;
    runKernelErrorMemsetFloat2 = 0;
    runKernelErrorIdentityMatrix = 0;
    runKernelErrorIdentityMatrixDouble = 0;
    runKernelErrorGetSubMatrix = 0;
    runKernelErrorGetSubMatrixDouble = 0;
    runKernelErrorPermuteMatrix = 0;
    runKernelErrorPermuteMatrixDouble = 0;
    runKernelErrorLogitMatrix = 0;
    runKernelErrorLogitMatrixDouble = 0;
    runKernelErrorMultiplyVolume = 0;
    runKernelErrorMultiplyVolumes = 0;
    runKernelErrorMultiplyVolumesOverwrite = 0;
    runKernelErrorMultiplyVolumesOverwriteDouble = 0;
    runKernelErrorAddVolume = 0;
    runKernelErrorAddVolumes = 0;
    runKernelErrorAddVolumesOverwrite = 0;
    runKernelErrorSubtractVolumes = 0;
    runKernelErrorSubtractVolumesOverwrite = 0;
    runKernelErrorSubtractVolumesOverwriteDouble = 0;
    runKernelErrorRemoveMean = 0;
    
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
    
    runKernelErrorSetStartClusterIndices = 0;
    runKernelErrorClusterizeScan = 0;
    runKernelErrorClusterizeRelabel = 0;
    runKernelErrorCalculateClusterSizes = 0;
    runKernelErrorCalculateClusterMasses = 0;
    runKernelErrorCalculateLargestCluster = 0;
    runKernelErrorCalculateTFCEValues = 0;
    runKernelErrorCalculatePermutationPValuesVoxelLevelInference = 0;
    runKernelErrorCalculatePermutationPValuesClusterExtentInference = 0;
    runKernelErrorCalculatePermutationPValuesClusterMassInference = 0;
    
    runKernelErrorCalculateBetaWeightsGLM = 0;
    runKernelErrorCalculateBetaWeightsGLMSlice = 0;
    runKernelErrorCalculateBetaWeightsAndContrastsGLM = 0;
    runKernelErrorCalculateBetaWeightsAndContrastsGLMSlice = 0;
    runKernelErrorCalculateBetaWeightsGLMFirstLevel = 0;
    runKernelErrorCalculateBetaWeightsGLMFirstLevelSlice = 0;
    runKernelErrorCalculateGLMResiduals = 0;
    runKernelErrorCalculateGLMResidualsSlice = 0;
    runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel = 0;
    runKernelErrorCalculateStatisticalMapsGLMFTestFirstLevel = 0;
    runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelSlice = 0;
    runKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelSlice = 0;
    runKernelErrorCalculateStatisticalMapsGLMTTest = 0;
    runKernelErrorCalculateStatisticalMapsGLMFTest = 0;
    runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelPermutation = 0;
    runKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelPermutation = 0;
    runKernelErrorCalculateStatisticalMapsGLMTTestSecondLevelPermutation = 0;
    runKernelErrorCalculateStatisticalMapsGLMFTestSecondLevelPermutation = 0;
    runKernelErrorCalculateStatisticalMapsMeanSecondLevelPermutation = 0;
    runKernelErrorCalculateStatisticalMapSearchlight = 0;
    runKernelErrorTransformData = 0;
    runKernelErrorRemoveLinearFit = 0;
    runKernelErrorRemoveLinearFitSlice = 0;
    
    runKernelErrorCalculateStatisticalMapsGLMBayesian = 0;
    
    runKernelErrorEstimateAR4Models = 0;
    runKernelErrorEstimateAR4ModelsSlice = 0;
    runKernelErrorApplyWhiteningAR4 = 0;
    runKernelErrorApplyWhiteningAR4Slice = 0;
    runKernelErrorGeneratePermutedVolumesFirstLevel = 0;
    
	getPlatformIDsError = 0;
	getDeviceIDsError = 0;		
	createContextError = 0;
	getContextInfoError = 0;
	createCommandQueueError = 0;
	createProgramError = 0;
	buildProgramError = 0;
	getProgramBuildInfoError = 0;

	NUMBER_OF_KERNEL_FILES = 12;

	for (int k = 0; k < NUMBER_OF_KERNEL_FILES; k++)
	{
		OpenCLPrograms[k] = NULL;
		binaryBuildProgramErrors[k] = FAIL;
		sourceBuildProgramErrors[k] = FAIL;
	}

	kernelFileNames.push_back("kernelConvolution.cpp");
	kernelFileNames.push_back("kernelRegistration.cpp");
	kernelFileNames.push_back("kernelClusterize.cpp");		
	kernelFileNames.push_back("kernelMisc.cpp");
	kernelFileNames.push_back("kernelStatistics1.cpp");
	kernelFileNames.push_back("kernelStatistics2.cpp");
    kernelFileNames.push_back("kernelStatistics3.cpp");
    kernelFileNames.push_back("kernelStatistics4.cpp");
    kernelFileNames.push_back("kernelStatistics5.cpp");
	kernelFileNames.push_back("kernelWhitening.cpp");
	kernelFileNames.push_back("kernelBayesian.cpp");
    kernelFileNames.push_back("kernelSearchlight.cpp");
    
	buildInfo.resize(12);
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
		deviceInfo.append("---------------------------------------------");
		deviceInfo.append("\n");
		deviceInfo.append("Platform number: ");
		temp_stream.str("");
		temp_stream.clear();
		temp_stream << i;
		deviceInfo.append(temp_stream.str());
		deviceInfo.append("\n");
		deviceInfo.append("---------------------------------------------");
		deviceInfo.append("\n");

	    // Get platform vendor
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, 0, NULL, &valueSize);
		value = (char*) malloc(valueSize);
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, valueSize, value, NULL);            
		deviceInfo.append("Platform vendor: ");
		deviceInfo.append(value);
		deviceInfo.append("\n");
		free(value);		

		// Get platform name
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, 0, NULL, &valueSize);
		value = (char*) malloc(valueSize);
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, valueSize, value, NULL);            
		deviceInfo.append("Platform name: ");
		deviceInfo.append(value);
		deviceInfo.append("\n");
		free(value);		

		// Get platform extensions
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &valueSize);
		value = (char*) malloc(valueSize);
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_EXTENSIONS, valueSize, value, NULL);            
		deviceInfo.append("Platform extentions: ");
		deviceInfo.append(value);
		deviceInfo.append("\n");
		free(value);		

		// Get platform profile
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_PROFILE, 0, NULL, &valueSize);
		value = (char*) malloc(valueSize);
		clGetPlatformInfo(platformIds[i], CL_PLATFORM_PROFILE, valueSize, value, NULL);            
		deviceInfo.append("Platform profile: ");
		deviceInfo.append(value);
		deviceInfo.append("\n");
		free(value);

		deviceInfo.append("---------------------------------------------");
		deviceInfo.append("\n");
		deviceInfo.append("\n");

		// Get devices for each platform
		cl_uint deviceIdCount = 0;
		clGetDeviceIDs (platformIds[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceIdCount);
		std::vector<cl_device_id> deviceIds (deviceIdCount);
		clGetDeviceIDs (platformIds[i], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), NULL);

		// Get information for for each device and save as a long string
		for (uint j = 0; j < deviceIdCount; j++) 
		{
			deviceInfo.append("---------------------------------------------");
			deviceInfo.append("\n");
			deviceInfo.append("Device number: ");
			temp_stream.str("");
			temp_stream.clear();
			temp_stream << j;
			deviceInfo.append(temp_stream.str());
			deviceInfo.append("\n");
			deviceInfo.append("---------------------------------------------");
			deviceInfo.append("\n");

	        // Get vendor name
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_VENDOR, 0, NULL, &valueSize);
			value = (char*) malloc(valueSize);
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_VENDOR, valueSize, value, NULL);            
			deviceInfo.append("Device vendor: ");
			deviceInfo.append(value);
			deviceInfo.append("\n");
			free(value);	
        	
			// Get device name
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
			value = (char*) malloc(valueSize);
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_NAME, valueSize, value, NULL);            
			deviceInfo.append("Device name: ");
			deviceInfo.append(value);
			deviceInfo.append("\n");
			free(value);

			// Get hardware device version
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
			value = (char*) malloc(valueSize);
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_VERSION, valueSize, value, NULL);
			deviceInfo.append("Hardware version: ");
			deviceInfo.append(value);
			deviceInfo.append("\n");
			free(value);

			// Get software driver version
			clGetDeviceInfo(deviceIds[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
			value = (char*) malloc(valueSize);
			clGetDeviceInfo(deviceIds[j], CL_DRIVER_VERSION, valueSize, value, NULL);
			deviceInfo.append("Software version: ");
			deviceInfo.append(value);
			deviceInfo.append("\n");
			free(value);

			// Get C version supported by compiler for device
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
			value = (char*) malloc(valueSize);
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
			deviceInfo.append("OpenCL C version: ");
			deviceInfo.append(value);
			deviceInfo.append("\n");
			free(value);
            
			// Get device extensions
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_EXTENSIONS, 0, NULL, &valueSize);
			value = (char*) malloc(valueSize);
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_EXTENSIONS, valueSize, value, NULL);            
			deviceInfo.append("Device extensions: ");
			deviceInfo.append(value);
			deviceInfo.append("\n");
			free(value);

			// Get global memory size
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memorySize), &memorySize, NULL);            
			deviceInfo.append("Global memory size in MB: ");
			temp_stream.str("");
			temp_stream.clear();
			temp_stream << memorySize/ (1024*1024);            
			deviceInfo.append(temp_stream.str());
			deviceInfo.append("\n");

			// Get size of largest memory object that can be allocated
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(memorySize), &memorySize, NULL);            
			deviceInfo.append("Size of largest memory object in MB: ");
			temp_stream.str("");
			temp_stream.clear();
			temp_stream << memorySize/ (1024*1024);            
			deviceInfo.append(temp_stream.str());
			deviceInfo.append("\n");
            
			// Get global memory cache size
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(memorySize), &memorySize, NULL);            
			deviceInfo.append("Global memory cache size in KB: ");
			temp_stream.str("");
			temp_stream.clear();
			temp_stream << memorySize/ (1024);            
			deviceInfo.append(temp_stream.str());
			deviceInfo.append("\n");            			

			// Get local (shared) memory size
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(memorySize), &memorySize, NULL);            
			deviceInfo.append("Local memory size in KB: ");
			temp_stream.str("");
			temp_stream.clear();
			temp_stream << memorySize/1024;            
			deviceInfo.append(temp_stream.str());
			deviceInfo.append("\n");
	            
			// Get constant memory size
		    clGetDeviceInfo(deviceIds[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(memorySize), &memorySize, NULL);            
			deviceInfo.append("Constant memory size in KB: ");
			temp_stream.str("");
			temp_stream.clear();
			temp_stream << memorySize/1024;            
			deviceInfo.append(temp_stream.str());
			deviceInfo.append("\n");                       
            
			// Get parallel compute units
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);            
			deviceInfo.append("Parallel compute units: ");
			temp_stream.str("");
			temp_stream.clear();
			temp_stream << maxComputeUnits;            
			deviceInfo.append(temp_stream.str());
			deviceInfo.append("\n");
            
			// Get clock frequency
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockFrequency), &clockFrequency, NULL);            
			deviceInfo.append("Clock frequency in MHz: ");
			temp_stream.str("");
			temp_stream.clear();
			temp_stream << clockFrequency;            
			deviceInfo.append(temp_stream.str());
			deviceInfo.append("\n");                                  

			// Get maximum number of threads per block
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(valueSize), &valueSize, NULL);            
			deviceInfo.append("Max number of threads per block: ");
			temp_stream.str("");
			temp_stream.clear();
			temp_stream << valueSize;            
			deviceInfo.append(temp_stream.str());
			deviceInfo.append("\n");                                  
        
			// Get maximum block dimensions
			clGetDeviceInfo(deviceIds[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(valueSizes), valueSizes, NULL);            
			deviceInfo.append("Max number of threads in each dimension: ");
			temp_stream.str("");
			temp_stream.clear();
			temp_stream << valueSizes[0];
			temp_stream << " ";
			temp_stream << valueSizes[1];
			temp_stream << " ";
			temp_stream << valueSizes[2];
			deviceInfo.append(temp_stream.str());
			deviceInfo.append("\n");                                  

			deviceInfo.append("\n");
		}
	}
}

void BROCCOLI_LIB::GetBandwidth()
{
	size_t elements = 131000000/2;

	// Allocate 250 MB on host
	float* h_Data = (float*)malloc(elements * sizeof(float));

	// Allocate 250 MB on device
	cl_mem d_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, elements * sizeof(float), NULL, NULL);
	
	// Copy data from host to device
	double start = GetTime();
	for (int i = 0; i < 10; i++)
	{
		clEnqueueWriteBuffer(commandQueue, d_Data, CL_TRUE, 0, elements * sizeof(float), h_Data, 0, NULL, NULL);
		clFinish(commandQueue);
	}
	double end = GetTime();
	double time = (end - start)/10.0;
	printf("On average it took %f seconds to transfer 250 MB from host to device, giving a bandwidth of %f MB/s\n",(float)(time),(float)(250.0/time));

	// Copy data from device to host
	start = GetTime();
	for (int i = 0; i < 10; i++)
	{
		clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, elements * sizeof(float), h_Data, 0, NULL, NULL);
		clFinish(commandQueue);
	}
	end = GetTime();
	time = (end - start)/10.0;
	printf("On average it took %f seconds to transfer 250 MB from device to host, giving a bandwidth of %f MB/s\n",(float)(time),(float)(250.0/time));

	cl_mem d_Data2 = clCreateBuffer(context, CL_MEM_READ_WRITE, elements * sizeof(float), NULL, NULL);

	// Copy data from device to device
	start = GetTime();
	for (int i = 0; i < 10; i++)
	{
		clEnqueueCopyBuffer(commandQueue, d_Data, d_Data2, 0, 0, elements * sizeof(float), 0, NULL, NULL);
		clFinish(commandQueue);
	}
	end = GetTime();
	time = (end - start)/10.0;
	printf("On average it took %f seconds to transfer 250 MB from device to device, giving a bandwidth of %f MB/s\n",(float)(time),(float)(250.0/time));


	free(h_Data);
	clReleaseMemObject(d_Data);
	clReleaseMemObject(d_Data2);
}

const char* BROCCOLI_LIB::GetOpenCLDeviceName()
{
	return deviceName.c_str();
}

const char* BROCCOLI_LIB::GetOpenCLPlatformName()
{
	return platformName.c_str();
}

// Creates OpenCL programs from binary files
void BROCCOLI_LIB::CreateProgramFromBinary(cl_context context, cl_device_id device, std::string filename)
{
	for (int k = 0; k < NUMBER_OF_KERNEL_FILES; k++)
	{
		std::string thisFilename = filename;

		// Get device name and remove spaces, add to filename
		char* value;
		size_t valueSize;
		std::string device_name;
		clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
		value = (char*) malloc(valueSize);
		clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);            
		thisFilename.append("_");
		device_name = value;
		device_name.erase(std::remove (device_name.begin(), device_name.end(), ' '), device_name.end());
		thisFilename.append(device_name);			

		// Remove ".cpp" and "kernel" from kernel name and add kernel name
		std::string name = kernelFileNames[k];
		name = name.substr(0,name.size()-4);
		name = name.substr(6,name.size());
		thisFilename.append("_");
		thisFilename.append(name);	
		thisFilename.append(".bin");

		free(value);

		FILE* fp = fopen(thisFilename.c_str(), "rb");
		if (fp == NULL)
		{
			if ((WRAPPER == BASH) && VERBOS)
			{
				printf("Unable to open binary kernel file %s \n",thisFilename.c_str());
			}
			OpenCLPrograms[k] = NULL;
			createProgramErrors[k] = FAIL;
			continue;
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

		OpenCLPrograms[k] = clCreateProgramWithBinary(context, 1, &device, &binarySize, (const unsigned char**)&programBinary, &binaryStatus, &createProgramErrors[k]);
		delete [] programBinary;

		/*
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
		*/
	}
}

// Saves a compiled program to a binary file
bool BROCCOLI_LIB::SaveProgramBinary(cl_device_id device, std::string filename, int k)
{
	std::string thisFilename = filename;		

	// Get number of devices for program
	cl_uint numDevices = 0;
	error = clGetProgramInfo(OpenCLPrograms[k], CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &numDevices, NULL);
	if (error != SUCCESS)
	{
		return false;
	}

	// Get device IDs
	cl_device_id* devices = new cl_device_id[numDevices];
	error = clGetProgramInfo(OpenCLPrograms[k], CL_PROGRAM_DEVICES, sizeof(cl_device_id) * numDevices, devices, NULL);
	if (error != SUCCESS)
	{
		// Cleanup
		delete [] devices;
		return false;
	}

	// Get size of each program binary
	size_t* programBinarySizes = new size_t[numDevices];
	error = clGetProgramInfo(OpenCLPrograms[k], CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * numDevices, programBinarySizes, NULL);
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
	error = clGetProgramInfo(OpenCLPrograms[k], CL_PROGRAM_BINARIES, sizeof(unsigned char*) * numDevices, programBinaries, NULL);
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
			thisFilename.append("_");
			device_name = value;
			// Remove spaces and add device name
			device_name.erase(std::remove (device_name.begin(), device_name.end(), ' '), device_name.end());
			thisFilename.append(device_name);

			// Remove ".cpp" and "kernel" from kernel name and add kernel name
			std::string name = kernelFileNames[k];
			name = name.substr(0,name.size()-4);
			name = name.substr(6,name.size());
			thisFilename.append("_");
			thisFilename.append(name);	
			thisFilename.append(".bin");
			free(value);

			// Write binary to file
			FILE* fp = fopen(thisFilename.c_str(), "wb");
			if (fp != NULL)
			{
				programBinarySize = programBinarySizes[i];
				writtenElements = fwrite(programBinaries[i], 1, programBinarySizes[i], fp);
				fclose(fp);
				break;				
			}
			else
			{
				if (WRAPPER == BASH)
				{
					printf("Unable to write to binary file for kernel %s, null file pointer!\n",kernelFileNames[k].c_str());
				}
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


std::string BROCCOLI_LIB::GetBROCCOLIDirectory()
{
    if (getenv("BROCCOLI_DIR") != NULL)
	{
		return std::string(getenv("BROCCOLI_DIR"));
	}	
	else
  	{
		return "ERROR"; 
 	}
}



bool BROCCOLI_LIB::OpenCLInitiate(cl_uint OPENCL_PLATFORM, cl_uint OPENCL_DEVICE)
{
	char* value = NULL;
	size_t valueSize;

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
		INITIALIZATION_ERROR = "Unable to create an OpenCL context for the selected device.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);
		return false;
	}

	// Get size of name of current platform
	error = clGetPlatformInfo(platformIds[OPENCL_PLATFORM], CL_PLATFORM_NAME, 0, NULL, &valueSize);

	if (error != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable to get size of name for selected OpenCL platform.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);		
		return false;
	}

	// Get name of current platform
	value = (char*) malloc(valueSize);
	error = clGetPlatformInfo(platformIds[OPENCL_PLATFORM], CL_PLATFORM_NAME, valueSize, value, NULL);

	if (error != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable to get name of selected OpenCL platform.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);
		free(value);
		return false;
	}

	// Convert name to string
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
		platformName = "Nvidia";
	}
	else if (intelPos != std::string::npos)
	{
		VENDOR = INTEL;
		binaryFilename = "broccoli_lib_kernel_Intel";
		platformName = "Intel";
	}
	else if (amdPos != std::string::npos)
	{
		VENDOR = AMD;
		binaryFilename = "broccoli_lib_kernel_AMD";
		platformName = "AMD";
	}
	else if (applePos != std::string::npos)
	{
		VENDOR = APPLE;
		binaryFilename = "broccoli_lib_kernel_Apple";
		platformName = "Apple";
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

	// Get device name

	// Get size of name
	error = clGetDeviceInfo(deviceIds[OPENCL_DEVICE], CL_DEVICE_NAME, 0, NULL, &valueSize);

	if (error != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable to get size of device name.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);		
		return false;
	}

	// Get the actual name
	value = (char*) malloc(valueSize);
	error = clGetDeviceInfo(deviceIds[OPENCL_DEVICE], CL_DEVICE_NAME, valueSize, value, NULL);            

	if (error != SUCCESS)
	{
		INITIALIZATION_ERROR = "Unable to get name of specified OpenCL device.";
		OPENCL_ERROR = GetOpenCLErrorMessage(error);
		free(value);
		return false;
	}
	deviceName = value;
	free(value);

	// Remove spaces
	deviceName.erase(std::remove (deviceName.begin(), deviceName.end(), ' '), deviceName.end());

	// Support for running BROCCOLI from any directory

	// Check if BROCCOLI_DIR environment variable is set
	if (WRAPPER == BASH)
	{
		if (GetBROCCOLIDirectory().compare("ERROR") == 0)
		{
			INITIALIZATION_ERROR = "BROCCOLI_DIR environment variable is not set!";
			OPENCL_ERROR = "";
			return false;		    
		}
	}

	// Get the location of the compiled kernels
	std::string binaryPathAndFileName;
	if (WRAPPER == BASH)
	{
		binaryPathAndFilename.append(GetBROCCOLIDirectory());
		binaryPathAndFilename.append("compiled/Kernels/");
	}
	binaryPathAndFilename.append(binaryFilename);

	// First try to compile from binary file for the selected device and platform
	CreateProgramFromBinary(context, deviceIds[OPENCL_DEVICE], binaryPathAndFilename);
	
	for (int k = 0; k < NUMBER_OF_KERNEL_FILES; k++)
	{
		if (createProgramErrors[k] == CL_SUCCESS)
		{	
			if ( (WRAPPER == BASH) && VERBOS )
			{
				printf("Building program from binary for %s \n",kernelFileNames[k].c_str());
			}

			// Build program for the selected device
			binaryBuildProgramErrors[k] = clBuildProgram(OpenCLPrograms[k], 1, &deviceIds[OPENCL_DEVICE], NULL, NULL, NULL);

			if ( (WRAPPER == BASH) && (binaryBuildProgramErrors[k] != CL_SUCCESS) )
			{
				printf("Binary build error for %s is %s \n",kernelFileNames[k].c_str(),GetOpenCLErrorMessage(binaryBuildProgramErrors[k]));
			}
		}
		else
		{
			if ( (WRAPPER == BASH) && VERBOS )
			{
				printf("Not building program from binary for %s since create program error was %s \n",kernelFileNames[k].c_str(),GetOpenCLErrorMessage(createProgramErrors[k]));
			}
		}
	}

	// Otherwise compile from source code

	// Get the location of the OpenCL kernel code
	std::string OpenCLPath;
	if (WRAPPER == BASH)
	{	
		OpenCLPath.append(GetBROCCOLIDirectory());
		OpenCLPath.append("code/Kernels/");
	}

	std::vector<std::string> kernelPathAndFileNames;

	for (int k = 0; k < NUMBER_OF_KERNEL_FILES; k++)
	{
		std::string temp = OpenCLPath;
		temp.append(kernelFileNames[k]);
		kernelPathAndFileNames.push_back(temp);
	}

	for (int k = 0; k < NUMBER_OF_KERNEL_FILES; k++)
	{
		// Check if kernel was built from binary
		if (binaryBuildProgramErrors[k] != CL_SUCCESS)
		{
			// Check if kernel file exists
			std::ifstream file(kernelPathAndFileNames[k].c_str());
			if ( !file.good() )
			{
				std::string temp = "Unable to open ";
				temp.append(kernelFileNames[k]);
				INITIALIZATION_ERROR = temp;
				OPENCL_ERROR = "";
				return false;
			}

			// Read the kernel code from file
			std::fstream kernelFile(kernelPathAndFileNames[k].c_str(),std::ios::in);

			std::ostringstream oss;
			oss << kernelFile.rdbuf();
			std::string src = oss.str();
			const char *srcstr = src.c_str();

			if ( (WRAPPER == BASH) && (VERBOS) )
			{
				printf("Creating program for %s \n",kernelFileNames[k].c_str());
			}

			// Create program 
			OpenCLPrograms[k] = clCreateProgramWithSource(context, 1, (const char**)&srcstr , NULL, &error);

			if ( (WRAPPER == BASH) && (error != SUCCESS) )
			{
				printf("Create program error for %s is %s \n",kernelFileNames[k].c_str(),GetOpenCLErrorMessage(error));
			}

			if (error == SUCCESS)
			{
				if ( (WRAPPER == BASH) && (VERBOS) )
				{
					printf("Building program from source for %s \n",kernelFileNames[k].c_str());
				}

				// Build program for the selected device
				sourceBuildProgramErrors[k] = clBuildProgram(OpenCLPrograms[k], 1, &deviceIds[OPENCL_DEVICE], NULL, NULL, NULL);

				if ( (WRAPPER == BASH) && (sourceBuildProgramErrors[k] != SUCCESS) )
				{
					printf("Source build error for %s is %s \n",kernelFileNames[k].c_str(),GetOpenCLErrorMessage(sourceBuildProgramErrors[k]));
				}

				// Always get build info

				// Get size of build info
	
				valueSize = 0;
				error = clGetProgramBuildInfo(OpenCLPrograms[k], deviceIds[OPENCL_DEVICE], CL_PROGRAM_BUILD_LOG, 0, NULL, &valueSize);

				if (error != SUCCESS)
				{
					INITIALIZATION_ERROR = "Unable to get size of build info .";
					OPENCL_ERROR = GetOpenCLErrorMessage(error);
					return false;
				}

				value = (char*)malloc(valueSize);
				error = clGetProgramBuildInfo(OpenCLPrograms[k], deviceIds[OPENCL_DEVICE], CL_PROGRAM_BUILD_LOG, valueSize, value, NULL);

				if (error != SUCCESS)
				{
					INITIALIZATION_ERROR = "Unable to get build info.";
					OPENCL_ERROR = GetOpenCLErrorMessage(error);
					free(value);
					return false;
				}

				buildInfo[k] = std::string(value);
				free(value);
			}
			else
			{
				buildInfo[k] = std::string("No build info available, since create program error occured");
			}

			// If successful build, save each program as a binary file
			if (sourceBuildProgramErrors[k] == CL_SUCCESS)
			{
				SaveProgramBinary(deviceIds[OPENCL_DEVICE],binaryPathAndFilename,k);		
			}
		}
		else
		{
			buildInfo[k] = std::string("Kernel was successfully built from binary!");
		}
	}

	// Get some info about the selected device

	// Find out the size of the global memory in MB
	clGetDeviceInfo(deviceIds[OPENCL_DEVICE], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemorySize), &globalMemorySize, NULL); 
	globalMemorySize /= (1024*1024);

	// Find out the size of the local (shared) memory in KB
	clGetDeviceInfo(deviceIds[OPENCL_DEVICE], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemorySize), &localMemorySize, NULL);            
	localMemorySize /= 1024;            
	
	// Find out the maximum number of threads per thread block
	clGetDeviceInfo(deviceIds[OPENCL_DEVICE], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxThreadsPerBlock), &maxThreadsPerBlock, NULL);            
       
	// Get maximum block dimensions
	clGetDeviceInfo(deviceIds[OPENCL_DEVICE], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxThreadsPerDimension), maxThreadsPerDimension, NULL);            

	if ( (WRAPPER == BASH) && VERBOS )
	{
		printf("The selected OpenCL device has %i KB of local memory, %i MB of global memory, and can run %i threads per thread block, max threads per dimension are %i %i %i\n",(int)localMemorySize,(int)globalMemorySize,(int)maxThreadsPerBlock,(int)maxThreadsPerDimension[0],(int)maxThreadsPerDimension[1],(int)maxThreadsPerDimension[2]);
	}

	// Create kernels
	
	// Non-separable convolution kernel using 32 KB of shared memory and 512 threads per thread block (32 * 16)
	if ( (localMemorySize >= 32) && (maxThreadsPerBlock >= 512) && (maxThreadsPerDimension[0] >= 32) && (maxThreadsPerDimension[1] >= 16)  )
	{
		NonseparableConvolution3DComplexThreeFiltersKernel = clCreateKernel(OpenCLPrograms[0],"Nonseparable3DConvolutionComplexThreeQuadratureFilters_32KB_512threads",&createKernelErrorNonseparableConvolution3DComplexThreeFilters);
	}
	// Non-separable convolution kernel using 24 KB of shared memory and 1024 threads per thread block (32 * 32)
	else if ( (localMemorySize >= 24) && (maxThreadsPerBlock >= 1024) && (maxThreadsPerDimension[0] >= 32) && (maxThreadsPerDimension[1] >= 32)  )
	{
		NonseparableConvolution3DComplexThreeFiltersKernel = clCreateKernel(OpenCLPrograms[0],"Nonseparable3DConvolutionComplexThreeQuadratureFilters_24KB_1024threads",&createKernelErrorNonseparableConvolution3DComplexThreeFilters);
	}
	// Non-separable convolution kernel using 32 KB of shared memory and 256 threads per thread block (16 * 16)
	else if ( (localMemorySize >= 32) && (maxThreadsPerBlock >= 256) && (maxThreadsPerDimension[0] >= 16) && (maxThreadsPerDimension[1] >= 16)  )
	{
		NonseparableConvolution3DComplexThreeFiltersKernel = clCreateKernel(OpenCLPrograms[0],"Nonseparable3DConvolutionComplexThreeQuadratureFilters_32KB_256threads",&createKernelErrorNonseparableConvolution3DComplexThreeFilters);
	}
	// Non-separable convolution kernel using global memory only (backup)
	else
	{
		NonseparableConvolution3DComplexThreeFiltersKernel = clCreateKernel(OpenCLPrograms[0],"Nonseparable3DConvolutionComplexThreeQuadratureFiltersGlobalMemory",&createKernelErrorNonseparableConvolution3DComplexThreeFilters);
	}

	// Separable convolution kernels using 16 KB of shared memory and 512 threads per thread block (32 * 8 * 2 and 32 * 2 * 8)
	if ( (localMemorySize >= 16) && (maxThreadsPerBlock >= 512) && (maxThreadsPerDimension[0] >= 32) && (maxThreadsPerDimension[1] >= 8) && (maxThreadsPerDimension[2] >= 8)  )
	{
		SeparableConvolutionRowsKernel = clCreateKernel(OpenCLPrograms[0],"SeparableConvolutionRows_16KB_512threads",&createKernelErrorSeparableConvolutionRows);
		SeparableConvolutionColumnsKernel = clCreateKernel(OpenCLPrograms[0],"SeparableConvolutionColumns_16KB_512threads",&createKernelErrorSeparableConvolutionColumns);
		SeparableConvolutionRodsKernel = clCreateKernel(OpenCLPrograms[0],"SeparableConvolutionRods_16KB_512threads",&createKernelErrorSeparableConvolutionRods);
	}
	// Separable convolution kernels using 16 KB of shared memory and 256 threads per thread block (32 * 8 * 1 and 32 * 1 * 8)
	else if ( (localMemorySize >= 16) && (maxThreadsPerBlock >= 256) && (maxThreadsPerDimension[0] >= 32) && (maxThreadsPerDimension[1] >= 8) && (maxThreadsPerDimension[2] >= 8)  )
	{
		SeparableConvolutionRowsKernel = clCreateKernel(OpenCLPrograms[0],"SeparableConvolutionRows_16KB_256threads",&createKernelErrorSeparableConvolutionRows);
		SeparableConvolutionColumnsKernel = clCreateKernel(OpenCLPrograms[0],"SeparableConvolutionColumns_16KB_256threads",&createKernelErrorSeparableConvolutionColumns);
		SeparableConvolutionRodsKernel = clCreateKernel(OpenCLPrograms[0],"SeparableConvolutionRods_16KB_256threads",&createKernelErrorSeparableConvolutionRods);
	}
	// Separable convolution kernels using global memory only (backup)
	else
	{
		SeparableConvolutionRowsKernel = clCreateKernel(OpenCLPrograms[0],"SeparableConvolutionRowsGlobalMemory",&createKernelErrorSeparableConvolutionRows);
		SeparableConvolutionColumnsKernel = clCreateKernel(OpenCLPrograms[0],"SeparableConvolutionColumnsGlobalMemory",&createKernelErrorSeparableConvolutionColumns);
		SeparableConvolutionRodsKernel = clCreateKernel(OpenCLPrograms[0],"SeparableConvolutionRodsGlobalMemory",&createKernelErrorSeparableConvolutionRods);
	}

	OpenCLKernels[0] = NonseparableConvolution3DComplexThreeFiltersKernel;
	OpenCLKernels[1] = SeparableConvolutionRowsKernel;
	OpenCLKernels[2] = SeparableConvolutionColumnsKernel;
	OpenCLKernels[3] = SeparableConvolutionRodsKernel;

	SliceTimingCorrectionKernel = clCreateKernel(OpenCLPrograms[3],"SliceTimingCorrection",&createKernelErrorSliceTimingCorrection);

	OpenCLKernels[4] = SliceTimingCorrectionKernel;

	// Kernels for linear registration
	CalculatePhaseDifferencesAndCertaintiesKernel = clCreateKernel(OpenCLPrograms[1],"CalculatePhaseDifferencesAndCertainties",&createKernelErrorCalculatePhaseDifferencesAndCertainties);
	CalculatePhaseGradientsXKernel = clCreateKernel(OpenCLPrograms[1],"CalculatePhaseGradientsX",&createKernelErrorCalculatePhaseGradientsX);
	CalculatePhaseGradientsYKernel = clCreateKernel(OpenCLPrograms[1],"CalculatePhaseGradientsY",&createKernelErrorCalculatePhaseGradientsY);
	CalculatePhaseGradientsZKernel = clCreateKernel(OpenCLPrograms[1],"CalculatePhaseGradientsZ",&createKernelErrorCalculatePhaseGradientsZ);
	CalculateAMatrixAndHVector2DValuesXKernel = clCreateKernel(OpenCLPrograms[1],"CalculateAMatrixAndHVector2DValuesX",&createKernelErrorCalculateAMatrixAndHVector2DValuesX);
	CalculateAMatrixAndHVector2DValuesYKernel = clCreateKernel(OpenCLPrograms[1],"CalculateAMatrixAndHVector2DValuesY",&createKernelErrorCalculateAMatrixAndHVector2DValuesY);
	CalculateAMatrixAndHVector2DValuesZKernel = clCreateKernel(OpenCLPrograms[1],"CalculateAMatrixAndHVector2DValuesZ",&createKernelErrorCalculateAMatrixAndHVector2DValuesZ);
	CalculateAMatrix1DValuesKernel = clCreateKernel(OpenCLPrograms[1],"CalculateAMatrix1DValues",&createKernelErrorCalculateAMatrix1DValues);
	CalculateHVector1DValuesKernel = clCreateKernel(OpenCLPrograms[1],"CalculateHVector1DValues",&createKernelErrorCalculateHVector1DValues);
	CalculateAMatrixKernel = clCreateKernel(OpenCLPrograms[1],"CalculateAMatrix",&createKernelErrorCalculateAMatrix);
	CalculateHVectorKernel = clCreateKernel(OpenCLPrograms[1],"CalculateHVector",&createKernelErrorCalculateHVector);

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

	// Kernels for non-linear registration
	CalculateTensorComponentsKernel = clCreateKernel(OpenCLPrograms[1], "CalculateTensorComponents", &createKernelErrorCalculateTensorComponents);
	CalculateTensorNormsKernel = clCreateKernel(OpenCLPrograms[1], "CalculateTensorNorms", &createKernelErrorCalculateTensorNorms);
	CalculateAMatricesAndHVectorsKernel = clCreateKernel(OpenCLPrograms[1], "CalculateAMatricesAndHVectors", &createKernelErrorCalculateAMatricesAndHVectors);
	CalculateDisplacementUpdateKernel = clCreateKernel(OpenCLPrograms[1], "CalculateDisplacementUpdate", &createKernelErrorCalculateDisplacementUpdate);
	AddLinearAndNonLinearDisplacementKernel = clCreateKernel(OpenCLPrograms[1], "AddLinearAndNonLinearDisplacement", &createKernelErrorAddLinearAndNonLinearDisplacement);

	OpenCLKernels[16] = CalculateTensorComponentsKernel;
	OpenCLKernels[17] = CalculateTensorNormsKernel;
	OpenCLKernels[18] = CalculateAMatricesAndHVectorsKernel;
	OpenCLKernels[19] = CalculateDisplacementUpdateKernel;
	OpenCLKernels[20] = AddLinearAndNonLinearDisplacementKernel;

	// Help kernels
	CalculateMagnitudesKernel = clCreateKernel(OpenCLPrograms[3],"CalculateMagnitudes",&createKernelErrorCalculateMagnitudes);
	CalculateColumnSumsKernel = clCreateKernel(OpenCLPrograms[3],"CalculateColumnSums",&createKernelErrorCalculateColumnSums);
	CalculateRowSumsKernel = clCreateKernel(OpenCLPrograms[3],"CalculateRowSums",&createKernelErrorCalculateRowSums);
	CalculateColumnMaxsKernel = clCreateKernel(OpenCLPrograms[3],"CalculateColumnMaxs",&createKernelErrorCalculateColumnMaxs);
	CalculateRowMaxsKernel = clCreateKernel(OpenCLPrograms[3],"CalculateRowMaxs",&createKernelErrorCalculateRowMaxs);
	CalculateMaxAtomicKernel = clCreateKernel(OpenCLPrograms[3],"CalculateMaxAtomic",&createKernelErrorCalculateMaxAtomic);
	ThresholdVolumeKernel = clCreateKernel(OpenCLPrograms[3],"ThresholdVolume",&createKernelErrorThresholdVolume);
	MemsetKernel = clCreateKernel(OpenCLPrograms[3],"Memset",&createKernelErrorMemset);
	MemsetDoubleKernel = clCreateKernel(OpenCLPrograms[3],"MemsetDouble",&createKernelErrorMemsetDouble);
	MemsetIntKernel = clCreateKernel(OpenCLPrograms[3],"MemsetInt",&createKernelErrorMemsetInt);
	MemsetFloat2Kernel = clCreateKernel(OpenCLPrograms[3],"MemsetFloat2",&createKernelErrorMemsetFloat2);
	IdentityMatrixKernel = clCreateKernel(OpenCLPrograms[3],"IdentityMatrix",&createKernelErrorIdentityMatrix);
	IdentityMatrixDoubleKernel = clCreateKernel(OpenCLPrograms[3],"IdentityMatrixDouble",&createKernelErrorIdentityMatrixDouble);
	GetSubMatrixKernel = clCreateKernel(OpenCLPrograms[3],"GetSubMatrix",&createKernelErrorGetSubMatrix);
	GetSubMatrixDoubleKernel = clCreateKernel(OpenCLPrograms[3],"GetSubMatrixDouble",&createKernelErrorGetSubMatrixDouble);
	PermuteMatrixKernel = clCreateKernel(OpenCLPrograms[3],"PermuteMatrix",&createKernelErrorPermuteMatrix);
	PermuteMatrixDoubleKernel = clCreateKernel(OpenCLPrograms[3],"PermuteMatrixDouble",&createKernelErrorPermuteMatrixDouble);
	LogitMatrixKernel = clCreateKernel(OpenCLPrograms[3],"LogitMatrix",&createKernelErrorLogitMatrix);
	LogitMatrixDoubleKernel = clCreateKernel(OpenCLPrograms[3],"LogitMatrixDouble",&createKernelErrorLogitMatrixDouble);
	MultiplyVolumeKernel = clCreateKernel(OpenCLPrograms[3],"MultiplyVolume",&createKernelErrorMultiplyVolume);
	MultiplyVolumesKernel = clCreateKernel(OpenCLPrograms[3],"MultiplyVolumes",&createKernelErrorMultiplyVolumes);
	MultiplyVolumesOverwriteKernel = clCreateKernel(OpenCLPrograms[3],"MultiplyVolumesOverwrite",&createKernelErrorMultiplyVolumesOverwrite);
	MultiplyVolumesOverwriteDoubleKernel = clCreateKernel(OpenCLPrograms[3],"MultiplyVolumesOverwriteDouble",&createKernelErrorMultiplyVolumesOverwriteDouble);
	AddVolumeKernel = clCreateKernel(OpenCLPrograms[3],"AddVolume",&createKernelErrorAddVolume);
	AddVolumesKernel = clCreateKernel(OpenCLPrograms[3],"AddVolumes",&createKernelErrorAddVolumes);
	AddVolumesOverwriteKernel = clCreateKernel(OpenCLPrograms[3],"AddVolumesOverwrite",&createKernelErrorAddVolumesOverwrite);
	SubtractVolumesKernel = clCreateKernel(OpenCLPrograms[3],"SubtractVolumes",&createKernelErrorSubtractVolumes);
	SubtractVolumesOverwriteKernel = clCreateKernel(OpenCLPrograms[3],"SubtractVolumesOverwrite",&createKernelErrorSubtractVolumesOverwrite);
	SubtractVolumesOverwriteDoubleKernel = clCreateKernel(OpenCLPrograms[3],"SubtractVolumesOverwriteDouble",&createKernelErrorSubtractVolumesOverwriteDouble);
	RemoveMeanKernel = clCreateKernel(OpenCLPrograms[3],"RemoveMean",&createKernelErrorRemoveMean);

	OpenCLKernels[21] = CalculateMagnitudesKernel;
	OpenCLKernels[22] = CalculateColumnSumsKernel;
	OpenCLKernels[23] = CalculateRowSumsKernel;
	OpenCLKernels[24] = CalculateColumnMaxsKernel;
	OpenCLKernels[25] = CalculateRowMaxsKernel;
	OpenCLKernels[26] = CalculateMaxAtomicKernel;
	OpenCLKernels[27] = ThresholdVolumeKernel;
	OpenCLKernels[28] = MemsetKernel;
	OpenCLKernels[29] = MemsetDoubleKernel;
	OpenCLKernels[30] = MemsetIntKernel;
	OpenCLKernels[31] = MemsetFloat2Kernel;
	OpenCLKernels[32] = IdentityMatrixKernel;
	OpenCLKernels[33] = IdentityMatrixDoubleKernel;
	OpenCLKernels[34] = GetSubMatrixKernel;
	OpenCLKernels[35] = GetSubMatrixDoubleKernel;
	OpenCLKernels[36] = PermuteMatrixKernel;
	OpenCLKernels[37] = PermuteMatrixDoubleKernel;
	OpenCLKernels[38] = LogitMatrixKernel;
	OpenCLKernels[39] = LogitMatrixDoubleKernel;
	OpenCLKernels[40] = MultiplyVolumeKernel;
	OpenCLKernels[41] = MultiplyVolumesKernel;
	OpenCLKernels[42] = MultiplyVolumesOverwriteKernel;
	OpenCLKernels[43] = MultiplyVolumesOverwriteDoubleKernel;
	OpenCLKernels[44] = AddVolumeKernel;
	OpenCLKernels[45] = AddVolumesKernel;
	OpenCLKernels[46] = AddVolumesOverwriteKernel;
	OpenCLKernels[47] = SubtractVolumesKernel;
	OpenCLKernels[48] = SubtractVolumesOverwriteKernel;
	OpenCLKernels[49] = SubtractVolumesOverwriteDoubleKernel;
	OpenCLKernels[50] = RemoveMeanKernel;

	// Interpolation kernels
	InterpolateVolumeNearestLinearKernel = clCreateKernel(OpenCLPrograms[1],"InterpolateVolumeNearestLinear",&createKernelErrorInterpolateVolumeNearestLinear);
	InterpolateVolumeLinearLinearKernel = clCreateKernel(OpenCLPrograms[1],"InterpolateVolumeLinearLinear",&createKernelErrorInterpolateVolumeLinearLinear);
	InterpolateVolumeCubicLinearKernel = clCreateKernel(OpenCLPrograms[1],"InterpolateVolumeCubicLinear",&createKernelErrorInterpolateVolumeCubicLinear);
	InterpolateVolumeNearestNonLinearKernel = clCreateKernel(OpenCLPrograms[1],"InterpolateVolumeNearestNonLinear",&createKernelErrorInterpolateVolumeNearestNonLinear);
	InterpolateVolumeLinearNonLinearKernel = clCreateKernel(OpenCLPrograms[1],"InterpolateVolumeLinearNonLinear",&createKernelErrorInterpolateVolumeLinearNonLinear);
	InterpolateVolumeCubicNonLinearKernel = clCreateKernel(OpenCLPrograms[1],"InterpolateVolumeCubicNonLinear",&createKernelErrorInterpolateVolumeCubicNonLinear);

	OpenCLKernels[51] = InterpolateVolumeNearestLinearKernel;
	OpenCLKernels[52] = InterpolateVolumeLinearLinearKernel;
	OpenCLKernels[53] = InterpolateVolumeCubicLinearKernel;
	OpenCLKernels[54] = InterpolateVolumeNearestNonLinearKernel;
	OpenCLKernels[55] = InterpolateVolumeLinearNonLinearKernel;
	OpenCLKernels[56] = InterpolateVolumeCubicNonLinearKernel;

	RescaleVolumeLinearKernel = clCreateKernel(OpenCLPrograms[1],"RescaleVolumeLinear",&createKernelErrorRescaleVolumeLinear);
	RescaleVolumeCubicKernel = clCreateKernel(OpenCLPrograms[1],"RescaleVolumeCubic",&createKernelErrorRescaleVolumeCubic);
	RescaleVolumeNearestKernel = clCreateKernel(OpenCLPrograms[1],"RescaleVolumeNearest",&createKernelErrorRescaleVolumeNearest);

	OpenCLKernels[57] = RescaleVolumeLinearKernel;
	OpenCLKernels[58] = RescaleVolumeCubicKernel;
	OpenCLKernels[59] = RescaleVolumeNearestKernel;

	CopyT1VolumeToMNIKernel = clCreateKernel(OpenCLPrograms[1],"CopyT1VolumeToMNI",&createKernelErrorCopyT1VolumeToMNI);
	CopyEPIVolumeToT1Kernel = clCreateKernel(OpenCLPrograms[1],"CopyEPIVolumeToT1",&createKernelErrorCopyEPIVolumeToT1);
	CopyVolumeToNewKernel = clCreateKernel(OpenCLPrograms[1],"CopyVolumeToNew",&createKernelErrorCopyVolumeToNew);

	OpenCLKernels[60] = CopyT1VolumeToMNIKernel;
	OpenCLKernels[61] = CopyEPIVolumeToT1Kernel;
	OpenCLKernels[62] = CopyVolumeToNewKernel;

	// Clusterize kernels	
	SetStartClusterIndicesKernel = clCreateKernel(OpenCLPrograms[2],"SetStartClusterIndicesKernel",&createKernelErrorSetStartClusterIndices);
	ClusterizeScanKernel = clCreateKernel(OpenCLPrograms[2],"ClusterizeScan",&createKernelErrorClusterizeScan);
	ClusterizeRelabelKernel = clCreateKernel(OpenCLPrograms[2],"ClusterizeRelabel",&createKernelErrorClusterizeRelabel);
	CalculateClusterSizesKernel = clCreateKernel(OpenCLPrograms[2],"CalculateClusterSizes",&createKernelErrorCalculateClusterSizes);
	CalculateClusterMassesKernel = clCreateKernel(OpenCLPrograms[2],"CalculateClusterMasses",&createKernelErrorCalculateClusterMasses);
	CalculateLargestClusterKernel = clCreateKernel(OpenCLPrograms[2],"CalculateLargestCluster",&createKernelErrorCalculateLargestCluster);
	CalculateTFCEValuesKernel = clCreateKernel(OpenCLPrograms[2],"CalculateTFCEValues",&createKernelErrorCalculateTFCEValues);
	CalculatePermutationPValuesVoxelLevelInferenceKernel = clCreateKernel(OpenCLPrograms[2],"CalculatePermutationPValuesVoxelLevelInference",&createKernelErrorCalculatePermutationPValuesVoxelLevelInference);
	CalculatePermutationPValuesClusterExtentInferenceKernel = clCreateKernel(OpenCLPrograms[2],"CalculatePermutationPValuesClusterExtentInference",&createKernelErrorCalculatePermutationPValuesClusterExtentInference);
	CalculatePermutationPValuesClusterMassInferenceKernel = clCreateKernel(OpenCLPrograms[2],"CalculatePermutationPValuesClusterMassInference",&createKernelErrorCalculatePermutationPValuesClusterMassInference);


	OpenCLKernels[63] = SetStartClusterIndicesKernel;
	OpenCLKernels[64] = ClusterizeScanKernel;
	OpenCLKernels[65] = ClusterizeRelabelKernel;
	OpenCLKernels[66] = CalculateClusterSizesKernel;
	OpenCLKernels[67] = CalculateClusterMassesKernel;
	OpenCLKernels[68] = CalculateLargestClusterKernel;
	OpenCLKernels[69] = CalculateTFCEValuesKernel;
	OpenCLKernels[70] = CalculatePermutationPValuesVoxelLevelInferenceKernel;
	OpenCLKernels[71] = CalculatePermutationPValuesClusterExtentInferenceKernel;
	OpenCLKernels[72] = CalculatePermutationPValuesClusterMassInferenceKernel;

	// Statistical kernels
	CalculateBetaWeightsGLMKernel = clCreateKernel(OpenCLPrograms[4],"CalculateBetaWeightsGLM",&createKernelErrorCalculateBetaWeightsGLM);
	CalculateBetaWeightsGLMSliceKernel = clCreateKernel(OpenCLPrograms[4],"CalculateBetaWeightsGLMSlice",&createKernelErrorCalculateBetaWeightsGLMSlice);
	CalculateBetaWeightsAndContrastsGLMKernel = clCreateKernel(OpenCLPrograms[4],"CalculateBetaWeightsAndContrastsGLM",&createKernelErrorCalculateBetaWeightsAndContrastsGLM);
	CalculateBetaWeightsAndContrastsGLMSliceKernel = clCreateKernel(OpenCLPrograms[4],"CalculateBetaWeightsAndContrastsGLMSlice",&createKernelErrorCalculateBetaWeightsAndContrastsGLMSlice);
	CalculateBetaWeightsGLMFirstLevelKernel = clCreateKernel(OpenCLPrograms[4],"CalculateBetaWeightsGLMFirstLevel",&createKernelErrorCalculateBetaWeightsGLMFirstLevel);
	CalculateBetaWeightsGLMFirstLevelSliceKernel = clCreateKernel(OpenCLPrograms[4],"CalculateBetaWeightsGLMFirstLevelSlice",&createKernelErrorCalculateBetaWeightsGLMFirstLevelSlice);
	CalculateGLMResidualsKernel = clCreateKernel(OpenCLPrograms[4],"CalculateGLMResiduals",&createKernelErrorCalculateGLMResiduals);
	CalculateGLMResidualsSliceKernel = clCreateKernel(OpenCLPrograms[4],"CalculateGLMResidualsSlice",&createKernelErrorCalculateGLMResidualsSlice);
	CalculateStatisticalMapsGLMTTestFirstLevelKernel = clCreateKernel(OpenCLPrograms[4],"CalculateStatisticalMapsGLMTTestFirstLevel",&createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel);
	CalculateStatisticalMapsGLMFTestFirstLevelKernel = clCreateKernel(OpenCLPrograms[4],"CalculateStatisticalMapsGLMFTestFirstLevel",&createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevel);
	CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel = clCreateKernel(OpenCLPrograms[4],"CalculateStatisticalMapsGLMTTestFirstLevelSlice",&createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelSlice);
	CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel = clCreateKernel(OpenCLPrograms[4],"CalculateStatisticalMapsGLMFTestFirstLevelSlice",&createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelSlice);
	CalculateStatisticalMapsGLMTTestKernel = clCreateKernel(OpenCLPrograms[4],"CalculateStatisticalMapsGLMTTest",&createKernelErrorCalculateStatisticalMapsGLMTTest);
	CalculateStatisticalMapsGLMFTestKernel = clCreateKernel(OpenCLPrograms[4],"CalculateStatisticalMapsGLMFTest",&createKernelErrorCalculateStatisticalMapsGLMFTest);
	
    
    CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel = clCreateKernel(OpenCLPrograms[6],"CalculateStatisticalMapsGLMTTestFirstLevelPermutation",&createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelPermutation);
	CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel = clCreateKernel(OpenCLPrograms[8],"CalculateStatisticalMapsGLMFTestFirstLevelPermutation",&createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelPermutation);
    
    
	CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel = clCreateKernel(OpenCLPrograms[5],"CalculateStatisticalMapsGLMTTestSecondLevelPermutation",&createKernelErrorCalculateStatisticalMapsGLMTTestSecondLevelPermutation);
	CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel = clCreateKernel(OpenCLPrograms[7],"CalculateStatisticalMapsGLMFTestSecondLevelPermutation",&createKernelErrorCalculateStatisticalMapsGLMFTestSecondLevelPermutation);
	CalculateStatisticalMapsMeanSecondLevelPermutationKernel = clCreateKernel(OpenCLPrograms[5],"CalculateStatisticalMapsMeanSecondLevelPermutation",&createKernelErrorCalculateStatisticalMapsMeanSecondLevelPermutation);
	
    TransformDataKernel = clCreateKernel(OpenCLPrograms[4],"TransformData",&createKernelErrorTransformData);
	RemoveLinearFitKernel = clCreateKernel(OpenCLPrograms[4],"RemoveLinearFit",&createKernelErrorRemoveLinearFit);
	RemoveLinearFitSliceKernel = clCreateKernel(OpenCLPrograms[4],"RemoveLinearFitSlice",&createKernelErrorRemoveLinearFitSlice);

	OpenCLKernels[73] = CalculateBetaWeightsGLMKernel;
	OpenCLKernels[74] = CalculateBetaWeightsGLMSliceKernel;
	OpenCLKernels[75] = CalculateBetaWeightsAndContrastsGLMKernel;
	OpenCLKernels[76] = CalculateBetaWeightsAndContrastsGLMSliceKernel;
	OpenCLKernels[77] = CalculateBetaWeightsGLMFirstLevelKernel;
	OpenCLKernels[78] = CalculateBetaWeightsGLMFirstLevelSliceKernel;
	OpenCLKernels[79] = CalculateGLMResidualsKernel;
	OpenCLKernels[80] = CalculateGLMResidualsSliceKernel;
	OpenCLKernels[81] = CalculateStatisticalMapsGLMTTestFirstLevelKernel;
	OpenCLKernels[82] = CalculateStatisticalMapsGLMFTestFirstLevelKernel;
	OpenCLKernels[83] = CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel;
	OpenCLKernels[84] = CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel;
	OpenCLKernels[85] = CalculateStatisticalMapsGLMTTestKernel;
	OpenCLKernels[86] = CalculateStatisticalMapsGLMFTestKernel;
	OpenCLKernels[87] = CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel;
	OpenCLKernels[88] = CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel;
	OpenCLKernels[89] = CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel;
	OpenCLKernels[90] = CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel;
	OpenCLKernels[91] = CalculateStatisticalMapsMeanSecondLevelPermutationKernel;
	OpenCLKernels[92] = TransformDataKernel;
	OpenCLKernels[93] = RemoveLinearFitKernel;
	OpenCLKernels[94] = RemoveLinearFitSliceKernel;

	// Bayesian kernels
	CalculateStatisticalMapsGLMBayesianKernel = clCreateKernel(OpenCLPrograms[10],"CalculateStatisticalMapsGLMBayesian",&createKernelErrorCalculateStatisticalMapsGLMBayesian);

	OpenCLKernels[95] = CalculateStatisticalMapsGLMBayesianKernel;

	// Whitening kernels	
	EstimateAR4ModelsKernel = clCreateKernel(OpenCLPrograms[9],"EstimateAR4Models",&createKernelErrorEstimateAR4Models);
	EstimateAR4ModelsSliceKernel = clCreateKernel(OpenCLPrograms[9],"EstimateAR4ModelsSlice",&createKernelErrorEstimateAR4ModelsSlice);
	ApplyWhiteningAR4Kernel = clCreateKernel(OpenCLPrograms[9],"ApplyWhiteningAR4",&createKernelErrorApplyWhiteningAR4);
	ApplyWhiteningAR4SliceKernel = clCreateKernel(OpenCLPrograms[9],"ApplyWhiteningAR4Slice",&createKernelErrorApplyWhiteningAR4Slice);
	GeneratePermutedVolumesFirstLevelKernel = clCreateKernel(OpenCLPrograms[9],"GeneratePermutedVolumesFirstLevel",&createKernelErrorGeneratePermutedVolumesFirstLevel);

	OpenCLKernels[96] = EstimateAR4ModelsKernel;
	OpenCLKernels[97] = EstimateAR4ModelsSliceKernel;
	OpenCLKernels[98] = ApplyWhiteningAR4Kernel;
	OpenCLKernels[99] = ApplyWhiteningAR4SliceKernel;
	OpenCLKernels[100] = GeneratePermutedVolumesFirstLevelKernel;

    // Searchlight kernels
    CalculateStatisticalMapSearchlightKernel = clCreateKernel(OpenCLPrograms[11],"CalculateStatisticalMapSearchlight",&createKernelErrorCalculateStatisticalMapSearchlight);
    
    OpenCLKernels[101] = CalculateStatisticalMapSearchlightKernel;
    
	OPENCL_INITIATED = true;

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
		if (WRAPPER == BASH)
		{
			printf("One or several kernels were not created correctly, check buildInfo* !\n");
		}
		return true;
	}
	else
	{
		INITIALIZATION_ERROR = "";
		OPENCL_ERROR = "";
		return true;
	}
}

const char* BROCCOLI_LIB::GetOpenCLKernelName(int kernel)
{
	switch (kernel)
	{
		case 0:
			return "NonseparableConvolution3DComplexThreeFilters";
			break;
		case 1:
			return "SeparableConvolutionRows";
			break;
		case 2:
			return "SeparableConvolutionColumns";
			break;
		case 3:
			return "SeparableConvolutionRods";
			break;
		case 4:
			return "SliceTimingCorrection";
			break;
		case 5:
			return "CalculatePhaseDifferencesAndCertainties";
			break;
		case 6:
			return "CalculatePhaseGradientsX";
			break;
		case 7:
			return "CalculatePhaseGradientsY";
			break;
		case 8:
			return "CalculatePhaseGradientsZ";
			break;
		case 9:
			return "CalculateAMatrixAndHVector2DValuesX";
			break;
		case 10:
			return "CalculateAMatrixAndHVector2DValuesY";
			break;
		case 11:
			return "CalculateAMatrixAndHVector2DValuesZ";
			break;
		case 12:
			return "CalculateAMatrix1DValues";
			break;
		case 13:
			return "CalculateHVector1DValues";
			break;
		case 14:
			return "CalculateAMatrix";
			break;
		case 15:
			return "CalculateHVector";
			break;
		case 16:
			return "CalculateTensorComponents";
			break;
		case 17:
			return "CalculateTensorNorms";
			break;
		case 18:
			return "CalculateAMatricesAndHVectors";
			break;
		case 19:
			return "CalculateDisplacementUpdate";
			break;
		case 20:
			return "AddLinearAndNonLinearDisplacement";
			break;
		case 21:
			return "CalculateMagnitudes";
			break;
		case 22:
			return "CalculateColumnSums";
			break;
		case 23:
			return "CalculateRowSums";
			break;
		case 24:
			return "CalculateColumnMaxs";
			break;
		case 25:
			return "CalculateRowMaxs";
			break;
		case 26:
			return "CalculateMaxAtomic";
			break;
		case 27:
			return "ThresholdVolume";
			break;
		case 28:
			return "Memset";
			break;
		case 29:
			return "MemsetDouble";
			break;
		case 30:
			return "MemsetInt";
			break;
		case 31:
			return "MemsetFloat2";
			break;
		case 32:
			return "IdentityMatrix";
			break;
		case 33:
			return "IdentityMatrixDouble";
			break;
		case 34:
			return "GetSubMatrix";
			break;
		case 35:
			return "GetSubMatrixDouble";
			break;
		case 36:
			return "PermuteMatrix";
			break;
		case 37:
			return "PermuteMatrixDouble";
			break;
		case 38:
			return "LogitMatrix";
			break;
		case 39:
			return "LogitMatrixDouble";
			break;
		case 40:
			return "MultiplyVolume";
			break;
		case 41:
			return "MultiplyVolumes";
			break;
		case 42:
			return "MultiplyVolumesOverwrite";
			break;
		case 43:
			return "MultiplyVolumesOverwriteDouble";
			break;
		case 44:
			return "AddVolume";
			break;
		case 45:
			return "AddVolumes";
			break;
		case 46:
			return "AddVolumesOverwrite";
			break;
		case 47:
			return "SubtractVolumes";
			break;
		case 48:
			return "SubtractVolumesOverwrite";
			break;
		case 49:
			return "SubtractVolumesOverwriteDouble";
			break;
		case 50:
			return "RemoveMean";
			break;

		case 51:
			return "InterpolateVolumeNearestLinear";
			break;
		case 52:
			return "InterpolateVolumeLinearLinear";
			break;
		case 53:
			return "InterpolateVolumeCubicLinear";
			break;
		case 54:
			return "InterpolateVolumeNearestNonLinear";
			break;
		case 55:
			return "InterpolateVolumeLinearNonLinear";
			break;
		case 56:
			return "InterpolateVolumeCubicNonLinear";
			break;
		case 57:
			return "RescaleVolumeLinear";
			break;
		case 58:
			return "RescaleVolumeCubic";
			break;
		case 59:
			return "RescaleVolumeNearest";
			break;
		case 60:
			return "CopyT1VolumeToMNI";
			break;
		case 61:
			return "CopyEPIVolumeToT1";
			break;
		case 62:
			return "CopyVolumeToNew";
			break;
		
		case 63:
			return "SetStartClusterIndices";
			break;
		case 64:
			return "ClusterizeScan";
			break;
		case 65:
			return "ClusterizeRelabel";
			break;
		case 66:
			return "CalculateClusterSizes";
			break;
		case 67:
			return "CalculateClusterMasses";
			break;
		case 68:
			return "CalculateLargestCluster";
			break;
		case 69:
			return "CalculateTFCEValues";
			break;
		case 70:
			return "CalculatePermutationPValuesVoxelLevelInference";
			break;
		case 71:
			return "CalculatePermutationPValuesClusterExtentInference";
			break;
		case 72:
			return "CalculatePermutationPValuesClusterMassInference";
			break;

		case 73:
			return "CalculateBetaWeightsGLM";
			break;
		case 74:
			return "CalculateBetaWeightsGLMSlice";
			break;
		case 75:
			return "CalculateBetaWeightsAndContrastsGLM";
			break;
		case 76:
			return "CalculateBetaWeightsAndContrastsGLMSlice";
			break;
		case 77:
			return "CalculateBetaWeightsGLMFirstLevel";
			break;
		case 78:
			return "CalculateBetaWeightsGLMFirstLevelSlice";
			break;
		case 79:
			return "CalculateGLMResiduals";
			break;
		case 80:
			return "CalculateGLMResidualsSlice";
			break;
		case 81:
			return "CalculateStatisticalMapsGLMTTestFirstLevel";
			break;
		case 82:
			return "CalculateStatisticalMapsGLMFTestFirstLevel";
			break;
		case 83:
			return "CalculateStatisticalMapsGLMTTestFirstLevelSlice";
			break;
		case 84:
			return "CalculateStatisticalMapsGLMFTestFirstLevelSlice";
			break;
		case 85:
			return "CalculateStatisticalMapsGLMTTest";
			break;
		case 86:
			return "CalculateStatisticalMapsGLMFTest";
			break;
		case 87:
			return "CalculateStatisticalMapsGLMTTestFirstLevelPermutation";
			break;
		case 88:
			return "CalculateStatisticalMapsGLMFTestFirstLevelPermutation";
			break;
		case 89:
			return "CalculateStatisticalMapsGLMTTestSecondLevelPermutation";
			break;
		case 90:
			return "CalculateStatisticalMapsGLMFTestSecondLevelPermutation";
			break;
		case 91:
			return "CalculateStatisticalMapsMeanSecondLevelPermutation";
			break;
		case 92:
			return "TransformData";
			break;
		case 93:
			return "RemoveLinearFit";
			break;
		case 94:
			return "RemoveLinearFitSlice";
			break;

		case 95:
			return "CalculateStatisticalMapsGLMBayesian";
			break;
		case 96:
			return "EstimateAR4Models";
			break;
		case 97:
			return "EstimateAR4ModelsSlice";
			break;
		case 98:
			return "ApplyWhiteningAR4";
			break;
		case 99:
			return "ApplyWhiteningAR4Slice";
			break;
		case 100:
			return "GeneratePermutedVolumesFirstLevel";
			break;
        case 101:
            return "CalculateStatisticalMapSearchlight";
            break;
            
            
		default:
			return "Unrecognized BROCCOLI kernel";
	}
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
	OpenCLCreateKernelErrors[28] = createKernelErrorMemset;
	OpenCLCreateKernelErrors[29] = createKernelErrorMemsetDouble;
	OpenCLCreateKernelErrors[30] = createKernelErrorMemsetInt;
	OpenCLCreateKernelErrors[31] = createKernelErrorMemsetFloat2;
	OpenCLCreateKernelErrors[32] = createKernelErrorIdentityMatrix;
	OpenCLCreateKernelErrors[33] = createKernelErrorIdentityMatrixDouble;
	OpenCLCreateKernelErrors[34] = createKernelErrorGetSubMatrix;
	OpenCLCreateKernelErrors[35] = createKernelErrorGetSubMatrixDouble;
	OpenCLCreateKernelErrors[36] = createKernelErrorPermuteMatrix;
	OpenCLCreateKernelErrors[37] = createKernelErrorPermuteMatrixDouble;
	OpenCLCreateKernelErrors[38] = createKernelErrorLogitMatrix;
	OpenCLCreateKernelErrors[39] = createKernelErrorLogitMatrixDouble;
	OpenCLCreateKernelErrors[40] = createKernelErrorMultiplyVolume;
	OpenCLCreateKernelErrors[41] = createKernelErrorMultiplyVolumes;
	OpenCLCreateKernelErrors[42] = createKernelErrorMultiplyVolumesOverwrite;
	OpenCLCreateKernelErrors[43] = createKernelErrorMultiplyVolumesOverwriteDouble;
	OpenCLCreateKernelErrors[44] = createKernelErrorAddVolume;
	OpenCLCreateKernelErrors[45] = createKernelErrorAddVolumes;
	OpenCLCreateKernelErrors[46] = createKernelErrorAddVolumesOverwrite;
	OpenCLCreateKernelErrors[47] = createKernelErrorSubtractVolumes;
	OpenCLCreateKernelErrors[48] = createKernelErrorSubtractVolumesOverwrite;
	OpenCLCreateKernelErrors[49] = createKernelErrorSubtractVolumesOverwriteDouble;
	OpenCLCreateKernelErrors[50] = createKernelErrorRemoveMean;

	OpenCLCreateKernelErrors[51] = createKernelErrorInterpolateVolumeNearestLinear;
	OpenCLCreateKernelErrors[52] = createKernelErrorInterpolateVolumeLinearLinear;
	OpenCLCreateKernelErrors[53] = createKernelErrorInterpolateVolumeCubicLinear;
	OpenCLCreateKernelErrors[54] = createKernelErrorInterpolateVolumeNearestNonLinear;
	OpenCLCreateKernelErrors[55] = createKernelErrorInterpolateVolumeLinearNonLinear;
	OpenCLCreateKernelErrors[56] = createKernelErrorInterpolateVolumeCubicNonLinear;
	OpenCLCreateKernelErrors[57] = createKernelErrorRescaleVolumeLinear;
	OpenCLCreateKernelErrors[58] = createKernelErrorRescaleVolumeCubic;
	OpenCLCreateKernelErrors[59] = createKernelErrorRescaleVolumeNearest;
	OpenCLCreateKernelErrors[60] = createKernelErrorCopyT1VolumeToMNI;
	OpenCLCreateKernelErrors[61] = createKernelErrorCopyEPIVolumeToT1;
	OpenCLCreateKernelErrors[62] = createKernelErrorCopyVolumeToNew;

	OpenCLCreateKernelErrors[63] = createKernelErrorSetStartClusterIndices;
	OpenCLCreateKernelErrors[64] = createKernelErrorClusterizeScan;
	OpenCLCreateKernelErrors[65] = createKernelErrorClusterizeRelabel;
	OpenCLCreateKernelErrors[66] = createKernelErrorCalculateClusterSizes;
	OpenCLCreateKernelErrors[67] = createKernelErrorCalculateClusterMasses;
	OpenCLCreateKernelErrors[68] = createKernelErrorCalculateLargestCluster;
	OpenCLCreateKernelErrors[69] = createKernelErrorCalculateTFCEValues;
	OpenCLCreateKernelErrors[70] = createKernelErrorCalculatePermutationPValuesVoxelLevelInference;
	OpenCLCreateKernelErrors[71] = createKernelErrorCalculatePermutationPValuesClusterExtentInference;
	OpenCLCreateKernelErrors[72] = createKernelErrorCalculatePermutationPValuesClusterMassInference;

	OpenCLCreateKernelErrors[73] = createKernelErrorCalculateBetaWeightsGLM;
	OpenCLCreateKernelErrors[74] = createKernelErrorCalculateBetaWeightsGLMSlice;
	OpenCLCreateKernelErrors[75] = createKernelErrorCalculateBetaWeightsAndContrastsGLM;
	OpenCLCreateKernelErrors[76] = createKernelErrorCalculateBetaWeightsAndContrastsGLMSlice;
	OpenCLCreateKernelErrors[77] = createKernelErrorCalculateBetaWeightsGLMFirstLevel;
	OpenCLCreateKernelErrors[78] = createKernelErrorCalculateBetaWeightsGLMFirstLevelSlice;
	OpenCLCreateKernelErrors[79] = createKernelErrorCalculateGLMResiduals;
	OpenCLCreateKernelErrors[80] = createKernelErrorCalculateGLMResidualsSlice;
	OpenCLCreateKernelErrors[81] = createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel;
	OpenCLCreateKernelErrors[82] = createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevel;
	OpenCLCreateKernelErrors[83] = createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelSlice;
	OpenCLCreateKernelErrors[84] = createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelSlice;
	OpenCLCreateKernelErrors[85] = createKernelErrorCalculateStatisticalMapsGLMTTest;
	OpenCLCreateKernelErrors[86] = createKernelErrorCalculateStatisticalMapsGLMFTest;
	OpenCLCreateKernelErrors[87] = createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelPermutation;
	OpenCLCreateKernelErrors[88] = createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelPermutation;
	OpenCLCreateKernelErrors[89] = createKernelErrorCalculateStatisticalMapsGLMTTestSecondLevelPermutation;
	OpenCLCreateKernelErrors[90] = createKernelErrorCalculateStatisticalMapsGLMFTestSecondLevelPermutation;
	OpenCLCreateKernelErrors[91] = createKernelErrorCalculateStatisticalMapsMeanSecondLevelPermutation;
	OpenCLCreateKernelErrors[92] = createKernelErrorTransformData;
	OpenCLCreateKernelErrors[93] = createKernelErrorRemoveLinearFit;
	OpenCLCreateKernelErrors[94] = createKernelErrorRemoveLinearFitSlice;

	OpenCLCreateKernelErrors[95] = createKernelErrorCalculateStatisticalMapsGLMBayesian;

	OpenCLCreateKernelErrors[96] = createKernelErrorEstimateAR4Models;
	OpenCLCreateKernelErrors[97] = createKernelErrorEstimateAR4ModelsSlice;
	OpenCLCreateKernelErrors[98] = createKernelErrorApplyWhiteningAR4;
	OpenCLCreateKernelErrors[99] = createKernelErrorApplyWhiteningAR4Slice;
	OpenCLCreateKernelErrors[100] = createKernelErrorGeneratePermutedVolumesFirstLevel;
    
    OpenCLCreateKernelErrors[101] = createKernelErrorCalculateStatisticalMapSearchlight;
    
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
	OpenCLRunKernelErrors[28] = runKernelErrorMemset;
	OpenCLRunKernelErrors[29] = runKernelErrorMemsetDouble;
	OpenCLRunKernelErrors[30] = runKernelErrorMemsetInt;
	OpenCLRunKernelErrors[31] = runKernelErrorMemsetFloat2;
	OpenCLRunKernelErrors[32] = runKernelErrorIdentityMatrix;
	OpenCLRunKernelErrors[33] = runKernelErrorIdentityMatrixDouble;
	OpenCLRunKernelErrors[34] = runKernelErrorGetSubMatrix;
	OpenCLRunKernelErrors[35] = runKernelErrorGetSubMatrixDouble;
	OpenCLRunKernelErrors[36] = runKernelErrorPermuteMatrix;
	OpenCLRunKernelErrors[37] = runKernelErrorPermuteMatrixDouble;
	OpenCLRunKernelErrors[38] = runKernelErrorLogitMatrix;
	OpenCLRunKernelErrors[39] = runKernelErrorLogitMatrixDouble;
	OpenCLRunKernelErrors[40] = runKernelErrorMultiplyVolume;
	OpenCLRunKernelErrors[41] = runKernelErrorMultiplyVolumes;
	OpenCLRunKernelErrors[42] = runKernelErrorMultiplyVolumesOverwrite;
	OpenCLRunKernelErrors[43] = runKernelErrorMultiplyVolumesOverwriteDouble;
	OpenCLRunKernelErrors[44] = runKernelErrorAddVolume;
	OpenCLRunKernelErrors[45] = runKernelErrorAddVolumes;
	OpenCLRunKernelErrors[46] = runKernelErrorAddVolumesOverwrite;
	OpenCLRunKernelErrors[47] = runKernelErrorSubtractVolumes;
	OpenCLRunKernelErrors[48] = runKernelErrorSubtractVolumesOverwrite;
	OpenCLRunKernelErrors[49] = runKernelErrorSubtractVolumesOverwriteDouble;
	OpenCLRunKernelErrors[50] = runKernelErrorRemoveMean;

	OpenCLRunKernelErrors[51] = runKernelErrorInterpolateVolumeNearestLinear;
	OpenCLRunKernelErrors[52] = runKernelErrorInterpolateVolumeLinearLinear;
	OpenCLRunKernelErrors[53] = runKernelErrorInterpolateVolumeCubicLinear;
	OpenCLRunKernelErrors[54] = runKernelErrorInterpolateVolumeNearestNonLinear;
	OpenCLRunKernelErrors[55] = runKernelErrorInterpolateVolumeLinearNonLinear;
	OpenCLRunKernelErrors[56] = runKernelErrorInterpolateVolumeCubicNonLinear;
	OpenCLRunKernelErrors[57] = runKernelErrorRescaleVolumeLinear;
	OpenCLRunKernelErrors[58] = runKernelErrorRescaleVolumeCubic;
	OpenCLRunKernelErrors[59] = runKernelErrorRescaleVolumeNearest;
	OpenCLRunKernelErrors[60] = runKernelErrorCopyT1VolumeToMNI;
	OpenCLRunKernelErrors[61] = runKernelErrorCopyEPIVolumeToT1;
	OpenCLRunKernelErrors[62] = runKernelErrorCopyVolumeToNew;

	OpenCLRunKernelErrors[63] = runKernelErrorSetStartClusterIndices;
	OpenCLRunKernelErrors[64] = runKernelErrorClusterizeScan;
	OpenCLRunKernelErrors[65] = runKernelErrorClusterizeRelabel;
	OpenCLRunKernelErrors[66] = runKernelErrorCalculateClusterSizes;
	OpenCLRunKernelErrors[67] = runKernelErrorCalculateClusterMasses;
	OpenCLRunKernelErrors[68] = runKernelErrorCalculateLargestCluster;
	OpenCLRunKernelErrors[69] = runKernelErrorCalculateTFCEValues;
	OpenCLRunKernelErrors[70] = runKernelErrorCalculatePermutationPValuesVoxelLevelInference;
	OpenCLRunKernelErrors[71] = runKernelErrorCalculatePermutationPValuesClusterExtentInference;
	OpenCLRunKernelErrors[72] = runKernelErrorCalculatePermutationPValuesClusterMassInference;

	OpenCLRunKernelErrors[73] = runKernelErrorCalculateBetaWeightsGLM;
	OpenCLRunKernelErrors[74] = runKernelErrorCalculateBetaWeightsGLMSlice;
	OpenCLRunKernelErrors[75] = runKernelErrorCalculateBetaWeightsAndContrastsGLM;
	OpenCLRunKernelErrors[76] = runKernelErrorCalculateBetaWeightsAndContrastsGLMSlice;
	OpenCLRunKernelErrors[77] = runKernelErrorCalculateBetaWeightsGLMFirstLevel;
	OpenCLRunKernelErrors[78] = runKernelErrorCalculateBetaWeightsGLMFirstLevelSlice;
	OpenCLRunKernelErrors[79] = runKernelErrorCalculateGLMResiduals;
	OpenCLRunKernelErrors[80] = runKernelErrorCalculateGLMResidualsSlice;
	OpenCLRunKernelErrors[81] = runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel;
	OpenCLRunKernelErrors[82] = runKernelErrorCalculateStatisticalMapsGLMFTestFirstLevel;
	OpenCLRunKernelErrors[83] = runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelSlice;
	OpenCLRunKernelErrors[84] = runKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelSlice;
	OpenCLRunKernelErrors[85] = runKernelErrorCalculateStatisticalMapsGLMTTest;
	OpenCLRunKernelErrors[86] = runKernelErrorCalculateStatisticalMapsGLMFTest;
	OpenCLRunKernelErrors[87] = runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelPermutation;
	OpenCLRunKernelErrors[88] = runKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelPermutation;
	OpenCLRunKernelErrors[89] = runKernelErrorCalculateStatisticalMapsGLMTTestSecondLevelPermutation;
	OpenCLRunKernelErrors[90] = runKernelErrorCalculateStatisticalMapsGLMFTestSecondLevelPermutation;
	OpenCLRunKernelErrors[91] = runKernelErrorCalculateStatisticalMapsMeanSecondLevelPermutation;
	OpenCLRunKernelErrors[92] = runKernelErrorTransformData;
	OpenCLRunKernelErrors[93] = runKernelErrorRemoveLinearFit;
	OpenCLRunKernelErrors[94] = runKernelErrorRemoveLinearFitSlice;

	OpenCLRunKernelErrors[95] = runKernelErrorCalculateStatisticalMapsGLMBayesian;

	OpenCLRunKernelErrors[96] = runKernelErrorEstimateAR4Models;
	OpenCLRunKernelErrors[97] = runKernelErrorEstimateAR4ModelsSlice;
	OpenCLRunKernelErrors[98] = runKernelErrorApplyWhiteningAR4;
	OpenCLRunKernelErrors[99] = runKernelErrorApplyWhiteningAR4Slice;
	OpenCLRunKernelErrors[100] = runKernelErrorGeneratePermutedVolumesFirstLevel;
    
    OpenCLRunKernelErrors[101] = runKernelErrorCalculateStatisticalMapSearchlight;
    
	return OpenCLRunKernelErrors;
}


// Cleans up all the OpenCL variables when the BROCCOLI instance is destroyed
void BROCCOLI_LIB::OpenCLCleanup()
{
	if (OPENCL_INITIATED)
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

		// Release programs, command queue and context
		for (int k = 0; k < NUMBER_OF_KERNEL_FILES; k++)
		{	
			cl_program temp = OpenCLPrograms[k];
			if (temp != NULL)
			{
				clReleaseProgram(temp);
			}
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
	// Separable convolution for 512 threads per thread block
	if ( (maxThreadsPerBlock >= 512) && (maxThreadsPerDimension[0] >= 32) && (maxThreadsPerDimension[1] >= 8) && (maxThreadsPerDimension[2] >= 8) )
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
	// Separable convolution for 256 threads per thread block
	else if ( (maxThreadsPerBlock >= 256) && (maxThreadsPerDimension[0] >= 32) && (maxThreadsPerDimension[1] >= 8) && (maxThreadsPerDimension[2] >= 8) )
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
	// Backup version for global memory
	else if ( (maxThreadsPerBlock >= 64) && (maxThreadsPerDimension[0] >= 64) )
	{
		//----------------------------------
		// Separable convolution rows
		//----------------------------------

		localWorkSizeSeparableConvolutionRows[0] = 64;
		localWorkSizeSeparableConvolutionRows[1] = 1;
		localWorkSizeSeparableConvolutionRows[2] = 1;

		// Calculate how many blocks are required
		xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeSeparableConvolutionRows[0]);
		yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeSeparableConvolutionRows[1]);
		zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeSeparableConvolutionRows[2]);

		// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
		globalWorkSizeSeparableConvolutionRows[0] = xBlocks * localWorkSizeSeparableConvolutionRows[0];
		globalWorkSizeSeparableConvolutionRows[1] = yBlocks * localWorkSizeSeparableConvolutionRows[1];
		globalWorkSizeSeparableConvolutionRows[2] = zBlocks * localWorkSizeSeparableConvolutionRows[2];

		//----------------------------------
		// Separable convolution columns
		//----------------------------------

		localWorkSizeSeparableConvolutionColumns[0] = 64;
		localWorkSizeSeparableConvolutionColumns[1] = 1;
		localWorkSizeSeparableConvolutionColumns[2] = 1;

		// Calculate how many blocks are required
		xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeSeparableConvolutionColumns[0]);
		yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeSeparableConvolutionColumns[1]);
		zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeSeparableConvolutionColumns[2]);

		// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
		globalWorkSizeSeparableConvolutionColumns[0] = xBlocks * localWorkSizeSeparableConvolutionColumns[0];
		globalWorkSizeSeparableConvolutionColumns[1] = yBlocks * localWorkSizeSeparableConvolutionColumns[1];
		globalWorkSizeSeparableConvolutionColumns[2] = zBlocks * localWorkSizeSeparableConvolutionColumns[2];

		//----------------------------------
		// Separable convolution rods
		//----------------------------------

		localWorkSizeSeparableConvolutionRods[0] = 64;
		localWorkSizeSeparableConvolutionRods[1] = 1;
		localWorkSizeSeparableConvolutionRods[2] = 1;

		// Calculate how many blocks are required
		// ConvolutionRods yields 32 * 8 * 8 valid filter responses per block (x,y,z)
		xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeSeparableConvolutionRods[0]);
		yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeSeparableConvolutionRods[1]);
		zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeSeparableConvolutionRods[2]);

		// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
		globalWorkSizeSeparableConvolutionRods[0] = xBlocks * localWorkSizeSeparableConvolutionRods[0];
		globalWorkSizeSeparableConvolutionRods[1] = yBlocks * localWorkSizeSeparableConvolutionRods[1];
		globalWorkSizeSeparableConvolutionRods[2] = zBlocks * localWorkSizeSeparableConvolutionRods[2];
	}
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesMemset(int N)
{
	if (maxThreadsPerDimension[1] >= 64)
	{
		localWorkSizeMemset[0] = 256;
		localWorkSizeMemset[1] = 1;
		localWorkSizeMemset[2] = 1;
	}
	else
	{
		localWorkSizeMemset[0] = 64;
		localWorkSizeMemset[1] = 1;
		localWorkSizeMemset[2] = 1;	
	}

	xBlocks = (size_t)ceil((float)(N) / (float)localWorkSizeMemset[0]);

	globalWorkSizeMemset[0] = xBlocks * localWorkSizeMemset[0];
	globalWorkSizeMemset[1] = 1;
	globalWorkSizeMemset[2] = 1;
}

void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesNonSeparableConvolution(int DATA_W, int DATA_H, int DATA_D)
{
	// 512 threads per block, as 32 * 16 threads
	if ( (maxThreadsPerBlock >= 512) && (maxThreadsPerDimension[0] >= 32) && (maxThreadsPerDimension[1] >= 16)  )
	{
		localWorkSizeNonseparableConvolution3DComplex[0] = 32;
		localWorkSizeNonseparableConvolution3DComplex[1] = 16;
		localWorkSizeNonseparableConvolution3DComplex[2] = 1;

		// Calculate how many blocks are required
		xBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_X_CONVOLUTION_2D_32KB);
		yBlocks = (size_t)ceil((float)DATA_H / (float)VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D_32KB);
		zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeNonseparableConvolution3DComplex[2]);

		// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
		globalWorkSizeNonseparableConvolution3DComplex[0] = xBlocks * localWorkSizeNonseparableConvolution3DComplex[0];
		globalWorkSizeNonseparableConvolution3DComplex[1] = yBlocks * localWorkSizeNonseparableConvolution3DComplex[1];
		globalWorkSizeNonseparableConvolution3DComplex[2] = zBlocks * localWorkSizeNonseparableConvolution3DComplex[2];
	}
	// 1024 threads per block, as 32 * 32 threads
	else if ( (maxThreadsPerBlock >= 1024) && (maxThreadsPerDimension[0] >= 32) && (maxThreadsPerDimension[1] >= 32)  )
	{
		localWorkSizeNonseparableConvolution3DComplex[0] = 32;
		localWorkSizeNonseparableConvolution3DComplex[1] = 32;
		localWorkSizeNonseparableConvolution3DComplex[2] = 1;

		// Calculate how many blocks are required
		xBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_X_CONVOLUTION_2D_24KB);
		yBlocks = (size_t)ceil((float)DATA_H / (float)VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D_24KB);
		zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeNonseparableConvolution3DComplex[2]);

		// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
		globalWorkSizeNonseparableConvolution3DComplex[0] = xBlocks * localWorkSizeNonseparableConvolution3DComplex[0];
		globalWorkSizeNonseparableConvolution3DComplex[1] = yBlocks * localWorkSizeNonseparableConvolution3DComplex[1];
		globalWorkSizeNonseparableConvolution3DComplex[2] = zBlocks * localWorkSizeNonseparableConvolution3DComplex[2];
	}
	// 256 threads per block, as 16 * 16 threads
	else if ( (maxThreadsPerBlock >= 256) && (maxThreadsPerDimension[0] >= 16) && (maxThreadsPerDimension[1] >= 16)  )
	{
		localWorkSizeNonseparableConvolution3DComplex[0] = 16;
		localWorkSizeNonseparableConvolution3DComplex[1] = 16;
		localWorkSizeNonseparableConvolution3DComplex[2] = 1;

		// Calculate how many blocks are required
		xBlocks = (size_t)ceil((float)DATA_W / (float)VALID_FILTER_RESPONSES_X_CONVOLUTION_2D_32KB);
		yBlocks = (size_t)ceil((float)DATA_H / (float)VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D_32KB);
		zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeNonseparableConvolution3DComplex[2]);

		// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
		globalWorkSizeNonseparableConvolution3DComplex[0] = xBlocks * localWorkSizeNonseparableConvolution3DComplex[0];
		globalWorkSizeNonseparableConvolution3DComplex[1] = yBlocks * localWorkSizeNonseparableConvolution3DComplex[1];
		globalWorkSizeNonseparableConvolution3DComplex[2] = zBlocks * localWorkSizeNonseparableConvolution3DComplex[2];
	}
	// Backup version for global memory, 128 threads per block, along one dimension (e.g. for Intel on the Apple platform)
	else if ( (maxThreadsPerBlock >= 64) && (maxThreadsPerDimension[0] >= 64)   )
	{
		localWorkSizeNonseparableConvolution3DComplex[0] = 64;
		localWorkSizeNonseparableConvolution3DComplex[1] = 1;
		localWorkSizeNonseparableConvolution3DComplex[2] = 1;

		// Calculate how many blocks are required
		xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeNonseparableConvolution3DComplex[0]);
		yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeNonseparableConvolution3DComplex[1]);
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

	if (maxThreadsPerDimension[1] >= 16)
	{
		localWorkSizeCalculatePhaseDifferencesAndCertainties[0] = 16;
		localWorkSizeCalculatePhaseDifferencesAndCertainties[1] = 16;
		localWorkSizeCalculatePhaseDifferencesAndCertainties[2] = 1;
	}
	else
	{
		localWorkSizeCalculatePhaseDifferencesAndCertainties[0] = 64;
		localWorkSizeCalculatePhaseDifferencesAndCertainties[1] = 1;
		localWorkSizeCalculatePhaseDifferencesAndCertainties[2] = 1;
	}

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

	if (maxThreadsPerDimension[1] >= 16)
	{
		localWorkSizeCalculatePhaseGradients[0] = 16;
		localWorkSizeCalculatePhaseGradients[1] = 16;
		localWorkSizeCalculatePhaseGradients[2] = 1;
	}
	else
	{
		localWorkSizeCalculatePhaseGradients[0] = 64;
		localWorkSizeCalculatePhaseGradients[1] = 1;
		localWorkSizeCalculatePhaseGradients[2] = 1;
	}

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

	if (maxThreadsPerDimension[1] >= 16)
	{
		localWorkSizeCalculateTensorNorms[0] = 16;
		localWorkSizeCalculateTensorNorms[1] = 16;
		localWorkSizeCalculateTensorNorms[2] = 1;
	}
	else
	{
		localWorkSizeCalculateTensorNorms[0] = 64;
		localWorkSizeCalculateTensorNorms[1] = 1;
		localWorkSizeCalculateTensorNorms[2] = 1;
	}

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

	if (maxThreadsPerDimension[1] >= 16)
	{
		localWorkSizeCalculateDisplacementAndCertaintyUpdate[0] = 16;
		localWorkSizeCalculateDisplacementAndCertaintyUpdate[1] = 16;
		localWorkSizeCalculateDisplacementAndCertaintyUpdate[2] = 1;
	}
	else
	{
		localWorkSizeCalculateDisplacementAndCertaintyUpdate[0] = 64;
		localWorkSizeCalculateDisplacementAndCertaintyUpdate[1] = 1;
		localWorkSizeCalculateDisplacementAndCertaintyUpdate[2] = 1;
	}

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

	if (maxThreadsPerDimension[1] >= 16)
	{
		localWorkSizeCalculateAMatricesAndHVectors[0] = 16;
		localWorkSizeCalculateAMatricesAndHVectors[1] = 16;
		localWorkSizeCalculateAMatricesAndHVectors[2] = 1;
	}
	else
	{
		localWorkSizeCalculateAMatricesAndHVectors[0] = 64;
		localWorkSizeCalculateAMatricesAndHVectors[1] = 1;
		localWorkSizeCalculateAMatricesAndHVectors[2] = 1;
	}

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
	if (maxThreadsPerDimension[1] >= 16)
	{
		localWorkSizeClusterize[0] = 16;
		localWorkSizeClusterize[1] = 16;
		localWorkSizeClusterize[2] = 1;
	}
	else
	{
		localWorkSizeClusterize[0] = 64;
		localWorkSizeClusterize[1] = 1;
		localWorkSizeClusterize[2] = 1;
	}

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
	if (maxThreadsPerDimension[1] >= 16)
	{
		localWorkSizeInterpolateVolume[0] = 16;
		localWorkSizeInterpolateVolume[1] = 16;
		localWorkSizeInterpolateVolume[2] = 1;
	}
	else
	{
		localWorkSizeInterpolateVolume[0] = 64;
		localWorkSizeInterpolateVolume[1] = 1;
		localWorkSizeInterpolateVolume[2] = 1;
	}

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
	if (maxThreadsPerDimension[1] >= 16)
	{
		localWorkSizeCopyVolumeToNew[0] = 16;
		localWorkSizeCopyVolumeToNew[1] = 16;
		localWorkSizeCopyVolumeToNew[2] = 1;
	}
	else
	{
		localWorkSizeCopyVolumeToNew[0] = 64;
		localWorkSizeCopyVolumeToNew[1] = 1;
		localWorkSizeCopyVolumeToNew[2] = 1;
	}

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
	if (maxThreadsPerDimension[1] >= 16)
	{
		localWorkSizeMultiplyVolumes[0] = 16;
		localWorkSizeMultiplyVolumes[1] = 16;
		localWorkSizeMultiplyVolumes[2] = 1;
	}
	else
	{
		localWorkSizeMultiplyVolumes[0] = 64;
		localWorkSizeMultiplyVolumes[1] = 1;
		localWorkSizeMultiplyVolumes[2] = 1;
	}

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
	if (maxThreadsPerDimension[1] >= 16)
	{
		localWorkSizeAddVolumes[0] = 16;
		localWorkSizeAddVolumes[1] = 16;
		localWorkSizeAddVolumes[2] = 1;
	}
	else
	{
		localWorkSizeAddVolumes[0] = 64;
		localWorkSizeAddVolumes[1] = 1;
		localWorkSizeAddVolumes[2] = 1;
	}

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
	if (maxThreadsPerDimension[1] >= 16)
	{
		localWorkSizeCalculateColumnSums[0] = 16;
		localWorkSizeCalculateColumnSums[1] = 16;
		localWorkSizeCalculateColumnSums[2] = 1;
	}
	else
	{
		localWorkSizeCalculateColumnSums[0] = 64;
		localWorkSizeCalculateColumnSums[1] = 1;
		localWorkSizeCalculateColumnSums[2] = 1;
	}

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
	if (maxThreadsPerDimension[1] >= 8)
	{
		localWorkSizeCalculateMaxAtomic[0] = 32;
		localWorkSizeCalculateMaxAtomic[1] = 8;
		localWorkSizeCalculateMaxAtomic[2] = 1;
	}
	else
	{
		localWorkSizeCalculateMaxAtomic[0] = 64;
		localWorkSizeCalculateMaxAtomic[1] = 1;
		localWorkSizeCalculateMaxAtomic[2] = 1;
	}

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculateMaxAtomic[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculateMaxAtomic[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateMaxAtomic[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculateMaxAtomic[0] = xBlocks * localWorkSizeCalculateMaxAtomic[0];
	globalWorkSizeCalculateMaxAtomic[1] = yBlocks * localWorkSizeCalculateMaxAtomic[1];
	globalWorkSizeCalculateMaxAtomic[2] = zBlocks * localWorkSizeCalculateMaxAtomic[2];


	if (maxThreadsPerDimension[1] >= 16)
	{
		localWorkSizeCalculateColumnMaxs[0] = 16;
		localWorkSizeCalculateColumnMaxs[1] = 16;
		localWorkSizeCalculateColumnMaxs[2] = 1;
	}
	else
	{
		localWorkSizeCalculateColumnMaxs[0] = 64;
		localWorkSizeCalculateColumnMaxs[1] = 1;
		localWorkSizeCalculateColumnMaxs[2] = 1;
	}

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
	if (maxThreadsPerDimension[1] >= 16)
	{
		localWorkSizeCalculateMagnitudes[0] = 16;
		localWorkSizeCalculateMagnitudes[1] = 16;
		localWorkSizeCalculateMagnitudes[2] = 1;
	}
	else
	{
		localWorkSizeCalculateMagnitudes[0] = 64;
		localWorkSizeCalculateMagnitudes[1] = 1;
		localWorkSizeCalculateMagnitudes[2] = 1;
	}

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
	if (maxThreadsPerDimension[1] >= 16)
	{
		localWorkSizeThresholdVolume[0] = 16;
		localWorkSizeThresholdVolume[1] = 16;
		localWorkSizeThresholdVolume[2] = 1;
	}
	else
	{
		localWorkSizeThresholdVolume[0] = 64;
		localWorkSizeThresholdVolume[1] = 1;
		localWorkSizeThresholdVolume[2] = 1;
	}

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
	if (maxThreadsPerDimension[1] >= 8)
	{
		localWorkSizeCalculateBetaWeightsGLM[0] = 32;
		localWorkSizeCalculateBetaWeightsGLM[1] = 8;
		localWorkSizeCalculateBetaWeightsGLM[2] = 1;
	}
	else
	{
		localWorkSizeCalculateBetaWeightsGLM[0] = 1;
		localWorkSizeCalculateBetaWeightsGLM[1] = 1;
		localWorkSizeCalculateBetaWeightsGLM[2] = 1;
	}

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculateBetaWeightsGLM[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculateBetaWeightsGLM[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateBetaWeightsGLM[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculateBetaWeightsGLM[0] = xBlocks * localWorkSizeCalculateBetaWeightsGLM[0];
	globalWorkSizeCalculateBetaWeightsGLM[1] = yBlocks * localWorkSizeCalculateBetaWeightsGLM[1];
	globalWorkSizeCalculateBetaWeightsGLM[2] = zBlocks * localWorkSizeCalculateBetaWeightsGLM[2];

	if (maxThreadsPerDimension[1] >= 8)
	{
		localWorkSizeCalculateStatisticalMapsGLM[0] = 32;
		localWorkSizeCalculateStatisticalMapsGLM[1] = 8;
		localWorkSizeCalculateStatisticalMapsGLM[2] = 1;
	}
	else
	{
		localWorkSizeCalculateStatisticalMapsGLM[0] = 1;
		localWorkSizeCalculateStatisticalMapsGLM[1] = 1;
		localWorkSizeCalculateStatisticalMapsGLM[2] = 1;
	}

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculateStatisticalMapsGLM[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculateStatisticalMapsGLM[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateStatisticalMapsGLM[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculateStatisticalMapsGLM[0] = xBlocks * localWorkSizeCalculateStatisticalMapsGLM[0];
	globalWorkSizeCalculateStatisticalMapsGLM[1] = yBlocks * localWorkSizeCalculateStatisticalMapsGLM[1];
	globalWorkSizeCalculateStatisticalMapsGLM[2] = zBlocks * localWorkSizeCalculateStatisticalMapsGLM[2];

	if (maxThreadsPerDimension[1] >= 8)
	{
		localWorkSizeEstimateAR4Models[0] = 32;
		localWorkSizeEstimateAR4Models[1] = 8;
		localWorkSizeEstimateAR4Models[2] = 1;
	}
	else
	{
		localWorkSizeEstimateAR4Models[0] = 64;
		localWorkSizeEstimateAR4Models[1] = 1;
		localWorkSizeEstimateAR4Models[2] = 1;
	}

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeEstimateAR4Models[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeEstimateAR4Models[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeEstimateAR4Models[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeEstimateAR4Models[0] = xBlocks * localWorkSizeEstimateAR4Models[0];
	globalWorkSizeEstimateAR4Models[1] = yBlocks * localWorkSizeEstimateAR4Models[1];
	globalWorkSizeEstimateAR4Models[2] = zBlocks * localWorkSizeEstimateAR4Models[2];

	if (maxThreadsPerDimension[1] >= 8)
	{
		localWorkSizeApplyWhiteningAR4[0] = 32;
		localWorkSizeApplyWhiteningAR4[1] = 8;
		localWorkSizeApplyWhiteningAR4[2] = 1;
	}
	else
	{
		localWorkSizeApplyWhiteningAR4[0] = 64;
		localWorkSizeApplyWhiteningAR4[1] = 1;
		localWorkSizeApplyWhiteningAR4[2] = 1;
	}

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeApplyWhiteningAR4[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeApplyWhiteningAR4[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeApplyWhiteningAR4[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeApplyWhiteningAR4[0] = xBlocks * localWorkSizeApplyWhiteningAR4[0];
	globalWorkSizeApplyWhiteningAR4[1] = yBlocks * localWorkSizeApplyWhiteningAR4[1];
	globalWorkSizeApplyWhiteningAR4[2] = zBlocks * localWorkSizeApplyWhiteningAR4[2];

	if (maxThreadsPerDimension[1] >= 8)
	{
		localWorkSizeGeneratePermutedVolumesFirstLevel[0] = 32;
		localWorkSizeGeneratePermutedVolumesFirstLevel[1] = 8;
		localWorkSizeGeneratePermutedVolumesFirstLevel[2] = 1;
	}
	else
	{
		localWorkSizeGeneratePermutedVolumesFirstLevel[0] = 64;
		localWorkSizeGeneratePermutedVolumesFirstLevel[1] = 1;
		localWorkSizeGeneratePermutedVolumesFirstLevel[2] = 1;
	}

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeGeneratePermutedVolumesFirstLevel[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeGeneratePermutedVolumesFirstLevel[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeGeneratePermutedVolumesFirstLevel[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeGeneratePermutedVolumesFirstLevel[0] = xBlocks * localWorkSizeGeneratePermutedVolumesFirstLevel[0];
	globalWorkSizeGeneratePermutedVolumesFirstLevel[1] = yBlocks * localWorkSizeGeneratePermutedVolumesFirstLevel[1];
	globalWorkSizeGeneratePermutedVolumesFirstLevel[2] = zBlocks * localWorkSizeGeneratePermutedVolumesFirstLevel[2];

	if (maxThreadsPerDimension[1] >= 8)
	{
		localWorkSizeRemoveLinearFit[0] = 32;
		localWorkSizeRemoveLinearFit[1] = 8;
		localWorkSizeRemoveLinearFit[2] = 1;
	}
	else
	{
		localWorkSizeRemoveLinearFit[0] = 64;
		localWorkSizeRemoveLinearFit[1] = 1;
		localWorkSizeRemoveLinearFit[2] = 1;
	}

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeRemoveLinearFit[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeRemoveLinearFit[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeRemoveLinearFit[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeRemoveLinearFit[0] = xBlocks * localWorkSizeRemoveLinearFit[0];
	globalWorkSizeRemoveLinearFit[1] = yBlocks * localWorkSizeRemoveLinearFit[1];
	globalWorkSizeRemoveLinearFit[2] = zBlocks * localWorkSizeRemoveLinearFit[2];

	if (maxThreadsPerDimension[1] >= 8)
	{
		localWorkSizeCalculatePermutationPValues[0] = 32;
		localWorkSizeCalculatePermutationPValues[1] = 8;
		localWorkSizeCalculatePermutationPValues[2] = 1;
	}
	else
	{
		localWorkSizeCalculatePermutationPValues[0] = 64;
		localWorkSizeCalculatePermutationPValues[1] = 1;
		localWorkSizeCalculatePermutationPValues[2] = 1;
	}

	// Calculate how many blocks are required
	xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculatePermutationPValues[0]);
	yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculatePermutationPValues[1]);
	zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculatePermutationPValues[2]);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	globalWorkSizeCalculatePermutationPValues[0] = xBlocks * localWorkSizeCalculatePermutationPValues[0];
	globalWorkSizeCalculatePermutationPValues[1] = yBlocks * localWorkSizeCalculatePermutationPValues[1];
	globalWorkSizeCalculatePermutationPValues[2] = zBlocks * localWorkSizeCalculatePermutationPValues[2];
}




void BROCCOLI_LIB::SetGlobalAndLocalWorkSizesSearchlight(int DATA_W, int DATA_H, int DATA_D)
{
    localWorkSizeCalculateStatisticalMapSearchlight[0] = 32;
    localWorkSizeCalculateStatisticalMapSearchlight[1] = 16;
    localWorkSizeCalculateStatisticalMapSearchlight[2] = 1;
    
    // Calculate how many blocks are required
    xBlocks = (size_t)ceil((float)DATA_W / (float)localWorkSizeCalculateStatisticalMapSearchlight[0]);
    yBlocks = (size_t)ceil((float)DATA_H / (float)localWorkSizeCalculateStatisticalMapSearchlight[1]);
    zBlocks = (size_t)ceil((float)DATA_D / (float)localWorkSizeCalculateStatisticalMapSearchlight[2]);
    
    // Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
    globalWorkSizeCalculateStatisticalMapSearchlight[0] = xBlocks * localWorkSizeCalculateStatisticalMapSearchlight[0];
    globalWorkSizeCalculateStatisticalMapSearchlight[1] = yBlocks * localWorkSizeCalculateStatisticalMapSearchlight[1];
    globalWorkSizeCalculateStatisticalMapSearchlight[2] = zBlocks * localWorkSizeCalculateStatisticalMapSearchlight[2];
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

void BROCCOLI_LIB::SetInputCertainty(float* data)
{
	h_Certainty = data;
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

void BROCCOLI_LIB::SetNumberOfSubjects(size_t N)
{
	NUMBER_OF_SUBJECTS = N;
}

void BROCCOLI_LIB::SetNumberOfSubjectsGroup1(int *N)
{
	NUMBER_OF_SUBJECTS_IN_GROUP1 = N;
}

void BROCCOLI_LIB::SetNumberOfSubjectsGroup2(int *N)
{
	NUMBER_OF_SUBJECTS_IN_GROUP2 = N;
}

void BROCCOLI_LIB::SetMask(float* data)
{
	h_Mask = data;
}

void BROCCOLI_LIB::SetEPIMask(float* data)
{
	h_EPI_Mask = data;
}

void BROCCOLI_LIB::SetAutoMask(bool mask)
{
	AUTO_MASK = mask;
}

void BROCCOLI_LIB::SetZScore(bool value)
{
	Z_SCORE = value;
}

void BROCCOLI_LIB::SetSmoothedEPIMask(float* data)
{
	h_Smoothed_EPI_Mask = data;
}


void BROCCOLI_LIB::SetTemporalDerivatives(size_t N)
{
	USE_TEMPORAL_DERIVATIVES = N;
}

void BROCCOLI_LIB::SetRegressConfounds(size_t R)
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

void BROCCOLI_LIB::SetBayesian(bool B)
{
	BAYESIAN = B;
}

void BROCCOLI_LIB::SetRegressOnly(bool R)
{
	REGRESS_ONLY = R;
}

void BROCCOLI_LIB::SetPreprocessingOnly(bool P)
{
	PREPROCESSING_ONLY = P;
}

void BROCCOLI_LIB::SetBetasOnly(bool R)
{
	BETAS_ONLY = R;
}

void BROCCOLI_LIB::SetContrastsOnly(bool R)
{
	CONTRASTS_ONLY = R;
}

void BROCCOLI_LIB::SetBetasAndContrastsOnly(bool R)
{
	BETAS_AND_CONTRASTS_ONLY = R;
}

void BROCCOLI_LIB::SetRegressMotion(size_t R)
{
	REGRESS_MOTION = R;
}

void BROCCOLI_LIB::SetRegressGlobalMean(size_t R)
{
	REGRESS_GLOBALMEAN = R;
}

void BROCCOLI_LIB::SetNumberOfConfoundRegressors(size_t N)
{
	NUMBER_OF_CONFOUND_REGRESSORS = N;
}

void BROCCOLI_LIB::SetNumberOfGLMRegressors(size_t N)
{
	NUMBER_OF_GLM_REGRESSORS = N;
}

void BROCCOLI_LIB::SetNumberOfDetrendingRegressors(size_t N)
{
	NUMBER_OF_DETRENDING_REGRESSORS = N;
}

void BROCCOLI_LIB::SetNumberOfContrasts(size_t N)
{
	NUMBER_OF_CONTRASTS = N;
}

void BROCCOLI_LIB::SetNumberOfICAComponents(int N)
{
	NUMBER_OF_ICA_COMPONENTS = N;
}

void BROCCOLI_LIB::SetVarianceToSaveBeforeICA(double p)
{
	PROPORTION_OF_VARIANCE_TO_SAVE_BEFORE_ICA = p;
}

void BROCCOLI_LIB::SetDesignMatrix(float* data1, float* data2)
{
	h_X_GLM_In = data1;
	h_xtxxt_GLM_In = data2;
}

void BROCCOLI_LIB::SetCorrectClasses(float* data1, float* data2)
{
    h_Correct_Classes_In = data1;
    h_d_In = data2;
}


void BROCCOLI_LIB::SetPermutationMatrix(unsigned short int* matrix)
{
	h_Permutation_Matrix = matrix;
}

void BROCCOLI_LIB::SetPermutationMatrices(unsigned short int** matrix)
{
	h_Permutation_Matrices = matrix;
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

void BROCCOLI_LIB::SetOutputMNIMask(float* data)
{
	h_MNI_Mask = data;
}

void BROCCOLI_LIB::SetGLMScalars(float* data)
{
	h_ctxtxc_GLM_In = data;
}

void BROCCOLI_LIB::SetContrasts(float* data)
{
	h_Contrasts_In = data;
}

void BROCCOLI_LIB::SetNumberOfPermutations(size_t N)
{
	NUMBER_OF_PERMUTATIONS = N;
}

void BROCCOLI_LIB::SetNumberOfGroupPermutations(size_t *N)
{
	NUMBER_OF_PERMUTATIONS_PER_CONTRAST = N;
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

void BROCCOLI_LIB::SetChangeMotionCorrectionReferenceVolume(bool change)
{
	CHANGE_MOTION_CORRECTION_REFERENCE_VOLUME = change;
}

void BROCCOLI_LIB::SetMotionCorrectionReferenceVolume(float* reference)
{
	h_Reference_Volume = reference;
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

void BROCCOLI_LIB::SetGroupDesigns(int *designs)
{
	GROUP_DESIGNS = designs;
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



void BROCCOLI_LIB::SetEPIWidth(size_t w)
{
	EPI_DATA_W = w;
}

void BROCCOLI_LIB::SetEPIHeight(size_t h)
{
	EPI_DATA_H = h;
}

void BROCCOLI_LIB::SetEPIDepth(size_t d)
{
	EPI_DATA_D = d;
}

void BROCCOLI_LIB::SetEPITimepoints(size_t t)
{
	EPI_DATA_T = t;
}

void BROCCOLI_LIB::SetEPITimepointsPerRun(size_t* t)
{
	EPI_DATA_T_PER_RUN = t;
}

void BROCCOLI_LIB::SetNumberOfRuns(size_t r)
{
	NUMBER_OF_RUNS = r;
}

void BROCCOLI_LIB::SetT1Width(size_t w)
{
	T1_DATA_W = w;
}

void BROCCOLI_LIB::SetT1Height(size_t h)
{
	T1_DATA_H = h;
}

void BROCCOLI_LIB::SetT1Depth(size_t d)
{
	T1_DATA_D = d;
}

void BROCCOLI_LIB::SetT1Timepoints(size_t t)
{
	T1_DATA_T = t;
}

void BROCCOLI_LIB::SetMNIWidth(size_t w)
{
	MNI_DATA_W = w;
}

void BROCCOLI_LIB::SetMNIHeight(size_t h)
{
	MNI_DATA_H = h;
}

void BROCCOLI_LIB::SetMNIDepth(size_t d)
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

void BROCCOLI_LIB::SetEPISliceOrder(int value)
{
	SLICE_ORDER = value;
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

void BROCCOLI_LIB::SetSaveDisplacementField(bool value)
{
	WRITE_DISPLACEMENT_FIELD = value;
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

void BROCCOLI_LIB::SetSaveMNIMask(bool value)
{
	WRITE_MNI_MASK = value;
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

void BROCCOLI_LIB::SetSaveResidualsEPI(bool value)
{
	WRITE_RESIDUALS_EPI = value;
}

void BROCCOLI_LIB::SetSaveResidualsMNI(bool value)
{
	WRITE_RESIDUALS_MNI = value;
}

void BROCCOLI_LIB::SetSaveResidualVariances(bool value)
{
	WRITE_RESIDUAL_VARIANCES = value;
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

void BROCCOLI_LIB::SetOutputResidualsEPI(float* data)
{
	h_Residuals_EPI = data;
}

void BROCCOLI_LIB::SetOutputResidualsMNI(float* data)
{
	h_Residuals_MNI = data;
}

void BROCCOLI_LIB::SetOutputfMRIVolumesMNI(float* data)
{
	h_fMRI_Volumes_MNI = data;
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

void BROCCOLI_LIB::SetOutputAlignedEPIVolumeMNILinear(float* aligned)
{
	h_Aligned_EPI_Volume_MNI_Linear = aligned;
}

void BROCCOLI_LIB::SetOutputAlignedEPIVolumeMNINonlinear(float* aligned)
{
	h_Aligned_EPI_Volume_MNI_Nonlinear = aligned;
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

void BROCCOLI_LIB::SetOutputPermutationDistributions(float** output)
{
	h_Permutation_Distributions = output;
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
	return deviceInfo.c_str();
}

std::vector<std::string> BROCCOLI_LIB::GetOpenCLBuildInfo()
{
	return buildInfo;
}

std::vector<std::string> BROCCOLI_LIB::GetKernelFileNames()
{
	return kernelFileNames;
}

int BROCCOLI_LIB::GetNumberOfKernelFiles()
{
	return NUMBER_OF_KERNEL_FILES;
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
	return deviceInfo;
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

std::string BROCCOLI_LIB::GetOpenCLInitializationError()
{
	return INITIALIZATION_ERROR;
}

const char* BROCCOLI_LIB::GetOpenCLError()
{
	return OPENCL_ERROR;
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
size_t BROCCOLI_LIB::GetEPIWidth()
{
	return EPI_DATA_W;
}

// Returns the height dimension (y) of the current fMRI dataset
size_t BROCCOLI_LIB::GetEPIHeight()
{
	return EPI_DATA_H;
}

// Returns the depth dimension (z) of the current fMRI dataset
size_t BROCCOLI_LIB::GetEPIDepth()
{
	return EPI_DATA_D;
}

// Returns the number of timepoints of the current fMRI dataset
size_t BROCCOLI_LIB::GetEPITimepoints()
{
	return EPI_DATA_T;
}

size_t BROCCOLI_LIB::GetT1Width()
{
	return T1_DATA_W;
}

size_t BROCCOLI_LIB::GetT1Height()
{
	return T1_DATA_H;
}

size_t BROCCOLI_LIB::GetT1Depth()
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

int BROCCOLI_LIB::GetNumberOfICAComponents()
{
	return NUMBER_OF_ICA_COMPONENTS;
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

	// Reset complex valued filter responses
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


void BROCCOLI_LIB::SetMemory(cl_mem memory, float value, size_t N)
{
	SetGlobalAndLocalWorkSizesMemset(N);
	clSetKernelArg(MemsetKernel, 0, sizeof(cl_mem), &memory);
	clSetKernelArg(MemsetKernel, 1, sizeof(float), &value);
	clSetKernelArg(MemsetKernel, 2, sizeof(int), &N);
	runKernelErrorMemset = clEnqueueNDRangeKernel(commandQueue, MemsetKernel, 1, NULL, globalWorkSizeMemset, localWorkSizeMemset, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::SetMemoryDouble(cl_mem memory, double value, size_t N)
{
	SetGlobalAndLocalWorkSizesMemset(N);
	clSetKernelArg(MemsetDoubleKernel, 0, sizeof(cl_mem), &memory);
	clSetKernelArg(MemsetDoubleKernel, 1, sizeof(double), &value);
	clSetKernelArg(MemsetDoubleKernel, 2, sizeof(int), &N);
	runKernelErrorMemsetDouble = clEnqueueNDRangeKernel(commandQueue, MemsetDoubleKernel, 1, NULL, globalWorkSizeMemset, localWorkSizeMemset, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::SetMemoryInt(cl_mem memory, int value, size_t N)
{
	SetGlobalAndLocalWorkSizesMemset(N);
	clSetKernelArg(MemsetIntKernel, 0, sizeof(cl_mem), &memory);
	clSetKernelArg(MemsetIntKernel, 1, sizeof(int), &value);
	clSetKernelArg(MemsetIntKernel, 2, sizeof(int), &N);
	runKernelErrorMemsetInt = clEnqueueNDRangeKernel(commandQueue, MemsetIntKernel, 1, NULL, globalWorkSizeMemset, localWorkSizeMemset, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::SetMemoryFloat2(cl_mem memory, float value, size_t N)
{
	SetGlobalAndLocalWorkSizesMemset(N);
	clSetKernelArg(MemsetFloat2Kernel, 0, sizeof(cl_mem), &memory);
	clSetKernelArg(MemsetFloat2Kernel, 1, sizeof(float), &value);
	clSetKernelArg(MemsetFloat2Kernel, 2, sizeof(int), &N);
	runKernelErrorMemsetFloat2 = clEnqueueNDRangeKernel(commandQueue, MemsetFloat2Kernel, 1, NULL, globalWorkSizeMemset, localWorkSizeMemset, 0, NULL, NULL);
	clFinish(commandQueue);
}


// This function is used by all linear registration functions, to setup necessary parameters
void BROCCOLI_LIB::AlignTwoVolumesLinearSetup(int DATA_W, int DATA_H, int DATA_D)
{
	// Set global and local work sizes
	SetGlobalAndLocalWorkSizesImageRegistration(DATA_W, DATA_H, DATA_D);

	// Create a 3D image (texture) for fast interpolation
	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_INTENSITY;

	/*
	cl_image_desc imageDesc;
	imageDesc.image_type = CL_MEM_OBJECT_IMAGE3D;
	imageDesc.image_width = DATA_W;
	imageDesc.image_height = DATA_H;
	imageDesc.image_depth = DATA_D;
	imageDesc.image_row_pitch = 0;
	imageDesc.image_slice_pitch = 0;
	imageDesc.num_mip_levels = 0;
	imageDesc.num_samples = 0;
	imageDesc.buffer = NULL;
	*/

	//d_Original_Volume = clCreateImage(context, CL_MEM_READ_ONLY, &format, &imageDesc, NULL, NULL);

	// Deprecated
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

	deviceMemoryAllocations += 18;

	// original, aligned, reference
	allocatedDeviceMemory += 3 * DATA_W * DATA_H * DATA_D * sizeof(float); 

	// filter responses, 6 complex valued
	allocatedDeviceMemory += 12 * DATA_W * DATA_H * DATA_D * sizeof(float);

	// phase differences, phase certainties, phase gradients
	allocatedDeviceMemory += 3 * DATA_W * DATA_H * DATA_D * sizeof(float);

	// A-matrix and h-vector
	allocatedDeviceMemory += NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float);
	allocatedDeviceMemory += NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float);
	allocatedDeviceMemory += DATA_H * DATA_D * NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS * sizeof(float);
	allocatedDeviceMemory += DATA_D * NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS * sizeof(float);
	allocatedDeviceMemory += DATA_H * DATA_D * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float);
	allocatedDeviceMemory += DATA_D * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float);

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





// This function is the foundation for all the linear image registration functions
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


// This function is used by all non-linear registration functions, to setup necessary parameters
void BROCCOLI_LIB::AlignTwoVolumesNonLinearSetup(int DATA_W, int DATA_H, int DATA_D)
{
	// Set global and local work sizes
	SetGlobalAndLocalWorkSizesImageRegistration(DATA_W, DATA_H, DATA_D);
	// a 3D image (texture) for fast interpolation
	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_INTENSITY;

	/*
	cl_image_desc imageDesc;
	imageDesc.image_type = CL_MEM_OBJECT_IMAGE3D;
	imageDesc.image_width = DATA_W;
	imageDesc.image_height = DATA_H;
	imageDesc.image_depth = DATA_D;
	imageDesc.image_row_pitch = 0;
	imageDesc.image_slice_pitch = 0;
	imageDesc.num_mip_levels = 0;
	imageDesc.num_samples = 0;
	imageDesc.buffer = NULL;
	*/

	//d_Original_Volume = clCreateImage(context, CL_MEM_READ_ONLY, &format, &imageDesc, NULL, NULL);

	// Deprecated
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

	deviceMemoryAllocations += 36;

	// original, aligned, reference
	allocatedDeviceMemory += 3 * DATA_W * DATA_H * DATA_D * sizeof(float); 

	// filter responses, 12 complex valued
	allocatedDeviceMemory += 24 * DATA_W * DATA_H * DATA_D * sizeof(float);

	// tensor components, 6 
	allocatedDeviceMemory += 6 * DATA_W * DATA_H * DATA_D * sizeof(float);

	// A-values
	allocatedDeviceMemory += 6 * DATA_W * DATA_H * DATA_D * sizeof(float);

	// h-values
	allocatedDeviceMemory += 3 * DATA_W * DATA_H * DATA_D * sizeof(float);

	// displacement fields
	allocatedDeviceMemory += 6 * DATA_W * DATA_H * DATA_D * sizeof(float);

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

	AlignTwoVolumesNonLinearCleanup(DATA_W,DATA_H,DATA_D);
}

// This function is the foundation for all the non-linear image registration functions
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

void BROCCOLI_LIB::AlignTwoVolumesNonLinearCleanup(int DATA_W, int DATA_H, int DATA_D)
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

	deviceMemoryDeallocations += 36;

	// original, aligned, reference
	allocatedDeviceMemory -= 3 * DATA_W * DATA_H * DATA_D * sizeof(float); 

	// filter responses, 12 complex valued
	allocatedDeviceMemory -= 24 * DATA_W * DATA_H * DATA_D * sizeof(float);

	// tensor components, 6 
	allocatedDeviceMemory -= 6 * DATA_W * DATA_H * DATA_D * sizeof(float);

	// A-values
	allocatedDeviceMemory -= 6 * DATA_W * DATA_H * DATA_D * sizeof(float);

	// h-values
	allocatedDeviceMemory -= 3 * DATA_W * DATA_H * DATA_D * sizeof(float);

	// displacement fields
	allocatedDeviceMemory -= 6 * DATA_W * DATA_H * DATA_D * sizeof(float);

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

	/*
	cl_image_desc imageDesc;
	imageDesc.image_type = CL_MEM_OBJECT_IMAGE3D;
	imageDesc.image_width = ORIGINAL_DATA_W;
	imageDesc.image_height = ORIGINAL_DATA_H;
	imageDesc.image_depth = ORIGINAL_DATA_D;
	imageDesc.image_row_pitch = 0;
	imageDesc.image_slice_pitch = 0;
	imageDesc.num_mip_levels = 0;
	imageDesc.num_samples = 0;
	imageDesc.buffer = NULL;
	*/

	//cl_mem d_Volume_Texture = clCreateImage(context, CL_MEM_READ_ONLY, &format, &imageDesc, NULL, NULL);

	// Deprecated
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

	/*
	cl_image_desc imageDesc;
	imageDesc.image_type = CL_MEM_OBJECT_IMAGE3D;
	imageDesc.image_width = ORIGINAL_DATA_W;
	imageDesc.image_height = ORIGINAL_DATA_H;
	imageDesc.image_depth = ORIGINAL_DATA_D;
	imageDesc.image_row_pitch = 0;
	imageDesc.image_slice_pitch = 0;
	imageDesc.num_mip_levels = 0;
	imageDesc.num_samples = 0;
	imageDesc.buffer = NULL;
	*/

	//cl_mem d_Volume_Texture = clCreateImage(context, CL_MEM_READ_ONLY, &format, &imageDesc, NULL, NULL);

	// Deprecated
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

// Runs linear registration over several scales, COARSEST_SCALE should be 16, 8, 4, 2 or 1
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
	CURRENT_DATA_W = (int)myround((float)DATA_W/((float)COARSEST_SCALE));
	CURRENT_DATA_H = (int)myround((float)DATA_H/((float)COARSEST_SCALE));
	CURRENT_DATA_D = (int)myround((float)DATA_D/((float)COARSEST_SCALE));

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
			AlignTwoVolumesLinearCleanup(CURRENT_DATA_W,CURRENT_DATA_H,CURRENT_DATA_D);

			// Prepare for the next scale  (the previous scale was current scale, so the next scale is times 2)
			CURRENT_DATA_W = (int)myround((float)DATA_W/((float)current_scale/2.0f));
			CURRENT_DATA_H = (int)myround((float)DATA_H/((float)current_scale/2.0f));
			CURRENT_DATA_D = (int)myround((float)DATA_D/((float)current_scale/2.0f));

			// Setup all parameters and allocate memory on host
			AlignTwoVolumesLinearSetup(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);

			PrintMemoryStatus("Inside align two volumes linear several scales");

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
			AlignTwoVolumesLinearCleanup(CURRENT_DATA_W,CURRENT_DATA_H,CURRENT_DATA_D);

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

// Runs non-linear registration over several scales, COARSEST_SCALE should be 8, 4, 2 or 1
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
	CURRENT_DATA_W = (int)myround((float)DATA_W/((float)COARSEST_SCALE));
	CURRENT_DATA_H = (int)myround((float)DATA_H/((float)COARSEST_SCALE));
	CURRENT_DATA_D = (int)myround((float)DATA_D/((float)COARSEST_SCALE));

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
			AlignTwoVolumesNonLinearCleanup(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);

			// Prepare for the next scale (the previous scale was current scale, so the next scale is times 2)
			CURRENT_DATA_W = (int)myround((float)DATA_W/((float)current_scale/2.0f));
			CURRENT_DATA_H = (int)myround((float)DATA_H/((float)current_scale/2.0f));
			CURRENT_DATA_D = (int)myround((float)DATA_D/((float)current_scale/2.0f));

			float scale_factor = 2.0f;

			// Setup all parameters and allocate memory on host
			AlignTwoVolumesNonLinearSetup(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);

			PrintMemoryStatus("Inside align two volumes non-linear several scales");

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
			AlignTwoVolumesNonLinearCleanup(CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D);
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
void BROCCOLI_LIB::AlignTwoVolumesLinearCleanup(int DATA_W, int DATA_H, int DATA_D)
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

	deviceMemoryDeallocations += 18;

	// original, aligned, reference
	allocatedDeviceMemory -= 3 * DATA_W * DATA_H * DATA_D * sizeof(float); 

	// filter responses, 6 complex valued
	allocatedDeviceMemory -= 12 * DATA_W * DATA_H * DATA_D * sizeof(float);

	// phase differences, phase certainties, phase gradients
	allocatedDeviceMemory -= 3 * DATA_W * DATA_H * DATA_D * sizeof(float);

	// A-matrix and h-vector
	allocatedDeviceMemory -= NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float);
	allocatedDeviceMemory -= NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float);
	allocatedDeviceMemory -= DATA_H * DATA_D * NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS * sizeof(float);
	allocatedDeviceMemory -= DATA_D * NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS * sizeof(float);
	allocatedDeviceMemory -= DATA_H * DATA_D * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float);
	allocatedDeviceMemory -= DATA_D * NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS * sizeof(float);
}





void BROCCOLI_LIB::CalculateGlobalMeans(float* h_Volumes)
{
	// Allocate temporary memory for mask
    float *h_Temp_Mask = (float*)malloc(EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float));

    // Copy the mask volume to host
	clEnqueueReadBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Temp_Mask, 0, NULL, NULL);

	// Loop over timepoints
    for (int t = 0; t < EPI_DATA_T; t++)
    {
		int	brainVoxels = 0;

	    float sum = 0.0f;
	    for (int i = 0; i < (EPI_DATA_W * EPI_DATA_H * EPI_DATA_D); i++)
	    {
			// Only use brain voxels
			if (h_Temp_Mask[i] == 1.0f)
			{
		    	sum += h_Volumes[i + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
				brainVoxels++;
			}
	    }
    
    	h_Global_Mean[t] = sum / (float)(brainVoxels);
	}

	free(h_Temp_Mask);
}


void BROCCOLI_LIB::CalculateCenterOfMass(float &rx, float &ry, float &rz, cl_mem d_Volume, size_t DATA_W, size_t DATA_H, size_t DATA_D)
{
    float *h_Temp_Volume = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));

    // Copy the volume to host
	clEnqueueReadBuffer(commandQueue, d_Volume, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Temp_Volume, 0, NULL, NULL);

    float totalMass = 0.0f;
    float mass = 0.0f;

    rx = 0.0f;
    ry = 0.0f;
    rz = 0.0f;

    for (size_t z = 0; z < DATA_D; z++)
    {
        for (size_t y = 0; y < DATA_H; y++)
        {
            for (size_t x = 0; x < DATA_W; x++)
            {
                mass = h_Temp_Volume[x + y * DATA_W + z * DATA_W * DATA_H];
                rx += mass * (float)x;
                ry += mass * (float)y;
                rz += mass * (float)z;
                totalMass += mass;
            }
        }
    }

    rx /= totalMass;
    ry /= totalMass;
    rz /= totalMass;

    free(h_Temp_Volume);
}



// Alters a volume such that the center of mass is in the middle of the volume
void BROCCOLI_LIB::CenterVolumeMass(cl_mem d_Volume,
                                    size_t DATA_W,
                                    size_t DATA_H,
                                    size_t DATA_D)

{
    float xCenter, yCenter, zCenter;
    CalculateCenterOfMass(xCenter, yCenter, zCenter, d_Volume, DATA_W, DATA_H, DATA_D);

    float xTrueCenter = (float)(DATA_W)/2.0f;
    float yTrueCenter = (float)(DATA_H)/2.0f;
    float zTrueCenter = (float)(DATA_D)/2.0f;
    
    // Calculate difference, myround to integers to avoid interpolation
    float xMassCenterDifference = myround(xTrueCenter - xCenter);
    float yMassCenterDifference = myround(yTrueCenter - yCenter);
    float zMassCenterDifference = myround(zTrueCenter - zCenter);
    
    float h_Parameters[12];
    h_Parameters[0] = -xMassCenterDifference;
    h_Parameters[1] = -yMassCenterDifference;
    h_Parameters[2] = -zMassCenterDifference;
    h_Parameters[3] = 0.0f;
    h_Parameters[4] = 0.0f;
    h_Parameters[5] = 0.0f;
    h_Parameters[6] = 0.0f;
    h_Parameters[7] = 0.0f;
    h_Parameters[8] = 0.0f;
    h_Parameters[9] = 0.0f;
    h_Parameters[10] = 0.0f;
    h_Parameters[11] = 0.0f;

    // Apply transformation
    TransformVolumesLinear(d_Volume, h_Parameters, DATA_W, DATA_H, DATA_D, 1, LINEAR);
}

// Alters a volume such that the center of mass is in the middle of the volume, returns translation parameters
void BROCCOLI_LIB::CenterVolumeMass(cl_mem d_Volume, 
 									float *h_Parameters, 
                                    size_t DATA_W,
                                    size_t DATA_H,
                                    size_t DATA_D)

{
    float xCenter, yCenter, zCenter;
    CalculateCenterOfMass(xCenter, yCenter, zCenter, d_Volume, DATA_W, DATA_H, DATA_D);

    float xTrueCenter = (float)(DATA_W)/2.0f;
    float yTrueCenter = (float)(DATA_H)/2.0f;
    float zTrueCenter = (float)(DATA_D)/2.0f;
    
    // Calculate difference, myround to integers to avoid interpolation
    float xMassCenterDifference = myround(xTrueCenter - xCenter);
    float yMassCenterDifference = myround(yTrueCenter - yCenter);
    float zMassCenterDifference = myround(zTrueCenter - zCenter);
    
    h_Parameters[0] = -xMassCenterDifference;
    h_Parameters[1] = -yMassCenterDifference;
    h_Parameters[2] = -zMassCenterDifference;
    h_Parameters[3] = 0.0f;
    h_Parameters[4] = 0.0f;
    h_Parameters[5] = 0.0f;
    h_Parameters[6] = 0.0f;
    h_Parameters[7] = 0.0f;
    h_Parameters[8] = 0.0f;
    h_Parameters[9] = 0.0f;
    h_Parameters[10] = 0.0f;
    h_Parameters[11] = 0.0f;

    // Apply transformation
    TransformVolumesLinear(d_Volume, h_Parameters, DATA_W, DATA_H, DATA_D, 1, LINEAR);
}

// Alters volume1 such that its center of mass matches volume2
void BROCCOLI_LIB::MatchVolumeMasses(cl_mem d_Volume_1,
									 cl_mem d_Volume_2,	
                                	 size_t DATA_W,
                                     size_t DATA_H,
                                     size_t DATA_D)

{
    float xCenter1, yCenter1, zCenter1;
    float xCenter2, yCenter2, zCenter2;

    CalculateCenterOfMass(xCenter1, yCenter1, zCenter1, d_Volume_1, DATA_W, DATA_H, DATA_D);
    CalculateCenterOfMass(xCenter2, yCenter2, zCenter2, d_Volume_2, DATA_W, DATA_H, DATA_D);

    // Calculate difference, myround to integers to avoid interpolation
    float xMassCenterDifference = myround(xCenter2 - xCenter1);
    float yMassCenterDifference = myround(yCenter2 - yCenter1);
    float zMassCenterDifference = myround(zCenter2 - zCenter1);
    
    float h_Parameters[12];

    h_Parameters[0] = -xMassCenterDifference;
    h_Parameters[1] = -yMassCenterDifference;
    h_Parameters[2] = -zMassCenterDifference;
    h_Parameters[3] = 0.0f;
    h_Parameters[4] = 0.0f;
    h_Parameters[5] = 0.0f;
    h_Parameters[6] = 0.0f;
    h_Parameters[7] = 0.0f;
    h_Parameters[8] = 0.0f;
    h_Parameters[9] = 0.0f;
    h_Parameters[10] = 0.0f;
    h_Parameters[11] = 0.0f;

    // Apply transformation
    TransformVolumesLinear(d_Volume_1, h_Parameters, DATA_W, DATA_H, DATA_D, 1, LINEAR);
}

// Alters volume1 such that its center of mass matches volume2, saves parameters
void BROCCOLI_LIB::MatchVolumeMasses(cl_mem d_Volume_1,
									 cl_mem d_Volume_2,	
									 float* h_Parameters,
                                	 size_t DATA_W,
                                     size_t DATA_H,
                                     size_t DATA_D)

{
    float xCenter1, yCenter1, zCenter1;
    float xCenter2, yCenter2, zCenter2;

    CalculateCenterOfMass(xCenter1, yCenter1, zCenter1, d_Volume_1, DATA_W, DATA_H, DATA_D);
    CalculateCenterOfMass(xCenter2, yCenter2, zCenter2, d_Volume_2, DATA_W, DATA_H, DATA_D);

    // Calculate difference, myround to integers to avoid interpolation
    float xMassCenterDifference = myround(xCenter2 - xCenter1);
    float yMassCenterDifference = myround(yCenter2 - yCenter1);
    float zMassCenterDifference = myround(zCenter2 - zCenter1);
    
    h_Parameters[0] = -xMassCenterDifference;
    h_Parameters[1] = -yMassCenterDifference;
    h_Parameters[2] = -zMassCenterDifference;
    h_Parameters[3] = 0.0f;
    h_Parameters[4] = 0.0f;
    h_Parameters[5] = 0.0f;
    h_Parameters[6] = 0.0f;
    h_Parameters[7] = 0.0f;
    h_Parameters[8] = 0.0f;
    h_Parameters[9] = 0.0f;
    h_Parameters[10] = 0.0f;
    h_Parameters[11] = 0.0f;

    // Apply transformation
    TransformVolumesLinear(d_Volume_1, h_Parameters, DATA_W, DATA_H, DATA_D, 1, LINEAR);
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
		                                          int INTERPOLATION_MODE,
		                                          int offset)
{
	// Calculate volume size for the same voxel size
	int DATA_W_INTERPOLATED = (int)myround((float)DATA_W * VOXEL_SIZE_X / NEW_VOXEL_SIZE_X);
	int DATA_H_INTERPOLATED = (int)myround((float)DATA_H * VOXEL_SIZE_Y / NEW_VOXEL_SIZE_Y);
	int DATA_D_INTERPOLATED = (int)myround((float)DATA_D * VOXEL_SIZE_Z / NEW_VOXEL_SIZE_Z);

	// Allocate memory for interpolated volume
	cl_mem d_Interpolated_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  DATA_W_INTERPOLATED * DATA_H_INTERPOLATED * DATA_D_INTERPOLATED * sizeof(float), NULL, NULL);

	// Create a 3D image (texture) for fast interpolation
	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_INTENSITY;

	/*
	cl_image_desc imageDesc;
	imageDesc.image_type = CL_MEM_OBJECT_IMAGE3D;
	imageDesc.image_width = DATA_W;
	imageDesc.image_height = DATA_H;
	imageDesc.image_depth = DATA_D;
	imageDesc.image_row_pitch = 0;
	imageDesc.image_slice_pitch = 0;
	imageDesc.num_mip_levels = 0;
	imageDesc.num_samples = 0;
	imageDesc.buffer = NULL;
	*/

	//cl_mem d_Volume_Texture = clCreateImage(context, CL_MEM_READ_ONLY, &format, &imageDesc, NULL, NULL);

	// Deprecated
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
		clEnqueueCopyBufferToImage(commandQueue, d_Volumes, d_Volume_Texture, (volume + offset) * DATA_W * DATA_H * DATA_D * sizeof(float), origin, region, 0, NULL, NULL);

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

void BROCCOLI_LIB::ScaleAffineRegistrationParameters(float* h_Parameters, float OLD_VOXEL_SIZE_X, float OLD_VOXEL_SIZE_Y, float OLD_VOXEL_SIZE_Z, float NEW_VOXEL_SIZE_X, float NEW_VOXEL_SIZE_Y, float NEW_VOXEL_SIZE_Z)
{
	// Put parameters in 4 x 4 affine transformation matrix

	// (p3 p4  p5  tx)
	// (p6 p7  p8  ty)
	// (p9 p10 p11 tz)
	// (0  0   0   1 )

	Eigen::MatrixXd Affine_Matrix(4,4);
	Eigen::MatrixXd Scaling_Matrix(4,4);

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

	Scaling_Matrix(0,0) = NEW_VOXEL_SIZE_X/OLD_VOXEL_SIZE_X;
	Scaling_Matrix(0,1) = 0.0f;
	Scaling_Matrix(0,2) = 0.0f;
	Scaling_Matrix(0,3) = 0.0f;

	Scaling_Matrix(1,0) = 0.0f;
	Scaling_Matrix(1,1) = NEW_VOXEL_SIZE_Y/OLD_VOXEL_SIZE_Y;
	Scaling_Matrix(1,2) = 0.0f;
	Scaling_Matrix(1,3) = 0.0f;

	Scaling_Matrix(2,0) = 0.0f;
	Scaling_Matrix(2,1) = 0.0f;
	Scaling_Matrix(2,2) = NEW_VOXEL_SIZE_Z/OLD_VOXEL_SIZE_Z;
	Scaling_Matrix(2,3) = 0.0f;

	Scaling_Matrix(3,0) = 0.0f;
	Scaling_Matrix(3,1) = 0.0f;
	Scaling_Matrix(3,2) = 0.0f;
	Scaling_Matrix(3,3) = 1.0f;

	Eigen::MatrixXd Total_Matrix = Affine_Matrix * Scaling_Matrix;

	// Subtract ones in the diagonal

	// Put back translation parameters into array
	h_Parameters[0] = (float)Total_Matrix(0,3);
	h_Parameters[1] = (float)Total_Matrix(1,3);
	h_Parameters[2] = (float)Total_Matrix(2,3);

	// First row
	h_Parameters[3] = (float)(Total_Matrix(0,0) - 1.0);
	h_Parameters[4] = (float)Total_Matrix(0,1);
	h_Parameters[5] = (float)Total_Matrix(0,2);

	// Second row
	h_Parameters[6] = (float)Total_Matrix(1,0);
	h_Parameters[7] = (float)(Total_Matrix(1,1) - 1.0);
	h_Parameters[8] = (float)Total_Matrix(1,2);

	// Third row
	h_Parameters[9] = (float)Total_Matrix(2,0);
	h_Parameters[10] = (float)Total_Matrix(2,1);
	h_Parameters[11] = (float)(Total_Matrix(2,2) - 1.0);
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
	Eigen::MatrixXd Inverse_Matrix(4,4);

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
	Inverse_Matrix = Affine_Matrix.inverse();

	// Subtract ones in the diagonal

	// Put back translation parameters into array
	h_Inverse_Parameters[0] = (float)Inverse_Matrix(0,3);
	h_Inverse_Parameters[1] = (float)Inverse_Matrix(1,3);
	h_Inverse_Parameters[2] = (float)Inverse_Matrix(2,3);

	// First row
	h_Inverse_Parameters[3] = (float)(Inverse_Matrix(0,0) - 1.0);
	h_Inverse_Parameters[4] = (float)Inverse_Matrix(0,1);
	h_Inverse_Parameters[5] = (float)Inverse_Matrix(0,2);

	// Second row
	h_Inverse_Parameters[6] = (float)Inverse_Matrix(1,0);
	h_Inverse_Parameters[7] = (float)(Inverse_Matrix(1,1) - 1.0);
	h_Inverse_Parameters[8] = (float)Inverse_Matrix(1,2);

	// Third row
	h_Inverse_Parameters[9] = (float)Inverse_Matrix(2,0);
	h_Inverse_Parameters[10] = (float)Inverse_Matrix(2,1);
	h_Inverse_Parameters[11] = (float)(Inverse_Matrix(2,2) - 1.0);
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
void BROCCOLI_LIB::MultiplyVolume(cl_mem d_Volume, float factor, size_t DATA_W, size_t DATA_H, size_t DATA_D)
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

// Multiplies all values in an array with a factor
void BROCCOLI_LIB::MultiplyArray(cl_mem d_Array, float factor, size_t N)
{
	SetGlobalAndLocalWorkSizesMultiplyVolumes(N, 1, 1);

	int one = 1;
	clSetKernelArg(MultiplyVolumeKernel, 0, sizeof(cl_mem), &d_Array);
	clSetKernelArg(MultiplyVolumeKernel, 1, sizeof(float), &factor);
	clSetKernelArg(MultiplyVolumeKernel, 2, sizeof(int), &N);
	clSetKernelArg(MultiplyVolumeKernel, 3, sizeof(int), &one);
	clSetKernelArg(MultiplyVolumeKernel, 4, sizeof(int), &one);

	runKernelErrorMultiplyVolumes = clEnqueueNDRangeKernel(commandQueue, MultiplyVolumeKernel, 3, NULL, globalWorkSizeMultiplyVolumes, localWorkSizeMultiplyVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

// Multiplies two volumes and saves result in a third volume
void BROCCOLI_LIB::MultiplyVolumes(cl_mem d_Result, cl_mem d_Volume_1, cl_mem d_Volume_2, size_t DATA_W, size_t DATA_H, size_t DATA_D)
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

// Multiplies two arrays and overwrites first array
void BROCCOLI_LIB::MultiplyArrays(cl_mem d_Array_1, cl_mem d_Array_2, size_t N)
{
	SetGlobalAndLocalWorkSizesMultiplyVolumes(N, 1, 1);

	int zero = 0;
	int one = 1;
	clSetKernelArg(MultiplyVolumesOverwriteKernel, 0, sizeof(cl_mem), &d_Array_1);
	clSetKernelArg(MultiplyVolumesOverwriteKernel, 1, sizeof(cl_mem), &d_Array_2);
	clSetKernelArg(MultiplyVolumesOverwriteKernel, 2, sizeof(int), &N);
	clSetKernelArg(MultiplyVolumesOverwriteKernel, 3, sizeof(int), &one);
	clSetKernelArg(MultiplyVolumesOverwriteKernel, 4, sizeof(int), &one);
	clSetKernelArg(MultiplyVolumesOverwriteKernel, 5, sizeof(int), &zero);

	runKernelErrorMultiplyVolumes = clEnqueueNDRangeKernel(commandQueue, MultiplyVolumesOverwriteKernel, 3, NULL, globalWorkSizeMultiplyVolumes, localWorkSizeMultiplyVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::MultiplyArraysDouble(cl_mem d_Array_1, cl_mem d_Array_2, size_t N)
{
	SetGlobalAndLocalWorkSizesMultiplyVolumes(N, 1, 1);

	int zero = 0;
	int one = 1;
	clSetKernelArg(MultiplyVolumesOverwriteDoubleKernel, 0, sizeof(cl_mem), &d_Array_1);
	clSetKernelArg(MultiplyVolumesOverwriteDoubleKernel, 1, sizeof(cl_mem), &d_Array_2);
	clSetKernelArg(MultiplyVolumesOverwriteDoubleKernel, 2, sizeof(int), &N);
	clSetKernelArg(MultiplyVolumesOverwriteDoubleKernel, 3, sizeof(int), &one);
	clSetKernelArg(MultiplyVolumesOverwriteDoubleKernel, 4, sizeof(int), &one);
	clSetKernelArg(MultiplyVolumesOverwriteDoubleKernel, 5, sizeof(int), &zero);

	runKernelErrorMultiplyVolumesOverwriteDouble = clEnqueueNDRangeKernel(commandQueue, MultiplyVolumesOverwriteDoubleKernel, 3, NULL, globalWorkSizeMultiplyVolumes, localWorkSizeMultiplyVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

// Multiplies two volumes and overwrites first volume
void BROCCOLI_LIB::MultiplyVolumes(cl_mem d_Volume_1, cl_mem d_Volume_2, size_t DATA_W, size_t DATA_H, size_t DATA_D)
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
void BROCCOLI_LIB::MultiplyVolumes(cl_mem d_Volume_1, cl_mem d_Volume_2, size_t DATA_W, size_t DATA_H, size_t DATA_D, size_t VOLUMES)
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
void BROCCOLI_LIB::AddVolume(cl_mem d_Volume, float value, size_t DATA_W, size_t DATA_H, size_t DATA_D)
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
void BROCCOLI_LIB::AddVolumes(cl_mem d_Result, cl_mem d_Volume_1, cl_mem d_Volume_2, size_t DATA_W, size_t DATA_H, size_t DATA_D)
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
void BROCCOLI_LIB::AddVolumes(cl_mem d_Volume_1, cl_mem d_Volume_2, size_t DATA_W, size_t DATA_H, size_t DATA_D)
{
	SetGlobalAndLocalWorkSizesAddVolumes(DATA_W, DATA_H, DATA_D);

	clSetKernelArg(AddVolumesOverwriteKernel, 0, sizeof(cl_mem), &d_Volume_1);
	clSetKernelArg(AddVolumesOverwriteKernel, 1, sizeof(cl_mem), &d_Volume_2);
	clSetKernelArg(AddVolumesOverwriteKernel, 2, sizeof(int), &DATA_W);
	clSetKernelArg(AddVolumesOverwriteKernel, 3, sizeof(int), &DATA_H);
	clSetKernelArg(AddVolumesOverwriteKernel, 4, sizeof(int), &DATA_D);

	runKernelErrorAddVolumesOverwrite = clEnqueueNDRangeKernel(commandQueue, AddVolumesOverwriteKernel, 3, NULL, globalWorkSizeAddVolumes, localWorkSizeAddVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

// Subtract values in second array, overwrites the first array
void BROCCOLI_LIB::SubtractArrays(cl_mem d_Array_1, cl_mem d_Array_2, size_t N)
{
	SetGlobalAndLocalWorkSizesAddVolumes(N, 1, 1);
	int one = 1;
	clSetKernelArg(SubtractVolumesOverwriteKernel, 0, sizeof(cl_mem), &d_Array_1);
	clSetKernelArg(SubtractVolumesOverwriteKernel, 1, sizeof(cl_mem), &d_Array_2);
	clSetKernelArg(SubtractVolumesOverwriteKernel, 2, sizeof(int), &N);
	clSetKernelArg(SubtractVolumesOverwriteKernel, 3, sizeof(int), &one);
	clSetKernelArg(SubtractVolumesOverwriteKernel, 4, sizeof(int), &one);

	runKernelErrorSubtractVolumes = clEnqueueNDRangeKernel(commandQueue, SubtractVolumesOverwriteKernel, 3, NULL, globalWorkSizeAddVolumes, localWorkSizeAddVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::SubtractArraysDouble(cl_mem d_Array_1, cl_mem d_Array_2, size_t N)
{
	SetGlobalAndLocalWorkSizesAddVolumes(N, 1, 1);
	int one = 1;
	clSetKernelArg(SubtractVolumesOverwriteDoubleKernel, 0, sizeof(cl_mem), &d_Array_1);
	clSetKernelArg(SubtractVolumesOverwriteDoubleKernel, 1, sizeof(cl_mem), &d_Array_2);
	clSetKernelArg(SubtractVolumesOverwriteDoubleKernel, 2, sizeof(int), &N);
	clSetKernelArg(SubtractVolumesOverwriteDoubleKernel, 3, sizeof(int), &one);
	clSetKernelArg(SubtractVolumesOverwriteDoubleKernel, 4, sizeof(int), &one);

	runKernelErrorSubtractVolumesOverwriteDouble = clEnqueueNDRangeKernel(commandQueue, SubtractVolumesOverwriteDoubleKernel, 3, NULL, globalWorkSizeAddVolumes, localWorkSizeAddVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::LogitMatrix(cl_mem d_Array, size_t N)
{
	SetGlobalAndLocalWorkSizesAddVolumes(N, 1, 1);

	clSetKernelArg(LogitMatrixKernel, 0, sizeof(cl_mem), &d_Array);
	clSetKernelArg(LogitMatrixKernel, 1, sizeof(int), &N);

	runKernelErrorLogitMatrix = clEnqueueNDRangeKernel(commandQueue, LogitMatrixKernel, 3, NULL, globalWorkSizeAddVolumes, localWorkSizeAddVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::LogitMatrixDouble(cl_mem d_Array, size_t N)
{
	SetGlobalAndLocalWorkSizesAddVolumes(N, 1, 1);

	clSetKernelArg(LogitMatrixDoubleKernel, 0, sizeof(cl_mem), &d_Array);
	clSetKernelArg(LogitMatrixDoubleKernel, 1, sizeof(int), &N);

	runKernelErrorLogitMatrixDouble = clEnqueueNDRangeKernel(commandQueue, LogitMatrixDoubleKernel, 3, NULL, globalWorkSizeAddVolumes, localWorkSizeAddVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

// Subtracts two volumes and saves as a third volume
void BROCCOLI_LIB::SubtractVolumes(cl_mem d_Result, cl_mem d_Volume_1, cl_mem d_Volume_2, size_t DATA_W, size_t DATA_H, size_t DATA_D)
{
	SetGlobalAndLocalWorkSizesAddVolumes(DATA_W, DATA_H, DATA_D);

	clSetKernelArg(SubtractVolumesKernel, 0, sizeof(cl_mem), &d_Result);
	clSetKernelArg(SubtractVolumesKernel, 1, sizeof(cl_mem), &d_Volume_1);
	clSetKernelArg(SubtractVolumesKernel, 2, sizeof(cl_mem), &d_Volume_2);
	clSetKernelArg(SubtractVolumesKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(SubtractVolumesKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(SubtractVolumesKernel, 5, sizeof(int), &DATA_D);

	runKernelErrorSubtractVolumes = clEnqueueNDRangeKernel(commandQueue, SubtractVolumesKernel, 3, NULL, globalWorkSizeAddVolumes, localWorkSizeAddVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

// Subtracts two volumes and overwrites the first volume
void BROCCOLI_LIB::SubtractVolumes(cl_mem d_Volume_1, cl_mem d_Volume_2, size_t DATA_W, size_t DATA_H, size_t DATA_D)
{
	SetGlobalAndLocalWorkSizesAddVolumes(DATA_W, DATA_H, DATA_D);

	clSetKernelArg(SubtractVolumesOverwriteKernel, 0, sizeof(cl_mem), &d_Volume_1);
	clSetKernelArg(SubtractVolumesOverwriteKernel, 1, sizeof(cl_mem), &d_Volume_2);
	clSetKernelArg(SubtractVolumesOverwriteKernel, 2, sizeof(int), &DATA_W);
	clSetKernelArg(SubtractVolumesOverwriteKernel, 3, sizeof(int), &DATA_H);
	clSetKernelArg(SubtractVolumesOverwriteKernel, 4, sizeof(int), &DATA_D);

	runKernelErrorSubtractVolumesOverwrite = clEnqueueNDRangeKernel(commandQueue, SubtractVolumesOverwriteKernel, 3, NULL, globalWorkSizeAddVolumes, localWorkSizeAddVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}


void BROCCOLI_LIB::IdentityMatrix(cl_mem d_Matrix, int N)
{
	SetGlobalAndLocalWorkSizesAddVolumes(N, N, 1);

	clSetKernelArg(IdentityMatrixKernel, 0, sizeof(cl_mem), &d_Matrix);
	clSetKernelArg(IdentityMatrixKernel, 1, sizeof(int), &N);

	runKernelErrorIdentityMatrix = clEnqueueNDRangeKernel(commandQueue, IdentityMatrixKernel, 3, NULL, globalWorkSizeAddVolumes, localWorkSizeAddVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::IdentityMatrixDouble(cl_mem d_Matrix, int N)
{
	SetGlobalAndLocalWorkSizesAddVolumes(N, N, 1);

	clSetKernelArg(IdentityMatrixDoubleKernel, 0, sizeof(cl_mem), &d_Matrix);
	clSetKernelArg(IdentityMatrixDoubleKernel, 1, sizeof(int), &N);

	runKernelErrorIdentityMatrixDouble = clEnqueueNDRangeKernel(commandQueue, IdentityMatrixDoubleKernel, 3, NULL, globalWorkSizeAddVolumes, localWorkSizeAddVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

void BROCCOLI_LIB::GetSubMatrix(cl_mem d_Small_Matrix, cl_mem d_Matrix, int startRow, int startColumn, int smallNumberOfRows, int smallNumberOfColumns, int largeNumberOfRows, int largeNumberOfColumns)
{
	SetGlobalAndLocalWorkSizesAddVolumes(smallNumberOfColumns, smallNumberOfRows, 1);

	clSetKernelArg(GetSubMatrixKernel, 0, sizeof(cl_mem), &d_Small_Matrix);
	clSetKernelArg(GetSubMatrixKernel, 1, sizeof(cl_mem), &d_Matrix);
	clSetKernelArg(GetSubMatrixKernel, 2, sizeof(int), &startRow);
	clSetKernelArg(GetSubMatrixKernel, 3, sizeof(int), &startColumn);
	clSetKernelArg(GetSubMatrixKernel, 4, sizeof(int), &smallNumberOfRows);
	clSetKernelArg(GetSubMatrixKernel, 5, sizeof(int), &smallNumberOfColumns);
	clSetKernelArg(GetSubMatrixKernel, 6, sizeof(int), &largeNumberOfRows);
	clSetKernelArg(GetSubMatrixKernel, 7, sizeof(int), &largeNumberOfColumns);

	runKernelErrorGetSubMatrix = clEnqueueNDRangeKernel(commandQueue, GetSubMatrixKernel, 3, NULL, globalWorkSizeAddVolumes, localWorkSizeAddVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}	

void BROCCOLI_LIB::GetSubMatrixDouble(cl_mem d_Small_Matrix, cl_mem d_Matrix, int startRow, int startColumn, int smallNumberOfRows, int smallNumberOfColumns, int largeNumberOfRows, int largeNumberOfColumns)
{
	SetGlobalAndLocalWorkSizesAddVolumes(smallNumberOfColumns, smallNumberOfRows, 1);

	clSetKernelArg(GetSubMatrixDoubleKernel, 0, sizeof(cl_mem), &d_Small_Matrix);
	clSetKernelArg(GetSubMatrixDoubleKernel, 1, sizeof(cl_mem), &d_Matrix);
	clSetKernelArg(GetSubMatrixDoubleKernel, 2, sizeof(int), &startRow);
	clSetKernelArg(GetSubMatrixDoubleKernel, 3, sizeof(int), &startColumn);
	clSetKernelArg(GetSubMatrixDoubleKernel, 4, sizeof(int), &smallNumberOfRows);
	clSetKernelArg(GetSubMatrixDoubleKernel, 5, sizeof(int), &smallNumberOfColumns);
	clSetKernelArg(GetSubMatrixDoubleKernel, 6, sizeof(int), &largeNumberOfRows);
	clSetKernelArg(GetSubMatrixDoubleKernel, 7, sizeof(int), &largeNumberOfColumns);

	runKernelErrorGetSubMatrixDouble = clEnqueueNDRangeKernel(commandQueue, GetSubMatrixDoubleKernel, 3, NULL, globalWorkSizeAddVolumes, localWorkSizeAddVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}	


void BROCCOLI_LIB::PermuteMatrix(cl_mem d_Permuted_Matrix, cl_mem d_Matrix, cl_mem d_Permutation, int numberOfRows, int numberOfColumns)
{
	SetGlobalAndLocalWorkSizesAddVolumes(numberOfColumns, numberOfRows, 1);

	clSetKernelArg(PermuteMatrixKernel, 0, sizeof(cl_mem), &d_Permuted_Matrix);
	clSetKernelArg(PermuteMatrixKernel, 1, sizeof(cl_mem), &d_Matrix);
	clSetKernelArg(PermuteMatrixKernel, 2, sizeof(cl_mem), &d_Permutation);
	clSetKernelArg(PermuteMatrixKernel, 3, sizeof(int), &numberOfRows);
	clSetKernelArg(PermuteMatrixKernel, 4, sizeof(int), &numberOfColumns);

	runKernelErrorPermuteMatrix = clEnqueueNDRangeKernel(commandQueue, PermuteMatrixKernel, 3, NULL, globalWorkSizeAddVolumes, localWorkSizeAddVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}


void BROCCOLI_LIB::PermuteMatrixDouble(cl_mem d_Permuted_Matrix, cl_mem d_Matrix, cl_mem d_Permutation, int numberOfRows, int numberOfColumns)
{
	SetGlobalAndLocalWorkSizesAddVolumes(numberOfColumns, numberOfRows, 1);

	clSetKernelArg(PermuteMatrixDoubleKernel, 0, sizeof(cl_mem), &d_Permuted_Matrix);
	clSetKernelArg(PermuteMatrixDoubleKernel, 1, sizeof(cl_mem), &d_Matrix);
	clSetKernelArg(PermuteMatrixDoubleKernel, 2, sizeof(cl_mem), &d_Permutation);
	clSetKernelArg(PermuteMatrixDoubleKernel, 3, sizeof(int), &numberOfRows);
	clSetKernelArg(PermuteMatrixDoubleKernel, 4, sizeof(int), &numberOfColumns);

	runKernelErrorPermuteMatrixDouble = clEnqueueNDRangeKernel(commandQueue, PermuteMatrixDoubleKernel, 3, NULL, globalWorkSizeAddVolumes, localWorkSizeAddVolumes, 0, NULL, NULL);
	clFinish(commandQueue);
}

// Not fully optimized, T1 is of MNI size
void BROCCOLI_LIB::PerformRegistrationEPIT1()
{
	// Reset total registration parameters
	for (int p = 0; p < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; p++)
	{
		h_Registration_Parameters_EPI_T1_Affine[p] = 0.0f;
		h_StartParameters_EPI[p] = 0.0f;
	}

	// Make sure that we start from the center, save the translation parameters
	CenterVolumeMass(d_EPI_Volume, h_StartParameters_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Make a segmentation of the EPI volume first
	//SegmentEPIData(d_EPI_Volume);

	// Interpolate EPI volume to T1 resolution and make sure it has the same size,
	// the registration is performed to the MNI aligned T1 volume, which has MNI size
	ChangeVolumesResolutionAndSize(d_T1_EPI_Volume, d_EPI_Volume, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, 0, INTERPOLATION_MODE, 0);

	// Make sure that the volumes overlap from start, save the translation parameters
	MatchVolumeMasses(d_T1_EPI_Volume, d_Skullstripped_T1_Volume, h_StartParameters_EPI_T1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	
	cl_mem d_T1_EPI_Tensor_Magnitude = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Skullstripped_T1_Tensor_Magnitude = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

/*
	CalculateTensorMagnitude(d_T1_EPI_Tensor_Magnitude, d_T1_EPI_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
	CalculateTensorMagnitude(d_Skullstripped_T1_Tensor_Magnitude, d_Skullstripped_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	// Do the registration between EPI and skullstripped T1 with several scales, first translation
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Translation, h_Rotations, d_T1_EPI_Volume, d_Skullstripped_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, TRANSLATION, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Tensor_Magnitude, h_Registration_Parameters_EPI_T1_Translation, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Translation);

	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Volume, d_Skullstripped_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Tensor_Magnitude, h_Registration_Parameters_EPI_T1_Rigid, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Rigid);

	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Tensor_Magnitude, d_Skullstripped_T1_Tensor_Magnitude, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 2, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Rigid, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Rigid);

	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid, h_Rotations, d_T1_EPI_Tensor_Magnitude, d_Skullstripped_T1_Tensor_Magnitude, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 2, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Rigid, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine,h_Registration_Parameters_EPI_T1_Rigid);
	*/

	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Affine, h_Rotations, d_T1_EPI_Volume, d_Skullstripped_T1_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_EPI_T1, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);

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


void BROCCOLI_LIB::PerformRegistrationEPIT1Original()
{
	// Make sure that we start from the center, save the translation parameters
	//CenterVolumeMass(d_EPI_Volume, h_StartParameters_EPI_Original, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	//CenterVolumeMass(d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D);

	// Reset total registration parameters
	for (int p = 0; p < NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS; p++)
	{
		h_Registration_Parameters_EPI_T1_Affine_Original[p] = 0.0f;
	}

	// Make a segmentation of the EPI volume first
	//SegmentEPIData(d_EPI_Volume);

	// Interpolate EPI volume to T1 resolution and make sure it has the same size,
	ChangeVolumesResolutionAndSize(d_T1_EPI_Volume, d_EPI_Volume, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, 0, INTERPOLATION_MODE, 0);

	// Make sure that the volumes overlap from start, save the translation parameters
	MatchVolumeMasses(d_T1_EPI_Volume, d_T1_Volume, h_StartParameters_EPI_T1_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D);

	cl_mem d_T1_EPI_Tensor_Magnitude = clCreateBuffer(context, CL_MEM_READ_WRITE, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Skullstripped_T1_Tensor_Magnitude = clCreateBuffer(context, CL_MEM_READ_WRITE, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);

	CalculateTensorMagnitude(d_T1_EPI_Tensor_Magnitude, d_T1_EPI_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D);
	CalculateTensorMagnitude(d_Skullstripped_T1_Tensor_Magnitude, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D);

	// Do the registration between EPI and skullstripped T1 with several scales, first translation		
	
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Translation_Original, h_Rotations, d_T1_EPI_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, COARSEST_SCALE_EPI_T1*2, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, TRANSLATION, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Tensor_Magnitude, h_Registration_Parameters_EPI_T1_Translation_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine_Original,h_Registration_Parameters_EPI_T1_Translation_Original);
	

	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid_Original, h_Rotations, d_T1_EPI_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, COARSEST_SCALE_EPI_T1*2, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Tensor_Magnitude, h_Registration_Parameters_EPI_T1_Rigid_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine_Original,h_Registration_Parameters_EPI_T1_Rigid_Original);

	
	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid_Original, h_Rotations, d_T1_EPI_Tensor_Magnitude, d_Skullstripped_T1_Tensor_Magnitude, T1_DATA_W, T1_DATA_H, T1_DATA_D, 2, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, RIGID, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Rigid_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine_Original,h_Registration_Parameters_EPI_T1_Rigid_Original);

	AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_EPI_T1_Rigid_Original, h_Rotations, d_T1_EPI_Tensor_Magnitude, d_Skullstripped_T1_Tensor_Magnitude, T1_DATA_W, T1_DATA_H, T1_DATA_D, 2, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);
	TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_EPI_T1_Rigid_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Affine_Original,h_Registration_Parameters_EPI_T1_Rigid_Original);
		

	clReleaseMemObject(d_T1_EPI_Tensor_Magnitude);
	clReleaseMemObject(d_Skullstripped_T1_Tensor_Magnitude);
}





void BROCCOLI_LIB::PerformRegistrationTwoVolumesWrapper()
{
	deviceMemoryAllocations = 0;
	deviceMemoryDeallocations = 0;
	allocatedDeviceMemory = 0;

	// Allocate memory for input volume, input volume of reference size and referencec volume
	cl_mem d_Input_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Reference_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Input_Volume_Reference_Size = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	allocatedDeviceMemory += T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float);
	allocatedDeviceMemory += 2 * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float);
	deviceMemoryAllocations += 3;

    clEnqueueWriteBuffer(commandQueue, d_Reference_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Volume , 0, NULL, NULL);

	for (int t = 0; t < T1_DATA_T; t++)
	{	
		if (T1_DATA_T > 1)
		{
			printf("Registering volume %i \n",t+1);
		}

		// Copy data to device
	    clEnqueueWriteBuffer(commandQueue, d_Input_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), &h_T1_Volume[t * T1_DATA_W * T1_DATA_H * T1_DATA_D] , 0, NULL, NULL);

		if (NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION > 0)
		{
			// Put input volume in the center of the volume
			CenterVolumeMass(d_Input_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D);
			CenterVolumeMass(d_Input_Volume, h_Center_Parameters, T1_DATA_W, T1_DATA_H, T1_DATA_D);

    		// Change resolution and size of input volume
    		ChangeVolumesResolutionAndSize(d_Input_Volume_Reference_Size, d_Input_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_T1_Z_CUT, INTERPOLATION_MODE, 0);
	
			// Make sure that the two volumes overlap from start
			MatchVolumeMasses(d_Input_Volume_Reference_Size, d_Reference_Volume, h_Match_Parameters, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

			// Copy the interpolated volume to host
			clEnqueueReadBuffer(commandQueue, d_Input_Volume_Reference_Size, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Interpolated_T1_Volume, 0, NULL, NULL);

			// Do Linear registration between the two volumes
			AlignTwoVolumesLinearSeveralScales(h_Registration_Parameters_T1_MNI_Out, h_Rotations, d_Input_Volume_Reference_Size, d_Reference_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION, AFFINE, DO_OVERWRITE, INTERPOLATION_MODE);

			// Copy the linearly aligned volume to host
			clEnqueueReadBuffer(commandQueue, d_Input_Volume_Reference_Size, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), &h_Aligned_T1_Volume_Linear[t * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D], 0, NULL, NULL);
		}
		else
		{
			clEnqueueWriteBuffer(commandQueue, d_Input_Volume_Reference_Size, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);
		}

		AddAffineRegistrationParameters(h_Registration_Parameters_T1_MNI_Out, h_Match_Parameters);

		// Perform non-Linear registration
		if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
		{
			AlignTwoVolumesNonLinearSeveralScales(d_Input_Volume_Reference_Size, d_Reference_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION, DO_OVERWRITE, INTERPOLATION_MODE, KEEP_DISPLACEMENT_FIELD);

			// Do total interpolation in one step, to reduce smoothness
			if (NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION > 0)
			{
				CreateCombinedDisplacementField(h_Registration_Parameters_T1_MNI_Out,d_Total_Displacement_Field_X,d_Total_Displacement_Field_Y,d_Total_Displacement_Field_Z,MNI_DATA_W,MNI_DATA_H,MNI_DATA_D);
	
				ChangeVolumesResolutionAndSize(d_Input_Volume_Reference_Size, d_Input_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_T1_Z_CUT, INTERPOLATION_MODE, 0);
	
				TransformVolumesNonLinear(d_Input_Volume_Reference_Size, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
			}

			// Copy the non-linearly aligned volume to host
			clEnqueueReadBuffer(commandQueue, d_Input_Volume_Reference_Size, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), &h_Aligned_T1_Volume_NonLinear[t * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D], 0, NULL, NULL);

			if (WRITE_DISPLACEMENT_FIELD && (T1_DATA_T == 1))
			{		    	
				// Copy the displacement field to host
				clEnqueueReadBuffer(commandQueue, d_Total_Displacement_Field_X, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Displacement_Field_X, 0, NULL, NULL);
				clEnqueueReadBuffer(commandQueue, d_Total_Displacement_Field_Y, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Displacement_Field_Y, 0, NULL, NULL);
				clEnqueueReadBuffer(commandQueue, d_Total_Displacement_Field_Z, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Displacement_Field_Z, 0, NULL, NULL);
			}
		
			clReleaseMemObject(d_Total_Displacement_Field_X);
			clReleaseMemObject(d_Total_Displacement_Field_Y);
			clReleaseMemObject(d_Total_Displacement_Field_Z);
		}
	}


	if (T1_DATA_T == 1)
	{
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
		else if (DO_SKULLSTRIP_ORIGINAL && (NUMBER_OF_ITERATIONS_FOR_LINEAR_IMAGE_REGISTRATION > 0))
		{
			// Copy brain mask from host
			cl_mem d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
			clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask, 0, NULL, NULL);

			// Copy back the interpolated volume from host
			clEnqueueWriteBuffer(commandQueue, d_Input_Volume_Reference_Size, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Interpolated_T1_Volume , 0, NULL, NULL);

			// Calculate inverse affine transform between T1 and MNI
			InvertAffineRegistrationParameters(h_Inverse_Registration_Parameters, h_Registration_Parameters_T1_MNI_Out);

			// Now apply the inverse transformation between MNI and T1, to transform the MNI brain mask to original T1 space
			TransformVolumesLinear(d_MNI_Brain_Mask, h_Inverse_Registration_Parameters, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, NEAREST);

			// Multiply the interpolated volume with the inverse transformed brain mask
			MultiplyVolumes(d_Input_Volume_Reference_Size, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

			// Copy the skullstripped volume to host
			clEnqueueReadBuffer(commandQueue, d_Input_Volume_Reference_Size, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Skullstripped_T1_Volume, 0, NULL, NULL);

			clReleaseMemObject(d_MNI_Brain_Mask);
		}
	}

	// Cleanup
	clReleaseMemObject(d_Input_Volume);
	clReleaseMemObject(d_Input_Volume_Reference_Size);
	clReleaseMemObject(d_Reference_Volume);

	allocatedDeviceMemory -= T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float);
	allocatedDeviceMemory -= 2 * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float);
	deviceMemoryDeallocations += 3;
}

void BROCCOLI_LIB::CreateCombinedDisplacementField(float* h_Registration_Parameters_,
		                                           cl_mem d_Displacement_Field_X,
		                                           cl_mem d_Displacement_Field_Y,
		                                           cl_mem d_Displacement_Field_Z,
		                                           size_t DATA_W,
		                                           size_t DATA_H,
		                                           size_t DATA_D)
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
	cl_mem d_Input_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * T1_DATA_T * sizeof(float), NULL, NULL);
	cl_mem d_Input_Volume_Reference_Size = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * T1_DATA_T * sizeof(float), NULL, NULL);
	d_Total_Displacement_Field_X = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Total_Displacement_Field_Y = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Total_Displacement_Field_Z = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_Input_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * T1_DATA_T * sizeof(float), h_T1_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_Total_Displacement_Field_X, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Displacement_Field_X , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_Total_Displacement_Field_Y, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Displacement_Field_Y , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_Total_Displacement_Field_Z, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Displacement_Field_Z , 0, NULL, NULL);

	// Change resolution and size of input volume
	ChangeVolumesResolutionAndSize(d_Input_Volume_Reference_Size, d_Input_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, T1_DATA_T, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_T1_Z_CUT, INTERPOLATION_MODE, 0);

	// Apply the transformation
	TransformVolumesNonLinear(d_Input_Volume_Reference_Size, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_DATA_T, INTERPOLATION_MODE);

	// Copy the transformed volume to host
	clEnqueueReadBuffer(commandQueue, d_Input_Volume_Reference_Size, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * T1_DATA_T * sizeof(float), h_Interpolated_T1_Volume, 0, NULL, NULL);

	// Release memory
	clReleaseMemObject(d_Input_Volume);
	clReleaseMemObject(d_Input_Volume_Reference_Size);
	clReleaseMemObject(d_Total_Displacement_Field_X);
	clReleaseMemObject(d_Total_Displacement_Field_Y);
	clReleaseMemObject(d_Total_Displacement_Field_Z);
}

void BROCCOLI_LIB::TransformVolumesLinearWrapper()
{
	// Allocate memory for volumes 
	cl_mem d_Input_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * T1_DATA_T * sizeof(float), NULL, NULL);
	cl_mem d_Input_Volume_Reference_Size = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * T1_DATA_T * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_Input_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * T1_DATA_T * sizeof(float), h_T1_Volume , 0, NULL, NULL);

	// Change resolution and size of input volume
	ChangeVolumesResolutionAndSize(d_Input_Volume_Reference_Size, d_Input_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, T1_DATA_T, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_T1_Z_CUT, INTERPOLATION_MODE, 0);

	// Apply the transformation
	TransformVolumesLinear(d_Input_Volume_Reference_Size, h_Registration_Parameters_T1_MNI_Out, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_DATA_T, INTERPOLATION_MODE);

	// Copy the transformed volume to host
	clEnqueueReadBuffer(commandQueue, d_Input_Volume_Reference_Size, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * T1_DATA_T * sizeof(float), h_Interpolated_T1_Volume, 0, NULL, NULL);

	// Release memory
	clReleaseMemObject(d_Input_Volume);
	clReleaseMemObject(d_Input_Volume_Reference_Size);
}

void BROCCOLI_LIB::CenterVolumesWrapper()
{
	// Allocate memory for volumes 
	cl_mem d_Input_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * T1_DATA_T * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_Input_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * T1_DATA_T * sizeof(float), h_T1_Volume , 0, NULL, NULL);

	CenterVolumeMass(d_Input_Volume, h_Center_Parameters, T1_DATA_W, T1_DATA_H, T1_DATA_D);
	
	// Copy first volume again
	clEnqueueWriteBuffer(commandQueue, d_Input_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);

	// Apply the transformation to all volumes
	TransformVolumesLinear(d_Input_Volume, h_Center_Parameters, T1_DATA_W, T1_DATA_H, T1_DATA_D, T1_DATA_T, INTERPOLATION_MODE);

	// Copy the centered volumes to host
	clEnqueueReadBuffer(commandQueue, d_Input_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * T1_DATA_T * sizeof(float), h_Interpolated_T1_Volume, 0, NULL, NULL);

	// Release memory
	clReleaseMemObject(d_Input_Volume);
}



// Performs registration between one high resolution skullstripped T1 volume and a high resolution skullstripped MNI volume (brain template)
void BROCCOLI_LIB::PerformRegistrationT1MNINoSkullstrip()
{
	// Make sure that we start from the center
	CenterVolumeMass(d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D);

	ChangeVolumesResolutionAndSize(d_MNI_T1_Volume, d_T1_Volume, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_T1_Z_CUT, INTERPOLATION_MODE, 0);

	// Make sure that the volumes overlap from start
	MatchVolumeMasses(d_MNI_T1_Volume, d_MNI_Brain_Volume, h_StartParameters_T1_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	// Copy result to skullstripped T1 volume, which will be used for the EPI-T1 registration
	clEnqueueCopyBuffer(commandQueue, d_MNI_T1_Volume, d_Skullstripped_T1_Volume, 0, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), 0, NULL, NULL);

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

    if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
	{
		// Perform non-Linear registration between registered skullstripped volume and MNI brain volume
		AlignTwoVolumesNonLinearSeveralScales(d_MNI_T1_Volume, d_MNI_Brain_Volume, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, COARSEST_SCALE_T1_MNI, NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION, DO_OVERWRITE, INTERPOLATION_MODE, KEEP_DISPLACEMENT_FIELD);

		if (WRITE_ALIGNED_T1_MNI_NONLINEAR)
		{
			clEnqueueReadBuffer(commandQueue, d_MNI_T1_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_T1_Volume_NonLinear, 0, NULL, NULL);
		}
	}
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
	
	/*
	cl_image_desc imageDesc;
	imageDesc.image_type = CL_MEM_OBJECT_IMAGE3D;
	imageDesc.image_width = DATA_W;
	imageDesc.image_height = DATA_H;
	imageDesc.image_depth = DATA_D;
	imageDesc.image_row_pitch = 0;
	imageDesc.image_slice_pitch = 0;
	imageDesc.num_mip_levels = 0;
	imageDesc.num_samples = 0;
	imageDesc.buffer = NULL;
	*/

	//cl_mem d_Volume_Texture = clCreateImage(context, CL_MEM_READ_ONLY, &format, &imageDesc, NULL, NULL);

	// Deprecated
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
	
	/*
	cl_image_desc imageDesc;
	imageDesc.image_type = CL_MEM_OBJECT_IMAGE3D;
	imageDesc.image_width = DATA_W;
	imageDesc.image_height = DATA_H;
	imageDesc.image_depth = DATA_D;
	imageDesc.image_row_pitch = 0;
	imageDesc.image_slice_pitch = 0;
	imageDesc.num_mip_levels = 0;
	imageDesc.num_samples = 0;
	imageDesc.buffer = NULL;
	*/

	//cl_mem d_Volume_Texture = clCreateImage(context, CL_MEM_READ_ONLY, &format, &imageDesc, NULL, NULL);

	// Deprecated
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

void BROCCOLI_LIB::PrintMemoryStatus(const char* text)
{
	if ((WRAPPER == BASH) && VERBOS)
	{
		printf("\n");
		printf("Code location is %s \n",text);
		//printf("Number of memory allocations is %i  \n",deviceMemoryAllocations);
		//printf("Number of memory deallocations is %i  \n",deviceMemoryDeallocations);
		printf("Total allocated device memory is %lu MB  \n",(unsigned long)(allocatedDeviceMemory/1024/1024));
		printf("Total allocated host memory is %lu MB  \n",(unsigned long)(allocatedHostMemory/1024/1024));
		printf("\n");
	}
}

void BROCCOLI_LIB::PerformFirstLevelAnalysisWrapper()
{
	Eigen::initParallel();

	deviceMemoryAllocations = 0;
	deviceMemoryDeallocations = 0;
	allocatedDeviceMemory = 0;

	// Save the first untouched fMRI volume, to be used for fMRI-T1 registration later (if needed)
	float* h_Temp_fMRI_Volume = (float*)malloc(EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float));
	memcpy(h_Temp_fMRI_Volume, h_fMRI_Volumes, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float));

	hostMemoryAllocations += 1;
	allocatedHostMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);

	//---------------------------------------------------------------------------------------------------------------------------------------
	// T1-MNI registration
	//---------------------------------------------------------------------------------------------------------------------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("\nPerforming registration between T1 and MNI\n");
	}

	// Allocate memory on device for registration
	d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_Brain_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_MNI_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Skullstripped_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	deviceMemoryAllocations += 4;
	allocatedDeviceMemory += T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float);
	allocatedDeviceMemory += 3 * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float);

	PrintMemoryStatus("Before T1-MNI registration");

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Volume , 0, NULL, NULL);

	PerformRegistrationT1MNINoSkullstrip();

	AddAffineRegistrationParameters(h_Registration_Parameters_T1_MNI_Out,h_Registration_Parameters_T1_MNI,h_StartParameters_T1_MNI);

	// Cleanup
	clReleaseMemObject(d_MNI_Brain_Volume);
	clReleaseMemObject(d_T1_Volume);

	deviceMemoryDeallocations += 2;
	allocatedDeviceMemory -= MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float);
	allocatedDeviceMemory -= T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float);

	PrintMemoryStatus("After T1-MNI registration");

	//---------------------------------------------------------------------------------------------------------------------------------------
	// fMRI-T1 registration
	//---------------------------------------------------------------------------------------------------------------------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("Performing registration between fMRI and T1\n");
	}

	// Allocate memory on device
	d_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_T1_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	deviceMemoryAllocations += 2;
	allocatedDeviceMemory += MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float);
	allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);

	PrintMemoryStatus("Before EPI-T1 registration");

	// Copy first fMRI volume to device
	clEnqueueWriteBuffer(commandQueue, d_EPI_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

	PerformRegistrationEPIT1();

	if (WRITE_ALIGNED_EPI_T1)
	{
		clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_EPI_Volume_T1, 0, NULL, NULL);
	}

	if (WRITE_ALIGNED_EPI_MNI)
	{
		TransformVolumesLinear(d_T1_EPI_Volume, h_Registration_Parameters_T1_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_EPI_Volume_MNI_Linear, 0, NULL, NULL);
		if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
		{
			TransformVolumesNonLinear(d_T1_EPI_Volume, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
			clEnqueueReadBuffer(commandQueue, d_T1_EPI_Volume, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Aligned_EPI_Volume_MNI_Nonlinear, 0, NULL, NULL);
		}		
	}

	// Cleanup
	clReleaseMemObject(d_MNI_T1_Volume);
	clReleaseMemObject(d_Skullstripped_T1_Volume);
	clReleaseMemObject(d_EPI_Volume);
	clReleaseMemObject(d_T1_EPI_Volume);

	allocatedDeviceMemory -= 3 * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float);
	allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);

	deviceMemoryDeallocations += 4;

	PrintMemoryStatus("After EPI-T1 registration");	

	// Concatenate transformation between T1 and MNI, and fMRI and T1, to get registration between fMRI and MNI
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_MNI,h_Registration_Parameters_T1_MNI,h_Registration_Parameters_EPI_T1_Affine);

	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Out,h_Registration_Parameters_EPI_T1_Affine,h_StartParameters_EPI_T1);
	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_MNI_Out,h_Registration_Parameters_T1_MNI,h_Registration_Parameters_EPI_T1_Out);

	//---------------------------------------------------------------------------------------------------------------------------------------
	// EPI - T1 original
	//---------------------------------------------------------------------------------------------------------------------------------------

	// Allocate memory on device
	d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
	d_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_T1_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);

	// Copy original T1 volume to device
	clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume, 0, NULL, NULL);

	// Copy first fMRI volume to device
	clEnqueueWriteBuffer(commandQueue, d_EPI_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_fMRI_Volumes, 0, NULL, NULL);

	// Register original fMRI volume to original T1 volume
	PerformRegistrationEPIT1Original();

	// Cleanup
	clReleaseMemObject(d_T1_Volume);
	clReleaseMemObject(d_EPI_Volume);
	clReleaseMemObject(d_T1_EPI_Volume);

	AddAffineRegistrationParameters(h_Registration_Parameters_EPI_T1_Out,h_Registration_Parameters_EPI_T1_Affine_Original,h_StartParameters_EPI_T1_Original);

	//---------------------------------------------------------------------------------------------------------------------------------------
	// Slice timing correction
	//---------------------------------------------------------------------------------------------------------------------------------------

	if (APPLY_SLICE_TIMING_CORRECTION)
	{
		if (SLICE_ORDER != UNDEFINED)
		{
			if ((WRAPPER == BASH) && PRINT)
			{
				printf("Performing slice timing correction \n");
			}

			PrintMemoryStatus("Before slice timing correction");

			PerformSliceTimingCorrectionHost(h_fMRI_Volumes);

			PrintMemoryStatus("After slice timing correction");

			if (WRITE_SLICETIMING_CORRECTED)
			{
				memcpy(h_Slice_Timing_Corrected_fMRI_Volumes, h_fMRI_Volumes, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float));
			}
		}
		else
		{
			if (WRAPPER == BASH)
			{
				printf("Warning: Not performing slice timing correction as the slice order is undefined.\n");
			}
		}
	}


	//---------------------------------------------------------------------------------------------------------------------------------------
	// Motion correction
	//---------------------------------------------------------------------------------------------------------------------------------------

	if (APPLY_MOTION_CORRECTION)
	{
		if ((WRAPPER == BASH) && PRINT)
		{
			printf("Performing motion correction");
			if (!VERBOS)
			{
				printf("\n");	
			}
		}

		PrintMemoryStatus("Before motion correction");

		h_Motion_Parameters = (float*)malloc(EPI_DATA_T * NUMBER_OF_MOTION_REGRESSORS * sizeof(float));
		allocatedHostMemory += EPI_DATA_T * NUMBER_OF_MOTION_REGRESSORS * sizeof(float);
		hostMemoryAllocations += 1;

		PerformMotionCorrectionHost(h_fMRI_Volumes);

		if ((WRAPPER == BASH) && VERBOS)
		{
			printf("\n");
		}

		PrintMemoryStatus("After motion correction");

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
			memcpy(h_Motion_Corrected_fMRI_Volumes, h_fMRI_Volumes, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float));
		}
	}

	//---------------------------------------------------------------------------------------------------------------------------------------
	// Segment EPI data
	//---------------------------------------------------------------------------------------------------------------------------------------

	if ((WRAPPER == BASH) && PRINT)
	{
		printf("Performing EPI segmentation\n");
	}

	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	deviceMemoryAllocations += 1;
	allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);

	SegmentEPIData();

	if (WRITE_EPI_MASK)
	{
		clEnqueueReadBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask, 0, NULL, NULL);

	}
	if (WRITE_MNI_MASK)
	{
		clEnqueueReadBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask, 0, NULL, NULL);
		TransformMaskToMNI();
	}

	//---------------------------------------------------------------------------------------------------------------------------------------
	// Smoothing
	//---------------------------------------------------------------------------------------------------------------------------------------

	d_Smoothed_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	deviceMemoryAllocations += 1;
	allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);

	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, EPI_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	PerformSmoothing(d_Smoothed_EPI_Mask, d_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

	if (APPLY_SMOOTHING)
	{
		if ((WRAPPER == BASH) && PRINT)
		{
			printf("Performing smoothing\n");
		}
	
		PrintMemoryStatus("Before smoothing");

		PerformSmoothingNormalizedHost(h_fMRI_Volumes, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

		PrintMemoryStatus("After smoothing");

		if (WRITE_SMOOTHED)
		{
			memcpy(h_Smoothed_fMRI_Volumes, h_fMRI_Volumes, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float));
		}
	}

	//---------------------------------------------------------------------------------------------------------------------------------------
	// GLM
	//---------------------------------------------------------------------------------------------------------------------------------------

	if (!REGRESS_ONLY && !BAYESIAN && !BETAS_ONLY && !PREPROCESSING_ONLY)
	{
		if ((WRAPPER == BASH) && PRINT)
		{
			printf("Performing statistical analysis\n");
		}

		NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + REGRESS_GLOBALMEAN + NUMBER_OF_CONFOUND_REGRESSORS*REGRESS_CONFOUNDS;

		CalculateNumberOfBrainVoxels(d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

		// Check amount of global memory, compared to required memory
		bool largeMemory = true;
		size_t totalRequiredMemory = allocatedDeviceMemory + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float) * 2 + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float) + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float) * 2 + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float) * 6 + NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float) + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float);
		totalRequiredMemory /= (1024*1024);

		if (totalRequiredMemory > globalMemorySize)
		{
			largeMemory = false;
			if ((WRAPPER == BASH) && VERBOS)
			{
				printf("Cannot run the GLM the whole volume at once, doing slice by slice. Required device memory for GLM is %zu MB, global memory is %zu MB ! \n",totalRequiredMemory,globalMemorySize);
			}
		}
		else
		{
			if ((WRAPPER == BASH) && VERBOS)
			{
				printf("Sufficient memory for running the GLM the whole volume at once! Required device memory for GLM is %zu MB, global memory is %zu MB ! \n",totalRequiredMemory,globalMemorySize);
			}
		}

		// Backup version
		if (!largeMemory)
		{
			// Allocate memory for one slice for all time points, loop over slices to save memory
			d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
			d_Whitened_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
			d_Residuals = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
			allocatedDeviceMemory += 3 * EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float);
			deviceMemoryAllocations += 3;
		}
		else
		{
			d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
			d_Whitened_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
			allocatedDeviceMemory += 2 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);
			deviceMemoryAllocations += 2;
		}

		c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
		c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
		c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
		c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

		d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
		d_Contrast_Volumes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
		d_Statistical_Maps = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
		d_Residual_Variances = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

		deviceMemoryAllocations += 4;
		allocatedDeviceMemory += (EPI_DATA_W * EPI_DATA_H * EPI_DATA_D)*(NUMBER_OF_TOTAL_GLM_REGRESSORS + NUMBER_OF_CONTRASTS + NUMBER_OF_CONTRASTS + 1) * sizeof(float);

		d_AR1_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
		d_AR2_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
		d_AR3_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
		d_AR4_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

		deviceMemoryAllocations += 4;
		allocatedDeviceMemory += 4 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);

		PrintMemoryStatus("Before GLM");

		h_X_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
		h_xtxxt_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
		h_Contrasts = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float));
		h_ctxtxc_GLM = (float*)malloc(NUMBER_OF_CONTRASTS * sizeof(float));
		h_X_GLM_With_Temporal_Derivatives = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * 2 * EPI_DATA_T * sizeof(float));
		h_X_GLM_Convolved = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * (USE_TEMPORAL_DERIVATIVES+1) * EPI_DATA_T * sizeof(float));
		h_Global_Mean = (float*)malloc(EPI_DATA_T * sizeof(float));

		if (REGRESS_GLOBALMEAN)
		{
			CalculateGlobalMeans(h_fMRI_Volumes);
		}

		SetupTTestFirstLevel();

		// Copy data to device
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

		// Run the actual GLM
		cl_int largeMemoryError = 0;
		if (largeMemory)
		{
			largeMemoryError = CalculateStatisticalMapsGLMTTestFirstLevel(h_fMRI_Volumes,3);
		}

		if (!largeMemory)
		{
			CalculateStatisticalMapsGLMTTestFirstLevelSlices(h_fMRI_Volumes,3);
		}
		else if (largeMemoryError)
		{
			clReleaseMemObject(d_fMRI_Volumes);
			clReleaseMemObject(d_Whitened_fMRI_Volumes);
			allocatedDeviceMemory -= 2 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);
			deviceMemoryDeallocations += 2;
			largeMemory = false;

			runKernelErrorCalculateBetaWeightsGLMFirstLevel = 0;
			runKernelErrorCalculateGLMResiduals = 0;
			runKernelErrorEstimateAR4Models = 0;
			runKernelErrorApplyWhiteningAR4 = 0;
			runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel = 0;

			printf("GLM error detected for full volume analysis, trying to loop over slices instead!\n");

			d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
			d_Whitened_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
			d_Residuals = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
			allocatedDeviceMemory += 3 * EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float);
			deviceMemoryAllocations += 3;

			CalculateStatisticalMapsGLMTTestFirstLevelSlices(h_fMRI_Volumes,3);
		}


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

		TransformFirstLevelResultsToMNI(true);

		if (WRITE_ACTIVITY_T1)
		{
			// Run the actual GLM again
			if (!largeMemory)
			{
				CalculateStatisticalMapsGLMTTestFirstLevelSlices(h_fMRI_Volumes,3);
			}
			else
			{
				CalculateStatisticalMapsGLMTTestFirstLevel(h_fMRI_Volumes,3);
			}

			// Allocate memory on device
			d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
			d_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
			d_T1_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);

			deviceMemoryAllocations += 3;
			allocatedDeviceMemory += 2 * T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float);
			allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);

			// Copy original T1 volume to device
			clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);

			// Copy first fMRI volume to device
			clEnqueueWriteBuffer(commandQueue, d_EPI_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Temp_fMRI_Volume , 0, NULL, NULL);

			// Register original fMRI volume to original T1 volume
			PerformRegistrationEPIT1Original();

			// Cleanup
			clReleaseMemObject(d_T1_Volume);
			clReleaseMemObject(d_EPI_Volume);
			clReleaseMemObject(d_T1_EPI_Volume);

			allocatedDeviceMemory -= 2 * T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float);
			allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);
			deviceMemoryDeallocations += 3;

			TransformFirstLevelResultsToT1(true);
		}


		// Do statistical analysis without whitening	
		if (WRITE_UNWHITENED_RESULTS)
		{
			// Calculate maps without whitening
			if (!largeMemory)
			{
				CalculateStatisticalMapsGLMTTestFirstLevelSlices(h_fMRI_Volumes,0);
			}
			else
			{
				CalculateStatisticalMapsGLMTTestFirstLevel(h_fMRI_Volumes,0);
			}

			// Copy data to host
			if (WRITE_ACTIVITY_EPI)
			{
				clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_No_Whitening_EPI, 0, NULL, NULL);
				clEnqueueReadBuffer(commandQueue, d_Contrast_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrast_Volumes_No_Whitening_EPI, 0, NULL, NULL);
				clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_No_Whitening_EPI, 0, NULL, NULL);
				//clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);
			}
	
			// Apply transformations and save to unwhitened pointers
			TransformFirstLevelResultsToMNI(false);

			if (WRITE_ACTIVITY_T1)
			{
				// Calculate maps without whitening again
				if (!largeMemory)
				{
					CalculateStatisticalMapsGLMTTestFirstLevelSlices(h_fMRI_Volumes,0);
				}
				else
				{
					CalculateStatisticalMapsGLMTTestFirstLevel(h_fMRI_Volumes,0);
				}

				TransformFirstLevelResultsToT1(false);
			}
		}

		//---------------------------------------------------------------------------------------------------------------------------------------
		// Single subject permutation test
		//---------------------------------------------------------------------------------------------------------------------------------------

		if (PERMUTE_FIRST_LEVEL)
		{
			// Check if there is enough memory first
			// Need to keep all whitened and permuted volumes in memory at the same time
			size_t totalRequiredMemory = EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float) * 2;

			if ( ((totalRequiredMemory + (cl_ulong)allocatedDeviceMemory) / (1024*1024)) > globalMemorySize)
			{
				if (WRAPPER == BASH)
				{
					printf("Cannot run permutation test on the selected device. Required memory for permutation test is %zu MB, global memory is %zu MB ! \n",totalRequiredMemory/(1024*1024),globalMemorySize);
				}
			}
			else
			{
				if ((WRAPPER == BASH) && PRINT)
				{
					printf("\nRunning permutation test\n\n");
				}
	
				// Try to allocate temporary memory
				cl_int memoryAllocationError1, memoryAllocationError2;
				d_Temp_fMRI_Volumes_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, &memoryAllocationError1);
				d_Temp_fMRI_Volumes_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, &memoryAllocationError2);

				if ( (memoryAllocationError1 != CL_SUCCESS) || (memoryAllocationError2 != CL_SUCCESS) )
				{
					if (WRAPPER == BASH)
					{	
						printf("Unable to allocate memory for permutation test, aborting. The error messages are %s and %s .\n",GetOpenCLErrorMessage(memoryAllocationError1),GetOpenCLErrorMessage(memoryAllocationError2));
					}
				}
				else
				{
					d_Cluster_Indices = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int), NULL, NULL);
					d_Cluster_Sizes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int), NULL, NULL);
					d_TFCE_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int), NULL, NULL);
					d_P_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	
					deviceMemoryAllocations += 6;
					allocatedDeviceMemory += 2 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);
					allocatedDeviceMemory += 3 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int);
					allocatedDeviceMemory += 1 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(int);

					c_Permutation_Vector = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(unsigned short int), NULL, NULL);
					c_Permutation_Distribution = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_PERMUTATIONS * sizeof(float), NULL, NULL);

					PrintMemoryStatus("Before permutation testing");
	
					// Run the actual permutation test
					ApplyPermutationTestFirstLevel(h_fMRI_Volumes); 
	
					// Free temporary memory
					clReleaseMemObject(d_Temp_fMRI_Volumes_1);
					clReleaseMemObject(d_Temp_fMRI_Volumes_2);
					allocatedDeviceMemory -= 2 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);
					deviceMemoryDeallocations += 2;

					// Calculate activity map without Cochrane-Orcutt
					CalculateStatisticalMapsGLMTTestFirstLevelSlices(h_fMRI_Volumes,0);
	
					// Calculate permutation p-values
					CalculatePermutationPValues(d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

					// Copy permutation p-values to host		
					if (WRITE_ACTIVITY_EPI)
					{
						clEnqueueReadBuffer(commandQueue, d_P_Values, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_P_Values_EPI, 0, NULL, NULL);
					}

					// Transform p-values to MNI space, without changing p-values
					TransformPValuesToMNI();

					// Transform p-values to T1 space
					if (WRITE_ACTIVITY_T1)
					{
						TransformPValuesToT1();
					}
	
					clReleaseMemObject(d_Cluster_Indices);
					clReleaseMemObject(d_Cluster_Sizes);
					clReleaseMemObject(d_TFCE_Values);
					clReleaseMemObject(d_P_Values);

					deviceMemoryDeallocations += 4;					
					allocatedDeviceMemory -= 3 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int);
					allocatedDeviceMemory -= 1 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(int);
	
					clReleaseMemObject(c_Permutation_Vector);
					clReleaseMemObject(c_Permutation_Distribution);
	
					PrintMemoryStatus("After permutation testing");
				}
			}
		}


		// Cleanup host memory
		free(h_X_GLM);
		free(h_xtxxt_GLM);
		free(h_Contrasts);
		free(h_ctxtxc_GLM);
		free(h_X_GLM_With_Temporal_Derivatives);
		free(h_X_GLM_Convolved);
		free(h_Global_Mean);

		// Cleanup device memory
		clReleaseMemObject(c_X_GLM);
		clReleaseMemObject(c_xtxxt_GLM);
		clReleaseMemObject(c_Contrasts);
		clReleaseMemObject(c_ctxtxc_GLM);
	
		clReleaseMemObject(d_fMRI_Volumes);
		clReleaseMemObject(d_Whitened_fMRI_Volumes);
		if (!largeMemory)
		{
			clReleaseMemObject(d_Residuals);
			deviceMemoryDeallocations += 3;
			allocatedDeviceMemory -= 3 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_T * sizeof(float);
		}
		else
		{
			deviceMemoryDeallocations += 2;
			allocatedDeviceMemory -= 2 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);		
		}

		clReleaseMemObject(d_Beta_Volumes);
		clReleaseMemObject(d_Contrast_Volumes);
		clReleaseMemObject(d_Statistical_Maps);
		clReleaseMemObject(d_Residual_Variances);

		allocatedDeviceMemory -= (EPI_DATA_W * EPI_DATA_H * EPI_DATA_D)*(NUMBER_OF_TOTAL_GLM_REGRESSORS + NUMBER_OF_CONTRASTS + NUMBER_OF_CONTRASTS + 1) * sizeof(float);
		deviceMemoryDeallocations += 4;

		clReleaseMemObject(d_AR1_Estimates);
		clReleaseMemObject(d_AR2_Estimates);
		clReleaseMemObject(d_AR3_Estimates);
		clReleaseMemObject(d_AR4_Estimates);

		allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * 4 * sizeof(float);
		deviceMemoryDeallocations += 4;

		PrintMemoryStatus("After GLM");
	}
	// Only transform the preprocessed fMRI data to MNI space
	else if (PREPROCESSING_ONLY)
	{
		TransformfMRIVolumesToMNI();
	}
	// Only estimate beta values, no t- or F-scores
	else if (!REGRESS_ONLY && !BAYESIAN && BETAS_ONLY)
	{
		if ((WRAPPER == BASH) && PRINT)
		{
			printf("Performing statistical analysis, only estimating beta values and contrasts\n");
		}

		NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + REGRESS_GLOBALMEAN + NUMBER_OF_CONFOUND_REGRESSORS*REGRESS_CONFOUNDS;

		// Check amount of global memory, compared to required memory
		bool largeMemory = true;
		size_t totalRequiredMemory = allocatedDeviceMemory + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float) + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float) + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float);
		totalRequiredMemory /= (1024*1024);

		if (totalRequiredMemory > globalMemorySize)
		{
			largeMemory = false;
			if ((WRAPPER == BASH) && VERBOS)
			{
				printf("Cannot calculate beta values for the whole volume at once, doing slice by slice. Required device memory for beta values is %zu MB, global memory is %zu MB ! \n",totalRequiredMemory,globalMemorySize);
			}
		}
		else
		{
			if ((WRAPPER == BASH) && VERBOS)
			{
				printf("Sufficient memory for calculating beta values for the whole volume at once! Required device memory for beta values is %zu MB, global memory is %zu MB ! \n",totalRequiredMemory,globalMemorySize);
			}
		}

		c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
		c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
		c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
		c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
		
		if (!largeMemory)
		{
			// Allocate memory for one slice for all time points, loop over slices to save memory
			d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
			allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float);
		}
		else
		{
			d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
			allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);
		}
		deviceMemoryAllocations += 1;

		d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
		d_Contrast_Volumes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

		deviceMemoryAllocations += 2;
		allocatedDeviceMemory += (EPI_DATA_W * EPI_DATA_H * EPI_DATA_D)*(NUMBER_OF_TOTAL_GLM_REGRESSORS + NUMBER_OF_CONTRASTS) * sizeof(float);

		PrintMemoryStatus("Before GLM");

		//SetMemory(d_EPI_Mask, 1.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D);

		h_X_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
		h_xtxxt_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
		h_Contrasts = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float));
		h_ctxtxc_GLM = (float*)malloc(NUMBER_OF_CONTRASTS * sizeof(float));
		h_X_GLM_With_Temporal_Derivatives = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * 2 * EPI_DATA_T * sizeof(float));
		h_X_GLM_Convolved = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * (USE_TEMPORAL_DERIVATIVES+1) * EPI_DATA_T * sizeof(float));
		h_Global_Mean = (float*)malloc(EPI_DATA_T * sizeof(float));

		if (REGRESS_GLOBALMEAN)
		{
			CalculateGlobalMeans(h_fMRI_Volumes);
		}

		SetupTTestFirstLevel();

		// Copy data to device
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

		// Run the actual GLM
		if (!largeMemory)
		{
			CalculateBetaWeightsAndContrastsFirstLevelSlices(h_fMRI_Volumes);
		}
		else
		{
			CalculateBetaWeightsAndContrastsFirstLevel(h_fMRI_Volumes);
		}

		// Copy data in EPI space to host

		if (WRITE_ACTIVITY_EPI)
		{
			clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_EPI, 0, NULL, NULL);
			clEnqueueReadBuffer(commandQueue, d_Contrast_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrast_Volumes_EPI, 0, NULL, NULL);
		}
		
		TransformFirstLevelResultsToMNI(true);

		if (WRITE_ACTIVITY_T1)
		{
			// Run the actual GLM again
			if (!largeMemory)
			{
				CalculateBetaWeightsAndContrastsFirstLevelSlices(h_fMRI_Volumes);
			}	
			else
			{
				CalculateBetaWeightsAndContrastsFirstLevel(h_fMRI_Volumes);
			}

			// Allocate memory on device
			d_T1_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);
			d_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
			d_T1_EPI_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE,  T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);

			deviceMemoryAllocations += 3;
			allocatedDeviceMemory += 2 * T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float);
			allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);

			// Copy original T1 volume to device
			clEnqueueWriteBuffer(commandQueue, d_T1_Volume, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_T1_Volume , 0, NULL, NULL);

			// Copy first fMRI volume to device
			clEnqueueWriteBuffer(commandQueue, d_EPI_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Temp_fMRI_Volume , 0, NULL, NULL);

			// Register original fMRI volume to original T1 volume
			PerformRegistrationEPIT1Original();

			// Cleanup
			clReleaseMemObject(d_T1_Volume);
			clReleaseMemObject(d_EPI_Volume);
			clReleaseMemObject(d_T1_EPI_Volume);

			allocatedDeviceMemory -= 2 * T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float);
			allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);
			deviceMemoryDeallocations += 3;

			TransformFirstLevelResultsToT1(true);
		}

		// Cleanup host memory
		free(h_X_GLM);
		free(h_xtxxt_GLM);
		free(h_Contrasts);
		free(h_ctxtxc_GLM);
		free(h_X_GLM_With_Temporal_Derivatives);
		free(h_X_GLM_Convolved);
		free(h_Global_Mean);

		// Cleanup device memory
		clReleaseMemObject(c_X_GLM);
		clReleaseMemObject(c_xtxxt_GLM);
		clReleaseMemObject(c_Contrasts);
		clReleaseMemObject(c_ctxtxc_GLM);
	
		clReleaseMemObject(d_fMRI_Volumes);

		if (!largeMemory)
		{
			allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_T * sizeof(float);
		}
		else
		{
			allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);
		}
		deviceMemoryDeallocations += 1;

		clReleaseMemObject(d_Beta_Volumes);
		clReleaseMemObject(d_Contrast_Volumes);

		allocatedDeviceMemory -= (EPI_DATA_W * EPI_DATA_H * EPI_DATA_D)*(NUMBER_OF_TOTAL_GLM_REGRESSORS + NUMBER_OF_CONTRASTS) * sizeof(float);
		deviceMemoryDeallocations += 2;

		PrintMemoryStatus("After GLM");
	}
	else if (BAYESIAN)
	{
		if ((WRAPPER == BASH) && PRINT)
		{
			printf("Performing statistical analysis\n");
		}

		NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + REGRESS_GLOBALMEAN + 		NUMBER_OF_CONFOUND_REGRESSORS*REGRESS_CONFOUNDS;

		c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
		c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
		c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
		c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

		d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * 10 * sizeof(float), NULL, NULL);
		d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * 6 * sizeof(float), NULL, NULL);
		d_AR1_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

		deviceMemoryAllocations += 3;
		allocatedDeviceMemory += (EPI_DATA_W * EPI_DATA_H * EPI_DATA_D)*(10 + 6 + 1) * sizeof(float);

		PrintMemoryStatus("Before Bayesian GLM");

		h_X_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
		h_xtxxt_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
		h_Contrasts = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float));
		h_ctxtxc_GLM = (float*)malloc(NUMBER_OF_CONTRASTS * sizeof(float));
		h_X_GLM_With_Temporal_Derivatives = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * 2 * EPI_DATA_T * sizeof(float));
		h_X_GLM_Convolved = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * (USE_TEMPORAL_DERIVATIVES+1) * EPI_DATA_T * sizeof(float));

		SetupTTestFirstLevel();

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

		// Run the Bayesian analysis
		CalculateStatisticalMapsGLMBayesianFirstLevel(h_fMRI_Volumes);

		// Copy data to host
		if (WRITE_ACTIVITY_EPI)
		{
			clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * 2 * sizeof(float), h_Beta_Volumes_EPI, 0, NULL, NULL);
			clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * 6 * sizeof(float), h_Statistical_Maps_EPI, 0, NULL, NULL);
		}

		if (WRITE_AR_ESTIMATES_EPI)
		{
			clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
		}

		TransformBayesianFirstLevelResultsToMNI();

		// Cleanup host memory
		free(h_X_GLM);
		free(h_xtxxt_GLM);
		free(h_Contrasts);
		free(h_ctxtxc_GLM);
		free(h_X_GLM_With_Temporal_Derivatives);
		free(h_X_GLM_Convolved);

		// Cleanup device memory
		clReleaseMemObject(c_X_GLM);
		clReleaseMemObject(c_xtxxt_GLM);
		clReleaseMemObject(c_Contrasts);
		clReleaseMemObject(c_ctxtxc_GLM);

		clReleaseMemObject(d_Beta_Volumes);
		clReleaseMemObject(d_Statistical_Maps);
		clReleaseMemObject(d_AR1_Estimates);

		allocatedDeviceMemory -= (EPI_DATA_W * EPI_DATA_H * EPI_DATA_D)*(10  + 6) * sizeof(float);
		allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);
		deviceMemoryDeallocations += 3;

		PrintMemoryStatus("After Bayesian GLM");
	}
	// Only regression
	else if (REGRESS_ONLY)
	{
		if ((WRAPPER == BASH) && PRINT)
		{
			printf("Performing regression\n");
		}

		NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + REGRESS_GLOBALMEAN + NUMBER_OF_CONFOUND_REGRESSORS*REGRESS_CONFOUNDS;

		// Check amount of global memory, compared to required memory
		bool largeMemory = true;
		size_t totalRequiredMemory = allocatedDeviceMemory + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float) * 2 + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float);
		totalRequiredMemory /= (1024*1024);

		if (totalRequiredMemory > globalMemorySize)
		{
			largeMemory = false;
			if ((WRAPPER == BASH) && VERBOS)
			{
				printf("Cannot run the regression the whole volume at once, doing slice by slice. Required device memory for regression is %zu MB, global memory is %zu MB ! \n",totalRequiredMemory,globalMemorySize);
			}
		}
		else
		{
			if ((WRAPPER == BASH) && VERBOS)
			{
				printf("Sufficient memory for running the regression the whole volume at once! Required device memory for regression is %zu MB, global memory is %zu MB ! \n",totalRequiredMemory,globalMemorySize);
			}
		}


		c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
		c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);

		if (!largeMemory)
		{
			// Allocate memory for one slice
			d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
			d_Residuals = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
			allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_T * 2 * sizeof(float);
		}
		else
		{
			d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
			d_Residuals = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
			allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * 2 * sizeof(float);
		}

		d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
		
		deviceMemoryAllocations += 3;
		allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float);

		PrintMemoryStatus("Before regression");

		h_X_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
		h_xtxxt_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
		h_Global_Mean = (float*)malloc(EPI_DATA_T * sizeof(float));

		if (REGRESS_GLOBALMEAN)
		{
			CalculateGlobalMeans(h_fMRI_Volumes);
		}

		SetupGLMRegressorsFirstLevel();

		// Copy data to device
		clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM , 0, NULL, NULL);
		clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM , 0, NULL, NULL);
	
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



		if (!largeMemory)
		{
			// Loop over slices, to save memory
			for (int slice = 0; slice < EPI_DATA_D; slice++)
			{
				// Copy fMRI data to the device, for the current slice
				CopyCurrentfMRISliceToDevice(d_fMRI_Volumes, h_fMRI_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
				// Perform the regression
				PerformRegressionSlice(d_Residuals, d_fMRI_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
				// Copy back the current slice
				CopyCurrentfMRISliceToHost(h_fMRI_Volumes, d_Residuals, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	
				if (WRITE_ACTIVITY_EPI)
				{
					// Copy residuals to the host, for the current slice			
					CopyCurrentfMRISliceToHost(h_Residuals_EPI, d_Residuals, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
				}
			}	
		}
		else
		{
			// Copy fMRI volumes to device
			clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes, 0, NULL, NULL);
			// Perform the regression
			PerformRegression(d_Residuals, d_fMRI_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
			// Copy back the residuals to the host
			clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes, 0, NULL, NULL);

			if (WRITE_ACTIVITY_EPI)
			{
				clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Residuals_EPI, 0, NULL, NULL);
			}
		}

		// Normalize residuals to have unit variance (the mean has already been removed)
		/*
		for (int x = 0; x < EPI_DATA_W; x++)
		{
			for (int y = 0; y < EPI_DATA_H; y++)
			{
				for (int z = 0; z < EPI_DATA_D; z++)
				{
					float var = 0.0f;
					for (int t = 0; t < EPI_DATA_T; t++)
					{
						float value = h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
						var += value * value;
					}
					var = var /(float)(EPI_DATA_T-1);
					float std = sqrt(var);

					if (var != 0.0f)
					{
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] /= std;
						}
					}
				}
			}
		}
		*/

		TransformResidualsToMNI();

		// Cleanup host memory
		free(h_X_GLM);
		free(h_xtxxt_GLM);
		free(h_Global_Mean);

		// Cleanup device memory
		clReleaseMemObject(c_X_GLM);
		clReleaseMemObject(c_xtxxt_GLM);

		clReleaseMemObject(d_Beta_Volumes);
		clReleaseMemObject(d_Residuals);
		clReleaseMemObject(d_fMRI_Volumes);

		deviceMemoryDeallocations += 3;
		if (!largeMemory)
		{
			allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_T * 2 * sizeof(float);
		}
		else
		{
			allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * 2 * sizeof(float);
		}
		allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float);

		PrintMemoryStatus("After regression");
	}

	if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
	{
		clReleaseMemObject(d_Total_Displacement_Field_X);
		clReleaseMemObject(d_Total_Displacement_Field_Y);
		clReleaseMemObject(d_Total_Displacement_Field_Z);
	}

	clReleaseMemObject(d_EPI_Mask);
	clReleaseMemObject(d_Smoothed_EPI_Mask);
	allocatedDeviceMemory -= 2 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);
	deviceMemoryDeallocations += 2;

	free(h_Temp_fMRI_Volume);
	allocatedHostMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);
	hostMemoryDeallocations += 1;
	
	if (APPLY_MOTION_CORRECTION)
	{
		free(h_Motion_Parameters);
		allocatedHostMemory -= EPI_DATA_T * NUMBER_OF_MOTION_REGRESSORS * sizeof(float);
		hostMemoryDeallocations += 1;
	}

	PrintMemoryStatus("After deallocating masks");
}








void BROCCOLI_LIB::TransformResidualsToMNI()
{
	// Allocate temporary memory
	cl_mem d_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Temp = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	// Loop over time points
	for (int i = 0; i < EPI_DATA_T; i++)
	{
		// Copy current volume to temp
		clEnqueueWriteBuffer(commandQueue, d_Temp, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), &h_fMRI_Volumes[i * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D], 0, NULL, NULL);

		// First apply initial translation before changing resolution and size 
		TransformVolumesLinear(d_Temp, h_StartParameters_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, INTERPOLATION_MODE);

		// Change resolution and size of volume
		ChangeVolumesResolutionAndSize(d_Data, d_Temp, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);

		// Now apply the same translation as applied before the EPI-T1 registration
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

		// Apply transformation
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)		
		{
			TransformVolumesNonLinear(d_Data, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		}

		// Write transformed volume to host
		clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), &h_Residuals_MNI[i * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D], 0, NULL, NULL);
	}

	clReleaseMemObject(d_Data);
	clReleaseMemObject(d_Temp);
}

void BROCCOLI_LIB::TransformMaskToMNI()
{
	// Allocate temporary memory
	cl_mem d_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Temp = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	// Copy mask volume to temp
	clEnqueueWriteBuffer(commandQueue, d_Temp, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask, 0, NULL, NULL);

	// First apply initial translation before changing resolution and size 
	TransformVolumesLinear(d_Temp, h_StartParameters_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, NEAREST);

	// Change resolution and size of volume
	ChangeVolumesResolutionAndSize(d_Data, d_Temp, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, NEAREST, 0);

	// Now apply the same translation as applied before the EPI-T1 registration
	TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, NEAREST);

	// Apply transformation
	TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, NEAREST);
	if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
	{
		TransformVolumesNonLinear(d_Data, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, NEAREST);
	}

	// Write transformed mask to host
	clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Mask, 0, NULL, NULL);

	clReleaseMemObject(d_Data);
	clReleaseMemObject(d_Temp);
}

void BROCCOLI_LIB::TransformfMRIVolumesToMNI()
{
	// Allocate temporary memory
	cl_mem d_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Temp = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	// Loop over time points
	for (int i = 0; i < EPI_DATA_T; i++)
	{
		// Copy current volume to temp
		clEnqueueWriteBuffer(commandQueue, d_Temp, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), &h_fMRI_Volumes[i * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D], 0, NULL, NULL);

		// First apply initial translation before changing resolution and size 
		TransformVolumesLinear(d_Temp, h_StartParameters_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, INTERPOLATION_MODE);

		// Change resolution and size of volume
		ChangeVolumesResolutionAndSize(d_Data, d_Temp, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);

		// Now apply the same translation as applied before the EPI-T1 registration
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

		// Apply transformation
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
		{
			TransformVolumesNonLinear(d_Data, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		}

		// Write transformed volume to host
		clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), &h_fMRI_Volumes_MNI[i * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D], 0, NULL, NULL);
	}

	clReleaseMemObject(d_Data);
	clReleaseMemObject(d_Temp);
}


// New version which uses less memory
void BROCCOLI_LIB::TransformFirstLevelResultsToMNI(bool WHITENED)
{
	// Allocate temporary memory
	cl_mem d_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Make a copy of results, to not ruin transformation to T1

	// First apply initial translation before changing resolution and size 
	TransformVolumesLinear(d_Beta_Volumes, h_StartParameters_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_TOTAL_GLM_REGRESSORS, INTERPOLATION_MODE);
	TransformVolumesLinear(d_Contrast_Volumes, h_StartParameters_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);
	if (!BETAS_ONLY)
	{
		TransformVolumesLinear(d_Statistical_Maps, h_StartParameters_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);
	}

	// Loop over regressors
	for (int i = 0; i < NUMBER_OF_TOTAL_GLM_REGRESSORS; i++)
	{
		// Change resolution and size of volume
		ChangeVolumesResolutionAndSize(d_Data, d_Beta_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, i);

		// Now apply the same translation as applied before the EPI-T1 registration
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

		// Apply transformation
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
		{
			TransformVolumesNonLinear(d_Data, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		}

		// Write transformed volume to host
		if (WHITENED)
		{
			clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), &h_Beta_Volumes_MNI[i * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D], 0, NULL, NULL);
		}
		else
		{
			clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), &h_Beta_Volumes_No_Whitening_MNI[i * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D], 0, NULL, NULL);
		}
	}

	// Loop over contrasts
	for (int i = 0; i < NUMBER_OF_CONTRASTS; i++)
	{
		// Change resolution and size of volume
		ChangeVolumesResolutionAndSize(d_Data, d_Contrast_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, i);

		// Now apply the same translation as applied before the EPI-T1 registration
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

		// Apply transformation
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
		{
			TransformVolumesNonLinear(d_Data, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		}

		// Write transformed volume to host
		if (WHITENED)
		{
			clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), &h_Contrast_Volumes_MNI[i * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D], 0, NULL, NULL);
		}
		else
		{
			clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), &h_Contrast_Volumes_No_Whitening_MNI[i * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D], 0, NULL, NULL);
		}
	}

	if (!BETAS_ONLY)
	{
		// Loop over contrasts, for statistical maps
		for (int i = 0; i < NUMBER_OF_CONTRASTS; i++)
		{
			// Change resolution and size of volume
			ChangeVolumesResolutionAndSize(d_Data, d_Statistical_Maps, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, i);

			// Now apply the same translation as applied before the EPI-T1 registration
			TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

			// Apply transformation
			TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
			if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
			{
				TransformVolumesNonLinear(d_Data, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
			}

			// Write transformed volume to host
			if (WHITENED)
			{
				clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), &h_Statistical_Maps_MNI[i * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D], 0, NULL, NULL);
			}
			else
			{
				clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), &h_Statistical_Maps_No_Whitening_MNI[i * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D], 0, NULL, NULL);
			}
		}
	}

	//ChangeVolumesResolutionAndSize(d_Residual_Variances_MNI, d_Residual_Variances, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	//TransformVolumesLinear(d_Residual_Variances_MNI, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
	//TransformVolumesNonLinear(d_Residual_Variances_MNI, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

	//MultiplyVolumes(d_Beta_Volumes_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_GLM_REGRESSORS);
	//MultiplyVolumes(d_Contrast_Volumes_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS);
	//MultiplyVolumes(d_Statistical_Maps_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, NUMBER_OF_CONTRASTS);
	//MultiplyVolumes(d_Residual_Variances_MNI, d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1);

	if (WRITE_AR_ESTIMATES_MNI && WHITENED && !BETAS_ONLY)
	{
		TransformVolumesLinear(d_AR1_Estimates, h_StartParameters_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, INTERPOLATION_MODE);
		ChangeVolumesResolutionAndSize(d_Data, d_AR1_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
		{
			TransformVolumesNonLinear(d_Data, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		}

		// Write transformed volume to host
		clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_AR1_Estimates_MNI, 0, NULL, NULL);

		TransformVolumesLinear(d_AR2_Estimates, h_StartParameters_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, INTERPOLATION_MODE);
		ChangeVolumesResolutionAndSize(d_Data, d_AR2_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
		{
			TransformVolumesNonLinear(d_Data, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		}

		// Write transformed volume to host
		clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_AR2_Estimates_MNI, 0, NULL, NULL);

		TransformVolumesLinear(d_AR3_Estimates, h_StartParameters_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, INTERPOLATION_MODE);
		ChangeVolumesResolutionAndSize(d_Data, d_AR3_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
		{
			TransformVolumesNonLinear(d_Data, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		}

		// Write transformed volume to host
		clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_AR3_Estimates_MNI, 0, NULL, NULL);

		TransformVolumesLinear(d_AR4_Estimates, h_StartParameters_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, INTERPOLATION_MODE);
		ChangeVolumesResolutionAndSize(d_Data, d_AR4_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
		{
			TransformVolumesNonLinear(d_Data, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		}

		// Write transformed volume to host
		clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_AR4_Estimates_MNI, 0, NULL, NULL);
	}

	clReleaseMemObject(d_Data);
}

// New version which uses less memory
void BROCCOLI_LIB::TransformFirstLevelResultsToT1(bool WHITENED)
{
	// Allocate temporary memory
	cl_mem d_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);

	// First apply initial translation before changing resolution and size 
	//TransformVolumesLinear(d_Beta_Volumes, h_StartParameters_EPI_Original, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_TOTAL_GLM_REGRESSORS, INTERPOLATION_MODE);
	//TransformVolumesLinear(d_Contrast_Volumes, h_StartParameters_EPI_Original, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);
	//TransformVolumesLinear(d_Statistical_Maps, h_StartParameters_EPI_Original, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);

	// Loop over regressors
	for (int i = 0; i < NUMBER_OF_TOTAL_GLM_REGRESSORS; i++)
	{
		// Change resolution and size of volume
		ChangeVolumesResolutionAndSize(d_Data, d_Beta_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, i);

		// Now apply the same translation as applied before the EPI-T1 registration
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);

		// Apply transformation
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_T1_Affine_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);

		// Write transformed volume to host
		if (WHITENED)
		{
			clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), &h_Beta_Volumes_T1[i * T1_DATA_W * T1_DATA_H * T1_DATA_D], 0, NULL, NULL);
		}	
		else
		{
			clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), &h_Beta_Volumes_No_Whitening_T1[i * T1_DATA_W * T1_DATA_H * T1_DATA_D], 0, NULL, NULL);
		}
	}

	// Loop over contrasts
	for (int i = 0; i < NUMBER_OF_CONTRASTS; i++)
	{
		// Change resolution and size of volume
		ChangeVolumesResolutionAndSize(d_Data, d_Contrast_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, i);

		// Now apply the same translation as applied before the EPI-T1 registration
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);

		// Apply transformation
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_T1_Affine_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);

		// Write transformed volume to host
		if (WHITENED)
		{
			clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), &h_Contrast_Volumes_T1[i * T1_DATA_W * T1_DATA_H * T1_DATA_D], 0, NULL, NULL);
		}
		else
		{
			clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), &h_Contrast_Volumes_No_Whitening_T1[i * T1_DATA_W * T1_DATA_H * T1_DATA_D], 0, NULL, NULL);
		}
	}

	if (!BETAS_ONLY)
	{
		// Loop over contrasts, for statistical maps
		for (int i = 0; i < NUMBER_OF_CONTRASTS; i++)
		{
			// Change resolution and size of volume
			ChangeVolumesResolutionAndSize(d_Data, d_Statistical_Maps, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, i);

			// Now apply the same translation as applied before the EPI-T1 registration
			TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);

			// Apply transformation
			TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_T1_Affine_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);

			// Write transformed volume to host
			if (WHITENED)
			{
				clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), &h_Statistical_Maps_T1[i * T1_DATA_W * T1_DATA_H * T1_DATA_D], 0, NULL, NULL);
			}
			else
			{
				clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), &h_Statistical_Maps_No_Whitening_T1[i * T1_DATA_W * T1_DATA_H * T1_DATA_D], 0, NULL, NULL);
			}
		}
	}

	//ChangeVolumesResolutionAndSize(d_Residual_Variances_T1, d_Residual_Variances, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE);
	//TransformVolumesLinear(d_Residual_Variances_T1, h_Registration_Parameters_EPI_T1_Affine, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);

	if (WRITE_AR_ESTIMATES_T1 && WHITENED && !BETAS_ONLY)
	{
		//TransformVolumesLinear(d_AR1_Estimates, h_StartParameters_EPI_Original, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, INTERPOLATION_MODE);
		ChangeVolumesResolutionAndSize(d_Data, d_AR1_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);
		// Now apply the same translation as applied before the EPI-T1 registration
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_T1_Affine_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
		clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_AR1_Estimates_T1, 0, NULL, NULL);

		//TransformVolumesLinear(d_AR2_Estimates, h_StartParameters_EPI_Original, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, INTERPOLATION_MODE);
		ChangeVolumesResolutionAndSize(d_Data, d_AR2_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);
		// Now apply the same translation as applied before the EPI-T1 registration
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_T1_Affine_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
		clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_AR2_Estimates_T1, 0, NULL, NULL);

		//TransformVolumesLinear(d_AR3_Estimates, h_StartParameters_EPI_Original, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, INTERPOLATION_MODE);
		ChangeVolumesResolutionAndSize(d_Data, d_AR3_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);
		// Now apply the same translation as applied before the EPI-T1 registration
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_T1_Affine_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
		clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_AR3_Estimates_T1, 0, NULL, NULL);

		//TransformVolumesLinear(d_AR4_Estimates, h_StartParameters_EPI_Original, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, INTERPOLATION_MODE);
		ChangeVolumesResolutionAndSize(d_Data, d_AR4_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);
		// Now apply the same translation as applied before the EPI-T1 registration
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_T1_Affine_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
		clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), h_AR4_Estimates_T1, 0, NULL, NULL);
	}

	clReleaseMemObject(d_Data);
}

// Transforms Bayesian results from EPI space to MNI space, updated to use less memory
void BROCCOLI_LIB::TransformBayesianFirstLevelResultsToMNI()
{
	// Allocate temporary memory
	cl_mem d_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	TransformVolumesLinear(d_Beta_Volumes, h_StartParameters_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 2, INTERPOLATION_MODE);
	TransformVolumesLinear(d_Statistical_Maps, h_StartParameters_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 6, INTERPOLATION_MODE);

	// Loop over regressors, for beta volumes
	for (int i = 0; i < 2; i++)
	{
		// Change resolution and size of volume
		ChangeVolumesResolutionAndSize(d_Data, d_Beta_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, i);

		// Now apply the same translation as applied before the EPI-T1 registration
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

		// Apply transformation
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
		{
			TransformVolumesNonLinear(d_Data, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		}

		// Write transformed volume to host
		clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), &h_Beta_Volumes_MNI[i * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D], 0, NULL, NULL);
	}

	// Loop over contrasts, for statistical maps
	for (int i = 0; i < 6; i++)
	{
		// Change resolution and size of volume
		ChangeVolumesResolutionAndSize(d_Data, d_Statistical_Maps, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, i);

		// Now apply the same translation as applied before the EPI-T1 registration
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);

		// Apply transformation
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
		{
			TransformVolumesNonLinear(d_Data, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		}

		// Write transformed volume to host
		clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), &h_Statistical_Maps_MNI[i * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D], 0, NULL, NULL);
	}

	if (WRITE_AR_ESTIMATES_MNI)
	{
		TransformVolumesLinear(d_AR1_Estimates, h_StartParameters_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, INTERPOLATION_MODE);
		ChangeVolumesResolutionAndSize(d_Data, d_AR1_Estimates, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, 0);
		TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
		{
			TransformVolumesNonLinear(d_Data, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
		}

		clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_AR1_Estimates_MNI, 0, NULL, NULL);
	}

	clReleaseMemObject(d_Data);
}


// Updated to use less memory
void BROCCOLI_LIB::TransformPValuesToMNI()
{	
	// Allocate temporary memory
	cl_mem d_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

	// Allocate temporary memory
	cl_mem d_Temp = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	clEnqueueCopyBuffer(commandQueue, d_P_Values, d_Temp, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), 0, NULL, NULL);

	// Nearest neighbour interpolation for cluster inference, since all voxels in the cluster should have the same p-value
	if ( (INFERENCE_MODE == CLUSTER_EXTENT) || (INFERENCE_MODE == CLUSTER_MASS) )
	{
		TransformVolumesLinear(d_Temp, h_StartParameters_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, NEAREST);

		// Loop over contrasts, for statistical maps
		for (int i = 0; i < NUMBER_OF_CONTRASTS; i++)
		{
			ChangeVolumesResolutionAndSize(d_Data, d_Temp, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, NEAREST, i);
			TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, NEAREST);
			TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, NEAREST);
			if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
			{
				TransformVolumesNonLinear(d_Data, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, NEAREST);
			}

			// Write transformed volume to host
			clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), &h_P_Values_MNI[i * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D], 0, NULL, NULL);
		}
	}
	// Linear interpolation otherwhise
	else
	{
		TransformVolumesLinear(d_Temp, h_StartParameters_EPI, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);

		// Loop over contrasts, for statistical maps
		for (int i = 0; i < NUMBER_OF_CONTRASTS; i++)
		{
			ChangeVolumesResolutionAndSize(d_Data, d_Temp, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, i);
			TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
			TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_MNI, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
			if (NUMBER_OF_ITERATIONS_FOR_NONLINEAR_IMAGE_REGISTRATION > 0)
			{
				TransformVolumesNonLinear(d_Data, d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D, 1, INTERPOLATION_MODE);
			}

			// Write transformed volume to host
			clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), &h_P_Values_MNI[i * MNI_DATA_W * MNI_DATA_H * MNI_DATA_D], 0, NULL, NULL);
		}
	}

	clReleaseMemObject(d_Data);
	clReleaseMemObject(d_Temp);
}

// Updated to use less memory
void BROCCOLI_LIB::TransformPValuesToT1()
{	
	// Allocate temporary memory
	cl_mem d_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), NULL, NULL);

	// Nearest neighbour interpolation for cluster inference, since all voxels in the cluster should have the same p-value
	if ( (INFERENCE_MODE == CLUSTER_EXTENT) || (INFERENCE_MODE == CLUSTER_MASS) )
	{
		//TransformVolumesLinear(d_P_Values, h_StartParameters_EPI_Original, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, NEAREST);

		// Loop over contrasts, for statistical maps
		for (int i = 0; i < NUMBER_OF_CONTRASTS; i++)
		{
			ChangeVolumesResolutionAndSize(d_Data, d_P_Values, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, NEAREST, i);
			TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, NEAREST);
			TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_T1_Affine_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, NEAREST);

			// Write transformed volume to host
			clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), &h_P_Values_T1[i * T1_DATA_W * T1_DATA_H * T1_DATA_D], 0, NULL, NULL);
		}
	}
	// Linear interpolation otherwhise
	else
	{
		//TransformVolumesLinear(d_P_Values, h_StartParameters_EPI_Original, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_CONTRASTS, INTERPOLATION_MODE);

		// Loop over contrasts, for statistical maps
		for (int i = 0; i < NUMBER_OF_CONTRASTS; i++)
		{
			ChangeVolumesResolutionAndSize(d_Data, d_P_Values, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1, T1_DATA_W, T1_DATA_H, T1_DATA_D, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z, T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z, MM_EPI_Z_CUT, INTERPOLATION_MODE, i);
			TransformVolumesLinear(d_Data, h_StartParameters_EPI_T1_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);
			TransformVolumesLinear(d_Data, h_Registration_Parameters_EPI_T1_Affine_Original, T1_DATA_W, T1_DATA_H, T1_DATA_D, 1, INTERPOLATION_MODE);

			// Write transformed volume to host
			clEnqueueReadBuffer(commandQueue, d_Data, CL_TRUE, 0, T1_DATA_W * T1_DATA_H * T1_DATA_D * sizeof(float), &h_P_Values_T1[i * T1_DATA_W * T1_DATA_H * T1_DATA_D], 0, NULL, NULL);
		}
	}

	clReleaseMemObject(d_Data);
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
	clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_Residuals_MNI, 0, NULL, NULL);
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




// Performs slice timing correction of an fMRI dataset, old, does not work
void BROCCOLI_LIB::PerformSliceTimingCorrection()
{
	SetGlobalAndLocalWorkSizesInterpolateVolume(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Allocate memory for slice differences
	c_Slice_Differences = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_D * sizeof(float), NULL, NULL);

	h_Slice_Differences = (float*)malloc(EPI_DATA_D * sizeof(float));

	float middle_slice;

	// Calculate slice differences
	if (SLICE_ORDER == UP)
	{
		middle_slice = myround((float)EPI_DATA_D / 2.0f) - 1.0f;

		for (int z = 0; z < EPI_DATA_D; z++)
		{
			h_Slice_Differences[z] = (middle_slice - (float)z)/((float)EPI_DATA_D);
		}
	}
	else if (SLICE_ORDER == DOWN)
	{
		middle_slice = myround((float)EPI_DATA_D / 2.0f) - 1.0f;

		for (int z = 0; z < EPI_DATA_D; z++)
		{
			h_Slice_Differences[z] = ((float)z - middle_slice)/(float)(EPI_DATA_D);
		}
	}
	else if (SLICE_ORDER == UP_INTERLEAVED)
	{
		middle_slice = (float)EPI_DATA_D - 1.0f;

		float* h_Times = (float*)malloc(EPI_DATA_D * sizeof(float));
		float timePerSlice = TR/(float)EPI_DATA_D;

		for (int z = 0; z < EPI_DATA_D; z++)
		{
			// Odd slice
			if (z % 2)
			{
				h_Times[z] = ceil((float)z/2.0f) * timePerSlice + TR/2.0f;		
			}
			// Even slice
			else
			{
				h_Times[z] = (float)z/2.0f * timePerSlice;		
			}
		}
		
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			h_Slice_Differences[z] = (h_Times[(int)middle_slice] - h_Times[z])/TR;
		}		
		free(h_Times);
	}
	else if (SLICE_ORDER == DOWN_INTERLEAVED)
	{
		middle_slice = 0.0f;

		float* h_Times = (float*)malloc(EPI_DATA_D * sizeof(float));
		float timePerSlice = TR/(float)EPI_DATA_D;

		int zz = 0;
		for (int z = EPI_DATA_D-1; z >= 0; z--)
		{
			// Odd slice
			if (zz % 2)
			{
				h_Times[z] = ceil((float)zz/2.0f) * timePerSlice + TR/2.0f;		
			}
			// Even slice
			else
			{
				h_Times[z] = (float)zz/2.0f * timePerSlice;		
			}
			zz++;
		}
		
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			h_Slice_Differences[z] = (h_Times[(int)middle_slice] - h_Times[z])/TR;
		}		
		free(h_Times);
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

// Performs slice timing correction of an fMRI dataset
// Updated to use less memory, loops over slices 
void BROCCOLI_LIB::PerformSliceTimingCorrectionHost(float* h_Volumes)
{
	SetGlobalAndLocalWorkSizesInterpolateVolume(EPI_DATA_W, EPI_DATA_H, 1);

	// Allocate temporary memory, one slice for all time points
	cl_mem d_Temp_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_T * sizeof(float), NULL, NULL);
	cl_mem d_Temp_Volumes_Corrected = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_T * sizeof(float), NULL, NULL);	

	deviceMemoryAllocations += 2;
	allocatedDeviceMemory += 2 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_T * sizeof(float);

	PrintMemoryStatus("Inside slice timing correction host");

	h_Slice_Differences = (float*)malloc(EPI_DATA_D * sizeof(float));

	float middle_slice;

	// Calculate slice differences
	if (SLICE_ORDER == UP)
	{
		middle_slice = myround((float)EPI_DATA_D / 2.0f) - 1.0f;

		for (int z = 0; z < EPI_DATA_D; z++)
		{
			h_Slice_Differences[z] = (middle_slice - (float)z)/((float)EPI_DATA_D);
		}
	}
	else if (SLICE_ORDER == DOWN)
	{
		middle_slice = myround((float)EPI_DATA_D / 2.0f) - 1.0f;

		for (int z = 0; z < EPI_DATA_D; z++)
		{
			h_Slice_Differences[z] = ((float)z - middle_slice)/(float)(EPI_DATA_D);
		}
	}
	else if (SLICE_ORDER == UP_INTERLEAVED)
	{
		middle_slice = (float)EPI_DATA_D - 1.0f;

		float* h_Times = (float*)malloc(EPI_DATA_D * sizeof(float));
		float timePerSlice = TR/(float)EPI_DATA_D;

		for (int z = 0; z < EPI_DATA_D; z++)
		{
			// Odd slice
			if (z % 2)
			{
				h_Times[z] = ceil((float)z/2.0f) * timePerSlice + TR/2.0f;		
			}
			// Even slice
			else
			{
				h_Times[z] = (float)z/2.0f * timePerSlice;		
			}
		}
		
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			h_Slice_Differences[z] = (h_Times[(int)middle_slice] - h_Times[z])/TR;
		}		
		free(h_Times);
	}
	else if (SLICE_ORDER == DOWN_INTERLEAVED)
	{
		middle_slice = 0.0f;

		float* h_Times = (float*)malloc(EPI_DATA_D * sizeof(float));
		float timePerSlice = TR/(float)EPI_DATA_D;

		int zz = 0;
		for (int z = EPI_DATA_D-1; z >= 0; z--)
		{
			// Odd slice
			if (zz % 2)
			{
				h_Times[z] = ceil((float)zz/2.0f) * timePerSlice + TR/2.0f;		
			}
			// Even slice
			else
			{
				h_Times[z] = (float)zz/2.0f * timePerSlice;		
			}
			zz++;
		}
		
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			h_Slice_Differences[z] = (h_Times[(int)middle_slice] - h_Times[z])/TR;
		}		
		free(h_Times);
	}
	else if (SLICE_ORDER == CUSTOM)
	{
		middle_slice = SLICE_CUSTOM_REF;
		
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			h_Slice_Differences[z] = (h_Custom_Slice_Times[(int)middle_slice] - h_Custom_Slice_Times[z])/TR;			
		}
	}

	// Flip data from x,y,z,t to x,y,t,z, to be able to copy one slice at a time
	//FlipVolumesXYZTtoXYTZ(h_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

	// Loop over slices
	for (int z = 0; z < EPI_DATA_D; z++)
	{
		// Copy a new slice of data to device, for all time points
		CopyCurrentfMRISliceToDevice(d_Temp_Volumes, h_Volumes, z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
				
		clSetKernelArg(SliceTimingCorrectionKernel, 0, sizeof(cl_mem), &d_Temp_Volumes_Corrected);
		clSetKernelArg(SliceTimingCorrectionKernel, 1, sizeof(cl_mem), &d_Temp_Volumes);
		clSetKernelArg(SliceTimingCorrectionKernel, 2, sizeof(float), &h_Slice_Differences[z]);
		clSetKernelArg(SliceTimingCorrectionKernel, 3, sizeof(int), &EPI_DATA_W);
		clSetKernelArg(SliceTimingCorrectionKernel, 4, sizeof(int), &EPI_DATA_H);
		clSetKernelArg(SliceTimingCorrectionKernel, 5, sizeof(int), &EPI_DATA_D);
		clSetKernelArg(SliceTimingCorrectionKernel, 6, sizeof(int), &EPI_DATA_T);

		runKernelErrorSliceTimingCorrection = clEnqueueNDRangeKernel(commandQueue, SliceTimingCorrectionKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
		clFinish(commandQueue);

		// Copy slice timing corrected slice from device, for all time points
		CopyCurrentfMRISliceToHost(h_Volumes, d_Temp_Volumes_Corrected, z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);		
	}

	// Flip data back from x,y,t,z to x,y,z,t
	//FlipVolumesXYTZtoXYZT(h_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

	clReleaseMemObject(d_Temp_Volumes);
	clReleaseMemObject(d_Temp_Volumes_Corrected);

	deviceMemoryDeallocations += 2;
	allocatedDeviceMemory -= 2 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_T * sizeof(float);

	free(h_Slice_Differences);
}


void BROCCOLI_LIB::PerformSliceTimingCorrectionWrapper()
{
	SetGlobalAndLocalWorkSizesInterpolateVolume(EPI_DATA_W, EPI_DATA_H, 1);

	// Allocate temporary memory, one slice for all time points
	cl_mem d_Temp_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_T * sizeof(float), NULL, NULL);
	cl_mem d_Temp_Volumes_Corrected = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_T * sizeof(float), NULL, NULL);	

	h_Slice_Differences = (float*)malloc(EPI_DATA_D * sizeof(float));

	float middle_slice;

	// Calculate slice differences
	if (SLICE_ORDER == UP)
	{
		middle_slice = myround((float)EPI_DATA_D / 2.0f) - 1.0f;

		for (int z = 0; z < EPI_DATA_D; z++)
		{
			h_Slice_Differences[z] = (middle_slice - (float)z)/((float)EPI_DATA_D);
		}
	}
	else if (SLICE_ORDER == DOWN)
	{
		middle_slice = myround((float)EPI_DATA_D / 2.0f) - 1.0f;

		for (int z = 0; z < EPI_DATA_D; z++)
		{
			h_Slice_Differences[z] = ((float)z - middle_slice)/(float)(EPI_DATA_D);
		}
	}
	else if (SLICE_ORDER == UP_INTERLEAVED)
	{
		middle_slice = (float)EPI_DATA_D - 1.0f;
		
		float* h_Times = (float*)malloc(EPI_DATA_D * sizeof(float));
		float timePerSlice = TR/(float)EPI_DATA_D;

		for (int z = 0; z < EPI_DATA_D; z++)
		{
			// Odd slice
			if (z % 2)
			{
				h_Times[z] = ceil((float)z/2.0f) * timePerSlice + TR/2.0f;		
			}
			// Even slice
			else
			{
				h_Times[z] = (float)z/2.0f * timePerSlice;		
			}
		}
		
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			h_Slice_Differences[z] = (h_Times[(int)middle_slice] - h_Times[z])/TR;
		}		
		free(h_Times);
	}
	else if (SLICE_ORDER == DOWN_INTERLEAVED)
	{
		middle_slice = 0.0f;

		float* h_Times = (float*)malloc(EPI_DATA_D * sizeof(float));
		float timePerSlice = TR/(float)EPI_DATA_D;

		int zz = 0;
		for (int z = EPI_DATA_D-1; z >= 0; z--)
		{
			// Odd slice
			if (zz % 2)
			{
				h_Times[z] = ceil((float)zz/2.0f) * timePerSlice + TR/2.0f;		
			}
			// Even slice
			else
			{
				h_Times[z] = (float)zz/2.0f * timePerSlice;		
			}
			zz++;
		}
		
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			h_Slice_Differences[z] = (h_Times[(int)middle_slice] - h_Times[z])/TR;
		}		
		free(h_Times);
	}
	else if (SLICE_ORDER == CUSTOM)
	{
		middle_slice = SLICE_CUSTOM_REF;
		
		for (int z = 0; z < EPI_DATA_D; z++)
		{
			h_Slice_Differences[z] = (h_Custom_Slice_Times[(int)middle_slice] - h_Custom_Slice_Times[z])/TR;			
		}
	}

	// Flip data from x,y,z,t to x,y,t,z, to be able to copy one slice at a time
	//FlipVolumesXYZTtoXYTZ(h_fMRI_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

	// Loop over slices
	for (int z = 0; z < EPI_DATA_D; z++)
	{
		// Copy a new slice of data to device, for all time points
		CopyCurrentfMRISliceToDevice(d_Temp_Volumes, h_fMRI_Volumes, z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
				
		clSetKernelArg(SliceTimingCorrectionKernel, 0, sizeof(cl_mem), &d_Temp_Volumes_Corrected);
		clSetKernelArg(SliceTimingCorrectionKernel, 1, sizeof(cl_mem), &d_Temp_Volumes);
		clSetKernelArg(SliceTimingCorrectionKernel, 2, sizeof(float), &h_Slice_Differences[z]);
		clSetKernelArg(SliceTimingCorrectionKernel, 3, sizeof(int), &EPI_DATA_W);
		clSetKernelArg(SliceTimingCorrectionKernel, 4, sizeof(int), &EPI_DATA_H);
		clSetKernelArg(SliceTimingCorrectionKernel, 5, sizeof(int), &EPI_DATA_D);
		clSetKernelArg(SliceTimingCorrectionKernel, 6, sizeof(int), &EPI_DATA_T);

		runKernelErrorSliceTimingCorrection = clEnqueueNDRangeKernel(commandQueue, SliceTimingCorrectionKernel, 3, NULL, globalWorkSizeInterpolateVolume, localWorkSizeInterpolateVolume, 0, NULL, NULL);
		clFinish(commandQueue);

		// Copy slice timing corrected slice from device, for all time points
		CopyCurrentfMRISliceToHost(h_fMRI_Volumes, d_Temp_Volumes_Corrected, z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);		
	}

	// Flip data back from x,y,t,z to x,y,z,t
	//FlipVolumesXYTZtoXYZT(h_fMRI_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

	clReleaseMemObject(d_Temp_Volumes);
	clReleaseMemObject(d_Temp_Volumes_Corrected);

	free(h_Slice_Differences);
}

// Only stores one fMRI volume in global memory, to reduce memory usage
void BROCCOLI_LIB::PerformMotionCorrectionWrapper()
{
	int startVolume;

	// Setup all parameters and allocate memory on device
	AlignTwoVolumesLinearSetup(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Set the first volume as the reference volume
	if (!CHANGE_MOTION_CORRECTION_REFERENCE_VOLUME)
	{
		startVolume = 1;
		clEnqueueWriteBuffer(commandQueue, d_Reference_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);
	}
	// Set user provided volume as reference
	else
	{
		startVolume = 0;
		clEnqueueWriteBuffer(commandQueue, d_Reference_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Reference_Volume, 0, NULL, NULL);
	}

	// Translations
	h_Motion_Parameters_Out[0 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters_Out[1 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters_Out[2 * EPI_DATA_T] = 0.0f;

	// Rotations
	h_Motion_Parameters_Out[3 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters_Out[4 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters_Out[5 * EPI_DATA_T] = 0.0f;

	// Run the registration for each volume
	for (size_t t = startVolume; t < EPI_DATA_T; t++)
	{
		// Set a new volume to be aligned
		clEnqueueWriteBuffer(commandQueue, d_Aligned_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), &h_fMRI_Volumes[t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D], 0, NULL, NULL);

		// Also copy the same volume to an image to interpolate from
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {EPI_DATA_W, EPI_DATA_H, EPI_DATA_D};
		clEnqueueCopyBufferToImage(commandQueue, d_Aligned_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);

		// Do rigid registration with only one scale
		AlignTwoVolumesLinear(h_Registration_Parameters_Motion_Correction, h_Rotations, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION, RIGID, INTERPOLATION_MODE);	

		// Copy the corrected volume back to the original pointer, to save host memory
		clEnqueueReadBuffer(commandQueue, d_Aligned_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), &h_fMRI_Volumes[t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D], 0, NULL, NULL);

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
	AlignTwoVolumesLinearCleanup(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
}

// Performs motion correction in place, only storing volumes in host memory
void BROCCOLI_LIB::PerformMotionCorrectionHost(float* h_Volumes)
{
	// Setup all parameters and allocate memory on device
	AlignTwoVolumesLinearSetup(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	PrintMemoryStatus("Inside motion correction host");

	// Set the first volume as the reference volume
	clEnqueueWriteBuffer(commandQueue, d_Reference_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Volumes , 0, NULL, NULL);

	// Translations
	h_Motion_Parameters[0 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters[1 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters[2 * EPI_DATA_T] = 0.0f;

	// Rotations
	h_Motion_Parameters[3 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters[4 * EPI_DATA_T] = 0.0f;
	h_Motion_Parameters[5 * EPI_DATA_T] = 0.0f;

	if ((WRAPPER == BASH) && VERBOS)
	{
		printf(", volume");
	}

	// Run the registration for each volume
	for (size_t t = 1; t < EPI_DATA_T; t++)
	{
		// Set a new volume to be aligned
		clEnqueueWriteBuffer(commandQueue, d_Aligned_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), &h_Volumes[t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D], 0, NULL, NULL);

		// Also copy the same volume to an image to interpolate from
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {EPI_DATA_W, EPI_DATA_H, EPI_DATA_D};
		clEnqueueCopyBufferToImage(commandQueue, d_Aligned_Volume, d_Original_Volume, 0, origin, region, 0, NULL, NULL);

		// Do rigid registration with only one scale
		AlignTwoVolumesLinear(h_Registration_Parameters_Motion_Correction, h_Rotations, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION, RIGID, INTERPOLATION_MODE);	

		// Copy the corrected volume to the corrected volumes
		clEnqueueReadBuffer(commandQueue, d_Aligned_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), &h_Volumes[t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D], 0, NULL, NULL);

		// Write the total parameter vector to host

		// Translations
		h_Motion_Parameters[t + 0 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[0] * EPI_VOXEL_SIZE_X;
		h_Motion_Parameters[t + 1 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[1] * EPI_VOXEL_SIZE_Y;
		h_Motion_Parameters[t + 2 * EPI_DATA_T] = h_Registration_Parameters_Motion_Correction[2] * EPI_VOXEL_SIZE_Z;

		// Rotations
		h_Motion_Parameters[t + 3 * EPI_DATA_T] = h_Rotations[0];
		h_Motion_Parameters[t + 4 * EPI_DATA_T] = h_Rotations[1];
		h_Motion_Parameters[t + 5 * EPI_DATA_T] = h_Rotations[2];

		if ((WRAPPER == BASH) && VERBOS)
		{
			printf(", %zu",t);
			fflush(stdout);
		}
	}

	// Cleanup allocated memory
	AlignTwoVolumesLinearCleanup(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
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
	for (size_t t = 1; t < EPI_DATA_T; t++)
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
	AlignTwoVolumesLinearCleanup(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
}


// Slow way of calculating the sum of a volume
float BROCCOLI_LIB::CalculateSum(cl_mem d_Volume, size_t DATA_W, size_t DATA_H, size_t DATA_D)
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
float BROCCOLI_LIB::CalculateMax(cl_mem d_Volume, size_t DATA_W, size_t DATA_H, size_t DATA_D)
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



// Ugly way of calculating max of floats, since there is no atomic function for floats
float BROCCOLI_LIB::CalculateMaxAtomic(cl_mem d_Array, size_t N)
{
	SetGlobalAndLocalWorkSizesCalculateMax(N, 1, 1);

	cl_mem d_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, NULL);
	SetMemory(d_Mask, 0.0f, N);

	cl_mem d_Max_Value = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	SetMemory(d_Max_Value, -1000000, 1);

	int one = 1;
	clSetKernelArg(CalculateMaxAtomicKernel, 0, sizeof(cl_mem), &d_Max_Value);
	clSetKernelArg(CalculateMaxAtomicKernel, 1, sizeof(cl_mem), &d_Array);
	clSetKernelArg(CalculateMaxAtomicKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateMaxAtomicKernel, 3, sizeof(int), &N);
	clSetKernelArg(CalculateMaxAtomicKernel, 4, sizeof(int), &one);
	clSetKernelArg(CalculateMaxAtomicKernel, 5, sizeof(int), &one);

	runKernelErrorCalculateMaxAtomic = clEnqueueNDRangeKernel(commandQueue, CalculateMaxAtomicKernel, 3, NULL, globalWorkSizeCalculateMaxAtomic, localWorkSizeCalculateMaxAtomic, 0, NULL, NULL);
	clFinish(commandQueue);

	int max;
	clEnqueueReadBuffer(commandQueue, d_Max_Value, CL_TRUE, 0, sizeof(int), &max, 0, NULL, NULL);

	clReleaseMemObject(d_Mask);
	clReleaseMemObject(d_Max_Value);

	return (float)((float)max/10000.0f);
}


// Ugly way of calculating max of floats, since there is no atomic function for floats
float BROCCOLI_LIB::CalculateMaxAtomic(cl_mem d_Volume, cl_mem d_Mask, size_t DATA_W, size_t DATA_H, size_t DATA_D)
{
	SetGlobalAndLocalWorkSizesCalculateMax(DATA_W, DATA_H, DATA_D);

	cl_mem d_Max_Value = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

	SetMemory(d_Max_Value, -1000000, 1);

	clSetKernelArg(CalculateMaxAtomicKernel, 0, sizeof(cl_mem), &d_Max_Value);
	clSetKernelArg(CalculateMaxAtomicKernel, 1, sizeof(cl_mem), &d_Volume);
	clSetKernelArg(CalculateMaxAtomicKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateMaxAtomicKernel, 3, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateMaxAtomicKernel, 4, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateMaxAtomicKernel, 5, sizeof(int), &DATA_D);

	runKernelErrorCalculateMaxAtomic = clEnqueueNDRangeKernel(commandQueue, CalculateMaxAtomicKernel, 3, NULL, globalWorkSizeCalculateMaxAtomic, localWorkSizeCalculateMaxAtomic, 0, NULL, NULL);
	clFinish(commandQueue);

	int max;
	clEnqueueReadBuffer(commandQueue, d_Max_Value, CL_TRUE, 0, sizeof(int), &max, 0, NULL, NULL);

	clReleaseMemObject(d_Max_Value);

	return (float)((float)max/10000.0f);
}

// Thresholds a volume
void BROCCOLI_LIB::ThresholdVolume(cl_mem d_Thresholded_Volume, cl_mem d_Volume_To_Threshold, float threshold, int DATA_W, int DATA_H, int DATA_D)
{
	SetGlobalAndLocalWorkSizesThresholdVolume(DATA_W, DATA_H, DATA_D);

	clSetKernelArg(ThresholdVolumeKernel, 0, sizeof(cl_mem), &d_Thresholded_Volume);
	clSetKernelArg(ThresholdVolumeKernel, 1, sizeof(cl_mem), &d_Volume_To_Threshold);
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

	// Copy the first fMRI volume from host
	clEnqueueWriteBuffer(commandQueue, d_EPI, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_fMRI_Volumes , 0, NULL, NULL);

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


// Performs normalized smoothing, loops over volumes and copies one volume to device, then copies back result
void BROCCOLI_LIB::PerformSmoothingNormalizedHost(float* h_Volumes,
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
	cl_mem d_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Convolved_Rows = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Convolved_Columns = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_W * DATA_H * DATA_D * sizeof(float), NULL, NULL);

	deviceMemoryAllocations += 3;
	allocatedDeviceMemory += 3 * DATA_W * DATA_H * DATA_D * sizeof(float);

	PrintMemoryStatus("Inside smoothing normalized host");

	int zero = 0;

	// Set arguments for the kernels
	clSetKernelArg(SeparableConvolutionRowsKernel, 0, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionRowsKernel, 1, sizeof(cl_mem), &d_Volume);
	clSetKernelArg(SeparableConvolutionRowsKernel, 2, sizeof(cl_mem), &d_Certainty);
	clSetKernelArg(SeparableConvolutionRowsKernel, 3, sizeof(cl_mem), &c_Smoothing_Filter_Y);
	clSetKernelArg(SeparableConvolutionRowsKernel, 4, sizeof(int), &zero);
	clSetKernelArg(SeparableConvolutionRowsKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionRowsKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionRowsKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionRowsKernel, 8, sizeof(int), &DATA_T);

	clSetKernelArg(SeparableConvolutionColumnsKernel, 0, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 1, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_X);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 3, sizeof(int), &zero);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 4, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 5, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 6, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 7, sizeof(int), &DATA_T);

	clSetKernelArg(SeparableConvolutionRodsKernel, 0, sizeof(cl_mem), &d_Volume);
	clSetKernelArg(SeparableConvolutionRodsKernel, 1, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionRodsKernel, 2, sizeof(cl_mem), &d_Smoothed_Certainty);
	clSetKernelArg(SeparableConvolutionRodsKernel, 3, sizeof(cl_mem), &c_Smoothing_Filter_Z);
	clSetKernelArg(SeparableConvolutionRodsKernel, 4, sizeof(int), &zero);
	clSetKernelArg(SeparableConvolutionRodsKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(SeparableConvolutionRodsKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(SeparableConvolutionRodsKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(SeparableConvolutionRodsKernel, 8, sizeof(int), &DATA_T);

	// Loop over volumes
	for (size_t v = 0; v < DATA_T; v++)
	{
		// Copy new volume to device
		clEnqueueWriteBuffer(commandQueue, d_Volume, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), &h_Volumes[v * DATA_W * DATA_H * DATA_D], 0, NULL, NULL);

		runKernelErrorSeparableConvolutionRows = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRowsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRows, localWorkSizeSeparableConvolutionRows, 0, NULL, NULL);
		clFinish(commandQueue);

		runKernelErrorSeparableConvolutionColumns = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionColumnsKernel, 3, NULL, globalWorkSizeSeparableConvolutionColumns, localWorkSizeSeparableConvolutionColumns, 0, NULL, NULL);
		clFinish(commandQueue);

		runKernelErrorSeparableConvolutionRods = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRodsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRods, localWorkSizeSeparableConvolutionRods, 0, NULL, NULL);
		clFinish(commandQueue);

		MultiplyVolumes(d_Volume, d_Certainty, DATA_W, DATA_H, DATA_D);

		// Copy smoothed volume back to host
		clEnqueueReadBuffer(commandQueue, d_Volume, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), &h_Volumes[v * DATA_W * DATA_H * DATA_D], 0, NULL, NULL);
	}

	// Free temporary memory
	clReleaseMemObject(c_Smoothing_Filter_X);
	clReleaseMemObject(c_Smoothing_Filter_Y);
	clReleaseMemObject(c_Smoothing_Filter_Z);

	clReleaseMemObject(d_Volume);
	clReleaseMemObject(d_Convolved_Rows);
	clReleaseMemObject(d_Convolved_Columns);

	deviceMemoryDeallocations += 3;
	allocatedDeviceMemory -= 3 * DATA_W * DATA_H * DATA_D * sizeof(float);
}


// Performs normalized smoothing, loops over volumes and copies one volume to device, then copies back result
void BROCCOLI_LIB::PerformSmoothingNormalizedHostWrapper()
{
	allocatedDeviceMemory = 0;

	SetGlobalAndLocalWorkSizesSeparableConvolution(EPI_DATA_W,EPI_DATA_H,EPI_DATA_D);

	// Allocate memory for certainty
	cl_mem d_Certainty = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Smoothed_Certainty = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	if (!AUTO_MASK)
	{
		// Copy certainty from host
		clEnqueueWriteBuffer(commandQueue, d_Certainty, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Certainty, 0, NULL, NULL);
	}
	// Make a mask to use as certainty
	else
	{
		d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

		SegmentEPIData();
		// Copy mask to certainty
		clEnqueueCopyBuffer(commandQueue, d_EPI_Mask, d_Certainty, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), 0, NULL, NULL);
		// Copy mask to host
		clEnqueueReadBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Certainty, 0, NULL, NULL);
		clReleaseMemObject(d_EPI_Mask);
	}

	// Create the smoothing filters for the requested FWHM
	CreateSmoothingFilters(h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, SMOOTHING_FILTER_SIZE, EPI_Smoothing_FWHM, EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z);
	PerformSmoothing(d_Smoothed_Certainty, d_Certainty, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

	// Allocate memory for smoothing filters
	c_Smoothing_Filter_X = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Y = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);
	c_Smoothing_Filter_Z = clCreateBuffer(context, CL_MEM_READ_ONLY, SMOOTHING_FILTER_SIZE * sizeof(float), NULL, NULL);

	// Copy smoothing filters to constant memory
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_X, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_X , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Y, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Y , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Smoothing_Filter_Z, CL_TRUE, 0, SMOOTHING_FILTER_SIZE * sizeof(float), h_Smoothing_Filter_Z , 0, NULL, NULL);

	// Allocate temporary memory
	cl_mem d_Volume = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Convolved_Rows = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	cl_mem d_Convolved_Columns = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	deviceMemoryAllocations += 5;
	allocatedDeviceMemory += 5 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);

	PrintMemoryStatus("Inside smoothing normalized host");

	int zero = 0;

	// Set arguments for the kernels
	clSetKernelArg(SeparableConvolutionRowsKernel, 0, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionRowsKernel, 1, sizeof(cl_mem), &d_Volume);
	clSetKernelArg(SeparableConvolutionRowsKernel, 2, sizeof(cl_mem), &d_Certainty);
	clSetKernelArg(SeparableConvolutionRowsKernel, 3, sizeof(cl_mem), &c_Smoothing_Filter_Y);
	clSetKernelArg(SeparableConvolutionRowsKernel, 4, sizeof(int), &zero);
	clSetKernelArg(SeparableConvolutionRowsKernel, 5, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(SeparableConvolutionRowsKernel, 6, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(SeparableConvolutionRowsKernel, 7, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(SeparableConvolutionRowsKernel, 8, sizeof(int), &EPI_DATA_T);

	clSetKernelArg(SeparableConvolutionColumnsKernel, 0, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 1, sizeof(cl_mem), &d_Convolved_Rows);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 2, sizeof(cl_mem), &c_Smoothing_Filter_X);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 3, sizeof(int), &zero);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 4, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 5, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 6, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(SeparableConvolutionColumnsKernel, 7, sizeof(int), &EPI_DATA_T);

	clSetKernelArg(SeparableConvolutionRodsKernel, 0, sizeof(cl_mem), &d_Volume);
	clSetKernelArg(SeparableConvolutionRodsKernel, 1, sizeof(cl_mem), &d_Convolved_Columns);
	clSetKernelArg(SeparableConvolutionRodsKernel, 2, sizeof(cl_mem), &d_Smoothed_Certainty);
	clSetKernelArg(SeparableConvolutionRodsKernel, 3, sizeof(cl_mem), &c_Smoothing_Filter_Z);
	clSetKernelArg(SeparableConvolutionRodsKernel, 4, sizeof(int), &zero);
	clSetKernelArg(SeparableConvolutionRodsKernel, 5, sizeof(int), &EPI_DATA_W);
	clSetKernelArg(SeparableConvolutionRodsKernel, 6, sizeof(int), &EPI_DATA_H);
	clSetKernelArg(SeparableConvolutionRodsKernel, 7, sizeof(int), &EPI_DATA_D);
	clSetKernelArg(SeparableConvolutionRodsKernel, 8, sizeof(int), &EPI_DATA_T);

	// Loop over volumes
	for (size_t v = 0; v < EPI_DATA_T; v++)
	{
		// Copy new volume to device
		clEnqueueWriteBuffer(commandQueue, d_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), &h_fMRI_Volumes[v * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D], 0, NULL, NULL);

		runKernelErrorSeparableConvolutionRows = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRowsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRows, localWorkSizeSeparableConvolutionRows, 0, NULL, NULL);
		clFinish(commandQueue);

		runKernelErrorSeparableConvolutionColumns = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionColumnsKernel, 3, NULL, globalWorkSizeSeparableConvolutionColumns, localWorkSizeSeparableConvolutionColumns, 0, NULL, NULL);
		clFinish(commandQueue);

		runKernelErrorSeparableConvolutionRods = clEnqueueNDRangeKernel(commandQueue, SeparableConvolutionRodsKernel, 3, NULL, globalWorkSizeSeparableConvolutionRods, localWorkSizeSeparableConvolutionRods, 0, NULL, NULL);
		clFinish(commandQueue);

		MultiplyVolumes(d_Volume, d_Certainty, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

		// Copy smoothed volume back to host
		clEnqueueReadBuffer(commandQueue, d_Volume, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), &h_fMRI_Volumes[v * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D], 0, NULL, NULL);

		if ((WRAPPER == BASH) && VERBOS)
		{
			printf(", %zu",v);
			fflush(stdout);
		}
	}

	// Free temporary memory
	clReleaseMemObject(c_Smoothing_Filter_X);
	clReleaseMemObject(c_Smoothing_Filter_Y);
	clReleaseMemObject(c_Smoothing_Filter_Z);

	clReleaseMemObject(d_Certainty);
	clReleaseMemObject(d_Smoothed_Certainty);

	clReleaseMemObject(d_Volume);
	clReleaseMemObject(d_Convolved_Rows);
	clReleaseMemObject(d_Convolved_Columns);

	deviceMemoryDeallocations += 5;
	allocatedDeviceMemory -= 5 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);
}

// Performs detrending of an fMRI dataset (removes mean, linear trend, quadratic trend, cubic trend)
void BROCCOLI_LIB::PerformDetrending(cl_mem d_Detrended_Volumes, cl_mem d_Volumes, size_t DATA_W, size_t DATA_H, size_t DATA_D, size_t DATA_T)
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

	h_Censored_Timepoints = (float*)malloc(DATA_T * sizeof(float));
	c_Censored_Timepoints = clCreateBuffer(context, CL_MEM_READ_ONLY, DATA_T * sizeof(float), NULL, NULL);

	for (int t = 0; t < DATA_T; t++)
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
	clSetKernelArg(RemoveLinearFitKernel, 2, sizeof(cl_mem), &d_Beta_Volumes);
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
	clReleaseMemObject(c_Censored_Timepoints);
	clReleaseMemObject(c_X_Detrend);
	clReleaseMemObject(c_xtxxt_Detrend);
}


// Performs detrending of an fMRI dataset (removes mean, linear trend, quadratic trend, cubic trend), for one slice
void BROCCOLI_LIB::PerformDetrendingSlice(cl_mem d_Detrended_Volumes, cl_mem d_Volumes, size_t slice, size_t DATA_W, size_t DATA_H, size_t DATA_D, size_t DATA_T)
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

	SetGlobalAndLocalWorkSizesStatisticalCalculations(DATA_W, DATA_H, 1);

	h_Censored_Timepoints = (float*)malloc(DATA_T * sizeof(float));
	c_Censored_Timepoints = clCreateBuffer(context, CL_MEM_READ_ONLY, DATA_T * sizeof(float), NULL, NULL);

	for (int t = 0; t < DATA_T; t++)
	{
		h_Censored_Timepoints[t] = 1.0f;
	}
	clEnqueueWriteBuffer(commandQueue, c_Censored_Timepoints, CL_TRUE, 0, EPI_DATA_T * sizeof(float), h_Censored_Timepoints , 0, NULL, NULL);

	// Estimate beta weights
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 3, sizeof(cl_mem), &c_xtxxt_Detrend);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 4, sizeof(cl_mem), &c_Censored_Timepoints);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 5, sizeof(int),    &DATA_W);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 6, sizeof(int),    &DATA_H);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 7, sizeof(int),    &DATA_D);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 8, sizeof(int),    &DATA_T);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 9, sizeof(int),    &NUMBER_OF_DETRENDING_REGRESSORS);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 10, sizeof(int),   &slice);

	runKernelErrorCalculateBetaWeightsGLM = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMSliceKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Remove linear fit
	clSetKernelArg(RemoveLinearFitSliceKernel, 0, sizeof(cl_mem), &d_Detrended_Volumes);
	clSetKernelArg(RemoveLinearFitSliceKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(RemoveLinearFitSliceKernel, 2, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(RemoveLinearFitSliceKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(RemoveLinearFitSliceKernel, 4, sizeof(cl_mem), &c_X_Detrend);
	clSetKernelArg(RemoveLinearFitSliceKernel, 5, sizeof(int),    &DATA_W);
	clSetKernelArg(RemoveLinearFitSliceKernel, 6, sizeof(int),    &DATA_H);
	clSetKernelArg(RemoveLinearFitSliceKernel, 7, sizeof(int),    &DATA_D);
	clSetKernelArg(RemoveLinearFitSliceKernel, 8, sizeof(int),    &DATA_T);
	clSetKernelArg(RemoveLinearFitSliceKernel, 9, sizeof(int),    &NUMBER_OF_DETRENDING_REGRESSORS);
	clSetKernelArg(RemoveLinearFitSliceKernel, 10, sizeof(int),   &slice);

	runKernelErrorRemoveLinearFit = clEnqueueNDRangeKernel(commandQueue, RemoveLinearFitSliceKernel, 3, NULL, globalWorkSizeRemoveLinearFit, localWorkSizeRemoveLinearFit, 0, NULL, NULL);
	clFinish(commandQueue);

	// Free host memory
	free(h_Censored_Timepoints);
	free(h_X_Detrend);
	free(h_xtxxt_Detrend);

	// Free memory
	clReleaseMemObject(c_Censored_Timepoints);
	clReleaseMemObject(c_X_Detrend);
	clReleaseMemObject(c_xtxxt_Detrend);
}

// Removes the linear fit between detrending regressors (mean, linear trend, quadratic trend, cubic trend) and motion regressors
void BROCCOLI_LIB::PerformDetrendingAndMotionRegression(cl_mem d_Regressed_Volumes, cl_mem d_Volumes, size_t DATA_W, size_t DATA_H, size_t DATA_D, size_t DATA_T)
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
	runKernelErrorRemoveLinearFitSlice = clEnqueueNDRangeKernel(commandQueue, RemoveLinearFitKernel, 3, NULL, globalWorkSizeRemoveLinearFit, localWorkSizeRemoveLinearFit, 0, NULL, NULL);
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


// Removes the linear fit between detrending regressors (mean, linear trend, quadratic trend, cubic trend) and motion regressors, for one slice
void BROCCOLI_LIB::PerformDetrendingAndMotionRegressionSlice(cl_mem d_Regressed_Volumes, cl_mem d_Volumes, size_t slice, size_t DATA_W, size_t DATA_H, size_t DATA_D, size_t DATA_T)
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

	SetGlobalAndLocalWorkSizesStatisticalCalculations(DATA_W, DATA_H, 1);

	h_Censored_Timepoints = (float*)malloc(EPI_DATA_T * sizeof(float));
	c_Censored_Timepoints = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(float), NULL, NULL);

	for (int t = 0; t < EPI_DATA_T; t++)
	{
		h_Censored_Timepoints[t] = 1.0f;
	}
	clEnqueueWriteBuffer(commandQueue, c_Censored_Timepoints, CL_TRUE, 0, EPI_DATA_T * sizeof(float), h_Censored_Timepoints , 0, NULL, NULL);

	// Estimate beta weights
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 3, sizeof(cl_mem), &c_xtxxt_Detrend);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 4, sizeof(cl_mem), &c_Censored_Timepoints);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 8, sizeof(int), &DATA_T);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 9, sizeof(int), &NUMBER_OF_DETRENDING_AND_MOTION_REGRESSORS);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel,10, sizeof(int), &slice);
	runKernelErrorCalculateBetaWeightsGLMSlice = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMSliceKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Remove linear fit
	clSetKernelArg(RemoveLinearFitSliceKernel, 0, sizeof(cl_mem), &d_Regressed_Volumes);
	clSetKernelArg(RemoveLinearFitSliceKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(RemoveLinearFitSliceKernel, 2, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(RemoveLinearFitSliceKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(RemoveLinearFitSliceKernel, 4, sizeof(cl_mem), &c_X_Detrend);
	clSetKernelArg(RemoveLinearFitSliceKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(RemoveLinearFitSliceKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(RemoveLinearFitSliceKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(RemoveLinearFitSliceKernel, 8, sizeof(int), &DATA_T);
	clSetKernelArg(RemoveLinearFitSliceKernel, 9, sizeof(int), &NUMBER_OF_DETRENDING_AND_MOTION_REGRESSORS);
	clSetKernelArg(RemoveLinearFitSliceKernel,10, sizeof(int), &slice);
	runKernelErrorRemoveLinearFitSlice = clEnqueueNDRangeKernel(commandQueue, RemoveLinearFitSliceKernel, 3, NULL, globalWorkSizeRemoveLinearFit, localWorkSizeRemoveLinearFit, 0, NULL, NULL);
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

// Removes the linear fit between regressors and data, regressors have already been setup
void BROCCOLI_LIB::PerformRegression(cl_mem d_Regressed_Volumes, cl_mem d_Volumes, size_t DATA_W, size_t DATA_H, size_t DATA_D, size_t DATA_T)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(DATA_W, DATA_H, DATA_D);

	h_Censored_Timepoints = (float*)malloc(DATA_T * sizeof(float));
	c_Censored_Timepoints = clCreateBuffer(context, CL_MEM_READ_ONLY, DATA_T * sizeof(float), NULL, NULL);

	for (int t = 0; t < DATA_T; t++)
	{
		h_Censored_Timepoints[t] = 1.0f;
	}
	clEnqueueWriteBuffer(commandQueue, c_Censored_Timepoints, CL_TRUE, 0, DATA_T * sizeof(float), h_Censored_Timepoints , 0, NULL, NULL);

	// Estimate beta weights
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 3, sizeof(cl_mem), &c_xtxxt_GLM);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 4, sizeof(cl_mem), &c_Censored_Timepoints);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 8, sizeof(int), &DATA_T);
	clSetKernelArg(CalculateBetaWeightsGLMKernel, 9, sizeof(int), &NUMBER_OF_TOTAL_GLM_REGRESSORS);

	runKernelErrorCalculateBetaWeightsGLM = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Remove linear fit
	clSetKernelArg(RemoveLinearFitKernel, 0, sizeof(cl_mem), &d_Regressed_Volumes);
	clSetKernelArg(RemoveLinearFitKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(RemoveLinearFitKernel, 2, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(RemoveLinearFitKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(RemoveLinearFitKernel, 4, sizeof(cl_mem), &c_X_GLM);
	clSetKernelArg(RemoveLinearFitKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(RemoveLinearFitKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(RemoveLinearFitKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(RemoveLinearFitKernel, 8, sizeof(int), &DATA_T);
	clSetKernelArg(RemoveLinearFitKernel, 9, sizeof(int), &NUMBER_OF_TOTAL_GLM_REGRESSORS);

	runKernelErrorRemoveLinearFit = clEnqueueNDRangeKernel(commandQueue, RemoveLinearFitKernel, 3, NULL, globalWorkSizeRemoveLinearFit, localWorkSizeRemoveLinearFit, 0, NULL, NULL);
	clFinish(commandQueue);

	// Free host memory
	free(h_Censored_Timepoints);

	// Free constant memory
	clReleaseMemObject(c_Censored_Timepoints);
}



// Removes the linear fit between regressors and data, regressors have already been setup, for one slice
void BROCCOLI_LIB::PerformRegressionSlice(cl_mem d_Regressed_Volumes, cl_mem d_Volumes, size_t slice, size_t DATA_W, size_t DATA_H, size_t DATA_D, size_t DATA_T)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(DATA_W, DATA_H, 1);

	h_Censored_Timepoints = (float*)malloc(DATA_T * sizeof(float));
	c_Censored_Timepoints = clCreateBuffer(context, CL_MEM_READ_ONLY, DATA_T * sizeof(float), NULL, NULL);

	for (int t = 0; t < DATA_T; t++)
	{
		h_Censored_Timepoints[t] = 1.0f;
	}
	clEnqueueWriteBuffer(commandQueue, c_Censored_Timepoints, CL_TRUE, 0, DATA_T * sizeof(float), h_Censored_Timepoints , 0, NULL, NULL);

	// Estimate beta weights
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 0, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 3, sizeof(cl_mem), &c_xtxxt_GLM);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 4, sizeof(cl_mem), &c_Censored_Timepoints);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 5, sizeof(int), &DATA_W);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 6, sizeof(int), &DATA_H);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 7, sizeof(int), &DATA_D);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 8, sizeof(int), &DATA_T);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 9, sizeof(int), &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(CalculateBetaWeightsGLMSliceKernel, 10, sizeof(int), &slice);

	runKernelErrorCalculateBetaWeightsGLM = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMSliceKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	// Remove linear fit
	clSetKernelArg(RemoveLinearFitSliceKernel, 0, sizeof(cl_mem), &d_Regressed_Volumes);
	clSetKernelArg(RemoveLinearFitSliceKernel, 1, sizeof(cl_mem), &d_Volumes);
	clSetKernelArg(RemoveLinearFitSliceKernel, 2, sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(RemoveLinearFitSliceKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(RemoveLinearFitSliceKernel, 4, sizeof(cl_mem), &c_X_GLM);
	clSetKernelArg(RemoveLinearFitSliceKernel, 5, sizeof(int),  &DATA_W);
	clSetKernelArg(RemoveLinearFitSliceKernel, 6, sizeof(int),  &DATA_H);
	clSetKernelArg(RemoveLinearFitSliceKernel, 7, sizeof(int),  &DATA_D);
	clSetKernelArg(RemoveLinearFitSliceKernel, 8, sizeof(int),  &DATA_T);
	clSetKernelArg(RemoveLinearFitSliceKernel, 9, sizeof(int),  &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(RemoveLinearFitSliceKernel, 10, sizeof(int), &slice);

	runKernelErrorRemoveLinearFit = clEnqueueNDRangeKernel(commandQueue, RemoveLinearFitSliceKernel, 3, NULL, globalWorkSizeRemoveLinearFit, localWorkSizeRemoveLinearFit, 0, NULL, NULL);
	clFinish(commandQueue);

	// Free host memory
	free(h_Censored_Timepoints);

	// Free constant memory
	clReleaseMemObject(c_Censored_Timepoints);
}

void BROCCOLI_LIB::PerformGLMTTestFirstLevelWrapper()
{
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + REGRESS_GLOBALMEAN + NUMBER_OF_MOTION_REGRESSORS * REGRESS_MOTION;

	// Copy mask to device
	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask , 0, NULL, NULL);

	deviceMemoryAllocations += 1;
	allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);

	CalculateNumberOfBrainVoxels(d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Check amount of global memory, compared to required memory
	bool largeMemory = true;
	bool ttest = true;
	size_t totalRequiredMemory;

	if (BETAS_ONLY || CONTRASTS_ONLY || BETAS_AND_CONTRASTS_ONLY)
	{
		totalRequiredMemory = allocatedDeviceMemory + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float) * 2 + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float) + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float) +  EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float) * 7; 
		ttest = false;
	}	
	else
	{
		totalRequiredMemory = allocatedDeviceMemory + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float) * 2 + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float) + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float) * 2 + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float) * 6 + NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float) + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float);
	}
	totalRequiredMemory /= (1024*1024);

	if (totalRequiredMemory > globalMemorySize)
	{
		largeMemory = false;
		if ((WRAPPER == BASH) && VERBOS)
		{
			printf("Cannot run the GLM the whole volume at once, doing slice by slice. Required device memory for GLM is %zu MB, global memory is %zu MB ! \n",totalRequiredMemory,globalMemorySize);
		}
	}
	else
	{
		if ((WRAPPER == BASH) && VERBOS)
		{
			printf("Sufficient memory for running the GLM the whole volume at once! Required device memory for GLM is %zu MB, global memory is %zu MB ! \n",totalRequiredMemory,globalMemorySize);
		}
	}

	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	// Allocate memory for one slice for all time points, loop over slices to save memory
	if (!largeMemory)
	{
		d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
		d_Whitened_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
		allocatedDeviceMemory += 2 * EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float);
		deviceMemoryAllocations += 2;

		// Slice based t-test stores residuals separately, volume based t-test does not
		if (ttest)
		{
			d_Residuals = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
			allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float);
			deviceMemoryAllocations += 1;
		}
	}
	else
	{
		d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
		d_Whitened_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
		allocatedDeviceMemory += 2 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);
		deviceMemoryAllocations += 2;
	}
	
	d_Smoothed_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Contrast_Volumes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	deviceMemoryAllocations += 5;
	allocatedDeviceMemory += (EPI_DATA_W * EPI_DATA_H * EPI_DATA_D)*(1 + NUMBER_OF_TOTAL_GLM_REGRESSORS + NUMBER_OF_CONTRASTS + NUMBER_OF_CONTRASTS + 1) * sizeof(float);

	d_AR1_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR2_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR3_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR4_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	deviceMemoryAllocations += 4;
	allocatedDeviceMemory += 4 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);

	PrintMemoryStatus("Before GLM");

	h_X_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
	h_xtxxt_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
	h_Contrasts = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float));
	h_ctxtxc_GLM = (float*)malloc(NUMBER_OF_CONTRASTS * sizeof(float));
	h_X_GLM_With_Temporal_Derivatives = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * 2 * EPI_DATA_T * sizeof(float));
	h_X_GLM_Convolved = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * (USE_TEMPORAL_DERIVATIVES+1) * EPI_DATA_T * sizeof(float));
	h_Global_Mean = (float*)malloc(EPI_DATA_T * sizeof(float));
	h_Motion_Parameters = (float*)malloc(EPI_DATA_T * NUMBER_OF_MOTION_REGRESSORS * sizeof(float));

	if (REGRESS_MOTION)
	{
		for (size_t i = 0; i < NUMBER_OF_MOTION_REGRESSORS * EPI_DATA_T; i++)
		{
			h_Motion_Parameters[i] = h_Motion_Parameters_Out[i];
		}
	}
	if (REGRESS_GLOBALMEAN)
	{
		CalculateGlobalMeans(h_fMRI_Volumes);
	}

	SetupTTestFirstLevel();

	// Copy mask to device
	clEnqueueWriteBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask , 0, NULL, NULL);

	// Copy model to device
	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM , 0, NULL, NULL);
	
	if (WRITE_DESIGNMATRIX)
	{
		for (int t = 0; t < EPI_DATA_T; t++)
		{
			for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
			{
				h_X_GLM_Out[t + r * EPI_DATA_T] = h_X_GLM[t + r * EPI_DATA_T];
				h_xtxxt_GLM_Out[t + r * EPI_DATA_T] = h_xtxxt_GLM[t + r * EPI_DATA_T];
			}
		}
	}

	// Run the actual GLM
	if (BETAS_ONLY || CONTRASTS_ONLY || BETAS_AND_CONTRASTS_ONLY)
	{
		if (!largeMemory)
		{
			CalculateBetaWeightsAndContrastsFirstLevelSlices(h_fMRI_Volumes);	

			if (WRITE_RESIDUALS_EPI)
			{
				// Loop over slices, to save memory
				for (int slice = 0; slice < EPI_DATA_D; slice++)
				{
					// Copy fMRI data to the device, for the current slice
					CopyCurrentfMRISliceToDevice(d_fMRI_Volumes, h_fMRI_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
					// Perform the regression
					PerformRegressionSlice(d_Whitened_fMRI_Volumes, d_fMRI_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
					// Copy back the current slice
					CopyCurrentfMRISliceToHost(h_Residuals_EPI, d_Whitened_fMRI_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
				}
			}
		}
		else
		{
			CalculateBetaWeightsAndContrastsFirstLevel(h_fMRI_Volumes);	

			if (WRITE_RESIDUALS_EPI)
			{
				PerformRegression(d_Whitened_fMRI_Volumes, d_fMRI_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
				clEnqueueReadBuffer(commandQueue, d_Whitened_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Residuals_EPI, 0, NULL, NULL);
			}			
		}
	}
	else
	{
		if (!largeMemory)
		{
			CalculateStatisticalMapsGLMTTestFirstLevelSlices(h_fMRI_Volumes,3);
		}
		else
		{
			CalculateStatisticalMapsGLMTTestFirstLevel(h_fMRI_Volumes,3);
		}
	}

	clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_EPI, 0, NULL, NULL);

	if (!BETAS_ONLY)
	{
		clEnqueueReadBuffer(commandQueue, d_Contrast_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrast_Volumes_EPI, 0, NULL, NULL);
	}
	if (!BETAS_ONLY && !BETAS_AND_CONTRASTS_ONLY)
	{
		clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_EPI, 0, NULL, NULL);
	}

	clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);
	
	if (WRITE_AR_ESTIMATES_EPI)
	{
		clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);
	}		

	// Cleanup host memory
	free(h_X_GLM);
	free(h_xtxxt_GLM);
	free(h_Contrasts);
	free(h_ctxtxc_GLM);
	free(h_X_GLM_With_Temporal_Derivatives);
	free(h_X_GLM_Convolved);
	free(h_Global_Mean);
	free(h_Motion_Parameters);

	// Cleanup device memory
	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);
	
	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Whitened_fMRI_Volumes);

	if (!largeMemory)
	{
		allocatedDeviceMemory -= 2 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_T * sizeof(float);
	}
	else
	{
		allocatedDeviceMemory -= 2 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);
	}
	deviceMemoryDeallocations += 2;

	if (!largeMemory && ttest)
	{
		clReleaseMemObject(d_Residuals);
		allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_T * sizeof(float);
		deviceMemoryDeallocations += 1;
	}

	clReleaseMemObject(d_EPI_Mask);
	clReleaseMemObject(d_Smoothed_EPI_Mask);
	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Contrast_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Residual_Variances);

	allocatedDeviceMemory -= (EPI_DATA_W * EPI_DATA_H * EPI_DATA_D)*(2 + NUMBER_OF_TOTAL_GLM_REGRESSORS + NUMBER_OF_CONTRASTS + NUMBER_OF_CONTRASTS + 1) * sizeof(float);
	deviceMemoryDeallocations += 6;

	clReleaseMemObject(d_AR1_Estimates);
	clReleaseMemObject(d_AR2_Estimates);
	clReleaseMemObject(d_AR3_Estimates);
	clReleaseMemObject(d_AR4_Estimates);

	allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * 4 * sizeof(float);
	deviceMemoryDeallocations += 4;

	PrintMemoryStatus("After GLM");
}


// Used for testing of F-test only
void BROCCOLI_LIB::PerformGLMFTestFirstLevelWrapper()
{
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + REGRESS_GLOBALMEAN + NUMBER_OF_MOTION_REGRESSORS * REGRESS_MOTION;

	// Copy mask to device
	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask , 0, NULL, NULL);

	deviceMemoryAllocations += 1;
	allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);

	CalculateNumberOfBrainVoxels(d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Check amount of global memory, compared to required memory
	bool largeMemory = true;
	size_t totalRequiredMemory;

	totalRequiredMemory = allocatedDeviceMemory + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float) * 2 + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float) + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float) * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float) * 6 + NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float) + EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float);
	
	totalRequiredMemory /= (1024*1024);

	if (totalRequiredMemory > globalMemorySize)
	{
		largeMemory = false;
		if ((WRAPPER == BASH) && VERBOS)
		{
			printf("Cannot run the GLM the whole volume at once, doing slice by slice. Required device memory for GLM is %zu MB, global memory is %zu MB ! \n",totalRequiredMemory,globalMemorySize);
		}
	}
	else
	{
		if ((WRAPPER == BASH) && VERBOS)
		{
			printf("Sufficient memory for running the GLM the whole volume at once! Required device memory for GLM is %zu MB, global memory is %zu MB ! \n",totalRequiredMemory,globalMemorySize);
		}
	}

	c_X_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	c_Contrasts = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	c_ctxtxc_GLM = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	if (!largeMemory)
	{
		// Allocate memory for one slice for all time points, loop over slices to save memory
		d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
		d_Whitened_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
		d_Residuals = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
		allocatedDeviceMemory += 3 * EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float);
		deviceMemoryAllocations += 3;
	}
	else
	{
		d_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
		d_Whitened_fMRI_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), NULL, NULL);
		allocatedDeviceMemory += 2 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);
		deviceMemoryAllocations += 2;
	}

	d_Smoothed_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	deviceMemoryAllocations += 4;
	allocatedDeviceMemory += (EPI_DATA_W * EPI_DATA_H * EPI_DATA_D)*(1 + NUMBER_OF_TOTAL_GLM_REGRESSORS + 1 + 1) * sizeof(float);

	d_AR1_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR2_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR3_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	d_AR4_Estimates = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	deviceMemoryAllocations += 4;
	allocatedDeviceMemory += 4 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);

	PrintMemoryStatus("Before GLM");

	h_X_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
	h_xtxxt_GLM = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float));
	h_Contrasts = (float*)malloc(NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float));
	h_ctxtxc_GLM = (float*)malloc(NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float));
	h_X_GLM_With_Temporal_Derivatives = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * 2 * EPI_DATA_T * sizeof(float));
	h_X_GLM_Convolved = (float*)malloc(NUMBER_OF_GLM_REGRESSORS * (USE_TEMPORAL_DERIVATIVES+1) * EPI_DATA_T * sizeof(float));
	h_Global_Mean = (float*)malloc(EPI_DATA_T * sizeof(float));
	h_Motion_Parameters = (float*)malloc(EPI_DATA_T * NUMBER_OF_MOTION_REGRESSORS * sizeof(float));

	if (REGRESS_MOTION)
	{
		for (size_t i = 0; i < NUMBER_OF_MOTION_REGRESSORS * EPI_DATA_T; i++)
		{
			h_Motion_Parameters[i] = h_Motion_Parameters_Out[i];
		}
	}
	if (REGRESS_GLOBALMEAN)
	{
		CalculateGlobalMeans(h_fMRI_Volumes);
	}

	SetupFTestFirstLevel();

	// Copy mask to device
	clEnqueueWriteBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask , 0, NULL, NULL);

	// Copy model to device
	clEnqueueWriteBuffer(commandQueue, c_X_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_X_GLM , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), h_xtxxt_GLM , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_Contrasts, CL_TRUE, 0, NUMBER_OF_TOTAL_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float), h_Contrasts , 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, c_ctxtxc_GLM, CL_TRUE, 0, NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float), h_ctxtxc_GLM , 0, NULL, NULL);
	
	if (WRITE_DESIGNMATRIX)
	{
		for (int t = 0; t < EPI_DATA_T; t++)
		{
			for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
			{
				h_X_GLM_Out[t + r * EPI_DATA_T] = h_X_GLM[t + r * EPI_DATA_T];
				h_xtxxt_GLM_Out[t + r * EPI_DATA_T] = h_xtxxt_GLM[t + r * EPI_DATA_T];
			}
		}
	}

	// Run the actual GLM
	if (!largeMemory)
	{
		CalculateStatisticalMapsGLMFTestFirstLevelSlices(h_fMRI_Volumes,3);
	}
	else
	{
		CalculateStatisticalMapsGLMFTestFirstLevel(h_fMRI_Volumes,3);
	}

	clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), h_Beta_Volumes_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Statistical_Maps_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);
	
	if (WRITE_AR_ESTIMATES_EPI)
	{
		clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
		clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);
	}		

	// Cleanup host memory
	free(h_X_GLM);
	free(h_xtxxt_GLM);
	free(h_Contrasts);
	free(h_ctxtxc_GLM);
	free(h_X_GLM_With_Temporal_Derivatives);
	free(h_X_GLM_Convolved);
	free(h_Global_Mean);
	free(h_Motion_Parameters);

	// Cleanup device memory
	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);
	
	clReleaseMemObject(d_fMRI_Volumes);
	clReleaseMemObject(d_Whitened_fMRI_Volumes);

	if (!largeMemory)
	{
		clReleaseMemObject(d_Residuals);
		allocatedDeviceMemory -= 3 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_T * sizeof(float);
		deviceMemoryDeallocations += 3;
	}
	else
	{
		allocatedDeviceMemory -= 2 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);
		deviceMemoryDeallocations += 2;
	}

	clReleaseMemObject(d_EPI_Mask);
	clReleaseMemObject(d_Smoothed_EPI_Mask);
	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Residual_Variances);

	allocatedDeviceMemory -= (EPI_DATA_W * EPI_DATA_H * EPI_DATA_D)*(2 + NUMBER_OF_TOTAL_GLM_REGRESSORS + 1 + 1) * sizeof(float);
	deviceMemoryDeallocations += 5;

	clReleaseMemObject(d_AR1_Estimates);
	clReleaseMemObject(d_AR2_Estimates);
	clReleaseMemObject(d_AR3_Estimates);
	clReleaseMemObject(d_AR4_Estimates);

	allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * 4 * sizeof(float);
	deviceMemoryDeallocations += 4;

	PrintMemoryStatus("After GLM");
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
	//ApplyPermutationTestFirstLevel(d_fMRI_Volumes);
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
	//ApplyPermutationTestFirstLevel(d_fMRI_Volumes);
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

	if (!BETAS_ONLY && !CONTRASTS_ONLY && !BETAS_AND_CONTRASTS_ONLY)
	{
		clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);
	}
	if (WRITE_RESIDUAL_VARIANCES)
	{
		clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);
	}
	if (WRITE_RESIDUALS_MNI)
	{
		clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_Residuals_MNI, 0, NULL, NULL);
	}

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

	if (!BETAS_ONLY && !CONTRASTS_ONLY && !BETAS_AND_CONTRASTS_ONLY)
	{
		clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);
	}

	if (WRITE_RESIDUAL_VARIANCES)
	{
		clEnqueueReadBuffer(commandQueue, d_Residual_Variances, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Residual_Variances, 0, NULL, NULL);
	}

	if (WRITE_RESIDUALS_MNI)
	{
		clEnqueueReadBuffer(commandQueue, d_Residuals, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_Residuals_MNI, 0, NULL, NULL);
	}

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




void BROCCOLI_LIB::PerformSearchlightWrapper()
{
    // Allocate memory for volumes
    d_First_Level_Results = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
    d_MNI_Brain_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
    
    // Allocate memory for classes
    c_Correct_Classes = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
    c_d = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
    
    // Allocate memory for results
    d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
    
    // Copy data to device
    clEnqueueWriteBuffer(commandQueue, d_First_Level_Results, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_First_Level_Results , 0, NULL, NULL);
    clEnqueueWriteBuffer(commandQueue, d_MNI_Brain_Mask, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_MNI_Brain_Mask , 0, NULL, NULL);

    // Copy model to constant memory
    clEnqueueWriteBuffer(commandQueue, c_Correct_Classes, CL_TRUE, 0, NUMBER_OF_SUBJECTS * sizeof(float), h_Correct_Classes_In , 0, NULL, NULL);
    clEnqueueWriteBuffer(commandQueue, c_d, CL_TRUE, 0, NUMBER_OF_SUBJECTS * sizeof(float), h_d_In , 0, NULL, NULL);
    
    // Run searchlight
    SetGlobalAndLocalWorkSizesSearchlight(MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);
    
    float n = 0.001;
    int EPOCS = 1;
    
    clSetKernelArg(CalculateStatisticalMapSearchlightKernel, 0, sizeof(cl_mem),  &d_Statistical_Maps);
    clSetKernelArg(CalculateStatisticalMapSearchlightKernel, 1, sizeof(cl_mem),  &d_First_Level_Results);
    clSetKernelArg(CalculateStatisticalMapSearchlightKernel, 2, sizeof(cl_mem),  &d_MNI_Brain_Mask);
    clSetKernelArg(CalculateStatisticalMapSearchlightKernel, 3, sizeof(cl_mem),  &c_d);
    clSetKernelArg(CalculateStatisticalMapSearchlightKernel, 4, sizeof(cl_mem),  &c_Correct_Classes);
    clSetKernelArg(CalculateStatisticalMapSearchlightKernel, 5, sizeof(int),     &MNI_DATA_W);
    clSetKernelArg(CalculateStatisticalMapSearchlightKernel, 6, sizeof(int),     &MNI_DATA_H);
    clSetKernelArg(CalculateStatisticalMapSearchlightKernel, 7, sizeof(int),     &MNI_DATA_D);
    clSetKernelArg(CalculateStatisticalMapSearchlightKernel, 8, sizeof(int),     &NUMBER_OF_SUBJECTS);
    clSetKernelArg(CalculateStatisticalMapSearchlightKernel, 9, sizeof(float),   &n);
    clSetKernelArg(CalculateStatisticalMapSearchlightKernel, 10, sizeof(int),    &EPOCS);
    
    runKernelErrorCalculateStatisticalMapSearchlight = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapSearchlightKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapSearchlight, localWorkSizeCalculateStatisticalMapSearchlight, 0, NULL, NULL);
    clFinish(commandQueue);

    // Copy results to  host
    clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);
    clFinish(commandQueue);    

    // Release memory
    clReleaseMemObject(d_First_Level_Results);
    clReleaseMemObject(d_MNI_Brain_Mask);
    
    clReleaseMemObject(c_Correct_Classes);
    clReleaseMemObject(c_d);
    
    clReleaseMemObject(d_Statistical_Maps);
}


void BROCCOLI_LIB::PerformMeanSecondLevelPermutationWrapper()
{
	NUMBER_OF_TOTAL_GLM_REGRESSORS = 1;

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
	c_Sign_Vector = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);

	// Allocate memory for results
	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Residuals = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
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
	unsigned short int* temp = (unsigned short int*)malloc(NUMBER_OF_SUBJECTS * sizeof(unsigned short int));
	for (int i = 0; i < NUMBER_OF_SUBJECTS; i++)
	{
		temp[i] = (unsigned short int)i;
	}
	clEnqueueWriteBuffer(commandQueue, c_Permutation_Vector, CL_TRUE, 0, NUMBER_OF_SUBJECTS * sizeof(unsigned short int), temp , 0, NULL, NULL);
	free(temp);

	// Run the actual permutation test
	ApplyPermutationTestSecondLevel();

	CalculateStatisticalMapsGLMTTestSecondLevel(d_First_Level_Results, d_MNI_Brain_Mask);

	CalculatePermutationPValues(d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	// Copy results to  host
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_P_Values, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_P_Values_MNI, 0, NULL, NULL);

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
	clReleaseMemObject(d_P_Values);
}


void BROCCOLI_LIB::PerformGLMTTestSecondLevelPermutationWrapper()
{
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS;

	// Allocate memory for volumes
	d_First_Level_Results = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_Transformed_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
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
	c_Transformation_Matrix = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_SUBJECTS * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);

	// Allocate memory for results
	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	d_Residuals = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_P_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

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

	// Copy data to device again
	clEnqueueWriteBuffer(commandQueue, d_First_Level_Results, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_First_Level_Results , 0, NULL, NULL);

	CalculateStatisticalMapsGLMTTestSecondLevel(d_First_Level_Results, d_MNI_Brain_Mask);

	CalculatePermutationPValues(d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	// Copy results to  host
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_P_Values, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_P_Values_MNI, 0, NULL, NULL);

	// Release memory
	clReleaseMemObject(d_First_Level_Results);
	clReleaseMemObject(d_Transformed_Volumes);
	clReleaseMemObject(d_MNI_Brain_Mask);
	clReleaseMemObject(d_Cluster_Indices);
	clReleaseMemObject(d_Cluster_Sizes);
	clReleaseMemObject(d_TFCE_Values);

	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);
	clReleaseMemObject(c_Permutation_Vector);
	clReleaseMemObject(c_Transformation_Matrix);

	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);
	clReleaseMemObject(d_P_Values);
}


// Used for testing of F-test only
void BROCCOLI_LIB::PerformGLMFTestSecondLevelPermutationWrapper()
{
	NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS;

	// Allocate memory for volumes
	d_First_Level_Results = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_Transformed_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
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
	c_Transformation_Matrix = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_SUBJECTS * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);

	// Allocate memory for results
	d_Beta_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_TOTAL_GLM_REGRESSORS * sizeof(float), NULL, NULL);
	d_Statistical_Maps = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_Residuals = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), NULL, NULL);
	d_Residual_Variances = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);
	d_P_Values = clCreateBuffer(context, CL_MEM_READ_WRITE, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), NULL, NULL);

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

	// Copy data to device again
	clEnqueueWriteBuffer(commandQueue, d_First_Level_Results, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_First_Level_Results , 0, NULL, NULL);

	CalculateStatisticalMapsGLMFTestSecondLevel(d_First_Level_Results, d_MNI_Brain_Mask);

	CalculatePermutationPValues(d_MNI_Brain_Mask, MNI_DATA_W, MNI_DATA_H, MNI_DATA_D);

	// Copy results to  host
	clEnqueueReadBuffer(commandQueue, d_Statistical_Maps, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);
	//clEnqueueReadBuffer(commandQueue, d_Beta_Volumes, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_Statistical_Maps_MNI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_P_Values, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * sizeof(float), h_P_Values_MNI, 0, NULL, NULL);

	// Release memory
	clReleaseMemObject(d_First_Level_Results);
	clReleaseMemObject(d_Transformed_Volumes);
	clReleaseMemObject(d_MNI_Brain_Mask);
	clReleaseMemObject(d_Cluster_Indices);
	clReleaseMemObject(d_Cluster_Sizes);
	clReleaseMemObject(d_TFCE_Values);

	clReleaseMemObject(c_X_GLM);
	clReleaseMemObject(c_xtxxt_GLM);
	clReleaseMemObject(c_Contrasts);
	clReleaseMemObject(c_ctxtxc_GLM);
	clReleaseMemObject(c_Permutation_Vector);
	clReleaseMemObject(c_Transformation_Matrix);

	clReleaseMemObject(d_Beta_Volumes);
	clReleaseMemObject(d_Statistical_Maps);
	clReleaseMemObject(d_Residuals);
	clReleaseMemObject(d_Residual_Variances);
	clReleaseMemObject(d_P_Values);
}

void BROCCOLI_LIB::CalculateNumberOfBrainVoxels(cl_mem d_Mask, size_t DATA_W, size_t DATA_H, size_t DATA_D)
{
	float* h_Mask = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));

	clEnqueueReadBuffer(commandQueue, d_Mask, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Mask, 0, NULL, NULL);

	float voxel_number = 0.0f;
	for (size_t z = 0; z < DATA_D; z++)
	{
		for (size_t y = 0; y < DATA_H; y++)
		{
			for (size_t x = 0; x < DATA_W; x++)
			{
				if ( h_Mask[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
				{
					voxel_number += 1.0f;
				}
			}
		}
	}

	NUMBER_OF_BRAIN_VOXELS = (int)voxel_number;

	free(h_Mask);
}

// Generates a number (index) for each brain voxel, for storing design matrices for brain voxels only
void BROCCOLI_LIB::CreateVoxelNumbers(cl_mem d_Voxel_Numbers, cl_mem d_Mask, size_t DATA_W, size_t DATA_H, size_t DATA_D)
{
	float* h_Voxel_Numbers = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));
	float* h_Mask = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));

	clEnqueueReadBuffer(commandQueue, d_Mask, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Mask, 0, NULL, NULL);

	float voxel_number = 0.0f;
	for (size_t z = 0; z < DATA_D; z++)
	{
		for (size_t y = 0; y < DATA_H; y++)
		{
			for (size_t x = 0; x < DATA_W; x++)
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

	if ((WRAPPER == BASH) && VERBOS)
	{
		printf("\nThe number of brain voxels is %zu \n",NUMBER_OF_BRAIN_VOXELS);
	}

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_Voxel_Numbers, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Voxel_Numbers, 0, NULL, NULL);

	free(h_Voxel_Numbers);
	free(h_Mask);
}


// Generates a number (index) for each brain voxel, for storing design matrices for brain voxels only, for one slice
void BROCCOLI_LIB::CreateVoxelNumbersSlice(cl_mem d_Voxel_Numbers, cl_mem d_Mask, size_t slice, size_t DATA_W, size_t DATA_H, size_t DATA_D)
{
	float* h_Voxel_Numbers = (float*)malloc(DATA_W * DATA_H * sizeof(float));
	float* h_Mask = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));

	clEnqueueReadBuffer(commandQueue, d_Mask, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Mask, 0, NULL, NULL);

	float voxel_number = 0.0f;
	for (size_t y = 0; y < DATA_H; y++)
	{
		for (size_t x = 0; x < DATA_W; x++)
		{
			h_Voxel_Numbers[x + y * DATA_W] = 0.0f;
			if ( h_Mask[x + y * DATA_W + slice * DATA_W * DATA_H] == 1.0f )
			{
				h_Voxel_Numbers[x + y * DATA_W] = voxel_number;
				voxel_number += 1.0f;
			}
		}
	}

	NUMBER_OF_BRAIN_VOXELS = (int)voxel_number;

	if ((WRAPPER == BASH) && VERBOS)
	{
		printf("\nThe number of brain voxels is %zu for slice %zu \n",NUMBER_OF_BRAIN_VOXELS,slice);
	}

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_Voxel_Numbers, CL_TRUE, 0, DATA_W * DATA_H * sizeof(float), h_Voxel_Numbers, 0, NULL, NULL);

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
											   cl_mem d_Voxel_Numbers,
											   size_t DATA_W,
		                                       size_t DATA_H,
		                                       size_t DATA_D,
		                                       size_t DATA_T,
		                                       size_t NUMBER_OF_REGRESSORS,
		                                       size_t NUMBER_OF_INVALID_TIMEPOINTS)
{
	float* h_Voxel_Numbers = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));
	float* h_Mask = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));

	// Map buffer to host memory, to for example avoid double the memory when using the CPU as device
	//float* h_xtxxt_GLM_ = (float*)malloc(NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float));
	float* h_xtxxt_GLM_ = (float*) clEnqueueMapBuffer(commandQueue, d_xtxxt_GLM, CL_TRUE, CL_MAP_WRITE, 0, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float),0,NULL,NULL,NULL); 

	clEnqueueReadBuffer(commandQueue, d_Mask, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Mask, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Voxel_Numbers, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Voxel_Numbers, 0, NULL, NULL);

	// Copy AR parameters to host
	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);
	
	// Loop over voxels
	#pragma omp parallel for
	for (size_t z = 0; z < DATA_D; z++)
	{
		for (size_t y = 0; y < DATA_H; y++)
		{
			for (size_t x = 0; x < DATA_W; x++)
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

					/*
					int MEAN_REGRESSOR;

	                if (!RAW_DESIGNMATRIX)
	                {   
	                    MEAN_REGRESSOR = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1);
	                }
	                else
	                {
	                    MEAN_REGRESSOR = NUMBER_OF_GLM_REGRESSORS;
	                }

	                // Demean regressors
	                for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	                {
	                    if (r != MEAN_REGRESSOR)
	                    {
	                        Eigen::VectorXd regressor = X.block(NUMBER_OF_INVALID_TIMEPOINTS,r,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS,1);
	                        DemeanRegressor(regressor,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS);
	                        X.block(NUMBER_OF_INVALID_TIMEPOINTS,r,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS,1) = regressor;
	                    }
	                }
					*/	

					// Calculate pseudo inverse in an ugly way
					Eigen::MatrixXd xtx(NUMBER_OF_REGRESSORS,NUMBER_OF_REGRESSORS);
					xtx = X.transpose() * X;
					Eigen::MatrixXd inv_xtx = xtx.inverse();
					Eigen::MatrixXd xtxxt = inv_xtx * X.transpose();

					int voxel_number = h_Voxel_Numbers[x + y * DATA_W + z * DATA_W * DATA_H];

					// Put whitened regressors into specific format, to copy to GPU
					// (takes too much memory to store regressors for all voxels, so only store for brain voxels)
					for (size_t r = 0; r < NUMBER_OF_REGRESSORS; r++)
					{
						for (size_t t = 0; t < DATA_T; t++)
						{
							h_xtxxt_GLM_[voxel_number * NUMBER_OF_REGRESSORS * DATA_T + r * DATA_T + t] = xtxxt(r,t);
						}
					}
				}
			}
		}
	}

	// Copy data to device
	//clEnqueueWriteBuffer(commandQueue, d_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float), h_xtxxt_GLM_, 0, NULL, NULL);

	// Unmap buffer
	clEnqueueUnmapMemObject(commandQueue, d_xtxxt_GLM, h_xtxxt_GLM_, 0, NULL, NULL);

	free(h_Mask);
	//free(h_xtxxt_GLM_);
	free(h_Voxel_Numbers);
}


// Applies whitening to design matrix, different for each voxel, saves the pseudo inverse, for one slice
void BROCCOLI_LIB::WhitenDesignMatricesInverseSlice(cl_mem d_xtxxt_GLM,
		                                       float* h_X_GLM,
		                                       cl_mem d_AR1_Estimates,
		                                       cl_mem d_AR2_Estimates,
		                                       cl_mem d_AR3_Estimates,
		                                       cl_mem d_AR4_Estimates,
		                                       cl_mem d_Mask,
											   cl_mem d_Voxel_Numbers,
											   size_t slice,
		                                       size_t DATA_W,
		                                       size_t DATA_H,
		                                       size_t DATA_D,
		                                       size_t DATA_T,
		                                       size_t NUMBER_OF_REGRESSORS,
		                                       size_t NUMBER_OF_INVALID_TIMEPOINTS)
{
	float* h_Voxel_Numbers = (float*)malloc(DATA_W * DATA_H * sizeof(float));
	float* h_Mask = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));

	// Map buffer to host memory, to for example avoid double the memory when using the CPU as device
	//float* h_xtxxt_GLM_ = (float*)malloc(NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float));
	float* h_xtxxt_GLM_ = (float*) clEnqueueMapBuffer(commandQueue, d_xtxxt_GLM, CL_TRUE, CL_MAP_WRITE, 0, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float),0,NULL,NULL,NULL); 

	clEnqueueReadBuffer(commandQueue, d_Voxel_Numbers, CL_TRUE, 0, DATA_W * DATA_H * sizeof(float), h_Voxel_Numbers, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Mask, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Mask, 0, NULL, NULL);

	// Copy AR parameters to host
	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);

	// Loop over voxels
	#pragma omp parallel for
	for (size_t y = 0; y < DATA_H; y++)
	{
		for (size_t x = 0; x < DATA_W; x++)
		{
			if ( h_Mask[x + y * DATA_W + slice * DATA_W * DATA_H] == 1.0f )
			{
				Eigen::MatrixXd X(DATA_T,NUMBER_OF_REGRESSORS);

				// Get AR parameters for current voxel
				float AR1 = h_AR1_Estimates_EPI[x + y * DATA_W + slice * DATA_W * DATA_H];
				float AR2 = h_AR2_Estimates_EPI[x + y * DATA_W + slice * DATA_W * DATA_H];
				float AR3 = h_AR3_Estimates_EPI[x + y * DATA_W + slice * DATA_W * DATA_H];
				float AR4 = h_AR4_Estimates_EPI[x + y * DATA_W + slice * DATA_W * DATA_H];

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

				/*
				int MEAN_REGRESSOR;

                if (!RAW_DESIGNMATRIX)
                {   
                    MEAN_REGRESSOR = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1);
                }
                else
                {
                    MEAN_REGRESSOR = NUMBER_OF_GLM_REGRESSORS;
                }

                // Demean regressors
                for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
                {
                    if (r != MEAN_REGRESSOR)
                    {
                        Eigen::VectorXd regressor = X.block(NUMBER_OF_INVALID_TIMEPOINTS,r,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS,1);
                        DemeanRegressor(regressor,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS);
                        X.block(NUMBER_OF_INVALID_TIMEPOINTS,r,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS,1) = regressor;
                    }
                }
				*/

				// Calculate pseudo inverse in an ugly way
				Eigen::MatrixXd xtx(NUMBER_OF_REGRESSORS,NUMBER_OF_REGRESSORS);
				xtx = X.transpose() * X;
				Eigen::MatrixXd inv_xtx = xtx.inverse();
				Eigen::MatrixXd xtxxt = inv_xtx * X.transpose();

				int voxel_number = h_Voxel_Numbers[x + y * DATA_W];

				// Put whitened regressors into specific format, to copy to GPU
				// (takes too much memory to store regressors for all voxels, so only store for brain voxels)
				for (size_t r = 0; r < NUMBER_OF_REGRESSORS; r++)
				{
					for (size_t t = 0; t < DATA_T; t++)
					{
						h_xtxxt_GLM_[voxel_number * NUMBER_OF_REGRESSORS * DATA_T + r * DATA_T + t] = xtxxt(r,t);
					}
				}
			}
		}
	}

	// Copy data to device
	//clEnqueueWriteBuffer(commandQueue, d_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float), h_xtxxt_GLM_, 0, NULL, NULL);

	// Unmap buffer
	clEnqueueUnmapMemObject(commandQueue, d_xtxxt_GLM, h_xtxxt_GLM_, 0, NULL, NULL);

	free(h_Mask);
	//free(h_xtxxt_GLM_);
	free(h_Voxel_Numbers);
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
										 	 cl_mem d_Voxel_Numbers,											 
		                                	 size_t DATA_W,
		                                	 size_t DATA_H,
		                                	 size_t DATA_D,
		                                	 size_t DATA_T,
		                                	 size_t NUMBER_OF_REGRESSORS,
		                                	 size_t NUMBER_OF_INVALID_TIMEPOINTS,
		                                	 size_t NUMBER_OF_CONTRASTS)
{
	float* h_Mask = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));
	float* h_GLM_Scalars = (float*)malloc(DATA_W * DATA_H * DATA_D * NUMBER_OF_CONTRASTS * sizeof(float));
	float* h_Voxel_Numbers = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));

	clEnqueueReadBuffer(commandQueue, d_Mask, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Mask, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Voxel_Numbers, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Voxel_Numbers, 0, NULL, NULL);

	//float* h_X_GLM_ = (float*)malloc(NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float));
	float* h_X_GLM_ = (float*) clEnqueueMapBuffer(commandQueue, d_X_GLM, CL_TRUE, CL_MAP_WRITE, 0, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float),0,NULL,NULL,NULL); 

	// Copy AR parameters to host
	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);

	// Loop over voxels	
	#pragma omp parallel for
	for (size_t z = 0; z < DATA_D; z++)
	{
		for (size_t y = 0; y < DATA_H; y++)
		{
			for (size_t x = 0; x < DATA_W; x++)
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

					/*
					int MEAN_REGRESSOR;

	                if (!RAW_DESIGNMATRIX)
	                {   
	                    MEAN_REGRESSOR = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1);
	                }
	                else
	                {
	                    MEAN_REGRESSOR = NUMBER_OF_GLM_REGRESSORS;
	                }

	                // Demean regressors
	                for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	                {
	                    if (r != MEAN_REGRESSOR)
	                    {
	                        Eigen::VectorXd regressor = X.block(NUMBER_OF_INVALID_TIMEPOINTS,r,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS,1);
	                        DemeanRegressor(regressor,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS);
	                        X.block(NUMBER_OF_INVALID_TIMEPOINTS,r,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS,1) = regressor;
	                    }
	                }
					*/

					// Calculate contrast scalars
					Eigen::MatrixXd xtx(NUMBER_OF_REGRESSORS,NUMBER_OF_REGRESSORS);
					xtx = X.transpose() * X;
					Eigen::MatrixXd inv_xtx = xtx.inverse();

					for (size_t c = 0; c < NUMBER_OF_CONTRASTS; c++)
					{
						Eigen::MatrixXd Contrast(NUMBER_OF_REGRESSORS,1);
	
						for (size_t r = 0; r < NUMBER_OF_REGRESSORS; r++)
						{
							Contrast(r) = (double)h_Contrasts[NUMBER_OF_REGRESSORS * c + r];
						}
	
						Eigen::MatrixXd GLM_scalar = Contrast.transpose() * inv_xtx * Contrast;
						h_GLM_Scalars[x + y * DATA_W + z * DATA_W * DATA_H + c * DATA_W * DATA_H * DATA_D] = GLM_scalar(0);
					}

					int voxel_number = h_Voxel_Numbers[x + y * DATA_W + z * DATA_W * DATA_H];

					// Put whitened regressors into specific format, to copy to GPU
					// (takes too much memory to store regressors for all voxels, so only store for brain voxels)
					for (size_t r = 0; r < NUMBER_OF_REGRESSORS; r++)
					{
						for (size_t t = 0; t < DATA_T; t++)
						{
							h_X_GLM_[voxel_number * NUMBER_OF_REGRESSORS * DATA_T + r * DATA_T + t] = X(t,r);
						}
					}
				}
			}
		}
	}

	// Copy data to device
	//clEnqueueWriteBuffer(commandQueue, d_X_GLM, CL_TRUE, 0, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float), h_X_GLM_, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_GLM_Scalars, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), h_GLM_Scalars, 0, NULL, NULL);

	free(h_Mask);
	//free(h_X_GLM_);
	free(h_GLM_Scalars);
	free(h_Voxel_Numbers);

	// Unmap buffer
	clEnqueueUnmapMemObject(commandQueue, d_X_GLM, h_X_GLM_, 0, NULL, NULL);
}




// Applies whitening to design matrix, different for each voxel, saves the whitened matrix, for one slice
void BROCCOLI_LIB::WhitenDesignMatricesTTestSlice(cl_mem d_X_GLM,
		                                	 cl_mem d_GLM_Scalars,
		                                	 float* h_X_GLM,
		                                	 float* h_Contrasts,
		                                	 cl_mem d_AR1_Estimates,
		                                	 cl_mem d_AR2_Estimates,
		                                	 cl_mem d_AR3_Estimates,
		                                	 cl_mem d_AR4_Estimates,
		                                	 cl_mem d_Mask,
										     cl_mem d_Voxel_Numbers,
											 size_t slice,
		                                	 size_t DATA_W,
		                                	 size_t DATA_H,
		                                	 size_t DATA_D,
		                                	 size_t DATA_T,
		                                	 size_t NUMBER_OF_REGRESSORS,
		                                	 size_t NUMBER_OF_INVALID_TIMEPOINTS,
		                                	 size_t NUMBER_OF_CONTRASTS)
{
	float* h_Mask = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));
	float* h_GLM_Scalars = (float*)malloc(DATA_W * DATA_H * NUMBER_OF_CONTRASTS * sizeof(float));
	float* h_Voxel_Numbers = (float*)malloc(DATA_W * DATA_H * sizeof(float));

	//float* h_X_GLM_ = (float*)malloc(NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float));
	float* h_X_GLM_ = (float*) clEnqueueMapBuffer(commandQueue, d_X_GLM, CL_TRUE, CL_MAP_WRITE, 0, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float),0,NULL,NULL,NULL); 

	clEnqueueReadBuffer(commandQueue, d_Mask, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Mask, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Voxel_Numbers, CL_TRUE, 0, DATA_W * DATA_H * sizeof(float), h_Voxel_Numbers, 0, NULL, NULL);

	// Copy AR parameters to host
	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);

	// Loop over voxels
	#pragma omp parallel for
	for (size_t y = 0; y < DATA_H; y++)
	{
		for (size_t x = 0; x < DATA_W; x++)
		{
			if ( h_Mask[x + y * DATA_W + slice * DATA_W * DATA_H] == 1.0f )
			{
				Eigen::MatrixXd X(DATA_T,NUMBER_OF_REGRESSORS);

				// Get AR parameters for current voxel
				float AR1 = h_AR1_Estimates_EPI[x + y * DATA_W + slice * DATA_W * DATA_H];
				float AR2 = h_AR2_Estimates_EPI[x + y * DATA_W + slice * DATA_W * DATA_H];
				float AR3 = h_AR3_Estimates_EPI[x + y * DATA_W + slice * DATA_W * DATA_H];
				float AR4 = h_AR4_Estimates_EPI[x + y * DATA_W + slice * DATA_W * DATA_H];

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

				/*
				int MEAN_REGRESSOR;

                if (!RAW_DESIGNMATRIX)
                {   
                    MEAN_REGRESSOR = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1);
                }
                else
                {
                    MEAN_REGRESSOR = NUMBER_OF_GLM_REGRESSORS;
                }

                // Demean regressors
                for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
                {
                    if (r != MEAN_REGRESSOR)
                    {
                        Eigen::VectorXd regressor = X.block(NUMBER_OF_INVALID_TIMEPOINTS,r,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS,1);
                        DemeanRegressor(regressor,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS);
                        X.block(NUMBER_OF_INVALID_TIMEPOINTS,r,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS,1) = regressor;
                    }
                }
				*/

				// Calculate contrast scalars
				Eigen::MatrixXd xtx(NUMBER_OF_REGRESSORS,NUMBER_OF_REGRESSORS);
				xtx = X.transpose() * X;
				Eigen::MatrixXd inv_xtx = xtx.inverse();

				for (size_t c = 0; c < NUMBER_OF_CONTRASTS; c++)
				{
					Eigen::MatrixXd Contrast(NUMBER_OF_REGRESSORS,1);

					for (size_t r = 0; r < NUMBER_OF_REGRESSORS; r++)
					{
						Contrast(r) = (double)h_Contrasts[NUMBER_OF_REGRESSORS * c + r];
					}

					Eigen::MatrixXd GLM_scalar = Contrast.transpose() * inv_xtx * Contrast;
					h_GLM_Scalars[x + y * DATA_W + c * DATA_W * DATA_H] = GLM_scalar(0);
				}

				int voxel_number = h_Voxel_Numbers[x + y * DATA_W];

				// Put whitened regressors into specific format, to copy to GPU
				// (takes too much memory to store regressors for all voxels, so only store for brain voxels)
				for (size_t r = 0; r < NUMBER_OF_REGRESSORS; r++)
				{
					for (size_t t = 0; t < DATA_T; t++)
					{
						h_X_GLM_[voxel_number * NUMBER_OF_REGRESSORS * DATA_T + r * DATA_T + t] = X(t,r);
					}
				}
			}
		}
	}

	// Copy data to device
	//clEnqueueWriteBuffer(commandQueue, d_X_GLM, CL_TRUE, 0, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float), h_X_GLM_, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_GLM_Scalars, CL_TRUE, 0, DATA_W * DATA_H * NUMBER_OF_CONTRASTS * sizeof(float), h_GLM_Scalars, 0, NULL, NULL);

	// Unmap buffer
	clEnqueueUnmapMemObject(commandQueue, d_X_GLM, h_X_GLM_, 0, NULL, NULL);

	free(h_Mask);
	//free(h_X_GLM_);
	free(h_GLM_Scalars);
	free(h_Voxel_Numbers);
}


void BROCCOLI_LIB::WhitenDesignMatricesFTestSlice(cl_mem d_X_GLM,
		                                	 cl_mem d_GLM_Scalars,
		                                	 float* h_X_GLM,
		                                	 float* h_Contrasts,
		                                	 cl_mem d_AR1_Estimates,
		                                	 cl_mem d_AR2_Estimates,
		                                	 cl_mem d_AR3_Estimates,
		                                	 cl_mem d_AR4_Estimates,
		                                	 cl_mem d_Mask,
											 cl_mem d_Voxel_Numbers,
		                                	 size_t slice,
		                                	 size_t DATA_W,
		                                	 size_t DATA_H,
		                                	 size_t DATA_D,
		                                	 size_t DATA_T,
		                                	 size_t NUMBER_OF_REGRESSORS,
		                                	 size_t NUMBER_OF_INVALID_TIMEPOINTS,
		                                	 size_t NUMBER_OF_CONTRASTS)
{
	float* h_Mask = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));
	float* h_GLM_Scalars = (float*)malloc(DATA_W * DATA_H * NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float));
	float* h_Voxel_Numbers = (float*)malloc(DATA_W * DATA_H * sizeof(float));

	//float* h_X_GLM_ = (float*)malloc(NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float));
	float* h_X_GLM_ = (float*) clEnqueueMapBuffer(commandQueue, d_X_GLM, CL_TRUE, CL_MAP_WRITE, 0, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float),0,NULL,NULL,NULL); 

	clEnqueueReadBuffer(commandQueue, d_Mask, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Mask, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_Voxel_Numbers, CL_TRUE, 0, DATA_W * DATA_H * sizeof(float), h_Voxel_Numbers, 0, NULL, NULL);

	// Copy AR parameters to host
	clEnqueueReadBuffer(commandQueue, d_AR1_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR1_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR2_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR2_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR3_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR3_Estimates_EPI, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_AR4_Estimates, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_AR4_Estimates_EPI, 0, NULL, NULL);

	// Loop over voxels
	#pragma omp parallel for
	for (size_t y = 0; y < DATA_H; y++)
	{
		for (size_t x = 0; x < DATA_W; x++)
		{
			if ( h_Mask[x + y * DATA_W + slice * DATA_W * DATA_H] == 1.0f )
			{
				// Insert contrast into eigen variable
				Eigen::MatrixXd Contrasts(NUMBER_OF_CONTRASTS,NUMBER_OF_REGRESSORS);

				for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
				{
					for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
					{
						Contrasts(c,r) = (double)h_Contrasts[NUMBER_OF_REGRESSORS * c + r];
					}
				}

				Eigen::MatrixXd X(DATA_T,NUMBER_OF_REGRESSORS);

				// Get AR parameters for current voxel
				float AR1 = h_AR1_Estimates_EPI[x + y * DATA_W + slice * DATA_W * DATA_H];
				float AR2 = h_AR2_Estimates_EPI[x + y * DATA_W + slice * DATA_W * DATA_H];
				float AR3 = h_AR3_Estimates_EPI[x + y * DATA_W + slice * DATA_W * DATA_H];
				float AR4 = h_AR4_Estimates_EPI[x + y * DATA_W + slice * DATA_W * DATA_H];

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

				/*
				int MEAN_REGRESSOR;

                if (!RAW_DESIGNMATRIX)
                {   
                    MEAN_REGRESSOR = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1);
                }
                else
                {
                    MEAN_REGRESSOR = NUMBER_OF_GLM_REGRESSORS;
                }

                // Demean regressors
                for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
                {
                    if (r != MEAN_REGRESSOR)
                    {
                        Eigen::VectorXd regressor = X.block(NUMBER_OF_INVALID_TIMEPOINTS,r,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS,1);
                        DemeanRegressor(regressor,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS);
                        X.block(NUMBER_OF_INVALID_TIMEPOINTS,r,DATA_T-NUMBER_OF_INVALID_TIMEPOINTS,1) = regressor;
                    }
                }
				*/

				// Calculate contrast values

				Eigen::MatrixXd xtx(NUMBER_OF_REGRESSORS,NUMBER_OF_REGRESSORS);
				xtx = X.transpose() * X;
				Eigen::MatrixXd inv_xtx = xtx.inverse();

				Eigen::MatrixXd temp = Contrasts * inv_xtx * Contrasts.transpose();
				Eigen::MatrixXd ctxtxc = temp.inverse();

				for (size_t c = 0; c < NUMBER_OF_CONTRASTS; c++)
				{
					for (size_t cc = 0; cc < NUMBER_OF_CONTRASTS; cc++)
					{
						h_GLM_Scalars[x + y * DATA_W + (cc + c * NUMBER_OF_CONTRASTS) * DATA_W * DATA_H] = (float)ctxtxc(c,cc);
					}
				}

				int voxel_number = h_Voxel_Numbers[x + y * DATA_W];

				// Put whitened regressors into specific format, to copy to GPU
				// (takes too much memory to store regressors for all voxels, so only store for brain voxels)
				for (size_t r = 0; r < NUMBER_OF_REGRESSORS; r++)
				{
					for (size_t t = 0; t < DATA_T; t++)
					{
						h_X_GLM_[voxel_number * NUMBER_OF_REGRESSORS * DATA_T + r * DATA_T + t] = X(t,r);
					}
				}
			}
		}
	}

	// Copy data to device
	//clEnqueueWriteBuffer(commandQueue, d_X_GLM, CL_TRUE, 0, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float), h_X_GLM_, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_GLM_Scalars, CL_TRUE, 0, DATA_W * DATA_H * NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float), h_GLM_Scalars, 0, NULL, NULL);

	// Unmap buffer
	clEnqueueUnmapMemObject(commandQueue, d_X_GLM, h_X_GLM_, 0, NULL, NULL);

	free(h_Mask);
	//free(h_X_GLM_);
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
											 cl_mem d_Voxel_Numbers,
		                                	 size_t DATA_W,
		                                	 size_t DATA_H,
		                                	 size_t DATA_D,
		                                	 size_t DATA_T,
		                                	 size_t NUMBER_OF_REGRESSORS,
		                                	 size_t NUMBER_OF_INVALID_TIMEPOINTS,
		                                	 size_t NUMBER_OF_CONTRASTS)
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
	for (size_t z = 0; z < DATA_D; z++)
	{
		for (size_t y = 0; y < DATA_H; y++)
		{
			for (size_t x = 0; x < DATA_W; x++)
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

					for (size_t c = 0; c < NUMBER_OF_CONTRASTS; c++)
					{
						for (size_t cc = 0; cc < NUMBER_OF_CONTRASTS; cc++)
						{
							h_GLM_Scalars[x + y * DATA_W + z * DATA_W * DATA_H + (cc + c * NUMBER_OF_CONTRASTS) * DATA_W * DATA_H * DATA_D] = (float)ctxtxc(c,cc);
						}
					}

					// Put whitened regressors into specific format, to copy to GPU
					// (takes too much memory to store regressors for all voxels, so only store for brain voxels)
					for (size_t r = 0; r < NUMBER_OF_REGRESSORS; r++)
					{
						for (size_t t = 0; t < DATA_T; t++)
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




// Changes the storage order of a 4D dataset, from x, y, z, t to x, y, t, z
void BROCCOLI_LIB::FlipVolumesXYZTtoXYTZ(float* h_Volumes, size_t DATA_W, size_t DATA_H, size_t DATA_D, size_t DATA_T)
{
	// Allocate temporary space
	float* h_Temp_Volumes = (float*)malloc(DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float));
	
	size_t i = 0;
    for (size_t t = 0; t < DATA_T ; t++)
    {
	    for (size_t z = 0; z < DATA_D ; z++)
	    {
            for (size_t y = 0; y < DATA_H ; y++)
            {
		        for (size_t x = 0; x < DATA_W ; x++)
		        {
	                h_Temp_Volumes[x + y * DATA_W + t * DATA_W * DATA_H + z * DATA_W * DATA_H * DATA_T] = h_Volumes[i];
	                i++;
	            }
	        }
	    }
	}

	// Copy back to original pointer
	memcpy(h_Volumes, h_Temp_Volumes, DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float));	
	free(h_Temp_Volumes);
}

// Changes the storage order of a 4D dataset, from x, y, t, z  to  x, y, z, t
void BROCCOLI_LIB::FlipVolumesXYTZtoXYZT(float* h_Volumes, size_t DATA_W, size_t DATA_H, size_t DATA_D, size_t DATA_T)
{
	// Allocate temporary space
	float* h_Temp_Volumes = (float*)malloc(DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float));
	
	size_t i = 0;
    for (size_t z = 0; z < DATA_D ; z++)
    {
	    for (size_t t = 0; t < DATA_T ; t++)
	    {
            for (size_t y = 0; y < DATA_H ; y++)
            {
		        for (size_t x = 0; x < DATA_W ; x++)
		        {
	                h_Temp_Volumes[x + y * DATA_W + z * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D] = h_Volumes[i];
	                i++;
	            }
	        }
	    }
	}

	// Copy back to original pointer
	memcpy(h_Volumes, h_Temp_Volumes, DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float));	
	free(h_Temp_Volumes);
}


void BROCCOLI_LIB::CopyCurrentfMRISliceToDevice(cl_mem d_Volumes, float* h_Volumes, size_t slice, size_t DATA_W, size_t DATA_H, size_t DATA_D, size_t DATA_T)
{
	// Allocate temporary space, for storing slice as x, y, t
	float* h_Temp_Data = (float*)malloc(DATA_W * DATA_H * DATA_T * sizeof(float));

	// Copy data to temporary space
    for (size_t t = 0; t < DATA_T ; t++)
    {
         for (size_t y = 0; y < DATA_H ; y++)
         {
	        for (size_t x = 0; x < DATA_W ; x++)
	        {
                h_Temp_Data[x + y * DATA_W + t * DATA_W * DATA_H] = h_Volumes[x + y * DATA_W + slice * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D];
            }
        }
    }

	// Copy the current slice for all time points
	clEnqueueWriteBuffer(commandQueue, d_Volumes, CL_TRUE, 0, DATA_W * DATA_H * DATA_T * sizeof(float), h_Temp_Data, 0, NULL, NULL);

	free(h_Temp_Data);
}

void BROCCOLI_LIB::CopyCurrentfMRISliceToHost(float* h_Volumes, cl_mem d_Volumes, size_t slice, size_t DATA_W, size_t DATA_H, size_t DATA_D, size_t DATA_T)
{
	// Allocate temporary space, for storing slice as x, y, t
	float* h_Temp_Data = (float*)malloc(DATA_W * DATA_H * DATA_T * sizeof(float));

	// Copy the current slice for all time points
	clEnqueueReadBuffer(commandQueue, d_Volumes, CL_TRUE, 0, DATA_W * DATA_H * DATA_T * sizeof(float), h_Temp_Data, 0, NULL, NULL);

	// Copy data to correct location in 4D array
    for (size_t t = 0; t < DATA_T ; t++)
    {
         for (size_t y = 0; y < DATA_H ; y++)
         {
	        for (size_t x = 0; x < DATA_W ; x++)
	        {				
                h_Volumes[x + y * DATA_W + slice * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D] = h_Temp_Data[x + y * DATA_W + t * DATA_W * DATA_H];
            }
        }
    }

	free(h_Temp_Data);
}

void BROCCOLI_LIB::CalculateBetaWeightsAndContrastsFirstLevel(float* h_Volumes)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// All timepoints are valid
	NUMBER_OF_INVALID_TIMEPOINTS = 0;
	c_Censored_Timepoints = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(float), NULL, NULL);
	SetMemory(c_Censored_Timepoints, 1.0f, EPI_DATA_T);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Volumes, 0, NULL, NULL);

	// Calculate beta values, using whitened data and the whitened voxel-specific models
	clSetKernelArg(CalculateBetaWeightsAndContrastsGLMKernel, 0,  sizeof(cl_mem), &d_Beta_Volumes);
	clSetKernelArg(CalculateBetaWeightsAndContrastsGLMKernel, 1,  sizeof(cl_mem), &d_Contrast_Volumes);
	clSetKernelArg(CalculateBetaWeightsAndContrastsGLMKernel, 2,  sizeof(cl_mem), &d_fMRI_Volumes);
	clSetKernelArg(CalculateBetaWeightsAndContrastsGLMKernel, 3,  sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateBetaWeightsAndContrastsGLMKernel, 4,  sizeof(cl_mem), &c_xtxxt_GLM);
	clSetKernelArg(CalculateBetaWeightsAndContrastsGLMKernel, 5,  sizeof(cl_mem), &c_Contrasts);
	clSetKernelArg(CalculateBetaWeightsAndContrastsGLMKernel, 6,  sizeof(cl_mem), &c_Censored_Timepoints);
	clSetKernelArg(CalculateBetaWeightsAndContrastsGLMKernel, 7,  sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(CalculateBetaWeightsAndContrastsGLMKernel, 8,  sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(CalculateBetaWeightsAndContrastsGLMKernel, 9,  sizeof(int),    &EPI_DATA_D);
	clSetKernelArg(CalculateBetaWeightsAndContrastsGLMKernel, 10, sizeof(int),    &EPI_DATA_T);
	clSetKernelArg(CalculateBetaWeightsAndContrastsGLMKernel, 11, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
	clSetKernelArg(CalculateBetaWeightsAndContrastsGLMKernel, 12, sizeof(int),    &NUMBER_OF_CONTRASTS);
	runKernelErrorCalculateBetaWeightsAndContrastsGLM = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsAndContrastsGLMKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
	clFinish(commandQueue);

	clReleaseMemObject(c_Censored_Timepoints);
}


// Calculates beta values for first level analysis
// Loops over slices to save memory

void BROCCOLI_LIB::CalculateBetaWeightsAndContrastsFirstLevelSlices(float* h_Volumes)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, 1);

	// All timepoints are valid
	NUMBER_OF_INVALID_TIMEPOINTS = 0;
	c_Censored_Timepoints = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_T * sizeof(float), NULL, NULL);
	SetMemory(c_Censored_Timepoints, 1.0f, EPI_DATA_T);

	for (size_t slice = 0; slice < EPI_DATA_D; slice++)
	{
		// Copy fMRI data to the device, for the current slice
		CopyCurrentfMRISliceToDevice(d_fMRI_Volumes, h_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

		// Calculate beta values, using whitened data and the whitened voxel-specific models
		clSetKernelArg(CalculateBetaWeightsAndContrastsGLMSliceKernel, 0,  sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateBetaWeightsAndContrastsGLMSliceKernel, 1,  sizeof(cl_mem), &d_Contrast_Volumes);
		clSetKernelArg(CalculateBetaWeightsAndContrastsGLMSliceKernel, 2,  sizeof(cl_mem), &d_fMRI_Volumes);
		clSetKernelArg(CalculateBetaWeightsAndContrastsGLMSliceKernel, 3,  sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateBetaWeightsAndContrastsGLMSliceKernel, 4,  sizeof(cl_mem), &c_xtxxt_GLM);
		clSetKernelArg(CalculateBetaWeightsAndContrastsGLMSliceKernel, 5,  sizeof(cl_mem), &c_Contrasts);
		clSetKernelArg(CalculateBetaWeightsAndContrastsGLMSliceKernel, 6,  sizeof(cl_mem), &c_Censored_Timepoints);
		clSetKernelArg(CalculateBetaWeightsAndContrastsGLMSliceKernel, 7,  sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(CalculateBetaWeightsAndContrastsGLMSliceKernel, 8,  sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(CalculateBetaWeightsAndContrastsGLMSliceKernel, 9,  sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(CalculateBetaWeightsAndContrastsGLMSliceKernel, 10, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(CalculateBetaWeightsAndContrastsGLMSliceKernel, 11, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateBetaWeightsAndContrastsGLMSliceKernel, 12, sizeof(int),    &NUMBER_OF_CONTRASTS);
		clSetKernelArg(CalculateBetaWeightsAndContrastsGLMSliceKernel, 13, sizeof(int),    &slice);
		runKernelErrorCalculateBetaWeightsAndContrastsGLMSlice = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsAndContrastsGLMSliceKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	clReleaseMemObject(c_Censored_Timepoints);
}



// Calculates a statistical map for first level analysis, using a Cochrane-Orcutt procedure

cl_int BROCCOLI_LIB::CalculateStatisticalMapsGLMTTestFirstLevel(float *h_Volumes, int iterations)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Volumes, 0, NULL, NULL);

	// Create a mapping between voxel coordinates and brain voxel number, since we cannot store the modified GLM design matrix for all voxels, only for the brain voxels
	cl_mem d_Voxel_Numbers = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);
	CreateVoxelNumbers(d_Voxel_Numbers, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Allocate memory for voxel specific design matrices (sufficient to store the pseudo inverses, since we only need to estimate beta weights with the voxel-specific models, not the residuals)
	cl_int memError1 = 0;
	cl_mem d_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);
	allocatedDeviceMemory += NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float);

	// Allocate memory for voxel specific GLM scalars
	cl_mem d_GLM_Scalars = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);
	allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float);
	PrintMemoryStatus("Inside GLM");

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
	WhitenDesignMatricesInverse(d_xtxxt_GLM, h_X_GLM, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, d_Voxel_Numbers, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS);

	// Set whitened volumes to original volumes
	clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Whitened_fMRI_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), 0, NULL, NULL);

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
		//clSetKernelArg(CalculateGLMResidualsKernel, 0, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(CalculateGLMResidualsKernel, 0, sizeof(cl_mem), &d_Whitened_fMRI_Volumes); // Save residuals in whitened fMRI volumes, not needed here
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
		//clSetKernelArg(EstimateAR4ModelsKernel, 4, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(EstimateAR4ModelsKernel, 4, sizeof(cl_mem), &d_Whitened_fMRI_Volumes); // Residuals being stored in whitened volumes
		clSetKernelArg(EstimateAR4ModelsKernel, 5, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(EstimateAR4ModelsKernel, 6, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(EstimateAR4ModelsKernel, 7, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(EstimateAR4ModelsKernel, 8, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(EstimateAR4ModelsKernel, 9, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(EstimateAR4ModelsKernel, 10, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorEstimateAR4Models = clEnqueueNDRangeKernel(commandQueue, EstimateAR4ModelsKernel, 3, NULL, globalWorkSizeEstimateAR4Models, localWorkSizeEstimateAR4Models, 0, NULL, NULL);

		// Smooth auto correlation estimates
		//PerformSmoothingNormalized(d_AR1_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		//PerformSmoothingNormalized(d_AR2_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		//PerformSmoothingNormalized(d_AR3_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		//PerformSmoothingNormalized(d_AR4_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

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
		WhitenDesignMatricesInverse(d_xtxxt_GLM, h_X_GLM, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, d_Voxel_Numbers, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS);
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
	WhitenDesignMatricesTTest(d_xtxxt_GLM, d_GLM_Scalars, h_X_GLM, h_Contrasts, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, d_Voxel_Numbers, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS, NUMBER_OF_CONTRASTS);

	// Finally calculate statistical maps using whitened model and whitened data
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 0,  sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 1,  sizeof(cl_mem), &d_Contrast_Volumes);
	//clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 2,  sizeof(cl_mem), &d_Residuals);
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelKernel, 2,  sizeof(cl_mem), &d_fMRI_Volumes); // Store residuals in original fMRI volumes
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

	if (WRITE_RESIDUALS_EPI)
	{
		clEnqueueReadBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Residuals_EPI, 0, NULL, NULL);
	}

	MultiplyVolumes(d_AR1_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR2_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR3_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	clReleaseMemObject(d_xtxxt_GLM);
	clReleaseMemObject(d_GLM_Scalars);
	clReleaseMemObject(d_Voxel_Numbers);
	clReleaseMemObject(c_Censored_Timepoints);

	allocatedDeviceMemory -= NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float);
	allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float);
	allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);

	if (runKernelErrorCalculateBetaWeightsGLMFirstLevel != CL_SUCCESS) 
	{
		return runKernelErrorCalculateBetaWeightsGLMFirstLevel;
	}
	else if (runKernelErrorCalculateGLMResiduals != CL_SUCCESS)
	{
		return runKernelErrorCalculateGLMResiduals;
	}
	else if (runKernelErrorEstimateAR4Models != CL_SUCCESS)
	{
		return runKernelErrorEstimateAR4Models;
	}
	else if (runKernelErrorApplyWhiteningAR4 != CL_SUCCESS)
	{
		return runKernelErrorApplyWhiteningAR4;
	}
	else if (runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel != CL_SUCCESS)
	{
		return runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel;
	}
	else
	{
		return CL_SUCCESS;
	}
}



// Calculates a statistical map for first level analysis, using a Cochrane-Orcutt procedure
// Loops over slices to save memory

void BROCCOLI_LIB::CalculateStatisticalMapsGLMTTestFirstLevelSlices(float* h_Volumes, int iterations)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, 1);

	// Allocate memory for voxel numbers
	cl_mem d_Voxel_Numbers = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * sizeof(float), NULL, NULL);

	// Allocate memory for voxel specific GLM scalars
	cl_mem d_GLM_Scalars = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * sizeof(float);
	allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * NUMBER_OF_CONTRASTS * sizeof(float);

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
	
	// Flip the fMRI data from x,y,z,t to x,y,t,z, to be able to copy all time points for one slice
	//FlipVolumesXYZTtoXYTZ(h_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	h_Whitened_fMRI_Volumes = (float*)malloc(EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float));
	memcpy(h_Whitened_fMRI_Volumes, h_Volumes, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float));	
	allocatedHostMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);

	int one = 1;

	// Cochrane-Orcutt procedure, iterate
	for (int it = 0; it < iterations; it++)
	{
		for (size_t slice = 0; slice < EPI_DATA_D; slice++)
		{
			// Create a mapping between voxel coordinates and brain voxel number, since we cannot store the modified GLM design matrix for all voxels, only for the brain voxels
			CreateVoxelNumbersSlice(d_Voxel_Numbers, d_EPI_Mask, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

			// Allocate memory for voxel specific design matrices (sufficient to store the pseudo inverses, since we only need to estimate beta weights with the voxel-specific models, not the residuals)
			// Only store for brain voxels, which differs for each slice
			cl_mem d_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);

			allocatedDeviceMemory += NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float);
			PrintMemoryStatus("Inside GLM");
			
			// Apply whitening to model and create voxel-specific models
			WhitenDesignMatricesInverseSlice(d_xtxxt_GLM, h_X_GLM, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, d_Voxel_Numbers, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS);

			// Copy fMRI data to the device, for the current slice
			CopyCurrentfMRISliceToDevice(d_Whitened_fMRI_Volumes, h_Whitened_fMRI_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

			// Calculate beta values, using whitened data and the whitened voxel-specific models
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 0,  sizeof(cl_mem), &d_Beta_Volumes);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 1,  sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 2,  sizeof(cl_mem), &d_EPI_Mask);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 3,  sizeof(cl_mem), &d_xtxxt_GLM);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 4,  sizeof(cl_mem), &d_Voxel_Numbers);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 5,  sizeof(cl_mem), &c_Censored_Timepoints);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 6,  sizeof(int),    &EPI_DATA_W);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 7,  sizeof(int),    &EPI_DATA_H);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 8,  sizeof(int),    &EPI_DATA_D);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 9,  sizeof(int),    &EPI_DATA_T);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 10, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 11, sizeof(int),    &NUMBER_OF_INVALID_TIMEPOINTS);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 12, sizeof(int),    &slice);
			runKernelErrorCalculateBetaWeightsGLMFirstLevelSlice = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMFirstLevelSliceKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
			clFinish(commandQueue);

			// Copy fMRI data to the device, for the current slice
			CopyCurrentfMRISliceToDevice(d_fMRI_Volumes, h_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

			// Calculate residuals, using original data and the original model
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 0, sizeof(cl_mem), &d_Residuals);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 1, sizeof(cl_mem), &d_fMRI_Volumes);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 2, sizeof(cl_mem), &d_Beta_Volumes);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 4, sizeof(cl_mem), &c_X_GLM);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 5, sizeof(int),    &EPI_DATA_W);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 6, sizeof(int),    &EPI_DATA_H);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 7, sizeof(int),    &EPI_DATA_D);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 8, sizeof(int),    &EPI_DATA_T);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 9, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 10, sizeof(int),   &slice);
			runKernelErrorCalculateGLMResidualsSlice = clEnqueueNDRangeKernel(commandQueue, CalculateGLMResidualsSliceKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
			clFinish(commandQueue);

			// Estimate auto correlation from residuals
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 0, sizeof(cl_mem), &d_AR1_Estimates);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 1, sizeof(cl_mem), &d_AR2_Estimates);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 2, sizeof(cl_mem), &d_AR3_Estimates);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 3, sizeof(cl_mem), &d_AR4_Estimates);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 4, sizeof(cl_mem), &d_Residuals);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 5, sizeof(cl_mem), &d_EPI_Mask);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 6, sizeof(int),    &EPI_DATA_W);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 7, sizeof(int),    &EPI_DATA_H);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 8, sizeof(int),    &EPI_DATA_D);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 9, sizeof(int),    &EPI_DATA_T);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 10, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 11, sizeof(int),   &slice);
			runKernelErrorEstimateAR4ModelsSlice = clEnqueueNDRangeKernel(commandQueue, EstimateAR4ModelsSliceKernel, 3, NULL, globalWorkSizeEstimateAR4Models, localWorkSizeEstimateAR4Models, 0, NULL, NULL);

			clReleaseMemObject(d_xtxxt_GLM);
			allocatedDeviceMemory -= NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float);
		}

		// Smooth auto correlation estimates
		//PerformSmoothingNormalized(d_AR1_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		//PerformSmoothingNormalized(d_AR2_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		//PerformSmoothingNormalized(d_AR3_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		//PerformSmoothingNormalized(d_AR4_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

		for (size_t slice = 0; slice < EPI_DATA_D; slice++)
		{
			// Copy fMRI data to the device, for the current slice
			CopyCurrentfMRISliceToDevice(d_fMRI_Volumes, h_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

			// Apply whitening to data
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 0,  sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 1,  sizeof(cl_mem), &d_fMRI_Volumes);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 2,  sizeof(cl_mem), &d_AR1_Estimates);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 3,  sizeof(cl_mem), &d_AR2_Estimates);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 4,  sizeof(cl_mem), &d_AR3_Estimates);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 5,  sizeof(cl_mem), &d_AR4_Estimates);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 6,  sizeof(cl_mem), &d_EPI_Mask);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 7,  sizeof(int),    &EPI_DATA_W);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 8,  sizeof(int),    &EPI_DATA_H);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 9,  sizeof(int),    &one);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 10, sizeof(int),    &EPI_DATA_T);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 11, sizeof(int),    &slice);
			runKernelErrorApplyWhiteningAR4Slice = clEnqueueNDRangeKernel(commandQueue, ApplyWhiteningAR4SliceKernel, 3, NULL, globalWorkSizeApplyWhiteningAR4, localWorkSizeApplyWhiteningAR4, 0, NULL, NULL);

			// Copy fMRI data to the host, for the current slice
			CopyCurrentfMRISliceToHost(h_Whitened_fMRI_Volumes, d_Whitened_fMRI_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
		}

		// First four timepoints are now invalid
		SetMemory(c_Censored_Timepoints, 0.0f, 4);
		NUMBER_OF_INVALID_TIMEPOINTS = 4;
	}

	for (size_t slice = 0; slice < EPI_DATA_D; slice++)
	{
		// Create a mapping between voxel coordinates and brain voxel number, since we cannot store the modified GLM design matrix for all voxels, only for the brain voxels
		CreateVoxelNumbersSlice(d_Voxel_Numbers, d_EPI_Mask, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

		// Allocate memory for voxel specific design matrices (sufficient to store the pseudo inverses, since we only need to estimate beta weights with the voxel-specific models, not the residuals)
		// Only store for brain voxels, which differs for each slice
		cl_mem d_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);

		// Apply whitening to model and create voxel-specific models
		WhitenDesignMatricesInverseSlice(d_xtxxt_GLM, h_X_GLM, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, d_Voxel_Numbers, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS);

		// Copy fMRI data to the device, for the current slice
		CopyCurrentfMRISliceToDevice(d_Whitened_fMRI_Volumes, h_Whitened_fMRI_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

		// Calculate beta values, using whitened data and the whitened voxel-specific models
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 0,  sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 1,  sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 2,  sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 3,  sizeof(cl_mem), &d_xtxxt_GLM);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 4,  sizeof(cl_mem), &d_Voxel_Numbers);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 5,  sizeof(cl_mem), &c_Censored_Timepoints);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 6,  sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 7,  sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 8,  sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 9,  sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 10, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 11, sizeof(int),    &NUMBER_OF_INVALID_TIMEPOINTS);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 12, sizeof(int),    &slice);
		runKernelErrorCalculateBetaWeightsGLMFirstLevelSlice = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMFirstLevelSliceKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
		clFinish(commandQueue);

		// d_xtxxt_GLM now contains X_GLM and not xtxxt_GLM ...
		WhitenDesignMatricesTTestSlice(d_xtxxt_GLM, d_GLM_Scalars, h_X_GLM, h_Contrasts, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, d_Voxel_Numbers, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS, NUMBER_OF_CONTRASTS);

		// Finally calculate statistical maps using whitened model and whitened data
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 0,  sizeof(cl_mem), &d_Statistical_Maps);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 1,  sizeof(cl_mem), &d_Contrast_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 2,  sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 3,  sizeof(cl_mem), &d_Residual_Variances);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 4,  sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 5,  sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 6,  sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 7,  sizeof(cl_mem), &d_xtxxt_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 8,  sizeof(cl_mem), &d_GLM_Scalars);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 9,  sizeof(cl_mem), &d_Voxel_Numbers);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 10, sizeof(cl_mem), &c_Contrasts);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 11, sizeof(cl_mem), &c_Censored_Timepoints);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 12, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 13, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 14, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 15, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 16, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 17, sizeof(int),    &NUMBER_OF_CONTRASTS);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 18, sizeof(int),    &NUMBER_OF_INVALID_TIMEPOINTS);
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 19, sizeof(int),    &slice);
		runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelSlice = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMTTestFirstLevelSliceKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
		clFinish(commandQueue);

		if (WRITE_RESIDUALS_EPI)
		{
			// Copy residuals to the host, for the current slice			
			CopyCurrentfMRISliceToHost(h_Residuals_EPI, d_Residuals, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
		}

		clReleaseMemObject(d_xtxxt_GLM);
	}

	MultiplyVolumes(d_AR1_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR2_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR3_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	clReleaseMemObject(d_GLM_Scalars);
	clReleaseMemObject(d_Voxel_Numbers);
	clReleaseMemObject(c_Censored_Timepoints);

	allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * sizeof(float);
	allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * NUMBER_OF_CONTRASTS * sizeof(float);

	free(h_Whitened_fMRI_Volumes);

	allocatedHostMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);
}


// Calculates a statistical map for first level analysis, using a Cochrane-Orcutt procedure

void BROCCOLI_LIB::CalculateStatisticalMapsGLMFTestFirstLevel(float *h_Volumes, int iterations)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Volumes, 0, NULL, NULL);

	// Create a mapping between voxel coordinates and brain voxel number, since we cannot store the modified GLM design matrix for all voxels, only for the brain voxels
	cl_mem d_Voxel_Numbers = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);
	CreateVoxelNumbers(d_Voxel_Numbers, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	// Allocate memory for voxel specific design matrices (sufficient to store the pseudo inverses, since we only need to estimate beta weights with the voxel-specific models, not the residuals)
	cl_mem d_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);

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
	WhitenDesignMatricesInverse(d_xtxxt_GLM, h_X_GLM, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, d_Voxel_Numbers, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS);

	// Set whitened volumes to original volumes
	clEnqueueCopyBuffer(commandQueue, d_fMRI_Volumes, d_Whitened_fMRI_Volumes, 0, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), 0, NULL, NULL);

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
		//clSetKernelArg(CalculateGLMResidualsKernel, 0, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(CalculateGLMResidualsKernel, 0, sizeof(cl_mem), &d_Whitened_fMRI_Volumes); // Store residuals in whitened fMRI volumes
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
		//clSetKernelArg(EstimateAR4ModelsKernel, 4, sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(EstimateAR4ModelsKernel, 4, sizeof(cl_mem), &d_Whitened_fMRI_Volumes); // Residuals being stored in whitened fMRI volumes
		clSetKernelArg(EstimateAR4ModelsKernel, 5, sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(EstimateAR4ModelsKernel, 6, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(EstimateAR4ModelsKernel, 7, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(EstimateAR4ModelsKernel, 8, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(EstimateAR4ModelsKernel, 9, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(EstimateAR4ModelsKernel, 10, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);
		runKernelErrorEstimateAR4Models = clEnqueueNDRangeKernel(commandQueue, EstimateAR4ModelsKernel, 3, NULL, globalWorkSizeEstimateAR4Models, localWorkSizeEstimateAR4Models, 0, NULL, NULL);

		// Smooth auto correlation estimates
		//PerformSmoothingNormalized(d_AR1_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		//PerformSmoothingNormalized(d_AR2_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		//PerformSmoothingNormalized(d_AR3_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		//PerformSmoothingNormalized(d_AR4_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

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
		WhitenDesignMatricesInverse(d_xtxxt_GLM, h_X_GLM, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, d_Voxel_Numbers, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS);
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
	WhitenDesignMatricesFTest(d_xtxxt_GLM, d_GLM_Scalars, h_X_GLM, h_Contrasts, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, d_Voxel_Numbers, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS, NUMBER_OF_CONTRASTS);

	// Finally calculate statistical maps using whitened model and whitened data
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 0, sizeof(cl_mem),  &d_Statistical_Maps);
	//clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 1, sizeof(cl_mem),  &d_Residuals);
	clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelKernel, 1, sizeof(cl_mem),  &d_fMRI_Volumes);  // Store residuals in original fMRI volumes
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

	if (WRITE_RESIDUALS_EPI)
	{
		clEnqueueReadBuffer(commandQueue, d_fMRI_Volumes, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_Residuals_EPI, 0, NULL, NULL);
	}

	MultiplyVolumes(d_AR1_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR2_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR3_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	clReleaseMemObject(d_xtxxt_GLM);
	clReleaseMemObject(d_GLM_Scalars);
	clReleaseMemObject(c_Censored_Timepoints);
	clReleaseMemObject(d_Voxel_Numbers);

	allocatedDeviceMemory -= NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float);
	allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * NUMBER_OF_CONTRASTS * sizeof(float);
	allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);
}

void BROCCOLI_LIB::CalculateStatisticalMapsGLMFTestFirstLevelSlices(float* h_Volumes, int iterations)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, 1);

	// Allocate memory for voxel numbers
	cl_mem d_Voxel_Numbers = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * sizeof(float), NULL, NULL);

	// Allocate memory for voxel specific GLM scalars
	cl_mem d_GLM_Scalars = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float), NULL, NULL);

	allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * sizeof(float);
	allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float);

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
	
	// Flip the fMRI data from x,y,z,t to x,y,t,z, to be able to copy all time points for one slice
	//FlipVolumesXYZTtoXYTZ(h_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	h_Whitened_fMRI_Volumes = (float*)malloc(EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float));
	memcpy(h_Whitened_fMRI_Volumes, h_Volumes, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float));	
	allocatedHostMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);

	int one = 1;

	// Cochrane-Orcutt procedure, iterate
	for (int it = 0; it < iterations; it++)
	{
		for (size_t slice = 0; slice < EPI_DATA_D; slice++)
		{
			// Create a mapping between voxel coordinates and brain voxel number, since we cannot store the modified GLM design matrix for all voxels, only for the brain voxels
			CreateVoxelNumbersSlice(d_Voxel_Numbers, d_EPI_Mask, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

			// Allocate memory for voxel specific design matrices (sufficient to store the pseudo inverses, since we only need to estimate beta weights with the voxel-specific models, not the residuals)
			// Only store for brain voxels, which differs for each slice
			cl_mem d_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);

			allocatedDeviceMemory += NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float);
			PrintMemoryStatus("Inside GLM");
			
			// Apply whitening to model and create voxel-specific models
			WhitenDesignMatricesInverseSlice(d_xtxxt_GLM, h_X_GLM, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, d_Voxel_Numbers, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS);

			// Copy fMRI data to the device, for the current slice
			CopyCurrentfMRISliceToDevice(d_Whitened_fMRI_Volumes, h_Whitened_fMRI_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

			// Calculate beta values, using whitened data and the whitened voxel-specific models
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 0,  sizeof(cl_mem), &d_Beta_Volumes);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 1,  sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 2,  sizeof(cl_mem), &d_EPI_Mask);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 3,  sizeof(cl_mem), &d_xtxxt_GLM);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 4,  sizeof(cl_mem), &d_Voxel_Numbers);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 5,  sizeof(cl_mem), &c_Censored_Timepoints);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 6,  sizeof(int),    &EPI_DATA_W);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 7,  sizeof(int),    &EPI_DATA_H);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 8,  sizeof(int),    &EPI_DATA_D);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 9,  sizeof(int),    &EPI_DATA_T);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 10, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 11, sizeof(int),    &NUMBER_OF_INVALID_TIMEPOINTS);
			clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 12, sizeof(int),    &slice);
			runKernelErrorCalculateBetaWeightsGLMFirstLevelSlice = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMFirstLevelSliceKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
			clFinish(commandQueue);

			// Copy fMRI data to the device, for the current slice
			CopyCurrentfMRISliceToDevice(d_fMRI_Volumes, h_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

			// Calculate residuals, using original data and the original model
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 0, sizeof(cl_mem), &d_Residuals);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 1, sizeof(cl_mem), &d_fMRI_Volumes);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 2, sizeof(cl_mem), &d_Beta_Volumes);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 4, sizeof(cl_mem), &c_X_GLM);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 5, sizeof(int),    &EPI_DATA_W);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 6, sizeof(int),    &EPI_DATA_H);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 7, sizeof(int),    &EPI_DATA_D);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 8, sizeof(int),    &EPI_DATA_T);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 9, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
			clSetKernelArg(CalculateGLMResidualsSliceKernel, 10, sizeof(int),   &slice);
			runKernelErrorCalculateGLMResidualsSlice = clEnqueueNDRangeKernel(commandQueue, CalculateGLMResidualsSliceKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
			clFinish(commandQueue);

			// Estimate auto correlation from residuals
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 0, sizeof(cl_mem), &d_AR1_Estimates);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 1, sizeof(cl_mem), &d_AR2_Estimates);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 2, sizeof(cl_mem), &d_AR3_Estimates);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 3, sizeof(cl_mem), &d_AR4_Estimates);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 4, sizeof(cl_mem), &d_Residuals);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 5, sizeof(cl_mem), &d_EPI_Mask);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 6, sizeof(int),    &EPI_DATA_W);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 7, sizeof(int),    &EPI_DATA_H);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 8, sizeof(int),    &EPI_DATA_D);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 9, sizeof(int),    &EPI_DATA_T);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 10, sizeof(int),   &NUMBER_OF_INVALID_TIMEPOINTS);
			clSetKernelArg(EstimateAR4ModelsSliceKernel, 11, sizeof(int),   &slice);
			runKernelErrorEstimateAR4ModelsSlice = clEnqueueNDRangeKernel(commandQueue, EstimateAR4ModelsSliceKernel, 3, NULL, globalWorkSizeEstimateAR4Models, localWorkSizeEstimateAR4Models, 0, NULL, NULL);

			clReleaseMemObject(d_xtxxt_GLM);
			allocatedDeviceMemory -= NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float);
		}

		// Smooth auto correlation estimates
		//PerformSmoothingNormalized(d_AR1_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		//PerformSmoothingNormalized(d_AR2_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		//PerformSmoothingNormalized(d_AR3_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);
		//PerformSmoothingNormalized(d_AR4_Estimates, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, 1);

		for (size_t slice = 0; slice < EPI_DATA_D; slice++)
		{
			// Copy fMRI data to the device, for the current slice
			CopyCurrentfMRISliceToDevice(d_fMRI_Volumes, h_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

			// Apply whitening to data
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 0,  sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 1,  sizeof(cl_mem), &d_fMRI_Volumes);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 2,  sizeof(cl_mem), &d_AR1_Estimates);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 3,  sizeof(cl_mem), &d_AR2_Estimates);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 4,  sizeof(cl_mem), &d_AR3_Estimates);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 5,  sizeof(cl_mem), &d_AR4_Estimates);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 6,  sizeof(cl_mem), &d_EPI_Mask);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 7,  sizeof(int),    &EPI_DATA_W);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 8,  sizeof(int),    &EPI_DATA_H);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 9,  sizeof(int),    &one);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 10, sizeof(int),    &EPI_DATA_T);
			clSetKernelArg(ApplyWhiteningAR4SliceKernel, 11, sizeof(int),    &slice);
			runKernelErrorApplyWhiteningAR4Slice = clEnqueueNDRangeKernel(commandQueue, ApplyWhiteningAR4SliceKernel, 3, NULL, globalWorkSizeApplyWhiteningAR4, localWorkSizeApplyWhiteningAR4, 0, NULL, NULL);

			// Copy fMRI data to the host, for the current slice
			CopyCurrentfMRISliceToHost(h_Whitened_fMRI_Volumes, d_Whitened_fMRI_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
		}

		// First four timepoints are now invalid
		SetMemory(c_Censored_Timepoints, 0.0f, 4);
		NUMBER_OF_INVALID_TIMEPOINTS = 4;
	}

	for (size_t slice = 0; slice < EPI_DATA_D; slice++)
	{
		// Create a mapping between voxel coordinates and brain voxel number, since we cannot store the modified GLM design matrix for all voxels, only for the brain voxels
		CreateVoxelNumbersSlice(d_Voxel_Numbers, d_EPI_Mask, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

		// Allocate memory for voxel specific design matrices (sufficient to store the pseudo inverses, since we only need to estimate beta weights with the voxel-specific models, not the residuals)
		// Only store for brain voxels, which differs for each slice
		cl_mem d_xtxxt_GLM = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_TOTAL_GLM_REGRESSORS * EPI_DATA_T * sizeof(float), NULL, NULL);

		// Apply whitening to model and create voxel-specific models
		WhitenDesignMatricesInverseSlice(d_xtxxt_GLM, h_X_GLM, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, d_Voxel_Numbers, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS);

		// Copy fMRI data to the device, for the current slice
		CopyCurrentfMRISliceToDevice(d_Whitened_fMRI_Volumes, h_Whitened_fMRI_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

		// Calculate beta values, using whitened data and the whitened voxel-specific models
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 0,  sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 1,  sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 2,  sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 3,  sizeof(cl_mem), &d_xtxxt_GLM);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 4,  sizeof(cl_mem), &d_Voxel_Numbers);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 5,  sizeof(cl_mem), &c_Censored_Timepoints);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 6,  sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 7,  sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 8,  sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 9,  sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 10, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 11, sizeof(int),    &NUMBER_OF_INVALID_TIMEPOINTS);
		clSetKernelArg(CalculateBetaWeightsGLMFirstLevelSliceKernel, 12, sizeof(int),    &slice);
		runKernelErrorCalculateBetaWeightsGLMFirstLevelSlice = clEnqueueNDRangeKernel(commandQueue, CalculateBetaWeightsGLMFirstLevelSliceKernel, 3, NULL, globalWorkSizeCalculateBetaWeightsGLM, localWorkSizeCalculateBetaWeightsGLM, 0, NULL, NULL);
		clFinish(commandQueue);

		// d_xtxxt_GLM now contains X_GLM and not xtxxt_GLM ...
		WhitenDesignMatricesFTestSlice(d_xtxxt_GLM, d_GLM_Scalars, h_X_GLM, h_Contrasts, d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates, d_EPI_Mask, d_Voxel_Numbers, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T, NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_INVALID_TIMEPOINTS, NUMBER_OF_CONTRASTS);

		// Finally calculate statistical maps using whitened model and whitened data
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 0,  sizeof(cl_mem), &d_Statistical_Maps);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 1,  sizeof(cl_mem), &d_Residuals);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 2,  sizeof(cl_mem), &d_Residual_Variances);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 3,  sizeof(cl_mem), &d_Whitened_fMRI_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 4,  sizeof(cl_mem), &d_Beta_Volumes);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 5,  sizeof(cl_mem), &d_EPI_Mask);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 6,  sizeof(cl_mem), &d_xtxxt_GLM);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 7,  sizeof(cl_mem), &d_GLM_Scalars);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 8,  sizeof(cl_mem), &d_Voxel_Numbers);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 9,  sizeof(cl_mem), &c_Contrasts);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 10, sizeof(cl_mem), &c_Censored_Timepoints);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 11, sizeof(int),    &EPI_DATA_W);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 12, sizeof(int),    &EPI_DATA_H);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 13, sizeof(int),    &EPI_DATA_D);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 14, sizeof(int),    &EPI_DATA_T);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 15, sizeof(int),    &NUMBER_OF_TOTAL_GLM_REGRESSORS);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 16, sizeof(int),    &NUMBER_OF_CONTRASTS);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 17, sizeof(int),    &NUMBER_OF_INVALID_TIMEPOINTS);
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 18, sizeof(int),    &slice);
		runKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelSlice = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMFTestFirstLevelSliceKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
		clFinish(commandQueue);

		if (WRITE_RESIDUALS_EPI)
		{
			// Copy residuals to the host, for the current slice			
			CopyCurrentfMRISliceToHost(h_Residuals_EPI, d_Residuals, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
		}

		clReleaseMemObject(d_xtxxt_GLM);
	}

	MultiplyVolumes(d_AR1_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR2_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR3_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
	MultiplyVolumes(d_AR4_Estimates, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);

	clReleaseMemObject(d_GLM_Scalars);
	clReleaseMemObject(d_Voxel_Numbers);
	clReleaseMemObject(c_Censored_Timepoints);

	allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * sizeof(float);
	allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float);

	free(h_Whitened_fMRI_Volumes);

	allocatedHostMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float);
}

// This function currently only works for 2 regressors
void BROCCOLI_LIB::CalculateStatisticalMapsGLMBayesianFirstLevel(float* h_Volumes)
{
	// Allocate memory for one slice, and all timepoints
	cl_mem d_Regressed_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
	cl_mem d_Volumes = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * 1 * EPI_DATA_T * sizeof(float), NULL, NULL);
	cl_mem d_Seeds = clCreateBuffer(context, CL_MEM_READ_ONLY, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int), NULL, NULL);

	allocatedDeviceMemory += 2 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_T * sizeof(float);
	allocatedDeviceMemory += EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);
	deviceMemoryAllocations += 3;

	PrintMemoryStatus("Inside Bayesian GLM");

	NUMBER_OF_TOTAL_GLM_REGRESSORS = 2;

	SetGlobalAndLocalWorkSizesStatisticalCalculations(EPI_DATA_W, EPI_DATA_H, 1);

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
	for (size_t i = 0; i < EPI_DATA_W * EPI_DATA_H * EPI_DATA_D; i++)
	{
		h_Seeds[i] = rand();
	}
	clEnqueueWriteBuffer(commandQueue, d_Seeds, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(int), h_Seeds, 0, NULL, NULL);
	clFinish(commandQueue);
	free(h_Seeds);

	// Flip the fMRI data from x,y,z,t to x,y,t,z, to be able to copy all time points for one slice
	//FlipVolumesXYZTtoXYTZ(h_Volumes, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

	// Loop over slices, to save memory
	for (size_t slice = 0; slice < EPI_DATA_D; slice++)
	{
		if ( (WRAPPER == BASH) && (VERBOS) )
		{
			printf("Bayesian GLM slice %zu\n",slice);
		}

		// Copy fMRI data to the device, for the current slice
		CopyCurrentfMRISliceToDevice(d_Volumes, h_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
		//CopyCurrentfMRISliceToDevice(d_Regressed_Volumes, h_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_T);

		// Remove linear fit of detrending regressors and motion regressors
		PerformDetrendingAndMotionRegressionSlice(d_Regressed_Volumes, d_Volumes, slice, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);

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
		clSetKernelArg(CalculateStatisticalMapsGLMBayesianKernel, 17, sizeof(int),   &slice);
		runKernelErrorCalculateStatisticalMapsGLMBayesian = clEnqueueNDRangeKernel(commandQueue, CalculateStatisticalMapsGLMBayesianKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, 	localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
		clFinish(commandQueue);
	}

	free(h_X_GLM_);
	free(h_S00);
	free(h_S01);
	free(h_S11);
	free(h_InvOmega0);

	clReleaseMemObject(d_Regressed_Volumes);
	clReleaseMemObject(d_Volumes);
	clReleaseMemObject(d_Seeds);
	clReleaseMemObject(c_InvOmega0);
	clReleaseMemObject(c_S00);
	clReleaseMemObject(c_S01);
	clReleaseMemObject(c_S11);

	allocatedDeviceMemory -= 2 * EPI_DATA_W * EPI_DATA_H * EPI_DATA_T * sizeof(float);
	allocatedDeviceMemory -= EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float);
	deviceMemoryDeallocations += 3;
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
	for (size_t i = 0; i < EPI_DATA_W * EPI_DATA_H * EPI_DATA_D; i++)
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
void BROCCOLI_LIB::PutWhitenedModelsIntoVolumes(cl_mem d_Mask, cl_mem d_xtxxt_GLM, size_t DATA_W, size_t DATA_H, size_t DATA_D, size_t DATA_T, size_t NUMBER_OF_REGRESSORS)
{
	float* h_Mask = (float*)malloc(DATA_W * DATA_H * DATA_D * sizeof(float));
	float* h_xtxxt_GLM_ = (float*)malloc(NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float));

	clEnqueueReadBuffer(commandQueue, d_Mask, CL_TRUE, 0, DATA_W * DATA_H * DATA_D * sizeof(float), h_Mask, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_xtxxt_GLM, CL_TRUE, 0, NUMBER_OF_BRAIN_VOXELS * NUMBER_OF_REGRESSORS * DATA_T * sizeof(float), h_xtxxt_GLM_, 0, NULL, NULL);

	// Loop over voxels
	size_t voxel_number = 0;
	for (size_t z = 0; z < DATA_D; z++)
	{
		for (size_t y = 0; y < DATA_H; y++)
		{
			for (size_t x = 0; x < DATA_W; x++)
			{
				for (size_t r = 0; r < NUMBER_OF_REGRESSORS; r++)
				{
					for (size_t t = 0; t < DATA_T; t++)
					{
						h_Whitened_Models[x + y * DATA_W + z * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D + r * DATA_W * DATA_H * DATA_D * DATA_T] = 0.0f;
					}
				}

				if ( h_Mask[x + y * DATA_W + z * DATA_W * DATA_H] == 1.0f )
				{
					for (size_t r = 0; r < NUMBER_OF_REGRESSORS; r++)
					{
						for (size_t t = 0; t < DATA_T; t++)
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
		                                         size_t DATA_W,
		                                         size_t DATA_H,
		                                         size_t DATA_D,
		                                         size_t DATA_T,
		                                         size_t NUMBER_OF_REGRESSORS)
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
	size_t voxel_number = 0;
	for (size_t z = 0; z < DATA_D; z++)
	{
		for (size_t y = 0; y < DATA_H; y++)
		{
			for (size_t x = 0; x < DATA_W; x++)
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

					for (size_t r = 0; r < NUMBER_OF_REGRESSORS; r++)
					{
						for (size_t t = 0; t < DATA_T; t++)
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
	clSetKernelArg(SeparableConvolutionRowsKernel, 1, sizeof(cl_mem), &d_Temp_fMRI_Volumes_2);
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

	clSetKernelArg(SeparableConvolutionRodsKernel, 0, sizeof(cl_mem), &d_Temp_fMRI_Volumes_2);
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
		clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel, 1, sizeof(cl_mem), &d_Temp_fMRI_Volumes_2);
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
		clSetKernelArg(CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel, 1, sizeof(cl_mem), &d_Temp_fMRI_Volumes_2);
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

	int zero = 0;

	clSetKernelArg(SetStartClusterIndicesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(SetStartClusterIndicesKernel, 1, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(SetStartClusterIndicesKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(SetStartClusterIndicesKernel, 3, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(SetStartClusterIndicesKernel, 4, sizeof(int),    &zero);
	clSetKernelArg(SetStartClusterIndicesKernel, 5, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(SetStartClusterIndicesKernel, 6, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(SetStartClusterIndicesKernel, 7, sizeof(int),    &EPI_DATA_D);

	clSetKernelArg(ClusterizeScanKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeScanKernel, 1, sizeof(cl_mem), &d_Updated);
	clSetKernelArg(ClusterizeScanKernel, 2, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(ClusterizeScanKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(ClusterizeScanKernel, 4, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(ClusterizeScanKernel, 5, sizeof(int),    &zero);
	clSetKernelArg(ClusterizeScanKernel, 6, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(ClusterizeScanKernel, 7, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(ClusterizeScanKernel, 8, sizeof(int),    &EPI_DATA_D);

	clSetKernelArg(ClusterizeRelabelKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeRelabelKernel, 1, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(ClusterizeRelabelKernel, 2, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(ClusterizeRelabelKernel, 3, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(ClusterizeRelabelKernel, 4, sizeof(int),    &zero);
	clSetKernelArg(ClusterizeRelabelKernel, 5, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(ClusterizeRelabelKernel, 6, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(ClusterizeRelabelKernel, 7, sizeof(int),    &EPI_DATA_D);

	clSetKernelArg(CalculateClusterSizesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(CalculateClusterSizesKernel, 1, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateClusterSizesKernel, 2, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateClusterSizesKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateClusterSizesKernel, 4, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(CalculateClusterSizesKernel, 5, sizeof(int),    &zero);
	clSetKernelArg(CalculateClusterSizesKernel, 6, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(CalculateClusterSizesKernel, 7, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(CalculateClusterSizesKernel, 8, sizeof(int),    &EPI_DATA_D);

	clSetKernelArg(CalculateClusterMassesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(CalculateClusterMassesKernel, 1, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateClusterMassesKernel, 2, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateClusterMassesKernel, 3, sizeof(cl_mem), &d_EPI_Mask);
	clSetKernelArg(CalculateClusterMassesKernel, 4, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(CalculateClusterMassesKernel, 5, sizeof(int),    &zero);
	clSetKernelArg(CalculateClusterMassesKernel, 6, sizeof(int),    &EPI_DATA_W);
	clSetKernelArg(CalculateClusterMassesKernel, 7, sizeof(int),    &EPI_DATA_H);
	clSetKernelArg(CalculateClusterMassesKernel, 8, sizeof(int),    &EPI_DATA_D);

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

	int zero = 0;

	clSetKernelArg(SetStartClusterIndicesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(SetStartClusterIndicesKernel, 1, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(SetStartClusterIndicesKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(SetStartClusterIndicesKernel, 3, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(SetStartClusterIndicesKernel, 4, sizeof(int),    &zero);
	clSetKernelArg(SetStartClusterIndicesKernel, 5, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(SetStartClusterIndicesKernel, 6, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(SetStartClusterIndicesKernel, 7, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(ClusterizeScanKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeScanKernel, 1, sizeof(cl_mem), &d_Updated);
	clSetKernelArg(ClusterizeScanKernel, 2, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(ClusterizeScanKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(ClusterizeScanKernel, 4, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(ClusterizeScanKernel, 5, sizeof(int),    &zero);
	clSetKernelArg(ClusterizeScanKernel, 6, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(ClusterizeScanKernel, 7, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(ClusterizeScanKernel, 8, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(ClusterizeRelabelKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeRelabelKernel, 1, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(ClusterizeRelabelKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(ClusterizeRelabelKernel, 3, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(ClusterizeRelabelKernel, 4, sizeof(int),    &zero);
	clSetKernelArg(ClusterizeRelabelKernel, 5, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(ClusterizeRelabelKernel, 6, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(ClusterizeRelabelKernel, 7, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(CalculateClusterSizesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(CalculateClusterSizesKernel, 1, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateClusterSizesKernel, 2, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateClusterSizesKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateClusterSizesKernel, 4, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(CalculateClusterSizesKernel, 5, sizeof(int),    &zero);
	clSetKernelArg(CalculateClusterSizesKernel, 6, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateClusterSizesKernel, 7, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateClusterSizesKernel, 8, sizeof(int),    &MNI_DATA_D);

	clSetKernelArg(CalculateClusterMassesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(CalculateClusterMassesKernel, 1, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateClusterMassesKernel, 2, sizeof(cl_mem), &d_Statistical_Maps);
	clSetKernelArg(CalculateClusterMassesKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateClusterMassesKernel, 4, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
	clSetKernelArg(CalculateClusterMassesKernel, 5, sizeof(int),    &zero);
	clSetKernelArg(CalculateClusterMassesKernel, 6, sizeof(int),    &MNI_DATA_W);
	clSetKernelArg(CalculateClusterMassesKernel, 7, sizeof(int),    &MNI_DATA_H);
	clSetKernelArg(CalculateClusterMassesKernel, 8, sizeof(int),    &MNI_DATA_D);

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

	if (STATISTICAL_TEST != GROUP_MEAN)
	{
		clSetKernelArg(TransformDataKernel, 0, sizeof(cl_mem), &d_Transformed_Volumes);
		clSetKernelArg(TransformDataKernel, 1, sizeof(cl_mem), &d_Volumes);
		clSetKernelArg(TransformDataKernel, 2, sizeof(cl_mem), &d_Mask);
		clSetKernelArg(TransformDataKernel, 3, sizeof(cl_mem), &c_Transformation_Matrix);
		clSetKernelArg(TransformDataKernel, 4, sizeof(int),    &MNI_DATA_W);
		clSetKernelArg(TransformDataKernel, 5, sizeof(int),    &MNI_DATA_H);
		clSetKernelArg(TransformDataKernel, 6, sizeof(int),    &MNI_DATA_D);
		clSetKernelArg(TransformDataKernel, 7, sizeof(int),    &NUMBER_OF_SUBJECTS);
	}
}

void BROCCOLI_LIB::CleanupPermutationTestSecondLevel()
{
	clReleaseMemObject(d_Largest_Cluster);
	clReleaseMemObject(d_Updated);
}

void BROCCOLI_LIB::CalculateStatisticalMapsFirstLevelPermutation(int contrast)
{
	if (STATISTICAL_TEST == TTEST)
	{
		CalculateStatisticalMapsGLMTTestFirstLevelPermutation(contrast);
	}
	else if (STATISTICAL_TEST == FTEST)
	{
		CalculateStatisticalMapsGLMFTestFirstLevelPermutation();
	}
}

// Calculates a statistical t-map for second level analysis, all kernel parameters have been set in SetupPermutationTestSecondLevel
void BROCCOLI_LIB::CalculateStatisticalMapsGLMTTestFirstLevelPermutation(int contrast)
{
	clSetKernelArg(CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel, 13, sizeof(int),   &contrast);
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
void BROCCOLI_LIB::CalculateStatisticalMapsSecondLevelPermutation(int p, int contrast)
{
   	if (STATISTICAL_TEST == GROUP_MEAN)
	{
   		// Copy a new sign vector to constant memory
	   	clEnqueueWriteBuffer(commandQueue, c_Sign_Vector, CL_TRUE, 0, NUMBER_OF_SUBJECTS * sizeof(float), &h_Sign_Matrix[p * NUMBER_OF_SUBJECTS], 0, NULL, NULL);
		CalculateStatisticalMapsMeanSecondLevelPermutation();
	}
   	else if (STATISTICAL_TEST == TTEST)
	{
		h_Permutation_Matrix = h_Permutation_Matrices[contrast];
   		// Copy a new permutation vector to constant memory
	   	clEnqueueWriteBuffer(commandQueue, c_Permutation_Vector, CL_TRUE, 0, NUMBER_OF_SUBJECTS * sizeof(unsigned short int), &h_Permutation_Matrix[p * NUMBER_OF_SUBJECTS], 0, NULL, NULL);
		// Set current contrast
		clSetKernelArg(CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel, 13, sizeof(int),   &contrast);
		CalculateStatisticalMapsGLMTTestSecondLevelPermutation();
	}
	else if (STATISTICAL_TEST == FTEST)
	{
		h_Permutation_Matrix = h_Permutation_Matrices[contrast];
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
		clFinish(commandQueue);		
		
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
		clSetKernelArg(ApplyWhiteningAR4Kernel, 0, sizeof(cl_mem), &d_Whitened_Volumes);
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
		clFinish(commandQueue);

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
void BROCCOLI_LIB::ApplyPermutationTestFirstLevel(float* h_fMRI_Volumes)
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
	SetMemory(d_Temp_fMRI_Volumes_1, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T);
	SetMemory(d_Temp_fMRI_Volumes_2, 0.0f, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T);

	// Copy fMRI data to first temporary location
	clEnqueueWriteBuffer(commandQueue, d_Temp_fMRI_Volumes_1, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * EPI_DATA_T * sizeof(float), h_fMRI_Volumes, 0, NULL, NULL);

	// Generate a random permutation matrix
	GeneratePermutationMatrixFirstLevel();

	// Remove mean and linear, quadratic and cubic trends
	//PerformDetrending(d_Temp_fMRI_Volumes_2, d_Temp_fMRI_Volumes_1, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	PerformRegression(d_Temp_fMRI_Volumes_2, d_Temp_fMRI_Volumes_1, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
	
	// Make the timeseries white prior to the random permutations
	PerformWhiteningPriorPermutations(d_Temp_fMRI_Volumes_1, d_Temp_fMRI_Volumes_2);

	// Setup parameters and memory prior to permutations, to save time in each permutation
	SetupPermutationTestFirstLevel();

	// Loop over contrasts
	for (size_t c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		if ((WRAPPER == BASH) && PRINT)
		{
			printf("Contrast %zu, permutation, ", c+1);
			fflush(stdout);
		}

		for (size_t p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
		{
			if (((p+1) % 100) == 0)
			{ 
				if ((WRAPPER == BASH) && PRINT)
				{
					printf("%zu, ",p+1);
					fflush(stdout);
				}
			}

			// Generate new fMRI volumes, through inverse whitening and permutation
		   	GeneratePermutedVolumesFirstLevel(d_Temp_fMRI_Volumes_2, d_Temp_fMRI_Volumes_1, p);

			// Smooth new fMRI volumes (smoothing needs to be done in each permutation, as it otherwise alters the AR parameters)
			//PerformSmoothingNormalized(d_Permuted_fMRI_Volumes, d_EPI_Mask, d_Smoothed_EPI_Mask, h_Smoothing_Filter_X, h_Smoothing_Filter_Y, h_Smoothing_Filter_Z, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T);
			//PerformSmoothingNormalizedPermutation();

			// Calculate statistical maps, for current contrast
			CalculateStatisticalMapsFirstLevelPermutation(c);

			// Voxel distribution
			if (INFERENCE_MODE == VOXEL)
			{
				// Get max test value
				h_Permutation_Distribution[p + c * NUMBER_OF_PERMUTATIONS] = CalculateMaxAtomic(d_Statistical_Maps, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
				if ( (WRAPPER == BASH) && VERBOS )
				{
					printf("Max test value is %f \n",h_Permutation_Distribution[p + c * NUMBER_OF_PERMUTATIONS]);
				}
			}
			// Cluster distribution, extent or mass
			else if ( (INFERENCE_MODE == CLUSTER_EXTENT) || (INFERENCE_MODE == CLUSTER_MASS) )
			{
				ClusterizeOpenCLPermutation(MAX_CLUSTER, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
				if ( (WRAPPER == BASH) && VERBOS )
				{
					printf("Max cluster is %f \n",MAX_CLUSTER);
				}
				h_Permutation_Distribution[p + c * NUMBER_OF_PERMUTATIONS] = MAX_CLUSTER;
			}
			// Threshold free cluster enhancement
			else if (INFERENCE_MODE == TFCE)
			{
				//maxActivation = CalculateMaxAtomic(d_Statistical_Maps, c, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D);
				float delta = 0.2846;
				//ClusterizeOpenCLTFCEPermutation(MAX_VALUE, d_EPI_Mask, EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, maxActivation, delta);
				//h_Permutation_Distribution[p + c * NUMBER_OF_PERMUTATIONS] = MAX_VALUE;
			}
		}

		std::vector<float> max_values (h_Permutation_Distribution + c * NUMBER_OF_PERMUTATIONS, h_Permutation_Distribution + (c + 1)*NUMBER_OF_PERMUTATIONS);
        std::sort (max_values.begin(), max_values.begin() + NUMBER_OF_PERMUTATIONS);
   
        // Find the threshold for the specified significance level
        SIGNIFICANCE_THRESHOLD = max_values[(int)(ceil((1.0f - SIGNIFICANCE_LEVEL) * (float)NUMBER_OF_PERMUTATIONS))-1];

        if (WRAPPER == BASH)
        {
            printf("\nPermutation threshold for contrast %zu for a significance level of %f is %f \n\n",c+1,SIGNIFICANCE_LEVEL, SIGNIFICANCE_THRESHOLD);
        }
	}

	CleanupPermutationTestFirstLevel();
}


Eigen::MatrixXd pinv(const Eigen::MatrixXd mat)
{
	int ncols = mat.cols();
	int nrows = mat.rows();

	if ((ncols > 1) && (nrows > 1) ) // Matrix
	{
	    Eigen::MatrixXd xtx = mat.transpose() * mat;
    	Eigen::MatrixXd inv_xtx = xtx.inverse();
    	return inv_xtx * mat.transpose();
	}
	else // Vector
	{
		Eigen::MatrixXd mag(1,1);
		if (nrows > ncols)  // Column vector
		{
	        mag = mat.transpose() * mat;
		}
		else	// Row vector
		{
			mag = mat * mat.transpose();
		}
        
		return mat.transpose() / mag(0,0);
	}
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

    // Copy design matrix to matrix object
    Eigen::MatrixXd X_GLM(NUMBER_OF_SUBJECTS,NUMBER_OF_TOTAL_GLM_REGRESSORS);
    for (int s = 0; s < NUMBER_OF_SUBJECTS; s++)
    {
        for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
        {
            X_GLM(s,r) = (double)h_X_GLM_In[NUMBER_OF_SUBJECTS * r + s];
        }
    }

    // Copy contrast matrix to matrix object
    Eigen::MatrixXd Contrasts(NUMBER_OF_CONTRASTS,NUMBER_OF_TOTAL_GLM_REGRESSORS);
    for (int c = 0; c < NUMBER_OF_CONTRASTS; c++)
    {
        for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
        {
            Contrasts(c,r) = (double)h_Contrasts_In[r + c * NUMBER_OF_TOTAL_GLM_REGRESSORS];
        }
    }

    // Loop over number of statistical maps
    for (size_t c = 0; c < NUMBER_OF_STATISTICAL_MAPS; c++)
    {
	    // Generate a random permutation matrix, unless one is provided
    	if (!USE_PERMUTATION_FILE)
    	{
			if (GROUP_DESIGNS[c] == TWOSAMPLE)
			{
		        GeneratePermutationMatrixSecondLevelTwoSample(c);
			}
			else if (GROUP_DESIGNS[c] == CORRELATION)
			{
		        GeneratePermutationMatrixSecondLevelCorrelation(c);
			}
	    }

		if (STATISTICAL_TEST == TTEST)
		{
			clEnqueueWriteBuffer(commandQueue, d_First_Level_Results, CL_TRUE, 0, MNI_DATA_W * MNI_DATA_H * MNI_DATA_D * NUMBER_OF_SUBJECTS * sizeof(float), h_First_Level_Results , 0, NULL, NULL);
			clFinish(commandQueue);

			if (NUMBER_OF_TOTAL_GLM_REGRESSORS > 1)
			{
	    	    // Extract current contrast vector
	    	    Eigen::MatrixXd contrastVector(1,NUMBER_OF_TOTAL_GLM_REGRESSORS);
	    	    contrastVector = Contrasts.block(c,0,1,NUMBER_OF_TOTAL_GLM_REGRESSORS);

	    	    // Partition design matrix into two sets of regressors, effects of interest and nuisance effects
	    	    int r = 1;
	    	    int p = NUMBER_OF_TOTAL_GLM_REGRESSORS;

	    	    Eigen::MatrixXd tmp(NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_TOTAL_GLM_REGRESSORS);
				Eigen::MatrixXd contrastVectorPinv = pinv(contrastVector.transpose());	
	    	    tmp = Eigen::MatrixXd::Identity(NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_TOTAL_GLM_REGRESSORS) - contrastVector.transpose() * pinv(contrastVector.transpose());
   
	    	    Eigen::JacobiSVD<Eigen::MatrixXd> svd(tmp, Eigen::ComputeFullU | Eigen::ComputeFullV);
	    	    Eigen::MatrixXd c2 = svd.matrixU().block(0,0,NUMBER_OF_TOTAL_GLM_REGRESSORS,p-r);
	    	    Eigen::MatrixXd c3=c2.transpose();
    	
	    	    Eigen::MatrixXd C(NUMBER_OF_TOTAL_GLM_REGRESSORS,NUMBER_OF_TOTAL_GLM_REGRESSORS);
	    	    C.block(0,0,1,NUMBER_OF_TOTAL_GLM_REGRESSORS) = contrastVector;
	    	    C.block(1,0,p-r,NUMBER_OF_TOTAL_GLM_REGRESSORS) = c3;
       		
    		    Eigen::MatrixXd W = X_GLM * C.inverse();
    		    Eigen::MatrixXd W1 = W.block(0,0,NUMBER_OF_SUBJECTS,r);
    		    Eigen::MatrixXd W2 = W.block(0,r,NUMBER_OF_SUBJECTS,p-r);
		
				// Setup matrix that transforms the data vector in each voxel
				Eigen::MatrixXd transformationMatrix = Eigen::MatrixXd::Identity(NUMBER_OF_SUBJECTS, NUMBER_OF_SUBJECTS) - W2 * pinv(W2);
				Eigen::MatrixXf transformationMatrixf = transformationMatrix.cast<float>();
		
	    	    // Copy transformation matrix to constant memory
	    	    clEnqueueWriteBuffer(commandQueue, c_Transformation_Matrix, CL_TRUE, 0, NUMBER_OF_SUBJECTS * NUMBER_OF_SUBJECTS * sizeof(float), transformationMatrixf.data(), 0, NULL, NULL);
				clFinish(commandQueue);

				// Transform the data, only needed once since the permutations are done by permuting the design matrix
				runKernelErrorTransformData = clEnqueueNDRangeKernel(commandQueue, TransformDataKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
			clFinish(commandQueue);
			}	
		}
		else if (STATISTICAL_TEST == FTEST)
		{	
			if (NUMBER_OF_TOTAL_GLM_REGRESSORS > NUMBER_OF_CONTRASTS)
			{
	    	    // Partition design matrix into two sets of regressors, effects of interest and nuisance effects
	    	    int r = NUMBER_OF_CONTRASTS;
	    	    int p = NUMBER_OF_TOTAL_GLM_REGRESSORS;

	    	    Eigen::MatrixXd tmp(NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_TOTAL_GLM_REGRESSORS);
	    	    tmp = Eigen::MatrixXd::Identity(NUMBER_OF_TOTAL_GLM_REGRESSORS, NUMBER_OF_TOTAL_GLM_REGRESSORS) - Contrasts.transpose() * pinv(Contrasts.transpose());
   
	    	    Eigen::JacobiSVD<Eigen::MatrixXd> svd(tmp, Eigen::ComputeFullU | Eigen::ComputeFullV);
	    	    Eigen::MatrixXd c2 = svd.matrixU().block(0,0,NUMBER_OF_TOTAL_GLM_REGRESSORS,p-r);
	    	    Eigen::MatrixXd c3=c2.transpose();
    	
	    	    Eigen::MatrixXd C(NUMBER_OF_TOTAL_GLM_REGRESSORS,NUMBER_OF_TOTAL_GLM_REGRESSORS);
	    	    C.block(0,0,r,NUMBER_OF_TOTAL_GLM_REGRESSORS) = Contrasts;
	    	    C.block(r,0,p-r,NUMBER_OF_TOTAL_GLM_REGRESSORS) = c3;
       		
    		    Eigen::MatrixXd W = X_GLM * C.inverse();
    		    Eigen::MatrixXd W1 = W.block(0,0,NUMBER_OF_SUBJECTS,r);
    		    Eigen::MatrixXd W2 = W.block(0,r,NUMBER_OF_SUBJECTS,p-r);
		
				// Setup matrix that transforms the data vector in each voxel
				Eigen::MatrixXd transformationMatrix = Eigen::MatrixXd::Identity(NUMBER_OF_SUBJECTS, NUMBER_OF_SUBJECTS) - W2 * pinv(W2);
				Eigen::MatrixXf transformationMatrixf = transformationMatrix.cast<float>();
		
	    	    // Copy transformation matrix to constant memory
	    	    clEnqueueWriteBuffer(commandQueue, c_Transformation_Matrix, CL_TRUE, 0, NUMBER_OF_SUBJECTS * NUMBER_OF_SUBJECTS * sizeof(float), transformationMatrixf.data(), 0, NULL, NULL);
				clFinish(commandQueue);

				// Transform the data, only needed once since the permutations are done by permuting the design matrix
				runKernelErrorTransformData = clEnqueueNDRangeKernel(commandQueue, TransformDataKernel, 3, NULL, globalWorkSizeCalculateStatisticalMapsGLM, localWorkSizeCalculateStatisticalMapsGLM, 0, NULL, NULL);
			clFinish(commandQueue);
			}					
		}
        
		h_Permutation_Distribution = h_Permutation_Distributions[c];

        // Loop over all the permutations, save the maximum test value from each permutation
        for (size_t p = 0; p < NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c]; p++)
        {
            if ((WRAPPER == BASH) && PRINT && (p%100 == 0))
            {
                printf("Starting permutation %lu \n",p+1);
            }
   
            // Calculate statistical maps
            CalculateStatisticalMapsSecondLevelPermutation(p,c);
   
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
   
        std::vector<float> max_values (h_Permutation_Distribution, h_Permutation_Distribution + NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c]);
        std::sort (max_values.begin(), max_values.begin() + NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c]);
   
        // Find the threshold for the specified significance level
        SIGNIFICANCE_THRESHOLD = max_values[(int)(ceil((1.0f - SIGNIFICANCE_LEVEL) * (float)NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c]))-1];

        if (WRAPPER == BASH)
        {
			if (STATISTICAL_TEST == TTEST)
			{
	            printf("Permutation threshold for contrast %zu for a significance level of %f is %f \n",c+1,SIGNIFICANCE_LEVEL, SIGNIFICANCE_THRESHOLD);
			}
			else if (STATISTICAL_TEST == FTEST)
			{
	            printf("Permutation threshold for F-test for a significance level of %f is %f \n",SIGNIFICANCE_LEVEL, SIGNIFICANCE_THRESHOLD);
			}
			else if (STATISTICAL_TEST == GROUP_MEAN)
			{
	            printf("Permutation threshold for group mean for a significance level of %f is %f \n",SIGNIFICANCE_LEVEL, SIGNIFICANCE_THRESHOLD);
			}

        }
    }

    CleanupPermutationTestSecondLevel();
}


// Calculates permutation based p-values in each voxel
void BROCCOLI_LIB::CalculatePermutationPValues(cl_mem d_Mask, int DATA_W, int DATA_H, int DATA_D)
{
	SetGlobalAndLocalWorkSizesStatisticalCalculations(DATA_W, DATA_H, DATA_D);

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

	SetMemory(d_P_Values, 0.0f, DATA_W * DATA_H * DATA_D * NUMBER_OF_STATISTICAL_MAPS);

	// Loop over contrasts
	for (size_t contrast = 0; contrast < NUMBER_OF_STATISTICAL_MAPS; contrast++)
	{
		c_Permutation_Distribution = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_PERMUTATIONS_PER_CONTRAST[contrast] * sizeof(float), NULL, NULL);

		// Copy max values to constant memory
		clEnqueueWriteBuffer(commandQueue, c_Permutation_Distribution, CL_TRUE, 0, NUMBER_OF_PERMUTATIONS_PER_CONTRAST[contrast] * sizeof(float), h_Permutation_Distributions[contrast], 0, NULL, NULL);
		clFinish(commandQueue);

		ClusterizeOpenCL(d_Cluster_Indices, d_Cluster_Sizes, d_Statistical_Maps, CLUSTER_DEFINING_THRESHOLD, d_Mask, DATA_W, DATA_H, DATA_D, contrast);

		if ( (INFERENCE_MODE == VOXEL) || (INFERENCE_MODE == TFCE) )
		{
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 0, sizeof(cl_mem), &d_P_Values);
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 1, sizeof(cl_mem), &d_Statistical_Maps);
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 2, sizeof(cl_mem), &d_Mask);
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 3, sizeof(cl_mem), &c_Permutation_Distribution);
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 4, sizeof(int),    &contrast);
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 5, sizeof(int),    &DATA_W);
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 6, sizeof(int),    &DATA_H);
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 7, sizeof(int),    &DATA_D);
			clSetKernelArg(CalculatePermutationPValuesVoxelLevelInferenceKernel, 8, sizeof(int),    &NUMBER_OF_PERMUTATIONS_PER_CONTRAST[contrast]);
			runKernelErrorCalculatePermutationPValuesVoxelLevelInference = clEnqueueNDRangeKernel(commandQueue, CalculatePermutationPValuesVoxelLevelInferenceKernel, 3, NULL, globalWorkSizeCalculatePermutationPValues, localWorkSizeCalculatePermutationPValues, 0, NULL, NULL);
		}
		else if (INFERENCE_MODE == CLUSTER_EXTENT)
		{
			clSetKernelArg(CalculatePermutationPValuesClusterExtentInferenceKernel, 0, sizeof(cl_mem), &d_P_Values);
			clSetKernelArg(CalculatePermutationPValuesClusterExtentInferenceKernel, 1, sizeof(cl_mem), &d_Statistical_Maps);
			clSetKernelArg(CalculatePermutationPValuesClusterExtentInferenceKernel, 2, sizeof(cl_mem), &d_Cluster_Indices);
			clSetKernelArg(CalculatePermutationPValuesClusterExtentInferenceKernel, 3, sizeof(cl_mem), &d_Cluster_Sizes);
			clSetKernelArg(CalculatePermutationPValuesClusterExtentInferenceKernel, 4, sizeof(cl_mem), &d_Mask);
			clSetKernelArg(CalculatePermutationPValuesClusterExtentInferenceKernel, 5, sizeof(cl_mem), &c_Permutation_Distribution);
			clSetKernelArg(CalculatePermutationPValuesClusterExtentInferenceKernel, 6, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
			clSetKernelArg(CalculatePermutationPValuesClusterExtentInferenceKernel, 7, sizeof(int),    &contrast);
			clSetKernelArg(CalculatePermutationPValuesClusterExtentInferenceKernel, 8, sizeof(int),    &DATA_W);
			clSetKernelArg(CalculatePermutationPValuesClusterExtentInferenceKernel, 9, sizeof(int),    &DATA_H);
			clSetKernelArg(CalculatePermutationPValuesClusterExtentInferenceKernel, 10, sizeof(int),   &DATA_D);
			clSetKernelArg(CalculatePermutationPValuesClusterExtentInferenceKernel, 11, sizeof(int),   &NUMBER_OF_PERMUTATIONS_PER_CONTRAST[contrast]);
			runKernelErrorCalculatePermutationPValuesClusterExtentInference = clEnqueueNDRangeKernel(commandQueue, CalculatePermutationPValuesClusterExtentInferenceKernel, 3, NULL, globalWorkSizeCalculatePermutationPValues, localWorkSizeCalculatePermutationPValues, 0, NULL, NULL);
		}
		else if (INFERENCE_MODE == CLUSTER_MASS)
		{
			clSetKernelArg(CalculatePermutationPValuesClusterMassInferenceKernel, 0, sizeof(cl_mem), &d_P_Values);
			clSetKernelArg(CalculatePermutationPValuesClusterMassInferenceKernel, 1, sizeof(cl_mem), &d_Statistical_Maps);
			clSetKernelArg(CalculatePermutationPValuesClusterMassInferenceKernel, 2, sizeof(cl_mem), &d_Cluster_Indices);
			clSetKernelArg(CalculatePermutationPValuesClusterMassInferenceKernel, 3, sizeof(cl_mem), &d_Cluster_Sizes);
			clSetKernelArg(CalculatePermutationPValuesClusterMassInferenceKernel, 4, sizeof(cl_mem), &d_Mask);
			clSetKernelArg(CalculatePermutationPValuesClusterMassInferenceKernel, 5, sizeof(cl_mem), &c_Permutation_Distribution);
			clSetKernelArg(CalculatePermutationPValuesClusterMassInferenceKernel, 6, sizeof(float),  &CLUSTER_DEFINING_THRESHOLD);
			clSetKernelArg(CalculatePermutationPValuesClusterMassInferenceKernel, 7, sizeof(int),    &contrast);
			clSetKernelArg(CalculatePermutationPValuesClusterMassInferenceKernel, 8, sizeof(int),    &DATA_W);
			clSetKernelArg(CalculatePermutationPValuesClusterMassInferenceKernel, 9, sizeof(int),    &DATA_H);
			clSetKernelArg(CalculatePermutationPValuesClusterMassInferenceKernel, 10, sizeof(int),   &DATA_D);
			clSetKernelArg(CalculatePermutationPValuesClusterMassInferenceKernel, 11, sizeof(int),   &NUMBER_OF_PERMUTATIONS_PER_CONTRAST[contrast]);
			runKernelErrorCalculatePermutationPValuesClusterMassInference = clEnqueueNDRangeKernel(commandQueue, CalculatePermutationPValuesClusterMassInferenceKernel, 3, NULL, globalWorkSizeCalculatePermutationPValues, localWorkSizeCalculatePermutationPValues, 0, NULL, NULL);

		}

		clReleaseMemObject(c_Permutation_Distribution);
	}
}


// Generates a permutation matrix for a single subject
void BROCCOLI_LIB::GeneratePermutationMatrixFirstLevel()
{
	// Create random permutation vector
	std::vector<unsigned short int> perm;
	for (int i = 0; i < EPI_DATA_T; i++) 
	{
	    perm.push_back((unsigned short int)i);
	}
	std::vector< std::vector<unsigned short int> > allPermutations;
	allPermutations.push_back(perm);

    for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
    {
		while(true)
		{
			// Make random permutation
			std::random_shuffle(perm.begin(), perm.end());			

			// Check for repetitions
			bool unique = true;
			for (int r = 0; r < (p+1); r++)
			{
				// Same permutation found, break
				if (allPermutations[r] == perm)
				{
					unique = false;
					break;
				}
			} 
				
			// Break while loop when we have found a new unique permutation
			if (unique)
			{
				allPermutations.push_back(perm);
				break;
			}
		}
	
		// Put permutation vector into matrix,
        // all permutations are valid since we have whitened the data
        for (int i = 0; i < EPI_DATA_T; i++)
        {            
            h_Permutation_Matrix[i + p * EPI_DATA_T] = perm[i];
        }
    }
}

// Generates a permutation matrix for group analysis, two sample design
void BROCCOLI_LIB::GeneratePermutationMatrixSecondLevelTwoSample(int contrast)
{
	h_Permutation_Matrix = h_Permutation_Matrices[contrast];

	// Create random permutation vector
	std::vector<unsigned short int> group1Subjects;
	std::vector<unsigned short int> group2Subjects;
	std::vector<int> groups;
	for (int i = 0; i < NUMBER_OF_SUBJECTS_IN_GROUP1[contrast]; i++) 
	{
	    groups.push_back(1);
		group1Subjects.push_back((unsigned short int)i);
	}
	for (int i = 0; i < NUMBER_OF_SUBJECTS_IN_GROUP2[contrast]; i++) 
	{
	    groups.push_back(-1);
		group2Subjects.push_back((unsigned short int)(i+NUMBER_OF_SUBJECTS_IN_GROUP1[contrast]));
	}

	std::vector<std::vector<int> >allPermutations;
	allPermutations.push_back(groups);

	// Save permutation vector in big array
	int group1Subject = 0; int group2Subject = 0;
	for (int i = 0; i < NUMBER_OF_SUBJECTS; i++)
	{
		// Pick any subject from the first group
		if (groups[i] == 1)
		{
           	h_Permutation_Matrix[i] = (unsigned short int)group1Subjects[group1Subject];
			group1Subject++;
		}
		// Pick any subject from second group
		else if (groups[i] == -1)
		{
           	h_Permutation_Matrix[i] = (unsigned short int)group2Subjects[group2Subject];
			group2Subject++;
		}			
	}

	printf("Number of permutations is %zu \n",NUMBER_OF_PERMUTATIONS_PER_CONTRAST[contrast]);

	// Loop over all remaining permutations
	for (int p = 1; p < NUMBER_OF_PERMUTATIONS_PER_CONTRAST[contrast]; p++)
	{
		while(true)
		{
			// Make random permutation
			std::random_shuffle(groups.begin(), groups.end());			

			// Check for repetitions
			bool unique = true;
			for (int r = 0; r < allPermutations.size(); r++)
			{
				// Same permutation found, break
				if (allPermutations[r] == groups)
				{
					unique = false;
					break;
				}
			} 
				
			// Break while loop when we have found a new unique permutation
			if (unique)
			{
				allPermutations.push_back(groups);
				break;
			}
		}

		// Save permutation vector in big array
		int group1Subject = 0; int group2Subject = 0;
		for (int i = 0; i < NUMBER_OF_SUBJECTS; i++)
		{
			// Pick any subject from the first group
			if (groups[i] == 1)
			{
            	h_Permutation_Matrix[i + p * NUMBER_OF_SUBJECTS] = (unsigned short int)group1Subjects[group1Subject];
				group1Subject++;
			}
			// Pick any subject from second group
			else if (groups[i] == -1)
			{
            	h_Permutation_Matrix[i + p * NUMBER_OF_SUBJECTS] = (unsigned short int)group2Subjects[group2Subject];
				group2Subject++;
			}			
		}
	}
}

// Generates a permutation matrix for group analysis, correlation
void BROCCOLI_LIB::GeneratePermutationMatrixSecondLevelCorrelation(int contrast)
{
	h_Permutation_Matrix = h_Permutation_Matrices[contrast];

	// Create random permutation vector
	std::vector<unsigned short int> perm;
	for (int i = 0; i < NUMBER_OF_SUBJECTS; i++) 
	{
	    perm.push_back((unsigned short int)i);
	}
	std::vector< std::vector<unsigned short int> > allPermutations;
	allPermutations.push_back(perm);

	// Put permutation vector into matrix
    for (int i = 0; i < NUMBER_OF_SUBJECTS; i++)
    {            
		h_Permutation_Matrix[i] = perm[i];
	}

    for (int p = 1; p < NUMBER_OF_PERMUTATIONS_PER_CONTRAST[contrast]; p++)
    {
		while(true)
		{
			// Make random permutation
			std::random_shuffle(perm.begin(), perm.end());			

			// Check for repetitions
			bool unique = true;
			for (int r = 0; r < p; r++)
			{
				// Same permutation found, break
				if (allPermutations[r] == perm)
				{
					unique = false;
					break;
				}
			} 
				
			// Break while loop when we have found a new unique permutation
			if (unique)
			{
				allPermutations.push_back(perm);
				break;
			}
		}
	
		// Put permutation vector into matrix
        for (int i = 0; i < NUMBER_OF_SUBJECTS; i++)
        {            
            h_Permutation_Matrix[i + p * NUMBER_OF_SUBJECTS] = perm[i];
        }
    }
}

// Generates a sign flipping matrix for group analysis, one sample t-test
void BROCCOLI_LIB::GenerateSignMatrixSecondLevel()
{
    std::vector<int> flips;
    for (int i = 0; i < NUMBER_OF_SUBJECTS; i++)
    {
        flips.push_back(1);
    }
    std::vector< std::vector<int> > allSignFlips;
    allSignFlips.push_back(flips);

    // Put sign vector into matrix
    for (int i = 0; i < NUMBER_OF_SUBJECTS; i++)
    {
        h_Sign_Matrix[i] = flips[i];
    }
    
    for (int p = 1; p < NUMBER_OF_PERMUTATIONS_PER_CONTRAST[0]; p++)
    {
        while(true)
        {
            // Make random sign flips
            for (int i = 0; i < NUMBER_OF_SUBJECTS; i++)
            {
                // Multiply with 1 or -1
                flips[i] *= (2*(int)(rand() % 2) - 1);
            }
         			
            // Check for repetitions
            bool unique = true;
            for (int r = 0; r < p; r++)
            {
                // Same flips found, break
                if (allSignFlips[r] == flips)
                {
                    unique = false;
                    break;
                }
            }
        
            // Break while loop when we have found a new unique permutation
            if (unique)
            {
                allSignFlips.push_back(flips);
                break;
            }
        }
            
        // Put sign vector into matrix
        for (int i = 0; i < NUMBER_OF_SUBJECTS; i++)
        {
            h_Sign_Matrix[i + p * NUMBER_OF_SUBJECTS] = flips[i];
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







// Solves an equation system using QR-factorization, used by the linear registration algorithm
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

void BROCCOLI_LIB::DemeanRegressor(Eigen::VectorXf& Regressor, int N)
{
	float mean = 0.0;
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
Eigen::MatrixXd BROCCOLI_LIB::SetupGLMRegressorsFirstLevel()
{
	// Calculate total number of regressors
	if (!RAW_DESIGNMATRIX)
	{
		NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + REGRESS_GLOBALMEAN + NUMBER_OF_CONFOUND_REGRESSORS*REGRESS_CONFOUNDS;
	}
	else
	{
		NUMBER_OF_TOTAL_GLM_REGRESSORS = NUMBER_OF_GLM_REGRESSORS + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + REGRESS_GLOBALMEAN;
	}

	std::vector<Eigen::VectorXd> allOneRegressors;
	std::vector<Eigen::VectorXd> allLinearRegressors;
	std::vector<Eigen::VectorXd> allQuadraticRegressors;
	std::vector<Eigen::VectorXd> allCubicRegressors;

	bool meanRegressor[NUMBER_OF_TOTAL_GLM_REGRESSORS];
	bool detrendingRegressor[NUMBER_OF_TOTAL_GLM_REGRESSORS];
	int  detrendingRegressorRun[NUMBER_OF_TOTAL_GLM_REGRESSORS];
	int totalTRs[NUMBER_OF_RUNS];

	for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
	{
		meanRegressor[r] = false;
		detrendingRegressor[r] = false;
		detrendingRegressorRun[r] = 0;
	}

	totalTRs[0] = 0;
	for (size_t run = 1; run < NUMBER_OF_RUNS; run++)
	{
		totalTRs[run] = totalTRs[run-1] + EPI_DATA_T_PER_RUN[run-1];
	}

	// Create detrending regressors
	for (size_t run = 0; run < NUMBER_OF_RUNS; run++)
	{
		int N = EPI_DATA_T_PER_RUN[run];

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
		double minn = std::abs(Quadratic.minCoeff());
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

		allOneRegressors.push_back(Ones);
		allLinearRegressors.push_back(Linear);
		allQuadraticRegressors.push_back(Quadratic);
		allCubicRegressors.push_back(Cubic);
	}


	if (!REGRESS_ONLY)
	{
		// Create temporal derivatives if requested and then convolve all regressors with HRF
		if (USE_TEMPORAL_DERIVATIVES && !RAW_REGRESSORS && !RAW_DESIGNMATRIX)
		{
			GenerateRegressorTemporalDerivatives(h_X_GLM_With_Temporal_Derivatives, h_X_GLM_In, EPI_DATA_T, NUMBER_OF_GLM_REGRESSORS);
			ConvolveRegressorsWithHRF(h_X_GLM_Convolved, h_X_GLM_With_Temporal_Derivatives, EPI_DATA_T, NUMBER_OF_GLM_REGRESSORS*2);
		}
		// Convolve regressors with HRF
		else if (!RAW_REGRESSORS && !RAW_DESIGNMATRIX)
		{
			ConvolveRegressorsWithHRF(h_X_GLM_Convolved, h_X_GLM_In, EPI_DATA_T, NUMBER_OF_GLM_REGRESSORS);
		}
		// Just copy raw regressors
		else if (RAW_REGRESSORS || RAW_DESIGNMATRIX)
		{
			// Loop over samples
			for (int i = 0; i < EPI_DATA_T; i++)
			{
				// Loop over regressors
				for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
				{
					h_X_GLM_Convolved[i + r * EPI_DATA_T] = h_X_GLM_In[i + r * EPI_DATA_T];
				}
			}
		}
	}

	// Setup total design matrix
	Eigen::MatrixXd X(EPI_DATA_T,NUMBER_OF_TOTAL_GLM_REGRESSORS);
	for (int i = 0; i < EPI_DATA_T; i++)
	{
		for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
		{
			X(i,r) = 0.0;		
		}	
	}

	// Detrending regressors
	size_t accumulatedTRs = 0;
	for (int run = 0; run < NUMBER_OF_RUNS; run++)
	{
		Eigen::VectorXd Ones = allOneRegressors[run];
		Eigen::VectorXd Linear = allLinearRegressors[run];
		Eigen::VectorXd Quadratic = allQuadraticRegressors[run];
		Eigen::VectorXd Cubic = allCubicRegressors[run];

		for (int i = 0; i < EPI_DATA_T_PER_RUN[run]; i++)
		{	
			X(i+accumulatedTRs,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 0) = Ones(i);

			if (NUMBER_OF_DETRENDING_REGRESSORS >= 2)
			{
				X(i+accumulatedTRs,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 1) = Linear(i);
			}
			if (NUMBER_OF_DETRENDING_REGRESSORS >= 3)
			{
				X(i+accumulatedTRs,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 2) = Quadratic(i);
			}
			if (NUMBER_OF_DETRENDING_REGRESSORS == 4)
			{
				X(i+accumulatedTRs,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 3) = Cubic(i);
			}
		}
		accumulatedTRs += EPI_DATA_T_PER_RUN[run];

		meanRegressor[NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 0] = true;
		if (NUMBER_OF_DETRENDING_REGRESSORS >= 2)
		{
			detrendingRegressor[NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 1] = true;
			detrendingRegressorRun[NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 1] = run;
		}
		if (NUMBER_OF_DETRENDING_REGRESSORS >= 3)
		{
			detrendingRegressor[NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 2] = true;
			detrendingRegressorRun[NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 2] = run;
		}
		if (NUMBER_OF_DETRENDING_REGRESSORS == 4)
		{
			detrendingRegressor[NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 3] = true;
			detrendingRegressorRun[NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 3] = run;
		}						
	}

	// Loop over samples
	for (int i = 0; i < EPI_DATA_T; i++)
	{
		// Regressors for paradigms (number of regressors is 0 for regressonly)
		for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1); r++)
		{
			X(i,r) = (double)h_X_GLM_Convolved[i + r * EPI_DATA_T];
		}

		if (REGRESS_MOTION)
		{
			// Motion regressors
			X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 0) = h_Motion_Parameters[i + 0 * EPI_DATA_T];
			X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 1) = h_Motion_Parameters[i + 1 * EPI_DATA_T];
			X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 2) = h_Motion_Parameters[i + 2 * EPI_DATA_T];
			X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 3) = h_Motion_Parameters[i + 3 * EPI_DATA_T];
			X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 4) = h_Motion_Parameters[i + 4 * EPI_DATA_T];
			X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 5) = h_Motion_Parameters[i + 5 * EPI_DATA_T];
		}

		if (REGRESS_GLOBALMEAN)
		{
			X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION) = (double)h_Global_Mean[i];
		}

		if (REGRESS_CONFOUNDS && !RAW_DESIGNMATRIX)
		{
			// Confounding regressors
			for (int r = 0; r < NUMBER_OF_CONFOUND_REGRESSORS; r++)
			{
				X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + REGRESS_GLOBALMEAN + r) = (double)h_X_GLM_Confounds[i + r * EPI_DATA_T];
			}
		}
	}

	// Calculate which regressor contains only ones
	int MEAN_REGRESSOR;

	if (NUMBER_OF_RUNS == 1)
	{
		if (!RAW_DESIGNMATRIX)
		{
			MEAN_REGRESSOR = NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1);
		}
		else
		{
			MEAN_REGRESSOR = NUMBER_OF_GLM_REGRESSORS;
		}
	
		// Demean regressors
		for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
		{
			if (r != MEAN_REGRESSOR)
			{
				Eigen::VectorXd regressor = X.block(0,r,EPI_DATA_T,1);
				DemeanRegressor(regressor,EPI_DATA_T);
				X.block(0,r,EPI_DATA_T,1) = regressor;
			}
		}
	}
	else
	{
		// Demean regressors
		for (int r = 0; r < NUMBER_OF_TOTAL_GLM_REGRESSORS; r++)
		{
			if (detrendingRegressor[r])
			{
				int run = detrendingRegressorRun[r];
				Eigen::VectorXd regressor = X.block(totalTRs[run],r,EPI_DATA_T_PER_RUN[run],1);
				DemeanRegressor(regressor,EPI_DATA_T_PER_RUN[run]);
				X.block(totalTRs[run],r,EPI_DATA_T_PER_RUN[run],1) = regressor;
			}
			else if (!meanRegressor[r])
			{
				Eigen::VectorXd regressor = X.block(0,r,EPI_DATA_T,1);
				DemeanRegressor(regressor,EPI_DATA_T);
				X.block(0,r,EPI_DATA_T,1) = regressor;
			}
		}

	}

	// Calculate pseudo inverse (could be done with SVD instead, or QR)
	Eigen::MatrixXd xtx(NUMBER_OF_TOTAL_GLM_REGRESSORS,NUMBER_OF_TOTAL_GLM_REGRESSORS);
	xtx = X.transpose() * X;
	Eigen::MatrixXd inv_xtx = xtx.inverse();
	Eigen::MatrixXd xtxxt = inv_xtx * X.transpose();

	// Finally store regressors in ordinary arrays
	for (int i = 0; i < EPI_DATA_T; i++)
	{
		// Regressors for paradigms (number of regressors is 0 for regressonly)
		for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1); r++)
		{
			h_X_GLM[i + r * EPI_DATA_T] = X(i,r);
		}

		// Detrending regressors
		for (int run = 0; run < NUMBER_OF_RUNS; run++)
		{
			h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 0) * EPI_DATA_T] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 0);
			if (NUMBER_OF_DETRENDING_REGRESSORS >= 2)
			{
				h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 1) * EPI_DATA_T] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 1);
			}
			if (NUMBER_OF_DETRENDING_REGRESSORS >= 3)
			{
				h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 2) * EPI_DATA_T] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 2);
			}
			if (NUMBER_OF_DETRENDING_REGRESSORS == 4)
			{
				h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 3) * EPI_DATA_T] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 3);
			}
		}

		if (REGRESS_MOTION)
		{
			// Motion regressors
			h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 0) * EPI_DATA_T] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 0);
			h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 1) * EPI_DATA_T] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 1);
			h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 2) * EPI_DATA_T] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 2);
			h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 3) * EPI_DATA_T] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 3);
			h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 4) * EPI_DATA_T] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 4);
			h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 5) * EPI_DATA_T] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 5);
		}

		if (REGRESS_GLOBALMEAN)
		{
			h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION) * EPI_DATA_T] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION);
		}

		if (REGRESS_CONFOUNDS && !RAW_DESIGNMATRIX)
		{
			for (int r = 0; r < NUMBER_OF_CONFOUND_REGRESSORS; r++)
			{
				h_X_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + REGRESS_GLOBALMEAN + r) * EPI_DATA_T] = (float)X(i,NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + REGRESS_GLOBALMEAN + r);
			}
		}

		// Regressors for paradigms (number of regressors is 0 for regressonly)
		for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1); r++)
		{
			h_xtxxt_GLM[i + r * EPI_DATA_T] = (float)xtxxt(r,i);
		}

		// Detrending regressors
		for (int run = 0; run < NUMBER_OF_RUNS; run++)
		{
			h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 0) * EPI_DATA_T] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 0,i);
			if (NUMBER_OF_DETRENDING_REGRESSORS >= 2)
			{
				h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 1) * EPI_DATA_T] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 1,i);
			}
			if (NUMBER_OF_DETRENDING_REGRESSORS >= 3)
			{
				h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 2) * EPI_DATA_T] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 2,i);
			}
			if (NUMBER_OF_DETRENDING_REGRESSORS == 4)
			{
				h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 3) * EPI_DATA_T] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + run*NUMBER_OF_DETRENDING_REGRESSORS + 3,i);
			}
		}

		if (REGRESS_MOTION)
		{
			// Motion regressors
			h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 0) * EPI_DATA_T] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 0,i);
			h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 1) * EPI_DATA_T] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 1,i);
			h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 2) * EPI_DATA_T] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 2,i);
			h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 3) * EPI_DATA_T] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 3,i);
			h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 4) * EPI_DATA_T] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 4,i);
			h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 5) * EPI_DATA_T] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + 5,i);
		}

		if (REGRESS_GLOBALMEAN)
		{
			h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION) * EPI_DATA_T] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION,i);
		}

		if (REGRESS_CONFOUNDS && !RAW_DESIGNMATRIX)
		{
			// Confounding regressors
			for (int r = 0; r < NUMBER_OF_CONFOUND_REGRESSORS; r++)
			{
				h_xtxxt_GLM[i + (NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + REGRESS_GLOBALMEAN + r) * EPI_DATA_T] = (float)xtxxt(NUMBER_OF_GLM_REGRESSORS*(USE_TEMPORAL_DERIVATIVES+1) + NUMBER_OF_DETRENDING_REGRESSORS*NUMBER_OF_RUNS + NUMBER_OF_MOTION_REGRESSORS*REGRESS_MOTION + REGRESS_GLOBALMEAN + r,i);
			}
		}
	}

	return inv_xtx;
}

// Setup variables for a t-test for first level analysis
void BROCCOLI_LIB::SetupTTestFirstLevel()
{
	// Setup GLM regressors
	Eigen::MatrixXd inv_xtx = SetupGLMRegressorsFirstLevel();

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
void BROCCOLI_LIB::SetupFTestFirstLevel()
{
	// Setup GLM regressors
	Eigen::MatrixXd inv_xtx = SetupGLMRegressorsFirstLevel();

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
				Contrasts(c,rr) = (double)h_Contrasts_In[NUMBER_OF_GLM_REGRESSORS * c + r];
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
				Contrasts(c,r) = (double)h_Contrasts_In[NUMBER_OF_GLM_REGRESSORS * c + r];
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
		                      size_t DATA_W,
		                      size_t DATA_H,
		                      size_t DATA_D,
		                      int GET_VOXEL_LABELS,
		                      int GET_CLUSTER_MASS)
{
	// Vector of clusters
	std::vector<std::vector<Coords3D> > clusters;

	// Keep track of labelled voxels
	int* ccMask = (int*)malloc(DATA_W * DATA_H * DATA_D * sizeof(int));
	for (size_t i = 0; i < (DATA_W * DATA_H * DATA_D); ++i)
	{
		ccMask[i] = 0;
		Cluster_Indices[i] = 0;
	}

	// Loop over volume voxels
	for (size_t z = 0; z < DATA_D; ++z)
	{
		for (size_t y = 0; y < DATA_H; ++y)
		{
			for (size_t x = 0; x < DATA_W; ++x)
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
	for (size_t cluster = 0; cluster < NUMBER_OF_CLUSTERS; cluster++)
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
			for (size_t voxel = 0; voxel < cluster_size; voxel++)
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
			for (size_t voxel = 0; voxel < cluster_size; voxel++)
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
		                            cl_mem d_Data,
		                            float Threshold,
		                            cl_mem d_Mask,
		                            int DATA_W,
		                            int DATA_H,
		                            int DATA_D,
									int contrast)
{
	SetGlobalAndLocalWorkSizesClusterize(DATA_W, DATA_H, DATA_D);

	cl_mem d_Updated = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);

	clSetKernelArg(SetStartClusterIndicesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(SetStartClusterIndicesKernel, 1, sizeof(cl_mem), &d_Data);
	clSetKernelArg(SetStartClusterIndicesKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(SetStartClusterIndicesKernel, 3, sizeof(float),  &Threshold);
	clSetKernelArg(SetStartClusterIndicesKernel, 4, sizeof(int),    &contrast);
	clSetKernelArg(SetStartClusterIndicesKernel, 5, sizeof(int),    &DATA_W);
	clSetKernelArg(SetStartClusterIndicesKernel, 6, sizeof(int),    &DATA_H);
	clSetKernelArg(SetStartClusterIndicesKernel, 7, sizeof(int),    &DATA_D);

	clSetKernelArg(ClusterizeScanKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeScanKernel, 1, sizeof(cl_mem), &d_Updated);
	clSetKernelArg(ClusterizeScanKernel, 2, sizeof(cl_mem), &d_Data);
	clSetKernelArg(ClusterizeScanKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(ClusterizeScanKernel, 4, sizeof(float),  &Threshold);
	clSetKernelArg(ClusterizeScanKernel, 5, sizeof(int),    &contrast);
	clSetKernelArg(ClusterizeScanKernel, 6, sizeof(int),    &DATA_W);
	clSetKernelArg(ClusterizeScanKernel, 7, sizeof(int),    &DATA_H);
	clSetKernelArg(ClusterizeScanKernel, 8, sizeof(int),    &DATA_D);

	clSetKernelArg(ClusterizeRelabelKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(ClusterizeRelabelKernel, 1, sizeof(cl_mem), &d_Data);
	clSetKernelArg(ClusterizeRelabelKernel, 2, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(ClusterizeRelabelKernel, 3, sizeof(float),  &Threshold);
	clSetKernelArg(ClusterizeRelabelKernel, 4, sizeof(int),    &contrast);
	clSetKernelArg(ClusterizeRelabelKernel, 5, sizeof(int),    &DATA_W);
	clSetKernelArg(ClusterizeRelabelKernel, 6, sizeof(int),    &DATA_H);
	clSetKernelArg(ClusterizeRelabelKernel, 7, sizeof(int),    &DATA_D);

	clSetKernelArg(CalculateClusterSizesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(CalculateClusterSizesKernel, 1, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateClusterSizesKernel, 2, sizeof(cl_mem), &d_Data);
	clSetKernelArg(CalculateClusterSizesKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateClusterSizesKernel, 4, sizeof(float),  &Threshold);
	clSetKernelArg(CalculateClusterSizesKernel, 5, sizeof(int),    &contrast);
	clSetKernelArg(CalculateClusterSizesKernel, 6, sizeof(int),    &DATA_W);
	clSetKernelArg(CalculateClusterSizesKernel, 7, sizeof(int),    &DATA_H);
	clSetKernelArg(CalculateClusterSizesKernel, 8, sizeof(int),    &DATA_D);

	clSetKernelArg(CalculateClusterMassesKernel, 0, sizeof(cl_mem), &d_Cluster_Indices);
	clSetKernelArg(CalculateClusterMassesKernel, 1, sizeof(cl_mem), &d_Cluster_Sizes);
	clSetKernelArg(CalculateClusterMassesKernel, 2, sizeof(cl_mem), &d_Data);
	clSetKernelArg(CalculateClusterMassesKernel, 3, sizeof(cl_mem), &d_Mask);
	clSetKernelArg(CalculateClusterMassesKernel, 4, sizeof(float),  &Threshold);
	clSetKernelArg(CalculateClusterMassesKernel, 5, sizeof(int),    &contrast);
	clSetKernelArg(CalculateClusterMassesKernel, 6, sizeof(int),    &DATA_W);
	clSetKernelArg(CalculateClusterMassesKernel, 7, sizeof(int),    &DATA_H);
	clSetKernelArg(CalculateClusterMassesKernel, 8, sizeof(int),    &DATA_D);

	SetMemoryInt(d_Cluster_Sizes, 0, DATA_W * DATA_H * DATA_D);
	SetMemoryInt(d_Cluster_Indices, 0, DATA_W * DATA_H * DATA_D);

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

	clReleaseMemObject(d_Updated);
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
		clFinish(commandQueue);
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
	clFinish(commandQueue);

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



// Small help functions

int BROCCOLI_LIB::CalculateMax(int *data, size_t N)
{
    int max = std::numeric_limits<int>::min();
	for (size_t i = 0; i < N; i++)
	{
	    if (data[i] > max)
		{
			max = data[i];
		}
	}
	return max;
}

float BROCCOLI_LIB::CalculateMax(float *data, size_t N)
{
    float max = std::numeric_limits<float>::min();
	for (size_t i = 0; i < N; i++)
	{
	    if (data[i] > max)
		{
			max = data[i];
		}
	}
	return max;
}

float BROCCOLI_LIB::CalculateMin(float *data, size_t N)
{
    float min = std::numeric_limits<float>::max();
	for (size_t i = 0; i < N; i++)
	{
	    if (data[i] < min)
		{
			min = data[i];
		}
	}
	return min;
}

void BROCCOLI_LIB::ResetEigenMatrix(Eigen::MatrixXd & matrix)
{
	int NUMBER_OF_COLUMNS = matrix.cols();
	int NUMBER_OF_ROWS = matrix.rows();

	for (int r = 0; r < NUMBER_OF_ROWS; r++)
	{	
		for (int c = 0; c < NUMBER_OF_COLUMNS; c++)
		{
			matrix(r,c) = 0.0;
		}
	}
}

void BROCCOLI_LIB::ResetEigenMatrix(Eigen::MatrixXf & matrix)
{
	int NUMBER_OF_COLUMNS = matrix.cols();
	int NUMBER_OF_ROWS = matrix.rows();

	for (int r = 0; r < NUMBER_OF_ROWS; r++)
	{	
		for (int c = 0; c < NUMBER_OF_COLUMNS; c++)
		{
			matrix(r,c) = 0.0f;
		}
	}
}

void BROCCOLI_LIB::IdentityEigenMatrix(Eigen::MatrixXd & matrix)
{
	int NUMBER_OF_COLUMNS = matrix.cols();
	int NUMBER_OF_ROWS = matrix.rows();

	for (int r = 0; r < NUMBER_OF_ROWS; r++)
	{	
		for (int c = 0; c < NUMBER_OF_COLUMNS; c++)
		{
			if (c == r)
			{
				matrix(r,c) = 1.0;
			}
			else
			{
				matrix(r,c) = 0.0;
			}
		}
	}
}

void BROCCOLI_LIB::IdentityEigenMatrix(Eigen::MatrixXf & matrix)
{
	int NUMBER_OF_COLUMNS = matrix.cols();
	int NUMBER_OF_ROWS = matrix.rows();

	for (int r = 0; r < NUMBER_OF_ROWS; r++)
	{	
		for (int c = 0; c < NUMBER_OF_COLUMNS; c++)
		{
			if (c == r)
			{
				matrix(r,c) = 1.0f;
			}
			else
			{
				matrix(r,c) = 0.0f;
			}
		}
	}
}

void BROCCOLI_LIB::PCAWhitenEigen(Eigen::MatrixXd & whitenedData,  Eigen::MatrixXd & inputData, int NUMBER_OF_COMPONENTS, bool demean)
{
	// inputData, NUMBER_OF_OBSERVATIONS x NUMBER_OF_VOXELS
	// whitenedData, NUMBER_OF_COMPONENTS x NUMBER_OF_VOXELS

	size_t NUMBER_OF_VOXELS = inputData.cols();
	size_t NUMBER_OF_OBSERVATIONS = inputData.rows();

	printf("Input data matrix size is %li x %li \n",inputData.rows(),inputData.cols());

	if (WRAPPER == BASH)
	{
		printf("Demeaning data\n");
	}

	if (demean)
	{
		#pragma omp parallel for
		for (size_t voxel = 0; voxel < NUMBER_OF_VOXELS; voxel++)
		{
			//printf("Demeaning data for voxel %i\n",voxel);
			Eigen::VectorXd values = inputData.block(0,voxel,NUMBER_OF_OBSERVATIONS,1);
			DemeanRegressor(values,NUMBER_OF_OBSERVATIONS);
			inputData.block(0,voxel,NUMBER_OF_OBSERVATIONS,1) = values;
		}
	}

	// Calculate covariance Matrix
	Eigen::MatrixXd covarianceMatrix(NUMBER_OF_OBSERVATIONS,NUMBER_OF_OBSERVATIONS);
	ResetEigenMatrix(covarianceMatrix);
	if (WRAPPER == BASH)
	{
		printf("Estimating the covariance matrix\n");
	}

	//for (int voxel = 0; voxel < NUMBER_OF_VOXELS; voxel++)
	//{
	//	Eigen::VectorXd values = inputData.col(voxel);
	//	covarianceMatrix += values * values.transpose();
	//}
	covarianceMatrix = inputData * inputData.transpose();
	covarianceMatrix *= 1.0/(double)(NUMBER_OF_VOXELS - 1);
	
	//std::cout << covarianceMatrix << std::endl;

	// Calculate eigen values of covariance matrix	
	if (WRAPPER == BASH)
	{
		printf("Calculating eigen values\n");
	}
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(covarianceMatrix);
	Eigen::VectorXd eigenValues = es.eigenvalues();
	Eigen::MatrixXd eigenVectors = es.eigenvectors();
	
	Eigen::VectorXd savedEigenValues(NUMBER_OF_COMPONENTS);
	Eigen::MatrixXd savedEigenVectors(NUMBER_OF_COMPONENTS,NUMBER_OF_OBSERVATIONS);
	
	double totalVariance = 0.0;
	for (int i = 0; i < NUMBER_OF_OBSERVATIONS; i++)
	{
		printf("Eigen value %i is %f\n",i,(float)eigenValues(i));
		totalVariance += eigenValues(i);
	}

	// Get a sub matrix of all the eigen vectors, to remove the smallest ones
	// Get a sub vector of the eigen values, to remove the smallest eigen values
	double savedVariance = 0.0;
	for (int i = 0; i < NUMBER_OF_COMPONENTS; i++)
	{	
		// Find largest eigen value for current component, and it's location
		int index = 0; 
		double largestEigenValue = eigenValues.maxCoeff(&index);
		savedEigenValues(i) = largestEigenValue;
		savedVariance += largestEigenValue;

		printf("Largest eigen value is %f \n",(float)largestEigenValue);

		// Get the corresponding eigen vector
		savedEigenVectors.row(i) = eigenVectors.col(index).transpose();

		// Set the previous largest eigen value to 0
		eigenValues(index) = 0.0;
	}

	if ((WRAPPER == BASH) && VERBOSE)
	{
		printf("Saved %f %% of the total variance at the dimensionality reduction\n",(float)savedVariance/(float)totalVariance*100.0);
	}

	// Calculate  ^(-1/2) for all saved eigen values
	Eigen::VectorXd scaledEigenValues(NUMBER_OF_COMPONENTS);
	for (int i = 0; i < NUMBER_OF_COMPONENTS; i++)
	{	
		double eigenValue = savedEigenValues(i);
		scaledEigenValues(i) = 1.0/sqrt(eigenValue);
	}

	// Calculate whitening matrix
	Eigen::MatrixXd whiteningMatrix = scaledEigenValues.asDiagonal() * savedEigenVectors;

	// Perform the actual whitening
	if (WRAPPER == BASH)
	{
		printf("Applying dimensionality reduction and whitening\n");
	}
	whitenedData = whiteningMatrix * inputData;
}



// Saves a certain percentage of the variance, instead of a fix number of components
Eigen::MatrixXd BROCCOLI_LIB::PCAWhitenEigen(Eigen::MatrixXd & inputData, bool demean)
{
	// inputData, NUMBER_OF_OBSERVATIONS x NUMBER_OF_VOXELS
	// whitenedData, NUMBER_OF_COMPONENTS x NUMBER_OF_VOXELS

	size_t NUMBER_OF_VOXELS = inputData.cols();
	size_t NUMBER_OF_OBSERVATIONS = inputData.rows();

	printf("Input data matrix size is %li x %li \n",inputData.rows(),inputData.cols());

	if (demean)
	{
		if (WRAPPER == BASH)
		{	
			printf("Demeaning data\n");
		}
		#pragma omp parallel for
		for (size_t voxel = 0; voxel < NUMBER_OF_VOXELS; voxel++)
		{
			//printf("Demeaning data for voxel %i\n",voxel);
			Eigen::VectorXd values = inputData.block(0,voxel,NUMBER_OF_OBSERVATIONS,1);
			DemeanRegressor(values,NUMBER_OF_OBSERVATIONS);
			inputData.block(0,voxel,NUMBER_OF_OBSERVATIONS,1) = values;
		}
	}

	// Calculate covariance Matrix
	if (WRAPPER == BASH)
	{
		printf("Estimating the covariance matrix\n");
	}

	double startTime = GetTime();
	Eigen::MatrixXd covarianceMatrix = inputData * inputData.transpose();
	covarianceMatrix *= 1.0/(double)(NUMBER_OF_VOXELS - 1);	
	double endTime = GetTime();
	if ((WRAPPER == BASH) && VERBOS)
	{
		printf("It took %f seconds to calculate the covariance matrix using Eigen\n",(float)(endTime - startTime));
	}

	// Calculate eigen values of covariance matrix	
	if (WRAPPER == BASH)
	{
		printf("Calculating eigen values\n");
	}
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(covarianceMatrix);
	Eigen::VectorXd eigenValues = es.eigenvalues();
	Eigen::MatrixXd eigenVectors = es.eigenvectors();
	
	double totalVariance = 0.0;
	for (int i = 0; i < NUMBER_OF_OBSERVATIONS; i++)
	{
		totalVariance += eigenValues(i);
	}

	// Calculate number of components to save
	double savedVariance = 0.0;
	Eigen::VectorXd temp = eigenValues;
	NUMBER_OF_ICA_COMPONENTS = 0;
	while (savedVariance/totalVariance*100.0 < (double)PROPORTION_OF_VARIANCE_TO_SAVE_BEFORE_ICA )
	{
		NUMBER_OF_ICA_COMPONENTS++;
		int index = 0;
		double largestEigenValue = temp.maxCoeff(&index);
		savedVariance += largestEigenValue;
		temp(index) = 0.0;
	}

	Eigen::VectorXd savedEigenValues(NUMBER_OF_ICA_COMPONENTS);
	Eigen::MatrixXd savedEigenVectors(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_OBSERVATIONS);

	// Get a sub matrix of all the eigen vectors, to remove the smallest ones
	// Get a sub vector of the eigen values, to remove the smallest eigen values
	for (int i = 0; i < NUMBER_OF_ICA_COMPONENTS; i++)
	{	
		// Find largest eigen value for current component, and it's location
		int index = 0; 
		double largestEigenValue = eigenValues.maxCoeff(&index);
		savedEigenValues(i) = largestEigenValue;

		printf("Largest eigen value is %f \n",(float)largestEigenValue);

		// Get the corresponding eigen vector
		savedEigenVectors.row(i) = eigenVectors.col(index).transpose();

		// Set the previous largest eigen value to 0
		eigenValues(index) = 0.0;
	}

	if ((WRAPPER == BASH) && VERBOSE)
	{
		printf("Saved %f %% of the total variance during the dimensionality reduction, using %zu components\n",(float)savedVariance/(float)totalVariance*100.0,NUMBER_OF_ICA_COMPONENTS);
	}

	// Calculate  ^(-1/2) for all saved eigen values
	Eigen::VectorXd scaledEigenValues(NUMBER_OF_ICA_COMPONENTS);
	for (int i = 0; i < NUMBER_OF_ICA_COMPONENTS; i++)
	{	
		double eigenValue = savedEigenValues(i);
		scaledEigenValues(i) = 1.0/sqrt(eigenValue);
	}

	// Calculate whitening matrix
	Eigen::MatrixXd whiteningMatrix = scaledEigenValues.asDiagonal() * savedEigenVectors;

	// Perform the actual whitening
	if (WRAPPER == BASH)
	{
		printf("Applying dimensionality reduction and whitening\n");
	}
	Eigen::MatrixXd whitenedData = whiteningMatrix * inputData;
	
	return whitenedData;
}


// Saves a certain percentage of the variance, instead of a fix number of components
Eigen::MatrixXf BROCCOLI_LIB::PCAWhitenEigen(Eigen::MatrixXf & inputData, bool demean)
{
	// inputData, NUMBER_OF_OBSERVATIONS x NUMBER_OF_VOXELS
	// whitenedData, NUMBER_OF_COMPONENTS x NUMBER_OF_VOXELS

	size_t NUMBER_OF_VOXELS = inputData.cols();
	size_t NUMBER_OF_OBSERVATIONS = inputData.rows();

	printf("Input data matrix size is %li x %li \n",inputData.rows(),inputData.cols());

	if (demean)
	{
		if (WRAPPER == BASH)
		{	
			printf("Demeaning data\n");
		}
		#pragma omp parallel for
		for (size_t voxel = 0; voxel < NUMBER_OF_VOXELS; voxel++)
		{
			//printf("Demeaning data for voxel %i\n",voxel);
			Eigen::VectorXf values = inputData.block(0,voxel,NUMBER_OF_OBSERVATIONS,1);
			DemeanRegressor(values,NUMBER_OF_OBSERVATIONS);
			inputData.block(0,voxel,NUMBER_OF_OBSERVATIONS,1) = values;
		}
	}

	// Calculate covariance Matrix	
	if (WRAPPER == BASH)
	{
		printf("Estimating the covariance matrix\n");
	}

	double startTime = GetTime();
	Eigen::MatrixXf covarianceMatrix = inputData * inputData.transpose();
	covarianceMatrix *= 1.0/(float)(NUMBER_OF_VOXELS - 1);	
	double endTime = GetTime();
	if ((WRAPPER == BASH) && VERBOS)
	{
		printf("It took %f seconds to calculate the covariance matrix using Eigen\n",(float)(endTime - startTime));
	}

	// Calculate eigen values of covariance matrix	
	if (WRAPPER == BASH)
	{
		printf("Calculating eigen values\n");
	}
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(covarianceMatrix);
	Eigen::VectorXf eigenValues = es.eigenvalues();
	Eigen::MatrixXf eigenVectors = es.eigenvectors();
	
	float totalVariance = 0.0f;
	for (int i = 0; i < NUMBER_OF_OBSERVATIONS; i++)
	{
		totalVariance += eigenValues(i);
	}

	// Calculate number of components to save
	float savedVariance = 0.0;
	Eigen::VectorXf temp = eigenValues;
	NUMBER_OF_ICA_COMPONENTS = 0;
	while (savedVariance/totalVariance*100.0 < (double)PROPORTION_OF_VARIANCE_TO_SAVE_BEFORE_ICA )
	{
		NUMBER_OF_ICA_COMPONENTS++;
		int index = 0;
		float largestEigenValue = temp.maxCoeff(&index);
		savedVariance += largestEigenValue;
		temp(index) = 0.0f;
	}

	Eigen::VectorXf savedEigenValues(NUMBER_OF_ICA_COMPONENTS);
	Eigen::MatrixXf savedEigenVectors(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_OBSERVATIONS);

	// Get a sub matrix of all the eigen vectors, to remove the smallest ones
	// Get a sub vector of the eigen values, to remove the smallest eigen values
	for (int i = 0; i < NUMBER_OF_ICA_COMPONENTS; i++)
	{	
		// Find largest eigen value for current component, and it's location
		int index = 0; 
		float largestEigenValue = eigenValues.maxCoeff(&index);
		savedEigenValues(i) = largestEigenValue;

		printf("Largest eigen value is %f \n",(float)largestEigenValue);

		// Get the corresponding eigen vector
		savedEigenVectors.row(i) = eigenVectors.col(index).transpose();

		// Set the previous largest eigen value to 0
		eigenValues(index) = 0.0f;
	}

	if ((WRAPPER == BASH) && VERBOSE)
	{
		printf("Saved %f %% of the total variance during the dimensionality reduction, using %zu components\n",(float)savedVariance/(float)totalVariance*100.0,NUMBER_OF_ICA_COMPONENTS);
	}

	// Calculate  ^(-1/2) for all saved eigen values
	Eigen::VectorXf scaledEigenValues(NUMBER_OF_ICA_COMPONENTS);
	for (int i = 0; i < NUMBER_OF_ICA_COMPONENTS; i++)
	{	
		float eigenValue = savedEigenValues(i);
		scaledEigenValues(i) = 1.0f/sqrt(eigenValue);
	}

	// Calculate whitening matrix
	Eigen::MatrixXf whiteningMatrix = scaledEigenValues.asDiagonal() * savedEigenVectors;

	// Perform the actual whitening
	if (WRAPPER == BASH)
	{
		printf("Applying dimensionality reduction and whitening\n");
	}
	Eigen::MatrixXf whitenedData = whiteningMatrix * inputData;
	
	return whitenedData;
}

void BROCCOLI_LIB::PCADimensionalityReductionEigen(Eigen::MatrixXd & reducedData,  Eigen::MatrixXd & inputData, int NUMBER_OF_COMPONENTS, bool demean)
{
	// inputData, NUMBER_OF_OBSERVATIONS x NUMBER_OF_VOXELS
	// whitenedData, NUMBER_OF_COMPONENTS x NUMBER_OF_VOXELS

	size_t NUMBER_OF_VOXELS = inputData.cols();
	size_t NUMBER_OF_OBSERVATIONS = inputData.rows();

	printf("Input data matrix size is %li x %li \n",inputData.rows(),inputData.cols());

	if (demean)
	{
		#pragma omp parallel for
		for (size_t voxel = 0; voxel < NUMBER_OF_VOXELS; voxel++)
		{
			Eigen::VectorXd values = inputData.block(0,voxel,NUMBER_OF_OBSERVATIONS,1);
			DemeanRegressor(values,NUMBER_OF_OBSERVATIONS);
			inputData.block(0,voxel,NUMBER_OF_OBSERVATIONS,1) = values;
		}
	}

	// Calculate covariance Matrix
	Eigen::MatrixXd covarianceMatrix(NUMBER_OF_OBSERVATIONS,NUMBER_OF_OBSERVATIONS);
	ResetEigenMatrix(covarianceMatrix);

	for (size_t voxel = 0; voxel < NUMBER_OF_VOXELS; voxel++)
	{
		//Eigen::VectorXd values = inputData.block(0,voxel,NUMBER_OF_OBSERVATIONS,1);
		Eigen::VectorXd values = inputData.col(voxel);
		covarianceMatrix += values * values.transpose();
	}
	//covarianceMatrix = inputData.adjoint() * inputData;
	covarianceMatrix *= 1.0/(double)(NUMBER_OF_VOXELS - 1);
	printf("Covariance matrix size is %li x %li \n",covarianceMatrix.rows(),covarianceMatrix.cols());
	
	// Calculate eigen values of covariance matrix	
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(covarianceMatrix);
	Eigen::VectorXd eigenValues = es.eigenvalues();
	Eigen::MatrixXd eigenVectors = es.eigenvectors();

	printf("Eigen values matrix size is %li x %li \n",eigenValues.rows(),eigenValues.cols());
	printf("Eigen vectors matrix size is %li x %li \n",eigenVectors.rows(),eigenVectors.cols());

	Eigen::VectorXd savedEigenValues(NUMBER_OF_COMPONENTS);
	Eigen::MatrixXd savedEigenVectors(NUMBER_OF_COMPONENTS,NUMBER_OF_OBSERVATIONS);
	
	double totalVariance = 0.0;
	for (int i = 0; i < NUMBER_OF_OBSERVATIONS; i++)
	{
		totalVariance += eigenValues(i);
	}

	// Get a sub matrix of all the eigen vectors, to remove the smallest ones
	// Get a sub vector of the eigen values, to remove the smallest eigen values
	double savedVariance = 0.0;
	for (int i = 0; i < NUMBER_OF_COMPONENTS; i++)
	{	
		// Find largest eigen value for current component, and it's location
		int index = 0;
		double largestEigenValue = eigenValues.maxCoeff(&index);
		savedEigenValues(i) = largestEigenValue;
		savedVariance += largestEigenValue;

		printf("Largest eigen value is %f \n",(float)largestEigenValue);

		// Get the corresponding eigen vector
		savedEigenVectors.row(i) = eigenVectors.col(index).transpose();

		// Set the previous largest eigen value to 0
		eigenValues(index) = 0.0;
	}

	if ((WRAPPER == BASH) && VERBOSE)
	{
		printf("Saved %f %% of the total variance at the dimensionality reduction\n",(float)savedVariance/(float)totalVariance*100.0);
	}

	// Perform the actual dimensionality reduction
	reducedData = savedEigenVectors * inputData;
}




// Saves a certain percentage of the variance, instead of a fix number of components
#ifdef __linux
Eigen::MatrixXf BROCCOLI_LIB::PCAWhiten(Eigen::MatrixXf & inputData, bool demean)
{
	// inputData, NUMBER_OF_OBSERVATIONS x NUMBER_OF_VOXELS
	// whitenedData, NUMBER_OF_COMPONENTS x NUMBER_OF_VOXELS

	printf("Input data matrix size is %i x %i \n",inputData.rows(),inputData.cols());

	if (demean)
	{
		if (WRAPPER == BASH)
		{	
			printf("Demeaning data\n");
		}
		#pragma omp parallel for
		for (size_t voxel = 0; voxel < NUMBER_OF_ICA_VARIABLES; voxel++)
		{
			//printf("Demeaning data for voxel %i\n",voxel);
			Eigen::VectorXf values = inputData.block(0,voxel,NUMBER_OF_ICA_OBSERVATIONS,1);
			DemeanRegressor(values,NUMBER_OF_ICA_OBSERVATIONS);
			inputData.block(0,voxel,NUMBER_OF_ICA_OBSERVATIONS,1) = values;
		}
	}

	// Calculate covariance Matrix
	if (WRAPPER == BASH)
	{
		printf("Estimating the covariance matrix using clBLAS\n");
	}

	double startTime, endTime;
	startTime = GetTime();

	cl_mem d_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_OBSERVATIONS *  NUMBER_OF_ICA_VARIABLES * sizeof(float), NULL, NULL);
	cl_mem d_Covariance_Matrix = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_OBSERVATIONS *  NUMBER_OF_ICA_OBSERVATIONS * sizeof(float), NULL, NULL);
	
	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_Data, CL_TRUE, 0, NUMBER_OF_ICA_OBSERVATIONS * NUMBER_OF_ICA_VARIABLES * sizeof(float), inputData.data(), 0, NULL, NULL);

	// C = alpha * A * B  + beta * C                                            
	//                      rows in d_Data             columns in d_Data          columns in d_Data       alpha   A matrix     leading dimension of A-matrix      B matrix     leading dimension of B-matrix    beta     C matrix
 	error = clblasSgemm (clblasColumnMajor, clblasNoTrans, clblasTrans, NUMBER_OF_ICA_OBSERVATIONS,   NUMBER_OF_ICA_OBSERVATIONS,    NUMBER_OF_ICA_VARIABLES, 1.0f/(float)(NUMBER_OF_ICA_VARIABLES - 1),  d_Data, 0,   NUMBER_OF_ICA_OBSERVATIONS,       d_Data, 0,      NUMBER_OF_ICA_OBSERVATIONS,    0.0f,   d_Covariance_Matrix, 0, NUMBER_OF_ICA_OBSERVATIONS, 1, &commandQueue, 0, NULL, NULL);
	clFinish(commandQueue);
	//printf("clBLAS error for covariance matrix is %s \n",GetOpenCLErrorMessage(error));

	// Copy covariance matrix back to host
	Eigen::MatrixXf covarianceMatrix(NUMBER_OF_ICA_OBSERVATIONS,NUMBER_OF_ICA_OBSERVATIONS);
	clEnqueueReadBuffer(commandQueue, d_Covariance_Matrix, CL_TRUE, 0, NUMBER_OF_ICA_OBSERVATIONS * NUMBER_OF_ICA_OBSERVATIONS * sizeof(float), covarianceMatrix.data(), 0, NULL, NULL);

	clReleaseMemObject(d_Data);
	clReleaseMemObject(d_Covariance_Matrix);

	endTime = GetTime();
	if ((WRAPPER == BASH) && VERBOS)
	{
		printf("It took %f seconds to calculate the covariance matrix using clBLAS\n",(float)(endTime - startTime));
	}

	// Calculate eigen values of covariance matrix	
	if (WRAPPER == BASH)
	{
		printf("Calculating eigen values \n");
	}
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(covarianceMatrix);
	Eigen::VectorXf eigenValues = es.eigenvalues();
	Eigen::MatrixXf eigenVectors = es.eigenvectors();


	float totalVariance = 0.0;
	for (int i = 0; i < NUMBER_OF_ICA_OBSERVATIONS; i++)
	{
		totalVariance += eigenValues(i);

		//printf("Eigen value for Eigen is %f and for clBLAS is %f \n",eigenValues(i),eigenValues2(i));
	}

	// Calculate number of components to save
	float savedVariance = 0.0;
	Eigen::VectorXf temp = eigenValues;
	NUMBER_OF_ICA_COMPONENTS = 0;
	while (savedVariance/totalVariance*100.0 < (double)PROPORTION_OF_VARIANCE_TO_SAVE_BEFORE_ICA )
	{
		NUMBER_OF_ICA_COMPONENTS++;
		int index = 0;
		float largestEigenValue = temp.maxCoeff(&index);
		savedVariance += largestEigenValue;
		temp(index) = 0.0;
	}

	Eigen::VectorXf savedEigenValues(NUMBER_OF_ICA_COMPONENTS);
	Eigen::MatrixXf savedEigenVectors(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_OBSERVATIONS);

	// Get a sub matrix of all the eigen vectors, to remove the smallest ones
	// Get a sub vector of the eigen values, to remove the smallest eigen values
	for (int i = 0; i < NUMBER_OF_ICA_COMPONENTS; i++)
	{	
		// Find largest eigen value for current component, and it's location
		int index = 0; 
		float largestEigenValue = eigenValues.maxCoeff(&index);
		savedEigenValues(i) = largestEigenValue;

		printf("Largest eigen value is %f \n",(float)largestEigenValue);

		// Get the corresponding eigen vector
		savedEigenVectors.row(i) = eigenVectors.col(index).transpose();

		// Set the previous largest eigen value to 0
		eigenValues(index) = 0.0;
	}

	if ((WRAPPER == BASH) && VERBOSE)
	{
		printf("Saved %f %% of the total variance during the dimensionality reduction, using %i components\n",(float)savedVariance/(float)totalVariance*100.0,NUMBER_OF_ICA_COMPONENTS);
	}

	// Calculate  ^(-1/2) for all saved eigen values
	Eigen::VectorXf scaledEigenValues(NUMBER_OF_ICA_COMPONENTS);
	for (int i = 0; i < NUMBER_OF_ICA_COMPONENTS; i++)
	{	
		float eigenValue = savedEigenValues(i);
		scaledEigenValues(i) = 1.0/sqrt(eigenValue);
	}

	// Calculate whitening matrix
	Eigen::MatrixXf whiteningMatrix = scaledEigenValues.asDiagonal() * savedEigenVectors;

	printf("Whitening matrix size is %i  x %i \n",whiteningMatrix.rows(),whiteningMatrix.cols());

	startTime = GetTime();

	d_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_OBSERVATIONS *  NUMBER_OF_ICA_VARIABLES * sizeof(float), NULL, NULL);
	cl_mem d_Whitening_Matrix = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_OBSERVATIONS * sizeof(float), NULL, NULL);
	cl_mem d_Whitened_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_VARIABLES *  sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_Data, CL_TRUE, 0, NUMBER_OF_ICA_OBSERVATIONS * NUMBER_OF_ICA_VARIABLES * sizeof(float), inputData.data(), 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, d_Whitening_Matrix, CL_TRUE, 0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_OBSERVATIONS * sizeof(float), whiteningMatrix.data(), 0, NULL, NULL);

	// C = alpha * A * B  + beta * C                                            
 	error = clblasSgemm (clblasColumnMajor, clblasNoTrans, clblasNoTrans, NUMBER_OF_ICA_COMPONENTS,   NUMBER_OF_ICA_VARIABLES,    NUMBER_OF_ICA_OBSERVATIONS, 1.0f,  d_Whitening_Matrix, 0,   NUMBER_OF_ICA_COMPONENTS,       d_Data, 0,      NUMBER_OF_ICA_OBSERVATIONS,    0.0f,   d_Whitened_Data, 0, NUMBER_OF_ICA_COMPONENTS, 1, &commandQueue, 0, NULL, NULL);
	clFinish(commandQueue);
	//printf("clBLAS error for whitening data is %s \n",GetOpenCLErrorMessage(error));

	// Copy whitened data back to host
	Eigen::MatrixXf whitenedData(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_VARIABLES);
	clEnqueueReadBuffer(commandQueue, d_Whitened_Data, CL_TRUE, 0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_VARIABLES * sizeof(float), whitenedData.data(), 0, NULL, NULL);

	clReleaseMemObject(d_Data);
	clReleaseMemObject(d_Whitening_Matrix);
	clReleaseMemObject(d_Whitened_Data);

	endTime = GetTime();
	if ((WRAPPER == BASH) && VERBOS)
	{
		printf("It took %f seconds to perform the whitening using clBLAS\n",(float)(endTime - startTime));
	}

	return whitenedData;
}
#elif __APPLE__
Eigen::MatrixXf BROCCOLI_LIB::PCAWhiten(Eigen::MatrixXf & inputData, bool demean)
{	
}
#endif




void BROCCOLI_LIB::LogitEigenMatrix(Eigen::MatrixXd & matrix)
{
	int ROWS = matrix.rows();
	int COLUMNS = matrix.cols();

	for (int r = 0; r < ROWS; r++)
	{
		for (int c = 0; c < COLUMNS; c++)
		{
			matrix(r,c) = 1.0-(2.0 / (1.0 + exp(-matrix(r,c) )) );
		}
	}
	// NOTE: gsl_expm1 computes exp(x)-1, hence the 2 + in denominator
    // return 1.0 - (2.0 / (2.0 + gsl_expm1(-in)));
}

void BROCCOLI_LIB::LogitEigenMatrix(Eigen::MatrixXf & matrix)
{
	int ROWS = matrix.rows();
	int COLUMNS = matrix.cols();

	for (int r = 0; r < ROWS; r++)
	{
		for (int c = 0; c < COLUMNS; c++)
		{
			matrix(r,c) = 1.0f-(2.0f / (1.0f + exp(-matrix(r,c) )) );
		}
	}
	// NOTE: gsl_expm1 computes exp(x)-1, hence the 2 + in denominator
    // return 1.0 - (2.0 / (2.0 + gsl_expm1(-in)));
}

void BROCCOLI_LIB::SetEigenVectorValues(Eigen::VectorXd & vector, double value)
{
	int N = 0;

	if (vector.rows() > vector.cols())
	{
		N = vector.rows();
	}
	else
	{
		N = vector.cols();
	}
	
	for (int i = 0; i < N; i++)
	{
		vector(i) = value;
	}
}

void BROCCOLI_LIB::SetEigenMatrixValues(Eigen::MatrixXd & matrix, double value)
{
	int ROWS = matrix.rows();
	int COLUMNS = matrix.cols();

	for (int r = 0; r < ROWS; r++)
	{
		for (int c = 0; c < COLUMNS; c++)
		{
			matrix(r,c) = value;
		}
	}
}

void BROCCOLI_LIB::SetEigenVectorValues(Eigen::VectorXf & vector, double value)
{
	int N = 0;

	if (vector.rows() > vector.cols())
	{
		N = vector.rows();
	}
	else
	{
		N = vector.cols();
	}
	
	for (int i = 0; i < N; i++)
	{
		vector(i) = value;
	}
}

void BROCCOLI_LIB::SetEigenMatrixValues(Eigen::MatrixXf & matrix, double value)
{
	int ROWS = matrix.rows();
	int COLUMNS = matrix.cols();

	for (int r = 0; r < ROWS; r++)
	{
		for (int c = 0; c < COLUMNS; c++)
		{
			matrix(r,c) = value;
		}
	}
}














int BROCCOLI_LIB::UpdateInfomaxWeightsEigen(Eigen::MatrixXd & weights, Eigen::MatrixXd & whitenedData, Eigen::MatrixXd & bias, Eigen::MatrixXd & shuffledWhitenedData, double updateRate)
{
	double MAX_W = 1.0e8;
	int error = 0;
	//size_t block = (size_t)floor(sqrt((float)NUMBER_OF_ICA_VARIABLES/3.0f));
	size_t block = NUMBER_OF_ICA_VARIABLES/10;

	// Create random permutation vector
	std::vector<int> perm;
	for (size_t i = 0; i < NUMBER_OF_ICA_VARIABLES; i++) 
	{
	    perm.push_back(i);
	}
	std::random_shuffle(perm.begin(), perm.end());

	// Loop over voxels, randomly permute each column
	for (size_t i = 0; i < NUMBER_OF_ICA_COMPONENTS; i++)
	{
		Eigen::VectorXd row = shuffledWhitenedData.row(i);
		Eigen::VectorXd permutedRow = row;

		for (size_t j = 0; j < NUMBER_OF_ICA_VARIABLES; j++)
		{
			permutedRow(j) = row(perm[j]);
		}		

		shuffledWhitenedData.row(i) = permutedRow.transpose();		
	}

	size_t start;
	Eigen::MatrixXd tempI(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_COMPONENTS);

	Eigen::MatrixXd *ib =  new Eigen::MatrixXd(1,block);
	SetEigenMatrixValues(*ib,1.0);	

	Eigen::MatrixXd * unmixed = new Eigen::MatrixXd(NUMBER_OF_ICA_COMPONENTS,block);
	Eigen::MatrixXd * unmLogit = new Eigen::MatrixXd(NUMBER_OF_ICA_COMPONENTS,block);
	Eigen::MatrixXd * ones = new Eigen::MatrixXd(block,1);
	SetEigenMatrixValues(*ones,1.0);

	for (start = 0; start < NUMBER_OF_ICA_VARIABLES; start = start + block) 
	{
		if (start + block > (NUMBER_OF_ICA_VARIABLES-1))
		{
			block = NUMBER_OF_ICA_VARIABLES - start;

			delete ib;
			delete unmixed;
			delete unmLogit;
			delete ones;

			ib =  new Eigen::MatrixXd(1,block);
			SetEigenMatrixValues(*ib,1.0);	
			unmixed = new Eigen::MatrixXd(NUMBER_OF_ICA_COMPONENTS,block);
			unmLogit = new Eigen::MatrixXd(NUMBER_OF_ICA_COMPONENTS,block);
			ones = new Eigen::MatrixXd(block,1);
			SetEigenMatrixValues(*ones,1.0);
		}	

		Eigen::MatrixXd subWhitenedData = shuffledWhitenedData.block(0,start,NUMBER_OF_ICA_COMPONENTS,block);

		// Compute unmixed = weights . sub_x_white + bias . ib
		
		*unmixed = weights * subWhitenedData + bias * *ib;

		*unmLogit = *unmixed;
	    // Compute 1-2*logit
		LogitEigenMatrix(*unmLogit);
		
		IdentityEigenMatrix(tempI);
	    // weights = weights + lrate*(block*I+(unmLogit*unmixed.T))*weights

	    // (1) temp_I = block*temp_I +unm_logit*unmixed.T
		tempI = (double)block * tempI + *unmLogit * (*unmixed).transpose();
		
	    // (2) weights = weights + lrate*temp_I*weights
		weights += updateRate * tempI * weights;

	    // Update the bias
		bias += updateRate * *unmLogit * *ones;

	    // Check if blows up
	    double max = weights.maxCoeff();

		if (max > MAX_W)
	    {
			if (updateRate < 1e-6) 
			{
				printf("\nERROR: Weight matrix may not be invertible\n");
				error = 2;
				break;
			}
			error = 1;
			break;
		}
	}

	delete ib;
	delete unmixed;
	delete unmLogit;
	delete ones;

	return(error);
}

int BROCCOLI_LIB::UpdateInfomaxWeightsEigen(Eigen::MatrixXf & weights, Eigen::MatrixXf & whitenedData, Eigen::MatrixXf & bias, Eigen::MatrixXf & shuffledWhitenedData, double updateRate)
{
	double MAX_W = 1.0e8;
	int error = 0;
	//size_t block = (size_t)floor(sqrt((float)NUMBER_OF_ICA_VARIABLES/3.0f))*2;
	size_t block = NUMBER_OF_ICA_VARIABLES/10;

	// Create random permutation vector
	std::vector<int> perm;
	for (size_t i = 0; i < NUMBER_OF_ICA_VARIABLES; i++) 
	{
	    perm.push_back(i);
	}
	std::random_shuffle(perm.begin(), perm.end());

	// Loop over voxels, randomly permute each column
	for (size_t i = 0; i < NUMBER_OF_ICA_COMPONENTS; i++)
	{
		Eigen::VectorXf row = shuffledWhitenedData.row(i);
		Eigen::VectorXf permutedRow = row;

		for (size_t j = 0; j < NUMBER_OF_ICA_VARIABLES; j++)
		{
			permutedRow(j) = row(perm[j]);
		}		

		shuffledWhitenedData.row(i) = permutedRow.transpose();		
	}

	size_t start;
	Eigen::MatrixXf tempI(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_COMPONENTS);

	Eigen::MatrixXf *ib =  new Eigen::MatrixXf(1,block);
	SetEigenMatrixValues(*ib,1.0);	

	Eigen::MatrixXf * unmixed = new Eigen::MatrixXf(NUMBER_OF_ICA_COMPONENTS,block);
	Eigen::MatrixXf * unmLogit = new Eigen::MatrixXf(NUMBER_OF_ICA_COMPONENTS,block);
	Eigen::MatrixXf * ones = new Eigen::MatrixXf(block,1);
	SetEigenMatrixValues(*ones,1.0);

	for (start = 0; start < NUMBER_OF_ICA_VARIABLES; start = start + block) 
	{
		//printf("Start is %zu \n",start);

		if (start + block > (NUMBER_OF_ICA_VARIABLES-1))
		{
			block = NUMBER_OF_ICA_VARIABLES - start;

			delete ib;
			delete unmixed;
			delete unmLogit;
			delete ones;

			ib =  new Eigen::MatrixXf(1,block);
			SetEigenMatrixValues(*ib,1.0);	
			unmixed = new Eigen::MatrixXf(NUMBER_OF_ICA_COMPONENTS,block);
			unmLogit = new Eigen::MatrixXf(NUMBER_OF_ICA_COMPONENTS,block);
			ones = new Eigen::MatrixXf(block,1);
			SetEigenMatrixValues(*ones,1.0);
		}	

		Eigen::MatrixXf subWhitenedData = shuffledWhitenedData.block(0,start,NUMBER_OF_ICA_COMPONENTS,block);

		// Compute unmixed = weights . sub_x_white + bias . ib
		
		*unmixed = weights * subWhitenedData + bias * *ib;

		*unmLogit = *unmixed;
	    // Compute 1-2*logit
		LogitEigenMatrix(*unmLogit);
		
		IdentityEigenMatrix(tempI);
	    // weights = weights + lrate*(block*I+(unmLogit*unmixed.T))*weights

	    // (1) temp_I = block*temp_I +unm_logit*unmixed.T
		tempI = (double)block * tempI + *unmLogit * (*unmixed).transpose();
		
	    // (2) weights = weights + lrate*temp_I*weights
		weights += updateRate * tempI * weights;

	    // Update the bias
		bias += updateRate * *unmLogit * *ones;

	    // Check if blows up
	    double max = weights.maxCoeff();

		if (max > MAX_W)
	    {
			if (updateRate < 1e-6) 
			{
				printf("\nERROR: Weight matrix may not be invertible\n");
				error = 2;
				break;
			}
			error = 1;
			break;
		}
	}

	delete ib;
	delete unmixed;
	delete unmLogit;
	delete ones;

	return(error);
}



#ifdef __linux
int BROCCOLI_LIB::UpdateInfomaxWeights(cl_mem d_Weights, cl_mem d_Whitened_Data, cl_mem d_Bias, cl_mem d_Permutation, cl_mem d_Shuffled_Whitened_Data, double updateRate)
{
	double MAX_W = 1.0e8;
	int error = 0;
	size_t i;
	//size_t block = (size_t)floor(sqrt((float)NUMBER_OF_ICA_VARIABLES/3.0f))*2;
	size_t block = NUMBER_OF_ICA_VARIABLES/10;

	// Create random permutation vector
	std::vector<unsigned int> perm;
	for (unsigned int i = 0; i < NUMBER_OF_ICA_VARIABLES; i++) 
	{
	    perm.push_back(i);
	}
	std::random_shuffle(perm.begin(), perm.end());

	// Copy permutation to device
	clEnqueueWriteBuffer(commandQueue, d_Permutation, CL_TRUE, 0, NUMBER_OF_ICA_VARIABLES * sizeof(unsigned int), perm.data(), 0, NULL, NULL);

	PermuteMatrix(d_Shuffled_Whitened_Data, d_Whitened_Data, d_Permutation, NUMBER_OF_ICA_COMPONENTS, NUMBER_OF_ICA_VARIABLES);

	size_t start;

	cl_mem d_TempI = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(float), NULL, NULL);

	cl_mem d_ib = clCreateBuffer(context, CL_MEM_READ_WRITE, block * sizeof(float), NULL, NULL);
	cl_mem d_unmixed = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * block * sizeof(float), NULL, NULL);
	cl_mem d_unmLogit = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * block * sizeof(float), NULL, NULL);
	cl_mem d_ones = clCreateBuffer(context, CL_MEM_READ_WRITE, block * sizeof(float), NULL, NULL);
	cl_mem d_Sub_Whitened_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * block * sizeof(float), NULL, NULL);

	SetMemory(d_ib, 1.0f, block);
	SetMemory(d_ones, 1.0f, block);

	for (start = 0; start < NUMBER_OF_ICA_VARIABLES; start = start + block) 
	{
		//printf("Start is %zu \n",start);

		if (start + block > (NUMBER_OF_ICA_VARIABLES-1))
		{
			block = NUMBER_OF_ICA_VARIABLES - start;

			clReleaseMemObject(d_ib);
			clReleaseMemObject(d_unmixed);
			clReleaseMemObject(d_unmLogit);
			clReleaseMemObject(d_ones);
			clReleaseMemObject(d_Sub_Whitened_Data);

			d_ib = clCreateBuffer(context, CL_MEM_READ_WRITE, block * sizeof(float), NULL, NULL);
			d_unmixed = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * block * sizeof(float), NULL, NULL);
			d_unmLogit = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * block * sizeof(float), NULL, NULL);
			d_ones = clCreateBuffer(context, CL_MEM_READ_WRITE, block * sizeof(float), NULL, NULL);
			d_Sub_Whitened_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * block * sizeof(float), NULL, NULL);

			SetMemory(d_ib, 1.0f, block);
			SetMemory(d_ones, 1.0f, block);		
		}	

		GetSubMatrix(d_Sub_Whitened_Data, d_Shuffled_Whitened_Data, 0, start, NUMBER_OF_ICA_COMPONENTS, block, NUMBER_OF_ICA_COMPONENTS, NUMBER_OF_ICA_VARIABLES);

		// Compute unmixed = weights * subWhitenedData + bias * ib
		
		// First unmixed = weights * subWhitenedData 
		// C = alpha * A * B  + beta * C                         
		error = clblasSgemm (clblasColumnMajor, clblasNoTrans, clblasNoTrans, NUMBER_OF_ICA_COMPONENTS,           block,               NUMBER_OF_ICA_COMPONENTS, 1.0f,  d_Weights, 0, NUMBER_OF_ICA_COMPONENTS,          d_Sub_Whitened_Data, 0,          NUMBER_OF_ICA_COMPONENTS,       0.0f,   d_unmixed, 0, NUMBER_OF_ICA_COMPONENTS, 1, &commandQueue, 0, NULL, NULL);
		if (error != CL_SUCCESS)
		{
			printf("Error for first Sgemm is %i\n",error);
		}

		// Then C = bias * ib + C
	 	error = clblasSgemm (clblasColumnMajor, clblasNoTrans, clblasNoTrans, NUMBER_OF_ICA_COMPONENTS,      block,          1,                1.0f,  d_Bias, 0,     NUMBER_OF_ICA_COMPONENTS,          d_ib, 0, 1, 1.0f,   d_unmixed, 0, NUMBER_OF_ICA_COMPONENTS, 1, &commandQueue, 0, NULL, NULL);
		if (error != CL_SUCCESS)
		{
			printf("Error for second Sgemm is %i\n",error);
		}

			
		//unmLogit = unmixed;
		clEnqueueCopyBuffer(commandQueue, d_unmixed, d_unmLogit, 0, 0, NUMBER_OF_ICA_COMPONENTS * block * sizeof(float), 0, NULL, NULL);

	    // Compute 1-2*logit
		LogitMatrix(d_unmLogit,NUMBER_OF_ICA_COMPONENTS * block);

		IdentityMatrix(d_TempI,NUMBER_OF_ICA_COMPONENTS);

	    // weights = weights + lrate*(block*I+(unmLogit*unmixed.T))*weights

		// (1) temp_I = block*temp_I +unm_logit*unmixed.T
		
		// C = alpha * A * B  + beta * C
	 	error = clblasSgemm (clblasColumnMajor, clblasNoTrans, clblasTrans, NUMBER_OF_ICA_COMPONENTS, NUMBER_OF_ICA_COMPONENTS, block, 1.0f, d_unmLogit, 0, NUMBER_OF_ICA_COMPONENTS, d_unmixed, 0, NUMBER_OF_ICA_COMPONENTS, (float)block,   d_TempI, 0, NUMBER_OF_ICA_COMPONENTS, 1, &commandQueue, 0, NULL, NULL);
		if (error != CL_SUCCESS)
		{
			printf("Error for third Sgemm is %i\n",error);
		}
		
	    // (2) weights = weights + lrate*temp_I*weights
		
		// C = alpha * A * B  + beta * C
	 	error = clblasSgemm (clblasColumnMajor, clblasNoTrans, clblasNoTrans, NUMBER_OF_ICA_COMPONENTS, NUMBER_OF_ICA_COMPONENTS, NUMBER_OF_ICA_COMPONENTS, (float)updateRate, d_TempI, 0, NUMBER_OF_ICA_COMPONENTS, d_Weights, 0, NUMBER_OF_ICA_COMPONENTS, 1.0f,   d_Weights, 0, NUMBER_OF_ICA_COMPONENTS, 1, &commandQueue, 0, NULL, NULL);
		if (error != CL_SUCCESS)
		{
			printf("Error for fourth Sgemm is %i\n",error);
		}


	    // Update the bias
		// bias += updateRate * *unmLogit * *ones;

		// y = alpha * A * x  + beta * y
		error = clblasSgemv(clblasColumnMajor, clblasNoTrans, NUMBER_OF_ICA_COMPONENTS, block, (float)updateRate, d_unmLogit, 0, NUMBER_OF_ICA_COMPONENTS, d_ones, 0, 1, 1.0f, d_Bias, 0, 1, 1, &commandQueue, 0, NULL, NULL);
		if (error != CL_SUCCESS)
		{
			printf("Error for Sgemv is %i\n",error);
		}

	    // Check if blows up
	    double max = CalculateMaxAtomic(d_Weights, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);

		if (max > MAX_W)
	    {
			if (updateRate < 1e-6) 
			{
				printf("\nERROR: Weight matrix may not be invertible\n");
				error = 2;
				break;
			}
			error = 1;
			break;
		}
	}

	clReleaseMemObject(d_ib);
	clReleaseMemObject(d_unmixed);
	clReleaseMemObject(d_unmLogit);
	clReleaseMemObject(d_ones);
	clReleaseMemObject(d_Sub_Whitened_Data);

	return(error);
}
#elif __APPLE__
int BROCCOLI_LIB::UpdateInfomaxWeights(cl_mem d_Weights, cl_mem d_Whitened_Data, cl_mem d_Bias, cl_mem d_Permutation, cl_mem d_Shuffled_Whitened_Data, double updateRate)
{
	return 0;
}
#endif



#ifdef __linux
int BROCCOLI_LIB::UpdateInfomaxWeightsDouble(cl_mem d_Weights, cl_mem d_Whitened_Data, cl_mem d_Bias, cl_mem d_Permutation, cl_mem d_Shuffled_Whitened_Data, double updateRate)
{
	double MAX_W = 1.0e8;
	int error = 0;
	size_t i;
	//size_t block = (size_t)floor(sqrt((float)NUMBER_OF_ICA_VARIABLES/3.0f))*2;
	size_t block = NUMBER_OF_ICA_VARIABLES/10;

	// Create random permutation vector
	std::vector<unsigned int> perm;
	for (unsigned int i = 0; i < NUMBER_OF_ICA_VARIABLES; i++) 
	{
	    perm.push_back(i);
	}
	std::random_shuffle(perm.begin(), perm.end());

	// Copy permutation to device
	clEnqueueWriteBuffer(commandQueue, d_Permutation, CL_TRUE, 0, NUMBER_OF_ICA_VARIABLES * sizeof(unsigned int), perm.data(), 0, NULL, NULL);

	PermuteMatrixDouble(d_Shuffled_Whitened_Data, d_Whitened_Data, d_Permutation, NUMBER_OF_ICA_COMPONENTS, NUMBER_OF_ICA_VARIABLES);

	size_t start;

	cl_mem d_TempI = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(double), NULL, NULL);

	cl_mem d_ib = clCreateBuffer(context, CL_MEM_READ_WRITE, block * sizeof(double), NULL, NULL);
	cl_mem d_unmixed = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * block * sizeof(double), NULL, NULL);
	cl_mem d_unmLogit = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * block * sizeof(double), NULL, NULL);
	cl_mem d_ones = clCreateBuffer(context, CL_MEM_READ_WRITE, block * sizeof(double), NULL, NULL);
	cl_mem d_Sub_Whitened_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * block * sizeof(double), NULL, NULL);

	SetMemoryDouble(d_ib, 1.0f, block);
	SetMemoryDouble(d_ones, 1.0f, block);

	for (start = 0; start < NUMBER_OF_ICA_VARIABLES; start = start + block) 
	{
		//printf("Start is %zu \n",start);

		if (start + block > (NUMBER_OF_ICA_VARIABLES-1))
		{
			block = NUMBER_OF_ICA_VARIABLES - start;

			clReleaseMemObject(d_ib);
			clReleaseMemObject(d_unmixed);
			clReleaseMemObject(d_unmLogit);
			clReleaseMemObject(d_ones);
			clReleaseMemObject(d_Sub_Whitened_Data);

			d_ib = clCreateBuffer(context, CL_MEM_READ_WRITE, block * sizeof(double), NULL, NULL);
			d_unmixed = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * block * sizeof(double), NULL, NULL);
			d_unmLogit = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * block * sizeof(double), NULL, NULL);
			d_ones = clCreateBuffer(context, CL_MEM_READ_WRITE, block * sizeof(double), NULL, NULL);
			d_Sub_Whitened_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * block * sizeof(double), NULL, NULL);

			SetMemoryDouble(d_ib, 1.0f, block);
			SetMemoryDouble(d_ones, 1.0f, block);		
		}	

		GetSubMatrixDouble(d_Sub_Whitened_Data, d_Shuffled_Whitened_Data, 0, start, NUMBER_OF_ICA_COMPONENTS, block, NUMBER_OF_ICA_COMPONENTS, NUMBER_OF_ICA_VARIABLES);

		// Compute unmixed = weights * subWhitenedData + bias * ib
		
		// First unmixed = weights * subWhitenedData 
		// C = alpha * A * B  + beta * C                         
		error = clblasDgemm (clblasColumnMajor, clblasNoTrans, clblasNoTrans, NUMBER_OF_ICA_COMPONENTS,           block,               NUMBER_OF_ICA_COMPONENTS, 1.0,  d_Weights, 0, NUMBER_OF_ICA_COMPONENTS,          d_Sub_Whitened_Data, 0,          NUMBER_OF_ICA_COMPONENTS,       0.0,   d_unmixed, 0, NUMBER_OF_ICA_COMPONENTS, 1, &commandQueue, 0, NULL, NULL);
		if (error != CL_SUCCESS)
		{
			printf("Error for first Dgemm is %i\n",error);
		}

		// Then C = bias * ib + C
	 	error = clblasDgemm (clblasColumnMajor, clblasNoTrans, clblasNoTrans, NUMBER_OF_ICA_COMPONENTS,      block,          1,                1.0f,  d_Bias, 0,     NUMBER_OF_ICA_COMPONENTS,          d_ib, 0, 1, 1.0,   d_unmixed, 0, NUMBER_OF_ICA_COMPONENTS, 1, &commandQueue, 0, NULL, NULL);
		if (error != CL_SUCCESS)
		{
			printf("Error for second Dgemm is %i\n",error);
		}
			
		//unmLogit = unmixed;
		clEnqueueCopyBuffer(commandQueue, d_unmixed, d_unmLogit, 0, 0, NUMBER_OF_ICA_COMPONENTS * block * sizeof(double), 0, NULL, NULL);

	    // Compute 1-2*logit
		LogitMatrixDouble(d_unmLogit,NUMBER_OF_ICA_COMPONENTS * block);

		IdentityMatrixDouble(d_TempI,NUMBER_OF_ICA_COMPONENTS);

	    // weights = weights + lrate*(block*I+(unmLogit*unmixed.T))*weights

		// (1) temp_I = block*temp_I +unm_logit*unmixed.T
		
		// C = alpha * A * B  + beta * C
	 	error = clblasDgemm (clblasColumnMajor, clblasNoTrans, clblasTrans, NUMBER_OF_ICA_COMPONENTS, NUMBER_OF_ICA_COMPONENTS, block, 1.0, d_unmLogit, 0, NUMBER_OF_ICA_COMPONENTS, d_unmixed, 0, NUMBER_OF_ICA_COMPONENTS, (double)block,   d_TempI, 0, NUMBER_OF_ICA_COMPONENTS, 1, &commandQueue, 0, NULL, NULL);
		if (error != CL_SUCCESS)
		{
			printf("Error for third Dgemm is %i\n",error);
		}
		
	    // (2) weights = weights + lrate*temp_I*weights
		
		// C = alpha * A * B  + beta * C
	 	error = clblasDgemm (clblasColumnMajor, clblasNoTrans, clblasNoTrans, NUMBER_OF_ICA_COMPONENTS, NUMBER_OF_ICA_COMPONENTS, NUMBER_OF_ICA_COMPONENTS, updateRate, d_TempI, 0, NUMBER_OF_ICA_COMPONENTS, d_Weights, 0, NUMBER_OF_ICA_COMPONENTS, 1.0,   d_Weights, 0, NUMBER_OF_ICA_COMPONENTS, 1, &commandQueue, 0, NULL, NULL);
		if (error != CL_SUCCESS)
		{
			printf("Error for fourth Dgemm is %i\n",error);
		}


	    // Update the bias
		// bias += updateRate * *unmLogit * *ones;

		// y = alpha * A * x  + beta * y
		error = clblasDgemv(clblasColumnMajor, clblasNoTrans, NUMBER_OF_ICA_COMPONENTS, block, updateRate, d_unmLogit, 0, NUMBER_OF_ICA_COMPONENTS, d_ones, 0, 1, 1.0, d_Bias, 0, 1, 1, &commandQueue, 0, NULL, NULL);
		if (error != CL_SUCCESS)
		{
			printf("Error for Dgemv is %i\n",error);
		}

	    // Check if blows up
	    double max = CalculateMaxAtomic(d_Weights, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);

		if (max > MAX_W)
	    {
			if (updateRate < 1e-6) 
			{
				printf("\nERROR: Weight matrix may not be invertible\n");
				error = 2;
				break;
			}
			error = 1;
			break;
		}
	}

	clReleaseMemObject(d_ib);
	clReleaseMemObject(d_unmixed);
	clReleaseMemObject(d_unmLogit);
	clReleaseMemObject(d_ones);
	clReleaseMemObject(d_Sub_Whitened_Data);

	return(error);
}
#elif __APPLE__
int BROCCOLI_LIB::UpdateInfomaxWeightsDouble(cl_mem d_Weights, cl_mem d_Whitened_Data, cl_mem d_Bias, cl_mem d_Permutation, cl_mem d_Shuffled_Whitened_Data, double updateRate)
{
	return 0;
}
#endif



#ifdef __linux
void BROCCOLI_LIB::InfomaxICA(Eigen::MatrixXf & whitenedData, Eigen::MatrixXf & weights, Eigen::MatrixXf & sourceMatrix)
{
  	// Computes ICA infomax in whitened data
    //	Decomposes x_white as x_white=AS
    //	*Input
    //	x_white: whitened data (Use PCAwhiten)
    //	*Output
    //	A : mixing matrix
    //	S : source matrix
  	
	double EPS = 1e-18;
	double MAX_W = 1.0e8;
	double ANNEAL = 0.9;
	double MIN_LRATE = 1e-6;
	double W_STOP = 1e-8;
	size_t MAX_STEP= 100;


	cl_mem d_Whitened_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_VARIABLES * sizeof(float), NULL, NULL);
	cl_mem d_Shuffled_Whitened_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_VARIABLES * sizeof(float), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_Whitened_Data, CL_TRUE, 0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_VARIABLES * sizeof(float), whitenedData.data(), 0, NULL, NULL);

	cl_mem d_Weights = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(float), NULL, NULL);

	cl_mem d_Bias = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * sizeof(float), NULL, NULL);

	cl_mem d_Old_Weights = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(float), NULL, NULL);
	cl_mem d_d_Weights = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(float), NULL, NULL);
	cl_mem d_Old_d_Weights = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(float), NULL, NULL);
	cl_mem d_Temp = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(float), NULL, NULL);

	cl_mem d_Scratch = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * 2 * sizeof(float), NULL, NULL);
	cl_mem d_Float = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
	cl_mem d_Ones = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(float), NULL, NULL);

	cl_mem d_Permutation = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_VARIABLES * sizeof(unsigned int), NULL, NULL);

	IdentityMatrix(d_Weights, NUMBER_OF_ICA_COMPONENTS);
	IdentityMatrix(d_Old_Weights, NUMBER_OF_ICA_COMPONENTS);

	// Set all values to 0
	SetMemory(d_Bias, 0.0f, NUMBER_OF_ICA_COMPONENTS);
	SetMemory(d_d_Weights, 0.0f, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);
	SetMemory(d_Old_d_Weights, 0.0f, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);
	SetMemory(d_Temp, 0.0f, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);

	SetMemory(d_Ones, 1.0f, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);

	double lrate = 0.00005/log((double)NUMBER_OF_ICA_COMPONENTS);
	float change = 1.0f;
	double angleDelta = 0.0;
    size_t step = 1;
	int error = 0;

	float tempsum, dweightsnorm, olddweightsnorm;

	while( (step < MAX_STEP) && (change > W_STOP))
	{		
		double start = GetTime();
		error = UpdateInfomaxWeights(d_Weights, d_Whitened_Data, d_Bias, d_Permutation, d_Shuffled_Whitened_Data, lrate);			
		double end = GetTime();

		if (VERBOS)
		{
			printf("One iteration took %f seconds \n",(float)(end-start));
		}

		if (error == 1 || error == 2)
		{
			// It blowed up! RESTART!
    	  	step = 1;
    	  	// change = 1;
    	  	error = 0;
    	 	lrate *= ANNEAL;		

			IdentityMatrix(d_Weights, NUMBER_OF_ICA_COMPONENTS);
			IdentityMatrix(d_Old_Weights, NUMBER_OF_ICA_COMPONENTS);

			SetMemory(d_Weights, 0.0f, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);
			SetMemory(d_Old_d_Weights, 0.0f, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);
			SetMemory(d_Bias, 0.0f, NUMBER_OF_ICA_COMPONENTS);
			
			if (lrate > MIN_LRATE)
			{
    	    	printf("\nLowering learning rate to %g and starting again.\n",lrate);
    	  	}
    	  	else
			{
		        printf("\nMatrix may not be invertible");
			}
    	}
    	else if (error == 0)
		{
			// dWeights = weights;	
			clEnqueueCopyBuffer(commandQueue, d_Weights, d_d_Weights, 0, 0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(float), 0, NULL, NULL);
	
			//dWeights -= oldWeights;
			SubtractArrays(d_d_Weights, d_Old_Weights, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);

			// Calculate norm of d weights
		  	error = clblasSnrm2(NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS, d_Float, 0, d_d_Weights, 0, 1, d_Scratch, 1, &commandQueue, 0, NULL, NULL);
			if (error != CL_SUCCESS)
			{
				printf("Error for first Snrm2 is %i\n",error);
			}
			clEnqueueReadBuffer(commandQueue, d_Float, CL_TRUE, 0, sizeof(float), &dweightsnorm, 0, NULL, NULL);
			change = dweightsnorm * dweightsnorm;
				
			if (step > 2)
			{
		        // Compute angle delta
				// temp = dWeights;
				clEnqueueCopyBuffer(commandQueue, d_d_Weights, d_Temp, 0, 0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(float), 0, NULL, NULL);

				// Pointwise multiplication
				// temp = temp.array() * oldDWeights.array();
				MultiplyArrays(d_Temp, d_Old_d_Weights, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);

				// Calculate sum of temp as a dot product with ones
				error = clblasSdot(NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS, d_Float, 0, d_Temp, 0, 1, d_Ones, 0, 1, d_Scratch, 1, &commandQueue, 0, NULL, NULL);
				if (error != CL_SUCCESS)
				{
					printf("Error for Sdot is %i\n",error);
				}

				//printf("clBLAS error for temp sum is %s \n",GetOpenCLErrorMessage(error));
				clEnqueueReadBuffer(commandQueue, d_Float, CL_TRUE, 0, sizeof(float), &tempsum, 0, NULL, NULL);						

				// Calculate norm of old d weights
				error = clblasSnrm2(NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS, d_Float, 0, d_Old_d_Weights, 0, 1, d_Scratch, 1, &commandQueue, 0, NULL, NULL);
				if (error != CL_SUCCESS)
				{
					printf("Error for second Snrm2 is %i\n",error);
				}
				clEnqueueReadBuffer(commandQueue, d_Float, CL_TRUE, 0, sizeof(float), &olddweightsnorm, 0, NULL, NULL);				

		        angleDelta = acos(tempsum / (dweightsnorm * olddweightsnorm) );
        		angleDelta *= (180.0 / M_PI);
			}

			//oldWeights = weights;
			clEnqueueCopyBuffer(commandQueue, d_Weights, d_Old_Weights, 0, 0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(float), 0, NULL, NULL);

			if (angleDelta > 60)
			{
        		lrate *= ANNEAL;
				// oldDWeights = dWeights;
				clEnqueueCopyBuffer(commandQueue, d_d_Weights, d_Old_d_Weights, 0, 0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(float), 0, NULL, NULL);
			} 
			else if (step == 1) 
			{
				// oldDWeights = dWeights;
				clEnqueueCopyBuffer(commandQueue, d_d_Weights, d_Old_d_Weights, 0, 0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(float), 0, NULL, NULL);
			}

			printf("\n\nStep %zu: Lrate %.1e, Wchange %.1e, Angle %.2f \n\n", step, lrate, change, angleDelta);

			step++;
    	}
  	}

	clEnqueueReadBuffer(commandQueue, d_Weights, CL_TRUE, 0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(float), weights.data(), 0, NULL, NULL);

	sourceMatrix = weights * whitenedData;	

	// C = alpha * A * B  + beta * C                                           rows in d_Weights  columns in d_Whitened_Data   columns in d_Weights   alpha   A matrix                                  B matrix                                   beta     C matrix
 	//error = clblasSgemm (clblasColumnMajor, clblasNoTrans, clblasNoTrans, NUMBER_OF_ICA_COMPONENTS, NUMBER_OF_ICA_VARIABLES, NUMBER_OF_ICA_COMPONENTS, 1.0f, d_Weights, 0, NUMBER_OF_ICA_COMPONENTS, d_Whitened_Data, 0, NUMBER_OF_ICA_COMPONENTS, 0.0f, d_Source_Matrix, 0, NUMBER_OF_ICA_COMPONENTS, 1, &commandQueue, 0, NULL, NULL);

	clReleaseMemObject(d_Whitened_Data);
	clReleaseMemObject(d_Shuffled_Whitened_Data);

	clReleaseMemObject(d_Weights);

	clReleaseMemObject(d_Bias);
	clReleaseMemObject(d_Old_Weights);
	clReleaseMemObject(d_d_Weights);
	clReleaseMemObject(d_Old_d_Weights);
	clReleaseMemObject(d_Temp);
	clReleaseMemObject(d_Scratch);
	clReleaseMemObject(d_Float);
	clReleaseMemObject(d_Ones);

	clReleaseMemObject(d_Permutation);
}
#elif __APPLE__
void BROCCOLI_LIB::InfomaxICA(Eigen::MatrixXf & whitenedData, Eigen::MatrixXf & weights, Eigen::MatrixXf & sourceMatrix)
{
	printf("Currently it is only possible to use the -cpu option for ICA on Mac platforms\n");
}
#endif



#ifdef __linux
void BROCCOLI_LIB::InfomaxICADouble(Eigen::MatrixXd & whitenedData, Eigen::MatrixXd & weights, Eigen::MatrixXd & sourceMatrix)
{
  	// Computes ICA infomax in whitened data
    //	Decomposes x_white as x_white=AS
    //	*Input
    //	x_white: whitened data (Use PCAwhiten)
    //	*Output
    //	A : mixing matrix
    //	S : source matrix
  	
	double EPS = 1e-18;
	double MAX_W = 1.0e8;
	double ANNEAL = 0.9;
	double MIN_LRATE = 1e-6;
	double W_STOP = 1e-8;
	size_t MAX_STEP= 100;


	cl_mem d_Whitened_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_VARIABLES * sizeof(double), NULL, NULL);
	cl_mem d_Shuffled_Whitened_Data = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_VARIABLES * sizeof(double), NULL, NULL);

	// Copy data to device
	clEnqueueWriteBuffer(commandQueue, d_Whitened_Data, CL_TRUE, 0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_VARIABLES * sizeof(double), whitenedData.data(), 0, NULL, NULL);

	cl_mem d_Weights = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(double), NULL, NULL);

	cl_mem d_Bias = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * sizeof(double), NULL, NULL);

	cl_mem d_Old_Weights = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(double), NULL, NULL);
	cl_mem d_d_Weights = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(double), NULL, NULL);
	cl_mem d_Old_d_Weights = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(double), NULL, NULL);
	cl_mem d_Temp = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(double), NULL, NULL);

	cl_mem d_Scratch = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * 2 * sizeof(double), NULL, NULL);
	cl_mem d_Double = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double), NULL, NULL);
	cl_mem d_Ones = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(double), NULL, NULL);

	cl_mem d_Permutation = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMBER_OF_ICA_VARIABLES * sizeof(unsigned int), NULL, NULL);

	IdentityMatrixDouble(d_Weights, NUMBER_OF_ICA_COMPONENTS);
	IdentityMatrixDouble(d_Old_Weights, NUMBER_OF_ICA_COMPONENTS);

	// Set all values to 0
	SetMemoryDouble(d_Bias, 0.0, NUMBER_OF_ICA_COMPONENTS);
	SetMemoryDouble(d_d_Weights, 0.0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);
	SetMemoryDouble(d_Old_d_Weights, 0.0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);
	SetMemoryDouble(d_Temp, 0.0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);

	SetMemoryDouble(d_Ones, 1.0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);

	double lrate = 0.00005/log((double)NUMBER_OF_ICA_COMPONENTS);
	double change = 1.0f;
	double angleDelta = 0.0;
    size_t step = 1;
	int error = 0;

	double tempsum, dweightsnorm, olddweightsnorm;

	while( (step < MAX_STEP) && (change > W_STOP))
	{		
		double start = GetTime();
		error = UpdateInfomaxWeightsDouble(d_Weights, d_Whitened_Data, d_Bias, d_Permutation, d_Shuffled_Whitened_Data, lrate);			
		double end = GetTime();

		if (VERBOS)
		{
			printf("One iteration took %f seconds \n",(float)(end-start));
		}

		if (error == 1 || error == 2)
		{
			// It blowed up! RESTART!
    	  	step = 1;
    	  	// change = 1;
    	  	error = 0;
    	 	lrate *= ANNEAL;		

			IdentityMatrixDouble(d_Weights, NUMBER_OF_ICA_COMPONENTS);
			IdentityMatrixDouble(d_Old_Weights, NUMBER_OF_ICA_COMPONENTS);

			SetMemoryDouble(d_Weights, 0.0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);
			SetMemoryDouble(d_Old_d_Weights, 0.0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);
			SetMemoryDouble(d_Bias, 0.0, NUMBER_OF_ICA_COMPONENTS);
			
			if (lrate > MIN_LRATE)
			{
    	    	printf("\nLowering learning rate to %g and starting again.\n",lrate);
    	  	}
    	  	else
			{
		        printf("\nMatrix may not be invertible");
			}
    	}
    	else if (error == 0)
		{
			// dWeights = weights;	
			clEnqueueCopyBuffer(commandQueue, d_Weights, d_d_Weights, 0, 0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(double), 0, NULL, NULL);
	
			//dWeights -= oldWeights;
			SubtractArraysDouble(d_d_Weights, d_Old_Weights, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);

			// Calculate norm of d weights
		  	error = clblasDnrm2(NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS, d_Double, 0, d_d_Weights, 0, 1, d_Scratch, 1, &commandQueue, 0, NULL, NULL);
			if (error != CL_SUCCESS)
			{
				printf("Error for first Dnrm2 is %i\n",error);
			}
			clEnqueueReadBuffer(commandQueue, d_Double, CL_TRUE, 0, sizeof(double), &dweightsnorm, 0, NULL, NULL);
			change = dweightsnorm * dweightsnorm;
				
			if (step > 2)
			{
		        // Compute angle delta
				// temp = dWeights;
				clEnqueueCopyBuffer(commandQueue, d_d_Weights, d_Temp, 0, 0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(double), 0, NULL, NULL);

				// Pointwise multiplication
				// temp = temp.array() * oldDWeights.array();
				MultiplyArraysDouble(d_Temp, d_Old_d_Weights, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS);

				// Calculate sum of temp as a dot product with ones
				error = clblasDdot(NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS, d_Double, 0, d_Temp, 0, 1, d_Ones, 0, 1, d_Scratch, 1, &commandQueue, 0, NULL, NULL);
				if (error != CL_SUCCESS)
				{
					printf("Error for Ddot is %i\n",error);
				}

				//printf("clBLAS error for temp sum is %s \n",GetOpenCLErrorMessage(error));
				clEnqueueReadBuffer(commandQueue, d_Double, CL_TRUE, 0, sizeof(double), &tempsum, 0, NULL, NULL);						

				// Calculate norm of old d weights
				error = clblasDnrm2(NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS, d_Double, 0, d_Old_d_Weights, 0, 1, d_Scratch, 1, &commandQueue, 0, NULL, NULL);
				if (error != CL_SUCCESS)
				{
					printf("Error for second Dnrm2 is %i\n",error);
				}
				clEnqueueReadBuffer(commandQueue, d_Double, CL_TRUE, 0, sizeof(double), &olddweightsnorm, 0, NULL, NULL);				

		        angleDelta = acos(tempsum / (dweightsnorm * olddweightsnorm) );
        		angleDelta *= (180.0 / M_PI);
			}

			//oldWeights = weights;
			clEnqueueCopyBuffer(commandQueue, d_Weights, d_Old_Weights, 0, 0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(double), 0, NULL, NULL);

			if (angleDelta > 60)
			{
        		lrate *= ANNEAL;
				// oldDWeights = dWeights;
				clEnqueueCopyBuffer(commandQueue, d_d_Weights, d_Old_d_Weights, 0, 0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(double), 0, NULL, NULL);
			} 
			else if (step == 1) 
			{
				// oldDWeights = dWeights;
				clEnqueueCopyBuffer(commandQueue, d_d_Weights, d_Old_d_Weights, 0, 0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(double), 0, NULL, NULL);
			}

			printf("\n\nStep %zu: Lrate %.1e, Wchange %.1e, Angle %.2f \n\n", step, lrate, change, angleDelta);

			step++;
    	}
  	}

	clEnqueueReadBuffer(commandQueue, d_Weights, CL_TRUE, 0, NUMBER_OF_ICA_COMPONENTS * NUMBER_OF_ICA_COMPONENTS * sizeof(double), weights.data(), 0, NULL, NULL);

	sourceMatrix = weights * whitenedData;	

	// C = alpha * A * B  + beta * C                                           rows in d_Weights  columns in d_Whitened_Data   columns in d_Weights   alpha   A matrix                                  B matrix                                   beta     C matrix
 	//error = clblasSgemm (clblasColumnMajor, clblasNoTrans, clblasNoTrans, NUMBER_OF_ICA_COMPONENTS, NUMBER_OF_ICA_VARIABLES, NUMBER_OF_ICA_COMPONENTS, 1.0f, d_Weights, 0, NUMBER_OF_ICA_COMPONENTS, d_Whitened_Data, 0, NUMBER_OF_ICA_COMPONENTS, 0.0f, d_Source_Matrix, 0, NUMBER_OF_ICA_COMPONENTS, 1, &commandQueue, 0, NULL, NULL);

	clReleaseMemObject(d_Whitened_Data);
	clReleaseMemObject(d_Shuffled_Whitened_Data);

	clReleaseMemObject(d_Weights);

	clReleaseMemObject(d_Bias);
	clReleaseMemObject(d_Old_Weights);
	clReleaseMemObject(d_d_Weights);
	clReleaseMemObject(d_Old_d_Weights);
	clReleaseMemObject(d_Temp);
	clReleaseMemObject(d_Scratch);
	clReleaseMemObject(d_Double);
	clReleaseMemObject(d_Ones);

	clReleaseMemObject(d_Permutation);
}
#elif __APPLE__
void BROCCOLI_LIB::InfomaxICADouble(Eigen::MatrixXd & whitenedData, Eigen::MatrixXd & weights, Eigen::MatrixXd & sourceMatrix)
{
	printf("Currently it is only possible to use the -cpu option for ICA on Mac platforms\n");
}
#endif


void BROCCOLI_LIB::InfomaxICAEigen(Eigen::MatrixXd & whitenedData, Eigen::MatrixXd & weights, Eigen::MatrixXd & sourceMatrix)
{
  	// Computes ICA infomax in whitened data
    //	Decomposes x_white as x_white=AS
    //	*Input
    //	x_white: whitened data (Use PCAwhiten)
    //	*Output
    //	A : mixing matrix
    //	S : source matrix
  	
	double EPS = 1e-18;
	double MAX_W = 1.0e8;
	double ANNEAL = 0.9;
	double MIN_LRATE = 1e-6;
	double W_STOP = 1e-8;
	size_t MAX_STEP= 100;

	Eigen::MatrixXd bias(NUMBER_OF_ICA_COMPONENTS,1);

	Eigen::MatrixXd oldWeights(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_COMPONENTS);
	Eigen::MatrixXd dWeights(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_COMPONENTS);
	Eigen::MatrixXd oldDWeights(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_COMPONENTS);
	Eigen::MatrixXd temp(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_COMPONENTS);
	Eigen::MatrixXd shuffledWhitenedData(whitenedData.rows(),whitenedData.cols());

	shuffledWhitenedData = whitenedData;

	IdentityEigenMatrix(weights);
	IdentityEigenMatrix(oldWeights);

	ResetEigenMatrix(bias);
	ResetEigenMatrix(dWeights);
	ResetEigenMatrix(oldDWeights);
	ResetEigenMatrix(temp);

	double lrate = 0.00005/log((double)NUMBER_OF_ICA_COMPONENTS);
	double change = 1.0;
	double angleDelta = 0.0;
    size_t step = 1;
	int error = 0;

	while( (step < MAX_STEP) && (change > W_STOP))
	{
		double start = GetTime();
	    error = UpdateInfomaxWeightsEigen(weights, whitenedData, bias, shuffledWhitenedData, lrate);
		double end = GetTime();

		if (VERBOS)
		{
			printf("One iteration took %f seconds \n",(float)(end-start));
		}

		if (error == 1 || error == 2)
		{
			// It blowed up! RESTART!
    	  	step = 1;
    	  	// change = 1;
    	  	error = 0;
    	 	lrate *= ANNEAL;
		
			IdentityEigenMatrix(weights);
			IdentityEigenMatrix(oldWeights);

			ResetEigenMatrix(dWeights);
			ResetEigenMatrix(oldDWeights);
			ResetEigenMatrix(bias);
			
			if (lrate > MIN_LRATE)
			{
    	    	printf("\nLowering learning rate to %g and starting again.\n",lrate);
    	  	}
    	  	else
			{
		        printf("\nMatrix may not be invertible");
			}
    	}
    	else if (error == 0)
		{
			dWeights = weights;	
			dWeights -= oldWeights;
		    change = dWeights.squaredNorm();
	
			if (step > 2)
			{
		        // Compute angle delta
				temp = dWeights;
				// Pointwise multiplication
				temp = temp.array() * oldDWeights.array();

		        angleDelta = acos(temp.sum() / (dWeights.norm() * oldDWeights.norm()) );
        		angleDelta *= (180.0 / M_PI);
			}

			oldWeights = weights;

			if (angleDelta > 60)
			{
        		lrate *= ANNEAL;
				oldDWeights = dWeights;
			} 
			else if (step == 1) 
			{
				oldDWeights = dWeights;
			}

			printf("\nStep %zu: Lrate %.1e, Wchange %.1e, Angle %.2f \n", step, lrate, change, angleDelta);

			step++;
    	}
  	}

	sourceMatrix = weights * whitenedData;	
}


void BROCCOLI_LIB::InfomaxICAEigen(Eigen::MatrixXf & whitenedData, Eigen::MatrixXf & weights, Eigen::MatrixXf & sourceMatrix)
{
  	// Computes ICA infomax in whitened data
    //	Decomposes x_white as x_white=AS
    //	*Input
    //	x_white: whitened data (Use PCAwhiten)
    //	*Output
    //	A : mixing matrix
    //	S : source matrix
  	
	double EPS = 1e-18;
	double MAX_W = 1.0e8;
	double ANNEAL = 0.9;
	double MIN_LRATE = 1e-6;
	double W_STOP = 1e-8;
	size_t MAX_STEP= 100;

	Eigen::MatrixXf bias(NUMBER_OF_ICA_COMPONENTS,1);

	Eigen::MatrixXf oldWeights(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_COMPONENTS);
	Eigen::MatrixXf dWeights(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_COMPONENTS);
	Eigen::MatrixXf oldDWeights(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_COMPONENTS);
	Eigen::MatrixXf temp(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_COMPONENTS);
	Eigen::MatrixXf shuffledWhitenedData(whitenedData.rows(),whitenedData.cols());

	shuffledWhitenedData = whitenedData;

	IdentityEigenMatrix(weights);
	IdentityEigenMatrix(oldWeights);

	ResetEigenMatrix(bias);
	ResetEigenMatrix(dWeights);
	ResetEigenMatrix(oldDWeights);
	ResetEigenMatrix(temp);

	double lrate = 0.00005/log((double)NUMBER_OF_ICA_COMPONENTS);
	double change = 1.0;
	double angleDelta = 0.0;
    size_t step = 1;
	int error = 0;

	while( (step < MAX_STEP) && (change > W_STOP))
	{
		double start = GetTime();
	    error = UpdateInfomaxWeightsEigen(weights, whitenedData, bias, shuffledWhitenedData, lrate);
		double end = GetTime();

		if (VERBOS)
		{
			printf("One iteration took %f seconds \n",(float)(end-start));
		}

		if (error == 1 || error == 2)
		{
			// It blowed up! RESTART!
    	  	step = 1;
    	  	// change = 1;
    	  	error = 0;
    	 	lrate *= ANNEAL;
		
			IdentityEigenMatrix(weights);
			IdentityEigenMatrix(oldWeights);

			ResetEigenMatrix(dWeights);
			ResetEigenMatrix(oldDWeights);
			ResetEigenMatrix(bias);
			
			if (lrate > MIN_LRATE)
			{
    	    	printf("\nLowering learning rate to %g and starting again.\n",lrate);
    	  	}
    	  	else
			{
		        printf("\nMatrix may not be invertible");
			}
    	}
    	else if (error == 0)
		{
			dWeights = weights;	
			dWeights -= oldWeights;
		    change = dWeights.squaredNorm();
	
			if (step > 2)
			{
		        // Compute angle delta
				temp = dWeights;
				// Pointwise multiplication
				temp = temp.array() * oldDWeights.array();

		        angleDelta = acos(temp.sum() / (dWeights.norm() * oldDWeights.norm()) );
        		angleDelta *= (180.0 / M_PI);
			}

			oldWeights = weights;

			if (angleDelta > 60)
			{
        		lrate *= ANNEAL;
				oldDWeights = dWeights;
			} 
			else if (step == 1) 
			{
				oldDWeights = dWeights;
			}

			printf("\nStep %zu: Lrate %.1e, Wchange %.1e, Angle %.2f \n", step, lrate, change, angleDelta);

			step++;
    	}
  	}

	sourceMatrix = weights * whitenedData;	
}




void BROCCOLI_LIB::PerformICACPUWrapper()
{
	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	if (!AUTO_MASK)
	{
		// Copy mask from host
		clEnqueueWriteBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask, 0, NULL, NULL);
	}
	else
	{
		// Make a mask
		SegmentEPIData();
		// Copy mask to host
		clEnqueueReadBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask, 0, NULL, NULL);
	}

	//--------------------------

	// Loop through mask to get number of voxels
	NUMBER_OF_ICA_VARIABLES = 0;
	for (int v = 0; v < EPI_DATA_W * EPI_DATA_H * EPI_DATA_D; v++)
	{
		if (h_EPI_Mask[v] == 1.0f)
		{
			NUMBER_OF_ICA_VARIABLES++;		
		}
	}

	NUMBER_OF_ICA_OBSERVATIONS = EPI_DATA_T;

	Eigen::MatrixXf inputData(NUMBER_OF_ICA_OBSERVATIONS,NUMBER_OF_ICA_VARIABLES);

	if (WRAPPER == BASH)
	{
		printf("Original number of voxels is %zu, reduced to %zu voxels using a mask\n",EPI_DATA_W*EPI_DATA_H*EPI_DATA_D,NUMBER_OF_ICA_VARIABLES);
	}

	// Put data into Eigen object
	int v = 0;
	for (int z = 0; z < EPI_DATA_D; z++)
	{
		for (int y = 0; y < EPI_DATA_H; y++)
		{
			for (int x = 0; x < EPI_DATA_W; x++)
			{
				if (h_EPI_Mask[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H] == 1.0f)
				{
					if (Z_SCORE)
					{
						// z-score each time series

						// Estimate mean
						float sum = 0.0f;
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							sum += h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
						}
						float mean = sum/(float)EPI_DATA_T;

						// Remove mean
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] -= mean;
						}				

						// Estimate variance					
						sum = 0.0f;
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							float value = h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
							sum += value * value;
						}
						float variance = sum/(float)(EPI_DATA_T-1);
						float std = sqrt(variance);

						// Divide by standard deviation
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] /= std;
						}
					}
	
					for (int t = 0; t < EPI_DATA_T; t++)
					{
						inputData(t,v) = h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
					}
					
					v++;
				}				
			}
		}
	}


	// First whiten the data and reduce the number of dimensions
	Eigen::MatrixXf whitenedData = PCAWhitenEigen(inputData, true);
	
	//Eigen::MatrixXd whitenedData(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_VARIABLES);
	//PCAWhitenEigen(whitenedData,  inputData, NUMBER_OF_ICA_COMPONENTS, true);
	//PCADimensionalityReduction(whitenedData,  inputData, NUMBER_OF_ICA_COMPONENTS, true);

	Eigen::MatrixXf weights(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_COMPONENTS);
	Eigen::MatrixXf sourceMatrix(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_VARIABLES);

	// Run the actual ICA algorithm
	InfomaxICAEigen(whitenedData, weights, sourceMatrix);

	//Eigen::MatrixXd inverseWeights = weights.inverse();

	// Put components back into fMRI volumes
	v = 0;
	for (int z = 0; z < EPI_DATA_D; z++)
	{
		for (int y = 0; y < EPI_DATA_H; y++)
		{
			for (int x = 0; x < EPI_DATA_W; x++)
			{
				if (h_EPI_Mask[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H] == 1.0f)
				{					
					for (int t = 0; t < NUMBER_OF_ICA_COMPONENTS; t++)
					{
						//h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] = whitenedData(t,x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H);
						h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] = sourceMatrix(t,v);
					}
					v++;
				}
				else
				{
					for (int t = 0; t < NUMBER_OF_ICA_COMPONENTS; t++)				
					{
						h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] = 0.0f;
					}
				}
			}
		}
	}

	clReleaseMemObject(d_EPI_Mask);
}

void BROCCOLI_LIB::PerformICADoubleCPUWrapper()
{
	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	if (!AUTO_MASK)
	{
		// Copy mask from host
		clEnqueueWriteBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask, 0, NULL, NULL);
	}
	else
	{
		// Make a mask
		SegmentEPIData();
		// Copy mask to host
		clEnqueueReadBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask, 0, NULL, NULL);
	}

	//--------------------------

	// Loop through mask to get number of voxels
	NUMBER_OF_ICA_VARIABLES = 0;
	for (int v = 0; v < EPI_DATA_W * EPI_DATA_H * EPI_DATA_D; v++)
	{
		if (h_EPI_Mask[v] == 1.0f)
		{
			NUMBER_OF_ICA_VARIABLES++;		
		}
	}

	NUMBER_OF_ICA_OBSERVATIONS = EPI_DATA_T;

	Eigen::MatrixXf inputData(NUMBER_OF_ICA_OBSERVATIONS,NUMBER_OF_ICA_VARIABLES);

	if (WRAPPER == BASH)
	{
		printf("Original number of voxels is %zu, reduced to %zu voxels using a mask\n",EPI_DATA_W*EPI_DATA_H*EPI_DATA_D,NUMBER_OF_ICA_VARIABLES);
	}

	// Put data into Eigen object
	int v = 0;
	for (int z = 0; z < EPI_DATA_D; z++)
	{
		for (int y = 0; y < EPI_DATA_H; y++)
		{
			for (int x = 0; x < EPI_DATA_W; x++)
			{
				if (h_EPI_Mask[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H] == 1.0f)
				{
					if (Z_SCORE)
					{
						// z-score each time series

						// Estimate mean
						float sum = 0.0f;
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							sum += h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
						}
						float mean = sum/(float)EPI_DATA_T;

						// Remove mean
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] -= mean;
						}				

						// Estimate variance					
						sum = 0.0f;
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							float value = h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
							sum += value * value;
						}
						float variance = sum/(float)(EPI_DATA_T-1);
						float std = sqrt(variance);

						// Divide by standard deviation
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] /= std;
						}
					}
	
					for (int t = 0; t < EPI_DATA_T; t++)
					{
						inputData(t,v) = h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
					}
					
					v++;
				}				
			}
		}
	}


	// First whiten the data and reduce the number of dimensions
	Eigen::MatrixXf whitenedData = PCAWhitenEigen(inputData, true);
	
	Eigen::MatrixXd weightsDouble(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_COMPONENTS);
	Eigen::MatrixXd sourceMatrixDouble(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_VARIABLES);

	Eigen::MatrixXd whitenedDataDouble = whitenedData.cast<double>();

	// Run the actual ICA algorithm
	InfomaxICAEigen(whitenedDataDouble, weightsDouble, sourceMatrixDouble);

	Eigen::MatrixXf sourceMatrix = sourceMatrixDouble.cast<float>();

	// Put components back into fMRI volumes
	v = 0;
	for (int z = 0; z < EPI_DATA_D; z++)
	{
		for (int y = 0; y < EPI_DATA_H; y++)
		{
			for (int x = 0; x < EPI_DATA_W; x++)
			{
				if (h_EPI_Mask[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H] == 1.0f)
				{					
					for (int t = 0; t < NUMBER_OF_ICA_COMPONENTS; t++)
					{
						//h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] = whitenedData(t,x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H);
						h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] = sourceMatrix(t,v);
					}
					v++;
				}
				else
				{
					for (int t = 0; t < NUMBER_OF_ICA_COMPONENTS; t++)				
					{
						h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] = 0.0f;
					}
				}
			}
		}
	}

	clReleaseMemObject(d_EPI_Mask);
}


void BROCCOLI_LIB::PerformICAWrapper()
{
	#ifdef __linux
	// Initiate clBLAS
	error = clblasSetup();
    if (error != CL_SUCCESS) 
	{
        printf("clblasSetup() failed with %s\n", GetOpenCLErrorMessage(error));
    }
	#endif

	//--------------------------

	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	if (!AUTO_MASK)
	{
		// Copy mask from host
		clEnqueueWriteBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask, 0, NULL, NULL);
	}
	else
	{
		// Make a mask
		SegmentEPIData();
		// Copy mask to host
		clEnqueueReadBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask, 0, NULL, NULL);
	}

	//--------------------------

	// Loop through mask to get number of voxels
	NUMBER_OF_ICA_VARIABLES = 0;
	for (int v = 0; v < EPI_DATA_W * EPI_DATA_H * EPI_DATA_D; v++)
	{
		if (h_EPI_Mask[v] == 1.0f)
		{
			NUMBER_OF_ICA_VARIABLES++;		
		}
	}

	NUMBER_OF_ICA_OBSERVATIONS = EPI_DATA_T;

	Eigen::MatrixXf inputData(NUMBER_OF_ICA_OBSERVATIONS,NUMBER_OF_ICA_VARIABLES);

	if (WRAPPER == BASH)
	{
		printf("Original number of voxels is %zu, reduced to %zu voxels using a mask\n",EPI_DATA_W*EPI_DATA_H*EPI_DATA_D,NUMBER_OF_ICA_VARIABLES);
	}

	// Put data into Eigen object
	int v = 0;
	for (int z = 0; z < EPI_DATA_D; z++)
	{
		for (int y = 0; y < EPI_DATA_H; y++)
		{
			for (int x = 0; x < EPI_DATA_W; x++)
			{
				if (h_EPI_Mask[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H] == 1.0f)
				{
					// z-score each time series
					if (Z_SCORE)
					{

						// Estimate mean
						float sum = 0.0f;
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							sum += h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
						}
						float mean = sum/(float)EPI_DATA_T;
	
						// Remove mean
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] -= mean;
						}				

						// Estimate variance					
						sum = 0.0f;
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							float value = h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
							sum += value * value;
						}
						float variance = sum/(float)(EPI_DATA_T-1);
						float std = sqrt(variance);

						// Divide by standard deviation
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] /= std;
						}	
					}

					for (int t = 0; t < EPI_DATA_T; t++)
					{
						inputData(t,v) = h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
					}

					v++;
				}				
			}
		}
	}

	//Eigen::MatrixXd whitenedData(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_VOXELS);


	// First whiten the data and reduce the number of dimensions
	Eigen::MatrixXf whitenedData = PCAWhiten(inputData, true);
	//PCAWhiten(whitenedData,  inputData, NUMBER_OF_ICA_COMPONENTS, true);
	//PCADimensionalityReduction(whitenedData,  inputData, NUMBER_OF_ICA_COMPONENTS, true);

	Eigen::MatrixXf weights(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_COMPONENTS);
	Eigen::MatrixXf sourceMatrix(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_VARIABLES);

	// Run the actual ICA algorithm
	InfomaxICA(whitenedData, weights, sourceMatrix);

	//Eigen::MatrixXd inverseWeights = weights.inverse();

	// Put components back into fMRI volumes
	v = 0;
	for (int z = 0; z < EPI_DATA_D; z++)
	{
		for (int y = 0; y < EPI_DATA_H; y++)
		{
			for (int x = 0; x < EPI_DATA_W; x++)
			{
				if (h_EPI_Mask[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H] == 1.0f)
				{					
					for (int t = 0; t < NUMBER_OF_ICA_COMPONENTS; t++)
					{
						//h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] = whitenedData(t,x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H);
						h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] = sourceMatrix(t,v);
					}
					v++;
				}
				else
				{
					for (int t = 0; t < NUMBER_OF_ICA_COMPONENTS; t++)				
					{
						h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] = 0.0f;
					}
				}
			}
		}
	}

	clReleaseMemObject(d_EPI_Mask);

	#ifdef __linux
	// Stop clBLAS
	clblasTeardown();
	#endif
}




void BROCCOLI_LIB::PerformICADoubleWrapper()
{
	#ifdef __linux
	// Initiate clBLAS
	error = clblasSetup();
    if (error != CL_SUCCESS) 
	{
        printf("clblasSetup() failed with %s\n", GetOpenCLErrorMessage(error));
    }
	#endif

	//--------------------------

	d_EPI_Mask = clCreateBuffer(context, CL_MEM_READ_WRITE, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), NULL, NULL);

	if (!AUTO_MASK)
	{
		// Copy mask from host
		clEnqueueWriteBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask, 0, NULL, NULL);
	}
	else
	{
		// Make a mask
		SegmentEPIData();
		// Copy mask to host
		clEnqueueReadBuffer(commandQueue, d_EPI_Mask, CL_TRUE, 0, EPI_DATA_W * EPI_DATA_H * EPI_DATA_D * sizeof(float), h_EPI_Mask, 0, NULL, NULL);
	}

	//--------------------------

	// Loop through mask to get number of voxels
	NUMBER_OF_ICA_VARIABLES = 0;
	for (int v = 0; v < EPI_DATA_W * EPI_DATA_H * EPI_DATA_D; v++)
	{
		if (h_EPI_Mask[v] == 1.0f)
		{
			NUMBER_OF_ICA_VARIABLES++;		
		}
	}

	NUMBER_OF_ICA_OBSERVATIONS = EPI_DATA_T;

	Eigen::MatrixXf inputData(NUMBER_OF_ICA_OBSERVATIONS,NUMBER_OF_ICA_VARIABLES);

	if (WRAPPER == BASH)
	{
		printf("Original number of voxels is %zu, reduced to %zu voxels using a mask\n",EPI_DATA_W*EPI_DATA_H*EPI_DATA_D,NUMBER_OF_ICA_VARIABLES);
	}

	// Put data into Eigen object
	int v = 0;
	for (int z = 0; z < EPI_DATA_D; z++)
	{
		for (int y = 0; y < EPI_DATA_H; y++)
		{
			for (int x = 0; x < EPI_DATA_W; x++)
			{
				if (h_EPI_Mask[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H] == 1.0f)
				{
					// z-score each time series
					if (Z_SCORE)
					{

						// Estimate mean
						float sum = 0.0f;
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							sum += h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
						}
						float mean = sum/(float)EPI_DATA_T;
	
						// Remove mean
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] -= mean;
						}				

						// Estimate variance					
						sum = 0.0f;
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							float value = h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
							sum += value * value;
						}
						float variance = sum/(float)(EPI_DATA_T-1);
						float std = sqrt(variance);

						// Divide by standard deviation
						for (int t = 0; t < EPI_DATA_T; t++)
						{
							h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] /= std;
						}	
					}

					for (int t = 0; t < EPI_DATA_T; t++)
					{
						inputData(t,v) = h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D];
					}

					v++;
				}				
			}
		}
	}

	//Eigen::MatrixXd whitenedData(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_VOXELS);


	// First whiten the data and reduce the number of dimensions
	Eigen::MatrixXf whitenedData = PCAWhiten(inputData, true);
	//PCAWhiten(whitenedData,  inputData, NUMBER_OF_ICA_COMPONENTS, true);
	//PCADimensionalityReduction(whitenedData,  inputData, NUMBER_OF_ICA_COMPONENTS, true);

	Eigen::MatrixXd weightsDouble(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_COMPONENTS);
	Eigen::MatrixXd sourceMatrixDouble(NUMBER_OF_ICA_COMPONENTS,NUMBER_OF_ICA_VARIABLES);
	
	// Run the actual ICA algorithm
	Eigen::MatrixXd whitenedDataDouble = whitenedData.cast<double>();
	InfomaxICADouble(whitenedDataDouble, weightsDouble, sourceMatrixDouble);

	Eigen::MatrixXf sourceMatrix = sourceMatrixDouble.cast<float>();

	//Eigen::MatrixXd inverseWeights = weights.inverse();

	// Put components back into fMRI volumes
	v = 0;
	for (int z = 0; z < EPI_DATA_D; z++)
	{
		for (int y = 0; y < EPI_DATA_H; y++)
		{
			for (int x = 0; x < EPI_DATA_W; x++)
			{
				if (h_EPI_Mask[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H] == 1.0f)
				{					
					for (int t = 0; t < NUMBER_OF_ICA_COMPONENTS; t++)
					{
						//h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] = whitenedData(t,x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H);
						h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] = sourceMatrix(t,v);
					}
					v++;
				}
				else
				{
					for (int t = 0; t < NUMBER_OF_ICA_COMPONENTS; t++)				
					{
						h_fMRI_Volumes[x + y * EPI_DATA_W + z * EPI_DATA_W * EPI_DATA_H + t * EPI_DATA_W * EPI_DATA_H * EPI_DATA_D] = 0.0f;
					}
				}
			}
		}
	}

	clReleaseMemObject(d_EPI_Mask);

	#ifdef __linux
	// Stop clBLAS
	clblasTeardown();
	#endif
}

