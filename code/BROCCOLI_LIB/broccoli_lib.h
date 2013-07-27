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

#ifndef BROCCOLILIB_H
#define BROCCOLILIB_H

#include "nifti1.h"
#include "nifti1_io.h"

//#include <vector_types.h>

#include <opencl.h>

#include <string>

//#include <cstdlib>


typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short int uint16;

#define CCA 0
#define GLM 1

#define SILENT 0
#define VERBOSE 1

#define NO 0
#define YES 1

#define RAW 0
#define NIFTI 1

#define FLOAT 0
#define INT32 1
#define INT16 2
#define UINT32 3
#define UINT16 4
#define DOUBLE 5
#define UINT8 6

#define SLICE_TIMING_CORRECTION 0
#define MOTION_CORRECTION 1
#define SMOOTHING 2
#define DETRENDING 3
#define STATISTICAL_ANALYSIS 4
#define PERMUTATION_TEST 5
#define COPY 6
#define CONVOLVE 7
#define PHASEDC 8
#define PHASEG 9
#define AH2D 10
#define EQSYSTEM 11
#define INTERPOLATION 12

#define QF1R 0
#define QF1I 1
#define QF2R 2
#define QF2I 3
#define QF3R 4
#define QF3I 5
#define QF1  6
#define QF2  7
#define QF3  8
#define CCA3D1 14
#define CCA3D2 15
#define CCA3D 16
#define fMRI_VOLUMES 17
#define XDETREND1 18
#define XDETREND2 19
#define CXX 20
#define SQRTINVCXX 21
#define XGLM1 22
#define XGLM2 23
#define CONTRAST_VECTOR 24
#define ACTIVITY_VOLUME 25
#define BRAIN_VOXELS 26
#define MOTION_CORRECTED_VOLUMES 27
#define REGISTRATION_PARAMETERS 28
#define SMOOTHED1 29
#define SMOOTHED2 30
#define SMOOTHED3 31
#define SMOOTHED4 32
#define DETRENDED1 33
#define DETRENDED2 34
#define DETRENDED3 35
#define DETRENDED4 36
#define X_SLICE_fMRI 37
#define Y_SLICE_fMRI 38
#define Z_SLICE_fMRI 39
#define X_SLICE_PREPROCESSED_fMRI 40
#define Y_SLICE_PREPROCESSED_fMRI 41
#define Z_SLICE_PREPROCESSED_fMRI 42
#define X_SLICE_ACTIVITY 43
#define Y_SLICE_ACTIVITY 44
#define Z_SLICE_ACTIVITY 45
#define MOTION_PARAMETERS_X 46
#define MOTION_PARAMETERS_Y 47
#define MOTION_PARAMETERS_Z 48
#define PLOT_VALUES_X 49
#define MOTION_CORRECTED_CURVE 50
#define SMOOTHED_CURVE 51
#define DETRENDED_CURVE 52	
#define ALPHAS1 53
#define ALPHAS2 54
#define ALPHAS3 55
#define ALPHAS4 56
#define SMOOTHED_ALPHAS1 57
#define SMOOTHED_ALPHAS2 58
#define SMOOTHED_ALPHAS3 59
#define SMOOTHED_ALPHAS4 60
#define SMOOTHED_CERTAINTY 61
#define BOLD_REGRESSED_VOLUMES 62
#define WHITENED_VOLUMES 63
#define PERMUTED_VOLUMES 64
#define MAXIMUM_TEST_VALUES 65
#define PERMUTATION_MATRIX 66

#define NUMBER_OF_HOST_POINTERS 100
#define NUMBER_OF_DEVICE_POINTERS 100

#define VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_ROWS 32
#define VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_ROWS 8
#define VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_ROWS 8

#define VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_COLUMNS 24
#define VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_COLUMNS 16
#define VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_COLUMNS 8

#define VALID_FILTER_RESPONSES_X_SEPARABLE_CONVOLUTION_RODS 32
#define VALID_FILTER_RESPONSES_Y_SEPARABLE_CONVOLUTION_RODS 8
#define VALID_FILTER_RESPONSES_Z_SEPARABLE_CONVOLUTION_RODS 8

//struct float2 { float x; float y; };

#define HALO 3
#define VALID_FILTER_RESPONSES_X_CONVOLUTION_2D 90
#define VALID_FILTER_RESPONSES_Y_CONVOLUTION_2D 58

#define TRANSLATION 0
#define RIGID 1
#define AFFINE 2

#define NEAREST 0
#define LINEAR 1

#define DO_OVERWRITE 0
#define NO_OVERWRITE 1

#define PI 3.1415f

class BROCCOLI_LIB
{
	public:

		// Constructor & destructor
		BROCCOLI_LIB();
		BROCCOLI_LIB(cl_uint OPENCL_PLATFORM);
		~BROCCOLI_LIB();

		// Set functions for GUI / Wrappers
		
		void SetOpenCLPlatform(int N);
		void SetMask(float* input);
		void SetNumberOfRegressors(int NR);
		void SetNumberOfContrasts(int NC);
		void SetDesignMatrix(float* X_GLM, float* xtxxt_GLM);
		void SetContrasts(float* contrasts);
		void SetGLMScalars(float* ctxtxc);
		void SetSmoothingFilters(float* smoothing_filter_x,float* smoothing_filter_y,float* smoothing_filter_z);
		void SetImageRegistrationFilterSize(int N);
		void SetImageRegistrationFilters(float* qf1r, float* qf1i, float* qf2r, float* qf2i, float* qf3r, float* qf3i);
		void SetNumberOfIterationsForImageRegistration(int N);
		void SetNumberOfIterationsForMotionCorrection(int N);
		void SetCoarsestScaleT1MNI(int N);
		void SetCoarsestScaleEPIT1(int N);
		void SetMMT1ZCUT(int mm);
		void SetMMEPIZCUT(int mm);

		void SetInputfMRIVolumes(float* input);
		void SetInputEPIVolume(float* input);
		void SetInputT1Volume(float* input);
		void SetInputMNIVolume(float* input);
		void SetInputMNIBrainMask(float* input);
		void SetOutputBetaVolumes(float* output);
		void SetOutputResiduals(float* output);
		void SetOutputResidualVariances(float* output);
		void SetOutputStatisticalMaps(float* output);		
		void SetOutputMotionParameters(float* output);
		void SetOutputT1MNIRegistrationParameters(float* output);
		void SetOutputEPIT1RegistrationParameters(float* output);
		void SetOutputQuadratureFilterResponses(float* qfr1r, float* qfr1i, float* qfr2r, float* qfr2i, float* qfr3r, float* qfr3i);
		void SetOutputPhaseDifferences(float*);
		void SetOutputPhaseCertainties(float*);
		void SetOutputPhaseGradients(float*);
		void SetOutputAlignedT1Volume(float*);
		void SetOutputAlignedEPIVolume(float*);
		void SetOutputSkullstrippedT1Volume(float*);
		void SetOutputInterpolatedT1Volume(float*);
		void SetOutputInterpolatedEPIVolume(float*);
		void SetOutputDownsampledVolume(float*);
		void SetOutputMotionCorrectedfMRIVolumes(float*);
		void SetOutputSmoothedfMRIVolumes(float*);
		
		void SetfMRIDataFilename(std::string filename);
			
		void SetfMRIParameters(float tr, float xs, float ys, float zs);		
		void SetSmoothingAmount(int value);
		void SetSmoothingDimensionality(int dimensionality);
		void SetNumberOfBasisFunctionsDetrending(int N);
		void SetAnalysisMethod(int method);
		void SetWriteStatus(bool status);
		void SetShowPreprocessedType(int value);

		void SetActivityThreshold(float threshold);
		void SetThresholdStatus(bool status);

		void SetfMRIDataSliceLocationX(int location);
		void SetfMRIDataSliceLocationY(int location);
		void SetfMRIDataSliceLocationZ(int location);
		void SetfMRIDataSliceTimepoint(int timepoint);

		void SetDataType(int type);
		void SetFileType(int type);

		void SetEPIVoxelSizeX(float value);
		void SetEPIVoxelSizeY(float value);
		void SetEPIVoxelSizeZ(float value);
		void SetEPITR(float value);

		void SetT1VoxelSizeX(float value);
		void SetT1VoxelSizeY(float value);
		void SetT1VoxelSizeZ(float value);

		void SetMNIVoxelSizeX(float value);
		void SetMNIVoxelSizeY(float value);
		void SetMNIVoxelSizeZ(float value);

		void SetEPIWidth(int w);
		void SetEPIHeight(int h);
		void SetEPIDepth(int d);
		void SetEPITimepoints(int t);

		void SetT1Width(int w);
		void SetT1Height(int h);
		void SetT1Depth(int d);

		void SetMNIWidth(int w);
		void SetMNIHeight(int h);
		void SetMNIDepth(int d);

		void SetNumberOfPermutations(int value);
		void SetSignificanceThreshold(float value);

		// Get functions for GUI / Wrappers
		
		const char* GetOpenCLDeviceInfoChar();
		const char* GetOpenCLBuildInfoChar();
		
		std::string GetOpenCLDeviceInfoString();
		std::string GetOpenCLBuildInfoString();

		int GetOpenCLError();
		int GetOpenCLCreateKernelError();
		int* GetOpenCLCreateBufferErrors();
		int* GetOpenCLRunKernelErrors();

		double GetProcessingTimeSliceTimingCorrection();
		double GetProcessingTimeMotionCorrection();
		double GetProcessingTimeSmoothing();
		double GetProcessingTimeDetrending();
		double GetProcessingTimeStatisticalAnalysis();
		double GetProcessingTimePermutationTest();

		double GetProcessingTimeCopy();
		double GetProcessingTimeConvolution();
		double GetProcessingTimePhaseDifferences();
		double GetProcessingTimePhaseGradients();
		double GetProcessingTimeAH();
		double GetProcessingTimeEquationSystem();
		double GetProcessingTimeInterpolation();

		int GetfMRIDataSliceLocationX();
		int GetfMRIDataSliceLocationY();
		int GetfMRIDataSliceLocationZ();

		int GetEPIWidth();
		int GetEPIHeight();
		int GetEPIDepth();
		int GetEPITimepoints();

		int GetT1Width();
		int GetT1Height();
		int GetT1Depth();
		int GetT1Timepoints();

		float GetEPIVoxelSizeX();
		float GetEPIVoxelSizeY();
		float GetEPIVoxelSizeZ();
		float GetEPITR();

		float GetT1VoxelSizeX();
		float GetT1VoxelSizeY();
		float GetT1VoxelSizeZ();
		
		double* GetMotionParametersX();
		double* GetMotionParametersY();
		double* GetMotionParametersZ();
		double* GetPlotValuesX();

		double* GetMotionCorrectedCurve();
		double* GetSmoothedCurve();
		double* GetDetrendedCurve();
			
		float GetPermutationThreshold();

		unsigned char* GetZSlicefMRIData();
		unsigned char* GetYSlicefMRIData();
		unsigned char* GetXSlicefMRIData();
		unsigned char* GetZSlicePreprocessedfMRIData();
		unsigned char* GetYSlicePreprocessedfMRIData();
		unsigned char* GetXSlicePreprocessedfMRIData();
		unsigned char* GetZSliceActivityData();
		unsigned char* GetYSliceActivityData();
		unsigned char* GetXSliceActivityData();

		std::string GetfMRIDataFilename();

		int GetNumberOfSignificantlyActiveVoxels();
		int GetNumberOfSignificantlyActiveClusters();

		std::string PrintDeviceInfo();

		// Read functions
		void ReadfMRIDataRAW();
		void ReadfMRIDataNIFTI();
		void ReadNIFTIHeader();

		// Write functions
		void WritefMRIDataNIFTI();
		
		// Preprocessing
		void TransformVolume(cl_mem d_Volume, float* h_Registration_Parameters, int DATA_W, int DATA_H, int DATA_D, int INTERPOLATION_MODE);
		void TransformVolumes(cl_mem d_Volume, float* h_Registration_Parameters, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_VOLUMES, int INTERPOLATION_MODE);
		void PerformRegistrationEPIT1();
		void PerformRegistrationT1MNI();
		void PerformRegistrationT1MNIWrapper();
		void PerformRegistrationEPIT1Wrapper();
		void PerformSliceTimingCorrection();
		void PerformMotionCorrection();
		void PerformMotionCorrectionWrapper();
		void PerformDetrending();
		void PerformSmoothing(cl_mem Smoothed_Volumes, cl_mem d_Volumes, cl_mem c_Smoothing_Filter_X, cl_mem c_Smoothing_Filter_Y, cl_mem c_Smoothing_Filter_Z, int DATA_W, int DATA_H, int DATA_D, int DATA_T);
		void PerformSmoothing(cl_mem d_Volumes, cl_mem c_Smoothing_Filter_X, cl_mem c_Smoothing_Filter_Y, cl_mem c_Smoothing_Filter_Z, int DATA_W, int DATA_H, int DATA_D, int DATA_T);
		void PerformSmoothingWrapper();

		// Processing
		void PerformFirstLevelAnalysis();				
		void PerformFirstLevelAnalysisWrapper();				
		void CalculateStatisticalMapsGLMFirstLevel(cl_mem Volumes);
		void CalculateStatisticalMapsGLMSecondLevel();
		void CalculatePermutationTestThresholdSingleSubject();
		void CalculatePermutationTestThresholdMultiSubject();
		
		void TransformFirstLevelResultsToMNI();

		void PerformGLMWrapper();

		
		// Permutation single subject	
		void SetupParametersPermutationSingleSubject();
		void GeneratePermutationMatrixSingleSubject();
		void PerformDetrendingPriorPermutation();
		void CreateBOLDRegressedVolumes();
		void WhitenfMRIVolumes();
		void GeneratePermutedfMRIVolumes();
		void PerformDetrendingPermutation();
		void PerformSmoothingPermutation(); 
		void CalculateStatisticalMapPermutation();

		// Permutation multi subject	
		void SetupParametersPermutationMultiSubject();
		void GeneratePermutationMatrixMultiSubject();
		

		void CalculateGroupMapPermutation();
		float FindMaxTestvaluePermutation();
		
		void CalculateSlicesfMRIData();
		void CalculateSlicesPreprocessedfMRIData();
		void CalculateSlicesActivityData();

		void GetOpenCLInfo();
		void OpenCLInitiate(cl_uint OPENCL_PLATFORM);
		void OpenCLTest();

		void ChangeT1VolumeResolutionAndSizeWrapper();


	private:

		void SetMemory(cl_mem memory, float value, int N);
		void SetGlobalAndLocalWorkSizesSeparableConvolution(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesNonSeparableConvolution(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesImageRegistration(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesStatisticalCalculations(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesRescaleVolume(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesInterpolateVolume(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesCopyVolumeToNew(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesMemset(int N);
		void SetGlobalAndLocalWorkSizesMultiplyVolumes(int DATA_W, int DATA_H, int DATA_D);
		

		void PerformSkullstrip(cl_mem d_Skullstripped_T1_Volume, cl_mem d_Aligned_T1_Volume, cl_mem d_MNI_Brain_Mask, int DATA_W, int DATA_H, int DATA_D);

		void Copy3DFiltersToConstantMemory(int z, int FILTER_SIZE);
		void NonseparableConvolution3D(cl_mem d_q1_Real, cl_mem d_q1_Imag, cl_mem d_q2_Real, cl_mem d_q2_Imag, cl_mem d_q3_Real, cl_mem d_q3_Imag, cl_mem d_Volume, int DATA_W, int DATA_H, int DATA_D);
		void AlignTwoVolumes(float* h_Registration_Parameters, float* h_Rotations, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_ITERATIONS, int ALIGNMENT_TYPE);
		void AlignTwoVolumesSeveralScales(float *h_Registration_Parameters, float* h_Rotations, cl_mem d_Al_Volume, cl_mem d_Ref_Volume, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_SCALES, int NUMBER_OF_ITERATIONS, int ALIGNMENT_TYPE, int OVERWRITE);
		void AlignTwoVolumesCleanup();
		void AlignTwoVolumesSetup(int DATA_W, int DATA_H, int DATA_D);
		void ChangeT1VolumeResolutionAndSize(cl_mem d_MNI_T1_Volume, cl_mem d_T1_Volume, int T1_DATA_W, int T1_DATA_H, int T1_DATA_D, int MNI_DATA_W, int MNI_DATA_H, int MNI_DATA_D, float T1_VOXEL_SIZE_X, float T1_VOXEL_SIZE_Y, float T1_VOXEL_SIZE_Z, float MNI_VOXEL_SIZE_X, float MNI_VOXEL_SIZE_Y, float MNI_VOXEL_SIZE_Z);
		void ChangeEPIVolumeResolutionAndSize(cl_mem d_T1_EPI_Volume, cl_mem d_EPI_Volume, int EPI_DATA_W, int EPI_DATA_H, int EPI_DATA_D, int T1_DATA_W, int T1_DATA_H, int T1_DATA_D, float EPI_VOXEL_SIZE_X, float EPI_VOXEL_SIZE_Y, float EPI_VOXEL_SIZE_Z, float T1_VOXEL_SIZE_X, float T1_VOXEL_SIZE_Y, float T1_VOXEL_SIZE_Z);
		void ChangeVolumeSize(cl_mem d_Current_Aligned_Volume, cl_mem d_Aligned_Volume, int DATA_W, int DATA_H, int DATA_D, int CURRENT_DATA_W, int CURRENT_DATA_H, int CURRENT_DATA_D);       
		void ChangeVolumesResolutionAndSize(cl_mem d_New_Volumes, cl_mem d_Volumes, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_VOLUMES, int NEW_DATA_W, int NEW_DATA_H, int NEW_DATA_D, float VOXEL_SIZE_X, float VOXEL_SIZE_Y, float VOXEL_SIZE_Z, float NEW_VOXEL_SIZE_X, float NEW_VOXEL_SIZE_Y, float NEW_VOXEL_SIZE_Z, int MM_Z_CUT);

		// Read functions
		void ReadRealDataInt32(int* data, std::string filename, int N);
		void ReadRealDataInt16(short int* data, std::string filename, int N);
		void ReadRealDataUint32(unsigned int* data, std::string filename, int N);
		void ReadRealDataUint16(unsigned short int* data, std::string filename, int N);
		void ReadRealDataFloat(float* data, std::string filename, int N);
		void ReadRealDataDouble(double* data, std::string filename, int N);
		//void ReadComplexData(Complex* data, std::string real_filename, std::string imag_filename, int N);
		void ReadImageRegistrationFilters();
		void ReadSmoothingFilters();
		void SetupParametersReadData();

		// Write functions
		void WriteRealDataUint16(unsigned short int* data, std::string filename, int N);
		void WriteRealDataFloat(float* data, std::string filename, int N);
		void WriteRealDataDouble(double* data, std::string filename, int N);
		//void WriteComplexData(Complex* data, std::string real_filename, std::string imag_filename, int N);

		// OpenCL help functions		
		
		void OpenCLCleanup();

		// Other help functions
		void SetStartValues();
		void ResetAllPointers();
		void AllocateMemoryForFilters();

		//void ConvertRealToComplex(Complex* complex_data, float* real_data, int N);
		//void ExtractRealData(float* real_data, Complex* complex_data, int N);
		//void Convert4FloatToFloat4(float4* floats, float* float_1, float* float_2, float* float_3, float* float_4, int N);
		//void Convert2FloatToFloat2(float2* floats, float* float_1, float* float_2, int N);
		void InvertMatrix(float* inverse_matrix, float* matrix, int N);
		void CalculateMatrixSquareRoot(float* sqrt_matrix, float* matrix, int N);
		void SolveEquationSystem(float* h_A_matrix, float* h_inverse_A_matrix, float* h_h_vector, float* h_Parameter_Vector, int N);
		void CalculateRotationAnglesFromRotationMatrix(float* h_Rotations, float* h_Registration_Parameters);

		void SetupDetrendingBasisFunctions();
		void SetupStatisticalAnalysisBasisFunctions();
		void SegmentBrainData();
		float CalculateMax(float *data, int N);
		float CalculateMin(float *data, int N);
		float Gpdf(double value, double shape, double scale);
		float loggamma(int value);

		void ConvolveWithHRF(float* temp_GLM);
		void CreateHRF();

		void InvertAffineRegistrationParameters(float* h_Inverse_Parameters, float* h_Parameters);

		// SVD 3x3
		double cbrt(double x); 
		void RemoveTransformationScaling(float* h_Registration_Parameters);
		void Cross(double* z, const double* x, const double* y); 
		void Sort3(double* x);
		void Unit3(double* x); 
		void LDUBSolve3(double *x, const double *y, const double *LDU, const int *P);
		void MatMul3x3(double* C, const double* A, const double* B);
		void MatVec3(double* y, const double* A, const double* x);
		void A_Transpose_A3x3(double* AA, const double* A);
		void A_A_Transpose3x3(double* AA, const double* A);
		void Transpose3x3(double* A); 
		void SolveCubic(double* c);
		void LDU3(double* A, int* P);
		void SVD3x3(double* U, double* S, double* V, const double* A);


		// OpenCL

		cl_context context;
		cl_command_queue commandQueue;
		cl_program program;
		cl_device_id device;
		
		std::string device_info;
		std::string build_info;
		
		cl_uint OPENCL_PLATFORM;
		int OPENCL_INITIATED;

		// OpenCL kernels

		cl_kernel MemsetKernel;
		cl_kernel SeparableConvolutionRowsKernel, SeparableConvolutionColumnsKernel, SeparableConvolutionRodsKernel, NonseparableConvolution3DComplexKernel;				
		cl_kernel CalculateBetaValuesGLMKernel, CalculateStatisticalMapsGLMKernel, RemoveLinearFitKernel;

		cl_kernel CalculatePhaseDifferencesAndCertaintiesKernel, CalculatePhaseGradientsXKernel, CalculatePhaseGradientsYKernel, CalculatePhaseGradientsZKernel;
		cl_kernel CalculateAMatrixAndHVector2DValuesXKernel, CalculateAMatrixAndHVector2DValuesYKernel,CalculateAMatrixAndHVector2DValuesZKernel; 
		cl_kernel CalculateAMatrix1DValuesKernel, CalculateHVector1DValuesKernel, CalculateHVectorKernel, ResetAMatrixKernel, CalculateAMatrixKernel;
		cl_kernel InterpolateVolumeNearestKernel, InterpolateVolumeLinearKernel, InterpolateVolumeCubicKernel; 
		cl_kernel RescaleVolumeNearestKernel, RescaleVolumeLinearKernel, RescaleVolumeCubicKernel;
		cl_kernel CopyT1VolumeToMNIKernel, CopyEPIVolumeToT1Kernel, CopyVolumeToNewKernel;
		cl_kernel MultiplyVolumesKernel, MultiplyVolumesOverwriteKernel;

		cl_kernel EstimateAR4ModelsKernel, ApplyWhiteningAR4Kernel, GeneratePermutedfMRIVolumesAR4Kernel;

		cl_kernel	AddKernel;

		cl_int error, kernel_error;

		// Create kernel errors
		cl_int createKernelErrorMemset;
		cl_int createKernelErrorSeparableConvolutionRows, createKernelErrorSeparableConvolutionColumns, createKernelErrorSeparableConvolutionRods, createKernelErrorNonseparableConvolution3DComplex; 
		cl_int createKernelErrorCalculatePhaseDifferencesAndCertainties, createKernelErrorCalculatePhaseGradientsX, createKernelErrorCalculatePhaseGradientsY, createKernelErrorCalculatePhaseGradientsZ;
		cl_int createKernelErrorCalculateAMatrixAndHVector2DValuesX, createKernelErrorCalculateAMatrixAndHVector2DValuesY, createKernelErrorCalculateAMatrixAndHVector2DValuesZ;
		cl_int createKernelErrorCalculateAMatrix1DValues, createKernelErrorCalculateHVector1DValues;
		cl_int createKernelErrorCalculateAMatrix, createKernelErrorCalculateHVector;
		cl_int createKernelErrorInterpolateVolumeNearest, createKernelErrorInterpolateVolumeLinear,  createKernelErrorInterpolateVolumeCubic;
		cl_int createKernelErrorRescaleVolumeNearest, createKernelErrorRescaleVolumeLinear, createKernelErrorRescaleVolumeCubic;
		cl_int createKernelErrorCopyT1VolumeToMNI, createKernelErrorCopyEPIVolumeToT1, createKernelErrorCopyVolumeToNew;
		cl_int createKernelErrorMultiplyVolumes;
		cl_int createKernelErrorMultiplyVolumesOverwrite;
		cl_int createKernelErrorCalculateBetaValuesGLM, createKernelErrorCalculateStatisticalMapsGLM;

		size_t threadsX, threadsY, threadsZ, xBlocks, yBlocks, zBlocks;

		// Create buffer errors
		cl_int createBufferErrorAlignedVolume, createBufferErrorReferenceVolume;
		cl_int createBufferErrorq11Real, createBufferErrorq11Imag, createBufferErrorq12Real, createBufferErrorq12Imag, createBufferErrorq13Real, createBufferErrorq13Imag;
		cl_int createBufferErrorq21Real, createBufferErrorq21Imag, createBufferErrorq22Real, createBufferErrorq22Imag, createBufferErrorq23Real, createBufferErrorq23Imag;
		cl_int createBufferErrorPhaseDifferences, createBufferErrorPhaseCertainties, createBufferErrorPhaseGradients;
		cl_int createBufferErrorAMatrix, createBufferErrorHVector, createBufferErrorAMatrix2DValues, createBufferErrorAMatrix1DValues, createBufferErrorHVector2DValues, createBufferErrorHVector1DValues;
		cl_int createBufferErrorQuadratureFilter1Real, createBufferErrorQuadratureFilter1Imag, createBufferErrorQuadratureFilter2Real, createBufferErrorQuadratureFilter2Imag, createBufferErrorQuadratureFilter3Real, createBufferErrorQuadratureFilter3Imag;    
		cl_int createBufferErrorRegistrationParameters;
		cl_int createBufferErrorBetaVolumesMNI;

		// Run kernel errors
		cl_int runKernelErrorNonseparableConvolution3DComplex;
		cl_int runKernelErrorMemset;
		cl_int runKernelErrorCalculatePhaseDifferencesAndCertainties, runKernelErrorCalculatePhaseGradientsX, runKernelErrorCalculatePhaseGradientsY, runKernelErrorCalculatePhaseGradientsZ;
		cl_int runKernelErrorCalculateAMatrixAndHVector2DValuesX, runKernelErrorCalculateAMatrixAndHVector2DValuesY, runKernelErrorCalculateAMatrixAndHVector2DValuesZ;
		cl_int runKernelErrorCalculateAMatrix1DValues, runKernelErrorCalculateHVector1DValues;
		cl_int runKernelErrorCalculateAMatrix, runKernelErrorCalculateHVector;
		cl_int runKernelErrorInterpolateVolume;
		cl_int runKernelErrorMultiplyVolumes;

		int OpenCLCreateBufferErrors[31];
		int OpenCLRunKernelErrors[30];

		double convolution_time;
		cl_event event;
		cl_ulong time_start, time_end;

		// OpenCL local work sizes

		size_t localWorkSizeMemset[3];
		size_t localWorkSizeSeparableConvolutionRows[3];
		size_t localWorkSizeSeparableConvolutionColumns[3];
		size_t localWorkSizeSeparableConvolutionRods[3];
		size_t localWorkSizeNonseparableConvolution3DComplex[3];
		
		size_t localWorkSizeCalculatePhaseDifferencesAndCertainties[3];
		size_t localWorkSizeCalculatePhaseGradients[3];
		size_t localWorkSizeCalculateAMatrixAndHVector2DValuesX[3];
		size_t localWorkSizeCalculateAMatrixAndHVector2DValuesY[3];
		size_t localWorkSizeCalculateAMatrixAndHVector2DValuesZ[3];
		size_t localWorkSizeCalculateAMatrix1DValues[3];
		size_t localWorkSizeCalculateHVector1DValues[3];
		size_t localWorkSizeResetAMatrix[3];
		size_t localWorkSizeCalculateAMatrix[3];
		size_t localWorkSizeCalculateHVector[3];
		size_t localWorkSizeRescaleVolumeLinear[3];
		size_t localWorkSizeRescaleVolumeCubic[3];
		size_t localWorkSizeInterpolateVolumeNearest[3];
		size_t localWorkSizeInterpolateVolumeLinear[3];
		size_t localWorkSizeInterpolateVolumeCubic[3];
		size_t localWorkSizeMultiplyVolumes[3];
		size_t localWorkSizeCopyVolumeToNew[3];

		size_t localWorkSizeCalculateBetaValuesGLM[3];		
		size_t localWorkSizeCalculateStatisticalMapsGLM[3];
		size_t localWorkSizeRemoveLinearFit[3];

		size_t localWorkSizeEstimateAR4Models[3];
		size_t localWorkSizeApplyWhiteningAR4[3];
		size_t localWorkSizeGeneratePermutedfMRIVolumesAR4[3];

		// OpenCL global work sizes

		size_t globalWorkSizeMemset[3];

		size_t globalWorkSizeSeparableConvolutionRows[3];
		size_t globalWorkSizeSeparableConvolutionColumns[3];
		size_t globalWorkSizeSeparableConvolutionRods[3];
		size_t globalWorkSizeNonseparableConvolution3DComplex[3];
		
		size_t globalWorkSizeCalculatePhaseDifferencesAndCertainties[3];
		size_t globalWorkSizeCalculatePhaseGradients[3];
		size_t globalWorkSizeCalculateAMatrixAndHVector2DValuesX[3];
		size_t globalWorkSizeCalculateAMatrixAndHVector2DValuesY[3];
		size_t globalWorkSizeCalculateAMatrixAndHVector2DValuesZ[3];
		size_t globalWorkSizeCalculateAMatrix1DValues[3];
		size_t globalWorkSizeCalculateHVector1DValues[3];
		size_t globalWorkSizeResetAMatrix[3];
		size_t globalWorkSizeCalculateAMatrix[3];
		size_t globalWorkSizeCalculateHVector[3];
		size_t globalWorkSizeRescaleVolumeLinear[3];
		size_t globalWorkSizeRescaleVolumeCubic[3];
		size_t globalWorkSizeInterpolateVolumeNearest[3];
		size_t globalWorkSizeInterpolateVolumeLinear[3];
		size_t globalWorkSizeInterpolateVolumeCubic[3];
		size_t globalWorkSizeMultiplyVolumes[3];
		size_t globalWorkSizeCopyVolumeToNew[3];

		size_t globalWorkSizeCalculateBetaValuesGLM[3];
		size_t globalWorkSizeCalculateStatisticalMapsGLM[3];
		size_t globalWorkSizeRemoveLinearFit[3];

		size_t globalWorkSizeEstimateAR4Models[3];
		size_t globalWorkSizeApplyWhiteningAR4[3];
		size_t globalWorkSizeGeneratePermutedfMRIVolumesAR4[3];

		// General
		int FILE_TYPE, DATA_TYPE;
		nifti_image *nifti_data;

		int EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T;
		int T1_DATA_W, T1_DATA_H, T1_DATA_D;
		int MNI_DATA_W, MNI_DATA_H, MNI_DATA_D;
		int CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D;
				
		int NUMBER_OF_SUBJECTS;
		int NUMBER_OF_CONTRASTS;
		int NUMBER_OF_REGRESSORS;
		float SEGMENTATION_THRESHOLD;
		float EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z;
		float T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z;
		float MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z;
		int PRINT;
		int WRITE_DATA;
		double processing_times[20];
		int PREPROCESSED;

		int X_SLICE_LOCATION_fMRI_DATA, Y_SLICE_LOCATION_fMRI_DATA, Z_SLICE_LOCATION_fMRI_DATA, TIMEPOINT_fMRI_DATA;

		unsigned char* x_slice_fMRI_data;
		unsigned char* y_slice_fMRI_data;
		unsigned char* z_slice_fMRI_data;
		unsigned char* x_slice_preprocessed_fMRI_data;
		unsigned char* y_slice_preprocessed_fMRI_data;
		unsigned char* z_slice_preprocessed_fMRI_data;
		unsigned char* x_slice_activity_data;
		unsigned char* y_slice_activity_data;
		unsigned char* z_slice_activity_data;

		double* plot_values_x;

		// Slice timing correction
		float TR;
		bool SLICE_TIMING_CORRECTED;

		// Image registration
		bool MOTION_CORRECTED;
		int IMAGE_REGISTRATION_FILTER_SIZE;
		int NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS;
		int NUMBER_OF_ITERATIONS_FOR_IMAGE_REGISTRATION;
		int NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION;
		int COARSEST_SCALE_T1_MNI, COARSEST_SCALE_EPI_T1;
		int MM_T1_Z_CUT, MM_EPI_Z_CUT;
		int	NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS;
		
		double* motion_parameters_x;
		double* motion_parameters_y;
		double* motion_parameters_z;

		double* motion_corrected_curve;

		// Smoothing
		int	SMOOTHING_FILTER_SIZE;
		int	SMOOTHING_AMOUNT_MM;
			
		double* smoothed_curve;

		int xBlockDifference, yBlockDifference, zBlockDifference;

		// Detrending
		int NUMBER_OF_DETRENDING_BASIS_FUNCTIONS;

		double* detrended_curve;
				 
		// Statistical analysis
		bool THRESHOLD_ACTIVITY_MAP;
		float ACTIVITY_THRESHOLD;
		int ANALYSIS_METHOD;
		int NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS;
		int NUMBER_OF_PERIODS;
		int PERIOD_TIME;
		

		// Random permutations
		int NUMBER_OF_PERMUTATIONS;
		float significance_threshold;
		float permutation_test_threshold;
		int NUMBER_OF_SIGNIFICANTLY_ACTIVE_VOXELS;
		int NUMBER_OF_SIGNIFICANTLY_ACTIVE_CLUSTERS;
		
		//--------------------------------------------------
		// Host pointers
		//--------------------------------------------------
			
		void		*host_pointers[NUMBER_OF_HOST_POINTERS];
		void		*host_pointers_static[NUMBER_OF_HOST_POINTERS];
		void		*host_pointers_permutation[NUMBER_OF_HOST_POINTERS];
		
		float		*h_Result;

		float		*h_fMRI_Volumes;
		float		*h_MNI_Brain_Mask;
		float		*h_Mask;
		float		*h_T1_Volume;
		float		*h_MNI_Volume;
		float	    *h_EPI_Volume;
		float		*h_Aligned_T1_Volume;
		float		*h_Aligned_EPI_Volume;
		float		*h_Skullstripped_T1_Volume;
		float		*h_Interpolated_T1_Volume;
		float		*h_Interpolated_EPI_Volume;
		float		*h_Downsampled_Volume;

		// Slice timing correction
		float		*h_Slice_Timing_Corrections_Real, *h_Slice_Timing_Corrections_Imag;
		//Complex		*h_Slice_Timing_Corrections;
		float		*h_Slice_Timing_Corrected_fMRI_Volumes;

		// Image Registration
		float		*h_Quadrature_Filter_1_Real, *h_Quadrature_Filter_1_Imag, *h_Quadrature_Filter_2_Real, *h_Quadrature_Filter_2_Imag, *h_Quadrature_Filter_3_Real, *h_Quadrature_Filter_3_Imag; 		
//		float2      *h_Quadrature_Filter_1, *h_Quadrature_Filter_2, *h_Quadrature_Filter_3;
//		float2      *h_Quadrature_Filter_Response_1, *h_Quadrature_Filter_Response_2, *h_Quadrature_Filter_Response_3; 
		float       *h_Quadrature_Filter_Response_1_Real, *h_Quadrature_Filter_Response_2_Real, *h_Quadrature_Filter_Response_3_Real; 
		float       *h_Quadrature_Filter_Response_1_Imag, *h_Quadrature_Filter_Response_2_Imag, *h_Quadrature_Filter_Response_3_Imag; 
		float		 h_A_Matrix[144], h_Inverse_A_Matrix[144], h_h_Vector[12];
		float 		 h_Registration_Parameters[12], h_Inverse_Registration_Parameters[12], h_Registration_Parameters_Old[12], h_Registration_Parameters_Temp[12], h_Registration_Parameters_EPI_T1_Affine[12], h_Registration_Parameters_Motion_Correction[12], h_Registration_Parameters_T1_MNI[12], h_Registration_Parameters_EPI_MNI[12], *h_Registration_Parameters_T1_MNI_Out, h_Registration_Parameters_EPI_T1[6], *h_Registration_Parameters_EPI_T1_Out;
		float		 h_Rotations[3], h_Rotations_Temp[3];
		float       *h_Phase_Differences, *h_Phase_Certainties, *h_Phase_Gradients;
	
		float *h_Registration_Parameters_Out;
		// Motion correction
		float		*h_Motion_Corrected_fMRI_Volumes;		
		float		*h_Motion_Parameters_Out, h_Motion_Parameters[2000];
		
		// fMRI - T1
		float		*h_Aligned_fMRI_Volume;


		// Smoothing
		float		*h_Smoothing_Filter_X, *h_Smoothing_Filter_Y, *h_Smoothing_Filter_Z;
		float		*h_Smoothed_fMRI_Volumes;
		
		// Detrending
		float		*h_X_Detrend, *h_xtxxt_Detrend;
		float		*h_Detrended_fMRI_Volumes;
			
		// Statistical analysis
		float		*hrf;
		int			 hrf_length;
		float       *h_Contrasts;
		float		*h_X_GLM, *h_xtxxt_GLM, *h_ctxtxc_GLM;
		float		*h_Statistical_Maps;
		float       *h_Beta_Volumes;
		float       *h_Residuals;
		float       *h_Residual_Variances;
			
		// Random permutations
		float		*h_Alpha_Smoothing_Kernel;
		float		*h_Smoothed_Alpha_Certainty;			
		uint16		*h_Permutation_Matrix;
		float		*h_Maximum_Test_Values;

		// Covariance pooling
		float		*h_Variance_Smoothing_Kernel;
		float		*h_Smoothed_Variance_Certainty;

		//--------------------------------------------------
		// Device pointers
		//--------------------------------------------------

		float		*device_pointers[NUMBER_OF_DEVICE_POINTERS];
		float		*device_pointers_permutation[NUMBER_OF_DEVICE_POINTERS];
		
		// Original data
		cl_mem		d_fMRI_Volumes;
		cl_mem		d_Volumes;
		cl_mem		d_Mask;
			
		// Slice timing correction
		cl_mem		d_fMRI_Volumes_Complex;
		cl_mem		d_Shifters;
		cl_mem		d_Slice_Timing_Corrected_fMRI_Volumes;

		// Image registration
		cl_mem      d_Reference_Volume, d_Aligned_Volume, d_Original_Volume;
		cl_mem		d_Current_Aligned_Volume, d_Current_Reference_Volume;
		cl_mem		d_A_Matrix, d_h_Vector, d_A_Matrix_2D_Values, d_A_Matrix_1D_Values, d_h_Vector_2D_Values, d_h_Vector_1D_Values;
		cl_mem 		d_Phase_Differences, d_Phase_Gradients, d_Phase_Certainties;
		cl_mem      d_q11, d_q12, d_q13, d_q21, d_q22, d_q23;
		cl_mem      d_q11_Real, d_q12_Real, d_q13_Real, d_q21_Real, d_q22_Real, d_q23_Real;
		cl_mem      d_q11_Imag, d_q12_Imag, d_q13_Imag, d_q21_Imag, d_q22_Imag, d_q23_Imag;
		cl_mem		c_Quadrature_Filter_1_Real, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_3_Real;
		cl_mem		c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Imag;
		cl_mem		c_Registration_Parameters;
	
		// Motion correction
		cl_mem		d_Motion_Corrected_fMRI_Volumes;
	
		//
		cl_mem		d_T1_Volume, d_Interpolated_T1_Volume, d_MNI_Volume, d_MNI_T1_Volume, d_Interpolated_fMRI_Volume, d_Skullstripped_T1_Volume, d_MNI_Brain_Mask;
		cl_mem		d_EPI_Volume, d_T1_EPI_Volume;
			
		// Smoothing
		cl_mem		d_Smoothed_Certainty;
		cl_mem		d_Smoothed_fMRI_Volumes;
		
		cl_mem		c_Smoothing_Filter_X;
		cl_mem		c_Smoothing_Filter_Y;
		cl_mem		c_Smoothing_Filter_Z;

		// Detrending
		cl_mem		d_Detrended_fMRI_Volumes;
		cl_mem		c_X_Detrend;

		// Statistical analysis		
		cl_mem		d_Beta_Volumes, d_Beta_Volumes_MNI;
		cl_mem		d_Statistical_Maps, d_Statistical_Maps_MNI;
		float		c_xtxxt_Detrend;
		cl_mem		c_Censor;
		cl_mem		c_xtxxt_GLM, c_X_GLM, c_Contrasts, c_ctxtxc_GLM;
		cl_mem		d_Beta_Contrasts;
		cl_mem		d_Residuals;
		cl_mem		d_Residual_Variances, d_Residual_Variances_MNI;

		// Paraneters for single subject permutations
		cl_mem		d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates;
		cl_mem		d_Smoothed_AR1_Estimates, d_Smoothed_AR2_Estimates, d_Smoothed_AR3_Estimates, d_Smoothed_AR4_Estimates;
			
		cl_mem		d_BOLD_Regressed_fMRI_Volumes;
		cl_mem		d_Whitened_fMRI_Volumes;
		cl_mem		d_Permuted_fMRI_Volumes;

		cl_mem		c_Permutation_Vector;
		cl_mem		c_AR_Smoothing_Filter_X, c_AR_Smoothing_Filter_Y, c_AR_Smoothing_Filter_Z;


		//--------------------------------------------------
		// Filenames
		//--------------------------------------------------

		std::string		filename_real_quadrature_filter_1;
		std::string		filename_real_quadrature_filter_2;
		std::string		filename_real_quadrature_filter_3;
		std::string		filename_imag_quadrature_filter_1;
		std::string		filename_imag_quadrature_filter_2;
		std::string		filename_imag_quadrature_filter_3;
		std::string		filename_GLM_filter;

		std::string		filename_fMRI_data_raw;
		std::string		filename_slice_timing_corrected_fMRI_volumes_raw;
		std::string		filename_registration_parameters_raw;
		std::string		filename_motion_corrected_fMRI_volumes_raw;
		std::string		filename_smoothed_fMRI_volumes_raw;
		std::string		filename_detrended_fMRI_volumes_raw;
		std::string		filename_activity_volume_raw;

		std::string		filename_fMRI_data_nifti;
		std::string		filename_slice_timing_corrected_fMRI_volumes_nifti;
		std::string		filename_registration_parameters_nifti;
		std::string		filename_motion_corrected_fMRI_volumes_nifti;
		std::string		filename_smoothed_fMRI_volumes_nifti;
		std::string		filename_detrended_fMRI_volumes_nifti;
		std::string		filename_activity_volume_nifti;

};


#endif
