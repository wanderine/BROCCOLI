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

#include "broccoli_constants.h"

#include "nifti1.h"
#include "nifti1_io.h"

#include <opencl.h>
#include <string>

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short int uint16;


class BROCCOLI_LIB
{
	public:

		// Constructor & destructor
		BROCCOLI_LIB();
		BROCCOLI_LIB(cl_uint platform, cl_uint device);
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
		void SetEPISmoothingAmount(float);
		void SetARSmoothingAmount(float);
		void SetImageRegistrationFilterSize(int N);
		void SetImageRegistrationFilters(float* qf1r, float* qf1i, float* qf2r, float* qf2i, float* qf3r, float* qf3i);
		void SetNumberOfIterationsForImageRegistration(int N);
		void SetNumberOfIterationsForMotionCorrection(int N);
		void SetCoarsestScaleT1MNI(int N);
		void SetCoarsestScaleEPIT1(int N);
		void SetMMT1ZCUT(int mm);
		void SetMMEPIZCUT(int mm);
		void SetBetaSpace(int space);
		void SetInterpolationMode(int mode);
		
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
		void SetOutputEPIMNIRegistrationParameters(float* output);
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
		void SetOutputAREstimates(float*, float*, float*, float*);
		void SetOutputSliceSums(float*);
		void SetOutputTopSlice(float*);
		
		void SetfMRIDataFilename(std::string filename);			
		void SetfMRIParameters(float tr, float xs, float ys, float zs);		
		void SetNumberOfBasisFunctionsDetrending(int N);
		void SetAnalysisMethod(int method);		
		
		void SetActivityThreshold(float threshold);
		void SetThresholdStatus(bool status);

		void SetfMRIDataSliceLocationX(int location);
		void SetfMRIDataSliceLocationY(int location);
		void SetfMRIDataSliceLocationZ(int location);
		void SetfMRIDataSliceTimepoint(int timepoint);

		void SetDataType(int type);
		void SetFileType(int type);

		void SetNumberOfPermutations(int value);
		void SetSignificanceThreshold(float value);

		// Get functions for GUI / Wrappers
		
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

		// OpenCL

		const char* GetOpenCLDeviceInfoChar();
		const char* GetOpenCLBuildInfoChar();
		
		std::string GetOpenCLDeviceInfoString();
		std::string GetOpenCLBuildInfoString();

		int* GetOpenCLCreateKernelErrors();
		int* GetOpenCLCreateBufferErrors();
		int* GetOpenCLRunKernelErrors();

		int GetOpenCLPlatformIDsError();
		int GetOpenCLDeviceIDsError();		
		int GetOpenCLCreateContextError();
		int GetOpenCLContextInfoError();
		int GetOpenCLCreateCommandQueueError();
		int GetOpenCLCreateProgramError();
		int GetOpenCLBuildProgramError();
		int GetOpenCLProgramBuildInfoError();

		int GetProgramBinarySize();
		int GetWrittenElements();

		// Processing times

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
				
		// Read functions
		void ReadfMRIDataRAW();
		void ReadfMRIDataNIFTI();
		void ReadNIFTIHeader();

		// Write functions
		void WritefMRIDataNIFTI();
		
		// Preprocessing
		void ChangeT1VolumeResolutionAndSizeWrapper();		
		void PerformRegistrationT1MNIWrapper();
		void PerformRegistrationEPIT1Wrapper();
		void PerformSliceTimingCorrectionWrapper();
		void PerformMotionCorrectionWrapper();		
		void PerformDetrending();
		void PerformSmoothingWrapper();
		void PerformGLMWrapper();
		void PerformFirstLevelAnalysisWrapper();				
														
		void CalculateSlicesfMRIData();
		void CalculateSlicesPreprocessedfMRIData();
		void CalculateSlicesActivityData();

		void GetOpenCLInfo();
		void OpenCLInitiate(cl_uint OPENCL_PLATFORM, cl_uint OPENCL_DEVICE);		
		
	private:
		
		//------------------------------------------------
		// High level functions
		//------------------------------------------------
		void PerformRegistrationEPIT1();
		void PerformRegistrationT1MNI();				
		void SegmentEPIData();		
		void PerformSliceTimingCorrection();		
		void PerformMotionCorrection();
		void PerformFirstLevelAnalysis();				

		void CalculateStatisticalMapsGLMFirstLevel(cl_mem Volumes);
		void CalculateStatisticalMapsGLMSecondLevel(cl_mem Volumes);
		void CalculatePermutationTestThresholdSingleSubject();
		void CalculatePermutationTestThresholdMultiSubject();				
		
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
		
		//------------------------------------------------
		// Convolution functions
		//------------------------------------------------
		void Copy3DFiltersToConstantMemory(int z, int FILTER_SIZE);
		void NonseparableConvolution3D(cl_mem d_q1_Real, cl_mem d_q1_Imag, cl_mem d_q2_Real, cl_mem d_q2_Imag, cl_mem d_q3_Real, cl_mem d_q3_Imag, cl_mem d_Volume, int DATA_W, int DATA_H, int DATA_D);		
		void PerformSmoothing(cl_mem Smoothed_Volumes, cl_mem d_Volumes, float* h_Smoothing_Filter_X, float* h_Smoothing_Filter_Y, float* h_Smoothing_Filter_Z, int DATA_W, int DATA_H, int DATA_D, int DATA_T);
		void PerformSmoothingNormalized(cl_mem Smoothed_Volumes, cl_mem d_Volumes, cl_mem d_Certainty, cl_mem d_Smoothed_Certainty, float* h_Smoothing_Filter_X, float* h_Smoothing_Filter_Y, float* h_Smoothing_Filter_Z, int DATA_W, int DATA_H, int DATA_D, int DATA_T);
		void PerformSmoothing(cl_mem d_Volumes, float* h_Smoothing_Filter_X, float* h_Smoothing_Filter_Y, float* h_Smoothing_Filter_Z, int DATA_W, int DATA_H, int DATA_D, int DATA_T);
		void PerformSmoothingNormalized(cl_mem d_Volumes, cl_mem d_Certainty, cl_mem d_Smoothed_Certainty, float* h_Smoothing_Filter_X, float* h_Smoothing_Filter_Y, float* h_Smoothing_Filter_Z, int DATA_W, int DATA_H, int DATA_D, int DATA_T);
				
		//------------------------------------------------
		// Functions for image registration		
		//------------------------------------------------
		void AlignTwoVolumes(float* h_Registration_Parameters, float* h_Rotations, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_ITERATIONS, int ALIGNMENT_TYPE, int INTERPOLATION_MODE);
		void AlignTwoVolumesDouble(float* h_Registration_Parameters, float* h_Rotations, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_ITERATIONS, int ALIGNMENT_TYPE, int INTERPOLATION_MODE);
		void AlignTwoVolumesSeveralScales(float *h_Registration_Parameters, float* h_Rotations, cl_mem d_Al_Volume, cl_mem d_Ref_Volume, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_SCALES, int NUMBER_OF_ITERATIONS, int ALIGNMENT_TYPE, int OVERWRITE, int INTERPOLATION_MODE);
		void AlignTwoVolumesCleanup();
		void AlignTwoVolumesCleanupDouble();
		void AlignTwoVolumesSetup(int DATA_W, int DATA_H, int DATA_D);
		void AlignTwoVolumesSetupDouble(int DATA_W, int DATA_H, int DATA_D);
		void ChangeT1VolumeResolutionAndSize(cl_mem d_MNI_T1_Volume, cl_mem d_T1_Volume, int T1_DATA_W, int T1_DATA_H, int T1_DATA_D, int MNI_DATA_W, int MNI_DATA_H, int MNI_DATA_D, float T1_VOXEL_SIZE_X, float T1_VOXEL_SIZE_Y, float T1_VOXEL_SIZE_Z, float MNI_VOXEL_SIZE_X, float MNI_VOXEL_SIZE_Y, float MNI_VOXEL_SIZE_Z, int INTERPOLATION_MODE);
		void ChangeEPIVolumeResolutionAndSize(cl_mem d_T1_EPI_Volume, cl_mem d_EPI_Volume, int EPI_DATA_W, int EPI_DATA_H, int EPI_DATA_D, int T1_DATA_W, int T1_DATA_H, int T1_DATA_D, float EPI_VOXEL_SIZE_X, float EPI_VOXEL_SIZE_Y, float EPI_VOXEL_SIZE_Z, float T1_VOXEL_SIZE_X, float T1_VOXEL_SIZE_Y, float T1_VOXEL_SIZE_Z, int INTERPOLATION_MODE);
		void ChangeVolumeSize(cl_mem d_Current_Aligned_Volume, cl_mem d_Aligned_Volume, int DATA_W, int DATA_H, int DATA_D, int CURRENT_DATA_W, int CURRENT_DATA_H, int CURRENT_DATA_D, int INTERPOLATION_MODE);       
		void ChangeVolumesResolutionAndSize(cl_mem d_New_Volumes, cl_mem d_Volumes, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_VOLUMES, int NEW_DATA_W, int NEW_DATA_H, int NEW_DATA_D, float VOXEL_SIZE_X, float VOXEL_SIZE_Y, float VOXEL_SIZE_Z, float NEW_VOXEL_SIZE_X, float NEW_VOXEL_SIZE_Y, float NEW_VOXEL_SIZE_Z, int MM_Z_CUT, int INTERPOLATION_MODE);

		void TransformVolume(cl_mem d_Volume, float* h_Registration_Parameters, int DATA_W, int DATA_H, int DATA_D, int INTERPOLATION_MODE);
		void TransformVolumes(cl_mem d_Volume, float* h_Registration_Parameters, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_VOLUMES, int INTERPOLATION_MODE);
		void TransformFirstLevelResultsToMNI();

		//------------------------------------------------
		// Help functions
		//------------------------------------------------
		void SetMemory(cl_mem memory, float value, int N);
		void SetMemoryDouble(cl_mem memory, double value, int N);
		void CalculateTopBrainSlice(int& slice, cl_mem d_Volume, int DATA_W, int DATA_H, int DATA_D, int z_cut);		
		void MultiplyVolumes(cl_mem d_Volume_1, cl_mem d_Volume_2, int DATA_W, int DATA_H, int DATA_D);
		void MultiplyVolumes(cl_mem d_Result, cl_mem d_Volume_1, cl_mem d_Volume_2, int DATA_W, int DATA_H, int DATA_D);	
		float CalculateSum(cl_mem Volume, int DATA_W, int DATA_H, int DATA_D);
		float CalculateMax(cl_mem Volume, int DATA_W, int DATA_H, int DATA_D);
		void ThresholdVolume(cl_mem d_Thresholded_Volume, cl_mem d_Volume, float threshold, int DATA_W, int DATA_H, int DATA_D);
		cl_int CreateProgramFromBinary(cl_program& program, cl_context context, cl_device_id device, std::string filename);
		bool SaveProgramBinary(cl_program program, cl_device_id device, std::string filename);
		void CreateSmoothingFilters(float* Smoothing_Filter_X, float* Smoothing_Filter_Y, float* Smoothing_Filter_Z, int size, float smoothing_FWHM, float voxel_size_x, float voxel_size_y, float voxel_size_z);
		//void ConvertRealToComplex(Complex* complex_data, float* real_data, int N);
		//void ExtractRealData(float* real_data, Complex* complex_data, int N);
		//void Convert4FloatToFloat4(float4* floats, float* float_1, float* float_2, float* float_3, float* float_4, int N);
		//void Convert2FloatToFloat2(float2* floats, float* float_1, float* float_2, int N);
		void InvertMatrix(float* inverse_matrix, float* matrix, int N);
		void InvertMatrixDouble(double* inverse_matrix, double* matrix, int N);
		void CalculateMatrixSquareRoot(float* sqrt_matrix, float* matrix, int N);
		void SolveEquationSystem(float* h_A_matrix, float* h_h_vector, float* h_Parameter_Vector, int N);
		void SolveSystemLUDouble(float* solution, float* matrix, float* b_vector, int N);
		void SolveSystemLUDoubleDouble(double* solution, double* matrix, double* b_vector, int N);
		void SolveEquationSystemDouble(double* h_A_matrix, double* h_h_vector, double* h_Parameter_Vector, int N);
		
		void InvertAffineRegistrationParameters(float* h_Inverse_Parameters, float* h_Parameters);
		void AddAffineRegistrationParameters(float* h_Old_Parameters, float* h_New_Parameters);
		void AddAffineRegistrationParametersNextScale(float* h_Old_Parameters, float* h_New_Parameters);
		void AddAffineRegistrationParameters(float* h_Resulting_Parameters, float* h_New_Parameters, float* h_Old_Parameters);
		void CalculateRotationAnglesFromRotationMatrix(float* h_Rotations, float* h_Registration_Parameters);					
		
		//------------------------------------------------
		// SVD 3x3
		//------------------------------------------------
		double cbrt(double x); 
		void RemoveTransformationScaling(float* h_Registration_Parameters);
		void RemoveTransformationScalingDouble(float* h_Registration_Parameters, double* h_Registration_Parameters_double);
		void Cross(double* z, const double* x, const double* y); 
		void Sort3(double* x);
		void Unit3(double* x); 
		void LDUBSolve3(double *x, const double *y, const double *LDU, const int *P);
		void MatMul3x3(double* C, const double* A, const double* B);
		void MatMul4x4(double* C, const double* A, const double* B);
		void MatVec3(double* y, const double* A, const double* x);
		void A_Transpose_A3x3(double* AA, const double* A);
		void A_A_Transpose3x3(double* AA, const double* A);
		void Transpose3x3(double* A); 
		void SolveCubic(double* c);
		void LDU3(double* A, int* P);
		void SVD3x3(double* U, double* S, double* V, const double* A);

		void SetupDetrendingBasisFunctions();
		void SetupStatisticalAnalysisBasisFunctions();
		float CalculateMax(float *data, int N);
		float CalculateMin(float *data, int N);
		float Gpdf(double value, double shape, double scale);
		float loggamma(int value);
		void CreateHRF();
		void ConvolveWithHRF(float* temp_GLM);

		//------------------------------------------------
		// Set functions
		//------------------------------------------------
		void SetGlobalAndLocalWorkSizesSeparableConvolution(int DATA_W, int DATA_H, int DATA_D);		
		void SetGlobalAndLocalWorkSizesNonSeparableConvolution(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesImageRegistration(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesStatisticalCalculations(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesRescaleVolume(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesInterpolateVolume(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesCopyVolumeToNew(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesMemset(int N);
		void SetGlobalAndLocalWorkSizesMultiplyVolumes(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesCalculateSum(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesThresholdVolume(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesCalculateMagnitudes(int DATA_W, int DATA_H, int DATA_D);		

		//------------------------------------------------
		// Read functions
		//------------------------------------------------
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

		//------------------------------------------------
		// Write functions
		//------------------------------------------------
		void WriteRealDataUint16(unsigned short int* data, std::string filename, int N);
		void WriteRealDataFloat(float* data, std::string filename, int N);
		void WriteRealDataDouble(double* data, std::string filename, int N);
		//void WriteComplexData(Complex* data, std::string real_filename, std::string imag_filename, int N);

		//------------------------------------------------
		// OpenCL help functions		
		//------------------------------------------------
		void OpenCLCleanup();

		void SetStartValues();
		void ResetAllPointers();
		void AllocateMemoryForFilters();
		
		//------------------------------------------------
		// OpenCL variables
		//------------------------------------------------
		size_t xBlocks, yBlocks, zBlocks;
		size_t programBinarySize, writtenElements;

		cl_context context;
		cl_command_queue commandQueue;
		cl_program program;
		cl_device_id device;

		std::string binaryFilename;
		std::string device_info;
		std::string build_info;
		
		cl_uint OPENCL_PLATFORM;
		int VENDOR, OPENCL_INITIATED;

		cl_int error;
		cl_int getPlatformIDsError;
		cl_int getDeviceIDsError;		
		cl_int createContextError;
		cl_int getContextInfoError;
		cl_int createCommandQueueError;
		cl_int createProgramError;
		cl_int buildProgramError;
		cl_int getProgramBuildInfoError;

		// OpenCL kernels
		cl_kernel MemsetKernel;
		cl_kernel MemsetDoubleKernel;
		cl_kernel SeparableConvolutionRowsKernel, SeparableConvolutionColumnsKernel, SeparableConvolutionRodsKernel, NonseparableConvolution3DComplexKernel;				
		cl_kernel CalculateBetaValuesGLMKernel, CalculateStatisticalMapsGLMKernel, RemoveLinearFitKernel;
		cl_kernel EstimateAR4ModelsKernel, ApplyWhiteningAR4Kernel, GeneratePermutedfMRIVolumesAR4Kernel;
		cl_kernel CalculatePhaseDifferencesAndCertaintiesKernel, CalculatePhaseGradientsXKernel, CalculatePhaseGradientsYKernel, CalculatePhaseGradientsZKernel;
		cl_kernel CalculateAMatrixAndHVector2DValuesXKernel, CalculateAMatrixAndHVector2DValuesYKernel,CalculateAMatrixAndHVector2DValuesZKernel; 
		cl_kernel CalculateAMatrixAndHVector2DValuesXDoubleKernel, CalculateAMatrixAndHVector2DValuesYDoubleKernel,CalculateAMatrixAndHVector2DValuesZDoubleKernel; 
		cl_kernel CalculateAMatrix1DValuesKernel, CalculateHVector1DValuesKernel, CalculateHVectorKernel, ResetAMatrixKernel, CalculateAMatrixKernel;
		cl_kernel CalculateAMatrix1DValuesDoubleKernel, CalculateHVector1DValuesDoubleKernel, CalculateHVectorDoubleKernel, ResetAMatrixDoubleKernel, CalculateAMatrixDoubleKernel;
		cl_kernel InterpolateVolumeNearestKernel, InterpolateVolumeLinearKernel, InterpolateVolumeCubicKernel; 
		cl_kernel RescaleVolumeNearestKernel, RescaleVolumeLinearKernel, RescaleVolumeCubicKernel;
		cl_kernel CopyT1VolumeToMNIKernel, CopyEPIVolumeToT1Kernel, CopyVolumeToNewKernel;
		cl_kernel MultiplyVolumesKernel, MultiplyVolumesOverwriteKernel;
		cl_kernel CalculateMagnitudesKernel;
		cl_kernel CalculateColumnSumsKernel, CalculateRowSumsKernel;
		cl_kernel CalculateColumnMaxsKernel, CalculateRowMaxsKernel;
		cl_kernel ThresholdVolumeKernel;
		
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
		cl_int createKernelErrorCalculateMagnitudes;
		cl_int createKernelErrorCalculateColumnSums;
		cl_int createKernelErrorCalculateRowSums;
		cl_int createKernelErrorCalculateColumnMaxs;
		cl_int createKernelErrorCalculateRowMaxs;
		cl_int createKernelErrorThresholdVolume;
		cl_int createKernelErrorCalculateBetaValuesGLM, createKernelErrorCalculateStatisticalMapsGLM;
		cl_int createKernelErrorEstimateAR4Models, createKernelErrorApplyWhiteningAR4;
		
		// Create buffer errors
		cl_int createBufferErrorAlignedVolume, createBufferErrorReferenceVolume;
		cl_int createBufferErrorq11Real, createBufferErrorq11Imag, createBufferErrorq12Real, createBufferErrorq12Imag, createBufferErrorq13Real, createBufferErrorq13Imag;
		cl_int createBufferErrorq21Real, createBufferErrorq21Imag, createBufferErrorq22Real, createBufferErrorq22Imag, createBufferErrorq23Real, createBufferErrorq23Imag;
		cl_int createBufferErrorPhaseDifferences, createBufferErrorPhaseCertainties, createBufferErrorPhaseGradients;
		cl_int createBufferErrorAMatrix, createBufferErrorHVector, createBufferErrorAMatrix2DValues, createBufferErrorAMatrix1DValues, createBufferErrorHVector2DValues, createBufferErrorHVector1DValues;
		cl_int createBufferErrorQuadratureFilter1Real, createBufferErrorQuadratureFilter1Imag, createBufferErrorQuadratureFilter2Real, createBufferErrorQuadratureFilter2Imag, createBufferErrorQuadratureFilter3Real, createBufferErrorQuadratureFilter3Imag;    
		cl_int createBufferErrorRegistrationParameters;
		cl_int createBufferErrorBetaVolumesMNI;
		cl_int createBufferErrorStatisticalMapsMNI;
		cl_int createBufferErrorResidualVariancesMNI;

		// Run kernel errors
		cl_int runKernelErrorSeparableConvolutionRows, runKernelErrorSeparableConvolutionColumns, runKernelErrorSeparableConvolutionRods;
		cl_int runKernelErrorNonseparableConvolution3DComplex;
		cl_int runKernelErrorMemset;
		cl_int runKernelErrorCalculatePhaseDifferencesAndCertainties, runKernelErrorCalculatePhaseGradientsX, runKernelErrorCalculatePhaseGradientsY, runKernelErrorCalculatePhaseGradientsZ;
		cl_int runKernelErrorCalculateAMatrixAndHVector2DValuesX, runKernelErrorCalculateAMatrixAndHVector2DValuesY, runKernelErrorCalculateAMatrixAndHVector2DValuesZ;
		cl_int runKernelErrorCalculateAMatrix1DValues, runKernelErrorCalculateHVector1DValues;
		cl_int runKernelErrorCalculateAMatrix, runKernelErrorCalculateHVector;
		cl_int runKernelErrorInterpolateVolume;
		cl_int runKernelErrorRescaleVolume;
		cl_int runKernelErrorMultiplyVolumes;
		cl_int runKernelErrorCopyVolume;
		cl_int runKernelErrorCalculateBetaValuesGLM;
		cl_int runKernelErrorCalculateStatisticalMapsGLM;
		cl_int runKernelErrorEstimateAR4Models;
		cl_int runKernelErrorApplyAR4Whitening;
		cl_int runKernelErrorCalculateMagnitudes;
		cl_int runKernelErrorCalculateColumnSums;
		cl_int runKernelErrorCalculateRowSums;
		cl_int runKernelErrorCalculateColumnMaxs;
		cl_int runKernelErrorCalculateRowMaxs;
		cl_int runKernelErrorThresholdVolume;

		int OpenCLCreateBufferErrors[50];
		int OpenCLRunKernelErrors[50];
		int OpenCLCreateKernelErrors[50];

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
		size_t localWorkSizeCalculateMagnitudes[3];
		size_t localWorkSizeCalculateColumnSums[3];
		size_t localWorkSizeCalculateRowSums[3];
		size_t localWorkSizeCalculateColumnMaxs[3];
		size_t localWorkSizeCalculateRowMaxs[3];
		size_t localWorkSizeThresholdVolume[3];
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
		size_t globalWorkSizeCalculateMagnitudes[3];
		size_t globalWorkSizeCalculateColumnSums[3];
		size_t globalWorkSizeCalculateRowSums[3];
		size_t globalWorkSizeCalculateColumnMaxs[3];
		size_t globalWorkSizeCalculateRowMaxs[3];
		size_t globalWorkSizeThresholdVolume[3];
		size_t globalWorkSizeCalculateBetaValuesGLM[3];
		size_t globalWorkSizeCalculateStatisticalMapsGLM[3];
		size_t globalWorkSizeRemoveLinearFit[3];		
		size_t globalWorkSizeEstimateAR4Models[3];
		size_t globalWorkSizeApplyWhiteningAR4[3];
		size_t globalWorkSizeGeneratePermutedfMRIVolumesAR4[3];

		//------------------------------------------------
		// General variables
		//------------------------------------------------
		int BETA_SPACE;
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

		// Resolution variables	
		float EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z;
		float TR;		
		float T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z;
		float MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z;

		double processing_times[20];
		
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

		
		// Image registration variables
		int INTERPOLATION_MODE;
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

		// Smoothing variables
		int	SMOOTHING_FILTER_SIZE;		
		float EPI_Smoothing_FWHM;
		float AR_Smoothing_FWHM;
			
		double* smoothed_curve;

		// Detrending variables
		int NUMBER_OF_DETRENDING_BASIS_FUNCTIONS;

		double* detrended_curve;
				 
		// Statistical analysis variables
		bool THRESHOLD_ACTIVITY_MAP;
		float ACTIVITY_THRESHOLD;
		int ANALYSIS_METHOD;
		int NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS;
		int NUMBER_OF_PERIODS;
		int PERIOD_TIME;
		
		// Random permutation variables
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
		
		// Data pointers
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

		// Slice timing correction pointers
		float		*h_Slice_Timing_Corrections_Real, *h_Slice_Timing_Corrections_Imag;
		//Complex		*h_Slice_Timing_Corrections;
		float		*h_Slice_Timing_Corrected_fMRI_Volumes;

		// Image Registration pointers
		float		*h_Quadrature_Filter_1_Real, *h_Quadrature_Filter_1_Imag, *h_Quadrature_Filter_2_Real, *h_Quadrature_Filter_2_Imag, *h_Quadrature_Filter_3_Real, *h_Quadrature_Filter_3_Imag; 		
//		float2      *h_Quadrature_Filter_1, *h_Quadrature_Filter_2, *h_Quadrature_Filter_3;
//		float2      *h_Quadrature_Filter_Response_1, *h_Quadrature_Filter_Response_2, *h_Quadrature_Filter_Response_3; 
		float       *h_Quadrature_Filter_Response_1_Real, *h_Quadrature_Filter_Response_2_Real, *h_Quadrature_Filter_Response_3_Real; 
		float       *h_Quadrature_Filter_Response_1_Imag, *h_Quadrature_Filter_Response_2_Imag, *h_Quadrature_Filter_Response_3_Imag; 
		float		 h_A_Matrix[144], h_h_Vector[12];
		double		 h_A_Matrix_double[144], h_h_Vector_double[12];
		float 		 h_Registration_Parameters[12], h_Inverse_Registration_Parameters[12], h_Registration_Parameters_Old[12], h_Registration_Parameters_Temp[12], h_Registration_Parameters_EPI_T1_Affine[12], h_Registration_Parameters_Motion_Correction[12], h_Registration_Parameters_T1_MNI[12], h_Registration_Parameters_EPI_MNI[12], *h_Registration_Parameters_T1_MNI_Out, h_Registration_Parameters_EPI_T1[6], *h_Registration_Parameters_EPI_T1_Out, *h_Registration_Parameters_EPI_MNI_Out;
		double       h_Registration_Parameters_double[12];
		float		 h_Rotations[3], h_Rotations_Temp[3];
		float       *h_Phase_Differences, *h_Phase_Certainties, *h_Phase_Gradients;
	
		float		*h_Slice_Sums, *h_Top_Slice;

		float *h_Registration_Parameters_Out;
		// Motion correction variables
		float		*h_Motion_Corrected_fMRI_Volumes;		
		float		*h_Motion_Parameters_Out, h_Motion_Parameters[2000];
		
		// fMRI - T1
		float		*h_Aligned_fMRI_Volume;

		// Smoothing pointers
		float		*h_Smoothing_Filter_X_In, *h_Smoothing_Filter_Y_In, *h_Smoothing_Filter_Z_In;
		float		h_Smoothing_Filter_X[9], h_Smoothing_Filter_Y[9], h_Smoothing_Filter_Z[9];
		float		*h_Smoothed_fMRI_Volumes;
		
		// Detrending pointers
		float		*h_X_Detrend, *h_xtxxt_Detrend;
		float		*h_Detrended_fMRI_Volumes;
			
		// Statistical analysis pointers
		float		*hrf;
		int			 hrf_length;
		float       *h_Contrasts;
		float		*h_X_GLM, *h_xtxxt_GLM, *h_ctxtxc_GLM;
		float		*h_Statistical_Maps;
		float       *h_Beta_Volumes;
		float       *h_Residuals;
		float       *h_Residual_Variances;
		float		*h_AR1_Estimates;
		float		*h_AR2_Estimates;
		float		*h_AR3_Estimates;
		float		*h_AR4_Estimates;
			
		// Random permutation pointers
		uint16		*h_Permutation_Matrix;
		float		*h_Maximum_Test_Values;

		//--------------------------------------------------
		// Device pointers
		//--------------------------------------------------

		float		*device_pointers[NUMBER_OF_DEVICE_POINTERS];
		float		*device_pointers_permutation[NUMBER_OF_DEVICE_POINTERS];
		
		// Original data 
		cl_mem		d_fMRI_Volumes;
		cl_mem		d_EPI_Mask;
		cl_mem		d_Smoothed_EPI_Mask;
			
		// Slice timing correction
		cl_mem		d_fMRI_Volumes_Complex;
		cl_mem		d_Shifters;
		cl_mem		d_Slice_Timing_Corrected_fMRI_Volumes;

		// Image registration
		cl_mem      d_Reference_Volume, d_Aligned_Volume, d_Original_Volume;
		cl_mem		d_Current_Aligned_Volume, d_Current_Reference_Volume;
		cl_mem		d_A_Matrix, d_h_Vector, d_A_Matrix_2D_Values, d_A_Matrix_1D_Values, d_h_Vector_2D_Values, d_h_Vector_1D_Values;
		cl_mem		d_A_Matrix_double, d_h_Vector_double, d_A_Matrix_2D_Values_double, d_A_Matrix_1D_Values_double, d_h_Vector_2D_Values_double, d_h_Vector_1D_Values_double;
		cl_mem 		d_Phase_Differences, d_Phase_Gradients, d_Phase_Certainties;
		cl_mem      d_q11, d_q12, d_q13, d_q21, d_q22, d_q23;
		cl_mem      d_q11_Real, d_q12_Real, d_q13_Real, d_q21_Real, d_q22_Real, d_q23_Real;
		cl_mem      d_q11_Imag, d_q12_Imag, d_q13_Imag, d_q21_Imag, d_q22_Imag, d_q23_Imag;
		cl_mem		c_Quadrature_Filter_1_Real, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_3_Real;
		cl_mem		c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Imag;
		cl_mem		c_Quadrature_Filter_1_Real_AMD, c_Quadrature_Filter_2_Real_AMD, c_Quadrature_Filter_3_Real_AMD;
		cl_mem		c_Quadrature_Filter_1_Imag_AMD, c_Quadrature_Filter_2_Imag_AMD, c_Quadrature_Filter_3_Imag_AMD;
		cl_mem		c_Registration_Parameters;
	
		// Motion correction
		cl_mem		d_Motion_Corrected_fMRI_Volumes;
	
		// T1-MNI and EPI-T1 registration
		cl_mem		d_T1_Volume, d_Interpolated_T1_Volume, d_MNI_Volume, d_MNI_T1_Volume, d_Interpolated_fMRI_Volume, d_Skullstripped_T1_Volume, d_MNI_Brain_Mask;
		cl_mem		d_EPI_Volume, d_T1_EPI_Volume;
			
		// Smoothing
		cl_mem		d_Certainty;
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
