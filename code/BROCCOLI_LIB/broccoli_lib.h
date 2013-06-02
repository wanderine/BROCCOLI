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

#ifndef BROCCOLILIB_H
#define BROCCOLILIB_H

#include "nifti1.h"
#include "nifti1_io.h"

#include <cuda.h>
#include <helper_functions.h> // Helper functions (utilities, parsing, timing)
#include "cuda_runtime.h"
#include <opencl.h>

#include <string>

//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/generate.h>
//#include <thrust/reduce.h>
//#include <thrust/functional.h>
//#include <thrust/transform_reduce.h>
//#include <cstdlib>


typedef float2 Complex;
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




class BROCCOLI_LIB
{
	public:

		// Constructor & destructor
		BROCCOLI_LIB();
		~BROCCOLI_LIB();

		// Set functions for GUI / Wrappers
		void SetfMRIDataFilename(std::string filename);
			
		void SetfMRIParameters(float tr, float xs, float ys, float zs);
		void SetNumberOfIterationsForMotionCorrection(int N);
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

		void SetfMRIVoxelSizeX(float value);
		void SetfMRIVoxelSizeY(float value);
		void SetfMRIVoxelSizeZ(float value);
		void SetTR(float value);

		void SetWidth(int w);
		void SetHeight(int h);
		void SetDepth(int d);
		void SetTimepoints(int t);

		void SetNumberOfPermutations(int value);
		void SetSignificanceThreshold(float value);

		// Get functions for GUI / Wrappers
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

		int GetWidth();
		int GetHeight();
		int GetDepth();
		int GetTimepoints();

		float GetfMRIVoxelSizeX();
		float GetfMRIVoxelSizeY();
		float GetfMRIVoxelSizeZ();
		float GetTR();

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
		void PerformRegistrationEPIT1(int t);
		void PerformRegistrationT1MNI();
		void PerformSliceTimingCorrection();
		void PerformMotionCorrection();
		void PerformDetrending();
		void PerformSmoothing(cl_mem Smoothed_Volumes, cl_mem d_Volumes, int NUMBER_OF_VOLUMES, cl_mem c_Smoothing_Filter_X, cl_mem c_Smoothing_Filter_Y, cl_mem c_Smoothing_Filter_Z);
		
		// Processing
		void PerformPreprocessingAndCalculateStatisticalMaps();				
		void CalculateStatisticalMapGLMFirstLevel();
		void CalculateStatisticalMapGLMSecondLevel();
		void CalculatePermutationTestThresholdSingleSubject();
		void CalculatePermutationTestThresholdMultiSubject();
		
		void CalculateStatisticalMapsGLMFirstLevel();
		void CalculateStatisticalMapsGLMSecondLevel();

		// Permutation single subject	
		void SetupParametersPermutationSingleSubject();
		void GeneratePermutationMatrixSingleSubject();
		void PerformDetrendingPriorPermutation();
		void CreateBOLDRegressedVolumes();
		void WhitenfMRIVolumes();
		void GeneratePermutedfMRIVolumes();
		void PerformDetrendingPermutation();
		void PerformSmoothingPermutation(); 
		void CalculateActivityMapPermutation();

		// Permutation multi subject	
		void SetupParametersPermutationMultiSubject();
		void GeneratePermutationMatrixMultiSubject();
		

		void CalculateGroupMapPermutation();
		float FindMaxTestvaluePermutation();
		
		void CalculateSlicesfMRIData();
		void CalculateSlicesPreprocessedfMRIData();
		void CalculateSlicesActivityData();

		void OpenCLTest();



	private:

		void AlignTwoVolumes(float* h_Registration_Parameters);
		void AlignTwoVolumesCleanup();
		void AlignTwoVolumesSetup(int DATA_W, int DATA_H, int DATA_D);
		void ChangeVolumeResolutionAndSize(cl_mem d_Original_Volume, cl_mem d_Interpolated_Volume, int ORIGINAL_DATA_W, int ORIGINAL_DATA_H, int ORIGINAL_DATA_D, int INTERPOLATED__DATA_W, int INTERPOLATED__DATA_H, int INTERPOLATED__DATA_D, float ORIGINAL_VOXEL_SIZE_X, float ORIGINAL_VOXEL_SIZE_Y, float ORIGINAL_VOXEL_SIZE_Z, float INTERPOLATED_VOXEL_SIZE_X, float INTERPOLATED_VOXEL_SIZE_Y, float INTERPOLATED_VOXEL_SIZE_Z);

		// Read functions
		void ReadRealDataInt32(int* data, std::string filename, int N);
		void ReadRealDataInt16(short int* data, std::string filename, int N);
		void ReadRealDataUint32(unsigned int* data, std::string filename, int N);
		void ReadRealDataUint16(unsigned short int* data, std::string filename, int N);
		void ReadRealDataFloat(float* data, std::string filename, int N);
		void ReadRealDataDouble(double* data, std::string filename, int N);
		void ReadComplexData(Complex* data, std::string real_filename, std::string imag_filename, int N);
		void ReadImageRegistrationFilters();
		void ReadSmoothingFilters();
		void SetupParametersReadData();

		// Write functions
		void WriteRealDataUint16(unsigned short int* data, std::string filename, int N);
		void WriteRealDataFloat(float* data, std::string filename, int N);
		void WriteRealDataDouble(double* data, std::string filename, int N);
		void WriteComplexData(Complex* data, std::string real_filename, std::string imag_filename, int N);

		// OpenCL help functions
		void OpenCLInitiate();
		void SetGlobalAndLocalWorkSizes();
		void OpenCLCleanup();

		// Other help functions
		void SetStartValues();
		void ResetAllPointers();
		void AllocateMemoryForFilters();

		void ConvertRealToComplex(Complex* complex_data, float* real_data, int N);
		void ExtractRealData(float* real_data, Complex* complex_data, int N);
		void Convert4FloatToFloat4(float4* floats, float* float_1, float* float_2, float* float_3, float* float_4, int N);
		void Convert2FloatToFloat2(float2* floats, float* float_1, float* float_2, int N);
		void Calculate_Block_Differences2D(int& xBlockDifference, int& yBlockDifference, int DATA_W, int DATA_H, int threadsInX, int threadsInY);
		void Calculate_Block_Differences3D(int& xBlockDifference, int& yBlockDifference, int& zBlockDifference, int DATA_W, int DATA_H, int DATA_D, int threadsInX, int threadsInY, int threadsInZ);
		void Invert_Matrix(float* inverse_matrix, float* matrix, int N);
		void Calculate_Square_Root_of_Matrix(float* sqrt_matrix, float* matrix, int N);
		void SolveEquationSystem(float* h_A_matrix, float* h_inverse_A_matrix, float* h_h_vector, float* h_Parameter_Vector, int N);
		void SetupDetrendingBasisFunctions();
		void SetupStatisticalAnalysisBasisFunctions();
		void SegmentBrainData();
		float CalculateMax(float *data, int N);
		float CalculateMin(float *data, int N);
		float Gpdf(double value, double shape, double scale);
		float loggamma(int value);

		void ConvolveWithHRF(float* temp_GLM);
		void CreateHRF();

		// OpenCL
		cl_context context;
		cl_command_queue commandQueue;
		cl_program program;
		cl_device_id device;
		
		// OpenCL kernels

		cl_kernel SeparableConvolutionRowsKernel, SeparableConvolutionColumnsKernel, SeparableConvolutionRodsKernel, NonseparableConvolution3DComplexKernel;				
		cl_kernel CalculateBetaValuesGLMKernel, CalculateStatisticalMapsGLMKernel, RemoveLinearFitKernel;

		cl_kernel CalculatePhaseDifferencesAndCertaintiesKernel, CalculatePhaseGradientsXKernel, CalculatePhaseGradientsYKernel, CalculatePhaseGradientsZKernel;
		cl_kernel CalculateAMatrixAndHVector2DValuesXKernel, CalculateAMatrixAndHVector2DValuesYKernel,CalculateAMatrixAndHVector2DValuesZKernel; 
		cl_kernel CalculateAMatrix1DValuesKernel, CalculateHVector1DValuesKernel, CalculateHVectorKernel, ResetAMatrixKernel, CalculateAMatrixKernel;
		cl_kernel InterpolateVolumeTrilinearKernel;

		cl_kernel EstimateAR4ModelsKernel, ApplyWhiteningAR4Kernel, GeneratePermutedfMRIVolumesAR4Kernel;

		cl_int errNum;

		// OpenCL local work sizes

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
		size_t localWorkSizeInterpolateVolumeTrilinear[3];

		size_t localWorkSizeCalculateBetaValuesGLM[3];		
		size_t localWorkSizeCalculateStatisticalMapsGLM[3];
		size_t localWorkSizeRemoveLinearFit[3];

		size_t localWorkSizeEstimateAR4Models[3];
		size_t localWorkSizeApplyWhiteningAR4[3];
		size_t localWorkSizeGeneratePermutedfMRIVolumesAR4[3];

		// OpenCL global work sizes

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
		size_t globalWorkSizeInterpolateVolumeTrilinear[3];

		size_t globalWorkSizeCalculateBetaValuesGLM[3];
		size_t globalWorkSizeCalculateStatisticalMapsGLM[3];
		size_t globalWorkSizeRemoveLinearFit[3];

		size_t globalWorkSizeEstimateAR4Models[3];
		size_t globalWorkSizeApplyWhiteningAR4[3];
		size_t globalWorkSizeGeneratePermutedfMRIVolumesAR4[3];

		// General
		int FILE_TYPE, DATA_TYPE;
		nifti_image *nifti_data;

		int DATA_W, DATA_H, DATA_D, DATA_T;
		int FMRI_DATA_W, FMRI_DATA_H, FMRI_DATA_D, FMRI_DATA_T;
		int T1_DATA_W, T1_DATA_H, T1_DATA_D;
		int MNI_DATA_W, MNI_DATA_H, MNI_DATA_D;

		int DATA_SIZE_VOLUME, DATA_SIZE_COMPLEX_VOLUME;
		int DATA_SIZE_T1_VOLUME, DATA_SIZE_FMRI_VOLUME, DATA_SIZE_FMRI_VOLUMES, DATA_SIZE_MNI_VOLUME;

		int DATA_SIZE_QUADRATURE_FILTER_REAL;
		int DATA_SIZE_QUADRATURE_FILTER_COMPLEX;
		int DATA_SIZE_SMOOTHING_FILTER_GLM;

		int NUMBER_OF_SUBJECTS;
		int NUMBER_OF_CONTRASTS;
		float SEGMENTATION_THRESHOLD;
		float FMRI_VOXEL_SIZE_X, FMRI_VOXEL_SIZE_Y, FMRI_VOXEL_SIZE_Z;
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
		
		float		*h_fMRI_Volumes;
		Complex	 	*h_fMRI_Volumes_Complex;	

		// Slice timing correction
		float		*h_Slice_Timing_Corrections_Real, *h_Slice_Timing_Corrections_Imag;
		Complex		*h_Slice_Timing_Corrections;
		float		*h_Slice_Timing_Corrected_fMRI_Volumes;

		// Image Registration
		float		*h_Quadrature_Filter_1_Real, *h_Quadrature_Filter_1_Imag, *h_Quadrature_Filter_2_Real, *h_Quadrature_Filter_2_Imag, *h_Quadrature_Filter_3_Real, *h_Quadrature_Filter_3_Imag; 		
		Complex     *h_Quadrature_Filter_1, *h_Quadrature_Filter_2, *h_Quadrature_Filter_3;
		float		*h_A_Matrix, *h_Inverse_A_Matrix, *h_h_Vector;
		float 		 h_Parameter_Vector[12], h_Parameter_Vector_Total[12];
	
		// Motion correction
		float		*h_Motion_Corrected_fMRI_Volumes;
		float		*h_Registration_Parameters;
		float		*h_Motion_Parameters;
		
		// fMRI - T1
		float		*h_Aligned_fMRI_Volume;
		//float		*h_

		// T1 - MNI
		float		*h_Aligned_T1_Volume;

		// Smoothing
		float		*h_GLM_Filter;
		float		*h_Smoothed_fMRI_Volumes;
		
		// Detrending
		float		*h_X_Detrend, *h_xtxxt_Detrend;
		float		*h_Detrended_fMRI_Volumes;
			
		// Statistical analysis
		float		*hrf;
		int			 hrf_length;
		float       *h_Contrast_Vectors;
		float		*h_X_GLM, *h_xtxxt_GLM;
		float		 h_ctxtxc;			
		float		*h_Statistical_Maps;
		float		*ctxtxc;
		
			
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
		cl_mem		d_A_Matrix, d_h_Vector, d_A_Matrix_2D_Values, d_A_Matrix_1D_Values, d_h_Vector_2D_Values, d_h_Vector_1D_Values;
		cl_mem 		d_Phase_Differences, d_Phase_Gradients, d_Phase_Certainties;
		cl_mem      d_q11, d_q12, d_q13, d_q21, d_q22, d_q23;
		cl_mem		c_Quadrature_Filter_1, c_Quadrature_Filter_2, c_Quadrature_Filter_3;
		cl_mem		c_Parameter_Vector;
	
		// Motion correction
		cl_mem		d_Motion_Corrected_fMRI_Volumes;
	
		//
		cl_mem		d_T1_Volume, d_Interpolated_T1_Volume, d_MNI_Volume, d_Interpolated_fMRI_Volume;
			
		// Smoothing
		cl_mem		d_Smoothed_Certainty;
		cl_mem		d_Smoothed_fMRI_Volumes;
		

		// Detrending
		cl_mem		d_Detrended_fMRI_Volumes;
		cl_mem		c_X_Detrend;

		// Statistical analysis
		cl_mem		d_Beta_Volumes;
		cl_mem		d_Statistical_Maps;
		float		c_xtxxt_Detrend;
		cl_mem		c_Censor;
		cl_mem		c_xtxxt_GLM, c_X_GLM, c_Contrast_Vectors;
		cl_mem		d_Beta_Contrasts;
		cl_mem		d_Residual_Volumes;
		cl_mem		d_Residual_Variances;

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