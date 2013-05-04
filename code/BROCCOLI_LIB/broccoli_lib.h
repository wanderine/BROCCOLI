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
#include <string>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <cstdlib>


typedef float2 Complex;
typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short int uint16;

#define CCA 0
#define GLM 1

#define SMOOTHING_2D 2
#define SMOOTHING_3D 3

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
#define MOTION_COMPENSATION 1
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
#define CCA2D1 9
#define CCA2D2 10
#define CCA2D3 11
#define CCA2D4 12
#define CCA2D 13
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
#define MOTION_COMPENSATED_VOLUMES 27
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
#define MOTION_COMPENSATED_CURVE 50
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

		// Set functions for GUI
		void SetfMRIDataFilename(std::string filename);
			
		void SetfMRIParameters(float tr, float xs, float ys, float zs);
		void SetNumberOfIterationsForMotionCompensation(int N);
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

		void SetXSize(float value);
		void SetYSize(float value);
		void SetZSize(float value);
		void SetTR(float value);

		void SetWidth(int w);
		void SetHeight(int h);
		void SetDepth(int d);
		void SetTimepoints(int t);

		void SetNumberOfPermutations(int value);
		void SetSignificanceThreshold(float value);

		// Get functions for GUI
		double GetProcessingTimeSliceTimingCorrection();
		double GetProcessingTimeMotionCompensation();
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

		float GetXSize();
		float GetYSize();
		float GetZSize();
		float GetTR();

		double* GetMotionParametersX();
		double* GetMotionParametersY();
		double* GetMotionParametersZ();
		double* GetPlotValuesX();

		double* GetMotionCompensatedCurve();
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

		void ReadfMRIDataRAW();
		void ReadfMRIDataNIFTI();
		void ReadNIFTIHeader();

		// Calculations
		void PerformPreprocessingAndCalculateActivityMap();
		void CalculatePermutationTestThreshold();

		std::string PrintGPUInfo();

		// Calculations		
		void PerformSliceTimingCorrection();
		void PerformMotionCompensation();
		void PerformDetrending();
		void PerformSmoothing(); 
		void CalculateActivityMap();

			
		void SetupParametersPermutation();
		void GeneratePermutationMatrix();
		void PerformDetrendingPriorPermutation();
		void CreateBOLDRegressedVolumes();
		void SmoothingSingleVolume(float* d_Smoothed_Volume, float* d_Volume, float* d_Certainty, float* d_Smoothed_Alpha_Certainty, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE);
		void PerformWhiteningPriorPermutation();
		void GeneratePermutedfMRIVolumes();
		void PerformDetrendingPermutation();
		void PerformSmoothingPermutation(); 
		void CalculateActivityMapPermutation();
		float FindMaxTestvaluePermutation();
		float FindMaxTestvaluePermutationOld();

		void CalculateSlicesfMRIData();
		void CalculateSlicesPreprocessedfMRIData();
		void CalculateSlicesActivityData();

	private:

		StopWatchInterface* hTimer;
		StopWatchInterface* hTimer2;

		// Read and write functions
		void SetupParametersReadData();

		void ReadRealDataInt32(int* data, std::string filename, int N);
		void ReadRealDataInt16(short int* data, std::string filename, int N);
		void ReadRealDataUint32(unsigned int* data, std::string filename, int N);
		void ReadRealDataUint16(unsigned short int* data, std::string filename, int N);

		void ReadRealDataFloat(float* data, std::string filename, int N);
		void ReadRealDataDouble(double* data, std::string filename, int N);
		void WriteRealDataUint16(unsigned short int* data, std::string filename, int N);
		void WriteRealDataFloat(float* data, std::string filename, int N);
		void WriteRealDataDouble(double* data, std::string filename, int N);
		void ReadComplexData(Complex* data, std::string real_filename, std::string imag_filename, int N);
		void WriteComplexData(Complex* data, std::string real_filename, std::string imag_filename, int N);
		void ReadMotionCompensationFilters();
		void ReadSmoothingFilters();

		// Help functions
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

		void ConvolveWithHRF(float* temp_GLM);
		void CreateHRF();

		// General
		nifti_image *nifti_data;

		int DATA_W, DATA_H, DATA_D, DATA_T, DATA_T_PADDED;
		float SEGMENTATION_THRESHOLD;
		float x_size, y_size, z_size;
		int PRINT;
		int WRITE_DATA;
		int FILE_TYPE;
		int DATA_TYPE;
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

		// Motion compensation
		bool MOTION_COMPENSATED;
		int NUMBER_OF_ITERATIONS_FOR_MOTION_COMPENSATION;
		int MOTION_COMPENSATION_FILTER_SIZE;
		int	NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS;
		
		double* motion_parameters_x;
		double* motion_parameters_y;
		double* motion_parameters_z;

		double* motion_compensated_curve;

		// Smoothing
		int SMOOTHING_DIMENSIONALITY;
		int	SMOOTHING_FILTER_SIZE;
		int	SMOOTHING_AMOUNT_MM;
			
		double* smoothed_curve;

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
		float* h_Contrast_Vector;

		// Random permutations
		int NUMBER_OF_PERMUTATIONS;
		float significance_threshold;
		float permutation_test_threshold;
		int NUMBER_OF_SIGNIFICANTLY_ACTIVE_VOXELS;
		
		/*--------------------------------------------------*/
		// Host pointers
		/*--------------------------------------------------*/
			
		void		*host_pointers[NUMBER_OF_HOST_POINTERS];
		void		*host_pointers_static[NUMBER_OF_HOST_POINTERS];
		void		*host_pointers_permutation[NUMBER_OF_HOST_POINTERS];
		
		float		*h_fMRI_Volumes;
		Complex	 	*h_fMRI_Volumes_Complex;	

		// Slice timing correction
		float		*h_Slice_Timing_Corrections_Real, *h_Slice_Timing_Corrections_Imag;
		Complex		*h_Slice_Timing_Corrections;
		float		*h_Slice_Timing_Corrected_fMRI_Volumes;

		// Motion compensation
		float		*h_Quadrature_Filter_1_Real, *h_Quadrature_Filter_1_Imag, *h_Quadrature_Filter_2_Real, *h_Quadrature_Filter_2_Imag, *h_Quadrature_Filter_3_Real, *h_Quadrature_Filter_3_Imag; 		
		Complex     *h_Quadrature_Filter_1, *h_Quadrature_Filter_2, *h_Quadrature_Filter_3;
		float		*h_Motion_Compensated_fMRI_Volumes;
		float		*h_Registration_Parameters;

		// Smoothing
		float		*h_CCA_2D_Filter_1;
		float		*h_CCA_2D_Filter_2;
		float		*h_CCA_2D_Filter_3;
		float		*h_CCA_2D_Filter_4;
		float4		*h_CCA_2D_Filters;
		float		*h_CCA_3D_Filter_1;
		float		*h_CCA_3D_Filter_2;
		float2		*h_CCA_3D_Filters;
		float		*h_GLM_Filter;
		float		*h_Smoothed_fMRI_Volumes_1;
		float		*h_Smoothed_fMRI_Volumes_2;
		float		*h_Smoothed_fMRI_Volumes_3;
		float		*h_Smoothed_fMRI_Volumes_4;

		// Detrending
		float		*h_X_Detrend, *h_xtxxt_Detrend;
		float		*h_Detrended_fMRI_Volumes_1;
		float		*h_Detrended_fMRI_Volumes_2;
		float		*h_Detrended_fMRI_Volumes_3;
		float		*h_Detrended_fMRI_Volumes_4;
			
		// Statistical analysis
		float		*hrf;
		int			 hrf_length;
		float		*h_Cxx, *h_sqrt_inv_Cxx;
		float		*h_X_GLM, *h_xtxxt_GLM;
		float		 h_ctxtxc;			
		float		*h_Activity_Volume;
			
		// Random permutations
		float		*h_Alpha_Smoothing_Kernel;
		float		*h_Smoothed_Alpha_Certainty;			
		uint16		*h_Permutation_Matrix;
		float		*h_Maximum_Test_Values;

		// Covariance pooling
		float		*h_Variance_Smoothing_Kernel;
		float		*h_Smoothed_Variance_Certainty;

		/*--------------------------------------------------*/
		// Device pointers
		/*--------------------------------------------------*/

		float		*device_pointers[NUMBER_OF_DEVICE_POINTERS];
		float		*device_pointers_permutation[NUMBER_OF_DEVICE_POINTERS];
		
		float		*d_fMRI_Volumes;
		float		*d_Brain_Voxels;
			
		// Slice timing correction
		Complex		*d_fMRI_Volumes_Complex;
		float		*d_Shifters;
		float		*d_Slice_Timing_Corrected_fMRI_Volumes;

		// Motion compensation
		float		*d_Motion_Compensated_fMRI_Volumes;
			
		// Smoothing
		float		*d_Smoothed_Certainty;
		float		*d_Smoothed_fMRI_Volumes_1;
		float		*d_Smoothed_fMRI_Volumes_2;
		float		*d_Smoothed_fMRI_Volumes_3;
		float		*d_Smoothed_fMRI_Volumes_4;

		// Detrending
		float		*d_Detrended_fMRI_Volumes_1;
		float		*d_Detrended_fMRI_Volumes_2;
		float		*d_Detrended_fMRI_Volumes_3;
		float		*d_Detrended_fMRI_Volumes_4;

		// Statistical analysis
		float		*d_Activity_Volume;

		// Random permutations
		float		*d_Alphas_1, *d_Alphas_2, *d_Alphas_3, *d_Alphas_4;
		float		*d_Smoothed_Alphas_1, *d_Smoothed_Alphas_2, *d_Smoothed_Alphas_3, *d_Smoothed_Alphas_4;
			
		float		*d_BOLD_Regressed_fMRI_Volumes;
		float		*d_Whitened_fMRI_Volumes;
		float		*d_Permuted_fMRI_Volumes;


		/*--------------------------------------------------*/
		// Filenames
		/*--------------------------------------------------*/
		std::string		filename_real_quadrature_filter_1;
		std::string		filename_real_quadrature_filter_2;
		std::string		filename_real_quadrature_filter_3;
		std::string		filename_imag_quadrature_filter_1;
		std::string		filename_imag_quadrature_filter_2;
		std::string		filename_imag_quadrature_filter_3;
		std::string		filename_GLM_filter;
		std::string		filename_CCA_2D_filter_1;
		std::string		filename_CCA_2D_filter_2;
		std::string		filename_CCA_2D_filter_3;
		std::string		filename_CCA_2D_filter_4;
		std::string		filename_CCA_3D_filter_1;
		std::string		filename_CCA_3D_filter_2;
		std::string		filename_fMRI_data;

		std::string		filename_slice_timing_corrected_fMRI_volumes;
		std::string		filename_registration_parameters;
		std::string		filename_motion_compensated_fMRI_volumes;
		std::string		filename_smoothed_fMRI_volumes_1;
		std::string		filename_smoothed_fMRI_volumes_2;
		std::string		filename_smoothed_fMRI_volumes_3;
		std::string		filename_smoothed_fMRI_volumes_4;
		std::string		filename_detrended_fMRI_volumes_1;
		std::string		filename_detrended_fMRI_volumes_2;
		std::string		filename_detrended_fMRI_volumes_3;
		std::string		filename_detrended_fMRI_volumes_4;
		std::string		filename_activity_volume;
};


#endif