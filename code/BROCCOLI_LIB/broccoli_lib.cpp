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
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>

//#include <shrUtils.h>
//#include <shrQATest.h>
#include "BROCCOLI_lib.h"
#include "BROCCOLI_kernel.cu"

#include "nifti1.h"
#include "nifti1_io.h"

#include <cstdlib>


/* 

to do

spm hrf
slice timing correction

*/


// public


// Constructor
BROCCOLI_LIB::BROCCOLI_LIB()
{
	PREPROCESSED = MOTION_COMPENSATION;

	FILE_TYPE = RAW;
	DATA_TYPE = FLOAT;

	DATA_W = 64;
	DATA_H = 64;
	DATA_D = 22;
	DATA_T = 79;

	x_size = 3.75f;
	y_size = 3.75f;
	z_size = 3.75f;
	TR = 2.0f;
	
	NUMBER_OF_PERMUTATIONS = 1000;
	significance_threshold = 0.05f;

	filename_real_quadrature_filter_1 = "filters\\quadrature_filter_1_real.raw";
	filename_real_quadrature_filter_2 = "filters\\quadrature_filter_2_real.raw";
	filename_real_quadrature_filter_3 = "filters\\quadrature_filter_3_real.raw";
	filename_imag_quadrature_filter_1 = "filters\\quadrature_filter_1_imag.raw";
	filename_imag_quadrature_filter_2 = "filters\\quadrature_filter_2_imag.raw";
	filename_imag_quadrature_filter_3 = "filters\\quadrature_filter_3_imag.raw";

	filename_GLM_filter = "filters\\GLM_smoothing_filter_";
	filename_CCA_2D_filter_1 = "filters\\CCA_2D_smoothing_filter_1_";
	filename_CCA_2D_filter_2 = "filters\\CCA_2D_smoothing_filter_2_";
	filename_CCA_2D_filter_3 = "filters\\CCA_2D_smoothing_filter_3_";
	filename_CCA_2D_filter_4 = "filters\\CCA_2D_smoothing_filter_4_";
	filename_CCA_3D_filter_1 = "filters\\CCA_3D_smoothing_filter_1_";
	filename_CCA_3D_filter_2 = "filters\\CCA_3D_smoothing_filter_2_";
	filename_fMRI_data = "fMRI_data.raw";

	filename_slice_timing_corrected_fMRI_volumes = "output\\slice_timing_corrected_fMRI_volumes.raw";
	filename_registration_parameters = "output\\registration_parameters.raw";
	filename_motion_compensated_fMRI_volumes = "output\\motion_compensated_fMRI_volumes.raw";
	filename_smoothed_fMRI_volumes_1 = "output\\smoothed_fMRI_volumes_1.raw";
	filename_smoothed_fMRI_volumes_2 = "output\\smoothed_fMRI_volumes_2.raw";
	filename_smoothed_fMRI_volumes_3 = "output\\smoothed_fMRI_volumes_3.raw";
	filename_smoothed_fMRI_volumes_4 = "output\\smoothed_fMRI_volumes_4.raw";
	filename_detrended_fMRI_volumes_1 = "output\\detrended_fMRI_volumes_1.raw";
	filename_detrended_fMRI_volumes_2 = "output\\detrended_fMRI_volumes_2.raw";
	filename_detrended_fMRI_volumes_3 = "output\\detrended_fMRI_volumes_3.raw";
	filename_detrended_fMRI_volumes_4 = "output\\detrended_fMRI_volumes_4.raw";
	filename_activity_volume = "output\\activity_volume.raw";

	THRESHOLD_ACTIVITY_MAP = false;
	ACTIVITY_THRESHOLD = 0.5f;

	MOTION_COMPENSATED = false;
	MOTION_COMPENSATION_FILTER_SIZE = 7;
	NUMBER_OF_ITERATIONS_FOR_MOTION_COMPENSATION = 3;
	NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS = 30;
	
	SMOOTHING_DIMENSIONALITY = 2;
	SMOOTHING_AMOUNT_MM = 8;
	SMOOTHING_FILTER_SIZE = 9;
	
	NUMBER_OF_DETRENDING_BASIS_FUNCTIONS = 4;

	SEGMENTATION_THRESHOLD = 600.0f;
	NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS = 2;
	ANALYSIS_METHOD = CCA;
	NUMBER_OF_PERIODS = 4;
	PERIOD_TIME = 20;

	PRINT = VERBOSE;
	WRITE_DATA = NO;

	int DATA_SIZE_QUADRATURE_FILTER_REAL = sizeof(float) * MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE;
	int DATA_SIZE_QUADRATURE_FILTER_COMPLEX = sizeof(Complex) * MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE;

	int DATA_SIZE_SMOOTHING_FILTER_2D_CCA = sizeof(float) * SMOOTHING_FILTER_SIZE * SMOOTHING_FILTER_SIZE;
	int DATA_SIZE_SMOOTHING_FILTERS_2D_CCA = sizeof(float4) * SMOOTHING_FILTER_SIZE * SMOOTHING_FILTER_SIZE;
	int DATA_SIZE_SMOOTHING_FILTER_3D_CCA = sizeof(float) * SMOOTHING_FILTER_SIZE * SMOOTHING_FILTER_SIZE * SMOOTHING_FILTER_SIZE;
	int DATA_SIZE_SMOOTHING_FILTERS_3D_CCA = sizeof(float2) * SMOOTHING_FILTER_SIZE * SMOOTHING_FILTER_SIZE;
	int DATA_SIZE_SMOOTHING_FILTER_GLM = sizeof(float) * SMOOTHING_FILTER_SIZE;

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
	
	h_Quadrature_Filter_1_Real = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL);
	h_Quadrature_Filter_1_Imag = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL);
	h_Quadrature_Filter_2_Real = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL);
	h_Quadrature_Filter_2_Imag = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL);
	h_Quadrature_Filter_3_Real = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL);
	h_Quadrature_Filter_3_Imag = (float*)malloc(DATA_SIZE_QUADRATURE_FILTER_REAL); 		
	h_Quadrature_Filter_1 = (Complex*)malloc(DATA_SIZE_QUADRATURE_FILTER_COMPLEX);
	h_Quadrature_Filter_2 = (Complex*)malloc(DATA_SIZE_QUADRATURE_FILTER_COMPLEX);
	h_Quadrature_Filter_3 = (Complex*)malloc(DATA_SIZE_QUADRATURE_FILTER_COMPLEX);

	h_CCA_2D_Filter_1 = (float*)malloc(DATA_SIZE_SMOOTHING_FILTER_2D_CCA);
	h_CCA_2D_Filter_2 = (float*)malloc(DATA_SIZE_SMOOTHING_FILTER_2D_CCA);
	h_CCA_2D_Filter_3 = (float*)malloc(DATA_SIZE_SMOOTHING_FILTER_2D_CCA);
	h_CCA_2D_Filter_4 = (float*)malloc(DATA_SIZE_SMOOTHING_FILTER_2D_CCA);
	h_CCA_2D_Filters = (float4*)malloc(DATA_SIZE_SMOOTHING_FILTERS_2D_CCA);
	h_CCA_3D_Filter_1 = (float*)malloc(DATA_SIZE_SMOOTHING_FILTER_3D_CCA);
	h_CCA_3D_Filter_2 = (float*)malloc(DATA_SIZE_SMOOTHING_FILTER_3D_CCA);
	h_CCA_3D_Filters = (float2*)malloc(DATA_SIZE_SMOOTHING_FILTERS_3D_CCA);
	h_GLM_Filter = (float*)malloc(DATA_SIZE_SMOOTHING_FILTER_GLM);

	host_pointers_static[QF1R]   = (void*)h_Quadrature_Filter_1_Real;
	host_pointers_static[QF1I]   = (void*)h_Quadrature_Filter_1_Imag;
	host_pointers_static[QF2R]   = (void*)h_Quadrature_Filter_2_Real;
	host_pointers_static[QF2I]   = (void*)h_Quadrature_Filter_2_Imag;
	host_pointers_static[QF3R]   = (void*)h_Quadrature_Filter_3_Real;
	host_pointers_static[QF3I]   = (void*)h_Quadrature_Filter_3_Imag;
	host_pointers_static[QF1]    = (void*)h_Quadrature_Filter_1;
	host_pointers_static[QF2]    = (void*)h_Quadrature_Filter_2;
	host_pointers_static[QF3]	 = (void*)h_Quadrature_Filter_3;
	host_pointers_static[CCA2D1] = (void*)h_CCA_2D_Filter_1;
	host_pointers_static[CCA2D2] = (void*)h_CCA_2D_Filter_2;
	host_pointers_static[CCA2D3] = (void*)h_CCA_2D_Filter_3;
	host_pointers_static[CCA2D4] = (void*)h_CCA_2D_Filter_4;
	host_pointers_static[CCA2D]  = (void*)h_CCA_2D_Filters;
	host_pointers_static[CCA3D1] = (void*)h_CCA_3D_Filter_1;
	host_pointers_static[CCA3D2] = (void*)h_CCA_3D_Filter_2;
	host_pointers_static[CCA3D]  = (void*)h_CCA_3D_Filters;
	
	ReadMotionCompensationFilters();
	ReadSmoothingFilters();	
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
			cudaFree(pointer);
		}
	}
	
	for (int i = 0; i < NUMBER_OF_DEVICE_POINTERS; i++)
	{
		float* pointer = device_pointers_permutation[i];
		if (pointer != NULL)
		{
			cudaFree(pointer);
		}
	}
}







// Set functions for GUI

void BROCCOLI_LIB::SetDataType(int type)
{
	DATA_TYPE = type;
}

void BROCCOLI_LIB::SetFileType(int type)
{
	FILE_TYPE = type;
}

void BROCCOLI_LIB::SetNumberOfIterationsForMotionCompensation(int N)
{
	NUMBER_OF_ITERATIONS_FOR_MOTION_COMPENSATION = N;
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

void BROCCOLI_LIB::SetSmoothingDimensionality(int dimensionality)
{
	SMOOTHING_DIMENSIONALITY = dimensionality;
	ReadSmoothingFilters();
}

void BROCCOLI_LIB::SetNumberOfBasisFunctionsDetrending(int N)
{
	NUMBER_OF_DETRENDING_BASIS_FUNCTIONS = N;
}

void BROCCOLI_LIB::SetfMRIDataFilename(std::string filename)
{
	filename_fMRI_data = filename;
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

void BROCCOLI_LIB::SetXSize(float value)
{
	x_size = value;
}

void BROCCOLI_LIB::SetYSize(float value)
{
	y_size = value;
}

void BROCCOLI_LIB::SetZSize(float value)
{
	z_size = value;
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









// Get functions for GUI

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
	return processing_times[MOTION_COMPENSATION];
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

double BROCCOLI_LIB::GetProcessingTimeCopy()
{
	return processing_times[COPY];
}

// Returns the processing time for convolution in the motion correction	
double BROCCOLI_LIB::GetProcessingTimeConvolution()
{
	return processing_times[CONVOLVE];
}

// Returns the processing time for calculation of phase differences in the motion correction
double BROCCOLI_LIB::GetProcessingTimePhaseDifferences()
{
	return processing_times[PHASEDC];
}

// Returns the processing time for calculation of phase gradients in the motion correction
double BROCCOLI_LIB::GetProcessingTimePhaseGradients()
{
	return processing_times[PHASEG];
}

// Returns the processing time for calculation of A-matrix and h-vector in the motion correction
double BROCCOLI_LIB::BROCCOLI_LIB::GetProcessingTimeAH()
{
	return processing_times[AH2D];
}

// Returns the processing time for solving the equation system in the motion correction
double BROCCOLI_LIB::GetProcessingTimeEquationSystem()
{
	return processing_times[EQSYSTEM];
}

// Returns the processing time for the interpolation step in the motion correction
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
float BROCCOLI_LIB::GetXSize()
{
	return x_size;
}

// Returns the voxel size (in mm) in the y direction
float BROCCOLI_LIB::GetYSize()
{
	return y_size;
}

// Returns the voxel size (in mm) in the z direction
float BROCCOLI_LIB::GetZSize()
{
	return z_size;
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
double* BROCCOLI_LIB::GetMotionCompensatedCurve()
{
	for (int t = 0; t < DATA_T; t++)
	{
		motion_compensated_curve[t] = (double)h_Motion_Compensated_fMRI_Volumes[X_SLICE_LOCATION_fMRI_DATA + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D];
	}

	return motion_compensated_curve;
}

// Returns the timeseries of the smoothed data for the current mouse location in the GUI
double* BROCCOLI_LIB::GetSmoothedCurve()
{
	for (int t = 0; t < DATA_T; t++)
	{
		smoothed_curve[t] = (double)h_Smoothed_fMRI_Volumes_1[X_SLICE_LOCATION_fMRI_DATA + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D];
	}

	return smoothed_curve;
}

// Returns the timeseries of the detrended data for the current mouse location in the GUI
double* BROCCOLI_LIB::GetDetrendedCurve()
{
	for (int t = 0; t < DATA_T; t++)
	{
		detrended_curve[t] = (double)h_Detrended_fMRI_Volumes_1[X_SLICE_LOCATION_fMRI_DATA + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D];
	}

	return detrended_curve;
}

// Returns the filename of the current fMRI dataset
std::string BROCCOLI_LIB::GetfMRIDataFilename()
{
	return filename_fMRI_data;
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

}













// Processing



// Preprocessing

// Performs registration between one low resolution fMRI volume and a high resolution T1 volume
void BROCCOLI_LIB::PerformRegistrationEPIT1()
{
	
}

// Performs registration between one high resolution T1 volume and a high resolution MNI volume
void BROCCOLI_LIB::PerformRegistrationT1MNI()
{
	
}

// Performs slice timing correction of an fMRI dataset
void BROCCOLI_LIB::PerformSliceTimingCorrection()
{
	
}

// Performs motion correction of an fMRI dataset
void BROCCOLI_LIB::PerformMotionCorrection()
{
	
}

// Performs smoothing of an fMRI dataset
void BROCCOLI_LIB::PerformSmoothing()
{
	
}

// Performs detrending of an fMRI dataset
void BROCCOLI_LIB::PerformDetrending()
{
	
}

// Processing

// Runs all the preprocessing steps and the statistical analysis for one subject
void BROCCOLI_LIB::PerformPreprocessingAndCalculateActivityMap()
{
	PerformRegistrationEPIT1();
	PerformRegistrationT1MNI();

    //PerformSliceTimingCorrection();
	PerformMotionCorrection();
	PerformSmoothing();	
	PerformDetrending();
	CalculateActivityMap();

	CalculateSlicesPreprocessedfMRIData();
}

// Calculates an activity map for a single subject
void BROCCOLI_LIB::CalculateActivityMap()
{
	
}

// Calculates an activity map for a group analysis
void BROCCOLI_LIB::CalculateGroupMap()
{
	// First calculate activity values for each subject ?

	PerformPreprocessingAndCalculateActivityMap();
}

// Calculates a significance threshold for a single subject
void BROCCOLI_LIB::CalculatePermutationTestThresholdSingleSubject()
{
	SetupParametersPermutation();
	GeneratePermutationMatrix();

	// Make the timeseries white prior to the random permutations (if single subject)
	PerformDetrendingPriorPermutation();
	CreateBOLDRegressedVolumes();
	PerformWhiteningPriorPermutation();
	
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
		if (h_Activity_Volume[i] >= permutation_test_threshold)
		{
			NUMBER_OF_SIGNIFICANTLY_ACTIVE_VOXELS++;
		}
	}
}

// Calculates a significance threshold for a group of subjects
void BROCCOLI_LIB::CalculatePermutationTestThresholdMultiSubject()
{
	SetupParametersPermutation();
	GeneratePermutationMatrix();

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
		if (h_Activity_Volume[i] >= permutation_test_threshold)
		{
			NUMBER_OF_SIGNIFICANTLY_ACTIVE_VOXELS++;
		}
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

void BROCCOLI_LIB::PerformWhiteningPriorPermutation()
{
}

void BROCCOLI_LIB::GeneratePermutedfMRIVolumes()
{
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
	cudaMemcpy(h_Activity_Volume, d_Activity_Volume, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyDeviceToHost);
	thrust::host_vector<float> h_vec(h_Activity_Volume, &h_Activity_Volume[DATA_W * DATA_H * DATA_D]); 
	
	thrust::device_vector<float> d_vec = h_vec;
	//thrust::device_vector<float> d_vec(d_Activity_Volume, &d_Activity_Volume[DATA_W * DATA_H * DATA_D]);

    return thrust::reduce(d_vec.begin(), d_vec.end(), -1000.0f, thrust::maximum<float>());
}

// Functions for permutations, single subject

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
		ReadRealDataFloat(h_fMRI_Volumes, filename_fMRI_data, DATA_W * DATA_H * DATA_D * DATA_T);
	}
	else if (DATA_TYPE == INT32)
	{
		int* h_Temp_Volumes = (int*)malloc(sizeof(int) * DATA_W * DATA_H * DATA_D * DATA_T);
		ReadRealDataInt32(h_Temp_Volumes, filename_fMRI_data, DATA_W * DATA_H * DATA_D * DATA_T);
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)h_Temp_Volumes[i];
		}
		free(h_Temp_Volumes);
	}
	else if (DATA_TYPE == INT16)
	{
		short int* h_Temp_Volumes = (short int*)malloc(sizeof(short int) * DATA_W * DATA_H * DATA_D * DATA_T);
		ReadRealDataInt16(h_Temp_Volumes, filename_fMRI_data, DATA_W * DATA_H * DATA_D * DATA_T);
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)h_Temp_Volumes[i];
		}
		free(h_Temp_Volumes);
	}
	else if (DATA_TYPE == UINT32)
	{
		unsigned int* h_Temp_Volumes = (unsigned int*)malloc(sizeof(unsigned int) * DATA_W * DATA_H * DATA_D * DATA_T);
		ReadRealDataUint32(h_Temp_Volumes, filename_fMRI_data, DATA_W * DATA_H * DATA_D * DATA_T);
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)h_Temp_Volumes[i];
		}
		free(h_Temp_Volumes);
	}
	else if (DATA_TYPE == UINT16)
	{
		unsigned short int* h_Temp_Volumes = (unsigned short int*)malloc(sizeof(unsigned short int) * DATA_W * DATA_H * DATA_D * DATA_T);
		ReadRealDataUint16(h_Temp_Volumes, filename_fMRI_data, DATA_W * DATA_H * DATA_D * DATA_T);
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)h_Temp_Volumes[i];
		}
		free(h_Temp_Volumes);
	}
	else if (DATA_TYPE == DOUBLE)
	{
		double* h_Temp_Volumes = (double*)malloc(sizeof(double) * DATA_W * DATA_H * DATA_D * DATA_T);
		ReadRealDataDouble(h_Temp_Volumes, filename_fMRI_data, DATA_W * DATA_H * DATA_D * DATA_T);
		for (int i = 0; i < (DATA_W * DATA_H * DATA_D * DATA_T); i++)
		{
			h_fMRI_Volumes[i] = (float)h_Temp_Volumes[i];
		}
		free(h_Temp_Volumes);
	}

	// Copy fMRI volumes to global memory, as floats
	cudaMemcpy(d_fMRI_Volumes, h_fMRI_Volumes, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyHostToDevice);

	for (int i = 0; i < DATA_T; i++)
	{
		plot_values_x[i] = (double)i * (double)TR;
	}


	SegmentBrainData();

	SetupStatisticalAnalysisBasisFunctions();
	SetupDetrendingBasisFunctions();

	CalculateSlicesfMRIData();
}

void BROCCOLI_LIB::ReadfMRIDataNIFTI()
{
	nifti_data = new nifti_image;
	// Read nifti data
	nifti_data = nifti_image_read(filename_fMRI_data.c_str(), 1);

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

	x_size = nifti_data->dx;
	y_size = nifti_data->dy;
	z_size = nifti_data->dz;
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
	cudaMemcpy(d_fMRI_Volumes, h_fMRI_Volumes, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyHostToDevice);

	for (int i = 0; i < DATA_T; i++)
	{
		plot_values_x[i] = (double)i * (double)TR;
	}

	SegmentBrainData();

	SetupStatisticalAnalysisBasisFunctions();
	SetupDetrendingBasisFunctions();

	CalculateSlicesfMRIData();
}

void BROCCOLI_LIB::ReadNIFTIHeader()
{
	// Read nifti header only
	nifti_data = nifti_image_read(filename_fMRI_data.c_str(), 0);

	// Get dimensions
	DATA_W = nifti_data->nx;
	DATA_H = nifti_data->ny;
	DATA_D = nifti_data->nz;
	DATA_T = nifti_data->nt;

	x_size = nifti_data->dx;
	y_size = nifti_data->dy;
	z_size = nifti_data->dz;
	TR = nifti_data->dt;
}

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

void BROCCOLI_LIB::ReadMotionCompensationFiltersFromFile()
{
	// Read the quadrature filters from file
	ReadComplexData(h_Quadrature_Filter_1, filename_real_quadrature_filter_1, filename_imag_quadrature_filter_1, MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE);
	ReadComplexData(h_Quadrature_Filter_2, filename_real_quadrature_filter_2, filename_imag_quadrature_filter_2, MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE);
	ReadComplexData(h_Quadrature_Filter_3, filename_real_quadrature_filter_3, filename_imag_quadrature_filter_3, MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE);
}

void BROCCOLI_LIB::ReadSmoothingFiltersFromFile()
{
	// Read smoothing filters from file
	std::string mm_string;
	std::stringstream out;
	out << SMOOTHING_AMOUNT_MM;
	mm_string = out.str();

	std::string filename_GLM = filename_GLM_filter + mm_string + "mm.raw";
	ReadRealDataFloat(h_GLM_Filter, filename_GLM, SMOOTHING_FILTER_SIZE);
	
	std::string filename_CCA_2D_1 = filename_CCA_2D_filter_1 + mm_string + "mm.raw";
	std::string filename_CCA_2D_2 = filename_CCA_2D_filter_2 + mm_string + "mm.raw";
	std::string filename_CCA_2D_3 = filename_CCA_2D_filter_3 + mm_string + "mm.raw";
	std::string filename_CCA_2D_4 = filename_CCA_2D_filter_4 + mm_string + "mm.raw";
	ReadRealDataFloat(h_CCA_2D_Filter_1, filename_CCA_2D_1, SMOOTHING_FILTER_SIZE * SMOOTHING_FILTER_SIZE);
	ReadRealDataFloat(h_CCA_2D_Filter_2, filename_CCA_2D_2, SMOOTHING_FILTER_SIZE * SMOOTHING_FILTER_SIZE);
	ReadRealDataFloat(h_CCA_2D_Filter_3, filename_CCA_2D_3, SMOOTHING_FILTER_SIZE * SMOOTHING_FILTER_SIZE);
	ReadRealDataFloat(h_CCA_2D_Filter_4, filename_CCA_2D_4, SMOOTHING_FILTER_SIZE * SMOOTHING_FILTER_SIZE);
	Convert4FloatToFloat4(h_CCA_2D_Filters, h_CCA_2D_Filter_1, h_CCA_2D_Filter_2, h_CCA_2D_Filter_3, h_CCA_2D_Filter_4, SMOOTHING_FILTER_SIZE * SMOOTHING_FILTER_SIZE);

	std::string filename_CCA_3D_1 = filename_CCA_3D_filter_1 + mm_string + "mm.raw";
	std::string filename_CCA_3D_2 = filename_CCA_3D_filter_2 + mm_string + "mm.raw";
	ReadRealDataFloat(h_CCA_3D_Filter_1, filename_CCA_3D_1, SMOOTHING_FILTER_SIZE);
	ReadRealDataFloat(h_CCA_3D_Filter_2, filename_CCA_3D_2, SMOOTHING_FILTER_SIZE);
	Convert2FloatToFloat2(h_CCA_3D_Filters, h_CCA_3D_Filter_1, h_CCA_3D_Filter_2, SMOOTHING_FILTER_SIZE);
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
			cudaFree(pointer);
			device_pointers[i] = NULL;
		}
	}

	MOTION_CORRECTED = false;

	X_SLICE_LOCATION_fMRI_DATA = DATA_W / 2;
	Y_SLICE_LOCATION_fMRI_DATA = DATA_H / 2;
	Z_SLICE_LOCATION_fMRI_DATA = DATA_D / 2;
	TIMEPOINT_fMRI_DATA = 0;

	int DATA_SIZE_fMRI_VOLUME = sizeof(float) * DATA_W * DATA_H * DATA_D;
	int DATA_SIZE_fMRI_VOLUMES = sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T;
	
	int DATA_SIZE_DETRENDING = sizeof(float) * DATA_T * NUMBER_OF_DETRENDING_BASIS_FUNCTIONS;
	
	int DATA_SIZE_TEMPORAL_BASIS_FUNCTIONS = sizeof(float) * DATA_T * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS;
	int DATA_SIZE_COVARIANCE_MATRIX = sizeof(float) * 4;

	h_fMRI_Volumes = (float*)malloc(DATA_SIZE_fMRI_VOLUMES);
	h_Motion_Compensated_fMRI_Volumes = (float*)malloc(DATA_SIZE_fMRI_VOLUMES);
	h_Smoothed_fMRI_Volumes_1 = (float*)malloc(DATA_SIZE_fMRI_VOLUMES);
	h_Detrended_fMRI_Volumes_1 = (float*)malloc(DATA_SIZE_fMRI_VOLUMES);
	
	h_X_Detrend = (float*)malloc(DATA_SIZE_DETRENDING);
	h_xtxxt_Detrend = (float*)malloc(DATA_SIZE_DETRENDING);	

	h_Cxx = (float*)malloc(DATA_SIZE_COVARIANCE_MATRIX);
	h_sqrt_inv_Cxx = (float*)malloc(DATA_SIZE_COVARIANCE_MATRIX);
	h_X_GLM = (float*)malloc(DATA_SIZE_TEMPORAL_BASIS_FUNCTIONS);
	h_xtxxt_GLM = (float*)malloc(DATA_SIZE_TEMPORAL_BASIS_FUNCTIONS);
	h_Contrast_Vector = (float*)malloc(sizeof(float) * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS);
	
	h_Activity_Volume = (float*)malloc(DATA_SIZE_fMRI_VOLUME);
	
	host_pointers[fMRI_VOLUMES] = (void*)h_fMRI_Volumes;
	host_pointers[MOTION_COMPENSATED_VOLUMES] = (void*)h_Motion_Compensated_fMRI_Volumes;
	host_pointers[SMOOTHED1] = (void*)h_Smoothed_fMRI_Volumes_1;
	host_pointers[DETRENDED1] = (void*)h_Detrended_fMRI_Volumes_1;
	host_pointers[XDETREND1] = (void*)h_X_Detrend;
	host_pointers[XDETREND2] = (void*)h_xtxxt_Detrend;
	host_pointers[CXX] = (void*)h_Cxx;
	host_pointers[SQRTINVCXX] = (void*)h_sqrt_inv_Cxx;
	host_pointers[XGLM1] = (void*)h_X_GLM;
	host_pointers[XGLM2] = (void*)h_xtxxt_GLM;
	host_pointers[CONTRAST_VECTOR] = (void*)h_Contrast_Vector;
	host_pointers[ACTIVITY_VOLUME] = (void*)h_Activity_Volume;

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
	motion_compensated_curve = (double*)malloc(sizeof(double) * DATA_T);
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
	host_pointers[MOTION_COMPENSATED_CURVE] = (void*)motion_compensated_curve;
	host_pointers[SMOOTHED_CURVE] = (void*)smoothed_curve;
	host_pointers[DETRENDED_CURVE] = (void*)detrended_curve;

	cudaMalloc((void **)&d_fMRI_Volumes, DATA_SIZE_fMRI_VOLUMES);
	cudaMalloc((void **)&d_Brain_Voxels, DATA_SIZE_fMRI_VOLUME);
	cudaMalloc((void **)&d_Smoothed_Certainty, DATA_SIZE_fMRI_VOLUME);
	cudaMalloc((void **)&d_Activity_Volume, DATA_SIZE_fMRI_VOLUME);

	device_pointers[fMRI_VOLUMES] = d_fMRI_Volumes;
	device_pointers[BRAIN_VOXELS] = d_Brain_Voxels;
	device_pointers[SMOOTHED_CERTAINTY] = d_Smoothed_Certainty;
	device_pointers[ACTIVITY_VOLUME] = d_Activity_Volume;

	int threadsInX = 32;
	int	threadsInY = 8;
	int	threadsInZ = 1;
		
	int	blocksInX = (DATA_W+threadsInX-1)/threadsInX;
	int	blocksInY = (DATA_H+threadsInY-1)/threadsInY;
	int	blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
	dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
	dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	MyMemset<<<dimGrid, dimBlock>>>(d_Smoothed_Certainty, 1.0f, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY));
}


// Write functions, public

void BROCCOLI_LIB::WritefMRIDataNIFTI()
{
	nifti_data = new nifti_image;
	// Read nifti data
	nifti_data = nifti_image_read(filename_fMRI_data.c_str(), 1);

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

	x_size = nifti_data->dx;
	y_size = nifti_data->dy;
	z_size = nifti_data->dz;
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
	cudaMemcpy(d_fMRI_Volumes, h_fMRI_Volumes, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyHostToDevice);

	for (int i = 0; i < DATA_T; i++)
	{
		plot_values_x[i] = (double)i * (double)TR;
	}

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





// Help functions


void BROCCOLI_LIB::CalculateSlicesActivityData()
{
	float max = CalculateMax(h_Activity_Volume, DATA_W * DATA_H * DATA_D);
	float min = CalculateMin(h_Activity_Volume, DATA_W * DATA_H * DATA_D);
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
	float* pointer;

	if (PREPROCESSED == MOTION_COMPENSATION)
	{
		pointer = h_Motion_Compensated_fMRI_Volumes;
	}
	else if (PREPROCESSED == SMOOTHING)
	{
		pointer = h_Smoothed_fMRI_Volumes_1;
	}
	else if (PREPROCESSED == DETRENDING)
	{
		pointer = h_Detrended_fMRI_Volumes_1;
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

void BROCCOLI_LIB::Convert2FloatToFloat2(float2* floats, float* float_1, float* float_2, int N)
{
	for (int i = 0; i < N; i++)
	{
		floats[i].x = float_1[i];
		floats[i].y = float_2[i];
	}
}

void BROCCOLI_LIB::ConvertRealToComplex(Complex* complex_data, float* real_data, int N)
{
	for (int i = 0; i < N; i++)
	{
		complex_data[i].x = real_data[i];
		complex_data[i].y = 0.0f;
	}
}

void BROCCOLI_LIB::ExtractRealData(float* real_data, Complex* complex_data, int N)
{
	for (int i = 0; i < N; i++)
	{
		real_data[i] = complex_data[i].x;
	}
}

void BROCCOLI_LIB::Calculate_Block_Differences2D(int& xBlockDifference, int& yBlockDifference, int DATA_W, int DATA_H, int threadsInX, int threadsInY)
{
	if ( (DATA_W % threadsInX) == 0)
	{
		xBlockDifference = 0;
	}
	else
	{
		xBlockDifference = threadsInX - (DATA_W % threadsInX);
	}

	if ( (DATA_H % threadsInY) == 0)
	{
		yBlockDifference = 0;
	}
	else
	{
		yBlockDifference = threadsInY - (DATA_H % threadsInY);
	}
}

void BROCCOLI_LIB::Calculate_Block_Differences3D(int& xBlockDifference, int& yBlockDifference, int& zBlockDifference, int DATA_W, int DATA_H, int DATA_D, int threadsInX, int threadsInY, int threadsInZ)
{
	if ( (DATA_W % threadsInX) == 0)
	{
		xBlockDifference = 0;
	}
	else
	{
		xBlockDifference = threadsInX - (DATA_W % threadsInX);
	}

	if ( (DATA_H % threadsInY) == 0)
	{
		yBlockDifference = 0;
	}
	else
	{
		yBlockDifference = threadsInY - (DATA_H % threadsInY);
	}

	if ( (DATA_D % threadsInZ) == 0)
	{
		zBlockDifference = 0;
	}
	else
	{
		zBlockDifference = threadsInZ - (DATA_D % threadsInZ);
	}
}

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

	cudaMemcpyToSymbol(c_X_Detrend, h_X_Detrend, sizeof(float) * NUMBER_OF_DETRENDING_BASIS_FUNCTIONS * DATA_T);
	cudaMemcpyToSymbol(c_xtxxt_Detrend, h_xtxxt_Detrend, sizeof(float) * NUMBER_OF_DETRENDING_BASIS_FUNCTIONS * DATA_T);
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
