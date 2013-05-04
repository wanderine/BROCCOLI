#include <cuda.h>
#include <helper_functions.h> // Helper functions (utilities, parsing, timing)
#include <helper_cuda.h>      // helper functions (cuda error checking and intialization)


#include "cufft.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>

//#include <cutil_inline.h>
//#include <shrUtils.h>
//#include <shrQATest.h>
#include "wabaacuda_lib.h"
#include "wabaacuda_kernel.cu"

#include "nifti1.h"
#include "nifti1_io.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <cstdlib>


/* 

to do

spm hrf
slice timing correction

*/


// public



WABAACUDA_LIB::WABAACUDA_LIB()
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

WABAACUDA_LIB::~WABAACUDA_LIB()
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


void WABAACUDA_LIB::SetDataType(int type)
{
	DATA_TYPE = type;
}

void WABAACUDA_LIB::SetFileType(int type)
{
	FILE_TYPE = type;
}

void WABAACUDA_LIB::SetNumberOfIterationsForMotionCompensation(int N)
{
	NUMBER_OF_ITERATIONS_FOR_MOTION_COMPENSATION = N;
}

void WABAACUDA_LIB::SetfMRIDataSliceLocationX(int location)
{
	X_SLICE_LOCATION_fMRI_DATA = location;
}
			
void WABAACUDA_LIB::SetfMRIDataSliceLocationY(int location)
{
	Y_SLICE_LOCATION_fMRI_DATA = location;
}
		
void WABAACUDA_LIB::SetfMRIDataSliceLocationZ(int location)
{
	Z_SLICE_LOCATION_fMRI_DATA = location;
}

void WABAACUDA_LIB::SetfMRIDataSliceTimepoint(int timepoint)
{
	TIMEPOINT_fMRI_DATA = timepoint;
}

int WABAACUDA_LIB::GetfMRIDataSliceLocationX()
{
	return X_SLICE_LOCATION_fMRI_DATA;
}
			
int WABAACUDA_LIB::GetfMRIDataSliceLocationY()
{
	return Y_SLICE_LOCATION_fMRI_DATA;
}
		
int WABAACUDA_LIB::GetfMRIDataSliceLocationZ()
{
	return Z_SLICE_LOCATION_fMRI_DATA;
}
			

void WABAACUDA_LIB::SetActivityThreshold(float threshold)
{
	ACTIVITY_THRESHOLD = threshold;
}

void WABAACUDA_LIB::SetThresholdStatus(bool status)
{
	THRESHOLD_ACTIVITY_MAP = status;
}

float CalculateMax(float *data, int N)
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

float CalculateMin(float *data, int N)
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




void WABAACUDA_LIB::CalculateSlicesActivityData()
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

void WABAACUDA_LIB::CalculateSlicesfMRIData()
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

void WABAACUDA_LIB::CalculateSlicesPreprocessedfMRIData()
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

double WABAACUDA_LIB::GetProcessingTimeSliceTimingCorrection()
{
	return processing_times[SLICE_TIMING_CORRECTION];
}

double WABAACUDA_LIB::GetProcessingTimeMotionCompensation()
{
	return processing_times[MOTION_COMPENSATION];
}

double WABAACUDA_LIB::GetProcessingTimeSmoothing()
{
	return processing_times[SMOOTHING];
}

double WABAACUDA_LIB::GetProcessingTimeDetrending()
{
	return processing_times[DETRENDING];
}

double WABAACUDA_LIB::GetProcessingTimeStatisticalAnalysis()
{
	return processing_times[STATISTICAL_ANALYSIS];
}

double WABAACUDA_LIB::GetProcessingTimePermutationTest()
{
	return processing_times[PERMUTATION_TEST];
}

double WABAACUDA_LIB::GetProcessingTimeCopy()
{
	return processing_times[COPY];
}

double WABAACUDA_LIB::GetProcessingTimeConvolution()
{
	return processing_times[CONVOLVE];
}

double WABAACUDA_LIB::GetProcessingTimePhaseDifferences()
{
	return processing_times[PHASEDC];
}

double WABAACUDA_LIB::GetProcessingTimePhaseGradients()
{
	return processing_times[PHASEG];
}

double WABAACUDA_LIB::WABAACUDA_LIB::GetProcessingTimeAH()
{
	return processing_times[AH2D];
}

double WABAACUDA_LIB::GetProcessingTimeEquationSystem()
{
	return processing_times[EQSYSTEM];
}

double WABAACUDA_LIB::GetProcessingTimeInterpolation()
{
	return processing_times[INTERPOLATION];
}


int WABAACUDA_LIB::GetWidth()
{
	return DATA_W;
}

int WABAACUDA_LIB::GetHeight()
{
	return DATA_H;
}

int WABAACUDA_LIB::GetDepth()
{
	return DATA_D;
}

int WABAACUDA_LIB::GetTimepoints()
{
	return DATA_T;
}

float WABAACUDA_LIB::GetXSize()
{
	return x_size;
}

float WABAACUDA_LIB::GetYSize()
{
	return y_size;
}

float WABAACUDA_LIB::GetZSize()
{
	return z_size;
}

float WABAACUDA_LIB::GetTR()
{
	return TR;
}

unsigned char* WABAACUDA_LIB::GetZSlicefMRIData()
{
	return z_slice_fMRI_data;
}

unsigned char* WABAACUDA_LIB::GetYSlicefMRIData()
{
	return y_slice_fMRI_data;
}

unsigned char* WABAACUDA_LIB::GetXSlicefMRIData()
{
	return x_slice_fMRI_data;
}

unsigned char* WABAACUDA_LIB::GetZSlicePreprocessedfMRIData()
{
	return z_slice_preprocessed_fMRI_data;
}

unsigned char* WABAACUDA_LIB::GetYSlicePreprocessedfMRIData()
{
	return y_slice_preprocessed_fMRI_data;
}

unsigned char* WABAACUDA_LIB::GetXSlicePreprocessedfMRIData()
{
	return x_slice_preprocessed_fMRI_data;
}

unsigned char* WABAACUDA_LIB::GetZSliceActivityData()
{
	return z_slice_activity_data;
}

unsigned char* WABAACUDA_LIB::GetYSliceActivityData()
{
	return y_slice_activity_data;
}

unsigned char* WABAACUDA_LIB::GetXSliceActivityData()
{
	return x_slice_activity_data;
}


double* WABAACUDA_LIB::GetMotionParametersX()
{
	return motion_parameters_x;
}

double* WABAACUDA_LIB::GetMotionParametersY()
{
	return motion_parameters_y;
}

double* WABAACUDA_LIB::GetMotionParametersZ()
{
	return motion_parameters_z;
}

double* WABAACUDA_LIB::GetPlotValuesX()
{
	return plot_values_x;
}


double* WABAACUDA_LIB::GetMotionCompensatedCurve()
{
	for (int t = 0; t < DATA_T; t++)
	{
		motion_compensated_curve[t] = (double)h_Motion_Compensated_fMRI_Volumes[X_SLICE_LOCATION_fMRI_DATA + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D];
	}

	return motion_compensated_curve;
}

double* WABAACUDA_LIB::GetSmoothedCurve()
{
	for (int t = 0; t < DATA_T; t++)
	{
		smoothed_curve[t] = (double)h_Smoothed_fMRI_Volumes_1[X_SLICE_LOCATION_fMRI_DATA + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D];
	}

	return smoothed_curve;
}

double* WABAACUDA_LIB::GetDetrendedCurve()
{
	for (int t = 0; t < DATA_T; t++)
	{
		detrended_curve[t] = (double)h_Detrended_fMRI_Volumes_1[X_SLICE_LOCATION_fMRI_DATA + Y_SLICE_LOCATION_fMRI_DATA * DATA_W + Z_SLICE_LOCATION_fMRI_DATA * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D];
	}

	return detrended_curve;
}

void WABAACUDA_LIB::SetSmoothingAmount(int amount)
{
	SMOOTHING_AMOUNT_MM = amount;
	ReadSmoothingFilters();
}

void WABAACUDA_LIB::SetSmoothingDimensionality(int dimensionality)
{
	SMOOTHING_DIMENSIONALITY = dimensionality;
	ReadSmoothingFilters();
}

void WABAACUDA_LIB::SetNumberOfBasisFunctionsDetrending(int N)
{
	NUMBER_OF_DETRENDING_BASIS_FUNCTIONS = N;
}

void WABAACUDA_LIB::SetfMRIDataFilename(std::string filename)
{
	filename_fMRI_data = filename;
}

std::string WABAACUDA_LIB::GetfMRIDataFilename()
{
	return filename_fMRI_data;
}

void WABAACUDA_LIB::SetAnalysisMethod(int method)
{
	ANALYSIS_METHOD = method;
	ReadSmoothingFilters();
}

void WABAACUDA_LIB::SetWriteStatus(bool status)
{
	WRITE_DATA = status;
}

void WABAACUDA_LIB::SetShowPreprocessedType(int value)
{
	PREPROCESSED = value;
}

void WABAACUDA_LIB::SetWidth(int w)
{
	DATA_W = w;
}
			
void WABAACUDA_LIB::SetHeight(int h)
{
	DATA_H = h;
}

void WABAACUDA_LIB::SetDepth(int d)
{
	DATA_D = d;
}

void WABAACUDA_LIB::SetTimepoints(int t)
{
	DATA_T = t;
}

void WABAACUDA_LIB::SetXSize(float value)
{
	x_size = value;
}

void WABAACUDA_LIB::SetYSize(float value)
{
	y_size = value;
}

void WABAACUDA_LIB::SetZSize(float value)
{
	z_size = value;
}

void WABAACUDA_LIB::SetTR(float value)
{
	TR = value;
}

void WABAACUDA_LIB::SetSignificanceThreshold(float value)
{
	significance_threshold = value;
}

void WABAACUDA_LIB::SetNumberOfPermutations(int value)
{
	NUMBER_OF_PERMUTATIONS = value;
}

float WABAACUDA_LIB::GetPermutationThreshold()
{
	return permutation_test_threshold;
}

void WABAACUDA_LIB::PerformPreprocessingAndCalculateActivityMap()
{
    //PerformSliceTimingCorrection();
	PerformMotionCompensation();
	PerformSmoothing();	
	PerformDetrending();
	CalculateActivityMap();

	CalculateSlicesPreprocessedfMRIData();
}

void WABAACUDA_LIB::SmoothingSingleVolume(float* d_Smoothed_Volume, float* d_Volume, float* d_Certainty, float* d_Smoothed_Alpha_Certainty, int DATA_W, int DATA_H, int DATA_D, int FILTER_SIZE)
{
	float  *d_Filter_Response_Rows, *d_Filter_Response_Columns;

	int DATA_SIZE_VOLUME  = DATA_W * DATA_H * DATA_D * sizeof(float);

	cudaMalloc((void **)&d_Filter_Response_Rows, DATA_SIZE_VOLUME);	
	cudaMalloc((void **)&d_Filter_Response_Columns, DATA_SIZE_VOLUME);	

	int threadsInX, threadsInY, threadsInZ;
	int blocksInX, blocksInY, blocksInZ;
	int xBlockDifference, yBlockDifference, zBlockDifference;
	dim3 dimGrid, dimBlock;

	// Convolve rows

	threadsInX = 32;
	threadsInY = 8;
	threadsInZ = 2;

	blocksInX = (DATA_W+threadsInX-1)/threadsInX;
	blocksInY = (DATA_H+threadsInY-1)/threadsInY;
	blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
	dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
	dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX, threadsInY, threadsInZ * 4);

	convolutionRows<<<dimGrid, dimBlock>>>(d_Filter_Response_Rows, d_Volume, d_Brain_Voxels, 0, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);
	
	// Convolve columns

	threadsInX = 32;
	threadsInY = 8;
	threadsInZ = 2;

	blocksInX = (DATA_W+threadsInX-1)/(threadsInX / 32 * 24);
	blocksInY = (DATA_H+threadsInY-1)/threadsInY / 2;
	blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
	dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
	dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX / 32 * 24, threadsInY * 2, threadsInZ * 4);

	convolutionColumns<<<dimGrid, dimBlock>>>(d_Filter_Response_Columns, d_Filter_Response_Rows, 0, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);
	
	// Convolve rods
	
	threadsInX = 32;
	threadsInY = 2;
	threadsInZ = 8;

	blocksInX = (DATA_W+threadsInX-1)/threadsInX;
	blocksInY = (DATA_H+threadsInY-1)/threadsInY / 4;
	blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
	dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
	dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX, threadsInY * 4, threadsInZ);

	convolutionRods<<<dimGrid, dimBlock>>>(d_Smoothed_Volume, d_Filter_Response_Columns, d_Smoothed_Alpha_Certainty, 0, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);
	
	// Free all the allocated memory on the graphics card
	cudaFree(d_Filter_Response_Rows);
	cudaFree(d_Filter_Response_Columns);
}

void WABAACUDA_LIB::GeneratePermutationMatrix()
{
    for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
    {
        for (int i = 0; i < DATA_T; i++)
        {
            h_Permutation_Matrix[i + p * DATA_T] = (unsigned short int)i;
        }

        for (int i = 0; i < DATA_T; i++)
        {
            int j = rand() % (DATA_T - i) + i;
            unsigned short int t = h_Permutation_Matrix[j + p * DATA_T];
            h_Permutation_Matrix[j + p * DATA_T] = h_Permutation_Matrix[i + p * DATA_T];
            h_Permutation_Matrix[i + p * DATA_T] = t;
        }
    }
}

void WABAACUDA_LIB::PerformDetrendingPriorPermutation()
{	
	int threadsInX = 32;
	int threadsInY = 8;
	int threadsInZ = 1;

	int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
	int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
	int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
	dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
	dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	// Calculate how many time multiples there are
	int timeMultiples = DATA_T / threadsInY;
	int timeRest = DATA_T - timeMultiples * threadsInY;

	// Do the detrending

	DetrendCubic<<<dimGrid, dimBlock>>>(d_Detrended_fMRI_Volumes_1, d_Motion_Compensated_fMRI_Volumes, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
}

void WABAACUDA_LIB::CreateBOLDRegressedVolumes()
{	
	int threadsInX = 32;
	int threadsInY = 8;
	int threadsInZ = 1;

	int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
	int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
	int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
	dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
	dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	// Calculate how many time multiples there are
	int timeMultiples = DATA_T / threadsInY;
	int timeRest = DATA_T - timeMultiples * threadsInY;

	Regress_BOLD<<<dimGrid, dimBlock>>>(d_BOLD_Regressed_fMRI_Volumes, d_Detrended_fMRI_Volumes_1, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/(float)blocksInY, timeMultiples, timeRest);
}


void WABAACUDA_LIB::PerformWhiteningPriorPermutation()
{
	int threadsInX = 32;
	int threadsInY = 8;
	int threadsInZ = 1;

	int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
	int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
	int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
	dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
	dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	EstimateAR4BrainVoxels<<<dimGrid, dimBlock>>>(d_Alphas_1, d_Alphas_2, d_Alphas_3, d_Alphas_4, d_BOLD_Regressed_fMRI_Volumes, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/(float)blocksInY);

	SmoothingSingleVolume(d_Smoothed_Alphas_1, d_Alphas_1, d_Brain_Voxels, d_Smoothed_Certainty, DATA_W, DATA_H, DATA_D, SMOOTHING_FILTER_SIZE);
	SmoothingSingleVolume(d_Smoothed_Alphas_2, d_Alphas_2, d_Brain_Voxels, d_Smoothed_Certainty, DATA_W, DATA_H, DATA_D, SMOOTHING_FILTER_SIZE);
	SmoothingSingleVolume(d_Smoothed_Alphas_3, d_Alphas_3, d_Brain_Voxels, d_Smoothed_Certainty, DATA_W, DATA_H, DATA_D, SMOOTHING_FILTER_SIZE);
	SmoothingSingleVolume(d_Smoothed_Alphas_4, d_Alphas_4, d_Brain_Voxels, d_Smoothed_Certainty, DATA_W, DATA_H, DATA_D, SMOOTHING_FILTER_SIZE);
	
	ApplyWhiteningAR4<<<dimGrid, dimBlock>>>(d_Whitened_fMRI_Volumes, d_BOLD_Regressed_fMRI_Volumes, d_Smoothed_Alphas_1, d_Smoothed_Alphas_2, d_Smoothed_Alphas_3, d_Smoothed_Alphas_4, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/(float)blocksInY);
}

void WABAACUDA_LIB::GeneratePermutedfMRIVolumes()
{
	int threadsInX = 32;
    int threadsInY = 8;
    int threadsInZ = 1;

    int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
    int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
    int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
    dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	ResetVolumes<<<dimGrid, dimBlock>>>(d_Permuted_fMRI_Volumes, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/(float)blocksInY);		

	GeneratePermutedfMRIVolumesAR4<<<dimGrid, dimBlock>>>(d_Permuted_fMRI_Volumes, d_Smoothed_Alphas_1, d_Smoothed_Alphas_2, d_Smoothed_Alphas_3, d_Smoothed_Alphas_4, d_Whitened_fMRI_Volumes, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY));
}

void WABAACUDA_LIB::SetupParametersPermutation()
{	
	for (int i = 0; i < NUMBER_OF_HOST_POINTERS; i++)
	{
		void* pointer = host_pointers_permutation[i];
		if (pointer != NULL)
		{
			free(pointer);
			host_pointers_permutation[i] = NULL;
		}
	}

	for (int i = 0; i < NUMBER_OF_DEVICE_POINTERS; i++)
	{
		float* pointer = device_pointers_permutation[i];
		if (pointer != NULL)
		{
			cudaFree(pointer);
			device_pointers_permutation[i] = NULL;
		}
	}

	int DATA_SIZE_fMRI_VOLUME = sizeof(float) * DATA_W * DATA_H * DATA_D;
	int DATA_SIZE_fMRI_VOLUMES = sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T;

	h_Permutation_Matrix = (unsigned short int*)malloc(sizeof(unsigned short int) * DATA_T * NUMBER_OF_PERMUTATIONS);
    h_Maximum_Test_Values = (float*)malloc(sizeof(float) * NUMBER_OF_PERMUTATIONS);

	host_pointers_permutation[PERMUTATION_MATRIX] = h_Permutation_Matrix;
	host_pointers_permutation[MAXIMUM_TEST_VALUES] = h_Maximum_Test_Values;
    
	cudaMalloc((void **)&d_Alphas_1, DATA_SIZE_fMRI_VOLUME);
	cudaMalloc((void **)&d_Alphas_2, DATA_SIZE_fMRI_VOLUME);
	cudaMalloc((void **)&d_Alphas_3, DATA_SIZE_fMRI_VOLUME);
	cudaMalloc((void **)&d_Alphas_4, DATA_SIZE_fMRI_VOLUME);

	cudaMalloc((void **)&d_Smoothed_Alphas_1, DATA_SIZE_fMRI_VOLUME);
	cudaMalloc((void **)&d_Smoothed_Alphas_2, DATA_SIZE_fMRI_VOLUME);
	cudaMalloc((void **)&d_Smoothed_Alphas_3, DATA_SIZE_fMRI_VOLUME);
	cudaMalloc((void **)&d_Smoothed_Alphas_4, DATA_SIZE_fMRI_VOLUME);

	cudaMalloc((void **)&d_BOLD_Regressed_fMRI_Volumes, DATA_SIZE_fMRI_VOLUMES);
	cudaMalloc((void **)&d_Whitened_fMRI_Volumes, DATA_SIZE_fMRI_VOLUMES);
	cudaMalloc((void **)&d_Permuted_fMRI_Volumes, DATA_SIZE_fMRI_VOLUMES);
	
	device_pointers_permutation[ALPHAS1] = d_Alphas_1;
	device_pointers_permutation[ALPHAS2] = d_Alphas_2;
	device_pointers_permutation[ALPHAS3] = d_Alphas_3;
	device_pointers_permutation[ALPHAS4] = d_Alphas_4;

	device_pointers_permutation[SMOOTHED_ALPHAS1] = d_Smoothed_Alphas_1;
	device_pointers_permutation[SMOOTHED_ALPHAS2] = d_Smoothed_Alphas_2;
	device_pointers_permutation[SMOOTHED_ALPHAS3] = d_Smoothed_Alphas_3;
	device_pointers_permutation[SMOOTHED_ALPHAS4] = d_Smoothed_Alphas_4;

	device_pointers_permutation[BOLD_REGRESSED_VOLUMES] = d_BOLD_Regressed_fMRI_Volumes;
	device_pointers_permutation[WHITENED_VOLUMES] = d_Whitened_fMRI_Volumes;
	device_pointers_permutation[PERMUTED_VOLUMES] = d_Permuted_fMRI_Volumes;
}

float WABAACUDA_LIB::FindMaxTestvaluePermutationOld()
{	
	float* h_Slice_Max = (float*)malloc(sizeof(float) * DATA_D);
	float* d_Slice_Max; 
	cudaMalloc((void **)&d_Slice_Max, sizeof(float) * DATA_D);

    int threadsInX = 32;
    int threadsInY = 16;
    int threadsInZ = 1;

    int blocksInX = 1;
    int blocksInY = 1;
    int blocksInZ = DATA_D;
    dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

    // Find the maximum test value in each slice
    FindSliceMax<<<dimGrid, dimBlock>>>(d_Slice_Max, d_Activity_Volume, DATA_W, DATA_H, DATA_D, blocksInY, 1.0f/(float)blocksInY);
    cudaMemcpy(h_Slice_Max, d_Slice_Max, DATA_D * sizeof(float), cudaMemcpyDeviceToHost);

    float max = 0.0f;
    for (int z = 0; z < DATA_D; z++)
    {
        if (h_Slice_Max[z] > max)
		{
	    	max = h_Slice_Max[z];
		}
    }

	free(h_Slice_Max);
	cudaFree(d_Slice_Max);

	return max;	
}

float WABAACUDA_LIB::FindMaxTestvaluePermutation()
{
	cudaMemcpy(h_Activity_Volume, d_Activity_Volume, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyDeviceToHost);
	thrust::host_vector<float> h_vec(h_Activity_Volume, &h_Activity_Volume[DATA_W * DATA_H * DATA_D]); 
	
	thrust::device_vector<float> d_vec = h_vec;
	//thrust::device_vector<float> d_vec(d_Activity_Volume, &d_Activity_Volume[DATA_W * DATA_H * DATA_D]);

    return thrust::reduce(d_vec.begin(), d_vec.end(), -1000.0f, thrust::maximum<float>());
}

int WABAACUDA_LIB::GetNumberOfSignificantlyActiveVoxels()
{
	return NUMBER_OF_SIGNIFICANTLY_ACTIVE_VOXELS;
}

void WABAACUDA_LIB::CalculatePermutationTestThreshold()
{
	SetupParametersPermutation();
	GeneratePermutationMatrix();

    // Make the timeseries white prior to the random permutations
    PerformDetrendingPriorPermutation();
    CreateBOLDRegressedVolumes();
    PerformWhiteningPriorPermutation();
   
	sdkCreateTimer(&hTimer);

	cudaThreadSynchronize();
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

    // Loop over all the permutations, save the maximum test value from each permutation
    for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
    {
         // Copy a new permutation vector to constant memory
        cudaMemcpyToSymbol(c_Permutation_Vector, &h_Permutation_Matrix[p * DATA_T], DATA_T * sizeof(uint16));

        GeneratePermutedfMRIVolumes();
        PerformSmoothingPermutation();
        PerformDetrendingPermutation();
        //PerformWhiteningPermutation();
        CalculateActivityMapPermutation();
		h_Maximum_Test_Values[p] = FindMaxTestvaluePermutation();  
    }

	cudaThreadSynchronize();
	sdkStopTimer(&hTimer);
	processing_times[PERMUTATION_TEST] = sdkGetTimerValue(&hTimer);
	sdkDeleteTimer(&hTimer);

    // Sort the maximum test values
	std::vector<float> h_Maximum_Test_Values_Vector(h_Maximum_Test_Values, &h_Maximum_Test_Values[NUMBER_OF_PERMUTATIONS]);              
	std::sort(h_Maximum_Test_Values_Vector.begin(), h_Maximum_Test_Values_Vector.end());
	
	// Find the threshold for the significance level
	int location = floor((1.0f - significance_threshold) * float(NUMBER_OF_PERMUTATIONS));
	permutation_test_threshold = h_Maximum_Test_Values_Vector.at(location);

	NUMBER_OF_SIGNIFICANTLY_ACTIVE_VOXELS = 0;
	for (int i = 0; i < DATA_W * DATA_H * DATA_D; i++)
	{
		if (h_Activity_Volume[i] >= permutation_test_threshold)
		{
			NUMBER_OF_SIGNIFICANTLY_ACTIVE_VOXELS++;
		}
	}
}


std::string WABAACUDA_LIB::PrintGPUInfo()
{
	int deviceCount = 0;
	int driverVersion = 0;
	int runtimeVersion = 0;
	std::stringstream s;

    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	for (int device = 0; device < deviceCount; device++)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		
		s << std::endl << "Printing stats for device " << device << " with name " << deviceProp.name << ", compute capability " << deviceProp.major << "." << deviceProp.minor << std::endl << std::endl;
		s << "The GPU has " << deviceProp.multiProcessorCount << " multiprocessors with " << _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) << " CUDA cores each, giving a total of " << _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount << " processor cores. ";
		s << "The clockrate of the processor cores is " << deviceProp.clockRate * 1e-6f << " GHz." << std::endl;
		s << "The GPU has " << (float)deviceProp.totalGlobalMem/1048576.0f << " MB of global memory." << std::endl;
		s << "Each multiprocessor has " << (float)deviceProp.totalConstMem/1024.0f << " KB of constant memory, " << (float)deviceProp.sharedMemPerBlock/1024.0f << " KB of shared memory and " << deviceProp.regsPerBlock << " registers." << std::endl;
		s << std::endl;
	}
	return s.str();
}


// private

// Read functions

void WABAACUDA_LIB::ReadRealDataInt32(int* data, std::string filename, int N)
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

void WABAACUDA_LIB::ReadRealDataInt16(short int* data, std::string filename, int N)
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

void WABAACUDA_LIB::ReadRealDataUint32(unsigned int* data, std::string filename, int N)
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

void WABAACUDA_LIB::ReadRealDataUint16(unsigned short int* data, std::string filename, int N)
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


void WABAACUDA_LIB::ReadRealDataFloat(float* data, std::string filename, int N)
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

void WABAACUDA_LIB::ReadRealDataDouble(double* data, std::string filename, int N)
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

void WABAACUDA_LIB::WriteRealDataUint16(unsigned short int* data, std::string filename, int N)
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

void WABAACUDA_LIB::WriteRealDataFloat(float* data, std::string filename, int N)
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

void WABAACUDA_LIB::WriteRealDataDouble(double* data, std::string filename, int N)
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

void WABAACUDA_LIB::ReadComplexData(Complex* data, std::string real_filename, std::string imag_filename, int N)
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

void WABAACUDA_LIB::WriteComplexData(Complex* data, std::string real_filename, std::string imag_filename, int N)
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

void WABAACUDA_LIB::SetupParametersReadData()
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

	MOTION_COMPENSATED = false;

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



void WABAACUDA_LIB::ReadfMRIDataRAW()
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

void WABAACUDA_LIB::ReadfMRIDataNIFTI()
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



void WABAACUDA_LIB::ReadNIFTIHeader()
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



void WABAACUDA_LIB::ReadMotionCompensationFilters()
{
	// Read the quadrature filters from file
	ReadComplexData(h_Quadrature_Filter_1, filename_real_quadrature_filter_1, filename_imag_quadrature_filter_1, MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE);
	ReadComplexData(h_Quadrature_Filter_2, filename_real_quadrature_filter_2, filename_imag_quadrature_filter_2, MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE);
	ReadComplexData(h_Quadrature_Filter_3, filename_real_quadrature_filter_3, filename_imag_quadrature_filter_3, MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE);

	// Copy the quadrature filters to constant memory
	cudaMemcpyToSymbol(c_Quadrature_Filter_1, h_Quadrature_Filter_1, sizeof(Complex) * MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE);
	cudaMemcpyToSymbol(c_Quadrature_Filter_2, h_Quadrature_Filter_2, sizeof(Complex) * MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE);
	cudaMemcpyToSymbol(c_Quadrature_Filter_3, h_Quadrature_Filter_3, sizeof(Complex) * MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE * MOTION_COMPENSATION_FILTER_SIZE);
}

void WABAACUDA_LIB::ReadSmoothingFilters()
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

	// Copy smoothing filters to constant memory
	cudaMemcpyToSymbol(c_Smoothing_Kernel, h_GLM_Filter, sizeof(float) * SMOOTHING_FILTER_SIZE);
	cudaMemcpyToSymbol(c_CCA_2D_Filters, h_CCA_2D_Filters, sizeof(float4) * SMOOTHING_FILTER_SIZE * SMOOTHING_FILTER_SIZE);
	cudaMemcpyToSymbol(c_CCA_3D_Filters, h_CCA_3D_Filters, sizeof(float2) * SMOOTHING_FILTER_SIZE);
}

// Help functions

void WABAACUDA_LIB::Convert4FloatToFloat4(float4* floats, float* float_1, float* float_2, float* float_3, float* float_4, int N)
{
	for (int i = 0; i < N; i++)
	{
		floats[i].x = float_1[i];
		floats[i].y = float_2[i];
		floats[i].z = float_3[i];
		floats[i].w = float_4[i];
	}
}

void WABAACUDA_LIB::Convert2FloatToFloat2(float2* floats, float* float_1, float* float_2, int N)
{
	for (int i = 0; i < N; i++)
	{
		floats[i].x = float_1[i];
		floats[i].y = float_2[i];
	}
}

void WABAACUDA_LIB::ConvertRealToComplex(Complex* complex_data, float* real_data, int N)
{
	for (int i = 0; i < N; i++)
	{
		complex_data[i].x = real_data[i];
		complex_data[i].y = 0.0f;
	}
}

void WABAACUDA_LIB::ExtractRealData(float* real_data, Complex* complex_data, int N)
{
	for (int i = 0; i < N; i++)
	{
		real_data[i] = complex_data[i].x;
	}
}

void WABAACUDA_LIB::Calculate_Block_Differences2D(int& xBlockDifference, int& yBlockDifference, int DATA_W, int DATA_H, int threadsInX, int threadsInY)
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

void WABAACUDA_LIB::Calculate_Block_Differences3D(int& xBlockDifference, int& yBlockDifference, int& zBlockDifference, int DATA_W, int DATA_H, int DATA_D, int threadsInX, int threadsInY, int threadsInZ)
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

void WABAACUDA_LIB::Invert_Matrix(float* inverse_matrix, float* matrix, int N)
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

void WABAACUDA_LIB::Calculate_Square_Root_of_Matrix(float* sqrt_matrix, float* matrix, int N)
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

void WABAACUDA_LIB::SolveEquationSystem(float* h_A_matrix, float* h_inverse_A_matrix, float* h_h_vector, float* h_Parameter_Vector, int N)
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

void WABAACUDA_LIB::SetupDetrendingBasisFunctions()
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

void WABAACUDA_LIB::SegmentBrainData()
{
	int threadsInX = 32;
	int threadsInY = 8;
	int threadsInZ = 1;

	int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
	int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
	int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
	dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
	dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	ThresholdfMRIData<<<dimGrid, dimBlock>>>(d_Brain_Voxels, d_fMRI_Volumes, SEGMENTATION_THRESHOLD, DATA_W, DATA_H, DATA_D, blocksInY, 1.0f/(float)blocksInY);
}

float loggamma(int value)
{
	int product = 1;
	for (int i = 1; i < value; i++)
	{
		product *= i;
	}
	return log((double)product);
}

float Gpdf(double value, double shape, double scale)
{
	//return pow(value, shape - scale) * exp(-value / scale) / (pow(scale,shape) * gamma((int)shape));

	return (exp( (shape - 1.0) * log(value) + shape * log(scale) - scale * value - loggamma(shape) ));
}



void WABAACUDA_LIB::CreateHRF()
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


void WABAACUDA_LIB::ConvolveWithHRF(float* temp_GLM)
{
	// Calculate derivative of hrf
	float* hrf_diff = (float*)malloc(sizeof(float) * hrf_length);
	hrf_diff[0] = 0.0f;
	for (int i = 1; i < hrf_length; i++)
	{
		hrf_diff[i] = hrf[i] - hrf[i-1];
	}

	// Do the convolution
	for (int t = 0; t < DATA_T; t++)
	{
		float sum1 = 0.0f;
		float sum2 = 0.0f;
		int t_offset = -(hrf_length-1)/2;
		for (int f = hrf_length - 1; f >= 0; f--)
		{
			if (((t + t_offset) > 0) && ((t + t_offset) < DATA_T))
			{
				sum1 += temp_GLM[t + t_offset] * hrf[f];
				sum2 += temp_GLM[t + t_offset] * hrf_diff[f];
			}
			t_offset++;
		}
		h_X_GLM[t + 0 * DATA_T] = sum1;
		h_X_GLM[t + 1 * DATA_T] = sum2;
	}

	free(hrf_diff);
}

void WABAACUDA_LIB::SetupStatisticalAnalysisBasisFunctions()
{
	/* Matlab equivalent
	% GLM
	X_GLM = zeros(st,2);
	X_GLM(1:10,1) = 1; X_GLM(1:10,2) = 0;
	X_GLM(11:20,1) = 0; X_GLM(11:20,2) = 1;
	X_GLM(21:30,1) = 1; X_GLM(21:30,2) = 0;
	X_GLM(31:40,1) = 0; X_GLM(31:40,2) = 1;
	X_GLM(41:50,1) = 1; X_GLM(41:50,2) = 0;
	X_GLM(51:60,1) = 0; X_GLM(51:60,2) = 1;
	X_GLM(61:70,1) = 1; X_GLM(61:70,2) = 0;
	X_GLM(71:end,1) = 0; X_GLM(71:end,2) = 1;

	hrf = spm_hrf(2);
	dhrf = diff(hrf);

	GLM1 = conv(X_GLM(:,1),hrf);
	GLM1 = GLM1(1:st);

	GLM2 = conv(X_GLM(:,1),dhrf);
	GLM2 = GLM2(1:st);

	X_GLM(:,1) = GLM1 - mean(GLM1);
	X_GLM(:,2) = GLM2 - mean(GLM2);

	% Normalize
	X_GLM(:,1) = X_GLM(:,1)/norm(X_GLM(:,1));
	X_GLM(:,2) = X_GLM(:,2)/norm(X_GLM(:,2));

	% Orthogonalize
	X_GLM(:,2) = X_GLM(:,2) - (X_GLM(:,1)'*X_GLM(:,2))*X_GLM(:,1);
	
	xtxxt_GLM = inv(X_GLM'*X_GLM)*X_GLM';

	c = [1 ; 0];

	ctxtx_GLM = c'*inv(X_GLM'*X_GLM)*c;
	*/

	float* temp_X_GLM = (float*)malloc(sizeof(float) * DATA_T);

	bool ACTIVITY = 0;
	bool REST = 1;
	bool STATE = ACTIVITY;

	// Calculate X_GLM
	int period_t = 1;
	for (int t = 0; t < DATA_T; t++)
	{
		if (STATE == ACTIVITY)
		{
			temp_X_GLM[t] = 1.0f;
		}
		else if (STATE == REST)
		{
			temp_X_GLM[t] = 0.0f;
		}

		period_t++;

		if (period_t > PERIOD_TIME/2)
		{
			STATE = !STATE;
			period_t = 1;
		}
	}

	// Convolve with HRF...
	CreateHRF();

	ConvolveWithHRF(temp_X_GLM);

	// Remove mean
	float2 means; means.x = 0.0f; means.y = 0.0f;
	for (int t = 0; t < DATA_T; t++)
	{
		means.x += h_X_GLM[t + 0 * DATA_T];
		means.y += h_X_GLM[t + 1 * DATA_T];
	}
	means.x /= (float)DATA_T;
	means.y /= (float)DATA_T;
	for (int t = 0; t < DATA_T; t++)
	{
		h_X_GLM[t + 0 * DATA_T] -= means.x;
		h_X_GLM[t + 1 * DATA_T] -= means.y;
	}

	// Normalize
	float2 norms; norms.x = 0.0f; norms.y = 0.0f;
	for (int t = 0; t < DATA_T; t++)
	{
		norms.x += h_X_GLM[t + 0 * DATA_T] * h_X_GLM[t + 0 * DATA_T];
		norms.y += h_X_GLM[t + 1 * DATA_T] * h_X_GLM[t + 1 * DATA_T];
	}
	norms.x = sqrt(norms.x);
	norms.y = sqrt(norms.y);

	for (int t = 0; t < DATA_T; t++)
	{
		h_X_GLM[t + 0 * DATA_T] /= norms.x;
		h_X_GLM[t + 1 * DATA_T] /= norms.y;
	}

	// Orthogonalize
	float scalar_product = 0.0f;
	for (int t = 0; t < DATA_T; t++)
	{
		scalar_product += h_X_GLM[t + 0 * DATA_T] * h_X_GLM[t + 1 * DATA_T];
	}

	for (int t = 0; t < DATA_T; t++)
	{
		h_X_GLM[t + 1 * DATA_T] -= scalar_product * h_X_GLM[t + 0 * DATA_T];
	}

	WriteRealDataFloat(h_X_GLM, "GLM_1.raw", DATA_T * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS);

	// Calculate X_GLM'*X_GLM
	float* xtx = (float*)malloc(sizeof(float) * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS);
	float* inv_xtx = (float*)malloc(sizeof(float) * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS);
	float* sqrt_inv_xtx = (float*)malloc(sizeof(float) * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS);

	for (int i = 0; i < NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS; i++)
	{
		for (int j = 0; j < NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS; j++)
		{
			xtx[i + j * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS] = 0.0f;
			for (int t = 0; t < DATA_T; t++)
			{
				xtx[i + j * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS] += h_X_GLM[t + i * DATA_T] * h_X_GLM[t + j * DATA_T];
			}
		}
	}

	// Calculate inverse of X_GLM'*X_GLM
	Invert_Matrix(inv_xtx, xtx, NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS);

	// Calculate inv(X_GLM'*X_GLM)*X_GLM'
	for (int t = 0; t < DATA_T; t++)
	{
		h_xtxxt_GLM[t + 0 * DATA_T] = inv_xtx[0] * h_X_GLM[t + 0 * DATA_T] + inv_xtx[1] * h_X_GLM[t + 1 * DATA_T];
		h_xtxxt_GLM[t + 1 * DATA_T] = inv_xtx[2] * h_X_GLM[t + 0 * DATA_T] + inv_xtx[3] * h_X_GLM[t + 1 * DATA_T];
	}

	cudaMemcpyToSymbol(c_X_GLM, h_X_GLM, sizeof(float) * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS * DATA_T);
	cudaMemcpyToSymbol(c_xtxxt_GLM, h_xtxxt_GLM, sizeof(float) * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS * DATA_T);

	h_Contrast_Vector[0] = 1.0f;
	h_Contrast_Vector[1] = 0.0f;

	float* xtxc = (float*)malloc(sizeof(float) * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS);
	xtxc[0] = inv_xtx[0] * h_Contrast_Vector[0] + inv_xtx[1] * h_Contrast_Vector[1];
	xtxc[1] = inv_xtx[2] * h_Contrast_Vector[0] + inv_xtx[3] * h_Contrast_Vector[1];
	h_ctxtxc = xtxc[0] * h_Contrast_Vector[0] + xtxc[1] * h_Contrast_Vector[1];

	cudaMemcpyToSymbol(c_Y, h_X_GLM, sizeof(float) * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS * DATA_T);
	
	xtx[0] /= ((float)DATA_T - 1);
	xtx[1] /= ((float)DATA_T - 1);
	xtx[2] /= ((float)DATA_T - 1);
	xtx[3] /= ((float)DATA_T - 1);

	Invert_Matrix(inv_xtx, xtx, NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS);

	// Calculate matrix square root of inverse of xtx
	Calculate_Square_Root_of_Matrix(sqrt_inv_xtx, inv_xtx, NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS);

	cudaMemcpyToSymbol(c_Cyy, xtx, sizeof(float) * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS);
	cudaMemcpyToSymbol(c_sqrt_inv_Cyy, sqrt_inv_xtx, sizeof(float) * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS * NUMBER_OF_STATISTICAL_BASIS_FUNCTIONS);

	free(temp_X_GLM);
	free(xtx);
	free(xtxc);
	free(inv_xtx);
	free(sqrt_inv_xtx);
}

// Preprocessing

void WABAACUDA_LIB::PerformSliceTimingCorrection()
{
	int NUMBER_OF_VOXELS = DATA_W * DATA_H * DATA_D;

	// Plan for batch 1D FFT's
    cufftHandle           FFTplan;
	cufftPlan1d(&FFTplan, DATA_T_PADDED, CUFFT_C2C, NUMBER_OF_VOXELS);

	int threadsInX = 32;
	int threadsInY = 4;
	int threadsInZ = 2;

	int blocksInX = (DATA_T_PADDED+threadsInX-1)/threadsInX;
	int blocksInY = (DATA_W+threadsInY-1)/threadsInY;
	int blocksInZ = (DATA_H+threadsInZ-1)/threadsInZ;
	dim3 dimGrid   = dim3(blocksInX, blocksInY*blocksInZ);
	dim3 dimBlock  = dim3(threadsInX, threadsInY, threadsInZ);

	int DATA_SIZE_COMPLEX_VOLUMES = DATA_W * DATA_H * DATA_D * DATA_T_PADDED * sizeof(Complex);
	Complex* d_Slice_Timing_Corrected_fMRI_Volumes_Padded;
	cudaMalloc((void **)&d_Slice_Timing_Corrected_fMRI_Volumes_Padded, DATA_SIZE_COMPLEX_VOLUMES);

	// Do all the forward 1D FFT's as a batch
	cufftExecC2C(FFTplan, (cufftComplex*)d_fMRI_Volumes, (cufftComplex *)d_fMRI_Volumes, CUFFT_FORWARD);

	// Launch the kernel for all the multiplications that do the phase shift
	for (int z = 0; z < DATA_D; z++)
	{
		//SliceTimingCorrection<<<dimGrid, dimBlock>>>(d_Slice_Timing_Corrected_fMRI_Volumes_Padded, d_fMRI_Volumes_Complex, d_Shifters, z, DATA_W, DATA_H, DATA_D, DATA_T_PADDED, blocksInY, 1.0f/((float)blocksInY), 1.0f/(float)DATA_T_PADDED);		
	}

	// Do all the inverse 1D FFT's as a batch		
	cufftExecC2C(FFTplan, (cufftComplex*)d_Slice_Timing_Corrected_fMRI_Volumes_Padded, (cufftComplex *)d_Slice_Timing_Corrected_fMRI_Volumes_Padded, CUFFT_INVERSE);
	
	
	threadsInX = 32;
	threadsInY = 4;
	threadsInZ = 2;

	blocksInX = (DATA_T+threadsInX-1)/threadsInX;
	blocksInY = (DATA_W+threadsInY-1)/threadsInY;
	blocksInZ = (DATA_H+threadsInZ-1)/threadsInZ;
	dimGrid   = dim3(blocksInX, blocksInY*blocksInZ);
	dimBlock  = dim3(threadsInX, threadsInY, threadsInZ);

	// Extract real data
	for (int z = 0; z < DATA_D; z++)
	{
		ExtractRealNonPaddedVolumes<<<dimGrid, dimBlock>>>(d_Slice_Timing_Corrected_fMRI_Volumes, d_Slice_Timing_Corrected_fMRI_Volumes_Padded, z, DATA_W, DATA_H, DATA_D, DATA_T, DATA_T_PADDED, blocksInY, 1.0f/((float)blocksInY));
	}

	cufftDestroy(FFTplan);
	cudaFree(d_Slice_Timing_Corrected_fMRI_Volumes_Padded);
}

void Set_float12_to_zeros(float12& x)
{
	x.a = 0.0f;
	x.b = 0.0f;
	x.c = 0.0f;
	x.d = 0.0f;
	x.e = 0.0f;
	x.f = 0.0f;
	x.g = 0.0f;
	x.h = 0.0f;
	x.i = 0.0f;
	x.j = 0.0f;
	x.k = 0.0f;
	x.l = 0.0f;
}

void Set_float30_to_zeros(float30& x)
{
	x.a = 0.0f;
	x.b = 0.0f;
	x.c = 0.0f;
	x.d = 0.0f;
	x.e = 0.0f;
	x.f = 0.0f;
	x.g = 0.0f;
	x.h = 0.0f;
	x.i = 0.0f;
	x.j = 0.0f;
	x.k = 0.0f;
	x.l = 0.0f;
	x.m = 0.0f;
	x.n = 0.0f;
	x.o = 0.0f;
	x.p = 0.0f;
	x.q = 0.0f;
	x.r = 0.0f;
	x.s = 0.0f;
	x.t = 0.0f;
	x.u = 0.0f;
	x.v = 0.0f;
	x.w = 0.0f;
	x.x = 0.0f;
	x.y = 0.0f;
	x.z = 0.0f;
	x.aa = 0.0f;
	x.bb = 0.0f;
	x.cc = 0.0f;
	x.dd = 0.0f;
}

void WABAACUDA_LIB::PerformMotionCompensation()
{
	if (!MOTION_COMPENSATED)
	{
		float                 *d_Reference_Volume, *d_Compensated_Volume;
		float				  *h_A_matrix, *h_inverse_A_matrix, *h_h_vector;
		float				  *d_A_matrix, *d_h_vector, *d_A_matrix_2D_values, *d_A_matrix_1D_values, *d_h_vector_2D_values, *d_h_vector_1D_values; 
		float30				  *d_A_matrix_3D_values, *h_A_matrix_3D_values;
		float12				  *d_h_vector_3D_values, *h_h_vector_3D_values;
		float 				  *d_Phase_Differences, *d_Phase_Gradients, *d_Certainties;
		cudaArray			  *d_Modified_Volume; 
		Complex               *d_q11, *d_q12, *d_q13, *d_q21, *d_q22, *d_q23;	
		cudaExtent            VOLUME_SIZE;
    
		h_Registration_Parameters = (float*)malloc(sizeof(float) * NUMBER_OF_PARAMETERS * DATA_T);
		host_pointers[REGISTRATION_PARAMETERS] = h_Registration_Parameters;
	
		float h_Parameter_Vector[12], h_Parameter_Vector_Total[12];

		int DATA_SIZE_COMPLEX = DATA_W * DATA_H * DATA_D * sizeof(Complex);
		int DATA_SIZE_VOLUME  = DATA_W * DATA_H * DATA_D * sizeof(float);
		int DATA_SIZE_VOLUMES  = DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float);
   
		h_A_matrix = (float *)malloc(sizeof(float) * NUMBER_OF_PARAMETERS * NUMBER_OF_PARAMETERS);
		h_inverse_A_matrix = (float *)malloc(sizeof(float) * NUMBER_OF_PARAMETERS * NUMBER_OF_PARAMETERS);
		h_h_vector = (float *)malloc(sizeof(float) * NUMBER_OF_PARAMETERS);	
		h_A_matrix_3D_values = (float30*)malloc(sizeof(float30) * DATA_W * DATA_H * DATA_D);
		h_h_vector_3D_values = (float12*)malloc(sizeof(float12) * DATA_W * DATA_H * DATA_D);
             
		 // Allocate memory on the graphics card
		cudaMalloc((void **)&d_Motion_Compensated_fMRI_Volumes, DATA_SIZE_VOLUMES);
		device_pointers[MOTION_COMPENSATED_VOLUMES] = d_Motion_Compensated_fMRI_Volumes;

		cudaMalloc((void **)&d_Compensated_Volume, DATA_SIZE_VOLUME);
		cudaMalloc((void **)&d_Reference_Volume, DATA_SIZE_VOLUME);

		cudaMalloc((void **)&d_q11,   DATA_SIZE_COMPLEX);
		cudaMalloc((void **)&d_q12,   DATA_SIZE_COMPLEX);
		cudaMalloc((void **)&d_q13,   DATA_SIZE_COMPLEX);
    
		cudaMalloc((void **)&d_q21,   DATA_SIZE_COMPLEX);
		cudaMalloc((void **)&d_q22,   DATA_SIZE_COMPLEX);
		cudaMalloc((void **)&d_q23,   DATA_SIZE_COMPLEX);

		cudaMalloc((void **)&d_Phase_Differences,   DATA_SIZE_VOLUME);
		cudaMalloc((void **)&d_Phase_Gradients,   DATA_SIZE_VOLUME);
		cudaMalloc((void **)&d_Certainties,   DATA_SIZE_VOLUME);

		cudaMalloc((void **)&d_A_matrix, sizeof(float) * NUMBER_OF_PARAMETERS * NUMBER_OF_PARAMETERS);
		cudaMalloc((void **)&d_h_vector, sizeof(float) * NUMBER_OF_PARAMETERS);

		cudaMalloc((void **)&d_A_matrix_3D_values, sizeof(float30) * DATA_W * DATA_H * DATA_D);

		cudaMalloc((void **)&d_A_matrix_2D_values, sizeof(float) * DATA_H * DATA_D * NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS);
		cudaMalloc((void **)&d_A_matrix_1D_values, sizeof(float) * DATA_D * NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS);
	
		cudaMalloc((void **)&d_h_vector_3D_values, sizeof(float12) * DATA_W * DATA_H * DATA_D);

		cudaMalloc((void **)&d_h_vector_2D_values, sizeof(float) * DATA_H * DATA_D * NUMBER_OF_PARAMETERS);
		cudaMalloc((void **)&d_h_vector_1D_values, sizeof(float) * DATA_D * NUMBER_OF_PARAMETERS);

		const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

		// Set texture parameters
		tex_Modified_Volume.normalized = false;                       // do not access with normalized texture coordinates
		tex_Modified_Volume.filterMode = cudaFilterModeLinear;        // linear interpolation

		// Allocate 3D array for modified volume (for fast interpolation)
		VOLUME_SIZE = make_cudaExtent(DATA_W, DATA_H, DATA_D);
		cudaMalloc3DArray(&d_Modified_Volume, &channelDesc, VOLUME_SIZE);
    
		// Bind the array to the 3D texture
		cudaBindTextureToArray(tex_Modified_Volume, d_Modified_Volume, channelDesc);    
           
		// ------------------------------------------------------	

		dim3 dimBlock(DATA_W, 1, 1);
		dim3 dimGrid(DATA_D, DATA_H);
    
		for (int p = 0; p < NUMBER_OF_PARAMETERS; p++)
		{
			h_Parameter_Vector_Total[p] = 0;
		}

		int threadsInX = 32;
		int threadsInY = 16;
		int threadsInZ = 1;

		int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;

		dim3 dG = dim3(blocksInX, blocksInY*blocksInZ);
		dim3 dB = dim3(threadsInX, threadsInY, threadsInZ);

		int xBlockDifference, yBlockDifference, zBlockDifference;

		int threadsInX_Conv3 = 8;
		int	threadsInY_Conv3 = 8;
		int	threadsInZ_Conv3 = 8;

		int	blocksInX_Conv3 = (DATA_W+threadsInX_Conv3-1)/threadsInX_Conv3;
		int	blocksInY_Conv3 = (DATA_H+threadsInY_Conv3-1)/threadsInY_Conv3;
		int	blocksInZ_Conv3 = (DATA_D+threadsInZ_Conv3-1)/threadsInZ_Conv3;
		
		dim3 dimGrid_Conv3 = dim3(blocksInX_Conv3, blocksInY_Conv3*blocksInZ_Conv3);
		dim3 dimBlock_Conv3 = dim3(threadsInX_Conv3, threadsInY_Conv3, threadsInZ_Conv3);

		if ( (DATA_W % threadsInX_Conv3) == 0)
		{
			xBlockDifference = 0;
		}
		else
		{
			xBlockDifference = threadsInX_Conv3 - (DATA_W % threadsInX_Conv3);
		}

		if ( (DATA_H % threadsInY_Conv3) == 0)
		{
			yBlockDifference = 0;
		}
		else
		{
			yBlockDifference = threadsInY_Conv3 - (DATA_H % threadsInY_Conv3);
		}

		if ( (DATA_D % threadsInZ_Conv3) == 0)
		{
			zBlockDifference = 0;
		}
		else
		{
			zBlockDifference = threadsInZ_Conv3 - (DATA_D % threadsInZ_Conv3);
		}

		// Set the first volume as the reference volume
		cudaMemcpy(d_Reference_Volume, d_fMRI_Volumes, DATA_SIZE_VOLUME, cudaMemcpyDeviceToDevice);
		//cutilCheckMsg(" ");

		// Calculate the filter responses for the reference volume (only needed once)
		Convolve_3D_Complex_7x7x7<<<dimGrid_Conv3, dimBlock_Conv3>>>(d_Reference_Volume, d_q11, d_q12, d_q13, DATA_W, DATA_H, DATA_D, blocksInY_Conv3, 1/(float)blocksInY_Conv3, xBlockDifference, yBlockDifference, zBlockDifference);
		//cutilCheckMsg(" ");

		sdkCreateTimer(&hTimer);
		sdkCreateTimer(&hTimer2);

		processing_times[COPY] = 0.0;
		processing_times[CONVOLVE] = 0.0;
		processing_times[PHASEDC] = 0.0;
		processing_times[PHASEG] = 0.0;
		processing_times[AH2D] = 0.0;
		processing_times[EQSYSTEM] = 0.0;
		processing_times[INTERPOLATION] = 0.0;

		cudaThreadSynchronize();
		sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);

		// Run the registration for each volume
		for (int t = 0; t < DATA_T; t++)
		{
			//std::cout << "Motion compensating volume " << t << std::endl;

			// Reset the parameter vector
			for (int p = 0; p < NUMBER_OF_PARAMETERS; p++)
			{
				h_Parameter_Vector_Total[p] = 0;
			}

			cudaThreadSynchronize();
			sdkResetTimer(&hTimer2);
			sdkStartTimer(&hTimer2);

			// Set a new volume as the modified volume
			cudaMemcpy(d_Compensated_Volume, &d_fMRI_Volumes[t * DATA_W * DATA_H * DATA_D], DATA_SIZE_VOLUME, cudaMemcpyDeviceToDevice);
			//cutilCheckMsg(" ");

			cudaMemcpy3DParms copyParams = {0};
			copyParams.srcPtr   = make_cudaPitchedPtr((void*)(&d_fMRI_Volumes[t * DATA_W * DATA_H * DATA_D]), sizeof(float)*DATA_W , DATA_W, DATA_H);
			copyParams.dstArray = d_Modified_Volume;
			copyParams.extent   = VOLUME_SIZE;
			copyParams.kind     = cudaMemcpyDeviceToDevice;
			cudaMemcpy3D(&copyParams);
			//cutilCheckMsg(" ");
		
			cudaThreadSynchronize();
			sdkStopTimer(&hTimer2);
			processing_times[COPY] += sdkGetTimerValue(&hTimer2);

			// Run the registration algorithm
			for (int it = 0; it < NUMBER_OF_ITERATIONS_FOR_MOTION_COMPENSATION; it++)
			{		
				dimBlock.x = DATA_H; dimBlock.y = 1; dimBlock.z = 1;
				dimGrid.x = DATA_D; dimGrid.y = 1;

				// Apply the three quadrature filters
			
				cudaThreadSynchronize();
				sdkResetTimer(&hTimer2);
				sdkStartTimer(&hTimer2);

				Convolve_3D_Complex_7x7x7<<<dimGrid_Conv3, dimBlock_Conv3>>>(d_Compensated_Volume, d_q21, d_q22, d_q23, DATA_W, DATA_H, DATA_D, blocksInY_Conv3, 1/(float)blocksInY_Conv3, xBlockDifference, yBlockDifference, 	zBlockDifference);			

				cudaThreadSynchronize();
				sdkStopTimer(&hTimer2);
				processing_times[CONVOLVE] += sdkGetTimerValue(&hTimer2);

				cudaThreadSynchronize();
				sdkResetTimer(&hTimer2);
				sdkStartTimer(&hTimer2);

				Calculate_Phase_Differences_And_Certainties<<<dG, dB>>>(d_Phase_Differences, d_Certainties, d_q11, d_q21, DATA_W, DATA_H, DATA_D, MOTION_COMPENSATION_FILTER_SIZE, blocksInY, 1.0f/(float)blocksInY);

				cudaThreadSynchronize();
				sdkStopTimer(&hTimer2);
				processing_times[PHASEDC] += sdkGetTimerValue(&hTimer2);


				cudaThreadSynchronize();
				sdkResetTimer(&hTimer2);
				sdkStartTimer(&hTimer2);

				Calculate_Phase_Gradients_X<<<dG, dB>>>(d_Phase_Gradients, d_q11, d_q21, DATA_W, DATA_H, DATA_D, MOTION_COMPENSATION_FILTER_SIZE, blocksInY, 1.0f/(float)blocksInY);			
				//Calculate_A_matrix_and_h_vector_3D_values_X<<<dimGrid, dimBlock>>>(d_A_matrix_3D_values, d_h_vector_3D_values, d_Phase_Differences, d_Phase_Gradients, d_Certainties, DATA_W, DATA_H, DATA_D, MOTION_COMPENSATION_FILTER_SIZE, blocksInY, 1.0f/(float)blocksInY);

				cudaThreadSynchronize();
				sdkStopTimer(&hTimer2);
				processing_times[PHASEG] += sdkGetTimerValue(&hTimer2);

				cudaThreadSynchronize();
				sdkResetTimer(&hTimer2);
				sdkStartTimer(&hTimer2);

				Calculate_A_matrix_and_h_vector_2D_values_X<<<dimGrid, dimBlock>>>(d_A_matrix_2D_values, d_h_vector_2D_values, d_Phase_Differences, d_Phase_Gradients, d_Certainties, DATA_W, DATA_H, DATA_D, MOTION_COMPENSATION_FILTER_SIZE);

				cudaThreadSynchronize();
				sdkStopTimer(&hTimer2);
				processing_times[AH2D] += sdkGetTimerValue(&hTimer2);

				cudaThreadSynchronize();
				sdkResetTimer(&hTimer2);
				sdkStartTimer(&hTimer2);

				Calculate_Phase_Differences_And_Certainties<<<dG, dB>>>(d_Phase_Differences, d_Certainties, d_q12, d_q22, DATA_W, DATA_H, DATA_D, MOTION_COMPENSATION_FILTER_SIZE, blocksInY, 1.0f/(float)blocksInY);

				cudaThreadSynchronize();
				sdkStopTimer(&hTimer2);
				processing_times[PHASEDC] += sdkGetTimerValue(&hTimer2);

				cudaThreadSynchronize();
				sdkResetTimer(&hTimer2);
				sdkStartTimer(&hTimer2);

				Calculate_Phase_Gradients_Y<<<dG, dB>>>(d_Phase_Gradients, d_q12, d_q22, DATA_W, DATA_H, DATA_D, MOTION_COMPENSATION_FILTER_SIZE, blocksInY, 1.0f/(float)blocksInY);			
				//Calculate_A_matrix_and_h_vector_3D_values_Y<<<dimGrid, dimBlock>>>(d_A_matrix_3D_values, d_h_vector_3D_values, d_Phase_Differences, d_Phase_Gradients, d_Certainties, DATA_W, DATA_H, DATA_D, MOTION_COMPENSATION_FILTER_SIZE, blocksInY, 1.0f/(float)blocksInY);				

				cudaThreadSynchronize();
				sdkStopTimer(&hTimer2);
				processing_times[PHASEG] += sdkGetTimerValue(&hTimer2);

				cudaThreadSynchronize();
				sdkResetTimer(&hTimer2);
				sdkStartTimer(&hTimer2);

				Calculate_A_matrix_and_h_vector_2D_values_Y<<<dimGrid, dimBlock>>>(d_A_matrix_2D_values, d_h_vector_2D_values, d_Phase_Differences, d_Phase_Gradients, d_Certainties, DATA_W, DATA_H, DATA_D, MOTION_COMPENSATION_FILTER_SIZE);

				cudaThreadSynchronize();
				sdkStopTimer(&hTimer2);
				processing_times[AH2D] += sdkGetTimerValue(&hTimer2);

				cudaThreadSynchronize();
				sdkResetTimer(&hTimer2);
				sdkStartTimer(&hTimer2);

				Calculate_Phase_Differences_And_Certainties<<<dG, dB>>>(d_Phase_Differences, d_Certainties, d_q13, d_q23, DATA_W, DATA_H, DATA_D, MOTION_COMPENSATION_FILTER_SIZE, blocksInY, 1.0f/(float)blocksInY);

				cudaThreadSynchronize();
				sdkStopTimer(&hTimer2);
				processing_times[PHASEDC] += sdkGetTimerValue(&hTimer2);

				cudaThreadSynchronize();
				sdkResetTimer(&hTimer2);
				sdkStartTimer(&hTimer2);

				Calculate_Phase_Gradients_Z<<<dG, dB>>>(d_Phase_Gradients, d_q13, d_q23, DATA_W, DATA_H, DATA_D, MOTION_COMPENSATION_FILTER_SIZE, blocksInY, 1.0f/(float)blocksInY);			
				//Calculate_A_matrix_and_h_vector_3D_values_Z<<<dimGrid, dimBlock>>>(d_A_matrix_3D_values, d_h_vector_3D_values, d_Phase_Differences, d_Phase_Gradients, d_Certainties, DATA_W, DATA_H, DATA_D, MOTION_COMPENSATION_FILTER_SIZE, blocksInY, 1.0f/(float)blocksInY);				

				cudaThreadSynchronize();
				sdkStopTimer(&hTimer2);
				processing_times[PHASEG] += sdkGetTimerValue(&hTimer2);

				cudaThreadSynchronize();
				sdkResetTimer(&hTimer2);
				sdkStartTimer(&hTimer2);

				Calculate_A_matrix_and_h_vector_2D_values_Z<<<dimGrid, dimBlock>>>(d_A_matrix_2D_values, d_h_vector_2D_values, d_Phase_Differences, d_Phase_Gradients, d_Certainties, DATA_W, DATA_H, DATA_D, MOTION_COMPENSATION_FILTER_SIZE);

				cudaThreadSynchronize();
				sdkStopTimer(&hTimer2);
				processing_times[AH2D] += sdkGetTimerValue(&hTimer2);

				// Sum all values
				
				//cudaMemcpy(h_h_vector_3D_values, d_h_vector_3D_values, sizeof(float12) * DATA_W * DATA_H * DATA_D, cudaMemcpyDeviceToHost);
				//cudaMemcpy(h_A_matrix_3D_values, d_A_matrix_3D_values, sizeof(float30) * DATA_W * DATA_H * DATA_D, cudaMemcpyDeviceToHost);
			
				//thrust::device_vector<float12> d_vec1(h_h_vector_3D_values, &h_h_vector_3D_values[DATA_W * DATA_H * DATA_D]);
				//thrust::device_vector<float30> d_vec2(h_A_matrix_3D_values, &h_A_matrix_3D_values[DATA_W * DATA_H * DATA_D]);
				
				//float12 zeros_12;
				//Set_float12_to_zeros(zeros_12);				
				//float12_unary_op  unary_op_12;
				//float12_binary_op binary_op_12;

				//float30 zeros_30;
				//Set_float30_to_zeros(zeros_30);				
				//float30_unary_op  unary_op_30;
				//float30_binary_op binary_op_30;

				//float12 h_h_vector_values = thrust::transform_reduce(d_vec1.begin(), d_vec1.end(), unary_op_12, zeros_12, binary_op_12);
				//float30 h_A_matrix_values = thrust::transform_reduce(d_vec2.begin(), d_vec2.end(), unary_op_30, zeros_30, binary_op_30);
				
				cudaThreadSynchronize();
				sdkResetTimer(&hTimer2);
				sdkStartTimer(&hTimer2);

				
    			// Setup final equation system

				// Sum in one direction to get 1D values
			
				dimBlock.x = DATA_D; dimBlock.y = 1; dimBlock.z = 1;
				dimGrid.x = NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS; dimGrid.y = 1;        
				Calculate_A_matrix_1D_values<<<dimGrid, dimBlock>>>(d_A_matrix_1D_values, d_A_matrix_2D_values, DATA_W, DATA_H, DATA_D, MOTION_COMPENSATION_FILTER_SIZE);
				
				// Sum in one direction to get 1D values
				dimBlock.x = DATA_D; dimBlock.y = 1; dimBlock.z = 1;
				dimGrid.x = NUMBER_OF_PARAMETERS; dimGrid.y = 1;  
				Calculate_h_vector_1D_values<<<dimGrid, dimBlock>>>(d_h_vector_1D_values, d_h_vector_2D_values, DATA_W, DATA_H, DATA_D, MOTION_COMPENSATION_FILTER_SIZE);
				
				dimBlock.x = NUMBER_OF_PARAMETERS * NUMBER_OF_PARAMETERS; dimBlock.y = 1; dimBlock.z = 1;
    			dimGrid.x = 1; dimGrid.y = 1;     
				Reset_A_matrix<<<dimGrid, dimBlock>>>(d_A_matrix);
				
				// Calculate final A-matrix
				dimBlock.x = NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS; dimBlock.y = 1; dimBlock.z = 1;
    			dimGrid.x = 1; dimGrid.y = 1;        	
				Calculate_A_matrix<<<dimGrid, dimBlock>>>(d_A_matrix, d_A_matrix_1D_values, DATA_W, DATA_H, DATA_D, MOTION_COMPENSATION_FILTER_SIZE);
				
				// Calculate final h-vector
				dimBlock.x = NUMBER_OF_PARAMETERS; dimBlock.y = 1; dimBlock.z = 1;
    			dimGrid.x = 1; dimGrid.y = 1; 
				Calculate_h_vector<<<dimGrid, dimBlock>>>(d_h_vector, d_h_vector_1D_values, DATA_W, DATA_H, DATA_D, MOTION_COMPENSATION_FILTER_SIZE);
				

				// Copy A-matrix and h-vector from device to host
				cudaMemcpy(h_A_matrix, d_A_matrix, sizeof(float) * NUMBER_OF_PARAMETERS * NUMBER_OF_PARAMETERS, cudaMemcpyDeviceToHost);
				cudaMemcpy(h_h_vector, d_h_vector, sizeof(float) * NUMBER_OF_PARAMETERS, cudaMemcpyDeviceToHost);

				// Mirror the matrix values
				for (int j = 0; j < NUMBER_OF_PARAMETERS; j++)
				{
					for (int i = 0; i < NUMBER_OF_PARAMETERS; i++)
					{
						h_A_matrix[j + i*NUMBER_OF_PARAMETERS] = h_A_matrix[i + j*NUMBER_OF_PARAMETERS];
					}
				}
		
				// Get the parameter vector 
				SolveEquationSystem(h_A_matrix, h_inverse_A_matrix, h_h_vector, h_Parameter_Vector, NUMBER_OF_PARAMETERS);

				// Update the total parameter vector
				h_Parameter_Vector_Total[0]  += h_Parameter_Vector[0];
				h_Parameter_Vector_Total[1]  += h_Parameter_Vector[1];
				h_Parameter_Vector_Total[2]  += h_Parameter_Vector[2];
				h_Parameter_Vector_Total[3]  += h_Parameter_Vector[3];
				h_Parameter_Vector_Total[4]  += h_Parameter_Vector[4];
				h_Parameter_Vector_Total[5]  += h_Parameter_Vector[5];
				h_Parameter_Vector_Total[6]  += h_Parameter_Vector[6];
				h_Parameter_Vector_Total[7]  += h_Parameter_Vector[7];
				h_Parameter_Vector_Total[8]  += h_Parameter_Vector[8];
				h_Parameter_Vector_Total[9]  += h_Parameter_Vector[9];
				h_Parameter_Vector_Total[10] += h_Parameter_Vector[10];
				h_Parameter_Vector_Total[11] += h_Parameter_Vector[11];

				cudaThreadSynchronize();
				sdkStopTimer(&hTimer2);
				processing_times[EQSYSTEM] += sdkGetTimerValue(&hTimer2);

				cudaThreadSynchronize();
				sdkResetTimer(&hTimer2);
				sdkStartTimer(&hTimer2);

		 		// Interpolate to get the new volume
				InterpolateVolumeTriLinear<<<dG, dB>>>(d_Compensated_Volume, h_Parameter_Vector_Total[0], h_Parameter_Vector_Total[1], h_Parameter_Vector_Total[2], h_Parameter_Vector_Total[3], h_Parameter_Vector_Total[4],
	       									           h_Parameter_Vector_Total[5], h_Parameter_Vector_Total[6], h_Parameter_Vector_Total[7], h_Parameter_Vector_Total[8], h_Parameter_Vector_Total[9], h_Parameter_Vector_Total[10],
	       									           h_Parameter_Vector_Total[11], DATA_W, DATA_H, DATA_D, blocksInY, 1.0f/(float)blocksInY);
				
				cudaThreadSynchronize();
				sdkStopTimer(&hTimer2);
				processing_times[INTERPOLATION] += sdkGetTimerValue(&hTimer2);
			}

			// Copy the compensated volume to the compensated volumes
			cudaMemcpy(&d_Motion_Compensated_fMRI_Volumes[t * DATA_W * DATA_H * DATA_D], d_Compensated_Volume, DATA_SIZE_VOLUME, cudaMemcpyDeviceToDevice);
			
			cudaThreadSynchronize();
			sdkStopTimer(&hTimer2);
			processing_times[COPY] += sdkGetTimerValue(&hTimer2);

			// Write the total parameter vector to host
			for (int i = 0; i < NUMBER_OF_PARAMETERS; i++)
			{
				h_Registration_Parameters[t + i * DATA_T] = h_Parameter_Vector_Total[i];
			}
		}
	
		cudaThreadSynchronize();
		sdkStopTimer(&hTimer);
		processing_times[MOTION_COMPENSATION] = sdkGetTimerValue(&hTimer);
		sdkDeleteTimer(&hTimer);
		sdkDeleteTimer(&hTimer2);

		cudaMemcpy(h_Motion_Compensated_fMRI_Volumes, d_Motion_Compensated_fMRI_Volumes, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToHost);
			
		for (int t = 0; t < DATA_T; t++)
		{
			motion_parameters_x[t] = (double)h_Registration_Parameters[t + 0 * DATA_T] * x_size;
			motion_parameters_y[t] = (double)h_Registration_Parameters[t + 1 * DATA_T] * y_size;
			motion_parameters_z[t] = (double)h_Registration_Parameters[t + 2 * DATA_T] * z_size;
		}



		if (WRITE_DATA == YES)
		{
			WriteRealDataFloat(h_Registration_Parameters, filename_registration_parameters, NUMBER_OF_PARAMETERS * DATA_T);
			WriteRealDataFloat(h_Motion_Compensated_fMRI_Volumes, filename_motion_compensated_fMRI_volumes, DATA_W * DATA_H * DATA_D * DATA_T);
		}

		cudaMemcpy(d_fMRI_Volumes, d_Motion_Compensated_fMRI_Volumes, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToDevice);

		// Free all the allocated memory on the graphics card

		cudaUnbindTexture(tex_Modified_Volume);
		cudaFreeArray(d_Modified_Volume);
 
		cudaFree(d_Reference_Volume);   
		cudaFree(d_Compensated_Volume);
   
		cudaFree(d_q11);
		cudaFree(d_q12);
		cudaFree(d_q13);
    
		cudaFree(d_q21);
 		cudaFree(d_q22);
		cudaFree(d_q23);

		cudaFree(d_Phase_Differences);	
		cudaFree(d_Phase_Gradients);
		cudaFree(d_Certainties);

		cudaFree(d_A_matrix);
		cudaFree(d_h_vector);

		cudaFree(d_A_matrix_3D_values);
		cudaFree(d_A_matrix_2D_values);
		cudaFree(d_A_matrix_1D_values);

		cudaFree(d_h_vector_3D_values);
		cudaFree(d_h_vector_2D_values);
		cudaFree(d_h_vector_1D_values); 
	
		// Free all allocated memory from Matlab

		free(h_A_matrix);
		free(h_inverse_A_matrix);
		free(h_h_vector);

		free(h_A_matrix_3D_values);
		free(h_h_vector_3D_values);

		MOTION_COMPENSATED = true;
	}
}



void WABAACUDA_LIB::PerformSmoothing()
{
	sdkCreateTimer(&hTimer);

	// CCA 2D
	if ( (ANALYSIS_METHOD == CCA) && (SMOOTHING_DIMENSIONALITY == SMOOTHING_2D) )
	{
		if (device_pointers[SMOOTHED1] == NULL)
		{
			cudaMalloc((void**)&d_Smoothed_fMRI_Volumes_1, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
			device_pointers[SMOOTHED1] = d_Smoothed_fMRI_Volumes_1;
		}
		if (device_pointers[SMOOTHED2] == NULL)
		{
			cudaMalloc((void**)&d_Smoothed_fMRI_Volumes_2, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
			device_pointers[SMOOTHED2] = d_Smoothed_fMRI_Volumes_2;
		}
		if (device_pointers[SMOOTHED3] == NULL)
		{
			cudaMalloc((void**)&d_Smoothed_fMRI_Volumes_3, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
			device_pointers[SMOOTHED3] = d_Smoothed_fMRI_Volumes_3;
		}
		if (device_pointers[SMOOTHED4] == NULL)
		{
			cudaMalloc((void**)&d_Smoothed_fMRI_Volumes_4, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
			device_pointers[SMOOTHED4] = d_Smoothed_fMRI_Volumes_4;
		}

		int threadsInX = 32;
		int	threadsInY = 16;
		int	threadsInT = 1;

		int	blocksInX = (DATA_W+threadsInX-1)/threadsInX / 3 * 2;
		int	blocksInY = (DATA_H+threadsInY-1)/threadsInY / 3;
		int	blocksInT = (DATA_T+threadsInT-1)/threadsInT;

		blocksInX = 2;
		blocksInY = 2;

		while (blocksInX * 48 < DATA_W)
		{
			blocksInX++;
		}

		while (blocksInY * 48 < DATA_H)
		{
			blocksInY++;
		}

		dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInT);
		dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInT);

		int xBlockDifference, yBlockDifference;

		Calculate_Block_Differences2D(xBlockDifference, yBlockDifference, DATA_W, DATA_H, threadsInX, threadsInY);

		xBlockDifference = 16;
		yBlockDifference = 16;

		cudaThreadSynchronize();
		sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);

		for (int z = 0; z < DATA_D; z++)
		{
			Smoothing_CCA_2D<<<dimGrid, dimBlock>>>(d_Smoothed_fMRI_Volumes_1, d_Smoothed_fMRI_Volumes_2, d_Smoothed_fMRI_Volumes_3, d_Smoothed_fMRI_Volumes_4, d_fMRI_Volumes, z, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), xBlockDifference, yBlockDifference);
		}

		cudaThreadSynchronize();
	    sdkStopTimer(&hTimer);
		processing_times[SMOOTHING] = sdkGetTimerValue(&hTimer);

		cudaMemcpy(h_Smoothed_fMRI_Volumes_1, d_Smoothed_fMRI_Volumes_1, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToHost);
	
		if (WRITE_DATA == YES)
		{
			if (host_pointers[SMOOTHED2] == NULL)
			{
				h_Smoothed_fMRI_Volumes_2 = (float*)malloc(sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
				host_pointers[SMOOTHED2] = h_Smoothed_fMRI_Volumes_2;
			}
			if (host_pointers[SMOOTHED3] == NULL)
			{
				h_Smoothed_fMRI_Volumes_3 = (float*)malloc(sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
				host_pointers[SMOOTHED3] = h_Smoothed_fMRI_Volumes_3;
			}
			if (host_pointers[SMOOTHED4] == NULL)
			{
				h_Smoothed_fMRI_Volumes_4 = (float*)malloc(sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
				host_pointers[SMOOTHED4] = h_Smoothed_fMRI_Volumes_4;
			}

			cudaMemcpy(h_Smoothed_fMRI_Volumes_2, d_Smoothed_fMRI_Volumes_2, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_Smoothed_fMRI_Volumes_3, d_Smoothed_fMRI_Volumes_3, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_Smoothed_fMRI_Volumes_4, d_Smoothed_fMRI_Volumes_4, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToHost);
			
			WriteRealDataFloat(h_Smoothed_fMRI_Volumes_1, filename_smoothed_fMRI_volumes_1, DATA_W * DATA_H * DATA_D * DATA_T);
			WriteRealDataFloat(h_Smoothed_fMRI_Volumes_2, filename_smoothed_fMRI_volumes_2, DATA_W * DATA_H * DATA_D * DATA_T);
			WriteRealDataFloat(h_Smoothed_fMRI_Volumes_3, filename_smoothed_fMRI_volumes_3, DATA_W * DATA_H * DATA_D * DATA_T);
			WriteRealDataFloat(h_Smoothed_fMRI_Volumes_4, filename_smoothed_fMRI_volumes_4, DATA_W * DATA_H * DATA_D * DATA_T);
		}
	}
	// CCA 3D
	else if ( (ANALYSIS_METHOD == CCA) && (SMOOTHING_DIMENSIONALITY == SMOOTHING_3D) )
	{
		float  *d_Filter_Response_Rows, *d_Filter_Response_Columns;

		int DATA_SIZE_fMRI_VOLUME  = DATA_W * DATA_H * DATA_D * sizeof(float);
		
		if (device_pointers[SMOOTHED1] == NULL)
		{
			cudaMalloc((void**)&d_Smoothed_fMRI_Volumes_1, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
			device_pointers[SMOOTHED1] = d_Smoothed_fMRI_Volumes_1;
		}
		if (device_pointers[SMOOTHED2] == NULL)
		{
			cudaMalloc((void**)&d_Smoothed_fMRI_Volumes_2, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
			device_pointers[SMOOTHED2] = d_Smoothed_fMRI_Volumes_2;
		}

		cudaMalloc((void **)&d_Filter_Response_Rows, DATA_SIZE_fMRI_VOLUME);	
		cudaMalloc((void **)&d_Filter_Response_Columns, DATA_SIZE_fMRI_VOLUME);	

		int threadsInX, threadsInY, threadsInZ;
		int blocksInX, blocksInY, blocksInZ;
		int xBlockDifference, yBlockDifference, zBlockDifference;
		dim3 dimGrid, dimBlock;
		
		threadsInX = 32;
		threadsInY = 8;
		threadsInZ = 1;
		
		blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		cudaThreadSynchronize();
		sdkResetTimer(&hTimer);
	    sdkStartTimer(&hTimer);

		// Loop over timepoints
		for (int t = 0; t < DATA_T; t++)
		{
			// Filter 1
			// Copy the smoothing kernel to constant memory
			cudaMemcpyToSymbol(c_Smoothing_Kernel, h_CCA_3D_Filter_1, SMOOTHING_FILTER_SIZE * sizeof(float));

			// Convolve rows

			threadsInX = 32;
			threadsInY = 8;
			threadsInZ = 2;

			blocksInX = (DATA_W+threadsInX-1)/threadsInX;
			blocksInY = (DATA_H+threadsInY-1)/threadsInY;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX, threadsInY, threadsInZ * 4);

			convolutionRows<<<dimGrid, dimBlock>>>(d_Filter_Response_Rows, d_fMRI_Volumes, d_Brain_Voxels, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);

			// Convolve columns

			threadsInX = 32;
			threadsInY = 8;
			threadsInZ = 2;

			blocksInX = (DATA_W+threadsInX-1)/(threadsInX / 32 * 24);
			blocksInY = (DATA_H+threadsInY-1)/threadsInY / 2;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX / 32 * 24, threadsInY * 2, threadsInZ * 4);

			convolutionColumns<<<dimGrid, dimBlock>>>(d_Filter_Response_Columns, d_Filter_Response_Rows, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);

			// Convolve rods
	
			threadsInX = 32;
			threadsInY = 2;
			threadsInZ = 8;

			blocksInX = (DATA_W+threadsInX-1)/threadsInX;
			blocksInY = (DATA_H+threadsInY-1)/threadsInY / 4;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);
	
			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX, threadsInY * 4, threadsInZ);

			convolutionRods<<<dimGrid, dimBlock>>>(d_Smoothed_fMRI_Volumes_1, d_Filter_Response_Columns, d_Smoothed_Certainty, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);
			
			// Filter 2
			// Copy the smoothing kernel to constant memory
		    cudaMemcpyToSymbol(c_Smoothing_Kernel, h_CCA_3D_Filter_2, SMOOTHING_FILTER_SIZE * sizeof(float));

			// Convolve rows

			threadsInX = 32;
			threadsInY = 8;
			threadsInZ = 2;

			blocksInX = (DATA_W+threadsInX-1)/threadsInX;
			blocksInY = (DATA_H+threadsInY-1)/threadsInY;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX, threadsInY, threadsInZ * 4);

			convolutionRows<<<dimGrid, dimBlock>>>(d_Filter_Response_Rows, d_fMRI_Volumes, d_Brain_Voxels, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);

			// Convolve columns

			threadsInX = 32;
			threadsInY = 8;
			threadsInZ = 2;

			blocksInX = (DATA_W+threadsInX-1)/(threadsInX / 32 * 24);
			blocksInY = (DATA_H+threadsInY-1)/threadsInY / 2;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX / 32 * 24, threadsInY * 2, threadsInZ * 4);

			convolutionColumns<<<dimGrid, dimBlock>>>(d_Filter_Response_Columns, d_Filter_Response_Rows, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);

			// Convolve rods
	
			threadsInX = 32;
			threadsInY = 2;
			threadsInZ = 8;

			blocksInX = (DATA_W+threadsInX-1)/threadsInX;
			blocksInY = (DATA_H+threadsInY-1)/threadsInY / 4;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX, threadsInY * 4, threadsInZ);

			convolutionRods<<<dimGrid, dimBlock>>>(d_Smoothed_fMRI_Volumes_2, d_Filter_Response_Columns, d_Smoothed_Certainty, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);
		}
    
		cudaThreadSynchronize();
	    sdkStopTimer(&hTimer);
		processing_times[SMOOTHING] = sdkGetTimerValue(&hTimer);
		
		// Free all the allocated memory on the graphics card
		cudaFree(d_Filter_Response_Rows);
		cudaFree(d_Filter_Response_Columns);

		cudaMemcpy(h_Smoothed_fMRI_Volumes_1, d_Smoothed_fMRI_Volumes_1, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToHost);
	
		if (WRITE_DATA == YES)
		{
			if (host_pointers[SMOOTHED2] == NULL)
			{
				h_Smoothed_fMRI_Volumes_2 = (float*)malloc(sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
				host_pointers[SMOOTHED2] = h_Smoothed_fMRI_Volumes_2;
			}
			cudaMemcpy(h_Smoothed_fMRI_Volumes_2, d_Smoothed_fMRI_Volumes_2, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToHost);
			WriteRealDataFloat(h_Smoothed_fMRI_Volumes_1, filename_smoothed_fMRI_volumes_1, DATA_W * DATA_H * DATA_D * DATA_T);
			WriteRealDataFloat(h_Smoothed_fMRI_Volumes_2, filename_smoothed_fMRI_volumes_2, DATA_W * DATA_H * DATA_D * DATA_T);
		}
	}
	// GLM 2D
	else if ( (ANALYSIS_METHOD == GLM) && (SMOOTHING_DIMENSIONALITY == SMOOTHING_2D) )
	{
		float  *d_Filter_Response_Rows;

		int DATA_SIZE_fMRI_VOLUME  = DATA_W * DATA_H * DATA_D * sizeof(float);

		if (device_pointers[SMOOTHED1] == NULL)
		{
			cudaMalloc((void**)&d_Smoothed_fMRI_Volumes_1, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
			device_pointers[SMOOTHED1] = d_Smoothed_fMRI_Volumes_1;
		}
		cudaMalloc((void **)&d_Filter_Response_Rows, DATA_SIZE_fMRI_VOLUME);	

		int threadsInX, threadsInY, threadsInZ;
		int blocksInX, blocksInY, blocksInZ;
		int xBlockDifference, yBlockDifference, zBlockDifference;
		dim3 dimGrid, dimBlock;

		threadsInX = 32;
		threadsInY = 8;
		threadsInZ = 1;
		
		blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		cudaThreadSynchronize();
	    sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);

		// Loop over timepoints
		for (int t = 0; t < DATA_T; t++)
		{
			// Convolve rows

			threadsInX = 32;
			threadsInY = 8;
			threadsInZ = 2;

			blocksInX = (DATA_W+threadsInX-1)/threadsInX;
			blocksInY = (DATA_H+threadsInY-1)/threadsInY;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX, threadsInY, threadsInZ * 4);

			convolutionRows<<<dimGrid, dimBlock>>>(d_Filter_Response_Rows, d_fMRI_Volumes, d_Brain_Voxels, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);
			//cutilCheckMsg(" ");

			// Convolve columns

			threadsInX = 32;
			threadsInY = 8;
			threadsInZ = 2;

			blocksInX = (DATA_W+threadsInX-1)/(threadsInX / 32 * 24);
			blocksInY = (DATA_H+threadsInY-1)/threadsInY / 2;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX / 32 * 24, threadsInY * 2, threadsInZ * 4);

			convolutionColumns2D<<<dimGrid, dimBlock>>>(d_Smoothed_fMRI_Volumes_1, d_Filter_Response_Rows, d_Smoothed_Certainty, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);
			//cutilCheckMsg(" ");
		}

		cudaThreadSynchronize();
	    sdkStopTimer(&hTimer);
		processing_times[SMOOTHING] = sdkGetTimerValue(&hTimer);
		
		// Free all the allocated memory on the graphics card
		cudaFree(d_Filter_Response_Rows);

		cudaMemcpy(h_Smoothed_fMRI_Volumes_1, d_Smoothed_fMRI_Volumes_1, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToHost);

		if (WRITE_DATA == YES)
		{
			WriteRealDataFloat(h_Smoothed_fMRI_Volumes_1, filename_smoothed_fMRI_volumes_1, DATA_W * DATA_H * DATA_D * DATA_T);
		}
	}
	// GLM 3D
	else if ( (ANALYSIS_METHOD == GLM) && (SMOOTHING_DIMENSIONALITY == SMOOTHING_3D) )
	{
		float  *d_Filter_Response_Rows, *d_Filter_Response_Columns;

		int DATA_SIZE_fMRI_VOLUME  = DATA_W * DATA_H * DATA_D * sizeof(float);

		if (device_pointers[SMOOTHED1] == NULL)
		{
			cudaMalloc((void**)&d_Smoothed_fMRI_Volumes_1, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
			device_pointers[SMOOTHED1] = d_Smoothed_fMRI_Volumes_1;
		}
		cudaMalloc((void **)&d_Filter_Response_Rows, DATA_SIZE_fMRI_VOLUME);	
		cudaMalloc((void **)&d_Filter_Response_Columns, DATA_SIZE_fMRI_VOLUME);	

		int threadsInX, threadsInY, threadsInZ;
		int blocksInX, blocksInY, blocksInZ;
		int xBlockDifference, yBlockDifference, zBlockDifference;
		dim3 dimGrid, dimBlock;

		threadsInX = 32;
		threadsInY = 8;
		threadsInZ = 1;
		
		blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		cudaThreadSynchronize();
	    sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);

		// Loop over timepoints
		for (int t = 0; t < DATA_T; t++)
		{
			// Convolve rows
	
			threadsInX = 32;
			threadsInY = 8;
			threadsInZ = 2;
		
			blocksInX = (DATA_W+threadsInX-1)/threadsInX;
			blocksInY = (DATA_H+threadsInY-1)/threadsInY;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);
	
			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX, threadsInY, threadsInZ * 4);
	
			convolutionRows<<<dimGrid, dimBlock>>>(d_Filter_Response_Rows, d_fMRI_Volumes, d_Brain_Voxels, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);
			//cutilCheckMsg(" ");
	
			// Convolve columns
				
			threadsInX = 32;
			threadsInY = 8;
			threadsInZ = 2;

			blocksInX = (DATA_W+threadsInX-1)/(threadsInX / 32 * 24);
			blocksInY = (DATA_H+threadsInY-1)/threadsInY / 2;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX / 32 * 24, threadsInY * 2, threadsInZ * 4);

			convolutionColumns<<<dimGrid, dimBlock>>>(d_Filter_Response_Columns, d_Filter_Response_Rows, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);
			//cutilCheckMsg(" ");

			// Convolve rods
	
			threadsInX = 32;
			threadsInY = 2;
			threadsInZ = 8;

			blocksInX = (DATA_W+threadsInX-1)/threadsInX;
			blocksInY = (DATA_H+threadsInY-1)/threadsInY / 4;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX, threadsInY * 4, threadsInZ);

			convolutionRods<<<dimGrid, dimBlock>>>(d_Smoothed_fMRI_Volumes_1, d_Filter_Response_Columns, d_Smoothed_Certainty, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);
			//cutilCheckMsg(" ");
		}
    
		cudaThreadSynchronize();
	    sdkStopTimer(&hTimer);
		processing_times[SMOOTHING] = sdkGetTimerValue(&hTimer);

		// Free all the allocated memory on the graphics card
		cudaFree(d_Filter_Response_Rows);
		cudaFree(d_Filter_Response_Columns);
		
		cudaMemcpy(h_Smoothed_fMRI_Volumes_1, d_Smoothed_fMRI_Volumes_1, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToHost);
			
		if (WRITE_DATA == YES)
		{
			WriteRealDataFloat(h_Smoothed_fMRI_Volumes_1, filename_smoothed_fMRI_volumes_1, DATA_W * DATA_H * DATA_D * DATA_T);
		}
		//cudaMemcpy(d_fMRI_Volumes, d_Smoothed_fMRI_Volumes_1, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToDevice);
		//cudaFree(d_Smoothed_fMRI_Volumes_1);
		//d_Smoothed_fMRI_Volumes_1 = NULL;
	}

	sdkDeleteTimer(&hTimer);
}

void WABAACUDA_LIB::PerformDetrending()
{
	sdkCreateTimer(&hTimer);

	if ( (ANALYSIS_METHOD == CCA) && (SMOOTHING_DIMENSIONALITY == SMOOTHING_2D) )
	{
		if (device_pointers[DETRENDED1] == NULL)
		{
			cudaMalloc((void **)&d_Detrended_fMRI_Volumes_1, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
			device_pointers[DETRENDED1] = d_Detrended_fMRI_Volumes_1;
		}
		if (device_pointers[DETRENDED2] == NULL)
		{
			cudaMalloc((void **)&d_Detrended_fMRI_Volumes_2, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
			device_pointers[DETRENDED2] = d_Detrended_fMRI_Volumes_2;
		}
		if (device_pointers[DETRENDED3] == NULL)
		{
			cudaMalloc((void **)&d_Detrended_fMRI_Volumes_3, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
			device_pointers[DETRENDED3] = d_Detrended_fMRI_Volumes_3;
		}
		if (device_pointers[DETRENDED4] == NULL)
		{
			cudaMalloc((void **)&d_Detrended_fMRI_Volumes_4, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
			device_pointers[DETRENDED4] = d_Detrended_fMRI_Volumes_4;
		}

		int threadsInX = 32;
		int threadsInY = 8;
		int threadsInZ = 1;

		int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		// Calculate how many time multiples there are
		int timeMultiples = DATA_T / threadsInY;
		int timeRest = DATA_T - timeMultiples * threadsInY;

		cudaThreadSynchronize();
	    sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);

		// Do the detrending
		DetrendCubic<<<dimGrid, dimBlock>>>(d_Detrended_fMRI_Volumes_1, d_Smoothed_fMRI_Volumes_1, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
		DetrendCubic<<<dimGrid, dimBlock>>>(d_Detrended_fMRI_Volumes_2, d_Smoothed_fMRI_Volumes_2, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
		DetrendCubic<<<dimGrid, dimBlock>>>(d_Detrended_fMRI_Volumes_3, d_Smoothed_fMRI_Volumes_3, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
		DetrendCubic<<<dimGrid, dimBlock>>>(d_Detrended_fMRI_Volumes_4, d_Smoothed_fMRI_Volumes_4, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);

		cudaThreadSynchronize();
	    sdkStopTimer(&hTimer);
		processing_times[DETRENDING] = sdkGetTimerValue(&hTimer);

		cudaMemcpy(h_Detrended_fMRI_Volumes_1, d_Detrended_fMRI_Volumes_1, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToHost);

		if (WRITE_DATA == YES)
		{
			if (host_pointers[DETRENDED2] == NULL)
			{
				h_Detrended_fMRI_Volumes_2 = (float*)malloc(sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
				host_pointers[DETRENDED2] = h_Detrended_fMRI_Volumes_2;
			}
			if (host_pointers[DETRENDED3] == NULL)
			{
				h_Detrended_fMRI_Volumes_3 = (float*)malloc(sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
				host_pointers[DETRENDED3] = h_Detrended_fMRI_Volumes_3;
			}
			if (host_pointers[DETRENDED4] == NULL)
			{
				h_Detrended_fMRI_Volumes_4 = (float*)malloc(sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
				host_pointers[DETRENDED4] = h_Detrended_fMRI_Volumes_4;
			}
			cudaMemcpy(h_Detrended_fMRI_Volumes_2, d_Detrended_fMRI_Volumes_2, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_Detrended_fMRI_Volumes_3, d_Detrended_fMRI_Volumes_3, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_Detrended_fMRI_Volumes_4, d_Detrended_fMRI_Volumes_4, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToHost);
			WriteRealDataFloat(h_Detrended_fMRI_Volumes_1, filename_detrended_fMRI_volumes_1, DATA_W * DATA_H * DATA_D * DATA_T);
			WriteRealDataFloat(h_Detrended_fMRI_Volumes_2, filename_detrended_fMRI_volumes_2, DATA_W * DATA_H * DATA_D * DATA_T);
			WriteRealDataFloat(h_Detrended_fMRI_Volumes_3, filename_detrended_fMRI_volumes_3, DATA_W * DATA_H * DATA_D * DATA_T);
			WriteRealDataFloat(h_Detrended_fMRI_Volumes_4, filename_detrended_fMRI_volumes_4, DATA_W * DATA_H * DATA_D * DATA_T);
		}
	}
	else if ( (ANALYSIS_METHOD == CCA) && (SMOOTHING_DIMENSIONALITY == SMOOTHING_3D) )
	{
		if (device_pointers[DETRENDED1] == NULL)
		{
			cudaMalloc((void **)&d_Detrended_fMRI_Volumes_1, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
			device_pointers[DETRENDED1] = d_Detrended_fMRI_Volumes_1;
		}
		if (device_pointers[DETRENDED2] == NULL)
		{
			cudaMalloc((void **)&d_Detrended_fMRI_Volumes_2, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
			device_pointers[DETRENDED2] = d_Detrended_fMRI_Volumes_2;
		}

		int threadsInX = 32;
		int threadsInY = 8;
		int threadsInZ = 1;

		int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		// Calculate how many time multiples there are
		int timeMultiples = DATA_T / threadsInY;
		int timeRest = DATA_T - timeMultiples * threadsInY;

		cudaThreadSynchronize();
	    sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);

		// Do the detrending
		DetrendCubic<<<dimGrid, dimBlock>>>(d_Detrended_fMRI_Volumes_1, d_Smoothed_fMRI_Volumes_1, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
		DetrendCubic<<<dimGrid, dimBlock>>>(d_Detrended_fMRI_Volumes_2, d_Smoothed_fMRI_Volumes_2, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
	
		cudaThreadSynchronize();
	    sdkStopTimer(&hTimer);
		processing_times[DETRENDING] = sdkGetTimerValue(&hTimer);

		cudaMemcpy(h_Detrended_fMRI_Volumes_1, d_Detrended_fMRI_Volumes_1, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToHost);

		if (WRITE_DATA == YES)
		{
			if (host_pointers[DETRENDED2] == NULL)
			{
				h_Detrended_fMRI_Volumes_2 = (float*)malloc(sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
				host_pointers[DETRENDED2] = h_Detrended_fMRI_Volumes_2;
			}
			cudaMemcpy(h_Detrended_fMRI_Volumes_2, d_Detrended_fMRI_Volumes_2, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToHost);
			WriteRealDataFloat(h_Detrended_fMRI_Volumes_1, filename_detrended_fMRI_volumes_1, DATA_W * DATA_H * DATA_D * DATA_T);
			WriteRealDataFloat(h_Detrended_fMRI_Volumes_2, filename_detrended_fMRI_volumes_2, DATA_W * DATA_H * DATA_D * DATA_T);
		}
	}
	else if (ANALYSIS_METHOD == GLM)
	{
		if (device_pointers[DETRENDED1] == NULL)
		{
			cudaMalloc((void **)&d_Detrended_fMRI_Volumes_1, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T);
			device_pointers[DETRENDED1] = d_Detrended_fMRI_Volumes_1;
		}

		int threadsInX = 32;
		int threadsInY = 8;
		int threadsInZ = 1;

		int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		// Calculate how many time multiples there are
		int timeMultiples = DATA_T / threadsInY;
		int timeRest = DATA_T - timeMultiples * threadsInY;

		cudaThreadSynchronize();
	    sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);

		// Do the detrending
		DetrendCubic<<<dimGrid, dimBlock>>>(d_Detrended_fMRI_Volumes_1, d_Smoothed_fMRI_Volumes_1, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);

		cudaThreadSynchronize();
	    sdkStopTimer(&hTimer);
		processing_times[DETRENDING] = sdkGetTimerValue(&hTimer);

		cudaMemcpy(h_Detrended_fMRI_Volumes_1, d_Detrended_fMRI_Volumes_1, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToHost);
		
		if (WRITE_DATA == YES)
		{
			WriteRealDataFloat(h_Detrended_fMRI_Volumes_1, filename_detrended_fMRI_volumes_1, DATA_W * DATA_H * DATA_D * DATA_T);
		}
		//cudaMemcpy(d_fMRI_Volumes, d_Detrended_fMRI_Volumes_1, sizeof(float) * DATA_W * DATA_H * DATA_D * DATA_T, cudaMemcpyDeviceToDevice);
		//cudaFree(d_Detrended_fMRI_Volumes_1);
		//d_Detrended_fMRI_Volumes_1 = NULL;
	}

	sdkDeleteTimer(&hTimer);
}

void WABAACUDA_LIB::CalculateActivityMap()
{
	sdkCreateTimer(&hTimer);

	// CCA 2D
	if ( (ANALYSIS_METHOD == CCA) && (SMOOTHING_DIMENSIONALITY == SMOOTHING_2D) )
	{
		int threadsInX = 32;
		int threadsInY = 4;
		int threadsInZ = 1;

		int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		// Calculate how many time multiples there are
		int timeMultiples = DATA_T / threadsInY;
		int timeRest = DATA_T - timeMultiples * threadsInY;
		
		cudaThreadSynchronize();
		sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);

		CalculateActivityMapCCA2D<<<dimGrid, dimBlock>>>(d_Activity_Volume, d_Detrended_fMRI_Volumes_1, d_Detrended_fMRI_Volumes_2, d_Detrended_fMRI_Volumes_3, d_Detrended_fMRI_Volumes_4, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
		//cutilCheckMsg(" ");
		
		cudaThreadSynchronize();
		sdkStopTimer(&hTimer);
		processing_times[STATISTICAL_ANALYSIS] = sdkGetTimerValue(&hTimer);
	}
	// CCA 3D
	else if ( (ANALYSIS_METHOD == CCA) && (SMOOTHING_DIMENSIONALITY == SMOOTHING_3D) )
	{
		int threadsInX = 32;
		int threadsInY = 4;
		int threadsInZ = 1;

		int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		// Calculate how many time multiples there are
		int timeMultiples = DATA_T / threadsInY;
		int timeRest = DATA_T - timeMultiples * threadsInY;

		cudaThreadSynchronize();
		sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);

		CalculateActivityMapCCA3D<<<dimGrid, dimBlock>>>(d_Activity_Volume, d_Detrended_fMRI_Volumes_1, d_Detrended_fMRI_Volumes_2, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
		//cutilCheckMsg(" ");

		cudaThreadSynchronize();
		sdkStopTimer(&hTimer);
		processing_times[STATISTICAL_ANALYSIS] = sdkGetTimerValue(&hTimer);
	}
	// GLM
	else if (ANALYSIS_METHOD == GLM)
	{
		int threadsInX = 32;
		int threadsInY = 8;
		int threadsInZ = 1;

		int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		// Calculate how many time multiples there are
		int timeMultiples = DATA_T / threadsInY;
		int timeRest = DATA_T - timeMultiples * threadsInY;

		cudaThreadSynchronize();
		sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);

		CalculateActivityMapGLM<<<dimGrid, dimBlock>>>(d_Activity_Volume, d_Detrended_fMRI_Volumes_1, d_Brain_Voxels, h_ctxtxc, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/(float)blocksInY, timeMultiples, timeRest);
		//cutilCheckMsg(" ");

		cudaThreadSynchronize();
		sdkStopTimer(&hTimer);
		processing_times[STATISTICAL_ANALYSIS] = sdkGetTimerValue(&hTimer);
	}

	cudaMemcpy(h_Activity_Volume, d_Activity_Volume, sizeof(float) * DATA_W * DATA_H * DATA_D, cudaMemcpyDeviceToHost);
	
	if (WRITE_DATA == YES)
	{
		WriteRealDataFloat(h_Activity_Volume, filename_activity_volume, DATA_W * DATA_H * DATA_D);
	}

	CalculateSlicesActivityData();

	sdkDeleteTimer(&hTimer);
}



void WABAACUDA_LIB::PerformSmoothingPermutation()
{
	// CCA 2D
	if ( (ANALYSIS_METHOD == CCA) && (SMOOTHING_DIMENSIONALITY == SMOOTHING_2D) )
	{
		int threadsInX = 32;
		int	threadsInY = 16;
		int	threadsInT = 1;

		int	blocksInX = (DATA_W+threadsInX-1)/threadsInX / 3 * 2;
		int	blocksInY = (DATA_H+threadsInY-1)/threadsInY / 3;
		int	blocksInT = (DATA_T+threadsInT-1)/threadsInT;

		blocksInX = 2;
		blocksInY = 2;

		while (blocksInX * 48 < DATA_W)
		{
			blocksInX++;
		}

		while (blocksInY * 48 < DATA_H)
		{
			blocksInY++;
		}

		dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInT);
		dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInT);

		int xBlockDifference, yBlockDifference;

		Calculate_Block_Differences2D(xBlockDifference, yBlockDifference, DATA_W, DATA_H, threadsInX, threadsInY);

		xBlockDifference = 16;
		yBlockDifference = 16;

		for (int z = 0; z < DATA_D; z++)
		{
			Smoothing_CCA_2D<<<dimGrid, dimBlock>>>(d_Smoothed_fMRI_Volumes_1, d_Smoothed_fMRI_Volumes_2, d_Smoothed_fMRI_Volumes_3, d_Smoothed_fMRI_Volumes_4, d_Permuted_fMRI_Volumes, z, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), xBlockDifference, yBlockDifference);
		}
	}
	// CCA 3D
	else if ( (ANALYSIS_METHOD == CCA) && (SMOOTHING_DIMENSIONALITY == SMOOTHING_3D) )
	{
		float  *d_Filter_Response_Rows, *d_Filter_Response_Columns;

		int DATA_SIZE_fMRI_VOLUME  = DATA_W * DATA_H * DATA_D * sizeof(float);

		cudaMalloc((void **)&d_Filter_Response_Rows, DATA_SIZE_fMRI_VOLUME);	
		cudaMalloc((void **)&d_Filter_Response_Columns, DATA_SIZE_fMRI_VOLUME);	

		int threadsInX, threadsInY, threadsInZ;
		int blocksInX, blocksInY, blocksInZ;
		int xBlockDifference, yBlockDifference, zBlockDifference;
		dim3 dimGrid, dimBlock;
		
		threadsInX = 32;
		threadsInY = 8;
		threadsInZ = 1;
		
		blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		// Loop over timepoints
		for (int t = 0; t < DATA_T; t++)
		{
			// Filter 1
			// Copy the smoothing kernel to constant memory
			cudaMemcpyToSymbol(c_Smoothing_Kernel, h_CCA_3D_Filter_1, SMOOTHING_FILTER_SIZE * sizeof(float));

			// Convolve rows

			threadsInX = 32;
			threadsInY = 8;
			threadsInZ = 2;

			blocksInX = (DATA_W+threadsInX-1)/threadsInX;
			blocksInY = (DATA_H+threadsInY-1)/threadsInY;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX, threadsInY, threadsInZ * 4);

			convolutionRows<<<dimGrid, dimBlock>>>(d_Filter_Response_Rows, d_Permuted_fMRI_Volumes, d_Brain_Voxels, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);

			// Convolve columns

			threadsInX = 32;
			threadsInY = 8;
			threadsInZ = 2;

			blocksInX = (DATA_W+threadsInX-1)/(threadsInX / 32 * 24);
			blocksInY = (DATA_H+threadsInY-1)/threadsInY / 2;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX / 32 * 24, threadsInY * 2, threadsInZ * 4);

			convolutionColumns<<<dimGrid, dimBlock>>>(d_Filter_Response_Columns, d_Filter_Response_Rows, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);

			// Convolve rods
	
			threadsInX = 32;
			threadsInY = 2;
			threadsInZ = 8;

			blocksInX = (DATA_W+threadsInX-1)/threadsInX;
			blocksInY = (DATA_H+threadsInY-1)/threadsInY / 4;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);
	
			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX, threadsInY * 4, threadsInZ);

			convolutionRods<<<dimGrid, dimBlock>>>(d_Smoothed_fMRI_Volumes_1, d_Filter_Response_Columns, d_Smoothed_Certainty, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);
			
			// Filter 2
			// Copy the smoothing kernel to constant memory
		    cudaMemcpyToSymbol(c_Smoothing_Kernel, h_CCA_3D_Filter_2, SMOOTHING_FILTER_SIZE * sizeof(float));

			// Convolve rows

			threadsInX = 32;
			threadsInY = 8;
			threadsInZ = 2;

			blocksInX = (DATA_W+threadsInX-1)/threadsInX;
			blocksInY = (DATA_H+threadsInY-1)/threadsInY;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX, threadsInY, threadsInZ * 4);

			convolutionRows<<<dimGrid, dimBlock>>>(d_Filter_Response_Rows, d_Permuted_fMRI_Volumes, d_Brain_Voxels, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);

			// Convolve columns

			threadsInX = 32;
			threadsInY = 8;
			threadsInZ = 2;

			blocksInX = (DATA_W+threadsInX-1)/(threadsInX / 32 * 24);
			blocksInY = (DATA_H+threadsInY-1)/threadsInY / 2;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX / 32 * 24, threadsInY * 2, threadsInZ * 4);

			convolutionColumns<<<dimGrid, dimBlock>>>(d_Filter_Response_Columns, d_Filter_Response_Rows, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);

			// Convolve rods
	
			threadsInX = 32;
			threadsInY = 2;
			threadsInZ = 8;

			blocksInX = (DATA_W+threadsInX-1)/threadsInX;
			blocksInY = (DATA_H+threadsInY-1)/threadsInY / 4;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX, threadsInY * 4, threadsInZ);

			convolutionRods<<<dimGrid, dimBlock>>>(d_Smoothed_fMRI_Volumes_2, d_Filter_Response_Columns, d_Smoothed_Certainty, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);
		}
    
		// Free all the allocated memory on the graphics card
		cudaFree(d_Filter_Response_Rows);
		cudaFree(d_Filter_Response_Columns);
	}
	// GLM 2D
	else if ( (ANALYSIS_METHOD == GLM) && (SMOOTHING_DIMENSIONALITY == SMOOTHING_2D) )
	{
		float  *d_Filter_Response_Rows;

		int DATA_SIZE_fMRI_VOLUME  = DATA_W * DATA_H * DATA_D * sizeof(float);

		cudaMalloc((void **)&d_Filter_Response_Rows, DATA_SIZE_fMRI_VOLUME);	

		int threadsInX, threadsInY, threadsInZ;
		int blocksInX, blocksInY, blocksInZ;
		int xBlockDifference, yBlockDifference, zBlockDifference;
		dim3 dimGrid, dimBlock;

		threadsInX = 32;
		threadsInY = 8;
		threadsInZ = 1;
		
		blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		// Loop over timepoints
		for (int t = 0; t < DATA_T; t++)
		{
			// Convolve rows

			threadsInX = 32;
			threadsInY = 8;
			threadsInZ = 2;

			blocksInX = (DATA_W+threadsInX-1)/threadsInX;
			blocksInY = (DATA_H+threadsInY-1)/threadsInY;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX, threadsInY, threadsInZ * 4);

			convolutionRows<<<dimGrid, dimBlock>>>(d_Filter_Response_Rows, d_Permuted_fMRI_Volumes, d_Brain_Voxels, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);
			//cutilCheckMsg(" ");

			// Convolve columns

			threadsInX = 32;
			threadsInY = 8;
			threadsInZ = 2;

			blocksInX = (DATA_W+threadsInX-1)/(threadsInX / 32 * 24);
			blocksInY = (DATA_H+threadsInY-1)/threadsInY / 2;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX / 32 * 24, threadsInY * 2, threadsInZ * 4);

			convolutionColumns2D<<<dimGrid, dimBlock>>>(d_Smoothed_fMRI_Volumes_1, d_Filter_Response_Rows, d_Smoothed_Certainty, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);
		}

		// Free all the allocated memory on the graphics card
		cudaFree(d_Filter_Response_Rows);
	}
	// GLM 3D
	else if ( (ANALYSIS_METHOD == GLM) && (SMOOTHING_DIMENSIONALITY == SMOOTHING_3D) )
	{
		float  *d_Filter_Response_Rows, *d_Filter_Response_Columns;

		int DATA_SIZE_fMRI_VOLUME  = DATA_W * DATA_H * DATA_D * sizeof(float);

		cudaMalloc((void **)&d_Filter_Response_Rows, DATA_SIZE_fMRI_VOLUME);	
		cudaMalloc((void **)&d_Filter_Response_Columns, DATA_SIZE_fMRI_VOLUME);	

		int threadsInX, threadsInY, threadsInZ;
		int blocksInX, blocksInY, blocksInZ;
		int xBlockDifference, yBlockDifference, zBlockDifference;
		dim3 dimGrid, dimBlock;

		threadsInX = 32;
		threadsInY = 8;
		threadsInZ = 1;
		
		blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		// Loop over timepoints
		for (int t = 0; t < DATA_T; t++)
		{
			// Convolve rows
	
			threadsInX = 32;
			threadsInY = 8;
			threadsInZ = 2;
		
			blocksInX = (DATA_W+threadsInX-1)/threadsInX;
			blocksInY = (DATA_H+threadsInY-1)/threadsInY;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);
	
			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX, threadsInY, threadsInZ * 4);
	
			convolutionRows<<<dimGrid, dimBlock>>>(d_Filter_Response_Rows, d_Permuted_fMRI_Volumes, d_Brain_Voxels, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);
	
			// Convolve columns
				
			threadsInX = 32;
			threadsInY = 8;
			threadsInZ = 2;

			blocksInX = (DATA_W+threadsInX-1)/(threadsInX / 32 * 24);
			blocksInY = (DATA_H+threadsInY-1)/threadsInY / 2;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ / 1;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX / 32 * 24, threadsInY * 2, threadsInZ * 4);

			convolutionColumns<<<dimGrid, dimBlock>>>(d_Filter_Response_Columns, d_Filter_Response_Rows, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);

			// Convolve rods
	
			threadsInX = 32;
			threadsInY = 2;
			threadsInZ = 8;

			blocksInX = (DATA_W+threadsInX-1)/threadsInX;
			blocksInY = (DATA_H+threadsInY-1)/threadsInY / 4;
			blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
			dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
			dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

			Calculate_Block_Differences3D(xBlockDifference, yBlockDifference, zBlockDifference, DATA_W, DATA_H, DATA_D, threadsInX, threadsInY * 4, threadsInZ);

			convolutionRods<<<dimGrid, dimBlock>>>(d_Smoothed_fMRI_Volumes_1, d_Filter_Response_Columns, d_Smoothed_Certainty, t, DATA_W, DATA_H, DATA_D, blocksInY, 1/((float)blocksInY), xBlockDifference + 8, yBlockDifference, zBlockDifference + 8);
		}
    
		// Free all the allocated memory on the graphics card
		cudaFree(d_Filter_Response_Rows);
		cudaFree(d_Filter_Response_Columns);					
	}
}

void WABAACUDA_LIB::PerformDetrendingPermutation()
{
	if ( (ANALYSIS_METHOD == CCA) && (SMOOTHING_DIMENSIONALITY == SMOOTHING_2D) )
	{
		int threadsInX = 32;
		int threadsInY = 8;
		int threadsInZ = 1;

		int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		// Calculate how many time multiples there are
		int timeMultiples = DATA_T / threadsInY;
		int timeRest = DATA_T - timeMultiples * threadsInY;

		// Do the detrending
		DetrendCubic<<<dimGrid, dimBlock>>>(d_Detrended_fMRI_Volumes_1, d_Smoothed_fMRI_Volumes_1, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
		DetrendCubic<<<dimGrid, dimBlock>>>(d_Detrended_fMRI_Volumes_2, d_Smoothed_fMRI_Volumes_2, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
		DetrendCubic<<<dimGrid, dimBlock>>>(d_Detrended_fMRI_Volumes_3, d_Smoothed_fMRI_Volumes_3, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
		DetrendCubic<<<dimGrid, dimBlock>>>(d_Detrended_fMRI_Volumes_4, d_Smoothed_fMRI_Volumes_4, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
	}
	else if ( (ANALYSIS_METHOD == CCA) && (SMOOTHING_DIMENSIONALITY == SMOOTHING_3D) )
	{
		int threadsInX = 32;
		int threadsInY = 8;
		int threadsInZ = 1;

		int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		// Calculate how many time multiples there are
		int timeMultiples = DATA_T / threadsInY;
		int timeRest = DATA_T - timeMultiples * threadsInY;

		// Do the detrending
		DetrendCubic<<<dimGrid, dimBlock>>>(d_Detrended_fMRI_Volumes_1, d_Smoothed_fMRI_Volumes_1, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
		DetrendCubic<<<dimGrid, dimBlock>>>(d_Detrended_fMRI_Volumes_2, d_Smoothed_fMRI_Volumes_2, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
	}
	else if (ANALYSIS_METHOD == GLM)
	{
		int threadsInX = 32;
		int threadsInY = 8;
		int threadsInZ = 1;

		int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		// Calculate how many time multiples there are
		int timeMultiples = DATA_T / threadsInY;
		int timeRest = DATA_T - timeMultiples * threadsInY;

		// Do the detrending
		DetrendCubic<<<dimGrid, dimBlock>>>(d_Detrended_fMRI_Volumes_1, d_Smoothed_fMRI_Volumes_1, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
	}
}

void WABAACUDA_LIB::CalculateActivityMapPermutation()
{
	// CCA 2D
	if ( (ANALYSIS_METHOD == CCA) && (SMOOTHING_DIMENSIONALITY == SMOOTHING_2D) )
	{
		int threadsInX = 32;
		int threadsInY = 4;
		int threadsInZ = 1;

		int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		// Calculate how many time multiples there are
		int timeMultiples = DATA_T / threadsInY;
		int timeRest = DATA_T - timeMultiples * threadsInY;

		CalculateActivityMapCCA2D<<<dimGrid, dimBlock>>>(d_Activity_Volume, d_Detrended_fMRI_Volumes_1, d_Detrended_fMRI_Volumes_2, d_Detrended_fMRI_Volumes_3, d_Detrended_fMRI_Volumes_4, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
	}
	// CCA 3D
	else if ( (ANALYSIS_METHOD == CCA) && (SMOOTHING_DIMENSIONALITY == SMOOTHING_3D) )
	{
		int threadsInX = 32;
		int threadsInY = 4;
		int threadsInZ = 1;

		int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		// Calculate how many time multiples there are
		int timeMultiples = DATA_T / threadsInY;
		int timeRest = DATA_T - timeMultiples * threadsInY;

		CalculateActivityMapCCA3D<<<dimGrid, dimBlock>>>(d_Activity_Volume, d_Detrended_fMRI_Volumes_1, d_Detrended_fMRI_Volumes_2, d_Brain_Voxels, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/((float)blocksInY), timeMultiples, timeRest);
	}
	// GLM
	else if (ANALYSIS_METHOD == GLM)
	{
		int threadsInX = 32;
		int threadsInY = 8;
		int threadsInZ = 1;

		int blocksInX = (DATA_W+threadsInX-1)/threadsInX;
		int blocksInY = (DATA_H+threadsInY-1)/threadsInY;
		int blocksInZ = (DATA_D+threadsInZ-1)/threadsInZ;
		dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
		dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

		// Calculate how many time multiples there are
		int timeMultiples = DATA_T / threadsInY;
		int timeRest = DATA_T - timeMultiples * threadsInY;

		CalculateActivityMapGLM<<<dimGrid, dimBlock>>>(d_Activity_Volume, d_Detrended_fMRI_Volumes_1, d_Brain_Voxels, h_ctxtxc, DATA_W, DATA_H, DATA_D, DATA_T, blocksInY, 1.0f/(float)blocksInY, timeMultiples, timeRest);
	}
}




