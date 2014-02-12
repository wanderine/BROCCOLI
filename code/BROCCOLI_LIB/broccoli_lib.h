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

//#include "nifti1.h"
//#include "nifti1_io.h"

#include <opencl.h>
#include <string>
#include <vector>
#include <Dense>

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short int uint16;

struct float2 {float x; float y;};

// Enumerated constants for axes
enum { X, Y, Z };

// Simple 3D coordinates class
class Coords3D
{
	public:

      // Default constructor
      Coords3D() {}

      // Initializing constructor
      Coords3D(int x,int y,int z)
      {
            coords_[X] = x;
            coords_[Y] = y;
            coords_[Z] = z;
      }

      // Mutable access
      int& operator[](int i)
      { return coords_[i]; }

      // Const access
      int const& operator[](int i) const
      { return coords_[i]; }

private:

      // 3D coordinates
      int coords_[3];
};

// Macro for 26-connected neighbors in 3D
#define Neighbors3D26(n,p) Coords3D const n[26] =\
{\
      Coords3D((p)[X] - 1, (p)[Y] - 1, (p)[Z] - 1),\
      Coords3D((p)[X]    , (p)[Y] - 1, (p)[Z] - 1),\
      Coords3D((p)[X] + 1, (p)[Y] - 1, (p)[Z] - 1),\
      Coords3D((p)[X] - 1, (p)[Y]    , (p)[Z] - 1),\
      Coords3D((p)[X]    , (p)[Y]    , (p)[Z] - 1),\
      Coords3D((p)[X] + 1, (p)[Y]    , (p)[Z] - 1),\
      Coords3D((p)[X] - 1, (p)[Y] + 1, (p)[Z] - 1),\
      Coords3D((p)[X]    , (p)[Y] + 1, (p)[Z] - 1),\
      Coords3D((p)[X] + 1, (p)[Y] + 1, (p)[Z] - 1),\
      Coords3D((p)[X] - 1, (p)[Y] - 1, (p)[Z]    ),\
      Coords3D((p)[X]    , (p)[Y] - 1, (p)[Z]    ),\
      Coords3D((p)[X] + 1, (p)[Y] - 1, (p)[Z]    ),\
      Coords3D((p)[X] - 1, (p)[Y]    , (p)[Z]    ),\
      Coords3D((p)[X] + 1, (p)[Y]    , (p)[Z]    ),\
      Coords3D((p)[X] - 1, (p)[Y] + 1, (p)[Z]    ),\
      Coords3D((p)[X]    , (p)[Y] + 1, (p)[Z]    ),\
      Coords3D((p)[X] + 1, (p)[Y] + 1, (p)[Z]    ),\
      Coords3D((p)[X] - 1, (p)[Y] - 1, (p)[Z] + 1),\
      Coords3D((p)[X]    , (p)[Y] - 1, (p)[Z] + 1),\
      Coords3D((p)[X] + 1, (p)[Y] - 1, (p)[Z] + 1),\
      Coords3D((p)[X] - 1, (p)[Y]    , (p)[Z] + 1),\
      Coords3D((p)[X]    , (p)[Y]    , (p)[Z] + 1),\
      Coords3D((p)[X] + 1, (p)[Y]    , (p)[Z] + 1),\
      Coords3D((p)[X] - 1, (p)[Y] + 1, (p)[Z] + 1),\
      Coords3D((p)[X]    , (p)[Y] + 1, (p)[Z] + 1),\
      Coords3D((p)[X] + 1, (p)[Y] + 1, (p)[Z] + 1),\
}



class BROCCOLI_LIB
{
	public:
		// Constructor & destructor
		BROCCOLI_LIB();
		BROCCOLI_LIB(cl_uint platform, cl_uint device);
		~BROCCOLI_LIB();

		// Set functions for GUI / Wrappers

		// void SetOpenCLPlatform(int N);

		void SetDebug(bool debug);

		void SetMask(float* input);
		void SetEPIMask(float* input);
		void SetSmoothedEPIMask(float* input);

		// Statistics
		void SetTemporalDerivatives(int TD);
		void SetRegressMotion(int R);
		void SetRegressConfounds(int R);
		void SetConfoundRegressors(float* X_GLM);
		void SetNumberOfGLMRegressors(int NR);
		void SetNumberOfDetrendingRegressors(int NR);
		void SetNumberOfConfoundRegressors(int NR);
		void SetNumberOfContrasts(int NC);
		void SetDesignMatrix(float* X_GLM, float* xtxxt_GLM);
		void SetContrasts(float* contrasts);
		void SetGLMScalars(float* ctxtxc);
		void SetNumberOfPermutations(int);
		void SetBetaSpace(int space);
		void SetStatisticalTest(int test);
		void SetInferenceMode(int mode);
		void SetClusterDefiningThreshold(float threshold);
		void SetPermutationMatrix(unsigned short int*);


		// Smoothing
		void SetSmoothingFilters(float* smoothing_filter_x,float* smoothing_filter_y,float* smoothing_filter_z);
		void SetSmoothingType(int);
		void SetEPISmoothingAmount(float);
		void SetARSmoothingAmount(float);

		// Image registration
		void SetImageRegistrationFilterSize(int N);
		void SetParametricImageRegistrationFilters(float* qf1r, float* qf1i, float* qf2r, float* qf2i, float* qf3r, float* qf3i);
		void SetNonParametricImageRegistrationFilters(float* qf1r, float* qf1i, float* qf2r, float* qf2i, float* q3r, float* q3i, float* qf4r, float* qf4i, float* qf5r, float* qf5i, float* q6r, float* q6i);
		void SetProjectionTensorMatrixFirstFilter(float m11, float m12, float m13, float m22, float m23, float m33);
		void SetProjectionTensorMatrixSecondFilter(float m11, float m12, float m13, float m22, float m23, float m33);
		void SetProjectionTensorMatrixThirdFilter(float m11, float m12, float m13, float m22, float m23, float m33);
		void SetProjectionTensorMatrixFourthFilter(float m11, float m12, float m13, float m22, float m23, float m33);
		void SetProjectionTensorMatrixFifthFilter(float m11, float m12, float m13, float m22, float m23, float m33);
		void SetProjectionTensorMatrixSixthFilter(float m11, float m12, float m13, float m22, float m23, float m33);
		void SetFilterDirections(float* x, float* y, float* z);
		void SetNumberOfIterationsForParametricImageRegistration(int N);
		void SetNumberOfIterationsForNonParametricImageRegistration(int N);
		void SetNumberOfIterationsForMotionCorrection(int N);
		void SetCoarsestScaleT1MNI(int N);
		void SetCoarsestScaleEPIT1(int N);
		void SetMMT1ZCUT(int mm);
		void SetMMEPIZCUT(int mm);
		void SetInterpolationMode(int mode);

		// EPI data
		void SetEPIVoxelSizeX(float value);
		void SetEPIVoxelSizeY(float value);
		void SetEPIVoxelSizeZ(float value);
		void SetEPITR(float value);

		void SetEPIWidth(int w);
		void SetEPIHeight(int h);
		void SetEPIDepth(int d);
		void SetEPITimepoints(int t);

		// T1 data
		void SetT1VoxelSizeX(float value);
		void SetT1VoxelSizeY(float value);
		void SetT1VoxelSizeZ(float value);

		void SetT1Width(int w);
		void SetT1Height(int h);
		void SetT1Depth(int d);

		// MNI data
		void SetMNIVoxelSizeX(float value);
		void SetMNIVoxelSizeY(float value);
		void SetMNIVoxelSizeZ(float value);

		void SetMNIWidth(int w);
		void SetMNIHeight(int h);
		void SetMNIDepth(int d);

		// Input data
		void SetInputfMRIVolumes(float* input);
		void SetInputEPIVolume(float* input);
		void SetInputT1Volume(float* input);
		void SetInputMNIVolume(float* input);
		void SetInputMNIBrainVolume(float* input);
		void SetInputMNIBrainMask(float* input);
		void SetInputFirstLevelResults(float* input);
		void SetNumberOfSubjects(int N);

		// Output statistics
		void SetOutputBetaVolumes(float* output);
		void SetOutputResiduals(float* output);
		void SetOutputResidualVariances(float* output);
		void SetOutputStatisticalMaps(float* output);
		void SetOutputEPIMask(float*);
		void SetOutputClusterIndices(int*);
		void SetOutputLargestCluster(int*);
		void SetOutputDesignMatrix(float* X_GLM, float* xtxxt_GLM);
		void SetOutputWhitenedModels(float* whitened_models);

		// Output image registration
		void SetOutputMotionParameters(float* output);
		void SetOutputT1MNIRegistrationParameters(float* output);
		void SetOutputEPIT1RegistrationParameters(float* output);
		void SetOutputEPIMNIRegistrationParameters(float* output);
		void SetOutputQuadratureFilterResponses(cl_float2* qfr1, cl_float2* qfr2, cl_float2* qfr3);
		void SetOutputQuadratureFilterResponses(cl_float2* qfr1, cl_float2* qfr2, cl_float2* qfr3, cl_float2* qfr4, cl_float2* qfr5, cl_float2* qfr6);
		void SetOutputTensorComponents(float*, float*, float*,float*, float*, float*);
		void SetOutputDisplacementField(float*, float*, float*);
		void SetOutputPhaseDifferences(float*);
		void SetOutputPhaseCertainties(float*);
		void SetOutputPhaseGradients(float*);
		void SetOutputAlignedT1Volume(float*);
		void SetOutputAlignedT1VolumeNonParametric(float*);
		void SetOutputAlignedEPIVolume(float*);
		void SetOutputSkullstrippedT1Volume(float*);
		void SetOutputInterpolatedT1Volume(float*);
		void SetOutputInterpolatedEPIVolume(float*);
		void SetOutputDownsampledVolume(float*);
		void SetOutputMotionCorrectedfMRIVolumes(float*);
		void SetOutputSliceTimingCorrectedfMRIVolumes(float*);

		void SetOutputSmoothedfMRIVolumes(float*);
		void SetOutputDetrendedfMRIVolumes(float*);
		void SetOutputWhitenedfMRIVolumes(float*);
		void SetOutputPermutedfMRIVolumes(float*);
		void SetOutputPermutedFirstLevelResults(float*);
		void SetOutputAREstimates(float*, float*, float*, float*);
		void SetOutputSliceSums(float*);
		void SetOutputTopSlice(float*);
		void SetOutputPermutationDistribution(float*);
		void SetOutputAMatrix(float*);
		void SetOutputHVector(float*);

                /**
                 * In oldstuff

		// void SetfMRIDataFilename(std::string filename);

		// void SetfMRIDataSliceLocationX(int location);
		// void SetfMRIDataSliceLocationY(int location);
		// void SetfMRIDataSliceLocationZ(int location);
		// void SetfMRIDataSliceTimepoint(int timepoint);

		// void SetDataType(int type);
		// void SetFileType(int type);

		*/

		void SetSignificanceLevel(float value);

		// Get functions for GUI / Wrappers

		int GetOpenCLInitiated();
		int GetNumberOfOpenCLKernels();

                /**
                 * In oldstuff
                 *
		// int GetfMRIDataSliceLocationX();
		// int GetfMRIDataSliceLocationY();
		// int GetfMRIDataSliceLocationZ();
		*/

		// EPI data
		float GetEPIVoxelSizeX();
		float GetEPIVoxelSizeY();
		float GetEPIVoxelSizeZ();
		float GetEPITR();

		int GetEPIWidth();
		int GetEPIHeight();
		int GetEPIDepth();
		int GetEPITimepoints();

		// T1 data
		int GetT1Width();
		int GetT1Height();
		int GetT1Depth();
		// int GetT1Timepoints();

                /**
                 * Nowhere

		// float GetT1VoxelSizeX();
		// float GetT1VoxelSizeY();
		// float GetT1VoxelSizeZ();
		*/

		float GetSignificanceThreshold();
		// std::string GetfMRIDataFilename();

		int GetNumberOfSignificantlyActiveVoxels();
		int GetNumberOfSignificantlyActiveClusters();

		// OpenCL

		const char* GetOpenCLDeviceInfoChar();
		const char* GetOpenCLBuildInfoChar();
		const char* GetOpenCLErrorMessage(int error);

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
		// void ReadfMRIDataRAW();
		// void ReadfMRIDataNIFTI();
		// void ReadNIFTIHeader();

		// Write functions
		// void WritefMRIDataNIFTI();

		// Wrappers
		void ChangeT1VolumeResolutionAndSizeWrapper();
		void PerformRegistrationT1MNIWrapper();
		void PerformRegistrationT1MNINoSkullstripWrapper();
		void PerformRegistrationTwoVolumesWrapper();
		void TransformVolumesNonParametricWrapper();
		void PerformRegistrationEPIT1Wrapper();
		void PerformSliceTimingCorrectionWrapper();
		void PerformMotionCorrectionWrapper();
		void PerformMotionCorrectionWrapperSeveralScales();
		void PerformDetrending(cl_mem, cl_mem, int, int, int, int);
		void PerformDetrendingAndMotionRegression(cl_mem, cl_mem, int, int, int, int);
		void PerformSmoothingWrapper();
		void PerformSmoothingNormalizedWrapper();
		void PerformGLMTTestFirstLevelWrapper();
		void PerformGLMFTestFirstLevelWrapper();
		void PerformGLMTTestSecondLevelWrapper();
		void PerformGLMFTestSecondLevelWrapper();
		void PerformGLMTTestFirstLevelPermutationWrapper();
		void PerformGLMFTestFirstLevelPermutationWrapper();
		void PerformGLMTTestSecondLevelPermutationWrapper();
		void PerformGLMFTestSecondLevelPermutationWrapper();
		void PerformBayesianFirstLevelWrapper();
		void PerformFirstLevelAnalysisWrapper();
		void PerformFirstLevelAnalysisBayesianWrapper();
		void PerformSecondLevelAnalysisWrapper();

		void ClusterizeOpenCLWrapper();
		void ClusterizeOpenCLWrapper2();
		void ClusterizeOpenCLWrapper3();

		// void CalculateSlicesfMRIData();
		// void CalculateSlicesPreprocessedfMRIData();
		// void CalculateSlicesActivityData();

		void GetOpenCLInfo();
		void OpenCLInitiate(cl_uint OPENCL_PLATFORM, cl_uint OPENCL_DEVICE);

	private:

		void CreateCombinedDisplacementField(float* h_Registration_Parameters, cl_mem d_Displacement_Field_X, cl_mem d_Displacement_Field_Y, cl_mem d_Displacement_Field_Z, int DATA_W, int DATA_H, int DATA_D);

		int Calculate3DIndex(int x, int y, int z, int DATA_W, int DATA_H);
		void Clusterize(int* Cluster_Indices, int& MAX_CLUSTER_SIZE, float& MAX_CLUSTER_MASS, int& NUMBER_OF_CLUSTERS, float* Data, float Threshold, float* Mask, int DATA_W, int DATA_H, int DATA_D, int GET_VOXEL_LABELS, int GET_CLUSTER_MASS);
		void ClusterizeOpenCL(cl_mem Cluster_Indices, int& MAX_CLUSTER_SIZE, cl_mem Data, float Threshold, cl_mem Mask, int DATA_W, int DATA_H, int DATA_D);
		void ClusterizeOpenCLPermutation(int& MAX_CLUSTER_SIZE, int DATA_W, int DATA_H, int DATA_D);

		//------------------------------------------------
		// High level functions
		//------------------------------------------------
		void PerformRegistrationEPIT1();
		void PerformRegistrationT1MNI();
		void PerformRegistrationT1MNINoSkullstrip();
		void SegmentEPIData();
		void SegmentEPIData(cl_mem Volume);
		void PerformSliceTimingCorrection();
		void PerformMotionCorrection();
		void PerformFirstLevelAnalysis();

		void CalculateStatisticalMapsGLMTTestFirstLevel(cl_mem Volumes);
		void CalculateStatisticalMapsGLMFTestFirstLevel(cl_mem Volumes);
		void CalculateStatisticalMapsGLMTTestSecondLevel(cl_mem Volumes, cl_mem Mask);
		void CalculateStatisticalMapsGLMFTestSecondLevel(cl_mem Volumes, cl_mem Mask);

		void CalculateStatisticalMapsGLMBayesianFirstLevel(cl_mem d_Volumes);

		void CreateVoxelNumbers(cl_mem d_Voxel_Numbers, cl_mem d_Mask, int DATA_W, int DATA_H, int DATA_D);

		void WhitenDesignMatricesInverse(cl_mem d_xtxxt_GLM, float* h_X_GLM, cl_mem d_AR1_Estimates, cl_mem d_AR2_Estimates, cl_mem d_AR3_Estimates, cl_mem d_AR4_Estimates, cl_mem d_Mask, int DATA_W, int DATA_H, int DATA_D, int DATA_T, int NUMBER_OF_REGRESSORS, int NUMBER_OF_INVALID_TIMEPOINTS);
		void WhitenDesignMatricesTTest(cl_mem d_xtxxt_GLM, cl_mem d_GLM_Scalars, float* h_X_GLM, float* h_Contrasts, cl_mem d_AR1_Estimates, cl_mem d_AR2_Estimates, cl_mem d_AR3_Estimates, cl_mem d_AR4_Estimates, cl_mem d_EPI_Mask, int EPI_DATA_W, int EPI_DATA_H, int EPI_DATA_D, int EPI_DATA_T, int NUMBER_OF_GLM_REGRESSORS, int NUMBER_OF_INVALID_TIMEPOINTS, int NUMBER_OF_CONTRASTS);
		void WhitenDesignMatricesFTest(cl_mem d_xtxxt_GLM, cl_mem d_GLM_Scalars, float* h_X_GLM, float* h_Contrasts, cl_mem d_AR1_Estimates, cl_mem d_AR2_Estimates, cl_mem d_AR3_Estimates, cl_mem d_AR4_Estimates, cl_mem d_EPI_Mask, int EPI_DATA_W, int EPI_DATA_H, int EPI_DATA_D, int EPI_DATA_T, int NUMBER_OF_GLM_REGRESSORS, int NUMBER_OF_INVALID_TIMEPOINTS, int NUMBER_OF_CONTRASTS);
		void PutWhitenedModelsIntoVolumes(cl_mem d_Mask, cl_mem d_xtxxt_GLM, int DATA_W, int DATA_H, int DATA_D, int DATA_T, int NUMBER_OF_REGRESSORS);
		void PutWhitenedModelsIntoVolumes2(cl_mem d_Mask, cl_mem d_AR1_Estimates, cl_mem d_AR2_Estimates, cl_mem d_AR3_Estimates, cl_mem d_AR4_Estimates, float* Regressors, int DATA_W, int DATA_H, int DATA_D, int DATA_T, int NUMBER_OF_REGRESSORS);

		void ApplyPermutationTestFirstLevel(cl_mem Volumes);
		void ApplyPermutationTestSecondLevel();

		// Permutation first level
		void SetupPermutationTestFirstLevel();
		void CleanupPermutationTestFirstLevel();
		void GeneratePermutationMatrixFirstLevel();
		void PerformDetrendingPriorPermutation();
		void PerformWhiteningPriorPermutations(cl_mem Whitened_volumes, cl_mem Volumes);
		void GeneratePermutedVolumesFirstLevel(cl_mem Permuted_Volumes, cl_mem Whitened_Volumes, int permutation);
		void CalculateStatisticalMapsFirstLevelPermutation();
		void CalculateStatisticalMapsGLMTTestFirstLevelPermutation();
		void CalculateStatisticalMapsGLMFTestFirstLevelPermutation();

		// Permutation second level
		void SetupPermutationTestSecondLevel(cl_mem Volumes, cl_mem Mask);
		void CleanupPermutationTestSecondLevel();
		void GeneratePermutationMatrixSecondLevel();
		void CalculateStatisticalMapsSecondLevelPermutation(int permutation);
		void CalculateStatisticalMapsGLMTTestSecondLevelPermutation();
		void CalculateStatisticalMapsGLMFTestSecondLevelPermutation();

		//------------------------------------------------
		// Convolution functions
		//------------------------------------------------
		void CopyThreeQuadratureFiltersToConstantMemory(cl_mem c_Quadrature_Filter_1_Real, cl_mem c_Quadrature_Filter_1_Imag, cl_mem c_Quadrature_Filter_2_Real, cl_mem c_Quadrature_Filter_2_Imag, cl_mem c_Quadrature_Filter_3_Real, cl_mem c_Quadrature_Filter_3_Imag, float* h_Quadrature_Filter_1_Real, float* h_Quadrature_Filter_1_Imag, float* h_Quadrature_Filter_2_Real, float* h_Quadrature_Filter_2_Imag, float* h_Quadrature_Filter_3_Real, float* Quadrature_h_Filter_3_Imag, int z, int FILTER_SIZE);
		void NonseparableConvolution3D(cl_mem d_q1, cl_mem d_q2, cl_mem d_q3, cl_mem d_Volume, cl_mem c_Filter_1_Real, cl_mem c_Filter_1_Imag, cl_mem c_Filter_2_Real, cl_mem c_Filter_2_Imag, cl_mem c_Filter_3_Real, cl_mem c_Filter_3_Imag, float* h_Filter_1_Real, float* h_Filter_1_Imag, float* h_Filter_2_Real, float* h_Filter_2_Imag, float* h_Filter_3_Real, float* h_Filter_3_Imag, int DATA_W, int DATA_H, int DATA_D);
		void PerformSmoothing(cl_mem Smoothed_Volumes, cl_mem d_Volumes, float* h_Smoothing_Filter_X, float* h_Smoothing_Filter_Y, float* h_Smoothing_Filter_Z, int DATA_W, int DATA_H, int DATA_D, int DATA_T);
		void PerformSmoothingNormalized(cl_mem Smoothed_Volumes, cl_mem d_Volumes, cl_mem d_Certainty, cl_mem d_Smoothed_Certainty, float* h_Smoothing_Filter_X, float* h_Smoothing_Filter_Y, float* h_Smoothing_Filter_Z, int DATA_W, int DATA_H, int DATA_D, int DATA_T);
		void PerformSmoothing(cl_mem d_Volumes, float* h_Smoothing_Filter_X, float* h_Smoothing_Filter_Y, float* h_Smoothing_Filter_Z, int DATA_W, int DATA_H, int DATA_D, int DATA_T);
		void PerformSmoothingNormalized(cl_mem d_Volumes, cl_mem d_Certainty, cl_mem d_Smoothed_Certainty, float* h_Smoothing_Filter_X, float* h_Smoothing_Filter_Y, float* h_Smoothing_Filter_Z, int DATA_W, int DATA_H, int DATA_D, int DATA_T);
		void PerformSmoothingNormalizedPermutation();

		//------------------------------------------------
		// Functions for image registration
		//------------------------------------------------
		void AlignTwoVolumesParametricSetup(int DATA_W, int DATA_H, int DATA_D);
		void AlignTwoVolumesParametric(float* h_Registration_Parameters, float* h_Rotations, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_ITERATIONS, int ALIGNMENT_TYPE, int INTERPOLATION_MODE);
		void AlignTwoVolumesParametricSeveralScales(float *h_Registration_Parameters, float* h_Rotations, cl_mem d_Al_Volume, cl_mem d_Ref_Volume, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_SCALES, int NUMBER_OF_ITERATIONS, int ALIGNMENT_TYPE, int OVERWRITE, int INTERPOLATION_MODE);
		void AlignTwoVolumesParametricCleanup();

		void AlignTwoVolumesNonParametricSetup(int DATA_W, int DATA_H, int DATA_D);
		void AlignTwoVolumesNonParametric(int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_ITERATIONS, int INTERPOLATION_MODE);
		void AlignTwoVolumesNonParametricSeveralScales(cl_mem d_Al_Volume, cl_mem d_Ref_Volume, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_SCALES, int NUMBER_OF_ITERATIONS, int OVERWRITE, int INTERPOLATION_MODE, int SAVE_DISPLACEMENT_FIELD);
		void AlignTwoVolumesNonParametricCleanup();

		void ChangeT1VolumeResolutionAndSize(cl_mem d_MNI_T1_Volume, cl_mem d_T1_Volume, int T1_DATA_W, int T1_DATA_H, int T1_DATA_D, int MNI_DATA_W, int MNI_DATA_H, int MNI_DATA_D, float T1_VOXEL_SIZE_X, float T1_VOXEL_SIZE_Y, float T1_VOXEL_SIZE_Z, float MNI_VOXEL_SIZE_X, float MNI_VOXEL_SIZE_Y, float MNI_VOXEL_SIZE_Z, int INTERPOLATION_MODE, int MNI_WITH_SKULL);
		void ChangeEPIVolumeResolutionAndSize(cl_mem d_T1_EPI_Volume, cl_mem d_EPI_Volume, int EPI_DATA_W, int EPI_DATA_H, int EPI_DATA_D, int T1_DATA_W, int T1_DATA_H, int T1_DATA_D, float EPI_VOXEL_SIZE_X, float EPI_VOXEL_SIZE_Y, float EPI_VOXEL_SIZE_Z, float T1_VOXEL_SIZE_X, float T1_VOXEL_SIZE_Y, float T1_VOXEL_SIZE_Z, int INTERPOLATION_MODE);
		void ChangeVolumeSize(cl_mem d_New_Volume, cl_mem d_Volume, int DATA_W, int DATA_H, int DATA_D, int CURRENT_DATA_W, int CURRENT_DATA_H, int CURRENT_DATA_D, int INTERPOLATION_MODE);
		void ChangeVolumeSize(cl_mem& d_Volume, int DATA_W, int DATA_H, int DATA_D, int CURRENT_DATA_W, int CURRENT_DATA_H, int CURRENT_DATA_D, int INTERPOLATION_MODE);
		void ChangeVolumesResolutionAndSize(cl_mem d_New_Volumes, cl_mem d_Volumes, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_VOLUMES, int NEW_DATA_W, int NEW_DATA_H, int NEW_DATA_D, float VOXEL_SIZE_X, float VOXEL_SIZE_Y, float VOXEL_SIZE_Z, float NEW_VOXEL_SIZE_X, float NEW_VOXEL_SIZE_Y, float NEW_VOXEL_SIZE_Z, int MM_Z_CUT, int INTERPOLATION_MODE);
		void CalculateTensorMagnitude(cl_mem d_Tensor_Magnitudes, cl_mem d_Volume, int DATA_W, int DATA_H, int DATA_D);

		void TransformVolumesParametric(cl_mem d_Volumes, float* h_Registration_Parameters, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_VOLUMES, int INTERPOLATION_MODE);
		void TransformVolumesNonParametric(cl_mem d_Volumes, cl_mem d_Displacement_Field_X, cl_mem d_Displacement_Field_Y, cl_mem d_Displacement_Field_Z, int DATA_W, int DATA_H, int DATA_D, int NUMBER_OF_VOLUMES, int INTERPOLATION_MODE);
		void TransformFirstLevelResultsToMNI();

		//------------------------------------------------
		// Help functions
		//------------------------------------------------

		void SetMemory(cl_mem memory, float value, int N);
		void SetMemoryInt(cl_mem memory, int value, int N);
		void SetMemoryFloat2(cl_mem memory, float value, int N);
		void CalculateTopBrainSlice(int& slice, cl_mem d_Volume, int DATA_W, int DATA_H, int DATA_D, int z_cut);
		void MultiplyVolume(cl_mem d_Volume_1, float value, int DATA_W, int DATA_H, int DATA_D);
		void MultiplyVolumes(cl_mem d_Volume_1, cl_mem d_Volume_2, int DATA_W, int DATA_H, int DATA_D);
		void MultiplyVolumes(cl_mem d_Volume_1, cl_mem d_Volume_2, int DATA_W, int DATA_H, int DATA_D, int VOLUMES);
		void MultiplyVolumes(cl_mem d_Result, cl_mem d_Volume_1, cl_mem d_Volume_2, int DATA_W, int DATA_H, int DATA_D);
		void AddVolume(cl_mem d_Volume, float value, int DATA_W, int DATA_H, int DATA_D);
		void AddVolumes(cl_mem d_Volume_1, cl_mem d_Volume_2, int DATA_W, int DATA_H, int DATA_D);
		void AddVolumes(cl_mem d_Result, cl_mem d_Volume_1, cl_mem d_Volume_2, int DATA_W, int DATA_H, int DATA_D);
		float CalculateSum(cl_mem Volume, int DATA_W, int DATA_H, int DATA_D);
		float CalculateMax(cl_mem Volume, int DATA_W, int DATA_H, int DATA_D);

		float CalculateMaxAtomic(cl_mem Volume, cl_mem Mask, int DATA_W, int DATA_H, int DATA_D);
		float CalculateMax(float *data, int N);
		int   CalculateMax(int *data, int N);
		float CalculateMin(float *data, int N);
		void ThresholdVolume(cl_mem d_Thresholded_Volume, cl_mem d_Volume, float threshold, int DATA_W, int DATA_H, int DATA_D);


		cl_int CreateProgramFromBinary(cl_program& program, cl_context context, cl_device_id device, std::string filename);
		bool SaveProgramBinary(cl_program program, cl_device_id device, std::string filename);
		void CreateSmoothingFilters(float* Smoothing_Filter_X, float* Smoothing_Filter_Y, float* Smoothing_Filter_Z, int size, float smoothing_FWHM, float voxel_size_x, float voxel_size_y, float voxel_size_z);
		void CreateSmoothingFilters(float* Smoothing_Filter_X, float* Smoothing_Filter_Y, float* Smoothing_Filter_Z, int size, double sigma);
		void SolveEquationSystem(float* h_Parameter_Vector, float* h_A_matrix, float* h_h_vector, int N);

		void InvertAffineRegistrationParameters(float* h_Inverse_Parameters, float* h_Parameters);
		void AddAffineRegistrationParameters(float* h_Old_Parameters, float* h_New_Parameters);
		void AddAffineRegistrationParametersNextScale(float* h_Old_Parameters, float* h_New_Parameters);
		void AddAffineRegistrationParameters(float* h_Resulting_Parameters, float* h_New_Parameters, float* h_Old_Parameters);
		void CalculateRotationAnglesFromRotationMatrix(float* h_Rotations, float* h_Registration_Parameters);
		void RemoveTransformationScaling(float* h_Registration_Parameters);

		void SetupDetrendingRegressors(int N);
		void SetupDetrendingAndMotionRegressors(int N);
		Eigen::MatrixXd SetupGLMRegressorsFirstLevel(int N);
		void SetupTTestFirstLevel(int N);
		void SetupFTestFirstLevel(int N);
		void DemeanRegressor(float* Regressor, int N);
		void DemeanRegressor(Eigen::VectorXd &Regressor, int N);


		float Gpdf(double value, double shape, double scale);
		double Gamma(double x);
		double LogGamma(double x);

		void CreateHRF();
		void ConvolveRegressorsWithHRF(float* Convolved_Regressors, float* Regressors, int NUMBER_OF_TIMEPOINTS, int NUMBER_OF_REGRESSORS);
		void GenerateRegressorTemporalDerivatives(float * Regressors_With_Temporal_Derivatives, float* Regressors, int NUMBER_OF_TIMEPOINTS, int NUMBER_OF_REGRESSORS);

		//------------------------------------------------
		// Set functions
		//------------------------------------------------

		void SetGlobalAndLocalWorkSizesSeparableConvolution(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesNonSeparableConvolution(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesImageRegistration(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesStatisticalCalculations(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesInterpolateVolume(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesCopyVolumeToNew(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesMemset(int N);
		void SetGlobalAndLocalWorkSizesMultiplyVolumes(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesAddVolumes(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesCalculateSum(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesCalculateMax(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesThresholdVolume(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesCalculateMagnitudes(int DATA_W, int DATA_H, int DATA_D);
		void SetGlobalAndLocalWorkSizesClusterize(int DATA_W, int DATA_H, int DATA_D);



		//------------------------------------------------
		// OpenCL help functions
		//------------------------------------------------

		void OpenCLCleanup();
		void SetStartValues();

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

		int NUMBER_OF_OPENCL_KERNELS;
		cl_kernel OpenCLKernels[200];

		// OpenCL kernels

		// Help kernels
		cl_kernel MemsetKernel;
		cl_kernel MemsetIntKernel;
		cl_kernel MemsetFloat2Kernel;
		cl_kernel MultiplyVolumeKernel, MultiplyVolumesKernel, MultiplyVolumesOverwriteKernel;
		cl_kernel AddVolumeKernel, AddVolumesKernel, AddVolumesOverwriteKernel;
		cl_kernel CalculateColumnSumsKernel, CalculateRowSumsKernel;
		cl_kernel CalculateColumnMaxsKernel, CalculateRowMaxsKernel;
		cl_kernel CalculateMaxAtomicKernel;
		cl_kernel ThresholdVolumeKernel;
		cl_kernel RemoveMeanKernel;
		cl_kernel SetStartClusterIndicesKernel;
		cl_kernel ClusterizeScanKernel;
		cl_kernel ClusterizeRelabelKernel;
		cl_kernel CalculateClusterSizesKernel;
		cl_kernel CalculateLargestClusterKernel;

		// Convolution kernels
		cl_kernel SeparableConvolutionRowsKernel, SeparableConvolutionColumnsKernel, SeparableConvolutionRodsKernel;
		cl_kernel NonseparableConvolution3DComplexThreeFiltersKernel;

		cl_kernel SliceTimingCorrectionKernel;

		// Image registration kernels
		cl_kernel CalculatePhaseDifferencesAndCertaintiesKernel, CalculatePhaseGradientsXKernel, CalculatePhaseGradientsYKernel, CalculatePhaseGradientsZKernel;
		cl_kernel CalculateAMatrixAndHVector2DValuesXKernel, CalculateAMatrixAndHVector2DValuesYKernel,CalculateAMatrixAndHVector2DValuesZKernel;
		cl_kernel CalculateAMatrix1DValuesKernel, CalculateHVector1DValuesKernel, CalculateHVectorKernel, ResetAMatrixKernel, CalculateAMatrixKernel;
		cl_kernel InterpolateVolumeNearestParametricKernel, InterpolateVolumeLinearParametricKernel, InterpolateVolumeCubicParametricKernel;
		cl_kernel InterpolateVolumeNearestNonParametricKernel, InterpolateVolumeLinearNonParametricKernel, InterpolateVolumeCubicNonParametricKernel;
		cl_kernel RescaleVolumeNearestKernel, RescaleVolumeLinearKernel, RescaleVolumeCubicKernel;
		cl_kernel CopyT1VolumeToMNIKernel, CopyEPIVolumeToT1Kernel, CopyVolumeToNewKernel;
		cl_kernel CalculateMagnitudesKernel;
		cl_kernel CalculateTensorComponentsKernel;
		cl_kernel CalculateTensorNormsKernel;
		cl_kernel CalculateAMatricesAndHVectorsKernel;
		cl_kernel CalculateDisplacementUpdateKernel;
		cl_kernel AddParametricAndNonParametricDisplacementKernel;

		// Statistical kernels
		cl_kernel CalculateBetaWeightsGLMKernel, CalculateBetaWeightsGLMFirstLevelKernel;
		cl_kernel CalculateGLMResidualsKernel;
		cl_kernel CalculateStatisticalMapsGLMTTestFirstLevelKernel, CalculateStatisticalMapsGLMFTestFirstLevelKernel;
		cl_kernel CalculateStatisticalMapsGLMTTestKernel, CalculateStatisticalMapsGLMFTestKernel, CalculateStatisticalMapsGLMBayesianKernel;
		cl_kernel CalculateStatisticalMapsGLMTTestFirstLevelPermutationKernel,CalculateStatisticalMapsGLMFTestFirstLevelPermutationKernel;
		cl_kernel CalculateStatisticalMapsGLMTTestSecondLevelPermutationKernel,CalculateStatisticalMapsGLMFTestSecondLevelPermutationKernel;
		cl_kernel RemoveLinearFitKernel;
		cl_kernel EstimateAR4ModelsKernel, ApplyWhiteningAR4Kernel, GeneratePermutedVolumesFirstLevelKernel;

		// Create kernel errors

		// Help kernels
		cl_int createKernelErrorMemset, createKernelErrorMemsetInt, createKernelErrorMemsetFloat2;
		cl_int createKernelErrorMultiplyVolume;
		cl_int createKernelErrorMultiplyVolumes;
		cl_int createKernelErrorMultiplyVolumesOverwrite;
		cl_int createKernelErrorAddVolume;
		cl_int createKernelErrorAddVolumes;
		cl_int createKernelErrorAddVolumesOverwrite;
		cl_int createKernelErrorRemoveMean;
		cl_int createKernelErrorSetStartClusterIndices;
		cl_int createKernelErrorClusterizeScan;
		cl_int createKernelErrorClusterizeRelabel;
		cl_int createKernelErrorCalculateClusterSizes;
		cl_int createKernelErrorCalculateLargestCluster;


		// Convolution kernels
		cl_int createKernelErrorSeparableConvolutionRows, createKernelErrorSeparableConvolutionColumns, createKernelErrorSeparableConvolutionRods;
		cl_int createKernelErrorNonseparableConvolution3DComplexThreeFilters;
		cl_int createKernelErrorCalculateColumnSums;
		cl_int createKernelErrorCalculateRowSums;
		cl_int createKernelErrorCalculateColumnMaxs;
		cl_int createKernelErrorCalculateRowMaxs;
		cl_int createKernelErrorCalculateMaxAtomic;
		cl_int createKernelErrorThresholdVolume;

		cl_int createKernelErrorSliceTimingCorrection;

		// Image registration kernels
		cl_int createKernelErrorCalculatePhaseDifferencesAndCertainties, createKernelErrorCalculatePhaseGradientsX, createKernelErrorCalculatePhaseGradientsY, createKernelErrorCalculatePhaseGradientsZ;
		cl_int createKernelErrorCalculateAMatrixAndHVector2DValuesX, createKernelErrorCalculateAMatrixAndHVector2DValuesY, createKernelErrorCalculateAMatrixAndHVector2DValuesZ;
		cl_int createKernelErrorCalculateAMatrix1DValues, createKernelErrorCalculateHVector1DValues;
		cl_int createKernelErrorCalculateAMatrix, createKernelErrorCalculateHVector;
		cl_int createKernelErrorInterpolateVolumeNearestParametric, createKernelErrorInterpolateVolumeLinearParametric,  createKernelErrorInterpolateVolumeCubicParametric;
		cl_int createKernelErrorInterpolateVolumeNearestNonParametric, createKernelErrorInterpolateVolumeLinearNonParametric,  createKernelErrorInterpolateVolumeCubicNonParametric;
		cl_int createKernelErrorRescaleVolumeNearest, createKernelErrorRescaleVolumeLinear, createKernelErrorRescaleVolumeCubic;
		cl_int createKernelErrorCopyT1VolumeToMNI, createKernelErrorCopyEPIVolumeToT1, createKernelErrorCopyVolumeToNew;
		cl_int createKernelErrorCalculateMagnitudes;
		cl_int createKernelErrorCalculateTensorComponents;
		cl_int createKernelErrorCalculateTensorNorms;
		cl_int createKernelErrorCalculateAMatricesAndHVectors;
		cl_int createKernelErrorCalculateDisplacementUpdate;
		cl_int createKernelErrorAddParametricAndNonParametricDisplacement;

		// Statistical kernels
		cl_int createKernelErrorCalculateBetaWeightsGLM,  createKernelErrorCalculateBetaWeightsGLMFirstLevel;
		cl_int createKernelErrorCalculateGLMResiduals;
		cl_int createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel, createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevel;
		cl_int createKernelErrorCalculateStatisticalMapsGLMTTest, createKernelErrorCalculateStatisticalMapsGLMFTest, createKernelErrorCalculateStatisticalMapsGLMBayesian;
		cl_int createKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelPermutation, createKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelPermutation;
		cl_int createKernelErrorCalculateStatisticalMapsGLMTTestSecondLevelPermutation, createKernelErrorCalculateStatisticalMapsGLMFTestSecondLevelPermutation;
		cl_int createKernelErrorEstimateAR4Models, createKernelErrorApplyWhiteningAR4;
		cl_int createKernelErrorGeneratePermutedVolumesFirstLevel;
		cl_int createKernelErrorRemoveLinearFit;

		// Create buffer errors
		cl_int createBufferErrorAlignedVolume, createBufferErrorReferenceVolume;
		cl_int createBufferErrorq11Real, createBufferErrorq11Imag, createBufferErrorq12Real, createBufferErrorq12Imag, createBufferErrorq13Real, createBufferErrorq13Imag, createBufferErrorq14Real, createBufferErrorq14Imag, createBufferErrorq15Real, createBufferErrorq15Imag, createBufferErrorq16Real, createBufferErrorq16Imag;
		cl_int createBufferErrorq21Real, createBufferErrorq21Imag, createBufferErrorq22Real, createBufferErrorq22Imag, createBufferErrorq23Real, createBufferErrorq23Imag, createBufferErrorq24Real, createBufferErrorq24Imag, createBufferErrorq25Real, createBufferErrorq25Imag, createBufferErrorq26Real, createBufferErrorq26Imag;
		cl_int createBufferErrorq11, createBufferErrorq12, createBufferErrorq13, createBufferErrorq14, createBufferErrorq15, createBufferErrorq16;
		cl_int createBufferErrorq21, createBufferErrorq22, createBufferErrorq23, createBufferErrorq24, createBufferErrorq25, createBufferErrorq26;
		cl_int createBufferErrort11, createBufferErrort12, createBufferErrort13, createBufferErrort22, createBufferErrort23, createBufferErrort33;
		cl_int createBufferErrorPhaseDifferences, createBufferErrorPhaseCertainties, createBufferErrorPhaseGradients;
		cl_int createBufferErrorAMatrix, createBufferErrorHVector, createBufferErrorAMatrix2DValues, createBufferErrorAMatrix1DValues, createBufferErrorHVector2DValues, createBufferErrorHVector1DValues;
		cl_int createBufferErrorQuadratureFilter1Real, createBufferErrorQuadratureFilter1Imag, createBufferErrorQuadratureFilter2Real, createBufferErrorQuadratureFilter2Imag, createBufferErrorQuadratureFilter3Real, createBufferErrorQuadratureFilter3Imag, createBufferErrorQuadratureFilter4Real, createBufferErrorQuadratureFilter4Imag, createBufferErrorQuadratureFilter5Real, createBufferErrorQuadratureFilter5Imag, createBufferErrorQuadratureFilter6Real, createBufferErrorQuadratureFilter6Imag;
		cl_int createBufferErrorRegistrationParameters;
		cl_int createBufferErrorBetaVolumesMNI;
		cl_int createBufferErrorStatisticalMapsMNI;
		cl_int createBufferErrorResidualVariancesMNI;
		cl_int createBufferErrorTensorNorms;

		// Run kernel errors

		// Help kernels
		cl_int runKernelErrorMemset, runKernelErrorMemsetInt, runKernelErrorMemsetFloat2;
		cl_int runKernelErrorMultiplyVolume;
		cl_int runKernelErrorMultiplyVolumes;
		cl_int runKernelErrorMultiplyVolumesOverwrite;
		cl_int runKernelErrorAddVolume;
		cl_int runKernelErrorAddVolumes;
		cl_int runKernelErrorAddVolumesOverwrite;
		cl_int runKernelErrorRemoveMean;
		cl_int runKernelErrorSetStartClusterIndices;
		cl_int runKernelErrorClusterizeScan;
		cl_int runKernelErrorClusterizeRelabel;
		cl_int runKernelErrorCalculateClusterSizes;
		cl_int runKernelErrorCalculateLargestCluster;


		// Convolution kernels
		cl_int runKernelErrorSeparableConvolutionRows, runKernelErrorSeparableConvolutionColumns, runKernelErrorSeparableConvolutionRods;
		cl_int runKernelErrorNonseparableConvolution3DComplexThreeFilters;
		cl_int runKernelErrorCalculateColumnSums;
		cl_int runKernelErrorCalculateRowSums;
		cl_int runKernelErrorCalculateColumnMaxs;
		cl_int runKernelErrorCalculateRowMaxs;
		cl_int runKernelErrorCalculateMaxAtomic;
		cl_int runKernelErrorThresholdVolume;

		cl_int runKernelErrorSliceTimingCorrection;

		// Image registration kernels
		cl_int runKernelErrorCalculatePhaseDifferencesAndCertainties, runKernelErrorCalculatePhaseGradientsX, runKernelErrorCalculatePhaseGradientsY, runKernelErrorCalculatePhaseGradientsZ;
		cl_int runKernelErrorCalculateAMatrixAndHVector2DValuesX, runKernelErrorCalculateAMatrixAndHVector2DValuesY, runKernelErrorCalculateAMatrixAndHVector2DValuesZ;
		cl_int runKernelErrorCalculateAMatrix1DValues, runKernelErrorCalculateHVector1DValues;
		cl_int runKernelErrorCalculateAMatrix, runKernelErrorCalculateHVector;
		cl_int runKernelErrorInterpolateVolumeNearestParametric, runKernelErrorInterpolateVolumeLinearParametric,  runKernelErrorInterpolateVolumeCubicParametric;
		cl_int runKernelErrorInterpolateVolumeNearestNonParametric, runKernelErrorInterpolateVolumeLinearNonParametric,  runKernelErrorInterpolateVolumeCubicNonParametric;
		cl_int runKernelErrorRescaleVolumeNearest, runKernelErrorRescaleVolumeLinear, runKernelErrorRescaleVolumeCubic;
		cl_int runKernelErrorCopyT1VolumeToMNI, runKernelErrorCopyEPIVolumeToT1, runKernelErrorCopyVolumeToNew;
		cl_int runKernelErrorCalculateMagnitudes;
		cl_int runKernelErrorCalculateTensorComponents;
		cl_int runKernelErrorCalculateTensorNorms;
		cl_int runKernelErrorCalculateAMatricesAndHVectors;
		cl_int runKernelErrorCalculateDisplacementUpdate;
		cl_int runKernelErrorAddParametricAndNonParametricDisplacement;

		// Statistical kernels
		cl_int runKernelErrorCalculateBetaWeightsGLM, runKernelErrorCalculateBetaWeightsGLMFirstLevel;
		cl_int runKernelErrorCalculateGLMResiduals;
		cl_int runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevel, runKernelErrorCalculateStatisticalMapsGLMFTestFirstLevel;
		cl_int runKernelErrorCalculateStatisticalMapsGLMTTest, runKernelErrorCalculateStatisticalMapsGLMFTest, runKernelErrorCalculateStatisticalMapsGLMBayesian;
		cl_int runKernelErrorCalculateStatisticalMapsGLMTTestFirstLevelPermutation, runKernelErrorCalculateStatisticalMapsGLMFTestFirstLevelPermutation;
		cl_int runKernelErrorCalculateStatisticalMapsGLMTTestSecondLevelPermutation, runKernelErrorCalculateStatisticalMapsGLMFTestSecondLevelPermutation;
		cl_int runKernelErrorEstimateAR4Models, runKernelErrorApplyWhiteningAR4;
		cl_int runKernelErrorGeneratePermutedVolumesFirstLevel;
		cl_int runKernelErrorRemoveLinearFit;



		int OpenCLCreateBufferErrors[200];
		int OpenCLRunKernelErrors[200];
		int OpenCLCreateKernelErrors[200];

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
		size_t localWorkSizeInterpolateVolume[3];
		size_t localWorkSizeMultiplyVolumes[3];
		size_t localWorkSizeAddVolumes[3];
		size_t localWorkSizeCopyVolumeToNew[3];
		size_t localWorkSizeCalculateMagnitudes[3];
		size_t localWorkSizeCalculateColumnSums[3];
		size_t localWorkSizeCalculateRowSums[3];
		size_t localWorkSizeCalculateColumnMaxs[3];
		size_t localWorkSizeCalculateRowMaxs[3];
		size_t localWorkSizeCalculateMaxAtomic[3];
		size_t localWorkSizeThresholdVolume[3];
		size_t localWorkSizeCalculateBetaWeightsGLM[3];
		size_t localWorkSizeCalculateStatisticalMapsGLM[3];
		size_t localWorkSizeRemoveLinearFit[3];
		size_t localWorkSizeEstimateAR4Models[3];
		size_t localWorkSizeApplyWhiteningAR4[3];
		size_t localWorkSizeGeneratePermutedVolumesFirstLevel[3];
		size_t localWorkSizeCalculateTensorNorms[3];
		size_t localWorkSizeCalculateAMatricesAndHVectors[3];
		size_t localWorkSizeCalculateDisplacementAndCertaintyUpdate[3];
		size_t localWorkSizeClusterize[3];

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
		size_t globalWorkSizeInterpolateVolume[3];
		size_t globalWorkSizeMultiplyVolumes[3];
		size_t globalWorkSizeAddVolumes[3];
		size_t globalWorkSizeCopyVolumeToNew[3];
		size_t globalWorkSizeCalculateMagnitudes[3];
		size_t globalWorkSizeCalculateColumnSums[3];
		size_t globalWorkSizeCalculateRowSums[3];
		size_t globalWorkSizeCalculateColumnMaxs[3];
		size_t globalWorkSizeCalculateRowMaxs[3];
		size_t globalWorkSizeCalculateMaxAtomic[3];
		size_t globalWorkSizeThresholdVolume[3];
		size_t globalWorkSizeCalculateBetaWeightsGLM[3];
		size_t globalWorkSizeCalculateStatisticalMapsGLM[3];
		size_t globalWorkSizeRemoveLinearFit[3];
		size_t globalWorkSizeEstimateAR4Models[3];
		size_t globalWorkSizeApplyWhiteningAR4[3];
		size_t globalWorkSizeGeneratePermutedVolumesFirstLevel[3];
		size_t globalWorkSizeCalculateTensorNorms[3];
		size_t globalWorkSizeCalculateAMatricesAndHVectors[3];
		size_t globalWorkSizeCalculateDisplacementAndCertaintyUpdate[3];
		size_t globalWorkSizeClusterize[3];

		//------------------------------------------------
		// General variables
		//------------------------------------------------
		int BETA_SPACE;
		int FILE_TYPE, DATA_TYPE;
		bool DEBUG;
		//nifti_image *nifti_data;

		int EPI_DATA_W, EPI_DATA_H, EPI_DATA_D, EPI_DATA_T;
		int T1_DATA_W, T1_DATA_H, T1_DATA_D;
		int MNI_DATA_W, MNI_DATA_H, MNI_DATA_D;
		int CURRENT_DATA_W, CURRENT_DATA_H, CURRENT_DATA_D;

		// Resolution variables
		float EPI_VOXEL_SIZE_X, EPI_VOXEL_SIZE_Y, EPI_VOXEL_SIZE_Z;
		float TR;
		float T1_VOXEL_SIZE_X, T1_VOXEL_SIZE_Y, T1_VOXEL_SIZE_Z;
		float MNI_VOXEL_SIZE_X, MNI_VOXEL_SIZE_Y, MNI_VOXEL_SIZE_Z;

		double processing_times[20];


		int SLICE_ORDER;

		// Image registration variables
		int INTERPOLATION_MODE;
		int IMAGE_REGISTRATION_FILTER_SIZE;
		int NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS;
		int NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION;
		int NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION;
		int NUMBER_OF_ITERATIONS_FOR_MOTION_CORRECTION;
		int COARSEST_SCALE_T1_MNI, COARSEST_SCALE_EPI_T1;
		int MM_T1_Z_CUT, MM_EPI_Z_CUT;
		int	NUMBER_OF_NON_ZERO_A_MATRIX_ELEMENTS;

		float M11_1, M12_1, M13_1, M22_1, M23_1, M33_1;
		float M11_2, M12_2, M13_2, M22_2, M23_2, M33_2;
		float M11_3, M12_3, M13_3, M22_3, M23_3, M33_3;
		float M11_4, M12_4, M13_4, M22_4, M23_4, M33_4;
		float M11_5, M12_5, M13_5, M22_5, M23_5, M33_5;
		float M11_6, M12_6, M13_6, M22_6, M23_6, M33_6;

		double* motion_parameters_x;
		double* motion_parameters_y;
		double* motion_parameters_z;

		double* motion_corrected_curve;
		double relativeErrorEquationSystemSolution;

		// Smoothing variables
		int	SMOOTHING_FILTER_SIZE;
		int SMOOTHING_TYPE;
		float EPI_Smoothing_FWHM;
		float AR_Smoothing_FWHM;

		// Statistical analysis variables
		int NUMBER_OF_SUBJECTS;
		int NUMBER_OF_CONTRASTS;
		int NUMBER_OF_STATISTICAL_MAPS;
		int NUMBER_OF_GLM_REGRESSORS;
		int NUMBER_OF_MOTION_REGRESSORS;
		int NUMBER_OF_TOTAL_GLM_REGRESSORS;
		int NUMBER_OF_DETRENDING_REGRESSORS;
		int NUMBER_OF_CONFOUND_REGRESSORS;
		int INFERENCE_MODE;
		int USE_TEMPORAL_DERIVATIVES;
		int REGRESS_MOTION;
		int REGRESS_CONFOUNDS;
		float CLUSTER_DEFINING_THRESHOLD;
		int NUMBER_OF_CLUSTERS;
		int MAX_CLUSTER_SIZE;
		float MAX_CLUSTER_MASS;
		float SIGNIFICANCE_LEVEL;
		float SIGNIFICANCE_THRESHOLD;
		int STATISTICAL_TEST;
		int NUMBER_OF_BRAIN_VOXELS;
		int NUMBER_OF_INVALID_TIMEPOINTS;

		// Random permutation variables
		int NUMBER_OF_PERMUTATIONS;
		int NUMBER_OF_SIGNIFICANTLY_ACTIVE_VOXELS;
		int NUMBER_OF_SIGNIFICANTLY_ACTIVE_CLUSTERS;

		//--------------------------------------------------
		// Host pointers
		//--------------------------------------------------

		// Data pointers
		float		*h_fMRI_Volumes;
		float		*h_MNI_Brain_Mask;
		float		*h_Mask;
		float		*h_EPI_Mask;
		float		*h_Smoothed_EPI_Mask;
        float       *h_T1_Volume;
		float		*h_MNI_Volume;
		float		*h_MNI_Brain_Volume;
		float	    *h_EPI_Volume;
		float		*h_Aligned_T1_Volume;
		float		*h_Aligned_T1_Volume_NonParametric;
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
		cl_float2	*h_Quadrature_Filter_1_Parametric_Registration, *h_Quadrature_Filter_2_Parametric_Registration, *h_Quadrature_Filter_3_Parametric_Registration;
		float       *h_Quadrature_Filter_1_Parametric_Registration_Real, *h_Quadrature_Filter_2_Parametric_Registration_Real, *h_Quadrature_Filter_3_Parametric_Registration_Real, *h_Quadrature_Filter_1_Parametric_Registration_Imag, *h_Quadrature_Filter_2_Parametric_Registration_Imag, *h_Quadrature_Filter_3_Parametric_Registration_Imag;
		float       *h_Quadrature_Filter_1_NonParametric_Registration_Real, *h_Quadrature_Filter_2_NonParametric_Registration_Real, *h_Quadrature_Filter_3_NonParametric_Registration_Real, *h_Quadrature_Filter_1_NonParametric_Registration_Imag, *h_Quadrature_Filter_2_NonParametric_Registration_Imag, *h_Quadrature_Filter_3_NonParametric_Registration_Imag;
		float       *h_Quadrature_Filter_4_NonParametric_Registration_Real, *h_Quadrature_Filter_5_NonParametric_Registration_Real, *h_Quadrature_Filter_6_NonParametric_Registration_Real, *h_Quadrature_Filter_4_NonParametric_Registration_Imag, *h_Quadrature_Filter_5_NonParametric_Registration_Imag, *h_Quadrature_Filter_6_NonParametric_Registration_Imag;
		cl_float2	*h_Quadrature_Filter_1_NonParametric_Registration, *h_Quadrature_Filter_2_NonParametric_Registration, *h_Quadrature_Filter_3_NonParametric_Registration, *h_Quadrature_Filter_4_NonParametric_Registration, *h_Quadrature_Filter_5_NonParametric_Registration, *h_Quadrature_Filter_6_NonParametric_Registration;
		float		*h_Quadrature_Filter_1_Real, *h_Quadrature_Filter_1_Imag, *h_Quadrature_Filter_2_Real, *h_Quadrature_Filter_2_Imag, *h_Quadrature_Filter_3_Real, *h_Quadrature_Filter_3_Imag, *h_Quadrature_Filter_4_Real, *h_Quadrature_Filter_4_Imag, *h_Quadrature_Filter_5_Real, *h_Quadrature_Filter_5_Imag, *h_Quadrature_Filter_6_Real, *h_Quadrature_Filter_6_Imag;
		cl_float2   *h_Quadrature_Filter_1, *h_Quadrature_Filter_2, *h_Quadrature_Filter_3, *h_Quadrature_Filter_4, *h_Quadrature_Filter_5, *h_Quadrature_Filter_6;
		float       *h_Quadrature_Filter_Response_1_Real, *h_Quadrature_Filter_Response_2_Real, *h_Quadrature_Filter_Response_3_Real;
		float       *h_Quadrature_Filter_Response_1_Imag, *h_Quadrature_Filter_Response_2_Imag, *h_Quadrature_Filter_Response_3_Imag;
		cl_float2   *h_Quadrature_Filter_Response_1, *h_Quadrature_Filter_Response_2, *h_Quadrature_Filter_Response_3, *h_Quadrature_Filter_Response_4, *h_Quadrature_Filter_Response_5, *h_Quadrature_Filter_Response_6;
		float		 h_A_Matrix[144], h_h_Vector[12];
		float		 *h_A_Matrix_Out, *h_h_Vector_Out;
		double		 h_A_Matrix_double[144], h_h_Vector_double[12];
		float 		 h_Registration_Parameters[12], h_Inverse_Registration_Parameters[12], h_Registration_Parameters_Old[12], h_Registration_Parameters_Temp[12], h_Registration_Parameters_EPI_T1_Affine[12], h_Registration_Parameters_EPI_T1_Translation[12], h_Registration_Parameters_EPI_T1_Rigid[12], h_Registration_Parameters_Motion_Correction[12], h_Registration_Parameters_T1_MNI[12], h_Registration_Parameters_EPI_MNI[12], *h_Registration_Parameters_T1_MNI_Out, h_Registration_Parameters_EPI_T1[6], *h_Registration_Parameters_EPI_T1_Out, *h_Registration_Parameters_EPI_MNI_Out;
		double       h_Registration_Parameters_double[12];
		float		 h_Rotations[3], h_Rotations_Temp[3];
		float       *h_Phase_Differences, *h_Phase_Certainties, *h_Phase_Gradients;

		float		*h_t11, *h_t12, *h_t13, *h_t22, *h_t23, *h_t33;
		float		*h_Displacement_Field_X, *h_Displacement_Field_Y, *h_Displacement_Field_Z;

		float		*h_Slice_Sums, *h_Top_Slice;

		float *h_Registration_Parameters_Out;
		// Motion correction variables
		float		*h_Motion_Corrected_fMRI_Volumes;
		float		*h_Motion_Parameters_Out, h_Motion_Parameters[10000];

		// fMRI - T1
		float		*h_Aligned_fMRI_Volum;

		// Smoothing pointers
		float		*h_Smoothing_Filter_X_In, *h_Smoothing_Filter_Y_In, *h_Smoothing_Filter_Z_In;
		float		h_Smoothing_Filter_X[9], h_Smoothing_Filter_Y[9], h_Smoothing_Filter_Z[9];
		float		*h_Smoothed_fMRI_Volumes;

		// Detrending pointers
		float		*h_X_Detrend, *h_xtxxt_Detrend;
		float		*h_Detrended_fMRI_Volumes;
		float		*h_Whitened_fMRI_Volumes;
		float		*h_Permuted_fMRI_Volumes;
		float		*h_Permuted_First_Level_Results;

		// Statistical analysis pointers
		float		*hrf;
		int			 HRF_LENGTH;
		float       *h_Contrasts, *h_Contrasts_In;
		float		*h_X_GLM_Out, *h_X_GLM_In, *h_X_GLM_Confounds, *h_xtxxt_GLM_In, *h_ctxtxc_GLM_In;
		float		*h_X_GLM, *h_X_GLM_With_Temporal_Derivatives, *h_X_GLM_Convolved, *h_xtxxt_GLM, *h_xtxxt_GLM_Out, *h_ctxtxc_GLM;
		float		*h_Censored_Timepoints, *h_Censored_Volumes;
		float		*h_Statistical_Maps;
		float       *h_Beta_Volumes;
		float		*h_First_Level_Results;
		float       *h_Residuals;
		float       *h_Residual_Variances;
		float		*h_AR1_Estimates;
		float		*h_AR2_Estimates;
		float		*h_AR3_Estimates;
		float		*h_AR4_Estimates;
		int			*h_Cluster_Indices;
		int 		*h_Largest_Cluster;
		cl_mem		 d_Cluster_Indices;
		cl_mem		 d_Cluster_Sizes;
		cl_mem		 d_Largest_Cluster;
		cl_mem		 d_Updated;
		int			*h_Cluster_Sizes;
		float		*h_Whitened_Models;

		// Random permutation pointers
		uint16		*h_Permutation_Matrix;
		float		*h_Permutation_Distribution;
		float		*h_Maximum_Test_Values;

		//--------------------------------------------------
		// Device pointers
		//--------------------------------------------------

		// Original data
		cl_mem		d_fMRI_Volumes;
		cl_mem		d_EPI_Mask;
		cl_mem		d_Smoothed_EPI_Mask;
		cl_mem		d_Group_Mask;

		// Slice timing correction
		float*		h_Slice_Differences;
		cl_mem		c_Slice_Differences;
		cl_mem		d_Slice_Timing_Corrected_fMRI_Volumes;

		// Image registration
		cl_mem		d_Tensor_Magnitude_T1, d_Tensor_Magnitude_T1_EPI;
		cl_mem      d_Reference_Volume, d_Aligned_Volume, d_Original_Volume;
		cl_mem		d_Current_Aligned_Volume, d_Current_Reference_Volume;
		cl_mem		d_A_Matrix, d_h_Vector, d_A_Matrix_2D_Values, d_A_Matrix_1D_Values, d_h_Vector_2D_Values, d_h_Vector_1D_Values;
		cl_mem		d_A_Matrix_double, d_h_Vector_double, d_A_Matrix_2D_Values_double, d_A_Matrix_1D_Values_double, d_h_Vector_2D_Values_double, d_h_Vector_1D_Values_double;
		cl_mem 		d_Phase_Differences, d_Phase_Gradients, d_Phase_Certainties;
		cl_mem      d_q11, d_q12, d_q13, d_q14, d_q15, d_q16, d_q21, d_q22, d_q23, d_q24, d_q25, d_q26;
		cl_mem      d_q11_Real, d_q12_Real, d_q13_Real, d_q14_Real, d_q15_Real, d_q16_Real, d_q21_Real, d_q22_Real, d_q23_Real, d_q24_Real, d_q25_Real, d_q26_Real;
		cl_mem      d_q11_Imag, d_q12_Imag, d_q13_Imag, d_q14_Imag, d_q15_Imag, d_q16_Imag, d_q21_Imag, d_q22_Imag, d_q23_Imag, d_q24_Imag, d_q25_Imag, d_q26_Imag;
		cl_mem		c_Quadrature_Filter_1_Real, c_Quadrature_Filter_2_Real, c_Quadrature_Filter_3_Real, c_Quadrature_Filter_4_Real, c_Quadrature_Filter_5_Real, c_Quadrature_Filter_6_Real;
		cl_mem		c_Quadrature_Filter_1_Imag, c_Quadrature_Filter_2_Imag, c_Quadrature_Filter_3_Imag, c_Quadrature_Filter_4_Imag, c_Quadrature_Filter_5_Imag, c_Quadrature_Filter_6_Imag;
		cl_mem		c_Quadrature_Filter_1, c_Quadrature_Filter_2, c_Quadrature_Filter_3, c_Quadrature_Filter_4, c_Quadrature_Filter_5, c_Quadrature_Filter_6;
		cl_mem		c_Registration_Parameters;
		cl_mem		d_Update_Displacement_Field_X, d_Update_Displacement_Field_Y, d_Update_Displacement_Field_Z, d_Update_Certainty;
		cl_mem		d_Temp_Displacement_Field_X, d_Temp_Displacement_Field_Y, d_Temp_Displacement_Field_Z;
		cl_mem		d_Total_Displacement_Field_X, d_Total_Displacement_Field_Y, d_Total_Displacement_Field_Z, d_Total_Certainty;
		cl_mem		d_t11, d_t12, d_t13, d_t22, d_t23, d_t33;
		cl_mem		d_Tensor_Norms, d_Smoothed_Tensor_Norms;
		cl_mem		d_a11, d_a12, d_a13, d_a22, d_a23, d_a33;
		cl_mem		d_h1, d_h2, d_h3;
		cl_mem		d_Displacement_Field;
		cl_mem		c_Filter_Directions_X, c_Filter_Directions_Y, c_Filter_Directions_Z;
		float		*h_Filter_Directions_X, *h_Filter_Directions_Y, *h_Filter_Directions_Z;
		double		TENSOR_NORM_SIGMA;
		double		EQUATION_SYSTEM_SIGMA;

		// Motion correction
		cl_mem		d_Motion_Corrected_fMRI_Volumes;

		// T1-MNI and EPI-T1 registration
		cl_mem		d_T1_Volume, d_Interpolated_T1_Volume, d_MNI_Volume, d_MNI_Brain_Volume, d_MNI_T1_Volume, d_Interpolated_fMRI_Volume, d_Skullstripped_T1_Volume, d_MNI_Brain_Mask;
		cl_mem		d_EPI_Volume, d_T1_EPI_Volume;

		// Smoothing
		cl_mem		d_Certainty;
		cl_mem		d_Smoothed_Certainty;
		cl_mem		d_Smoothed_fMRI_Volumes;

		cl_mem		d_Rows_Temp, d_Columns_Temp;

		cl_mem		c_Smoothing_Filter_X;
		cl_mem		c_Smoothing_Filter_Y;
		cl_mem		c_Smoothing_Filter_Z;

		// Detrending
		cl_mem		d_Detrended_fMRI_Volumes;
		cl_mem		c_xtxxt_Detrend, c_X_Detrend;

		// Statistical analysis
		cl_mem		d_First_Level_Results;
		cl_mem		d_Beta_Volumes, d_Beta_Volumes_MNI;
		cl_mem		d_Statistical_Maps, d_Statistical_Maps_MNI;
		cl_mem		c_Censor;
		cl_mem		c_xtxxt_GLM, c_X_GLM, c_Contrasts, c_ctxtxc_GLM;
		cl_mem		d_Residuals;
		cl_mem		d_Residual_Variances, d_Residual_Variances_MNI;
		cl_mem		c_Censored_Timepoints, c_Censored_Volumes;

		// Paraneters for single subject permutations
		cl_mem		d_AR1_Estimates, d_AR2_Estimates, d_AR3_Estimates, d_AR4_Estimates;

		cl_mem		d_BOLD_Regressed_fMRI_Volumes;
		cl_mem		d_Whitened_fMRI_Volumes;
		cl_mem		d_Permuted_fMRI_Volumes;
		cl_mem		d_Permuted_First_Level_Results;

		cl_mem		c_Permutation_Vector;


};



#endif
