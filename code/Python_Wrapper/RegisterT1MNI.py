#!/usr/bin/env python

import broccoli
import numpy
import scipy
from nibabel import nifti1

import argparse

import matplotlib.pyplot as plot
import matplotlib.cm as cm

from operator import mul

def flatSize(a):
  if hasattr(a, 'shape'):
    a = a.shape
  return reduce(mul, a, 1)

def plotVolume(data):
  sliceY = int(round(0.45 * data.shape[0]))
  
  # Data is first ordered [y][x][z]
  plot.imshow(numpy.flipud(data[sliceY].transpose()), cmap = cm.Greys_r, interpolation="nearest")
  plot.show()
  
  sliceZ = int(round(0.47 * data.shape[2])) - 1
  
  # We want it ordered [z][x][y]
  data_t = data.transpose()
  plot.imshow(numpy.fliplr(data_t[sliceZ]).transpose(), cmap = cm.Greys_r, interpolation="nearest")
  plot.show()

def registerT1MNI(
    h_T1_Data,          # Array
    h_T1_Voxel_Sizes,   # 3 elements
    h_MNI_Data,         # Array
    h_MNI_Voxel_Sizes,   # 3 elements
    h_MNI_Brain,        # double
    h_MNI_Brain_Mask,   # double
    h_Quadrature_Filter_Parametric_Registration,            # 3 elements, complex arrays
    h_Quadrature_Filter_NonParametric_Registration,         # 6 elements, complex arrays
    h_Projection_Tensor,             # 6 elements
    h_Filter_Directions,             # 3 elements
    NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION,     # int
    NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION,  # int
    COARSEST_SCALE,         # int
    MM_T1_Z_CUT,            # int
    OPENCL_PLATFORM,        # int
    OPENCL_DEVICE,          # int
  ):
  
  BROCCOLI = broccoli.BROCCOLI_LIB()
  # BROCCOLI.GetOpenCLInfo()
  # print(BROCCOLI.GetOpenCLDeviceInfoChar())
  print("Initializing OpenCL...")
  
  BROCCOLI.OpenCLInitiate(OPENCL_PLATFORM, OPENCL_DEVICE)
  ok = BROCCOLI.GetOpenCLInitiated()
  
  if ok == 0:
    BROCCOLI.printSetupErrors()
    print("OpenCL initialization failed, aborting")
    return

  print("OpenCL initialization successful, proceeding...")
  
  ## Set constants
  NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS = 12
  MNI_DATA_SIZE = flatSize(h_MNI_Data)
  
  MNI_DATA_SHAPE = h_MNI_Data.shape
  T1_DATA_SHAPE = h_T1_Data.shape
  T1_INTERPOLATED_DATA_SHAPE = [int(round(float(T1_DATA_SHAPE[i]) * T1_voxel_sizes[i] / MNI_voxel_sizes[i])) for i in range(3)]
  
  ## Make all arrays contiguous
  h_T1_Data = broccoli.packArray(h_T1_Data)
  h_MNI_Data = broccoli.packArray(h_MNI_Data)
  h_MNI_Brain = broccoli.packArray(h_MNI_Brain)
  h_MNI_Brain_Mask = broccoli.packArray(h_MNI_Brain_Mask)
  
  h_MNI_Voxel_Sizes = [round(i) for i in h_MNI_Voxel_Sizes]
  h_T1_Voxel_Sizes = [round(i) for i in h_T1_Voxel_Sizes]
  
  ## Pass input parameters to BROCCOLI
  print("Setting up input parameters...")
  
  print("T1 size is %s" % ' x '.join([str(i) for i in h_T1_Data.shape]))
  print("MNI size is %s" % ' x '.join([str(i) for i in h_MNI_Data.shape]))
  
  BROCCOLI.SetT1Data(h_T1_Data, h_T1_Voxel_Sizes)
  BROCCOLI.SetMNIData(h_MNI_Data, h_MNI_Voxel_Sizes)

  BROCCOLI.SetInputMNIBrainVolume(broccoli.packVolume(h_MNI_Brain))
  BROCCOLI.SetInputMNIBrainMask(broccoli.packVolume(h_MNI_Brain_Mask))
  
  BROCCOLI.SetInterpolationMode(broccoli.LINEAR) # Linear
  BROCCOLI.SetNumberOfIterationsForParametricImageRegistration(NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION)
  BROCCOLI.SetNumberOfIterationsForNonParametricImageRegistration(NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION)
  
  BROCCOLI.SetImageRegistrationFilterSize(h_Quadrature_Filter_Parametric_Registration[0][0].shape[0])
  BROCCOLI.SetParametricImageRegistrationFilters(h_Quadrature_Filter_Parametric_Registration)
  BROCCOLI.SetNonParametricImageRegistrationFilters(h_Quadrature_Filter_NonParametric_Registration)

  BROCCOLI.SetProjectionTensorMatrixFilters(h_Projection_Tensor)
  BROCCOLI.SetFilterDirections(*[broccoli.packArray(a) for a in h_Filter_Directions])
  
  BROCCOLI.SetCoarsestScaleT1MNI(COARSEST_SCALE)
  BROCCOLI.SetMMT1ZCUT(MM_T1_Z_CUT)
    
  ## Set up output parameters
  print("Setting up output parameters...")
  
  h_Aligned_T1_Volume = broccoli.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputAlignedT1Volume(h_Aligned_T1_Volume)
  
  h_Aligned_T1_Volume_NonParametric = broccoli.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputAlignedT1VolumeNonParametric(h_Aligned_T1_Volume_NonParametric)
  
  h_Skullstripped_T1_Volume = broccoli.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputSkullstrippedT1Volume(h_Skullstripped_T1_Volume)
  
  h_Interpolated_T1_Volume = broccoli.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputInterpolatedT1Volume(h_Interpolated_T1_Volume)

  # Not used in broccoli_lib.cpp  
  # h_Downsampled_Volume = numpy.empty(MNI_DATA_SIZE, dtype=numpy.float32)
  # BROCCOLI.SetOutputDownsampledVolume(h_Downsampled_Volume)
  
  h_Registration_Parameters = broccoli.createOutputArray(12)
  h_Registration_Parameters[3] = 1
  print(h_Registration_Parameters)
  BROCCOLI.SetOutputT1MNIRegistrationParameters(h_Registration_Parameters)
  
  # Not used in broccoli_lib.cpp
  # h_Quadrature_Filter_Response = [broccoli.cl_float2Array(10, dtype=numpy.float32) for i in range(6)]
  # BROCCOLI.SetOutputQuadratureFilterResponses(*h_Quadrature_Filter_Response)
  
  # h_OutputTensor = [numpy.empty(10, dtype=numpy.float32) for i in range(6)]
  # BROCCOLI.SetOutputTensorComponents(*h_OutputTensor)
  
  # h_Displacement_Field = [numpy.empty(10, dtype=numpy.float32) for i in range(3)]
  # BROCCOLI.SetOutputDisplacementField(*h_Displacement_Field)
  
  h_Phase_Differences = broccoli.createOutputArray(h_MNI_Data.shape)
  BROCCOLI.SetOutputPhaseDifferences(h_Phase_Differences)
  
  h_Phase_Certainties = broccoli.createOutputArray(h_MNI_Data.shape)
  BROCCOLI.SetOutputPhaseCertainties(h_Phase_Certainties)
  
  h_Phase_Gradients = broccoli.createOutputArray(h_MNI_Data.shape)
  BROCCOLI.SetOutputPhaseGradients(h_Phase_Gradients)
  
  h_Slice_Sums = broccoli.createOutputArray(h_MNI_Data.shape[2])
  BROCCOLI.SetOutputSliceSums(h_Slice_Sums)
  
  h_Top_Slice = broccoli.createOutputArray(1)
  BROCCOLI.SetOutputTopSlice(h_Top_Slice)
  
  h_A_Matrix = broccoli.createOutputArray(NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS * NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS)
  BROCCOLI.SetOutputAMatrix(h_A_Matrix)
  
  h_h_Vector = broccoli.createOutputArray(NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS)
  BROCCOLI.SetOutputHVector(h_h_Vector)
    
  ## Perform registration
  print("Performing registration...")
  BROCCOLI.PerformRegistrationT1MNINoSkullstripWrapper()
    
  print(h_Registration_Parameters)
  
  plot_results = (
    broccoli.unpackOutputVolume(h_Interpolated_T1_Volume, MNI_DATA_SHAPE),
    broccoli.unpackOutputVolume(h_Aligned_T1_Volume, MNI_DATA_SHAPE),
    h_MNI_Brain,
    broccoli.unpackOutputVolume(h_Aligned_T1_Volume_NonParametric, MNI_DATA_SHAPE),
  )
  
  for r in plot_results:
    plotVolume(r)
  
  return (h_Aligned_T1_Volume, h_Aligned_T1_Volume_NonParametric, h_Skullstripped_T1_Volume, h_Interpolated_T1_Volume, 
          h_Registration_Parameters, h_Phase_Differences, h_Phase_Certainties, h_Phase_Gradients, h_Slice_Sums, h_Top_Slice, h_A_Matrix, h_h_Vector)

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description='Performs T1 MNI registration')
  
  parser.add_argument('--opencl-platform', type=int, default=0)
  parser.add_argument('--opencl-device', type=int, default=0)
  
  parser.add_argument('--mni-file', type=str, required=True)
  parser.add_argument('--mni-brain-file', type=str)
  parser.add_argument('--mni-brain-mask-file', type=str)
  parser.add_argument('--t1-file', type=str, required=True)
  
  parser.add_argument('--iterations-parametric', type=int, default=10)
  parser.add_argument('--iterations-nonparametric', type=int, default=15)
  
  parser.add_argument('--filters-parametric-file', type=str, default="../Matlab_Wrapper/filters_for_parametric_registration.mat")
  parser.add_argument('--filters-nonparametric-file', type=str, default="../Matlab_Wrapper/filters_for_nonparametric_registration.mat")
  
  parser.add_argument('--mm-t1-z-cut', type=int, default=30)
  
  args = parser.parse_args()
  
  (MNI, MNI_brain, MNI_brain_mask, MNI_voxel_sizes) = broccoli.load_MNI_templates(args.mni_file)
  (T1, T1_voxel_sizes) = broccoli.load_T1(args.t1_file)
  
  coarsest_scale = int(round(8 / MNI_voxel_sizes[0]))
  
  filters_parametric_mat = scipy.io.loadmat(args.filters_parametric_file)
  filters_nonparametric_mat = scipy.io.loadmat(args.filters_nonparametric_file)
  
  parametric_filters = [filters_parametric_mat['f%d_parametric_registration' % (i+1)] for i in range(3)]
  nonparametric_filters = [filters_nonparametric_mat['f%d_nonparametric_registration' % (i+1)] for i in range(6)]
  
  results = registerT1MNI(T1, T1_voxel_sizes, MNI, MNI_voxel_sizes, MNI_brain, MNI_brain_mask, parametric_filters, nonparametric_filters,
                [filters_nonparametric_mat['m%d' % (i+1)][0] for i in range(6)], 
                [filters_nonparametric_mat['filter_directions_%s' % d][0] for d in ['x', 'y', 'z']],
                args.iterations_parametric,
                args.iterations_nonparametric,
                coarsest_scale,
                args.mm_t1_z_cut,
                args.opencl_platform,
                args.opencl_device)

  