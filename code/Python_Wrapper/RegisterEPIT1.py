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
  
  sliceZ = int(round(0.62 * data.shape[2])) - 1
  
  # We want it ordered [z][x][y]
  data_t = data.transpose()
  plot.imshow(numpy.fliplr(data_t[sliceZ]).transpose(), cmap = cm.Greys_r, interpolation="nearest")
  plot.show()

def registerT1MNI(
    h_EPI_Data,
    h_EPI_Voxel_Sizes,
    h_T1_Data,          # Array
    h_T1_Voxel_Sizes,   # 3 elements
    h_Quadrature_Filter_Parametric_Registration,            # 3 elements, complex arrays
    h_Quadrature_Filter_NonParametric_Registration,         # 6 elements, complex arrays
    h_Projection_Tensor,             # 6 elements
    h_Filter_Directions,             # 3 elements
    NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION,     # int
    COARSEST_SCALE,         # int
    MM_EPI_Z_CUT,            # int
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
  T1_DATA_SHAPE = h_T1_Data.shape
  EPI_DATA_SHAPE = h_EPI_Data.shape
  EPI_INTERPOLATED_DATA_SHAPE = [int(round(float(EPI_DATA_SHAPE[i]) * EPI_voxel_sizes[i] / T1_voxel_sizes[i])) for i in range(3)]
  
  ## Make all arrays contiguous
  h_T1_Data = broccoli.packArray(h_T1_Data)
  h_EPI_Data = broccoli.packArray(h_EPI_Data)
  
  h_EPI_Voxel_Sizes = [round(i) for i in h_EPI_Voxel_Sizes]
  h_T1_Voxel_Sizes = [round(i) for i in h_T1_Voxel_Sizes]
  
  ## Pass input parameters to BROCCOLI
  print("Setting up input parameters...")
  
  print("EPI size is %s" % ' x '.join([str(i) for i in h_EPI_Data.shape]))
  print("T1 size is %s" % ' x '.join([str(i) for i in h_T1_Data.shape]))

  BROCCOLI.SetEPIData(h_EPI_Data, h_EPI_Voxel_Sizes)
  BROCCOLI.SetT1Data(h_T1_Data, h_T1_Voxel_Sizes)

  BROCCOLI.SetInterpolationMode(broccoli.LINEAR) # Linear
  BROCCOLI.SetNumberOfIterationsForParametricImageRegistration(NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION)
  
  BROCCOLI.SetImageRegistrationFilterSize(h_Quadrature_Filter_Parametric_Registration[0][0].shape[0])
  BROCCOLI.SetParametricImageRegistrationFilters(h_Quadrature_Filter_Parametric_Registration)
  BROCCOLI.SetNonParametricImageRegistrationFilters(h_Quadrature_Filter_NonParametric_Registration)

  BROCCOLI.SetProjectionTensorMatrixFilters(h_Projection_Tensor)
  BROCCOLI.SetFilterDirections(*[broccoli.packArray(a) for a in h_Filter_Directions])
  
  BROCCOLI.SetCoarsestScaleEPIT1(COARSEST_SCALE)
  BROCCOLI.SetMMEPIZCUT(MM_EPI_Z_CUT)
    
  ## Set up output parameters
  print("Setting up output parameters...")
  
  h_Aligned_EPI_Volume = broccoli.createOutputArray(T1_DATA_SHAPE)
  BROCCOLI.SetOutputAlignedEPIVolume(h_Aligned_EPI_Volume)
  
  h_Interpolated_EPI_Volume = broccoli.createOutputArray(T1_DATA_SHAPE)
  BROCCOLI.SetOutputInterpolatedEPIVolume(h_Interpolated_EPI_Volume)

  h_Registration_Parameters = broccoli.createOutputArray(6)
  BROCCOLI.SetOutputEPIT1RegistrationParameters(h_Registration_Parameters)
  
  h_Phase_Differences = broccoli.createOutputArray(h_T1_Data.shape)
  BROCCOLI.SetOutputPhaseDifferences(h_Phase_Differences)
  
  h_Phase_Certainties = broccoli.createOutputArray(h_T1_Data.shape)
  BROCCOLI.SetOutputPhaseCertainties(h_Phase_Certainties)
  
  h_Phase_Gradients = broccoli.createOutputArray(h_T1_Data.shape)
  BROCCOLI.SetOutputPhaseGradients(h_Phase_Gradients)
    
  ## Perform registration
  print("Performing registration...")
  BROCCOLI.PerformRegistrationEPIT1Wrapper()
  
  print(h_Registration_Parameters)
  
  plot_results = (
    broccoli.unpackOutputVolume(h_Interpolated_EPI_Volume, T1_DATA_SHAPE),
    broccoli.unpackOutputVolume(h_Aligned_EPI_Volume, T1_DATA_SHAPE),
    h_T1_Data,
  )
  
  for r in plot_results:
    plotVolume(r)
  
  return (h_Aligned_EPI_Volume, h_Interpolated_EPI_Volume, 
          h_Registration_Parameters, h_Phase_Differences, h_Phase_Certainties, h_Phase_Gradients)
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Performs T1 MNI registration')
  
  parser.add_argument('--opencl-platform', type=int, default=0)
  parser.add_argument('--opencl-device', type=int, default=0)
  
  parser.add_argument('--epi-file', type=str, required=True)
  parser.add_argument('--t1-file', type=str, required=True)
  
  parser.add_argument('--iterations-parametric', type=int, default=10)
  parser.add_argument('--iterations-nonparametric', type=int, default=15)
  
  parser.add_argument('--filters-parametric-file', type=str, default="../Matlab_Wrapper/filters_for_parametric_registration.mat")
  parser.add_argument('--filters-nonparametric-file', type=str, default="../Matlab_Wrapper/filters_for_nonparametric_registration.mat")
  
  parser.add_argument('--mm-epi-z-cut', type=int, default=30)
  
  args = parser.parse_args()
  
  (T1, T1_voxel_sizes) = broccoli.load_T1(args.t1_file)
  (EPI, EPI_voxel_sizes) = broccoli.load_EPI(args.epi_file)
  
  coarsest_scale = int(round(8 / T1_voxel_sizes[0]))
  
  filters_parametric_mat = scipy.io.loadmat(args.filters_parametric_file)
  filters_nonparametric_mat = scipy.io.loadmat(args.filters_nonparametric_file)
  
  parametric_filters = [filters_parametric_mat['f%d_parametric_registration' % (i+1)] for i in range(3)]
  nonparametric_filters = [filters_nonparametric_mat['f%d_nonparametric_registration' % (i+1)] for i in range(6)]
  
  results = registerT1MNI(EPI, EPI_voxel_sizes, T1, T1_voxel_sizes, parametric_filters, nonparametric_filters,
                [filters_nonparametric_mat['m%d' % (i+1)][0] for i in range(6)], 
                [filters_nonparametric_mat['filter_directions_%s' % d][0] for d in ['x', 'y', 'z']],
                args.iterations_parametric,
                coarsest_scale,
                args.mm_epi_z_cut,
                args.opencl_platform,
                args.opencl_device)

  