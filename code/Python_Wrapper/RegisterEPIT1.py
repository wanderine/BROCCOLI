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

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Performs T1 MNI registration')
  
  parser.add_argument('--opencl-platform', type=int, default=0)
  parser.add_argument('--opencl-device', type=int, default=0)
  
  parser.add_argument('--epi-file', type=str, required=True)
  parser.add_argument('--t1-file', type=str, required=True)
  
  parser.add_argument('--iterations-parametric', type=int, default=20)
  
  parser.add_argument('--filters-parametric-file', type=str, default="../Matlab_Wrapper/filters_for_parametric_registration.mat")
  parser.add_argument('--filters-nonparametric-file', type=str, default="../Matlab_Wrapper/filters_for_nonparametric_registration.mat")
  
  parser.add_argument('--mm-epi-z-cut', type=int, default=30)
  parser.add_argument('--show-results', action='store_true')
  
  args = parser.parse_args()
  
  (T1, T1_voxel_sizes) = broccoli.load_T1(args.t1_file)
  (EPI, EPI_voxel_sizes) = broccoli.load_EPI(args.epi_file)
  
  coarsest_scale = 8
  
  filters_parametric_mat = scipy.io.loadmat(args.filters_parametric_file)
  filters_nonparametric_mat = scipy.io.loadmat(args.filters_nonparametric_file)
  
  parametric_filters = [filters_parametric_mat['f%d_parametric_registration' % (i+1)] for i in range(3)]
  nonparametric_filters = [filters_nonparametric_mat['f%d_nonparametric_registration' % (i+1)] for i in range(6)]
  
  results = broccoli.registerEPIT1(EPI, EPI_voxel_sizes, T1, T1_voxel_sizes, parametric_filters, nonparametric_filters,
                [filters_nonparametric_mat['m%d' % (i+1)][0] for i in range(6)], 
                [filters_nonparametric_mat['filter_directions_%s' % d][0] for d in ['x', 'y', 'z']],
                args.iterations_parametric,
                coarsest_scale,
                args.mm_epi_z_cut,
                args.opencl_platform,
                args.opencl_device,
                args.show_results)

  