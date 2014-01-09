#!/usr/bin/env python

import broccoli
import numpy
import scipy
from nibabel import nifti1

import argparse

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
  parser.add_argument('--show-results', action='store_true')

  args = parser.parse_args()

  (MNI, MNI_brain, MNI_brain_mask, MNI_voxel_sizes) = broccoli.load_MNI_templates(args.mni_file)
  (T1, T1_voxel_sizes) = broccoli.load_T1(args.t1_file)

  coarsest_scale = int(round(8 / MNI_voxel_sizes[0]))

  filters_parametric_mat = scipy.io.loadmat(args.filters_parametric_file)
  filters_nonparametric_mat = scipy.io.loadmat(args.filters_nonparametric_file)

  parametric_filters = [filters_parametric_mat['f%d_parametric_registration' % (i+1)] for i in range(3)]
  nonparametric_filters = [filters_nonparametric_mat['f%d_nonparametric_registration' % (i+1)] for i in range(6)]

  results = broccoli.registerT1MNI(T1, T1_voxel_sizes, MNI, MNI_voxel_sizes, MNI_brain, MNI_brain_mask, parametric_filters, nonparametric_filters,
                [filters_nonparametric_mat['m%d' % (i+1)][0] for i in range(6)],
                [filters_nonparametric_mat['filter_directions_%s' % d][0] for d in ['x', 'y', 'z']],
                args.iterations_parametric,
                args.iterations_nonparametric,
                coarsest_scale,
                args.mm_t1_z_cut,
                args.opencl_platform,
                args.opencl_device,
                args.show_results)

