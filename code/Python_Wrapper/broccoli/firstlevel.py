import broccoli.common as broccoli

def performFirstLevelAnalysis(
  fMRI_data, fMRI_voxel_sizes,
  T1_data, T1_voxel_sizes,
  MNI_data, MNI_brain_data, MNI_brain_mask_data, MNI_voxel_sizes,
  parametric_filters, nonparametric_filters, projection_tensor, filter_directions,
  iterations_parametric, iterations_nonparametric, iterations_motion_correction,
  coarsest_scale_T1_MNI, coarsest_scale_EPI_T1, mm_T1_z_cut, mm_EPI_z_cut,
  regress_motion, EPI_smoothing, AR_smoothing,
  X_GLM, xtxxt_GLM, contrasts, ctxtxc_GLM,
  use_temporal_derivatives, beta_space, X_GLM_confounds, regress_confounds,
  opencl_platform, opencl_device,
  ):
    
  BROCCOLI = broccoli.BROCCOLI_LIB()
  print("Initializing OpenCL...")

  BROCCOLI.OpenCLInitiate(opencl_platform, opencl_device)
  ok = BROCCOLI.GetOpenCLInitiated()

  if ok == 0:
    BROCCOLI.printSetupErrors()
    print("OpenCL initialization failed, aborting")
    return

  print("OpenCL initialization successful, proceeding...")

  BROCCOLI.PerformFirstLevelAnalysisWrapper()