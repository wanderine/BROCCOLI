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
  
  

  BROCCOLI.SetfMRIData(fMRI_data, fMRI_voxel_sizes)
  BROCCOLI.SetT1Data(T1_data, T1_voxel_sizes)
  BROCCOLI.SetMNIData(MNI_data, MNI_voxel_sizes)
  BROCCOLI.SetInputMNIBrainVolume(BROCCOLI.packVolume(MNI_brain_data))
  BROCCOLI.SetInputMNIBrainMask(BROCCOLI.packVolume(MNI_brain_mask_data))

  BROCCOLI.SetNumberOfIterationsForParametricImageRegistration(iterations_parametric)
  BROCCOLI.SetNumberOfIterationsForNonParametricImageRegistration(iterations_nonparametric)
  BROCCOLI.SetImageRegistrationFilterSize(parametric_filters[0][0].shape[0])
  BROCCOLI.SetParametricImageRegistrationFilters(parametric_filters)
  BROCCOLI.SetProjectionTensorMatrixFilters(projection_tensor)
  BROCCOLI.SetFilterDirections(filter_directions)
  
  
  BROCCOLI.SetNumberOfIterationsForMotionCorrection(iterations_motion_correction)
  BROCCOLI.SetCoarsestScaleT1MNI(coarsest_scale_T1_MNI)
  BROCCOLI.SetCoarsestScaleEPIT1(coarsest_scale_EPI_T1)
  BROCCOLI.SetMMT1ZCUT(mm_T1_z_cut)  
  BROCCOLI.SetMMEPIZCUT(mm_EPI_z_cut)
  
  NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_RIGID = 6
  NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_AFFINE = 12
 
  T1_MNI_registration_parameters = BROCCOLI.createOutputArray(NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_AFFINE)
  BROCCOLI.SetOutputT1MNIRegistrationParameters(T1_MNI_registration_parameters)
  
  EPI_T1_registration_parameters = BROCCOLI.createOutputArray(NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_RIGID)
  BROCCOLI.SetOutputEPIT1RegistrationParameters(EPI_T1_registration_parameters)
  
  EPI_MNI_registration_parameters = BROCCOLI.createOutputArray(NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_AFFINE)
  BROCCOLI.SetOutputEPIMNIRegistrationParameters(EPI_MNI_registration_parameters)
  
  motion_corrected_fMRI_data = BROCCOLI.createOutputArray(fMRI_data.shape)
  BROCCOLI.SetOutputMotionCorrectedfMRIVolumes(motion_corrected_fMRI_data)
  
  motion_parameters = BROCCOLI.createOutputArray(NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_RIGID * fMRI_data.shape[3])
  BROCCOLI.SetOutputMotionParameters(motion_parameters)
  
  BROCCOLI.SetEPISmoothingAmount(EPI_smoothing)
  BROCCOLI.SetARSmoothingAmount(AR_smoothing)
  
  smoother_fMRI_data = BROCCOLI.createOutputArray(fMRI_data.shape)
  BROCCOLI.SetOutputSmoothedfMRIVolumes(smoother_fMRI_data)

  BROCCOLI.PerformFirstLevelAnalysisWrapper()