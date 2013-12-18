import broccoli_common as broccoli
import numpy

import matplotlib.pyplot as plot
import matplotlib.cm as cm
    
def plotVolume(data, sliceYrel, sliceZrel):
  sliceY = int(round(sliceYrel * data.shape[0]))

  # Data is first ordered [y][x][z]
  plot.imshow(numpy.flipud(data[sliceY].transpose()), cmap = cm.Greys_r, interpolation="nearest")
  plot.draw()
  plot.figure()

  sliceZ = int(round(sliceZrel * data.shape[2])) - 1

  plot.imshow(numpy.fliplr(data[:,:,sliceZ]), cmap = cm.Greys_r, interpolation="nearest")
  plot.draw()
  plot.figure()

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
  opencl_platform, opencl_device, show_results = False,
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
  
  fMRI_data = numpy.flipud(fMRI_data)
  
  brain_max = MNI_brain_data.max()
  MNI_brain_data = MNI_brain_data / brain_max

  BROCCOLI.SetfMRIData(fMRI_data, fMRI_voxel_sizes)
  BROCCOLI.SetT1Data(T1_data, T1_voxel_sizes)
  BROCCOLI.SetMNIData(MNI_data, MNI_voxel_sizes)
  BROCCOLI.SetInputMNIBrainVolume(BROCCOLI.packVolume(MNI_brain_data))
  BROCCOLI.SetInputMNIBrainMask(BROCCOLI.packVolume(MNI_brain_mask_data))

  BROCCOLI.SetNumberOfIterationsForParametricImageRegistration(iterations_parametric)
  BROCCOLI.SetNumberOfIterationsForNonParametricImageRegistration(iterations_nonparametric)
  BROCCOLI.SetImageRegistrationFilterSize(parametric_filters[0][0].shape[0])
  BROCCOLI.SetParametricImageRegistrationFilters(parametric_filters)
  BROCCOLI.SetNonParametricImageRegistrationFilters(nonparametric_filters)
  BROCCOLI.SetProjectionTensorMatrixFilters(projection_tensor)
  BROCCOLI.SetFilterDirections(*[BROCCOLI.packArray(f) for f in filter_directions])
  
  
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
  
  smoothed_fMRI_data = BROCCOLI.createOutputArray(fMRI_data.shape)
  BROCCOLI.SetOutputSmoothedfMRIVolumes(smoothed_fMRI_data)
  
  BROCCOLI.SetTemporalDerivatives(use_temporal_derivatives)
  BROCCOLI.SetRegressMotion(regress_motion)
  BROCCOLI.SetRegressConfounds(regress_confounds)
  BROCCOLI.SetBetaSpace(beta_space)
  
  number_of_GLM_regressors = X_GLM.shape[1]
  
  NUMBER_OF_DETRENDING_REGRESSORS = 4
  NUMBER_OF_MOTION_REGRESSORS = 6
  number_of_contrasts = len(contrasts)

  if regress_confounds == 1:
      number_of_confound_regressors = X_GLM_confounds.shape[1]
      BROCCOLI.SetNumberOfConfoundRegressors(number_of_confound_regressors)
      BROCCOLI.SetConfoundRegressors(X_GLM_confounds)
  else:
      number_of_confound_regressors = 1

  BROCCOLI.SetNumberOfGLMRegressors(number_of_GLM_regressors)
  BROCCOLI.SetNumberOfContrasts(number_of_contrasts)
  BROCCOLI.SetDesignMatrix(BROCCOLI.packVolume(X_GLM), BROCCOLI.packVolume(xtxxt_GLM))
  
  number_of_total_GLM_regressors = number_of_GLM_regressors * (use_temporal_derivatives+1) + NUMBER_OF_DETRENDING_REGRESSORS + NUMBER_OF_MOTION_REGRESSORS * regress_motion + number_of_confound_regressors * regress_confounds;
  design_matrix_shape = (number_of_total_GLM_regressors, fMRI_data.shape[3])
  
  design_matrix1 = BROCCOLI.createOutputArray(design_matrix_shape)
  design_matrix2 = BROCCOLI.createOutputArray(design_matrix_shape)
  BROCCOLI.SetOutputDesignMatrix(design_matrix1, design_matrix2)
  
  ctxtxc_GLM = numpy.array(ctxtxc_GLM)
  BROCCOLI.SetContrasts(BROCCOLI.packVolume(contrasts))
  BROCCOLI.SetGLMScalars(BROCCOLI.packVolume(ctxtxc_GLM))
  
  # print("ctxtxc_GLM : ", ctxtxc_GLM)
  # print("xtxxt_GLM: ", xtxxt_GLM)
  print("Number of GLM regressors : ", number_of_GLM_regressors)
  print("Number of confound regressors : ", number_of_confound_regressors) 
  print("Number of total GLM regressors : ", number_of_total_GLM_regressors) 
  print("Number of contrasts : ", number_of_contrasts)
  
  if beta_space == broccoli.MNI:
      beta_shape = MNI_data.shape + (number_of_total_GLM_regressors,)
      statistical_maps_shape = MNI_data.shape + (number_of_contrasts,)
      residual_variances_shape = MNI_data.shape
  elif beta_space == broccoli.EPI:
      beta_shape = fMRI_data.shape + (number_of_total_GLM_regressors,)
      statistical_maps_shape = fMRI_data.shape[0:3] + (number_of_contrasts,)
      residual_variances_shape = fMRI_data.shape
  
  beta_volumes = BROCCOLI.createOutputArray(beta_shape)
  BROCCOLI.SetOutputBetaVolumes(beta_volumes)
  
  residuals = BROCCOLI.createOutputArray(fMRI_data.shape)
  BROCCOLI.SetOutputResiduals(residuals)
  
  residual_variances = BROCCOLI.createOutputArray(residual_variances_shape)
  BROCCOLI.SetOutputResidualVariances(residual_variances)
  
  statistical_maps = BROCCOLI.createOutputArray(statistical_maps_shape)
  BROCCOLI.SetOutputStatisticalMaps(statistical_maps)
  
  ar_shape = fMRI_data.shape[0:3]
  ar_estimates = [BROCCOLI.createOutputArray(ar_shape) for i in range(4)]
  BROCCOLI.SetOutputAREstimates(*ar_estimates)
  
  whitened_models = BROCCOLI.createOutputArray(fMRI_data.shape + (number_of_total_GLM_regressors,))
  BROCCOLI.SetOutputWhitenedModels(whitened_models)

  aligned_T1_Volume = BROCCOLI.createOutputArray(MNI_data.shape)
  BROCCOLI.SetOutputAlignedT1Volume(aligned_T1_Volume)
  
  aligned_T1_Volume_nonparametric = BROCCOLI.createOutputArray(MNI_data.shape)
  BROCCOLI.SetOutputAlignedT1VolumeNonParametric(aligned_T1_Volume_nonparametric)
  
  aligned_EPI_volume = BROCCOLI.createOutputArray(MNI_data.shape)
  BROCCOLI.SetOutputAlignedEPIVolume(aligned_EPI_volume)

  cluster_indices = BROCCOLI.createOutputArray(fMRI_data.shape[0:3], dtype=numpy.int32)
  BROCCOLI.SetOutputClusterIndices(cluster_indices)
  
  EPI_mask = BROCCOLI.createOutputArray(fMRI_data.shape[0:3])
  BROCCOLI.SetOutputEPIMask(EPI_mask)

  print("Parameters set, now performing analysis...")
  BROCCOLI.PerformFirstLevelAnalysisWrapper()
  print("First level analysis performed!")
  
  print("T1_MNI_registration_parameters = ", T1_MNI_registration_parameters)
  print("EPI_T1_registration_parameters = ", EPI_T1_registration_parameters)
  print("EPI_MNI_registration_parameters = ", EPI_MNI_registration_parameters)
  
  motion_corrected_fMRI_data = BROCCOLI.unpackOutputVolume(motion_corrected_fMRI_data, fMRI_data.shape)
  
  motion_parameters = BROCCOLI.unpackOutputArray(motion_parameters, shape=(NUMBER_OF_IMAGE_REGISTRATION_PARAMETERS_RIGID, fMRI_data.shape[3]))
  
  smoothed_fMRI_data = BROCCOLI.unpackOutputVolume(smoothed_fMRI_data, fMRI_data.shape)
  design_matrix1 = BROCCOLI.unpackOutputVolume(design_matrix1, design_matrix_shape)
  design_matrix2 = BROCCOLI.unpackOutputVolume(design_matrix2, design_matrix_shape)
  
  beta_volumes = BROCCOLI.unpackOutputVolume(beta_volumes, beta_shape)
  residuals = BROCCOLI.unpackOutputVolume(residuals, fMRI_data.shape)
  residual_variances = BROCCOLI.unpackOutputVolume(residual_variances, residual_variances_shape)
  statistical_maps = BROCCOLI.unpackOutputVolume(statistical_maps, statistical_maps_shape)
  
  ar_estimates = [BROCCOLI.unpackOutputVolume(i, ar_shape) for i in ar_estimates]
  whitened_models = BROCCOLI.unpackOutputVolume(whitened_models, fMRI_data.shape + (number_of_total_GLM_regressors,))
  
  aligned_T1_Volume = BROCCOLI.unpackOutputVolume(aligned_T1_Volume, MNI_data.shape)
  aligned_T1_Volume_nonparametric = BROCCOLI.unpackOutputVolume(aligned_T1_Volume_nonparametric, MNI_data.shape)
  aligned_EPI_volume = BROCCOLI.unpackOutputVolume(aligned_EPI_volume, MNI_data.shape)
  cluster_indices = BROCCOLI.unpackOutputVolume(cluster_indices, fMRI_data.shape[0:3])
  EPI_mask = BROCCOLI.unpackOutputVolume(EPI_mask, fMRI_data.shape[0:3])
  
  if show_results:
    for volume in [aligned_T1_Volume_nonparametric, aligned_EPI_volume, MNI_brain_data]:
      plotVolume(volume, 0.45, 0.47)
      
  plot.plot(motion_parameters[0],'g')
  plot.plot(motion_parameters[1],'r')
  plot.plot(motion_parameters[2],'b')
  plot.title('Translation (mm)')
  plot.legend('X','Y','Z')
  plot.draw()
  plot.figure()
  
  plot.plot(motion_parameters[3],'g')
  plot.plot(motion_parameters[4],'r')
  plot.plot(motion_parameters[5],'b')
  plot.title('Rotation (degrees)')
  plot.legend('X','Y','Z')
  plot.draw()
  plot.figure()
  
  
  if beta_space == broccoli.MNI:
      slice = int(MNI_brain_data.shape[2] / 2)
  else:
      slice = int(fMRI_data.shape[2] / 2)
  
  plot.imshow(numpy.flipud(MNI_brain_data[:,:,slice]), cmap = cm.Greys_r, interpolation="nearest")
  plot.draw()
  plot.figure()
  
  print(statistical_maps.shape)
  print(statistical_maps[..., 0].shape)
  plot.imshow(numpy.flipud(statistical_maps[:,:,slice,0]), interpolation="nearest")
  plot.draw()
  plot.figure()

  plot.close()
  plot.show()
  
  # TODO: Return more parameters in proper order
  return statistical_maps
