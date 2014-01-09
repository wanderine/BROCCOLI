import broccoli_common as broccoli
import matplotlib.pyplot as plot

def performMotionCorrection(
  h_fMRI_Volumes, h_fMRI_Voxel_Sizes,
  h_Quadrature_Filters,
  iterations,
  OPENCL_PLATFORM,
  OPENCL_DEVICE,
  show_results = False,
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
  
  fMRI_DATA_SHAPE = h_fMRI_Volumes.shape
  DATA_W, DATA_H, DATA_D, DATA_T = fMRI_DATA_SHAPE
  VOLUME_SHAPE = (DATA_W, DATA_H, DATA_D)
  print(fMRI_DATA_SHAPE)
  
  FILTER_SIZE = len(h_Quadrature_Filters[0])
  
  BROCCOLI.SetfMRIData(h_fMRI_Volumes, h_fMRI_Voxel_Sizes)
  
  BROCCOLI.SetImageRegistrationFilterSize(FILTER_SIZE)
  BROCCOLI.SetParametricImageRegistrationFilters(h_Quadrature_Filters)
  BROCCOLI.SetNumberOfIterationsForMotionCorrection(iterations)
  
  h_Motion_Corrected_fMRI_Volumes = BROCCOLI.createOutputArray(fMRI_DATA_SHAPE)
  BROCCOLI.SetOutputMotionCorrectedfMRIVolumes(h_Motion_Corrected_fMRI_Volumes)
  
  h_Motion_Parameters = BROCCOLI.createOutputArray((6, DATA_T))
  BROCCOLI.SetOutputMotionParameters(h_Motion_Parameters)
  
  h_Phase_Differences = BROCCOLI.createOutputArray(VOLUME_SHAPE)
  BROCCOLI.SetOutputPhaseDifferences(h_Phase_Differences)
  
  h_Phase_Certainties = BROCCOLI.createOutputArray(VOLUME_SHAPE)
  BROCCOLI.SetOutputPhaseCertainties(h_Phase_Certainties)
  
  h_Phase_Gradients = BROCCOLI.createOutputArray(VOLUME_SHAPE)
  BROCCOLI.SetOutputPhaseGradients(h_Phase_Gradients)
        
  BROCCOLI.PerformMotionCorrectionWrapper()
  
  h_Motion_Parameters = BROCCOLI.unpackOutputArray(h_Motion_Parameters, shape=(6, DATA_T))
  
  if show_results:
    for p in h_Motion_Parameters:
      plot.plot(p)
      plot.draw()
      plot.figure()
    plot.close()
    plot.show()
      
  h_Motion_Corrected_fMRI_Volumes = BROCCOLI.unpackOutputVolume(h_Motion_Corrected_fMRI_Volumes, fMRI_DATA_SHAPE)
  h_Phase_Differences = BROCCOLI.unpackOutputVolume(h_Phase_Differences, VOLUME_SHAPE)
  h_Phase_Certainties = BROCCOLI.unpackOutputVolume(h_Phase_Certainties, VOLUME_SHAPE)
  h_Phase_Gradients = BROCCOLI.unpackOutputVolume(h_Phase_Gradients, VOLUME_SHAPE)
  
  return h_Motion_Corrected_fMRI_Volumes, h_Motion_Parameters, h_Phase_Differences, h_Phase_Certainties, h_Phase_Gradients
