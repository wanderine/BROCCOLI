import broccoli

def registerT1MNI(
    h_T1_Volume,        # double
    h_MNI_Volume,       # double
    h_MNI_Brain,        # double
    h_MNI_Brain_Mask,   # double
    T1_VOXEL_SIZE,      # 3 elements 
    MNI_VOXEL_SIZE,     # 3 elements
    h_Quadrature_Filter_Parametric_Registration,            # 3 elements, complex
    h_Quadrature_Filter_NonParametric_Registration,         # 6 elements, complex
    h_Projection_Tensor,             # 6 elements
    h_Filter_Directions,             # 3 elements
    NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION,     # int
    NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION,  # int
    COARSEST_SCALE,         # int
    MM_T1_Z_CUT,            # int
    OPENCL_PLATFORM,        # int
    OPENCL_DEVICE,          # int
  ):
  
  b = broccoli.BROCCOLI_LIB(OPENCL_PLATFORM, OPENCL_DEVICE)
  ok = b.GetOpenCLInitiated()
  
  if ok == 0:
    print("Get platform IDs error is %d" % b.GetOpenCLPlatformIDsError())
    print("Get device IDs error is %d" % b.GetOpenCLDeviceIDsError())
    print("Create context error is %d" % b.GetOpenCLCreateContextError())
    print("Get create context info error is %d" % b.GetOpenCLContextInfoError())
    print("Create command queue error is %d" % b.GetOpenCLCreateCommandQueueError())
    print("Create program error is %d" % b.GetOpenCLCreateProgramError())
    print("Build program error is %d" % b.GetOpenCLBuildProgramError())
    print("Get program build info error is %d" % b.GetOpenCLProgramBuildInfoError())
    
    numOpenKernels = b.GetNumberOfOpenCLKernels()
    createKernelErrors = b.GetOpenCLCreateKernelErrors()
    
    for i in range(numOpenKernels):
      error = createKernelErrors[i]
      if error:
        print("Run kernel error %d is %d" % (i, error))

    print("OpenCL initialization failed, aborting")
  else:
    print("OpenCL initialization successful, proceeding...")

if __name__ == "__main__":
  zeros = [0] * 16
  registerT1MNI(*zeros)
