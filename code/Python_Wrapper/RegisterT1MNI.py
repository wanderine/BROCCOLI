import broccoli

class Array:
  def __init__(self, data, dimensions, voxel_sizes = None):
    self.data = data
    self.dimensions = dimensions
    if voxel_sizes:
      self.voxel_sizes = voxel_sizes
    else:
      self.voxel_sizes = [1 for i in dimensions]
    
class BROCCOLI_EXT(broccoli.BROCCOLI_LIB):
  def __init__(self, opencl_platform, opencl_device):
    broccoli.BROCCOLI_LIB.__init__(self, opencl_platform, opencl_device)
    
  def SetT1Data(self, array):
    self.SetT1Width(array.dimensions[0])
    self.SetT1Height(array.dimensions[1])
    self.SetT1Depth(array.dimensions[2])
    self.SetT1VoxelSizeX(array.dimensions[0])
    self.SetT1VoxelSizeY(array.dimensions[1])
    self.SetT1VoxelSizeZ(array.dimensions[2])
    self.SetInputT1Volume(array.data)
    
  def SetMNIData(self, array):
    self.SetMNIWidth(array.dimensions[0])
    self.SetMNIHeight(array.dimensions[1])
    self.SetMNIDepth(array.dimensions[2])
    self.SetMNIVoxelSizeX(array.dimensions[0])
    self.SetMNIVoxelSizeY(array.dimensions[1])
    self.SetMNIVoxelSizeZ(array.dimensions[2])
    self.SetInputMNIVolume(array.data)
    
  def SetParametricImageRegistrationFilters(self, filters):
    args = []
    for i in range(3):
      real = [c.real for c in filters[i].data]
      imag = [c.imag for c in filters[i].data]
      args.append(real)
      args.append(imag)
    broccoli.BROCCOLI_LIB.SetParametricImageRegistrationFilters(self, *args)
    
  def SetNonParametricImageRegistrationFilters(self, filters):
    args = []
    for i in range(6):
      real = [c.real for c in filters[i].data]
      imag = [c.imag for c in filters[i].data]
      args.append(real)
      args.append(imag)
    broccoli.BROCCOLI_LIB.SetNonParametricImageRegistrationFilters(self, *args)
    
  def SetProjectionTensorMatrixFilters(self, filters):
    self.SetProjectionTensorMatrixFirstFilter(*filters[0])
    self.SetProjectionTensorMatrixSecondFilter(*filters[1])
    self.SetProjectionTensorMatrixThirdFilter(*filters[2])
    self.SetProjectionTensorMatrixFourthFilter(*filters[3])
    self.SetProjectionTensorMatrixFifthFilter(*filters[4])
    self.SetProjectionTensorMatrixSixthFilter(*filters[5])

  def printErrors(self):
    print("Get platform IDs error is %d" % self.GetOpenCLPlatformIDsError())
    print("Get device IDs error is %d" % self.GetOpenCLDeviceIDsError())
    print("Create context error is %d" % self.GetOpenCLCreateContextError())
    print("Get create context info error is %d" % self.GetOpenCLContextInfoError())
    print("Create command queue error is %d" % self.GetOpenCLCreateCommandQueueError())
    print("Create program error is %d" % self.GetOpenCLCreateProgramError())
    print("Build program error is %d" % self.GetOpenCLBuildProgramError())
    print("Get program build info error is %d" % self.GetOpenCLProgramBuildInfoError())
    
    numOpenKernels = self.GetNumberOfOpenCLKernels()
    createKernelErrors = self.GetOpenCLCreateKernelErrors()
    
    for i in range(numOpenKernels):
      error = createKernelErrors[i]
      if error:
        print("Run kernel error %d is %d" % (i, error))

def registerT1MNI(
    h_T1_Data,          # Array
    h_MNI_Data,         # Array
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
  
  BROCCOLI = BROCCOLI_EXT(OPENCL_PLATFORM, OPENCL_DEVICE)
  ok = BROCCOLI.GetOpenCLInitiated()
  
  if ok == 0:
    BROCCOLI.printErrors()
    print("OpenCL initialization failed, aborting")
    return

  print("OpenCL initialization successful, proceeding...")
  
  BROCCOLI.SetT1Data(h_T1_Data)
  BROCCOLI.SetMNIData(h_MNI_Data)
  BROCCOLI.SetInputMNIBrainVolume(h_MNI_Brain)
  BROCCOLI.SetInputMNIBrainMask(h_MNI_Brain_Mask)
  
  BROCCOLI.SetInterpolationMode(broccoli.LINEAR) # Linear
  BROCCOLI.SetNumberOfIterationsForParametricImageRegistration(NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION)
  BROCCOLI.SetNumberOfIterationsForNonParametricImageRegistration(NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION)
  BROCCOLI.SetImageRegistrationFilterSize(h_Quadrature_Filter_Parametric_Registration[0].dimensions[0])
  
  BROCCOLI.SetParametricImageRegistrationFilters(h_Quadrature_Filter_Parametric_Registration)
  BROCCOLI.SetNonParametricImageRegistrationFilters(h_Quadrature_Filter_NonParametric_Registration)
  
  BROCCOLI.SetFilterDirections(*h_Filter_Directions)
  
if __name__ == "__main__":
  size3 = [1, 1, 1]
  size1 = [1]
  data = [1.0]
  registerT1MNI(
    Array(data, size3),
    Array(data, size3),
    data,
    data,
    [Array(data, size3) for i in range(3)],
    [Array(data, size3) for i in range(6)],
    [data for i in range(6)],
    [data for i in range(3)],
    10,
    10,
    10,
    10,
    0,
    0
  )
