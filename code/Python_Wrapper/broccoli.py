from broccoli_base import *
import numpy
    
BROCCOLI_LIB_BASE = BROCCOLI_LIB

def packArray(array):
  return array.astype(numpy.float32).flatten()
    
class BROCCOLI_LIB(BROCCOLI_LIB_BASE):
  def SetT1Data(self, array, voxel_sizes):
    self.SetT1Width(array.shape[0])
    self.SetT1Height(array.shape[1])
    self.SetT1Depth(array.shape[2])
    self.SetT1VoxelSizeX(voxel_sizes[0])
    self.SetT1VoxelSizeY(voxel_sizes[1])
    self.SetT1VoxelSizeZ(voxel_sizes[2])
    self.SetInputT1Volume(packArray(array))
    
  def SetMNIData(self, array, voxel_sizes):
    self.SetMNIWidth(array.shape[0])
    self.SetMNIHeight(array.shape[1])
    self.SetMNIDepth(array.shape[2])
    self.SetMNIVoxelSizeX(voxel_sizes[0])
    self.SetMNIVoxelSizeY(voxel_sizes[1])
    self.SetMNIVoxelSizeZ(voxel_sizes[2])
    self.SetInputMNIVolume(packArray(array))
    
  def SetParametricImageRegistrationFilters(self, filters):
    args = []
    for i in range(3):
      args.append(packArray(numpy.real(filters[i][0])))
      args.append(packArray(numpy.imag(filters[i][0])))
    BROCCOLI_LIB_BASE.SetParametricImageRegistrationFilters(self, *args)
    
  def SetNonParametricImageRegistrationFilters(self, filters):
    args = []
    for i in range(6):
      args.append(packArray(numpy.real(filters[i][0])))
      args.append(packArray(numpy.imag(filters[i][0])))
    BROCCOLI_LIB_BASE.SetNonParametricImageRegistrationFilters(self, *args)
    
  def SetProjectionTensorMatrixFilters(self, filters):
    self.SetProjectionTensorMatrixFirstFilter(*filters[0])
    self.SetProjectionTensorMatrixSecondFilter(*filters[1])
    self.SetProjectionTensorMatrixThirdFilter(*filters[2])
    self.SetProjectionTensorMatrixFourthFilter(*filters[3])
    self.SetProjectionTensorMatrixFifthFilter(*filters[4])
    self.SetProjectionTensorMatrixSixthFilter(*filters[5])

  def printSetupErrors(self):
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
        
  def printRunErrors(self):
    numOpenKernels = self.GetNumberOfOpenCLKernels()
    createBufferErrors = self.GetOpenCLCreateBufferErrors()
    runKernelErrors = self.GetOpenCLRunKernelErrors()
    
    for i in range(numOpenKernels):
      if createBufferErrors[i]:
        print("Create buffer error %d is %d" % (i, createBufferErrors[i]))
      if runKernelErrors[i]:
        print("Run kernel error %d is %d" % (i, runKernelErrors[i]))
