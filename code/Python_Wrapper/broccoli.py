from broccoli_base import *
import numpy
    
BROCCOLI_LIB_BASE = BROCCOLI_LIB

"""
  This is a hack to prevent Python from free()-ing arrays
  that have been packed and then passed to C
"""
_input_arrays = []

def packArray(array):
  return numpy.ascontiguousarray(array, dtype=numpy.float32)

def packVolume(array):
  t = array.transpose((1, 0, 2))
  t = numpy.fliplr(t)
  return packArray(t)

def createOutputArray(shape, dtype=numpy.float32):
  return numpy.empty(shape=shape, dtype=dtype).flatten()

def unpackOutputArray(array, shape):
  if len(shape) == 3:
    t_shape = (shape[2], shape[0], shape[1])
    t = array.reshape(t_shape)
    return t.transpose((2, 0, 1))
  else:
    return array.reshape(shape)
  
def unpackOutputVolume(array, shape = None):
  if shape:
    array = unpackOutputArray(array, shape)
  return numpy.fliplr(array).transpose((1, 0, 2))
    
class BROCCOLI_LIB(BROCCOLI_LIB_BASE):
  def SetEPIData(self, array, voxel_sizes):
    t = packVolume(array)
    _input_arrays.append(t)
    BROCCOLI_LIB_BASE.SetInputEPIData(self, t)
    self.SetEPIVoxelSizeX(voxel_sizes[0])
    self.SetEPIVoxelSizeY(voxel_sizes[1])
    self.SetEPIVoxelSizeZ(voxel_sizes[2])
    
  def SetT1Data(self, array, voxel_sizes):
    t = packVolume(array)
    _input_arrays.append(t)
    BROCCOLI_LIB_BASE.SetInputT1Data(self, t)
    self.SetT1VoxelSizeX(voxel_sizes[0])
    self.SetT1VoxelSizeY(voxel_sizes[1])
    self.SetT1VoxelSizeZ(voxel_sizes[2])
    
  def SetMNIData(self, array, voxel_sizes):
    t = packVolume(array)
    _input_arrays.append(t)
    BROCCOLI_LIB_BASE.SetInputMNIData(self, t)
    self.SetMNIVoxelSizeX(voxel_sizes[0])
    self.SetMNIVoxelSizeY(voxel_sizes[1])
    self.SetMNIVoxelSizeZ(voxel_sizes[2])
    
  def SetParametricImageRegistrationFilters(self, filters):
    args = []
    for i in range(3):
      args.append(packArray(numpy.real(filters[i][0])).flatten())
      args.append(packArray(numpy.imag(filters[i][0])).flatten())
    BROCCOLI_LIB_BASE.SetParametricImageRegistrationFilters(self, *args)
    
  def SetNonParametricImageRegistrationFilters(self, filters):
    args = []
    for i in range(6):
      args.append(packArray(numpy.real(filters[i][0])).flatten())
      args.append(packArray(numpy.imag(filters[i][0])).flatten())
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
