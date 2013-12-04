from broccoli_base import *
import numpy
    
BROCCOLI_LIB_BASE = BROCCOLI_LIB

# DONE: Check that passing arrays to C method as 1D packed arrays is the same as passing arays using the 3D array wrappers
# DONE: Check that packing (packVolume) and unpacking (unpackOutputVolume) results in the original input array
# DONE: Transpose and reshape until the two conditions above are not met

"""
  This is a hack to prevent Python from free()-ing arrays
  that have been packed and then passed to C
"""
_input_arrays = []

_pack_permutation = (2, 0, 1)
_unpack_permutation = numpy.argsort(_pack_permutation)

def _permute(permutation, array):
  n = len(array)
  return [array[permutation[i]] for i in range(n)] 

def packArray(array):
  return numpy.ascontiguousarray(array, dtype=numpy.float32)

def packVolume(array):
  t = array.transpose(_pack_permutation)
  t = numpy.fliplr(t)
  t = packArray(t.flatten())
  _input_arrays.append(t)
  return t

def createOutputArray(shape, dtype=numpy.float32):
  return numpy.empty(shape=shape, dtype=dtype).flatten()

def unpackOutputArray(array, shape):
  return array.reshape(shape)
  
def unpackOutputVolume(array, shape = None):
  if shape:
    t_shape = _permute(_pack_permutation, shape)
    array = unpackOutputArray(array, t_shape)
    array = numpy.fliplr(array)
  return array.transpose(_unpack_permutation)
    
class BROCCOLI_LIB(BROCCOLI_LIB_BASE): 
  def SetEPIData(self, array, voxel_sizes):
    self.SetEPIHeight(array.shape[0])
    self.SetEPIWidth(array.shape[1])
    self.SetEPIDepth(array.shape[2])

    t = packVolume(array)
    self.SetInputEPIVolume(t)

    self.SetEPIVoxelSizeX(voxel_sizes[0])
    self.SetEPIVoxelSizeY(voxel_sizes[1])
    self.SetEPIVoxelSizeZ(voxel_sizes[2])
    
  def SetT1Data(self, array, voxel_sizes):
    self.SetT1Height(array.shape[0])
    self.SetT1Width(array.shape[1])
    self.SetT1Depth(array.shape[2])

    t = packVolume(array)
    self.SetInputT1Volume(t)

    self.SetT1VoxelSizeX(voxel_sizes[0])
    self.SetT1VoxelSizeY(voxel_sizes[1])
    self.SetT1VoxelSizeZ(voxel_sizes[2])
    
  def SetMNIData(self, array, voxel_sizes):
    self.SetMNIHeight(array.shape[0])
    self.SetMNIWidth(array.shape[1])
    self.SetMNIDepth(array.shape[2])

    t = packVolume(array)
    self.SetInputMNIVolume(t)

    self.SetMNIVoxelSizeX(voxel_sizes[0])
    self.SetMNIVoxelSizeY(voxel_sizes[1])
    self.SetMNIVoxelSizeZ(voxel_sizes[2])
    
  def SetParametricImageRegistrationFilters(self, filters):
    args = []
    for i in range(3):
      args.append(packVolume(numpy.real(filters[i])))
      args.append(packVolume(numpy.imag(filters[i])))
    BROCCOLI_LIB_BASE.SetParametricImageRegistrationFilters(self, *args)
    
  def SetNonParametricImageRegistrationFilters(self, filters):
    args = []
    for i in range(6):
      args.append(packVolume(numpy.real(filters[i])))
      args.append(packVolume(numpy.imag(filters[i])))
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
