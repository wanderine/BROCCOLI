from broccoli_base import *
import numpy

def floatArrayFromList(lst):
  n = len(lst)
  array = floatArray(n)
  
  if isinstance(lst, numpy.ndarray):
    lst = lst.flatten()
    
  for i in range(n):
    array[i] = float(lst[i])
  return array

class Array:
  def __init__(self, data, dimensions = None, voxel_sizes = None):
    self.data = data
    if dimensions is None:
      self.dimensions = data.shape
    else:
      self.dimensions = dimensions
    if voxel_sizes:
      self.voxel_sizes = voxel_sizes
    else:
      self.voxel_sizes = [1 for i in self.dimensions]
      
  def toFloatArray(self):
    return floatArrayFromList(self.data)
  
def arrayFromNifti(img, voxel_sizes = None):
  return Array(img.get_data(), img.shape, voxel_sizes)
    
BROCCOLI_LIB_BASE = BROCCOLI_LIB
    
class BROCCOLI_LIB(BROCCOLI_LIB_BASE):
  def SetT1Data(self, array):
    self.SetT1Width(array.dimensions[0])
    self.SetT1Height(array.dimensions[1])
    self.SetT1Depth(array.dimensions[2])
    self.SetT1VoxelSizeX(array.dimensions[0])
    self.SetT1VoxelSizeY(array.dimensions[1])
    self.SetT1VoxelSizeZ(array.dimensions[2])
    self.SetInputT1Volume(array.toFloatArray())
    
  def SetMNIData(self, array):
    self.SetMNIWidth(array.dimensions[0])
    self.SetMNIHeight(array.dimensions[1])
    self.SetMNIDepth(array.dimensions[2])
    self.SetMNIVoxelSizeX(array.dimensions[0])
    self.SetMNIVoxelSizeY(array.dimensions[1])
    self.SetMNIVoxelSizeZ(array.dimensions[2])
    self.SetInputMNIVolume(array.toFloatArray())
    
  def SetParametricImageRegistrationFilters(self, filters):
    args = []
    for i in range(3):
      real = floatArrayFromList([c.real for c in filters[i][0].data.flatten()])
      imag = floatArrayFromList([c.imag for c in filters[i][0].data.flatten()])
      args.append(real)
      args.append(imag)
    BROCCOLI_LIB_BASE.SetParametricImageRegistrationFilters(self, *args)
    
  def SetNonParametricImageRegistrationFilters(self, filters):
    args = []
    for i in range(6):
      real = floatArrayFromList([c.real for c in filters[i][0].data.flatten()])
      imag = floatArrayFromList([c.imag for c in filters[i][0].data.flatten()])
      args.append(real)
      args.append(imag)
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
