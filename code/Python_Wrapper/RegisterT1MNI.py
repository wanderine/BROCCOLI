import broccoli
import numpy
from nibabel import nifti1

def floatArrayFromList(lst):
  n = len(lst)
  array = broccoli.floatArray(n)
  
  if isinstance(lst, numpy.ndarray):
    lst = lst.flatten()
  
  for i in range(n):
    array[i] = float(lst[i])
  return array

class Array:
  def __init__(self, data, dimensions, voxel_sizes = None):
    self.data = data
    self.dimensions = dimensions
    if voxel_sizes:
      self.voxel_sizes = voxel_sizes
    else:
      self.voxel_sizes = [1 for i in dimensions]
      
  def toFloatArray(self):
    return floatArrayFromList(self.data)
  
def arrayFromNifti(img, voxel_sizes = None):
  return Array(img.get_data(), img.shape, voxel_sizes)
    
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
      real = floatArrayFromList([c.real for c in filters[i].data])
      imag = floatArrayFromList([c.imag for c in filters[i].data])
      args.append(real)
      args.append(imag)
    broccoli.BROCCOLI_LIB.SetParametricImageRegistrationFilters(self, *args)
    
  def SetNonParametricImageRegistrationFilters(self, filters):
    args = []
    for i in range(6):
      real = floatArrayFromList([c.real for c in filters[i].data])
      imag = floatArrayFromList([c.imag for c in filters[i].data])
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
  
  ## Set constants
  NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS = 12
  
  ## Pass input parameters to BROCCOLI
  print("Setting up input parameters")
  
  BROCCOLI.SetT1Data(h_T1_Data)
  BROCCOLI.SetMNIData(h_MNI_Data)
  BROCCOLI.SetInputMNIBrainVolume(h_MNI_Brain.toFloatArray())
  BROCCOLI.SetInputMNIBrainMask(h_MNI_Brain_Mask.toFloatArray())
  
  BROCCOLI.SetInterpolationMode(broccoli.LINEAR) # Linear
  BROCCOLI.SetNumberOfIterationsForParametricImageRegistration(NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION)
  BROCCOLI.SetNumberOfIterationsForNonParametricImageRegistration(NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION)
  
  BROCCOLI.SetImageRegistrationFilterSize(h_Quadrature_Filter_Parametric_Registration[0].dimensions[0])
  BROCCOLI.SetParametricImageRegistrationFilters(h_Quadrature_Filter_Parametric_Registration)
  BROCCOLI.SetNonParametricImageRegistrationFilters(h_Quadrature_Filter_NonParametric_Registration)
  
  BROCCOLI.SetProjectionTensorMatrixFilters(h_Projection_Tensor)
  BROCCOLI.SetFilterDirections(*h_Filter_Directions)
    
  ## Set up output parameters
  print("Setting up output parameters")
  
  h_Aligned_T1_Volume = broccoli.floatArray(10)
  BROCCOLI.SetOutputAlignedT1Volume(h_Aligned_T1_Volume)
  
  h_Aligned_T1_Volume_NonParametric = broccoli.floatArray(10)
  BROCCOLI.SetOutputAlignedT1VolumeNonParametric(h_Aligned_T1_Volume_NonParametric)
  
  h_Skullstripped_T1_Volume = broccoli.floatArray(10)
  BROCCOLI.SetOutputSkullstrippedT1Volume(h_Skullstripped_T1_Volume)
  
  h_Interpolated_T1_Volume = broccoli.floatArray(10)
  BROCCOLI.SetOutputInterpolatedT1Volume(h_Interpolated_T1_Volume)
  
  h_Downsampled_Volume = broccoli.floatArray(10)
  BROCCOLI.SetOutputDownsampledVolume(h_Downsampled_Volume)
  
  h_Registration_Parameters = broccoli.floatArray(10)
  BROCCOLI.SetOutputT1MNIRegistrationParameters(h_Registration_Parameters)
  
  h_Quadrature_Filter_Response = [broccoli.cl_float2Array(10) for i in range(6)]
  BROCCOLI.SetOutputQuadratureFilterResponses(*h_Quadrature_Filter_Response)
  
  h_OutputTensor = [broccoli.floatArray(10) for i in range(6)]
  BROCCOLI.SetOutputTensorComponents(*h_OutputTensor)
  
  h_Displacement_Field = [broccoli.floatArray(10) for i in range(3)]
  BROCCOLI.SetOutputDisplacementField(*h_Displacement_Field)
  
  h_Phase_Differences = broccoli.floatArray(10)
  BROCCOLI.SetOutputPhaseDifferences(h_Phase_Differences)
  
  h_Phase_Certainties = broccoli.floatArray(10)
  BROCCOLI.SetOutputPhaseCertainties(h_Phase_Certainties)
  
  h_Phase_Gradients = broccoli.floatArray(10)
  BROCCOLI.SetOutputPhaseGradients(h_Phase_Gradients)
  
  h_Slice_Sums = broccoli.floatArray(10)
  BROCCOLI.SetOutputSliceSums(h_Slice_Sums)
  
  h_Top_Slice = broccoli.floatArray(10)
  BROCCOLI.SetOutputTopSlice(h_Top_Slice)
  
  h_A_Matrix = broccoli.floatArray(NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS * NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS)
  BROCCOLI.SetOutputAMatrix(h_A_Matrix)
  
  h_h_Vector = broccoli.floatArray(NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS)
  BROCCOLI.SetOutputHVector(h_h_Vector)

  ## Perform registration
  print("Performing registration")
  BROCCOLI.PerformRegistrationT1MNINoSkullstripWrapper()

  
if __name__ == "__main__":
  
  study = 'Cambridge'
  subject = 'sub00156'
  voxel_size = 2
  
  number_of_iterations_for_parametric_image_registration = 10
  number_of_iterations_for_nonparametric_image_registration = 15
  coarsest_scale = 8 / voxel_size
  MM_T1_Z_CUT = 30
  
  MNI_nni = nifti1.load('../../brain_templates/MNI152_T1_%dmm.nii' % voxel_size)
  MNI = arrayFromNifti(MNI_nni)
  
  MNI_brain_nii = nifti1.load('../../brain_templates/MNI152_T1_%dmm_brain.nii' % voxel_size)
  MNI_brain = arrayFromNifti(MNI_brain_nii)
  
  MNI_brain_mask_nii = nifti1.load('../../brain_templates/MNI152_T1_%dmm_brain_mask.nii' % voxel_size)
  MNI_brain_mask = arrayFromNifti(MNI_brain_mask_nii)
  
  size3 = [3, 1, 1]
  size1 = [3]
  data = [1.0, 2.9, 3.7]
  
  T1_nni = nifti1.load('../../test_data/fcon1000/classic/%s/%s/anat/mprage_skullstripped.nii.gz' % (study, subject))
  T1 = arrayFromNifti(T1_nni)
  
  registerT1MNI(T1, MNI, MNI_brain, MNI_brain_mask,
    [Array(data, size3) for i in range(3)],
    [Array(data, size3) for i in range(6)],
    [[1, 1, 1, 0, 0, 0] for i in range(6)],
    [floatArrayFromList(data) for j in range(3)],
    10,
    10,
    10,
    10,
    0,
    0
  )
