import broccoli_common as broccoli

import matplotlib.pyplot as plot
import matplotlib.cm as cm

from operator import mul

def flatSize(a):
  if hasattr(a, 'shape'):
    a = a.shape
  return reduce(mul, a, 1)

def plotVolume(data, sliceYrel, sliceZrel):
  sliceY = int(round(sliceYrel * data.shape[0]))

  # Data is first ordered [y][x][z]
  plot.imshow(numpy.flipud(data[sliceY].transpose()), cmap = cm.Greys_r, interpolation="nearest")
  plot.show()

  sliceZ = int(round(sliceZrel * data.shape[2])) - 1

  # We want it ordered [z][x][y]
  data_t = data.transpose()
  plot.imshow(numpy.fliplr(data_t[sliceZ]).transpose(), cmap = cm.Greys_r, interpolation="nearest")
  plot.show()


def registerEPIT1(
    h_EPI_Data,
    h_EPI_Voxel_Sizes,
    h_T1_Data,          # Array
    h_T1_Voxel_Sizes,   # 3 elements
    h_Quadrature_Filter_Parametric_Registration,            # 3 elements, complex arrays
    h_Quadrature_Filter_NonParametric_Registration,         # 6 elements, complex arrays
    h_Projection_Tensor,             # 6 elements
    h_Filter_Directions,             # 3 elements
    NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION,     # int
    COARSEST_SCALE,         # int
    MM_EPI_Z_CUT,            # int
    OPENCL_PLATFORM,        # int
    OPENCL_DEVICE,          # int
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

  ## Set constants
  T1_DATA_SHAPE = h_T1_Data.shape
  EPI_DATA_SHAPE = h_EPI_Data.shape
  EPI_INTERPOLATED_DATA_SHAPE = [int(round(float(EPI_DATA_SHAPE[i]) * h_EPI_Voxel_Sizes[i] / h_T1_Voxel_Sizes[i])) for i in range(3)]

  ## Make all arrays contiguous
  h_T1_Data = broccoli.packArray(h_T1_Data)
  h_EPI_Data = broccoli.packArray(h_EPI_Data)

  h_EPI_Voxel_Sizes = [round(i) for i in h_EPI_Voxel_Sizes]
  h_T1_Voxel_Sizes = [round(i) for i in h_T1_Voxel_Sizes]

  ## Pass input parameters to BROCCOLI
  print("Setting up input parameters...")

  print("EPI size is %s" % ' x '.join([str(i) for i in h_EPI_Data.shape]))
  print("T1 size is %s" % ' x '.join([str(i) for i in h_T1_Data.shape]))

  BROCCOLI.SetEPIData(h_EPI_Data, h_EPI_Voxel_Sizes)
  BROCCOLI.SetT1Data(h_T1_Data, h_T1_Voxel_Sizes)

  BROCCOLI.SetInterpolationMode(broccoli.LINEAR) # Linear
  BROCCOLI.SetNumberOfIterationsForParametricImageRegistration(NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION)

  BROCCOLI.SetImageRegistrationFilterSize(h_Quadrature_Filter_Parametric_Registration[0][0].shape[0])
  BROCCOLI.SetParametricImageRegistrationFilters(h_Quadrature_Filter_Parametric_Registration)
  BROCCOLI.SetNonParametricImageRegistrationFilters(h_Quadrature_Filter_NonParametric_Registration)

  BROCCOLI.SetProjectionTensorMatrixFilters(h_Projection_Tensor)
  BROCCOLI.SetFilterDirections(*[broccoli.packArray(a) for a in h_Filter_Directions])

  BROCCOLI.SetCoarsestScaleEPIT1(COARSEST_SCALE)
  BROCCOLI.SetMMEPIZCUT(MM_EPI_Z_CUT)

  ## Set up output parameters
  print("Setting up output parameters...")

  h_Aligned_EPI_Volume = broccoli.createOutputArray(T1_DATA_SHAPE)
  BROCCOLI.SetOutputAlignedEPIVolume(h_Aligned_EPI_Volume)

  h_Interpolated_EPI_Volume = broccoli.createOutputArray(T1_DATA_SHAPE)
  BROCCOLI.SetOutputInterpolatedEPIVolume(h_Interpolated_EPI_Volume)

  h_Registration_Parameters = broccoli.createOutputArray(6)
  BROCCOLI.SetOutputEPIT1RegistrationParameters(h_Registration_Parameters)

  h_Phase_Differences = broccoli.createOutputArray(h_T1_Data.shape)
  BROCCOLI.SetOutputPhaseDifferences(h_Phase_Differences)

  h_Phase_Certainties = broccoli.createOutputArray(h_T1_Data.shape)
  BROCCOLI.SetOutputPhaseCertainties(h_Phase_Certainties)

  h_Phase_Gradients = broccoli.createOutputArray(h_T1_Data.shape)
  BROCCOLI.SetOutputPhaseGradients(h_Phase_Gradients)

  ## Perform registration
  print("Performing registration...")
  BROCCOLI.PerformRegistrationEPIT1Wrapper()

  print("Registration done, unpacking output volumes...")

  h_Interpolated_EPI_Volume = broccoli.unpackOutputVolume(h_Interpolated_EPI_Volume, T1_DATA_SHAPE)
  h_Aligned_EPI_Volume = broccoli.unpackOutputVolume(h_Aligned_EPI_Volume, T1_DATA_SHAPE)

  h_Phase_Differences = broccoli.unpackOutputVolume(h_Phase_Differences, T1_DATA_SHAPE)
  h_Phase_Certainties = broccoli.unpackOutputVolume(h_Phase_Certainties, T1_DATA_SHAPE)
  h_Phase_Gradients = broccoli.unpackOutputVolume(h_Phase_Gradients, T1_DATA_SHAPE)

  print(h_Registration_Parameters)

  if show_results:
    plot_results = (
      h_Interpolated_EPI_Volume,
      h_Aligned_EPI_Volume,
      h_T1_Data,
    )

    sliceY = 0.45
    sliceZ = 0.62

    for r in plot_results:
      plotVolume(r, sliceY, sliceZ)

  return (h_Aligned_EPI_Volume, h_Interpolated_EPI_Volume,
          h_Registration_Parameters, h_Phase_Differences, h_Phase_Certainties, h_Phase_Gradients)

def registerT1MNI(
    h_T1_Data,          # Array
    h_T1_Voxel_Sizes,   # 3 elements
    h_MNI_Data,         # Array
    h_MNI_Voxel_Sizes,   # 3 elements
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

  ## Set constants
  NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS = 12
  MNI_DATA_SIZE = flatSize(h_MNI_Data)

  MNI_DATA_SHAPE = h_MNI_Data.shape
  T1_DATA_SHAPE = h_T1_Data.shape
  T1_INTERPOLATED_DATA_SHAPE = [int(round(float(T1_DATA_SHAPE[i]) * h_T1_Voxel_Sizes[i] / h_MNI_Voxel_Sizes[i])) for i in range(3)]

  ## Make all arrays contiguous
  h_T1_Data = broccoli.packArray(h_T1_Data)
  h_MNI_Data = broccoli.packArray(h_MNI_Data)
  h_MNI_Brain = broccoli.packArray(h_MNI_Brain)
  h_MNI_Brain_Mask = broccoli.packArray(h_MNI_Brain_Mask)

  h_MNI_Voxel_Sizes = [round(i) for i in h_MNI_Voxel_Sizes]
  h_T1_Voxel_Sizes = [round(i) for i in h_T1_Voxel_Sizes]

  ## Pass input parameters to BROCCOLI
  print("Setting up input parameters...")

  print("T1 size is %s" % ' x '.join([str(i) for i in h_T1_Data.shape]))
  print("MNI size is %s" % ' x '.join([str(i) for i in h_MNI_Data.shape]))

  BROCCOLI.SetT1Data(h_T1_Data, h_T1_Voxel_Sizes)
  BROCCOLI.SetMNIData(h_MNI_Data, h_MNI_Voxel_Sizes)

  BROCCOLI.SetInputMNIBrainVolume(broccoli.packVolume(h_MNI_Brain))
  BROCCOLI.SetInputMNIBrainMask(broccoli.packVolume(h_MNI_Brain_Mask))

  BROCCOLI.SetInterpolationMode(broccoli.LINEAR) # Linear
  BROCCOLI.SetNumberOfIterationsForParametricImageRegistration(NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION)
  BROCCOLI.SetNumberOfIterationsForNonParametricImageRegistration(NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION)

  BROCCOLI.SetImageRegistrationFilterSize(h_Quadrature_Filter_Parametric_Registration[0][0].shape[0])
  BROCCOLI.SetParametricImageRegistrationFilters(h_Quadrature_Filter_Parametric_Registration)
  BROCCOLI.SetNonParametricImageRegistrationFilters(h_Quadrature_Filter_NonParametric_Registration)

  BROCCOLI.SetProjectionTensorMatrixFilters(h_Projection_Tensor)
  BROCCOLI.SetFilterDirections(*[broccoli.packArray(a) for a in h_Filter_Directions])

  BROCCOLI.SetCoarsestScaleT1MNI(COARSEST_SCALE)
  BROCCOLI.SetMMT1ZCUT(MM_T1_Z_CUT)

  ## Set up output parameters
  print("Setting up output parameters...")

  h_Aligned_T1_Volume = broccoli.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputAlignedT1Volume(h_Aligned_T1_Volume)

  h_Aligned_T1_Volume_NonParametric = broccoli.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputAlignedT1VolumeNonParametric(h_Aligned_T1_Volume_NonParametric)

  h_Skullstripped_T1_Volume = broccoli.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputSkullstrippedT1Volume(h_Skullstripped_T1_Volume)

  h_Interpolated_T1_Volume = broccoli.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputInterpolatedT1Volume(h_Interpolated_T1_Volume)

  h_Registration_Parameters = broccoli.createOutputArray(12)
  BROCCOLI.SetOutputT1MNIRegistrationParameters(h_Registration_Parameters)

  h_Phase_Differences = broccoli.createOutputArray(h_MNI_Data.shape)
  BROCCOLI.SetOutputPhaseDifferences(h_Phase_Differences)

  h_Phase_Certainties = broccoli.createOutputArray(h_MNI_Data.shape)
  BROCCOLI.SetOutputPhaseCertainties(h_Phase_Certainties)

  h_Phase_Gradients = broccoli.createOutputArray(h_MNI_Data.shape)
  BROCCOLI.SetOutputPhaseGradients(h_Phase_Gradients)

  h_Slice_Sums = broccoli.createOutputArray(h_MNI_Data.shape[2])
  BROCCOLI.SetOutputSliceSums(h_Slice_Sums)

  h_Top_Slice = broccoli.createOutputArray(1)
  BROCCOLI.SetOutputTopSlice(h_Top_Slice)

  h_A_Matrix = broccoli.createOutputArray(NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS * NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS)
  BROCCOLI.SetOutputAMatrix(h_A_Matrix)

  h_h_Vector = broccoli.createOutputArray(NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS)
  BROCCOLI.SetOutputHVector(h_h_Vector)

  ## Perform registration
  print("Performing registration...")
  BROCCOLI.PerformRegistrationT1MNINoSkullstripWrapper()

  print("Registration done, unpacking output volumes...")

  h_Aligned_T1_Volume = broccoli.unpackOutputVolume(h_Aligned_T1_Volume, MNI_DATA_SHAPE)
  h_Interpolated_T1_Volume = broccoli.unpackOutputVolume(h_Interpolated_T1_Volume, MNI_DATA_SHAPE)
  h_Aligned_T1_Volume_NonParametric = broccoli.unpackOutputVolume(h_Aligned_T1_Volume_NonParametric, MNI_DATA_SHAPE)
  h_Skullstripped_T1_Volume = broccoli.unpackOutputVolume(h_Skullstripped_T1_Volume, MNI_DATA_SHAPE)
  h_Phase_Differences = broccoli.unpackOutputVolume(h_Phase_Differences, MNI_DATA_SHAPE)
  h_Phase_Certainties = broccoli.unpackOutputVolume(h_Phase_Certainties, MNI_DATA_SHAPE)
  h_Phase_Gradients = broccoli.unpackOutputVolume(h_Phase_Gradients, MNI_DATA_SHAPE)

  print(h_Registration_Parameters)

  if show_results:
    plot_results = (
      h_Interpolated_T1_Volume,
      h_Aligned_T1_Volume,
      h_MNI_Brain,
      h_Aligned_T1_Volume_NonParametric,
    )

    sliceY = 0.45
    sliceZ = 0.47

    for r in plot_results:
      plotVolume(r, sliceY, sliceZ)

  return (h_Aligned_T1_Volume, h_Aligned_T1_Volume_NonParametric, h_Skullstripped_T1_Volume, h_Interpolated_T1_Volume,
          h_Registration_Parameters, h_Phase_Differences, h_Phase_Certainties, h_Phase_Gradients, h_Slice_Sums, h_Top_Slice, h_A_Matrix, h_h_Vector)
