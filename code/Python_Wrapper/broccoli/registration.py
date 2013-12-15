import broccoli_common as broccoli

import matplotlib.pyplot as plot
import matplotlib.cm as cm
import numpy

from operator import mul

def flatSize(a):
  if hasattr(a, 'shape'):
    a = a.shape
  return reduce(mul, a, 1)

def plotVolume(data, sliceYrel, sliceZrel):
  sliceY = int(round(sliceYrel * data.shape[0]))

  # Data is ordered [y][x][z]
  plot.imshow(numpy.flipud(data[sliceY].transpose()), cmap = cm.Greys_r, interpolation="nearest")
  plot.draw()
  plot.figure()

  sliceZ = int(round(sliceZrel * data.shape[2])) - 1

  plot.imshow(numpy.fliplr(data[:,:,sliceZ]), cmap = cm.Greys_r, interpolation="nearest")
  plot.draw()
  plot.figure()

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
  h_T1_Data = BROCCOLI.packArray(h_T1_Data)
  h_EPI_Data = BROCCOLI.packArray(h_EPI_Data)

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
  BROCCOLI.SetFilterDirections(*[BROCCOLI.packArray(a) for a in h_Filter_Directions])

  BROCCOLI.SetCoarsestScaleEPIT1(COARSEST_SCALE)
  BROCCOLI.SetMMEPIZCUT(MM_EPI_Z_CUT)

  ## Set up output parameters
  print("Setting up output parameters...")

  h_Aligned_EPI_Volume = BROCCOLI.createOutputArray(T1_DATA_SHAPE)
  BROCCOLI.SetOutputAlignedEPIVolume(h_Aligned_EPI_Volume)

  h_Interpolated_EPI_Volume = BROCCOLI.createOutputArray(T1_DATA_SHAPE)
  BROCCOLI.SetOutputInterpolatedEPIVolume(h_Interpolated_EPI_Volume)

  h_Registration_Parameters = BROCCOLI.createOutputArray(6)
  BROCCOLI.SetOutputEPIT1RegistrationParameters(h_Registration_Parameters)

  h_Phase_Differences = BROCCOLI.createOutputArray(h_T1_Data.shape)
  BROCCOLI.SetOutputPhaseDifferences(h_Phase_Differences)

  h_Phase_Certainties = BROCCOLI.createOutputArray(h_T1_Data.shape)
  BROCCOLI.SetOutputPhaseCertainties(h_Phase_Certainties)

  h_Phase_Gradients = BROCCOLI.createOutputArray(h_T1_Data.shape)
  BROCCOLI.SetOutputPhaseGradients(h_Phase_Gradients)

  ## Perform registration
  print("Performing registration...")
  BROCCOLI.PerformRegistrationEPIT1Wrapper()

  print("Registration done, unpacking output volumes...")

  h_Interpolated_EPI_Volume = BROCCOLI.unpackOutputVolume(h_Interpolated_EPI_Volume, T1_DATA_SHAPE)
  h_Aligned_EPI_Volume = BROCCOLI.unpackOutputVolume(h_Aligned_EPI_Volume, T1_DATA_SHAPE)

  h_Phase_Differences = BROCCOLI.unpackOutputVolume(h_Phase_Differences, T1_DATA_SHAPE)
  h_Phase_Certainties = BROCCOLI.unpackOutputVolume(h_Phase_Certainties, T1_DATA_SHAPE)
  h_Phase_Gradients = BROCCOLI.unpackOutputVolume(h_Phase_Gradients, T1_DATA_SHAPE)

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
    plot.close()
    plot.show()

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
  h_T1_Data = BROCCOLI.packArray(h_T1_Data)
  h_MNI_Data = BROCCOLI.packArray(h_MNI_Data)
  h_MNI_Brain = BROCCOLI.packArray(h_MNI_Brain)
  h_MNI_Brain_Mask = BROCCOLI.packArray(h_MNI_Brain_Mask)

  h_MNI_Voxel_Sizes = [round(i) for i in h_MNI_Voxel_Sizes]
  h_T1_Voxel_Sizes = [round(i) for i in h_T1_Voxel_Sizes]

  ## Pass input parameters to BROCCOLI
  print("Setting up input parameters...")

  print("T1 size is %s" % ' x '.join([str(i) for i in h_T1_Data.shape]))
  print("MNI size is %s" % ' x '.join([str(i) for i in h_MNI_Data.shape]))

  BROCCOLI.SetT1Data(h_T1_Data, h_T1_Voxel_Sizes)
  BROCCOLI.SetMNIData(h_MNI_Data, h_MNI_Voxel_Sizes)

  BROCCOLI.SetInputMNIBrainVolume(BROCCOLI.packVolume(h_MNI_Brain))
  BROCCOLI.SetInputMNIBrainMask(BROCCOLI.packVolume(h_MNI_Brain_Mask))

  BROCCOLI.SetInterpolationMode(broccoli.LINEAR) # Linear
  BROCCOLI.SetNumberOfIterationsForParametricImageRegistration(NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION)
  BROCCOLI.SetNumberOfIterationsForNonParametricImageRegistration(NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION)

  BROCCOLI.SetImageRegistrationFilterSize(h_Quadrature_Filter_Parametric_Registration[0][0].shape[0])
  BROCCOLI.SetParametricImageRegistrationFilters(h_Quadrature_Filter_Parametric_Registration)
  BROCCOLI.SetNonParametricImageRegistrationFilters(h_Quadrature_Filter_NonParametric_Registration)

  BROCCOLI.SetProjectionTensorMatrixFilters(h_Projection_Tensor)
  BROCCOLI.SetFilterDirections(*[BROCCOLI.packArray(a) for a in h_Filter_Directions])

  BROCCOLI.SetCoarsestScaleT1MNI(COARSEST_SCALE)
  BROCCOLI.SetMMT1ZCUT(MM_T1_Z_CUT)

  ## Set up output parameters
  print("Setting up output parameters...")

  h_Aligned_T1_Volume = BROCCOLI.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputAlignedT1Volume(h_Aligned_T1_Volume)

  h_Aligned_T1_Volume_NonParametric = BROCCOLI.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputAlignedT1VolumeNonParametric(h_Aligned_T1_Volume_NonParametric)

  h_Skullstripped_T1_Volume = BROCCOLI.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputSkullstrippedT1Volume(h_Skullstripped_T1_Volume)

  h_Interpolated_T1_Volume = BROCCOLI.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputInterpolatedT1Volume(h_Interpolated_T1_Volume)

  h_Registration_Parameters = BROCCOLI.createOutputArray(12)
  BROCCOLI.SetOutputT1MNIRegistrationParameters(h_Registration_Parameters)

  h_Phase_Differences = BROCCOLI.createOutputArray(h_MNI_Data.shape)
  BROCCOLI.SetOutputPhaseDifferences(h_Phase_Differences)

  h_Phase_Certainties = BROCCOLI.createOutputArray(h_MNI_Data.shape)
  BROCCOLI.SetOutputPhaseCertainties(h_Phase_Certainties)

  h_Phase_Gradients = BROCCOLI.createOutputArray(h_MNI_Data.shape)
  BROCCOLI.SetOutputPhaseGradients(h_Phase_Gradients)

  h_Slice_Sums = BROCCOLI.createOutputArray(h_MNI_Data.shape[2])
  BROCCOLI.SetOutputSliceSums(h_Slice_Sums)

  h_Top_Slice = BROCCOLI.createOutputArray(1)
  BROCCOLI.SetOutputTopSlice(h_Top_Slice)

  h_A_Matrix = BROCCOLI.createOutputArray(NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS * NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS)
  BROCCOLI.SetOutputAMatrix(h_A_Matrix)

  h_h_Vector = BROCCOLI.createOutputArray(NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS)
  BROCCOLI.SetOutputHVector(h_h_Vector)

  ## Perform registration
  print("Performing registration...")
  BROCCOLI.PerformRegistrationT1MNINoSkullstripWrapper()

  print("Registration done, unpacking output volumes...")

  h_Aligned_T1_Volume = BROCCOLI.unpackOutputVolume(h_Aligned_T1_Volume, MNI_DATA_SHAPE)
  h_Interpolated_T1_Volume = BROCCOLI.unpackOutputVolume(h_Interpolated_T1_Volume, MNI_DATA_SHAPE)
  h_Aligned_T1_Volume_NonParametric = BROCCOLI.unpackOutputVolume(h_Aligned_T1_Volume_NonParametric, MNI_DATA_SHAPE)
  h_Skullstripped_T1_Volume = BROCCOLI.unpackOutputVolume(h_Skullstripped_T1_Volume, MNI_DATA_SHAPE)
  h_Phase_Differences = BROCCOLI.unpackOutputVolume(h_Phase_Differences, MNI_DATA_SHAPE)
  h_Phase_Certainties = BROCCOLI.unpackOutputVolume(h_Phase_Certainties, MNI_DATA_SHAPE)
  h_Phase_Gradients = BROCCOLI.unpackOutputVolume(h_Phase_Gradients, MNI_DATA_SHAPE)

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
    plot.close()
    plot.show()

  return (h_Aligned_T1_Volume, h_Aligned_T1_Volume_NonParametric, h_Skullstripped_T1_Volume, h_Interpolated_T1_Volume,
          h_Registration_Parameters, h_Phase_Differences, h_Phase_Certainties, h_Phase_Gradients, h_Slice_Sums, h_Top_Slice, h_A_Matrix, h_h_Vector)
