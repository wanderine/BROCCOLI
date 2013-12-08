from nipype.interfaces.base import TraitedSpec, BaseInterface, File
from nipype.utils.filemanip import split_filename
import scipy.io
import os.path as op
import nibabel as nb
import numpy as np

import broccoli

class CommonRegistrationInputSpec(TraitedSpec):
    filters_parametric = File(exists=True, mandatory=True,
                              desc='Matlab file with filters for parametric registration')
    filters_nonparametric = File(exists=True, mandatory=True,
                                 desc='Matlab file with filters for nonparametric registration')

class RegistrationT1MNIInputSpec(CommonRegistrationInputSpec):
    t1_file = File(exists=True, desc="Input T1 file", mandatory=True)
    mni_file = File(exists=True, desc="Input MNI file", mandatory=True)
    mni_brain_file = File(exists=True, desc="Input MNI Brain file")
    mni_brain_mask_file = File(exists=True, desc="Input MNI Brain Mask file")

class RegistrationT1MNIOutputSpec(TraitedSpec):
    aligned_t1_file = File(exists=False)
    interpolated_t1_file = File(exists=False)

class RegistrationT1MNI(BaseInterface):
    input_spec = RegistrationT1MNIInputSpec
    output_spec = RegistrationT1MNIOutputSpec

    def _run_interface(self, runtime):
        T1_data, T1_voxel_sizes = broccoli.load_T1(self.inputs.t1_file)
        MNI_data, MNI_brain_data, MNI_brain_mask_data, MNI_voxel_sizes = broccoli.load_MNI_templates(self.inputs.mni_file)
        print(MNI_voxel_sizes)

        filters_parametric_mat = scipy.io.loadmat(self.inputs.filters_parametric)
        filters_nonparametric_mat = scipy.io.loadmat(self.inputs.filters_nonparametric)

        filters_parametric = [filters_parametric_mat['f%d_parametric_registration' % (i+1)] for i in range(3)]
        filters_nonparametric = [filters_nonparametric_mat['f%d_nonparametric_registration' % (i+1)] for i in range(6)]

        projection_tensor = [filters_nonparametric_mat['m%d' % (i+1)][0] for i in range(6)]
        filter_directions = [filters_nonparametric_mat['filter_directions_%s' % d][0] for d in ['x', 'y', 'z']]

        (Aligned_T1_Volume, Aligned_T1_Volume_NonParametric, Skullstripped_T1_Volume, Interpolated_T1_Volume,
        Registration_Parameters, Phase_Differences, Phase_Certainties, Phase_Gradients, Slice_Sums, Top_Slice, A_Matrix, h_Vector) = broccoli.registerT1MNI(
            T1_data, T1_voxel_sizes,
            MNI_data, MNI_voxel_sizes, MNI_brain_data, MNI_brain_mask_data,
            filters_parametric, filters_nonparametric, projection_tensor, filter_directions,
            10, 15, coarsest_scale, 30, 0, 0, False,
        )

        MNI_nni = nb.load(self.inputs.mni_file)
        aligned_T1_nni = nb.Nifti1Image(Aligned_T1_Volume, None, MNI_nni.get_header())
        nb.save(aligned_T1_nni, self.outputs.aligned_t1_file)

        interpolated_T1_nni = nb.Nifti1Image(Interpolated_T1_Volume, None, MNI_nni.get_header())
        nb.save(interpolated_T1_nni, self.outputs.interpolated_t1_file)

        return runtime