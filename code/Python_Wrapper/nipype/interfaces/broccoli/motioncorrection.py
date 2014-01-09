from nipype.interfaces.base import TraitedSpec, BaseInterface, File, isdefined, traits
from nipype.utils.filemanip import split_filename

import scipy.io
import os.path as op
import nibabel as nb
import numpy as np

import broccoli
from base import BroccoliInputSpec, BroccoliInterface

class MotionCorrectionInputSpec(BroccoliInputSpec):
    filters_parametric = File(exists=True, mandatory=True,
                              desc='Matlab file with filters for parametric registration')
    fMRI_file = File(exists=True, desc='Input fMRI file', mandatory = True)

class MotionCorrectionOutputSpec(TraitedSpec):
    motion_corrected_fMRI_file = File()

class MotionCorrection(BroccoliInterface):
    input_spec = MotionCorrectionInputSpec
    output_spec = MotionCorrectionOutputSpec

    def _run_interface(self, runtime):
        fMRI_data, fMRI_voxel_sizes = broccoli.load_EPI(self.inputs.fMRI_file, only_volume=False)

        filters_parametric_mat = scipy.io.loadmat(self.inputs.filters_parametric)
        filters_parametric = [filters_parametric_mat['f%d_parametric_registration' % (i+1)] for i in range(3)]

        (Motion_Corrected_fMRI_Volumes, Motion_Parameters, Phase_Differences, Phase_Certainties, Phase_Gradients) = broccoli.performMotionCorrection(
            fMRI_data, [int(round(s)) for s in fMRI_voxel_sizes], filters_parametric, 10, self.inputs.opencl_platform, self.inputs.opencl_device, self.inputs.show_results,
        )

        fMRI_nii = nb.load(self.inputs.fMRI_file)
        corrected_fMRI_nii = nb.Nifti1Image(Motion_Corrected_fMRI_Volumes, None, fMRI_nii.get_header())
        nb.save(corrected_fMRI_nii, self._get_output_filename('_motion_corrected.nii'))

        return runtime
      
      
    def _list_outputs(self):
        outputs = self.output_spec().get()
        for k in outputs.keys():
            outputs[k] = self._get_output_filename(k.replace('_fMRI_file', '.nii'))
        return outputs