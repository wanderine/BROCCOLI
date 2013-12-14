from nipype.interfaces.base import TraitedSpec, BaseInterface, File, isdefined, traits
from nipype.utils.filemanip import split_filename

import scipy.io
import os.path as op
import nibabel as nb
import numpy as np

import broccoli
import base

class MotionCorrectionInputSpec(base.BroccoliInputSpec):
    filters_parametric = File(exists=True, mandatory=True,
                              desc='Matlab file with filters for parametric registration')
    fmri_file = File(exists=True, desc='Input fMRI file', mandatory = True)

class MotionCorrectionOutputSpec(TraitedSpec):
    motion_corrected_fmri_file = File()

class MotionCorrection(BaseInterface):
    input_spec = MotionCorrectionInputSpec
    output_spec = MotionCorrectionOutputSpec

    def _run_interface(self, runtime):
        fMRI_nii = nb.load(self.inputs.fmri_file)
        fMRI_data = fMRI_nii.get_data()

        filters_parametric_mat = scipy.io.loadmat(self.inputs.filters_parametric)
        filters_parametric = [filters_parametric_mat['f%d_parametric_registration' % (i+1)] for i in range(3)]

        (Motion_Corrected_fMRI_Volumes, Motion_Parameters, Phase_Differences, Phase_Certainties, Phase_Gradients) = broccoli.performMotionCorrection(
            fMRI_data, filters_parametric, 10, self.inputs.opencl_platform, self.inputs.opencl_device, self.inputs.show_results,
        )

        corrected_fMRI_nni = nb.Nifti1Image(Motion_Corrected_fMRI_Volumes, None, fMRI_nni.get_header())
        nb.save(corrected_fMRI_nni, self.inputs.base_name + '_motion_corrected.nii')

        return runtime
      
      
    def _list_outputs(self):
        outputs = self.output_spec().get()
        for k in outputs.keys():
            outputs[k] = self.inputs.base_name + '_' + k.replace('_t1_file', '.nii')
        return outputs