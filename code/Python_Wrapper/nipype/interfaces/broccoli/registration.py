

from nipype.interfaces.base import TraitedSpec, BaseInterface, File
from nipype.utils.filemanip import split_filename
import scipy.io
import os.path as op
import nibabel as nb
import numpy as np

from RegisterT1MNI import RegisterT1MNI

class CommonRegistrationInputSpec(TraitedSpec):
    filters_parametric = File(exists=True, mandatory=True, desc='Matlab file with filters for parametric registration')
    filters_nonparametric = File(exists=True, mandatory=True, desc='Matlab file with filters for nonparametric registration')
  
class RegistrationT1MNIInputSpec(CommonRegistrationInputSpec):
    t1_file = File(exists=True, desc="Input T1 file")
    mni_file = File(exists=True, desc="Input MNI file")
    mni_brain_file = File(exists=True, desc="Input MNI Brain file")
    mni_brain_mask_file = File(exists=True, desc="Input MNI Brain Mask file")
  
  
class RegistrationT1MNIOutputSpec(TraitedSpec):
  out_file = File(exists=True)
  
class RegistrationT1MNI(BaseInterface):
    input_spec = RegistrationT1MNIInputSpec
    output_spec = RegistrationT1MNIOutputSpec

    def _run_interface(self, runtime):
        T1_nni = nb.load(self.inputs.t1_file)
        T1_data = T1_nni.get_data()
        T1_voxel_sizes = T1_nni.get_header()['pixdim'][1:4]
        
        MNI_nni = nb.load(self.inputs.mni_file)
        MNI_data = MNI_nni.get_data()
        MNI_brain_nni = nb.load(self.inputs.mni_brain_file)
        MNI_brain = MNI_brain_nni.get_data()
        MNI_brain_mask_nni = nb.load(self.inputs.mni_brain_mask_file)
        MNI_brain_bask = MNI_brain_mask_nni.get_data()
        MNI_voxel_sizes = MNI_nni.get_header()['pixdim'][1:4]
            
        filters_parametric_mat = scipy.io.loadmat("../Matlab_Wrapper/filters_for_parametric_registration.mat")
        filters_nonparametric_mat = scipy.io.loadmat("../Matlab_Wrapper/filters_for_nonparametric_registration.mat")
        
        filters_parametric = [filters_parametric_mat['f%d_parametric_registration' % (i+1)] for i in range(3)]
        filters_nonparametric = [filters_nonparametric_mat['f%d_nonparametric_registration' % (i+1)] for i in range(6)]
        
        projection_tensor = [filters_nonparametric_mat['m%d' % (i+1)][0] for i in range(6)]
        filter_directions = [filters_nonparametric_mat['filter_directions_%s' % d][0] for d in ['x', 'y', 'z']]
                
        results = RegisterT1MNI(
            T1_data, T1_voxel_sizes,
            MNI_data, MNI_voxel_sizes, MNI_brain, MNI_brain_bask,
            filters_parametric, filters_nonparametric, projection_tensor, filter_directions,
            10, 15, round(8 / MNI_voxel_sizes[0]), 30, 0, 0
        )
        
        return runtime