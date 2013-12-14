from nipype.interfaces.base import TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File, Directory, isdefined, traits
from nipype.utils.filemanip import split_filename

import scipy.io
import scipy.signal
import os
import os.path as op
import nibabel as nb
import numpy as np

import broccoli
import base

class FirstLevelAnalysisInputSpec(base.BroccoliInputSpec):
    fMRI_file = File(exists=True, mandatory=True)
    MNI_file = File(exists=True, mandatory=True)
    MNI_brain_file = File(exists=True)
    MNI_brain_mask_file = File(exists=True)
    T1_file = File(exists=True, mandatory=True)
    GLM_path = Directory(exists=True, mandatory=True)
    
    filters_parametric = File(exists=True, mandatory=True,
                              desc='Matlab file with filters for parametric registration')
    filters_nonparametric = File(exists=True, mandatory=True,
                                 desc='Matlab file with filters for nonparametric registration')
    
    iterations_parametric = traits.Int(15)
    iterations_nonparametric = traits.Int(10)
    iterations_motion_correction = traits.Int(3)
    
    beta_space = traits.Enum('EPI', 'MNI', desc='either EPI or MNI', usedefault=True)
    
    regress_motion = traits.Bool()
    regress_confounds = traits.Bool()
    use_temporal_derivatives = traits.Bool()
    
    EPI_smoothing = traits.Float(5.5)
    AR_smoothing = traits.Float(7.0)
    
class FirstLevelAnalysisOutputSpec(TraitedSpec):
    pass
  
class FirstLevelAnalysis(BaseInterface):
    input_spec = FirstLevelAnalysisInputSpec
    output_spec = FirstLevelAnalysisOutputSpec
    
    def load_regressor(self, filename, samples):
        d = np.loadtxt(filename)
        hr = np.empty(shape=(samples * len(d),))
        tr = 2
        
        for row in d:
            start = int(round(row[0] * samples / tr))
            duration = int(round(row[1] * samples / tr))
            for i in range(duration):
              hr[start + i] = row[2]
        
        return scipy.signal.decimate(hr, samples)
        
    
    def load_regressors(self):
        files = [f for f in os.listdir(self.inputs.GLM_path) if op.isfile(op.join(self.inputs.GLM_path, f))]
        data = [self.load_regressor(f, 100000) for f in files]
        return np.array(data)
        
    def _run_interface(self, runtime):
        
        MNI, MNI_brain, MNI_brain_mask, MNI_voxel_sizes = broccoli.load_MNI_templates(self.inputs.MNI_file, self.inputs.MNI_brain_file, self.inputs.mni_brain_mask_file)
        fMRI, fMRI_voxel_sizes = broccoli.load_EPI(self.inputs.fMRI_file, only_volume=False)
        T1, T1_voxel_sizes = broccoli.load_T1(self.inputs.T1_file)
        
        filters_parametric_mat = scipy.io.loadmat(self.inputs.filters_parametric)
        filters_nonparametric_mat = scipy.io.loadmat(self.inputs.filters_nonparametric)
        filters_parametric = [filters_parametric_mat['f%d_parametric_registration' % (i+1)] for i in range(3)]
        filters_nonparametric = [filters_nonparametric_mat['f%d_nonparametric_registration' % (i+1)] for i in range(6)]
        projection_tensor = [filters_nonparametric_mat['m%d' % (i+1)][0] for i in range(6)]
        filter_directions = [filters_nonparametric_mat['filter_directions_%s' % d][0] for d in ['x', 'y', 'z']]
        
        X_GLM = self.load_regressors()
        xtxxt_GLM = np.linalg.inv(X_GLM.transpose() * X_GLM) * X_GLM.transpose()

        confounds = 1
        if self.inputs.regress_confounds:
            confounds = np.loadtxt(self.inputs.confounds_file)

        contrasts = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
        ctxtxc_GLM = [contrasts[i:i+1].transpose() * np.linalg.inv(X_GLM.transpose() * X_GLM) * contrasts[0:1] for i in range(len(contrasts))]
        
        broccoli.performFirstLevelAnalysis(
            fMRI, fMRI_voxel_sizes, T1, T1_voxel_sizes, MNI, MNI_brain, MNI_brain_mask, MNI_voxel_sizes,
            filters_parametric, filters_nonparametric, projection_tensor, filter_directions,
            self.inputs.iterations_parametric, self.inputs.iterations_nonparametric, self.inputs.iterations_motion_correction, 4, 4, 0, 0,
            self.inputs.regress_motion, self.inputs.EPI_smoothing, self.inputs.AR_smoothing, X_GLM, xtxxt_GLM.transpose(), contrasts, ctxtxc_GLM,
            self.inputs.use_temporal_derivatives, getattr(broccoli, self.inputs.beta_space), confounds, self.inputs.regress_confounds,
            self.inputs.opencl_platform, self.inputs.opencl_device,
        )
      
        return runtime
