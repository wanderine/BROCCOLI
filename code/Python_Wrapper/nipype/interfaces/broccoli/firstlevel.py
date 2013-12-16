from nipype.interfaces.base import TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File, Directory, isdefined, traits
from nipype.utils.filemanip import split_filename

import scipy.io
import scipy.signal
import os
import os.path as op
import nibabel as nb
import numpy as np

import broccoli
from base import BroccoliInputSpec, BroccoliInterface

class FirstLevelAnalysisInputSpec(BroccoliInputSpec):
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
    
    iterations_parametric = traits.Int(15, usedefault=True)
    iterations_nonparametric = traits.Int(10, usedefault=True)
    iterations_motion_correction = traits.Int(3, usedefault=True)
    
    beta_space = traits.Enum('EPI', 'MNI', desc='either EPI or MNI', usedefault=True)
    
    regress_motion = traits.Bool(usedefault=True)
    regress_confounds = traits.Bool(usedefault=True)
    use_temporal_derivatives = traits.Bool(usedefault=True)
    
    EPI_smoothing = traits.Float(5.5, usedefault=True)
    AR_smoothing = traits.Float(7.0, usedefault=True)
    
class FirstLevelAnalysisOutputSpec(TraitedSpec):
    statistical_map = File()
  
class FirstLevelAnalysis(BroccoliInterface):
    input_spec = FirstLevelAnalysisInputSpec
    output_spec = FirstLevelAnalysisOutputSpec
    
    def load_regressor(self, filename, st, samples):
        d = np.loadtxt(filename)
        hr = np.zeros(samples * st)
        tr = 2
        
        for row in d:
            start = int(round(row[0] * samples / tr))
            duration = int(round(row[1] * samples / tr))
            for i in range(duration):
                hr[start + i] = row[2]
        
        print(hr.shape)
        print(np.count_nonzero(hr))
        print(hr)
        lr = scipy.signal.decimate(hr, samples)
        return lr
    
    def load_regressors(self, st):
        files = [f for f in os.listdir(self.inputs.GLM_path) if op.isfile(op.join(self.inputs.GLM_path, f))]
        data = [self.load_regressor(op.join(self.inputs.GLM_path, f), st, 10) for f in files]
        return np.array(data).transpose()
        
    def _run_interface(self, runtime):
        
        MNI, MNI_brain, MNI_brain_mask, MNI_voxel_sizes = broccoli.load_MNI_templates(self.inputs.MNI_file, self.inputs.MNI_brain_file, self.inputs.MNI_brain_mask_file)
        fMRI, fMRI_voxel_sizes = broccoli.load_EPI(self.inputs.fMRI_file, only_volume=False)
        T1, T1_voxel_sizes = broccoli.load_T1(self.inputs.T1_file)
        
        filters_parametric_mat = scipy.io.loadmat(self.inputs.filters_parametric)
        filters_nonparametric_mat = scipy.io.loadmat(self.inputs.filters_nonparametric)
        filters_parametric = [filters_parametric_mat['f%d_parametric_registration' % (i+1)] for i in range(3)]
        filters_nonparametric = [filters_nonparametric_mat['f%d_nonparametric_registration' % (i+1)] for i in range(6)]
        projection_tensor = [filters_nonparametric_mat['m%d' % (i+1)][0] for i in range(6)]
        filter_directions = [filters_nonparametric_mat['filter_directions_%s' % d][0] for d in ['x', 'y', 'z']]
        
        X_GLM = self.load_regressors(fMRI.shape[3])
        xtx = np.linalg.inv(np.dot(X_GLM.T, X_GLM))
        # print(xtx)
        xtxxt_GLM = xtx.dot(X_GLM.T)

        confounds = 1
        if self.inputs.regress_confounds:
            confounds = np.loadtxt(self.inputs.confounds_file)

        contrasts = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
        ctxtxc_GLM = [contrasts[i:i+1].dot(xtx).dot(contrasts[i:i+1].T) for i in range(len(contrasts))]
        
        fMRI_voxel_sizes = [int(round(v)) for v in T1_voxel_sizes]
        T1_voxel_sizes = [int(round(v)) for v in T1_voxel_sizes]
        MNI_voxel_sizes = [int(round(v)) for v in T1_voxel_sizes]
        
        
        
        statistical_maps = broccoli.performFirstLevelAnalysis(
            fMRI, fMRI_voxel_sizes, T1, T1_voxel_sizes, MNI, MNI_brain, MNI_brain_mask, MNI_voxel_sizes,
            filters_parametric, filters_nonparametric, projection_tensor, filter_directions,
            self.inputs.iterations_parametric, self.inputs.iterations_nonparametric, self.inputs.iterations_motion_correction, 4, 4, 0, 0,
            self.inputs.regress_motion, self.inputs.EPI_smoothing, self.inputs.AR_smoothing, X_GLM, xtxxt_GLM.transpose(), contrasts, ctxtxc_GLM,
            self.inputs.use_temporal_derivatives, getattr(broccoli, self.inputs.beta_space), confounds, self.inputs.regress_confounds,
            self.inputs.opencl_platform, self.inputs.opencl_device, self.inputs.show_results,
        )
        
        
        EPI_nni = nb.load(self.inputs.fMRI_file)
        aligned_EPI_nni = nb.Nifti1Image(statistical_maps, None, EPI_nni.get_header())
        nb.save(aligned_EPI_nni, self._get_output_filename('statistical_map.nii'))
      
        return runtime
            
    def _list_outputs(self):
        outputs = self.output_spec().get()
        for k in outputs.keys():
            outputs[k] = self._get_output_filename(k + '.nii')
        return outputs
