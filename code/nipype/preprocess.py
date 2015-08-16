# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The BROCCOLI module provides classes for interfacing with the `BROCCOLI
<http://github.com/wanderine/BROCCOLI>`_ command line tools.

    Change directory to provide relative paths for doctests
    >>> import os
    >>> filepath = os.path.dirname( os.path.realpath( __file__ ) )
    >>> datadir = os.path.realpath(os.path.join(filepath, '../../testing/data'))
    >>> os.chdir(datadir)
"""

import os
import os.path as op
import warnings

import numpy as np

from nipype.interfaces.broccoli.base import BROCCOLICommand, BROCCOLICommandInputSpec, BROCCOLICommandOutputSpec
from nipype.interfaces.base import (TraitedSpec, File, InputMultiPath,
                                    OutputMultiPath, Undefined, traits,
                                    isdefined, OutputMultiPath)
from nipype.utils.filemanip import split_filename

from nibabel import load


warn = warnings.warn
warnings.filterwarnings('always', category=UserWarning)



class RegisterTwoVolumesInputSpec(BROCCOLICommandInputSpec):
    in_file = File(desc='input volume to align',
                   argstr='%s',
                   position=0,
                   mandatory=True,
                   exists=True,
                   copyfile=False)

    reference = File(desc='reference volume to align to',
                   argstr='%s',
                   position=1,
                   mandatory=True,
                   exists=True,
                   copyfile=False)


    iterationslinear = traits.Int(argstr='-iterationslinear %d', desc='Number of iterations for linear registration')

    iterationsnonlinear = traits.Int(argstr='-iterationsnonlinear %d', desc='Number of iterations for non-linear registration')

    sigma = traits.Float(argstr='-sigma %s', desc='Amount of Gaussian smoothing for regularization of displacement field (default 5.0)')

    mask = traits.Str(argstr='-mask %s', desc='Mask to apply after linear and non-linear registration')

    maskoriginal = traits.Str(argstr='-maskoriginal %s', desc='Mask to apply on original volume after (inverted) linear registration (but after interpolation to reference volume size)')

    saveinterpolated = traits.Bool(argstr='-saveinterpolated', desc='Save input volume after interpolation to reference volume size')

class RegisterTwoVolumesOutputSpec(TraitedSpec):
    interpolated_file = File(
        desc="path/name of volume interpolated to reference volume size (if generated)")
    aligned_linear_file = File(
        desc="path/name of linearly aligned volume (if generated)")
    aligned_nonlinear_file = File(
        desc="path/name of non-linearly aligned volume (if generated)")
    masked_file = File(
        desc="path/name of masked registered volume (if generated)")


class RegisterTwoVolumes(BROCCOLICommand):
    """This function performs linear and non-linear registration of two volumes

    Examples
    ========

    >>> from nipype.interfaces import broccoli
    >>> from nipype.testing import example_data
    >>> reg = broccoli.RegisterTwoVolumes()
    >>> reg.inputs.in_file = 'structural.nii'
    >>> reg.inputs.reference = 'mni.nii'
    >>> reg.inputs.platform = 1
    >>> reg.inputs.device = 2
    >>> reg.cmdline
    'RegisterTwoVolumes structural.nii mni.nii -device 2 -platform 1'
    >>> res = reg.run() # doctest: +SKIP

    """

    _cmd = 'RegisterTwoVolumes'
    input_spec = RegisterTwoVolumesInputSpec
    output_spec = RegisterTwoVolumesOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if (not isdefined(self.inputs.output)):
            if ((not isdefined(self.inputs.iterationslinear)) or (self.inputs.iterationslinear > 0)):
                outputs['aligned_linear_file'] = self._gen_fname(self.inputs.in_file,
                                               suffix='_aligned_linear')

            if ((not isdefined(self.inputs.iterationsnonlinear)) or (self.inputs.iterationsnonlinear > 0)):
                outputs['aligned_nonlinear_file'] = self._gen_fname(self.inputs.in_file,
                                               suffix='_aligned_nonlinear')

            if (isdefined(self.inputs.saveinterpolated)):
                outputs['interpolated_file'] = self._gen_fname(self.inputs.in_file,
                                               suffix='_interpolated')

            if (isdefined(self.inputs.mask) or isdefined(self.inputs.maskoriginal)):
                outputs['masked_file'] = self._gen_fname(self.inputs.in_file,
                                               suffix='_masked')

        else:
            if ((not isdefined(self.inputs.iterationslinear)) or (self.inputs.iterationslinear > 0)):
                outputs['aligned_linear_file'] = self._gen_fname(self.inputs.output,
                                               suffix='_aligned_linear')

            if ((not isdefined(self.inputs.iterationsnonlinear)) or (self.inputs.iterationsnonlinear > 0)):
                outputs['aligned_nonlinear_file'] = self._gen_fname(self.inputs.output,
                                               suffix='_aligned_nonlinear')

            if (isdefined(self.inputs.saveinterpolated)):
                outputs['interpolated_file'] = self._gen_fname(self.inputs.output,
                                               suffix='_interpolated')

            if (isdefined(self.inputs.mask) or isdefined(self.inputs.maskoriginal)):
                outputs['masked_file'] = self._gen_fname(self.inputs.output,
                                               suffix='_masked')
        return outputs




class SmoothingInputSpec(BROCCOLICommandInputSpec):
    in_file = File(desc='input volume(s) to smooth',
                   argstr='%s',
                   position=0,
                   mandatory=True,
                   exists=True,
                   copyfile=False)


    fwhm = traits.Float(argstr='-fwhm %s', desc='Amount of Gaussian smoothing, in mm FWHM')

    mask = traits.Str(argstr='-mask %s', desc='Perform smoothing inside mask (normalized convolution)')

    automask = traits.Bool(argstr='-automask', desc='Generate a mask and apply smoothing inside mask (normalized convolution)')


class SmoothingOutputSpec(TraitedSpec):
    smoothed_file = File(exists=True,
        desc="path/name of smoothed volume(s)")
    mask_file = File(exists=True,
        desc="path/name of generated mask (if generated)")

class Smoothing(BROCCOLICommand):
    """This function performs smoothing of one or several volumes

    Examples
    ========

    >>> from nipype.interfaces import broccoli
    >>> from nipype.testing import example_data
    >>> sm = broccoli.Smoothing()
    >>> sm.inputs.in_file = 'functional.nii'
    >>> sm.inputs.platform = 1
    >>> sm.inputs.device = 2
    >>> sm.inputs.fwhm = 4
    >>> sm.cmdline
    'Smoothing functional.nii -device 2 -fwhm 4.0 -platform 1'
    >>> res = sm.run() # doctest: +SKIP

    """

    _cmd = 'Smoothing'
    input_spec = SmoothingInputSpec
    output_spec = SmoothingOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if (not isdefined(self.inputs.output)):

            smoothed_file = self._gen_fname(self.inputs.in_file,
                                               suffix='_sm')

            if (isdefined(self.inputs.automask)):
                    outputs['mask_file'] = self._gen_fname(self.inputs.in_file,
                                                   suffix='_mask')

        else:
            smoothed_file = self._gen_fname(self.inputs.output)

            if (isdefined(self.inputs.automask)):
                    outputs['mask_file'] = self._gen_fname(self.inputs.output,
                                                   suffix='_mask')
        return outputs

class MotionCorrectionInputSpec(BROCCOLICommandInputSpec):
    in_file = File(desc='input volumes to apply motion correction to',
                   argstr='%s',
                   position=0,
                   mandatory=True,
                   exists=True,
                   copyfile=False)



    iterations = traits.Int(argstr='-iterations %d', desc='Number of iterations for motion correction algorithm')

    referencevolume = traits.Str(argstr='-referencevolume %s', desc='Volume to align all other volumes to')


class MotionCorrectionOutputSpec(TraitedSpec):
    motioncorrected_file = File(exists=True,
        desc="path/name of motion corrected volume")
    parameters_file = File(exists=True,
        desc="path/name of motion parameters file")


class MotionCorrection(BROCCOLICommand):
    """This function performs motion correction for an fMRI dataset

    Examples
    ========

    >>> from nipype.interfaces import broccoli
    >>> from nipype.testing import example_data
    >>> mc = broccoli.MotionCorrection()
    >>> mc.inputs.in_file = 'functional.nii'
    >>> mc.inputs.platform = 1
    >>> mc.inputs.device = 2
    >>> mc.cmdline
    'MotionCorrection functional.nii -device 2 -platform 1'
    >>> res = mc.run() # doctest: +SKIP

    """

    _cmd = 'MotionCorrection'
    input_spec = MotionCorrectionInputSpec
    output_spec = MotionCorrectionOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if (not isdefined(self.inputs.output)):

            motioncorrected_file = self._gen_fname(self.inputs.in_file,
                                               suffix='_mc')

            parameters_file = self._gen_fname(self.inputs.in_file,
                                               suffix='_motionparameters',
                                               ext='.1D')
        else:
            motioncorrected_file = self._gen_fname(self.inputs.output)


            parameters_file = self._gen_fname(self.inputs.output,
                                               suffix='_motionparameters',
                                               ext='.1D')

        return outputs


