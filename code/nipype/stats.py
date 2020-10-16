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


class RandomiseGroupLevelInputSpec(BROCCOLICommandInputSpec):
    in_file = File(desc='input volume to use for randomisation testing',
                   argstr='%s',
                   position=0,
                   mandatory=True,
                   exists=True,
                   copyfile=False)

    design = traits.Str(argstr='-design %s', desc='The design matrix to apply in each permutation')

    contrasts = traits.Str(argstr='-contrasts %s', desc='The contrast vector(s) to apply to the estimated beta values')

    groupmean = traits.Bool(argstr='-groupmean', desc='Test for group mean, using sign flipping (design and contrast not needed)')

    mask = traits.Str(argstr='-mask %s', desc='A mask that defines which voxels to permute (default none)')

    permutations = traits.Int(argstr='-permutations %d', desc='Number of permutations to use (default 5,000)')

    teststatistics = traits.Int(argstr='-teststatistics %d', desc='Test statistics to use, 0 = GLM t-test, 1 = GLM F-test  (default 0)')

    inferencemode = traits.Int(argstr='-inferencemode %d', desc='Inference mode to use, 0 = voxel, 1 = cluster extent, 2 = cluster mass, 3 = TFCE (default 1)')

    cdt = traits.Float(argstr='-cdt %s', desc='Cluster defining threshold for cluster inference (default 2.5)')

    significance = traits.Float(argstr='-significance %s', desc='The significance level to calculate the threshold for (default 0.05)')

    output = traits.Str(argstr='-output %s', desc='Set output filename (default volumes_perm_tvalues.nii and volumes_perm_pvalues.nii)')

    writepermutationvalues = traits.Str(argstr='-writepermutationvalues %s', desc='Write all the permutation values to a text file')

    writepermutations = traits.Str(argstr='-writepermutations %s', desc='Write all the random permutations (or sign flips) to a text file')

    permutationfile = traits.Str(argstr='-permutationfile', desc='Use a specific permutation file or sign flipping file (e.g. from FSL)')

    quiet = traits.Bool(argstr='-quiet', desc="Use a specific permutation file or sign flipping file (e.g. from FSL)")

    verbose = traits.Bool(argstr='-verbose', desc='Print extra stuff (default false)')


class RandomiseGroupLevelOutputSpec(TraitedSpec):
    pvalues_file = File(exists=True,
        desc="path/name of volume(s) showing p values of contrast(s)")
    stats_file = File(exists=True,
        desc="path/name of volume(s) showing test statistics (t or F) of contrast(s)")
    permutationvalues_file = File(
        desc="path/name of text file with permutation values")
    permutations_file = File(
        desc="path/name of text file with random permutations (or sign flips)")
    

class RandomiseGroupLevel(BROCCOLICommand):
    """The function performs permutation testing for group analyses.

    Examples
    ========

    General usage:

    >>> from nipype.interfaces import broccoli
    >>> rgl = broccoli.RandomiseGroupLevel()
    >>> rgl.inputs.in_file = 'volumes.nii'
    >>> rgl.inputs.design = 'design.mat'
    >>> rgl.inputs.contrasts = 'design.con'
    >>> rgl.inputs.platform = 1
    >>> rgl.inputs.device = 2
    >>> rgl.cmdline
    'RandomiseGroupLevel volumes.nii -design design.mat -contrasts design.con -device 2 -platform 1'

    Testing a group mean:

    >>> from nipype.interfaces import broccoli
    >>> rgl = broccoli.RandomiseGroupLevel()
    >>> rgl.inputs.in_file = 'volumes.nii'
    >>> rgl.inputs.groupmean = True
    >>> rgl.inputs.platform = 1
    >>> rgl.inputs.device = 2
    >>> rgl.cmdline
    'RandomiseGroupLevel volumes.nii -design design.mat -contrasts design.con -device 2 -platform 1' 
    
    """

    _cmd = 'RandomiseGroupLevel'
    input_spec = RandomiseGroupLevelInputSpec
    output_spec = RandomiseGroupLevelOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if (not isdefined(self.inputs.output)):
            pvalues_file = self._gen_fname(self.inputs.in_file,
                                           suffix='_pvalues')

            stats_file = self._gen_fname(self.inputs.in_file,
                                         suffix='_tvalues')

        else:
            pvalues_file = self._gen_fname(self.inputs.output,
                                           suffix='_pvalues')

            stats_file = self._gen_fname(self.inputs.output,
                                         suffix='_tvalues')

        if (isdefined(self.inputs.writepermutationvalues)):
            outputs['permutationvalues_file'] = self._gen_fname(self.inputs.writepermutationvalues,
                                                                ext='.txt')

        if (isdefined(self.inputs.writepermutations)):
            outputs['permutations_file'] = self._gen_fname(self.inputs.writepermutations,
                                                           ext='.txt')

        return outputs
