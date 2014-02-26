BROCCOLI: Software for Fast fMRI Analysis on Many-Core CPUs and GPUs
========

BROCCOLI is a software for analysis of fMRI (functional magnetic resonance imaging) data and is written in OpenCL (Open Computing Language). The analysis can thereby be performed in parallel on many types of hardware, such as CPUs, Nvidia GPUs and AMD GPUs. The result is a significantly faster analysis than possible with existing software packages for fMRI analysis (SPM, FSL, AFNI). For example, non-linear normalization of an anatomical T1 volume to MNI space (1mm resolution) takes only 4-8 seconds with a GPU. A permutation test with 10,000 permutations can be done within a minute, to empirically estimate a null distribution. Additionally, BROCCOLI includes support for Bayesian first-level fMRI analysis using a Gibbs sampler.

For more information, see

Eklund, A., Dufort, P., Villani, M., LaConte, S., BROCCOLI: Software for Fast fMRI Analysis on Many-Core CPUs and GPUs, Frontiers in Neuroinformatics, 8:24, 2014, http://journal.frontiersin.org/Journal/10.3389/fninf.2014.00024/abstract

--------------------------------------------------------------------

OpenCL drivers need to be installed to be able to run BROCCOLI

Intel: http://software.intel.com/en-us/vcsource/tools/opencl-sdk

AMD: http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/downloads/

Nvidia: http://www.nvidia.com/download/index.aspx

--------------------------------------------------------------------

This software is based on the following publications

Eklund, A., Andersson, M., Knutsson, H., fMRI Analysis on the GPU - Possibilities and Challenges, Computer Methods and Programs in Biomedicine, 105, 145-161, 2012, http://dx.doi.org/10.1016/j.cmpb.2011.07.007

Eklund, A., Andersson, M., Knutsson, H, Fast Random Permutation Tests Enable Objective Evaluation of Methods for Single Subject fMRI Analysis, International Journal of Biomedical Imaging, Article ID 627947, 2011, http://dx.doi.org/10.1155/2011/627947

Eklund, A., Andersson, M., Knutsson, H., Phase Based Volume Registration Using CUDA, International Conference on Acoustics, Speech & Signal Processing (ICASSP), p. 658-661, 2010, http://dx.doi.org/10.1109/ICASSP.2010.5495134

The code has previously been used in this publication

Eklund, A., Andersson, M., Josephson, C., Johannesson, M., Knutsson, H., Does Parametric fMRI Analysis with SPM Yield Valid Results? - An Empirical Study of 1484 Rest Datasets, NeuroImage, 61, 565-578, 2012, http://dx.doi.org/10.1016/j.neuroimage.2012.03.093

This paper presents an overview of GPUs in medical imaging

Eklund, A., Dufort, P., Forsberg, D., LaConte, S., Medical image processing on the GPU - Past, present and future, Medical Image Analysis, 17, 1073-1094, 2013, http://dx.doi.org/10.1016/j.media.2013.05.008

--------------------------------------------------------------------


