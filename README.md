BROCCOLI: Software for Fast fMRI Analysis on Many-Core CPUs and GPUs
========

BROCCOLI is a software for analysis of fMRI (functional magnetic resonance imaging) data and is written in OpenCL (Open Computing Language). The analysis can thereby be performed in parallel on many types of hardware, such as CPUs, Nvidia GPUs and AMD GPUs. The result is a significantly faster analysis than possible with existing software packages for fMRI analysis (SPM, FSL, AFNI). For example, non-linear normalization of an anatomical T1 volume to MNI space (1mm resolution) takes only 4-8 seconds with a GPU. A permutation test with 10,000 permutations can be done within a minute, to empirically estimate a null distribution. Additionally, BROCCOLI includes support for Bayesian first-level fMRI analysis using a Gibbs sampler.

For more information, see

[Eklund, A., Dufort, P., Villani, M., LaConte, S., BROCCOLI: Software for Fast fMRI Analysis on Many-Core CPUs and GPUs, Frontiers in Neuroinformatics, 8:24, 2014](http://journal.frontiersin.org/Journal/10.3389/fninf.2014.00024/abstract)

--------------------------------------------------------------------

[Documentation](https://github.com/wanderine/BROCCOLI/raw/master/documentation/broccoli.pdf)





