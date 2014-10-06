function [motion_corrected_volumes, motion_parameters] = MotionCorrection(filename,varargin)

% The function performs motion correction of an fMRI dataset. 
%
% [motion_corrected_volumes, motion_parameters] = MotionCorrection(filename,iterations,opencl_platform,opencl_device)
%
% Required input parameters
% filename - The filename of the fMRI data to be motion corrected
%
% Optional input parameters
% iterations - The number of iterations to use for the algorithm (default 5)
% opencl_platform - The OpenCL platform to use (default 0)
% opencl_device - The OpenCL device to use (default 0)

if length(varargin) > 0
    iterations = varargin{1};
else
    iterations = 5;
end

if length(varargin) > 1
    opencl_platform = varargin{2};
else
    opencl_platform = 0;
end

if length(varargin) > 2
    opencl_device = varargin{3};
else
    opencl_device = 0;
end

    
%---------------------------------------------------------------------------------------------------------------------
% README
% If you run this code in Windows, your graphics driver might stop working
% for large volumes / large filter sizes. This is not a bug in my code but is due to the
% fact that the Nvidia driver thinks that something is wrong if the GPU
% takes more than 2 seconds to complete a task. This link solved my problem
% https://forums.geforce.com/default/topic/503962/tdr-fix-here-for-nvidia-driver-crashing-randomly-in-firefox/
%---------------------------------------------------------------------------------------------------------------------

error = 0;

% Load fMRI data
try
    start = clock;    
    EPI_nii = load_nii(filename);
    fMRI_volumes = double(EPI_nii.img);        
    voxel_size_x = EPI_nii.hdr.dime.pixdim(2);
    voxel_size_y = EPI_nii.hdr.dime.pixdim(3);
    voxel_size_z = EPI_nii.hdr.dime.pixdim(4);
    [sy sx sz st] = size(fMRI_volumes);
    loadtime = etime(clock,start);
    disp(sprintf('It took %f seconds to load the fMRI data \n',loadtime'));
catch
    error = 1;
    disp('Unable to load fMRI data!')
end
    
% Load quadrature filters
try
    load filters_for_parametric_registration.mat
catch
    error = 1;
    disp('Unable to load quadrature filters!')
end

% Do motion correction on OpenCL device
if error == 0
    try
        start = clock;
        [motion_corrected_volumes,motion_parameters] = MotionCorrectionMex(fMRI_volumes,voxel_size_x,voxel_size_y,voxel_size_z,f1_parametric_registration,f2_parametric_registration,f3_parametric_registration,iterations,opencl_platform,opencl_device);
        runtime = etime(clock,start);
        disp(sprintf('It took %f seconds to run the motion correction \n',runtime'));
    catch
        disp('Failed to run motion correction!')
    end
end
        
        
