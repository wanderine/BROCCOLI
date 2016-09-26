function [aligned_volume_nonlinear, varargout] = RegisterTwoVolumes(filename_source_volume,filename_reference_volume,varargin)

% 
% The function registers two volumes by using both linear and non-linear
% transformations. The function automagically resizes and rescales the
% source volume to match the reference volume. 
%
% [aligned_volume_nonlinear, aligned_volume_linear, registration_parameters, ... 
% reference_volume, interpolated_volume, displacement_x, displacement_y, displacement_z] = ...
% RegisterTwoVolumes(filename_source_volume,filename_reference_volume, ... 
% number_of_iterations_for_linear_image_registration, ...
% number_of_iterations_for_nonlinear_image_registration, ...
% MM_Z_CUT,SIGMA,opencl_platform,opencl_device)
%
% Required input parameters
% filename_source_volume    - the filename of the volume to be transformed
% filename_reference_volume - the filename of the reference volume
% 
% Optional input parameters
% number_of_iterations_for_linear_image_registration (default 10)
% number_of_iterations_for_nonlinear_image_registration (default 10)
% MM_Z_CUT - millimeters to cut from the source volume in the z-direction (default 0)
% SIGMA - amount of smoothing applied to displacement field (default 5)
% opencl_platform - the OpenCL platform to use (default 0)
% opencl_device   - the OpenCL device to use (default 0)
%
% Required output parameters
% aligned_volume_nonlinear - The result after linear and non-linear registration
%
% Optional output parameters
% aligned_volume_linear - The result after linear registration
% registration_parameters - The estimated affine registration parameters
% reference_volume - The reference volume
% interpolated_volume - The source volume after resizing and rescaling
% displacement_x - Non-linear displacement field in x-direction
% displacement_y - Non-linear displacement field in y-direction
% displacement_z - Non-linear displacement field in z-direction

if length(varargin) > 0
    number_of_iterations_for_linear_image_registration = varargin{1};
else
    number_of_iterations_for_linear_image_registration = 10;
end

if length(varargin) > 1
    number_of_iterations_for_nonlinear_image_registration = varargin{2};
else
    number_of_iterations_for_nonlinear_image_registration = 10;
end

if length(varargin) > 2
    MM_Z_CUT = varargin{3};
else
    MM_Z_CUT = 0;
end

if length(varargin) > 3
    SIGMA = varargin{4};
else
    SIGMA = 5;
end

if length(varargin) > 4
    opencl_platform = varargin{5};
else
    opencl_platform = 0;
end

if length(varargin) > 5
    opencl_device = varargin{6};
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
start = clock;    
% Load volume to transform
try
    volume_nii = load_nii(filename_source_volume);
    volume = double(volume_nii.img);
    [volume_sy volume_sx volume_sz] = size(volume);
    volume_voxel_size_x = volume_nii.hdr.dime.pixdim(2);
    volume_voxel_size_y = volume_nii.hdr.dime.pixdim(3);
    volume_voxel_size_z = volume_nii.hdr.dime.pixdim(4);
catch
    error = 1;
    disp('Failed to load source volume!')
end
   
% Load reference volume
try
    reference_volume_nii = load_nii(filename_reference_volume);
    reference_volume = double(reference_volume_nii.img);
    [reference_volume_sy reference_volume_sx reference_volume_sz] = size(reference_volume);
    reference_volume_voxel_size_x = reference_volume_nii.hdr.dime.pixdim(2);
    reference_volume_voxel_size_y = reference_volume_nii.hdr.dime.pixdim(3);
    reference_volume_voxel_size_z = reference_volume_nii.hdr.dime.pixdim(4);   
catch
    error = 1;
    disp('Failed to load reference volume!')
end

load_time = etime(clock,start);
disp(sprintf('It took %f seconds to load the volumes \n',load_time'));

% Load quadrature filters
try
    load filters_for_parametric_registration.mat
    load filters_for_nonparametric_registration.mat 
catch
    error = 1;
    disp('Failed to load quadrature filters!')
end
    
% Run the registration with OpenCL           
if error == 0
    try
        start = clock;
        [aligned_volume_linear, aligned_volume_nonlinear, interpolated_volume, registration_parameters, displacement_x, displacement_y, displacement_z] = ...
            RegisterTwoVolumesMex(volume,reference_volume,volume_voxel_size_x,volume_voxel_size_y,volume_voxel_size_z,reference_volume_voxel_size_x,reference_volume_voxel_size_y,reference_volume_voxel_size_z, ...
            f1_parametric_registration,f2_parametric_registration,f3_parametric_registration, ...
            f1_nonparametric_registration,f2_nonparametric_registration,f3_nonparametric_registration,f4_nonparametric_registration,f5_nonparametric_registration,f6_nonparametric_registration, ...
            m1, m2, m3, m4, m5, m6, ...
            filter_directions_x, filter_directions_y, filter_directions_z, ...
            number_of_iterations_for_linear_image_registration,number_of_iterations_for_nonlinear_image_registration,MM_Z_CUT, SIGMA, opencl_platform, opencl_device);
        registration_time = etime(clock,start);
        disp(sprintf('It took %f seconds to run the registration \n',registration_time'));
    catch
        disp('Failed to run the registration!') 
    end
end

if nargout > 1
    varargout{1} = aligned_volume_linear;
end

if nargout > 2
    varargout{2} = registration_parameters;
end

if nargout > 3
    varargout{3} = reference_volume;
end

if nargout > 4
    varargout{4} = interpolated_volume;
end

if nargout > 5
    varargout{5} = displacement_x;
end

if nargout > 6
    varargout{6} = displacement_y;
end

if nargout > 7
    varargout{7} = displacemeny_z;
end

