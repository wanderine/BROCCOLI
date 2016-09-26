function [statistical_maps, p_values] = RandomiseGroupLevel(volumes,design_matrix,analysis_type,contrasts,broccoli_location,varargin)

% The function performs permutation testing at the group level.
%
% [statistical_maps, p_values, null_distribution] = ...
% RandomiseGroupLevel(volumes,design_matrix, analysis_type, contrasts, ...
% broccoli_location, brain_mask, ...
% number_of_permutations, inference_mode, cluster_defining_threshold, ...
% opencl_platform,opencl_device)
%
% Required input parameters
%
% volumes       - Volumes to permute (filenames or 4D Matlab array)
% design_matrix - The design matrix to apply, use [] for group mean
% analysis_type - t-test (0), F-test (1), group mean (2)
% contrasts     - The contrasts to use, one contrast per row, use [] for group mean
% broccoli_location - Where BROCCOLI is installed, a string
%
% Optional input parameters
%
% brain_mask             - Only permute the voxels in the mask (default none)
% number_of_permutations - The number of permutations to use (default 5000)
% inference_mode         - 0 = voxel, 1 = cluster extent, 2 = cluster mass (default 1)
% cluster_defining_threshold - Voxel-wise threshold to apply to define
% clusters (default 2.5)
% opencl_platform - the OpenCL platform to use (default 0)
% opencl_device   - the OpenCL device to use (default 0)


if length(varargin) > 1
    number_of_permutations = varargin{2};
else
    number_of_permutations = 5000;
end

if length(varargin) > 2
    inference_mode = varargin{3};
else
    inference_mode = 1;
end

if length(varargin) > 3
    cluster_defining_threshold = varargin{4};
else
    cluster_defining_threshold = 2.5;
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

%--------------------------------------------------------------------------------------
% Load first level results
%--------------------------------------------------------------------------------------

error = 0;

if iscell(volumes)
    
    number_of_subjects = length(volumes);
    
    % Read one subject first, to get size of data
    beta_volume = load_nii(volumes{1});
    beta_volume = double(beta_volume.img);
    [MNI_sy MNI_sx MNI_sz] = size(beta_volume);
    
    first_level_results = zeros(MNI_sy,MNI_sx,MNI_sz,number_of_subjects);
    for subject = 1:number_of_subjects
        try
            beta_volume = load_nii(volumes{subject});
            beta_volume = double(beta_volume.img);
            [sy sx sz] = size(beta_volume);
            if ( (sx ~= MNI_sx) || (sy ~= MNI_sy) || (sz ~= MNI_sz) )
                error = 1;
                disp(sprintf('Error: the size of the data for subject %i does not match the size of the first subject!\n',subject));
            end
            first_level_results(:,:,:,subject) = beta_volume;
        catch
            error = 1;
            disp(sprintf('Unable to read first level results for subject %i !',subject));
        end
    end
    
else
    [MNI_sy MNI_sx MNI_sz number_of_subjects] = size(volumes);
    first_level_results = volumes;
end


%--------------------------------------------------------------------------------------
% Load brain mask
%--------------------------------------------------------------------------------------

if length(varargin) > 0
    brain_mask = varargin{1};
    
    if ischar(brain_mask)
        try
            MNI_brain_mask_nii = load_nii(brain_mask);
            MNI_brain_mask = double(MNI_brain_mask_nii.img);
            
            [sy sx sz] = size(MNI_brain_mask);
            
            if ( (sx ~= MNI_sx) || (sy ~= MNI_sy) || (sz ~= MNI_sz) )
                error = 1;
                disp('Error: the size of the brain mask does not match the size of the subject data!');
            end
            
            brain_voxels = sum(MNI_brain_mask(:));
            disp(sprintf('Brain mask contains %i voxels, out of the total %i voxels \n',brain_voxels, MNI_sx * MNI_sy * MNI_sz));
        catch
            error = 1;
            disp('Unable to read brain mask!')
        end
    else
        MNI_brain_mask = brain_mask;
    end
else
    MNI_brain_mask = ones(MNI_sy,MNI_sx,MNI_sz);
    disp('Warning: No mask being used, doing permutations for all voxels')
end

%--------------------------------------------------------------------------------------
% Setup GLM regressors
%--------------------------------------------------------------------------------------

if analysis_type ~= 2
    X_GLM = design_matrix;
    k = rank(X_GLM);
    number_of_regressors = size(X_GLM,2);
    
    if size(X_GLM,1) ~= number_of_subjects
        error = 1;
        disp('Error: the number of rows in the design matrix does not match the number of subjects!')
    end
    if k < number_of_regressors
        disp('Warning: design matrix does not have full rank!')
    end
else
    X_GLM = ones(number_of_subjects,1);
end

xtxxt_GLM = inv(X_GLM'*X_GLM)*X_GLM';

%--------------------------------------------------------------------------------------
% Setup contrasts
%--------------------------------------------------------------------------------------

% Not group mean
if analysis_type ~= 2
    if size(contrasts,2) ~= number_of_regressors
        error = 1;
        disp('Error: the number of columns in the contrast matrix does not match the number of regressors in the design matrix!')
    else
        for i = 1:size(contrasts,1)
            contrast = contrasts(i,:)';
            ctxtxc_GLM(i) = contrast'*inv(X_GLM'*X_GLM)*contrast;
        end
    end
    % Group mean
else
    contrasts = 1;
    ctxtxc_GLM = 1;
end

%--------------------------------------------------------------------------------------
% Run permutation based second level analysis with OpenCL
%--------------------------------------------------------------------------------------

if error == 0
    start = clock;
    [statistical_maps, p_values] = ...
        RandomiseGroupLevelMex(first_level_results, MNI_brain_mask, analysis_type, X_GLM,xtxxt_GLM',contrasts',ctxtxc_GLM, number_of_permutations, inference_mode, cluster_defining_threshold, opencl_platform, opencl_device, broccoli_location);
    permutation_time = etime(clock,start);
    disp(sprintf('It took %f seconds to run the permutation test \n',permutation_time'));                
end




