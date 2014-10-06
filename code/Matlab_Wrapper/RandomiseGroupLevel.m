 function [statistical_maps, p_values, null_distribution] = RandomiseGroupLevel(filenames,design_matrix,contrasts,varargin)

% The function performs permutation testing at the group level.
%
% [statistical_maps, p_values, null_distribution] = ...
% RandomiseGroupLevel(filenames,design_matrix, contrasts, brain_mask, ... 
% number_of_permutations, inference_mode, cluster_defining_threshold, ...
% opencl_platform,opencl_device)
%
% Required input parameters
% filenames     - A vector of filenames with first level results
% design_matrix - The design matrix to apply
% contrasts     - The contrasts to use, one contrast per row
% 
% Optional input parameters
% brain_mask             - Only permute the voxels in the mask (default none)
% number_of_permutations - The number of permutations to use (default 1000)
% inference_mode         - 0 = voxel, 1 = cluster extent, 2 = cluster mass (default
% 1)
% cluster_defining_threshold - Voxel-wise threshold to apply to define
% clusters (default 2.5)
% opencl_platform - the OpenCL platform to use (default 0)
% opencl_device   - the OpenCL device to use (default 0)

  
 if length(varargin) > 1
    number_of_permutations = varargin{2}; 
 else
    number_of_permutations = 1000; 
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

start = clock;
number_of_subjects = length(filenames);

% Read one subject first, to get size of data
beta_volume = load_nii(filenames{1});
beta_volume = double(beta_volume.img);
[MNI_sy MNI_sx MNI_sz] = size(beta_volume);
        
first_level_results = zeros(MNI_sy,MNI_sx,MNI_sz,number_of_subjects);
for subject = 1:number_of_subjects
    try
        beta_volume = load_nii(filenames{subject});
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
loadtime = etime(clock,start);
disp(sprintf('It took %f seconds to load the first level results \n',loadtime));

%--------------------------------------------------------------------------------------
% Load brain mask
%--------------------------------------------------------------------------------------

if length(varargin) > 0
    brain_mask = varargin{1};
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
    MNI_brain_mask = ones(MNI_sy,MNI_sx,MNI_sz); 
end

%--------------------------------------------------------------------------------------
% Setup GLM regressors
%--------------------------------------------------------------------------------------

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

xtxxt_GLM = inv(X_GLM'*X_GLM)*X_GLM';

%--------------------------------------------------------------------------------------
% Setup contrasts
%--------------------------------------------------------------------------------------

if size(contrasts,2) ~= number_of_regressors
    error = 1;
    disp('Error: the number of columns in the contrast matrix does not match the number of regressors in the design matrix!') 
else
    for i = 1:size(contrasts,1)
        contrast = contrasts(i,:)';
        ctxtxc_GLM(i) = contrast'*inv(X_GLM'*X_GLM)*contrast;
    end
end

%--------------------------------------------------------------------------------------
% Generate permutation matrix
%--------------------------------------------------------------------------------------

if error == 0
    permutation_matrix = zeros(number_of_permutations,number_of_subjects);
    values = 1:number_of_subjects;
    for p = 1:number_of_permutations
        permutation = randperm(number_of_subjects);
        permutation_matrix(p,:) = values(permutation);
    end
    permutation_matrix = permutation_matrix - 1;
end

%--------------------------------------------------------------------------------------
% Run permutation based second level analysis with OpenCL
%--------------------------------------------------------------------------------------

if error == 0
    start = clock;
    [statistical_maps, p_values, null_distribution] = ...
        GLMTTest_SecondLevel_Permutation(first_level_results,MNI_brain_mask, X_GLM,xtxxt_GLM',contrasts,ctxtxc_GLM, uint16(permutation_matrix'), number_of_permutations, inference_mode, cluster_defining_threshold, opencl_platform, opencl_device);
    permutation_time = etime(clock,start);
    disp(sprintf('It took %f seconds to run the permutation test \n',permutation_time'));

    s = sort(null_distribution);
    significance_threshold = s(round(0.95*number_of_permutations));
    disp(sprintf('The permutation threshold for a significance level of 5%% is %f \n',significance_threshold'));
end




