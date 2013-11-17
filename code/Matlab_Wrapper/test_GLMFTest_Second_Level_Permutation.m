%  	 BROCCOLI: An open source multi-platform software for parallel analysis of fMRI data on many core CPUs and GPUS
%    Copyright (C) <2013>  Anders Eklund, andek034@gmail.com
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%-----------------------------------------------------------------------------

%---------------------------------------------------------------------------------------------------------------------
% README
% If you run this code in Windows, your graphics driver might stop working
% for large volumes / large filter sizes. This is not a bug in my code but is due to the
% fact that the Nvidia driver thinks that something is wrong if the GPU
% takes more than 2 seconds to complete a task. This link solved my problem
% https://forums.geforce.com/default/topic/503962/tdr-fix-here-for-nvidia-driver-crashing-randomly-in-firefox/
%---------------------------------------------------------------------------------------------------------------------

clear all
clc
close all

if ispc
    addpath('D:\nifti_matlab')
    basepath = 'D:\OpenfMRI\';
    opencl_platform = 0; % 0 Nvidia, 1 Intel, 2 AMD
    opencl_device = 0;
    
    %mex -g GLMFTest_SecondLevel_Permutation.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib    -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
    mex GLMFTest_SecondLevel_Permutation.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib    -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab')
    basepath = '/data/andek/OpenfMRI/';
    opencl_platform = 2;
    opencl_device = 0;
    
    %mex -g GLMFTest_SecondLevel_Permutation.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Debug -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
    mex GLMFTest_SecondLevel_Permutation.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
end

study = 'RhymeJudgment/ds003_models';
number_of_subjects = 30;

voxel_size = 2;
do_permutations_in_Matlab = 0;

%--------------------------------------------------------------------------------------
% Statistical settings
%--------------------------------------------------------------------------------------

number_of_regressors = 25;
number_of_contrasts = 10;
number_of_permutations = 1000;
inference_mode = 0; % 0 = voxel, 1 = cluster extent, 2 = cluster mass
cluster_defining_threshold = 5;

%--------------------------------------------------------------------------------------
% Load MNI templates
%--------------------------------------------------------------------------------------

MNI_brain_mask_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm_brain_mask.nii']);
MNI_brain_mask = double(MNI_brain_mask_nii.img);
MNI_brain_mask = MNI_brain_mask/max(MNI_brain_mask(:));

[MNI_sy MNI_sx MNI_sz] = size(MNI_brain_mask);
[MNI_sy MNI_sx MNI_sz]

%--------------------------------------------------------------------------------------
% Load first level results
%--------------------------------------------------------------------------------------

first_level_results = zeros(MNI_sy,MNI_sx,MNI_sz,number_of_subjects);
for subject = 1:13
    if subject < 10
        beta_volume = load_nii([basepath study '/sub00' num2str(subject) '/model/model001/task001.gfeat/cope1.feat/stats/cope1.nii.gz']);
    else
        beta_volume = load_nii([basepath study '/sub0' num2str(subject) '/model/model001/task001.gfeat/cope1.feat/stats/cope1.nii.gz']);
    end
    beta_volume = double(beta_volume.img);
    first_level_results(:,:,:,subject) = beta_volume;
end

for subject = 14:number_of_subjects
    first_level_results(:,:,:,subject) = beta_volume;
end


%--------------------------------------------------------------------------------------
% Create GLM regressors
%--------------------------------------------------------------------------------------

X_GLM = zeros(number_of_subjects,number_of_regressors);

for subject = 1:number_of_subjects
    
    X_GLM(subject,1) = 1;
    
    for r = 2:number_of_regressors
        X_GLM(subject,r) = randn;
    end
    
end

xtxxt_GLM = inv(X_GLM'*X_GLM)*X_GLM';



%--------------------------------------------------------------------------------------
% Setup contrasts
%--------------------------------------------------------------------------------------

contrasts = zeros(number_of_contrasts,number_of_regressors);

for c = 1:number_of_contrasts
    contrasts(c,:) = randn(1,number_of_regressors);
end

ctxtxc_GLM = inv(contrasts*inv(X_GLM'*X_GLM)*contrasts');

%--------------------------------------------------------------------------------------
% Generate permutation matrix
%--------------------------------------------------------------------------------------

permutation_matrix = zeros(number_of_permutations,number_of_subjects);
values = 1:number_of_subjects;
for p = 1:number_of_permutations
    permutation = randperm(number_of_subjects);
    permutation_matrix(p,:) = values(permutation);
end



%-----------------------------------------------------------------------
% Run permutation based second level analysis in Matlab
%-----------------------------------------------------------------------

[sy sx sz st] = size(first_level_results);

statistical_maps_cpu = zeros(sy,sx,sz);
betas_cpu = zeros(sy,sx,sz,size(X_GLM,2));
residuals_cpu = zeros(sy,sx,sz,st);
residual_variances_cpu = zeros(sy,sx,sz);

null_distribution_cpu = zeros(number_of_permutations,1);

if do_permutations_in_Matlab == 1
    disp('Calculating statistical maps')
    
    % Loop over permutations
    for p = 1:number_of_permutations
        
        statistical_maps_cpu = zeros(sy,sx,sz);
        
        % Loop over voxels
        for x = 1:sx
            for y = 1:sy
                for z = 1:sz
                    if MNI_brain_mask(y,x,z) == 1
                        
                        % Calculate beta values, using permuted model
                        data = squeeze(first_level_results(y,x,z,:));
                        permuted_xtxxt_GLM = zeros(size(xtxxt_GLM));
                        permutation = permutation_matrix(p,:);
                        for r = 1:number_of_regressors
                            permuted_xtxxt_GLM(r,:) = xtxxt_GLM(r,permutation);
                        end
                        beta = permuted_xtxxt_GLM * data;
                        betas_cpu(y,x,z,:) = beta;
                        
                        % Calculate t-values and residuals, using permuted model
                        permuted_X_GLM = zeros(size(X_GLM));
                        permutation = permutation_matrix(p,:);
                        for r = 1:number_of_regressors
                            permuted_X_GLM(:,r) = X_GLM(permutation,r);
                        end
                        residuals = data - permuted_X_GLM*beta;
                        residuals_cpu(y,x,z,:) = residuals;
                        residual_variances_cpu(y,x,z) = sum((residuals-mean(residuals)).^2)/(st - 1);
                        
                        %F-test
                        statistical_maps_cpu(y,x,z) = (contrasts*beta)' * 1/residual_variances_cpu(y,x,z) * ctxtxc_GLM * (contrasts*beta) / size(contrasts,1);
                        
                    end
                end
            end
        end
        
        % Voxel
        if (inference_mode == 0)
            null_distribution_cpu(p) = max(statistical_maps_cpu(:));
            % Cluster extent
        elseif (inference_mode == 1)
            a = statistical_maps_cpu(:,:,:,1);
            [labels,N] = bwlabeln(a > cluster_defining_threshold);
            
            cluster_extents = zeros(N,1);
            for i = 1:N
                cluster_extents(i) = sum(labels(:) == i);
            end
            null_distribution_cpu(p) = max(cluster_extents);
            % Cluster mass
        elseif (inference_mode == 2)
            a = statistical_maps_cpu(:,:,:,1);
            [labels,N] = bwlabeln(a > cluster_defining_threshold);
            
            cluster_masses = zeros(N,1);
            for i = 1:N
                cluster_masses(i) = sum(a(labels(:) == i));
            end
            null_distribution_cpu(p) = max(cluster_masses);
        end
    end
    toc
end

%--------------------------------------------------------------------------------------
% Run permutation based second level analysis with OpenCL
%--------------------------------------------------------------------------------------

permutation_matrix = permutation_matrix - 1;

tic
[beta_volumes, residuals, residual_variances, statistical_maps_opencl, cluster_indices, null_distribution_opencl, permuted_first_level_results] = ...
    GLMFTest_SecondLevel_Permutation(first_level_results,MNI_brain_mask, X_GLM,xtxxt_GLM',contrasts,ctxtxc_GLM, uint16(permutation_matrix'), number_of_permutations, inference_mode, cluster_defining_threshold, opencl_platform, opencl_device);
toc



slice = round(0.5*MNI_sz);

figure
imagesc(beta_volumes(:,:,slice)); colormap gray; colorbar
title('Beta')

figure
imagesc(statistical_maps_opencl(:,:,slice,1)); colorbar
title('t-values')

figure
imagesc(statistical_maps_opencl(:,:,slice,1) > cluster_defining_threshold); colorbar
title('t-values above cluster defining threshold')

figure
imagesc(cluster_indices(:,:,slice,1)); colorbar
title('Cluster indices')


s = sort(null_distribution_cpu);
threshold_cpu = s(round(0.95*number_of_permutations))

s = sort(null_distribution_opencl);
threshold_opencl = s(round(0.95*number_of_permutations))

if number_of_permutations >= 100    
    figure
    hist(null_distribution_opencl,50)
end

if number_of_permutations < 10
    [null_distribution_cpu  null_distribution_opencl]
end

% Compare clustering to Matlab

% Cluster extent
if (inference_mode == 1)
    
    a = statistical_maps_opencl(:,:,:,1);
    [labels,N] = bwlabeln(a > cluster_defining_threshold);
    
    matlab_sums = zeros(N,1);
    for i = 1:N
        matlab_sums(i) = sum(labels(:) == i);
    end
    
    N = max(cluster_indices(:));
    for i = 1:N
        broccoli_sums(i) = sum(cluster_indices(:) == i);
    end
    
    matlab_sums = sort(matlab_sums);
    broccoli_sums = sort(broccoli_sums);
    
    %[matlab_sums'; broccoli_sums]'
    
    cluster_sum_error = sum(matlab_sums(:) - broccoli_sums(:))
    
% Cluster mass
elseif (inference_mode == 2)
    
    a = statistical_maps_opencl(:,:,:,1);
    [labels,N] = bwlabeln(a > cluster_defining_threshold);
    
    matlab_sums = zeros(N,1);
    for i = 1:N
        matlab_sums(i) = sum(a(labels(:) == i));
    end
    
    N = max(cluster_indices(:));
    for i = 1:N
        broccoli_sums(i) = sum(a(cluster_indices(:) == i));
    end
    
    matlab_sums = sort(matlab_sums);
    broccoli_sums = sort(broccoli_sums);
    
    %[matlab_sums'; broccoli_sums]'
    
    cluster_sum_error = sum(matlab_sums(:) - broccoli_sums(:))
    
end
