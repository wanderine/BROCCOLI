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
    %mex -g ../code/Matlab_Wrapper/SecondLevelAnalysis.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib    -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
    mex ../code/Matlab_Wrapper/SecondLevelAnalysis.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib    -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab')
    basepath = '/data/andek/OpenfMRI/';
    opencl_platform = 0;
    opencl_device = 0;
    mex ../code/Matlab_Wrapper/SecondLevelAnalysis.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
end

study = 'RhymeJudgment/ds003_models';
number_of_subjects = 52;

voxel_size = 2;

%--------------------------------------------------------------------------------------
% Statistical settings
%--------------------------------------------------------------------------------------

number_of_permutations = 10000;
inference_mode = 0; % 0 = voxel, 1 = cluster extent, 2 = cluster mass
cluster_defining_threshold = 2;
statistical_test = 0; % 0 = t-test, 1 = F-test

%--------------------------------------------------------------------------------------
% Load MNI templates
%--------------------------------------------------------------------------------------

MNI_brain_mask_nii = load_nii(['../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm_brain_mask.nii']);
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

first_level_results(:,:,:,14:26) = first_level_results(:,:,:,1:13);
first_level_results(:,:,:,27:39) = first_level_results(:,:,:,1:13);
first_level_results(:,:,:,40:52) = first_level_results(:,:,:,1:13);

%--------------------------------------------------------------------------------------
% Create GLM regressors
%--------------------------------------------------------------------------------------

nr = 1;
X_GLM = zeros(number_of_subjects,nr);

for subject = 1:number_of_subjects
    X_GLM(subject,1) = subject;
    
    for r = 2:nr
        X_GLM(subject,r) = randn;
    end
    
end

% X_GLM(1:13,1:13) = eye(13);
% for subject = 1:number_of_subjects
%     
%     for r = 14:nr
%         X_GLM(subject,r) = 1;
%     end        
%     
% end
% 
% subject = 1;
% for r = 14:nr
%     X_GLM(subject,r) = -1;
%     subject = subject + 1;
% end



%xtxxt_GLM = inv(X_GLM'*X_GLM)*X_GLM';
xtxxt_GLM = pinv(X_GLM);



%--------------------------------------------------------------------------------------
% Setup contrasts
%--------------------------------------------------------------------------------------

if nr == 1
    
    %contrasts = [1];
    contrasts = [1; -1; 3; 8; 99];
    
elseif nr == 2
    
    contrasts = [1 0];
    
    %contrasts = [1 0;
    %             0 1];
    
    
elseif nr == 3
    
    contrasts = [1 0 0];
    
elseif nr == 4
    
    contrasts = [1 0 0 0];
    
elseif nr == 5
    
    contrasts = [1 0 0 0 0];
    
elseif nr == 6
    
    contrasts = [1 0 0 0 0 0];
    
elseif nr == 7
    
    contrasts = [1 0 0 0 0 0 0];
    
elseif nr == 8
    
    contrasts = [1 0 0 0 0 0 0 0];
    
elseif nr == 9
    
    contrasts = [1 0 0 0 0 0 0 0 0];
    
elseif nr == 10
    
    contrasts = [1 0 0 0 0 0 0 0 0 0];
    
    %contrasts = [1 0 0 0 0 0 0 0 0 0;
    %             0 1 0 0 0 0 0 0 0 0];
    
elseif nr == 11
    
    contrasts = [1 0 0 0 0 0 0 0 0 0 0];
    
elseif nr == 12
    
    contrasts = [1 0 0 0 0 0 0 0 0 0 0 0];
    
elseif nr == 13
    
    contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0];
    
    %contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0;
    %             0 1 0 0 0 0 0 0 0 0 0 0 0];
    
elseif nr == 14
    
    contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0 0];
    
elseif nr == 15
    
    contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
    
elseif nr == 16
    
    contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
    
elseif nr == 17
    
    contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
    
elseif nr == 18
    
    contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
    
elseif nr == 19
    
    contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
    
elseif nr == 20
    
    contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
    
    %contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
    %             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
    
    
elseif nr == 21
    
    contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
    
    %contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
    %             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
    
elseif nr == 22
    
    contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
    
elseif nr == 23
    
    contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
    
    %contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
    %             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
    
elseif nr == 24
    
    contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
    
elseif nr == 25
    
    %contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
    %             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
    
    contrasts = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] ;
    
end

for i = 1:size(contrasts,1)
    contrast = contrasts(i,:)';
    %ctxtxc_GLM(i) = contrast'*inv(X_GLM'*X_GLM)*contrast;
    ctxtxc_GLM(i) = 1;
end


%--------------------------------------------------------------------------------------
% Run second level analysis
%--------------------------------------------------------------------------------------
%fid = fopen([basepath 'RhymeJudgment/permtest__perm_tstat1.txt']);
%text = textscan(fid,'%f%f%f%f%f%f%f%f%f%f%f%f%f');
%fclose(fid);
permutation_matrix = zeros(10000,13);
%permutation_matrix(:,1) = text{1};
%permutation_matrix(:,2) = text{2};
%permutation_matrix(:,3) = text{3};
%permutation_matrix(:,4) = text{4};
%permutation_matrix(:,5) = text{5};
%permutation_matrix(:,6) = text{6};
%permutation_matrix(:,7) = text{7};
%permutation_matrix(:,8) = text{8};
%permutation_matrix(:,9) = text{9};
%permutation_matrix(:,10) = text{10};
%permutation_matrix(:,11) = text{11};
%permutation_matrix(:,12) = text{12};
%permutation_matrix(:,13) = text{13};
%permutation_matrix = permutation_matrix - 1;

tic
[beta_volumes, residuals, residual_variances, statistical_maps, design_matrix1, design_matrix2, cluster_indices, null_distribution, permuted_first_level_results] = ...
    SecondLevelAnalysis(first_level_results,MNI_brain_mask, X_GLM,xtxxt_GLM',contrasts,ctxtxc_GLM, statistical_test, uint16(permutation_matrix'), number_of_permutations, inference_mode, cluster_defining_threshold, opencl_platform, opencl_device);
toc



slice = round(0.5*MNI_sz);


%figure
%imagesc(MNI(:,:,slice)); colormap gray

figure
imagesc(beta_volumes(:,:,slice)); colormap gray; colorbar
title('Beta')

figure
imagesc(statistical_maps(:,:,slice,1)); colorbar
title('t-values')

figure
imagesc(statistical_maps(:,:,slice,1) > cluster_defining_threshold); colorbar

figure
imagesc(cluster_indices(:,:,slice,1)); colorbar
title('Cluster indices')



slice = round(0.45*MNI_sz);

%figure
%imagesc(flipud(squeeze(MNI(slice,:,:))')); colormap gray

figure
imagesc(flipud(squeeze(beta_volumes(slice,:,:,1))')); colormap gray
title('Beta')

figure
imagesc(flipud(squeeze(statistical_maps(slice,:,:,1))')); colorbar
title('t-values')

figure
imagesc(residual_variances(:,:,slice)); colorbar
title('Residual variances')

s = sort(null_distribution);
threshold = s(round(0.95*number_of_permutations))

figure
hist(null_distribution,50)
%hist(null_distribution,5:0.75:25)
%N_BROCCOLI = hist(null_distribution,5:0.75:25);

% fid = fopen([basepath 'RhymeJudgment/permtest__vox_corrp_tstat1.txt']);
% text = textscan(fid,'%f');
% fclose(fid);
% FSL_permutation_values = text{1};
% N_FSL = hist(FSL_permutation_values,5:0.75:25);
%
% figure
% plot(5:0.75:25,N_BROCCOLI/sum(N_BROCCOLI),'b')
% hold on
% plot(5:0.75:25,N_FSL/sum(N_FSL),'g')
% hold off
%
% legend('BROCCOLI null distribution','FSL null distribution')
%
% xlabel('Maximum t-value','FontSize',15)
% ylabel('Probability','FontSize',15)
%
% set(gca,'FontSize',15)

%print -dpng /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/permutation_distributions.png


%close all

if (inference_mode == 1)
    
    a = statistical_maps(:,:,:,1);
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
    
    sum(matlab_sums(:) - broccoli_sums(:))
    
    max(matlab_sums(:))
    
elseif (inference_mode == 2)

    a = statistical_maps(:,:,:,1);
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
    
    sum(matlab_sums(:) - broccoli_sums(:))
    
    max(matlab_sums(:))


end
        