%       BROCCOLI: An open source multi-platform software for parallel analysis of fMRI data on many core CPUs and GPUS
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
   
    %mex -g GLMTTest_SecondLevel_Permutation.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib    -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
    mex GLMTTest_SecondLevel_Permutation.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib    -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab')
    basepath = '/home/andek/Research_projects/RandomGroupAnalyses/Results/';
    opencl_platform = 2;
    opencl_device = 0;
   
    %mex -g GLMTTest_SecondLevel_Permutation.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Debug -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
    %mex GLMTTest_SecondLevel_Permutation.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
end

study = 'Beijing';
number_of_subjects = 10;

voxel_size = 2;
do_permutations_in_Matlab = 1;

%--------------------------------------------------------------------------------------
% Statistical settings
%--------------------------------------------------------------------------------------

number_of_regressors = 9;
number_of_permutations = 6;
inference_mode = 0; % 0 = voxel, 1 = cluster extent, 2 = cluster mass
cluster_defining_threshold = 3;

mytimes = zeros(25,1);
for number_of_regressors = 9:9
   
    %--------------------------------------------------------------------------------------
    % Load MNI templates
    %--------------------------------------------------------------------------------------
   
    MNI_brain_mask_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm_brain_mask.nii']);
    MNI_brain_mask = double(MNI_brain_mask_nii.img);
    MNI_brain_mask = MNI_brain_mask/max(MNI_brain_mask(:));
   
    MNI_brain_mask(:,:,1) = 0;
    MNI_brain_mask(:,:,end) = 0;
   
    [MNI_sy MNI_sx MNI_sz] = size(MNI_brain_mask);
    [MNI_sy MNI_sx MNI_sz]
   
    %--------------------------------------------------------------------------------------
    % Load first level results
    %--------------------------------------------------------------------------------------
   
    first_level_results = zeros(MNI_sy,MNI_sx,MNI_sz,number_of_subjects);
   
%     subjects{1} = 'sub00440';
%     subjects{2} = 'sub01018';
%     subjects{3} = 'sub01244';
%     subjects{4} = 'sub02403';
%     subjects{5} = 'sub04050';
%     subjects{6} = 'sub04191';
%     subjects{7} = 'sub05267';
%     subjects{8} = 'sub06880';
%     subjects{9} = 'sub06899';
%     subjects{10} = 'sub07144';
%    
%     for subject = 1:10
%         beta_volume = load_nii([basepath study '/4mm/Event1/' subjects{subject} '.feat/reg_standard/stats/cope1.nii.gz']);
%         beta_volume = double(beta_volume.img);
%         first_level_results(:,:,:,subject) = beta_volume;
%     end

    beta_volumes = load_nii('volumes.nii.gz');
    first_level_results = double(beta_volumes.img);
   
   
   
    %--------------------------------------------------------------------------------------
    % Create GLM regressors
    %--------------------------------------------------------------------------------------
   
    % X_GLM = zeros(number_of_subjects,number_of_regressors);
    %
    % for subject = 1:number_of_subjects
    %
    %     X_GLM(subject,1) = randn;
    %
    %     for r = 2:number_of_regressors
    %         X_GLM(subject,r) = randn;
    %     end
    %
    % end
   
    X_GLM = [
       
    0.5377    1.1093   -0.4390   -0.5336    1.5270   -1.5651    0.0012    0.7477   -0.5700    0.5455   -0.6169     1.8339   -0.8637   -1.7947   -2.0026    0.4669   -0.0845   -0.0708   -0.2730   -1.0257   -1.0516    0.2748      0.3539   -0.5861   -0.6300
    -2.2588    0.0774    0.8404    0.9642   -0.2097    1.6039   -2.4863    1.5763   -0.9087    0.3975    0.6011     0.8622   -1.2141   -0.8880    0.5201    0.6252    0.0983    0.5812   -0.4809   -0.2099   -0.7519    0.0923 1.5970    0.7449   -0.0469
    0.3188   -1.1135    0.1001   -0.0200    0.1832    0.0414   -2.1924    0.3275   -1.6989    1.5163    1.7298    -1.3077   -0.0068   -0.5445   -0.0348   -1.0298   -0.7342   -2.3193    0.6647    0.6076   -0.0326   -0.6086  0.5275   -0.8282    2.6830
    -0.4336    1.5326    0.3035   -0.7982    0.9492   -0.0308    0.0799    0.0852   -0.1178    1.6360   -0.7371     0.3426   -0.7697   -0.6003    1.0187    0.3071    0.2323   -0.9485    0.8810    0.6992   -0.4251   -1.7499  0.8542    0.5745   -1.1467
    3.5784    0.3714    0.4900   -0.1332    0.1352    0.4264    0.4115    0.3232    0.2696    0.5894    0.9105     2.7694   -0.2256    0.7394   -0.7145    0.5152   -0.3728    0.6770   -0.7841    0.4943   -0.0628    0.8671 1.3418    0.2818    0.5530
    -1.3499    1.1174    1.7119    1.3514    0.2614   -0.2365    0.8577   -1.8054   -1.4831   -2.0220   -0.0799     3.0349   -1.0891   -0.1941   -0.2248   -0.9415    2.0237   -0.6912    1.8586   -1.0203   -0.9821    0.8985 -2.4995    1.1393   -1.0765
    0.7254    0.0326   -2.1384   -0.5890   -0.1623   -2.2584    0.4494   -0.6045   -0.4470    0.6125    0.1837    -0.0631    0.5525   -0.8396   -0.2938   -0.1461    2.2294    0.1006    0.1034    0.1097   -0.0549    0.2908 -0.1676   -0.4259    1.0306
    0.7147    1.1006    1.3546   -0.8479   -0.5320    0.3376    0.8261    0.5632    1.1287   -1.1187    0.1129    -0.2050    1.5442   -1.0722   -1.1201    1.6821    1.0001    0.5362    0.1136   -0.2900   -0.6264    0.4400 0.3530    0.6361    0.3275
    -0.1241    0.0859    0.9610    2.5260   -0.8757   -1.6642    0.8979   -0.9047    1.2616    0.2495    0.1017     1.4897   -1.4916    0.1240    1.6555   -0.4838   -0.5900   -0.1319   -0.4677    0.4754   -0.9930    2.7873 0.7173    0.7932    0.6521
    1.4090   -0.7423    1.4367    0.3075   -0.7120   -0.2781   -0.1472   -0.1249    1.1741    0.9750   -1.1667     1.4172   -1.0616   -1.9609   -1.2571   -1.1742    0.4227    1.0078    1.4790    0.1269   -0.6407   -1.8543 -1.3049   -0.8984   -0.2789
    ];

    X_GLM = X_GLM(:,1:number_of_regressors);

    xtxxt_GLM = inv(X_GLM'*X_GLM)*X_GLM';


%--------------------------------------------------------------------------------------
% Setup contrasts
%--------------------------------------------------------------------------------------

%contrasts = [1 zeros(1,number_of_regressors-1)];
%contrasts = [1 -1 2 -2 3 0];
%contrasts = [1 -1 0];
contrasts = [1.0 -1.0 0.0 0.0 2.0 3.0 -5.0 0.0 0.0];

for i = 1:size(contrasts,1)
    contrast = contrasts(i,:)';
    ctxtxc_GLM(i) = contrast'*inv(X_GLM'*X_GLM)*contrast;
end

%--------------------------------------------------------------------------------------
% Generate permutation matrix
%--------------------------------------------------------------------------------------

permutation_matrix = zeros(number_of_permutations,number_of_subjects);
%values = 1:number_of_subjects;
%for p = 1:number_of_permutations
%    permutation = randperm(number_of_subjects);
%    permutation_matrix(p,:) = values(permutation);
%end


permutation_matrix(1,:) = [1 2 3 4 5 6 7 8 9 10];
permutation_matrix(2,:) = [5 4 8 9 1 6 3 2 7 10];
permutation_matrix(3,:) = [1 6 8 9 5 4 10 3 2 7];
permutation_matrix(4,:) = [6 8 7 4 9 5 3 1 2 10];
permutation_matrix(5,:) = [5 7 1 3 9 2 4 10 6 8];
permutation_matrix(6,:) = [5 1 6 2 8 10 7 3 9 4];



%-----------------------------------------------------------------------
% Run permutation based second level analysis in Matlab
%-----------------------------------------------------------------------

[sy sx sz st] = size(first_level_results);

statistical_maps_cpu = zeros(sy,sx,sz);
betas_cpu = zeros(sy,sx,sz,size(X_GLM,2));
residuals_cpu = zeros(sy,sx,sz,st);
residual_variances_cpu = zeros(sy,sx,sz);

disp('Calculating statistical maps')

null_distribution_cpu = zeros(number_of_permutations,1);

if do_permutations_in_Matlab == 1
   
    contrast = contrast';
    r = size(contrast,1);
    p = size(contrast,2);
    tmp = diag(ones(p,1)) - contrast'*pinv(contrast');
    [U,D,V] = svd(tmp);
    c2 = U(:,1:p-r);
    c2 = c2';
    C = [contrast ; c2]
   
    W = X_GLM * inv(C);
    W1 = W(:,1:r);
    W2 = W(:,(r+1):end);
    nuisanceModel = W2;
    confounds = (size(W2,2) > 0)
   
    outputModel = W1;
    outputContrast = diag(ones(r,1));
    nuisanceContrast = zeros(r,size(W2,2));
    outputContrast = [outputContrast nuisanceContrast];
    outputModel = [outputModel W2];
   
    contrast = contrast';
    %contrast = outputContrast';
    %X_GLM = outputModel;
    %xtxxt_GLM = inv(X_GLM'*X_GLM)*X_GLM';
    %ctxtxc_GLM = contrast'*inv(X_GLM'*X_GLM)*contrast;
   
    tic
    % Loop over permutations
    for perm = 1:number_of_permutations
       
        statistical_maps_cpu = zeros(sy,sx,sz);
       
        % Loop over voxels
        for x = 1:sx
            for y = 1:sy
                for z = 1:sz
                    if MNI_brain_mask(y,x,z) == 1
                       
                        % Calculate beta values, using permuted model
                        data = single(squeeze(first_level_results(y,x,z,:)));
                       
                        temp = single((diag(ones(size(W2,1),1))-W2*pinv(W2)));
                        outputData = single(temp*data);
                        %outputData = data;
                       
                        permuted_xtxxt_GLM = zeros(size(xtxxt_GLM));
                        permutation = permutation_matrix(perm,:);
                        for reg = 1:number_of_regressors
                            permuted_xtxxt_GLM(reg,:) = xtxxt_GLM(reg,permutation);
                        end
                       
                        beta = single(permuted_xtxxt_GLM) * outputData;
                        betas_cpu(y,x,z,:) = beta;
                       
                        % Calculate t-values and residuals, using permuted model
                        permuted_X_GLM = zeros(size(X_GLM));
                        permutation = permutation_matrix(perm,:);
                        for reg = 1:number_of_regressors
                            permuted_X_GLM(:,reg) = X_GLM(permutation,reg);
                        end
                        residuals = single(outputData - single(permuted_X_GLM)*single(beta));
                        residuals_cpu(y,x,z,:) = residuals;
                        %residual_variances_cpu(y,x,z) = sum((residuals-mean(residuals)).^2)/(st - 1);
                        residual_variances_cpu(y,x,z) = single(sum((residuals).^2)/(st - number_of_regressors));
                       
                        %t-tests
                        for i = 1:size(contrasts,1)
                            %contrast = contrasts(i,:)';
                            statistical_maps_cpu(y,x,z,i) = single(contrast)'*single(beta) / sqrt( single(residual_variances_cpu(y,x,z)) * single(ctxtxc_GLM(i)));
                        end
                       
                    end
                end
            end
        end
       
        % Voxel
        if (inference_mode == 0)
            null_distribution_cpu(perm) = max(statistical_maps_cpu(:));
            % Cluster extent
        elseif (inference_mode == 1)
            a = statistical_maps_cpu(:,:,:,1);
            [labels,N] = bwlabeln(a > cluster_defining_threshold);
           
            cluster_extents = zeros(N,1);
            for i = 1:N
                cluster_extents(i) = sum(labels(:) == i);
            end
            if (not(isempty(cluster_extents)))
                null_distribution_cpu(perm) = max(cluster_extents);
            else
                null_distribution_cpu(perm) = 0;
            end
            % Cluster mass
        elseif (inference_mode == 2)
            a = statistical_maps_cpu(:,:,:,1);
            [labels,N] = bwlabeln(a > cluster_defining_threshold);
           
            cluster_masses = zeros(N,1);
            for i = 1:N
                cluster_masses(i) = sum(a(labels(:) == i));
            end
            null_distribution_cpu(perm) = max(cluster_masses);
        end
    end
    toc
end

end

%--------------------------------------------------------------------------------------
% Run permutation based second level analysis with OpenCL
%--------------------------------------------------------------------------------------

%permutation_matrix = permutation_matrix - 1;

%start = clock;
%[beta_volumes, residuals, residual_variances, statistical_maps_opencl, cluster_indices, null_distribution_opencl, permuted_first_level_results] = ...
%    GLMTTest_SecondLevel_Permutation(first_level_results,MNI_brain_mask, X_GLM,xtxxt_GLM',contrasts,ctxtxc_GLM, uint16(permutation_matrix'), number_of_permutations, inference_mode, cluster_defining_threshold, opencl_platform, opencl_device);
%elapsed_time = etime(clock,start)

%mytimes(number_of_regressors) = elapsed_time;

%s = sort(null_distribution_opencl);
%threshold_opencl = s(round(0.95*number_of_permutations))

%end


% slice = round(0.5*MNI_sz);
%
% figure
% imagesc(beta_volumes(:,:,slice)); colormap gray; colorbar
% title('Beta')
%
% figure
% imagesc(statistical_maps_opencl(:,:,slice,1)); colorbar
% title('t-values')
%
% figure
% imagesc(statistical_maps_opencl(:,:,slice,1) > cluster_defining_threshold); colorbar
% title('t-values above cluster defining threshold')
%
% figure
% imagesc(cluster_indices(:,:,slice,1)); colorbar
% title('Cluster indices')

null_distribution_cpu
s = sort(null_distribution_cpu);
threshold_cpu = s(round(0.95*number_of_permutations))

% s = sort(null_distribution_opencl);
% threshold_opencl = s(round(0.95*number_of_permutations))
%
% if number_of_permutations >= 100
%     figure
%     hist(null_distribution_opencl,50)
% end

% if number_of_permutations < 10
%     [null_distribution_cpu  null_distribution_opencl]
% end

% Compare clustering to Matlab

% Cluster extent
% if (inference_mode == 1)
%
%     a = statistical_maps_opencl(:,:,:,1);
%     [labels,N] = bwlabeln(a > cluster_defining_threshold);
%
%     matlab_sums = zeros(N,1);
%     for i = 1:N
%         matlab_sums(i) = sum(labels(:) == i);
%     end
%
%     N = max(cluster_indices(:));
%     for i = 1:N
%         broccoli_sums(i) = sum(cluster_indices(:) == i);
%     end
%
%     matlab_sums = sort(matlab_sums);
%     broccoli_sums = sort(broccoli_sums);
%
%     %[matlab_sums'; broccoli_sums]'
%
%     cluster_sum_error = sum(matlab_sums(:) - broccoli_sums(:))
%
%     % Cluster mass
% elseif (inference_mode == 2)
%
%     a = statistical_maps_opencl(:,:,:,1);
%     [labels,N] = bwlabeln(a > cluster_defining_threshold);
%
%     matlab_sums = zeros(N,1);
%     for i = 1:N
%         matlab_sums(i) = sum(a(labels(:) == i));
%     end
%
%     N = max(cluster_indices(:));
%     for i = 1:N
%         broccoli_sums(i) = sum(a(cluster_indices(:) == i));
%     end
%
%     matlab_sums = sort(matlab_sums);
%     broccoli_sums = sort(broccoli_sums);
%
%     %[matlab_sums'; broccoli_sums]'
%
%     cluster_sum_error = sum(matlab_sums(:) - broccoli_sums(:))
%
% end
%
%
% volume = load_nii(['/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/testing_scripts/randomise/permtest_tstat1.nii.gz']);
% volume = double(volume.img);
%
% imagesc([ statistical_maps_cpu(:,:,50) - volume(:,:,50)  ]); colorbar
%
% imagesc([ statistical_maps_cpu(:,:,50) - statistical_maps_opencl(:,:,50)  ]); colorbar
%
