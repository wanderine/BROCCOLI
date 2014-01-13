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

% Make a mex-file
if ispc
    addpath('D:\nifti_matlab')
    addpath('D:\BROCCOLI_test_data')
    basepath = 'D:\OpenfMRI\';
    opencl_platform = 0;
    opencl_device = 0;
    
    %mex -g GLMFTestFirstLevelPermutation.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
    mex GLMFTestFirstLevelPermutation.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab/')
    basepath = '/data/andek/OpenfMRI/';
    opencl_platform = 2;
    opencl_device = 0;
    
    %mex -g GLMFTestFirstLevelPermutation.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Debug -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
    mex GLMFTestFirstLevelPermutation.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
end

study = 'RhymeJudgment/ds003';

subject = 2;

if subject < 10
    subject = ['/sub00' num2str(subject)];
else
    subject = ['/sub0' num2str(subject)];
end

%-----------------------------------------------------------------------
% Settings
%-----------------------------------------------------------------------

do_Matlab_permutations = 0;
EPI_smoothing_amount = 5.5;
AR_smoothing_amount = 7.0;
number_of_permutations = 1000;
inference_mode = 2; % 0 = voxel, 1 = cluster extent, 2 = cluster mass
cluster_defining_threshold = 2;
number_of_regressors = 5;

%-----------------------------------------------------------------------
% Load data
%-----------------------------------------------------------------------

EPI_nii = load_nii([basepath study subject '/BOLD/task001_run001/bold.nii.gz']);

EPI_voxel_size_x = EPI_nii.hdr.dime.pixdim(2);
EPI_voxel_size_y = EPI_nii.hdr.dime.pixdim(3);
EPI_voxel_size_z = EPI_nii.hdr.dime.pixdim(4);
TR = 2;

fMRI_volumes = double(EPI_nii.img);
[sy sx sz st] = size(fMRI_volumes);
[sy sx sz st]

%-----------------------------------------------------------------------
% Create smoothing filters for data
%-----------------------------------------------------------------------

sigma_x = EPI_smoothing_amount / 2.354 / EPI_voxel_size_x;
sigma_y = EPI_smoothing_amount / 2.354 / EPI_voxel_size_y;
sigma_z = EPI_smoothing_amount / 2.354 / EPI_voxel_size_z;

smoothing_filter_x = fspecial('gaussian',9,sigma_x);
smoothing_filter_x = smoothing_filter_x(:,5);
smoothing_filter_x = smoothing_filter_x / sum(abs(smoothing_filter_x));

smoothing_filter_y = fspecial('gaussian',9,sigma_y);
smoothing_filter_y = smoothing_filter_y(:,5);
smoothing_filter_y = smoothing_filter_y / sum(abs(smoothing_filter_y));

smoothing_filter_z = fspecial('gaussian',9,sigma_z);
smoothing_filter_z = smoothing_filter_z(:,5);
smoothing_filter_z = smoothing_filter_z / sum(abs(smoothing_filter_z));

temp = zeros(1,9,1);
temp(1,:,1) = smoothing_filter_x;
smoothing_filter_data_xx = temp;

temp = zeros(9,1,1);
temp(:,1,1) = smoothing_filter_y;
smoothing_filter_data_yy = temp;

temp = zeros(1,1,9);
temp(1,1,:) = smoothing_filter_z;
smoothing_filter_data_zz = temp;


%-----------------------------------------------------------------------
% Create mask
%-----------------------------------------------------------------------

% Smooth with 4 mm filter
sigma_x = 4 / 2.354 / EPI_voxel_size_x;
sigma_y = 4 / 2.354 / EPI_voxel_size_y;
sigma_z = 4 / 2.354 / EPI_voxel_size_z;

smoothing_filter_x = fspecial('gaussian',9,sigma_x);
smoothing_filter_x = smoothing_filter_x(:,5);
smoothing_filter_x = smoothing_filter_x / sum(abs(smoothing_filter_x));

smoothing_filter_y = fspecial('gaussian',9,sigma_y);
smoothing_filter_y = smoothing_filter_y(:,5);
smoothing_filter_y = smoothing_filter_y / sum(abs(smoothing_filter_y));

smoothing_filter_z = fspecial('gaussian',9,sigma_z);
smoothing_filter_z = smoothing_filter_z(:,5);
smoothing_filter_z = smoothing_filter_z / sum(abs(smoothing_filter_z));

temp = zeros(1,9,1);
temp(1,:,1) = smoothing_filter_x;
smoothing_filter_xx = temp;

temp = zeros(9,1,1);
temp(:,1,1) = smoothing_filter_y;
smoothing_filter_yy = temp;

temp = zeros(1,1,9);
temp(1,1,:) = smoothing_filter_z;
smoothing_filter_zz = temp;

volume = fMRI_volumes(:,:,:,1);
smoothed_volume = convn(volume,smoothing_filter_xx,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_yy,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_zz,'same');

threshold = 0.9 * mean(smoothed_volume(:));
brain_mask = double(volume > threshold);
brain_mask(brain_mask == 0) = 0.0001;

volume = brain_mask;
smoothed_volume = convn(volume,smoothing_filter_data_xx,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_data_yy,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_data_zz,'same');
smoothed_mask_data = smoothed_volume;

%--------------------------------------------------------------------------------------
% Create AR smoothing filters
%--------------------------------------------------------------------------------------

sigma_x = AR_smoothing_amount / 2.354 / EPI_voxel_size_x;
sigma_y = AR_smoothing_amount / 2.354 / EPI_voxel_size_y;
sigma_z = AR_smoothing_amount / 2.354 / EPI_voxel_size_z;

smoothing_filter_x = fspecial('gaussian',9,sigma_x);
smoothing_filter_x = smoothing_filter_x(:,5);
smoothing_filter_x = smoothing_filter_x / sum(abs(smoothing_filter_x));

smoothing_filter_y = fspecial('gaussian',9,sigma_y);
smoothing_filter_y = smoothing_filter_y(:,5);
smoothing_filter_y = smoothing_filter_y / sum(abs(smoothing_filter_y));

smoothing_filter_z = fspecial('gaussian',9,sigma_z);
smoothing_filter_z = smoothing_filter_z(:,5);
smoothing_filter_z = smoothing_filter_z / sum(abs(smoothing_filter_z));

temp = zeros(1,9,1);
temp(1,:,1) = smoothing_filter_x;
smoothing_filter_AR_xx = temp;

temp = zeros(9,1,1);
temp(:,1,1) = smoothing_filter_y;
smoothing_filter_AR_yy = temp;

temp = zeros(1,1,9);
temp(1,1,:) = smoothing_filter_z;
smoothing_filter_AR_zz = temp;

volume = brain_mask;
smoothed_volume = convn(volume,smoothing_filter_AR_xx,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_AR_yy,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_AR_zz,'same');
smoothed_mask_AR = smoothed_volume;

%--------------------------------------------------------------------------------------
% Create GLM regressors
%--------------------------------------------------------------------------------------

% Make high resolution regressors first
X_GLM_highres = zeros(st*1000,2);
X_GLM = zeros(st,number_of_regressors);

for regressor = 1:2
    
    fid = fopen([basepath study subject '/model/model001/onsets/task001_run001/cond00' num2str(regressor) '.txt']);
    text = textscan(fid,'%f%f%f');
    fclose(fid);
    
    onsets = text{1};
    durations = text{2};
    values = text{3};
    
    for i = 1:length(onsets)
        start = round(onsets(i)*1000/TR);
        activity_length = round(durations(i)*1000/TR);
        value = values(i);
        
        for j = 1:activity_length
            X_GLM_highres(start+j,regressor) = value;
        end
    end
    
    % Downsample
    temp = decimate(X_GLM_highres(:,regressor),1000);
    X_GLM(:,regressor) = temp(1:st);
end

for r = 3:number_of_regressors
    X_GLM(:,r) = randn(st,1);
end

xtxxt_GLM = inv(X_GLM'*X_GLM)*X_GLM';

%-----------------------------------------------------------------------
% Create contrasts
%-----------------------------------------------------------------------

%contrasts = [2 1 zeros(1,number_of_regressors-2);
%             1 -1 zeros(1,number_of_regressors-2);
%             ];

contrasts = [2 1 randn(1,number_of_regressors-2);
    1 -1 randn(1,number_of_regressors-2);
    ];

ctxtxc_GLM = inv(contrasts*inv(X_GLM'*X_GLM)*contrasts');

%-----------------------------------------------------------------------
% Create permutation matrix
%-----------------------------------------------------------------------

permutation_matrix = zeros(number_of_permutations,st);
values = 1:st;
for p = 1:number_of_permutations
    permutation = randperm(st);
    permutation_matrix(p,:) = values(permutation);
end

%-----------------------------------------------------------------------
% Detrend and whiten data prior to permutations
%-----------------------------------------------------------------------

if do_Matlab_permutations == 1
    
    detrended_volumes_cpu = zeros(size(fMRI_volumes));
    whitened_volumes_cpu = zeros(size(fMRI_volumes));
    permuted_volumes_cpu = zeros(size(fMRI_volumes));
    statistical_maps_cpu = zeros(sy,sx,sz);
    betas_cpu = zeros(sy,sx,sz,size(X_GLM,2));
    
    ar1_estimates_cpu = zeros(sy,sx,sz);
    ar2_estimates_cpu = zeros(sy,sx,sz);
    ar3_estimates_cpu = zeros(sy,sx,sz);
    ar4_estimates_cpu = zeros(sy,sx,sz);
    
    total_ar1_estimates_cpu = zeros(sy,sx,sz);
    total_ar2_estimates_cpu = zeros(sy,sx,sz);
    total_ar3_estimates_cpu = zeros(sy,sx,sz);
    total_ar4_estimates_cpu = zeros(sy,sx,sz);
    
    X_Detrend = zeros(st,4);
    X_Detrend(:,1) = ones(st,1);
    X_Detrend(:,2) = -(st-1)/2:(st-1)/2; a = X_Detrend(:,2); X_Detrend(:,2) = X_Detrend(:,2) / max(a(:));
    X_Detrend(:,3) = X_Detrend(:,2) .* X_Detrend(:,2); a = X_Detrend(:,3); X_Detrend(:,3) = X_Detrend(:,3) / max(a(:));
    X_Detrend(:,4) = X_Detrend(:,2) .* X_Detrend(:,2) .* X_Detrend(:,2) ; a = X_Detrend(:,4); X_Detrend(:,4) = X_Detrend(:,4) / max(a(:));
    
    xtxxt_Detrend = inv(X_Detrend'*X_Detrend)*X_Detrend';
    
    % Detrend
    disp('Detrending')
    for x = 1:sx
        for y = 1:sy
            for z = 1:sz
                if brain_mask(y,x,z) == 1
                    % Calculate beta values
                    timeseries = squeeze(fMRI_volumes(y,x,z,:));
                    beta = xtxxt_Detrend * timeseries;
                    
                    % Calculate detrended timeseries
                    detrended_volumes_cpu(y,x,z,:) = timeseries - X_Detrend*beta;
                end
            end
        end
    end
    
    INVALID_TIMEPOINTS = 0;
    
    whitened_volumes_cpu = detrended_volumes_cpu;
    
    for i = 1:3
        
        % Estimate auto correlation, loop over voxels
        disp('Estimating AR parameters')
        for x = 1:sx
            for y = 1:sy
                for z = 1:sz
                    if brain_mask(y,x,z) == 1
                        
                        c0 = 0; c1 = 0; c2 = 0; c3 = 0; c4 = 0;
                        
                        old_value_1 = whitened_volumes_cpu(y,x,z,1+INVALID_TIMEPOINTS);
                        c0 = c0 + old_value_1 * old_value_1;
                        
                        old_value_2 = whitened_volumes_cpu(y,x,z,2+INVALID_TIMEPOINTS);
                        c0 = c0 + old_value_2 * old_value_2;
                        c1 = c1 + old_value_2 * old_value_1;
                        
                        old_value_3 = whitened_volumes_cpu(y,x,z,3+INVALID_TIMEPOINTS);
                        c0 = c0 + old_value_3 * old_value_3;
                        c1 = c1 + old_value_3 * old_value_2;
                        c2 = c2 + old_value_3 * old_value_1;
                        
                        old_value_4 = whitened_volumes_cpu(y,x,z,4+INVALID_TIMEPOINTS);
                        c0 = c0 + old_value_4 * old_value_4;
                        c1 = c1 + old_value_4 * old_value_3;
                        c2 = c2 + old_value_4 * old_value_2;
                        c3 = c3 + old_value_4 * old_value_1;
                        
                        % Estimate c0, c1, c2, c3, c4
                        for t = (5 + INVALID_TIMEPOINTS):st
                            
                            % Read old value
                            old_value_5 = whitened_volumes_cpu(y,x,z,t);
                            
                            % Sum and multiply the values
                            c0 = c0 + old_value_5 * old_value_5;
                            c1 = c1 + old_value_5 * old_value_4;
                            c2 = c2 + old_value_5 * old_value_3;
                            c3 = c3 + old_value_5 * old_value_2;
                            c4 = c4 + old_value_5 * old_value_1;
                            
                            % Save old values
                            old_value_1 = old_value_2;
                            old_value_2 = old_value_3;
                            old_value_3 = old_value_4;
                            old_value_4 = old_value_5;
                        end
                        
                        c0 = c0 / (st - 1 - INVALID_TIMEPOINTS);
                        c1 = c1 / (st - 2 - INVALID_TIMEPOINTS);
                        c2 = c2 / (st - 3 - INVALID_TIMEPOINTS);
                        c3 = c3 / (st - 4 - INVALID_TIMEPOINTS);
                        c4 = c4 / (st - 5 - INVALID_TIMEPOINTS);
                        
                        r1 = c1/c0;
                        r2 = c2/c0;
                        r3 = c3/c0;
                        r4 = c4/c0;
                        
                        R = zeros(4,4);
                        R(1,1) = 1;
                        R(1,2) = r1;
                        R(1,3) = r2;
                        R(1,4) = r3;
                        
                        R(2,1) = r1;
                        R(2,2) = 1;
                        R(2,3) = r1;
                        R(2,4) = r2;
                        
                        R(3,1) = r2;
                        R(3,2) = r1;
                        R(3,3) = 1;
                        R(3,4) = r1;
                        
                        R(4,1) = r3;
                        R(4,2) = r2;
                        R(4,3) = r1;
                        R(4,4) = 1;
                        
                        % Get AR parameters
                        AR = inv(R) * [r1; r2; r3; r4];
                        ar1_estimates_cpu(y,x,z) = AR(1);
                        ar2_estimates_cpu(y,x,z) = AR(2);
                        ar3_estimates_cpu(y,x,z) = AR(3);
                        ar4_estimates_cpu(y,x,z) = AR(4);
                        
                    end
                end
            end
        end
        
        % Smooth AR estimates, using normalized convolution
        disp('Smoothing AR parameters')
        smoothed_volume = convn(ar1_estimates_cpu .* brain_mask,smoothing_filter_AR_xx,'same');
        smoothed_volume = convn(smoothed_volume,smoothing_filter_AR_yy,'same');
        ar1_estimates_cpu = convn(smoothed_volume,smoothing_filter_AR_zz,'same') ./ smoothed_mask_AR;
        ar1_estimates_cpu = ar1_estimates_cpu .* brain_mask;
        
        smoothed_volume = convn(ar2_estimates_cpu .* brain_mask,smoothing_filter_AR_xx,'same');
        smoothed_volume = convn(smoothed_volume,smoothing_filter_AR_yy,'same');
        ar2_estimates_cpu = convn(smoothed_volume,smoothing_filter_AR_zz,'same') ./ smoothed_mask_AR;
        ar2_estimates_cpu = ar2_estimates_cpu .* brain_mask;
        
        smoothed_volume = convn(ar3_estimates_cpu .* brain_mask,smoothing_filter_AR_xx,'same');
        smoothed_volume = convn(smoothed_volume,smoothing_filter_AR_yy,'same');
        ar3_estimates_cpu = convn(smoothed_volume,smoothing_filter_AR_zz,'same') ./ smoothed_mask_AR;
        ar3_estimates_cpu = ar3_estimates_cpu .* brain_mask;
        
        smoothed_volume = convn(ar4_estimates_cpu .* brain_mask,smoothing_filter_AR_xx,'same');
        smoothed_volume = convn(smoothed_volume,smoothing_filter_AR_yy,'same');
        ar4_estimates_cpu = convn(smoothed_volume,smoothing_filter_AR_zz,'same') ./ smoothed_mask_AR;
        ar4_estimates_cpu = ar4_estimates_cpu .* brain_mask;
        
        % Add to total parameters
        total_ar1_estimates_cpu = total_ar1_estimates_cpu + ar1_estimates_cpu;
        total_ar2_estimates_cpu = total_ar2_estimates_cpu + ar2_estimates_cpu;
        total_ar3_estimates_cpu = total_ar3_estimates_cpu + ar3_estimates_cpu;
        total_ar4_estimates_cpu = total_ar4_estimates_cpu + ar4_estimates_cpu;
        
        % Apply whitening to data
        
        % Loop over voxels
        disp('Whitening data')
        for x = 1:sx
            for y = 1:sy
                for z = 1:sz
                    if brain_mask(y,x,z) == 1
                        
                        old_value_1 = detrended_volumes_cpu(y,x,z,1);
                        whitened_volumes_cpu(y,x,z,1) = old_value_1;
                        old_value_2 = detrended_volumes_cpu(y,x,z,2);
                        whitened_volumes_cpu(y,x,z,2) = old_value_2  - total_ar1_estimates_cpu(y,x,z) * old_value_1;
                        old_value_3 = detrended_volumes_cpu(y,x,z,3);
                        whitened_volumes_cpu(y,x,z,3) = old_value_3 - total_ar1_estimates_cpu(y,x,z) * old_value_2 - total_ar2_estimates_cpu(y,x,z) * old_value_1;
                        old_value_4 = detrended_volumes_cpu(y,x,z,4);
                        whitened_volumes_cpu(y,x,z,4) = old_value_4 - total_ar1_estimates_cpu(y,x,z) * old_value_3 - total_ar2_estimates_cpu(y,x,z) * old_value_2 - total_ar3_estimates_cpu(y,x,z) * old_value_1;
                        
                        for t = 5:st
                            
                            old_value_5 = detrended_volumes_cpu(y,x,z,t);
                            
                            whitened_volumes_cpu(y,x,z,t) = old_value_5 - total_ar1_estimates_cpu(y,x,z) * old_value_4 - total_ar2_estimates_cpu(y,x,z) * old_value_3 - total_ar3_estimates_cpu(y,x,z) * old_value_2 - total_ar4_estimates_cpu(y,x,z) * old_value_1;
                            
                            % Save old values
                            old_value_1 = old_value_2;
                            old_value_2 = old_value_3;
                            old_value_3 = old_value_4;
                            old_value_4 = old_value_5;
                            
                        end
                    end
                end
            end
        end
        
        INVALID_TIMEPOINTS = 4;
    end
    
    
    %-----------------------------------------------------------------------
    % Calculate statistical maps for each permutation
    %-----------------------------------------------------------------------
    
    null_distribution_cpu = zeros(number_of_permutations,1);
    
    
    
    for p = 1:number_of_permutations
        
        p
        start = clock;
        
        permutation_vector = permutation_matrix(p,:);
        
        % Do inverse whitening transform and randomly permute volumes
        disp('Generating new fMRI data')
        for x = 1:sx
            for y = 1:sy
                for z = 1:sz
                    if brain_mask(y,x,z) == 1
                        
                        old_value_1 = whitened_volumes_cpu(y,x,z,permutation_vector(1));
                        old_value_2 = total_ar1_estimates_cpu(y,x,z) * old_value_1 + whitened_volumes_cpu(y,x,z,permutation_vector(2));
                        old_value_3 = total_ar1_estimates_cpu(y,x,z) * old_value_2 + total_ar2_estimates_cpu(y,x,z) * old_value_1 + whitened_volumes_cpu(y,x,z,permutation_vector(3));
                        old_value_4 = total_ar1_estimates_cpu(y,x,z) * old_value_3 + total_ar2_estimates_cpu(y,x,z) * old_value_2 + total_ar3_estimates_cpu(y,x,z) * old_value_1 + whitened_volumes_cpu(y,x,z,permutation_vector(4));
                        
                        permuted_volumes_cpu(y,x,z,1) = old_value_1;
                        permuted_volumes_cpu(y,x,z,2) = old_value_2;
                        permuted_volumes_cpu(y,x,z,3) = old_value_3;
                        permuted_volumes_cpu(y,x,z,4) = old_value_4;
                        
                        for t = 5:st
                            
                            old_value_5 = total_ar1_estimates_cpu(y,x,z) * old_value_4 + total_ar2_estimates_cpu(y,x,z) * old_value_3 + total_ar3_estimates_cpu(y,x,z) * old_value_2 + total_ar4_estimates_cpu(y,x,z) * old_value_1 + whitened_volumes_cpu(y,x,z,permutation_vector(t));
                            
                            permuted_volumes_cpu(y,x,z,t) = old_value_5;
                            
                            % Save old values
                            old_value_1 = old_value_2;
                            old_value_2 = old_value_3;
                            old_value_3 = old_value_4;
                            old_value_4 = old_value_5;
                            
                        end
                    end
                end
            end
        end
        
        
        % Smooth data
        disp('Smoothing')
        smoothed_volumes_cpu = zeros(size(fMRI_volumes));
        for t = 1:size(fMRI_volumes,4)
            volume = permuted_volumes_cpu(:,:,:,t) .* brain_mask;
            smoothed_volume = convn(volume,smoothing_filter_data_xx,'same');
            smoothed_volume = convn(smoothed_volume,smoothing_filter_data_yy,'same');
            smoothed_volume = convn(smoothed_volume,smoothing_filter_data_zz,'same') ./ smoothed_mask_data;
            smoothed_volumes_cpu(:,:,:,t) = smoothed_volume .* brain_mask;
        end
        
        % Loop over voxels
        disp('Calculating statistical maps')
        for x = 1:sx
            for y = 1:sy
                for z = 1:sz
                    if brain_mask(y,x,z) == 1
                        
                        % Calculate beta values
                        timeseries = squeeze(smoothed_volumes_cpu(y,x,z,:));
                        beta = xtxxt_GLM * timeseries;
                        
                        % Calculate residuals
                        residuals = timeseries - X_GLM*beta;
                        
                        % F-test
                        statistical_maps_cpu(y,x,z) = (contrasts*beta)' * 1/var(residuals) * ctxtxc_GLM * (contrasts*beta) / size(contrasts,1);
                        
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
        
        etime(clock,start)
        
    end
end

permutation_matrix = permutation_matrix - 1;

tic
[betas_opencl, residuals_opencl, residual_variances_opencl, statistical_maps_opencl, ...
    ar1_estimates_opencl, ar2_estimates_opencl, ar3_estimates_opencl, ar4_estimates_opencl, ...
    cluster_indices, detrended_volumes_opencl, whitened_volumes_opencl, permuted_volumes_opencl, null_distribution_opencl] = ...
    GLMFTestFirstLevelPermutation(fMRI_volumes,brain_mask,smoothed_mask_data,X_GLM,xtxxt_GLM',contrasts,ctxtxc_GLM,EPI_smoothing_amount,AR_smoothing_amount,...
    EPI_voxel_size_x,EPI_voxel_size_y,EPI_voxel_size_z,uint16(permutation_matrix'),number_of_permutations,inference_mode,cluster_defining_threshold,opencl_platform,opencl_device);
toc

if do_Matlab_permutations == 1
    
    figure
    imagesc([ total_ar1_estimates_cpu(:,:,20) ar1_estimates_opencl(:,:,20) ]); colorbar
    
    figure
    imagesc([ total_ar2_estimates_cpu(:,:,20) ar2_estimates_opencl(:,:,20) ]); colorbar
    
    figure
    imagesc([ total_ar3_estimates_cpu(:,:,20) ar3_estimates_opencl(:,:,20) ]); colorbar
    
    figure
    imagesc([ total_ar4_estimates_cpu(:,:,20) ar4_estimates_opencl(:,:,20) ]); colorbar
       
    null_distribution_cpu = sort(null_distribution_cpu);
    threshold_cpu = null_distribution_cpu(round(0.95*number_of_permutations))
    
    null_distribution_opencl = sort(null_distribution_opencl);
    threshold_opencl = null_distribution_opencl(round(0.95*number_of_permutations))

    [null_distribution_opencl null_distribution_cpu]
    
end

figure
hist(null_distribution_opencl,50)


%  for z = 1:sz
%      z
%      a = statistical_maps_cpu(:,:,z);
%      b = statistical_maps_opencl(:,:,z);
%      %max(a(:))
%      %max(b(:))
%      max(a(:) - b(:))
%  end


