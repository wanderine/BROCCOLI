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
    
    %mex -g GLMFTestFirstLevel.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
    mex GLMFTestFirstLevel.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab/')
    basepath = '/data/andek/OpenfMRI/';    
    opencl_platform = 2;
    opencl_device = 0;
    
    %mex -g GLMFTestFirstLevel.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Debug -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
    mex GLMFTestFirstLevel.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
end

study = 'RhymeJudgment/ds003';

subject = 1; 

if subject < 10
    subject = ['/sub00' num2str(subject)];
else
    subject = ['/sub0' num2str(subject)];
end


%-----------------------------------------------------------------------
% Settings
%-----------------------------------------------------------------------

EPI_smoothing_amount = 5.5;
AR_smoothing_amount = 6.0;

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
% Smooth data
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
smoothing_filter_xx = temp;

temp = zeros(9,1,1);
temp(:,1,1) = smoothing_filter_y;
smoothing_filter_yy = temp;

temp = zeros(1,1,9);
temp(1,1,:) = smoothing_filter_z;
smoothing_filter_zz = temp;

smoothed_volumes_cpu = zeros(size(fMRI_volumes));
for t = 1:size(fMRI_volumes,4)
    volume = fMRI_volumes(:,:,:,t);
    smoothed_volume = convn(volume,smoothing_filter_xx,'same');
    smoothed_volume = convn(smoothed_volume,smoothing_filter_yy,'same');
    smoothed_volume = convn(smoothed_volume,smoothing_filter_zz,'same');
    fMRI_volumes(:,:,:,t) = smoothed_volume;
end

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
smoothing_filter_xx = temp;

temp = zeros(9,1,1);
temp(:,1,1) = smoothing_filter_y;
smoothing_filter_yy = temp;

temp = zeros(1,1,9);
temp(1,1,:) = smoothing_filter_z;
smoothing_filter_zz = temp;

volume = brain_mask;
smoothed_volume = convn(volume,smoothing_filter_xx,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_yy,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_zz,'same');
smoothed_mask = smoothed_volume;

%--------------------------------------------------------------------------------------
% Create GLM regressors
%--------------------------------------------------------------------------------------

% Make high resolution regressors first
X_GLM_highres = zeros(st*100000,2);
X_GLM = zeros(st,6);

for regressor = 1:2
    
    fid = fopen([basepath study subject '/model/model001/onsets/task001_run001/cond00' num2str(regressor) '.txt']);
    text = textscan(fid,'%f%f%f');
    fclose(fid);

    onsets = text{1};
    durations = text{2};
    values = text{3};

    for i = 1:length(onsets)    
        start = round(onsets(i)*100000/TR); 
        activity_length = round(durations(i)*100000/TR);
        value = values(i);

        for j = 1:activity_length
            X_GLM_highres(start+j,regressor) = value; 
        end                    
    end
    
    % Downsample
    temp = decimate(X_GLM_highres(:,regressor),100000);
    X_GLM(:,regressor) = temp(1:st);
end

X_GLM(:,3) = ones(st,1);
X_GLM(:,4) = -(st-1)/2:(st-1)/2; a = X_GLM(:,4); X_GLM(:,4) = X_GLM(:,4) / max(a(:));
X_GLM(:,5) = X_GLM(:,4) .* X_GLM(:,4); a = X_GLM(:,5); X_GLM(:,5) = X_GLM(:,5) / max(a(:));
X_GLM(:,6) = X_GLM(:,4) .* X_GLM(:,4) .* X_GLM(:,4) ; a = X_GLM(:,6); X_GLM(:,6) = X_GLM(:,6) / max(a(:));

xtxxt_GLM = inv(X_GLM'*X_GLM)*X_GLM';

%-----------------------------------------------------------------------
% Create contrasts
%-----------------------------------------------------------------------

contrasts = zeros(3,size(X_GLM,2));
contrasts(1,:) = [1 0 0 0 0 0];
contrasts(2,:) = [0 1 0 0 0 0];
contrasts(3,:) = [1 0 0 1 0 0];

ctxtxc_GLM = inv(contrasts*inv(X_GLM'*X_GLM)*contrasts')

%-----------------------------------------------------------------------
% Allocate memory
%-----------------------------------------------------------------------

statistical_maps_cpu_no_whitening = zeros(sy,sx,sz);
statistical_maps_cpu_co = zeros(sy,sx,sz);
betas_cpu_no_whitening = zeros(sy,sx,sz,size(X_GLM,2));
betas_cpu = zeros(sy,sx,sz,size(X_GLM,2));

residuals_cpu = zeros(sy,sx,sz,st);
residuals_cpu_co = zeros(sy,sx,sz,st-4);
residual_variances_cpu_no_whitening = zeros(sy,sx,sz);
residual_variances_cpu = zeros(sy,sx,sz);
residual_variances_cpu_co = zeros(sy,sx,sz);
ar1_estimates_cpu = zeros(sy,sx,sz);
ar2_estimates_cpu = zeros(sy,sx,sz);
ar3_estimates_cpu = zeros(sy,sx,sz);
ar4_estimates_cpu = zeros(sy,sx,sz);
ar1_estimates_cpu_co = zeros(sy,sx,sz);
ar2_estimates_cpu_co = zeros(sy,sx,sz);
ar3_estimates_cpu_co = zeros(sy,sx,sz);
ar4_estimates_cpu_co = zeros(sy,sx,sz);

%-----------------------------------------------------------------------
% Calculate statistical maps, no whitening
%-----------------------------------------------------------------------

% Loop over voxels
disp('Calculating statistical maps, no whitening')
tic
for x = 1:sx
    for y = 1:sy
        for z = 1:sz
            if brain_mask(y,x,z) == 1
                
                % Calculate beta values
                timeseries = squeeze(fMRI_volumes(y,x,z,:));                
                beta = xtxxt_GLM * timeseries;
                betas_cpu_no_whitening(y,x,z,:) = beta;
                
                % Calculate t-values and residuals                
                residuals = timeseries - X_GLM*beta;
                residual_variances_cpu_no_whitening(y,x,z) = sum((residuals-mean(residuals)).^2)/(st - 1);
                
                % F-test
                statistical_maps_cpu_no_whitening(y,x,z) = (contrasts*beta)' * 1/residual_variances_cpu_no_whitening(y,x,z) * ctxtxc_GLM * (contrasts*beta) / size(contrasts,1);
                
            end
        end
    end
end
toc

%-----------------------------------------------------------------------
% Calculate statistical maps, Cochrane-Orcutt
%-----------------------------------------------------------------------


whitened_fMRI_volumes = fMRI_volumes;

whitened_X_GLMs_cpu = zeros(sy,sx,sz,st,6);
whitened_xtxxt_GLMs = zeros(sy,sx,sz,6,st);

% Loop over voxels
for x = 1:sx
    for y = 1:sy
        for z = 1:sz
            whitened_xtxxt_GLMs(y,x,z,:,:) = xtxxt_GLM;
        end
    end
end

INVALID_TIMEPOINTS = 0;

% Cochrane-Orcutt, iterate
for it = 1:3
    
    % Calculate statistical maps
    
    % Loop over voxels
    disp('Doing Cochrane-Orcutt iterations')
    tic
    for x = 1:sx
        for y = 1:sy
            for z = 1:sz
                if brain_mask(y,x,z) == 1
                    
                    % Calculate beta values, using whitened data and the whitened voxel-specific models
                    whitened_timeseries = squeeze(whitened_fMRI_volumes(y,x,z,(INVALID_TIMEPOINTS+1):end));
                    whitened_xtxxt_GLM = squeeze(whitened_xtxxt_GLMs(y,x,z,:,(INVALID_TIMEPOINTS+1):end));
                    beta = whitened_xtxxt_GLM * whitened_timeseries;
                    betas_cpu(y,x,z,:) = beta;
                    
                    % Calculate t-values and residuals, using original data and the original model
                    timeseries = squeeze(fMRI_volumes(y,x,z,:));
                    residuals = timeseries - X_GLM*beta;
                    residuals_cpu(y,x,z,:) = residuals;                                                            
                    
                end
            end
        end
    end
    toc
    
    % Estimate auto correlation from residuals
    % Loop over voxels
    disp('Estimating AR parameters')
    tic
    for x = 1:sx
        for y = 1:sy
            for z = 1:sz
                if brain_mask(y,x,z) == 1
                    
                    c0 = 0; c1 = 0; c2 = 0; c3 = 0; c4 = 0;
                    
                    old_value_1 = residuals_cpu(y,x,z,1+INVALID_TIMEPOINTS);
                    c0 = c0 + old_value_1 * old_value_1;
                    
                    old_value_2 = residuals_cpu(y,x,z,2+INVALID_TIMEPOINTS);
                    c0 = c0 + old_value_2 * old_value_2;
                    c1 = c1 + old_value_2 * old_value_1;
                    
                    old_value_3 = residuals_cpu(y,x,z,3+INVALID_TIMEPOINTS);
                    c0 = c0 + old_value_3 * old_value_3;
                    c1 = c1 + old_value_3 * old_value_2;
                    c2 = c2 + old_value_3 * old_value_1;
                    
                    old_value_4 = residuals_cpu(y,x,z,4+INVALID_TIMEPOINTS);
                    c0 = c0 + old_value_4 * old_value_4;
                    c1 = c1 + old_value_4 * old_value_3;
                    c2 = c2 + old_value_4 * old_value_2;
                    c3 = c3 + old_value_4 * old_value_1;
                    
                    % Estimate c0, c1, c2, c3, c4
                    for t = (5 + INVALID_TIMEPOINTS):st
                        
                        % Read old value
                        old_value_5 = residuals_cpu(y,x,z,t);
                        
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
    toc
    
    % Smooth AR estimates, using normalized convolution
    disp('Smoothing AR parameters')
    tic
    smoothed_volume = convn(ar1_estimates_cpu .* brain_mask,smoothing_filter_xx,'same');
    smoothed_volume = convn(smoothed_volume,smoothing_filter_yy,'same');
    ar1_estimates_cpu = convn(smoothed_volume,smoothing_filter_zz,'same') ./ (smoothed_mask+0.01);
    ar1_estimates_cpu = ar1_estimates_cpu .* brain_mask;
    
    smoothed_volume = convn(ar2_estimates_cpu .* brain_mask,smoothing_filter_xx,'same');
    smoothed_volume = convn(smoothed_volume,smoothing_filter_yy,'same');
    ar2_estimates_cpu = convn(smoothed_volume,smoothing_filter_zz,'same') ./ (smoothed_mask+0.01);
    ar2_estimates_cpu = ar2_estimates_cpu .* brain_mask;
    
    smoothed_volume = convn(ar3_estimates_cpu .* brain_mask,smoothing_filter_xx,'same');
    smoothed_volume = convn(smoothed_volume,smoothing_filter_yy,'same');
    ar3_estimates_cpu = convn(smoothed_volume,smoothing_filter_zz,'same') ./ (smoothed_mask+0.01);
    ar3_estimates_cpu = ar3_estimates_cpu .* brain_mask;
    
    smoothed_volume = convn(ar4_estimates_cpu .* brain_mask,smoothing_filter_xx,'same');
    smoothed_volume = convn(smoothed_volume,smoothing_filter_yy,'same');
    ar4_estimates_cpu = convn(smoothed_volume,smoothing_filter_zz,'same') ./ (smoothed_mask+0.01);
    ar4_estimates_cpu = ar4_estimates_cpu .* brain_mask;
    toc
    
    %ar1_estimates_cpu = ar1_estimates_cpu * 0;
    %ar2_estimates_cpu = ar2_estimates_cpu * 0;
    %ar3_estimates_cpu = ar3_estimates_cpu * 0;
    %ar4_estimates_cpu = ar4_estimates_cpu * 0;
    
    % Apply whitening to data
    
    % Loop over voxels
    disp('Whitening data')
    tic
    for x = 1:sx
        for y = 1:sy
            for z = 1:sz
                if brain_mask(y,x,z) == 1
                    
                    old_value_1 = fMRI_volumes(y,x,z,1);
                    whitened_fMRI_volumes(y,x,z,1) = old_value_1;
                    old_value_2 = fMRI_volumes(y,x,z,2);
                    whitened_fMRI_volumes(y,x,z,2) = old_value_2  - ar1_estimates_cpu(y,x,z) * old_value_1;
                    old_value_3 = fMRI_volumes(y,x,z,3);
                    whitened_fMRI_volumes(y,x,z,3) = old_value_3 - ar1_estimates_cpu(y,x,z) * old_value_2 - ar2_estimates_cpu(y,x,z) * old_value_1;
                    old_value_4 = fMRI_volumes(y,x,z,4);
                    whitened_fMRI_volumes(y,x,z,4) = old_value_4 - ar1_estimates_cpu(y,x,z) * old_value_3 - ar2_estimates_cpu(y,x,z) * old_value_2 - ar3_estimates_cpu(y,x,z) * old_value_1;
                    
                    for t = 5:st
                        
                        old_value_5 = fMRI_volumes(y,x,z,t);
                        
                        whitened_fMRI_volumes(y,x,z,t) = old_value_5 - ar1_estimates_cpu(y,x,z) * old_value_4 - ar2_estimates_cpu(y,x,z) * old_value_3 - ar3_estimates_cpu(y,x,z) * old_value_2 - ar4_estimates_cpu(y,x,z) * old_value_1;
                        
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
    toc
    
    % Apply whitening to models
    % Loop over voxels
    disp('Whitening GLM models')
    tic
    for x = 1:sx
        for y = 1:sy
            for z = 1:sz
                if brain_mask(y,x,z) == 1
                    
                    whitened_X_GLM = zeros(st,6);
                    
                    for r = 1:6
                        
                        old_value_1 = X_GLM(1,r);
                        old_value_2 = X_GLM(2,r);
                        old_value_3 = X_GLM(3,r);
                        old_value_4 = X_GLM(4,r);
                        
                        for t = 5:st
                            
                            old_value_5 = X_GLM(t,r);
                            whitened_X_GLM(t,r) = old_value_5 - ar1_estimates_cpu(y,x,z) * old_value_4 - ar2_estimates_cpu(y,x,z) * old_value_3 - ar3_estimates_cpu(y,x,z) * old_value_2 - ar4_estimates_cpu(y,x,z) * old_value_1;
                            
                            % Save old values
                            old_value_1 = old_value_2;
                            old_value_2 = old_value_3;
                            old_value_3 = old_value_4;
                            old_value_4 = old_value_5;
                            
                        end
                        
                    end
                    
                    whitened_X_GLMs_cpu(y,x,z,:,:) = whitened_X_GLM;
                    
                    % Calculate pseudo inverse
                    whitened_xtxxt_GLM = inv(whitened_X_GLM'*whitened_X_GLM)*whitened_X_GLM';
                    whitened_xtxxt_GLMs(y,x,z,:,:) = whitened_xtxxt_GLM;
                    
                end
            end
        end
    end
    toc
    
    INVALID_TIMEPOINTS = 4;
    
end

INVALID_TIMEPOINTS = 4;


% Loop over voxels
disp('Calculating statistical maps, Cochrane-Orcutt')
tic
for x = 1:sx
    for y = 1:sy
        for z = 1:sz
            if brain_mask(y,x,z) == 1
                
                % Calculate beta values, using whitened data and the whitened voxel-specific models
                whitened_timeseries = squeeze(whitened_fMRI_volumes(y,x,z,(INVALID_TIMEPOINTS+1):end));
                whitened_xtxxt_GLM = squeeze(whitened_xtxxt_GLMs(y,x,z,:,(INVALID_TIMEPOINTS+1):end));
                beta = whitened_xtxxt_GLM * whitened_timeseries;
                betas_cpu(y,x,z,:) = beta;
                
                % Calculate t-values and residuals, using whitened data and the whitened voxel-specific models
                whitened_X_GLM = squeeze(whitened_X_GLMs_cpu(y,x,z,(INVALID_TIMEPOINTS+1):end,:));
                residuals = whitened_timeseries - whitened_X_GLM*beta;
                residuals_cpu_co(y,x,z,:) = residuals;
                residual_variances_cpu_co(y,x,z) = sum((residuals-mean(residuals)).^2)/(st - 1);
                
                % F-test
                contrast_values = inv(contrasts*inv(whitened_X_GLM'*whitened_X_GLM)*contrasts');  
                statistical_maps_cpu_co(y,x,z) = (contrasts*beta)' * 1/residual_variances_cpu_co(y,x,z) * contrast_values * (contrasts*beta) / size(contrasts,1);
                
            end
        end
    end
end
toc

% Estimate auto correlation from residuals
% Loop over voxels
disp('Estimating AR parameters,co')
tic
for x = 1:sx
    for y = 1:sy
        for z = 1:sz
            if brain_mask(y,x,z) == 1
                
                c0 = 0; c1 = 0; c2 = 0; c3 = 0; c4 = 0;
                
                old_value_1 = residuals_cpu_co(y,x,z,1);
                c0 = c0 + old_value_1 * old_value_1;
                
                old_value_2 = residuals_cpu_co(y,x,z,2);
                c0 = c0 + old_value_2 * old_value_2;
                c1 = c1 + old_value_2 * old_value_1;
                
                old_value_3 = residuals_cpu_co(y,x,z,3);
                c0 = c0 + old_value_3 * old_value_3;
                c1 = c1 + old_value_3 * old_value_2;
                c2 = c2 + old_value_3 * old_value_1;
                
                old_value_4 = residuals_cpu_co(y,x,z,4);
                c0 = c0 + old_value_4 * old_value_4;
                c1 = c1 + old_value_4 * old_value_3;
                c2 = c2 + old_value_4 * old_value_2;
                c3 = c3 + old_value_4 * old_value_1;
                
                % Estimate c0, c1, c2, c3, c4
                for t = 1:(st-INVALID_TIMEPOINTS)
                    
                    % Read old value
                    old_value_5 = residuals_cpu_co(y,x,z,t);
                    
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
                ar1_estimates_cpu_co(y,x,z) = AR(1);
                ar2_estimates_cpu_co(y,x,z) = AR(2);
                ar3_estimates_cpu_co(y,x,z) = AR(3);
                ar4_estimates_cpu_co(y,x,z) = AR(4);
                
            end
        end
    end
end
toc

% Smooth AR estimates, using normalized convolution
disp('Smoothing AR parameters, co')
tic
smoothed_volume = convn(ar1_estimates_cpu_co .* brain_mask,smoothing_filter_xx,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_yy,'same');
ar1_estimates_cpu_co = convn(smoothed_volume,smoothing_filter_zz,'same') ./ (smoothed_mask+0.01);
ar1_estimates_cpu_co = ar1_estimates_cpu_co .* brain_mask;

smoothed_volume = convn(ar2_estimates_cpu_co.* brain_mask,smoothing_filter_xx,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_yy,'same');
ar2_estimates_cpu_co = convn(smoothed_volume,smoothing_filter_zz,'same') ./ (smoothed_mask+0.01);
ar2_estimates_cpu_co = ar2_estimates_cpu_co .* brain_mask;

smoothed_volume = convn(ar3_estimates_cpu_co .* brain_mask,smoothing_filter_xx,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_yy,'same');
ar3_estimates_cpu_co = convn(smoothed_volume,smoothing_filter_zz,'same') ./ (smoothed_mask+0.01);
ar3_estimates_cpu_co = ar3_estimates_cpu_co .* brain_mask;

smoothed_volume = convn(ar4_estimates_cpu_co .* brain_mask,smoothing_filter_xx,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_yy,'same');
ar4_estimates_cpu_co = convn(smoothed_volume,smoothing_filter_zz,'same') ./ (smoothed_mask+0.01);
ar4_estimates_cpu_co = ar4_estimates_cpu_co .* brain_mask;



tic
[betas_opencl, residuals_opencl, residual_variances_opencl, statistical_maps_opencl, ...
    ar1_estimates_opencl, ar2_estimates_opencl, ar3_estimates_opencl, ar4_estimates_opencl, ...
    design_matrix, design_matrix2, whitened_X_GLMs_opencl] = ...
    GLMFTestFirstLevel(fMRI_volumes,brain_mask,smoothed_mask,X_GLM,xtxxt_GLM',contrasts,ctxtxc_GLM,EPI_smoothing_amount,AR_smoothing_amount,...
    EPI_voxel_size_x,EPI_voxel_size_y,EPI_voxel_size_z,opencl_platform,opencl_device);
toc

slice = 20;

for r = 1:size(X_GLM,2)
    betas_cpu(:,:,:,r) = betas_cpu(:,:,:,r) .* brain_mask;
    betas_opencl(:,:,:,r) = betas_opencl(:,:,:,r) .* brain_mask;
end

statistical_maps_cpu_co = statistical_maps_cpu_co .* brain_mask;
statistical_maps_opencl = statistical_maps_opencl .* brain_mask;

figure
imagesc([betas_cpu(:,:,slice,1) betas_opencl(:,:,slice,1)]); colorbar
title('Beta')

figure(1)
imagesc([betas_cpu(:,:,slice,1) - betas_opencl(:,:,slice,1)]); colorbar
title('Beta diff')

figure
imagesc([residual_variances_cpu_co(:,:,slice) residual_variances_opencl(:,:,slice)]); colorbar
title('Residual variances')

figure
imagesc([residual_variances_cpu_co(:,:,slice) - residual_variances_opencl(:,:,slice)]); colorbar
title('Residual variances diff')

for slice = 20:20
figure(1)
imagesc([statistical_maps_cpu_co(:,:,slice)  statistical_maps_opencl(:,:,slice)]); colorbar
title('Statistical map')
pause(1)
end

slice = 20;

figure
imagesc([statistical_maps_cpu_co(:,:,slice) - statistical_maps_opencl(:,:,slice)]); colorbar
title('Statistical map diff')

figure
imagesc([ (statistical_maps_cpu_co(:,:,slice) - statistical_maps_opencl(:,:,slice)) ./ statistical_maps_cpu_co(:,:,slice)]); colorbar
title('Relative statistical map diff')

figure
imagesc([ar1_estimates_cpu(:,:,slice) ar1_estimates_opencl(:,:,slice) ar1_estimates_cpu_co(:,:,slice)]); colorbar

figure
imagesc([ar2_estimates_cpu(:,:,slice) ar2_estimates_opencl(:,:,slice) ar2_estimates_cpu_co(:,:,slice)]); colorbar

figure
imagesc([ar3_estimates_cpu(:,:,slice) ar3_estimates_opencl(:,:,slice) ar3_estimates_cpu_co(:,:,slice)]); colorbar

figure
imagesc([ar4_estimates_cpu(:,:,slice) ar4_estimates_opencl(:,:,slice) ar4_estimates_cpu_co(:,:,slice)]); colorbar





beta_tot_error = sum(abs(betas_cpu(:) - betas_opencl(:)))
beta_max_error = max(abs(betas_cpu(:) - betas_opencl(:)))


residual_tot_error = sum(abs(residuals_cpu(:) - residuals_opencl(:)))
residual_max_error = max(abs(residuals_cpu(:) - residuals_opencl(:)))

residual_variances_tot_error = sum(abs(residual_variances_cpu(:) - residual_variances_opencl(:)))
residual_variances_max_error = max(abs(residual_variances_cpu(:) - residual_variances_opencl(:)))

stat_tot_error = sum(abs(statistical_maps_cpu_co(:) - statistical_maps_opencl(:)))
stat_max_error = max(abs(statistical_maps_cpu_co(:) - statistical_maps_opencl(:)))
stat_mean_error = sum(abs(statistical_maps_cpu_co(:) - statistical_maps_opencl(:)))/sum(brain_mask(:))

% for slice = 1:sz
%    a = betas_cpu(:,:,slice,1);
%    b = betas_opencl(:,:,slice,1);
%    
%    max(abs(a(:) - b(:)))
% end


