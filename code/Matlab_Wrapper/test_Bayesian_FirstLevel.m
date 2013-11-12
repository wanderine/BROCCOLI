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
    opencl_platform = 2; % 0 Nvidia, 1 Intel, 2 AMD
    opencl_device = 1;
    
    %mex -g BayesianFirstLevel.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
    mex BayesianFirstLevel.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab/')
    basepath = '/data/andek/OpenfMRI/';
    opencl_platform = 1;
    opencl_device = 0;
    
    %mex -g BayesianFirstLevel.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Debug -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
    mex BayesianFirstLevel.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
end

run_Matlab_equivalent = 0;

study = 'RhymeJudgment/ds003';

subject = 1; %5 has bad skullstrip

if subject < 10
    subject = ['/sub00' num2str(subject)];
else
    subject = ['/sub0' num2str(subject)];
end


%-----------------------------------------------------------------------
% Settings
%-----------------------------------------------------------------------

EPI_smoothing_amount = 5.5;

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

volume = fMRI_volumes(:,:,:,t);
smoothed_volume = convn(volume,smoothing_filter_xx,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_yy,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_zz,'same');

threshold = 0.9 * mean(smoothed_volume(:));
brain_mask = double(volume > threshold);

%--------------------------------------------------------------------------------------
% Create GLM regressors
%--------------------------------------------------------------------------------------

% Make high resolution regressors first
X_GLM_highres = zeros(st*100000,2);
X_GLM = zeros(st,2);

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

xtxxt_GLM = inv(X_GLM'*X_GLM)*X_GLM';

X_GLM_Detrend = zeros(st,4);
X_GLM_Detrend(:,1) = ones(st,1);
X_GLM_Detrend(:,2) = -(st-1)/2:(st-1)/2; a = X_GLM_Detrend(:,2); X_GLM_Detrend(:,2) = X_GLM_Detrend(:,2) / max(a(:));
X_GLM_Detrend(:,3) = X_GLM_Detrend(:,2) .* X_GLM_Detrend(:,2); a = X_GLM_Detrend(:,3); X_GLM_Detrend(:,3) = X_GLM_Detrend(:,3) / max(a(:));
X_GLM_Detrend(:,4) = X_GLM_Detrend(:,2) .* X_GLM_Detrend(:,2) .* X_GLM_Detrend(:,2) ; a = X_GLM_Detrend(:,3); X_GLM_Detrend(:,3) = X_GLM_Detrend(:,3) / max(a(:));
xtxxt_GLM_Detrend = inv(X_GLM_Detrend'*X_GLM_Detrend)*X_GLM_Detrend';


%-----------------------------------------------------------------------
% Create contrasts
%-----------------------------------------------------------------------

contrasts = [1; 0 ; 3 ; 3];

%-----------------------------------------------------------------------
% Calculate statistical maps
%-----------------------------------------------------------------------

[sy sx sz st] = size(fMRI_volumes)

statistical_maps_cpu = zeros(sy,sx,sz);

% Model and Data options
modelOpt.k = 1;                      % Order of autoregressive process for the noise
modelOpt.stimulusPos = 1;            % The index (position) in the beta vector corresponding to the stimulus covariate.

% Prior options
priorOpt.beta0 = 0;                  % Prior mean of beta. p-vector. Scalar value here will be replicated in the vector
priorOpt.tau = 100;                  % Prior covariance matrix of beta is tau^2*(X'X)^-1
priorOpt.iota = 1;                   % Decay factor for lag length in prior for rho.
priorOpt.r = 0.0;                    % Prior mean on rho1
priorOpt.c = 0.3;                    % Prior standard deviation on first lag.
priorOpt.a0 = 0.01;                  % First parameter in IG prior for sigma^2
priorOpt.b0 = 0.01;                  % Second parameter in IG prior for sigma^2

% Algorithmic options
algoOpt.nIter = 1000;               % Number of Gibbs sampling iterations.
algoOpt.prcBurnin = 10;

[T, p] = size(X_GLM);

% Calculate statistical maps

if run_Matlab_equivalent == 1
    
    % Loop over voxels
    disp('Calculating statistical maps')
    tic
    for x = 1:sx
        x
        for y = 1:sy
            for z = 1:sz
                if brain_mask(y,x,z) == 1
                    
                    yy = squeeze(fMRI_volumes(y,x,z,:));
                    beta = xtxxt_GLM_Detrend * yy;
                    detrended = yy - X_GLM_Detrend*beta;
                    [PrActive, paramDraws] = GibbsDynGLM(detrended, X_GLM, modelOpt, priorOpt, algoOpt);
                    statistical_maps_cpu(y,x,z) = PrActive;
                    
                end
            end
        end
    end
    toc
    
end

tic
statistical_maps_opencl = BayesianFirstLevel(fMRI_volumes,brain_mask,X_GLM,xtxxt_GLM',contrasts,opencl_platform,opencl_device);
toc

slice = 20;

figure
imagesc([statistical_maps_cpu(:,:,slice,1) statistical_maps_opencl(:,:,slice,1)]); colorbar
title('Posterior probability map, left = Matlab, right = OpenCL')

figure
imagesc([statistical_maps_cpu(:,:,slice,1) - statistical_maps_opencl(:,:,slice,1)]); colorbar
title('Posterior probability map diff')

a = statistical_maps_opencl(:,:,:,1);
stat_tot_error = sum(abs(statistical_maps_cpu(:) - a(:)))
stat_max_error = max(abs(statistical_maps_cpu(:) - a(:)))

