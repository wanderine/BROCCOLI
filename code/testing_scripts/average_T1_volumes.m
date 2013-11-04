%--------------------------------------------------------------------------------
% This script compares normalized T1 volumes to the used MNI template, 
% for SPM, FSL, AFNI and BROCCOLI
%--------------------------------------------------------------------------------

clear all
close all
clc

addpath('/home/andek/exportfig')

if ispc
    addpath('D:\nifti_matlab')
    addpath('D:\BROCCOLI_test_data\FSL')    
    basepath = 'D:\';    
elseif isunix
    addpath('/data/andek/MIToolbox/')
    addpath('/home/andek/Research_projects/nifti_matlab')
    basepath_SPM_Normalize = '/data/andek/BROCCOLI_test_data/SPM/normalization/normalize';
    basepath_SPM_Segment = '/data/andek/BROCCOLI_test_data/SPM/normalization/segment';
    basepath_FSL = '/data/andek/BROCCOLI_test_data/FSL/normalization';
    basepath_AFNI = '/data/andek/BROCCOLI_test_data/AFNI/normalization';    
    basepath_BROCCOLI = '/data/andek/BROCCOLI_test_data/BROCCOLI/normalization';    
end

N = 198;

calculate_std = 0;  % Calculate voxel-wise standard deviation or not

% Load MNI brain template
MNI_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(1) 'mm_brain.nii']);
MNI = double(MNI_nii.img);
MNI = MNI/max(MNI(:));
MNI_ = MNI/max(MNI(:)) * 256;

% Load MNI brain mask
MNI_mask_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(1) 'mm_brain_mask.nii']);
MNI_mask = double(MNI_mask_nii.img);

MNI_masked = MNI(MNI_mask == 1);
MNI_masked_ = MNI_masked/max(MNI_masked(:)) * 256;

%-----------------------------------------------------------------
% SPM Normalize
%-------------------------------------------------------------------

mutual_information_SPM_Normalize = zeros(N,1);
correlation_SPM_Normalize = zeros(N,1);
ssd_SPM_Normalize = zeros(N,1);

mean_T1_volume_SPM_Normalize = zeros(182,218,182);
for s = 1:N
    s
    T1 = load_nii([basepath_SPM_Normalize '/SPM_warped_subject_'  num2str(s) '.nii']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));
    mean_T1_volume_SPM_Normalize = mean_T1_volume_SPM_Normalize + T1; 
    T1_masked = T1(MNI_mask == 1);    
    % Calculate NCC, SSD and mutual information
    correlation_SPM_Normalize(s) = corr2(T1_masked(:),MNI_masked(:));
    ssd_SPM_Normalize(s) = sum( (T1_masked(:) - MNI_masked(:)).^2 );
    T1_masked_ = T1_masked/max(T1_masked(:)) * 256;    
    mutual_information_SPM_Normalize(s) = mi(T1_masked_(:),MNI_masked_(:));     
end
mean_T1_volume_SPM_Normalize = mean_T1_volume_SPM_Normalize/N;

if calculate_std == 1
    std_T1_volume_SPM_Normalize = zeros(182,218,182);
    for s = 1:N
        s
        T1 = load_nii([basepath_SPM_Normalize '/SPM_warped_subject_'  num2str(s) '.nii']);
        T1 = double(T1.img);
        T1 = T1/max(T1(:));    
        std_T1_volume_SPM_Normalize = std_T1_volume_SPM_Normalize + sqrt((T1 - MNI) .* (T1 - MNI));
    end
    std_T1_volume_SPM_Normalize = std_T1_volume_SPM_Normalize / N;
end

%-----------------------------------------------------------------
% SPM Segment
%-------------------------------------------------------------------

mutual_information_SPM_Segment = zeros(N,1);
correlation_SPM_Segment = zeros(N,1);
ssd_SPM_Segment = zeros(N,1);

mean_T1_volume_SPM_Segment = zeros(182,218,182);
for s = 1:N
    s
    T1 = load_nii([basepath_SPM_Segment '/SPM_warped_subject_'  num2str(s) '.nii']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));
    mean_T1_volume_SPM_Segment = mean_T1_volume_SPM_Segment + T1; 
    T1_masked = T1(MNI_mask == 1);    
    % Calculate NCC, SSD and mutual information
    correlation_SPM_Segment(s) = corr2(T1_masked(:),MNI_masked(:));
    ssd_SPM_Segment(s) = sum( (T1_masked(:) - MNI_masked(:)).^2 );
    T1_masked_ = T1_masked/max(T1_masked(:)) * 256;    
    mutual_information_SPM_Segment(s) = mi(T1_masked_(:),MNI_masked_(:));     
end
mean_T1_volume_SPM_Segment = mean_T1_volume_SPM_Segment/N;

if calculate_std == 1
    std_T1_volume_SPM_Segment = zeros(182,218,182);
    for s = 1:N
        s
        T1 = load_nii([basepath_SPM_Segment '/SPM_warped_subject_'  num2str(s) '.nii']);
        T1 = double(T1.img);
        T1 = T1/max(T1(:));    
        std_T1_volume_SPM_Segment = std_T1_volume_SPM_Segment + sqrt((T1 - MNI) .* (T1 - MNI));
    end
    std_T1_volume_SPM_Segment = std_T1_volume_SPM_Segment / N;
end


%-----------------------------------------------------------------
% FSL
%-------------------------------------------------------------------

mutual_information_FSL = zeros(N,1);
correlation_FSL = zeros(N,1);
ssd_FSL = zeros(N,1);

mean_T1_volume_FSL = zeros(182,218,182);
for s = 1:N
    s
    T1 = load_nii([basepath_FSL '/FSL_warped_'  num2str(s) '.nii.gz']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));
    mean_T1_volume_FSL = mean_T1_volume_FSL + T1; 
    T1_masked = T1(MNI_mask == 1);    
    % Calculate NCC, SSD and mutual information
    correlation_FSL(s) = corr2(T1_masked(:),MNI_masked(:));
    ssd_FSL(s) = sum( (T1_masked(:) - MNI_masked(:)).^2 );
    T1_masked_ = T1_masked/max(T1_masked(:)) * 256;    
    mutual_information_FSL(s) = mi(T1_masked_(:),MNI_masked_(:));        
end
mean_T1_volume_FSL = mean_T1_volume_FSL/N;

if calculate_std == 1
    std_T1_volume_FSL = zeros(182,218,182);
    for s = 1:N
        s
        T1 = load_nii([basepath_FSL '/FSL_warped_'  num2str(s) '.nii.gz']);
        T1 = double(T1.img);
        T1 = T1/max(T1(:));    
        std_T1_volume_FSL = std_T1_volume_FSL + sqrt((T1 - MNI) .* (T1 - MNI));
    end
    std_T1_volume_FSL = std_T1_volume_FSL / N;
end

%-----------------------------------------------------------------------
% AFNI
%-------------------------------------------------------------------

mutual_information_AFNI = zeros(N,1);
correlation_AFNI = zeros(N,1);
ssd_AFNI = zeros(N,1);

mean_T1_volume_AFNI = zeros(182,218,182);
for s = 1:N
    s
    T1 = load_nii([basepath_AFNI '/AFNI_warped_subject'  num2str(s) '.nii']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));
    mean_T1_volume_AFNI = mean_T1_volume_AFNI + T1;
    T1_masked = T1(MNI_mask == 1);    
    % Calculate NCC, SSD and mutual information
    correlation_AFNI(s) = corr2(T1_masked(:),MNI_masked(:));
    ssd_AFNI(s) = sum( (T1_masked(:) - MNI_masked(:)).^2 );
    T1_masked_ = T1_masked/max(T1_masked(:)) * 256;    
    mutual_information_AFNI(s) = mi(T1_masked_(:),MNI_masked_(:));         
end
mean_T1_volume_AFNI = mean_T1_volume_AFNI/N;

if calculate_std == 1
    std_T1_volume_AFNI = zeros(182,218,182);
    for s = 1:N
        s
        T1 = load_nii([basepath_AFNI '/AFNI_warped_subject'  num2str(s) '.nii']);
        T1 = double(T1.img);
        T1 = T1/max(T1(:));    
        std_T1_volume_AFNI = std_T1_volume_AFNI + sqrt((T1 - MNI) .* (T1 - MNI));
    end
    std_T1_volume_AFNI = std_T1_volume_AFNI / N;
end

%-------------------------------------------------------------------
% BROCCOLI
%-------------------------------------------------------------------

mutual_information_BROCCOLI = zeros(N,1);
correlation_BROCCOLI = zeros(N,1);
ssd_BROCCOLI = zeros(N,1);

mean_T1_volume_BROCCOLI = zeros(182,218,182);
for s = 1:N
    s
    %load([basepath_BROCCOLI '/BROCCOLI_warped_subject' num2str(s) '.mat']);    
    %T1 = aligned_T1_nonparametric_opencl;
    T1 = load_nii([basepath_BROCCOLI '/BROCCOLI_warped_subject'  num2str(s) '.nii']);
    T1 = double(T1.img);    
    T1 = T1/max(T1(:));
    mean_T1_volume_BROCCOLI = mean_T1_volume_BROCCOLI + T1;  
    T1_masked = T1(MNI_mask == 1);        
    % Calculate NCC, SSD and mutual information
    correlation_BROCCOLI(s) = corr2(T1_masked(:),MNI_masked(:));
    ssd_BROCCOLI(s) = sum( (T1_masked(:) - MNI_masked(:)).^2 );
    T1_masked_ = T1_masked/max(T1_masked(:)) * 256;    
    mutual_information_BROCCOLI(s) = mi(T1_masked_(:),MNI_masked_(:));        
end
mean_T1_volume_BROCCOLI = mean_T1_volume_BROCCOLI/N;

if calculate_std == 1
    std_T1_volume_BROCCOLI = zeros(182,218,182);
    for s = 1:N
        s
        %load([basepath_BROCCOLI '/BROCCOLI_warped_subject'  num2str(s) '.mat']);    
        %T1 = aligned_T1_nonparametric_opencl;
        T1 = load_nii([basepath_BROCCOLI '/BROCCOLI_warped_subject'  num2str(s) '.nii']);
        T1 = double(T1.img);    
        T1 = T1/max(T1(:));    
        std_T1_volume_BROCCOLI = std_T1_volume_BROCCOLI + sqrt((T1 - MNI) .* (T1 - MNI));
    end
    std_T1_volume_BROCCOLI = std_T1_volume_BROCCOLI / N;
end

%-------------------------------------------------------------------



close all

% Show average normalized T1 volumes
figure(1)
image([ flipud(MNI(:,:,85)')*50  flipud(mean_T1_volume_SPM_Normalize(:,:,85)')*75 flipud(mean_T1_volume_SPM_Segment(:,:,85)')*75   ; flipud(mean_T1_volume_FSL(:,:,85)')*75 flipud(mean_T1_volume_AFNI(:,:,85)')*75  flipud(mean_T1_volume_BROCCOLI(:,:,85)')*75 ]); colormap gray
axis equal
axis off
text(10,25,'A','Color','White','FontSize',17)
text(200,25,'B','Color','White','FontSize',17)
text(380,25,'C','Color','White','FontSize',17)
text(10,225,'D','Color','White','FontSize',17)
text(200,225,'E','Color','White','FontSize',17)
text(380,225,'F','Color','White','FontSize',17)
%print -dpng /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/axial_.png
export_fig /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/axial_.png -png -native

% Show voxel-wise standard deviation
if calculate_std == 1
    figure(2)
    imagesc([ std_T1_volume_SPM_Normalize(:,:,85) std_T1_volume_SPM_Segment(:,:,85) std_T1_volume_FSL(:,:,85) std_T1_volume_AFNI(:,:,85)  std_T1_volume_BROCCOLI(:,:,85) ]); colormap gray
    axis equal
    axis off
    %print -dpng /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/axial_std.png
end

% Show average normalized T1 volumes
figure(3)
image([ flipud(squeeze(MNI(85,:,:))')*50 flipud(squeeze(mean_T1_volume_SPM_Normalize(85,:,:))')*75 flipud(squeeze(mean_T1_volume_SPM_Segment(85,:,:))')*75    ; flipud(squeeze(mean_T1_volume_FSL(85,:,:))')*75 flipud(squeeze(mean_T1_volume_AFNI(85,:,:))')*75 flipud(squeeze(mean_T1_volume_BROCCOLI(85,:,:))')*75  ]); colormap gray
axis equal
axis off
text(10,25,'A','Color','White','FontSize',18)
text(230,25,'B','Color','White','FontSize',18)
text(450,25,'C','Color','White','FontSize',18)
text(10,225,'D','Color','White','FontSize',18)
text(230,225,'E','Color','White','FontSize',18)
text(450,225,'F','Color','White','FontSize',18)
%print -dpng /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/sagittal_.png
export_fig /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/sagittal_.png -png -native

% Show voxel-wise standard deviation
if calculate_std == 1
    figure(4)
    imagesc([ flipud(squeeze(std_T1_volume_SPM_Normalize(85,:,:))') flipud(squeeze(std_T1_volume_SPM_Segment(85,:,:))') flipud(squeeze(std_T1_volume_FSL(85,:,:))') flipud(squeeze(std_T1_volume_AFNI(85,:,:))') flipud(squeeze(std_T1_volume_BROCCOLI(85,:,:))')  ]); colormap gray
    axis equal
    axis off
    %print -dpng /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/sagittal_std.png
end

mean(correlation_SPM_Normalize)
mean(correlation_SPM_Segment)
mean(correlation_FSL)
mean(correlation_AFNI)
mean(correlation_BROCCOLI)

std(correlation_SPM_Normalize)
std(correlation_SPM_Segment)
std(correlation_FSL)
std(correlation_AFNI)
std(correlation_BROCCOLI)

mean(mutual_information_SPM_Normalize)
mean(mutual_information_SPM_Segment)
mean(mutual_information_FSL)
mean(mutual_information_AFNI)
mean(mutual_information_BROCCOLI)

std(mutual_information_SPM_Normalize)
std(mutual_information_SPM_Segment)
std(mutual_information_FSL)
std(mutual_information_AFNI)
std(mutual_information_BROCCOLI)


mean(ssd_SPM_Normalize/100000)
mean(ssd_SPM_Segment/100000)
mean(ssd_FSL/100000)
mean(ssd_AFNI/100000)
mean(ssd_BROCCOLI/100000)

std(ssd_SPM_Normalize/100000)
std(ssd_SPM_Segment/100000)
std(ssd_FSL/100000)
std(ssd_AFNI/100000)
std(ssd_BROCCOLI/100000)





