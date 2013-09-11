clear all
close all
clc

if ispc
    addpath('D:\nifti_matlab')
    addpath('D:\BROCCOLI_test_data\FSL')    
    basepath = 'D:\';    
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab')
    basepath_FSL = '/data/andek/BROCCOLI_test_data/FSL';
    basepath_AFNI = '/data/andek/BROCCOLI_test_data/AFNI';    
    basepath_BROCCOLI = '/data/andek/BROCCOLI_test_data/BROCCOLI';    
end

N = 100;

MNI_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(1) 'mm_brain.nii']);
MNI = double(MNI_nii.img);
MNI = MNI/max(MNI(:));

MNI_mask_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(1) 'mm_brain_mask.nii']);
MNI_mask = double(MNI_mask_nii.img);

%-----------------------------------------------------------------------
% AFNI
%-------------------------------------------------------------------

mean_T1_volume_AFNI = zeros(182,218,182);
for s = 1:N
    T1 = load_nii([basepath_AFNI '/AFNI_warped_subject'  num2str(s) '.nii']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));
    mean_T1_volume_AFNI = mean_T1_volume_AFNI + T1;    
end
mean_T1_volume_AFNI = mean_T1_volume_AFNI/N;

std_T1_volume_AFNI = zeros(182,218,182);
for s = 1:N
    T1 = load_nii([basepath_AFNI '/AFNI_warped_subject'  num2str(s) '.nii']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));
    %std_T1_volume_AFNI = std_T1_volume_AFNI + sqrt((T1 - mean_T1_volume_AFNI) .* (T1 - mean_T1_volume_AFNI));
    std_T1_volume_AFNI = std_T1_volume_AFNI + sqrt((T1 - MNI) .* (T1 - MNI));
end
std_T1_volume_AFNI = std_T1_volume_AFNI / N;

%-----------------------------------------------------------------
% FSL
%-------------------------------------------------------------------

mean_T1_volume_FSL = zeros(182,218,182);
for s = 1:N
    T1 = load_nii([basepath_FSL '/FSL_warped_'  num2str(s) '.nii.gz']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));
    mean_T1_volume_FSL = mean_T1_volume_FSL + T1;    
end
mean_T1_volume_FSL = mean_T1_volume_FSL/N;

std_T1_volume_FSL = zeros(182,218,182);
for s = 1:N
    T1 = load_nii([basepath_FSL '/FSL_warped_'  num2str(s) '.nii.gz']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));
    %std_T1_volume_FSL = std_T1_volume_FSL + sqrt((T1 - mean_T1_volume_FSL) .* (T1 - mean_T1_volume_FSL));
    std_T1_volume_FSL = std_T1_volume_FSL + sqrt((T1 - MNI) .* (T1 - MNI));
end
std_T1_volume_FSL = std_T1_volume_FSL / N;

%-------------------------------------------------------------------
% BROCCOLI
%-------------------------------------------------------------------

mean_T1_volume_BROCCOLI = zeros(182,218,182);
for s = 1:N
    load([basepath_BROCCOLI '/BROCCOLI_warped_subject' num2str(s) '.mat']);    
    T1 = aligned_T1_nonparametric_opencl;
    T1 = T1/max(T1(:));
    mean_T1_volume_BROCCOLI = mean_T1_volume_BROCCOLI + T1;    
end
mean_T1_volume_BROCCOLI = mean_T1_volume_BROCCOLI/N;

std_T1_volume_BROCCOLI = zeros(182,218,182);
for s = 1:N
    load([basepath_BROCCOLI '/BROCCOLI_warped_subject'  num2str(s) '.mat']);    
    T1 = aligned_T1_nonparametric_opencl;
    T1 = T1/max(T1(:));
    %std_T1_volume_BROCCOLI = std_T1_volume_BROCCOLI + sqrt((T1 - mean_T1_volume_BROCCOLI) .* (T1 - mean_T1_volume_BROCCOLI));
    std_T1_volume_BROCCOLI = std_T1_volume_BROCCOLI + sqrt((T1 - MNI) .* (T1 - MNI));
end
std_T1_volume_BROCCOLI = std_T1_volume_BROCCOLI / N;

%-------------------------------------------------------------------



close all

figure
imagesc([ MNI(:,:,85) mean_T1_volume_FSL(:,:,85)*2 mean_T1_volume_AFNI(:,:,85)*2  mean_T1_volume_BROCCOLI(:,:,85)*2 ]); colormap gray
axis off
print -dpng /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/axial.png

figure
imagesc([ std_T1_volume_FSL(:,:,85)*3 std_T1_volume_AFNI(:,:,85)*3  std_T1_volume_BROCCOLI(:,:,85)*3 ]); colormap gray
axis off


figure
imagesc([ flipud(squeeze(MNI(85,:,:))') flipud(squeeze(mean_T1_volume_FSL(85,:,:))')*2 flipud(squeeze(mean_T1_volume_AFNI(85,:,:))')*2 flipud(squeeze(mean_T1_volume_BROCCOLI(85,:,:))')*2       ]); colormap gray
axis off
print -dpng /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/sagittal.png


figure
imagesc([ flipud(squeeze(std_T1_volume_FSL(85,:,:))')*3 flipud(squeeze(std_T1_volume_AFNI(85,:,:))')*3 flipud(squeeze(std_T1_volume_BROCCOLI(85,:,:))')*3       ]); colormap gray
axis off


sum(std_T1_volume_AFNI(:)) / sum(MNI_mask(:))

sum(std_T1_volume_FSL(:)) / sum(MNI_mask(:))

sum(std_T1_volume_BROCCOLI(:)) / sum(MNI_mask(:))




