clear all
close all
clc

if ispc
    addpath('D:\nifti_matlab')
    addpath('D:\BROCCOLI_test_data\AFNI')
    %basepath = 'D:\BROCCOLI_test_data\';
    basepath = 'D:\';
    %basepath = '../../test_data/fcon1000/classic/';   
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab')
    basepath = '/data/andek/BROCCOLI_test_data/AFNI';
    %basepath = '../../test_data/fcon1000/classic/';    
end

mean_T1_volume_AFNI = zeros(182,218,182);
for s = 1:19
    T1 = load_nii([basepath '/AFNI_warped_subject'  num2str(s) '.nii']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));
    mean_T1_volume_AFNI = mean_T1_volume_AFNI + T1/192;    
end

MNI_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(1) 'mm.nii']);
MNI = double(MNI_nii.img);
MNI = MNI/max(MNI(:));

figure
imagesc(mean_T1_volume_AFNI(:,:,85)); colormap gray

figure
imagesc(MNI(:,:,85)); colormap gray


