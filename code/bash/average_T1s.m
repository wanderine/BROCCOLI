clear all
close all
clc

if ispc
    addpath('D:\nifti_matlab')
    addpath('D:\BROCCOLI_test_data\FSL')    
    basepath = 'D:\';    
elseif isunix
    addpath('/data/andek/MIToolbox/')
    addpath('/home/andek/Research_projects/nifti_matlab')
    basepath_SPM = '/data/andek/BROCCOLI_test_data/SPM';
    basepath_FSL = '/data/andek/BROCCOLI_test_data/FSL';
    basepath_AFNI = '/data/andek/BROCCOLI_test_data/AFNI';    
    basepath_BROCCOLI = '/data/andek/BROCCOLI_test_data/BROCCOLI';    
end

N = 198;

MNI_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(1) 'mm_brain.nii']);
MNI = double(MNI_nii.img);
MNI = MNI/max(MNI(:));
MNI_ = MNI/max(MNI(:)) * 256;

MNI_mask_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(1) 'mm_brain_mask.nii']);
MNI_mask = double(MNI_mask_nii.img);

MNI_masked = MNI(MNI_mask == 1);
MNI_masked_ = MNI_masked/max(MNI_masked(:)) * 256;

%-----------------------------------------------------------------
% SPM
%-------------------------------------------------------------------

mutual_information_SPM = zeros(N,1);
correlation_SPM = zeros(N,1);
ssd_SPM = zeros(N,1);

mean_T1_volume_SPM = zeros(182,218,182);
for s = 1:N
    s
    T1 = load_nii([basepath_SPM '/SPM_warped_subject_'  num2str(s) '.nii']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));
    mean_T1_volume_SPM = mean_T1_volume_SPM + T1; 
    T1_masked = T1(MNI_mask == 1);    
    correlation_SPM(s) = corr2(T1_masked(:),MNI_masked(:));
    ssd_SPM(s) = sum( (T1_masked(:) - MNI_masked(:)).^2 );
    T1_masked_ = T1_masked/max(T1_masked(:)) * 256;    
    mutual_information_SPM(s) = mi(T1_masked_(:),MNI_masked_(:));    
    %correlation_SPM(s) = corr2(T1(:),MNI(:));
    %ssd_SPM(s) = sum( (T1(:) - MNI(:)).^2 );
    %T1_ = T1/max(T1(:)) * 256;    
    %mutual_information_SPM(s) = mi(T1_(:),MNI_(:));        
end
mean_T1_volume_SPM = mean_T1_volume_SPM/N;

std_T1_volume_SPM = zeros(182,218,182);
for s = 1:N
    s
    T1 = load_nii([basepath_SPM '/SPM_warped_subject_'  num2str(s) '.nii']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));    
    std_T1_volume_SPM = std_T1_volume_SPM + sqrt((T1 - MNI) .* (T1 - MNI));
end
std_T1_volume_SPM = std_T1_volume_SPM / N;



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
    correlation_FSL(s) = corr2(T1_masked(:),MNI_masked(:));
    ssd_FSL(s) = sum( (T1_masked(:) - MNI_masked(:)).^2 );
    T1_masked_ = T1_masked/max(T1_masked(:)) * 256;    
    mutual_information_FSL(s) = mi(T1_masked_(:),MNI_masked_(:));    
    %correlation_FSL(s) = corr2(T1(:),MNI(:));
    %ssd_FSL(s) = sum( (T1(:) - MNI(:)).^2 );
    %T1_ = T1/max(T1(:)) * 256;    
    %mutual_information_FSL(s) = mi(T1_(:),MNI_(:));    
end
mean_T1_volume_FSL = mean_T1_volume_FSL/N;

std_T1_volume_FSL = zeros(182,218,182);
for s = 1:N
    s
    T1 = load_nii([basepath_FSL '/FSL_warped_'  num2str(s) '.nii.gz']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));    
    std_T1_volume_FSL = std_T1_volume_FSL + sqrt((T1 - MNI) .* (T1 - MNI));
end
std_T1_volume_FSL = std_T1_volume_FSL / N;

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
    correlation_AFNI(s) = corr2(T1_masked(:),MNI_masked(:));
    ssd_AFNI(s) = sum( (T1_masked(:) - MNI_masked(:)).^2 );
    T1_masked_ = T1_masked/max(T1_masked(:)) * 256;    
    mutual_information_AFNI(s) = mi(T1_masked_(:),MNI_masked_(:));    
    %correlation_AFNI(s) = corr2(T1(:),MNI(:));
    %ssd_AFNI(s) = sum( (T1(:) - MNI(:)).^2 );
    %T1_ = T1/max(T1(:)) * 256;    
    %mutual_information_AFNI(s) = mi(T1_(:),MNI_(:));    
end
mean_T1_volume_AFNI = mean_T1_volume_AFNI/N;

std_T1_volume_AFNI = zeros(182,218,182);
for s = 1:N
    s
    T1 = load_nii([basepath_AFNI '/AFNI_warped_subject'  num2str(s) '.nii']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));    
    std_T1_volume_AFNI = std_T1_volume_AFNI + sqrt((T1 - MNI) .* (T1 - MNI));
end
std_T1_volume_AFNI = std_T1_volume_AFNI / N;


%-------------------------------------------------------------------
% BROCCOLI
%-------------------------------------------------------------------

mutual_information_BROCCOLI = zeros(N,1);
correlation_BROCCOLI = zeros(N,1);
ssd_BROCCOLI = zeros(N,1);

mean_T1_volume_BROCCOLI = zeros(182,218,182);
for s = 1:N
    s
    load([basepath_BROCCOLI '/BROCCOLI_warped_subject' num2str(s) '.mat']);    
    T1 = aligned_T1_nonparametric_opencl;
    T1 = T1/max(T1(:));
    mean_T1_volume_BROCCOLI = mean_T1_volume_BROCCOLI + T1;  
    T1_masked = T1(MNI_mask == 1);        
    correlation_BROCCOLI(s) = corr2(T1_masked(:),MNI_masked(:));
    ssd_BROCCOLI(s) = sum( (T1_masked(:) - MNI_masked(:)).^2 );
    T1_masked_ = T1_masked/max(T1_masked(:)) * 256;    
    mutual_information_BROCCOLI(s) = mi(T1_masked_(:),MNI_masked_(:));    
    %correlation_BROCCOLI(s) = corr2(T1(:),MNI(:));
    %ssd_BROCCOLI(s) = sum( (T1(:) - MNI(:)).^2 );
    %T1_ = T1/max(T1(:)) * 256;    
    %mutual_information_BROCCOLI(s) = mi(T1_(:),MNI_(:));        
end
mean_T1_volume_BROCCOLI = mean_T1_volume_BROCCOLI/N;

std_T1_volume_BROCCOLI = zeros(182,218,182);
for s = 1:N
    s
    load([basepath_BROCCOLI '/BROCCOLI_warped_subject'  num2str(s) '.mat']);    
    T1 = aligned_T1_nonparametric_opencl;
    T1 = T1/max(T1(:));    
    std_T1_volume_BROCCOLI = std_T1_volume_BROCCOLI + sqrt((T1 - MNI) .* (T1 - MNI));
end
std_T1_volume_BROCCOLI = std_T1_volume_BROCCOLI / N;

%-------------------------------------------------------------------



close all

figure
image([ MNI(:,:,85)*50  mean_T1_volume_SPM(:,:,85)*75 mean_T1_volume_FSL(:,:,85)*75 mean_T1_volume_AFNI(:,:,85)*75  mean_T1_volume_BROCCOLI(:,:,85)*75 ]); colormap gray
axis off

%text(50,13,'MNI','FontSize',15,'Color','w')
%text(270,13,'SPM','FontSize',15,'Color','w')
%text(490,13,'FSL','FontSize',15,'Color','w')
%text(710,13,'AFNI','FontSize',15,'Color','w')
%text(880,13,'BROCCOLI','FontSize',15,'Color','w')
%print -dpng /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/axial.png

figure
imagesc([ std_T1_volume_SPM(:,:,85) std_T1_volume_FSL(:,:,85) std_T1_volume_AFNI(:,:,85)  std_T1_volume_BROCCOLI(:,:,85) ]); colormap gray
axis off
%print -dpng /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/axial_std.png

figure
image([ flipud(squeeze(MNI(85,:,:))')*50 flipud(squeeze(mean_T1_volume_SPM(85,:,:))')*75 flipud(squeeze(mean_T1_volume_FSL(85,:,:))')*75 flipud(squeeze(mean_T1_volume_AFNI(85,:,:))')*75 flipud(squeeze(mean_T1_volume_BROCCOLI(85,:,:))')*75  ]); colormap gray
axis off
%print -dpng /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/sagittal.png


figure
imagesc([ flipud(squeeze(std_T1_volume_SPM(85,:,:))') flipud(squeeze(std_T1_volume_FSL(85,:,:))') flipud(squeeze(std_T1_volume_AFNI(85,:,:))') flipud(squeeze(std_T1_volume_BROCCOLI(85,:,:))')  ]); colormap gray
axis off
%print -dpng /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/sagittal_std.png

% Calculate mean standard deviation from MNI template
%sum(std_T1_volume_AFNI(:)) / sum(MNI_mask(:))
%sum(std_T1_volume_FSL(:)) / sum(MNI_mask(:))
%sum(std_T1_volume_BROCCOLI(:)) / sum(MNI_mask(:))

mean(mutual_information_SPM)
mean(mutual_information_FSL)
mean(mutual_information_AFNI)
mean(mutual_information_BROCCOLI)

std(mutual_information_SPM)
std(mutual_information_FSL)
std(mutual_information_AFNI)
std(mutual_information_BROCCOLI)

mean(correlation_SPM)
mean(correlation_FSL)
mean(correlation_AFNI)
mean(correlation_BROCCOLI)

std(correlation_SPM)
std(correlation_FSL)
std(correlation_AFNI)
std(correlation_BROCCOLI)

mean(ssd_SPM)
mean(ssd_FSL)
mean(ssd_AFNI)
mean(ssd_BROCCOLI)

std(ssd_SPM)
std(ssd_FSL)
std(ssd_AFNI)
std(ssd_BROCCOLI)





