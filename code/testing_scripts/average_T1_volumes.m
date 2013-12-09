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
    basepath_BROCCOLI = '/data/andek/BROCCOLI_test_data/BROCCOLI/normalization_10its';    
end

N = 198;    % Number of subjects

calculate_std = 0;  % Calculate voxel-wise standard deviation or not

show_SPM_Normalize = 0;
show_SPM_Segment = 0;
show_FSL = 0;
show_AFNI = 0;
show_BROCCOLI = 0;

% Load MNI brain template
MNI_nii = load_nii(['../../brain_templates/MNI152_T1_1mm_brain.nii']);
MNI = double(MNI_nii.img);
MNI = MNI/max(MNI(:));
MNI_ = MNI/max(MNI(:)) * 256;

% Load MNI brain mask
MNI_mask_nii = load_nii(['../../brain_templates/MNI152_T1_1mm_brain_mask.nii']);
MNI_mask = double(MNI_mask_nii.img);

MNI_masked = MNI(MNI_mask == 1);
MNI_masked_ = MNI_masked/max(MNI_masked(:)) * 256;

%-----------------------------------------------------------------
% SPM Normalize Linear interpolation (default)
%-------------------------------------------------------------------

mutual_information_SPM_Normalize = zeros(N,1);
correlation_SPM_Normalize = zeros(N,1);
ssd_SPM_Normalize = zeros(N,1);

mean_T1_volume_SPM_Normalize = zeros(182,218,182);

% Loop over subjects
disp('SPM Normalize')
for s = 1:N
    s
    T1 = load_nii([basepath_SPM_Normalize '/SPM_warped_subject_'  num2str(s) '.nii']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));
    if show_SPM_Normalize == 1
        figure(1)
        imagesc(T1(:,:,85)); colormap gray; pause(0.5); title(['SPM Normalize ' num2str(s)] )
        figure(2)    
        imagesc(flipud(squeeze(T1(85,:,:))')); colormap gray; pause(0.2); title(['SPM Normalize ' num2str(s)])
    end
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
% SPM Segment Linear interpolation (default)
%-------------------------------------------------------------------

mutual_information_SPM_Segment = zeros(N,1);
correlation_SPM_Segment = zeros(N,1);
ssd_SPM_Segment = zeros(N,1);

mean_T1_volume_SPM_Segment = zeros(182,218,182);

% Loop over subjects
disp('SPM Segment')
for s = 1:N
    s
    T1 = load_nii([basepath_SPM_Segment '/SPM_warped_subject_'  num2str(s) '.nii']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));
    if show_SPM_Segment == 1
        figure(1)
        imagesc(T1(:,:,85)); colormap gray; pause(0.5); title(['SPM Segment ' num2str(s)] )
        figure(2)    
        imagesc(flipud(squeeze(T1(85,:,:))')); colormap gray; pause(0.2); title(['SPM Segment ' num2str(s)])
    end   
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
% FSL Linear interpolation (default)
%-------------------------------------------------------------------

mutual_information_FSL = zeros(N,1);
correlation_FSL = zeros(N,1);
ssd_FSL = zeros(N,1);

mean_T1_volume_FSL = zeros(182,218,182);

% Loop over subjects
disp('FSL')
for s = 1:N
    s
    T1 = load_nii([basepath_FSL '/FSL_warped_subject'  num2str(s) '.nii.gz']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));
    if show_FSL == 1
        figure(1)
        imagesc(T1(:,:,85)); colormap gray; pause(0.5); title(['FSL ' num2str(s)] )
        figure(2)    
        imagesc(flipud(squeeze(T1(85,:,:))')); colormap gray; pause(0.2); title(['FSL ' num2str(s)])
    end     
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
        T1 = load_nii([basepath_FSL '/FSL_warped_subject'  num2str(s) '.nii.gz']);
        T1 = double(T1.img);
        T1 = T1/max(T1(:));    
        std_T1_volume_FSL = std_T1_volume_FSL + sqrt((T1 - MNI) .* (T1 - MNI));
    end
    std_T1_volume_FSL = std_T1_volume_FSL / N;
end

%-----------------------------------------------------------------------
% AFNI Linear interpolation
%-------------------------------------------------------------------

mutual_information_AFNI_Linear = zeros(N,1);
correlation_AFNI_Linear = zeros(N,1);
ssd_AFNI_Linear = zeros(N,1);

mean_T1_volume_AFNI_Linear = zeros(182,218,182);

% Loop over subjects
disp('AFNI Linear')
for s = 1:N
    s
    T1 = load_nii([basepath_AFNI '/AFNI_warped_subject'  num2str(s) '.nii']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));
    mean_T1_volume_AFNI_Linear = mean_T1_volume_AFNI_Linear + T1;
    T1_masked = T1(MNI_mask == 1);    
    % Calculate NCC, SSD and mutual information
    correlation_AFNI_Linear(s) = corr2(T1_masked(:),MNI_masked(:));
    ssd_AFNI_Linear(s) = sum( (T1_masked(:) - MNI_masked(:)).^2 );
    T1_masked_ = T1_masked/max(T1_masked(:)) * 256;    
    mutual_information_AFNI_Linear(s) = mi(T1_masked_(:),MNI_masked_(:));         
end
mean_T1_volume_AFNI_Linear = mean_T1_volume_AFNI_Linear/N;

if calculate_std == 1
    std_T1_volume_AFNI_Linear = zeros(182,218,182);
    for s = 1:N
        s
        T1 = load_nii([basepath_AFNI '/AFNI_warped_subject'  num2str(s) '.nii']);
        T1 = double(T1.img);
        T1 = T1/max(T1(:));    
        std_T1_volume_AFNI_Linear = std_T1_volume_AFNI_Linear + sqrt((T1 - MNI) .* (T1 - MNI));
    end
    std_T1_volume_AFNI_Linear = std_T1_volume_AFNI_Linear / N;
end


%-----------------------------------------------------------------------
% AFNI Sinc interpolation (default)
%-------------------------------------------------------------------

mutual_information_AFNI_Sinc = zeros(N,1);
correlation_AFNI_Sinc = zeros(N,1);
ssd_AFNI_Sinc = zeros(N,1);

mean_T1_volume_AFNI_Sinc = zeros(182,218,182);

% Loop over subjects
disp('AFNI Sinc')
for s = 1:N
    s
    T1 = load_nii([basepath_AFNI '/AFNI_warped_subject'  num2str(s) '_sinc.nii']);
    T1 = double(T1.img);
    T1 = T1/max(T1(:));
    if show_AFNI == 1
        figure(1)
        imagesc(T1(:,:,85)); colormap gray; pause(0.5); title(['AFNI ' num2str(s)] )
        figure(2)    
        imagesc(flipud(squeeze(T1(85,:,:))')); colormap gray; pause(0.2); title(['AFNI ' num2str(s)])
    end     
    mean_T1_volume_AFNI_Sinc = mean_T1_volume_AFNI_Sinc + T1;
    T1_masked = T1(MNI_mask == 1);    
    % Calculate NCC, SSD and mutual information
    correlation_AFNI_Sinc(s) = corr2(T1_masked(:),MNI_masked(:));
    ssd_AFNI_Sinc(s) = sum( (T1_masked(:) - MNI_masked(:)).^2 );
    T1_masked_ = T1_masked/max(T1_masked(:)) * 256;    
    mutual_information_AFNI_Sinc(s) = mi(T1_masked_(:),MNI_masked_(:));         
end
mean_T1_volume_AFNI_Sinc = mean_T1_volume_AFNI_Sinc/N;

if calculate_std == 1
    std_T1_volume_AFNI_Sinc = zeros(182,218,182);
    for s = 1:N
        s
        T1 = load_nii([basepath_AFNI '/AFNI_warped_subject'  num2str(s) '_sinc.nii']);
        T1 = double(T1.img);
        T1 = T1/max(T1(:));    
        std_T1_volume_AFNI_Sinc = std_T1_volume_AFNI_Sinc + sqrt((T1 - MNI) .* (T1 - MNI));
    end
    std_T1_volume_AFNI_Sinc = std_T1_volume_AFNI_Sinc / N;
end



%-------------------------------------------------------------------
% BROCCOLI Linear interpolation (default)
%-------------------------------------------------------------------

mutual_information_BROCCOLI = zeros(N,1);
correlation_BROCCOLI = zeros(N,1);
ssd_BROCCOLI = zeros(N,1);

mean_T1_volume_BROCCOLI = zeros(182,218,182);

% Loop over subjects
disp('BROCCOLI')
for s = 1:N
    s    
    T1 = load_nii([basepath_BROCCOLI '/BROCCOLI_warped_subject'  num2str(s) '.nii']);
    T1 = double(T1.img);    
    T1 = T1/max(T1(:));
    if show_BROCCOLI == 1
        figure(1)
        imagesc(T1(:,:,85)); colormap gray; pause(0.5); title(['BROCCOLIL ' num2str(s)] )
        figure(2)    
        imagesc(flipud(squeeze(T1(85,:,:))')); colormap gray; pause(0.2); title(['BROCCOLI ' num2str(s)])
    end     
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
image([ flipud(MNI(:,:,85)')*50  flipud(mean_T1_volume_SPM_Normalize(:,:,85)')*75 flipud(mean_T1_volume_SPM_Segment(:,:,85)')*75   ; flipud(mean_T1_volume_FSL(:,:,85)')*75 flipud(mean_T1_volume_AFNI_Sinc(:,:,85)')*75  flipud(mean_T1_volume_BROCCOLI(:,:,85)')*75 ; zeros(15,546) ]); colormap gray
axis equal
axis off
text(10,25,'A','Color','White','FontSize',17)
text(200,25,'B','Color','White','FontSize',17)
text(380,25,'C','Color','White','FontSize',17)
text(10,225,'D','Color','White','FontSize',17)
text(200,225,'E','Color','White','FontSize',17)
text(380,225,'F','Color','White','FontSize',17)

text(5,120,'L','Color','White','FontSize',18)
text(5,340,'L','Color','White','FontSize',18)
text(460,430,'z = 13mm','Color','White','FontSize',18)

%print -dpng /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/axial_.png
%export_fig /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/axial_.png -png -native

% Show voxel-wise standard deviation
if calculate_std == 1
    figure(2)
    imagesc([ std_T1_volume_SPM_Normalize(:,:,85) std_T1_volume_SPM_Segment(:,:,85) std_T1_volume_FSL(:,:,85) std_T1_volume_AFNI_Sinc(:,:,85)  std_T1_volume_BROCCOLI(:,:,85) ]); colormap gray
    axis equal
    axis off
    %print -dpng /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/axial_std.png
end

% Show average normalized T1 volumes
figure(3)
image([ flipud(squeeze(MNI(85,:,:))')*50 flipud(squeeze(mean_T1_volume_SPM_Normalize(85,:,:))')*75 flipud(squeeze(mean_T1_volume_SPM_Segment(85,:,:))')*75    ; flipud(squeeze(mean_T1_volume_FSL(85,:,:))')*75 flipud(squeeze(mean_T1_volume_AFNI_Sinc(85,:,:))')*75 flipud(squeeze(mean_T1_volume_BROCCOLI(85,:,:))')*75 ; zeros(30,654) ]); colormap gray
axis equal
axis off
text(10,25,'A','Color','White','FontSize',18)
text(230,25,'B','Color','White','FontSize',18)
text(450,25,'C','Color','White','FontSize',18)
text(10,225,'D','Color','White','FontSize',18)
text(230,225,'E','Color','White','FontSize',18)
text(450,225,'F','Color','White','FontSize',18)

text(575,380,'x = 8mm','Color','White','FontSize',18)

%print -dpng /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/sagittal_.png
%export_fig /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/sagittal_.png -png -native

% Show voxel-wise standard deviation
if calculate_std == 1
    figure(4)
    imagesc([ flipud(squeeze(std_T1_volume_SPM_Normalize(85,:,:))') flipud(squeeze(std_T1_volume_SPM_Segment(85,:,:))') flipud(squeeze(std_T1_volume_FSL(85,:,:))') flipud(squeeze(std_T1_volume_AFNI_Sinc(85,:,:))') flipud(squeeze(std_T1_volume_BROCCOLI(85,:,:))')  ]); colormap gray
    axis equal
    axis off
    %print -dpng /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/sagittal_std.png
end

figure(5)
%image([ flipud(squeeze(MNI(85,:,:))')*50 flipud(squeeze(mean_T1_volume_SPM_Normalize(85,:,:))')*75 flipud(squeeze(mean_T1_volume_SPM_Segment(85,:,:))')*75    ; flipud(squeeze(mean_T1_volume_FSL(85,:,:))')*75 flipud(squeeze(mean_T1_volume_AFNI_Sinc(85,:,:))')*75 flipud(squeeze(mean_T1_volume_BROCCOLI(85,:,:))')*75 ; zeros(30,654) ]); colormap gray
%image([ flipud(MNI(:,:,85)')*50  flipud(mean_T1_volume_SPM_Normalize(:,:,85)')*75 flipud(mean_T1_volume_SPM_Segment(:,:,85)')*75   ; flipud(mean_T1_volume_FSL(:,:,85)')*75 flipud(mean_T1_volume_AFNI_Sinc(:,:,85)')*75  flipud(mean_T1_volume_BROCCOLI(:,:,85)')*75 ; zeros(15,546) ]); colormap gray
%image([ [flipud(MNI(:,:,85)')*50 zeros(1,36)]  [flipud(squeeze(MNI(85,:,:))')*50 zeros(36,1)] [flipud(mean_T1_volume_SPM_Normalize(:,:,85)')*75 zeros(1,36)] [flipud(squeeze(mean_T1_volume_SPM_Normalize(85,:,:))')*75 zeros(36,1)]]); colormap gray
MNI_axial = [flipud(MNI(10:end-10,:,85)')*50 zeros(218,1)];
MNI_sagittal = [zeros(18,218); flipud(squeeze(MNI(85,:,:))')*50 ; zeros(18,218)];
SPM_Normalize_axial = [flipud(mean_T1_volume_SPM_Normalize(10:end-10,:,85)')*75 zeros(218,1)];
SPM_Normalize_sagittal = [zeros(18,218); flipud(squeeze(mean_T1_volume_SPM_Normalize(85,:,:))')*75 ; zeros(18,218)];
SPM_Segment_axial = [flipud(mean_T1_volume_SPM_Segment(10:end-10,:,85)')*75 zeros(218,1)];
SPM_Segment_sagittal = [zeros(18,218); flipud(squeeze(mean_T1_volume_SPM_Segment(85,:,:))')*75 ; zeros(18,218)];
FSL_axial = [flipud(mean_T1_volume_FSL(10:end-10,:,85)')*75 zeros(218,1)];
FSL_sagittal = [zeros(18,218); flipud(squeeze(mean_T1_volume_FSL(85,:,:))')*75 ; zeros(18,218)];
AFNI_axial = [flipud(mean_T1_volume_AFNI_Sinc(10:end-10,:,85)')*75 zeros(218,1)];
AFNI_sagittal = [zeros(18,218); flipud(squeeze(mean_T1_volume_AFNI_Sinc(85,:,:))')*75 ; zeros(18,218)];
BROCCOLI_axial = [flipud(mean_T1_volume_BROCCOLI(10:end-10,:,85)')*75 zeros(218,1)];
BROCCOLI_sagittal = [zeros(18,218); flipud(squeeze(mean_T1_volume_BROCCOLI(85,:,:))')*75 ; zeros(18,218)];



image([zeros(30,804); zeros(218,20) MNI_axial MNI_sagittal SPM_Normalize_axial SPM_Normalize_sagittal zeros(218,20); zeros(218,20) SPM_Segment_axial SPM_Segment_sagittal FSL_axial FSL_sagittal zeros(218,20); zeros(218,20) AFNI_axial AFNI_sagittal BROCCOLI_axial BROCCOLI_sagittal zeros(218,20); zeros(30,804)]); colormap gray
axis equal
axis off
text(10,30,'A','Color','White','FontSize',17)
text(400,30,'B','Color','White','FontSize',17)
text(10,260,'C','Color','White','FontSize',17)
text(400,260,'D','Color','White','FontSize',17)
text(10,490,'E','Color','White','FontSize',17)
text(400,490,'F','Color','White','FontSize',17)

text(10,140,'L','Color','White','FontSize',16)
text(10,360,'L','Color','White','FontSize',16)
text(10,580,'L','Color','White','FontSize',16)

text(265,700,'x = 8mm','Color','White','FontSize',16)
text(88,700,'z = 13mm','Color','White','FontSize',16)

%print -dpng /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/axial_.png
export_fig /home/andek/Dropbox/Dokument/VirginiaTech/papers/Frontiers_in_NeuroInformatics_Parallel/combined.png -png -native


% Calculate mean and standard deviation of correlation
mean(correlation_SPM_Normalize)
mean(correlation_SPM_Segment)
mean(correlation_FSL)
mean(correlation_AFNI_Linear)
mean(correlation_AFNI_Sinc)
mean(correlation_BROCCOLI)

std(correlation_SPM_Normalize)
std(correlation_SPM_Segment)
std(correlation_FSL)
std(correlation_AFNI_Linear)
std(correlation_AFNI_Sinc)
std(correlation_BROCCOLI)

% Calculate mean and standard deviation of mutual information
mean(mutual_information_SPM_Normalize)
mean(mutual_information_SPM_Segment)
mean(mutual_information_FSL)
mean(mutual_information_AFNI_Linear)
mean(mutual_information_AFNI_Sinc)
mean(mutual_information_BROCCOLI)

std(mutual_information_SPM_Normalize)
std(mutual_information_SPM_Segment)
std(mutual_information_FSL)
std(mutual_information_AFNI_Linear)
std(mutual_information_AFNI_Sinc)
std(mutual_information_BROCCOLI)

% Calculate mean and standard deviation of ssd
mean(ssd_SPM_Normalize/300000)
mean(ssd_SPM_Segment/300000)
mean(ssd_FSL/300000)
mean(ssd_AFNI_Linear/300000)
mean(ssd_AFNI_Sinc/300000)
mean(ssd_BROCCOLI/300000)

std(ssd_SPM_Normalize/300000)
std(ssd_SPM_Segment/300000)
std(ssd_FSL/300000)
std(ssd_AFNI_Linear/300000)
std(ssd_AFNI_Sinc/300000)
std(ssd_BROCCOLI/300000)





