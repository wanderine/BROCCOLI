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
    basepath_FSL = '/data/andek/BROCCOLI_test_data/FSL';
    basepath_AFNI = '/data/andek/BROCCOLI_test_data/AFNI';
    basepath_BROCCOLI = '/data/andek/BROCCOLI_test_data/BROCCOLI/original_motion';
    basepath_none = '/data/andek/BROCCOLI_test_data/Cambridge/';
end

N = 198;


%-----------------------------------------------------------------------
% AFNI
%-------------------------------------------------------------------

mutual_information_AFNI = zeros(N,1);
correlation_AFNI = zeros(N,1);
ssd_AFNI = zeros(N,1);

for s = 1:N
    
    s
    mc = load_nii([basepath_AFNI '/AFNI_motion_corrected_subject'  num2str(s) '_original_motion.nii']);
    mc = double(mc.img);
    mc(isnan(mc)) = 0;
    mc = mc/max(mc(:));
    reference = mc(:,:,:,1);    
    
    for t = 2:119
        volume = mc(:,:,:,t);        
        correlation_AFNI(s) = correlation_AFNI(s) + corr2(reference(:),volume(:));
        ssd_AFNI(s) = ssd_AFNI(s) + sum( (reference(:) - volume(:)).^2 );
        mutual_information_AFNI(s) = mutual_information_AFNI(s) + mi(reference(:)*256,volume(:)*256);
    end
    
    correlation_AFNI(s) = correlation_AFNI(s) / 118;
    ssd_AFNI(s) = ssd_AFNI(s) / 118;
    mutual_information_AFNI(s) = mutual_information_AFNI(s) / 118;
    
end
 
%-----------------------------------------------------------------
% FSL
%-------------------------------------------------------------------
 
mutual_information_FSL = zeros(N,1);
correlation_FSL = zeros(N,1);
ssd_FSL = zeros(N,1);

for s = 1:N

    s
    mc = load_nii([basepath_FSL '/FSL_motion_corrected_subject'  num2str(s) '_original_motion.nii.gz']);
    mc = double(mc.img);
    mc(isnan(mc)) = 0;
    mc = mc/max(mc(:));
    reference = mc(:,:,:,1);    

    for t = 2:119
        volume = mc(:,:,:,t);        
        correlation_FSL(s) = correlation_FSL(s) + corr2(reference(:),volume(:));
        ssd_FSL(s) = ssd_FSL(s) + sum( (reference(:) - volume(:)).^2 );
        mutual_information_FSL(s) = mutual_information_FSL(s) + mi(reference(:)*256,volume(:)*256);
     end

    correlation_FSL(s) = correlation_FSL(s) / 118;
    ssd_FSL(s) = ssd_FSL(s) / 118;
    mutual_information_FSL(s) = mutual_information_FSL(s) / 118;

end


%-------------------------------------------------------------------
% BROCCOLI
%-------------------------------------------------------------------

mutual_information_BROCCOLI = zeros(N,1);
correlation_BROCCOLI = zeros(N,1);
ssd_BROCCOLI = zeros(N,1);

dirs = dir([basepath_BROCCOLI]);

for s = 1:N
    
    s
    subject = dirs(s+2).name;
    
    load([basepath_BROCCOLI '/' subject]);
    mc = motion_corrected_volumes_opencl;    
    mc(isnan(mc)) = 0;
    mc = mc / max(mc(:));        
    reference = mc(:,:,:,1);    
    
    for t = 2:119
        volume = mc(:,:,:,t);        
        correlation_BROCCOLI(s) = correlation_BROCCOLI(s) + corr2(volume(:),reference(:));        
        ssd_BROCCOLI(s) = ssd_BROCCOLI(s) + sum( (volume(:) - reference(:)).^2 );
        mutual_information_BROCCOLI(s) = mutual_information_BROCCOLI(s) + mi(volume(:)*256,reference(:)*256);
    end
    
    correlation_BROCCOLI(s) = correlation_BROCCOLI(s) / 118;
    ssd_BROCCOLI(s) = ssd_BROCCOLI(s) / 118;
    mutual_information_BROCCOLI(s) = mutual_information_BROCCOLI(s) / 118;
    
end


%-------------------------------------------------------------------
% No motion correction
%-------------------------------------------------------------------

mutual_information_none = zeros(N,1);
correlation_none = zeros(N,1);
ssd_none = zeros(N,1);

dirs = dir([basepath_none]);

for s = 1:N
    
    s
    subject = dirs(s+2).name;
    
    mc1 = load_nii([basepath_none '/' subject '/func/rest.nii.gz']);
    mc1 = double(mc1.img);
    mc1(isnan(mc1)) = 0;
    mc1 = mc1 / max(mc1(:));    
    reference = mc1(:,:,:,1);    
    
    for t = 2:119
        volume = mc1(:,:,:,t);
        correlation_none(s) = correlation_none(s) + corr2(volume(:),reference(:));
        ssd_none(s) = ssd_none(s) + sum( (volume(:) - reference(:)).^2 );
        mutual_information_none(s) = mutual_information_none(s) + mi(volume(:)*256,reference(:)*256);
    end
    
    correlation_none(s) = correlation_none(s) / 118;
    ssd_none(s) = ssd_none(s) / 118;
    mutual_information_none(s) = mutual_information_none(s) / 118;
    
end

%-------------------------------------------------------------------



mean(mutual_information_AFNI)
mean(mutual_information_FSL)
mean(mutual_information_BROCCOLI)
mean(mutual_information_none)

std(mutual_information_AFNI)
std(mutual_information_FSL)
std(mutual_information_BROCCOLI)
std(mutual_information_none)

mean(correlation_AFNI)
mean(correlation_FSL)
mean(correlation_BROCCOLI)
mean(correlation_none)

std(correlation_AFNI)
std(correlation_FSL)
std(correlation_BROCCOLI)
std(correlation_none)

mean(ssd_AFNI)
mean(ssd_FSL)
mean(ssd_BROCCOLI)
mean(ssd_none)

std(ssd_AFNI)
std(ssd_FSL)
std(ssd_BROCCOLI)
std(ssd_none)



