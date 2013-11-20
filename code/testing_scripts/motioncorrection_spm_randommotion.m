
clear all
clc
close all

if ispc
    addpath('D:/spm8')
    data_path = 'D:/BROCCOLI_test_data/Cambridge/with_random_motion';
    results_directory = '/data/andek/BROCCOLI_test_data/SPM/motion_correction/';    
elseif isunix
    addpath('/data/andek/spm8/')
    data_path = '/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion';
    results_directory = '/data/andek/BROCCOLI_test_data/SPM/motion_correction/';
end

%noise_level = '_no_noise';
%noise_level = '_2percent_noise';
noise_level = '_shading';

%interpolation = 1; % Linear interpolation
interpolation = 2; % B-spline interpolation

try
    system(['rm' ' batch_preprocessing.mat']);
end
    
N = 198;

% Loop over subjects

motion_correction_times = zeros(N,1);

tic
for s = 1:N
    
    
    %% Initialise SPM defaults
    %--------------------------------------------------------------------------
    spm('Defaults','fMRI');
    spm_jobman('initcfg'); 
    
    
    %% WORKING DIRECTORY (useful for .ps only)
    %--------------------------------------------------------------------------
    clear pjobs
    pjobs{1}.util{1}.cdir.directory = cellstr(data_path);                      
    
    filename = [data_path '/cambridge_rest_subject_' num2str(s) '_with_random_motion' noise_level '.nii'];    
    subject = ['cambridge_rest_subject_' num2str(s) '_with_random_motion' noise_level '.nii']
            
    %% Motion correction settings
    
    % Set data to use
    pjobs{2}.spatial{1}.realign{1}.estwrite.data{1} = cellstr(filename);
    
    % Register to first volume
    pjobs{2}.spatial{1}.realign{1}.estwrite.eoptions.rtm = 0;      
        
    if interpolation == 1
        % Trilinear interpolation for estimation
        pjobs{2}.spatial{1}.realign{1}.estwrite.eoptions.interp = 1;     
        % Trilinear interpolation for reslice
        pjobs{2}.spatial{1}.realign{1}.estwrite.roptions.interp = 1;          
    elseif interpolation == 2
        % 2nd degree B-spline for estimation (default)
        pjobs{2}.spatial{1}.realign{1}.estwrite.eoptions.interp = 2;     
        % 4th degree B-spline interpolation for reslice (default)
        pjobs{2}.spatial{1}.realign{1}.estwrite.roptions.interp = 4;          
    end
    
    % Only reslice images, not mean
    pjobs{2}.spatial{1}.realign{1}.estwrite.roptions.which = [2 0];

    save('batch_preprocessing.mat','pjobs');
        
    error1 = 0;
    start = clock;
    try        
        % Run preprocessing        
        spm_jobman('run',pjobs);                
    catch err
        err
        error1 = 1;
    end
    motion_correction_times(s) = etime(clock,start)
    
    % Move files to results directory 
    
    motion_corrected_data = ['r' subject ];
    motion_corrected_matrix = ['r' subject(1:end-4) '.mat' ];
    motion_parameters = ['rp_' subject(1:end-4) '.txt'];    
        
    if interpolation == 1
        new_motion_corrected_data = motion_corrected_data;
        new_motion_corrected_matrix = motion_corrected_matrix;
        new_motion_parameters = motion_parameters;    
    elseif interpolation == 2
        new_motion_corrected_data = ['r' subject(1:end-4)  '_spline.nii'];
        new_motion_corrected_matrix = ['r' subject(1:end-4) '_spline.mat' ];
        new_motion_parameters = ['rp_' subject(1:end-4) '_spline.txt'];    
    end
    mat = [subject(1:end-4) '.mat'];
    
    system(['mv ' motion_corrected_data ' ' results_directory new_motion_corrected_data]);
    system(['mv ' motion_corrected_matrix ' ' results_directory new_motion_corrected_matrix]);
    system(['mv ' motion_parameters ' ' results_directory new_motion_parameters]);    
    system(['rm ' mat]);
    
        
    try
        system(['rm' ' batch_preprocessing.mat']);
    end
    
end
toc









