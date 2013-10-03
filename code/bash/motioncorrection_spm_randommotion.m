
clear all
clc
close all

if ispc
    addpath('D:/spm8')
    data_path = 'D:/BROCCOLI_test_data/Cambridge/with_random_motion';
    results_directory = '/data/andek/BROCCOLI_test_data/SPM/';    
elseif isunix
    addpath('/data/andek/spm8/')
    data_path = '/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion';
    results_directory = '/data/andek/BROCCOLI_test_data/SPM/';
end

noise_level = '_1percent_noise';
%noise_level = '_no_noise';
%noise_level = '';

try
    system(['rm' ' batch_preprocessing.mat']);
end
    
% Loop over subjects

tic
for s = 1:10
    
    
    %% Initialise SPM defaults
    %--------------------------------------------------------------------------
    spm('Defaults','fMRI');
    spm_jobman('initcfg'); % useful in SPM8 only
    
    
    %% WORKING DIRECTORY (useful for .ps only)
    %--------------------------------------------------------------------------
    clear pjobs
    pjobs{1}.util{1}.cdir.directory = cellstr(data_path);    
        
    %subject = dirs(s+2).name
    
    %filename = [data_path '/' subject];                
    
    filename = [data_path '/cambridge_rest_subject_' num2str(s) '_with_random_motion' noise_level '.nii'];    
    subject = ['cambridge_rest_subject_' num2str(s) '_with_random_motion' noise_level '.nii']
            
    %% Motion correction settings
    
    % Set data to use
    pjobs{2}.spatial{1}.realign{1}.estwrite.data{1} = cellstr(filename);
    % Register to first volume
    pjobs{2}.spatial{1}.realign{1}.estwrite.eoptions.rtm = 0;  
    % Trilinear interpolation for estimation
    pjobs{2}.spatial{1}.realign{1}.estwrite.eoptions.interp = 1;          
    % Trilinear interpolation for reslice
    pjobs{2}.spatial{1}.realign{1}.estwrite.roptions.interp = 1;          
    % Only reslice images, not mean
    pjobs{2}.spatial{1}.realign{1}.estwrite.roptions.which = [2 0];

    save('batch_preprocessing.mat','pjobs');
        
    error1 = 0;
    try        
        % Run preprocessing
        spm_jobman('run',pjobs);        
    catch err
        err
        error1 = 1;
    end
    
    
    % Move files to results directory 
    motion_corrected_data = ['r' subject ];
    motion_corrected_matrix = ['r' subject(1:end-4) '.mat' ];
    motion_parameters = ['rp_' subject(1:end-4) '.txt'];    
    mat = [subject(1:end-4) '.mat'];
    
    system(['mv ' motion_corrected_data ' ' results_directory]);
    system(['mv ' motion_corrected_matrix ' ' results_directory]);
    system(['mv ' motion_parameters ' ' results_directory]);    
    system(['rm ' mat]);
    
        
    try
        system(['rm' ' batch_preprocessing.mat']);
    end
    
end
toc









